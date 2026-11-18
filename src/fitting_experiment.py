"""Separately versioned, paired raw-history fitting ablation for Report 4.

--preflight reads local model metadata and saved prompts, but generates no tokens.
--run requires explicit pilot IDs and saves two new predictions per selected ID.
The original pilot, caches, and prior reports are read only. This diagnostic
experiment changes fitting only; it does not establish architecture accuracy.
"""
import argparse
import fcntl
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from audit_progress import writer_active
from compare import records
from experiment import ROOT, Ollama, atomic_json, blocks, digest, fit, history, prompt
from run_experiment import load_data
from tokenizer_probe import (TEMPLATE_SHA256, build_tokenizer, check_template,
                             fit_exact, render_generate)

VARIANTS = ('legacy_byte_bound', 'exact_token_count')
FROZEN_SOURCES = ('experiment.py', 'memories.py', 'run_experiment.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_output(path, allowed=None):
    """Refuse any output outside the new report's directory, including symlinks."""
    allowed = (allowed or ROOT / 'results/progress_report_4').resolve()
    path = Path(path).resolve()
    if path == allowed or allowed not in path.parents:
        raise ValueError('Use a separate subdirectory under results/progress_report_4')
    return path


def verify_protocol(protocol, cfg):
    base = ROOT / protocol['base_config']
    if hashlib.sha256(base.read_bytes()).hexdigest() != protocol['base_config_sha256']:
        raise ValueError('Base configuration changed; version a new protocol')
    source_hash = digest({name: (ROOT / 'src' / name).read_text() for name in FROZEN_SOURCES})
    if source_hash != protocol['expected_inference_source_hash']:
        raise ValueError('Frozen pilot inference source changed')
    manifest = json.loads((ROOT / protocol['pilot_manifest']).read_text())
    expected_cfg = dict(cfg, cache_dir=str(ROOT / cfg['cache_dir']))
    if (manifest['config'] != expected_cfg or manifest['code_hash'] != source_hash
            or manifest['identity'] != protocol['expected_identity']):
        raise ValueError('Protocol does not match the preserved pilot manifest')
    if protocol['variants'] != list(VARIANTS):
        raise ValueError('Unrecognized fitting variants')
    return source_hash


def validated_counter(cfg, expected_identity):
    """Verify installed identity/template and every saved prompt count first."""
    import tokenizers
    client = Ollama(cfg)
    identity = client.identity({cfg['answer_model']})
    if identity != expected_identity:
        raise ValueError('Installed runtime/model differs from the frozen protocol')
    data = client.request('/api/show', {'model': cfg['answer_model'], 'verbose': True})
    check_template(data)
    tokenizer = build_tokenizer(data['model_info'])
    counter = lambda text: len(tokenizer.encode(render_generate(text), add_special_tokens=False).ids)
    comparisons = []
    for approach in ('raw_history', 'vector_retrieval', 'running_summary'):
        manifest, rows = records(ROOT / 'results/pilot' / approach)
        if any(manifest['identity'][k] != identity[k] for k in identity):
            raise ValueError('Saved model/runtime differs from installed model/runtime')
        for row in rows:
            if row['status'] == 'ok' and row.get('input_tokens') is not None:
                observed = counter(row['prompt'])
                if observed != row['input_tokens']:
                    raise ValueError(f"Tokenizer mismatch: {approach}/{row['question_id']}")
                comparisons.append({'approach': approach, 'question_id': row['question_id'],
                    'prompt_hash': digest(row['prompt']), 'runtime_tokens': row['input_tokens'],
                    'offline_tokens': observed})
    if not comparisons:
        raise ValueError('No saved runtime prompt counts for counter validation')
    return client, counter, {'identity': identity, 'tokenizers_version': tokenizers.__version__,
        'template_sha256': TEMPLATE_SHA256, 'vocabulary_hash': digest(data['model_info']),
        'matched_saved_prompts': len(comparisons), 'saved_prompt_comparisons': comparisons}


def fit_pair(row, cfg, counter):
    """Create both prompts from allowlisted turns; annotations never enter fitting."""
    parts = blocks(history(row))
    make_prompt = lambda memory: prompt(cfg, memory, row['question'], row['question_date'])
    budget = cfg['context_tokens'] - cfg['output_tokens'] - cfg['template_margin_tokens']
    if budget <= 0:
        raise ValueError('Nonpositive input budget')
    legacy_text, legacy_selection = fit(parts, make_prompt, budget, True)
    exact_text, exact_selection = fit_exact(parts, make_prompt, counter, budget)
    result = {}
    for name, text, selected in (
            (VARIANTS[0], legacy_text, legacy_selection),
            (VARIANTS[1], exact_text, exact_selection)):
        count = counter(text)
        if count > budget or count + cfg['output_tokens'] > cfg['context_tokens']:
            raise ValueError('Fitted prompt exceeds validated input/output budget')
        result[name] = {'prompt': text, 'prompt_hash': digest(text),
            'local_prompt_tokens': count, 'input_budget_tokens': budget,
            'selection': {**selected, 'fitting_method': name,
                'retained_turn_fraction': selected['retained_blocks'] / len(parts) if parts else None}}
    return result


def validate_measurement(response, expected_tokens, cfg):
    count = response.get('prompt_eval_count')
    if type(count) is not int or count != expected_tokens:
        raise ValueError('Runtime prompt count does not match the validated local counter')
    if count + cfg['output_tokens'] > cfg['context_tokens']:
        raise ValueError('Runtime prompt plus output allowance exceeds context')
    if not response.get('done') or not response.get('response', '').strip():
        raise ValueError('Incomplete or empty generation')


def execute_variant(row, variant, fitted, cfg, client, manifest_hash, order):
    """One answer call, with errors preserved rather than converted to successes."""
    started = time.perf_counter()
    client.calls = []
    record = {'question_id': row['question_id'], 'question_type': row['question_type'],
        'category': 'abstention' if row['question_id'].endswith('_abs') else row['question_type'],
        'approach': 'raw_history', 'fitting_variant': variant, 'manifest_hash': manifest_hash,
        'started_at_utc': now(), 'status': 'running', 'pair_order': order,
        **fitted, 'prediction': None, 'input_tokens': None, 'output_tokens': None,
        'answer_seconds': None, 'official_accuracy': None}
    try:
        answer = client.generate(fitted['prompt'], cfg['answer_model'], cfg['output_tokens'], 'answer')
        record.update(prediction=answer.get('response', '').strip(),
            input_tokens=answer.get('prompt_eval_count'), output_tokens=answer.get('eval_count'),
            done_reason=answer.get('done_reason'))
        validate_measurement(answer, fitted['local_prompt_tokens'], cfg)
        record.update(status='ok', runtime_count_matches=True)
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                      error=f'{type(exc).__name__}: {exc}', runtime_count_matches=False)
    record.update(answer_seconds=time.perf_counter() - started, calls=client.calls,
                  finished_at_utc=now())
    return record


def existing_record(path, expected_manifest_hash, fitted):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record.get('manifest_hash') != expected_manifest_hash
            or record.get('prompt_hash') != fitted['prompt_hash']
            or digest(record.get('prompt')) != fitted['prompt_hash']):
        raise ValueError('Incompatible/stale experiment record; use a new output directory')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/fitting_experiment.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--question-ids', nargs='+', help='Explicit diagnostic pilot IDs for generation')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if args.run and not args.question_ids:
        parser.error('--run requires explicit --question-ids; no implicit full-pilot generation')
    protocol = json.loads(args.config.read_text())
    cfg = json.loads((ROOT / protocol['base_config']).read_text())
    frozen_hash = verify_protocol(protocol, cfg)
    by_id, ids = load_data(cfg)
    chosen = args.question_ids or ids
    if len(chosen) != len(set(chosen)) or any(qid not in ids for qid in chosen):
        raise ValueError('Question IDs must be unique members of the frozen pilot')
    output = safe_output(args.output_dir or ROOT / protocol['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.run and any(writer_active(ROOT / 'results/pilot' / a)
                            for a in ('raw_history', 'vector_retrieval', 'running_summary')):
            raise ValueError('Finish active pilot inference before the paired latency experiment')
        client, counter, validation = validated_counter(cfg, protocol['expected_identity'])
        source_files = (*FROZEN_SOURCES, 'tokenizer_probe.py', 'fitting_experiment.py')
        manifest = {'version': protocol['version'], 'protocol': protocol, 'base_config': cfg,
            'pilot_ids': ids, 'python': platform.python_version(), 'identity': validation['identity'],
            'template_sha256': validation['template_sha256'],
            'vocabulary_hash': validation['vocabulary_hash'],
            'tokenizers_version': validation['tokenizers_version'], 'frozen_source_hash': frozen_hash,
            'source_hash': digest({name: (ROOT / 'src' / name).read_text() for name in source_files})}
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Incompatible experiment manifest; use a new output directory')
        atomic_json(manifest_path, manifest)
        fingerprint = digest(manifest)
        fitted = {qid: fit_pair(by_id[qid], cfg, counter) for qid in ids}
        _, prior_rows = records(ROOT / 'results/pilot/raw_history')
        prior_prompts = {row['question_id']: row['prompt'] for row in prior_rows
                         if row['status'] == 'ok'}
        if any(fitted[qid]['legacy_byte_bound']['prompt'] != prior_prompts.get(qid) for qid in ids):
            raise ValueError('Reconstructed byte-bound prompts differ from the original pilot')
        preflight = {'generated_at_utc': now(), 'manifest_hash': fingerprint,
            'generation_calls': 0, 'validation': validation,
            'questions': {qid: {variant: {k: v for k, v in item.items() if k != 'prompt'}
                               for variant, item in pair.items()} for qid, pair in fitted.items()},
            'note': 'Offline fits, not answer predictions. Same original turns and answer instructions; '
                    'the exact counter includes the verified runtime template. Fitting budget also '
                    'retains the original template margin as a conservative extra reserve.'}
        atomic_json(output / 'preflight.json', preflight)
        if args.preflight:
            print(f"Validated {validation['matched_saved_prompts']} saved counts; fitted {len(ids)} histories; no generation.")
            return
        atomic_json(output / 'runtime_request.json', {
            'started_at_utc': now(), 'manifest_hash': fingerprint,
            'requested_question_ids': chosen, 'requested_pairs': len(chosen),
            'protocol_note': 'Explicit diagnostic cases; full pilot fitting is offline only.'})
        for qid in chosen:
            order = list(VARIANTS if ids.index(qid) % 2 == 0 else VARIANTS[::-1])
            for variant in order:
                path = output / variant / (qid + '.json')
                prior = existing_record(path, fingerprint, fitted[qid][variant])
                if prior is not None:
                    if prior['status'] == 'ok' or not args.retry_errors:
                        print(f"Skip {qid}/{variant}: {prior['status']}", flush=True)
                        continue
                    atomic_json(output / 'attempts' / f'{qid}-{variant}-{time.time_ns()}.json', prior)
                atomic_json(path, {'question_id': qid, 'fitting_variant': variant,
                    'manifest_hash': fingerprint, **fitted[qid][variant], 'status': 'running',
                    'started_at_utc': now()})
                print(f"Generate {qid}/{variant}: {fitted[qid][variant]['local_prompt_tokens']} prompt tokens", flush=True)
                record = execute_variant(by_id[qid], variant, fitted[qid][variant], cfg, client, fingerprint, order)
                atomic_json(path, record)
                print(f"  {record['status']}: {record['answer_seconds']:.2f} seconds", flush=True)
                if record['status'] != 'ok':
                    raise SystemExit(f"Stopped after {qid}/{variant}: {record.get('error')}")
        status = {}
        for qid in ids:
            status[qid] = {}
            for variant in VARIANTS:
                record = existing_record(output / variant / (qid + '.json'), fingerprint, fitted[qid][variant])
                status[qid][variant] = record['status'] if record else 'pending'
        atomic_json(output / 'coverage.json', {'generated_at_utc': now(), 'manifest_hash': fingerprint,
            'status_by_question': status,
            'successful_pairs': sum(all(s == 'ok' for s in pair.values()) for pair in status.values()),
            'candidate_pairs': len(ids), 'requested_question_ids': chosen,
            'requested_pairs': len(chosen), 'official_accuracy': None,
            'note': 'A small explicitly selected paired diagnostic, not a full three-method rerun. '
                    'Prompt cache/loading conditions and one observation per variant limit latency interpretation.'})


if __name__ == '__main__':
    main()
