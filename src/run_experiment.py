"""Run fixed pilot with atomic per-question records, manifest, and resume checks."""
import argparse
import fcntl
import hashlib
import json
import platform
import time
from pathlib import Path
from experiment import ROOT, VERSION, Ollama, atomic_json, digest, fit, history, prompt
from memories import APPROACHES


def load_data(cfg):
    path = ROOT / cfg['dataset']
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    rows = json.loads(path.read_text())
    by_id = {r['question_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate dataset IDs')
    ids = json.loads((ROOT / cfg['pilot_ids']).read_text())
    if len(ids) != len(set(ids)) or any(q not in by_id for q in ids):
        raise ValueError('Duplicate or missing pilot IDs')
    return by_id, ids


def check_manifest(path, expected):
    if path.exists() and json.loads(path.read_text()) != expected:
        raise ValueError('Incompatible resume: use a new output directory')
    atomic_json(path, expected)


def run_question(row, approach, cfg, client, identity, fingerprint):
    started = time.perf_counter()
    client.calls = []
    record = {'question_id': row['question_id'], 'question_type': row.get('question_type'),
        'category': 'abstention' if row['question_id'].endswith('_abs') else row.get('question_type'),
        'approach': approach, 'model': cfg['answer_model'], 'identity': identity,
        'config_hash': fingerprint, 'status': 'running', 'prediction': None,
        'input_tokens': None, 'output_tokens': None, 'answer_seconds': None,
        'preparation_seconds': None}
    try:
        memory = APPROACHES[approach](cfg, client, identity)
        parts, metadata = memory.prepare(history(row), row['question'])
        text, selection = fit(parts, lambda m: prompt(cfg, m, row['question'], row['question_date']),
            cfg['context_tokens'] - cfg['output_tokens'] - cfg['template_margin_tokens'], memory.drop_oldest)
        record.update(memory=metadata, selection=selection, prompt=text,
                      preparation_seconds=time.perf_counter() - started)
        answer_started = time.perf_counter()
        result = client.generate(text, cfg['answer_model'], cfg['output_tokens'], 'answer')
        record.update(prediction=result['response'].strip(), input_tokens=result.get('prompt_eval_count'),
            output_tokens=result.get('eval_count'), answer_seconds=time.perf_counter() - answer_started,
            done_reason=result.get('done_reason'), status='ok')
        if record['input_tokens'] is not None and record['input_tokens'] + cfg['output_tokens'] > cfg['context_tokens']:
            raise ValueError('Runtime prompt count exceeds configured input allowance')
        record['missing_measurements'] = [k for k in ('input_tokens', 'output_tokens') if record[k] is None]
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                      error=f'{type(exc).__name__}: {exc}')
    record.update(total_seconds=time.perf_counter() - started, calls=client.calls)
    if record['preparation_seconds'] is None:
        record['preparation_seconds'] = record['total_seconds']
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--approach', choices=APPROACHES, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    cfg['cache_dir'] = str(ROOT / cfg['cache_dir'])
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    # Hold advisory lock through entire run: parallel writers cannot overwrite work.
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        by_id, ids = load_data(cfg)
        client = Ollama(cfg)
        models = {cfg['answer_model']}
        if args.approach == 'vector_retrieval':
            models.add(cfg['retrieval']['embedding_model'])
        if args.approach == 'running_summary':
            models.add(cfg['summary']['model'])
        identity = client.identity(models)
        code_hash = digest({p.name: p.read_text() for p in [ROOT / 'src' / name for name in ('experiment.py', 'memories.py', 'run_experiment.py')]})
        manifest = {'version': VERSION, 'config': cfg, 'identity': identity,
            'approach': args.approach, 'pilot_ids': ids, 'code_hash': code_hash,
            'python': platform.python_version()}
        check_manifest(output / 'manifest.json', manifest)
        fingerprint = digest(manifest)
        selected = ids[:args.limit] if args.limit is not None else ids
        for qid in selected:
            path = output / (qid + '.json')
            if path.exists():
                old = json.loads(path.read_text())
                if old['config_hash'] != fingerprint:
                    raise ValueError('Record provenance mismatch')
                if old['status'] == 'ok' or not args.retry_errors:
                    continue
                # Preserve failed attempts before retry.
                atomic_json(output / 'attempts' / f'{qid}-{time.time_ns()}.json', old)
            print(f'{args.approach}: {qid}', flush=True)
            atomic_json(path, {'question_id': qid, 'config_hash': fingerprint, 'status': 'running'})
            record = run_question(by_id[qid], args.approach, cfg, client, identity, fingerprint)
            atomic_json(path, record)
            print(f"  {record['status']} total={record['total_seconds']:.1f}s", flush=True)
            if record['status'] == 'interrupted':
                break
        status = {qid: json.loads((output / (qid + '.json')).read_text())['status']
                  if (output / (qid + '.json')).exists() else 'pending' for qid in ids}
        atomic_json(output / 'coverage.json', {'expected': len(ids), 'status_by_id': status,
            'complete': all(v == 'ok' for v in status.values())})


if __name__ == '__main__':
    main()
