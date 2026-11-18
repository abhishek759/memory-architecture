"""Offline evidence diagnostics and compute planning; this never grades answers."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from audit_progress import audit_run, supporting_turns
from compare import average, paired_measurements, records
from experiment import ROOT, atomic_json, blocks, digest, history
from run_experiment import load_data

APPROACHES = ('raw_history', 'vector_retrieval', 'running_summary')
METRICS = ('input_tokens', 'output_tokens', 'answer_seconds', 'preparation_seconds', 'total_seconds')


def summary_batches(sessions, batch_chars):
    """Reproduce frozen pilot-v1 segmentation without constructing any memories."""
    batches = []
    for session in sessions:
        for text in blocks([session]):
            for start in range(0, len(text), batch_chars):
                part = text[start:start + batch_chars]
                if (batches and batches[-1][0] == session['session_id']
                        and len(batches[-1][1]) + len(part) + 1 <= batch_chars):
                    batches[-1][1] += '\n' + part
                else:
                    batches.append([session['session_id'], part])
    return len(batches)


def usage(calls):
    return {field: {'sum': sum(values) if values else None, 'measured_n': len(values)}
            for field in ('input_tokens', 'output_tokens', 'wall_seconds')
            for values in [[c[field] for c in calls if c.get(field) is not None]]}


def checkpoint_state(row, manifest, record):
    cfg = manifest['config']
    key = digest([manifest['version'], history(row), cfg['summary'], cfg['generation'],
                  cfg['context_tokens'], cfg['template_margin_tokens'], manifest['identity']])
    path = Path(cfg['cache_dir']) / 'summary' / (key + '.json')
    expected = summary_batches(history(row), cfg['summary']['batch_chars'])
    if path.exists():
        state = json.loads(path.read_text())
        count, calls = state['next_batch'], state['calls']
        if state['total_batches'] != expected or count != len(calls) or not 0 <= count <= expected:
            raise ValueError(f'Invalid summary checkpoint: {path}')
        return count, calls, 'checkpoint'
    if record.get('status') == 'ok':
        calls = record.get('memory', {}).get('construction_cost')
        if calls is not None and len(calls) == expected:
            return expected, calls, 'completed_record'
        # Do not pretend successful answering establishes recoverable preparation usage.
        return expected, [], 'successful_record_without_preparation_ledger'
    return 0, [], 'unavailable'


def summary_plan(rows, manifest, saved):
    details, calls = {}, []
    cfg = manifest['config']
    for row in rows:
        qid = row['question_id']
        expected = summary_batches(history(row), cfg['summary']['batch_chars'])
        completed, observed, source = checkpoint_state(row, manifest, saved.get(qid, {}))
        details[qid] = {'expected_updates': expected, 'completed_updates': completed,
                        'remaining_updates': expected - completed, 'cost_source': source,
                        'checkpointed_update_usage': usage(observed)}
        calls.extend(observed)
    remaining = sum(d['remaining_updates'] for d in details.values())
    seconds = [c['wall_seconds'] for c in calls if c.get('wall_seconds') is not None]
    p90 = sorted(seconds)[min(len(seconds) - 1, int(.9 * len(seconds)))] if seconds else None
    return {'histories': details, 'expected_updates': sum(d['expected_updates'] for d in details.values()),
            'completed_updates': sum(d['completed_updates'] for d in details.values()),
            'remaining_updates': remaining, 'observed_calls': len(calls),
            'output_limit_stops': sum(c.get('done_reason') == 'length' for c in calls),
            'character_clipping_events': sum(c.get('clipped_chars', 0) > 0 for c in calls),
            'checkpointed_update_usage': usage(calls),
            'seconds_per_update': {'mean': mean(seconds) if seconds else None,
                                   'median': median(seconds) if seconds else None, 'p90': p90},
            'remaining_preparation_hours_at_observed_mean': remaining * mean(seconds) / 3600 if seconds else None,
            'remaining_preparation_hours_at_observed_p90': remaining * p90 / 3600 if seconds else None,
            'note': 'Checkpoint and completed-record costs are deduplicated by history. '
                    'These completed-update costs overlap record costs; never add the two. '
                    'Failed or uncheckpointed calls may be absent. Estimates are scenarios, not confidence intervals.'}


def evidence_check(row, record):
    evidence = supporting_turns(row, record.get('prompt', ''))
    return {'annotated_turns': len(evidence),
            'full_annotated_turns_in_prompt': sum(t['full_text_in_prompt'] for t in evidence),
            'note': 'Literal text checks do not detect paraphrases, partial fragments, or prove answer correctness.'}


def analyze(cfg, baseline=None):
    by_id, ids = load_data(cfg)
    for row in by_id.values():
        history(row)
    source_hash = digest({name: (ROOT / 'src' / name).read_text()
                          for name in ('experiment.py', 'memories.py', 'run_experiment.py')})
    reports, runs, shared = {}, {}, None
    for approach in APPROACHES:
        directory = ROOT / 'results/pilot' / approach
        manifest, rows = records(directory)
        if manifest['approach'] != approach or manifest['config']['dataset_sha256'] != cfg['dataset_sha256']:
            raise ValueError('Run/dataset identity mismatch')
        current = {k: manifest['config'][k] for k in ('answer_model', 'answer_instruction', 'generation',
                   'context_tokens', 'output_tokens', 'template_margin_tokens', 'dataset_sha256')}
        current.update(identity={k: manifest['identity'][k] for k in ('runtime', current['answer_model'])},
                       pilot_ids=manifest['pilot_ids'], code_hash=manifest['code_hash'])
        if shared is not None and shared != current:
            raise ValueError('Incompatible shared settings')
        shared = current
        audit, _ = audit_run(directory, by_id, ids, source_hash)
        ok = [r for r in rows if r['status'] == 'ok']
        runs[approach] = rows
        details = {}
        for record in ok:
            selection = record['selection']
            details[record['question_id']] = {
                'record_hash': digest(record), 'category': record['category'],
                'prompt_context_fraction': record['input_tokens'] / manifest['config']['context_tokens']
                    if record.get('input_tokens') is not None else None,
                'retained_block_fraction': selection['retained_blocks'] / selection['original_blocks']
                    if selection['original_blocks'] else None,
                'truncated': selection['truncated'],
                'evidence': evidence_check(by_id[record['question_id']], record)}
        reports[approach] = {'expected': len(ids), 'successful': len(ok),
            'status_counts': audit['status_counts'], 'writer_active_at_check': audit['writer_active_at_check'],
            'successful_by_category': dict(Counter(r['category'] for r in ok)),
            'accuracy': None, 'grading_status': 'ungraded',
            'measurements': {field: average(ok, field) for field in METRICS},
            'missing_usage_ids': [r['question_id'] for r in ok if any(r.get(k) is None for k in ('input_tokens', 'output_tokens'))],
            'questions': details}
        if baseline:
            reports[approach]['new_successes_since_baseline'] = len(ok) - baseline['approaches'][approach]['successful']
        if approach == 'running_summary':
            reports[approach]['preparation_plan'] = summary_plan([by_id[q] for q in ids], manifest,
                                                               {r['question_id']: r for r in rows})
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset_questions': len(by_id), 'dataset_checksum_verified': True,
            'inference_source_hash': source_hash, 'pilot_ids': ids, 'approaches': reports,
            'paired_successful_questions': paired_measurements(runs, ids),
            'note': 'Post-inference diagnostics only. Reference annotations are never used to build memories. '
                    'Evidence presence and output-limit stops are diagnostics, not grades or proven failure causes.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/pilot.json')
    p.add_argument('--baseline', type=Path, help='Optional prior analysis JSON')
    p.add_argument('--output', type=Path, default=ROOT / 'results/analysis.json')
    args = p.parse_args()
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    report = analyze(json.loads(args.config.read_text()), baseline)
    atomic_json(args.output, report)
    for name, result in report['approaches'].items():
        print(name, result['status_counts'])
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
