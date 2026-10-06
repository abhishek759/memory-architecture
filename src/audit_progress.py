"""Export a read-only midpoint audit and review worksheet; never perform inference."""
import argparse
import fcntl
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from compare import records
from experiment import ROOT, atomic_json, digest, history
from run_experiment import load_data


def writer_active(directory):
    """Sample the runner's advisory lock; this is not a persistent health check."""
    path = directory / '.lock'
    if not path.exists():
        return False
    with path.open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def summary_checkpoint(row, manifest):
    # Reproduce pilot-v1's existing key without changing its inference sources.
    cfg = manifest['config']
    key = digest([manifest['version'], history(row), cfg['summary'], cfg['generation'],
                  cfg['context_tokens'], cfg['template_margin_tokens'], manifest['identity']])
    path = Path(cfg['cache_dir']) / 'summary' / (key + '.json')
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    calls = state['calls']
    result = {'cache_key': key, 'completed_updates': state['next_batch'],
              'total_updates': state['total_batches'], 'calls': len(calls)}
    if not 0 <= result['completed_updates'] <= result['total_updates'] or len(calls) != result['completed_updates']:
        raise ValueError(f'Invalid summary checkpoint: {path}')
    for field in ('input_tokens', 'output_tokens', 'wall_seconds'):
        values = [c[field] for c in calls if c.get(field) is not None]
        result[field] = {'sum': sum(values) if values else None, 'measured_n': len(values)}
    return result


def supporting_turns(row, prompt):
    """Literal checks assist review; paraphrases and fragments require human review."""
    result = []
    for session, date, sid in zip(row['haystack_sessions'], row['haystack_dates'], row['haystack_session_ids']):
        for index, turn in enumerate(session):
            if turn.get('has_answer'):
                result.append({'session_id': sid, 'date': date, 'turn': index,
                    'role': turn['role'], 'content': turn['content'],
                    'full_text_in_prompt': bool(turn['content']) and turn['content'] in prompt})
    return result


def audit_run(directory, by_id, expected, source_hash):
    manifest, rows = records(directory)
    if manifest['pilot_ids'] != expected:
        raise ValueError(f'Pilot selection mismatch: {directory}')
    if manifest['code_hash'] != source_hash:
        raise ValueError(f'Inference source differs from saved run: {directory}')
    active = writer_active(directory)
    saved = {r['question_id']: r for r in rows}
    statuses, checkpoints, worksheet = {}, {}, []
    for qid in expected:
        row = by_id[qid]
        record = saved.get(qid, {})
        status = record.get('status', 'pending')
        statuses[qid] = 'unfinished' if status == 'running' and not active else status
        if status == 'ok' and (not record.get('prediction') or not record.get('prompt')):
            raise ValueError(f'Success record lacks prediction/prompt: {qid}')
        if manifest['approach'] == 'running_summary':
            checkpoint = summary_checkpoint(row, manifest)
            if checkpoint is not None:
                checkpoints[qid] = checkpoint
        worksheet.append({'question_id': qid, 'approach': manifest['approach'],
            'category': 'abstention' if qid.endswith('_abs') else row['question_type'],
            'status': statuses[qid], 'question': row['question'],
            'question_date': row['question_date'], 'reference_answer': row['answer'],
            'prediction': record.get('prediction'),
            'record_path': str(directory / (qid + '.json')),
            'supporting_turns': supporting_turns(row, record.get('prompt', '')),
            'assessment': None, 'failure_modes': [], 'reviewer': None, 'review_notes': ''})
    return {'manifest_verified': True, 'inference_source_matches': True,
        'writer_active_at_check': active, 'status_by_id': statuses,
        'status_counts': dict(Counter(statuses.values())),
        'successful': sum(s == 'ok' for s in statuses.values()), 'expected': len(expected),
        'summary_checkpoints': checkpoints}, worksheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot.json')
    parser.add_argument('--runs', nargs='+', type=Path, default=[ROOT / 'results/pilot' / a
        for a in ('raw_history', 'vector_retrieval', 'running_summary')])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/midpoint/audit.json')
    parser.add_argument('--review-output', type=Path,
                        help='New JSONL worksheet path; refuses to overwrite an existing review')
    args = parser.parse_args()
    if args.review_output and args.review_output.exists():
        raise ValueError('Review output already exists; choose a new path to preserve annotations')
    cfg = json.loads((ROOT / args.config).read_text())
    by_id, ids = load_data(cfg)
    for row in by_id.values():
        history(row)
    source_hash = digest({name: (ROOT / 'src' / name).read_text()
                         for name in ('experiment.py', 'memories.py', 'run_experiment.py')})
    runs, reviews = {}, []
    for directory in args.runs:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['config']['dataset_sha256'] != cfg['dataset_sha256']:
            raise ValueError(f'Dataset provenance mismatch: {directory}')
        if manifest['approach'] in runs:
            raise ValueError('Duplicate approach')
        runs[manifest['approach']], worksheet = audit_run(directory, by_id, ids, source_hash)
        reviews.extend(worksheet)
    atomic_json(args.output, {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'reporting_period': 'Weeks 1–7', 'dataset_questions': len(by_id),
        'dataset_checksum_verified': True, 'history_metadata_verified': True,
        'pilot_questions': len(ids), 'inference_source_hash': source_hash,
        'note': 'Checkpoint costs overlap completed-record costs; do not add them together. '
                'Lock state is a point-in-time observation. Literal evidence matching is not grading.',
        'approaches': runs})
    if args.review_output:
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        with args.review_output.open('x') as stream:
            for row in reviews:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({a: r['status_counts'] for a, r in runs.items()}, indent=2))


if __name__ == '__main__':
    main()
