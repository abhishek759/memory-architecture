"""Audit coverage, export official evaluator inputs, and compare measured costs."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from experiment import atomic_json, digest

METRICS = ('input_tokens', 'output_tokens', 'answer_seconds', 'preparation_seconds', 'total_seconds')


def records(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = manifest['pilot_ids']
    if len(expected) != len(set(expected)):
        raise ValueError(f'Duplicate pilot IDs in {directory}')
    rows = []
    for qid in expected:
        path = directory / (qid + '.json')
        if path.exists():
            r = json.loads(path.read_text())
            if r['question_id'] != qid or r['config_hash'] != digest(manifest):
                raise ValueError(f'Invalid record provenance: {path}')
            rows.append(r)
    return manifest, rows


def average(rows, field):
    values = [r.get(field) for r in rows if r.get(field) is not None]
    return {'mean': mean(values) if values else None, 'measured_n': len(values)}


def paired_measurements(runs, expected):
    """Use the same successful questions for every supplied approach."""
    common = set(expected)
    for rows in runs.values():
        common &= {r['question_id'] for r in rows if r['status'] == 'ok'}
    ids = [qid for qid in expected if qid in common]
    return {'question_ids': ids, 'n': len(ids), 'approaches': {
        approach: {field: average([r for r in rows if r['question_id'] in common], field)
                   for field in METRICS}
        for approach, rows in runs.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, default=Path('results/comparison.json'))
    p.add_argument('--grades', type=Path, help='JSON mapping approach to official evaluator result JSONL path')
    args = p.parse_args()
    grade_paths = json.loads(args.grades.read_text()) if args.grades else {}
    report, common, evaluator, run_rows = {}, None, None, {}
    for directory in args.runs:
        manifest, rows = records(directory)
        cfg = manifest['config']
        signature = {k: cfg[k] for k in ('answer_model', 'answer_instruction', 'generation', 'context_tokens', 'output_tokens', 'template_margin_tokens', 'dataset_sha256')}
        signature.update(pilot_ids=manifest['pilot_ids'], answer_digest=manifest['identity'][cfg['answer_model']], runtime=manifest['identity']['runtime'], code_hash=manifest['code_hash'])
        if common is not None and common != signature:
            raise ValueError('Incompatible comparison settings')
        common = signature
        approach = manifest['approach']
        if approach in report:
            raise ValueError('Duplicate approach')
        run_rows[approach] = rows
        ok = [r for r in rows if r['status'] == 'ok']
        hypotheses = [{'question_id': r['question_id'], 'hypothesis': r['prediction']} for r in ok]
        (directory / 'hypotheses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in hypotheses))
        costs = {}
        for phase in ('index_embedding', 'query_embedding', 'summarization', 'answer'):
            calls = [c for r in rows for c in r.get('calls', []) if c['phase'] == phase]
            costs[phase] = {'calls': len(calls), **{k: {'sum': sum(c[k] for c in calls if c.get(k) is not None) if any(c.get(k) is not None for c in calls) else None,
                'measured_n': sum(c.get(k) is not None for c in calls)} for k in ('input_tokens', 'output_tokens', 'wall_seconds')}}
        result = {'expected': len(manifest['pilot_ids']), 'saved': len(rows), 'successful': len(ok),
            'status_counts': dict(Counter(r['status'] for r in rows)),
            'pending_ids': [qid for qid in manifest['pilot_ids'] if qid not in {r['question_id'] for r in rows}],
            'errors': {r['question_id']: r.get('error') for r in rows if r['status'] != 'ok'},
            'accuracy': None, 'grading_status': 'ungraded',
            'measurements_successful_only': {f: average(ok, f) for f in METRICS},
            'current_attempt_costs': costs, 'cache_hits': sum(r.get('memory', {}).get('cache_hit', False) for r in ok)}
        if approach in grade_paths:
            grades = [json.loads(line) for line in Path(grade_paths[approach]).read_text().splitlines() if line]
            mapping = {g['question_id']: g for g in grades}
            if len(mapping) != len(grades) or set(mapping) != {r['question_id'] for r in ok}:
                raise ValueError('Grades must exactly match successful predictions without duplicates')
            for r in ok:
                g = mapping[r['question_id']]
                label = g['autoeval_label']
                if g['hypothesis'] != r['prediction'] or type(label['label']) is not bool:
                    raise ValueError('Grade hypothesis or label mismatch')
                if evaluator is not None and evaluator != label['model']:
                    raise ValueError('Different grading models')
                evaluator = label['model']
            if len(ok) == len(manifest['pilot_ids']):
                result.update(accuracy=mean(g['autoeval_label']['label'] for g in grades), grading_status='graded', evaluator=evaluator,
                    accuracy_by_category={cat: mean(mapping[r['question_id']]['autoeval_label']['label'] for r in ok if r['category'] == cat) for cat in sorted({r['category'] for r in ok})})
            else:
                result['grading_status'] = 'partial: full-pilot accuracy withheld'
        report[approach] = result
    # Comparison accuracy requires a consistent completed grade set for every approach.
    if any(r['grading_status'] != 'graded' for r in report.values()):
        for r in report.values():
            r['accuracy'] = None
            r.pop('accuracy_by_category', None)
    atomic_json(args.output, {'caution': 'Exploratory pilot; ungraded answers are not accuracy results. Cached preparation costs are excluded from current-attempt totals; original costs remain in per-question memory metadata.', 'approaches': report,
        'paired_successful_questions': paired_measurements(run_rows, common['pilot_ids'])})
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
