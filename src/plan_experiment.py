"""Select a fixed, disjoint larger sample and estimate its costs; never run it."""
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analyze_pilot import analyze, summary_batches
from experiment import ROOT, atomic_json, digest, history
from run_experiment import load_data


def select_sample(rows, excluded, per_category, seed):
    if per_category < 1:
        raise ValueError('per_category must be positive')
    groups = defaultdict(list)
    for row in rows:
        if row['question_id'] not in excluded:
            category = 'abstention' if row['question_id'].endswith('_abs') else row['question_type']
            groups[category].append(row['question_id'])
    rng = random.Random(seed)
    selected = []
    for category in sorted(groups):
        if len(groups[category]) < per_category:
            raise ValueError(f'Insufficient examples in {category}')
        selected.extend(rng.sample(sorted(groups[category]), per_category))
    return selected


def create_plan(cfg, spec, report):
    by_id, pilot_ids = load_data(cfg)
    ids = select_sample(list(by_id.values()), set(pilot_ids), spec['per_category'], spec['seed'])
    batches = sum(summary_batches(history(by_id[q]), cfg['summary']['batch_chars']) for q in ids)
    observed = report['approaches']['running_summary']['preparation_plan']['seconds_per_update']
    estimates = {}
    for approach in ('raw_history', 'vector_retrieval'):
        metric = report['approaches'][approach]['measurements']['total_seconds']
        estimates[approach] = {'hours_at_pilot_mean': len(ids) * metric['mean'] / 3600
                               if metric['mean'] is not None else None, 'observed_questions': metric['measured_n']}
    estimates['running_summary'] = {
        'preparation_hours_at_observed_mean': batches * observed['mean'] / 3600 if observed['mean'] is not None else None,
        'preparation_hours_at_observed_p90': batches * observed['p90'] / 3600 if observed['p90'] is not None else None,
        'expected_updates': batches}
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'planned_not_run',
            'seed': spec['seed'], 'per_category': spec['per_category'], 'questions': len(ids),
            'pilot_excluded': pilot_ids, 'question_ids': ids, 'sample_hash': digest(ids),
            'dataset_sha256': cfg['dataset_sha256'], 'cost_scenarios': estimates,
            'protocol_status': 'Do not launch until the pilot is completed, grading is resolved, '
                               'and tokenizer/fitting settings are validated and frozen.',
            'caution': 'Balanced exploratory sample; estimates extrapolate few histories and are not guarantees. '
                       'Question IDs are disjoint; session/content independence is not guaranteed. '
                       'A revised protocol needs new configuration, source/cache version, and result directories. '
                       'Official grading requires 3 x questions judgments, plus any review.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, default=ROOT / 'configs/main_sample_plan.json')
    p.add_argument('--output', type=Path, default=ROOT / 'results/main_sample_plan.json')
    args = p.parse_args()
    cfg = json.loads((ROOT / 'configs/pilot.json').read_text())
    plan = create_plan(cfg, json.loads(args.spec.read_text()), analyze(cfg))
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous['sample_hash'] != plan['sample_hash'] or previous['dataset_sha256'] != plan['dataset_sha256']:
            raise ValueError('Refusing to replace an existing selection; choose a new output path')
    atomic_json(args.output, plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'question_ids'}, indent=2))


if __name__ == '__main__':
    main()
