"""Export and validate explicit human assessments, separate from official grades."""
import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze_pilot import APPROACHES
from audit_progress import supporting_turns
from compare import records
from experiment import ROOT, atomic_json, digest
from run_experiment import load_data

FAILURE_MODES = {'missing_evidence', 'outdated_information', 'reasoning_or_interpretation',
                 'unsupported_claim', 'inappropriate_abstention', 'missed_abstention',
                 'summary_compression_suspected', 'uncertain'}


def rubric_function(directory=ROOT / 'vendor/longmemeval'):
    """Load only the pinned pure rubric function, without importing API clients."""
    source = directory / 'evaluate_qa.py'
    provenance = json.loads((directory / 'provenance.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance['sha256']['evaluate_qa.py']:
        raise ValueError('Pinned evaluator checksum mismatch')
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'get_anscheck_prompt']
    if len(functions) != 1:
        raise ValueError('Expected one pinned rubric function')
    scope = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    return scope['get_anscheck_prompt'], provenance['revision']


def review_rows(cfg):
    by_id, ids = load_data(cfg)
    rubric, revision = rubric_function()
    result = []
    shared = None
    for approach in APPROACHES:
        manifest, rows = records(ROOT / 'results/pilot' / approach)
        if manifest['pilot_ids'] != ids or manifest['config']['dataset_sha256'] != cfg['dataset_sha256']:
            raise ValueError('Review pilot/dataset mismatch')
        signature = {k: manifest['config'][k] for k in ('answer_model', 'answer_instruction', 'generation',
                     'context_tokens', 'output_tokens', 'template_margin_tokens', 'dataset_sha256')}
        signature.update(model_digest=manifest['identity'][signature['answer_model']],
                         runtime=manifest['identity']['runtime'], code_hash=manifest['code_hash'])
        if shared is not None and shared != signature:
            raise ValueError('Incompatible review settings')
        shared = signature
        for record in rows:
            if record['status'] != 'ok':
                continue
            row = by_id[record['question_id']]
            result.append({'question_id': row['question_id'], 'approach': approach,
                'category': record['category'], 'record_hash': digest(record),
                'rubric_revision': revision, 'question': row['question'],
                'reference_answer': row['answer'], 'prediction': record['prediction'],
                'rubric_prompt': rubric(row['question_type'], row['question'], row['answer'],
                                       record['prediction'], abstention=row['question_id'].endswith('_abs')),
                'supporting_turns': supporting_turns(row, record['prompt']),
                'correct': None, 'reviewer': '', 'review_notes': '', 'failure_modes': []})
    return result, len(ids) * len(APPROACHES)


def validate_reviews(rows, expected):
    key = lambda r: (r['approach'], r['question_id'])
    saved = {key(r): r for r in rows}
    current = {key(r): r for r in expected}
    if len(saved) != len(rows) or set(saved) != set(current):
        raise ValueError('Review IDs must exactly match successful saved predictions, without duplicates')
    editable = {'correct', 'reviewer', 'review_notes', 'failure_modes'}
    for pair, row in saved.items():
        if set(row) != set(current[pair]) or any(row[k] != current[pair][k] for k in row if k not in editable):
            raise ValueError(f'Stale or modified review evidence: {pair}')
        if row['correct'] is not None and type(row['correct']) is not bool:
            raise ValueError('Review correctness must be true, false, or null')
        if (not isinstance(row['failure_modes'], list)
                or any(mode not in FAILURE_MODES for mode in row['failure_modes'])):
            raise ValueError('Unknown failure mode')
        if not all(isinstance(row[k], str) for k in ('reviewer', 'review_notes')):
            raise ValueError('Reviewer and notes must be text')
        if row['correct'] is not None and not all(row[k].strip() for k in ('reviewer', 'review_notes')):
            raise ValueError('Every assessment requires a reviewer and evidence-backed notes')
    return saved


def summarize_reviews(rows, planned):
    assessed = [r for r in rows if r['correct'] is not None]
    return {'method': 'Explicit human assessment using pinned official prompts; not official automated grading',
            'planned_predictions': planned, 'successful_predictions': len(rows),
            'assessed': len(assessed), 'unassessed': len(rows) - len(assessed),
            'reviewers': dict(Counter(r['reviewer'] for r in assessed)),
            'full_comparison_assessed': len(assessed) == planned,
            'manual_correct_fraction': sum(r['correct'] for r in assessed) / planned
                if len(assessed) == planned else None,
            'official_accuracy': None,
            'note': 'Do not treat blank fields or literal evidence checks as assessments. '
                    'An implementing assistant is not an independent human reviewer.'}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export', type=Path, help='Create a NEW worksheet; existing files are protected')
    mode.add_argument('--validate', type=Path, help='Validate completed/partial human assessments')
    p.add_argument('--output', type=Path, default=ROOT / 'results/manual_review_status.json')
    args = p.parse_args()
    cfg = json.loads((ROOT / 'configs/pilot.json').read_text())
    expected, planned = review_rows(cfg)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open('x') as stream:
            for row in expected:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        rows = expected
    else:
        rows = read_rows(args.validate)
        validate_reviews(rows, expected)
    status = summarize_reviews(rows, planned)
    status['worksheet_path'] = str(args.export or args.validate)
    status['worksheet_hash'] = digest(rows)
    status['successful_records_hash'] = digest({f"{r['approach']}:{r['question_id']}": r['record_hash'] for r in rows})
    atomic_json(args.output, status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
