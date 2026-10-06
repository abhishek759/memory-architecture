"""Render the comparison JSON as a concise, explicitly provisional Markdown table."""
import argparse
import json
from pathlib import Path


def show(value, digits=2):
    return 'unavailable' if value is None else f'{value:,.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('results/comparison.json'))
    p.add_argument('--output', type=Path, default=Path('results/comparison.md'))
    args = p.parse_args()
    report = json.loads(args.input.read_text())
    lines = ['# Preliminary pilot comparison', '', report['caution'], '',
        'Means below use successful questions only. Unequal coverage is not a paired comparison.', '',
        '| Approach | Success / expected | Accuracy | Prompt tokens | Output tokens | Answer seconds | Preparation seconds | Total seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in report['approaches'].items():
        m = r['measurements_successful_only']
        metrics = [show(m[k]['mean']) for k in ('input_tokens', 'output_tokens', 'answer_seconds', 'preparation_seconds', 'total_seconds')]
        lines.append('| ' + ' | '.join([name, f"{r['successful']} / {r['expected']}",
            'ungraded' if r['accuracy'] is None else f"{r['accuracy']:.1%}", *metrics]) + ' |')
    paired = report.get('paired_successful_questions')
    if paired:
        lines += ['', '## Matched successful questions', '',
            f"All supplied approaches completed the same {paired['n']} questions: " + ', '.join(paired['question_ids']),
            'This subset may cover only some categories and is not full-pilot evidence.', '',
            '| Approach | Prompt tokens | Answer seconds | Preparation seconds | Total seconds |',
            '|---|---:|---:|---:|---:|']
        for name, m in paired['approaches'].items():
            lines.append('| ' + ' | '.join([name, *[show(m[k]['mean']) for k in
                ('input_tokens', 'answer_seconds', 'preparation_seconds', 'total_seconds')]]) + ' |')
    lines += ['', '## Measured preparation calls in current attempts', '',
        '| Approach / phase | Calls | Input tokens (sum) | Output tokens (sum) | Wall seconds (sum) |',
        '|---|---:|---:|---:|---:|']
    for name, r in report['approaches'].items():
        for phase, cost in r['current_attempt_costs'].items():
            if phase == 'answer' or not cost['calls']:
                continue
            lines.append('| ' + ' | '.join([f'{name} / {phase}', str(cost['calls']),
                *[show(cost[k]['sum']) for k in ('input_tokens', 'output_tokens', 'wall_seconds')]]) + ' |')
    lines += ['', '## Coverage and limitations', '']
    for name, r in report['approaches'].items():
        lines.append(f"- {name}: statuses {r['status_counts']}; {len(r['pending_ids'])} pending; {r['cache_hits']} completed cache hits.")
        for qid, error in r['errors'].items():
            lines.append(f'  - {qid}: {error or "unfinished record; a saved running status does not establish an active process"}')
    lines += ['', 'Actual runtime usage is reported, not fitting bounds. Missing fields remain unavailable;',
        'the JSON includes the measured denominator for every aggregate. Preparation totals exclude',
        'cached earlier work; per-question cache metadata retains original construction costs.',
        'An interrupted question can have checkpointed work not yet represented in its final record.',
        '', 'The conservative fitting rule substantially underuses the 32K context ceiling. This is',
        'a pilot of these configured implementations, not a claim about each architecture’s best',
        'possible performance. See `notes/manual_review.md` for qualitative examples.']
    args.output.write_text('\n'.join(lines) + '\n')

if __name__ == '__main__': main()
