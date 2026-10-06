import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment import atomic_json, digest
from compare import records, paired_measurements

CFG = json.loads(Path('configs/pilot.json').read_text())

class ComparisonTests(unittest.TestCase):
    def test_paired_metrics_use_common_successes_and_keep_missing_usage(self):
        result = paired_measurements({
            'raw': [{'question_id': 'one', 'status': 'ok', 'input_tokens': 10},
                    {'question_id': 'two', 'status': 'ok', 'input_tokens': 1000}],
            'summary': [{'question_id': 'one', 'status': 'ok', 'input_tokens': None},
                        {'question_id': 'two', 'status': 'running'}]}, ['one', 'two'])
        self.assertEqual(result['question_ids'], ['one'])
        self.assertEqual(result['approaches']['raw']['input_tokens']['mean'], 10)
        self.assertEqual(result['approaches']['summary']['input_tokens']['measured_n'], 0)
        self.assertIsNone(result['approaches']['summary']['input_tokens']['mean'])

    def test_disjoint_successes_have_no_paired_mean(self):
        result = paired_measurements({
            'raw': [{'question_id': 'one', 'status': 'ok', 'input_tokens': 10}],
            'summary': [{'question_id': 'two', 'status': 'ok', 'input_tokens': 20}]}, ['one', 'two'])
        self.assertEqual(result['n'], 0)
        self.assertIsNone(result['approaches']['raw']['input_tokens']['mean'])

    def make_run(self, root, approach):
        directory = root / approach
        manifest = {'config': CFG, 'pilot_ids': ['one', 'two'], 'approach': approach,
                    'identity': {'llama3.1:8b': 'modeldigest', 'runtime': 'test'}, 'code_hash': 'source'}
        atomic_json(directory / 'manifest.json', manifest)
        for qid in manifest['pilot_ids']:
            atomic_json(directory / (qid + '.json'), {'question_id': qid,
                'config_hash': digest(manifest), 'status': 'ok', 'prediction': 'Paris',
                'category': 'knowledge-update', 'input_tokens': None})
        return directory

    def command(self, directories, output, grades=None):
        command = [sys.executable, 'src/compare.py', *map(str, directories), '--output', str(output)]
        if grades: command += ['--grades', str(grades)]
        return subprocess.run(command, capture_output=True, text=True)

    def test_coverage_no_invented_usage_or_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root, 'raw_history')
            (run / 'two.json').unlink()
            out = root / 'report.json'
            result = self.command([run], out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text())['approaches']['raw_history']
            self.assertEqual(report['pending_ids'], ['two'])
            self.assertIsNone(report['accuracy'])
            self.assertIsNone(report['measurements_successful_only']['input_tokens']['mean'])
            self.assertEqual(len((run / 'hypotheses.jsonl').read_text().splitlines()), 1)

    def test_grade_hypothesis_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.make_run(root, 'raw_history')
            grades = root / 'grades.jsonl'
            grades.write_text('\n'.join(json.dumps({'question_id': qid, 'hypothesis': 'Wrong prediction',
                'autoeval_label': {'model': 'same-judge', 'label': True}}) for qid in ('one', 'two')))
            mapping = root / 'mapping.json'
            atomic_json(mapping, {'raw_history': str(grades)})
            result = self.command([run], root / 'report.json', mapping)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('hypothesis or label mismatch', result.stderr)

    def test_incompatible_answer_settings_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.make_run(root, 'raw_history')
            second = self.make_run(root, 'vector_retrieval')
            manifest = json.loads((second / 'manifest.json').read_text())
            manifest['config']['generation']['temperature'] = 1
            atomic_json(second / 'manifest.json', manifest)
            for qid in ('one', 'two'):
                path = second / (qid + '.json')
                rec = json.loads(path.read_text())
                rec['config_hash'] = digest(manifest)
                atomic_json(path, rec)
            result = self.command([first, second], root / 'report.json')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Incompatible comparison', result.stderr)
