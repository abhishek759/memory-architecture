import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment import blocks, digest, history
from fitting_experiment import (FROZEN_SOURCES, execute_variant, existing_record,
                                fit_pair, safe_output, validate_measurement,
                                verify_protocol)


CFG = {'context_tokens': 100, 'output_tokens': 16, 'template_margin_tokens': 8,
       'answer_instruction': 'Memory:{memory}\nQuestion:{question}', 'answer_model': 'fake'}
ROW = {'question_id': 'q', 'question_type': 'single-session-user', 'question': 'where?',
       'question_date': '2026/01/03 (Sat) 00:00', 'answer': 'SECRET_REFERENCE',
       'haystack_sessions': [[{'role': 'user', 'content': 'old dated fact', 'has_answer': True}],
                             [{'role': 'user', 'content': 'new dated fact', 'has_answer': False}]],
       'haystack_dates': ['2026/01/01 (Thu) 00:00', '2026/01/02 (Fri) 00:00'],
       'haystack_session_ids': ['old', 'new']}


class FittingExperimentTests(unittest.TestCase):
    def test_paired_fits_keep_complete_recent_turns_and_exclude_labels(self):
        counter = lambda text: (len(text.encode()) + 3) // 4 + 2
        pair = fit_pair(ROW, CFG, counter)
        original = blocks(history(ROW))
        self.assertGreater(pair['exact_token_count']['selection']['retained_blocks'],
                           pair['legacy_byte_bound']['selection']['retained_blocks'])
        for entry in pair.values():
            kept = entry['selection']['retained_blocks']
            expected_memory = '\n'.join(original[-kept:]) if kept else ''
            self.assertEqual(entry['prompt'], CFG['answer_instruction'].format(
                memory=expected_memory, question=ROW['question']))
            self.assertNotIn('SECRET_REFERENCE', entry['prompt'])
            self.assertNotIn('has_answer', entry['prompt'])
            self.assertLessEqual(entry['local_prompt_tokens'] + CFG['output_tokens'], CFG['context_tokens'])

    def test_oversized_question_cannot_silently_truncate_instructions(self):
        row = copy.deepcopy(ROW)
        row['question'] = 'X' * 1000
        with self.assertRaisesRegex(ValueError, 'exceed'):
            fit_pair(row, CFG, len)

    def test_runtime_counter_mismatch_or_missing_usage_prevents_success(self):
        fitted = {'prompt': 'memory', 'prompt_hash': digest('memory'), 'local_prompt_tokens': 10}
        class Client:
            calls = []
            def generate(self, *args):
                return {'done': True, 'response': 'answer', 'prompt_eval_count': 11, 'eval_count': 1}
        record = execute_variant(ROW, 'exact_token_count', fitted, CFG, Client(), 'manifest', ['exact_token_count'])
        self.assertEqual(record['status'], 'error')
        self.assertFalse(record['runtime_count_matches'])
        self.assertIn('does not match', record['error'])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_measurement({'done': True, 'response': 'answer'}, 10, CFG)

    def test_stale_manifest_and_modified_prompt_are_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'record.json'
            entry = {'prompt': 'original', 'prompt_hash': digest('original')}
            row = {**entry, 'manifest_hash': 'one', 'status': 'ok'}
            path.write_text(json.dumps(row))
            self.assertEqual(existing_record(path, 'one', entry), row)
            with self.assertRaisesRegex(ValueError, 'Incompatible'):
                existing_record(path, 'two', entry)
            row['prompt'] = 'tampered'
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'stale'):
                existing_record(path, 'one', entry)

    def test_output_guard_protects_prior_results_and_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prior, allowed = root / 'pilot', root / 'progress_report_4'
            prior.mkdir()
            allowed.mkdir()
            protected = prior / 'answer.json'
            protected.write_text('original evidence')
            (allowed / 'escape').symlink_to(prior, target_is_directory=True)
            for target in (prior, allowed, allowed / 'escape' / 'new'):
                with self.assertRaisesRegex(ValueError, 'separate subdirectory'):
                    safe_output(target, allowed)
            self.assertEqual(safe_output(allowed / 'ablation', allowed), (allowed / 'ablation').resolve())
            self.assertEqual(protected.read_text(), 'original evidence')

    def test_changed_frozen_source_invalidates_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'src').mkdir()
            (root / 'configs').mkdir()
            cfg = {'cache_dir': 'cache'}
            base = root / 'configs/pilot.json'
            base.write_text(json.dumps(cfg))
            for name in FROZEN_SOURCES:
                (root / 'src' / name).write_text('# original ' + name)
            source_hash = digest({name: (root / 'src' / name).read_text() for name in FROZEN_SOURCES})
            identity = {'runtime': 'test', 'fake': 'digest'}
            (root / 'manifest.json').write_text(json.dumps({'config': {**cfg, 'cache_dir': str(root / 'cache')},
                'code_hash': source_hash, 'identity': identity}))
            protocol = {'base_config': 'configs/pilot.json',
                'base_config_sha256': hashlib.sha256(base.read_bytes()).hexdigest(),
                'expected_inference_source_hash': source_hash, 'pilot_manifest': 'manifest.json',
                'expected_identity': identity, 'variants': ['legacy_byte_bound', 'exact_token_count']}
            with patch('fitting_experiment.ROOT', root):
                self.assertEqual(verify_protocol(protocol, cfg), source_hash)
                (root / 'src/experiment.py').write_text('# changed')
                with self.assertRaisesRegex(ValueError, 'Frozen pilot inference source'):
                    verify_protocol(protocol, cfg)


if __name__ == '__main__':
    unittest.main()
