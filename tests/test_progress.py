import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_pilot import checkpoint_state, evidence_check, summary_batches, summary_plan
from experiment import atomic_json, history
from memories import RunningSummary
from plan_experiment import select_sample
from review_pilot import rubric_function, summarize_reviews, validate_reviews
from test_experiment import CFG, ROW, Fake
from tokenizer_probe import build_tokenizer, check_template, fit_exact, render_generate


class ProgressTests(unittest.TestCase):
    def test_estimated_batches_match_frozen_summary_implementation(self):
        row = copy.deepcopy(ROW)
        row['haystack_sessions'][0][0]['content'] = 'é important dated fact ' * 12
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            cfg['summary']['batch_chars'] = 30
            _, metadata = RunningSummary(cfg, Fake(), {}).prepare(history(row), 'ignored')
            self.assertEqual(summary_batches(history(row), 30), metadata['summary_updates'])

    def test_checkpoint_costs_are_not_added_to_completed_record_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            identity = {'llama3.1:8b': 'digest'}
            _, metadata = RunningSummary(cfg, Fake(), identity).prepare(history(ROW), 'ignored')
            record = {'status': 'ok', 'memory': metadata}
            manifest = {'version': 'pilot-v1', 'config': cfg, 'identity': identity}
            result = summary_plan([ROW], manifest, {'test': record})
            self.assertEqual(result['observed_calls'], 2)
            self.assertEqual(result['remaining_updates'], 0)
            self.assertIsNone(result['checkpointed_update_usage']['input_tokens']['sum'])
            self.assertEqual(result['output_limit_stops'], 0)

    def test_summary_costs_remain_available_without_local_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            manifest = {'version': 'pilot-v1', 'config': cfg, 'identity': {}}
            record = {'status': 'ok', 'memory': {'construction_cost': [
                {'wall_seconds': 10, 'done_reason': 'length'}, {'wall_seconds': 20}]}}
            result = summary_plan([ROW], manifest, {'test': record})
            self.assertEqual(result['observed_calls'], 2)
            self.assertEqual(result['output_limit_stops'], 1)
            self.assertEqual(result['checkpointed_update_usage']['wall_seconds']['sum'], 30)

    def test_bad_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            identity = {}
            RunningSummary(cfg, Fake(), identity).prepare(history(ROW), 'ignored')
            path = next((Path(temp) / 'summary').glob('*.json'))
            state = json.loads(path.read_text())
            state['total_batches'] += 1
            atomic_json(path, state)
            with self.assertRaisesRegex(ValueError, 'Invalid summary checkpoint'):
                checkpoint_state(ROW, {'version': 'pilot-v1', 'config': cfg, 'identity': identity}, {})

    def test_absent_evidence_is_not_a_correctness_judgment(self):
        result = evidence_check(ROW, {'prompt': 'A paraphrase of the reference'})
        self.assertEqual(result['full_annotated_turns_in_prompt'], 0)
        self.assertNotIn('correct', result)

    def test_sample_is_balanced_disjoint_deterministic_and_order_independent(self):
        rows = [{'question_id': f'{cat}{i}', 'question_type': cat}
                for cat in ('a', 'b') for i in range(8)]
        excluded = {'a0', 'b0'}
        selected = select_sample(rows, excluded, 3, 42)
        self.assertEqual(selected, select_sample(rows[::-1], excluded, 3, 42))
        self.assertEqual(len(set(selected)), 6)
        self.assertFalse(set(selected) & excluded)
        self.assertEqual(sum(q.startswith('a') for q in selected), 3)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_sample(rows, excluded, 8, 42)

    def test_pinned_rubric_has_category_and_abstention_rules(self):
        rubric, _ = rubric_function()
        self.assertIn('unanswerable', rubric('single-session-user', 'q', 'a', 'p', True))
        self.assertIn('off-by-one', rubric('temporal-reasoning', 'q', 'a', 'p'))
        self.assertIn('personal information', rubric('single-session-preference', 'q', 'a', 'p'))
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'evaluate_qa.py').write_text('tampered')
            atomic_json(directory / 'provenance.json', {'sha256': {'evaluate_qa.py': 'wrong'}})
            with self.assertRaisesRegex(ValueError, 'checksum'):
                rubric_function(directory)

    def review(self):
        return {'approach': 'raw_history', 'question_id': 'one', 'record_hash': 'hash',
                'prediction': 'Paris', 'correct': None, 'reviewer': '', 'review_notes': '', 'failure_modes': []}

    def test_blank_review_never_becomes_a_grade(self):
        rows = [self.review()]
        validate_reviews(rows, copy.deepcopy(rows))
        summary = summarize_reviews(rows, 42)
        self.assertEqual(summary['assessed'], 0)
        self.assertIsNone(summary['manual_correct_fraction'])
        self.assertIsNone(summary['official_accuracy'])

    def test_stale_duplicate_and_unattributed_reviews_rejected(self):
        original = self.review()
        row = copy.deepcopy(original)
        row['prediction'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'modified'):
            validate_reviews([row], [original])
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            validate_reviews([original, original], [original])
        row = copy.deepcopy(original)
        row['correct'] = True
        with self.assertRaisesRegex(ValueError, 'reviewer'):
            validate_reviews([row], [original])
        row.update(reviewer='Human reviewer', review_notes='The answer matches the dated update.')
        validate_reviews([row], [original])
        self.assertEqual(summarize_reviews([row], 1)['manual_correct_fraction'], 1)
        row['correct'] = 1
        with self.assertRaisesRegex(ValueError, 'true, false, or null'):
            validate_reviews([row], [original])

    def test_local_fitter_preserves_whole_recent_turns_and_bounds(self):
        text, selected = fit_exact(['older fact', 'new fact'], lambda memory: 'P' + memory, len, 10)
        self.assertEqual(text, 'Pnew fact')
        self.assertEqual(selected['retained_blocks'], 1)
        self.assertLessEqual(selected['offline_prompt_tokens'], 10)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            fit_exact([], lambda memory: 'large instruction', len, 2)

    def test_tokenizer_reconstruction_handles_unicode_and_preserves_special_ids(self):
        from tokenizers.pre_tokenizers import ByteLevel
        vocabulary = sorted(ByteLevel.alphabet()) + ['<|begin_of_text|>']
        info = {'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'llama-bpe',
                'tokenizer.ggml.tokens': vocabulary, 'tokenizer.ggml.merges': [],
                'tokenizer.ggml.token_type': [1] * 256 + [3]}
        tokenizer = build_tokenizer(info)
        self.assertEqual(len(tokenizer.encode('é🧠', add_special_tokens=False).ids), len('é🧠'.encode()))
        self.assertEqual(tokenizer.encode('<|begin_of_text|>', add_special_tokens=False).ids, [256])
        info['tokenizer.ggml.token_type'].pop()
        with self.assertRaisesRegex(ValueError, 'alignment'):
            build_tokenizer(info)

    def test_unrecognized_template_and_tokenizer_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported runtime template'):
            check_template({'template': 'changed runtime template'})
        with self.assertRaisesRegex(ValueError, 'Unsupported tokenizer'):
            build_tokenizer({'tokenizer.ggml.model': 'other'})
        self.assertTrue(render_generate('Question').startswith('<|begin_of_text|>'))


if __name__ == '__main__':
    unittest.main()
