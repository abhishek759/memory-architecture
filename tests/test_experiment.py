import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment import atomic_json, bound, digest, fit, history
from memories import RunningSummary, Retrieval
from run_experiment import check_manifest, run_question

CFG = json.loads(Path('configs/pilot.json').read_text())
ROW = {'question_id': 'test', 'question_type': 'knowledge-update', 'question': 'Where?',
       'question_date': '2024/01/03 (Wed) 12:00', 'answer': 'SECRET', 'answer_session_ids': ['SECRET'],
       'haystack_sessions': [[{'role': 'user', 'content': 'Paris', 'has_answer': True}],
                             [{'role': 'user', 'content': 'Rome'}]],
       'haystack_dates': ['2024/01/02 (Tue) 12:00', '2024/01/01 (Mon) 12:00'],
       'haystack_session_ids': ['new', 'old']}

class Fake:
    def __init__(self):
        self.calls = []
        self.prompts = []
    def generate(self, text, model, limit, phase):
        self.prompts.append(text)
        self.calls.append({'phase': phase, 'input_tokens': None, 'output_tokens': None, 'wall_seconds': 0})
        return {'response': 'Paris', 'done': True}
    def embed(self, texts, phase):
        self.calls.append({'phase': phase})
        return {'embeddings': [[float('Paris' in t), float('Rome' in t), 0.1] for t in texts]}

class Tests(unittest.TestCase):
    def test_allowlist_and_chronology(self):
        sessions = history(ROW)
        self.assertEqual(sessions[0]['session_id'], 'old')
        self.assertNotIn('SECRET', json.dumps(sessions))
        self.assertNotIn('has_answer', json.dumps(sessions))
        bad = copy.deepcopy(ROW)
        bad['haystack_dates'].pop()
        with self.assertRaises(ValueError): history(bad)

    def test_complete_turn_truncation_and_unicode(self):
        text, selection = fit(['old turn', 'é new'], lambda m: 'P' + m, 8)
        self.assertEqual(text, 'Pé new')
        self.assertEqual(selection['dropped_blocks'], 1)
        self.assertLessEqual(bound(text), 8)
        with self.assertRaises(ValueError): fit([], lambda m: 'too long', 2)
        self.assertEqual(fit(['best', 'worst'], lambda m: m, 5, False)[0], 'best')

    def test_resume_manifest_and_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'manifest.json'
            check_manifest(p, {'a': 1})
            check_manifest(p, {'a': 1})
            with self.assertRaises(ValueError): check_manifest(p, {'a': 2})
            self.assertEqual(json.loads(p.read_text()), {'a': 1})

    def test_missing_usage_is_not_zero(self):
        rec = run_question(ROW, 'raw_history', CFG, Fake(), {}, 'hash')
        self.assertEqual(rec['status'], 'ok')
        self.assertIsNone(rec['input_tokens'])
        self.assertEqual(rec['missing_measurements'], ['input_tokens', 'output_tokens'])
        self.assertNotIn('SECRET', rec['prompt'])

    def test_summary_cache_invalidation_and_no_question(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            client = Fake()
            memory = RunningSummary(cfg, client, {'llama3.1:8b': 'digest'})
            _, first = memory.prepare(history(ROW), 'FUTURE SECRET QUESTION')
            self.assertFalse(first['cache_hit'])
            self.assertNotIn('SECRET', ''.join(client.prompts))
            calls = len(client.calls)
            _, second = memory.prepare(history(ROW), 'different question')
            self.assertTrue(second['cache_hit'])
            self.assertEqual(calls, len(client.calls))
            cfg['summary']['output_tokens'] += 1
            _, third = memory.prepare(history(ROW), 'x')
            self.assertNotEqual(first['cache_key'], third['cache_key'])

    def test_real_chroma_history_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            client = Fake()
            memory = Retrieval(cfg, client, {'nomic-embed-text': 'digest'})
            first, a = memory.prepare(history(ROW), 'Paris')
            other = copy.deepcopy(ROW)
            other['haystack_sessions'] = [[{'role': 'user', 'content': 'Tokyo'}], []]
            second, b = memory.prepare(history(other), 'Paris')
            self.assertNotEqual(a['cache_key'], b['cache_key'])
            self.assertIn('Paris', ''.join(first))
            self.assertNotIn('Paris', ''.join(second))
            _, cached = memory.prepare(history(ROW), 'Rome')
            self.assertTrue(cached['cache_hit'])


class FailureTests(unittest.TestCase):
    def test_generation_failure_saved_with_null_measurements(self):
        class Broken(Fake):
            def generate(self, *args):
                raise TimeoutError('simulated timeout')
        rec = run_question(ROW, 'raw_history', CFG, Broken(), {}, 'h')
        self.assertEqual(rec['status'], 'error')
        self.assertIn('simulated timeout', rec['error'])
        self.assertIsNone(rec['input_tokens'])
        self.assertIsNotNone(rec['total_seconds'])

    def test_summary_resumes_after_failed_update(self):
        class Interrupted(Fake):
            def generate(self, *args):
                if len(self.calls) == 1:
                    raise TimeoutError('interrupted after first update')
                return super().generate(*args)
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            identity = {'llama3.1:8b': 'digest'}
            with self.assertRaises(TimeoutError):
                RunningSummary(cfg, Interrupted(), identity).prepare(history(ROW), 'unused')
            client = Fake()
            _, meta = RunningSummary(cfg, client, identity).prepare(history(ROW), 'unused')
            self.assertEqual(meta['resumed_batches'], 1)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(len(meta['construction_cost']), 2)

if __name__ == '__main__': unittest.main()
