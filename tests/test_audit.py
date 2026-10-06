import copy
import fcntl
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_progress import audit_run, summary_checkpoint, supporting_turns, writer_active
from experiment import atomic_json, digest
from memories import RunningSummary
from test_experiment import CFG, ROW, Fake
from experiment import history


class AuditTests(unittest.TestCase):
    def test_running_record_without_lock_is_unfinished_and_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = {'pilot_ids': ['test'], 'code_hash': 'source', 'approach': 'raw_history'}
            atomic_json(directory / 'manifest.json', manifest)
            record = {'question_id': 'test', 'config_hash': digest(manifest), 'status': 'running'}
            path = directory / 'test.json'
            atomic_json(path, record)
            audit, review = audit_run(directory, {'test': ROW}, ['test'], 'source')
            self.assertEqual(audit['status_counts'], {'unfinished': 1})
            self.assertIsNone(review[0]['assessment'])
            self.assertEqual(json.loads(path.read_text()), record)
            with self.assertRaisesRegex(ValueError, 'Inference source'):
                audit_run(directory, {'test': ROW}, ['test'], 'changed-source')

    def test_held_runner_lock_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with (directory / '.lock').open('w') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(writer_active(directory))
                fcntl.flock(stream, fcntl.LOCK_UN)
                self.assertFalse(writer_active(directory))

    def test_checkpoint_matches_actual_memory_key_and_excludes_other_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(CFG)
            cfg['cache_dir'] = temp
            identity = {'llama3.1:8b': 'digest'}
            RunningSummary(cfg, Fake(), identity).prepare(history(ROW), 'ignored')
            manifest = {'version': 'pilot-v1', 'config': cfg, 'identity': identity}
            checkpoint = summary_checkpoint(ROW, manifest)
            self.assertEqual(checkpoint['completed_updates'], 2)
            self.assertEqual(checkpoint['total_updates'], 2)
            self.assertIsNone(checkpoint['input_tokens']['sum'])
            cfg['summary']['output_tokens'] += 1
            self.assertIsNone(summary_checkpoint(ROW, manifest))

    def test_literal_evidence_matching_does_not_count_session_alone(self):
        evidence = supporting_turns(ROW, 'session new without the supporting content')
        self.assertFalse(evidence[0]['full_text_in_prompt'])
        self.assertTrue(supporting_turns(ROW, 'Paris')[0]['full_text_in_prompt'])
