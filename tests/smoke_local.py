"""Real local smoke test; synthetic data is never counted as benchmark evidence."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from experiment import Ollama, atomic_json, digest
from memories import APPROACHES
from run_experiment import run_question
cfg = json.loads(Path('configs/pilot.json').read_text())
row = {'question_id': 'synthetic-smoke', 'question_type': 'knowledge-update',
       'question': 'What city do I live in now?', 'question_date': '2024/01/03 (Wed) 12:00',
       'haystack_sessions': [[{'role': 'user', 'content': 'I live in Rome.'}],
                             [{'role': 'user', 'content': 'I moved to Paris today.'}]],
       'haystack_dates': ['2024/01/01 (Mon) 12:00', '2024/01/02 (Tue) 12:00'],
       'haystack_session_ids': ['s1', 's2']}
client = Ollama(cfg)
identity = client.identity({cfg['answer_model'], cfg['retrieval']['embedding_model']})
for approach in APPROACHES:
    result = run_question(row, approach, cfg, client, identity, digest(cfg))
    atomic_json(Path('results/smoke') / (approach + '.json'), result)
    print(approach, result['status'], result.get('prediction'), flush=True)
    if result['status'] != 'ok' or 'Paris' not in result['prediction']:
        raise RuntimeError(result.get('error', 'Failed simple factual smoke test'))
