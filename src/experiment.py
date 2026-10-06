"""Shared local inference, provenance, history formatting, and durable storage."""
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'pilot-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def history(row):
    """Allowlist inference fields; never pass annotations or answers to memories."""
    arrays = [row[k] for k in ('haystack_sessions', 'haystack_dates', 'haystack_session_ids')]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError('Misaligned history metadata')
    sessions = []
    for turns, date, sid in zip(*arrays):
        sessions.append({'date': date, 'session_id': sid,
                         'turns': [{'role': t['role'], 'content': t['content']} for t in turns]})
    def date_key(s):
        return datetime.strptime(s['date'], '%Y/%m/%d (%a) %H:%M')
    return sorted(sessions, key=date_key)


def blocks(sessions):
    return [f"[{s['date']} | session {s['session_id']} | {t['role']}] {t['content']}"
            for s in sessions for t in s['turns']]


def bound(text):
    # Byte-level BPE cannot use more text tokens than UTF-8 bytes. Reserve the
    # configured template margin separately. This is a bound, NOT measured usage.
    return len(text.encode('utf-8'))


def prompt(cfg, memory, question, date):
    return cfg['answer_instruction'].format(memory=memory, question=question, question_date=date)


def fit(parts, make_prompt, budget, drop_oldest=True):
    parts = list(parts)
    original = len(parts)
    while parts and bound(make_prompt('\n'.join(parts))) > budget:
        parts.pop(0 if drop_oldest else -1)
    text = make_prompt('\n'.join(parts))
    if bound(text) > budget:
        raise ValueError('Question/instructions exceed input budget')
    return text, {'original_blocks': original, 'retained_blocks': len(parts),
                  'dropped_blocks': original - len(parts), 'truncated': original != len(parts),
                  'prompt_token_upper_bound': bound(text), 'fitting_method': 'utf8_byte_bound'}


class Ollama:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []

    def request(self, route, payload=None):
        req = urllib.request.Request(self.cfg['base_url'] + route,
              data=None if payload is None else json.dumps(payload).encode(),
              headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.cfg['timeout_seconds']) as r:
            result = json.load(r)
        if result.get('error'):
            raise RuntimeError(result['error'])
        return result

    def identity(self, models):
        tags = self.request('/api/tags')['models']
        result = {'runtime': self.request('/api/version')['version']}
        for model in models:
            found = [m for m in tags if m['name'] in (model, model + ':latest')]
            if not found:
                raise RuntimeError(f'Model not installed: {model}')
            result[model] = found[0]['digest']
        return result

    def generate(self, text, model, limit, phase):
        return self._call('/api/generate', {'model': model, 'prompt': text, 'stream': False,
            'keep_alive': self.cfg['keep_alive'], 'options': {
                **self.cfg['generation'], 'num_ctx': self.cfg['context_tokens'], 'num_predict': limit}}, phase)

    def embed(self, texts, phase):
        return self._call('/api/embed', {'model': self.cfg['retrieval']['embedding_model'],
            'input': texts, 'truncate': False, 'keep_alive': self.cfg['keep_alive']}, phase)

    def _call(self, route, payload, phase):
        started = time.perf_counter()
        record = {'phase': phase, 'model': payload['model'], 'input_tokens': None,
                  'output_tokens': None, 'status': 'error'}
        try:
            result = self.request(route, payload)
            record.update(status='ok', input_tokens=result.get('prompt_eval_count'),
                output_tokens=result.get('eval_count'), done_reason=result.get('done_reason'),
                prompt_eval_seconds=ns(result.get('prompt_eval_duration')),
                generation_seconds=ns(result.get('eval_duration')),
                load_seconds=ns(result.get('load_duration')))
            if route == '/api/generate' and (not result.get('done') or not result.get('response', '').strip()):
                raise RuntimeError('Incomplete or empty generation')
            return result
        except Exception as exc:
            record.update(status='error', error=str(exc))
            raise
        finally:
            record['wall_seconds'] = time.perf_counter() - started
            self.calls.append(record)


def ns(value):
    return None if value is None else value / 1e9
