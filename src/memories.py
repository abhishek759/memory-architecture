"""Three memories share prepare(sessions, question) -> text blocks + metadata."""
import json
import time
from pathlib import Path
from experiment import VERSION, atomic_json, blocks, bound, digest


class RawHistory:
    drop_oldest = True
    def __init__(self, cfg, client, identity):
        pass

    def prepare(self, sessions, question):
        return blocks(sessions), {'cache_hit': False}


class Retrieval:
    drop_oldest = False  # discard least similar chunks first
    def __init__(self, cfg, client, identity):
        self.cfg, self.client, self.identity = cfg, client, identity

    def prepare(self, sessions, question):
        import chromadb
        from chromadb.config import Settings
        cfg = self.cfg['retrieval']
        key = digest([VERSION, sessions, cfg, self.identity[cfg['embedding_model']]])
        root = Path(self.cfg['cache_dir']) / 'retrieval' / key
        db = chromadb.PersistentClient(path=str(root), settings=Settings(anonymized_telemetry=False))
        collection = db.get_or_create_collection('history', metadata={'hnsw:space': 'cosine'}, embedding_function=None)
        chunks, metadata = [], []
        for s in sessions:
            for ti, turn in enumerate(s['turns']):
                # Bound chunks by bytes, retain metadata on each fragment.
                content = turn['content']
                fragments, fragment = [], ''
                for char in content:
                    if bound(fragment + char) > cfg['chunk_bytes']:
                        fragments.append(fragment)
                        fragment = ''
                    fragment += char
                fragments.append(fragment)
                for fi, fragment in enumerate(fragments):
                    chunks.append(f"[{s['date']} | session {s['session_id']} | {turn['role']} | turn {ti} fragment {fi}] {fragment}")
                    metadata.append({'session_id': s['session_id'], 'date': s['date'], 'role': turn['role'], 'turn': ti, 'fragment': fi})
        marker = root / 'complete.json'
        hit = marker.exists() and collection.count() == len(chunks)
        start = len(self.client.calls)
        if not hit:
            for i in range(0, len(chunks), cfg['batch_size']):
                batch = chunks[i:i + cfg['batch_size']]
                embedded = self.client.embed(['search_document: ' + t for t in batch], 'index_embedding')
                collection.upsert(ids=[str(j) for j in range(i, i + len(batch))], documents=batch,
                    metadatas=metadata[i:i + len(batch)], embeddings=embedded['embeddings'])
            atomic_json(marker, {'calls': self.client.calls[start:], 'chunks': len(chunks)})
        if not chunks:
            return [], {'cache_hit': hit, 'chunks': 0}
        embedded = self.client.embed(['search_query: ' + question], 'query_embedding')
        result = collection.query(query_embeddings=embedded['embeddings'],
            n_results=min(cfg['top_k'], len(chunks)), include=['documents', 'metadatas', 'distances'])
        return result['documents'][0], {'cache_hit': hit, 'cache_key': key, 'chunks': len(chunks),
            'retrieved': [{'id': i, 'distance': d, **m} for i, d, m in zip(result['ids'][0], result['distances'][0], result['metadatas'][0])],
            'construction_cost': json.loads(marker.read_text())}


class RunningSummary:
    drop_oldest = True
    def __init__(self, cfg, client, identity):
        self.cfg, self.client, self.identity = cfg, client, identity

    def prepare(self, sessions, question):
        # Deliberately ignore question: only allowlisted conversation is summarized.
        cfg = self.cfg['summary']
        key = digest([VERSION, sessions, cfg, self.cfg['generation'], self.cfg['context_tokens'],
                      self.cfg['template_margin_tokens'], self.identity])
        path = Path(self.cfg['cache_dir']) / 'summary' / (key + '.json')
        state = json.loads(path.read_text()) if path.exists() else {'next_batch': 0, 'summary': '', 'calls': []}
        batches = []
        # Each session is updated separately; oversized turns are split losslessly.
        for s in sessions:
            for text in blocks([s]):
                # Chunk without dropping any history; UTF-8 bound checked again below.
                for start in range(0, len(text), cfg['batch_chars']):
                    part = text[start:start + cfg['batch_chars']]
                    if batches and batches[-1][0] == s['session_id'] and len(batches[-1][1]) + len(part) + 1 <= cfg['batch_chars']:
                        batches[-1][1] += '\n' + part
                    else:
                        batches.append([s['session_id'], part])
        initial = state['next_batch']
        for i in range(initial, len(batches)):
            text = cfg['instruction'].format(summary=state['summary'], session=batches[i][1])
            budget = self.cfg['context_tokens'] - cfg['output_tokens'] - self.cfg['template_margin_tokens']
            if bound(text) > budget:
                raise ValueError('Summary update exceeds safe input bound; reduce batch_chars/summary max_chars')
            result = self.client.generate(text, cfg['model'], cfg['output_tokens'], 'summarization')
            summary = result['response'].strip()
            # Explicit bounded memory; log any clipping rather than hiding it.
            state.update(summary=summary[:cfg['max_chars']], next_batch=i + 1, total_batches=len(batches))
            state['calls'].append({**self.client.calls[-1], 'clipped_chars': max(0, len(summary) - cfg['max_chars'])})
            atomic_json(path, state)
            print(f'  summary update {i + 1}/{len(batches)}', flush=True)
        return [state['summary']] if state['summary'] else [], {'cache_key': key,
            'cache_hit': initial == len(batches), 'resumed_batches': initial,
            'summary_updates': len(batches), 'construction_cost': state['calls']}


APPROACHES = {'raw_history': RawHistory, 'vector_retrieval': Retrieval, 'running_summary': RunningSummary}
