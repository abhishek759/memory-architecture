"""Reconstruct the installed Llama tokenizer and preflight fitting, without inference.

Only the frozen pilot model's no-system/no-tools generate template is supported.
The counter is validated against saved runtime counts before proposing new fits.
It is deliberately not integrated into the original pilot runner.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from compare import records
from experiment import ROOT, Ollama, atomic_json, blocks, digest, history, prompt
from run_experiment import load_data
from audit_progress import writer_active

TEMPLATE_SHA256 = '948af2743fc78a328dcb3b0f5a31b3d75f415840fdb699e8b1235978392ecf85'
# Llama 3 pattern documented in upstream llama.cpp/src/llama-vocab.cpp.
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def build_tokenizer(info):
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers
    if info.get('tokenizer.ggml.model') != 'gpt2' or info.get('tokenizer.ggml.pre') != 'llama-bpe':
        raise ValueError('Unsupported tokenizer metadata')
    vocabulary = info['tokenizer.ggml.tokens']
    types = info['tokenizer.ggml.token_type']
    if len(vocabulary) != len(types) or len(set(vocabulary)) != len(vocabulary):
        raise ValueError('Invalid vocabulary/type alignment')
    tokenizer = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(vocabulary)},
        merges=[tuple(s.split(' ')) for s in info['tokenizer.ggml.merges']], ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.add_special_tokens([AddedToken(s, special=True, normalized=False)
                                  for s, kind in zip(vocabulary, types) if kind in (3, 4)])
    if any(tokenizer.token_to_id(token) != i for i, token in enumerate(vocabulary)):
        raise ValueError('Reconstructed vocabulary IDs changed')
    return tokenizer


def check_template(data):
    actual = hashlib.sha256(data['template'].encode()).hexdigest()
    if actual != TEMPLATE_SHA256 or data.get('system'):
        raise ValueError('Unsupported runtime template or default system prompt')


def render_generate(text):
    # API /generate adds the BOS and this single-user/assistant prefix for the
    # checksum-verified template, with no system message or tools in pilot-v1.
    return ('<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' + text
            + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')


def fit_exact(parts, make_prompt, counter, budget):
    """Find a bounded recent-turn suffix using inexpensive local tokenization."""
    if counter(make_prompt('')) > budget:
        raise ValueError('Question/instructions exceed input budget')
    low, high = 0, len(parts)
    while low < high:
        middle = (low + high) // 2
        text = make_prompt('\n'.join(parts[middle:]))
        if counter(text) <= budget:
            high = middle
        else:
            low = middle + 1
    text = make_prompt('\n'.join(parts[low:]))
    measured = counter(text)
    if measured > budget:
        raise ValueError('Final local token count exceeds allowance')
    return text, {'original_blocks': len(parts), 'retained_blocks': len(parts) - low,
                  'dropped_blocks': low, 'offline_prompt_tokens': measured}


def probe(cfg, validate_runtime=False):
    import tokenizers
    started = time.perf_counter()
    client = Ollama(cfg)
    identity = client.identity({cfg['answer_model']})
    data = client.request('/api/show', {'model': cfg['answer_model'], 'verbose': True})
    check_template(data)
    tokenizer = build_tokenizer(data['model_info'])
    counter = lambda text: len(tokenizer.encode(render_generate(text), add_special_tokens=False).ids)
    comparisons = []
    saved_raw = {}
    for approach in ('raw_history', 'vector_retrieval', 'running_summary'):
        manifest, rows = records(ROOT / 'results/pilot' / approach)
        if any(manifest['identity'][k] != identity[k] for k in ('runtime', cfg['answer_model'])):
            raise ValueError('Installed runtime/model differs from saved predictions')
        for row in rows:
            if row['status'] == 'ok' and row.get('input_tokens') is not None:
                count = counter(row['prompt'])
                comparisons.append({'approach': approach, 'question_id': row['question_id'],
                    'prompt_hash': digest(row['prompt']), 'runtime_tokens': row['input_tokens'],
                    'offline_tokens': count, 'matches': count == row['input_tokens']})
                if approach == 'raw_history':
                    saved_raw[row['question_id']] = row
    if not comparisons or not all(c['matches'] for c in comparisons):
        raise ValueError('Counter does not match all available saved runtime prompt counts')
    by_id, ids = load_data(cfg)
    fitted, proposed = {}, {}
    for qid in ids:
        row = by_id[qid]
        text, selected = fit_exact(blocks(history(row)),
            lambda memory: prompt(cfg, memory, row['question'], row['question_date']), counter,
            cfg['context_tokens'] - cfg['output_tokens'] - cfg['template_margin_tokens'])
        fitted[qid] = {**selected, 'proposed_prompt_hash': digest(text),
            'original_runtime_prompt_tokens': saved_raw.get(qid, {}).get('input_tokens'),
            'original_retained_blocks': saved_raw.get(qid, {}).get('selection', {}).get('retained_blocks')}
        proposed[qid] = text
    live = []
    if validate_runtime:
        if any(writer_active(ROOT / 'results/pilot' / approach)
               for approach in ('raw_history', 'vector_retrieval', 'running_summary')):
            raise ValueError('Finish active pilot inference before running token-count probes')
        samples = [
            ('unicode', "Return one word. Memory: café, café\u0301, 東京, नमस्ते, 🧠.\n\tQuestion: what?"),
            ('numbers_whitespace', "Return one word.\r\n\r\nI'M sure we've counted 1234567890.  \n\t  End."),
            ('literal_special_tokens', 'Return one word. Literal data: <|begin_of_text|> <|eot_id|> <|start_header_id|>.'),
            ('near_context', proposed[ids[0]])]
        for name, text in samples:
            before = time.perf_counter()
            # A one-token diagnostic is not an answer prediction. Empty generated
            # text is acceptable; this probe validates the returned prompt count.
            result = client.request('/api/generate', {'model': cfg['answer_model'], 'prompt': text,
                'stream': False, 'keep_alive': cfg['keep_alive'], 'options': {
                    **cfg['generation'], 'num_ctx': cfg['context_tokens'], 'num_predict': 1}})
            actual = result.get('prompt_eval_count')
            offline = counter(text)
            live.append({'name': name, 'prompt_hash': digest(text), 'offline_tokens': offline,
                'runtime_tokens': actual, 'matches': actual == offline,
                'generation_tokens': result.get('eval_count'), 'done': result.get('done'),
                'done_reason': result.get('done_reason'), 'wall_seconds': time.perf_counter() - before})
            print(name, 'offline=', offline, 'runtime=', actual, flush=True)
    return {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'identity': identity,
        'tokenizers_version': tokenizers.__version__, 'template_sha256': TEMPLATE_SHA256,
        'vocabulary_hash': digest(data['model_info']), 'counter_source_hash': digest(Path(__file__).read_text()),
        'matched_prompts': len(comparisons), 'all_matched': True, 'comparisons': comparisons,
        'fitting_preflight': fitted, 'wall_seconds': time.perf_counter() - started,
        'model_generation_calls': len(live), 'runtime_validation': live,
        'all_runtime_probes_matched': all(p['matches'] and p['done'] for p in live) if live else None,
        'status': 'Counter validated against saved prompts; proposed fits are not benchmark answer predictions',
        'note': 'Default preflight performs no inference. Optional runtime probes generate one diagnostic token each; '
                'they are not benchmark answer predictions or accuracy results. Counter is not integrated into the original pilot. '
                'Validation covers these recorded prompts, not all possible inputs or model/template versions.',
        'technical_sources': [
            'https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp',
            'https://huggingface.co/docs/tokenizers/python/latest/api/reference.html']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/progress_report_3/tokenizer_preflight.json')
    p.add_argument('--validate-runtime', action='store_true',
                   help='Run four one-token local probes, including Unicode and a near-context prompt')
    args = p.parse_args()
    report = probe(json.loads((ROOT / 'configs/pilot.json').read_text()), args.validate_runtime)
    atomic_json(args.output, report)
    print('Matched', report['matched_prompts'], 'saved runtime prompts;', report['model_generation_calls'], 'diagnostic generation calls.')
    print('Wrote', args.output)
    if args.validate_runtime and not report['all_runtime_probes_matched']:
        raise SystemExit('Runtime validation failed; do not integrate this counter')


if __name__ == '__main__':
    main()
