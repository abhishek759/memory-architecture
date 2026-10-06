import json
import time
import urllib.request
from pathlib import Path

MODEL = "llama3.1:8b"
CONTEXT_BUDGET = 32768
OUTPUT_TOKEN_LIMIT = 512
GENERATION_OPTIONS = {
    "temperature": 0, "seed": 42,
    "num_ctx": CONTEXT_BUDGET, "num_predict": OUTPUT_TOKEN_LIMIT,
}

ANSWER_INSTRUCTION = """Answer the question using only the supplied conversation memory.
Use the session dates and question date when interpreting time.
If the memory does not support an answer, say that it does not
contain enough information. Give a concise answer.

Conversation memory: {memory}
Question date: {question_date}
Question: {question}"""


def ollama_generate(prompt, options):
    # keep_alive prevents the runner from unloading mid-pipeline (the default
    # 5m keep_alive expired during a multi-call fitting sequence and caused a
    # multi-minute hang waiting on a reload - see development_log.md).
    payload = {"model": MODEL, "prompt": prompt, "stream": False,
               "options": options, "keep_alive": "30m"}
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def count_tokens(prompt):
    # num_predict=1 forces a real prompt_eval pass with minimal generation cost.
    # num_predict=0 was tested and does NOT suppress generation on this Ollama version.
    result = ollama_generate(prompt, {"num_ctx": CONTEXT_BUDGET, "num_predict": 1})
    return result["prompt_eval_count"]


def estimate_tokens(text):
    # Rough ~4-chars-per-token heuristic, used only to narrow the search
    # before a real call confirms the actual count. A real prompt_eval call
    # at 60K+ tokens takes tens of seconds, so a naive real-token binary
    # search (tested and abandoned - see development_log.md) is impractical
    # across a batch. This estimate is never what gets reported as input_tokens.
    return len(text) / 4.0


def flatten_turns(row):
    turns = []
    for session, date, session_id in zip(
        row["haystack_sessions"], row["haystack_dates"], row["haystack_session_ids"]
    ):
        for turn in session:
            turns.append({
                "date": date, "session_id": session_id,
                "role": turn["role"], "content": turn["content"],
            })
    return turns


def format_memory(turns):
    lines = [
        f"[{t['date']} | session {t['session_id']} | {t['role']}] {t['content']}"
        for t in turns
    ]
    return "\n".join(lines)


def format_prompt(turns, question, question_date):
    return ANSWER_INSTRUCTION.format(
        memory=format_memory(turns), question_date=question_date, question=question
    )


def fit_turns(turns, question, question_date, verbose=False):
    """Drop the oldest complete turns until the prompt fits the context budget
    (token count is monotonic in the number of retained turns, so this always
    converges to the same result as the guide's pop-from-front loop).

    A pure real-token binary search was tried first and abandoned: each check
    near 60K+ tokens takes tens of seconds, so ~log2(turns) real calls made a
    single question impractically slow (see development_log.md). Instead:
    binary-search a cheap chars-per-token estimate (no model calls, so this
    part is instant regardless of history length), then calibrate that ratio
    against one real measurement of the actual text and re-search - which
    converges in a couple of real calls because the second estimate uses the
    tokenizer's true ratio for this specific text, not a generic guess.
    """
    if not turns:
        return turns, False

    def prompt_for(subset):
        return format_prompt(subset, question, question_date)

    budget = CONTEXT_BUDGET - OUTPUT_TOKEN_LIMIT

    def estimate_drop(chars_per_token):
        lo, hi = 0, len(turns)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(prompt_for(turns[mid:])) / chars_per_token <= budget:
                hi = mid
            else:
                lo = mid + 1
        return lo

    chars_per_token = 4.0
    drop = estimate_drop(chars_per_token)
    for attempt in range(4):
        candidate = turns[drop:]
        if not candidate:
            return [], True
        text = prompt_for(candidate)
        actual = count_tokens(text)
        if verbose:
            print(f"    [fit attempt {attempt}] drop={drop}/{len(turns)} "
                  f"chars={len(text)} real_tokens={actual} budget={budget}", flush=True)
        if actual <= budget:
            return candidate, drop > 0
        chars_per_token = len(text) / actual  # calibrate to this text's real ratio
        drop = max(estimate_drop(chars_per_token), drop + 1)  # force forward progress
    return [], True


def run_one(row, verbose=False):
    turns = flatten_turns(row)
    original_count = len(turns)
    retained_turns, truncated = fit_turns(turns, row["question"], row["question_date"], verbose=verbose)
    base_record = {
        "question_id": row["question_id"],
        "question_type": row["question_type"],
        "is_abstention": row["question_id"].endswith("_abs"),
        "question": row["question"],
        "question_date": row["question_date"],
        "reference_answer": row["answer"],
        "original_turns": original_count,
    }
    if not retained_turns:
        return {**base_record, "status": "context_error",
                "error": "no turns fit even after dropping all but the last one"}

    prompt = format_prompt(retained_turns, row["question"], row["question_date"])
    started = time.perf_counter()
    result = ollama_generate(prompt, GENERATION_OPTIONS)
    elapsed = time.perf_counter() - started
    retained_session_ids = sorted(set(t["session_id"] for t in retained_turns))
    return {
        **base_record,
        "status": "ok",
        "prediction": result["response"].strip(),
        "prompt": prompt,
        "retained_turns": len(retained_turns),
        "retained_session_ids": retained_session_ids,
        "truncated": truncated,
        "input_tokens": result.get("prompt_eval_count"),
        "output_tokens": result.get("eval_count"),
        "elapsed_seconds": elapsed,
        "model": MODEL,
        "context_budget_tokens": CONTEXT_BUDGET,
    }


def load_rows():
    data_path = Path("data/longmemeval_s_cleaned.json")
    with data_path.open(encoding="utf-8") as f:
        rows = json.load(f)
    return {r["question_id"]: r for r in rows}


def main():
    pilot_ids = json.loads(Path("results/pilot_ids.json").read_text())
    by_id = load_rows()

    results = []
    for i, qid in enumerate(pilot_ids):
        row = by_id[qid]
        print(f"[{i + 1}/{len(pilot_ids)}] {qid} ({row['question_type']}) ...", flush=True)
        record = run_one(row)
        results.append(record)
        if record["status"] == "ok":
            print(f"  truncated={record['truncated']} retained_turns={record['retained_turns']}"
                  f"/{record['original_turns']} input_tokens={record['input_tokens']}"
                  f" elapsed={record['elapsed_seconds']:.2f}s", flush=True)
        else:
            print(f"  ERROR: {record['error']}", flush=True)

    Path("results").mkdir(exist_ok=True)
    Path("results/baseline_raw_history_predictions.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("Saved results/baseline_raw_history_predictions.json")


if __name__ == "__main__":
    main()
