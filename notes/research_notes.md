# Research Notes — Step 1 (Define a Controlled Experiment)

> Draft generated while scaffolding the project on 2026-09-15. Read the LongMemEval paper (arXiv:2410.10813) yourself and rewrite this in your own words before treating it as submitted evidence — that's what the assignment asks for, and it's worth actually understanding before building on it.

## Research question

How do three memory architectures for a conversational LLM agent — full raw history, vector retrieval over history, and a running summary — compare on:

- **answer accuracy** (overall and broken down by LongMemEval's question types: information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention),
- **token usage** (input, output, and memory-construction tokens), and
- **latency** (memory construction and question-answering measured separately),

and what characteristic failure modes does each approach produce? All three approaches share the same answering model, the same exact model version, the same generation settings, and the same benchmark question set, so any difference in outcome is attributable to the memory mechanism rather than to the model or the questions.

## Approaches

| Approach | Planned behavior | Hypothesis (untested until the pilot runs) |
|---|---|---|
| **Raw history** | Feed the model as much of the dated conversation history as fits in a fixed input budget; when it doesn't fit, drop the oldest complete turns first. | Early-session facts get truncated away on long histories, so accuracy should drop on questions whose evidence sits early in the timeline. |
| **Vector retrieval** | Chunk the history (e.g., per exchange or bounded turn group), embed the chunks, embed the question, and retrieve the top-k most similar chunks as memory. | Semantic similarity can miss a session that's *relevant* but not *lexically/semantically close* to the question — e.g., a later session that updates or contradicts an earlier fact without restating it verbatim. |
| **Running summary** | Process sessions in chronological order; after each one, fold it into a bounded running summary, without ever seeing the eventual test question. | Compression should lose low-salience details (hurting narrow extraction questions) and may retain facts that a later session invalidated (a knowledge-update failure). |

## Metrics to record per question/run

- Correct / attempted, grouped by `question_type` and by abstention (`_abs` IDs)
- Input tokens, output tokens (from the runtime's own usage fields — never estimated when an exact count is available)
- Memory-construction time and tokens (indexing/embedding for retrieval; summarization calls for running summary), kept **separate** from question-answering latency
- Question-answering wall-clock latency
- A short failure-mode label for incorrect answers (truncation / retrieval miss / summarization loss / reasoning error / inappropriate or missed abstention)

## Comparison rules (fixed before any pilot run)

- Same answering model, exact version/digest, answer instruction, generation settings (temperature 0, seed 42 where supported), and question set across all three approaches.
- All methods start from the same original dated history (`haystack_sessions` + `haystack_dates` + `haystack_session_ids`); resulting prompt lengths are allowed to differ, and that difference is itself a result to report, not something to normalize away.
- A fresh memory store per benchmark history — no cross-contamination between examples.
- Speaker roles, session timestamps, and the question date are preserved in whatever is shown to the model.
- Summaries and retrieval indices are built without access to the eventual test question.
- Reference answers (`answer`) and evidence annotations (`answer_session_ids`, `has_answer`) are used only for grading, never fed into the memory pipeline.

See `experiment_config.json` for the machine-readable version of this design (model identity, generation settings, context policy, dataset revision, and grading method).
