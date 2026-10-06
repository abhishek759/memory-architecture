# Weeks 3–4 implementation progress

The three pipelines, shared runner, and evaluation integration are implemented.
Raw-history and retrieval pilots are complete (14/14 each). The full summary
pilot is running and incomplete: at the 09:31 CDT snapshot, 32 of 49 updates
for its first history were checkpointed, with 0/14 final answers saved.
No graded accuracy comparison is available yet.

## Completed and validated

- Shared local Ollama client and configuration-controlled answering across all approaches.
- Raw history with chronological metadata, complete-turn removal, and separate fitting bounds/runtime counts.
- Chroma retrieval with local nomic embeddings, isolated histories, metadata, configurable chunks/top-k, and cache provenance.
- Running summaries built only from chronological history, with bounded updates, separate costs, and per-update checkpoints.
- Atomic per-question records, run locks, model/config/source fingerprints, error archives, coverage audits, and resumable execution.
- Pinned official evaluator, compatible exports, strict result imports, comparison JSON and Markdown reports.
- Eleven focused tests passed; all three real synthetic smoke tests returned Paris after the recorded Rome-to-Paris move.

## Evidence and preliminary measurements

`results/test_validation.txt` and `results/smoke/` contain validation evidence.
`results/pilot/raw_history/coverage.json` and
`results/pilot/vector_retrieval/coverage.json` each confirm all 14 fixed IDs succeeded.
Each prediction has a manifest fingerprint, supplied prompt, actual usage, timing,
selection metadata, and request costs. No old smoke prediction was reused.

Raw-history means: 7,059.79 prompt tokens, 30.64 output tokens, 61.01 seconds answering,
0.13 seconds preparation, and 61.14 seconds total. All 14 histories were truncated;
20–55 complete turns were retained. This is **not** an accuracy result.
Retrieval means: 2,541.86 prompt tokens, 26.71 output tokens, 18.37 seconds
answering, 19.09 seconds preparation, and 37.47 seconds total. All 14 indexes
were newly constructed (no completed cache hits). Across them, indexing used
1,846,030 runtime-reported embedding input tokens. No embedding generation
tokens are reported by the runtime; these fields remain null.

`results/comparison.json` and `results/comparison.md` contain the current
comparison, with summary explicitly incomplete. Regenerate them after summary
finishes using the README commands. `results/summary_preparation_progress.json`
captures the ongoing checkpoint state: 92,965 prompt tokens, 16,231 output tokens,
and 1,827.82 seconds across 32 completed preparation calls. These are preparation
costs, not a completed benchmark answer. The active summary runner was verified
as PID 56599; per-question files and cache checkpoints continue to update.

`notes/manual_review.md` documents selected reference/supporting-turn checks:
appropriate but generic abstentions, a sneaker-location failure with both supporting
sessions omitted, and a restaurant-count answer supported by the retained update.
These observations are not a systematic independent grading pass.

## Challenges and limitations

The local server was stopped and had to be started. Sandbox access to localhost
required execution approval. Dependencies and the embedding model were installed
locally; no paid service, publication, or GitHub push was used.

Conservative UTF-8-bound fitting substantially underuses the 32,768-token ceiling
and changes the old baseline's fitting policy. This must be disclosed in comparison;
it can disadvantage raw history. Runtime usage is never replaced by this bound.
The next methodological improvement is a verified local tokenizer to use more of
the available window without introducing hidden runtime truncation.

Summary generation requires 719 sequential update calls for the full pilot under
this configuration. Checkpoints preserve completed preparation if interrupted.
The installed 8B model is used for summarization, with 512 output tokens and a
4,000-character bound, so compression loss is an expected risk.

The official grading route requires an authorized external judge or a separately
served large local judge. Neither is invoked here. Ungraded predictions must not be
reported as accuracy, and a 14-question pilot cannot establish full-benchmark ranking.

## Next milestone

Let the active summary run finish (719 updates, likely many hours), then audit
all 42 records, regenerate the comparison, obtain
consistent authorized grading, and compare category outcomes and memory-selection
failures. Resolve tokenizer fitting before considering a larger evaluation.
Hybrid memory and the 500-question evaluation remain outside this milestone.
