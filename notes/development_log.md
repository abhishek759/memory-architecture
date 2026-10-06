# Development Log

## 2026-10-06 — Cumulative Weeks 1–7 midpoint update

The current assignment covers the complete project history through Week 7,
superseding the earlier Weeks 5–6 planning scope. Reviewed the supplied topic,
proposal, progress reports, and seven-section midpoint requirements against
repository evidence. The current project plan uses the user's earlier Week 12
completion target and identifies the original proposal's 14-week timeline.

Added `audit_progress.py` for dataset/history verification, saved-record/source
provenance checks, point-in-time runner-lock detection, question-specific summary
checkpoint inspection, and an optional review worksheet. It leaves inference
records intact, reports unlocked `running` records as unfinished in the audit,
and refuses to overwrite existing review worksheets. Reference/evidence fields
remain confined to post-inference review outputs.

Extended comparison and Markdown rendering with statistics on the intersection
of successful question IDs. Refreshed comparison outputs and created midpoint
audit/worksheet artifacts. Added a report/figure builder and optional document
dependencies, then authored the cumulative report with all seven required sections.
The inference source/configuration fingerprints remain unchanged and compatible
with the saved runs.

Validation: 17 unit tests pass, including six new audit and matched-comparison
tests. Dataset checksum, all 500 histories, pilot membership, and all saved record
fingerprints validate. Coverage is raw history 14/14, retrieval 14/14, summary 2/14;
the third summary question has 28/55 updates checkpointed and no final answer.
No runner lock was held at audit. All 30 completed answers remain ungraded.

No new inference or paid grading was invoked during this midpoint update. The
summary pilot, systematic grading, main experiment, and final error analysis
remain future work. Older dated entries below are historical snapshots.

Dated entries: symptom/change, what was inspected or changed, outcome, and what's left. Add one entry per work session.

## 2026-09-15 — Project kickoff

**Did:**
- Read the LongMemEval Project Process Guide end to end.
- Scaffolded the repository: `src/`, `data/`, `results/`, `notes/`, `evidence/`, `.venv`, local git init, `.gitignore`, `requirements.txt`.
- Drafted `notes/research_notes.md` and `experiment_config.json` for Step 1 (still needs a personal read of the paper and a rewrite in my own words — see the note at the top of research_notes.md).
- Downloaded `longmemeval_s_cleaned.json` from HuggingFace (revision commit `98d7416c24c778c2fee6e6f3006e7a073259d48f`). Verified size (277,383,467 bytes) and sha256 (`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`) by computing them locally — matches what the server reported. Both recorded in `experiment_config.json`.
- Wrote and ran `src/inspect_data.py` (Step 4). All assertions passed (no duplicate question IDs; `haystack_sessions`/`haystack_dates`/`haystack_session_ids` arrays aligned on every row). Results:
  - 500 questions total, sessions-per-question ranges 38-62 (mean 47.7) — confirms the guide's warning not to assume a fixed 40.
  - Reporting-group counts: single-session-user 64, single-session-assistant 56, single-session-preference 30, multi-session 121, temporal-reasoning 127, knowledge-update 72, abstention 30.
  - Saved `results/dataset_summary.json` and a fixed 14-question pilot sample (seed 42) to `results/pilot_ids.json`.
- Chose the answering model/runtime for Step 3: **local Ollama, `llama3.1:8b`**. Recorded for the report: this machine (Apple M4, 16GB unified memory) cannot run the paper's ~115K-token "S" context setting locally — KV-cache alone would need roughly 15GB at that length — so the baseline will use a smaller, explicitly recorded context budget instead. This is a documented deviation from the paper's setup, not a defect, and should be stated as a limitation in Progress Report 1.
- `brew install ollama` failed on the first attempt: Homebrew's auto-update step couldn't resolve `ghcr.io` (DNS). Retested a handful of hosts individually — `ollama.com`, `github.com`, `formulae.brew.sh`, and (on retry) `ghcr.io` itself all responded normally, so this looks like transient network flakiness in this environment rather than a real block. Retried with `HOMEBREW_NO_AUTO_UPDATE=1` and it succeeded: Ollama 0.34.0 installed (with Apple MLX acceleration as a dependency).
- Started the server manually (not via `brew services`, so the performance env vars from the install caveats apply): `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`. Confirmed listening on 127.0.0.1:11434.
- `ollama pull llama3.1:8b` (4.9GB) failed to converge on the first two attempts — progress climbed to ~1.9GB then reset to 0% and restarted, repeatedly, over many minutes, with the reported transfer rate decaying toward a stall each time before each reset. Diagnosed by pulling a small model (`qwen2.5:0.5b`, 397MB) instead, which completed in one clean pass at ~68MB/s — ruling out a general network block and pointing at something specific to the large, multi-stream transfer (Ollama splits big blobs across `OLLAMA_MAX_TRANSFER_STREAMS` (4) parallel connections by default). Retried with `OLLAMA_MAX_TRANSFER_STREAMS=1` and it completed cleanly in one continuous run.
- Ran `src/check_model.py`: got a real response ("Memory is a complex process..."), 18 input tokens, 21 output tokens, 9.112s elapsed (this includes cold-start model load — not a warm-latency number). Saved to `results/model_check.json`.
- Investigated the actual local context ceiling empirically (`ollama ps` after loading at each size) instead of relying on the earlier back-of-envelope KV-cache math:

  | num_ctx | total memory | processor |
  |---|---|---|
  | 4,096 | 5.0 GB | 100% GPU |
  | 32,768 | 7.1 GB | 100% GPU |
  | 65,536 | 9.6 GB | 100% GPU |
  | 81,920 | 10 GB | 100% GPU |
  | 114,688 (paper's ~115K setting) | 13 GB | 18%/82% CPU/GPU |

  The paper's own setting *does* load, but spills off the GPU, which would make raw-history latency incomparable to the (GPU-resident) retrieval/summary pipelines later. **Decision: context budget = 65,536 tokens** — the largest tested size that stays 100% GPU-resident, with ~6.4GB of headroom left on this 16GB machine. Recorded in `experiment_config.json` along with the full table and rationale.

**Outcome:** Steps 1, 2, 3, and 4 all done (Step 1 still needs the personal rewrite noted above).

**Remaining / next:**
- Implement the raw-history baseline (`format_prompt`, the drop-oldest-turn loop, the shared answer instruction) and test it on one pilot question before batching.
- Run the full 14-question pilot, grade manually, and write Progress Report 1.

## 2026-09-15 (continued) — Baseline implementation and a real performance wall

**Token counting resolved without HF's gated Llama tokenizer:** `num_predict: 1` in an `/api/generate` call returns an accurate `prompt_eval_count` after ~0.6s (confirmed against a fixed test string), using the exact same tokenizer/chat-template the model actually runs — no need to accept Meta's license and pull `meta-llama/Llama-3.1-8B-Instruct` from HuggingFace. (`num_predict: 0` was tried first and does *not* suppress generation on Ollama 0.34.0 — it ran a full unconstrained generation anyway. Worth knowing if reused elsewhere.)

**First truncation-search design was impractical.** Implemented `fit_turns()` as a real-token binary search (pop-from-front, guide's spec, done as O(log n) real calls instead of O(n) for speed). On the first pilot question (`e5ba910e_abs`, 48 sessions / 522 flattened turns) this didn't converge in over 5 minutes — each check near the ~65K budget took a genuinely long time by itself. Killed it and rewrote `fit_turns()` to binary-search a cheap chars-per-token *estimate* first (instant, no model calls), then calibrate that ratio against one real measurement and re-search — this is the version in `src/baseline_raw_history.py` now.

**Hit a second, unrelated bug immediately after:** the recalibrated version *also* hung, and `ollama ps` showed the runner status as `Stopping...` right when it happened — the default 5-minute `keep_alive` had expired mid-pipeline (multiple sequential calls without an explicit keep_alive), and the next request got stuck behind the reload. Fixed by sending `"keep_alive": "30m"` on every request in `ollama_generate()`.

**With both fixes in place, the real bottleneck turned out to be raw throughput, not a bug.** `tail`ing `/tmp/ollama_serve.log` mid-run showed llama-server's own `prompt processing` timing: **~37 tokens/sec** with `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` (the settings Homebrew's install caveats recommended). At that rate a ~60K-token prompt takes ~25+ minutes — impractical for a 14-question pilot, let alone the full 500-question comparison later.

**Counterintuitive fix: turning those "optimizations" off is ~3x faster here.** Restarted `ollama serve` with no special env vars (plain fp16 KV cache, flash attention off) and measured prompt-eval directly via the API's own `prompt_eval_count`/`prompt_eval_duration` fields on an 8,194-token test prompt: **107 tokens/sec** — about 3x the flash-attention+q8_0 number. Re-ran the GPU-memory-ceiling test under these default settings since fp16 KV cache is ~2x the size of q8_0:

| num_ctx | total memory | processor |
|---|---|---|
| 8,192 | 5.8 GB | 100% GPU |
| 16,384 | 6.9 GB | 100% GPU |
| 32,768 | 9.1 GB | 100% GPU |
| 49,152 | 11 GB | 100% GPU |
| 65,536 | 14 GB | 18%/82% CPU/GPU |
| 81,920 | 16 GB | 29%/71% CPU/GPU |

So on this machine, with the flash-attention/q8_0 combination working against us for prompt-eval speed, the practical choice is smaller-and-fast (default settings, ≤~49K context, 100+ tok/s) rather than larger-and-slow (flash-attn+q8_0, ≤~80K context, ~37 tok/s). Both are well under the paper's ~115K setting either way. **Paused here to decide the actual number with the user rather than pick unilaterally, since it now affects timing for the whole remaining project, not just this baseline.**

**Decision: default Ollama settings (flash attention off, fp16 KV cache), context budget 32,768 tokens.** Updated `CONTEXT_BUDGET` in `src/baseline_raw_history.py` and the full rationale plus both measurement tables into `experiment_config.json`.

Re-ran the single-question test (`e5ba910e_abs`, 522 turns) end to end: **285s total**, and the calibrated estimate needed zero recalibration rounds — the first real check (drop=400/522 turns, 126,200 chars → 28,105 real tokens) already fit the 32,256-token budget (context budget minus the 512-token output allowance), so it went straight to the final generation call (1.8s, 18 output tokens). Retained only 122 of 522 turns — this question's full history is far larger than 32K tokens, so raw-history is truncating hard here, which is itself a real result worth keeping. (This is an `_abs` question; the reference answer is an appropriate abstention noting a headphone purchase was mentioned but no iPad. The model's prediction — "I don't have access to your personal shopping history" — is also a non-answer, but phrased as a capability refusal rather than reasoning from the supplied memory. Worth flagging in Step 7's error analysis: is this actually grounded in the truncated memory it saw, or a generic refusal reflex? Can't tell without inspecting what the retained 122 turns actually contained.)

**Outcome:** The raw-history baseline pipeline works end-to-end. Moving to the full 14-question pilot batch.

## 2026-09-30 — Weeks 3–4 implementation and pilot execution

Inspected all existing code, configuration, notes, dataset summary, pilot IDs,
and model-check output. Confirmed no saved baseline pilot, retrieval, summary,
or evaluator implementation. The README's 65,536-token status was stale; the
executable baseline and research configuration already specified 32,768.
Preserved the dataset and 14 saved IDs (checksum and history alignment verified).
Archived the original baseline implementation in `notes/baseline_raw_history_legacy.py`.

Implemented `experiment.py`, `memories.py`, `run_experiment.py`, and `compare.py`:
shared Ollama answering, allowlisted history fields, chronological ordering,
complete-turn truncation, Chroma per-history retrieval, running summary
checkpoints, manifest fingerprints/model digests, atomic per-question saves,
resume compatibility checks, errors, coverage, official evaluation exports and
strict graded-result import. Model responses report runtime counts separately
from prompt-fitting estimates/bounds. No monetary local-inference cost assigned.

Executable parameters are centralized in `configs/pilot.json`. Local embeddings
use Ollama nomic-embed-text (137M), with the required search_document/search_query
prefixes; ChromaDB 1.5.5 is installed and telemetry disabled. Summary generation
uses llama3.1:8b, 512 output tokens, a 4,000-character bound, and chronological
session segments. Cache keys bind conversation, memory settings and model identity;
summary updates save after every call. References and evidence labels are excluded
from all inference and memory-construction inputs.

**Method change:** replaced unaccounted expensive baseline fitting probes with a
conservative UTF-8-byte bound and a 256-token template allowance. It preserves the
32K ceiling and whole-turn removal but substantially underutilizes the window.
This limits the raw-history interpretation; September 15 smoke is not reused.
An exact locally matched tokenizer is a priority improvement before scaling up.

Started Ollama 0.34.0 with default runtime settings; downloaded local embedding
weights and installed dependencies in `.venv`. Sandbox denied Python's localhost
connection, so smoke and benchmark processes use approved local-server access.
No paid API calls, GitHub push, or publication occurred.

Validation: 11 focused tests pass (see `results/test_validation.txt`), including
real Chroma isolation, annotation exclusion, Unicode fitting, failed-call usage,
summary checkpoint recovery, manifest compatibility, missing coverage, and stale
grading rejection. Real synthetic smoke passes for all three approaches, each
answering Paris after a Rome-to-Paris update; saved under `results/smoke/`.
First real benchmark raw-history question succeeded with 7,310 prompt tokens,
8 output tokens and 48.45 seconds total; saved under `results/pilot/raw_history/`.
Longer runs follow sequentially to avoid concurrency bias. `scripts/run_pilot.sh`
provides the sequential resumable workflow.

Inspected the official LongMemEval README and evaluation source, then pinned the
unaltered evaluator and license at commit
`9e0b455f4ef0e2ab8f2e582289761153549043fc` under `vendor/longmemeval/`.
Its category/abstention rubrics are the shared grading procedure. No authorized
external judge is available; accuracy remains ungraded rather than substituting
exact match or the small answering model. See `notes/evaluation.md` for integration.
Manual spot checks against references/supporting turns are in `notes/manual_review.md`.

### Pilot execution outcome at 2026-09-30 09:31 CDT

Raw history and vector retrieval each completed **14/14** fixed pilot questions
without runtime errors or missing answer token measurements. Coverage and
uniqueness audits passed. No completed memory cache hits were used.

| Approach | Mean prompt tokens | Mean output tokens | Mean answer seconds | Mean preparation seconds | Mean total seconds |
|---|---:|---:|---:|---:|---:|
| raw_history | 7,059.79 | 30.64 | 61.01 | 0.13 | 61.14 |
| vector_retrieval | 2,541.86 | 26.71 | 18.37 | 19.09 | 37.47 |

These are ungraded pilot measurements, not accuracy or a definitive ranking.
The comparison is saved in `results/comparison.json` and `.md`. First-answer load
times differ; per-call load durations remain available for interpretation.

The summary pilot is **still running, not complete**. Process 56599 was verified
alive. At the snapshot it had checkpointed 32/49 updates for `e5ba910e_abs`, with
0/14 final answers saved. The full pilot requires 719 updates; observed update
costs imply many hours of local inference. Preparation so far consumed 92,965
prompt tokens, 16,231 output tokens and 1,827.82 seconds. See
`results/summary_preparation_progress.json`; this snapshot does not claim a final
answer or include the currently executing update. The active runner will keep
saving question records and caches. Regenerate the comparison after it finishes.

A detached shell continuation attempt did not survive its launching shell;
process inspection confirmed it was absent before the retrieval runner was
started through a managed session. No duplicate inference run was left active.
The standard sequential script remains available for future terminal runs.

Remaining: finish summary, audit 42 completed predictions, consistently grade
with an authorized official evaluator, then update comparison and failure analysis.
No external grading was invoked, and no accuracy result has been fabricated.
