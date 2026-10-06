# Memory Architecture Study

A local, reproducible 14-question pilot comparing raw history, Chroma vector
retrieval, and running summaries on LongMemEval. The shared answering model is
Ollama `llama3.1:8b`, with temperature 0, seed 42, a 32,768-token context ceiling,
and a 512-token answer allowance. Hybrid memory and the full 500-question run are
outside the current pilot milestone; the main experiment's size remains to be decided.

## Midpoint milestone: cumulative Weeks 1–7

The current deliverable is a cumulative midpoint report covering project inception
through Week 7. See the [PDF report](reports/Midpoint_Project_Report.pdf),
[editable Word report](reports/Midpoint_Project_Report.docx),
[report source](reports/midpoint_project_report.md), and
[current project plan](notes/project_plan.md). The earlier Weeks 5–6 planning note
is superseded for the current assignment.

At the October 6, 2026 audit, raw history and retrieval each have 14/14 successful
predictions. Summary has 2/14 successful predictions, one unfinished question with
28/55 updates checkpointed, and 11 pending questions. No runner lock was held.
All 30 saved answers remain ungraded. The current unit suite has 17 passing tests.

Refresh the comparison and provenance/checkpoint audit without invoking inference:

```sh
sh scripts/refresh_midpoint.sh
# Optional: create a NEW worksheet; an existing path is rejected to protect reviews.
.venv/bin/python src/audit_progress.py --review-output results/midpoint/review_new.jsonl
```

`results/midpoint/audit.json` derives status from saved records and samples runner
locks. A saved `running` record with no held lock is displayed as `unfinished`;
original inference records are preserved. Summary checkpoint costs overlap costs
in completed records and must not be added again. `compare.py` also reports the
intersection of successful question IDs, avoiding comparisons of different subsets.
Currently the intersection across all three approaches contains only two abstention
questions and cannot represent full-pilot performance.

The review worksheet contains references and annotated supporting turns for
post-inference evaluation only. Its blank assessment fields are not grades.
Literal text presence does not detect paraphrases or prove evidence sufficiency.

To rebuild the figures and PDF, install the optional report dependencies separately:

```sh
python3 -m venv /tmp/memory-study-report-env
/tmp/memory-study-report-env/bin/python -m pip install -r requirements-report.txt
/tmp/memory-study-report-env/bin/python scripts/build_midpoint_report.py
# If pandoc is installed, add --docx to also regenerate the Word document.
```

The written report is an authored snapshot. Review its numerical claims when
refreshing experiment results; the PDF builder renders the text and regenerates
figures, but does not rewrite the narrative. Historical notes retain their original
snapshots; use the refreshed audit for present coverage.

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
ollama serve
# In another terminal:
ollama pull llama3.1:8b
ollama pull nomic-embed-text
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_local.py
```

The lock file captures this macOS/Python 3.13 environment. `requirements.txt`
pins the direct dependency for other supported environments. Ollama must be
reachable at the configured local URL. Use default flash-attention/KV-cache
settings as documented in `experiment_config.json`; original measurements found
these faster on this M4/16 GB machine. Do not run the three experiments concurrently
when comparing latency. This project neither uploads benchmark data nor invokes
paid inference. Chroma telemetry is disabled and embeddings are supplied locally.

The existing benchmark file is `data/longmemeval_s_cleaned.json`, revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f` from
[the cleaned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
The runner verifies its recorded SHA-256 before running. Preserve the fixed
`results/pilot_ids.json` (seed 42); do not regenerate it for comparison runs.

## Run and resume

```sh
# First benchmark smoke; remove --limit to finish the remaining questions.
.venv/bin/python src/run_experiment.py --approach raw_history --output results/pilot/raw_history --limit 1
.venv/bin/python src/run_experiment.py --approach raw_history --output results/pilot/raw_history
.venv/bin/python src/run_experiment.py --approach vector_retrieval --output results/pilot/vector_retrieval
.venv/bin/python src/run_experiment.py --approach running_summary --output results/pilot/running_summary
.venv/bin/python src/compare.py results/pilot/raw_history results/pilot/vector_retrieval results/pilot/running_summary
```

Each directory has a manifest, one atomic JSON record per attempted question, and
a coverage report with all 14 expected IDs. Successful records are skipped on
resume. Add `--retry-errors` to retry errors/interrupted/running records; previous
attempts are archived. A lock prevents concurrent writers. Configuration,
inference source hashes, model digests, runtime version, and pilot IDs must match
before resuming; changed experiments need a new output directory. A killed process
may leave a `running` record; coverage is rebuilt at orderly completion and the
comparison script reads records directly. Do not equate saved records with success.

## Configuration and method

`configs/pilot.json` is the executable configuration; `experiment_config.json`
preserves research rationale and historical runtime measurements.

- Shared: model, answer instructions, generation settings, context ceiling,
  output allowance, dataset checksum, and saved question IDs.
- Raw history: stable chronological sessions and original turn order, retaining
  dates/session IDs/speakers. Drop oldest complete turns to fit.
- Retrieval: local `nomic-embed-text` (137M; downloaded Ollama digest recorded),
  Chroma cosine similarity, 1,800-byte turn fragments, top 12, embedding batches
  of 32. Each fragment retains date/session/speaker/turn/fragment metadata.
  Use the model's `search_document:` and `search_query:` prefixes. Selection
  drops lowest-ranked complete fragments if needed. All speakers are indexed.
- Summary: local `llama3.1:8b`; chronological session updates, with oversized
  sessions split into bounded segments. Previous summary plus conversation only;
  no eventual question, reference answer, or evidence annotation. Configurable
  update instruction, 512 output tokens, 4,000-character hard bound, and
  18,000-character input batches. Character clipping and output-limit stops are
  recorded; compression can lose information.
- Fitting: UTF-8 byte count is a conservative upper bound on text token count
  for the Llama byte-level tokenizer, with 256 additional tokens reserved for
  the runtime template. This avoids expensive fitting inference and hidden
  runtime truncation, but **underuses the 32K window substantially**. It changes
  the old calibrated baseline policy; the old smoke is not reused. Recorded
  `input_tokens`/`output_tokens` always come from Ollama, separately from the
  fitting bound. A matching exact tokenizer is a valuable next improvement.
- Cache: `results/cache/` uses conversation/configuration/model hashes. Chroma
  histories are isolated, and summary updates are checkpointed after each call.
  Cache keys exclude questions and benchmark labels. Original preparation costs
  are stored with cached artifacts; current-run calls and wall time are separate.

`answer_seconds` includes the final request's prompt processing and generation;
`preparation_seconds` includes indexing, retrieval, summary updates and fitting;
`total_seconds` covers both. Per-call records separate embedding, summary, and
answer usage, runtime prompt/generation/load durations, and wall time. Missing
usage stays null. Costs are tokens and seconds, not invented dollar estimates.

## Evidence and grading

See [evaluation procedure](notes/evaluation.md), [development log](notes/development_log.md),
and [progress summary](notes/progress_weeks_3_4.md). The comparison script exports
inputs for the pinned official evaluator and can import consistently graded logs.
No accuracy is claimed for ungraded answers. Synthetic smoke results under
`results/smoke/` are validation only. September 15 results and the original
baseline source (`notes/baseline_raw_history_legacy.py`) are preserved.
