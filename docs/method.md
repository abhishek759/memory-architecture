# Method and evaluation

The pilot uses 14 fixed question IDs, selected with seed 42. `configs/pilot.json`
defines the dataset checksum, model, prompts, and memory settings. All approaches
share `llama3.1:8b`, temperature 0, seed 42, a 32,768-token context ceiling, and
512 output tokens.

- Raw history keeps recent complete turns in chronological order.
- Retrieval uses local `nomic-embed-text` embeddings, Chroma cosine similarity,
  1,800-byte fragments, and the top 12 matches.
- Running summaries process dated sessions without the eventual question, with
  512 output tokens per update and a 4,000-character summary limit.

Memory construction excludes reference answers and evidence annotations. The
pilot fits prompts using a conservative UTF-8 byte bound plus a template margin;
this underuses the context window. Recorded token counts come from Ollama.
Preparation and answer timings are separate. Cached construction costs can overlap
saved records and must not be added twice. Compare latency using sequential runs
and the same successful question IDs across approaches.

## Analysis and review

After creating all three pilot runs:

```sh
.venv/bin/python src/audit_progress.py
.venv/bin/python src/analyze_pilot.py
.venv/bin/python src/plan_experiment.py
.venv/bin/python src/review_pilot.py --export results/review.jsonl
```

The planner selects a disjoint sample; it does not run inference. To assess the
pilot, fill in `correct`, `reviewer`, `review_notes`, and optional `failure_modes`
in the review worksheet, then run:

```sh
.venv/bin/python src/review_pilot.py --validate results/review.jsonl
```

Leave other fields unchanged. Partial reviews and evidence-presence checks do not
establish full-pilot accuracy. The pinned official evaluator is under
`vendor/longmemeval/`; its external grading dependencies and service are separate
from local inference. `compare.py --grades mapping.json` accepts a JSON mapping
from approach names to evaluator result JSONL files. Grades must match saved
predictions and use the same evaluator across a complete comparison.

## Tokenizer and fitting experiments

Install `requirements-tokenizer.txt` to use `tokenizer_probe.py` and
`fitting_experiment.py`. Both require saved pilot runs and the matching local
model. The fitting protocol pins the pilot configuration, inference source, and
model identity. It must match the local baseline before use.

```sh
.venv/bin/python src/tokenizer_probe.py
.venv/bin/python src/fitting_experiment.py --preflight
```

Preflight reads local model metadata and saved prompts. Generation requires
`--run --question-ids ID ...`. The fitting experiment retains its historical
`results/progress_report_4/fitting_experiment` output path for compatibility with
existing runs. The tokenizer probe also retains its historical output path under
`results/progress_report_3/`. Those outputs stay local. Exact token fitting is separate from
the original pilot runner.
