# Memory Architecture

Compare raw conversation history, vector retrieval, and running summaries on
[LongMemEval](https://github.com/xiaowu0162/LongMemEval), using local Ollama models.
The runner records token usage, latency, predictions, and run configuration.

## Setup

Requires Python 3.13, Ollama, and macOS or Linux (the runner uses POSIX file locks).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
ollama serve
```

In another terminal:

```sh
ollama pull llama3.1:8b
ollama pull nomic-embed-text
.venv/bin/python -m unittest discover -s tests -v
```

`requirements-lock.txt` records the original macOS environment. Optional tokenizer
experiments require `pip install -r requirements-tokenizer.txt` inside the venv.

Download `longmemeval_s_cleaned.json` from
[longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)
at revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`, and place it in `data/`.
The runner verifies the SHA-256 in `configs/pilot.json`.

## Run

Run commands from the repository root. Start with one question:

```sh
.venv/bin/python src/run_experiment.py --approach raw_history \
  --output results/pilot/raw_history --limit 1
```

Run or resume all three approaches sequentially:

```sh
sh scripts/run_pilot.sh
```

Successful questions are skipped. The script retries failed or interrupted
questions and preserves previous attempts. Resuming requires matching configuration,
model identities, inference source, and Python version. Use a new output directory
for a changed experiment.

To regenerate the comparison from saved runs:

```sh
.venv/bin/python src/compare.py results/pilot/raw_history \
  results/pilot/vector_retrieval results/pilot/running_summary
.venv/bin/python src/render_report.py
```

Outputs are written to `results/comparison.json` and `results/comparison.md`.
Accuracy remains unset until complete, consistent evaluator grades are imported.
See [the method and evaluation guide](docs/method.md).

## Layout

- `src/`: inference, memory implementations, comparison, and analysis tools.
- `configs/`: pilot settings, sample planning, and the paired fitting protocol.
- `tests/`: offline tests and an optional local model smoke test.
- `scripts/run_pilot.sh`: sequential experiment workflow.
- `vendor/longmemeval/`: pinned evaluator source, license, and provenance.
- `results/pilot_ids.json`: fixed 14-question pilot selection.

Datasets, generated results, reports, research notes, and report-authoring tools
stay local and are ignored by Git. The pilot selection is the only tracked file
under `results/`.
