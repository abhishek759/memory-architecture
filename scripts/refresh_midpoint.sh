#!/bin/sh
# Refresh evidence from saved runs. Does not invoke a model or grade answers.
set -eu
cd "$(dirname "$0")/.."
.venv/bin/python src/compare.py results/pilot/raw_history \
  results/pilot/vector_retrieval results/pilot/running_summary
.venv/bin/python src/render_report.py
.venv/bin/python src/audit_progress.py
