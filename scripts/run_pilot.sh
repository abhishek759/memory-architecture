#!/bin/sh
# Run sequentially on one local GPU. Safe to repeat after interruption.
set -eu
cd "$(dirname "$0")/.."
mkdir -p results/pilot
for approach in raw_history vector_retrieval running_summary; do
  .venv/bin/python src/run_experiment.py --approach "$approach" \
    --output "results/pilot/$approach" --retry-errors
 done
.venv/bin/python src/compare.py results/pilot/raw_history \
  results/pilot/vector_retrieval results/pilot/running_summary
.venv/bin/python src/render_report.py
