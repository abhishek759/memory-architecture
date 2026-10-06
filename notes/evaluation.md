# Pilot evaluation procedure

Use the same pinned [official LongMemEval evaluator](../vendor/longmemeval/evaluate_qa.py)
for every approach. [Upstream guidance](https://github.com/xiaowu0162/LongMemEval#-testing-your-system)
specifies JSONL records containing `question_id` and `hypothesis`.
`src/compare.py` exports precisely those records from successful predictions,
checks matching pilot/configuration/model provenance, and reports missing IDs.
Never count an error as a generated answer or report ungraded outputs as accuracy.

The [pinned evaluator source](../vendor/longmemeval/PROVENANCE.md) handles ordinary
fact questions, temporal tolerance, updated facts, preferences, and abstention
separately. Apply its rubric unchanged across all three approaches. The official
script chooses the abstention prompt from the question ID. Inspect a small
sample manually against reference answers and dated supporting turns; distinguish
observed answer errors from hypothesized retrieval/compression causes.

## Integration (not executed; paid use requires separate authorization)

No paid API requests are part of the experiment runner. After all runs finish,
export inputs:

```sh
.venv/bin/python src/compare.py results/pilot/raw_history results/pilot/vector_retrieval results/pilot/running_summary
```

For an authorized judge environment with `openai`, `backoff`, `tqdm`, and `numpy`
installed, the upstream invocation is:

```sh
python vendor/longmemeval/evaluate_qa.py gpt-4o results/pilot/raw_history/hypotheses.jsonl data/longmemeval_s_cleaned.json
```

Repeat for retrieval and summary using the same judge. The GPT-4o route requires
an API key and incurs charges; do not run it without authorization. The upstream
local `llama-3.1-70b-instruct` option requires a separate server on port 8001 and
resources not available in the current 16 GB setup. Merely exporting hypotheses
is not grading. Save grader identity, revision, and raw logs.

To import completed official result files, create JSON mapping approach names to
the `.eval-results-gpt-4o` file paths, then pass `--grades path/to/mapping.json`
to `src/compare.py`. Import rejects duplicate/missing IDs, changed hypotheses,
non-boolean labels, and mixed judge models. Full-pilot accuracy is withheld unless
all supplied approaches have all pilot questions graded. Category breakdowns
use a distinct abstention group. No exact-match proxy is used.

Fourteen questions (two per group) are exploratory, not a statistically reliable
ranking or a reproduction of the benchmark's full evaluation. Compare cold and
cached preparation separately. Answer time includes prompt processing; total time
includes memory preparation and answering. Actual runtime token counts can be
missing and are never replaced with estimated counts or zeros.
