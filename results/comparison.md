# Preliminary pilot comparison

Exploratory pilot; ungraded answers are not accuracy results. Cached preparation costs are excluded from current-attempt totals; original costs remain in per-question memory metadata.

Means below use successful questions only. Unequal coverage is not a paired comparison.

| Approach | Success / expected | Accuracy | Prompt tokens | Output tokens | Answer seconds | Preparation seconds | Total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_history | 14 / 14 | ungraded | 7,059.79 | 30.64 | 61.01 | 0.13 | 61.14 |
| vector_retrieval | 14 / 14 | ungraded | 2,541.86 | 26.71 | 18.37 | 19.09 | 37.47 |
| running_summary | 2 / 14 | ungraded | 607.00 | 8.00 | 5.41 | 3,232.32 | 3,237.73 |

## Matched successful questions

All supplied approaches completed the same 2 questions: e5ba910e_abs, 0ddfec37_abs
This subset may cover only some categories and is not full-pilot evidence.

| Approach | Prompt tokens | Answer seconds | Preparation seconds | Total seconds |
|---|---:|---:|---:|---:|
| raw_history | 6,998.00 | 46.23 | 0.08 | 46.31 |
| vector_retrieval | 2,622.00 | 17.97 | 18.13 | 36.11 |
| running_summary | 607.00 | 5.41 | 3,232.32 | 3,237.73 |

## Measured preparation calls in current attempts

| Approach / phase | Calls | Input tokens (sum) | Output tokens (sum) | Wall seconds (sum) |
|---|---:|---:|---:|---:|
| vector_retrieval / index_embedding | 276 | 1,846,030.00 | unavailable | 256.34 |
| vector_retrieval / query_embedding | 14 | 380.00 | unavailable | 0.17 |
| running_summary / summarization | 100 | 286,375.00 | 48,441.00 | 6,463.80 |

## Coverage and limitations

- raw_history: statuses {'ok': 14}; 0 pending; 0 completed cache hits.
- vector_retrieval: statuses {'ok': 14}; 0 pending; 0 completed cache hits.
- running_summary: statuses {'ok': 2, 'running': 1}; 11 pending; 0 completed cache hits.
  - 07741c44: unfinished record; a saved running status does not establish an active process

Actual runtime usage is reported, not fitting bounds. Missing fields remain unavailable;
the JSON includes the measured denominator for every aggregate. Preparation totals exclude
cached earlier work; per-question cache metadata retains original construction costs.
An interrupted question can have checkpointed work not yet represented in its final record.

The conservative fitting rule substantially underuses the 32K context ceiling. This is
a pilot of these configured implementations, not a claim about each architecture’s best
possible performance. See `notes/manual_review.md` for qualitative examples.
