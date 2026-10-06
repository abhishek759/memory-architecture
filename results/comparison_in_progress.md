# Preliminary pilot comparison

14-question exploratory pilot; ungraded answers are not accuracy results. Cached preparation costs are excluded from current-attempt totals; original costs remain in per-question memory metadata.

Means below use successful questions only. Unequal coverage is not a paired comparison.

| Approach | Success / expected | Accuracy | Prompt tokens | Output tokens | Answer seconds | Preparation seconds | Total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_history | 14 / 14 | ungraded | 7,059.79 | 30.64 | 61.01 | 0.13 | 61.14 |

## Measured preparation calls in current attempts

| Approach / phase | Calls | Input tokens (sum) | Output tokens (sum) | Wall seconds (sum) |
|---|---:|---:|---:|---:|

## Coverage and limitations

- raw_history: statuses {'ok': 14}; 0 pending; 0 completed cache hits.

Actual runtime usage is reported, not fitting bounds. Missing fields remain unavailable;
the JSON includes the measured denominator for every aggregate. Preparation totals exclude
cached earlier work; per-question cache metadata retains original construction costs.
An interrupted question can have checkpointed work not yet represented in its final record.

The conservative fitting rule substantially underuses the 32K context ceiling. This is
a pilot of these configured implementations, not a claim about each architecture’s best
possible performance. See `notes/manual_review.md` for qualitative examples.
