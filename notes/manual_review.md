# Qualitative pilot spot checks — 2026-09-30

These are exploratory checks by the implementing assistant, not a complete,
independent, blinded grading pass. They must not be aggregated into accuracy.
Reference answers and annotated supporting turns were consulted **after** the
predictions were generated. Inference code never receives those annotations.

| Approach / ID | Observation | Evidence and interpretation |
|---|---|---|
| raw_history / e5ba910e_abs | “It does not contain enough information.” | Reference says headphones were mentioned but not an iPad. Appropriate abstention wording, but the generic response does not establish that the model identified the specific missing fact. |
| raw_history / 0ddfec37_abs | “It does not contain enough information.” | Reference concerns baseball rather than football collectibles. Same caution about generic abstention. |
| raw_history / 07741c44 | Abstains on initial sneaker storage location. | Reference is “under my bed.” An annotated earlier user turn explicitly states that location; a later turn concerns sneakers in a closet shoe rack. Check saved prompt membership below to distinguish memory omission from reasoning failure. |
| raw_history / 6aeb4375 | Answers four Korean restaurants. | The earlier supporting user turn says three; the later one says four. The answer matches the update and the reference. |

Each answer and the exact supplied memory are in
`results/pilot/raw_history/<question_id>.json`; the preserved benchmark provides
the reference and supporting turns. Manual observations here are preliminary and
cannot establish relative architecture quality across the full benchmark.

Prompt membership was verified directly: neither supporting session is present
in the sneaker question's saved prompt. For the restaurant question, the older
three-restaurant session is absent and the newer four-restaurant session is
present. This supports truncation as the sneaker failure mechanism and confirms
that the restaurant answer had the updated evidence available.

Retrieval checks on the same two non-abstention examples:

- `07741c44`: retrieved chunks come from both annotated supporting sessions, yet
  the answer abstains and gives generic shoe-storage advice. Session-level recall
  alone must not be interpreted as answer correctness; inspect the exact turn
  content before attributing this to retrieval versus reasoning.
- `6aeb4375`: both supporting sessions were retrieved and the model answers four,
  matching the later update and reference.

Timing caution: the first saved raw-history answer reports only 0.014 seconds
of model loading, while the first retrieval answer reports 4.168 seconds. The
runtime's per-call load durations are saved; these exploratory means are not a
fully standardized warm-start latency benchmark.

Exact sneaker retrieval inspection: the direct user statement “under my bed” is
absent, but a retrieved assistant turn explicitly discusses keeping the sneakers
“under the bed.” The answer changes the historical question into where sneakers
*should* be stored and gives advice. This suggests a mixture of weak evidence
selection and question interpretation; it is not justified to label it a pure
session-retrieval miss.
