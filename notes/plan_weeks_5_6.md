# Weeks 5–6 work plan

> Superseded for the current assignment: the reporting scope is cumulative Weeks
> 1–7. See [current project plan](project_plan.md) and the
> [midpoint report](../reports/midpoint_project_report.md). The original planning
> text below is retained as history, not as the current reporting requirement.

Prepared October 6, 2026. This is a plan, not a record of completed work.
Week numbers follow the user's 12-week project schedule; calendar deadlines were
not supplied.

## Course requirements and project commitments

The supplied course instructions require Progress Report 2 to document work
actually completed during weeks 5 and 6: overview, completed work, evidence,
challenges, and next steps. The report must be a PDF and must be accompanied by
a 2–5 minute video demonstration, following Progress Report 1's requirements.
These are requirements for the eventual deliverables, not instructions to submit
anything now. There is no requirement in the supplied instructions to finish all
500 benchmark questions by week 6.

The project proposal commits to comparing raw history, vector retrieval, and
running summaries under shared answering settings, measuring accuracy, tokens,
and latency, and analyzing errors. Hybrid memory and an improvement experiment
are conditional extensions. The proposal's timeline spans 14 weeks; this plan
uses the user's current 12-week schedule instead.

The supplied Progress Report 2 PDF already describes weeks 5–6 as completed.
Its submission status is unknown. It can inform the eventual report, but its
claims must be checked against artifacts. In particular, implementing a pilot
workflow is different from completing every pilot prediction and grading it.

## Verified starting point

- All three pipelines, shared runner, checkpointing, and comparison tooling exist.
- Raw history and retrieval each have 14 successful predictions.
- Summary has two successful predictions. Its third question, `07741c44`, has
  28 of 55 summary updates checkpointed and a stale `running` record. Eleven
  additional questions have not been attempted. No runner or Ollama process was
  active at inspection.
- All 11 existing unit tests pass. Dataset checksum, 500 question IDs, history
  alignment, and date parsing validate. Saved inference source hashes match.
- Existing comparison reports and progress notes are stale. No official accuracy
  grades have been produced.
- Byte-bound prompt fitting uses substantially less than the 32K context ceiling;
  this limits the interpretation of raw-history performance.

## Two-week objective

Produce a complete, auditable comparison on the fixed 14-question pilot, with a
documented grading method, preparation and answering costs, representative error
analysis, and evidence suitable for Progress Report 2. Establish a feasible main
experiment plan for weeks 7–9. Larger runs and hybrid memory are not prerequisites
for this milestone.

## Week 5: finish the pilot and establish evaluation

| Work block | Action | Evidence / completion condition |
|---|---|---|
| Days 1–2 | Record starting coverage; restore local Ollama; resume summary with the original configuration and `--retry-errors`. Run approaches sequentially. | Existing successful answers remain preserved; checkpoints resume; coverage and failures are inspected directly. |
| Days 2–3 | Allow remaining summary inference to finish while reviewing completed records and measuring observed update time. | Target: 14/14 successful records per approach, 42 total. If interrupted, record actual coverage and the reason. |
| Days 3–4 | Audit IDs, manifests, shared settings, saved prompts, usage fields, timing boundaries, and cached preparation costs. Regenerate comparison exports and reports. | A record of the audit and an up-to-date cost comparison, with measured denominators and missing values retained. |
| Days 4–5 | Select and document a consistent grading route using the pinned official rubric; try a small calibration set before grading the pilot. | Judge identity/settings or human-review procedure recorded; reference/hypothesis pairs traceable to saved predictions. |

The existing evaluation route uses an external model judge or a separately served
large local judge. Its feasibility and any paid-use authorization must be resolved
before invoking it. A useful fallback is a separate manual assessment of all 42
answers using the official rubric, with an independent second review where
available. Label it as manual assessment and record disagreements; do not import
invented official evaluator logs. The current importer expects actual official
JSONL labels. If grading remains unavailable, report costs and qualitative errors
and explicitly identify accuracy as pending.

The runner's inference source files are fingerprinted in saved manifests. Finish
the existing pilot before changing them. A fitting or memory-method change belongs
in a separately named experiment; do not overwrite or combine incompatible runs.

## Week 6: interpret the pilot and prepare evidence

| Work block | Action | Evidence / completion condition |
|---|---|---|
| Days 1–2 | Complete consistent grading or the explicitly labeled manual assessment; inspect all 14 matched question sets. | Per-question judgments and supporting evidence; overall/category counts with denominators if fully assessed. |
| Days 2–3 | Investigate errors using reference-supporting turns and the exact supplied memory. | For each reviewed failure, distinguish missing evidence, stale information, compression loss, reasoning/interpretation errors, and abstention behavior. Record uncertainty and multiple causes. |
| Days 3–4 | Produce comparison tables and charts; investigate prompt fitting as a bounded methodological task. | Prompt tokens, answer latency, preparation cost, and total cost reported separately. Accuracy charts only when supported. Tokenizer feasibility decision recorded. |
| Days 4–5 | Draft Progress Report 2 from actual new work and assemble a video demonstration. Freeze the next experiment's scope. | PDF report, 2–5 minute video, linked evidence files, and a documented weeks 7–9 experiment plan. |

For error analysis, compare the same questions across approaches. An annotated
session being retrieved is not proof that the specific answer-bearing turn reached
the model. Record whether evidence was present, then assess how the answer used it.
Review preference questions against the category rubric rather than exact match.

The fitting investigation should first establish whether an exact tokenizer can
be matched to the installed model and runtime template. If feasible, validate it
on representative prompts and plan a separate rerun of all three methods under
the revised shared fitting policy. If it cannot be verified within this period,
retain the current method, disclose its limitation, and defer the change. Do not
let this investigation block completion of the original pilot.

## Reporting and demonstration

Keep a dated log of new work so the report distinguishes inherited implementation
from work performed during this reporting period. Support claims with coverage,
individual records, validation results, tables, charts, and reviewed failures.
Saved local artifacts satisfy the evidence requirement; a GitHub link is one
possible evidence type, not a prerequisite in the supplied instructions.

A practical video sequence is: research question and methods (30 seconds), shared
configuration and resumable workflow (45 seconds), saved comparison results
(60 seconds), one matched error example (60 seconds), and limitations/next steps
(30 seconds). Demonstrate saved results rather than waiting for long inference.

## End-of-week-6 acceptance criteria

- Target: all 42 pilot predictions complete and provenance checked. Any remaining
  failures or pending IDs explicitly listed.
- Grading or manual assessment method documented; ungraded answers never presented
  as accuracy results. Only two questions per category means category findings
  remain exploratory.
- Comparison artifacts refreshed; current-run and original cached preparation
  costs distinguished; unequal coverage and warm-start differences disclosed.
- Representative failures traced to actual supplied evidence.
- Progress Report 2 and video describe work that actually occurred.
- Main experiment sample size and compute budget selected from measured costs.

## Fit within the remaining 12-week schedule

Weeks 7–9: freeze the protocol and run the main comparison on a preselected,
category-balanced sample whose size is feasible from measured summary preparation
costs. Preserve the pilot IDs and use a separate larger-sample configuration.
Estimate both inference and grading work before committing to all 500 questions;
document any reduced scope relative to the proposal's full-comparison ambition.

Week 10: deepen error analysis and, if the main comparison is complete, test one
targeted improvement on a separate evaluation set. Dates are already preserved
in the current implementation, so adding timestamps alone would not be a new
improvement.

Weeks 11–12: finish analysis, reproducibility documentation, the final written
report, and the presentation. Keep hybrid memory optional until the three-method
comparison and reporting are secure.
