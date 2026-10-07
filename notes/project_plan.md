# Current project plan — midpoint, Weeks 1–7

Updated October 6, 2026. The user corrected the reporting scope from Weeks 5–6
to the cumulative Weeks 1–7 milestone. Work is organized by verified milestones,
not invented week-by-week completion dates. The earlier 12-week overall deadline
is the working assumption; the original proposal had a 14-week schedule.

## Assignment scope

The midpoint report includes all seven required sections: project overview,
work completed during Weeks 1–7, evidence, progress against the proposal,
challenges and solutions, current status, and the plan for the second half.
The supplied video instructions also require a 3–5 minute narrated demonstration
of the actual implementation, settings, saved outputs, and next steps.

## Verified midpoint implementation

| Component | State | Evidence |
|---|---|---|
| Study design, local environment, dataset and fixed pilot | Complete | Configuration, dataset summary, recorded model check |
| Three memory methods and shared resumable runner | Implemented | Source code, tests, saved synthetic smoke outputs |
| Raw-history and retrieval pilots | 14/14 each | Provenance-checked question records |
| Summary pilot | 2/14; third history at 28/55 updates | Checkpoints and audit; no held runner lock at audit |
| Official grading | Integrated, not executed | Pinned evaluator and strict import/export code |
| Cumulative evidence tooling | Implemented | Offline audit, matched-question statistics, blank review worksheet, and saved figures |
| Unit validation | 17 tests pass | results/midpoint/test_validation.txt |
| Midpoint report | Written from current evidence | reports/Midpoint_Project_Report.pdf and .docx |

The midpoint update preserves the existing inference source files and configuration
so their fingerprints continue to match the saved pilot. It adds audit, evaluation
support, and reporting capability. It does not claim to have finished summary
inference or produced official grades.

## Remaining milestones

- Week 8: finish summary preparation and answer generation using the existing
  checkpoints; inspect all 42 records; establish and apply a consistent grading
  procedure. If official judging is unavailable, distinguish manual assessment
  from official grades and document its procedure and limitations.
- Weeks 9–10: verify tokenizer/fitting improvements before freezing the main
  experiment protocol. Select a feasible, category-balanced larger sample and
  record compute/grading estimates. Run sequentially under consistent settings.
- Week 11: complete category and evidence-level error analysis; test one targeted
  improvement on a separate evaluation set if the core comparison is complete.
- Week 12: final report, figures, reproducibility materials, and presentation.

The original proposal allocated writing/presentation to Weeks 13–14. A Week 12
target compresses that schedule; hybrid memory remains optional. A full 500-question
run is an ambition subject to measured resources, not a completed milestone or
a prerequisite for this midpoint report.

## Reproducible operations

Use `sh scripts/refresh_midpoint.sh` to refresh offline evidence. It does not call a
model. `src/audit_progress.py --review-output NEW_PATH.jsonl` creates a worksheet
without overwriting prior reviews. Reports and figures are saved evidence
snapshots; use the editable Word report for revisions and verify its claims
against the refreshed comparison. Optional report-authoring sources and old drafts
are retained locally and in Git history rather than in the published file list.

When ready to resume benchmark inference, start Ollama with the recorded runtime
settings and run the existing `scripts/run_pilot.sh`; it retries unfinished records
and preserves successful ones. Verify model identities and configuration rather
than bypassing manifest checks. Method changes require separate experiment outputs.
