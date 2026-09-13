You are the executor. Implement the earlier `PLAN_COMPLETE` contract as a checklist. Preserve every requirement and invariant, and run the associated validation.

Before further edits, record the current commit with `git rev-parse HEAD > /tmp/switchyard-review-base`. This preserves the task's starting point even if you commit later.

You may deviate when execution reveals contradictory repository behavior, a changed failure signature, an invalid assumption, or a necessary implementation constraint. Before acting on a deviation, emit `PLAN_DEVIATION` with the new evidence, affected contract items, revised approach, and required validation. Never silently omit a contract item.

Before finishing, run the final relevant checks again. Prefix each check's shell command with the exact comment `# switchyard_review_test_evidence` so its output reaches the reviewer.

Then materialize the complete task patch with `git diff --binary "$(cat /tmp/switchyard-review-base)" -- > /tmp/switchyard-review.patch`. Split patches larger than 24 KB into numbered files. Show every file in a separate tool call whose shell command contains the exact comment `# switchyard_review_patch_chunk`. Do not summarize or omit patch content.

After inspecting that patch, emit a completion ledger that marks every contract item as satisfied, changed with rationale, or blocked. Include the validation evidence used for each item.
