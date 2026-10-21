# Independent final focused correctness audit

`receipt.json` records an independent read-only audit of the final complex-mass
and mixed-coupling controls. It invokes no compiler, numerical worker or test
suite. The existing live source-role/runtime validators and complete-series
reader validate all retained inputs; `policy.compare_rows` supplies the unchanged
physical comparison tolerance.

The complex control uses the four new final candidate backends in
`../final-focused-controls-v1/runs/complex`. The mixed control uses the three new
legacy candidate backends in `runs/mixed-legacy` and the already measured final
NinjaRS target in `../final-representative-v1/runs/mixed`. Every candidate has the
full reducer descriptor `8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff`.
No historical candidate measurements are reused.

Each case also includes the three corrected-original helper backends selected
by `../final-focused-reference-reuse-v1/receipt.json`, whose exact SHA256 is
checked. The audit independently revalidates their live source, library, module,
executable and original MODEL-correction bindings. Historical complex helper
measurements contain two persistent commands and no fresh-process command;
that limitation remains explicit. All eight final candidate targets and the
three mixed helper targets have exact fresh/persistent rows and status parity.

Both persistent commands are compared across all 21 pairs among the seven
physical groups. Every point includes the total and every split order, with
all four coefficients: Born, finite, single pole and double pole. There are
20 rows per complex target and 50 rows per mixed target. The common scale at
each point is the largest absolute Born/Laurent coefficient across those seven
groups and all their orders. Original uncorrected model results do not enter
that scale. The tolerance is exactly
`1e-7 * max(abs(a), abs(b)) + 1e-10 * S`.

All **11,760 scalar comparisons pass**. The largest fraction of tolerance is
`1.7102944298016923e-5` for complex-mass point 8, total single pole, CutTools
versus Collier. The mixed maximum is `9.075192625481982e-6` for point 1, split
order 4 single pole, CutTools versus NinjaRS. Every target has ten DP point
statuses in each command: `211`, `216`, `217` or `218` for its selected backend.
No rescued point is observed in these two controls.

The first read-only audit attempt is retained as `audit.py`, its stdout/stderr
and `failure.json`. It stopped while reporting historical generation metadata:
an independently verified original archive has a source identity rather than
the newer `tree_state_sha256` field. `audit-v2.py` records both identity forms
without changing any validation, input, numerical result or tolerance. It exits
zero and writes the final receipt. Both attempts are retained unchanged.

This ten-point focused evidence does not replace the eight-process 100-point
gate or establish any performance claim. The receipt binds the precise reader
sources, raw results, helper receipt and runtime inventory. `manifest.json`
and `retained-files.txt` identify the complete audit package; they include no
compiled binary or cache.
