# Executed final variant diagnostics

The preparation state described in `README.md` was followed by actual private
compilation and execution on CPU190. `ledgers/{prepare,run,compare-rotation}`
retain the six-hour phase limits, controller identities, every directly
launched child command, results and child-exit checks. All three phases
exited successfully. Every output remains bound to the two actual final CLI
parents and the complete final reduction runtime.

All twelve streams passed the unchanged comparison policy:

| Diagnostic | Backends | Coefficient comparisons | Result |
|---|---|---:|---|
| Rotated HH | 1, 6, 7, 8 | 480 | Pass |
| Changing top mass and scale | 1, 6, 7, 8 | 480 | Pass |
| Reused versus finalized NinjaRS contexts | 8 | 80 | Pass |
| Near-singular top-pair points | 1, 6, 8 | 240 | Pass |
| Rotated versus fresh unrotated HH first ten points | 1, 6, 7, 8 | 320 | Pass |

The unrotated oracle comes from the freshly source-bound 100-point final
candidate HH collection; only its byte-identical first ten input points are
selected. `rotation-reference.json` and `rotation-reference-inputs/` retain
both the complete original source bindings and the separately selected rows.
Old candidate rows were not used.

NinjaRS reported stable DP status218 at all rotation, mass/scale and
context-reset points. Those drivers assign both independent DP and MP model
state variables, but these successful DP results alone do not establish
executed MP rebinding. The separate forced-MP diagnostic addresses that
coverage. The near-singular case did exercise automatic full QP rescue:
CutTools returned329, Ninja328 and NinjaRS322 at all ten points, with
agreement and no rejected measurements.

These short fixed-point measurements are correctness and stability
diagnostics. They do not enter the runtime acceptance gate.
