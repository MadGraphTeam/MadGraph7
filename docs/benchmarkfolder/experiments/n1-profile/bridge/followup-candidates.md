# Remaining bridge costs: source audit only

These are hypotheses after the split-workspace experiment, not implemented
changes or measured gains. The production source under review has SHA256
`455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c`.
The split/original whole-amplitude improvement does not close the baseline gap.
The Fortran allocation experiment has higher priority because the original
adapter's seven heap allocations and frees lie outside its detailed timer
brackets. Changing only its compiler optimization to O3 did not establish a
gain: the diagnostic interval includes unity.

| Candidate | Exact behavior and constraint | Relevant observed cost |
| --- | --- | --- |
| Reserve DP input capacity | `dp_batch` collects 4N complex shifts, then appends N(N-1)/2 invariants, N masses and one scale. Reserve `5N+N(N-1)/2+1` before packing to avoid vector growth. Keep supplied-invariant checks, ordering and all finite/width checks. | Complete input validation/packing is about 0.49 us per amplitude in the original level-1 profile; only a fraction is allocation. |
| Avoid repeated same-budget C calls | The private Fortran context can retain its last successfully applied byte budget; invalidate that marker on creation/finalization and preserve negative/error handling. The Rust setter already avoids clearing unchanged budgets, but still crosses the ABI and locks the state. | About 0.10 us per amplitude in the original level-1 profile. |
| Retain output packing scratch | A context/workspace-owned buffer could avoid the fresh `pack_dp` allocation, but values must be completely validated before any caller output is written. Preserve slot filtering, unevaluated finite padding, failure cleanup, and no partial output on errors. | Allocation/packing about 0.19 us and final copy about 0.08 us per amplitude originally. |
| Avoid unchanged store attachment | Retain attachment state per directional workspace and only change it when cache-enabled mode changes; new workspaces and failed/recreated workspaces must start with explicit attachment. Budget and numerical-cache clearing change shared store contents, not its identity. | About 0.16 us per amplitude originally; retaining attachment must not retain numerical results for checks. |

All costs include timer overhead and are descriptive, not independent additive
upper bounds. None of these small regions alone explains the remaining
whole-amplitude gap. Do not remove integrity checks or independent-check store
bypass to save their cost.

A useful experiment changes one concern at a time in an isolated bridge or
wrapper, reuses the 90-call input/mode/budget/failure validator, and checks native
QP and pole-only smoke as appropriate. A budget-cache test must additionally
cover rejected negative updates, a changed valid budget, unchanged repeated
settings, finalization/recreation and calls from two generated subprocesses.
Only a separate uninstrumented whole-amplitude comparison can establish an
application gain. Final acceptance requires the frozen cross-backend numerical
and runtime policy across the full process set.
