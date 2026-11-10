# Proposed bounded status diagnostic — not executed

The closed pilot identifies affected tensor topologies, but it does not retain
the exact C-boundary input arrays or per-call bridge status/error text. Saved
physical momenta plus generated wavefunction permutations are insufficient to
claim an already verified bit-identical reconstruction of every on-shell-aware
invariant. This proposal first captures that missing boundary evidence in a
fresh private diagnostic, then evaluates a small standalone tensor matrix.
It changes neither production nor the admitted pilot and requires a separate
native-stage admission.

1. Bind the original executable/library/source hashes, physical card, exact
   point 25 of Zgg and point 2 of ZZgg, scale, full finite/pole settings, and
   cache-disabled singleton backend cards. Work only in new private copies.
   Capture the adapter input at groups 59/61 (`N=5,R=4`) and 408/422
   (`N=6,R=5`), for the original, reversed and signed-rotation probes. Use the
   existing generated kinematic construction; do not substitute recomputed
   invariants or a different momentum convention.
2. Record branch-stage identifiers around existing adapter guards, then exact
   integer bit patterns of `PL`, shifts, supplied invariants, complex squared
   masses and `MU_R**2` immediately before the tensor call. Record the returned
   integer bridge status and copy `LAST_ERROR` **immediately**, before another
   bridge call can clear or replace it. Observe output-buffer bits only after
   success. Keep before/after software latches and any separately observed IEEE
   flags distinct. Do not reset flags or read uninitialized tensor storage.
3. Replay the resulting at most 12 captured tensor requests through the existing
   shared-library ABI. For each, compare one fresh-context evaluation with one
   same-context repeated evaluation: at most 24 tensor calls, serial, cache
   disabled, a fixed short execution/RSS cap, and full status/error/bit records.
   Register the existing plans and retain exact ABI/library/runtime hashes.
   No numerator contraction or amplitude calculation belongs in this stage.
4. Preserve every failure and timeout. A fresh-context success does not refute
   the original failure: the original context had structural workspace history.
   Reproducing that history would be a separately bounded and admitted follow-up,
   using captured requests rather than guessed state. Internal numerical truth
   would also require an independent higher-precision reference and remains a
   separate acceptance task.

This diagnostic can distinguish an early Fortran guard, a bridge error/unstable
status and a rejected nonfinite tensor. It cannot retrospectively turn the
closed unchecked-110 amplitudes into qualified stability samples. No build,
loader invocation, tensor evaluation or numerical source change was performed
to prepare this proposal.

## Separate proposed driver-boundary diagnostic

Direct tensor replay does not measure the effect of the host software latch.
A separate future private-driver study could record the incoming latch, then
start each independent amplitude probe with a fresh DP/QP **software** failure
flag, recording the reset and the resulting flags explicitly. The current
entry path is `STABILITY_EVALUATE` → `SLOOPMATRIX_THRES` → `SLOOPMATRIX` →
`SLOOPMATRIX_RETRY_IMPL`. In the bound Zgg generated source these calls are at
driver line 185 and matrix lines 3045/26; the matrix's automatic stability
branch is gated by `CTModeRun=-1` at line 1409, while this pilot uses positive
forced modes. Its software-flag reset sites are matrix lines 1567/1638/1653.
There is no per-probe flag reset on the current positive-mode path; this is
not a claim that production's automatic rescue path follows the pilot path.

Such a reset would be a new private diagnostic policy, never a silent repair
of the closed data. It must occur at an independent probe boundary, not inside
a reduction or between groups. Preserve bootstrap failures and all per-group
stability/nonfinite checks. An active group with `STABLE=.FALSE.` still rejects
its probe even after an initial latch reset; clearing the latch cannot turn
those contributions into valid results. Hardware IEEE flags would remain a
separate observation. Freeze any follow-up point set and labels before running,
retain the original unknown bounds unchanged, and keep this host-lifecycle
study separate from standalone tensor status replay and numerical truth tests.
