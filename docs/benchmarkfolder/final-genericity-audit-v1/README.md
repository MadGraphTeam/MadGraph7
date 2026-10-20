# Bounded production genericity audit

No new process-name, model-name, benchmark-case, fixed-seed, fixed-point or stored
physical-input dispatch was found in the reviewed production implementation.
The source census binds all changed paths in the generator/templates/interface
scope relative to the untouched original revision. It records current file
hashes and the small set of text matches for manual classification; its regex
search is a supporting census, not a semantic proof.

The source-only review ran during the allocation experiment's quiet window.
It generated no process, changed no implementation, ran no test or compiler,
and started no numerical engine. Previous sealed audits remain unchanged.

## Dispatch and fallback findings

- **Fortran traversal changes** operate on polynomial ranks, coefficient counts
  and array dimensions. Loop interchange keeps each output's accumulation order.
  Finite sparse QP skipping is guarded by runtime operand classification and
  rounding checks; exceptional inputs use the dense updater. No process or
  benchmark identifier controls selection.
- **QP FFV factorization** recognizes two generated fermion-vector vertex
  orientations at tensor ranks 2–4. The actual stored derivative rows must obey
  the signed real/imaginary relations at runtime, with bounded finite operands,
  supported binary128 layout, nearest rounding, disabled traps and sticky entry
  INEXACT. Otherwise it calls the generic public updater. The 4-by-4 dimensions
  are spinor dimensions, not the number of final-state particles.
- **Terminal contraction selection** follows the complete generated wavefunction
  graph, rejects aliases/unknown graph members and requires roots with no further
  wavefunction consumer. The exporter separately checks every rendered COMMON
  consumer and regenerates ordinary calls when that proof fails. Selection uses
  graph structure and numerical guards, not any hardcoded diagram number.
- **Compensated DP retry** is an explicit MadLoop retry for its currently chosen
  backend, with unchanged stability checks. The special callback is T3/mindeg=2
  at ranks 2–8 and a safe IEEE/numerical domain. Every other callback and unsafe
  input delegates to the pinned upstream implementation. These are tensor and
  arithmetic capability conditions, not favored external states or momenta.
- **Bridge evaluation and state** receive n, rank, all tensor coefficients,
  momenta, squared masses, invariant matrix and scale from the caller. Caches use
  topology/rank/precision; all numerical inputs are rebound and accumulation is
  reset on every evaluation. Invalid, unsupported, unstable or nonfinite calls
  return status into the normal MadLoop rescue machinery. No production capture
  stream, expected output, seed, point number or benchmark filename is consulted.
- **Precision and supported range** remain explicit: native DP and compensated
  DP, genuine binary128-input 256-bit generic rescue, with rank at most n+2 and
  first validation on Linux x86_64. Numeric library capability and guarded
  optimization limits are not a claim of support for arbitrary ranks/platforms.
  The full ordinary path remains available outside private optimization domains.
- **UFO/model precision fixes** repair generic generated REAL/CMPLX kinds and the
  shared MP continuation constant. They do not dispatch on loop_sm, the EW model
  name or a benchmark process. Born/interference/loop-induced/mixed-order/UV/R2
  assembly stays in the existing general MadLoop flow.

Physical fixture names and deterministic random seeds in Rust regression modules
are cfg(test)-only. The installed Fortran smoke fixture is a separate installation
test and cannot select a production amplitude path. An existing ggH mention in
`loop_exporters.py` explains the general empty-color-projection case in a comment;
its branch tests an empty color-flow list, and predates this integration.

The new challenging 2-to-4 QCD/EW campaign is independent validation of these
general paths. This source review does not predict agreement or performance for
those processes, establish arbitrary-model correctness, or replace their actual
CLI generation, fixed-point comparisons, rotations/scales and rescue checks.

The separate numerical audit also records unexplained microscopic generic-engine
internal history dependence below the public ASCII output resolution. That is
not a process-specific dispatch and is not silently treated as exact raw-state
equivalence here.
