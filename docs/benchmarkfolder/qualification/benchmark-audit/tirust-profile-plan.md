# TIRust runtime attribution plan

This plan follows a read-only inspection of integration commit
`c91cb85f3aa942c90299b3518bb1f195f55334a5`. Pilot runtimes are not acceptance
evidence. Diagnostic builds and results must remain separate from the frozen
baseline, generated campaign outputs, installed libraries and paired gates.

## Scope and sequence

The generated `qcd_n1_uux_z` process has one loop group: a massless triangle
with three propagators and rank two. `NC = LOOPMAXCOEFS = 15`, so its output
slice is contiguous. It has one squared split order and reduces the summed
numerator, so the exporter does not enable its Fortran tensor cache. Neither
lower-rank array packing nor cache-hit contraction explains this process.

1. Preserve the first paired gate and its failures. Obtain the campaign
   controller's explicit release before compiling or running native diagnostics.
2. Clone the exact generated candidate into an ignored directory. Retain input,
   source and executable hashes, compiler commands, cards and the same 100 points.
   Hold `.tirust-build/numerical-tests.lock` exclusively during numerical work.
3. Count calls by `(CTMODE, propagators, rank)`, status and precision. Attribute
   Fortran time to initialization, ordered kinematics, on-shell-aware matrix
   construction, bridge call, mapping/validation and contraction. Record batch
   CPU/wall time and timer calibration; compare instrumentation enabled/disabled.
4. Attribute Rust time to packing, store attachment, reduction, output packing
   and the FFI boundary. Enable the existing `TensorWorkspace::set_profiling`
   only after workspace construction, then collect `last_profile()` and
   `last_lane_routes()`. This preserves embedded algorithm selection and reveals
   direct, method-choice and internal precision-tier use.
5. Select changes from measured costs, qualify numerical behavior, then rerun
   uninstrumented paired gates. Instrumented times are attribution evidence only.

## Candidates

Paths and lines refer to the integration revision above.

| Candidate | Location | Applicability and constraints |
|---|---|---|
| Reuse DP input storage and reserve its complete length | `vendor/tirust/src/lib.rs:449` | Every reduction currently collects shifts, then grows the vector for invariants, masses and scale. Preserve validation and exact values. |
| Reuse output storage or pack into the validated caller buffer | `vendor/tirust/src/lib.rs:475` | Avoid allocating/zeroing an intermediate vector and copying it at line 803. Preserve complete outputs, pole selection and failed-result handling. |
| Avoid repeated store attachment/registration | `vendor/tirust/src/lib.rs:106` | Upstream attachment re-registers nested gathers. Consider distinct reusable primary/check workspaces; never let checks read primary results. |
| Remember the applied cache budget in the shared context | `madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc:33` | Avoid a redundant C call and mutex acquisition. State must follow context generation and updates from every subprocess, not a subprocess-local saved budget. |
| Use contiguous storage for lower-rank outputs | `madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc:78` | Applies when `NC < LOOPMAXCOEFS`, not the n1 triangle. Existing native array-temporary warnings confirm the packing path. |
| Contract directly from cached Fortran coefficients | `madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc:312` | Only where Fortran caching is enabled. Preserve coefficient summation order, normalization, invalidation and finite checks. Measure layout changes separately. |
| Use generated coefficient counts | `madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc:232` | Replace the repeated rank sum with existing `NCOEF_R` after a supported-rank guard; likely small. |
| Reduce kinematic intermediates | `madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc:62` | Only if measurements justify it. Preserve ordered mass subtraction/addition, complex masses and sequential `OSThres` snaps from `CT_interface.inc:192`. |

## Numerical boundaries

Keep numerator construction and contraction in Fortran. Full and poles-only,
DP and QP, and primary and independent stability calculations require separate
result stores. CTMODE 2/5 must recompute rather than reuse primary tensor results;
failed reductions must never become successful cache entries. Do not disable
host stability checks, change the reduction algorithm or alter rescue policy to
obtain a faster profile. Retain native-DP and rescued results separately.

Mapping tables are already static `DATA`, ordered prefix sums are already
linear, and backend 9 already skips the legacy preprocessing block at
`TIR_interface.inc:172`. Repeating these changes cannot explain a new gain.
