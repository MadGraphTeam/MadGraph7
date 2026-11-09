# Production qualification outline for the CMS CutTools diagnosis

The retained experiments identify two causes of the CMS point-20 CutTools
disagreement, but they do not yet qualify a production correction. No production
source change, compiler invocation or native evaluation is made by this audit.
The original `complex-mass/` evidence and artifact index remain unchanged.

The existing evidence supports the following limited conclusions:

| Evidence | Established | Still missing |
|---|---|---|
| Scalar-cutoff intervention | CutTools' independent absolute `1e-6 GeV²` cutoff erases physical `t≈-2.5e-7 GeV²`; `1e-9` and `1e-10` give identical corrected results with identical scalar inputs | A general scale/precision policy and automatic checked results |
| QP momentum intervention | Preserving repaired QP denominator shifts removes the remaining point-20 discrepancy; other scalar algorithms, masses and numerator remain unchanged | Safe DP initialization, automatic rescue, ordinary and other-process regression evidence |
| Independent QED double pole | The four massless charged leptons give the MadGraph FKS charge-sum residue; the repaired diagnostic agrees at recorded precision | Independent finite and single-pole checks, and complete point-21 qualification |
| Primary automatic campaign | The original first 19 points pass the retained `3e-6` comparison; the 12 ordinary points meet the stated target | Complete extreme-point automatic qualification; the original 21-point CutTools attempt was interrupted, not shown numerically wrong by that interruption |

All four private QP-transport observations return status `120` and accuracy
`-1`. Their `CTModeRun=4/5/6` agreement, including direction reversal, is not a
checked automatic result. The successful scalar-cutoff/QP-transport combination
agrees with the retained QP references to at most `2.11e-11` relative, but the
original primary extreme-point failures remain unqualified. Preserve both
those failures and every private intervention as separate observations.

**Production boundaries.** The optimized adapter declares DP `PL/PCT` and
complex-DP masses/results in
`madgraph/iolibs/template_files/loop_optimized/CT_interface.inc:25` and calls
legacy `CTSXCUT` at line 114. The generated `helas_loop_amplitude.inc:110`
already constructs `MP_PL` when doing QP, but the CutTools dispatch at lines
142 and 182 sends the DP array for both interference and loop-induced paths.
The native signature in `vendor/CutTools/src/cts/cts_cutroutines.f90:69`
declares DP momenta; lines 305 and 346 promote already rounded denominator
momenta. Passing a REAL16 array to that existing symbol would break its ABI.

A Fortran-only correction can add a distinct `MP_CTLOOP` adapter and a distinct
QP-capable CutTools entrypoint while retaining the original DP symbol. The QP
adapter must receive the existing QP momentum array, perform QP prefix sums,
and preserve the current mass, scale, callback, result, normalization and error
interfaces unless separately qualified. The entrypoint may live in the
vendored CutTools build or a repository-owned Fortran companion built against
the matching internal modules; a superficial wrapper around the unchanged
DP `CTSXCUT` signature cannot recover lost bits. A companion must be packaged
and linked reproducibly, with matching `cts_mpr.h`, compiler/module ABI and
internal dependencies. Export capability checks must account for externally
provided old CutTools archives without the new symbol.

The private CMS prototype replaces `CTLOOP` in place and explicitly rejects
modes 0–3; it therefore cannot serve as the production implementation. The EW
private diagnostic demonstrates a split DP/QP dispatch shape, but does not
qualify its production behavior. Keep ordinary DP initialization and modes
1/2/3 on the old ABI. Define and test legacy mode 0 separately: its internal
DP-to-QP rescue does not automatically receive the host's repaired QP momenta.
The nonoptimized `loop/CT_interface.inc` and `loop/helas_loop_amplitude.inc`
are separate DP-only generation paths and require an explicit supported scope
or their own integration. Test both `UseQPIntegrandForCutTools` settings;
genuine QP denominator transport must not silently mismatch the numerator's
chosen kinematics. Masses and `MU_R` are still DP inputs in the demonstrated
prototype; it is not a claim of fully QP model-parameter input.

**Scalar-cutoff policy.** `CTSINIT` sets the absolute cutoff at
`vendor/CutTools/src/cts/cts_cuttools.f90:82`; the alternate QCDLoop path also
sets it for OneLOop rank-one/two bubbles at lines 91–95. The public
`avh_olo_forCutTools::olo_onshell` setter calls the DP, QP and MP setters
(`vendor/CutTools/src/avh/avh_olo.f90:35251`). Consequently a post-initialization
call changes ordinary DP and all QP modes, including legacy internal rescue.
Precision-specific setters exist at lines 6551 and 18248; any deliberate
precision split must be source-bound and tested. The state is saved in the
scalar modules. Check initialization order, repeated initialization, mixed
backend calls and multiple subprocesses before assuming a setting stays local
to one amplitude.

For this pinned build, the actual archive/ELF symbols establish that the
CutTools scalar state is isolated from Ninja and COLLIER. CutTools uses
`__avh_olo_forcuttools_*` module symbols; Ninja uses its ordinary OneLOop
namespace with local DP/QP scalar state; COLLIER exposes no CutTools/OneLOop
symbol dependency and has its own module state. Thus this setter does not
name-bind to backend 6 or 7 scalar state in the inspected build. It still
changes the shared CutTools state for every subprocess in the same process.
Recheck other external-library/build configurations instead of generalizing
the observed isolation to them.

A precision-specific setter can be imported as
`USE avh_olo_forCutTools_qp, ONLY: qp_onshell => olo_onshell`. The outer wrapper's
similarly named `OLO_QP_ONSHELL` is a getter, not that setter. The inspected GNU
configuration uses kind 16 for the CutTools QP path; arbitrary-MP configurations
need their own module/type compatibility checks. The frozen CMS prototype ELF
contains `ctsxcut_qpinput_` and `ctsinit_` but lacks the original `ctsxcut_`,
confirming that its successful run does not exercise the legacy DP entrypoint.

Do not promote the point-specific `1e-10` value into a universal default.
At this point it lies above the observed DP external virtuality `2.64e-12`
and below physical `abs(t)≈2.5e-7`; that interval is not guaranteed at other
energies, angles or precisions. QP transport reduces the observed external
virtuality to `2.52e-29`. The still smaller physical transfer of point 21 and
explicit rescalings must be retained. Setting the public cutoff to zero is
also not a restoration of automatic relative tolerance: both setters
unconditionally set `nonzerothrs=.true.` before assigning the value. A
qualified policy must distinguish physical small invariants from roundoff,
preserve exact/scaleless limits and complex-mass branches, and document what
happens when the two scales cannot be separated reliably. Keep the original
policy available in new evidence so each intervention remains attributable.

**Required automatic qualification.** Use `CTModeRun=-1`, the documented
exhaustive host policy, with the frozen physical stream, model, widths,
renormalization scale, card settings and requested tolerances. Record a separate
CutTools-only run and the materialized production rescue chain, including the
actual reducer/precision used per point. Positive fixed modes are insufficient;
legacy mode 0's internal checks also do not substitute for MadLoop's checked
return classification. The host collects direct/reverse evaluations and the
configured rotations at
`loop_optimized/loop_matrix_standalone.inc:1713`, computes accuracy and enforces
QP FPE failures at line 1784, and advances DP-to-QP rescue at line 1904.
The unchecked path assigns `H=1` and accuracy `-1` at line 1983. Require final
accepted `H=2` or `H=3`, completed `T=1` or `T=2`, finite nonnegative reported
accuracy and the intended backend use; retain failures rather than weakening
the acceptance threshold. Fixed-QP `120` combined with forced stability checks
can be remapped to initialization code `140` and stall the existing worker, so
that protocol combination must not be used as the production certificate.

First reproduce the original source/card behavior, then compare four separately
bound implementations: original, cutoff-only, transport-only and both. Test
the retained points 20 and 21 individually and in the full 21-point stream,
including initialization and point-order changes. Restore the primary
`OSThres=1e-8` when assessing that primary policy; the private proofs used an
alternate card. CutTools' current entry does not consume MadLoop's
`BUILD_KINEMATIC_MATRIX`, so correcting CutTools does not itself repair the
other reducers' independent on-shell-guard behavior. Distinguish a qualified
CutTools result from complete five-backend primary-policy agreement.
The original automatic outcome must first be established: a fixed-mode wrong
answer does not show whether the automatic policy accepts that error, rejects
the point or fails initialization. If an inaccurate automatic result is
accepted, the candidate must demonstrate that the accepted error is repaired;
if the original rejects it, report the candidate's checked recovery separately.

Keep the original no-rotation policy for the direct before/after comparison;
add separately labeled nonzero DP/QP rotation checks. Verify that a wrong or
nonfinite native result is rejected/rescued, not merely that two correlated
evaluations agree. In addition to the FKS double pole, use an independent
finite/single-pole adjudication and higher-precision checks where necessary.
Agreement among related scalar implementations alone is insufficient.

**Further gates before promotion.**

1. Check DP and QP initialization, modes 1–6 and defined mode-0 behavior,
   `UseQPIntegrandForCutTools` true/false, direct/reverse/rotated kinematics,
   split-order/helicity reuse, event boundaries, loop-induced amplitudes,
   complex and real masses, thresholds, soft/collinear limits and scaleless
   residues. Retain exact input and native-output traces for representative
   boundary cases, including external shell residuals and the physical transfer.
2. Sweep small physical invariants across the old and proposed cutoffs; vary
   boosts, hard scale and renormalization scale, and scale dimensionful inputs
   consistently. A cutoff that fixes one 1 TeV point must not turn on-shell
   roundoff into a physical regulator or erase a genuine channel elsewhere.
3. Rebuild actual CLI exports and test installation/internal and external
   CutTools linkage, relocated libraries, mixed backend initialization and
   subprocess isolation. Check availability behavior for unsupported archive
   ABIs and scalar-library choices rather than silently claiming the new QP path.
4. Repeat the frozen main processes and relevant real/CMS controls with the
   exact MadLoop regression comparator and the ordinary-point target. Verify
   existing-backend outcomes and actual rescue usage. Run the agreed paired
   single-core performance campaign separately; the private fixed-QP elapsed
   times provide no evidence for the per-process 95% bound requirement.

This is a qualification outline, not a recommendation to change defaults now.
The decisive missing evidence is a checked result that resolves the retained
fixed-mode discrepancy and establishes the automatic outcome while preserving
ordinary correctness, rescue semantics and performance. Namespace-specific
source findings and audit hashes are recorded
alongside this outline; no frozen CMS artifact is rewritten.
