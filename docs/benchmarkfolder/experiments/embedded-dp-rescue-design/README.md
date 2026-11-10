# Embedded DP rescue: source-only experiment design

**Classify the current full-QP rescues before enabling another numerical
path.** This is a hypothesis and measurement design, not an implementation,
timing result, or execution authorization. No production or registered
campaign source, card, library, protocol, or acceptance criterion changes.
The current formal campaign takes priority over every proposed job here.

The closed 100-point numerical records for `split-dp-terminal-1` report:

| Process | NinjaRS accepted DP / QP | TIRust accepted DP / QP |
| --- | ---: | ---: |
| `u u~ > z g g [virt=QCD]` | 100 / 0 | 87 / 13 |
| `e+ e- > t t~ a [virt=QED]` | 100 / 0 | 82 / 18 |

These are final public-call classifications from the untimed numerical phase.
They do not count internal subintegrals, identify why rescue occurred, or
establish either reducer's runtime. Cold initialization elapsed times are not
used for a performance inference. `source-audit.json` binds the records and
the source inspection to the registered native bridge identity.

## What the pinned options actually do

The pinned upstream revision is
`854b79dcb7fffcb35e92b4a8a3dd443369b798c4`, with the separately pinned
full/poles patch. The bridge calls `TensorWorkspace::new_embedded` with an
`f64` prototype and `EmbedderOptions::default()`. Thus method choice is
already enabled; partial rescue and precision tiers are disabled. The
registered bridge is `eb3d67fd…`; it is not one of the historical profiling
libraries below.

| Option | Actual numerical scope | Implication for a future experiment |
| --- | --- | --- |
| `method_choice=true` | After direct conditioning refusal, retry at the same 53-bit precision using the a priori triangle/box choice, small-Gram expansion or alternative PV. Initial gate defaults to `1e-12`; method-choice gate is `1e-15`, rank boost 12. No internal rotation certificate. | Already on. Enabling it is not a new optimization. Do not lower gates or disable it without a separately justified experiment. |
| `partial_rescue=true` | In the first DP pass, eligible **non-top** triangle/box PV subsets with failed conditioning are recomputed in double-double, then their coefficients are rounded to DP and injected into the remaining DP recurrence. | Smallest candidate if such subsets cause the observed rescues, or expensive method-choice retries. It is mixed 53/106-bit work, not pure DP and not QP. |
| `precision_tiers=true` | After method choice, remaining conditioning-refused lanes run at 106, 256, then 512 bits as needed. Exactly lift the supplied **53-bit** inputs, round accepted outputs back to DP, and leave host checking to MadLoop. | Separate, broader candidate for native conditioning refusals; can duplicate expensive work and must report its actual tier. It does not replace the genuine 113-bit-input QP path. |

The partial-rescue implementation is narrower than its short API comment:
`direct/exec.rs::geometry` requires no jet and `lambda_shift==0`, a
two- or three-dimensional Gram system, and a subset other than the top
subset. An exactly singular solve returns before the partial-rescue branch.
Consequently it does not directly rescue a top-level triangle or box gate,
an exact singular pivot, a pentagon/hexagon-specific failure, or a scalar
master failure. In the method-choice pass the Gram-expansion branch has
precedence over the partial-rescue branch. It may improve internal triangles
of boxes and internal triangles/boxes of higher-point integrals, but the
current 13/18 final-QP counts do not establish that those cases occur.

Embedded partial rescue uses double-double gate `2^-66`; the embedded tier
gates are `2^-66`, `2^-216`, and `2^-472`, with eight bits of margin when
choosing the first tier. A zero conditioning estimate starts at 256 bits.
These are the embedded path's existing constants, not the standalone
facade's different double-double headroom/rotation policy. No proposed arm
changes these thresholds. Native invalid-input/master failures are not
silently treated as conditioning retries.

Neither option is triggered merely because a successful native DP result
later disagrees with its MadLoop reversal/rotation. In particular, enabling
partial rescue or tiers is not an established fix for the retained default
correlated-error example. Higher precision cannot reconstruct information
lost before the DP inputs were formed. The separate genuine QP path must
retain its 113-bit decimal input parsing, 256-bit working arithmetic,
binary128 output and independent host acceptance.

## Existing evidence and missing attribution

The retained n1 bridge profiles counted direct/method-choice/tier/refused
routes and upstream geometry/master/recurrence time. All observed n1 calls
were direct DP, with no failures or method-choice/tier activity. Their
direction-workspace improvement is already incorporated. Repeating those
profiles does not explain the n3 rescues.

The historical bridge fixture qualification retained 14 native DP refusals
among 27 method-choice stress fixtures and successful QP evaluation. Those
fixtures and the correlated-error host-helper example are useful boundary
controls, not evidence about the current physical n3 points. The separate
QP scalar-shadow pilot found repeated scalar keys, but their matched work
was only 14.29% of observed scalar-kernel work in its small historical EW3
subset; it did not measure parent/amplitude time or qualify a cache.
It is a different hypothesis from avoiding full-amplitude QP rescue.

The existing C ABI already exposes native status and `ml_tirust_last_error`.
Status 1 combines upstream `Unstable` and `MasterFailure`; status 2 covers
invalid inputs and other errors. `dp_checked` also turns nonfinite returned
DP tensors into failure and clears the numerical store/directional
workspaces. Capture the error immediately after failure: every guarded API
call, including statistics/clear/budget calls, resets the error string.
The error getter itself only copies the thread-local string.

This text is not necessarily the complete retry history: an unsuccessful
method-choice attempt appends its own status to upstream `last_diagnostics`
but retains the original direct failure in the returned lane status. A later
master failure can therefore be absent from the C error text. The native
return/error pair identifies the public failure; full internal attempt
attribution needs the existing Rust diagnostics through a private observer.

Native status alone is insufficient. Adapter validation/initialization can
return before a native call; mapping or final contraction can also fail.
The common TIR wrapper converts failures to a reduction-FPE flag. The host
resets both precision flags before QP rescue, so final code `323` and final
cohort IEEE flags lose the DP cause. Capture point, group, squared order,
CT mode, native-call presence/status/error, adapter exit stage, and the
already computed DP `ACC`, `DP_RES`, and FPE flags **before** that reset.
All native calls successful, no adapter/FPE failure, and over-threshold
accuracy identifies a host-consistency rescue; otherwise keep multiple
contributing causes rather than force one label.

Reusable patterns already exist: the lifecycle GNU `--wrap` observer records
native calls without changing the numerical DSO, and the historical QCD
kinematics observer demonstrates insertion-only stage/status/error hooks.
Neither already supplies all these current-DP records. A private adapted
observer needs observed/unobserved bitwise output, status, IEEE and complete
trace parity before its explanation is accepted. Never read failed output
buffers. No instrumentation is added to the registered acceptance workers.

Upstream Rust exposes `last_lane_routes`, `last_diagnostics`,
`engine_execution`, `set_profiling` and `last_profile`, but the production C
ABI exposes none of those route/profile records. The historical profiling
DSO is a reusable design, not the current binary. `LaneRoute::Direct` may
include partial DD rescue, so route counts alone cannot certify pure DP.
The C cache counters omit upstream DD-master counters; even Rust DD cache
counters cannot count uncached/check-pass DD execution. A partial-rescue
experiment needs a separate bounded private counter/journal for actual DD
subset evaluations, reuse, and failures, with an independently demonstrated
observation-only control. An enabled option is not evidence it was exercised.

## Bounded sequence after the formal campaign

1. **Cause-only pilot, unchanged numerical library.** Use private copies of
   the two current n3 exports, original 100-point order/cards and the full
   prescribed warmup/filter history. At most one unobserved/observed pair per
   process, each with a preregistered wall limit and no automatic retry. Keep
   all 100 final statuses and trace every native visit associated with the
   31 existing QP points plus controls; do not replace the original stream
   with independently initialized selected points. Use the shared lock and
   a root-approved quiet window; this is correctness attribution, not timing.
2. **One candidate at a time.** If refused non-top triangle/box subsets or
   their method-choice cost dominate, compare the immutable control with
   only `partial_rescue=true`; retain method choice, gate defaults, master
   provider, finite/pole routing, input formation and host policy. If top
   conditioning refusals dominate instead, a separately registered tiers-only
   candidate may be justified. Do not sweep thresholds or enable both as an
   unexplained bundle. If all failures are native-success consistency/FPE
   cases, these flags have no demonstrated causal target; stop this proposal
   and investigate the observed source of disagreement.
3. **Correctness before speed.** Keep `MLStabThres=1e-6`, original independent
   direction checks and the same external `3e-6`/ordinary `1e-8` criteria.
   Compare all Laurent coefficients and accepted statuses against retained
   references. Preserve failures and every DP/QP migration, with genuine QP
   still available. Include the retained correlated-error fixture, top-gate
   refusals, and one demonstrated eligible non-top DD subset as controls.
   Full physical parity and precision transport remain separate checks.
4. **Cache and precision accounting.** Use distinct contexts/processes for
   each immutable option set. Primary/check workspaces remain separate;
   check passes detach **all** numerical stores, including partial DD
   stores. Qualify on/off, repeated primary versus reversed checks,
   changed mass/scale/shifts, bounded/zero budget, failed-call clearing and
   finalize/recreate. Pole-only mode remains unchanged and separate. Report
   pure-53 work, mixed-53/106 subset work, 106/256/512 whole-lane retries,
   and explicit 113-bit-input QP separately. A DP-interface success after DD
   or MPFR work must not be labeled native-DP-only; DD is never labeled QP.
5. **Only then a bounded uninstrumented runtime pilot.** At most the control
   and one justified candidate, original complete 100-point traversals,
   seven alternating warmed paired batches, approximately two seconds per
   batch, existing eligibility floor and two-sided confidence policy.
   No concurrent owned compiler/native work. Count extra work on formerly
   fast DP points as well as saved QP reruns; fewer QP codes alone is not a
   speedup. A gain needs a new registered implementation and all required
   physical/regression gates before promotion, never a retry of the frozen
   current formal result.

The maximum opportunity is the measured cost of the avoidable full-QP reruns,
minus added work on every affected DP tensor and check. Neither that cost nor
the avoidable fraction is known yet. The 13%/18% final-QP fractions are not a
speedup ceiling or a prediction.

`source-audit.json` records hashes and inspected line anchors; `index.json`
binds this package. All observations here come from reading existing source
and retained text/JSON. No compiler, native reducer, test suite or timing was
run to prepare it.
