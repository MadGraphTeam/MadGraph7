# Bounded genuine-QP conditioning retry

This is a source-bound design, not a production patch or a qualified adaptive
implementation. No compiler, reducer, or amplitude is run by this package.
The already authorized standalone 512-bit QCD replay remains a separate task.

The narrow candidate is one ordinary 256-bit QP evaluation followed by at most
one 512-bit evaluation of the **same binary128 input**, and only after a known
conditioning refusal. It leaves DP evaluation, MadLoop's independent direction
and rotation checks, model parameters, scalar formulas, supplied invariants,
on-shell classification, and effective-point/Born policy unchanged. It cannot
repair inconsistent input geometry or Born normalization. The private graph,
mass, and effective-point work must retain their separate qualifications.

## What the evidence supports

In the retained topology-C QCD test, the historical 256-bit bridge serves181
of232 group/direction calls and refuses51. All51 have nonzero exact supplied
and Cartesian Gram determinants;39 are massless. The native messages identify
the Gram conditioning gate, not an exactly singular pivot. This justifies a
bounded conditioning-retry experiment. It does not prove that more precision
is necessary or sufficient.

The first private512 process returned232 successful contractions and host323,
but its actual-library capture failed. Its numerical interpretation remains
provisional. All696 complex contracted slots are retained:46 exceed a3e-6
relative diagnostic in at least one real/imaginary component or complex norm.
The projected absolute-error sums are small relative to the host and group L1
magnitudes, and the host finite/single/double values agree with NinjaRS to about
7.25e-16 or better. This is contracted coverage, not every unit-numerator tensor
component, and NinjaRS is not an analytic finite/single truth reference. Only
the universal QCD double pole has a separate analytic check here. Exact squared
norms are explicitly serialized for the46 flagged slots; all other norms are
exactly reconstructible from retained rational components.

Even a successful corrected512 replay would test a uniformly512 calculation,
not this proposed mixture of unchanged256 successes and retried512 refusals.
The adaptive path therefore still needs its own same-input native comparison.
The separate EW512 experiment also shows that higher precision and admission
alone do not cure a supplied-invariant/Cartesian mismatch.

## Admission policy and arithmetic

Keep the current256 gate exactly `2^-136`. The private512 experiment uses
`2^-392 = 2^-(512-120)`. This changes the raw admission threshold. It preserves
the existing heuristic arithmetic-headroom relation

```
condition^coefficient_rank >= 2^-(working_bits - 120)
unit_roundoff / condition^coefficient_rank <= approximately 2^-120
```

The second line is a motivation, not a proved forward-error bound. It omits
recurrence constants, cancellation, scalar errors and representation effects.
The113-bit inputs have seven nominal headroom bits below120; this does not
guarantee113 correct output bits. Keeping the literal `2^-136` at512 instead
would preserve the same geometric refusal for these inputs and has no observed
benefit. No proposal relaxes the256 guard or retries at256 with a smaller gate.

Parse each existing64-byte decimal record at113 bits once, retaining signed
zero, then copy the exact value to the selected256/512 working precision. Never
round these inputs through binary64. Existing `UseQPIntegrand=false` modes keep
their explicitly promoted-DP numerator policy; genuine512 scalar/tensor work
does not upgrade that numerator or make it a genuine-QP integrand. Preserve
that distinction in telemetry and qualification. Reuse the existing
`set_input_precision(113)` on both workspaces; the lightlike tolerance
therefore stays `2^-100`. Fixed affine tests at40/33/80 bits are unchanged.
Working-precision-dependent roundoff checks naturally use their own workspace
precision; they are not frozen to a fabricated binary64 representation.

`DirectWorkspace<Float>` constructs the actual native scalar evaluators with
`prototype.get_precision()`. The poles-only executor likewise constructs
`ScalarPoleEvaluator<Float>` with that precision and skips full scalar kernels.
Both full and pole retry arithmetic can therefore be genuinely512-bit without
changing scalar backends. Existing finite-result decimal formatting and the
Fortran binary128 conversion remain the public output boundary.

## Minimal dispatch

Preserve the native outcome until eligibility is decided; the current C status1
alone is insufficient because it conflates instability and scalar failure.
The upstream public API already exposes `last_conditions()` for retry choices.
For the first experiment, admit only all of:

- The actual returned lane is `LaneStatus::Unstable`.
- Its exact pinned category is `ill-conditioned Gram matrix for ...:
  conditioning ...`, with a valid reported subset in the qualified N/rank domain.
- The same call's `last_conditions()[0]` is finite and strictly positive.

The recorded condition is already the quantity compared with the gate, whereas
the diagnostic text prints the unpowered condition. Do not exponentiate the
recorded quantity again. Never use the process-wide last-error string alone or
a stale condition from a prior call. A small private classifier over the pinned
native string plus the public condition is sufficient for this experiment;
unknown wording fails closed. A future typed upstream reason would be cleaner,
but is not a prerequisite for adding a new framework.

Do not retry exactly singular/zero conditions, underflowed condition telemetry,
relation/lightlike incompatibility, invalid input, unsupported mode, scalar
master failure, nonfinite output, formatting/conversion failure, preparation
errors, or panic/poisoned context. Some otherwise useful cases may be missed by
this deliberately narrow rule. Cayley/pentagon/six-point condition categories
are excluded initially because the retained QCD evidence does not qualify them.

On an uncached request, evaluate256. Return an ordinary256 success unchanged.
On an eligible refusal, discard that attempt's output and use a lazy512
workspace once. Return512 only after its own normal status and finite packing
checks pass. If it refuses or fails, preserve a non-success result and both
attempt diagnostics. No third tier, alternative method, hidden unchecked result,
or automatic retry of an unrelated failure is introduced. A successful retry
must clear the public last-error text rather than leaving the old256 message
beside success; retain the prior reason in explicit retry telemetry instead.

## Workspace and cache ownership

Share only the immutable `DirectReductionPlan` for each `(N,rank)`. Keep mutable
workspaces separate by `(N,rank, working_bits, primary_or_check)` and by full
versus poles-only executor type. Allocate512 lazily. The existing QP path uses
no upstream MPFR tensor/subintegral store; do not attach the binary64 store or
claim QP subintegral hits. Whole-result QP caching remains the only reuse here.

The exact result key includes input records, N/rank, pole selection, finite-mode,
input bits113, actual working bits256/512, and the fixed admission-policy
identity. Share the existing overall QP byte budget across both tiers and modes;
do not multiply it by the number of workspaces. Full and poles-only entries
cannot alias. The uncomputed finite placeholder in pole mode remains explicitly
unphysical and is never used for validation.

Primary requests may first consume an exact256 cache hit. Otherwise run256.
Only after that request's eligible refusal may an exact512 cache entry be used;
there is no negative cache or remembered permission to bypass256 on future
inputs. This keeps the first-tier contract simple, at the cost of repeating the
256 refusal before a512 hit. Independent check passes read and write neither
tier's result cache and use their own mutable workspaces. They independently
decide whether512 is needed; do not copy a primary success or its tier decision.

Insert only a completely successful packed result under its actual tier. A
failed tier's partial buffers are never a result or a cache entry. Discard both
direction workspaces at the failed tier for that plan/mode before reuse; retain
the immutable plan. Invalidate any matching primary result for the failed input
and tier when an independent check fails. If the complete adaptive request
fails, invalidate matching input/mode entries at both tiers and discard their
direction workspaces. These failure actions do not turn another precision or
full/pole result into a success. Context panic retains the existing poisoned
context error; do not catch it and continue inside the same state.

Event clear, budget changes, model/effective-point changes and finalize must
clear both precision result populations through the existing lifecycle. The
current Fortran unpack failure already clears all numerical caches because
native insertion precedes binary128 conversion; retain that behavior. Structural
workspace storage is not currently part of the numerical cache-byte accounting;
the lazy512 footprint must be measured and reported separately.

## Host checks, visibility and qualification

The Fortran adapter sees one ordinary success/failure and performs its unchanged
mapping, contraction and independent stability checks. A native success is not
a host accuracy certificate. Do not change MLStabThres, manufacture H/T/U digits,
or mark a whole amplitude512 merely because one tensor used512. Keep the IEEE
boundary around the complete attempt sequence, preserve caller traps/rounding,
and never erase a256 exception merely to accept a finite512 result. Any existing
Fortran transport or host check failure still rejects the result.

Add only small explicit diagnostic accounting: actual256 evaluations,
eligible refusals,512 evaluations/successes/failures/cache hits, and the selected
working precision for the last request. Count attempts independently of cache
hits; retain independent-check usage. The current aggregate cache counters
cannot reveal this, so a future additive query is required before benchmark
claims. No timers or general precision framework are needed in the hot path.

Required private qualification is limited to this behavior:

1. Unchanged ordinary256 outputs/status/IEEE state; no512 construction on success.
2. Known nonsingular Gram refusal retries once with exact113-bit inputs; all other
   failure classes fail closed, including stale/zero/nonfinite condition data.
3. Same input at256, unconditional512, adaptive and NinjaRS; independent forward
   and reversed/rotated checks. The adaptive mixed result must be measured,
   not inferred from the existing all512 process.
4. Full and true poles-only parity, unchanged finite-skipping, precision/mode
   cache separation, primary/check bypass, budget0, input/model/event changes,
   failure-then-valid, finalization and poisoned-context behavior.
5. Binary128 last-bit/signed-zero transport; output overflow/nonfinite rejection;
   caller flags, rounding and enabled-trap behavior. No silent exception reset.
6. Actual host accepted/rejected status and the independent physics criterion,
   with same-effective-point normalization and explicit finite/single limits.
   Ordinary-path and rescue-cost measurements remain required before promotion.

`source-bindings.json` records the inspected current bridge and pinned historical
direct engine, plus frozen evidence identities. `verify.py` checks this portable
inventory; `--sources` additionally checks those external bytes and source
anchors. Neither operation imports MadGraph, loads a DSO, or runs a numerical
kernel. This proposal is not apply-ready, compile-ready, or production-ready.
