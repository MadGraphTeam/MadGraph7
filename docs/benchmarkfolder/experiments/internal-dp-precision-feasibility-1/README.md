# Internal precision choices before genuine QP rescue

This is a static feasibility study, not an implementation or a benchmark. No
dependency was changed, compiled or loaded; no active numerical output was
read. The source identities and selected excerpts are frozen in
`source-bindings.json`. Independent read-only review of numeric traits and
downcasts reached the same conclusions.

The smallest existing numerical experiment is **partial double-double (DD)
rescue inside the DP reduction**. The pinned core already implements it. It
could avoid some complete Fortran QP reconstructions when loss occurs in an
eligible tensor recurrence, but that is a hypothesis. It cannot recover digits
already lost in DP kinematics, numerator coefficients or final contraction.
Enabling a general internal precision ladder before measuring the existing
path would add more work and obscure its cause.

The reviewed references are TIRust
`854b79dcb7fffcb35e92b4a8a3dd443369b798c4` and oneloopmaster
`bd703d09a78f7849a6b86a07096e6f222fc01193`; neither was modified.

## Supported choices

| Choice | Existing implementation | Qualification issue |
|---|---|---|
| Current finite DP | `TensorWorkspace<f64>::new_embedded` with default options; method choice enabled, partial DD and precision tiers disabled | Keep the independent MadLoop check and ordinary QP rescue. |
| Partial DD | `EmbedderOptions.partial_rescue=true`; lifts selected non-top triangle/box subsets, reduces them in DD, then injects rounded coefficients into DP | Subset success is not complete tensor or amplitude success. This is the existing source-only private proposal, not an installed feature. |
| Whole-lane internal tiers | `precision_tiers=true` provides 106-bit DD, then 256/512-bit MPFR; conditioning selects the starting tier | A single option also enables internal MPFR work; it is not a DD-only toggle. The inputs and returned tensor remain DP. |
| Dedicated DD workspace | `DirectReductionPlan::workspace(DoubleFloat)` and generic native master evaluators already exist | A new bridge policy, workspace/cache ownership and rounding path would need review. There is no demonstrated advantage over the existing partial option. |
| Genuine QP | Existing binary128 Fortran transport, 113-bit input interpretation, 256-bit MPFR reduction and QP numerator contraction | Remains separate and unchanged. DD is not a substitute. |

Sources: TIRust `dispatch/embedded.rs:45–78,133–174,217–252,323–386`,
`dispatch/direct_hybrid.rs:117–124,303–319,1657–1724`, `direct.rs:186–235`;
bridge `vendor/tirust/src/lib.rs:91–118,162–222`.

`NativeFloat` explicitly supports `f64`, `DoubleFloat` and MPFR `Float`.
DoubleFloat has **106 significand bits**, below binary128's 113, and two
binary64 limbs rather than a wider exponent range. Its exact rational
conversion and complex product preserve both limbs. DD scalar masters use
native DD dilogarithms; they do not invariably call MPFR for every dilog.
However, fixed domains normally select handwritten C0/D0 sectors whereas MPFR
normally selects generated expressions with special C0 routes. Comparing DD
and MPFR therefore changes the scalar execution path as well as precision.
Sources: oneloopmaster `native/primitives.rs:51–68,194–267` and
`native/mod.rs:88–161`.

There is no supplied x87/80-bit `NativeFloat` implementation or corresponding
bridge transport. An extended-real carrier would require a new numeric domain
and its complex arithmetic, transcendental functions, exact predicates and
conversion contract. Changing a Fortran kind alone would not provide it.
The existing DD carrier is therefore the lower-complexity source-supported
candidate; it still needs physical qualification and measurement.

## Scope and failure boundaries

The partial route is limited to binary64 workspaces, ordinary undotted
unshifted plans, failed conditioning after a nonzero Gram solve, and non-top
PV triangle/box subsets. The configured Gram method has precedence. A top
box, an exact singular pivot, scalar-master failure or invalid input does not
become an eligible partial retry merely because DD is enabled. Its DD
acceptance threshold is `2^-66`; input precision remains 53 bits. The
existing whole-lane tier runner also explicitly sets 53-bit input precision
and rounds accepted output back to DP. Sources: TIRust
`direct/exec.rs:1502,1623–1672,1239–1280`,
`dispatch/embedded.rs:103–107,236–252` and
`dispatch/direct_hybrid.rs:1657–1724`.

DD can reduce arithmetic error for represented inputs, but it cannot guarantee
that a native-DP refusal or failed host test disappears. It cannot fix a
physical error inherited from DP numerator construction, repair a wrong
branch prescription or turn an unchecked status into a checked result. A
successful public DP status can include DD-derived intermediates once this
option is enabled; cached intermediates can preserve that ancestry even if
the last call starts no DD evaluator. Stability plots must label that arm
separately and retain actual tier/subset counters, successful tensors and
accepted amplitudes as different observations.

Higher precision also does **not** remove the retained trap mechanisms:

- Generic Gram/Cayley indicators form magnitudes in the carrier, convert
  squared norms to binary64, then divide before testing finiteness. DD or
  MPFR arithmetic upstream cannot make those binary64 operations range safe
  (`direct/exec/gram.rs:699–725,759–788,796–841`).
- The generic native dilog computes its truncation count using a binary64
  modulus and `4*pi*pi/modulus`; that count controls Horner terms
  (`oneloopmaster/native/dilog.rs:72–75,123–144`). Blind clamping or rewriting
  this expression can change the numerical recurrence.
- A separate source-only DD concern is `factor` calling `split(hi)` before
  its range test; splitting multiplies by `2^27+1`. A sufficiently large
  finite limb can overflow before the intended product fallback. This has
  **not** been observed in a Standard Model amplitude
  (`oneloopmaster/native/double_double.rs:24–33,66–86`).

The closed private adaptive-QP experiment instead retries selected genuine-QP
reductions from 256 to 512 MPFR bits; it is a different intervention from
53-bit-input partial DD. Its private Cayley guard retained masked comparator
results but did not complete either trapped physical worker. The remaining
localized Gram and dilog DIVSD sites are consistent with the control
expressions above. The actual operands and exception subtypes are still
unknown. These observations justify diagnostic work, not promotion of that
development library or repinning dependencies. See the retained
[physical qualification](../cayley-guarded-adaptive-physical-qualification-1/README.md)
and [source audit](../post-cayley-trap-source-audit-1/README.md).

## Stores, mapping and host behavior to preserve

The core has limb-exact DD master keys and DD values; it must not be described
as a DP-rounded master cache. Partial-result keys include method settings,
input precision and threshold. Primary and independent-check numerical stores
must remain separate; configurations and precision paths cannot share a
context/store accidentally. Current bridge failure handling clears the shared
DP store and both directional workspaces, because unsuccessful intermediate
results may already have been inserted. Removing that protection is not a
precision optimization. Sources: TIRust `store.rs:15–89`,
`direct/exec.rs:1023–1054,1239–1280,3683–3710,3730–3766`;
bridge `lib.rs:113–160,291–348`.

Keep `Convention::MadLoop`, component weights, supplied on-shell-aware
invariants and squared scale throughout every tier. Do not apply the odd-rank
sign or normalization twice. The Fortran adapter maps the four native slots
to finite, UV+IR simple pole and IR double pole; its DP/QP contraction remains
in Fortran. Genuine QP must still transport 113-bit inputs without a DD
intermediary and contract QP numerator coefficients before rounding the result.
Sources: bridge `lib.rs:357–363,493–513`, TIRust `convention.rs:70–111`,
`TIRUST_interface.inc:104–123,230–249,294–316` and
`mtirust.f90:296–364`.

## Profile first, in this order

1. Record native failures, host-check failures, actual partial subsets and
   genuine-QP calls separately. Attribute time to numerator construction,
   host-cache lookup, Fortran kinematics, transport, native geometry/masters/
   recurrences and contraction. Accepted DP/QP point counts alone do not say
   where time is spent. Keep all failures and unchanged independent checks.
2. Measure low-complexity Fortran costs before changing arithmetic: an unused
   backend's event clear currently creates a context; QP transport allocates
   buffers on every call, including exact native-cache hits; host-cache lookup
   and contraction copy whole coefficient arrays. Candidate improvements are
   a null-context clear fast return, reusable transport buffers, and direct
   contraction from a contiguous cached slice. Each has lifecycle/alias/IEEE
   obligations and is **not implemented or shown faster** here.
3. Reuse the existing
   [partial-DD private proposal](../partial-dd-private-proposal/README.md)
   for the smallest isolated numerical comparison. First qualify the
   identically instrumented control against the installed library, then
   compare candidate physical tensors/amplitudes, input bits, host acceptance,
   IEEE state, caches and failure invalidation. Count selected subsets,
   actual evaluations and complete injections separately.
4. Consider a DD-only whole-lane policy only if remaining measured failures
   fit that mechanism. Simply enabling `precision_tiers` is broader. Treat
   MPFR adaptive and trap fixes as independent experiments. There is no
   evidence here that any option meets the per-process runtime requirement.

The existing [cache audit](../higher-multiplicity-cache-reuse-audit-1/README.md)
also identifies fixed budget partitions, broad failure invalidation and the
distinction between native cache0 and the separate Fortran host cache. Profile
those effects before changing key semantics, frame alignment or reuse across
checks. Do not skip the rest of a failed DP pass based solely on a failure
latch: output/filter/stability/IEEE state can still change. Final precise
paired timing and its confidence bound remain Hua-Sheng's local TODO.

Verify this text inventory and retained source excerpts without numerical code:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/internal-dp-precision-feasibility-1/verify.py
```

Add `--source-check` to compare the bound excerpts and complete source hashes
with the exact checked-out source trees. Neither mode compiles or loads a
numerical library.
