# QP lifecycle and high-multiplicity cost audit

No production change, compilation, native execution, precision change, or
timing was performed. The next useful step is a bounded private profile of
QP transport and repeated scalar requests, not enabling the existing generic
Fortran cache. No speedup or cache hit rate is claimed.

Upstream source anchors below are relative to
`.tirust-build/pole-development/crates/iregi-rust/src/`; `evidence.json`
records exact hashes and their agreement with retained installation metadata.

## What the retained traces establish

The historical explicit-9 EW4 initialization stopped without READY after 66
completed public calls: 24 DP and 42 QP. QCD4 also stopped without READY after
28 completed public calls: 6 DP and 22 QP. The coordinator reported unchanged
1800-second bounds; the original failure records contain protocol timeouts
but not those numeric durations. The earlier 21-call QCD4 snapshot remains
retained separately. Neither trace contains tensor counts, cache statistics,
or per-call times. They cannot attribute cost to Rust, transport, numerator
construction, or any proposed change.

`evidence.json` binds these observations to the historical c91 generated
sources and DSO `49af12f0…71ca12`. The audited current QP bridge functions and
Fortran transport are byte-identical to c91; the newer direction-workspace
split changes DP only. Explicit backend 9 rescues to its own QP after DP
failure. This does not measure the five-backend rescue chain, which tries the
remaining DP reducers first (`../rescue-routing.md`).

## Existing reuse and the Fortran-cache question

QP plans, workspaces, scalar evaluator constants, and structural variants are
already retained. `vendor/tirust/src/lib.rs:80–87,162–215,644–654` creates each
plan/workspace once per `(N,rank)` and clears numerical caches without removing
them. `mp_records` checks the exact whole-tensor cache before MPFR parsing.
Independent check passes bypass this cache. Upstream MPFR scalar/subset
sharing is explicitly unsupported (`direct/exec.rs:1046–1048`).

`loop_exporters.py:884–889` enables `TIRCaching` only for amplitude reduction
or multiple squared split orders. Both audited Born/single-order exports
therefore generate a dummy `CLEAR_TIR_CACHE`. Even where enabled, the cache
in `TIR_interface.inc:63–66,238–315` holds **DP** coefficients. QP dispatch in
the generated `CT_interface.f:1281–1283` calls `MP_TIRUST_LOOP` directly,
bypassing that storage. Merely enabling `TIRCaching` would not cache QP data.

Steady Born evaluation sums helicities before its loop-reduction calls
(`loop_matrix.f:1026–1144`); QP numerator construction is separately guarded
by `MP_DONE` at lines 1012–1023. The 144 QCD4 or 64 EW4 helicity combinations
must not be multiplied by the loop-group count to invent repeated QP tensor
calls. Initialization and explicit-helicity interfaces have different reuse
opportunities, which require call-level evidence.

A new mapped-QP Fortran cache would need an actual repeat rate before it is
justified. It must preserve subprocess/context identity, exact inputs and
scale, topology/rank, pole selection, precision, event/rotation invalidation,
and failure behavior. Initially bypass check passes, as the present bridge
does; never let them reuse primary tensors. Existing event boundaries call
`CLEAR_CACHES` at generated lines 928–929, 964–971, and 1800–1818. Generic
cache mode wrapping additionally clears at template lines 1044–1045 and
1357–1358; its DP indexing is not a ready-made QP policy.
Insertion must not turn a failed/nonfinite tensor or failed transport into a
successful hit, and mapped coefficient validity is distinct from raw-provider
success. Public diagnostics and final contraction checks must remain intact.

Two dense QP direction slots would require `3 × coefficient_capacity ×
loop_groups × 32 × 2` bytes: 18,023,040 bytes for QCD4 and 16,369,920 for EW4,
before keys/flags. Such storage would need to share the existing
`TIRustCacheMaxBytes` allowance and respect `TIRustGlobalCache`; adding it
outside that budget would violate the intended per-worker cache limit.

## Prioritized proposals

1. **Measure transport and exact repeat opportunities first.** Every
   `mtirust.f90:296–359` QP call allocates five buffers, writes
   `6N+N²+1` input decimal records, and reads `8*binomial(rank+4,4)` output
   records, including whole-tensor cache hits. At the generated maximum ranks
   those outputs are 1008 records for QCD4 and 560 for EW4. Add private stage
   counters around packing, bridge lookup/parsing, reduction, output
   formatting, and Fortran unpacking, plus exact-key repeat counts. Preserve
   the 113-bit parse followed by promotion to 256 bits; parsing directly at
   256 bits would change the transported value. Reusing scratch buffers or
   avoiding the cache-hit `values.to_vec()` copy are possible after these
   costs are measured. No numerical-result cache is needed merely to reuse
   allocation capacity.
2. **Assess a full-QP scalar cache through a pinned hook.** This can reuse
   A0/B0/C0/D0 masters across distinct parent tensor groups, unlike a
   group-indexed mapped cache. The existing provider accepts binary64 only;
   a bridge-only call to the present API cannot enable MPFR sharing. See
   `scalar-cache-feasibility.md` for the exact boundary, key, isolation, and
   bounded shadow-profile proposal. First count repeated eligible requests;
   do not assume common topology implies identical numerical inputs.
3. **Secondary small engine candidates.** `direct/exec.rs:1658–1663` computes
   two square roots for Gram size three and then discards them in favor of
   the existing cube-root expression. Removing only discarded work needs
   numerical and ambient MPFR/IEEE flag checks. At lines 644–678 and 876–903,
   copy-only `clone()` assignments into retained MPFR buffers could use
   precision-preserving `FloatLike::set_from`. Plan-only divisors at line
   2819 and local lightlike scratch vectors at lines 2025–2027 are further
   candidates. None is a measured explanation for the warmup timeout.

Required work remains required: 256-bit master arithmetic, numerical Gram
construction/inversion, conditioning and independent stability tests, genuine
QP numerator/contraction, and all public finite/error checks. Full MPFR
misses currently recompute scalar masters; structural variant plans persist,
although incompatible successive inputs can force relation checks and
variant rediscovery. Measure those events before proposing a larger
workspace partition.
