# Static audit of higher-multiplicity cache reuse

This is a source-only audit, with no implementation, native execution or new
performance measurement. No active numerical values, cards or filters were
read. The closed 76-call lifecycle proof covers native-DP real-SM processes
through N=3/rank=2; its QP requests are explicit tensor diagnostics. It does
not exercise physical QP rescues, repeated DP failures or cache saturation.

## What is already reused

Automatic event entry/exit and Lorentz rotations clear numerical caches
(`loop_matrix_standalone.inc:954–957,993–997,2046–2073`). There is no unconditional
native clear inside the helicity or split-order loops. Immutable plans and
structural workspaces survive ordinary native clears (`lib.rs:644–654`).
Generated plan registration is unique by `(N,rank)` and skipped after a
subprocess's context generation is registered (`loop_exporters.py:2680–2687`,
`TIRUST_interface.inc:325–334`). Context destruction changes the generation.

Born interference normally accumulates numerator coefficients over helicities
before reducing, then loops over split orders (`loop_matrix_standalone.inc:
1118–1319,1391–1402`). Amplitude reduction, including loop-induced processes,
reduces inside the helicity loop instead (1198–1209). Consequently, many
helicities do not automatically imply repeated native integrals in Born mode.

| Path | Reuse and independent-check behavior |
|---|---|
| DP host tensor cache | Enabled for amplitude reduction or multiple squared split orders; keys are loop group and CT-mode slot, independent of numerator/helicity. Hits bypass the TIRust adapter. |
| Full DP native | Shared subintegral store on primary passes; independent checks attach no numerical store. Two structural workspaces per `(N,rank)`, distinguished by check flag. |
| Pole DP native | Separate pole store on primary passes; checks attach no numerical store. Primary/check share a structural workspace per `(N,rank)`. |
| QP native | Exact decimal-request whole-tensor cache on primary passes; checks bypass it. No MPFR subintegral sharing. Full/poles keys are distinct; each precision path retains a structural workspace per `(N,rank)`. |

These paths are defined in `TIR_interface.inc:237–328`,
`helas_loop_amplitude.inc:174–180,213–219` and `lib.rs:91–118,162–268,337–348`.
QP calls bypass the DP host tensor cache and still perform Fortran kinematic
preparation and decimal transport, including on a native exact-request hit.

## Findings that need measurement

1. **Failure invalidation can discard successful work elsewhere in the event.**
   Any full-DP non-success clears the entire shared DP store and drops both
   directional workspaces for the failing `(N,rank)` (`lib.rs:128–160`). This
   also happens after a failed check or cache-disabled call, although that call
   attached no shared store. Earlier successful topologies lose their native
   entries; their immutable plans remain. The store clear resets counters as
   well as entries (pinned TIRust `direct/shared.rs:509–535`), so a final snapshot
   can conceal the discarded work. This is conservative correctness protection,
   not an established defect. Do not remove it without a proof that failed
   intermediate entries cannot survive. The small lifecycle proof has no
   physical failure cascade.

2. **The native budget is partitioned without borrowing.** With pole support,
   the default 256 MiB becomes 64 MiB full DP, 64 MiB pole DP and 128 MiB QP
   (`lib.rs:59–76,279–284`). Full DP cannot use an idle pole/QP share. The limits
   account for numerical entries, not all plans, geometry variants, workspaces
   and transport buffers. No saturation or memory-pressure conclusion follows
   from this source fact.

3. **Native cache0 does not disable the DP host tensor cache.**
   `TIRustGlobalCache` supplies the native cache flag
   (`TIRUST_interface.inc:106–108,232–234`); the host cache does not consult it.
   `TIRCaching` is independently enabled by `loop_exporters.py:884–889`.
   Therefore a cache0/−1 comparison for loop-induced or mixed-order processes
   can retain host tensor hits in both arms. Single-split-order Born processes
   have no such host tensor cache. This distinction must appear in profiling
   labels and hit counts.

4. **Unselected backends still perform empty TIRust lifecycle work.**
   Whenever TIRust was available at export, `CLEAR_CACHES` calls its clear even
   if backend9 is never selected (2064–2066). `mtirust.f90:137–142` ensures a
   context before clearing, so a first event can create an otherwise unused
   context; later event entry/exit calls still cross the FFI and mutex.
   A null-context fast return would change generation/error observability and
   is not implemented here. Measure this separately in explicit1/6/7/8 lanes.

Other source-visible costs worth separating in that profile are repeated
kinematic construction, same-store attachment/registration, QP key allocation
and full-answer copying, and QP/pole geometry changes between directions.
For example, N=6/rank=6 has 210 coefficients: the four-slot QP answer is
107,520 ASCII bytes. Its Fortran transport allocation payload is 139,008 bytes
per call before allocator overhead (`mtirust.f90:319–320`), including hits.
These are dimensional calculations, not observed allocations or time costs.

## Minimal local profiling TODO, after the main campaign

- Record a monotonic event timeline before every clear: reason, subprocess,
  CT mode/precision, loop group, host hits, native hits/misses, entries/bytes,
  budget refusals, workspace recreation and native failure status. Snapshot
  before clear because native counters reset; keep failure and event clears
  distinct. Include n3/n4, mixed orders and a loop-induced case.
- Attribute time to Fortran numerator work, host-cache lookup, kinematics,
  registration/budget, QP encoding/decoding and key/answer copies, native
  geometry/reduction and contraction. Count repeated exact requests by
  helicity/order/check direction and geometry restarts. An instrumented profile
  explains work; it cannot itself establish full-amplitude speedup.
- Only after those observations, review one isolated change at a time. Preserve
  independent checks, all failure protection and exact input/IEEE semantics.
  Measure candidate/control full amplitudes and existing-backend regressions
  with the frozen policy. Precise acceptance remains Hua-Sheng's local TODO.

`audit.json` binds the source lines and closed proof. Verify these source/text
bindings without loading a numerical artifact:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/higher-multiplicity-cache-reuse-audit-1/verify.py
```
