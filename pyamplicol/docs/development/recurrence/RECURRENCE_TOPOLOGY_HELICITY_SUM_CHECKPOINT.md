# Recurrence Topology-Replay Helicity-Sum Checkpoint

This checkpoint records the first native one-pass topology-replay helicity-sum
slice. It is subordinate to
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`; it does not claim that recurrence
artifacts, all-flow union, public APIs, or performance qualification are
complete.

## Compact Schedule Shape

- Current identity uses local source-state ancestry. A complete numerical
  helicity is not part of every partial-current key.
- Rust constructs one shared current/contribution/finalization schedule and
  derives complete resolved helicities only at live closures.
- Each live `(physical sector, resolved helicity)` pair is interned as one
  authenticated `RecurrenceAmplitudeDestination` with a packed closure range.
- Structural-zero helicities have no destination or amplitude storage. The
  schedule records retained, active, and structural-zero counts separately.
- Native amplitude workspace is
  `live_destination_count * point_tile_capacity`; it is not the Cartesian
  product of all helicities and all sectors.
- Closure validation recomputes source ancestry from every parent current and
  requires an exact match with the destination's resolved-helicity record.

The word "authenticated" here means only cheap construction/load-time
structural validation: packed ranges are in bounds, IDs are canonical, source
permutations are bijections, and closure ancestry matches its destination. It
does not add cryptographic work to evaluation, duplicate the proof graph, or
hash tables in the hot loop.

## Physical-Flow Replay

Topology replay materializes one exact representative recurrence for each
proven topology class. Every retained public flow stores a compiler-certified
mapping to that representative:

- a source-slot and momentum-slot permutation;
- an exact amplitude phase and fermion-exchange sign;
- representative and target physical-sector IDs.

Rusticol gathers the representative inputs into fixed reusable buffers,
executes the compact recurrence, and scatters the incoherent norm into the
target public flow. A sector without a certified replay relation remains its
own identity representative, so one failed proof cannot disable independent
sharing elsewhere.

## Native Reduction Semantics

Closure kernels accumulate into sparse destination IDs. Resolved output is
destination-major and carries an explicit destination-to-sector/helicity map.
The helicity-sum lane scatters destinations to physical sectors and computes

```text
sum_h |A(flow, h)|^2
```

rather than the incorrect coherent expression `|sum_h A(flow, h)|^2`.
Homogeneous recurrence invocations remain packetized by prepared kernel over
edge-by-point lanes; the runtime does not call a backend once per edge.

## Model-Generic Numerical Check

The prepared-JIT canary runs the same recurrence projection, Rust constructor,
and native runtime for built-in SM and UFO-SM `d d~ > z g`. Both models have
matching schedule counts and reproduce the same physically normalized
component. Their raw amplitudes are not compared directly across models:
built-in SM applies a separate global coupling normalization, while UFO-SM
includes the corresponding coupling factors in its prepared kernels.

Within each model, representative one-pass resolved amplitudes agree with
separate selected-helicity recurrence evaluations. The built-in raw amplitude
is additionally locked to an independent compiled-mode oracle.

A second prepared-JIT canary covers both physical flows of `d d~ > z g g`.
It independently fills source wavefunctions from compiled runtime metadata,
uses a nontrivial replay permutation for the second flow, sums all retained
helicities incoherently, applies the full coupling/color/averaging/identical-
particle normalization, and agrees with the preserved original-AmpliCol
generated-library oracle for both built-in SM and UFO-SM. The UFO comparison
retains the established roughly `2.81e-12` parameter-rounding offset already
present between compiled UFO-SM and that legacy oracle; recurrence-to-compiled
checks remain at the stricter project tolerance.

## Locked Regressions

- sparse destination storage uses three live rows for a synthetic two-sector,
  two-helicity case whose dense Cartesian layout would require four;
- a synthetic non-identity replay target locks source/momentum gathering,
  exact replay weight, and physical-sector scatter;
- complete `d d~ > z g g` helicity sums for two physical flows agree with an
  independent legacy oracle in both model sources;
- helicity reduction distinguishes incoherent and coherent sums;
- a closure routed to a destination with different source ancestry is
  rejected;
- retained, active, and structural-zero helicity counts agree for built-in and
  UFO-SM;
- the exact-source prepared-JIT integration canary exercises the newly built
  native extension under the 30 GiB guard.

## Remaining Before Public Recurrence Execution

- compiler-certified runtime-helicity source dispatch and helicity-independent
  current identity for `all-flow-union`;
- recurrence PACBIN serialization/loading and transactional artifact writing;
- public `Runtime` and Python/Rust/C/C++/Fortran dispatch;
- exact Python execution over the compact schedule;
- process-set schedule interning;
- broader component correctness and performance qualification;
- a fresh independent audit against every unchecked item in
  `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`.
