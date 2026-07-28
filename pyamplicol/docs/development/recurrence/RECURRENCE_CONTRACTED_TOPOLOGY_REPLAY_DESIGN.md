# Contracted Topology Replay For Full-Color Pure Gluons

## Scope

This design addresses the unexpected cold-generation cliff for process 8
(`g g >` pure gluons) in full color, first visible at five final-state gluons.
It applies to recurrence, compiled, and eager generation.  It is not a
process-name exception: every optimization is selected by an exact
color-topology and model-equivariance certificate.

The target is stronger than preserving the current final runtime schedule:
pyAmpliCol must not materialize substantially more static currents or
interactions than the original AmpliCol library when an exact orbit-template
representation exists.

## Authenticated N4/N5 Evidence

The original generated library is not one amplitude graph per physical color
ordering.  It generates a small set of crossing templates and replays them
under external-label permutations:

| N | Physical sectors | Legacy templates | Replay multiplicity/template |
|---:|---:|---:|---:|
| 4 | 120 | 5 | 24 |
| 5 | 720 | 6 | 120 |
| 6 | 5,040 | 7 | 720 |

For N5, every one of the six generated Fortran modules has:

- 406 current slots;
- 2,440 interaction slots;
- 128 helicity amplitudes;
- the same call inventory and loop-count digest.

Consequently:

- legacy static materialization is 2,436 currents and 14,640 interactions;
- legacy dynamic replay is 292,320 current evaluations and 1,756,800
  interaction evaluations per point;
- recurrence final work is 101,942 currents and 955,368 contributions;
- recurrence construction peaks at 372,422 currents and 4,868,016
  contribution attempts.

The recurrence runtime therefore does not miss recycling relative to the
legacy dynamic replay: final current and contribution work are approximately
34.9% and 54.4% of legacy replay.  The defect is cold materialization:
recurrence construction reaches approximately 153 times the legacy static
current inventory and 333 times its static interaction inventory before
discarding most candidates.

The same-host N5 runtime comparison must use like-for-like totals.  Legacy's
Mac measurement separates approximately 11.49 ms of amplitude evaluation and
36.33 ms of color contraction, for a 48.04 ms total.  Recurrence takes
approximately 36.05 ms total (26.65 ms in the recurrence core).  Comparing
recurrence total against only the legacy amplitude subcomponent creates the
apparent three-to-four-times slowdown.  This reporting issue does not excuse
the real 370.9 s recurrence generation time.

## Exact Orbit Certificate

The color-generic topology partitioner acts on canonical sector words and
external-label maps.  On the full-color N5 plan it deterministically derives:

- six partitions of 120 physical sectors;
- representative sector IDs `0, 120, 144, 150, 152, 153`;
- initial-state-set-preserving source permutations for every target;
- complete, disjoint coverage of all 720 physical sectors.

These representatives match the six legacy crossing templates exactly.  The
proof layer additionally binds:

- requested color accuracy;
- canonical source contracts and initial/final roles;
- canonical model lowering and evaluation-equivalence contracts;
- exact representative and target color-sector contracts;
- external fermion permutation sign;
- every target label permutation;
- one canonical SHA-256 proof digest per partition.

The structural partitioner is shared by all generation modes.  A mode may use
it only after authenticating its own color-reduction route.

## Runtime Shape

### Representative lanes

The artifact carries one independently executable lane per proven
representative, plus independently materialized residual sectors if a proof
fails closed.  It must not place all representatives in one unconditional
schedule: executing a six-representative schedule for each of 720 target
sectors would recreate a factor-six runtime excess.

Each lane has:

- a representative-sector ID;
- a liveness-pruned current/contribution schedule;
- a deterministic local current namespace;
- exact target source-slot permutations and momentum signs;
- a target-to-representative amplitude phase and fermion sign;
- a schedule digest and proof digest.

### Contracted replay

For each retained helicity representative and physical color target:

1. select the target's representative lane;
2. fill momentum forms using the authenticated source permutation;
3. execute only that representative lane;
4. write the physical-sector amplitude into the preallocated color-transform
   scratch;
5. run the existing authenticated Hermitian/Walsh color contraction;
6. expand proven helicity aliases and apply public weights.

Only the physical-sector amplitude scratch scales with the 720-sector color
axis.  Current and contribution arenas scale with the largest representative
lane, not with all physical sectors.

The color reducer remains authoritative.  Replay changes how its physical
amplitude inputs are produced; it does not alter color factors, owner maps,
Hermitian factorization, normalization, or resolved-output semantics.

### Helicity parity

Pure-gluon N5 has 128 public helicities, 16 structural zeros, and 112 computed
assignments.  Exact global-flip symmetry gives 56 nonzero representative
orbits.  Contracted topology replay composes with the generic
`helicity-equivalence:global-flip-v1` proof:

- color target-to-representative maps remain independent of helicity;
- the materialized lanes retain only proven helicity representatives;
- physics metadata expands each dropped public helicity through its exact
  representative and coefficient.

No color-replay schema may invent a second helicity-alias contract.

## Construction Path

The recurrence constructor currently has two independent avoidable
superlinear scans:

1. finalization rescans every pending closure for every amplitude
   destination;
2. closure construction scans all currents repeatedly to find anchor and
   complement supports.

The first is replaced by one monotonic iterator over the already ordered
pending-closure key.  The second is replaced by one support-signature index
and source-slot index.  Both transformations preserve deterministic row and ID
order.  These changes reduce constructor overhead before orbit replay lands,
but they do not solve static materialization by themselves.

The final constructor must instantiate certified representative templates
directly.  It must not first construct all physical-sector candidates and then
deduplicate them.  Representative-local current keys are interned once;
target permutations live in the replay table, not in current identity.

## Compiled And Eager Modes

The color certificate belongs above a generation-mode backend:

- recurrence lowers each representative into a Direct-Arena lane;
- compiled mode lowers each representative into an independently callable
  compiled selector lane;
- eager mode lowers the same representative closure into an eager invocation
  lane.

All three modes consume identical target permutations, signs, proof digests,
color-contraction ownership, and helicity aliases.  Their static-work censuses
are compared with the same legacy template inventory.

Compiled/eager global-helicity parity already demonstrates that the proof can
halve representative amplitude roots.  Recurrence must use the same theorem,
not a mode-specific pure-gluon shortcut.

## Required Structural Gates

For every supported pure-gluon full-color multiplicity:

1. partition coverage is exact, disjoint, and deterministic;
2. target permutations preserve the initial-state label set;
3. every target sector remaps exactly from its representative contract;
4. every representative lane is independently liveness-pruned;
5. final and peak materialized current counts are no greater than 1.05 times
   the authenticated legacy static template inventory;
6. final and peak materialized interaction counts are no greater than 1.05
   times the authenticated legacy static template inventory;
7. no lane execution evaluates an unrelated representative;
8. N-to-N+1 growth, normalized to the legacy template growth, is no greater
   than 1.05 for final work and 1.25 for construction peaks.

A temporary diagnostic certificate may report a failing static-work ratio,
but release acceptance must not downgrade or waive gates 5 and 6.

## Numerical And Performance Acceptance

N4 and N5 are both mandatory to prevent shifting the cliff:

- every resolved helicity total agrees with the current full-sector
  recurrence exact executor;
- native totals agree with exact resolved sums at `rtol=1e-12`,
  `atol=1e-15`;
- built-in and UFO-SM schedules agree after explicit model-state mapping;
- compiled, eager, and recurrence use the same orbit proof;
- normal and cyclically permuted physical points agree with their independent
  full-sector oracle;
- generated artifact and PACBIN digests are deterministic across two builds;
- no warmed evaluator allocates;
- generation time and peak RSS improve at N5 without regressing N4;
- total evaluator time is compared with legacy total evaluator time, while
  amplitude and color-contraction subcomponents remain separately visible.

No existing campaign cell is relabeled or silently replaced.  Performance
tables may adopt the optimization only through the normal scoped descendant
continuity mechanism.

## Landing Sequence

1. Land the color-generic structural partition and proof certificate.
2. Land the deterministic linear closure/finalization constructor changes
   after native tests and exact schedule-digest comparison.
3. Add contracted replay tables and independently executable recurrence lanes.
4. Compose the generic helicity-parity proof.
5. Add compiled and eager representative lanes using the same certificate.
6. Make static-template parity mandatory in the structural-work gate.
7. Run N4/N5 numerical, deterministic-build, memory, generation-time, and
   total-evaluator benchmarks on both architectures.

Until step 7 passes, the existing full-sector runtime remains the correctness
fallback.
