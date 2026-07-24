# Recurrence And AmpliCol Execution Audit

This read-only audit compares the first end-to-end recurrence implementation
against original AmpliCol's generated mode-1 execution. It was completed after
the Z+2g through Z+4g canaries and supplements
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`.

## Main Finding

The compact recurrence schedule is stored once, but a global selected-flow
request was dispatched through the resolved-component lane. That lane evaluates
the complete graph independently for every nonzero helicity. The intended
topology-replay path evaluates all retained helicity representatives in one
compact pass.

| Case | Sources / contributions / finalizations / closures | Nonzero helicities | Backend calls per graph/tile | Calls before fix |
|---|---:|---:|---:|---:|
| Z+2g | 11 / 126 / 58 / 24 | 24 | 12 | 288 |
| Z+3g | 13 / 426 / 142 / 48 | 48 | 20 | 960 |
| Z+4g | 15 / 1298 / 318 / 96 | 96 | 28 | 2688 |

For Z+4g, the erroneous lane performs 672 external wavefunction fills, 1,440
source-current copies, 124,608 contribution rows, 30,528 finalizations, and
9,216 closures per point. The intended pass performs 7 fills, 15 copies, 1,298
contributions, 318 finalizations, and 96 closures.

The scheduler itself already batches homogeneous recurrence rows. It makes one
backend call per `(stage, role, prepared-kernel ID)` packet with
`edge_count * point_count` lanes. It does not call the backend once per point,
current, or contribution. The dispatch into resolved execution was the
multiplier.

## Original AmpliCol

AmpliCol mode 1 includes local helicity ancestry in compact current identity.
It fills source rows, accumulates all contributions to each current, and
propagates that current exactly once. Generated libraries store static
current-major and interaction-major integer tables grouped by primitive type.
The benchmark allocates outside the timing loop and calls the generated
amplitude evaluator once per point.

The recurrence schedule follows the same high-level structure:

- compact current and contribution tables;
- one finalization per propagated current;
- direct closure reduction;
- model-generic prepared kernel IDs;
- homogeneous backend packets across recurrence rows and points.

Its process generation performs no process-specific JIT compilation. JIT
compilation belongs to prepared-model creation, where portable SymJIT O2
applications are compiled once for the canonical model kernels. A process
artifact references those kernels by ID.

## Remaining Runtime Gap

Removing the accidental helicity multiplier is expected to reduce the Z+4g
evaluator cost from about 3.94 ms/point to roughly 41 us/point. That is still
approximately 3.3 times original AmpliCol's 12.27 us/point and roughly 12 times
the current compiled evaluator time. These estimates must be replaced by fresh
native measurements after the dispatch fix.

Remaining measured owners and likely improvements are:

1. Dense workspace clearing and source copying for every graph pass.
2. Recomputing current momentum sums instead of interning momentum forms.
3. SoA-to-AoS gathers and AoS-to-SoA scatters around prepared kernel calls.
4. Invocation rows that remain one-to-one with contribution attachments.
5. Missing selector-specific active plans and warmed-allocation assertions.

These are secondary to correctness and the graph-pass fix.

## Correctness Frontier

Z+2g agrees with compiled mode to floating-point precision. Z+3g is the first
failing multiplicity:

- all structural-zero helicities agree;
- all 48 nonzero helicities differ;
- the selected flow is the materialized sector with identity replay mapping and
  unit factor;
- recurrence and compiled proofs have matching high-level current/root counts;
- simple source-helicity permutation and replay phase errors are excluded.

The leading suspect is orientation, sign, or factor handling when the
antisymmetric two-vector tensor transition feeds a tensor-vector transition.
The decisive diagnostic sequence is:

1. Verify `K_tensor(A,B) = -K_tensor(B,A)`.
2. Verify canonicalized `(V,T)` gives the exact oriented
   `K_tensor_vector(T,V)` factor.
3. Compare each stage-2 tensor current as a complex vector.
4. Split the first differing stage-3 vector current by kernel and contribution.
5. Compare complex closure amplitudes before norm-squaring.

Do not compensate this discrepancy in closure normalization.

## Required Counters

Add runtime counters for:

- graph passes and nonzero helicities;
- replay targets;
- source fills and source copies;
- contribution, finalization, and closure rows by stage/kernel;
- backend calls and lanes;
- gather/scatter bytes and time;
- unique momentum forms;
- stable selector reorder skipped/applied;
- warmed native allocations.

Correctness gates must include per-helicity Z+2g, Z+3g, and Z+4g comparisons,
compact-total versus resolved-total agreement, and both representative and
replayed physical flows.

## Ranked Follow-Up

### P0

1. Dispatch recurrence total evaluation with global selectors through the
   compact global-selector lane.
2. Add the execution counters above.
3. Isolate and fix the first Z+3g tensor/contact discrepancy.
4. Gate performance work on component-wise parity.

### P1

1. Persist selector-specific active plans and caller-owned output buffers.
2. Intern unique momentum forms.
3. Compact identical numeric invocation rows.
4. Reduce gather/scatter through a recurrence-oriented prepared packet ABI.
5. Prove zero warmed allocations, then implement process-set schedule sharing
   and the all-flow-union schedule.
