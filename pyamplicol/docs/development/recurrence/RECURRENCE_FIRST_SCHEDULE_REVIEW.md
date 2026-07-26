# Recurrence First-Schedule Review

This checkpoint records the independent review performed after composite
authentication landed on `codex/recurrence-execution`. It supplements
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`; the original audit remains the
acceptance reference for the first executable recurrence schedule.

## Status At The Checkpoint

The branch has a sound model-generic, pre-`GenericDAG` foundation, but it does
not yet construct or execute a recurrence schedule. A correct first schedule
must store compact unique current, contribution, finalization, and closure
rows. It must not materialize sector-by-helicity interaction rows as eager mode
does.

Composite authentication proves that the process projection and prepared
template catalog belong together. It does not prove that recurrence states can
yet be constructed: four compiler/process-owned semantic contracts are still
required.

## Contracts Required Before Schedule Construction

1. **Source color seed.** Every source template must carry an explicit,
   compiler-owned LC color seed. The seed states whether it creates an empty
   forest or a singleton active component, its component kind and role, and
   its exact proof provenance. The builder must not infer this from PDG codes,
   model identity, or representation alone.
2. **Transition result role.** An LC transition witness must state whether its
   result component is active, passive, or absent. For example, a completed
   quark-antiquark open string in a color-singlet current is passive; treating
   every joined component as active is incorrect.
3. **Closure result kind.** A closure witness must state whether two active
   components close into an open string, a trace, or no new component. Closure
   code cannot hard-code a trace: a physical `d d~ > z g` closure can produce
   the open string `[d,g,d~]`.
4. **Closure anchor.** Every physical LC construction sector needs a certified
   source-slot anchor. The builder constructs the non-anchor recurrence and
   closes its full-support current with that source. This prevents duplicate
   enumeration of equivalent full-support bipartitions and mirrors the compact
   anchored construction used by original AmpliCol. Replay mappings must map
   the anchor exactly.

Each field must originate in compiler/color-plan contracts, carry exact proof
or provenance data, survive JSON and columnar serialization, and be validated
again by Rust. Unsupported or ambiguous semantics fail closed.

## Minimal Program And Builder

The first immutable program should contain:

- source routes;
- unique current states;
- exact contribution rows grouped by result current;
- one finalization for each live non-source propagated current;
- exact signed closure terms;
- interned dynamic LC color states and exact factors.

`CurrentValueKey` is the current core identity plus the sorted exact aggregate
of `(ContributionKey, coefficient)` terms. Physical sector IDs and complete
helicity IDs must never enter current identity.

The first builder should:

1. authenticate and index the process/template inputs;
2. choose the materialized replay sector and its certified closure anchor;
3. seed source currents from compiler-owned source color seeds;
4. enumerate non-anchor subsets by cardinality;
5. combine disjoint parents only when transition, flow, coupling, and color
   contracts match;
6. aggregate every exact contribution before finalizing a current once;
7. close only the live full non-anchor current with the anchor source;
8. apply exact backward liveness from physical closures;
9. remap deterministically and validate dependencies forward.

## First Structural Fixture

A synthetic `d d~ > z g` fixture should exercise the complete construction:

- four source currents;
- two live size-two fermion currents;
- one size-three current with two exact contributions;
- four live contribution rows;
- three finalizations;
- one anchored closure;
- one otherwise legal `d d~ -> g` branch removed by backward liveness.

Required tests include missing or inferred source-seed rejection, active/passive
mutation rejection, wrong open-string/trace closure rejection, closure-anchor
mapping rejection, contribution/fan-in equality, one finalization per live
non-source current, deterministic output under shuffled inputs, absence of
sector/helicity IDs from current identity, and generation succeeding while
`GenericDAG` and evaluator construction entry points are patched to raise.

## AmpliCol Parity Check

Before claiming the architecture gap closed, a fresh independent reviewer must
compare the implemented schedule against both this checkpoint and
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`. In particular it must confirm:

- topology replay shares partial currents by local source-state ancestry;
- all-flow union uses helicity-independent source placeholders and dynamic
  ordered LC words/open-string tuples;
- no flow-by-helicity-by-edge table is materialized;
- contributions are accumulated before one current finalization;
- closures preserve reversals, exchange signs, multi-line partners, and
  same-flavor reconstruction;
- process support is metadata, not state identity;
- backend execution packetizes homogeneous rows rather than calling once per
  recurrence edge.

