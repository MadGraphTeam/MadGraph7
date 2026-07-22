# Recurrence Constructor Audit

This checkpoint reviews the first compact Rust constructor against
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`. It is an implementation gate, not
a claim that recurrence execution is complete.

## Confirmed Foundations

- Construction does not use `GenericDAG` or a flow-by-helicity edge table.
- Topology-replay current identity uses local source-state ancestry.
- Physical sector IDs are absent from `CurrentCoreKey`.
- Dynamic LC words and open-string forests are explicitly interned.
- Backward reachability is structurally present.
- A retained non-source current has exactly one finalization.

## Blocking Findings

1. All-flow-union source identity still expands retained source states instead
   of using a complete runtime-helicity dispatch contract. Runtime source and
   contribution selector domains must be explicit before that layout executes.
2. Topology replay targets, public-flow mappings, selector coverage,
   permutations, and signs are not yet retained by `RecurrenceProgram`.
3. Parent orientation, `canonical_input_order`, `input_exchange_factor`,
   coupling, and `output_factor_source` semantics must be resolved before the
   first numerical topology-replay execution.
4. Process-level closure reconstruction is incomplete for reflected pure-gluon
   phases, multi-line partner contractions, and same-flavour reconstruction.
5. Direct closure templates have no eligible quantum-flow rows and therefore
   require an explicit direct-closure constructor path.
6. Process-support masks and generation-selected flow/source coverage are
   authenticated but not yet applied during construction.
7. Exact-zero aggregated contributions and closures must be excluded before
   backward liveness.
8. The initial two-source constructor fixture was not a useful execution test:
   it contained no internal contribution or finalization. A real model process
   and a four-source dead-branch fixture are required instead.

## Required Sequence

Before numerical topology replay, resolve item 3 and prove an internal-current
fixture with packed fan-in, dead-branch pruning, and exactly one finalization.
A representative-flow vertical slice may then proceed before replay selectors
are complete.

Do not begin all-flow-union execution until item 1 is closed. Multi-quark-line
acceptance also requires complete process-level closure scheduling from item 4.

After those corrections, an independent agent must compare the implementation
again against both this checkpoint and
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` before performance claims are made.

## Follow-Up Audit

The independent follow-up after the first real built-in/UFO canary found that
the constructor now closes items 3, 5, 7, and 8 above for the restricted
topology-replay slice. Exact mirrored-transition canonicalization removes the
previous built-in/UFO inflation, and both models produce 31 currents, 34
contributions, and 12 closures for `d d~ > z g`, matching the corresponding
`GenericDAG` structural counts without constructing that DAG in recurrence
mode.

Items 1, 2, and 6 remain runtime/integration work. Item 4 is proven only for
the anchored low-multiplicity closure used by the canary; reflected pure-gluon,
multiple-open-line, partner, and same-flavour reconstruction closures still
require direct schedule and numerical tests. In particular, all-flow-union
execution must not start until numerical source helicity is replaced by an
authenticated runtime-helicity placeholder in current identity.

The next permitted vertical slice is therefore numerical topology replay for
`d d~ > z g`. It must retain the compact program, packetize prepared-kernel
calls, accumulate every contribution before one finalization, and compare each
resolved component against compiled mode. This checkpoint does not authorize
general recurrence support or performance claims.
