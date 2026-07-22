# Recurrence/AmpliCol Semantic Checkpoint Audit

This read-only checkpoint review was performed after the first recurrence
semantic-contract milestone and before recurrence schedule construction.  It
must be read together with `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`.

## Blocking Findings

1. The LC operation language does not yet describe general multi-open-line
   evolution.  A join needs explicit component operands and an optional
   component permutation; closure must validate the resulting physical color
   topology instead of dropping both parent forests.
2. An all-flow runtime-helicity domain is not yet certified to use one
   helicity-independent full-current contract.  The builder must use either a
   certified full-current template or bounded exact static-chirality classes,
   and graph-size tests must prove that adding retained helicities does not
   duplicate the recurrence.
3. Model-owned LC witnesses are declarations, not yet independent proofs of
   the exact color tensor.  Standard rule-name/representation inference is not
   sufficient for orientation-dependent fundamental-generator operations.
4. Native validation must bind each color witness to the transition or closure
   input-state tuple and result state, including representations and LC shape
   contracts.
5. Dynamic color-state IDs and witness IDs need interner-owned newtypes.
   Physical sector IDs must never be accepted where an interned partial-color
   state is required, and contribution identity must include the applied
   witness ordinal.
6. The process input and prepared recurrence-template input are currently
   validated independently.  The schedule-builder entry point must consume
   both authenticated inputs and cross-check template IDs and digests before
   state construction.

## Scaling Risks

- The columnar process input itself does not materialize a flow-by-helicity
  product.  Sector expansion can nevertheless reappear if physical sector IDs
  leak into dynamic current identity.
- Helicity expansion can reappear if runtime domains are split by numerical
  source state rather than exact static recurrence class.
- The semantic color operation implementation clones forests and computes all
  trace rotations.  The production builder must use interned slices and
  reusable scratch storage in high-cardinality loops.

## Required Closure Evidence

Before the recurrence builder milestone is accepted, tests must demonstrate:

- process/template cross-authentication;
- exact transition-to-witness state compatibility;
- explicit multi-line component selection and physical closure checks;
- no sector ID in dynamic color-state identity;
- no growth in all-flow-union recurrence rows when only retained runtime
  helicity coverage increases;
- contribution counts equal current fan-in and exactly one finalization exists
  per propagated non-source current.

No production schedule should be constructed until these contracts fail
closed.
