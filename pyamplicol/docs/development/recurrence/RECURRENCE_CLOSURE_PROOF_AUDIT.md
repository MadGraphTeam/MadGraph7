# Recurrence Closure-Proof Audit

This note records the independent follow-up audit of the Direct-Arena
recurrence closure representation. It supplements
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` and is an acceptance checklist,
not an implementation plan.

## Three Open Lines

For a three-open-line physical sector, reconstruct each complete line block as

```text
[fundamental source, ordered adjoint sources..., antifundamental source]
```

The sector color word, rather than open-string row order, defines the reference
block order. It must be parsed as an exact concatenation of the three blocks.
The unique block ending at the closure anchor is the sink. Rotate both the
reference order and every unaggregated closure witness traversal cyclically so
that this sink is last. Exactly two traversals are valid:

```text
direct:  [A, B, sink]
partner: [B, A, sink]
```

Anything else fails closed. The builder must retain an authenticated
certificate for each realized traversal, including the sector, kind, sink,
reference and witness block orders, block permutation, flattened colored-source
orders, colored-position permutation, closure anchor, physical pairing rule,
and semantic digest. A bare `0/1` tag is insufficient.

The partner traversal changes closure traversal order; it does not relabel
physical sources. Pairing, topology replay, and reflection source permutations
remain independent contracts.

## Fermion Pairings

Match each physical open-line sector to exactly one authenticated fermion
pairing rule by comparing its sorted `(fundamental, antifundamental)` endpoints
with the pairing catalog. A raw closure contribution may claim only the
intersection of:

- pairing rules realized by both parent currents; and
- the rule represented by the physical sector.

For every retained `(sector, complete source-state destination, pairing rule)`
with three open lines, every unaggregated closure witness must carry either a
direct or partner traversal certificate. The builder must exhaustively visit
all compatible current pairs, closure templates, quantum-flow rows, and color
witnesses before aggregation.

Original AmpliCol keeps one representative and an optional partner for a
physical three-line flow. A fixed exact source-state/chirality destination can
therefore legitimately realize only one traversal. Absence of the other
traversal is certified by the exhaustive exact closure search; it must not be
inferred by grouping unrelated source states, physical color sectors, or
fermion pairings. This distinction is exercised by
`d d~ > u u~ s s~`, where some exact source-state destinations contain both
traversals and others contain only the partner.

The exact pairing parity and multiplicity must remain authenticated. They
classify the realized Wick lineage but are not an additional closure
multiplier: canonical closure/input ordering already carries the physical
fermion-exchange sign. Applying the pairing parity again double-counts that
sign for same-flavour processes. Raw contributions retain the pairing
certificate and remain available to the proof graph even when exact
aggregation cancels their runtime row.

## Folded Pure-Gluon Reflection

Reflection applies only when all of the following hold:

1. The target is a single-trace sector with more than two entries.
2. The closure returns one trace.
3. A retained public flow for the same construction sector is the fixed-anchor
   reflection of the canonical trace.
4. Exactly one closure parent carries the folded reciprocal current orbit.
5. The reflected word is distinct from the canonical word.

The retained canonical current should carry its reflection-certificate ID
through current compaction. A closure must verify both the parent dynamic-color
identity and reflection lineage against that certificate. Its source
permutation, followed by canonical trace rotation, must produce the reflected
public flow.

The exported phase direction must be explicit and tested with a
non-self-inverse phase, such as `i`; a phase of `-1` cannot detect reversal of
the convention.

Palindromic and two-entry traces are fixed points rather than reciprocal
two-cycles. Topology replay must compose reflection through the replay source
permutation. More than three open lines requires a generalized cyclic
traversal certificate rather than the two-class rule above.

## Acceptance

The closure proof is complete only when:

- the cold proof rows retain every pre-aggregation witness and sum exactly to
  each executed closure row;
- three-line traversal certificates and physical pairing rules are explicit
  and authenticated;
- same-flavor signs are represented numerically exactly once;
- applicable folded traces carry a verified reciprocal reflection certificate;
- built-in and UFO-SM schedules match after semantic model-state mapping; and
- the independent McClintock follow-up audit confirms that these gaps are
  closed without changing the Direct-Arena hot path.
