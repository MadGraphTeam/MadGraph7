# Recurrence LC Color-Witness Audit

This note preserves the independent McClintock/Faraday audit of the exact LC
color transition contract required by recurrence generation. It is an
actionable compiler-to-builder checklist, not a claim that the contract is
already implemented or validated.

Read it together with:

- `RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`, which defines the compact
  recurrence and scaling requirements;
- `RECURRENCE_AMPLICOL_SEMANTIC_CHECKPOINT_AUDIT.md`, which identifies the
  remaining dynamic-color-state, helicity-domain, and native-authentication
  gaps.

## Non-Negotiable Ownership Rule

The model compiler must emit exact, executable LC transition witnesses. The
recurrence builder and runtime must never reconstruct them from model names,
particle IDs, particle representations, benchmark-process identities, or
generic color rule kinds such as `fundamental-generator` or
`adjoint-structure-constant`.

Representations and rule kinds remain useful validation inputs, but they do
not determine orientation, input ordering, relative signs, contact-fragment
history, or which open-string components an operation acts upon. Missing or
inconsistent compiler-owned witness data must therefore fail recurrence
preflight rather than trigger an inferred fallback.

## Fundamental-Generator Orientation

For the three oriented fundamental-generator cases, the exact LC operation is
fixed by the result representation and ordered source slots:

1. **Antifundamental result from fundamental plus adjoint inputs:** append the
   adjoint word to the fundamental-side word.
2. **Fundamental result from adjoint plus antifundamental inputs:** prepend the
   adjoint word to the antifundamental-side word.
3. **Adjoint result from fundamental plus antifundamental inputs:** join the
   fundamental and antifundamental components into the resulting adjoint
   segment.

The witness must identify the selected parent components, their canonical
ordering, any component reversal, and the resulting component. An operation
label such as `append`, `prepend`, or `join` without explicit operands is not
sufficient for multiple-open-line color forests.

Tests must cover every source-slot orientation and verify that built-in SM and
equivalent UFO-SM kernels emit semantically identical witnesses after explicit
model-state mapping.

## Adjoint Commutator Terms

An adjoint structure constant is an exact two-term commutator, not a generic
concatenation rule:

```text
+ concat(parent_a, parent_b)
- concat(parent_b, parent_a)
```

The compiler must adjust these signs by the parity of the permutation that
maps concrete source slots to the canonical oriented-kernel order. Each term
must independently retain:

- its canonical input permutation;
- parent-component selectors and reversal mask;
- exact signed factor reference;
- resulting dynamic-color-state operation;
- proof digest tying the term to the exact projected tensor expression.

The builder must intern and apply both terms as separate exact contributions.
It must not collapse the commutator to one unsigned rule or infer its ordering
from representations.

## Contact-Decomposition Provenance

Higher-point color structures decomposed into trivalent auxiliary-current
fragments must preserve the proof information carried by each
`CompiledContactDecompositionSplit`. Lowering may not reduce those fragments
to an unqualified oriented-kernel rule.

Every emitted LC witness term originating from a contact decomposition must
retain, at minimum:

- split identity and decomposition stage;
- original and current fragment leg ordering;
- introduced dummy or auxiliary state identity;
- source-slot permutation and its parity;
- fragment orientation and parent-reversal data;
- exact coefficient/factor reference;
- auxiliary and result color-shape contracts;
- digest of the compiler proof connecting the fragment to the original exact
  contact tensor.

This provenance must survive prepared-model serialization, process lowering,
schedule construction, and deep artifact inspection.

## Mandatory `lc_color_transition_terms`

Every recurrence-capable `CompiledOrientedKernel` must carry compiler-owned
`lc_color_transition_terms`. Each term must encode a closed, versioned record
with at least:

- operation: `inherit`, `prepend`, `append`, `join`, `keep`, or `close`;
- canonical input permutation;
- explicit parent component/open-line selectors;
- parent reversal mask and optional result-component permutation;
- input, auxiliary, and result LC color-shape contracts;
- exact factor-catalog reference, with f64 bits only as a derived payload;
- witness ordinal, unique within the kernel;
- exact proof digest;
- contact split/stage/legs/dummy/parity/orientation provenance when applicable.

The witness ordinal and oriented-kernel identity together form the semantic
witness ID used in contribution identity. Generic rule kind, representation,
or a physical flow/sector ID is not a substitute.

Closure witnesses require the same discipline. They must select concrete
components, encode exact signed closure terms, and certify the resulting
physical topology, including reflected pure-gluon phases, multiple-open-line
partner contractions, same-flavour reconstruction, and external-fermion
exchange signs where applicable.

## Built-In And UFO-SM Parity

Built-in and external-model compilers must pass through the same witness ABI
and certification path. Equivalent SM interactions must produce matching:

- term counts and operations;
- input permutations and component selectors;
- reversal masks and commutator signs;
- exact factors and proof identities;
- contact-decomposition provenance;
- auxiliary/result color shapes.

Parity must be checked from canonical model contracts, not by recognizing
either model as the SM. Any difference must be explained by an explicit
canonical input mapping; model-specific optimization exceptions are forbidden.

## Native Authentication

Rust validation must authenticate a witness against the actual transition or
closure state tuple before the builder applies it. For every term, validate:

- oriented-kernel ID, witness ordinal, and semantic digest;
- concrete ordered input-state IDs and their representation/color shapes;
- the declared canonical input permutation and selected parent components;
- auxiliary-state contract, when present;
- result-state representation and dynamic LC color shape;
- exact factor reference and proof digest;
- contact provenance and permutation parity, when present.

The operation must then produce the declared interned dynamic-color-state ID.
The builder must reject out-of-range components, incompatible open-string
endpoints, invalid joins, mismatched shapes, missing commutator partners,
unphysical closures, and any process/template digest mismatch. Validation may
not accept a witness merely because its generic rule kind is compatible with
the input representations.

## Acceptance Checklist

- [ ] `CompiledOrientedKernel` serializes complete
      `lc_color_transition_terms` without orientation inference downstream.
- [ ] Fundamental append, prepend, and join orientations are covered for all
      source-slot permutations.
- [ ] Adjoint commutators retain two exact terms with permutation-adjusted
      signs.
- [ ] Contact splits retain stage, legs, dummy state, parity, orientation, and
      proof provenance end to end.
- [ ] Multi-open-line operations use explicit component selectors and closure
      topology checks.
- [ ] Built-in and equivalent UFO-SM witnesses match after canonical mapping.
- [ ] Native validation binds every witness to concrete input, auxiliary,
      result, and closure state tuples.
- [ ] Dynamic color-state and contribution identities include the applied
      witness ordinal and never substitute physical sector identity.
- [ ] Negative tests prove that absent, malformed, inferred, or
      representation-only witnesses fail closed.
- [ ] A fresh independent review confirms that the recurrence builder applies
      only compiler-certified terms and does not reintroduce model-specific or
      rule-kind inference.
