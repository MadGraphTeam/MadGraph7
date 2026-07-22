# Recurrence LC Port-Wiring Audit

This audit records the focused follow-up to
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md`. It is a design gate for the
recurrence builder, not a description of a completed implementation.

## Finding

The current ordered component forest is not sufficient for multi-open-line LC
recurrences. It records one active component, but an adjoint current exposes
two ordered color-flow ports. Losing the result-port bindings can preserve the
same source word while changing which endpoint a later current consumes.

For zero-based all-outgoing slots

```text
0 = dbar(A), 1 = d(F), 2 = u(F), 3 = ubar(A)
```

the crossed sector requires

```text
[1,3] + [2,0].
```

The intermediate `u(2) + ubar(3) -> g*` has two independently bound adjoint
ports. In `d(1) + g* -> d*`, one port closes `[1,3]` while the other leaves the
strand beginning at `2` active. Treating the gluon as an undirected `[2,3]`
component cannot distinguish that wiring and can produce the wrong traversal.

## Required Representation

The proof-level LC state must be a compiler-certified port-wired ordered strand
forest:

- singlets expose no color ports;
- fundamentals and antifundamentals expose one oriented port;
- adjoints expose two ordered ports;
- every result port binds to one exact strand endpoint;
- passive strands retain construction order in state identity;
- traces may fold cyclic rotation only;
- reflection requires an exact signed proof.

The model compiler must derive transition wiring from tensor roles and emit:

- parent and result port pairings;
- result-port endpoint bindings;
- parent permutation and reversal;
- exact factor;
- proof digest and witness ordinal.

Rust applies only generic splice and rebind operations. It must not infer
built-in vertex kinds, particle names, benchmark processes, or SM-specific
particle lists.

## Closure Certificates

Closure matching must return an authenticated certificate rather than a
boolean. The certificate retains:

- exact closed strands;
- closed-strand to physical-open-string permutation;
- color witness term ID;
- source and parent permutations;
- exact factor and positive multiplicity;
- reconstruction-rule proof digest.

Independent physical open-string blocks may be permuted with factor `+1` at
closure. Partial states must never be sorted or aliased on that basis.
Distinct partner, reflection, direct, and exchange terms must remain distinct
in the pending closure identity even when parent current IDs coincide.

### Rooted closure invariant

Every physical amplitude is closed through one fixed external source and its
connected `N-1` complement. This is the compact rooted-tree convention used by
AmpliCol. Admitting arbitrary complementary support partitions would count the
same unrooted tree once for each eligible internal cut. Fermion-pairing rules
may remap the selected root and add exact reconstruction terms, but they must
not replace the singleton-root partition.

The `N-1` current must retain accumulated flavour ancestry. Quantum-flow
branch admission is certified from particle state and spin, independently of
that accumulated ancestry; the declared flavour operation is then applied to
the actual parent ancestries. Comparing an accumulated flow such as
`[u, ubar, g]` with the seed probe flow `[g]` incorrectly removes valid
multi-line currents.

## Same-Flavour Reconstruction

Direct and exchanged Wick routings must be derived from canonical species IDs,
antiparticle relations, source orientations, and fermion statistics. Each
routing carries its source-state bijection, physical open-string routing,
fermion permutation parity, crossing or basis phase, exact factor, proof
digest, and multiplicity. No numerical equivalence probe may become a
production proof.

## Acceptance Tests

1. `u + ubar -> g*` exposes two ordered result-port bindings.
2. The crossed two-line example closes exactly as `[[1,3],[2,0]]`.
3. Wrong wiring or reversal fails before schedule construction.
4. Built-in and equivalent UFO-SM produce identical wiring and closure terms
   after explicit state mapping.
5. Three-line representative and partner closures remain separate terms with
   multiplicity one unless an independent orbit proof states otherwise.
6. Same-flavour direct and exchange terms remain separate even for equal
   parent IDs.
7. Every tested `(flow, helicity)` component agrees with GenericDAG.
8. State identity contains neither physical sector ID nor a
   flow-by-helicity-expanded edge table.

The recurrence builder and public runtime remain gated until these tests pass.
