<!-- SPDX-License-Identifier: 0BSD -->

# Internal identity-order loop flavour transport

`loops.flavor_transport.IdentityFlavorTransport.admit` connects an existing
`FlavorSymmetry` to two actual `NativeSourceModelProgram` instances. The
representative and independently generated member must share the exact retained
source model and scalar program. The complete ordinary generation allow-list,
UV/R2 owner, request scope, correction selection, ordered external states,
normalization, order pairings and graph conventions are checked. External PDGs
come from the actual native Born graph as well as the source request; supplied
process metadata alone cannot admit a relation.

The species proof remains species-only, with
`external_transport_validated=False`. The separate transport record retains both
prepared identities, actual requests and the existing `LegMapping`. Its first
supported relation is orientation-preserving species substitution with unchanged
external port order and unit factor. This imposes no bound on the number of legs,
orders, histories or related members. Nonidentity permutations, charge
conjugation, signed resolved colour components and spin-source transport need
separate transport proofs.

`relation.bind(representative, directed=prepared_directed)` creates a member view
of the representative's existing batched numerical executors. It passes through
whole result objects: Born, every squared order and Laurent coefficient, bare,
UV, R2, local and external-field pieces, and both complex directed interference
terms. It neither fits an overall factor nor relabels numerical coefficients.
The directed operation order and physical/auxiliary labels are unchanged because
the external ports are identical. Original point IDs remain the executor's opaque
IDs. Empty batches retain all existing result axes.

Updates act on the representative's atomic source owner. Structurally shared
parameters stay shared; parameters fixed during generation cannot be changed
through the member view. Rebinding precision uses retained instructions and the
original exact source inputs. The independently generated member is an admission
and validation input, not a second mutable numerical state consulted during
mapped evaluation.

## Validation and current boundary

The independent source controls are QCD `u u~ -> d d~` and `u u~ -> s s~` from one
prepared `loop_sm` request. Its complete retained source scope contains nine
ordinary rules, 64 UV/R2 rules and six external-field rules. Each process is
independently generated with one Born graph, 14 loop graphs and six families.
Their entire component/order/pole outputs and four directed histories agree in
binary64, DoubleFloat and ArbPrec70. The histories include a swapped ordered pair
with nonzero imaginary interference, and compare first/second terms separately.
Exact coupling perturbations beyond binary64 survive; a mixed update attempting
to change a generation-fixed parameter fails before publishing any state.

The existing species-proof tests additionally reject independent equal-default
masses and Yukawas, different charges, UV logarithms, loop-group multiplicities,
positive-epsilon CT coefficients, unequal external-field terms and overridden
native tensors. Numerical equality of tree samples is never a certificate.

This is an internal full-colour, model-R2 source-program feature. It does not yet
permit saved-artifact aliases, SDK aliases, public process grouping, spin/mapping
composition or LC/NLC transport. Public admission must be recorded in the existing
immutable loop payload and checked against existing artifact selection; mutable
`process-groups.json` alone is insufficient. No new manifest or hash is required.
The species proof still explicitly rejects unsupported whole-rule analytic
continuation and diagnostic-bearing parameter substitutions; these require the
existing continuation/paired-scalar lowering interfaces before those domains can
be admitted. The validated fixed-full-colour control does not use those paths.
