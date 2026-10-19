<!-- SPDX-License-Identifier: 0BSD -->

# Identity-order loop flavour transport

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
conjugation, projected-colour components and spin-source composition need
separate transport proofs. Ordinary full-colour resolved helicity axes are
checked for equality during admission and preserved by identity-port transport.

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

## Saved artifacts and public selection

For virtual generation, `process.flavor_blind = true` requests exact candidate
grouping. It does not authorize tree-style numerical or flavour-blind reuse.
Every process is independently generated first. The complete retained model
proof and actual prepared-program admission decide which identity-port members
can share a representative. This first implementation therefore saves stored
runtime instructions but does not avoid member generation costs.

Proved members use the existing `ProcessGroupMap` and the `exact-loop-flavor`
grouping basis. They are not manifest `ProcessAlias` records, whose same-PDG
permutation meaning is unchanged. The existing immutable loop PACBIN records
admitted member IDs, expressions, external PDGs and the proved species relation.
Its producer checks the actual prepared owner. Native artifact selection matches
mutable process-group records against that admission before binding any numeric
state. Changed species, factors, permutations, conjugation or unadmitted IDs
cannot obtain representative access by editing `process-groups.json`.

Public process ID, expression and external-particle metadata describe the selected
member. `representative_process_key` identifies the stored process. Ordinary
resolved and directed histories retain every order, pole and component; the
member uses the same exact scalar owner and parameter-update guards.

Unproved candidates remain independent. Requests combining flavour grouping with
LC/NLC or spin sources also retain independent generation and their normal
evaluation capabilities. The existing generation filters report whether grouping
was requested, transport was checked and reuse actually applied, with a reason
when the process remains independent. A request flag is never a claim of a proved
mapping. Ordinary group inspection retains each member's actual proof basis.

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

Fresh public generation additionally compares the saved admitted member with an
independently serialized member PACBIN in all three carriers. The ordinary case
retains all 16 resolved helicities; the directed case checks separate first and
second terms for ordered complex histories. Both cover every component, order
and Laurent coefficient, per-point scales and opaque point IDs. Native selected
member metadata, atomic fixed-parameter rejection and modified group-record
rejection are checked on the saved artifact. These are correctness controls, not
runtime performance measurements.

This is a full-colour, model-R2 source-program feature. Spin/mapping composition,
LC/NLC transport and nonidentity external permutations remain uncertified. Saved
member selection uses the shared native artifact loader; it introduces no new
manifest or hash.
The species proof still explicitly rejects unsupported whole-rule analytic
continuation and diagnostic-bearing parameter substitutions; these require the
existing continuation/paired-scalar lowering interfaces before those domains can
be admitted. The validated fixed-full-colour control does not use those paths.
