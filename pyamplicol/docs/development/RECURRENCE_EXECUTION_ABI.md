# Recurrence Execution ABI

This document freezes the first implementation boundary for LC recurrence
execution. It is normative for the feature branch until an explicit ABI version
change is committed.

## Ownership Boundary

Generation forks after process expansion, complete LC color planning, coupling
order resolution, and LC topology replay proof construction, but before
`GenericDAG` construction.

Python owns:

- Symbolica algebra and exact-expression canonicalization;
- model compilation and prepared evaluator construction;
- process expansion and the physical LC color plan;
- exact model and replay proof catalogs;
- compact column extraction and artifact transactions.

Rust owns:

- recurrence state construction and interning;
- exact contribution aggregation;
- backward liveness and forward dependency validation;
- schedule lowering and semantic schedule digests;
- direct PACBIN writing and loading;
- native f64 recurrence execution and selector planning.

The production recurrence path must not construct a `GenericDAG`, expanded
Python recurrence rows, or process-specific symbolic evaluator applications.

## Versioned Contracts

The v1 implementation introduces these exact identifiers:

```text
pyamplicol-recurrence-template-v1
pyamplicol-recurrence-builder-input-v1
pyamplicol-recurrence-builder-result-v1
pyamplicol-recurrence-plan-v1
pyamplicol-recurrence-runtime-layout-v1
pyamplicol-runtime-recurrence-execution
rusticol.recurrence-runtime.complex-f64.v1
rusticol.recurrence-color.lc.v1
```

Process artifacts retain schema v3 and PACBIN retains `pacbin-v1` framing.
Pre-release recurrence contracts require no compatibility loader. Existing
compiled and eager artifacts and prepared-kernel ABI v1 remain valid.

## Exact Scalar Contract

Proof coefficients use canonical exact complex rationals:

```text
ExactComplexRationalV1 {
  real_numerator: signed decimal integer,
  real_denominator: positive decimal integer,
  imag_numerator: signed decimal integer,
  imag_denominator: positive decimal integer
}
```

Each fraction is reduced, has a positive denominator, and encodes zero as
`0/1`. A binary64 source is converted to its exact dyadic rational. No proof
coefficient is aggregated through binary64 arithmetic or `fsum`. Runtime f64
coefficients are derived only after the exact proof and schedule digest have
been finalized.

## Prepared Recurrence Template

The optional prepared-model companion is a content-addressed semantic catalog
above the existing callable-kernel catalog. Old prepared bundles remain valid
for eager execution and fail recurrence preflight with an exact recompilation
command.

The catalog contains:

```text
CatalogHeader
ParameterTemplate[]
CurrentStateTemplate[]
SourceTemplate[]
QuantumFlowTemplate[]
TransitionTemplate[]
PropagatorTemplate[]
ClosureTemplate[]
ColorContractionTemplate[]
SymmetryProof[]
EvaluatorBinding[]
```

Every record has a canonical semantic digest. Evaluator bindings additionally
bind the prepared kernel ID, callable signature, input/output layout, and exact
expression digest. Distinct semantic states may share one callable evaluator;
they must not become one semantic template merely because their evaluator is
identical.

Source templates carry their canonical flavour and quantum-number flows in
addition to helicity and spin state. Quantum-flow templates carry the result
spin state and bind their input/result state tuple and coupling-order set into
their predicate identity. They also carry one exactly certified flavour-flow
operation from this finite set:

```text
constant-result
append-left-result
append-right-result
concat-left-right-result
```

The model compiler owns the operation declaration. Template preparation then
evaluates the live model callback on deterministic, distinct input-flow,
quantum-number-flow, and coupling-order sentinels as an independent consistency
check. Admission must be independent of those accumulated values, and the
declared operation must reproduce every observed result exactly. These probes
do not promote sampled behavior into a proof. A model that overrides the live
flow predicate without also declaring its recurrence contract, or whose
declaration and callback disagree, fails closed before process construction.
The callback is never approximated in Rust.

The v1 quantum-number-flow operation is `particle-static-result`: every
transition to one current-state template must carry the same canonical result
quantum-number flow. Both the declared flavour operation and this result-state
invariant are revalidated against the stored witness columns in Python and
Rust.

Prepared closure templates reference the complete nonempty set of quantum-flow
predicates that admit their input currents; each reference must have the same
input-state, result-state, and coupling-order contract as the closure. Direct
Rusticol closure templates carry neither a prepared quantum-flow witness nor a
result-state contract.

Unknown fields, duplicate semantic keys, duplicate evaluator resolver keys,
stale digests, incomplete state contracts, or unsupported proof algorithms fail
closed.

## Builder Input

`pyamplicol-recurrence-builder-input-v1` is passed to the private
`_lower_recurrence_runtime_v1` binding as contiguous, read-only columns.

Logical sections are:

```text
header and semantic digests
external legs and source-state coverage
physical LC sectors and ordered open strings
topology replay partitions and exact factors
selected generation coverage
coupling-order names and limits
multiword momentum/support masks
prepared semantic template references
process normalization and parameter projection
```

Identifiers are checked `u32`. Offsets and counts are checked `u64`. Bitsets are
catalogued arrays of little-endian `u64` words and are never narrowed to one
machine word. Strings and variable-length integer sequences use flat byte/value
arrays plus `u64` ranges.

Python validates shape, contiguity, byte order, bounds, and the canonical input
digest before releasing the GIL. Rust revalidates all of them before building
state.

## State And Contribution Identity

The recurrence builder uses these semantic identities:

```text
CurrentCoreKey = (
  catalog_digest,
  node_kind,
  current_state_template_id,
  interned_dynamic_lc_color_state_id,
  sorted_support_source_slots,
  canonical_momentum_linear_form,
  layout_specific_helicity_identity,
  canonical_flavour_flow,
  interned_canonical_quantum_number_flow_id,
  coupling_order_vector,
  source_binding,
  propagator_template_id_or_null
)

ContributionKey = (
  transition_template_id,
  canonical_parent_value_class_ids,
  canonical_parent_state_ids,
  canonical_parent_momentum_forms,
  result_state_template_id,
  quantum_flow_witness_id,
  color_flow_rule_id,
  runtime_coupling_binding_digest,
  output_projection_id
)
```

`layout_specific_helicity_identity` is not an opaque selector ID. For
`topology-replay` it contains the static result spin class and one ordered
`(source_slot, source_state_index)` assignment for every source in the local
current support. This is AmpliCol's local source ancestry: it shares a partial
current across complete helicity configurations only when that partial current
depends on the same local source states. A replay source binds one fixed source
template.

For `all-flow-union`, the identity contains only the static transition spin
class. It contains no public or numerical helicity assignment. Source currents
instead bind a runtime source-dispatch domain covering the retained states of
their external leg. Non-source currents never carry source bindings. These
constraints are checked when keys are constructed, preventing either layout
from accidentally adopting the other's state multiplication.

Node IDs, allocation slots, physical sector IDs, and selector IDs are not value
identities. Reuse is certified only after exact contribution vectors agree.

### Dynamic LC color state

The interned dynamic state is an output color-shape ID plus an ordered forest of
components and an ordered binding from every result-current color port to one
forest endpoint. Components are open strings, adjoint segments, or traces and
contain colored source slots in exact order. Fundamental, antifundamental, and
adjoint sources expose one, one, and two oriented ports respectively. Color-
singlet sources remain in current support and helicity ancestry but do not enter
color identity. Physical sector IDs never enter current identity.

Open strings, adjoint segments, and separate component blocks preserve their
construction order. Traces are canonicalized only under cyclic rotation.
Reversal, component permutation, and fermion-line exchange are aliases only when
an exact proof supplies their phase/sign.

At a physical open-line closure, the process colour plan separately certifies
that complete independent open-string blocks may be compared as an unordered
forest with unit phase. This closure-only relation does not canonicalize partial
states: every ordered partner remains a distinct current/closure term, so its
multiplicity and exact coefficient survive aggregation. A closure must already
contain the exact physical blocks; no flat-word splitting or post-hoc line
reconstruction is permitted.

The model compiler supplies executable LC transition witnesses. Each witness
binds a prepared color contraction, input permutation, parent-reversal mask,
exact input-port pairings, ordered result-port bindings, result color-shape
contract, nonzero exact factor, and proof digest. Every parent port is consumed
exactly once, either by a pairing or by a result binding. Rust applies this
certified wiring; it never infers connectivity from `rule_kind`, particle IDs,
model names, or benchmark processes. Applying a witness must conserve every
colored source slot exactly once and produce a state admitted by the declared
output shape.

For each contribution key, Rust aggregates the complete exact coefficient:

```text
color * symmetry * kernel equivalence * exchange * parent phases * flow coupling
```

`CurrentValueKey` hashes the core key, sorted exact contribution vector, and
propagator template. A value-class relation `candidate = phase * representative`
requires topologically prior parent witnesses, exact coefficient equality for
every contribution, and a certified homogeneous-linear propagator. Recurrence
v1 permits only nonzero, parameter-independent exact phases.

Closures additionally bind the closure template, ordered parent classes and
states, coupling binding, exact LC topology, exact `Nc` polynomial, external
fermion permutation sign, and selector weight.

## Model-Generic Transition Contract

The catalog must project every dynamic model callback used by recurrence
construction. In particular, transition outcomes are keyed by the full input
state contract, including spin state, chirality, flavor flow class, quantum
number flow, result spin state, and coupling orders. Closure eligibility uses
the same exact model callback predicates rather than particle-only matching.
Transition records and their quantum-flow witnesses must have identical input,
result, and coupling-order contracts. Caches may not omit fields visible to the
model's transition predicate.

Built-in SM and compiled UFO models enter the same catalog encoder and Rust
binding. Branching on model name, built-in identity, benchmark process, or
hard-coded SM particle lists is forbidden.

## Layouts And Proof Witnesses

`topology-replay` builds one recurrence per exactly proven physical-flow
topology class. A replay witness contains process and catalog digests, source
bijection, inductive current and contribution bijections, closure mapping,
external-label permutation, exact phase, fermion sign, and complete/residual
coverage.

`all-flow-union` builds one helicity-independent current union. Its witness
embeds every independently valid physical-sector recurrence into the union,
proving current/contribution identity, closure coverage, attachment
reachability, and absence of cross-sector leakage.

Each independently proven simplification remains active. An unsupported proof
is localized to its smallest residual schedule and cannot disable other proven
reuse. No numerical probe equality is a production proof.

## Builder Result And PACBIN

Rust writes `recurrence-runtime.pacbin` directly to a unique staged path and
returns only bounded metadata:

```text
kind and ABI identifiers
input and semantic digests
payload path, size, SHA-256, index SHA-256
member and unpacked-byte counts
process, schedule, state, contribution, closure, proof, and residual counts
phase timings
inspection summary
```

The root container stores schedules by semantic digest and process bindings by
process ID. Prepared evaluator payloads remain in the separate root
`evaluators.pacbin`.

PACBIN member kinds are:

```text
RecurrenceRuntimeMetadata = 6
RecurrenceRuntimeTable = 7
```

The exact member inventory is versioned by the runtime-layout ABI. The initial
inventory contains metadata, string and sequence catalogs, exact and f64 factor
tables, process bindings, selector axes/domains, source routes, current states,
contributions, finalizations, closures, reductions, proofs, and inspection
summaries.

The container is authenticated and memory-mapped. Loading decodes directly into
the final immutable `RecurrenceProgram` and shares it through `Arc`; mutable
parameters, warnings, selector caches, aliases, scratch space, and execution
slots remain runtime-local.

## Runtime Contract

`RecurrenceExecutionRuntime` is a separate lane selected once at load time. Its
inner loop does not branch through compiled or eager behavior.

It uses component-major storage with points contiguous, kernel-homogeneous
microprogram ranges, one backend call per packet, direct accumulation, and one
finalization per current. Stable selector grouping is automatic; already
contiguous selector groups bypass reordering.

`evaluate_f64_into` guarantees caller-owned output. Zero warmed native heap
allocation is promised only by a prepared selector plan through
`evaluate_prepared_f64_into`; unseen shapes or selector signatures return
`WouldAllocate` instead of allocating.

## Differential Verification

Until the recurrence path is independently established, tests may emit a
bounded diagnostic graph snapshot. Production artifacts never contain that
snapshot and normal generation never constructs a Python `GenericDAG`.

Required differential checks include:

- every physical flow/helicity component against compiled and eager;
- replay and union witnesses against independently generated no-reuse sectors;
- exact coefficient mutation tests, including the `1 + 2^-54` collision;
- built-in/UFO semantic catalog parity after explicit state mapping;
- pure-gluon reversal, chiral/massive fermions, and three-open-line exchange;
- deterministic bytes and semantic digests across worker counts and hash seeds;
- malformed, truncated, stale-digest, and out-of-bounds inputs in Python and Rust.
