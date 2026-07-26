# Recurrence All-Flow Runtime-Helicity Audit

This checkpoint records the independent design audit requested after the first
compact topology-replay runtime milestone. It is subordinate to
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` and must be rechecked before the
all-flow-union runtime milestone is accepted.

## Finding

The current all-flow-union constructor is not yet equivalent to AmpliCol mode
2. It iterates over every retained numerical source state and therefore still
helicity-expands source currents. The native runtime correctly rejects that
incomplete schedule.

AmpliCol mode 2 instead constructs helicity-independent full currents. The
runtime-selected helicity chooses only the source wavefunction and its exact
crossing/embedding factor; full vertex and propagator algebra then carries the
selected components through a single union recurrence.

## Required Static Contract

An all-flow-union current must retain these static properties:

- full execution-state template, including particle, orientation,
  representation, basis, dimension, auxiliary kind, mass, and width;
- chirality-zero full-current representation where runtime helicity can change
  a massless fermion's active chiral components;
- dynamic LC word/open-string state, momentum subset, coupling orders, flavour
  and quantum-number flow, propagator contract, and process support;
- a compiler-certified runtime-helicity execution class.

Numerical public helicity, crossing-adjusted source helicity, source
wavefunction variant, and exact crossing/embedding factor remain runtime data.

## Required Compiler Proof

Add a model-generic `RecurrenceRuntimeHelicityContract` derived through the
same canonical IR and exact-expression checks for built-in and UFO models. It
must certify that:

1. every physical source wavefunction embeds into the declared full state;
2. full-state propagator, vertex, and closure kernels are the unprojected model
   expressions;
3. full-state transitions contain every physical chiral branch without
   spurious terms;
4. crossing phases and source-state mappings are exact.

If this proof is absent, recurrence all-flow union must fail preflight. It must
not silently fall back to a helicity-expanded schedule.

## Compiler-Contract Status

The prepared recurrence catalog now has a model-generic runtime-helicity
contract with deterministic source variants, full-state embeddings and
projections, exact factors, and proof metadata. Both Python and native input
validators check its compact table relationships at construction/load time.
This validation is intentionally lightweight and is not repeated during
evaluation.

Non-chiral identity families can already declare the contract. Chiral
all-flow union still fails closed because the current prepared built-in/UFO
packs do not yet contain every required full-state vertex, propagator, and
closure callable with certified component ordering. The union builder must not
start until model compilation emits those callables; expanding the schedule
over numerical helicities is not an acceptable fallback.

## Required Schedule Shape

- One source-dispatch domain and one source current per external leg/domain.
- Source variants map public helicities to source-fill callables, full-state
  embeddings, and exact factors.
- Internal current identity contains no numerical helicity.
- Selector planning maps each requested helicity to one source variant per
  domain and stably groups equal assignments.
- No table scales as `currents x helicities`, and retained-helicity count does
  not change union current counts.

## Acceptance Checks

- Source variants agree with existing helicity-specific wavefunctions,
  including crossing.
- Every resolved `(flow, helicity)` agrees with compiled/eager for built-in SM
  and equivalent UFO-SM.
- Built-in/UFO schedules match after canonical state-ID mapping.
- Chiral `qq_Z6g`, massive top, longitudinal vectors, pure gluons, and multiple
  quark lines are covered.
- Homogeneous, alternating, random, and pre-grouped per-point selectors remain
  allocation-free after warm-up.
- A post-implementation audit confirms there is one union source per leg,
  chirality-zero full fermion states where required, no numerical helicity in
  current identity, and no helicity-multiplied recurrence graph.
