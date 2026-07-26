# Recurrence Direct-Arena Intrinsic Census

## Purpose

This document identifies the smallest model-generic set of Direct-Arena
intrinsics that can materially close the recurrence runtime gap to original
AmpliCol. It is a design census, not an implementation plan or a claim that the
projected speedups have already been obtained.

The central result is:

- the current `u u~ > Z+6g` topology-replay schedule has exact structural
  parity with AmpliCol after filtering;
- 8,338 contribution rows reduce to seven prepared executor groups and five
  canonical tensor-algebra classes;
- two chiral vector-Weyl classes plus the color-ordered three-vector class
  cover 80.38% of contribution rows and an estimated 54.3-56.5 us of the
  measured 67.56 us contribution time;
- adding the two antisymmetric-tensor auxiliary classes covers every
  contribution row in this process;
- three massless finalization classes cover every finalization row;
- eligibility can be certified from exact prepared tensor contracts without
  model-name, process-name, particle-name, or auxiliary-state-name matching.

The recommended first useful intrinsic set is therefore:

1. vector + Weyl -> Weyl, chirality `-1`;
2. vector + Weyl -> Weyl, chirality `+1`;
3. color-ordered vector + vector -> vector.

The recommended parity set adds:

4. vector + vector -> antisymmetric rank-two tensor;
5. antisymmetric rank-two tensor + vector -> vector;
6. the two massless Weyl propagators;
7. the massless vector propagator.

## Scope And Evidence

The census uses existing artifacts only. No process was generated or rebuilt.

### Process artifacts

| Process | Layout | Currents | Contributions | Finalizations | Closure terms |
|---|---|---:|---:|---:|---:|
| `u u~ > Zg` | topology replay | 31 | 34 | 22 | 12 |
| `d d~ > Zgg` | topology replay | 69 | 126 | 34 | 24 |
| `u u~ > Z+6g` | topology replay | 1,425 | 8,338 | 858 | 384 |

The Z+6g artifact is the current Direct-Arena v2 artifact under
`.artifacts/recurrence-z6g-post-rebase-68c`. Its schedule contains 69
homogeneous row groups, 720 physical-flow replay targets, 768 retained
helicities, and 384 nonzero helicity representatives. Its contribution and
current counts match the corresponding AmpliCol after-filter counts.

The Zg and Zgg artifacts establish the low-multiplicity structural sequence.
The current native loader cannot reopen their older candidate-version exact
sections, so they are used for total-count evidence, not for executor-level
cost attribution.

### Prepared-model evidence

The built-in and UFO-SM recurrence-template review bundles contain:

| Prepared catalog | Kernels | Current states | Transitions | Propagators | Closures | Evaluator bindings |
|---|---:|---:|---:|---:|---:|---:|
| built-in SM | 57 | 62 | 1,059 | 62 | 75 | 531 |
| UFO-SM | 297 | 93 | 1,963 | 93 | 52 | 689 |

The raw semantic digests do not intersect across these catalogs because model
namespaces, parameter slots, state identities, parent orientation, and
coupling placement are deliberately part of those digests. Raw digest equality
is therefore neither necessary nor suitable as an intrinsic eligibility rule.
The comparison below instead uses exact tensor contracts after certified
canonicalization.

### Runtime profile

The paired native batch-1024 profile for Z+6g reports:

| Phase | Time per point | Share of wall |
|---|---:|---:|
| Headline wall | 75.26 us | 100.0% |
| Profiled recurrence schedule | 74.63 us | 99.2% |
| Contribution kernels | 67.56 us | 89.8% |
| Finalization kernels | 5.42 us | 7.2% |
| Other recurrence work | 1.65 us | 2.2% |

The same-host AmpliCol result is 37.85 us/point. The current recurrence graph
therefore no longer has a missing-reuse explanation for the factor-of-two
runtime gap. The material owner is the execution of many small generic
prepared kernels.

The profiler currently attributes time by executor role, not by individual
executor class. Class-level cost ranges below use two transparent allocation
models:

1. contribution-row share; and
2. prepared SymJIT application-byte share.

They are bounds for prioritization, not measured per-class timers. Each
intrinsic must be accepted with an isolated generic-versus-intrinsic A/B
measurement on identical rows and points.

## Contribution Census

### Z+6g executor groups

The current schedule contains exactly these contribution groups:

| Audit executor | Prepared kernel | Canonical contract | Rows | Row share |
|---:|---:|---|---:|---:|
| 138 | 24 | adjoint vector + Weyl -> Weyl, `chi=-1` | 2,568 | 30.80% |
| 152 | 7 | adjoint vector + Weyl -> Weyl, `chi=+1` | 2,568 | 30.80% |
| 121 | 37 | antisymmetric tensor + vector -> vector | 1,152 | 13.82% |
| 525 | 4 | color-ordered vector + vector -> vector | 804 | 9.64% |
| 325 | 42 | vector + vector -> antisymmetric tensor | 484 | 5.80% |
| 77 | 24 | singlet vector + Weyl -> Weyl, `chi=-1` | 381 | 4.57% |
| 80 | 7 | singlet vector + Weyl -> Weyl, `chi=+1` | 381 | 4.57% |
|  |  | **Total** | **8,338** | **100.00%** |

The adjoint and singlet vector rows use the same Lorentz-spinor algebra. Their
different color and coupling semantics belong in the exact scalar projection
and recurrence proof metadata, not in separate intrinsic implementations.
Combining them gives:

| Canonical class | Rows | Row share | Estimated contribution cost |
|---|---:|---:|---:|
| vector + Weyl -> Weyl, both chiralities | 5,898 | 70.74% | 39.1-47.8 us |
| color-ordered vector + vector -> vector | 804 | 9.64% | 6.5-17.4 us |
| antisymmetric tensor + vector -> vector | 1,152 | 13.82% | 5.7-9.3 us |
| vector + vector -> antisymmetric tensor | 484 | 5.80% | 3.9-5.4 us |

The cost intervals overlap only as alternative attribution models; they must
not be summed by independently choosing each lower or upper endpoint. The
combined vector-Weyl plus three-vector set accounts for 6,702 rows, 80.38% of
all contribution rows, and 54.3-56.5 us of the 67.56 us contribution phase
under the two complete attribution models.

### Exact canonical algebra

The intrinsic reference contracts should be written over abstract component
bases and one extracted exact complex scalar `alpha`. This keeps couplings,
normalization factors, color factors, exchange phases, and model parameters
outside the tensor primitive while still proving the complete row operation.

#### Vector-Weyl contract

For a Weyl current `q=(q0,q1)` and a Lorentz vector
`V=(V0,V1,V2,V3)`, define:

```text
A = V0 + V3
B = V0 - V3
C = V1 + i*V2
D = V1 - i*V2
```

The two chiral maps are:

```text
chi=+1:
  out0 = alpha * (B*q0 - C*q1)
  out1 = alpha * (A*q1 - D*q0)

chi=-1:
  out0 = alpha * (A*q0 + C*q1)
  out1 = alpha * (B*q1 + D*q0)
```

This is the same tensor family as AmpliCol's
`QuarkGluontoQuark_weyl`/`GluonQuarktoQuark_weyl` and coupled variants.
Parent order is not part of the family identity; it is a certified input
permutation.

#### Color-ordered three-vector contract

For vectors `V`, `W` and their momenta `p`, `q`, with the declared Minkowski
metric:

```text
out_mu = alpha * (
    dot(V,W) * (p-q)_mu
  + 2 * (dot(V,q) * W_mu - dot(W,p) * V_mu)
)
```

This matches AmpliCol's color-ordered `ThreeGluon` family after exact scalar,
metric, momentum-direction, and parent-order normalization.

#### Antisymmetric auxiliary contracts

For the six-component upper-triangle basis
`(01,02,03,12,13,23)`:

```text
T_mu_nu = alpha * (V_mu*W_nu - V_nu*W_mu),  mu < nu
```

The inverse vertex is the exact contraction of that antisymmetric tensor with
a vector under the certified component-sign and metric map:

```text
out_mu = alpha * T_mu_nu * V^nu
```

The explicit six-to-four sign map is part of the certificate. These are the
same algebraic families as AmpliCol's `TwoGluontoTensor`,
`TensorGluontoGluon`, and `GluonTensortoGluon` primitives.

## Finalization Census

The current schedule contains:

| Audit executor | Prepared kernel | Canonical contract | Rows | Row share | Approximate role cost |
|---:|---:|---|---:|---:|---:|
| 559 | 0 | massless Weyl propagator, `chi=+1` | 315 | 36.71% |  |
| 624 | 35 | massless Weyl propagator, `chi=-1` | 315 | 36.71% | about 4.0 us combined |
| 575 | 38 | massless vector propagator | 228 | 26.57% | about 1.4 us |
|  |  | **Total** | **858** | **100.00%** | **5.42 us** |

The two Weyl finalizers are the two `sigma.p / p^2` chiral maps, including the
declared momentum and crossing convention. The vector finalizer is the
massless vector propagator in the declared metric and phase convention.
These map to AmpliCol's `QuarkPropagator_weyl` and `GluonPropagator`
families.

Finalization is worth specializing after contribution intrinsics, but cannot
close the runtime gap alone. Even a twofold speedup of the complete
finalization phase saves only about 2.71 us/point.

The 384 closures split into two 192-row direct Weyl contraction groups.
Closure and reduction time is already part of the 1.65 us non-contribution,
non-finalization remainder, so closure intrinsics are not in the smallest
useful set.

## Built-In And UFO-SM Equivalence

### What is shared

The exact prepared metadata demonstrates the following model-independent
equivalences:

- built-in kernels 24 and 7 and the corresponding UFO q-vector kernels use
  the same two chiral two-component maps;
- the UFO catalog has flavor- and orientation-specific q-vector kernels, but
  their tensor contracts collapse to the same two chiral classes after parent
  permutation and scalar-coupling extraction;
- built-in kernel 4 and UFO kernel 104 implement the same color-ordered
  three-vector tensor map after momentum and scalar normalization;
- built-in kernels 42 and 37 use a six-component antisymmetric tensor;
- the relevant UFO four-gluon decomposition contains one six-component
  color-octet auxiliary state with the same `(01,02,03,12,13,23)` algebra,
  even though its model-owned auxiliary identity is different;
- built-in propagator kernels 0, 35, and 38 match UFO kernels 110, 54, and 118,
  respectively, after role-symbol and state-basis normalization.

The numeric IDs above are audit-local evidence only. They must never enter the
eligibility predicate.

### What differs

The prepared models legitimately differ in:

- canonical state and species identifiers;
- parent orientation and input order;
- model parameter namespaces;
- whether a coupling appears inside the exact kernel expression or as a row
  scalar;
- how coupling order is divided across the two auxiliary edges;
- auxiliary-state names and model-owned proof identities.

In particular, the built-in auxiliary decomposition and UFO contact
decomposition distribute QCD coupling powers differently across
`vector+vector -> tensor` and `tensor+vector -> vector`. Matching one edge's
raw coupling order would incorrectly reject a valid shared intrinsic.

Eligibility must therefore prove:

1. the normalized tensor map for each edge;
2. the exact scalar projection used by that edge;
3. the composed coupling-power accounting required by the recurrence proof;
4. the parent, basis, metric, crossing, and destination maps.

This permits different model representations to use the same intrinsic
without weakening the exact physics contract.

## Fail-Closed Eligibility Certificate

Introduce a prepared-model-owned certificate conceptually named:

```text
pyamplicol-recurrence-direct-intrinsic-certificate-v1
```

It should be emitted during model preparation, where Symbolica expressions and
the complete prepared contract are still available. Process generation should
only bind an already certified intrinsic ID to recurrence executor rows.

### Required certificate fields

```text
certificate ABI and version
intrinsic family and intrinsic implementation version
executor role
destination operation
parent arity and canonical parent order
parent component dimensions
destination component dimension
canonical input and output basis descriptors
chirality or other static algebra class
momentum operand count and momentum convention
metric signature
parent permutation and component maps
output component map
exact scalar/coupling projection
total coupling-power accounting
allowed destination aliasing
allowed row flags
normalized exact output-expression digest
prepared-kernel exact-expression digest
prepared-template semantic digest
parameter-layout digest
proof transcript digest
```

The basis descriptors must describe algebraic objects such as
`weyl-chiral(2)`, `lorentz-vector(4)`, and
`antisymmetric-lorentz-pair(6;01,02,03,12,13,23)`. They must not depend on
particle names or model-specific auxiliary labels.

### Exact proof procedure

For every candidate prepared template:

1. Resolve the model-owned input/output bases and component dimensions.
2. Enumerate only dimension-compatible canonical intrinsic families.
3. Canonicalize dummy indices and role symbols.
4. Apply the candidate parent permutation and exact component basis maps.
5. Extract a single exact complex scalar projection where the tensor map is
   linear in that scalar.
6. Account for parameter slots, coupling slots, row factors, and any certified
   auxiliary-edge coupling repartition.
7. Substitute canonical symbolic component and momentum variables.
8. Prove every normalized prepared output minus the intrinsic reference output
   is exactly zero.
9. Prove linearity in each parent current and in the extracted row scalar.
10. Record the exact transcript and all source digests in the certificate.

Numerical probes may be retained as corruption tests, but they cannot confer
eligibility.

### Load-time and run-time behavior

At prepared-pack load:

- authenticate the certificate and referenced prepared-template digests;
- verify dimensions, basis maps, scalar projection, parameter layout,
  destination operation, and intrinsic implementation version;
- bind a typed intrinsic handle only if every check passes.

At process-plan load:

- verify every row-group contract is a subset of the certificate;
- reject illegal row flags, aliases, or scalar projections.

On any mismatch:

- do not partially apply an intrinsic;
- bind the existing generic prepared Direct-Arena executor for that group;
- record the rejected reason in deep inspection/profiling metadata.

This fallback is exact and local. An uncertified template cannot disable
intrinsics for independently certified groups.

## Minimal Direct-Arena Intrinsic ABI

The existing Direct-Arena call shape is already sufficient. An intrinsic
should implement the same typed contribution or finalization function pointer:

```text
call(
    immutable_intrinsic_context,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    rows,
    row_count,
    point_count
) -> status
```

The intrinsic implementation must:

- loop over all rows in one homogeneous group;
- loop over SIMD point blocks internally;
- load parent components directly from the split-complex current arena using
  row offsets;
- load momenta, parameters, and exact factors directly from their views;
- execute the certified fixed tensor algebra;
- initialize or add directly into destination arena planes;
- handle scalar tails without allocating;
- return a fail-closed status for malformed views or unsupported flags.

No recurrence packet, packed input buffer, output buffer, attachment list,
scatter pass, or per-row backend call is permitted.

### Immutable intrinsic context

The load-time context should contain only data that is constant for the
certified class:

```text
intrinsic ID and implementation version
parent permutation
parent and destination component maps
momentum convention
metric/sign convention
static chirality
parameter/coupling projection
destination operation
allowed row flag mask
certificate digest
```

Current recurrence rows already provide the required dynamic values:

```text
parent component bases
parent momentum-form IDs
destination component base
exact-factor ID
selector-domain ID
flags
```

No new process-specific row format is required for the Z+6g dominant classes.
If a future model needs a scalar projection that cannot be represented by the
immutable context plus existing parameter/factor views, that class remains on
the generic path until a model-generic row extension is designed.

### Backend policy

The intrinsic family is a Rusticol execution optimization, not a new model
definition:

- JIT prepared packs use the certificate to select a host-native intrinsic
  implementation while retaining the portable exact prepared contract.
- C++ and ASM packs may provide the same typed intrinsic ABI, with their
  existing scalar-batch expectations.
- Exact Python continues to execute the exact recurrence contract, not the
  f64 intrinsic.
- Built-in and UFO models share an implementation only after independent exact
  certification.

## Coverage And Runtime Ceiling

The following projections keep all untargeted phases fixed and assume a
twofold speedup of the targeted class set. They are not benchmark results.

| Intrinsic set | Contribution rows covered | Estimated targeted wall | Projected wall at 2x | Ratio to AmpliCol |
|---|---:|---:|---:|---:|
| vector-Weyl, both chiralities | 70.74% | 39.1-47.8 us | 51.4-55.7 us | 1.36-1.47x |
| vector-Weyl + three-vector | 80.38% | 54.3-56.5 us | 47.0-48.1 us | 1.24-1.27x |
| all contribution classes | 100.00% | 67.56 us | 41.48 us | 1.10x |
| all contributions + finalizations | all material kernel rows | 72.98 us | 38.77 us | 1.02x |

For the smallest useful set, the theoretical floor obtained by making the
targeted classes free is 18.8-21.0 us/point. That limit is not physically
attainable, but it shows that the selected classes have enough coverage to
matter. To reach the existing `1.20 * AmpliCol = 45.42 us/point` gate with
only that set, its aggregate class throughput must improve by approximately
2.12-2.22x.

The practical interpretation is:

- vector-Weyl alone is a useful first implementation test but is unlikely to
  pass the runtime gate;
- vector-Weyl plus three-vector is the smallest set that can approach the
  gate and establish whether intrinsic execution is the correct direction;
- full contribution coverage at a twofold speedup should pass the gate;
- adding the three finalizers should bring recurrence close to AmpliCol if the
  contribution estimates hold;
- a table-aware generic SymJIT call alone is expected to remain around
  63-67 us/point and is therefore insufficient without algebra-specific
  intrinsics.

The prior architecture audit's broader estimate of 38-46 us/point for
canonical contract-class Direct-Arena execution is consistent with this
class census.

## Acceptance Sequence

### Milestone 1: smallest useful class set

Implement and certify:

1. vector-Weyl `chi=-1`;
2. vector-Weyl `chi=+1`;
3. color-ordered three-vector.

Require:

- exact built-in and UFO-SM certificate generation;
- every eligible prepared template maps without model-name exceptions;
- component agreement against the generic executor at `rtol=1e-12`,
  `atol=1e-15`;
- identical totals and resolved components for Zg, Zgg, and Z+6g;
- zero warmed native allocations;
- intrinsic row counters equal 6,702 for the audited Z+6g artifact;
- generic fallback counters account for the remaining 1,636 contribution
  rows exactly;
- class-level A/B timing on identical row tables and point batches;
- no compiled or eager runtime change.

### Milestone 2: complete audited contribution coverage

Add the two antisymmetric auxiliary contracts. Require:

- exact six-component basis and sign-map proof for both built-in and UFO-SM;
- exact auxiliary-edge coupling accounting;
- 8,338/8,338 contribution rows intrinsic-eligible in the audited artifact;
- no eligibility rule referring to the built-in tensor name or UFO contact
  state name.

### Milestone 3: finalization coverage

Add both Weyl propagators and the massless vector propagator. Require:

- 858/858 finalization rows intrinsic-eligible;
- exact momentum, crossing, metric, and phase proofs;
- matched paired native profiling at batches 128 and 1024.

### Broader validation

Before treating the class set as generally useful:

- repeat the census for all-flow-union Zgg and Z+6g;
- test built-in and UFO-SM with identical physics;
- test massive fermion, scalar, and multiple-open-quark-line processes, which
  are expected to introduce additional classes;
- report eligible, rejected, and fallback rows by canonical reason;
- verify that selector grouping and topology replay do not change eligibility;
- retain generic prepared executors for every unrecognized exact contract.

## Conclusion

The current recurrence runtime is not dominated by recurrence construction,
selector handling, closure reduction, or missing current reuse. It is
dominated by executing a small number of canonical tensor contracts through a
very large number of tiny generic prepared-kernel calls.

A model-generic intrinsic layer is feasible because the prepared built-in and
UFO-SM catalogs expose the same underlying algebra after exact parent, basis,
momentum, and scalar normalization. The safe implementation boundary is an
exact prepared-model certificate plus the existing typed Direct-Arena row-group
ABI. The first useful implementation should cover the two chiral
vector-Weyl maps and the color-ordered three-vector map; full performance
parity likely requires the two antisymmetric-tensor maps and the three
massless finalizers as well.
