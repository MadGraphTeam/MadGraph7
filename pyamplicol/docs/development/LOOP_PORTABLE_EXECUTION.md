<!-- SPDX-License-Identifier: 0BSD -->

# Internal graph-free loop execution

`loops.portable.write_loop_pacbin` lowers a retained, complete full-colour
source/legacy model-R2 or certified direct-R2 four-gluon program to the existing
`evaluators.pacbin` envelope.
`PortableLoopPlan.open(...).bind(...)` creates a fresh numerical view in double,
DoubleFloat or arbitrary precision. This is an internal execution roundtrip;
public artifact generation/loading and standalone SDK loop dispatch remain gated.

The ordinary physical controls are `u u~ -> d d~` and `u u~ -> t t~`.
The latter retains nonzero external-field contributions, independently of the
local UV and R2 currents. The direct control is `g g -> g g`: all three N4 and
dimensional-remainder epsilon sectors survive persistence, with UV currents and
external fields retained and model R2 excluded. Source-owned model-R2 plans retain
one scalar owner and ordinary/CT scale views, including anonymous external-field
coefficients. Source dimensional/direct, formal colour projection, correlations
and rescue remain outside this boundary.

## Persisted data

The container adds one `LOOP_RUNTIME_PLAN` member to the existing PACBIN member
catalogue. Its versioned, typed JSON record references existing
`SYMBOLICA_EXACT_STATE` members. The shared Rust core parses the record and
validates member kinds, exact evaluator input/output sizes, schedule indices,
input/output slots, external-state layouts, order axes, reduction-family shapes,
scalar value/diagnostic pairs and assembly sizes. Existing PACBIN integrity and
path validation are reused. There is no second manifest, digest or container.

Persisted numerical data are:

- Physical state kinds, structural masslessness, component offsets and helicity
  rows and external PDGs; exact model-input names and integer column maps.
- Symbolica's saved local evaluator bytes and native coefficient arena slots.
- The already proven execution-sharing groups. Aliases reference the exact
  representative schedule and ordered columns that the original hot path uses;
  no symbolic sharing proof is reconstructed on loading.
- Numerical loop-output and order-sector indices, denominator counts, coefficient
  powers, exact kinematic/mass/projector evaluator states and numerator strings.
- Saved Born, counterterm, final-assembly and order-summing evaluator states.
  Counterterm execution retains source-kind tags, scalar diagnostics and input
  columns, without source insertion or tree objects.

The internal v3 descriptor is regenerated rather than migrated. Its model
binding is explicitly either legacy evaluator input or a source scalar owner;
scalar current bindings are correspondingly either saved evaluators or selected
source-output columns. The native reader rejects mixed ownership, invalid
parameter types/names, non-finite exact override literals, real parameters with
any nonzero imaginary part, mismatched scale selection and invalid column maps.
Source owner values, defaults and real-type guards retain their paired diagnostic
instructions. Both scale views refer to the same owner by descriptor structure;
no raw UFO, scalar library or producer reference is required to reconstruct it.
The source `model_inputs` labels are qualified display metadata; named updates
and input order are defined solely by the scalar owner's external names/types.

The internal format retains the captured initial exact external override strings
and fixed parameter values. `bind(parameters=...)` overlays supplied values on
those captured overrides. Source `update_parameters(...)` keeps the existing
atomic checked update semantics; `rebind(parameters=...)` captures the current
owner snapshot once and overlays any supplied values without dropping omitted
keys. Every view has a fresh owner, while its ordinary and CT scale evaluations
share one captured event-batch snapshot and the per-point renormalization scale.
The existing callback registry and exact decimal parser now live in the shared
core decoder, with the runtime delegating to that one implementation.

The native reader checks that direct selection is dimensional, uses the certified QCD
four-gluon state layout and QCD4/QCD6 Born/virtual sectors, and contains no
effective R2 currents. Coefficient labels
retain weighted rank `q_degree + 2*mu2_degree` and epsilon degree at most two.
NinjaRS continues to own numerator parsing and epsilon-sector reduction; no
private Laurent or tensor algebra is introduced by loading.

Producer graphs, recurrence objects, symbolic Python objects, colour tensors,
live evaluator workspaces, rounded model values and event results are excluded.
Normalization factors already belong to the saved projection evaluators; the
integer initial-average field is metadata and is not reapplied.

## Execution and remaining boundaries

The Rust-owned descriptor is immutable. Each arithmetic binding creates fresh
coefficient schedules, model/projector evaluators and NinjaRS reducer workspaces.
Repeated schedules may share one workspace only within a single bound view,
where the existing coordinator invokes them sequentially. Views do not share
mutable evaluator scratch or result buffers.

The experimental Python binding is a thin owner of the core descriptor. Both
the existing Python coordinator and the shared native coordinator below invoke
the same numerical operations. There is no reverse dependency from the shared
core to PyO3. Public SDK dispatch remains a separate step.
Loaded views explicitly reject producer-dependent colour mutation, producer
metadata access, retained-producer preparation and partial/rescue evaluation.
Ordinary complete evaluation and fresh arithmetic rebinding remain available.

Typed buffer storage, arithmetic selection, decimal conversion, gathers,
concatenation, regrouping, exact rational ingress and kinematic evaluation,
dependency wavefunction blocks, saved scalar evaluators and the tiled coefficient
schedule now live in the non-Python `rusticol_loop_reduction::runtime` module.
The polynomial reduction facade, epsilon-sector assembly, successful-lane
buffers and optional fast-engine profiling use that same shared module.
The Python classes own these same Rust values and delegate their operations.
The existing dependency interpreter/JIT and cold-template cache have one
implementation. Fixed scalar callbacks register at shared decoding, so native
callers restore the same saved instructions without importing the Python
consumer. The complete native coordinator below reuses these operations for
legacy and source-owned bindings; no public SDK loop entry point is activated.

Family projectors retain the same saved bilinear instructions and packed
input/output order under both numerical bindings. The default
`projector_engine="interpreted"` keeps the native batch interpreter in every
carrier. An explicit `projector_engine="jit"` binds each saved family projector
as one node in the existing tiled `CoefficientPlan`; this option requires
`arithmetic="double"`. It is independent of `coefficient_engine`, and is accepted
at initial construction, retained rebinding and portable binding. DoubleFloat
and arbitrary precision keep the interpreter; requesting projector JIT for
those carriers raises an error. This changes neither the bilinear algebra nor
the PACBIN format, and introduces no arena sharing or dual-Born factorization.
Every bound view owns fresh evaluator scratch. Engine selection is a binding
choice and is not an accuracy certificate.

NinjaRS currently exposes no public serialization/rebinding interface for its
private generic Laurent expansion plan. Loading therefore constructs a fresh
`PolynomialPlan` from retained exact numerator strings and ordered coefficient
parameters. This preserves dependency-owned algebra, but does not remove that
cold setup cost or establish cross-version NinjaRS binary-plan compatibility.

`tests/unit/test_loop_portable.py` exercises the complete controls in all three
carriers, native result/old-buffer lifetime, changed exact parameters/scales,
empty batches and noncontiguous IDs. It also loads in a fresh process without a
source UFO, with producer constructors disabled, and checks malformed layouts
through the Rust reader. Independent MadLoop components are checked when the
corresponding `PYAMPLICOL_UUDD_COMPONENT_REFERENCE`,
`PYAMPLICOL_TOP_PAIR_COMPONENT_REFERENCE` or `PYAMPLICOL_GG_COMPONENT_REFERENCE`
is provided. Native success remains
numerical status rather than an independent accuracy certificate.

`tests/unit/test_loop_source_portable.py` covers the generic source-compiled
four-gluon model-R2 program, including nonzero field contributions. It compares
loaded and retained execution in all three carriers, saved/current exact override
overlays, per-point scales, retained snapshots, owner isolation, empty batches,
malformed scalar descriptors and loading in a fresh process with source producers
disabled. Independent component checks use the existing binary64 MadLoop capture;
roundtrip agreement itself is a transport/control-flow check, not a new physics
or arbitrary-precision accuracy certificate.

## Shared native coordinator

`rusticol_core::loop_execution::LoopExecutor` owns the complete ordinary execution
sequence over one validated descriptor. Its narrow numerical backend is implemented
in `rusticol_loop_reduction::runtime::physical` using the same saved evaluators,
native coefficient plans, exact kinematics, FeynKit wavefunctions and NinjaRS
facades as the Python consumer. The core has no dependency on PyO3 or NinjaRS.

The experimental `_loop_reduction.NativeLoopProgram(path, ...)` binds this
coordinator; `evaluate(momenta, scales_squared, model_values, point_ids=...)` runs
the entire batch in Rust. `rebind(...)` shares the immutable validated descriptor
and constructs fresh arithmetic workspaces. Whole-schedule aliases share immutable
native output blocks. Returned Born values, all six component batches and order
sums retain their requested carrier and original point identities; each sector
keeps finite, single-pole and double-pole columns. The result records the executed
R2 strategy. No finite-only coercion or second normalization is applied.
Family projectors use the existing interpreter in this initial native entry
point; the separate Python opt-in projector JIT is not exposed here.

This native coordinator admits full-colour legacy/source model-rule programs
and the certified legacy direct four-gluon descriptor. Correlations, formal colour,
automatic rescue and standalone SDK dispatch remain separate work. The default
entry points raise on failed reductions or nonfinite assembled components;
successful reducer statuses alone do not establish numerical accuracy.

The additive `rusticol_core::source_bindings` helper binds the validated source
scalar record over this same numerical backend. `SourceOwner` publishes immutable
snapshots containing native arguments/selected values, original exact overrides
and a revision. Checked updates serialize only the owner's evaluator workspace;
all value, active diagnostic and real-input checks finish before one publication.
Failed updates retain the prior snapshot. Initial overrides overlay the artifact's
captured values, and later updates overlay the current exact override map.

`SourceScale` owns separate ordinary or counterterm evaluator scratch and requires
an explicit snapshot from its owner. It validates positive real per-point
`mu_r2`, applies the saved scale substitution, preserves row IDs and checks paired
diagnostics before selecting values. A retained snapshot stays valid after later
global updates; a snapshot from another owner is rejected, including at a
coordinator's optional early `require_snapshot` check. No source definitions or
Python objects are consulted by these operations. The focused standalone Rust
test `source_scalar_owner` uses the existing source gg PACBIN to exercise all
three carriers, exact parameter bits, old snapshots, scale branches, failed
updates and concurrent publication.

Source-owned `NativeLoopProgram` bindings accept exact complex pairs through
`parameters={name: [real, imaginary]}` and expose a separate `source_state`
handle for atomic updates. They reject positional `model_values`. Each ordinary
evaluation captures one snapshot before numerical work; ordinary coefficients,
pointwise masses, local UV/R2 rules and external fields all use that snapshot.
The result reports its `source_revision`. The internal `snapshot=` execution
argument supports retained-state checks and rejects foreign owners before any
lane is evaluated, including empty calls. Rebinding copies the captured exact
override strings into a fresh owner and workspaces, never rounded native values.
`test_loop_native_source.py` compares complete source gg outputs and nonzero
external fields against the saved-plan Python consumer in all three carriers,
including point IDs, distinct scales, exact parameter changes and retained output
lifetimes. `PYAMPLICOL_NATIVE_SOURCE_PACBIN` can select an already generated
ordinary source gg fixture for these integration checks.

## Ordinary artifact boundary

`generation.loop_artifact.LoopProcessArtifact` publishes a retained source plan
through the existing transactional schema-v3 writer. Its execution capability is
`rusticol.loop-runtime.v3`, with contribution kind `virtual` and execution mode
`loop_recurrence`. The ordinary physics payload carries the order catalogue,
Laurent convention and explicit scheme declaration. A declaration is not a new
physics certification. Source model-rule R2 plus UV, full colour and the three
arithmetic carriers form the initial admitted boundary; aliases, correlated or
resolved output, mixed tree/virtual artifacts and direct R2 are rejected there.
Internal bounded direct controls remain available through their existing API.

`rusticol_core::loop_artifact::open_loop_artifact` (or `load_loop_artifact` with an
already verified artifact) resolves the normal execution-set envelope and opens
the declared PACBIN through its checked file handle. The loader compares order,
state, normalization and strategy metadata against the validated execution plan.
It keeps no producer graphs or source-model import. Existing scalar tree runtime
loads reject the virtual capability. The additive loop consumer owns execution;
the core does not acquire a dependency on the reducer.

Transport checks reuse the saved complete source gg fixture rather than repeat
its physics generation. `PYAMPLICOL_PUBLIC_SOURCE_ARTIFACT` selects the resulting
ordinary two-process artifact for standalone Rust checks of metadata rejection,
process selection, payload replacement and scalar-tree rejection. Public process
generation and the complete SDK/API surface remain separate integration gates.

## Structured reduction failures

The shared coordinator also provides `evaluate_partial_with_snapshot`, exposed
by the internal `NativeLoopProgram.evaluate(..., partial=True)` and native
`VirtualRuntime.evaluate_partial` adapters. It invokes the same execution path
and keeps a compact set of successful **whole physical points**. If any family,
coupling sector or resolved helicity fails, every numerical result row for that
physical point is excluded. A failed channel never contributes a stale buffer,
a zero placeholder or a NaN labelled as a successful value.

`requested_point_ids` retains all original physical IDs. The compact `point_ids`
retains the requested order of successful points; resolved rows retain the
existing fused point/helicity identity. Each typed failure separately records
the original physical `point_id`, optional zero-based `helicity`, `order_index`
in the virtual sector catalogue, `family_index`, execution `stage` and the
reducer's opaque `detail`. These identities come from the scheduled channel
layout, never from parsing reducer error text. Empty and all-failed batches
retain their typed Born/component columns.

Public `Runtime.evaluate` and `evaluate_resolved` continue to raise by default.
Their `EvaluationError.failures` tuple contains immutable
`VirtualFailureDiagnostic` records for native reduction failures; point IDs in
this facade are the original input row indices. Successful public result shapes
and all coupling/Laurent axes are unchanged. There is no public partial-result
policy or hidden precision retry in this slice.

The partial boundary currently starts at reduction. Parameter, scale,
wavefunction/current and nonfinite-assembly errors retain strict whole-call
rejection. Invalid structure and global dependency errors also remain errors.
Extending those early stages to typed lane outcomes and recomputing only failed
points from the original exact inputs are separate requirements for rescue.
Finite output and a successful reducer status still do not certify accuracy.

The focused saved-source test uses exactly on-shell near-forward gluon
scattering where the generic reducer reports a degenerate Gram determinant,
with ordinary points on either side. It checks inclusive/resolved compaction,
sparse original IDs, successful-row agreement, default rejection, all-failed
and empty buffers, and typed public DoubleFloat errors. The binary64 fast
engine succeeds on this particular control; that is not treated as a failure.
