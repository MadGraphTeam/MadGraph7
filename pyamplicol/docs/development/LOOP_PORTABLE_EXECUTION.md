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

The native
reader checks that direct selection is dimensional, uses the certified QCD
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

The experimental Python binding is a thin owner of the core descriptor. The
Python numerical coordinator currently invokes the existing native operations;
there is no reverse dependency from the shared core to PyO3. Moving that
coordinator into the core is a later step required for standalone SDK execution.
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
consumer. These are extraction steps toward native coordination; the complete
process coordinator still runs in Python, and no SDK loop entry point is
activated by this move.

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
