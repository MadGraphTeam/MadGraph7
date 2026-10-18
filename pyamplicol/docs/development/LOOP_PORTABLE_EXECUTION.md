<!-- SPDX-License-Identifier: 0BSD -->

# Internal graph-free loop execution

`loops.portable.write_loop_pacbin` lowers a retained, complete full-colour
legacy model-R2 program to the existing `evaluators.pacbin` envelope.
`PortableLoopPlan.open(...).bind(...)` creates a fresh numerical view in double,
DoubleFloat or arbitrary precision. This is an internal execution roundtrip;
public artifact generation/loading and standalone SDK loop dispatch remain gated.

The initial physical control is `u u~ -> d d~`. It has nonzero UV and R2
contributions but **no external-field currents**. External-field payload slots
are represented; this control does not certify their nonzero execution. Direct
R2, source-owned scalar plans, formal colour projection, correlations and rescue
are not admitted by this first portable boundary.

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
  rows; exact model-input names and integer column maps.
- Symbolica's saved local evaluator bytes and native coefficient arena slots.
- The already proven execution-sharing groups. Aliases reference the exact
  representative schedule and ordered columns that the original hot path uses;
  no symbolic sharing proof is reconstructed on loading.
- Numerical loop-output and order-sector indices, denominator counts, coefficient
  powers, exact kinematic/mass/projector evaluator states and numerator strings.
- Saved Born, counterterm, final-assembly and order-summing evaluator states.
  Counterterm execution retains scalar diagnostics and input columns, without
  source insertion or tree objects.

Producer graphs, recurrence objects, symbolic Python objects, colour tensors,
live evaluator workspaces, model values and event results are excluded.
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

NinjaRS currently exposes no public serialization/rebinding interface for its
private generic Laurent expansion plan. Loading therefore constructs a fresh
`PolynomialPlan` from retained exact numerator strings and ordered coefficient
parameters. This preserves dependency-owned algebra, but does not remove that
cold setup cost or establish cross-version NinjaRS binary-plan compatibility.

`tests/unit/test_loop_portable.py` exercises the complete control in all three
carriers, native result/old-buffer lifetime, changed exact parameters/scales,
empty batches and noncontiguous IDs. It also loads in a fresh process without a
source UFO, with producer constructors disabled, and checks malformed layouts
through the Rust reader. Independent MadLoop components are checked when
`PYAMPLICOL_UUDD_COMPONENT_REFERENCE` is provided. Native success remains
numerical status rather than an independent accuracy certificate.
