<!-- SPDX-License-Identifier: 0BSD -->

# Internal graph-free loop execution

> Historical: this describes the plan v3 runtime of the per-diagram legacy
> loop engine, removed on 2026-10-29 (MQ-LEGACY-REMOVE; `loops/portable.py`
> and the native `PortableLoopPlan`/`NativeLoopProgram` no longer exist).
> Plan v4 (the v2 engine) is described in `loop_v2/DESIGN.md` §12.

`loops.portable.write_loop_pacbin` lowers a retained, complete source/legacy
model-R2 or certified direct-R2 four-gluon program to the existing
`evaluators.pacbin` envelope.
`PortableLoopPlan.open(...).bind(...)` creates a fresh numerical view in double,
DoubleFloat or arbitrary precision. The same checked execution plan feeds public
saved-artifact loading and the standalone virtual SDK; public generation admits
only its declared supported source-model domains.

The ordinary physical controls are `u u~ -> d d~` and `u u~ -> t t~`.
The latter retains nonzero external-field contributions, independently of the
local UV and R2 currents. The direct control is `g g -> g g`: all three N4 and
dimensional-remainder epsilon sectors survive persistence, with UV currents and
external fields retained and model R2 excluded. Source-owned model-R2 plans retain
one scalar owner and ordinary/CT scale views, including anonymous external-field
coefficients. Explicitly continued full/LC/NLC model-rule colour projections are
persisted with their structural sector degrees. Source dimensional/direct
execution remains outside this boundary. Saved full-colour directed kernels use
explicit native mode selection. Source-owned precision verification/rescue is
explicit and uses the native coordinator described below.

### Cold directed projections

An internal writer may attach `directed=PreparedDirectedModel` to the same
complete model-rule plan. The optional `directed_projection` record declares
`kind="directed-complex"`, the existing colour convention and zero-based physical
colour ports. It references the prepared Born, family, post-integration,
CT/external-field, final-assembly and order-sum evaluator bytes. Current
schedules, scalar owners, denominator families and order catalogues are shared
with the ordinary plan; no producer algebra is repeated during loading.

The two complex interference directions remain separate through integration.
Inputs and outputs pack histories in columns. Order totals sum only the coupling
order axis within each history, returning Born plus eight Laurent triples per
history (full, bare, UV, R2, local UV, external fields, first and second term).
The shared reader derives every evaluator arity from the existing order/current
layouts. It also checks all referenced evaluator members as for ordinary plans.

`directed_histories` may supply existing `ColorCorrelator` declarations. Their
unique IDs and exact ordered operations must match the prepared columns, so
request labels survive serialization. Without declarations, the internal writer
uses `history-0`, `history-1`, etc.; these are column labels, never point IDs.

This cold format currently requires full colour, model R2 and inclusive physical
helicities. Spin sources, resolved directed output and LC/NLC directed projection
remain unimplemented. Old readers reject the unknown optional field. The legacy
Python binder continues to reject these plans. The shared native executor binds
both ordinary and directed projectors and selects the observable explicitly;
ordinary evaluation still uses its original kernels. Public correlated
capabilities require matching artifact metadata, not just extra saved states.

### Native directed execution and verification

`NativeLoopProgram.evaluate_directed` executes all saved histories in one native
call. Physical state/current/source-scalar work is shared with ordinary execution.
Each family reduces channels in history-major, paired-direction, squared-order
order; complex conjugation of the second term occurs in the prepared
post-integration kernel. UV, R2 and external-field kernels retain both directions.
There is no point loop or numerical colour contraction in Python.

The result retains six ordinary component batches, with Born columns
`history × born_order` and virtual columns `history × virtual_order × epsilon`.
Its explicit `observable_kind="directed-complex"`, `history_ids`, `directed_first`
and `directed_second` preserve the meaning of these wider columns. Totals contain
25 values per history: one Born total and eight Laurent triples, including both
cross terms. Ordinary results have no directed fields and keep their existing
19-value totals. Physical point IDs are opaque; internal numerical rows remain
dense, with no multiplication of caller IDs by the history count.

Partial execution removes complete failed physical points and maps reducer
failures to the original point, history, direction, order and denominator family.
Earlier state/scalar failures remain typed at their applicable stage. Survivor
replay uses the same immutable source snapshot. The explicit
`evaluate_directed_verified` path compares every complex per-order component,
both cross terms and the per-history totals, then rescues failing points from
original exact inputs through the existing precision policy. Successful retries
retain earlier diagnostic attempts. Empty/all-failed outputs preserve all widths.

Actual source QCD controls pass in double, DoubleFloat and AP70, including exact
parameter changes, retained snapshots, nonzero imaginary ordered histories,
swapped-history conjugation, Casimir factors and independent analytic Born/UV
checks. Maximum-u64 identities, mixed early/reduction failures and AP60/AP80
subset rescue pass. A distinguishable complex saved-assembly control checks
history labels and first/second output placement. These are execution/conformance
controls; sharing evaluator bytes is not an independent general correlated
physics oracle. Spin sources, resolved directed output, projected-colour directed
output, flavour mappings and public generation composition remain separate gates.

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

This native coordinator admits full/LC/NLC legacy/source model-rule programs
and the certified legacy direct four-gluon descriptor. Correlations remain
separate work; the standalone virtual SDK uses this coordinator. Explicit
source-owned precision verification and rescue are described below. The default
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
physics certification. Source model-rule R2 plus UV and the three arithmetic
carriers form the admitted boundary. Full/NLC resolved output has a contracted
colour component; LC requires saved canonical Born components and resolved
projection kernels. Aliases, correlations, mixed tree/virtual artifacts and
direct R2 remain rejected. Public LC/NLC generation still needs explicit
continuation ingress in prepared models; changing metadata cannot supply it.
Internal certified direct controls remain available through their existing API.

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
coupling sector or resolved helicity/colour component fails, every numerical
result row for that physical point is excluded. A failed channel never contributes a stale buffer,
a zero placeholder or a NaN labelled as a successful value.

`requested_point_ids` retains all original physical IDs. The compact `point_ids`
retains the requested order of successful points; resolved rows retain the
fused identity `((point_id * H + helicity) * C + component)`. Each typed failure
separately records the original physical `point_id`, optional zero-based `helicity`,
`color_component_index`, `order_index`
in the virtual sector catalogue, `family_index`, execution `stage` and the
reducer's opaque `detail`. These identities come from the scheduled channel
layout, never from parsing reducer error text. Empty and all-failed batches
retain their typed Born/component columns.

Public `Runtime.evaluate` and `evaluate_resolved` continue to raise by default.
Their `EvaluationError.failures` tuple contains immutable
`VirtualFailureDiagnostic` records for native numerical failures; point IDs in
this facade are the original input row indices. Successful public result shapes
and all coupling/Laurent axes are unchanged. There is no public partial-result
policy. Precision retry is explicit through the policy below.

The same boundary includes external-state preparation, per-point scale-dependent
scalar evaluation, nonfinite currents, family projections/kinematics and final
assembly. State failures also carry `external_leg`; order/family fields are
absent when a stage has no corresponding identity. Exact original input literals
and positive scales are validated before removing any point. Active scalar
value/diagnostic pairs are checked together, including their imaginary parts.

State and scale blocks compact before current construction. A later nonfinite
stage returns a typed loop-local outcome with all failures observed so far. The
same executor removes those whole physical points and replays the surviving
batch from its original exact inputs. This correctness-first rare-failure path
can repeat work for survivors. Each replay must strictly reduce the active
physical-point set, and there is no arbitrary retry cap. All-failed execution
finishes with a typed empty result. One source snapshot is captured before the
first attempt and reused for every replay; parameter updates cannot change the
calculation between stages. Unexpected structural/dependency errors still raise.

Global source-owner construction and parameter updates retain whole-handle
checks. They can reject a numerically overflowing lower-carrier scalar even
when its exact model expression is valid; this is distinct from a malformed
model. Recovery from a failed global binding is not supplied by this boundary.
Finite output and successful reducer status alone do not certify accuracy.

The focused saved-source test uses exactly on-shell near-forward gluon
scattering where the generic reducer reports a degenerate Gram determinant,
with ordinary points on either side. It checks inclusive/resolved compaction,
sparse original IDs, successful-row agreement, default rejection, all-failed
and empty buffers, and typed public DoubleFloat errors. The binary64 fast
engine succeeds on this particular control; that is not treated as a failure.

The stage tests combine distinct sparse-ID failures at state construction,
scale ingress, real Gram reduction and an explicitly injected late counterterm
current. They compare the surviving output with a separate run and ensure the
late replay retains earlier failure records. An injected resolved Born
projection also checks all-failed point/helicity diagnostics. These injected
controls test execution transport, not physics accuracy.


## Explicit cross-precision verification and rescue

`Runtime.load` accepts an optional retained `VirtualStabilityPolicy`, applied
before exact source binding. Ordinary, resolved, directed and spin evaluations
use it unless the call supplies an explicit replacement. Without either policy,
evaluation performs no extra precision work. For example:

```python
from pyamplicol import Runtime, VirtualArithmeticSpec, VirtualStabilityPolicy

policy = VirtualStabilityPolicy(
    higher=(VirtualArithmeticSpec("arbitrary", 60),
            VirtualArithmeticSpec("arbitrary", 80)),
    relative_tolerance="1e-8",
    absolute_tolerance="1e-20",
    verify_successful=False,
)
runtime = Runtime.load(artifact, stability=policy)
result = runtime.evaluate(points, mu_r2=scales)
```

The native coordinator validates actual increasing precision bits: double has
53 bits, DoubleFloat 106, and arbitrary precision uses the dependency's actual
requested-bit conversion. A numerically lower arbitrary setting cannot follow
DoubleFloat just because its enum is different. Tolerances are exact finite,
nonnegative decimal strings, not both zero. There is no hidden policy or
per-event Python retry loop.

With `verify_successful=False`, initially successful points are returned
unchanged and explicitly remain accuracy-unestablished. Only failed points
enter the higher-carrier batches. With `True`, successful points also enter
verification; agreement with a higher evaluation preserves the original result.
When the primary failed or disagreed, rescue requires two successful increasing
precision evaluations that agree. Failed intermediate attempts do not erase a
previous successful candidate. Comparison includes all Born values, every
physical/local component, order and Laurent coefficient, and every requested
resolved helicity and colour component. It uses the shared native
`Batch.agreement_rows` rule on both real and imaginary parts:
`abs(a-b) <= atol + rtol*max(abs(a),abs(b))`.

A rescued output is converted to the caller's requested carrier only after
agreement. The converted result must satisfy the same bound against the
accepted higher value; a bound that cannot be represented by the requested
output precision leaves the point unresolved. The requested carrier and R2
strategy never change implicitly. Agreement is empirical numerical evidence,
not an independent physics proof or an estimated number of reliable digits.

One immutable source snapshot is captured at call entry. Each higher owner
re-evaluates that snapshot's exact overrides and saved default/formula
programs, without merging previous retry state or publishing a parameter
update. Native retries select original decimal momentum/scale rows; they never
promote the rounded primary buffers. Higher workspaces are cached by explicit
carrier/precision and cleared by `clear`, while source parameter state remains.
With a configured policy, a numerical primary source-binding failure can retain
the exact overrides and defaults when a configured higher realization succeeds.
That admission establishes no amplitude accuracy: the ordinary complete-result
agreement and output-conversion checks still apply. Failed source snapshots
expose their failure stage but refuse numerical `arguments` and `values` access.
Malformed inputs and unsupported settings remain errors. Parameter updates
validate before publishing; a rejected update preserves the previous revision.

An arithmetic-changing call supplies its explicit policy before constructing
the alternate source owner. The alternate handle is local to the call, so it
cannot replace the original runtime's retained policy. `clear` releases work
buffers while preserving exact parameter state and policy. Standalone Rust,
C, C++ and Fortran expose additive policy-bearing open operations and retain
binary64 result carriers. Raw native complete-result operations reject a
configured policy rather than silently discard verification diagnostics.

Each public lane can carry a `VirtualVerificationDiagnostic` with original
point ID, attempted carriers/bits, numerical failures, comparison precision,
agreement and conversion outcomes. Accepted rescue histories retain earlier
failures without making the completed call fail. Unresolved points populate
`EvaluationError.failures` and `EvaluationError.verification`; successful
numerical attempts without accepted agreement are distinguished as
`accuracy-not-established`. No unavailable values are filled with zeros or NaNs.

The source-binding transport control adds an overflowing exponential diagnostic
to an existing real-input guard, leaving all saved physical instructions and
scalar values unchanged. Public resolved LC components and complex joint-spin
results agree with their unmodified controls after AP60/AP80 rescue. The native
owner tests cover eager rejection, exact snapshot ownership, atomic updates and
policy replacement; a standalone Rust control checks the ordinary physical
result and owned rescue diagnostics. Focused C, C++, Fortran and installed Rust
clients exercise the new policy-bearing open operation and empty-result lifetime.
These are correctness controls, not performance measurements or new physics
certification. The live-producer stability helper also preserves exact source
bindings through numerical failures in its full-colour, real-mass domain,
including the retained massive-W control; its complete six-field controls and remaining
limits are described in [the model implementation notes](LOOP_MODEL_IR.md#live-producer-source-binding-rescue).

## Explicit partial-value returns

All public virtual evaluation methods accept the per-call option
`on_failure="return"`; the default `"raise"` retains `EvaluationError` and its
failure/verification payload. The option applies to ordinary, resolved,
precision-specific, correlated and grouped source requests. It does not change
the arithmetic, configured verification policy or runtime defaults.

```python
result = runtime.evaluate_resolved(points, mu_r2=scales, on_failure="return")
for point, diagnostic in enumerate(result.diagnostics):
    if diagnostic.valid:
        coefficients = result.values[point]
    else:
        assert result.values[point] is result.born_values[point] is None
        failures = diagnostic.failures
```

The existing immutable `VirtualEvaluation` and `ResolvedVirtualEvaluation`
types retain every original input position. Native execution still returns
compact complete successes; Python scatters whole-point output blocks without
an event-evaluation loop. Missing values are `None`, never zero, NaN, a surviving
helicity subset or an unaccepted last precision attempt. A physical zero remains
a valid numerical value. Empty input yields empty values/diagnostics; an
all-failed batch retains its input length and complete order/shape metadata.

Each unavailable lane retains the existing `VirtualFailureDiagnostic` records,
including original point ID and available helicity, colour component, squared
order, family, external leg, history and direction. If numerical values were
obtained without accepted precision agreement, its status is
`accuracy-not-established`, still invalid with absent values. Successful rescue
keeps earlier failures in its verification attempts and exposes no unrecovered
lane failures. Source admission alone cannot turn one high-precision amplitude
into an accepted result.

Finite/single/double-pole access, `for_order()`, `sum_orders()` and resolved
`.total()` preserve missing entries and immutable diagnostics. Channel indices
inside diagnostics retain the original generated-catalogue meaning after output
selection; they are not silently renumbered. High-precision successful values
remain Decimal or complex Decimal pairs, including directed outputs.

Grouped requests keep their label order. Histories sharing one source assignment
share native whole-point validity; distinct assignments can have different
failure masks. A failing explicit assignment does not alter the retained spin
setter or prevent other admitted source assignments from returning their values.
Malformed inputs, invalid scales, unavailable capabilities/histories, corrupted
ownership/metadata and rejected source updates remain hard errors under either
option. This Python option does not replace the SDK's existing successful-ID
and per-request status arrays. CLI policy transport is a separate open gate.

## Signed LC Born components

The plan's optional `born_color_components` record is the sole partition
catalogue: `kind="canonical-born-tensor-partition"`, deterministic component
`ids`, and `members` containing coherent Born-column indices. Core validation
requires every Born column exactly once and checks the expanded projection
kernel widths. The dependency canonicalizes exact colour tensors with fixed
external ports; pyAmpliCol does not solve new colour identities. Equal canonical
tensors share an anchor, so the groups can remain overcomplete.

The common leading colour degree of each complete squared-order sector is
chosen before anchor masking. Born components are `Re(B_anchor† G B)` and virtual
components are `2 Re(B_anchor† G_BL L)` after complex loop integration. UV, R2 and
external-field projections use the same partition. Zero ordinary overlaps remain
explicit components. These signed contributions are not tree flows or sampling
probabilities, and do not claim to form a minimal independent colour basis.

Native currents run once on point/helicity rows; only projection outputs expand
to point/helicity/component rows. The Python and standalone transports reorder
these into `[point, order, helicity, component, epsilon]`, with the Born epsilon
axis omitted. All three arithmetic carriers preserve the same layout. The
immutable `.total()` result sums helicity/component axes and preserves squared
orders, Laurent coefficients and colour accuracy. Text and JSON output keep
component identities and signs. Public `ProcessPhysics` exposes virtual anchors
through `born_anchored_color_components` while tree `color_flows` stay unchanged.
Native errors and verification histories identify original physical points and
optional colour components; one failed component removes or rescues the whole
point using its original exact inputs.

### Public saved directed colour histories

`Runtime.load` validates that the artifact's `correlations` capability matches
its saved directed descriptor. `available_color_correlations()` exposes the
ordered bra/ket catalogue using existing one-based public leg labels. Native
nonnegative labels shift by one, while negative auxiliary labels stay unchanged;
this is a label conversion and applies no charge sign or crossing operation.
`evaluate_correlated` and
`evaluate_correlated_many` dispatch to the same native virtual handle; one bulk
call evaluates the saved histories and Python selects labelled output columns.
They return `VirtualEvaluation(correlated=True)` with every squared coupling
order and finite/single-pole/double-pole coefficient. Binary64 values are complex;
DoubleFloat/arbitrary values use decimal-preserving `CorrelatedValue` pairs.
Ordinary `evaluate` still selects its ordinary real projection.

The reserved public `born` request selects an explicitly named empty history,
or the unique saved history with empty bra and ket. Ambiguous unnamed identities
are not guessed. This derived alias does not add a native history or change its
column order. Model parameters, `mu_r2`, arithmetic, strategy and optional native
verification use the ordinary exact-input interfaces. Errors preserve physical
point identity and available history/direction labels. Verification compares
both complex cross terms as well as their complete sum.

This initial saved-public domain is full colour, model R2 plus UV and inclusive
physical helicities. Spin replacement, selected helicities, resolved directed
outputs, LC/NLC composition, FFT and mapped processes are rejected. Public
source generation now accepts the existing `CorrelatorConfig`, with staged
ordinary and directed validation; see `LOOP_PUBLIC_GENERATION.md`. Fresh CLI
quark-scattering generation additionally checks the nontrivial T12 dipole against
Recola at two points/scales in all three carriers. This current developer-runtime
proof is distinct from the earlier installed-wheel ordinary/LC/NLC milestone.
Tree correlated execution and its defaults are unchanged.
