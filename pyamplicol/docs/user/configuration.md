---
title: "Configuration"
nav_order: 3
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Configuration

pyAmpliCol uses one typed configuration schema for TOML run cards, direct CLI
options, and Python configuration classes. Unknown fields are rejected rather
than ignored, with a nearest-field suggestion when a likely spelling exists.

The complete field-by-field reference is the shipped
[`examples/all_options.toml`](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/all_options.toml).
This page focuses on practical configurations.

## Minimal generation card

```toml
schema_version = 1
action = "generate"

[model]
source = "built-in-sm"

[process]
entries = [{ expression = "d d~ > Z g", name = "ddbar_Zg" }]

[color]
accuracy = "lc"

[generation]
output = "artifacts/ddbar_Zg"
```

Run it with:

```console
pyamplicol run.toml
```

Equivalent direct command:

```console
pyamplicol generate 'd d~ > Z g' artifacts/ddbar_Zg \
  --model built-in-sm --color-accuracy lc
```

## Configuration precedence

Values are resolved in this order:

1. schema defaults;
2. TOML card values;
3. dedicated command-line flags;
4. repeated `--set section.field=value` overrides, from left to right;
5. effective generation requirements and license/resource adjustments.

For example:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml \
  --set generation.workers=2 \
  --set generation.mode=replace
```

Requested and effective configurations are retained separately in generated
artifact provenance. A restricted Symbolica environment may reduce effective
generation resources without rewriting the requested card.

Inspect resolution without running the action:

```console
pyamplicol config resolve generate_pp_zjj_from_ufo_sm.toml
pyamplicol config resolve generate_pp_zjj_from_ufo_sm.toml --json
```

Create a commented template:

```console
pyamplicol config template run.toml
```

## Paths

Paths in a TOML card are resolved relative to that card, including:

- model sources;
- artifact outputs and inputs;
- model-parameter cards;
- momenta files;
- cache paths.

This makes copied example workspaces movable. Direct CLI paths are resolved
from the current working directory.

Generation output policy is explicit:

| `generation.mode` | Behavior |
| --- | --- |
| `error` | Default; refuse an existing destination. |
| `replace` | Atomically replace the generated artifact. |
| `append` | Add to an existing compatible artifact. |

Use `replace` deliberately while iterating:

```console
pyamplicol generate --card run.toml --set generation.mode=replace
```

## External JSON or UFO model

The primary example uses serialized JSON, which does not execute Python while
loading:

```toml
[model]
source = "models/json/sm/sm.json"
restriction = "default"
simplify = true
cache = true
```

A trusted UFO directory uses the same field:

```toml
[model]
source = "/absolute/path/to/MyUFO"
restriction = "default"
```

UFO modules are Python code and execute during loading. See
[Models and Processes](models-and-processes.md) before using an external model.

## Processes and multiparticles

One card may contain one inclusive multiparticle request or multiple named
requests:

```toml
[process]
entries = [
  { expression = "p p > Z j j" },
  { expression = "u u~ > Z g", name = "uubar_Zg" },
]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]
```

Every model provides the generic `all` multiparticle for valid propagating
physical external particles. It excludes ghosts, Goldstones,
non-propagating records, and auxiliary states. An explicit `all` entry replaces
the default; other user labels are merged over model defaults.

Broad products such as `p p > all all` can expand combinatorially. Prefer a
narrow user-defined label for production scans.

`flavor_scheme`, `max_quark_lines`, and coupling-order settings constrain that
expansion. Coupling-order names come from the selected model: the generic
engine treats them as model-defined filtering and scheduling data and does not
assign arbitrary names fixed QCD or electroweak meanings.

### Process groups

Four `process` fields control which subprocesses are recorded as sharing one
matrix element in the artifact's `process-groups.json`:

```toml
[process]
flavor_blind = false                 # generate one subprocess per flavour class
consider_charge_conjugation = true   # also relate charge-conjugate subprocesses
only_unit_multiplier = false         # accept numerical relations only with factor 1
analyze_process_groups = "none"      # or "numerical", run after generation
```

`flavor_blind` is an assumption, not a check: members whose flavour relabelling
is not a symmetry of the model couplings are reported with a warning.
`consider_charge_conjugation` matters only with `flavor_blind` or numerical
analysis. See
[Group equivalent subprocesses](command-line-interface.md#group-equivalent-subprocesses)
for the CLI flags and the `group-processes` command, and
[Process Grouping](process-grouping.md) for the full description.

## Color accuracy and LC layout

```toml
[color]
accuracy = "lc"                 # lc, nlc, or full
contraction = "auto"            # auto, direct, or symmetric-group-fft
fft_basis = "adjoint"           # trace or adjoint; used by FFT contraction
lc_flow_layout = "topology-replay"
```

| Accuracy | Resolved color output |
| --- | --- |
| `lc` | One entry per physical leading-color flow. |
| `nlc` | One contracted color entry per helicity. |
| `full` | One contracted color entry per helicity. |

LC offers two complete-coverage layouts:

| Layout | Best suited to |
| --- | --- |
| `topology-replay` | Default: one selected flow with a helicity sum. |
| `all-flow-union` | All flows with one selected helicity. |

Both preserve all physical flows and helicities for runtime selection.
`all-flow-union` is LC-only and is incompatible with generation-fixed or
truncated color/helicity coverage.

For contracted NLC/full output, `symmetric-group-fft` evaluates the same exact
colour interference as `direct` while Fourier-transforming certified
permutation orbits and retaining all other terms as direct residuals. It is
available for `recurrence` and `on-the-fly`; compiled/eager execution and LC
flows deliberately reject explicitly forced FFT.

The default `contraction = "auto"` tries FFT for NLC/full recurrence and
on-the-fly execution, falling back to direct contraction when an FFT plan is
unsupported. LC, compiled/eager execution, and correlated generation use
direct contraction and the trace basis. The effective configuration can retain
`"auto"` until per-process selection.

The default `fft_basis = "adjoint"` enables adaptive exact-basis selection.
The requested option and the mathematical representation are
distinct: generation chooses a certified Del Duca–Dixon–Maltoni (DDM) reduction
where available, and otherwise keeps the exact representation appropriate to
the process.

| Process domain | Actual representation |
| --- | --- |
| Certified Yang–Mills trees | Two-anchor DDM half-ladders. |
| Certified single-insertion scalar HEFT, with gluons and one Higgs | DDM half-ladders; the two-gluon case is the single delta tensor. Use explicit `HIG = 1`. |
| One quark–antiquark line, with optional colour singlets | Fundamental chain, already fixing both quark endpoints. |
| Multiple quark lines | Exact products of fundamental chains and all endpoint routings; not a Melia/Johansson–Ochirov primitive decomposition. |
| Colour singlets only | The trivial singlet tensor. |
| Adjoint interactions without a DDM certificate | Original trace tensors, without assuming amplitude identities. |

For `n` external gluons, a certified DDM reduction retains `(n-2)!` ordered
amplitudes instead of `(n-1)!`. Count only the gluons, not a colourless Higgs.
This removes redundant orderings without changing colour accuracy. A quark
process need not become smaller or faster: retaining its existing fundamental
chains is a deliberate exact choice, not an adjoint reduction.

Generation records `fft_basis_selection`, with `requested`, `actual_basis`,
`name`, `reason`, `permutation_degree`, and `tensor_count`. For fundamental
chains, `actual_basis = "trace"` identifies the existing metric implementation,
while `name = "fundamental-chain"` or `"fundamental-chain-products"` identifies
the tensors. The tensor count excludes traversal aliases and is not a rank
claim at fixed `Nc`. Use these recorded fields when comparing artifacts.

Explicit `fft_basis = "trace"` keeps the original FFT representation.
`contraction = "direct"` opts out of FFT and ignores `fft_basis`. Explicit
`contraction = "symmetric-group-fft"` forces FFT and retains errors for
unsupported plans, LC, and compiled/eager execution. Forced adjoint FFT cannot
be combined with correlated generation; correlations retain their separate
direct contraction path.

This is an exact contraction algorithm, not an approximation. Its speedup is
process-dependent: a small certified symmetry subgroup or a residual-dominated
contraction can make the FFT and direct curves scale similarly. Keep the
direct result as the baseline when characterizing a new process family.
For the adjoint basis, single-helicity full-colour recurrence benchmarks of
`g g > (n-2) g` on one core showed it faster than trace for `n >= 6`: at
`n = 10` it evaluated 3.9x faster per sample (57.8 versus 227 ms), generated
4.2x faster and used 2.4x less memory during generation. Below `n = 6` the two
bases are comparable, so compare setup and warmed runtime for other workloads. See
[FullColor FFT Profiling](fullcolor-fft-profiling.md) for the resumable
comparison driver and published snapshots.

```toml
[color]
accuracy = "full"
contraction = "symmetric-group-fft"
fft_basis = "adjoint"  # for example, built-in-sm with g g > g g g

[evaluator]
execution_mode = "recurrence"
```

This example forces FFT. Omit `contraction` and `fft_basis` to use automatic
adjoint selection with direct fallback. Use `execution_mode = "on-the-fly"`
for the compact OTF lane. The matching forced-FFT CLI shorthand is
`--fft adjoint --color-accuracy full`; `--fft trace` forces the original trace
path. Both require an explicit NLC/full accuracy setting (on the command line
or in the card). Use `--color-contraction direct` to opt out.

## Execution mode and evaluator backend

The accuracy, contraction, and execution choices on this page describe
ordinary Born evaluation. [Born Correlations](../correlators.md) are a separate
opt-in declaration: pass `CorrelatorConfig` as `correlators=` in Python or use
`generate --correlators declaration.json`. There is no `[correlators]` TOML
section. `color.accuracy` selects LC, NLC or full colour for the correlated
matrix contraction; all three retain complete underlying amplitudes. The
path records direct contraction, compiled execution, and numerical current
reuse off. Conflicting requested settings
are retained in provenance with those explicit adjustments.

Correlated generation rejects partial colour/helicity coverage and append
mode. Numerical spin vectors are supplied later through the Python runtime,
not through the generation card or declaration file.

```toml
[evaluator]
execution_mode = "compiled"     # recurrence, compiled, eager, on-the-fly
backend = "jit"                 # jit, asm, cpp
batch_size = 128

[evaluator.jit]
optimization_level = 2
compress = "auto"               # true for compiled JIT O2; false for other modes/levels
```

### Execution modes

| Mode | What is stored in the process artifact |
| --- | --- |
| `recurrence` | Compact current-recursion schedules over prepared local kernels. |
| `compiled` | Process-wide stage evaluators compiled during generation. |
| `eager` | Compact DAG invocation tables over prepared local kernels. |
| `on-the-fly` | A compact process seed; the selected LC query family, or contracted NLC/full family, is constructed on first use. |

`recurrence` is the default. Eager, recurrence, and on-the-fly require a
compatible prepared `.pyamplicol-model` kernel pack; `built-in-sm` selects the
packaged JIT O2 pack automatically. Raw JSON/UFO model IR normally uses
`compiled` unless you first create a prepared bundle.

Ordinary recurrence, compiled, and eager generation may fix
`process.selected_color_sector_ids`, `process.selected_source_helicities`, or
both at generation time when a deliberately specialized artifact is useful.
Omit them to retain reusable runtime selectors. On-the-fly keeps selection at
runtime instead: its last warmed family is retained until another selector is
requested.

For reusable contracted recurrence artifacts, generation persists one shared
all-helicity physical-colour plan plus an exact per-helicity dispatch sidecar.
Cold selector binding copies only the dependency-closed active row groups;
warmed evaluation does not scan helicity-support masks. This companion belongs
to recurrence execution. On-the-fly instead constructs the requested family
from its compact process seed on first use and caches that family; compiled and
eager artifacts retain their own execution layouts.

On-the-fly is native binary64 only. LC retains physical flow and helicity
selection without materializing either LC layout. NLC and full colour expose
one contracted color component, accept helicity selection, and reject LC-flow
selectors. Their cold family is correctness-oriented and may become
impractical at high multiplicity. For on-the-fly generation,
`evaluator.optimization.cores` requests query-construction workers for the cold
warm-up; it does not make steady-state numerical evaluation use that many
threads. See [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
for the distinct lifecycle of each mode.

### Backends

| Backend | Notes |
| --- | --- |
| `jit` | Default direct SymJIT application. Compiled O1/O2 artifacts are portable on supported 64-bit little-endian hosts; prepared packs use exact O2. |
| `asm` | Target-native Symbolica assembly evaluator. |
| `cpp` | Target-native generated C++ evaluator. |

Explicit JIT O0/O3 and ASM/C++ artifacts remain target-native. See
[Artifacts and Portability](artifacts-and-portability.md).

## Generation validation

```toml
[generation.validation]
enabled = true
samples = 2
seed = 12345
relative_tolerance = 1e-12
absolute_tolerance = 1e-300
post_build_validation = false
```

Generation validation checks symbolic/numerical construction before
publication. Structural artifact checks always run. The optional
`post_build_validation` reopens the finished artifact and evaluates it again;
it is disabled by default because its cost can dominate large process builds.
Enable it when you explicitly want that additional runtime smoke:

```console
pyamplicol generate --card run.toml \
  --set generation.validation.post_build_validation=true
```

Known limitation of the legacy loop engine: with colour-correlated outputs
declared (for example the Casimir and dipole histories of `g g > g g`), its
post-build validation can fail at the default tolerances ("saved virtual
execution differs from source execution for 'gg/casimir'", an absolute
difference of about 3e-10). Validate such legacy artifacts with
`relative_tolerance = 3e-9` and `absolute_tolerance = 2e-12`; the v2 engine
is not affected.

## Numerical current reuse

Certified current-relation reuse is an opt-in optimization. It is off by
default because its high-precision probes are costly at generation time;
enable it to search for additional reuse. It prefers exact
structural proofs and may apply independently verified high-precision equal,
opposite, or zero relations. The artifact records the evidence, tolerances,
probe derivation, replay identity, and whether a structural proof was present.

This configurable discovery pass applies to ordinary recurrence, compiled,
and eager generation. On-the-fly uses its compact source projection instead
and does not run relation discovery. Correlated generation always disables
this pass to preserve arbitrary external vector-source dependence.

`diagnostic` records deterministic candidates and their exact replay without
changing the generated evaluator. `certified-reuse` may apply a relation only
after its independent verification pass succeeds; it emits one warning when
an applied relation has numerical evidence but no structural proof. Malformed,
non-finite, unstable, or stale evidence fails closed and is never reused.

Discovery and application are reported separately. For contracted NLC/full-colour
recurrence, independently verified equal and zero-current relations can be
applied even when the same pass also finds opposite-current relations. The
opposite relations remain recorded but are not applied in that layout; their
currents retain their original computation. Existing helicity/flow-selector
restrictions still apply. No relation search is deferred to the first evaluation.

Enable the search with:

```console
pyamplicol generate --card run.toml --numerical-current-reuse
```

Or in TOML:

```toml
[generation.relation_discovery]
mode = "certified-reuse"        # off (default), diagnostic, certified-reuse
```

`--no-numerical-current-reuse` restates the default and overrides a card that
enables the search.

The exhaustive probe, precision, seed, and tolerance fields are documented in
`all_options.toml`.

## Evaluation card

```toml
schema_version = 1
action = "evaluate"

[evaluation]
artifact = "artifacts/pp_zjj"
process = "d d~ > g z g"
precision = 16
resolved = true
model_parameters = "data/model_parameters.json"
momenta = "data/pp_zjj_momenta.json"

[output]
format = "human"                # programmatic/config provenance default
color = "auto"                  # auto, always, never
progress = "off"                # auto, tty, log, off
```

`helicity_ids` and `color_flow_ids` may be added under `[evaluation]`. Empty
lists mean complete retained coverage. `orders = true` (instead of `resolved`)
splits recurrence tree totals by squared coupling order.

For a saved virtual artifact, use the same card with `resolved = false` and
explicit scale input. The optional arithmetic choice applies before model
parameter binding:

```toml
[evaluation]
artifact = "artifacts/virtual"
process = "gg"
momenta = "data/points.json"
arithmetic = "arbitrary"        # double, double-double, arbitrary
precision = 70
mu_r2 = "1000000"              # exact text, positive scale squared
# mu_r2_file = "data/scales.json"  # alternative: one exact value per point

[loop]
r2_strategy = "model"          # auto or an available complete artifact strategy
```

`mu_r2` and `mu_r2_file` are mutually exclusive. Quote scalar `mu_r2` in TOML
to preserve its decimal value exactly. The scale file accepts numeric or string
JSON entries. No `[process].loop_coupling_orders` declaration is needed to load
an existing virtual artifact. See [virtual CLI evaluation](command-line-interface.md#saved-virtual-artifacts)
for output shapes and currently available features.

Optional virtual verification and partial numerical returns use the same policy
as the Python runtime:

```toml
[evaluation]
artifact = "artifacts/virtual"
momenta = "data/points.json"
arithmetic = "double"
mu_r2 = "1000000"
on_failure = "return"         # default: "raise"

[evaluation.stability]
higher = ["arbitrary:60", "arbitrary:80"]
relative_tolerance = "1e-8"
absolute_tolerance = "1e-20"
verify_successful = false
```

The policy is bound before the model parameters. `higher` is an ordered list of
`double`, `double-double`, or `arbitrary:DIGITS`; only arbitrary arithmetic takes
a digit suffix. Quote both tolerance strings to preserve their exact values.
Higher levels require both tolerances, and policy fields without higher levels
are rejected. Native admission checks increasing actual precision; configuration
does not infer a carrier from a digit count.

The default empty `[evaluation.stability]` enables no verification or rescue.
With `on_failure = "return"`, numerical failures become unavailable whole-point
rows with diagnostics (`null` in JSON); valid zero values remain zero. Invalid
inputs and unsupported features still raise errors. These settings apply only to
virtual `evaluate`, including resolved output, and are not silently used by
profiling or generation.

### The v2 loop engine

`[loop] engine = "v2"` generates virtual processes with the shared-recursion
one-loop engine (no loop diagram is listed) and publishes them as plan v4
processes (runtime capability `rusticol.loop-runtime.v4`). Until the counterterm
recursion is integrated, this development build closes the counterterms of a v2
process per site (a development stage, not the final production path). The
default `engine = "legacy"` keeps the per-diagram engine, which remains
available as an oracle. A v2 evaluation can skip the poles and return stability
estimates:

```toml
[evaluation]
artifact = "artifacts/virtual-v2"
momenta = "data/points.json"
mu_r2 = "1000000"
poles = "none"                 # finite part only; poles are absent (null)

[evaluation.stability_checks]
rotations = ["z:3/5:4/5"]      # exact "<x|y|z>:<cos>:<sin>" or "<ax>,<ay>,<az>:<radians>"
# rotation_count = 2           # alternative: that many default rotations
reduction_direction = true     # re-reduce every loop numerator, reversed frame
```

Each check re-evaluates or re-reduces every point; the result then carries one
`stability` estimate per point (relative accuracy of the whole virtual, of the
finite part and, with poles, of each pole, plus each check's own budget).
Without checks no estimate is returned. Escalating to `arithmetic = "arbitrary"`
is left to the caller; `[evaluation.stability]` rescue policies belong to the
legacy engine. `poles` is a pole selection (`PoleSelection` in Python):
`"all"` (the default), `"none"`, `"uv"` or `"ir"`; the finite part is the same
under every selection, and `"none"` saves the pole work. The CLI spells these
`--loop-engine v2`, `--poles {all,none,uv,ir}`,
`--stability-rotation SPEC` (repeatable), `--stability-rotations COUNT` and
`--stability-reduction-direction`.

#### Ultraviolet and infrared poles

With `poles = "uv"` or `"ir"` every pole of the Born-virtual interference is
split by origin, `P/eps = P_UV/eps_UV + P_IR/eps_IR`; the double pole is
infrared. The returned coefficients are then those of the selected origin:

- `"ir"`: the infrared poles of the renormalised virtual. They equal the poles
  of `"all"` (Catani's operator), because the renormalised virtual has no
  ultraviolet pole.
- `"uv"`: the ultraviolet poles of the renormalised virtual, zero up to
  rounding. A model states its external-field factors with one eps, so their
  ultraviolet part is fixed by this zero; an incomplete ultraviolet
  renormalisation shows instead in the `"ir"` (and `"all"`) poles, which then
  miss Catani's operator.

Both selections cost the same and both also return each origin's parts by
component, `VirtualEvaluation.pole_parts` (per point, `uv` and `ir`, each
component a Laurent triple per sector; `pole_part(origin, component)` reads one):
`virtual` (renormalised), `bare` (the bare loop), `r2`, `uv_local` (the model's
coupling and mass counterterms), `uv_external_fields` (the on-shell field
factors, whose ultraviolet part follows from the ultraviolet finiteness of the
renormalised virtual) and `uv` (`uv_local + uv_external_fields`). So `"uv"`
shows the bare-loop UV poles and their cancellation against the counterterms
term by term. In Python:

```python
result = runtime.evaluate(points, mu_r2=scales, poles="uv")
bare_uv = result.pole_part("uv", "bare")          # [point][sector] (finite, 1/eps_UV, 0)
counterterms_uv = result.pole_part("uv", "uv")    # the opposite, to rounding
```

The selection is a runtime option: the same artifact serves all four (it stores
the companion plan of its certified-scaleless loop families, which only `"uv"`
and `"ir"` evaluate; at g g > g g g g they cost about 20 % more than `"all"`).
Artifacts written before the split serve `"all"` and `"none"` and refuse
`"uv"`/`"ir"` with a request to regenerate. Correlated and resolved outputs
take every selection too; their `pole_parts` follow their own axes (resolved:
per sector, helicity and colour component).

Current v2 scope: model-supplied R2; full, NLC or LC colour (LC/NLC with the
model's source colour declaration); double, double-double and arbitrary
arithmetic; inclusive results per squared coupling order; colour and spin
correlations declared at generation (`correlators`); helicity-resolved output
(also per Born colour component at LC; one contracted component at NLC and full
colour); exact flavour grouping (`process.flavor_blind`: an identity-port member
such as `s s~ > g g` of `d d~ > g g` is served by its representative). `poles`
and `[evaluation.stability_checks]` apply to correlated and resolved evaluations
too (resolved estimates are per helicity row and colour component).

Coupling orders. The loop orders follow the process's Born amplitude orders:
one loop amplitude per Born amplitude order (within the process's coupling-order
limits) and correction in `process.loop_coupling_orders`, so mixed Born orders
(e.g. `u u~ > d d~` with QCD and QED Born amplitudes) and several corrections are
generated together, each squared coupling order a separate partial result. A
correction the model does not expand perturbatively is refused. What the engine
does not admit yet fails at generation with the reason, for example:

- Z or W couplings on a quark line inside a QCD loop (`u u~ > d d~` with mixed
  Born orders in the SM: "not a pure vector coupling; its sign needs the
  electroweak oracles"), and electroweak bosons in the final state with QCD
  corrections (`u u~ > e- e+ g g`);
- QED corrections to quark processes (`u u~ > d d~` with QCD and QED corrections:
  "divergent witnesses without a counterterm rule or certified zero").

MG5_aMC's `loop_qcd_qed_sm` needs
`[loop.preparation] compatibility = "gmu_particle_anti"` with either engine (its
`G-` loses the Goldstone flag of `G+`: "antiparticle metadata mismatch for G+:
goldstoneboson"); `loop.preparation.covariant_multiplets` and
`color.contraction = "symmetric-group-fft"` need `engine = "legacy"`. A dry run
(`generate --dry-run`) reports what the model preparation refuses; loop
admission is checked by the generation only.

Widths. v2 virtuals use real on-shell masses: loop propagators carry no width and
the model counterterms renormalise the real mass. A coloured particle that a
tree current propagates (the top in `g g > t t~`) must therefore have a zero
width, fixed in the card, otherwise generation and the dry run refuse the
request:

```toml
[loop.preparation]
fixed_parameters = {MB = "0", WT = "0"}
```

The legacy engine takes every propagator without width; a resonant coloured
particle needs the complex-mass scheme (not available yet). An evaluation refuses
unknown parameter names, the renormalisation-scale parameter (`mu_r2` sets it
per point) and a value other than the one a parameter was fixed to at
generation. Other known parameters are accepted even where the process does not
use them (for example a width only loop propagators would carry), as with the
legacy engine. The renormalisation-scale parameter is the external
parameter at UFO/SLHA code `LOOP 1`, or the only one in block `LOOP`.

Loading a v2 process rebuilds its per-point tree inputs from the artifact, which
takes about the process's generation time, until those inputs run natively; the
standalone SDK bundle needs `engine = "legacy"` until then. A v2 artifact holds
no absolute path, user or host name of the generating host, and a copy moved to
another directory loads with bitwise-identical results. Loading regenerates the
stored plan and requires it to be identical, so the copy needs the same
pyAmpliCol build (Python package and native extension) as the one that
generated it; relocation is tested on the generating host only, not yet on
another machine.

To evaluate numerical colour/spin requests, set
`evaluation.correlated_requests = "data/requests.json"`. The file maps output
labels to the existing `color_correlation` and `spin_vectors` fields described
in [correlated CLI requests](command-line-interface.md#correlated-requests).
It works for tree and virtual artifacts and cannot combine with `resolved = true`
or nonempty `color_flow_ids`. Tree correlated arithmetic follows its existing
API default (arbitrary precision); virtual arithmetic and optional stability are
still bound before parameter evaluation. Request JSON preserves exact decimal
components and rejects duplicate fields or colliding integer leg keys.

The model-parameter card is a flat JSON object:

```json
{
  "aS": 0.117,
  "MZ": 91.1876,
  "complex_parameter": [1.0, -0.25]
}
```

Values must be finite real numbers or `[real, imaginary]` pairs. Direct
parameter overrides are applied after the card and win atomically.

## Profiling card

Run cards retain `action = "benchmark"`; the preferred direct CLI spelling is
`pyamplicol profile`:

```toml
action = "benchmark"

[evaluation]
artifact = "artifacts/pp_zjj"
process = "d d~ > g z g"
momenta = "data/pp_zjj_momenta.json"

[benchmark]
target_runtime = 1.0
batch_size = 128
precision = 16
warmup_runs = 2
minimum_samples = 5
color_flow_ids = ["1"]
```

With both benchmark selector lists empty, LC profiling infers the stored
layout's deterministic hot workload. Explicit subsets and selected-axis lists
override that default; a complete summed-axis list is normalized to equivalent
omission. A valid shape outside the optimized layout emits at most one pre-loop
warning per loaded process.

See [Runtime and Selectors](runtime-and-selectors.md) for evaluation and profiling
semantics.

## Output and progress

```toml
[output]
format = "human"
color = "auto"
progress = "auto"
log_level = "info"
```

- The `pyamplicol` CLI always produces aligned colored terminal tables unless
  the invocation includes `--json`. The run-card `format` field remains part
  of the typed configuration used by the Python API and recorded provenance;
  it does not silently switch CLI stdout to JSON.
- `pyamplicol ... --json` reserves standard output for stable, uncolored,
  machine-readable data.
- `auto` progress uses a TTY progress bar interactively and rate-limited log
  messages otherwise.
- `Ctrl-C` during profiling retains completed timing blocks and marks the result
  partial.

## Resolve configuration from Python

```python
from pathlib import Path

from pyamplicol.config import resolve_config

resolution = resolve_config(
    {
        "schema_version": 1,
        "action": "generate",
        "model": {"source": "models/json/sm/sm.json"},
        "process": {"entries": [{"expression": "d d~ > z g"}]},
        "generation": {"output": "artifacts/builtin_ddbar_to_zg"},
    },
    base_dir=Path("pyamplicol-examples"),
    overrides=("generation.workers=2",),
)
print(resolution.effective.generation.output)
```

For a mapping, relative paths use `base_dir`; without it they use the current
working directory. Values supplied through `--set` or `overrides=` use TOML
syntax, so quote strings containing spaces.

## Further reading

- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [LC workloads and execution modes](lc-workloads-and-execution-modes.md)
- [Models and Processes](models-and-processes.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Exhaustive current schema](https://github.com/MadGraphTeam/pyamplicol/blob/main/examples/all_options.toml)
