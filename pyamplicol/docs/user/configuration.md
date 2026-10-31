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
accuracy = "lc"                 # lc, nlc, or full; unset: lc for tree
                                # requests, full for one-loop requests
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

The v2 engine's post-build validation compares the reopened artifact with
its reference executor in binary64 at `relative_tolerance` (Borns per
coupling order, virtual Laurent coefficients relative to their sector's
largest one). A validation point where the two binary64 values miss that
tolerance is checked again, alone, against the reference at 32 digits, in
both carriers: the artifact in double-double must agree to the same
`relative_tolerance`, its binary64 Borns to `relative_tolerance` and its
binary64 virtuals to the larger of `relative_tolerance` and their own
stability estimate (one exact rotation and the reduction-direction check),
an estimate that must exist and not exceed 1e-9. A binary64 virtual off by
more than the tolerance is thus accepted only where binary64 itself
estimates that it cannot resolve the tolerance (an ill-conditioned point),
and a defect of either carrier beyond that is refused. An accepted point is
reported as a warning of the `pyamplicol.generation` logger with its
deviations and estimates. This costs one double-double and one checked
binary64 evaluation of the artifact and one 32-digit reference evaluation
per re-checked point; the 32-digit reference dominates, about 9 to 11 times
the binary64 one (7.8 s against 0.7 s for d d~ > g g at NLC with five
colour outputs, 60 s against 6.9 s for u u~ > g g g at NLC with four). The
estimate cap is the larger of `relative_tolerance` and 1e-9: at high
multiplicity, where binary64 estimates can exceed 1e-9, validate with a
`relative_tolerance` that binary64 resolves.

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

Partial numerical returns use the same option as the Python runtime:

```toml
[evaluation]
artifact = "artifacts/virtual"
momenta = "data/points.json"
arithmetic = "double"
mu_r2 = "1000000"
on_failure = "return"         # default: "raise"
```

With `on_failure = "return"`, numerical failures become unavailable whole-point
rows with diagnostics (`null` in JSON); valid zero values remain zero. Invalid
inputs and unsupported features still raise errors for the whole call; on v2
runtimes this includes a non-positive `mu_r2` and a scale at which a
counterterm coupling is not finite in the lane's arithmetic (for example
`mu_r2 = "1e-400"` in binary64), which raise a `ValueError` for the whole batch
(returning such points one by one is an open item). This setting applies only
to virtual `evaluate`, including resolved output, and is not silently used by
profiling or generation.

The precision-rescue policy `[evaluation.stability]` (`higher` carriers, the
two tolerances, `verify_successful`, and the CLI options `--stability-higher`,
`--stability-relative-tolerance`, `--stability-absolute-tolerance` and
`--verify-successful`) was removed with the legacy loop engine, the only one
that served it: any value other than the defaults is refused ("evaluation.stability
(the precision-rescue policy ...) was removed ..."), and a Python
`VirtualStabilityPolicy` is refused by v2 runtimes at load. v2 evaluations
return stability estimates instead and leave escalating the arithmetic to the
caller (below).

### The v2 loop engine

One-loop requests (`process.loop_coupling_orders`) generate virtual processes
with the shared-recursion one-loop engine v2 (no loop diagram is listed),
the only one-loop engine, and publish them as plan v4 processes (runtime
capability `rusticol.loop-runtime.v4`). The per-diagram legacy engine was
removed: a card, override or command line naming `loop.engine` (or
`--loop-engine`) is refused ("configuration field 'loop.engine' was removed
..."), and an artifact of the legacy engine (`rusticol.loop-runtime.v3`) is
refused at load and inspection with the reason; regenerate it. Its UV and R2
counterterms come from the same tree recursion as the Born, over the model's
counterterm vertices (no counterterm diagram is listed either): every
counterterm vertex of the model, two-point ones included, is an ordinary tree
vertex inserted exactly once per contribution, within the requested coupling
orders, and is not matched to the loops. A v2 evaluation can skip the poles
and return stability estimates:

```toml
[evaluation]
artifact = "artifacts/virtual-v2"
momenta = "data/points.json"
mu_r2 = "1000000"
poles = "none"                 # finite part only; poles are absent (null)

[evaluation.stability_checks]
rotation_count = 1             # that many default (generic-axis) rotations
# rotations = ["x:3/5:4/5"]    # alternative: exact "<x|y|z>:<cos>:<sin>" or "<ax>,<ay>,<az>:<radians>"
reduction_direction = true     # re-reduce every loop numerator, reversed frame
rescaled = true                # ... at rescaled inputs: master integrals evaluated again
```

This is the recommended configuration (Python `VirtualStabilityChecks(1, True,
rescaled=True)`). Avoid a rotation about the beam axis (`"z:..."` for beams
along z): it leaves both incoming momenta and every energy and p_z bitwise
unchanged, and has under-reported a binary64 double pole 23-fold
(`u u~ > u u~ e+ e-`); only the rescaled check sees the master integrals' own
rounding (`g g > b b~ h` with a massive b: up to 96-fold below the error
without it).

Each check re-evaluates or re-reduces every point; the result then carries one
`stability` estimate per point (relative accuracy of the whole virtual, of the
finite part and, with poles, of each pole, plus each check's own budget).
Without checks no estimate is returned. Escalating to `arithmetic = "arbitrary"`
is left to the caller (the `[evaluation.stability]` rescue policy was removed,
above). `poles` is a pole selection (`PoleSelection` in Python):
`"all"` (the default), `"none"`, `"uv"` or `"ir"`; the finite part is the same
under every selection, and `"none"` saves the pole work. The CLI spells these
`--poles {all,none,uv,ir}`,
`--stability-rotation SPEC` (repeatable), `--stability-rotations COUNT`,
`--stability-reduction-direction` and `--stability-rescaled`.

**Momenta and arithmetic.** Every arithmetic (`double`, `double-double`,
`arbitrary`) takes each external leg's invariant p² in the loop integrals to be
the leg's squared mass at the evaluation's model parameters (zero, or a mass
parameter such as `MT`), exactly, rather than the p² of the supplied momenta;
everything else (trees, wavefunctions, loop-momentum shifts, invariants of
several legs) uses the momenta as given, without projection or tolerance. This
is always on and changes no input: it is what an external state means. Momenta rounded to binary64 sit off
their mass shells by about 1e-16 relative, so a point copied from a binary64
evaluation gives the same finite part and poles in `double-double` and
`arbitrary` arithmetic, to the accuracy of the binary64 value (its stability
estimate); `double-double` and `arbitrary` agree with each other to
double-double accuracy. To go beyond binary64 accuracy, supply momenta that are
on shell and conserve momentum to the digits wanted (exact decimals or
rationals): a point off its shells by a relative δ moves the result at order δ.

**Light masses in binary64.** The fast binary64 reduction compares squared
masses and invariants with an absolute constant, 1e-9 in the model's units: it
would treat a smaller one as zero and an invariant that close to a squared mass
as on that shell. A `double` evaluation never applies these: a loop whose
squared masses or invariants at a point fall in that window (a keV-scale
internal mass in GeV, the electron mass in TeV) is reduced by the exact generic
binary64 engine instead, so these comparisons no longer depend on the units;
only the cost of those loops changes.
The exact engine uses Symbolica in the native code, under pyAmpliCol's package
key, so a v2 binary64 evaluation with such a loop needs the key, which other v2
binary64 evaluations do not (see
[Symbolica and licensing](symbolica-and-licensing.md)).
The artifact's generation filters list the masses this light at the generation
parameters (`binary64_exact_engine`: the threshold and each squared mass; v2
artifacts); a parameter override is checked again at evaluation. Binary64
itself limits light loops far below the other scales: a box with three or four
light lines can lose several digits, and one beside a massive leg can fail with
a non-finite master integral (the point then fails rather than returning the
massless value). Such a loop is as accurate as binary64's master integrals at
its arguments, and the rotation and reduction-direction checks reuse those
masters, so they do not see that error (`u u~ > t t~` with a 1e-6 GeV top:
2.5e-6 off with estimates of 4e-15); only the rescaled reduction check, which
recomputes the masters (`rescaled = true`; Python
`VirtualStabilityChecks(..., reduction_direction=True, rescaled=True)`), does. Use that check for light loops, and `double-double`
where binary64 fails.

**Units and the beam axis in binary64.** One absolute constant of the fast
binary64 engine remains until an upstream fix (NinjaRS): its light-like spinors
take a momentum within 1e-9, in the model's units, of the z axis as lying on
it, and form the others with a square root that cancels near the -z axis. In
GeV-like units this is harmless for beams along z. With energies of order 1e6
or more in the model's units (keV or eV units, or MeV with multi-TeV beams) the
rounding of a beam along z exceeds that window, and a `double` virtual can be
silently off by several per cent with the checks off (measured 3-22 % for
`u u~ > d d~ g` and `g g > t t~` with every momentum and dimensionful
parameter scaled from GeV by factors from 1e4 to 1e9);
the stability checks see most such points but not all, and can flag correct
ones. For `double`, use units in which the energies stay below about 1e6 (GeV
for collider energies), or `double-double`. In any units, a final-state leg
within about 1e-3 rad of the z axis can lose up to about 1e-5 of relative
accuracy silently; use a rotation or the reduction-direction check for such
kinematics.

**Native libraries and glibc.** On Linux, the native libraries built from a
checkout with `tools/developer/build_loop_native.py` (`_rusticol.so`,
`librusticol_loop_sdk.so`) need glibc 2.38 or newer; the manylinux 2.28 wheels
need glibc 2.28. On an older glibc, importing the extension fails at once with
`NativePlatformError` naming the missing versions and the running glibc. C,
C++ or Fortran programs fail to link against `librusticol_loop_sdk.so` with a
toolchain on an older glibc
(``undefined reference to `__isoc23_strtol@GLIBC_2.38'``), and a program built
on a newer glibc is stopped by the dynamic loader on an older one
(``version `GLIBC_2.38' not found``); see
[Native libraries and glibc](installation.md#native-libraries-and-glibc).

#### Leading-colour helicity and colour-flow selection

At leading colour (`[color] accuracy = "lc"`) a v2 virtual offers the tree
engine's selectors, with the tree engine's identifiers and selector
conventions (per-flow values at a cross order of several Born orders follow
the loop's convention, below).
At evaluation, `Runtime.evaluate(..., helicities=[...], color_flows=[...])`
returns the sum of the selected Born helicity rows and colour flows (Born
included): helicities by tree ID (`h:-1,+1,...`, values in leg order) or the
runtime's `h<index>`, flows by tree ID (`flow:2,4,3,5,1`) or 1-based ordinal.
`runtime.physics.born_anchored_color_components` lists the flows the
artifact computes, numbered as the tree runtime with the same `[process]`
selectors numbers its flows: in the order of the tree engine's
leading-colour plan (after `reference_color_order` and `max_color_sectors`;
`reference_color_order` alone selects nothing and only orders), and under
a generation-time colour selection only the selected flows. The physics
helicity entries keep the runtime's `h<index>` IDs (the helicity axis of
`evaluate_resolved` at every colour accuracy); their tree ID is `h:`
followed by their `values`. A complete set of selections sums to the
inclusive value; a (helicity, flow) selection equals its `evaluate_resolved`
entry. `evaluate_resolved` takes the same selectors and returns the selected
helicity rows and the selected flows' Born basis elements, each entry as
without selectors. At generation, `process.selected_source_helicities` (leg label to
helicity) and `process.selected_color_sector_ids` (sectors of the tree
engine's leading-colour plan, after `reference_color_order` and
`max_color_sectors`) prune the recursion to the selection: fewer Born rows,
skeletons and tree currents. The artifact's `evaluate()` then returns the
selection, and its physics reports `helicity_coverage` and `color_coverage`
`"selected"`. A sector of a folded trace holds two flows, the sector's and
its reflection, as in the tree engine. As in the tree engine, declared
colour or spin correlations with any of these selectors are refused at
generation (`selected-correlators`): generate the correlations in an
inclusive artifact. An evaluation-time selection costs at most one inclusive
evaluation: it runs only the loop families its flows read, and only the
helicity rows its helicities reach in the loop dressing and the tree
currents. A generation-time selection is somewhat cheaper still (one
helicity and one flow, binary64: u u~ > g g g about 0.7 ms per point at
evaluation and 0.6 ms at generation, against 2.0 ms inclusive at leading
colour; u u~ > g g g g 4.0 ms and 2.9 ms, against 30 ms). The first call of
each distinct evaluation-time selection also forms and compiles its column
set once (u u~ > g g g g: about 1 s and 66 MB for a new (helicity, flow)
pair). The runtime keeps the 8 most recently used selections for later
calls; the environment variable `PYAMPLICOL_LOOP_SELECTION_CACHE` sets
another count (0 keeps none), and `Runtime.clear()` releases them. Sampling
many distinct selections one event at a time therefore pays that setup per
selection; the kept sets bound the memory, apart from about 1.5 MB that each
formation leaves in the native heap (as every rebuild of a native executor
does today, e.g. `Runtime.clear()` followed by an evaluation). A
generation-time selection has no such setup.

```toml
[process]
selected_color_sector_ids = [2]
selected_source_helicities = {"1" = -1, "5" = 1}
[color]
accuracy = "lc"
```

The loop's leading colour is the leading N_c power of the squared matrix
element. A tree flow below it, such as a flow of u u~ > d d~ g that joins
each quark to its own antiquark, has no leading-colour Born or virtual. Its
selection is zero at evaluation. At generation it contributes exact zeros
next to a leading flow (`max_color_sectors = 3` on u u~ > d d~ g keeps two
such flows and a leading one), and a selection without a leading flow is
refused (`subleading-colour-flows`). With several Born orders (QCD and
electroweak Born amplitudes, e.g. u u~ > d d~ with photon exchange) a flow
can interfere at the leading power with another flow between the two orders
(there QCD^2 QED^2): its evaluation-time selection keeps that term, and a
generation-time selection that would drop it is refused
(`interfering-colour-flows`); select the interfering flows together, or
select at evaluation. At such a cross order a flow's value is the selected
flow interfering with the whole Born of the other order (bra the flow, ket
every flow), not the tree engine's per-flow `evaluate_orders` value: it can
be nonzero where the inclusive leading-colour value vanishes (u u~ > d d~
at QCD^2 QED^2: the two flows give +-8.5e-3 and sum to 0), and complete
sets of flows sum to the inclusive value. Identical-flavour quark lines (u u~ >
u u~ g) take every selection. A member of an exact flavour group
(`process.flavor_blind`) lists no flows and takes no evaluation-time
selection (`CompatibilityError`): select on its representative process,
which evaluates the member. Next-to-leading and full colour expose no
selection: they evaluate every helicity and colour, and their API and plans
are unchanged.

#### External photons: alpha(0) per leg

With electroweak (QED) corrections, an on-shell external photon either couples
with alpha(0) (a *tagged* photon, MG5_aMC's `!a!`: an isolated photon) or keeps
the loops' coupling, alpha_Gmu on a G_mu model (an *untagged* photon: a photon
from a PDF or in a jet). `[loop] external_photons` chooses per leg:

```toml
[loop]
external_photons = "final"     # default: final-state photons tagged, initial ones untagged
# external_photons = "all"     # every photon tagged (MG5_aMC with every photon !a!, e.g. UPC/EPA)
# external_photons = "none"    # no photon tagged
# external_photons = [1, 3]    # exactly these 1-based legs (initial legs allowed)
```

The override is `--set loop.external_photons=none` (or `=[1,3]`), and Python
takes the same values (`LoopConfig(external_photons=...)`, the v2
`SliceRequest(external_photons=...)`). For n tagged photons the engine applies
MG5_aMC's formula (the G_mu branch of `export_fks.write_rescale_a0gmu_file`):
the Born and the virtual are multiplied by (alpha(0)/alpha_Gmu)^n, and the
virtual of the electromagnetic correction gains -2 n DGmua0 Born, with DGmua0 =
delta Z_e^Gmu - delta Z_e^alpha(0) the model's `eCoup_DGmua0_UV_EW`; their IR
poles carry no gamma_gamma. This needs a G_mu loop model with an adjustable
1/alpha(0): MG5_aMC's `loop_qcd_qed_sm_Gmu` under `restriction = "a0"`
(`aEW0M1`, a runtime parameter of the stored artifact). Untagged photons need no
model input and keep their -gamma_gamma/eps pole, which cancels against the
photon PDF's or the photon-to-fermion collinear counterterm. MadLoop standalone
converts no photon: its raw values are those of `external_photons = "none"`.

The default `"final"` follows the physics of the usual cases: an initial-state
photon is a PDF photon (its coupling and pole belong to the PDF's scheme), so
converting it would shift the result by about 3.3 % per photon
(alpha(0)/alpha_Gmu ~ 0.967) and give the wrong IR pole; a final-state photon
is an isolated photon, for which alpha(0) is the physical coupling. Choose
`"all"` or a list for photons from the photon flux of a charged beam (UPC/EPA).
Only requests with the electromagnetic correction convert photons (as in
MG5_aMC: QCD corrections leave every photon at the model's coupling).

The conversion counterterm and 1/alpha(0) are found by MG5_aMC's conventions
(`eCoup_DGmua0_UV_EW` and the unique adjustable SLHA `SMINPUTS 1` parameter); a
model that spells them otherwise names them:

```toml
[loop]
alpha0_conversion_parameter = "dZe_Gmu_a0"   # default "eCoup_DGmua0_UV_EW"
alpha0_inverse_parameter = "ainv0"            # default: the SMINPUTS 1 parameter
```

A named 1/alpha(0) whose inverse is the loops' alpha declares a model whose
loops already couple with alpha(0): tagged photons then take neither the
rescaling nor the conversion term (and no conversion counterterm is needed).
Both names are recorded in the artifact's request only when set.

Refusals (listed, typed): tagged photons on a model that cannot take the
convention, `external-photon-scheme`; also any on-shell external photon of an
electroweak request on a model with several photon candidates (massless,
neutral, colourless vectors: which one is the photon is ambiguous). MG5_aMC's `loop_qcd_qed_sm` declares no
conversion (MG5_aMC refuses tagged photons there too): use `external_photons =
"none"` or the G_mu model with `restriction = "a0"`; on the G_mu model's default
card 1/alpha(0) is fixed (zero): use `restriction = "a0"` or `"none"`. A listed
leg that is no on-shell external photon of the process: `external-photon-legs`.
The artifact records the tagged and untagged legs among the electroweak
conventions (`inspect`: `tagged_photon_legs`, `untagged_photon_legs`).

| MG5_aMC | pyAmpliCol |
| --- | --- |
| `generate a a > e+ e- [QED]` (no tags) | `external_photons = "none"` (the default leaves both initial photons untagged too) |
| `generate !a! !a! > e+ e- [QED]` | `external_photons = "all"` |
| `generate u u~ > !a! g [QCD QED]` | the default `"final"` |
| `generate a u > !a! u [QCD QED]` | the default `"final"` (or `[3]`) |

#### Ultraviolet and infrared poles

With `poles = "uv"` or `"ir"` every pole of the Born-virtual interference is
split by origin, `P/eps = P_UV/eps_UV + P_IR/eps_IR`; the double pole is
infrared. The finite part is always that of the renormalised virtual; the
returned poles are:

- `"uv"`: the ultraviolet poles of the bare (unrenormalised) one-loop
  amplitude interfered with the Born, i.e. the reduction's UV poles (the
  double pole is zero). The counterterms' UV poles are returned separately
  (below), so that their cancellation can be checked.
- `"ir"`: the infrared poles of the renormalised virtual, which Catani's
  operator predicts. Where the ultraviolet renormalisation is complete they
  equal the poles of `"all"`. They are the poles of the computed loops: a
  QCD request, like MadLoop's `[virt=QCD]`, has no loop with a W, Z or H
  line, so where the Born trees join quark lines in different ways through
  such lines (`u d > u d h` at `QCD=0`: a Z or a W between the lines; `u b >
  d t g`: the gluon on either line) the gluon exchanges across the lines are
  left out and the poles differ from the full Catani operator's. Such
  sectors carry no Catani prediction (`LoopSlice.unpredicted_sectors()`
  states why for each).

Both selections cost the same and both also return each origin's parts by
component, `VirtualEvaluation.pole_parts` (per point, `uv` and `ir`, each
component a Laurent triple per sector; `pole_part(origin, component)` reads one):
`virtual` (renormalised), `bare` (the bare loop), `r2`, `uv_local` (the model's
coupling and mass counterterms), `uv_external_fields` (the on-shell field
factors) and `uv` (`uv_local + uv_external_fields`). A model states its
external-field factors with one eps (a massless leg's vanishes), so their
ultraviolet part is fixed by the ultraviolet finiteness of the renormalised
virtual: the bare and counterterm UV poles then cancel by construction, and
the field factors' UV part is the quantity to compare with a field
renormalisation of your own (for loop_sm in Feynman gauge it equals the
textbook constants of the external legs). An incomplete ultraviolet
renormalisation shows in the `"ir"` (and `"all"`) poles, which then miss
Catani's operator. A model that writes its counterterms with pole regulators
(MadLoop's electroweak UFOs `loop_qcd_qed_sm` and `loop_qcd_qed_sm_Gmu`:
`epsUV` and `epsIR` in block `TECHNICAL`, codes 1 and 2) states the origin
itself: as long as both stay adjustable (not in `fixed_parameters`; the
complex-mass scheme keeps them), every counterterm component's UV part is its
value at `epsIR = 0` (MadLoop's reading of that card setting) and its IR part
the rest, so for these models the bare and counterterm UV poles cancel as a
check, not by construction (UV finiteness then also needs consistent inputs,
e.g. the top Yukawa `ymt` at the top mass `MT`). An electroweak process
records which definition its `"uv"` and `"ir"` parts use in its conventions
(`Runtime.inspect()["conventions"]["pole_split"]`: `"regulators"`, or
`"remainder"` with a regulator fixed); an electroweak artifact without that
entry predates this split and uses the remainder: regenerate it. This costs
one more pass of the tree program per point in `"uv"` and `"ir"`, also for
correlated and resolved outputs (g g > t t~ with QCD and QED corrections:
about 3.1 against 2.1 ms per point under `"all"`, binary64, one thread). In
Python:

```python
result = runtime.evaluate(points, mu_r2=scales, poles="uv")
bare_uv = result.values                           # [point][sector] (finite, 1/eps_UV, 0)
counterterms_uv = result.pole_part("uv", "uv")    # UV_local + field factors' UV part
local_uv = result.pole_part("uv", "uv_local")     # the model's coupling and mass CTs
```

The strong coupling of Catani's QCD operator is the model parameter named by
`[loop] alpha_s_parameter` (an adjustable external parameter), else the unique
adjustable SLHA `SMINPUTS 3` parameter; a model with neither (or with it fixed)
generates the same virtual without a QCD pole prediction, whatever the
correction set. Masses are the model's runtime values, a derived (internal)
mass included.

The selection is a runtime option: the same artifact serves all four (it stores
the companion plan of its certified-scaleless loop families, which only `"uv"`
and `"ir"` evaluate; they cost about 1 % (u u~ > e- e+ g g), 7 % (g g > g g g)
and 13–18 % (g g > g g g g) more than `"all"`).
An artifact written before the split is refused at load with a request to
regenerate it.

Stability estimates follow the selection. With `"uv"`, request at least one
rotation check (`rotations` or `rotation_count`) to bracket the selected single
pole: it is the bare loop's UV residue, which does not depend on the reduction
frame, so `reduction_direction` alone (also rescaled) sees only its rounding
and not the error of the loop numerators (at a `u u~ > d d~ g` point in
double precision it underestimated that error about 8 times; a rotation
bracketed it). With `"all"`, `"ir"` and `"none"` either check works. The
`"uv"` double pole is exactly zero, and so is its estimate.

Correlated and resolved outputs
take every selection too; their `pole_parts` follow their own axes (resolved:
per sector, helicity and colour component).

A one-loop request without `color.accuracy` is full colour (MadLoop's
accuracy); LC and NLC are opt-in. MG5_aMC's loop_sm ships with pyAmpliCol
(`model.source = "loop_sm"`, with its `[loop.model]` declaration): see
[One-loop QCD on loop_sm](loop-sm-and-madloop.md).

Current v2 scope: model-supplied R2; full, NLC or LC colour (LC/NLC with the
model's formal-N_c colour declaration, shipped for loop_sm and its
restrictions; formal-N_c projections for any number of
quark lines, identical flavours such as `u u~ > u u~ g` included); double,
double-double and arbitrary
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
correction the model does not expand perturbatively is refused. Coupling-order
names are case-insensitive: `process.loop_coupling_orders`,
`loop.model.loop_coupling_orders` and `process.max_coupling_orders` resolve
against the model's declared names (`qcd` names a declared `QCD`, `qe` a
declared `Qe`), and loop results carry the declared spelling; a name the model
does not declare is refused, and so is a model declaring two names that differ
only in case. Tree-level per-coupling-order partials still key orders by the
tree engine's upper-case spelling (`QE` for a declared `Qe`); every shipped
model declares upper-case names, where both spellings agree. The strong
correction is recognised structurally, as the coupling order of the gluon's
vertices (the massless adjoint vector's), whatever the model calls it; any
other correction, or a perturbative colour-singlet species, makes the request
electroweak. Electroweak requests take 't Hooft-Feynman gauge throughout
(Born, trees attached to the loops, loop lines and counterterms; a model
whose source declares no such gauge is refused) and need zero widths in
their trees (see Widths below). Catani's QCD pole prediction needs the
strong coupling parameter described above (`ir_poles` is absent otherwise;
`poles = "ir"` still selects the computed virtual's poles).
What the engine does not admit fails at generation with a typed error that
names the rule, for example:

- leading and next-to-leading colour over tree currents that are not exact
  in N_c (currents mixing U(1)-subtraction grades): "colour-accuracy-nc-exact:
  ... LC needs N_c-exact tree currents ...". Leading and next-to-leading
  colour split such currents by grade, so identical-flavour quark lines (from
  `u u~ > u u~ g` up: `u u~ > u u~ d d~`, `g g > u u~ u u~`, ...) are exact
  and generate, as distinct flavours do; the rule guards any request whose
  currents would still mix grades. The check runs right after the trees are
  built, and every export repeats it, so no artifact carries a projection that
  is not exact;
- a `particle_veto` that removes a species running in the loops of a
  requested correction, on a model some of whose counterterm terms of that
  correction declare no loop content (`loop_particles = []`):
  "veto-unlabelled-counterterms: ... particle_veto removes the species
  [23, 24, 25, 250, 251], perturbative in the corrections ['QED'], and the
  model's counterterm vertices V_UV... write terms of those corrections
  without loop content ...", in the model preparation, before any tree is
  built, so a dry run reports it too (photon-only QED corrections on
  MG5_aMC's `loop_qcd_qed_sm` and its G_mu variant; see the particle veto
  below). Request the complete corrections;
- a vertex joining a coloured and a colour-singlet fermion that the request
  can use (a leptoquark `u e lq` vertex: one fermion line changing colour
  representation): "fermion-line-representation: ... the vertex V (u~ e- lq)
  joins a coloured and a colour-singlet fermion ...", before any tree is
  built. Veto a particle of the vertex (`particle_veto`) or bound its
  coupling order so that the request cannot use it;
- a loop vertex that the request's restricted tree model does not hold while
  its couplings are not shown to vanish at the model's restriction:
  "loop-vertex-outside-tree-model: ... the loop vertex V (...) at the
  coupling orders {...} has no term in the request's tree model ...". A
  vertex whose couplings vanish identically at the restriction (decided from
  their values: e.g. the W vertices of the off-diagonal CKM elements at the
  diagonal-CKM default restriction of `loop_qcd_qed_sm_Gmu`) is simply no
  loop vertex, so `g t > t u u~` with QCD and QED corrections passes the loop
  catalogue (it is then refused at native export with "plane-mismatch" at
  its `UV_GTTLEW` counterterm, as `g g > t t~` is, see the stored programs
  below). Choose a restriction under which the vertex's couplings vanish
  exactly or one that keeps it, or veto or bound it out of the loops;
- fermions outside the loop sign rule: vertices joining two or more fermion
  lines (four-fermion contact vertices: "a loop vertex joins more than one
  fermion line"), fermions in colour representations other than singlets and
  (anti)triplets ("a fermion of colour representation 8 has no sign rule"), and
  Majorana fermions (refused when the model is loaded). Any number of
  colour-singlet (lepton) fermion lines, mixed with any number of quark lines
  and running through the loop or not, is admitted: `e- e+ > m- m+` and
  `e- e+ > e- e+` with QED corrections, `u u~ > e- e+ m- m+` with QCD
  corrections.

A process whose programs cannot be stored natively is refused with the
feature's key (see the stored programs below), for example a program whose
binary64 SymJIT application disagrees with its exact evaluator at the
generation's check ("plane-mismatch", naming the output: a model coupling by
its UFO coupling and vertex, a counterterm coupling with its epsilon power, or
a kernel). Tagged on-shell external photons with electroweak corrections
couple with alpha(0) and are stored natively, with 1/alpha(0) (`aEW0M1` of
MG5_aMC's G_mu loop models under `restrict_a0`) a runtime parameter
([external photons](#external-photons-alpha0-per-leg)). The check compares
at complex points off the real axis and at the request's parameter values,
so it depends neither on the parameters' names nor on their order; it
refuses a branch cut taken on different sides at those values, or a branch
cut of a program's inputs that no real point certifies.

A `particle_veto` drops every counterterm whose declared loop particles include
a vetoed particle. MG5_aMC's `loop_qcd_qed_sm` declares some R2 couplings for
the photon and Z loops together (e.g. the `u u~ g` R2, `R2_GUU2Cp`, with loop
particles `u` and `Z`), and its electroweak UV counterterms carry nominal loop
particles (none on the vertices, one representative on the two-point ones),
so photon-only QED corrections (the Z, W, Higgs and Goldstones vetoed) would
drop the photon's part of the former and keep the W and Z loops of the latter:
with lepton lines (`e- e+ > m- m+`) the single pole misses the QED Catani
operator's, and the finite part is wrong. Such a veto is therefore refused at
generation ("veto-unlabelled-counterterms", above) on every model with a
counterterm term of a requested correction that declares no loop content.
Photon-only corrections are not expressible on this model; full electroweak
corrections are.

The model's UV and R2 counterterms are taken as declared (`uv_complete`,
`model_r2_complete`): the engine does not check them against the loops.

MG5_aMC's `loop_qcd_qed_sm` and `loop_qcd_qed_sm_Gmu` (shipped, see
[One-loop QCD+QED on loop_qcd_qed_sm](electroweak-and-madloop.md)) need no
compatibility setting: their `G-` loses the Goldstone flag of `G+`, which
model preparation repairs by structure and records
(`compatibility = "gmu_particle_anti"` is still accepted);
`loop.preparation.covariant_multiplets` is not supported by v2 (refused,
`covariant-multiplets`). `color.contraction` selects the seed contraction of NLC and full-colour virtuals: `auto` (default) contracts
the certified symmetric-group orbit pairs of the external gluons with the FFT
and the rest directly, `direct` contracts everything directly, and
`symmetric-group-fft` requires a certified pair (a process with fewer than two
external gluons has none and is refused at generation with the listed
`forced-seed-fft-unserved`, as are declared correlations with a forced FFT,
since observable columns are contracted directly). The FFT and the direct
contraction are equal up to rounding in every arithmetic; each evaluation
reports the algorithm it ran. The orbits are those of the loop's own seed
tables (the external gluons' permutations, certified on the exact colour
weights): loop requests keep the default `color.fft_basis = "adjoint"`, and
`"trace"` is refused unless the contraction is `direct`. A dry run
(`generate --dry-run`) reports what the configuration, the model preparation
and the Born need (the veto refusal "veto-unlabelled-counterterms", the
external-photon convention and the widths of the Born currents of an
electroweak request included), each with its listed key; the rest of loop
admission is checked by the generation only, and the dry run lists every key
with its remedy (`estimated_coverage.loop_configuration_refusals`,
`loop_admission_refusals`, `loop_internal_refusals`) and what the native
export refuses (`native_export_refusals`).

Particle veto. `[generation] particle_veto = ["t"]` (CLI `--particle-veto t`,
repeatable; Python `GenerationConfig(particle_veto=("t",))`) removes the named
particles, with their antiparticles, from every tree vertex of the v2
generation: the Born, the trees attached to the loops, the loop vertices and
propagators, and the counterterm program; every counterterm term whose loop
content (the UFO `loop_particles`) holds a vetoed particle is dropped, and so is
its external-field renormalisation. With loop_sm, `particle_veto = ["t"]` gives
massless five-flavour QCD (no top loop, no top decoupling term), the convention
of NJet. A vetoed vector boson takes its Faddeev-Popov ghosts with it (the ghosts
of its colour, charge and mass: vetoing `w+` also vetoes `ghWp` and `ghWm`),
whose loops without the boson would break the gauge cancellations. A vetoed
particle cannot be external. The veto is stored with the artifact's generation
request and in its generation filters; it needs a one-loop request.

Anomalous vetoes. A veto can keep model counterterms from cancelling that
cancel with every flavour admitted: vetoing `d s b` (or `t`) in a request with
a Z and gluons leaves the axial Levi-Civita R2 of the Z with gluons
(`R2_GGZ`, `R2_GGGVa`), which cancels between the members of complete quark
doublets, uncancelled. Such a request is refused by default
(`AnomalousVetoError`, naming the counterterm and the override). `[generation]
allow_anomalous_veto = true` (CLI `--allow-anomalous-veto`; Python
`SliceRequest(allow_anomalous_veto=True)`) lowers those counterterms as tree
vertices and records the choice: the artifact's request record and its
generation filters (`allow_anomalous_veto`, `counterterms_anomalous_veto`:
rule -> cells). It needs a `particle_veto`.

Widths. By default (`mass_scheme = "real-on-shell"`) v2 virtuals use real
on-shell masses: loop propagators carry no width and the model counterterms
renormalise the real mass. A coloured particle that a tree current propagates
(the top in `g g > t t~`) must therefore have a zero width, fixed in the card,
otherwise generation and the dry run refuse the request (`coloured-width`; an
electroweak request refuses any width in its trees, `electroweak-width`):

```toml
[loop.preparation]
fixed_parameters = {MB = "0", WT = "0"}
```

A resonant particle needs the complex-mass scheme, MG5_aMC's `set
complex_mass_scheme True`:

```toml
[loop.model.scheme]
mass_scheme = "complex-mass"
```

Stated alone, `mass_scheme` selects the scheme over the model's own (source or
shipped) loop declaration. Generation then runs on MG5_aMC's transform of the
model: every particle with a nonzero width at the card propagates with its
complex pole mass `sqrt(M^2 - i M W)`, which the parameters, counterterms and
couplings read too (`CMSParam = 1`; internal widths take their real part; the
Yukawa couplings follow the masses). Masses and the nonzero widths stay real
inputs of the evaluation; a width that is zero at the card (fixed, or zero on
the restriction card) gives its particle no pole mass and stays zero. The
scheme refuses, with listed keys: an external
particle with a width (`cms-external-width`: external states are stable), a
width-carrying particle whose mass is not an external parameter
(`cms-internal-mass`, e.g. loop_sm's W), a fixed Yukawa input
(`cms-yukawa-input`), a value of `mass_scheme` other than `"real-on-shell"` and
`"complex-mass"` (`mass-scheme`), and a complex fixed parameter in either
scheme (`complex-fixed-parameter`). Stored artifacts take the complex pole
masses in their native point inputs (the loop propagators read M^2 - i M W; the
binary64 fast-engine threshold reads |re| + |im| of it), so they keep native
point inputs and relocate as any other artifact. A width changed at run time
may be zero (the real-mass limit), which puts counterterm arguments on their
branch cuts: an argument that is real when the zeroed widths vanish reads its
real part there and every zero imaginary part of a cut argument is made
positive, so every precision takes the principal value (checked against
MadLoop CMS on `e- e+ > mu- mu+` with all, some and none of the widths zero;
where MadLoop's double and quadruple precision disagree, pyAmpliCol's
precisions agree with each other and with one of them). In double-double a width must
stay at least 1e-150 of its mass, or be zero (`complex-mass-master` otherwise,
a model-parameter error naming the plan mass). An evaluation
(the Python API and the native SDKs alike) refuses an empty parameter name, a
value that is not a finite decimal number (a number, or ASCII decimal text:
surrounding whitespace and underscores are ignored, fractions and non-ASCII
digits are refused), unknown parameter names, the renormalisation-scale
parameter (`mu_r2` sets it per point) and a value other than the one a parameter
was fixed to at generation (the card's `fixed_parameters` and those the model
preparation fixes, e.g. a colour declaration's external scheme constants, which
the refusal names as such: they are not run-time parameters); a parameter
update with one refused entry changes nothing, and every client checks the
entries by name, so a mapping with several refused entries gets the same
message everywhere. Other known parameters are accepted even where
the process does not use them (for example a width only loop propagators would
carry). The renormalisation-scale parameter is the
external parameter at UFO/SLHA code `LOOP 1`, or the only one in block `LOOP`.
Generation records which model-derived values depend on it: for the usual NLO
models only the counterterms do, and a batch of points at several scales is
evaluated at once; for a model whose Born couplings or masses run with the scale
(a running-coupling or MSbar-mass UFO) an ordinary evaluation (`evaluate`, every
SDK) is evaluated per scale, so every point equals its evaluation at that scale
alone. Colour- and spin-correlated and helicity-resolved evaluations of such a
model take one scale per call: a batch of several scales is refused with a
`ValueError` naming the scale-dependent couplings, constants or masses (call
them once per scale).

A v2 artifact stores its whole program: the loop plan and its per-point tree,
Born, counterterm and parameter programs (`point_inputs = "native-v2"`), the
latter as SymJIT IR recompiled for the loading CPU and as saved Symbolica states
for double-double and arbitrary precision, with the companion of the
certified-scaleless families that the `"uv"` and `"ir"` pole selections
evaluate and the colour- and spin-correlated and helicity-resolved columns. No
call of a loaded process regenerates anything: every pole selection, stability
check, arithmetic, model-parameter change, per-order partial, exact flavour
member, correlated, resolved and catalogue call reads the stored programs, and a
load takes a small fraction of the generation time. A process whose programs
cannot be stored natively is refused at generation with the reason, never stored
for a regeneration at load. The SymJIT IR of the kernels is compiled at -O2,
whatever `evaluator.jit.optimization_level` says (a program whose -O2 code fails
the generation's plane check is kept at level 0 instead: a SymJIT SIMD defect,
see the portability guide), and that of the parameter program at level 0. The
artifact holds no absolute path, user or host name of the generating host; a
copy moved to another directory, user or machine loads without regeneration and,
with the same pyAmpliCol build (Python package and native extension), gives
bitwise-identical binary64 results on CPUs of the same SymJIT class (x86-64 with
AVX and FMA3; AArch64 is expected to behave the same but is untested); other CPU
levels (checked under emulation) run the saved states and agree to rounding
([portability](artifacts-and-portability.md#cpu-requirements-of-symjit-kernels)).
Binary64 evaluation needs no Symbolica licence, at any model parameters, pole
selection and scale, except for loops routed to the exact engine at light
masses (above: they use pyAmpliCol's own Symbolica key, which expires on
2028-01-01, after which they need a personal licence); double-double and
arbitrary precision use pyAmpliCol's own Symbolica key as well. The native SDKs
(C, C++, Fortran, Rust) open a v2 artifact through `rusticol_virtual_open` and
evaluate binary64 virtuals, bitwise equal to the Python API, without a licence
(with the same exception):
`rusticol_virtual_set_poles(handle, poles)` takes the same pole selections as
the Python API: `"all"` (the default) every pole, `"none"` the finite part alone
(its pole-only work skipped natively, the pole entries quiet NaNs), `"uv"` and
`"ir"` the split selections (the finite part is the same in all four);
model parameters (at open and through `rusticol_virtual_update_parameters`)
follow the Python API's rules above, with the same messages
(`RUSTICOL_VIRTUAL_INVALID_ARGUMENT`); the default R2 strategy `"auto"` selects
the artifact's model R2; correlated, resolved and extended-precision calls go
through the Python API. A
v2 artifact carries no API
bundle (a requested `emit_api_bundle` is dropped with a recorded configuration
adjustment).

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
