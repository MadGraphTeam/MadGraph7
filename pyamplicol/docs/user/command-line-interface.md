---
title: "Command-Line Interface"
nav_order: 5
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Command-Line Interface

The `pyamplicol` command covers the complete user workflow: inspect or compile a
model, plan or generate process artifacts, inspect their physics axes, evaluate
phase-space points, and profile the optimized runtime. The same schema also
drives the typed [Python API](python-api.md).

> [!TIP]
> Start with the four-card walkthrough in [Quick Start](quick-start.md). This page
> is a command map, not a requirement to configure every available switch.

## Command map

| Command | Purpose | Typical use |
| --- | --- | --- |
| `generate` | Plan or write a process artifact | `pyamplicol generate "d d~ > z g" artifacts/builtin_ddbar_to_zg --model built-in-sm` |
| `evaluate` | Evaluate totals or resolved components | `pyamplicol evaluate artifacts/builtin_ddbar_to_zg --momenta point.json` |
| `profile` | Calibrate and measure the optimized total path | `pyamplicol profile artifacts/builtin_ddbar_to_zg --target-runtime 1` |
| `benchmark` | Compatibility alias for `profile` | Existing cards use `action = "benchmark"` |
| `check` | Pole (Catani; loop-induced: finiteness) and Ward checks of a one-loop request, regenerated in memory | `pyamplicol check loop_gg_zgg_generate.toml` or `pyamplicol check artifacts/loop_gg_zgg` |
| `inspect` | Show artifact and process metadata without running it | `pyamplicol inspect artifacts/builtin_ddbar_to_zg` |
| `group-processes` | Find generated subprocesses with equal matrix elements | `pyamplicol group-processes artifacts/pp_jj` |
| `model inspect` | Summarize a built-in, JSON, UFO, or compiled model | `pyamplicol model inspect models/json/sm/sm.json` |
| `model compile` | Compile portable model IR or a prepared kernel bundle | See [Models and Processes](models-and-processes.md) |
| `model processes` | Expand and inspect a process request without generating it | `pyamplicol model processes "p p > all all" --model built-in-sm` |
| `config template` | Write the exhaustive schema-v1 TOML template | `pyamplicol config template run.toml` |
| `config resolve` | Show requested and effective settings | `pyamplicol config resolve run.toml` |
| `examples list` | List shipped card names, actions, and descriptions | `pyamplicol examples list` |
| `examples copy` | Create an editable, self-contained example workspace | `pyamplicol examples copy ./pyamplicol-examples --force` |
| `examples run` | Run one named packaged card in a private cache workspace | `pyamplicol examples run evaluate_total` |
| `profiling-campaign copy` | Create a self-contained performance campaign | See [Profiling Campaigns](profiling-campaigns.md) |
| `doctor` | Diagnose the installed runtime and SDK | `pyamplicol doctor` |
| `self-test` | Run the installed-package smoke test | `pyamplicol self-test` |

License-request helpers are documented in
[Symbolica and Licensing](symbolica-and-licensing.md).

Run `pyamplicol COMMAND --help` for the options accepted by a command. For the
model family, place `--help` after the second verb, for example
`pyamplicol model compile --help`.

## Cards, direct options, and overrides

There are three equivalent ways to supply configuration:

1. a schema-v1 TOML card;
2. dedicated command-line options;
3. repeated `--set PATH=VALUE` overrides.

Configuration precedence is:

```text
defaults < TOML card < dedicated CLI options < repeated --set overrides
```

Any license or resource clamp is applied last and reported in the effective
configuration. Unknown fields are errors rather than silently ignored.

Run a card directly:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

Use an explicit action when the same card supplies shared settings:

```console
pyamplicol generate --card qq_z6g_recurrence_jit_o2.toml
pyamplicol profile --card qq_z6g_recurrence_jit_o2.toml
```

Override one leaf without editing the card:

```console
pyamplicol generate_pp_zjj_from_ufo_sm.toml \
  --set generation.workers=2 \
  --set generation.mode=replace
```

`--set` is deliberately repeatable and order-sensitive. For commonly used
fields, prefer the readable dedicated form—for example `--workers 2`,
`--execution-mode eager`, or `--color-accuracy nlc`.

For the complete field reference, see [Configuration](configuration.md) or create
an annotated local template:

```console
pyamplicol config template pyamplicol.toml
```

## Generate

The shortest built-in-model command is:

```console
pyamplicol generate "d d~ > z g" artifacts/builtin_ddbar_to_zg \
  --model built-in-sm
```

Useful generation options include:

- `--process EXPRESSION` to add another request;
- `--name NAME` to name a request;
- `--multiparticle 'p=d,d~,g'` to define a multiparticle label;
- `--flavor-scheme N` and `--max-quark-lines N` to constrain expansion;
- `--flavor-blind` to generate one subprocess per flavour-relabelling class
  (see [Group equivalent subprocesses](#group-equivalent-subprocesses));
- `--analyze-process-groups {none,numerical}`, `--only-unit-multiplier`, and
  `--[no-]consider-charge-conjugation` to search the generated subprocesses
  for equal matrix elements;
- `--color-accuracy {lc,nlc,full}`;
- `--color-contraction {direct,symmetric-group-fft}`;
- `--fft {trace,adjoint}` as a shorthand selecting symmetric-group FFT and its basis;
- `--correlators FILE.json` to prepare named colour operators and allowed spin
  replacements for the [Python and native correlated APIs](../correlators.md);
- `--lc-flow-layout {topology-replay,all-flow-union}`;
- `--execution-mode {recurrence,compiled,eager,on-the-fly}`;
- `--backend {jit,asm,cpp}`;
- `--workers auto|N`;
- `--mode {error,append,replace}` or the `--force` shortcut for replacement;
- `--no-emit-api-bundle` when standalone drivers are not wanted;
- `--numerical-current-reuse` to search for certified numerical current
  relations at generation time (off by default because the high-precision
  probes are costly);
- `--post-build-validation` for an optional immediate native total-versus-
  resolved smoke after writing the artifact;
- `--loop-induced-probes` to run the generation-time finiteness, Ward and
  Goldstone probes of loop-induced processes (binary64, double-double where
  a binary64 probe fails or is refused; off by default because the written
  artifact does not depend on them; they validate a new or edited loop UFO,
  see [Configuration](configuration.md)), and `--allow-failed-probes` to
  store a process whose probe failed (it acts only with the probes on).

Planning is non-writing:

```console
pyamplicol generate "p p > Z j j" artifacts/unused \
  --model models/json/sm/sm.json \
  --multiparticle 'p=d,d~,g' --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2 \
  --execution-mode compiled --dry-run
```

Adaptive `--fft adjoint` is the recommended starting point for exact FFT colour
contraction. It selects certified DDM where valid and otherwise retains exact
fundamental-chain or trace tensors. Select contracted colour and
a supported execution lane (`recurrence` or `on-the-fly`) explicitly:

```console
pyamplicol generate "g g > g g g" artifacts/ggg_adjoint_fft \
  --model built-in-sm --color-accuracy full \
  --fft adjoint --execution-mode recurrence
```

Use `--execution-mode on-the-fly` for the compact OTF counterpart. The same
option works for quark processes, using their existing fundamental chains:

```console
pyamplicol generate "d d~ > z g g" artifacts/quark_zgg_fft \
  --model built-in-sm --color-accuracy full \
  --fft adjoint --execution-mode recurrence
```

For a certified `n`-gluon tree, DDM fixes two gluon anchors and retains
`(n-2)!` orderings instead of `(n-1)!`. Certified single-insertion HEFT also
uses DDM: see `examples/builtin_sm_heft.toml`, which explicitly sets `HIG = 1`.
Other interactions retain their exact trace representation; multi-quark-line
processes retain fundamental-chain products, not a JO primitive basis.
The saved `fft_basis_selection` states the requested and actual basis and the
selection reason. `--fft trace` forces the original representation.
Both requests preserve `full` or `nlc` accuracy; forced adjoint FFT rejects correlations.
In the benchmarked pure-gluon family the adjoint basis was faster than trace
for six or more external gluons (3.9x per sample and 4x faster generation at
ten gluons); at lower multiplicity the two are comparable, so compare warmed
evaluation and setup costs for other workloads.
`--color-contraction symmetric-group-fft` remains valid and defaults to adjoint.
Without an explicit contraction flag, NLC/full recurrence and on-the-fly
generation automatically try adaptive adjoint FFT and fall back to direct if
the FFT plan is unsupported. LC, compiled/eager, and correlated generation
remain direct/trace. Use `--color-contraction direct` to opt out.
Do not combine `--fft` and `--color-contraction`; use either spelling. Card
settings are overridden by dedicated flags, then by ordered `--set` options
as usual.

For the dedicated direct/FFT/reference scaling scan, see
[FullColor FFT Profiling](fullcolor-fft-profiling.md).

`--dry-run` performs the operation exposed as `Generator.plan()`. It creates no
artifact, output directory, or compiled-model cache entry. A raw UFO or JSON
source must already have a reusable compiled-model cache entry, or be compiled
explicitly, because planning never compiles trusted model input as a side
effect. A one-loop dry run runs the request-model preparations and so fills the
one-loop preparation cache (unless it is off), preparing a later generation
ahead.

See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) before
combining execution modes and prepared-model backends.

### Correlated Born quantities

The [Born Correlations guide](../correlators.md#json-and-command-line) gives the
declaration format and complete runtime examples:

```console
pyamplicol generate 'g g > g g' artifacts/gg_correlated \
  --model built-in-sm --color-accuracy full --correlators correlators.json
```

`--correlators` is a generation option, separate from the ordinary TOML colour
settings. `--color-accuracy lc|nlc|full` selects the correlated approximation
(the ordinary default is LC). Complete amplitudes are generated in all cases;
direct contraction and generic compiled execution are selected, with those
adjustments recorded in the effective configuration. Partial
helicity/colour generation, append, and `--dry-run` are not supported on this
path. Numerical spin-vector setting and correlated evaluation use
`Runtime.set_spin_correlation_vectors(...)` and
`Runtime.evaluate_correlated(...)` or `Runtime.evaluate_correlated_many(...)`
in Python, or the corresponding [native SDK calls](native-apis.md#correlated-born-evaluations),
not the CLI `evaluate` command.

## Inspect and select a process

List an artifact's stable process IDs and coverage:

```console
pyamplicol inspect artifacts/pp_zjj
```

Focus on one public process ordering:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

The selector may be a stable process ID, an explicit alias ID, an exact stored
expression, or a unique side-preserving permutation of one. See
[Process Selection and Permutations](process-selection-and-permutations.md) for
the ordering contract.

Inspection does not execute evaluator state. Human output uses compact colored
tables; machine output is available with `--json`.

The `Process groups` section shows the artifact's
[process-group map](process-grouping.md): an overview row, then one table of
groups, each representative in bold followed by its members. Colours mark
`generated` (green) and `routed` (cyan) members, `C` (magenta), factors other
than 1 and warnings (yellow), and the basis of each relation. For example, for
`p p > j j j j` generated with `--flavor-blind --analyze-process-groups
numerical` (abridged):

```text
Process groups

+--------+--------------+-----------+--------+------------+-------------+------------------+-----------+----------+
| groups | subprocesses | generated | routed | conjugated | factor != 1 | bases            | analyses  | warnings |
+--------+--------------+-----------+--------+------------+-------------+------------------+-----------+----------+
|     23 |          486 |        23 |    463 |        140 |           0 | flavor-blind 463 | numerical |        0 |
+--------+--------------+-----------+--------+------------+-------------+------------------+-----------+----------+

+--------------------+---------------------------------------------------------------------------------------------+
| representative     | members                                                                                     |
+--------------------+---------------------------------------------------------------------------------------------+
| d d > d d d d~     |  4 routed    u u > u u u u~      s s > s s s s~      c c > c c c c~      b b > b b b b~     |
| p_p_to_j_j_j_j_1   |  5 routed C  d~ d~ > d d~ d~ d~  u~ u~ > u u~ u~ u~  s~ s~ > s s~ s~ s~  c~ c~ > c c~ c~ c~ |
| 10 subprocesses    |              b~ b~ > b b~ b~ b~                                                             |
+--------------------+---------------------------------------------------------------------------------------------+
...
+--------------------+---------------------------------------------------------------------------------------------+

routed: not generated, evaluated through its representative; C: charge-conjugated relation
compact view of 486 subprocesses; IDs and leg permutations: --json; one group row by row: --process REPRESENTATIVE
numerical analysis: 3 points, tolerance 1e-09
```

`routed` members are not generated and are evaluated through their
representative; `C` marks a charge-conjugated relation. A map with at most 80
subprocesses is listed one member per row, with its process ID, role, and
basis, plus a factor column when a factor differs from 1, a `C` column when a
relation is conjugated, a permutation column when a leg permutation is not the
identity, and a `note` column with `[n]` references to the warnings printed
below the table. Larger maps list each group's members as a grid of process
strings, one block per relation; subprocesses alone in their group are listed
as `ungrouped`. When every process is its own group, the section is the single
line `Process groups: none; each of the N processes is its own group`. With
`--process`, only the selected process's group is shown, so `--process
REPRESENTATIVE` lists a large group member by member. A routed member has no
runtime process of its own: without `--full-physics`, selecting it stops with
an error that names its representative. `--json` reports the
complete map under `process_groups`: `stored`, `analyses`, and per group the
`representative_id`, `representative`, and `members` (`id`, `process`, `pdgs`,
`initial_count`, `factor` as a fraction string, `permutation`, `conjugated`,
`basis`, `generated`, `warning`).

`--process MEMBER --full-physics` on a grouped process adds a `Process Group`
table, and a `process_group` object to the JSON. For a member, it gives the
representative, factor, conjugation and the leg permutation from the
representative to the requested leg order; for a representative, the size of
its group, and that permutation when the requested leg order is not the stored
one:

```text
+-----------------------------------------------------------------------------------------+
|                                      Process Group                                      |
+----------------+------------------------------------------------------------------------+
| field          | value                                                                  |
+----------------+------------------------------------------------------------------------+
| role           | routed member: not generated, evaluated through the representative     |
| representative | p_p_to_j_j_j_j_52  d u~ > d d d~ u~                                    |
| factor         | 1                                                                      |
| conjugated     | yes (C)                                                                |
| permutation    | p = [1 0 4 5 3 2]: representative leg r takes the momentum of leg p[r] |
| basis          | flavor-blind                                                           |
| group          | 40 subprocesses: 1 generated, 39 routed, 20 conjugated                 |
+----------------+------------------------------------------------------------------------+
```

## Group equivalent subprocesses

Every artifact records in `process-groups.json` which subprocesses share one
matrix element, up to a rational factor and a leg permutation. By default each
generated subprocess is its own group.

`generate --flavor-blind` assumes that subprocesses related by a relabelling of
flavours with equal spin, colour, mass, and width (and, unless
`--no-consider-charge-conjugation`, by charge conjugation) are equal. Only one
representative per class is generated; the others are recorded as its members
and exported to UMAMI through the representative's runtime. The assumption is
not verified: a warning names the members whose relabelling is not a symmetry
of the model couplings, for example electroweak couplings of different quark
flavours.

Generation logs the outcome of this pre-grouping before any representative is
generated:

```text
INFO pyamplicol.generation: Flavor-blind grouping: 486 subprocesses -> 23 generated representatives; 463 routed members (140 by charge conjugation); 0 relabelling warnings
```

The summary table printed at the end has a `process grouping` row, coloured
like the `inspect` display, read from the written `process-groups.json`
(abridged):

```text
| processes              | 23 requests                                                                              |
| process grouping       | 486 subprocesses -> 23 groups: 23 generated, 463 routed                                  |
|                        | 463 flavor-blind relations (140 charge-conjugated); 0 warnings                           |
|                        | numerical analysis: 3 points, tolerance 1e-09                                            |
```

Its first line counts the subprocesses, groups, and generated and routed
subprocesses; the second, the relations by basis, those that are
charge-conjugated or have a factor other than 1, and the relabelling warnings;
the last ones, each recorded numerical analysis. The `Generated Processes`
table gains a `group` column: the size of each representative's group, or
`member of REPRESENTATIVE` for a member generated in its own right, followed by
its factor when it is not 1 and `C` when the relation is conjugated, as in
`member of d_dbar_to_a_a (x16)`. When
every subprocess is its own group, the column is omitted, and so is the row
unless an analysis was recorded; the row then reads `no relation found; each of
the N processes is its own group`.

The numerical analysis compares generated subprocesses with equal external
particle types at three random phase-space points and merges those whose
resolved components agree up to one common rational factor:

```console
pyamplicol generate "p p > j j" artifacts/pp_jj --analyze-process-groups numerical
pyamplicol group-processes artifacts/pp_jj --only-unit-multiplier
```

`group-processes` runs the numerical analysis (its default) on an existing
artifact and rewrites `process-groups.json` in place; the artifact ID and the
UMAMI export are unchanged. `--only-unit-multiplier` accepts only relations
with factor 1. Subprocesses generated with a selected helicity or colour subset
are never grouped numerically. [Process Grouping](process-grouping.md)
describes the relations, the warnings and the file format in full.

## Evaluate

Evaluate a momenta file with optional parameter updates:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --model-parameters data/model_parameters.json \
  --momenta data/pp_zjj_momenta.json
```

Request all physical components rather than only the optimized sum:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --resolved
```

Split recurrence totals by squared coupling order with `--orders`; the JSON
output carries `values` `[point][order]`, the unchanged `total`, the squared
`orders` with their contributing amplitude-order pairs, and
`amplitude_orders`:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --orders
```

Global selectors may be repeated:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process p_p_to_z_j_j_3 \
  --helicity 'h:-1,+1,-1,+1,+1' \
  --color-flow 1 \
  --momenta data/pp_zjj_momenta.json
```

`--color-flow` accepts either a stable flow ID or the one-based ordinal shown
by `inspect`. Flow selection is LC-only; NLC and full-color artifacts expose a
contracted color output and accept helicity selection only. More examples are
in [Runtime and Selectors](runtime-and-selectors.md).

### Saved virtual artifacts

The artifact selects the virtual execution lane; loading it does not require
repeating its loop coupling orders. Supply the renormalization scale squared:

```console
pyamplicol evaluate artifacts/virtual --process gg --momenta points.json \
  --mu-r2 1000000 --arithmetic double --r2-strategy model --json
```

`--arithmetic` accepts `double`, `double-double` (DoubleFloat) or `arbitrary`.
For arbitrary precision, add `--precision 70`, for example. These choices apply
before model parameters are bound. Without an explicit arithmetic choice, the
existing precision convention applies: 16 digits selects double; other requested
precisions select arbitrary precision. DoubleFloat is always selected explicitly.

`--loop-reduction {auto,ninjars,tirust}` selects the reduction backend for the
evaluation, overriding the artifact's default (`generate --loop-reduction`
records that default): `auto` takes TIRust for a loop-induced and NinjaRS for
a Born-interfered process, falling back to NinjaRS with the reason recorded
where TIRust cannot serve the call; on TIRust a point it refuses or fails is
evaluated again on NinjaRS (the per-point fallback, recorded in that point's
lane diagnostic), and in double-double and arbitrary precision each point it
serves is guarded by a rotation re-test in the evaluation's
arithmetic: a point whose estimate exceeds the target (never below 1e-5)
fails with the reason (the estimate is in the lane's `reduction` record,
`guard_estimate`; [Python API](python-api.md), "Near-planar NinjaRS
hazard"); a binary64 fallback point is served unguarded; `ninjars` and `tirust` request one backend
without a fallback or guard (`tirust` serves every arithmetic; `ninjars`
without stability checks was exposed to that hazard until NinjaRS's
conditioning bound, 71eddb15); an explicit backend that cannot serve the call is refused.
`--reduction-target X` sets TIRust's target accuracy (the relative accuracy
of each point's result, default 1e-5; it sets the calibrated acceptance of
TIRust's checked method choice and, never below 1e-5, is the fallback
guard's threshold).
`--kinematics-improvement VARIANT` (`z-shift`, `rescaling`, `projection` or `off`; default
`z-shift`, MadLoop's `ImprovePSPoint = 2` with its fallback to 1, finished by
the projection where MadLoop leaves the point inexact) and
`--[no-]kinematics-improvement-binary64` (default off) set the improvement
of the input points to exact conservation and on-shell externals before a
double-double or arbitrary-precision evaluation (binary64 only with the
switch); the JSON output's lane diagnostics carry the per-point record
(`kinematics`; [Configuration](configuration.md#kinematics-improvement-of-the-input-points)).
Both are `evaluate`, `profile` and `check` flags (for `check`, below) and
the run card's `[evaluation] kinematics_improvement` /
`kinematics_improvement_binary64`; an unknown variant is refused with the
native policy's one refusal text when the configuration resolves.
`--reduction-unchecked` (default off) reduces without the backends' own
acceptance and refusal checks: TIRust's gates and method choice select the
method without re-checking its values, plain Passarino-Veltman at gate 0
serves what they refuse, NinjaRS runs with zero stability thresholds (its
box mu-expansion's conditioning bound stays active); only points without a
finite value or refused by that bound still fail, `auto` keeps its backend choice and
its fallback acts on those only. It is a timing and benchmark mode with no
accuracy certificate (the measured accuracy per process: [Python
API](python-api.md), "Unchecked reductions"). The JSON output's lane diagnostics name the requested and
effective backends, the reason, the fallback, the target and the unchecked
setting. A
loop-induced artifact (`contribution: "loop-induced"` in the JSON, empty
`born_values`) takes `--poles all` or `none` only.

Use `--mu-r2-file scales.json` for one positive scale squared per point, instead
of `--mu-r2`. A scale file such as `[1000000, "900000.0000000000000000001"]`
accepts JSON numbers or strings. Virtual momenta, parameter and scale files
preserve decimal digits before native arithmetic conversion. Parameter pairs
`[real, imaginary]` also preserve their exact inputs.

Human output lists every Born and virtual squared coupling order and all three
coefficients, with contributing Born–loop pairs and per-point diagnostics. JSON
adds explicit `epsilon_powers: [0, -1, -2]`, coefficient names and shape metadata;
high-precision values are decimal strings. Neither format silently sums orders
or divides by the Born result. A successful lane can still report
`accuracy_established: false`.

`--orders` splits a virtual evaluation by squared coupling order: one
evaluation (`Runtime.evaluate_orders`, a `VirtualOrderEvaluation` view), every
sector's Laurent coefficients and every Born order apart as above, followed
by a separately titled table of the explicit sums over the orders (JSON:
`evaluation`, `orders`, `born_orders`, `sum_orders`, `sum_born_orders`). The
sums are named and printed apart; nothing is summed silently.

The precision-rescue options (`--stability-higher`,
`--stability-relative-tolerance`, `--stability-absolute-tolerance`,
`--verify-successful`, card section `[evaluation.stability]`) were removed with
the legacy loop engine, the only one that served them; naming one is refused
with the reason. v2 evaluations return stability estimates
(`--stability-rotations`, `--stability-reduction-direction`,
`--stability-rescaled`) and leave escalating `--arithmetic` to the caller.

`--on-failure raise` is the default. `--on-failure return` preserves successful
points and returns an unavailable whole-point row for each numerical failure:
`null` in JSON, with its failure diagnostics. An actual zero remains a
numerical zero. A scale at which a counterterm coupling is not finite in the
chosen arithmetic (e.g. `1e-400` in binary64) is such a failure of its own
points. Malformed inputs (a non-positive `--mu-r2`) and unsupported
capabilities remain errors for the whole call. Both inclusive and `--resolved`
evaluation use this option; resolved axes follow the artifact's retained
layout. It requires virtual evaluation and is rejected on other commands,
including `profile`; configuration inspection can still display it.

Only strategies and observables present in the artifact may be selected.

### Correlated requests

Supply a JSON object of labelled numerical requests to evaluate colour and spin
correlations together over the entire point batch:

```console
pyamplicol evaluate artifacts/virtual --momenta points.json --mu-r2 1000000 \
  --correlated-requests requests.json --on-failure return --json
```

```json
{
  "physical": {"color_correlation": "born", "spin_vectors": {}},
  "joint": {
    "color_correlation": "born",
    "spin_vectors": {
      "3": ["0", ["0.8", "-0.16"], "0", ["-0.6", "0.12"]],
      "4": ["0", "0", ["1", "0.2"], "0"]
    }
  }
}
```

Request labels become output keys. `color_correlation` defaults to `born`; use
stored colour-history IDs for other operators. The supplied vector-leg set must
be a class declared in the artifact. Leg keys use public one-based labels, and
each value is a four-vector or one four-vector per point. Components may be
exact real numbers, strings, or `[real, imaginary]` pairs. Vectors are used
literally, without normalization or projection. Unknown fields, duplicate JSON
keys and colliding leg spellings such as `"3"` and `"03"` are errors.

Missing or null `spin_vectors` inherits the runtime's default; a fresh CLI
runtime has no replacement sources. `{}` explicitly selects physical helicities.
One entry requests one observable; multiple entries use the existing grouped
runtime call, preserving each label and its independent failure mask. Human
output prints each result under its label; virtual JSON retains every coupling
order, Laurent coefficient and diagnostic, including unavailable rows.

Tree artifacts use the same request file, without virtual scales, R2, stability
or partial-return options. Their correlated API defaults to arbitrary precision;
`--arithmetic double-double` selects its existing DoubleFloat path. `--precision`
and `--helicity` follow the underlying correlated API. Virtual requests inherit
the arithmetic and stability policy selected at load time.

`--resolved` and `--color-flow` cannot be combined with correlated requests:
the grouped API returns contracted observables. Generation-time `--correlators`
declares capabilities; evaluation-time `--correlated-requests` supplies numerical
values and selects among those capabilities.

## Profile

The profiler measures the same optimized total path as `Runtime.evaluate()`:

```console
pyamplicol profile artifacts/pp_zjj \
  --process 'd d~ > g z g' \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0 \
  --batch-size 128 \
  --color-flow 1 \
  --precision 16
```

It calibrates repetitions, keeps at least the requested independent sample
count, and reports uncertainty. `Ctrl-C` during sampling preserves completed
blocks and prints a result marked partial. See
[Profiling and Benchmarking](profiling-and-benchmarking.md) for metric meanings
and selector-aware profiling. If both profile selector axes are omitted, the
stored LC layout supplies a deterministic optimized selector; explicit
non-hot shapes are retained and produce at most one pre-loop warning per
loaded process.

A one-loop (v2 virtual) artifact is profiled per stage: `pyamplicol profile
artifacts/loop_gg_zgg --momenta points.json` times `--minimum-samples` calls
of one batch of `--batch-size` points (the artifact's validation points
repeated when `--momenta` is absent), finite part only, after
`--warmup-runs` warm-up calls, and prints the per-point time (median call,
fastest and slowest) and a coloured table of the time per point in each stage
group (trees: the native trees and counterterm kets; dressing: seeds, factors
and dressing of a Born-interfered process, the dressing of a loop-induced
one; scatter; reduction; rest: every other stage and the untimed remainder),
read from the native executor's always-on stage reports of the same timed
calls, every native batch of a call counted (a per-point fallback, a rescue
level; no other timer runs). `--loop-reduction` selects the backend,
`--reduction-unchecked` the unchecked timing mode (no accuracy certificate)
and `--mu-r2` the scale (default: per point the squared invariant mass of its
initial state). JSON: `benchmarking.VirtualProfileResult` (`seconds_per_point`,
`low`, `high`, `stages`, `reduction`, `unchecked`, `batch`, `samples`,
`failed_points`).

## Check

`pyamplicol check` runs the pole and Ward checks of a one-loop request:

```console
pyamplicol check loop_gg_zgg_generate.toml              # a run card's request
pyamplicol check artifacts/loop_gg_zgg --points 4 --json  # an artifact's request
pyamplicol loop_gg_zgg_check.toml                      # action = "check"
```

The target is a run card (its process entries as its generation would make
them) or an artifact directory: every process its manifest stores (`id` and
expression, `artifact.json`), regenerated with the configuration its
`config/effective.toml` records, from the model the artifact stores. The
prepared model must be the stored one (its manifest record: name, source
kind, content hash and restriction) and the regenerated processes exactly the
stored ones, else `check` refuses the artifact before generating anything
(exit 2). An artifact generated from a UFO directory records only the
directory's name; name the model with `--model SOURCE` (`check.model`, a
source as `model.source` takes it; it must still be the stored model).
Every process is **regenerated in memory** through the generation
service (nothing is written; one model preparation shared by the process
and its probe slices, the persistent preparation cache serving it as at
generation): the request on the model's full tree kernels,
with every width zeroed outside the complex-mass scheme (a fixed width in a
tree propagator breaks the identities; the complex-mass scheme is gauge
invariant and keeps its widths), as the generation-time probes do. Checks run
at `--points N` (default 3, `check.points`) seeded phase-space points
(`--seed S`, default `generation.validation.seed`) of the generation's
validation convention (decays at rest; scatterings in the centre-of-mass
frame at sqrt(s) = max(1000, final threshold + 100) for massless beams,
twice the larger threshold for massive ones; `--sqrt-s E`, `check.sqrt_s`,
sets a scattering's energy in the model's mass unit, above its threshold),
lifted to exact momentum conservation, at mu_R^2 = the squared invariant mass
of the initial state, on NinjaRS at every carrier:

| Check | Born-interfered | Loop-induced | Criterion |
| --- | --- | --- | --- |
| poles | every squared-order sector's 1/eps and 1/eps^2 against Catani's operator of the requested corrections (QCD; QED where requested), formed natively | the finiteness probe: the amplitude's poles against the row term floor S_k(h) (the pre-cancellation scale) | binary64 with the recommended checks: \|pole - Catani\| <= 10 E + 1e-12 \|Catani\|; double-double: 1e-12 relative; finiteness: within 10 x its own estimate |
| ward | epsilon_j -> c p_j / p_j^0 on every external gluon and photon, c = 1 and i, Born and virtual parts (`ward.ward_row`) | epsilon_j -> p_j / E_j on one leg of every gluon and photon species, amplitude level | binary64: \|V\| <= max(1e-12 S, E) and \|V\| <= 10 E with E <= 1e-9 S (scope); double-double: 1e-25 S; Born 1e-12 / 1e-24; probes: 10 x their estimate |
| goldstone | not applicable (the probe is amplitude-level) | k.A(V) = eta M_V A(G) on one leg of every massive-vector species with a Goldstone partner, in 't Hooft-Feynman gauge | within 10 x its estimate |

`--precision auto` (default) runs binary64 first and double-double only for a
row whose binary64 run fails, is refused or is out of its criterion's scope
(a pole row whose relative estimate exceeds the default target accuracy
1e-5 or that carries no estimate; a Ward row with E above 1e-9 of the
physical scale or without an estimate); `double` or `double-double` runs one
carrier alone, and there a row out of its carrier's scope reads "not run"
(exit 3), never a failure or a pass. When binary64 evaluated a point and
the double-double run is refused, the rows binary64 left undecided keep its
verdict and say that double-double was refused. In `auto`, a row
double-double fails while its residual is within 10 x double-double's own
estimate (asked for only then: the recommended checks for poles, the
reduction-direction check for Ward) has run out of digits: only such rows,
at their points, are re-run in **arbitrary precision** at the digit count
derived from the shortfall, d = ceil(32.0 + log10(10 E / B)) for an
estimate E and a bound B (32.0: double-double's decimal digits, 1 +
log10(0.5 / 2^-104); the largest over the point's rows; `--arbitrary-digits N`,
`check.arbitrary_digits`, sets it), the point lifted again at those digits,
and judged by their double-double criterion (no new bound; the row shows
carrier `arbitrary`, its digits and the double-double numbers); a
double-double residual far above its own estimate stays a failure. A row
still undecided there, or undecided at an explicit carrier, reads
"undecided" with both carriers' numbers and counts as a failure (exit 1).
The loop-induced probes' criterion is their own estimate, so they never
take this rung. Each double-double and arbitrary-precision rung (and the
binary64 one under `--kinematics-improvement-binary64`) first improves its
lifted point per `--kinematics-improvement` (`evaluation.
kinematics_improvement`, default `z-shift`) in its own carrier, exactly as
`evaluate` does; `check` lifts its points onto exact conservation along the
beam axis at twice the rung's digits already, so this moves them by about
the rung's rounding, and
each row records it (`CheckRow.kinematics`: the producing variant, `exact` or
`not-improved`). The Ward rows of a request whose loop corrections include an
electroweak one (any correction other than the model's strong one) and with
an on-shell zero-width external massive quark (a top or a massive b, quark or
antiquark) read "not applicable" and are not evaluated: the Ward identity of
the asymmetric check is not guaranteed with electroweak corrections there;
Ward rows are not split by squared-order sector, so that request's QCD-loop
sectors are not Ward-checked either. A sector Catani's operators
do not predict (a QCD correction of a process with photon-exchange Borns,
...) is listed as "no prediction" with its reason; a sector they predict
that got no pole row at a point reads "not run" (exit 3: an applicable
check did not run); a check that does not apply (no
external gluon or photon, no massive vector, a unitary-gauge request) reads
"not applicable" with the reason; a Goldstone side the engine refuses reads
"not run". None of these is a pass.

Every row names the check, its subject (sector and pole, leg and factor),
the point, the carrier that decided, `ratio` (the measured value over its
bound: <= 1 passes), the deviation and E over the row's scale where formed
(`dev / scale`, `E / scale`: a binary64 pole row's pre-cancellation scale S;
a double-double pole row's |Catani|, its criterion being relative; a
Born-interfered Ward row's physical scale; a probe's row term floor
S_k(h)), the fallback's reason and the verdict. The exit code is 0 iff
every applicable check passed, 1 on any failure, and 3 when a process had
nothing to check (or an applicable check did not run); an "undecided" row
counts as a failure (1). `--json` returns the
`CheckReport` (`source`, `source_kind`, `points`, `seed`, `precision`,
`carriers`, `regeneration`, `seconds`, `exit_code`, and per process `process`,
`expression`, `contribution`, `widths`, `zeroed_widths`, `reduction`,
`status`, `seconds` and `rows` with the fields above and `digits`, the
arbitrary-precision carrier's digit count); `carriers` ends with `arbitrary`
under `auto`; `source` names the run card or artifact checked. Python:
`pyamplicol.api.check(target, points=..., seed=..., precision=..., model=...,
sqrt_s=..., arbitrary_digits=...)`.

Cost: one probe-slice generation per process (about the generation's own
cost: the request-model preparation included) plus a few evaluations per
point; the generation-time probes (`generation.loop_induced_probes`, off by
default) are the same loop-induced checks at one point during generation.

## Models

Inspect a trusted UFO directory without generating a process:

```console
pyamplicol model inspect models/ufo/sm
```

Compile portable model IR:

```console
pyamplicol model compile models/ufo/sm models/sm.pyAmplicol-model.json
```

Prepare a reusable JIT-O2 kernel bundle for eager or recurrence execution:

```console
pyamplicol model compile \
  models/json/sm/sm.json models/ufo-sm-jit-o2.pyamplicol-model \
  --backend jit --jit-optimization-level 2
```

Enumerate a broad request without writing a process artifact:

```console
pyamplicol model processes "p p > all all" \
  --model built-in-sm --flavor-scheme 1 --max-quark-lines 0
```

The distinction between portable model IR and prepared kernel bundles is
explained in [Models and Processes](models-and-processes.md).

The model caches are controlled by `--model-cache` / `--no-model-cache`,
`--model-cache-dir DIR` and, for one-loop generation, `--loop-preparation-cache`
/ `--no-loop-preparation-cache` (the persistent cache of the lowered model
programs, on by default, `[model] loop_preparation_cache`; also off with
`--no-model-cache` or `PYAMPLICOL_LOOP_PREPARATION_CACHE=0`). See
[Model caches and the one-loop preparation cache](configuration.md#model-caches-and-the-one-loop-preparation-cache).

## Examples and installed workspaces

List the available names before running one:

```console
pyamplicol examples list
pyamplicol examples run generate_pp_zjj_from_ufo_sm
```

The list uses a colored table on a terminal. Add `--json` for stable,
uncolored machine-readable output.

`examples run` accepts the stem of a shipped `.toml` card. `all_options` is a
reference template, not a runnable example. Unknown names fail with the full
available-name list.

For an editable and inspectable workspace, copying is usually clearer:

```console
pyamplicol examples copy ./pyamplicol-examples --force
cd pyamplicol-examples
pyamplicol generate_pp_zjj_from_ufo_sm.toml
```

See [Examples Gallery](examples-gallery.md) for the complete card inventory and
recommended sequences.

## Output, color, progress, and logging

The common presentation options are:

| Option | Values | Meaning |
| --- | --- | --- |
| `--json` | flag | Emit stable machine-readable stdout instead of the human-facing table |
| `--color` | `auto`, `always`, `never` | Detect a terminal, force ANSI color, or disable it |
| `--progress` | `auto`, `tty`, `log`, `off` | Select live bars, rate-limited log progress, or silence |
| `--log-level` | `debug`, `info`, `warning`, `error` | Control diagnostic verbosity |

JSON mode keeps stdout machine-readable; progress and diagnostics use stderr.
This makes pipelines predictable:

```console
pyamplicol inspect artifacts/pp_zjj --json > artifact.json
```

Use `--color always` only when the consumer understands ANSI escapes. The
default `auto` mode produces colored terminal tables and plain redirected
output.

## Diagnostics

Start with the two installed-package checks:

```console
pyamplicol doctor
pyamplicol self-test
```

Then inspect the relevant artifact and effective configuration:

```console
pyamplicol inspect artifacts/pp_zjj --json > inspect.json
pyamplicol config resolve run.toml --json > config.json
```

For common failures and the information to include in a bug report, see
[Troubleshooting](troubleshooting.md) and [Release and Support](release-and-support.md).

## Next steps

- [Configuration](configuration.md)
- [Models and Processes](models-and-processes.md)
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Native APIs](native-apis.md)
