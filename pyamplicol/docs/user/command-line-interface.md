---
title: "Command-Line Interface"
nav_order: 4
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
| `inspect` | Show artifact and process metadata without running it | `pyamplicol inspect artifacts/builtin_ddbar_to_zg` |
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
- `--color-accuracy {lc,nlc,full}`;
- `--lc-flow-layout {topology-replay,all-flow-union}`;
- `--execution-mode {recurrence,compiled,eager}`;
- `--backend {jit,asm,cpp}`;
- `--workers auto|N`;
- `--mode {error,append,replace}` or the `--force` shortcut for replacement;
- `--no-emit-api-bundle` when standalone drivers are not wanted;
- `--no-numerical-current-reuse` for an unoptimized diagnostic build;
- `--post-build-validation` for an optional immediate native total-versus-
  resolved smoke after writing the artifact.

Planning is non-writing:

```console
pyamplicol generate "p p > Z j j" artifacts/unused \
  --model models/json/sm/sm.json \
  --multiparticle 'p=d,d~,g' --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 --max-quark-lines 2 \
  --execution-mode compiled --dry-run
```

`--dry-run` performs the operation exposed as `Generator.plan()`. It creates no
artifact, output directory, or model-cache entry. A raw UFO or JSON source must
already have a reusable compiled-model cache entry, or be compiled explicitly,
because planning never compiles trusted model input as a side effect.

See [Generation Modes and Evaluators](generation-modes-and-evaluators.md) before
combining execution modes and prepared-model backends.

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

Global selectors may be repeated:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process p_p_to_z_j_j_4 \
  --helicity 'h:-1,+1,-1,+1,+1' \
  --color-flow 1 \
  --momenta data/pp_zjj_momenta.json
```

`--color-flow` accepts either a stable flow ID or the one-based ordinal shown
by `inspect`. Flow selection is LC-only; NLC and full-color artifacts expose a
contracted color output and accept helicity selection only. More examples are
in [Runtime and Selectors](runtime-and-selectors.md).

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
  --backend jit --jit-optimization-level 2 --jit-compress
```

Enumerate a broad request without writing a process artifact:

```console
pyamplicol model processes "p p > all all" \
  --model built-in-sm --flavor-scheme 1 --max-quark-lines 0
```

The distinction between portable model IR and prepared kernel bundles is
explained in [Models and Processes](models-and-processes.md).

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
