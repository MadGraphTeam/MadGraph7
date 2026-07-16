<!-- SPDX-License-Identifier: 0BSD -->

# Configuration

TOML schema version 1 is shared by the CLI and Python configuration classes.
`examples/all_options.toml` is the exhaustive commented reference for the
current field registry.

## Run Cards

```toml
schema_version = 1
action = "generate"

[model]
source = "built-in-sm"

[process]
entries = [{ expression = "d d~ > z g", name = "ddbar_zg" }]

[color]
accuracy = "lc"

[generation]
output = "artifacts/ddbar_zg"
```

Invoke a card as the first argument:

```console
pyamplicol run.toml --set generation.workers=2
```

The resolver applies defaults, card values, dedicated CLI flags, repeated
`--set` overrides in order, then effective license/resource clamps. It keeps
requested and effective configurations separately. Unknown dotted paths are
errors and may include a nearest-field suggestion.

`process.entries` is the only process-request list. Every entry has an
`expression` and may have a unique stable `name`; unnamed entries receive their
runtime name from the normalized expression. A mixed process set uses the same
field:

```toml
[process]
entries = [
  { expression = "d d~ > z g", name = "ddbar_zg" },
  { expression = "d d~ > z g g", name = "ddbar_zgg" },
]
```

TOML paths and path-valued `--set` overrides on a card are relative to the
card. In-memory mappings use `base_dir` when provided and otherwise use the
current directory. A `--set` value follows TOML syntax, so strings with spaces
need quotes:

```console
pyamplicol run.toml \
  --set 'process.entries=[{ expression = "d d~ > z g", name = "ddbar_zg" }]' \
  --set color.accuracy=full \
  --set output.format=json
```

## Direct Commands

The current parser supports:

```text
generate
evaluate
benchmark
inspect
model inspect|compile|processes
request-symbolica-trial-license
request-symbolica-hobbyist-license
config template|resolve
examples list|copy|run
doctor
self-test
```

The direct generation form accepts one positional process and repeatable
`--process` values. Repeat `--name` in the same order when assigning stable
names. Use `pyamplicol COMMAND --help` for the dedicated flags accepted by that
command.

Use `generate --dry-run` for the same non-writing planning operation exposed as
`Generator.plan()` in Python. `config template` writes the fully commented
schema reference; `config resolve` reports requested/effective values and
resource adjustments. Packaged examples can be listed, copied, or run without
assuming a source-checkout layout.

## Color And Evaluation

`color.accuracy` chooses the generated contraction:

| Value | Resolved color dimension |
| --- | --- |
| `lc` | One entry per physical leading-color flow |
| `nlc` | One contracted color entry per helicity |
| `full` | One contracted color entry per helicity |

LC generation always includes complete physical flow coverage. Runtime
selectors are supplied through `evaluation.helicity_ids` and
`evaluation.color_flow_ids`; benchmark selectors use the corresponding
`benchmark` fields. Color-flow selectors are valid only for LC; NLC/full expose
one contracted output component and reject them. These selectors do not limit
generated artifact coverage.

`evaluation.resolved = false` selects the optimized summed path. With
`resolved = true`, the returned physical components can be explicitly summed
and compared with the total.

## Evaluators

| Backend | Use |
| --- | --- |
| `jit` | Default Symbolica JIT, optimization level 3 |
| `asm` | Symbolica assembly evaluator |
| `cpp` | C++ evaluator using `[evaluator.cpp]` compiler options |

The default batch size and output chunk size are 128. Optimization defaults
are 10 Horner iterations, backend-default common-pair elimination iterations,
1000 Horner variables, 5,000,000 common-pair cache entries, and pair distance
1000. Stage-local parameters are mandatory and are not a user toggle.

## Python Resolution

```python
from pyamplicol.config import resolve_config

resolution = resolve_config(
    "run.toml",
    dedicated={"generation.workers": 4},
    overrides=("generation.workers=2", "generation.workers=1"),
)
assert resolution.requested.generation.workers == 1
```
