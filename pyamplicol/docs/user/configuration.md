<!-- SPDX-License-Identifier: 0BSD -->

# Configuration

TOML schema version 1 is shared by the CLI and Python configuration classes.
`examples/all_options.toml` is the exhaustive commented field reference.

## Primary Run Card

The recommended card uses a serialized external model so loading does not
execute UFO Python:

```toml
schema_version = 1
action = "generate"

[model]
source = "models/json/sm/sm.json"
restriction = "default"

[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]

[color]
accuracy = "lc"

[generation]
output = "artifacts/pp_zjj"
emit_api_bundle = true
```

Run it and override fields without editing the card:

```console
pyamplicol external_json_sm.toml \
  --set generation.workers=2 \
  --set evaluator.jit.optimization_level=3
```

The resolver applies defaults, card values, dedicated flags, repeated `--set`
overrides in order, then effective license/resource clamps. Requested and
effective configurations are retained separately. Unknown fields are errors
and may include a nearest-field suggestion.

Model, cache, output, artifact, momenta, and parameter-card paths in TOML are
resolved relative to the card. `model.restriction` is a loader restriction name
such as `default`, `no_widths`, or `none`; omitting it applies a model-provided
default when available. The typed `ModelSource` API can instead receive an
explicit restriction file path.

## Processes

`process.entries` is the only process-request list. An entry may have a stable
name; unnamed multiparticle requests receive generated concrete names:

```toml
[process]
entries = [{ expression = "p p > Z j j" }]
```

For the primary two-flavor expansion, the current physical filter produces 19
processes named `p_p_to_z_j_j_1` through `p_p_to_z_j_j_19`. The name is not a
filesystem path. `p_p_to_z_j_j_4` selects `d d~ > Z g g` from the shared
`artifacts/pp_zjj` root.

Explicit named requests can mix multiplicities in one artifact:

```toml
[process]
entries = [
  { expression = "u u~ > Z g", name = "uubar_Zg" },
  { expression = "u u~ > Z g g", name = "uubar_Zgg" },
]
```

`flavor_scheme`, `max_quark_lines`, and coupling-order policy constrain
expansion. Coupling-order names are model data and are filtering/scheduling
hints; the generic engine does not assign fixed QCD or electroweak semantics to
arbitrary names.

## Direct Commands

The direct equivalent of the primary card is:

```console
pyamplicol generate "p p > Z j j" artifacts/pp_zjj \
  --model models/json/sm/sm.json \
  --restriction default \
  --multiparticle 'p=d,d~,g' \
  --multiparticle 'j=d,d~,g' \
  --flavor-scheme 2 \
  --max-quark-lines 2
```

The command families are:

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

Use `generate --dry-run` for the non-writing operation exposed by
`Generator.plan()`. An external UFO/JSON source must first be compiled or
present in the configured model cache; dry-run intentionally does not compile
trusted external input as a side effect.

## Color And Evaluation

| `color.accuracy` | Resolved color dimension |
| --- | --- |
| `lc` | One entry per physical leading-color flow |
| `nlc` | One contracted color entry per helicity |
| `full` | One contracted color entry per helicity |

LC generation always includes complete physical flow coverage. Runtime
selectors use `evaluation.helicity_ids` and `evaluation.color_flow_ids`;
benchmark selectors use the corresponding `benchmark` fields. Color-flow
selectors are valid only for LC. NLC/full currently describe the supported
contracted SU(3) calculations rather than an arbitrary UFO color basis.

`evaluation.resolved = false` selects the optimized total. With
`resolved = true`, all selected physical components are returned and their
explicit sum must agree with the total.

## Evaluators

| Backend | Use |
| --- | --- |
| `jit` | Default direct SymJIT application, optimization level 3 |
| `asm` | Symbolica assembly evaluator |
| `cpp` | Generated/compiled C++ evaluator with `[evaluator.cpp]` options |

The default batch and output chunk sizes are 128. Optimization defaults are 10
Horner iterations, backend-selected common-pair iterations, 1000 Horner
variables, 5,000,000 common-pair cache entries, and pair distance 1000.
Stage-local parameters are mandatory and are not a user toggle.

## Python Resolution

```python
from pyamplicol.config import resolve_config

resolution = resolve_config(
    "external_json_sm.toml",
    dedicated={"generation.workers": 4},
    overrides=("generation.workers=2", "generation.workers=1"),
)
assert resolution.requested.generation.workers == 1
assert resolution.effective.model.source.endswith("models/json/sm/sm.json")
```

In-memory mappings accept a `base_dir`; otherwise their relative paths use the
current directory. A `--set` value follows TOML syntax, so strings containing
spaces must be quoted.
