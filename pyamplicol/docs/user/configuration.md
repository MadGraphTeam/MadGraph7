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
pyamplicol generate_pp_zjj_from_ufo_sm.toml \
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
filesystem path. Either `p_p_to_z_j_j_4` or the concrete expression
`d d~ > z g g` selects that process from the shared `artifacts/pp_zjj` root.

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
profile
benchmark (compatibility alias for profile)
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

The direct `profile` command resolves internally to the schema-v1 `benchmark`
action. Existing cards therefore keep `action = "benchmark"`; both direct
spellings accept the same runtime, process, batch-size, sampling, selector, and
output options.

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

`color.lc_flow_layout` chooses how complete LC coverage is organized:

| LC flow layout | Optimized workload |
| --- | --- |
| `topology-replay` | Default. One runtime-selected flow with a helicity sum. |
| `all-flow-union` | All physical flows with one runtime-selected helicity. |

Both layouts retain every physical flow and helicity and accept runtime
selectors. The union layout constructs one shared cross-flow recurrence; enable
it in a card with `lc_flow_layout = "all-flow-union"` under `[color]`, or use:

```console
pyamplicol generate --card run.toml --lc-flow-layout all-flow-union
```

`all-flow-union` is rejected for NLC/full. It is also incompatible with an LC
request that fixes `process.selected_color_sector_ids` or
`process.selected_source_helicities`, or truncates coverage with
`process.max_color_sectors`. Use the default topology-replay layout for those
generation-selected or truncated artifacts.

`evaluation.resolved = false` selects the optimized total. With
`resolved = true`, all selected physical components are returned and their
explicit sum must agree with the total.

## Evaluators

Execution mode and evaluator backend are independent choices:

| Execution mode | Process artifact |
| --- | --- |
| `compiled` | Default. Compiles process-wide stage evaluators during generation. |
| `eager` | Uses a prepared model's local kernels and writes compact DAG invocation tables. |

| Backend | Use |
| --- | --- |
| `jit` | Default direct SymJIT application, optimization level 3 |
| `asm` | Symbolica assembly evaluator |
| `cpp` | Generated/compiled C++ evaluator with `[evaluator.cpp]` options |

Eager mode normally requires a `.pyamplicol-model` bundle already prepared for
exactly one backend. The `built-in-sm` source is the exception: installed
wheels carry `x86_64` and `aarch64` JIT O3 packs and select the host architecture
automatically. Generation never compiles missing eager kernels. The prepared
backend and code-shaping optimization settings are authoritative; conflicting
requests are retained in the requested configuration, adjusted in the effective
configuration, and reported once. Pass an explicit prepared-model path to
select built-in C++ or ASM instead of the packaged JIT O3 pack.

`.pyAmplicol-model.json` model IR is architecture-independent. SymJIT
storage-v3 prepared packs are instead portable only within their architecture
class; same-architecture transfer across supported operating systems is tested.
An explicit `x86_64`/`aarch64` mismatch is rejected before DAG construction or
SymJIT loading. A future SymJIT storage ABI may widen this contract.

`evaluator.eager.point_tile_size` defaults to 1024 and is an upper bound. The
runtime reduces it as needed to keep reusable storage within
`evaluator.eager.workspace_mib`, which defaults to 256 MiB. Arbitrarily large
input batches are processed through those fixed-size tiles.

The default batch size is 128 and the default output chunk size is 512.
Optimization defaults are 10
Horner iterations, backend-selected common-pair iterations, 1000 Horner
variables, 5,000,000 common-pair cache entries, and pair distance 1000.
Stage-local parameters are mandatory and are not a user toggle.

## Python Resolution

```python
from pyamplicol.config import resolve_config

resolution = resolve_config(
    "generate_pp_zjj_from_ufo_sm.toml",
    dedicated={"generation.workers": 4},
    overrides=("generation.workers=2", "generation.workers=1"),
)
assert resolution.requested.generation.workers == 1
assert resolution.effective.model.source.endswith("models/json/sm/sm.json")
```

In-memory mappings accept a `base_dir`; otherwise their relative paths use the
current directory. A `--set` value follows TOML syntax, so strings containing
spaces must be quoted.
