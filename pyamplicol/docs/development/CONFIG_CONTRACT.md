---
title: "Configuration Contract"
nav_order: 2
parent: "Development Documentation"
---
<!-- SPDX-License-Identifier: 0BSD -->
# Configuration Contract

TOML schema version 1 and the Python configuration dataclasses share one field
registry. Unknown fields are errors. Paths are resolved relative to the card.

Precedence is defaults, card, dedicated CLI flags, repeatable `--set`
overrides in order, then recorded license/resource clamping.

## Top Level

- `schema_version: int = 1`
- `action: generate | evaluate | benchmark | inspect | model-inspect |
  model-compile | model-processes`

## Model

- `source: str = "built-in-sm"`
- `restriction: str | null`
- `simplify: bool = true`
- `cache: bool = true`
- `cache_dir: path | null`

## Process

- `entries: list[{ expression: str, name: str | null }]`
- `multiparticles: table[str, list[str]] = {}`
- `flavor_scheme: int = 5`
- `max_quark_lines: int | null`
- `coupling_order_policy: minimal | explicit = minimal`
- `max_coupling_orders: table[str, int] = {}`
- `max_color_sectors: int | null`
- `reference_color_order: list[int] = []`
- `selected_color_sector_ids: list[int] = []`
- `selected_source_helicities: table[int, int] = {}`

Model defaults and this explicit table are merged by name. Every model supplies
`all` for its declaration-ordered valid physical external states; an explicit
`all` entry overrides it. Broad products such as `p p > all all` may expand
combinatorially.

The explicit color-sector, reference-order, and source-helicity controls are
developer-facing generation constraints. Ordinary users should leave them at
their defaults and use the runtime selectors below.

## Color

- `accuracy: lc | nlc | full = lc`
- `lc_flow_layout: topology-replay | all-flow-union = topology-replay`

LC generation always includes complete physical flow coverage. Runtime flow
selectors are configured under `evaluation` or `benchmark`; internal sector,
topology, replay, and reference-order IDs are not configurable.

## Generation

- `output: path | null`
- `mode: error | append | replace = error`
- `workers: auto | int = auto`
- `emit_api_bundle: bool = true`

### Generation Validation

- `enabled: bool = true`
- `samples: int = 2`
- `seed: int = 12345`
- `relative_tolerance: float = 1e-12`
- `absolute_tolerance: float = 1e-300`
- `post_build_validation: bool = false`

### Relation Discovery

- `mode: off | diagnostic | certified-reuse = certified-reuse`
- `precision_digits: int >= 80 = 96`
- `probe_count: int >= 2 = 4`
- `verification_probe_count: int >= 2 = 4`
- `relative_tolerance: float >= 0 = 1e-70`
- `absolute_tolerance: float >= 0 = 1e-80`
- `seed: int >= 0 = 1348026701`

Candidate and independent verification probes are bounded and deterministic.
Artifact writing always validates the schema, declared payloads, references,
and digests. Optional post-build validation additionally re-opens the completed
artifact and compares native binary64 optimized and resolved evaluation. It is
off by default because that second runtime pass does not alter the artifact and
can be disproportionately expensive for large resolved axes; enable it with
`--post-build-validation` when an immediate runtime smoke is wanted. Its
configured samples are independent of the 96-digit relation-discovery
certification probes below.
Exact binary64 term-vector or `ExactComplexRational` schedule proofs remain the
preferred promotion path. When no exact structural proof exists,
`certified-reuse` may apply equal, opposite, or zero-current reuse only after
the independent current-value probes pass both configured tolerances and the
complete certification input and mapping are persisted for replay. One warning
is emitted per generated artifact when such proof-less mappings are applied.
Malformed, non-finite, unstable, or stale evidence fails closed.

The feature is enabled by default for LC, NLC, and full colour, for compiled,
eager, and recurrence generation using built-in or prepared external/UFO
models. The compact on-the-fly source projection does not run this configurable
relation-discovery pass. `mode = "off"`—or the public
`--no-numerical-current-reuse` flag—selects the unoptimized path without
changing numerical results. Direct vertex-kernel equivalence remains
model-certificate-owned.

## Evaluator

- `backend: jit | asm | cpp = jit`
- `execution_mode: compiled | eager | recurrence | on-the-fly = recurrence`
- `batch_size: int = 128`
- `output_chunk_size: int | null = 512`
- Stage-local parameter layout is mandatory and is not a public toggle.

### Eager Execution

- `point_tile_size: int = 1024`
- `workspace_mib: int = 256`

Eager mode requires a prepared model bundle before DAG construction. A
`built-in-sm` source resolves automatically to the wheel-owned portable
`built-in-sm-jit-o2` pack; other models and built-in C++/ASM execution require
an explicit prepared path. The prepared pack is
authoritative for backend and code-shaping optimization settings. The runtime
may reduce `point_tile_size` to honor the workspace limit, but never increases
it.

`.pyAmplicol-model.json` IR is architecture-independent. SymJIT application
storage-v3 prepared packs use optimization level 2 and are portable across the
supported `x86_64` and `aarch64` targets. Loading rebuilds executable code for
the receiving CPU. Prepared-pack compilation forces O2 and records any
requested-to-effective adjustment. C++ and ASM prepared packs remain
target-native.

### Recurrence Execution

- `point_tile_size: int = 1024`
- `workspace_mib: int = 256`

Recurrence is the global default. The runtime may reduce `point_tile_size` to
honor the recurrence workspace limit, but never increases it. Recurrence JIT
kernels use the same portable prepared O2 contract as eager execution. A
missing prepared pack fails closed; configuration resolution never falls back
to compiled execution. Cards that require process-local compiled DAGs must set
`execution_mode = "compiled"` explicitly.

### On-The-Fly Execution

On-the-fly execution requires a prepared model bundle and native `f64`
precision. It stores a compact process seed and constructs recurrence schedules
for the requested selector instead of materializing a process layout. LC does
not materialize topology replay or all-flow union: static inspection exposes
the physical helicity/color-flow census and the runtime supplies both selector
families. NLC and full colour expose one contracted component, accept helicity
selectors, and reject LC-flow selectors.

`evaluator.optimization.cores` resolves to a positive requested
query-construction thread count for this mode. It is not a numerical-runtime
thread guarantee. LC numerical execution sizes its warmed family to the
requested batch capacity. Contracted NLC/full execution applies the
authenticated recurrence point tile. There is no on-the-fly-specific tile or
workspace configuration section.

### Evaluator Optimization

- `horner_iterations: int = 10`
- `cpe_iterations: int | null = null`
- `cores: auto | int = auto`
- `max_horner_variables: int = 1000`
- `max_common_pair_cache_entries: int = 5000000`
- `max_common_pair_distance: int = 1000`
- `collect_factors: auto | bool = auto`

For on-the-fly execution, only `cores` has a mode-specific runtime meaning;
the prepared kernel pack remains authoritative for code-shaping optimization.

### JIT

- `optimization_level: 0 | 1 | 2 | 3 = 2`
- `compress: bool = true`

JIT artifacts embed direct SymJIT applications. The defaults above apply to
process-local compiled DAG evaluators. Prepared JIT kernel packs used by eager,
recurrence, and on-the-fly execution force optimization level 2 to preserve
their cross-architecture storage contract.

### C++

- `optimization: str = "O3"`
- `compiler: str | null`
- `native_arch: bool = false`
- `extra_flags: list[str] = []`

Portable C++ generation is the default. Setting `native_arch = true` opts into
host-native code and records Rusticol's canonical, sorted runtime CPU-feature set
as an artifact requirement. Loaders reject that artifact before reading evaluator
state on a target without every declared feature. Additional flags are restricted
to the documented non-ISA allowlist; arbitrary `-march`, `-mcpu`, `-m*`, or target
flags are rejected because they could introduce unrecorded requirements.
Schema-v3 evaluator payload portability is currently defined for macOS arm64,
macOS x86_64, and glibc Linux x86_64. Other targets are rejected explicitly.

## Evaluation

- `artifact: path | null`
- `process: str | null`
- `precision: int = 16`
- `resolved: bool = false`
- `helicity_ids: list[str] = []`
- `color_flow_ids: list[str] = []`
- `model_parameters: path | null`
- `momenta: path | null`

## Benchmark

- `target_runtime: float = 10.0`
- `batch_size: int = 128`
- `precision: int = 16`
- `warmup_runs: int = 2`
- `minimum_samples: int = 5`
- `helicity_ids: list[str] = []`
- `color_flow_ids: list[str] = []`

For LC profiling, empty benchmark selector lists resolve at runtime to the
generated layout's optimized workload: one physical flow and the helicity sum
for `topology-replay`, or all physical flows and one computed helicity for
`all-flow-union`. Explicit subsets and selected-axis lists are preserved; a
complete summed-axis list is normalized to equivalent omission. A valid
non-hot shape emits at most one pre-loop warning per loaded process. Evaluation
selector defaults remain the complete matrix element.

## Output And Symbolica

- `output.format: human | json = human` for typed API/config provenance. The
  public CLI normalizes this to human tables unless `--json` is explicitly
  present.
- `output.color: auto | always | never = auto`
- `output.progress: auto | tty | log | off = auto`
- `output.log_level: debug | info | warning | error = info`
- `symbolica.suggest_license: bool = true`
