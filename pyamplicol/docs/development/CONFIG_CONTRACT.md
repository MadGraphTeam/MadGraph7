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

## Color

- `accuracy: lc | nlc | full = lc`

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
- `samples: int = 10`
- `seed: int = 12345`
- `relative_tolerance: float = 1e-12`
- `absolute_tolerance: float = 1e-300`
- `post_build_validation: bool = true`

Generation applies only proven structural helicity reduction. Schema-v3 does
not expose numerical current pruning or merging because model parameters remain
runtime-mutable.

## Evaluator

- `backend: jit | asm | cpp = jit`
- `execution_mode: compiled | eager = compiled`
- `batch_size: int = 128`
- `output_chunk_size: int | null = 512`
- Stage-local parameter layout is mandatory and is not a public toggle.

### Eager Execution

- `point_tile_size: int = 1024`
- `workspace_mib: int = 256`

Eager mode requires a prepared model bundle before DAG construction. A
`built-in-sm` source resolves automatically to the wheel-owned JIT O3 pack for
the host's `x86_64` or `aarch64` architecture class; other models and built-in
C++/ASM execution require an explicit prepared path. The prepared pack is
authoritative for backend and code-shaping optimization settings. The runtime
may reduce `point_tile_size` to honor the workspace limit, but never increases
it.

`.pyAmplicol-model.json` IR is architecture-independent. SymJIT application
storage-v3 prepared packs are architecture-class-specific, although
same-architecture transfer across supported operating systems is tested. Pack
target validation precedes DAG construction and SymJIT loading, so a cross-
architecture mismatch cannot reach dependency code. A future SymJIT storage ABI
may relax this restriction without changing the prepared-bundle interface.

### Evaluator Optimization

- `horner_iterations: int = 10`
- `cpe_iterations: int | null = null`
- `cores: auto | int = auto`
- `max_horner_variables: int = 1000`
- `max_common_pair_cache_entries: int = 5000000`
- `max_common_pair_distance: int = 1000`
- `collect_factors: auto | bool = auto`

### JIT

- `optimization_level: 0 | 1 | 2 | 3 = 3`

JIT artifacts always use indirect SymJIT translation because direct
translation is not a stable serialized-application ABI.

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
- `warmup_runs: int = 2`
- `minimum_samples: int = 5`
- `helicity_ids: list[str] = []`
- `color_flow_ids: list[str] = []`

## Output And Symbolica

- `output.format: human | json = human`
- `output.color: auto | always | never = auto`
- `output.progress: auto | tty | log | off = auto`
- `output.log_level: debug | info | warning | error = info`
- `symbolica.suggest_license: bool = true`
