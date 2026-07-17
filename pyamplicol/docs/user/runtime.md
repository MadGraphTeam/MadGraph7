<!-- SPDX-License-Identifier: 0BSD -->

# Runtime

`Runtime` loads a schema-v3 process artifact and exposes stable process,
particle, helicity, color, reduction, and model-parameter metadata. The native
extension is imported only when an artifact is loaded.

## Inspect An Artifact

List every generated process and stable ID without loading executable evaluator
state:

```console
pyamplicol inspect artifacts/pp_zjj
```

The terminal view uses aligned colored tables for artifact/model/runtime
metadata, concrete processes, crossing aliases, helicity and color coverage,
payload size, and dependencies. The same complete inventory is available as
JSON:

```console
pyamplicol inspect artifacts/pp_zjj --format json
```

To inspect the detailed resolved-physics metadata for one process, select it by
expression or stable ID:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > z g g'
pyamplicol inspect artifacts/pp_zjj --process p_p_to_z_j_j_4
```

## Load A Concrete Process

Multiprocess artifacts accept either a readable concrete expression or a
stable process/alias ID:

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > z g g")
print(runtime.physics.process)  # d d~ > Z g g
print(runtime.physics.external_particles)
print(runtime.physics.helicity_ids)
print(runtime.physics.color_flow_ids)
```

The equivalent stable selector is `process="p_p_to_z_j_j_4"`. Expression
matching normalizes whitespace but preserves concrete particle names and
ordering.

Artifact schema, payload paths and hashes, target compatibility, and runtime ABI
are validated before executable state is loaded.

## Model Parameters

The artifact records mutable UFO external parameters and their defaults:

```python
for parameter in runtime.physics.model_parameters:
    if parameter.mutable:
        print(parameter.name, parameter.default_real)
```

Apply a complete JSON object while loading:

```python
runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > z g g",
    model_parameters={"aS": 0.117, "MZ": 91.188},
)
```

Or update genuine UFO inputs atomically after loading:

```python
runtime.set_model_parameters({"aS": 0.1165, "MZ": 91.1876})
runtime.set_model_parameter("MT", 172.5)
```

Derived couplings and dependent parameters are refreshed before the update is
committed. An unknown, immutable, non-finite, or otherwise invalid entry rejects
the full batch; no prefix is applied. The CLI and all generated API drivers use
the same contract:

```console
pyamplicol evaluate artifacts/pp_zjj \
  --process 'd d~ > z g g' \
  --model-parameters data/model_parameters.json \
  --momenta data/pp_zjj_momenta.json
```

## Total And Resolved Evaluation

Momenta have shape `(point, external particle, [E, px, py, pz])`:

```python
import json
from pathlib import Path

momenta = json.loads(Path("data/pp_zjj_momenta.json").read_text())
total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)

for summed, explicit in zip(total, resolved.total(), strict=True):
    scale = max(1.0, abs(summed))
    assert abs(summed - explicit) <= 1.0e-12 * scale
```

At LC, resolved values have shape `(point, helicity, physical color flow)`. At
NLC/full, the final dimension has length one because color is contracted.

Selectors use IDs reported by `runtime.physics`:

```python
selected = runtime.evaluate(
    momenta,
    helicities=[runtime.physics.helicity_ids[0]],
    color_flows=[runtime.physics.color_flow_ids[0]],
)
```

Color-flow selection is available only for LC artifacts. NLC/full accept
helicity selectors and reject color-flow selectors.

## Precision And Capabilities

Precision 16 uses Rusticol's Symbolica-independent f64 runtime. For the default
JIT backend, Rusticol loads the embedded direct SymJIT application through the
separate MIT-licensed SymJIT runtime. ASM and C++ backends instead load their
compiled evaluator library when the artifact target triple matches the runtime
and every recorded CPU feature is available. Neither f64 path imports
Symbolica, reads its license state, or applies its generation-time resource
clamp. Python, Rust, C++, and Fortran share these f64 capabilities.

Other positive Python precision requests lazily load retained Symbolica
evaluator states and replay the recorded stage-local plan. Decimal input keeps
its supplied digits. Values originating as binary64 are extended with trailing
zeros; requesting more arithmetic digits does not reconstruct input information
or certify that many physically accurate digits. Results are rounded to the
requested decimal precision after guard-digit evaluation.

Direct SymJIT applications are lowered to native code when loaded. ASM/C++
artifacts contain target-specific native libraries and fail compatibility
validation before executable state is loaded when the target or required CPU
features do not match. Rust, C++, and Fortran expose f64 only and reject
precision requests other than 16.

## Runtime Profiling

The intended CLI command profiles a selected process through the same optimized
total path used by `Runtime.evaluate()`:

```console
pyamplicol profile artifacts/pp_zjj \
  --process p_p_to_z_j_j_4 \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0 \
  --batch-size 128 \
  --precision 16
```

`--process` accepts a stable process or alias ID, or the exact concrete
expression such as `--process 'd d~ > z g g'`. If `--momenta` is omitted, the
artifact's deterministic validation point is repeated to the requested runtime
batch size.

The profiler performs configured warmups and uses their timing, or a dedicated
probe when warmups are disabled, to calibrate both the number of independent
timed blocks and the repetitions in each block. Fast runtimes use more
repetitions per block; slow runtimes reduce the block count without going below
`--minimum-samples`. The target duration applies to wall-time measurements.
Native component profiling is measured separately when the runtime supports it,
with exactly one native profile per statistical block. It never inherits the
potentially large wall-time repetition count.

The human result is a colorized PrettyTable showing the selected process, wall
and evaluator means with standard errors, wall standard deviation and relative
standard error, calibrated sampling geometry, target versus measured time, and
timing provenance. TTY progress uses a colored thread-safe progress bar.
Non-TTY progress uses typed, rate-limited log messages on stderr. With
`--format json`, stdout contains only the machine-readable result.

When native profiling is available, additional Rusticol tables report profile
wall time, source fill, momentum setup, stage input packing/evaluator calls and
output assignment, amplitude input packing/evaluator calls, reduction, and
per-stage packing/evaluator/output timings. `BenchmarkResult.timing_breakdown`
preserves the same data as typed component and stage timing objects, including
sample counts and uncertainty.

The same operation is available through the typed Python API:

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner, Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="p_p_to_z_j_j_4")
runner = BenchmarkRunner(
    BenchmarkConfig(target_runtime=1.0, batch_size=128, precision=16)
)
result = runner.run(runtime, points=momenta)
print(
    result.wall_time_per_point,
    result.uncertainty.standard_error,
    result.repetitions_per_sample,
)
```

Python benchmarks may request higher precision. Double-precision input points
are converted through their exact decimal spelling and padded with trailing
zeros before evaluation at the requested precision. The native evaluator
profiler is f64-only, so non-f64 benchmarks report wall time for both timing
fields.

`pyamplicol benchmark` is retained as a compatibility alias for `profile` with
the same flags and output. Existing run cards remain valid and continue to use
`action = "benchmark"`; no `action = "profile"` card value is introduced.

## Artifact Trust

A process artifact is executable input. Direct SymJIT applications are lowered
to native code at load time, and compiled backends may contain native
libraries. Manifest hashes and path confinement establish internal consistency,
not publisher identity. Generate artifacts yourself or obtain them through a
trusted channel.

Rusticol uses SymJIT's own trusted-input `Application::load(...).seal()` path
with an empty external-function map. It does not maintain a second application
decoder.

## Concurrency And Warnings

Parameter and warning state belongs to one runtime handle. Do not call the same
mutable handle concurrently. Independent handles may run in separate threads.
Symbolica's restricted-generation clamp does not limit independent handles
executing direct SymJIT or target-compatible ASM/C++ f64 artifacts.
