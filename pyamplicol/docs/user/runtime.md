<!-- SPDX-License-Identifier: 0BSD -->

# Runtime

`Runtime` loads a schema-v3 process artifact and exposes stable process,
particle, helicity, color, reduction, and model-parameter metadata. The native
extension is imported only when an artifact is loaded.

## Load A Concrete Process

Multiprocess artifacts require a stable process name or alias:

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="p_p_to_z_j_j_4")
print(runtime.physics.process)  # d d~ > Z g g
print(runtime.physics.external_particles)
print(runtime.physics.helicity_ids)
print(runtime.physics.color_flow_ids)
```

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
    process="p_p_to_z_j_j_4",
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
  --process p_p_to_z_j_j_4 \
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

Precision 16 executes the direct SymJIT f64 application through Rusticol. It
does not import Symbolica or consult Symbolica's runtime license state. This
Symbolica-independent path is shared by Python, Rust, C++, and Fortran and uses
the separate MIT-licensed SymJIT runtime.

Other positive Python precision requests lazily load retained Symbolica
evaluator states and replay the recorded stage-local plan. Decimal input keeps
its supplied digits. Values originating as binary64 are extended with trailing
zeros; requesting more arithmetic digits does not reconstruct input information
or certify that many physically accurate digits. Results are rounded to the
requested decimal precision after guard-digit evaluation.

ASM and C++ evaluator artifacts advertise a Symbolica-backed runtime
capability. The lightweight native SDK rejects unsupported capabilities before
partial loading. Rust, C++, and Fortran expose f64 only and reject precision
requests other than 16.

## Benchmarking

Benchmark a selected runtime through the same optimized total path:

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner, Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="p_p_to_z_j_j_4")
runner = BenchmarkRunner(BenchmarkConfig(target_runtime=1.0, batch_size=128))
result = runner.run(runtime, points=momenta)
print(result.wall_time_per_point, result.uncertainty)
```

The direct equivalent is:

```console
pyamplicol benchmark artifacts/pp_zjj \
  --set evaluation.process=p_p_to_z_j_j_4 \
  --momenta data/pp_zjj_momenta.json \
  --target-runtime 1.0
```

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
executing direct JIT f64 artifacts.
