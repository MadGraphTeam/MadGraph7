---
title: "Python API"
nav_order: 4
has_children: true
---
<!-- SPDX-License-Identifier: 0BSD -->
# Python API

The public Python interface offers typed services for model compilation,
process planning/generation, runtime evaluation, selectors, and profiling.
Importing `pyamplicol` is lightweight: Symbolica and model tooling are loaded
only when the requested operation needs them.

```python
import pyamplicol

print(pyamplicol.__version__)
```

## Public service map

| Service | Purpose |
| --- | --- |
| `ModelSource` | Resolve and compile built-in, JSON, UFO, compiled, or prepared models. |
| `Generator` | Plan or generate schema-v3 process artifacts. |
| `Runtime` | Load one concrete process and evaluate totals/resolved components. |
| `BenchmarkRunner` | Profile an artifact or loaded runtime. |
| `generate`, `load`, `benchmark` | One-shot convenience functions. |

The shipped scripts under
[`examples/python/`](https://github.com/mg5amcnlo/pyamplicol/tree/main/examples/python)
are complete runnable programs.

## Plan and generate with the built-in model

```python
from pyamplicol import GenerationConfig, Generator, ModelSource

generator = Generator(GenerationConfig(workers=2))
model = ModelSource.built_in_sm()

plan = generator.plan("d d~ > z g", model=model)
for process in plan.concrete_processes:
    print(process.name, process.expression)

result = generator.generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    model=model,
    mode="replace",
)
print(result.output)
```

`plan()` resolves the model/process/color/evaluator contract without writing a
process artifact. `generate()` returns an immutable `GenerationResult` with the
absolute output, stored process set, schema version, and file inventory.

The packaged scalar HEFT model uses the same public selector:

```python
from pyamplicol import Generator, ModelSource

heft = ModelSource.built_in_sm_heft()
Generator().plan("g g > H g g", model=heft)
```

Use an explicit `HIG = 1` coupling-order limit in a resolved run configuration
when generating the process; see [Models and Processes](models-and-processes.md#built-in-scalar-heft-model)
and the packaged `builtin_sm_heft.toml` card.

## Generate a named process set

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet

processes = ProcessSet(
    requests=(
        ProcessRequest.parse("u u~ > Z g", name="uubar_Zg"),
        ProcessRequest.parse("u u~ > Z g g", name="uubar_Zgg"),
    )
)

result = Generator().generate(
    processes,
    "artifacts/z_ladder",
    model=ModelSource.built_in_sm(),
    mode="replace",
)
```

Process names must be unique. For custom aliases and crossing metadata, use
`ProcessAlias`; runtime expression matching already handles unique
side-preserving permutations automatically.

## Compile and reuse an external model

```python
from pyamplicol import Generator, ModelSource

source = ModelSource.from_path(
    "models/json/sm/sm.json",
    restriction="default",
)
model = source.compile()

print(model.info.name)
print(model.capabilities.supported_color_accuracies)
print(model.supported)

Generator().plan("d d~ > z g", model=model)
```

For a trusted UFO directory:

```python
source = ModelSource.from_path(
    "/path/to/MyUFO",
    restriction="restrict_default.dat",
)
model = source.compile()
```

UFO modules execute Python while loading. See
[Models and Processes](models-and-processes.md) for trust, serialization, and
prepared bundles.

## Use a fully resolved run card

For complex configuration, resolve the same schema used by the CLI and pass the
result to `Generator`:

```python
from pyamplicol import Generator, ModelSource, ProcessRequest, ProcessSet
from pyamplicol.config import resolve_config

card = {
    "schema_version": 1,
    "action": "generate",
    "model": {"source": "models/json/sm/sm.json", "restriction": "default"},
    "process": {
        "entries": [{"expression": "p p > Z j j"}],
        "multiparticles": {"p": ["d", "d~", "g"], "j": ["d", "d~", "g"]},
        "flavor_scheme": 2,
        "max_quark_lines": 2,
    },
    "color": {"accuracy": "lc"},
    "generation": {"output": "artifacts/pp_zjj", "mode": "replace"},
    "evaluator": {"execution_mode": "compiled", "backend": "jit"},
}

resolution = resolve_config(card)
model = ModelSource.from_config(resolution.effective.model).compile()
processes = ProcessSet(
    tuple(
        ProcessRequest.parse(entry.expression, name=entry.name)
        for entry in resolution.effective.process.entries
    )
)
result = Generator(resolution).generate(
    processes,
    resolution.effective.generation.output,
    model=model,
    mode=resolution.effective.generation.mode,
)
```

The complete maintained implementation is
[`examples/python/typed_generation.py`](https://github.com/mg5amcnlo/pyamplicol/blob/main/examples/python/typed_generation.py).

## Load a runtime

```python
from pyamplicol import Runtime

runtime = Runtime.load(
    "artifacts/pp_zjj",
    process="d d~ > g z g",
    model_parameters={"aS": 0.117, "MZ": 91.1876},
)

print(runtime.artifact_id)
print(runtime.execution_mode)
print(runtime.physics.process_id)
print(runtime.physics.process)
print(runtime.representative_process_key)
print(runtime.external_permutation)
```

`process` may be a stable process ID, explicit alias ID, exact stored expression,
or unique permutation-equivalent expression within each side. The loaded
runtime exposes metadata in the requested public order.

## Evaluate totals and resolved components

```python
import json
import math
from pathlib import Path

points = json.loads(Path("data/pp_zjj_momenta.json").read_text())

totals = runtime.evaluate(points)
resolved = runtime.evaluate_resolved(points)

print(resolved.helicity_ids)
print(resolved.color_ids)
for optimized, explicit in zip(totals, resolved.total(), strict=True):
    assert math.isclose(optimized.real, explicit.real, rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(optimized.imag, explicit.imag, rel_tol=1e-12, abs_tol=1e-15)
```

Input shape is:

```text
(point, external particle, [E, px, py, pz])
```

At LC, resolved shape is `(point, helicity, color_flow)`. At NLC/full,
the color dimension has length one because color is contracted.

## Select helicities and color flows

Batch-global selectors accept stable IDs or the typed objects exposed by
`runtime.physics`:

```python
selected = runtime.evaluate(
    points,
    helicities=[runtime.physics.helicities[0]],
    color_flows=[runtime.physics.color_flows[0]],
)
```

For one selector per point:

```python
mixed = runtime.evaluate(
    points,
    helicity_by_point=[
        runtime.physics.helicity_ids[index % 2]
        for index in range(len(points))
    ],
    color_flow_by_point=[
        runtime.physics.color_flow_ids[(index // 2) % 2]
        for index in range(len(points))
    ],
)
```

Batch-global and per-point selectors are mutually exclusive on the same axis.
Per-point color selection is LC-only. Rusticol groups mixed selectors while
returning values in original point order.

See [Runtime and Selectors](runtime-and-selectors.md) for the complete selector
contract.

## Update model parameters

```python
runtime.set_model_parameters({"aS": 0.1165, "MZ": 91.1876})
runtime.set_model_parameter("MT", 172.5)
```

Updates are atomic and refresh dependent derived parameters. Unknown,
immutable, non-finite, or invalid values reject the entire update.

`Runtime.load(..., model_parameters=...)` applies a complete mapping before the
runtime is returned. Complex inputs may be represented as Python complex values.

## Precision

```python
f64 = runtime.evaluate(points, precision=16)
high_precision = runtime.evaluate(points, precision=80)
```

Precision 16 uses the native Rusticol runtime and does not import Symbolica.
Other positive precision requests use retained exact evaluator state when the
artifact supports it and load Symbolica lazily. Decimal input preserves the
supplied decimal digits; binary64 input cannot gain information merely by
requesting more arithmetic precision.

Generated C/C++/Fortran/Rust standalone APIs support f64 (`precision=16`) only.

## Profile from Python

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(
    target_runtime=1.0,
    batch_size=128,
    precision=16,
    warmup_runs=2,
    minimum_samples=5,
    color_flow_ids=("1",),
)

result = BenchmarkRunner(config).run(runtime, points=points)
print(result.wall_time_per_point)
print(result.evaluator_time_per_point)
print(result.uncertainty.standard_error)
print(result.uncertainty.relative_standard_error)
```

`BenchmarkResult` also contains calibrated repetitions, sample count,
environment metadata, native timing breakdown, stage attribution, and work
counters when the runtime exposes them.

The runnable reference is
[`examples/python/benchmark.py`](https://github.com/mg5amcnlo/pyamplicol/blob/main/examples/python/benchmark.py).

## Convenience functions

```python
from pyamplicol import benchmark, generate, load

generated = generate(
    "d d~ > z g",
    "artifacts/builtin_ddbar_to_zg",
    model=ModelSource.built_in_sm(),
    mode="replace",
)
runtime = load(generated.output)
profile = benchmark(runtime, points=points)
```

The class-based services are preferable when reusing configuration, progress
sinks, compiled models, or loaded runtimes.

## Errors

All public failures derive from `PyAmpliColError`:

```python
from pyamplicol import EvaluationError, GenerationError, PyAmpliColError

try:
    runtime = Runtime.load("artifacts/pp_zjj", process="missing")
except EvaluationError as error:
    print(f"could not load runtime: {error}")
```

Useful subclasses include configuration, model, generation, artifact,
compatibility, dependency, and evaluation errors. Catch the narrowest type you
can handle; catch `PyAmpliColError` at an application boundary.

## API design notes

- Public result and metadata objects are immutable dataclasses or typed facades.
- Artifact paths are normalized to absolute paths at service boundaries.
- `Generator.generate(..., mode="error")` refuses an existing destination.
- `Runtime` is bound to one selected process; load another instance for another
  process in the same multiprocess artifact.
- Process artifacts are trusted executable inputs. Normal loading validates the
  schema, path confinement, references, target compatibility, and runtime ABI.
- Use explicit payload validation only for an intentional whole-artifact
  corruption audit; it is not required before every load.

## Further reading

- [Configuration](configuration.md)
- [Generation Modes and Evaluators](generation-modes-and-evaluators.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Artifacts and Portability](artifacts-and-portability.md)
- [Native APIs](native-apis.md)
