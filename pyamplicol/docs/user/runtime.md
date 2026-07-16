<!-- SPDX-License-Identifier: 0BSD -->

# Runtime

The public `Runtime` facade loads a schema-v3 process artifact and exposes
stable physical metadata and evaluation methods. Its lazy adapter imports the
wheel's `pyamplicol._rusticol` extension only when an artifact is loaded.
Schema-v3 generation emits the runtime payloads and root API examples described
below.

## Load And Inspect

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/mixed", process="ddbar_zg")
print(runtime.physics.external_particles)
print(runtime.physics.helicity_ids)
print(runtime.physics.color_flow_ids)
```

For a process-set artifact, `process` selects a stable process name or alias.
Artifact schema, payload hashes, paths, target, and ABI are validated before
executable state is loaded.

## Total And Resolved Evaluation

Momenta have shape `(point, external particle, [E, px, py, pz])`:

```python
momenta = [[
    [500.0, 0.0, 0.0, 500.0],
    [500.0, 0.0, 0.0, -500.0],
    [504.157625672, -304.1084262865, 208.7602652353, 331.3561179451],
    [495.842374328, 304.1084262865, -208.7602652353, -331.3561179451],
]]

total = runtime.evaluate(momenta)
resolved = runtime.evaluate_resolved(momenta)
assert resolved.total() == total
```

At LC, resolved values have shape `(point, helicity, physical color flow)`. At
NLC/full, the last dimension has length one because color is contracted.

Selectors use IDs reported by `runtime.physics`:

```python
selected = runtime.evaluate(
    momenta,
    helicities=[runtime.physics.helicity_ids[0]],
    color_flows=[runtime.physics.color_flow_ids[0]],
)
```

Color-flow selection is available only for LC artifacts. NLC/full artifacts
advertise helicity selection alone; their singleton contracted-color axis is an
output descriptor, and passing `color_flows` is an error. Precision 16 uses the
native Rusticol path. Other Python precision requests lazily load the retained
evaluator states with Symbolica's own `Evaluator.load()` API and return decimal
results. Decimal kinematics retain their supplied digits. Binary64-origin
kinematics and runtime model values are padded with trailing zeros to the
requested precision; this stabilizes the higher-precision replay without
claiming to reconstruct information absent from the input. Stage inputs are
re-padded after every evaluator boundary. Evaluation uses guard digits and
round-to-nearest-even before publishing the requested number of decimal digits;
that number controls arithmetic precision and is not a claim of certified
physical accuracy. Resolved Decimal components are summed exactly rather than
through Python's ambient Decimal context. The C++ and Fortran APIs remain
f64-only.

Every artifact records the evaluator capabilities required to load it. The
default JIT backend stores a self-contained SymJIT application under
`symjit.application.complex-f64.v1`; Rusticol executes that f64 payload without
importing Symbolica or consulting a Symbolica license. This makes a generated
JIT process independently deployable and permits normal multi-handle runtime
concurrency even when generation ran in restricted mode.

That guarantee is backend- and precision-specific. Arbitrary-precision Python
evaluation uses the retained Symbolica evaluator state and replays the recorded
stage-local execution plan. ASM and C++ artifacts also advertise
Symbolica-backed runtime capabilities; the lightweight native SDK rejects them
before loading and reports that a Symbolica-capable runtime is required.

## Artifact Trust

A process artifact is executable input: direct SymJIT payloads are translated
to native code at load time, and compiled backends may contain native
libraries. Payload hashes and path checks establish internal consistency, not
publisher authenticity. Load only artifacts whose producer and transport you
trust. Corrupted or untrusted artifacts should be discarded rather than used
as a parser-security boundary.

Rusticol deliberately follows Symbolica's trusted-input implementation here:
it supplies the empty external-function map, calls SymJIT's own
`Application::load`, and seals the returned application. It does not maintain
a second application decoder.

## Model Parameters

Updates validate as one transaction before committing:

```python
runtime.set_model_parameters(
    {
        "normalization.alpha_s_me_check": 0.118,
        "normalization.alpha_ew": 1.0 / 132.507,
    }
)
runtime.set_model_parameter("normalization.alpha_s_me_check", 0.120)
```

The CLI accepts a JSON object with `--model-parameters`. Invalid or unknown
entries reject the update rather than applying a prefix.

## Benchmarking

```python
from pyamplicol import BenchmarkConfig, BenchmarkRunner

runner = BenchmarkRunner(BenchmarkConfig(target_runtime=1.0, batch_size=128))
result = runner.run("artifacts/ddbar_zg", points=momenta)
print(result.wall_time_per_point, result.uncertainty)
```

The direct equivalent is:

```console
pyamplicol benchmark artifacts/ddbar_zg \
  --momenta examples/data/ddbar_zg_momenta.json \
  --target-runtime 1.0
```

The benchmark uses the same runtime adapter and optimized summed evaluation
path as `Runtime.evaluate()`.

## Concurrency And Warnings

Model parameter and warning state belongs to one runtime handle. A mutable
handle must not be called concurrently. Independent handles may be evaluated
from separate threads. Symbolica's restricted-mode generation clamp does not
apply to independent Rusticol runtime handles evaluating a direct SymJIT f64
payload.
