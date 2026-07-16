# Public API Contract

This document is normative for Python API version 1. Implementations may add
private helpers but may not widen the package-root export list without an API
review.

## Package-Root Exports

`pyamplicol` exports:

- Model and process types: `ModelSource`, `CompiledModel`, `ProcessRequest`,
  and `ProcessSet`.
- Configuration: `RunConfig`, `GenerationConfig`, `EvaluationConfig`, and
  `BenchmarkConfig`.
- Services: `Generator`, `Runtime`, and `BenchmarkRunner`.
- Results: `GenerationPlan`, `GenerationResult`, `BenchmarkResult`,
  `BenchmarkStatistics`, `ProcessPhysics`, `ExternalParticle`,
  `HelicityConfiguration`, `ColorComponent`, `ColorFlow`,
  `ContractedColorComponent`, `PhysicsReduction`, `ReductionGroup`,
  `ModelParameter`, and `ResolvedEvaluation`.
- Process metadata: `ProcessAlias`.
- Helpers: `generate`, `load`, and `benchmark`.
- Errors: `PyAmpliColError`, `ConfigurationError`, `ModelError`,
  `GenerationError`, `ArtifactError`, `CompatibilityError`,
  `EvaluationError`, and `DependencyError`.

No DAG, evaluator, stage, color-sector, Symbolica, or legacy-reference class is
exported from the package root.

## Model And Process Requests

`ModelSource.built_in_sm()` identifies the built-in Standard Model.
`ModelSource.from_path(path, restriction=None, simplify=True)` accepts a UFO
directory, serialized JSON, or compiled-model file and records a resolved
source kind without importing Symbolica.

`ModelSource.compile(*, cache_dir=None, use_cache=True,
require_supported=True) -> CompiledModel` is the explicit public model
compilation operation. `CompiledModel` is the canonical compiled payload used
by model loading and generation; there is no separate path-only public model
handle. It exposes the compiled IR and metadata, can be serialized with
`write(path)`, and is also returned when a compiled-model source is loaded.

`ProcessRequest.parse(expression, *, name=None)` validates one process string.
`ProcessSet` contains a non-empty tuple of uniquely named requests and
explicit aliases. Public aliases use physical process metadata; internal
representatives remain hidden.

## Generation

`Generator(config=None, progress=None)` accepts immutable typed configuration
or a `ConfigResolution`, plus an optional typed progress sink. Passing the
resolution preserves requested and effective TOML plus every clamp reason in
the generated artifact.

`Generator.plan(processes, *, model=None) -> GenerationPlan` is strictly
non-writing: it creates no model-cache or output files, directories, locks, or
temporary trees. It expands concrete processes and validates structural color
coverage without compiling a model, process DAG, evaluator, or artifact. A
built-in model, `CompiledModel`, or compiled-model file can always be planned.
For a UFO/JSON `ModelSource`, planning may read a valid existing model-cache
entry but never populates one; on a cache miss callers must first invoke
`ModelSource.compile()` and pass the returned `CompiledModel`.

Planning performs the same effective Symbolica license and resource resolution
as generation. Restricted mode clamps generation workers and Symbolica cores
to one. Licensed mode partitions the affinity-aware CPU budget after concrete
process expansion. `GenerationPlan.requested_settings` and
`GenerationPlan.effective_settings` retain both typed configurations, while
`GenerationPlan.adjustments` contains typed `ConfigClamp` values with the
requested value, effective value, path, and reason. `generate --dry-run` uses
this operation and has identical non-writing behavior.

`Generator.generate(processes, output, *, model=None, mode="error")
-> GenerationResult` writes a transactional schema-v3 artifact. `mode` is
`error`, `append`, or `replace`. Both `model` arguments accept a `ModelSource`,
the canonical `CompiledModel`, or `None` for the configured/default source.

`generate(...)` is a convenience wrapper with the same generation semantics.

## Runtime

`Runtime.load(artifact, *, process=None, model_parameters=None,
mute_warnings=False) -> Runtime` validates schema, checksums, target, and ABI
before loading executable state.

`Runtime.physics` returns `ProcessPhysics` with stable particles, physical
helicity IDs, physical color-flow IDs, contraction metadata, coverage, and
selector capabilities. LC advertises `("helicity", "color_flow")`; NLC/full
advertise only `("helicity",)`. Their singleton contracted-color output axis is
metadata, not a selectable color flow.

`Runtime.evaluate(momenta, *, helicities=None, color_flows=None)` returns one
fully summed value per point. Momenta have shape
`(point, particle, [E, px, py, pz])`.

`Runtime.evaluate_resolved(...) -> ResolvedEvaluation` returns LC values with
shape `(point, physical_helicity, physical_color_flow)` and NLC/full values
with shape `(point, physical_helicity, 1)`. `ResolvedEvaluation.total()` must
reproduce `evaluate()`.

`Runtime.set_model_parameters(mapping)` validates the complete update before
committing it. `set_model_parameter(name, value)` is a convenience wrapper.
`mute_warnings()` and `unmute_warnings()` modify only that runtime handle.

`load(...)` is an alias for `Runtime.load(...)`.

## Benchmarking

`BenchmarkRunner(config=None, progress=None).run(target, *, points=None)
-> BenchmarkResult` accepts a runtime or artifact path. Results contain
requested/effective configuration, sample count, wall time per point, pure
evaluator time where available, uncertainty statistics, and environment
provenance.

`benchmark(...)` is the convenience wrapper.

## Stability Rules

All public dataclasses are frozen. Public collections are tuples or immutable
mappings. Paths are accepted as `os.PathLike[str]` and returned as absolute
`Path` objects only when they describe user-created files.

Machine-readable methods return typed objects or documented JSON, never
volatile dictionaries. JSON output writes only to stdout; diagnostics use
logging/stderr.

## Typing Gate

`just typing` is the normative public Python typing gate and is also a
dependency of `just check`. It runs strict mypy over an explicit source list:
the package root, `pyamplicol.api`, the public configuration models/registry
and `pyamplicol.config` facade, `pyamplicol.reporting`, and
`pyamplicol.runtime`. Imported implementation modules are followed for type
information with their diagnostics silenced; no public target uses
`ignore_errors`.

A second mypy pass stages a temporary installed-style package, including
`py.typed` and the maintained `_rusticol.pyi`, then checks external consumers.
Those consumers assert exact Generator, Runtime, configuration, protocol, and
result types. Every name in the `__all__` lists of `pyamplicol` and its public
`api`, `config`, `reporting`, and `runtime` facades is also rejected if it
degrades to `Any`. The native consumer checks the maintained extension stub,
and the metadata check compares that stub with the Rust binding export list
without rebuilding Rust.

This is intentionally not a project-wide strictness claim. Dynamic generation,
model compilation, evaluator, color, artifact, CLI, and other private
implementation modules are outside the source target. Bundled vendor UFO
assets under `src/pyamplicol/assets/models/ufo/` and generated standalone API
templates under `src/pyamplicol/assets/api_templates/` are explicitly
excluded. Their exclusion does not weaken checks of the documented public
signatures that expose their results.
