# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pyamplicol.api.errors import ConfigurationError
from pyamplicol.config import ConfigResolution, RunConfig
from pyamplicol.reporting import ProgressSink

if TYPE_CHECKING:
    from pyamplicol.api import CompiledModel, ModelSource, ProcessSet


class CliServices(Protocol):
    """Command-level service boundary used by CLI handlers."""

    def generate(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def plan(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def inspect(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def model_inspect(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def model_compile(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def model_processes(self, config: RunConfig, progress: ProgressSink) -> object: ...


def dispatch(
    config: RunConfig,
    services: CliServices,
    progress: ProgressSink,
    *,
    dry_run: bool = False,
) -> object:
    """Dispatch a typed config without importing a concrete domain backend."""

    if config.action == "generate":
        return (
            services.plan(config, progress)
            if dry_run
            else services.generate(config, progress)
        )
    if config.action == "evaluate":
        return services.evaluate(config, progress)
    if config.action == "benchmark":
        return services.benchmark(config, progress)
    if config.action == "inspect":
        return services.inspect(config, progress)
    if config.action == "model-inspect":
        return services.model_inspect(config, progress)
    if config.action == "model-compile":
        return services.model_compile(config, progress)
    if config.action == "model-processes":
        return services.model_processes(config, progress)
    raise ConfigurationError(f"unsupported action {config.action!r}")


def _model_source(config: RunConfig) -> ModelSource:
    from pyamplicol.api import ModelSource

    if config.model.source == "built-in-sm":
        return ModelSource.built_in_sm()
    return ModelSource.from_path(
        config.model.source,
        restriction=config.model.restriction,
        simplify=config.model.simplify,
    )


def _process_set(config: RunConfig) -> ProcessSet:
    from pyamplicol.api import ProcessRequest, ProcessSet

    requests: list[ProcessRequest] = []
    errors: list[str] = []
    for index, entry in enumerate(config.process.entries):
        try:
            requests.append(ProcessRequest.parse(entry.expression, name=entry.name))
        except (TypeError, ValueError) as exc:
            errors.append(f"process.entries[{index}]: {exc}")
    if errors:
        raise ConfigurationError(errors)
    try:
        return ProcessSet(tuple(requests))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}") from exc


def _compile_configured_model(
    config: RunConfig,
    *,
    require_supported: bool,
) -> CompiledModel:
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        config.model.source,
        restriction=config.model.restriction or "default",
        simplify=config.model.simplify,
        cache_dir=config.model.cache_dir,
        use_cache=config.model.cache,
        require_supported=require_supported,
    )


def _model_inspection_payload(compiled: CompiledModel) -> dict[str, object]:
    return {
        "name": compiled.name,
        "supported": compiled.supported,
        "source": dict(compiled.source),
        "producer": dict(compiled.producer),
        "capabilities": dict(compiled.capabilities),
        "issues": [issue.to_dict() for issue in compiled.issues],
        "conversion_seconds": compiled.conversion_seconds,
        "phase_timings": dict(compiled.phase_timings),
        "contents": {
            "orders": [order.name for order in compiled.ir.orders],
            "particles": [particle.to_dict() for particle in compiled.ir.particles],
            "runtime_parameter_names": sorted(compiled.parameter_defaults),
        },
    }


class DefaultCliServices:
    """Thin adapters over the public API; backend imports remain first-use only."""

    def __init__(self, *, resolution: ConfigResolution | None = None) -> None:
        self._resolution = resolution

    def _generation_config(self, config: RunConfig) -> RunConfig | ConfigResolution:
        if self._resolution is None:
            return config
        if self._resolution.effective != config:
            raise ConfigurationError(
                "CLI generation config does not match its resolved provenance"
            )
        return self._resolution

    def generate(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import Generator

        if not config.process.entries:
            raise ConfigurationError("generate requires at least one process request")
        if config.generation.output is None:
            raise ConfigurationError("generate requires generation.output")
        processes = _process_set(config)
        return Generator(
            config=self._generation_config(config), progress=progress
        ).generate(
            processes,
            config.generation.output,
            model=_model_source(config),
            mode=cast(
                Literal["error", "append", "replace"],
                str(config.generation.mode),
            ),
        )

    def plan(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import Generator

        if not config.process.entries:
            raise ConfigurationError("generate --dry-run requires a process request")
        processes = _process_set(config)
        return Generator(
            config=self._generation_config(config), progress=progress
        ).plan(
            processes,
            model=_model_source(config),
        )

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        from pyamplicol.api import Runtime

        if config.evaluation.artifact is None:
            raise ConfigurationError("evaluate requires evaluation.artifact")
        if config.evaluation.momenta is None:
            raise ConfigurationError("evaluate requires evaluation.momenta")
        parameters: Mapping[str, complex | float | int] | None = None
        if config.evaluation.model_parameters is not None:
            raw_parameters = _read_json(config.evaluation.model_parameters)
            if not isinstance(raw_parameters, Mapping):
                raise ConfigurationError("model parameter JSON must be an object")
            parameters = raw_parameters
        momenta = _read_json(config.evaluation.momenta)
        if not isinstance(momenta, list):
            raise ConfigurationError("momenta JSON must be a point list")
        runtime = Runtime.load(
            config.evaluation.artifact,
            process=config.evaluation.process,
            model_parameters=parameters,
        )
        selectors = {
            "helicities": config.evaluation.helicity_ids or None,
            "color_flows": config.evaluation.color_flow_ids or None,
        }
        if config.evaluation.resolved:
            return runtime.evaluate_resolved(
                momenta,
                precision=config.evaluation.precision,
                **selectors,
            )
        return runtime.evaluate(
            momenta,
            precision=config.evaluation.precision,
            **selectors,
        )

    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import BenchmarkRunner

        if config.evaluation.artifact is None:
            raise ConfigurationError("benchmark requires an artifact target")
        points = None
        if config.evaluation.momenta is not None:
            raw_points = _read_json(config.evaluation.momenta)
            if not isinstance(raw_points, list):
                raise ConfigurationError("momenta JSON must be a point list")
            points = raw_points
        return BenchmarkRunner(config=config, progress=progress).run(
            config.evaluation.artifact,
            points=points,
        )

    def inspect(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if config.evaluation.artifact is None:
            raise ConfigurationError("inspect requires evaluation.artifact")
        if config.evaluation.process is None:
            from pyamplicol.artifacts import inspect_artifact

            return inspect_artifact(config.evaluation.artifact)

        from pyamplicol.api import Runtime

        return Runtime.load(
            config.evaluation.artifact,
            process=config.evaluation.process,
        ).physics

    def model_inspect(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        return _model_inspection_payload(
            _compile_configured_model(config, require_supported=False)
        )

    def model_compile(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if config.generation.output is None:
            raise ConfigurationError("model compile requires generation.output")
        compiled = _compile_configured_model(config, require_supported=True)
        output = compiled.write(config.generation.output)
        return {
            "model": compiled.name,
            "supported": compiled.supported,
            "output": str(output),
            "conversion_seconds": compiled.conversion_seconds,
            "phase_timings": dict(compiled.phase_timings),
            "source": dict(compiled.source),
            "capabilities": dict(compiled.capabilities),
        }

    def model_processes(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if not config.process.entries:
            raise ConfigurationError("model processes requires a process request")

        from pyamplicol.processes.model import (
            ModelParticleCatalog,
            build_model_process_ir,
            expand_model_processes,
        )

        compiled = _compile_configured_model(config, require_supported=True)
        catalog = ModelParticleCatalog(compiled.ir.name, compiled.ir.particles)
        multiparticles = {
            **catalog.default_multiparticles(),
            **config.process.multiparticles,
        }
        entries: dict[str, dict[str, object]] = {}
        for entry in config.process.entries:
            request = entry.expression
            for process in expand_model_processes(
                request,
                catalog,
                multiparticles=multiparticles,
            ):
                process_ir = build_model_process_ir(
                    process,
                    compiled.ir,
                    color_accuracy=config.color.accuracy,
                )
                entries.setdefault(
                    process_ir.key,
                    {
                        "key": process_ir.key,
                        "process": process_ir.process,
                        "ir": process_ir.to_json_dict(),
                    },
                )
        concrete = list(entries.values())
        return {
            "available": True,
            "model": compiled.name,
            "model_source": dict(compiled.source),
            "requests": [entry.expression for entry in config.process.entries],
            "default_key": concrete[0]["key"] if concrete else None,
            "n_entries": len(concrete),
            "multiparticles": {
                name: list(values) for name, values in sorted(multiparticles.items())
            },
            "entries": concrete,
        }


__all__ = ["CliServices", "DefaultCliServices", "dispatch"]
