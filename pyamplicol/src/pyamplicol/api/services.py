# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pyamplicol as _pyamplicol
from pyamplicol.config import (
    Action,
    BenchmarkConfig,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
)
from pyamplicol.reporting import ProgressSink

from .errors import DependencyError, EvaluationError, GenerationError
from .protocols import (
    BenchmarkBackend,
    BenchmarkFactory,
    GeneratorBackend,
    GeneratorFactory,
    ModelParameters,
    Momenta,
    RuntimeBackend,
    RuntimeLoader,
)
from .requests import ModelSource, ProcessRequest, ProcessSet
from .results import (
    BenchmarkResult,
    ColorFlow,
    GenerationPlan,
    GenerationResult,
    HelicityConfiguration,
    ProcessPhysics,
    ResolvedEvaluation,
)

_generator_factory: GeneratorFactory | None = None
_runtime_loader: RuntimeLoader | None = None
_benchmark_factory: BenchmarkFactory | None = None


def install_backend_factories(
    *,
    generator: GeneratorFactory | None = None,
    runtime: RuntimeLoader | None = None,
    benchmark: BenchmarkFactory | None = None,
) -> None:
    """Install backend adapters without importing domain dependencies eagerly."""

    global _generator_factory, _runtime_loader, _benchmark_factory
    if generator is not None:
        _generator_factory = generator
    if runtime is not None:
        _runtime_loader = runtime
    if benchmark is not None:
        _benchmark_factory = benchmark


def _discover(attribute: str, modules: Sequence[str]) -> Any:
    failures: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        factory = getattr(module, attribute, None)
        if factory is not None:
            return factory
        failures.append(f"{module_name}: missing {attribute}")
    detail = "; ".join(failures)
    raise DependencyError(
        f"no pyAmpliCol backend provides {attribute}; backend discovery tried {detail}"
    )


def _get_generator_factory() -> GeneratorFactory:
    if _generator_factory is not None:
        return _generator_factory
    return cast(
        GeneratorFactory,
        _discover(
            "create_generator_backend",
            ("pyamplicol.generation", "pyamplicol.generation.service"),
        ),
    )


def _generation_resource_resolution(
    config: GenerationConfig | RunConfig | ConfigResolution | None,
) -> ConfigResolution:
    from pyamplicol import licensing

    if isinstance(config, ConfigResolution):
        effective = config.effective
    else:
        effective = (
            RunConfig(action=Action.GENERATE)
            if config is None
            else RunConfig(action=Action.GENERATE, generation=config)
            if isinstance(config, GenerationConfig)
            else config
        )
    state = licensing.detect_symbolica_license(
        suggest=effective.symbolica.suggest_license,
        json_mode=str(effective.output.format) == "json",
    )
    return licensing.resolve_symbolica_resource_config(
        config,
        state,
    )


def _get_runtime_loader() -> RuntimeLoader:
    if _runtime_loader is not None:
        return _runtime_loader
    return cast(
        RuntimeLoader,
        _discover(
            "load_runtime_backend",
            ("pyamplicol.artifact", "pyamplicol.runtime"),
        ),
    )


def _get_benchmark_factory() -> BenchmarkFactory:
    if _benchmark_factory is not None:
        return _benchmark_factory
    return cast(
        BenchmarkFactory,
        _discover(
            "create_benchmark_backend",
            ("pyamplicol.benchmarking", "pyamplicol.benchmark"),
        ),
    )


def _process_set(
    processes: ProcessSet | ProcessRequest | str | Iterable[ProcessRequest | str],
) -> ProcessSet:
    if isinstance(processes, ProcessSet):
        return processes
    if isinstance(processes, ProcessRequest):
        return ProcessSet((processes,))
    if isinstance(processes, str):
        return ProcessSet((ProcessRequest.parse(processes),))
    requests = tuple(
        entry if isinstance(entry, ProcessRequest) else ProcessRequest.parse(entry)
        for entry in processes
    )
    return ProcessSet(requests)


def _validate_progress(progress: ProgressSink | None) -> None:
    if progress is not None and not isinstance(progress, ProgressSink):
        raise TypeError("progress must implement ProgressSink.emit(event)")


def _validate_precision(precision: int) -> int:
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise TypeError("precision must be a positive integer number of decimal digits")
    if precision < 1:
        raise ValueError(
            "precision must be a positive integer number of decimal digits"
        )
    return precision


def _selector_ids(
    values: Sequence[str | HelicityConfiguration | ColorFlow] | None,
    *,
    expected_type: type[HelicityConfiguration] | type[ColorFlow],
    name: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    identifiers: list[str] = []
    for value in values:
        if isinstance(value, str) and value:
            identifiers.append(value)
        elif isinstance(value, expected_type):
            identifiers.append(value.id)
        else:
            raise TypeError(f"{name} selectors must be IDs or {expected_type.__name__}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} selectors must be unique")
    return tuple(identifiers)


class Generator:
    """Plan and generate process artifacts with typed configuration.

    A :class:`~pyamplicol.config.ConfigResolution` preserves both requested and
    effective settings, including license/resource clamps, in the output.
    """

    def __init__(
        self,
        config: GenerationConfig | RunConfig | ConfigResolution | None = None,
        progress: ProgressSink | None = None,
    ) -> None:
        if config is not None and not isinstance(
            config, (GenerationConfig, RunConfig, ConfigResolution)
        ):
            raise TypeError(
                "Generator config must be GenerationConfig, RunConfig, "
                "ConfigResolution, or null"
            )
        _validate_progress(progress)
        self._config = config
        self._progress = progress
        self._backend: GeneratorBackend | None = None

    def _implementation(self) -> GeneratorBackend:
        if self._backend is None:
            self._backend = _get_generator_factory()(self._config, self._progress)
        return self._backend

    def _resolve_generation_resources(self) -> None:
        resource_config = _generation_resource_resolution(self._config)
        if resource_config != self._config:
            self._config = resource_config
            self._backend = None

    def plan(
        self,
        processes: ProcessSet | ProcessRequest | str | Iterable[ProcessRequest | str],
        *,
        model: ModelSource | _pyamplicol.CompiledModel | None = None,
    ) -> GenerationPlan:
        """Resolve concrete processes and coverage without writing an artifact."""

        process_set = _process_set(processes)
        self._resolve_generation_resources()
        result = self._implementation().plan(process_set, model=model)
        if not isinstance(result, GenerationPlan):
            raise GenerationError(
                "generator backend returned an invalid GenerationPlan"
            )
        return result

    def generate(
        self,
        processes: ProcessSet | ProcessRequest | str | Iterable[ProcessRequest | str],
        output: os.PathLike[str] | str,
        *,
        model: ModelSource | _pyamplicol.CompiledModel | None = None,
        mode: Literal["error", "append", "replace"] = "error",
    ) -> GenerationResult:
        """Generate an artifact in ``error``, ``append``, or ``replace`` mode."""

        if mode not in ("error", "append", "replace"):
            raise ValueError("generation mode must be 'error', 'append', or 'replace'")
        process_set = _process_set(processes)
        self._resolve_generation_resources()
        destination = Path(os.fspath(output)).expanduser().resolve(strict=False)
        result = self._implementation().generate(
            process_set, destination, model=model, mode=mode
        )
        if not isinstance(result, GenerationResult):
            raise GenerationError(
                "generator backend returned an invalid GenerationResult"
            )
        return result


class Runtime:
    """Typed Python facade for one process in a generated Rusticol artifact."""

    def __init__(self, backend: RuntimeBackend) -> None:
        if not isinstance(backend, RuntimeBackend):
            raise TypeError("Runtime backend does not implement RuntimeBackend")
        self._backend = backend

    @classmethod
    def load(
        cls,
        artifact: os.PathLike[str] | str,
        *,
        process: str | None = None,
        model_parameters: ModelParameters | None = None,
        mute_warnings: bool = False,
    ) -> Runtime:
        """Load one process by stable ID, alias ID, or exact expression.

        ``model_parameters`` is applied atomically before the runtime is
        returned. Omit ``process`` only for a single-process artifact or to use
        the artifact's declared default.
        """

        path = Path(os.fspath(artifact)).expanduser().resolve(strict=False)
        parameters = dict(model_parameters) if model_parameters is not None else None
        backend = _get_runtime_loader()(
            path,
            process=process,
            model_parameters=parameters,
            mute_warnings=mute_warnings,
        )
        return backend if isinstance(backend, cls) else cls(backend)

    @property
    def physics(self) -> ProcessPhysics:
        result = self._backend.physics
        if not isinstance(result, ProcessPhysics):
            raise EvaluationError("runtime backend returned invalid process physics")
        return result

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        precision: int = 16,
    ) -> tuple[complex | Decimal, ...]:
        """Return one fully summed matrix element for every input point.

        Selectors accept stable string IDs or the typed objects exposed by
        :attr:`physics`. Precision 16 uses the native f64 path; larger values
        use the retained high-precision evaluator state when available.
        """

        precision = _validate_precision(precision)
        values = self._backend.evaluate(
            momenta,
            helicities=_selector_ids(
                helicities,
                expected_type=HelicityConfiguration,
                name="helicity",
            ),
            color_flows=_selector_ids(
                color_flows,
                expected_type=ColorFlow,
                name="color-flow",
            ),
            precision=precision,
        )
        return tuple(
            value if isinstance(value, Decimal) else complex(value) for value in values
        )

    def evaluate_with_prec(
        self,
        momenta: Momenta,
        precision: int,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
    ) -> tuple[complex | Decimal, ...]:
        return self.evaluate(
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
        )

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        precision: int = 16,
    ) -> ResolvedEvaluation:
        """Return physical values resolved by helicity and color component.

        LC output has shape ``(point, helicity, color_flow)``. NLC/full output
        has shape ``(point, helicity, 1)`` because color is already contracted.
        Summing the non-point axes reproduces :meth:`evaluate`.
        """

        precision = _validate_precision(precision)
        result = self._backend.evaluate_resolved(
            momenta,
            helicities=_selector_ids(
                helicities,
                expected_type=HelicityConfiguration,
                name="helicity",
            ),
            color_flows=_selector_ids(
                color_flows,
                expected_type=ColorFlow,
                name="color-flow",
            ),
            precision=precision,
        )
        if not isinstance(result, ResolvedEvaluation):
            raise EvaluationError(
                "runtime backend returned an invalid ResolvedEvaluation"
            )
        return result

    def evaluate_resolved_with_prec(
        self,
        momenta: Momenta,
        precision: int,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
    ) -> ResolvedEvaluation:
        return self.evaluate_resolved(
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
        )

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        """Validate and atomically apply a batch of runtime model parameters."""

        self._backend.set_model_parameters(dict(mapping))

    def set_model_parameter(self, name: str, value: complex | float | int) -> None:
        self.set_model_parameters({name: value})

    def mute_warnings(self) -> None:
        self._backend.mute_warnings()

    def unmute_warnings(self) -> None:
        self._backend.unmute_warnings()


class BenchmarkRunner:
    """Profile summed process evaluation with a typed benchmark configuration."""

    def __init__(
        self,
        config: BenchmarkConfig | RunConfig | None = None,
        progress: ProgressSink | None = None,
    ) -> None:
        if config is not None and not isinstance(config, (BenchmarkConfig, RunConfig)):
            raise TypeError(
                "BenchmarkRunner config must be BenchmarkConfig, RunConfig, or null"
            )
        _validate_progress(progress)
        self._config = config
        self._progress = progress
        self._backend: BenchmarkBackend | None = None

    def _implementation(self) -> BenchmarkBackend:
        if self._backend is None:
            self._backend = _get_benchmark_factory()(self._config, self._progress)
        return self._backend

    def run(
        self,
        target: Runtime | os.PathLike[str] | str,
        *,
        points: Momenta | None = None,
    ) -> BenchmarkResult:
        """Profile an artifact path or an already loaded :class:`Runtime`."""

        backend_target: RuntimeBackend | os.PathLike[str] | str
        if isinstance(target, Runtime):
            backend_target = target._backend
        else:
            backend_target = Path(os.fspath(target)).expanduser().resolve(strict=False)
        result = self._implementation().run(backend_target, points=points)
        if not isinstance(result, BenchmarkResult):
            raise EvaluationError(
                "benchmark backend returned an invalid BenchmarkResult"
            )
        return result


def generate(
    processes: ProcessSet | ProcessRequest | str | Iterable[ProcessRequest | str],
    output: os.PathLike[str] | str,
    *,
    model: ModelSource | _pyamplicol.CompiledModel | None = None,
    mode: Literal["error", "append", "replace"] = "error",
    config: GenerationConfig | RunConfig | ConfigResolution | None = None,
    progress: ProgressSink | None = None,
) -> GenerationResult:
    """Generate a process artifact using a one-shot convenience function."""

    return Generator(config=config, progress=progress).generate(
        processes, output, model=model, mode=mode
    )


load = Runtime.load


def benchmark(
    target: Runtime | os.PathLike[str] | str,
    *,
    points: Momenta | None = None,
    config: BenchmarkConfig | RunConfig | None = None,
    progress: ProgressSink | None = None,
) -> BenchmarkResult:
    """Profile a generated artifact using a one-shot convenience function."""

    return BenchmarkRunner(config=config, progress=progress).run(target, points=points)


__all__ = [
    "BenchmarkRunner",
    "Generator",
    "Runtime",
    "benchmark",
    "generate",
    "install_backend_factories",
    "load",
]
