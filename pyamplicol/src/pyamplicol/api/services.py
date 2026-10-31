# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pyamplicol as _pyamplicol
from pyamplicol.config import (
    Action,
    BenchmarkConfig,
    ConfigResolution,
    GenerationConfig,
    PoleSelection,
    R2Strategy,
    RunConfig,
)
from pyamplicol.reporting import ProgressSink

from .errors import (
    ArtifactError,
    CompatibilityError,
    DependencyError,
    EvaluationError,
    GenerationError,
)
from .protocols import (
    BenchmarkBackend,
    BenchmarkFactory,
    GeneratorBackend,
    GeneratorFactory,
    ModelParameters,
    Momenta,
    RuntimeBackend,
    RuntimeLoader,
    ScalarInput,
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
    TreeGradedEvaluation,
    TreeOrderEvaluation,
    WarmUpResult,
    _sum_entries,
)
from .virtual_results import (
    ResolvedVirtualEvaluation,
    VirtualEvaluation,
    VirtualStabilityChecks,
    VirtualStabilityPolicy,
    pole_selection,
)

_generator_factory: GeneratorFactory | None = None
_runtime_loader: RuntimeLoader | None = None
_benchmark_factory: BenchmarkFactory | None = None

if TYPE_CHECKING:
    from pyamplicol.runtime.correlations import CorrelatorCatalogue, CorrelatorEvaluator


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
    return tuple(identifiers) or None


def _point_selector_ids(
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
    return tuple(identifiers) or None


def _accepts_keyword_arguments(operation: object, *names: str) -> bool:
    if not callable(operation):
        return False
    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    declared = {parameter.name for parameter in parameters}
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    return accepts_arbitrary or all(name in declared for name in names)


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
        correlators: _pyamplicol.CorrelatorConfig | None = None,
    ) -> GenerationResult:
        """Generate an artifact in ``error``, ``append``, or ``replace`` mode."""

        if mode not in ("error", "append", "replace"):
            raise ValueError("generation mode must be 'error', 'append', or 'replace'")
        process_set = _process_set(processes)
        destination = Path(os.fspath(output)).expanduser().resolve(strict=False)
        if destination.exists() and not destination.is_dir():
            raise GenerationError(
                f"artifact destination is not a directory: {destination}"
            )
        if mode == "error" and destination.exists():
            raise FileExistsError(f"artifact already exists: {destination}")
        if mode == "append" and not destination.is_dir():
            raise FileNotFoundError(f"cannot append to missing artifact: {destination}")
        self._resolve_generation_resources()
        if correlators is not None:
            run = (
                self._config.effective
                if isinstance(self._config, ConfigResolution)
                else self._config
            )
            if (
                isinstance(run, RunConfig)
                and run.process.loop_coupling_orders is not None
            ):
                operation = self._implementation().generate
                if not _accepts_keyword_arguments(operation, "correlators"):
                    raise GenerationError(
                        "generator backend does not support virtual "
                        "correlator declarations"
                    )
                result = cast(Any, operation)(
                    process_set,
                    destination,
                    model=model,
                    mode=mode,
                    correlators=correlators,
                )
                if not isinstance(result, GenerationResult):
                    raise GenerationError(
                        "generator backend returned an invalid GenerationResult"
                    )
                return result
            from pyamplicol.generation.correlated import generate_correlated

            return generate_correlated(
                process_set,
                destination,
                model=model,
                mode=mode,
                config=self._config,
                progress=self._progress,
                declarations=correlators,
            )
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
        # Python 3.11 runtime Protocol checks invoke properties. Match the
        # static lookup used by newer Python versions, preserving lazy physics.
        # Like the Protocol check, require presence: a member may hold None.
        missing = object()
        if not all(
            inspect.getattr_static(backend, name, missing) is not missing
            for name in (
                "physics",
                "evaluate",
                "evaluate_resolved",
                "set_model_parameters",
                "clear",
                "mute_warnings",
                "unmute_warnings",
            )
        ):
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
        arithmetic: str | None = None,
        precision: int | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
    ) -> Runtime:
        """Load one process by stable ID, alias ID, or exact expression.

        ``model_parameters`` is applied atomically before the runtime is
        returned. Omit ``process`` only for a single-process artifact or to use
        the artifact's declared default.

        Virtual runtimes accept initial ``arithmetic``, ``precision`` and
        ``r2_strategy`` defaults, selected before exact model parameters are
        evaluated. Omitted evaluation options retain these defaults; tree
        runtimes select precision at evaluation time instead. ``stability`` (a
        ``VirtualStabilityPolicy`` rescue) is refused by v2 loop runtimes, the
        only ones since the legacy engine was removed: they return estimates
        (``stability_checks``) and leave escalation to the caller.
        """

        path = Path(os.fspath(artifact)).expanduser().resolve(strict=False)
        parameters = dict(model_parameters) if model_parameters is not None else None
        options: dict[str, object] = {}
        if arithmetic is not None:
            options["arithmetic"] = arithmetic
        if precision is not None:
            options["precision"] = _validate_precision(precision)
        if r2_strategy is not None:
            options["r2_strategy"] = r2_strategy
        if stability is not None:
            if not isinstance(stability, VirtualStabilityPolicy):
                raise TypeError("stability requires a VirtualStabilityPolicy")
            options["stability"] = stability
        loader = _get_runtime_loader()
        if options and not _accepts_keyword_arguments(loader, *options):
            raise CompatibilityError(
                "runtime loader does not support load-time "
                + ", ".join(options)
                + "; install a loader supporting these options"
            )
        backend = cast(Callable[..., RuntimeBackend], loader)(
            path,
            process=process,
            model_parameters=parameters,
            mute_warnings=mute_warnings,
            **options,
        )
        return backend if isinstance(backend, cls) else cls(backend)

    @property
    def physics(self) -> ProcessPhysics:
        """Return the complete public process metadata.

        On-the-fly runtimes materialize their compact helicity and color-flow
        axes on first access and retain that compatibility view for the life of
        the runtime. Use :meth:`inspect` for compact high-multiplicity runtime
        metadata and cache-state observation.
        """

        result = self._backend.physics
        if not isinstance(result, ProcessPhysics):
            raise EvaluationError("runtime backend returned invalid process physics")
        return result

    @property
    def artifact_id(self) -> str:
        """Identity of the artifact manifest authenticated by the native loader."""

        try:
            value = getattr(self._backend, "artifact_id", None)
        except ArtifactError as error:
            raise EvaluationError(
                "runtime backend does not expose an authenticated artifact identity"
            ) from error
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value != value.lower()
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise EvaluationError(
                "runtime backend does not expose an authenticated artifact identity"
            )
        return value

    @property
    def execution_mode(
        self,
    ) -> Literal["compiled", "eager", "recurrence", "on-the-fly", "loop_recurrence"]:
        """Native execution lane authenticated while loading the artifact."""

        value = getattr(self._backend, "execution_mode", None)
        if value not in {
            "compiled",
            "eager",
            "recurrence",
            "on-the-fly",
            "loop_recurrence",
        }:
            raise EvaluationError(
                "runtime backend does not expose a valid execution mode"
            )
        return cast(
            Literal["compiled", "eager", "recurrence", "on-the-fly", "loop_recurrence"],
            value,
        )

    @property
    def contribution_kind(self) -> Literal["tree", "virtual"]:
        """Observable kind declared by the loaded backend.

        Existing injected tree backends may omit this additive capability.
        The loop execution lane always requires virtual result semantics.
        """
        mode = getattr(self._backend, "execution_mode", None)
        kind = getattr(
            self._backend,
            "contribution_kind",
            "virtual" if mode == "loop_recurrence" else "tree",
        )
        if kind not in {"tree", "virtual"} or (
            mode is not None and (mode == "loop_recurrence") != (kind == "virtual")
        ):
            raise EvaluationError(
                "runtime backend has inconsistent contribution metadata"
            )
        return cast(Literal["tree", "virtual"], kind)

    def inspect(self) -> Mapping[str, object]:
        """Return compact authenticated runtime metadata and live cache state.

        This optional facade capability does not form part of the minimum
        :class:`RuntimeBackend` protocol. The built-in backend implements it
        without opening dense process physics.
        """

        operation = getattr(self._backend, "inspect", None)
        if not callable(operation):
            raise CompatibilityError(
                "runtime backend does not expose compact runtime inspection"
            )
        result = operation()
        if not isinstance(result, Mapping):
            raise EvaluationError("runtime backend returned invalid runtime inspection")
        return dict(result)

    def save(self, path: os.PathLike[str] | str) -> None:
        """Save the retained on-the-fly recursion after completed evaluations.

        This saves structural warm-up work, not momenta, evaluated matrix
        elements, or model-parameter values. Load the same process normally,
        then use :meth:`load_cache` to restore it. Only the currently retained
        selector family is saved; other selectors keep their usual cold path.
        """

        operation = getattr(self._backend, "save", None)
        if not callable(operation):
            raise CompatibilityError(
                "runtime backend does not support saving on-the-fly caches"
            )
        operation(Path(os.fspath(path)).expanduser().resolve(strict=False))

    def load_cache(self, path: os.PathLike[str] | str) -> None:
        """Restore a saved on-the-fly recursion into the same loaded process.

        The current model parameters remain unchanged. A failed restore leaves
        the existing cache intact. Restoring does not specialize the process:
        new momenta, batch sizes and selector families remain supported.
        """

        operation = getattr(self._backend, "load_cache", None)
        if not callable(operation):
            raise CompatibilityError(
                "runtime backend does not support restoring on-the-fly caches"
            )
        operation(Path(os.fspath(path)).expanduser().resolve(strict=False))

    def evaluate(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        helicity_by_point: Sequence[str | HelicityConfiguration] | None = None,
        color_flow_by_point: Sequence[str | ColorFlow] | None = None,
        precision: int | None = None,
        arithmetic: Literal["double", "double-double", "arbitrary"] | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
    ) -> tuple[complex | Decimal, ...] | VirtualEvaluation:
        """Return tree values or a typed Born-loop interference result.

        Virtual results retain their squared-order and Laurent axes; no finite
        coefficient or coupling-order sum is selected implicitly.

        Selectors accept stable string IDs or the typed objects exposed by
        :attr:`physics`. Omitted precision uses the runtime's load-time default,
        or 16 when the backend has no precision default. Tree precision 16 uses
        the native f64 path; larger values use the retained high-precision
        evaluator state. Explicit ``arithmetic`` selects double, DoubleFloat or
        arbitrary precision. Virtual runtimes retain their loaded arithmetic
        and R2 defaults when omitted, and accept scalar or per-point positive
        ``mu_r2``; exact inputs are forwarded unchanged. An explicit ``stability``
        policy replaces the load-time verification/subset-rescue policy for this
        call. With neither policy there are no hidden precision retries.
        ``on_failure="return"`` retains numerical failures as ``None`` at their
        original point positions, with typed diagnostics; the default raises.
        Invalid inputs and unsupported capabilities still raise.

        Virtual runtimes of the v2 engine also accept ``poles`` (a
        :class:`PoleSelection`: ``"all"``, the default; ``"none"``, which
        computes the finite part only, the poles are then ``None``; ``"uv"``
        or ``"ir"``, the ultraviolet or infrared parts of the poles, with both
        origins' parts by component in ``VirtualEvaluation.pole_parts``), and
        ``stability_checks`` (rotations and/or the reduction-direction check),
        which add a stability estimate per point; without checks no estimate
        is returned.
        """

        helicity_ids = _selector_ids(
            helicities,
            expected_type=HelicityConfiguration,
            name="helicity",
        )
        color_ids = _selector_ids(
            color_flows,
            expected_type=ColorFlow,
            name="color-flow",
        )
        point_helicity_ids = _point_selector_ids(
            helicity_by_point,
            expected_type=HelicityConfiguration,
            name="per-point helicity",
        )
        point_color_ids = _point_selector_ids(
            color_flow_by_point,
            expected_type=ColorFlow,
            name="per-point color-flow",
        )
        if helicity_ids is not None and point_helicity_ids is not None:
            raise ValueError("helicities and helicity_by_point are mutually exclusive")
        if color_ids is not None and point_color_ids is not None:
            raise ValueError(
                "color_flows and color_flow_by_point are mutually exclusive"
            )
        precision = _validate_precision(
            getattr(self._backend, "default_precision", 16)
            if precision is None
            else precision
        )
        contribution = self.contribution_kind
        options = self._evaluation_options(
            arithmetic, mu_r2, r2_strategy, stability, on_failure
        )
        options.update(self._v2_options(poles, stability_checks))
        self._require_evaluation_options(self._backend.evaluate, options)
        if point_helicity_ids is None and point_color_ids is None:
            values = self._backend.evaluate(
                momenta,
                helicities=helicity_ids,
                color_flows=color_ids,
                precision=precision,
                **options,
            )
        else:
            operation = self._backend.evaluate
            if not _accepts_keyword_arguments(
                operation,
                "helicity_by_point",
                "color_flow_by_point",
            ):
                raise EvaluationError(
                    "runtime backend does not support per-point helicity/color-flow "
                    "selectors; use batch-global selectors or install a "
                    "selector-capable backend"
                )
            values = cast(Any, operation)(
                momenta,
                helicities=helicity_ids,
                color_flows=color_ids,
                helicity_by_point=point_helicity_ids,
                color_flow_by_point=point_color_ids,
                precision=precision,
                **options,
            )
        if contribution == "virtual":
            if not isinstance(values, VirtualEvaluation) or values.correlated:
                raise EvaluationError(
                    "virtual runtime backend returned an invalid ordinary "
                    "VirtualEvaluation"
                )
            return values
        if isinstance(values, VirtualEvaluation):
            raise EvaluationError("tree runtime backend returned a virtual result")
        return tuple(
            value if isinstance(value, Decimal) else complex(value) for value in values
        )

    def _evaluation_options(
        self,
        arithmetic: str | None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None,
        r2_strategy: R2Strategy | str | None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
    ) -> dict[str, object]:
        """Forward explicit options without changing old backend call signatures."""
        options: dict[str, object] = {}
        if on_failure not in {"raise", "return"}:
            raise ValueError("on_failure must be 'raise' or 'return'")
        if on_failure == "return":
            if self.contribution_kind != "virtual":
                raise CompatibilityError(
                    "partial-value return requires a virtual runtime"
                )
            options["on_failure"] = on_failure
        if arithmetic is not None:
            if arithmetic not in {"double", "double-double", "arbitrary"}:
                raise ValueError("unknown evaluation arithmetic")
            options["arithmetic"] = arithmetic
        if mu_r2 is not None:
            options["mu_r2"] = mu_r2
        if r2_strategy is not None:
            options["r2_strategy"] = R2Strategy(r2_strategy).value
        if stability is not None:
            if not isinstance(stability, VirtualStabilityPolicy):
                raise TypeError("stability requires a VirtualStabilityPolicy")
            options["stability"] = stability
        if self.contribution_kind != "virtual" and (
            mu_r2 is not None or r2_strategy is not None or stability is not None
        ):
            raise CompatibilityError(
                "mu_r2, r2_strategy and stability require a virtual runtime"
            )
        return options

    def _v2_options(
        self,
        poles: PoleSelection | str | None,
        checks: VirtualStabilityChecks | None,
    ) -> dict[str, object]:
        """Pole-selection and stability-check options, forwarded only when set."""
        options: dict[str, object] = {}
        if poles is not None:
            options["poles"] = pole_selection(poles)
        if checks is not None:
            if not isinstance(checks, VirtualStabilityChecks):
                raise TypeError("stability_checks requires VirtualStabilityChecks")
            options["stability_checks"] = checks
        if options and self.contribution_kind != "virtual":
            raise CompatibilityError("poles and stability_checks are virtual options")
        return options

    @staticmethod
    def _require_evaluation_options(
        operation: object, options: Mapping[str, object]
    ) -> None:
        if options and not _accepts_keyword_arguments(operation, *options):
            raise CompatibilityError(
                "runtime backend does not support the requested evaluation options: "
                + ", ".join(options)
            )

    def warm_up(
        self,
        momenta: Momenta,
        *,
        precision: int = 16,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        progress: ProgressSink | None = None,
        n_cores: int | None = None,
    ) -> WarmUpResult:
        """Warm one on-the-fly selector family using exactly one f64 point.

        This is an explicit structural warm-up, not a throughput evaluation:
        batches and high-precision execution are intentionally rejected. LC
        accepts the same helicity and color-flow selectors as :meth:`evaluate`;
        contracted NLC/full execution accepts only helicity selectors.
        ``n_cores`` overrides the process's query-construction worker count for
        this call only; omitted, it uses the generation-time setting. Changing
        the count neither discards a warm family nor changes evaluation threads.
        """

        if n_cores is not None:
            if isinstance(n_cores, bool) or not isinstance(n_cores, int):
                raise TypeError("n_cores must be a positive integer or None")
            if n_cores < 1:
                raise ValueError("n_cores must be a positive integer or None")
        try:
            point_count = len(momenta)
        except TypeError as exc:
            raise TypeError("warm_up momenta must be a sized sequence") from exc
        if point_count != 1:
            raise ValueError(
                "warm_up requires exactly one phase-space point; "
                f"received {point_count}"
            )
        precision = _validate_precision(precision)
        if precision != 16:
            raise CompatibilityError(
                "on-the-fly warm_up supports only precision=16 (native f64); "
                f"received precision={precision}"
            )
        _validate_progress(progress)
        if self.execution_mode != "on-the-fly":
            raise CompatibilityError(
                "warm_up is available only for on-the-fly runtimes"
            )
        operation = getattr(self._backend, "warm_up", None)
        if not callable(operation):
            raise CompatibilityError(
                "installed on-the-fly runtime does not expose explicit warm-up; "
                "reinstall pyAmpliCol from the current source revision"
            )
        result = operation(
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
            progress=progress,
            n_cores=n_cores,
        )
        if not isinstance(result, WarmUpResult):
            raise EvaluationError("runtime backend returned an invalid WarmUpResult")
        return result

    def evaluate_with_prec(
        self,
        momenta: Momenta,
        precision: int,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        arithmetic: Literal["double", "double-double", "arbitrary"] | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
    ) -> tuple[complex | Decimal, ...] | VirtualEvaluation:
        return self.evaluate(
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            arithmetic=arithmetic,
            mu_r2=mu_r2,
            r2_strategy=r2_strategy,
            stability=stability,
            on_failure=on_failure,
            poles=poles,
            stability_checks=stability_checks,
        )

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        precision: int | None = None,
        arithmetic: Literal["double", "double-double", "arbitrary"] | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
    ) -> ResolvedEvaluation | ResolvedVirtualEvaluation:
        """Return physical values resolved by helicity and color component.

        LC output has shape ``(point, helicity, color_flow)``. NLC/full output
        has shape ``(point, helicity, 1)`` because color is already contracted.
        For tree results, summing the non-point axes reproduces :meth:`evaluate`.
        Virtual results additionally retain squared-order and Laurent axes;
        their ``total()`` sums only the helicity and colour axes.
        ``on_failure="return"`` preserves failed point entries and diagnostics
        through that sum, as in :meth:`evaluate`.
        Omitted precision uses the runtime's load-time default, or 16 when the
        backend has no precision default, as in :meth:`evaluate`.
        ``poles`` and ``stability_checks`` follow :meth:`evaluate` (v2 virtual
        runtimes; the estimates and the pole parts are per resolved entry).
        """

        precision = _validate_precision(
            getattr(self._backend, "default_precision", 16)
            if precision is None
            else precision
        )
        contribution = self.contribution_kind
        options = self._evaluation_options(
            arithmetic, mu_r2, r2_strategy, stability, on_failure
        )
        options.update(self._v2_options(poles, stability_checks))
        self._require_evaluation_options(self._backend.evaluate_resolved, options)
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
            **options,
        )
        if contribution == "virtual":
            if not isinstance(result, ResolvedVirtualEvaluation) or result.correlated:
                raise EvaluationError(
                    "virtual runtime backend returned an invalid ordinary "
                    "ResolvedVirtualEvaluation"
                )
            return result
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
        arithmetic: Literal["double", "double-double", "arbitrary"] | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
    ) -> ResolvedEvaluation | ResolvedVirtualEvaluation:
        return self.evaluate_resolved(
            momenta,
            helicities=helicities,
            color_flows=color_flows,
            precision=precision,
            arithmetic=arithmetic,
            mu_r2=mu_r2,
            r2_strategy=r2_strategy,
            stability=stability,
            on_failure=on_failure,
            poles=poles,
            stability_checks=stability_checks,
        )

    def evaluate_orders(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        color_flows: Sequence[str | ColorFlow] | None = None,
        precision: int | None = None,
    ) -> TreeOrderEvaluation:
        """Return tree values split by squared coupling order.

        Every squared coupling order of the generated amplitude, including
        vanishing interference, is a separate ``values[point][order]`` column
        alongside ``total``, which is exactly :meth:`evaluate`'s value for the
        same global selectors and precision; per-point selectors are not
        supported. Orders follow the Born order axis of virtual results
        (:attr:`VirtualEvaluation.born_orders`). Virtual runtimes already
        return their order axes from :meth:`evaluate`.
        """

        if self.contribution_kind != "tree":
            raise CompatibilityError(
                "virtual results already carry Born and virtual order axes; "
                "use evaluate()"
            )
        operation = getattr(self._backend, "evaluate_orders", None)
        if not callable(operation):
            raise CompatibilityError(
                "runtime backend does not support squared coupling-order partials"
            )
        precision = _validate_precision(
            getattr(self._backend, "default_precision", 16)
            if precision is None
            else precision
        )
        result = operation(
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
        if not isinstance(result, TreeOrderEvaluation):
            raise EvaluationError(
                "runtime backend returned an invalid TreeOrderEvaluation"
            )
        return result

    def evaluate_graded(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        precision: int | None = None,
    ) -> TreeGradedEvaluation:
        """Return tree values split by power of N_c: ``values[point][degree]``.

        Available for artifacts graded by power of N_c: tree artifacts of
        ``color.accuracy = "strict-lc"`` or ``"strict-nlc"`` (the formal 1/N_c
        expansion truncated at the leading degree P, or at P, P-1 and P-2),
        or of full colour with ``color.graded`` (every degree). Each degree's
        term is evaluated at N_c = 3 and summed over the selected helicities;
        the degrees of a point sum to :meth:`evaluate` up to rounding. The
        same split per helicity is the contracted colour axis of
        :meth:`evaluate_resolved`. Precision follows :meth:`evaluate`.
        """

        if self.contribution_kind != "tree":
            raise CompatibilityError("graded output is implemented for tree artifacts")
        physics = self.physics
        components = tuple(physics.contracted_color_components)
        if not components or any(item.nc_degree is None for item in components):
            raise CompatibilityError(
                "graded output needs an artifact generated with "
                "color.accuracy='strict-lc' or 'strict-nlc', or with "
                "color.graded and full colour"
            )
        resolved = self.evaluate_resolved(
            momenta, helicities=helicities, precision=precision
        )
        if not isinstance(resolved, ResolvedEvaluation):
            raise EvaluationError("tree runtime returned an invalid resolved result")
        slot = {
            identifier: index for index, identifier in enumerate(resolved.color_ids)
        }
        try:
            slots = tuple(slot[item.id] for item in components)
        except KeyError as error:
            raise EvaluationError(
                "resolved evaluation does not report every N_c degree"
            ) from error
        values = tuple(
            tuple(
                _sum_entries(tuple(helicity[index] for helicity in point))
                for index in slots
            )
            for point in resolved.values
        )
        degrees = tuple(int(cast(int, item.nc_degree)) for item in components)
        return TreeGradedEvaluation(
            values=values,
            total=tuple(_sum_entries(point) for point in values),
            degrees=degrees,
            color_ids=tuple(resolved.color_ids[index] for index in slots),
            color_accuracy=cast(
                Literal["full", "strict-lc", "strict-nlc"], physics.color_accuracy
            ),
            reference_degree=degrees[0],
        )

    def _diagnostic_project_onshell(
        self,
        momenta: Momenta,
        *,
        precision: int,
    ) -> tuple[object, Mapping[str, object]]:
        """Forward the private report-only kinematic projection capability."""

        precision = _validate_precision(precision)
        operation = getattr(self._backend, "_diagnostic_project_onshell", None)
        if not callable(operation):
            raise EvaluationError(
                "runtime backend does not support diagnostic on-shell projection"
            )
        result = operation(momenta, precision=precision)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[1], Mapping)
        ):
            raise EvaluationError(
                "runtime backend returned an invalid diagnostic on-shell projection"
            )
        return result[0], dict(result[1])

    def set_model_parameters(self, mapping: ModelParameters) -> None:
        """Validate and atomically apply a batch of runtime model parameters."""

        self._backend.set_model_parameters(dict(mapping))

    def _correlator_catalogue(self) -> CorrelatorCatalogue:
        if self.contribution_kind == "virtual":
            raise CompatibilityError(
                "public virtual correlation dispatch is not integrated yet"
            )
        catalogue: CorrelatorCatalogue | None = getattr(
            self, "_correlated_catalogue", None
        )
        if catalogue is None:
            from pyamplicol.runtime.correlations import load_correlator_catalogue

            catalogue = load_correlator_catalogue(self._backend)
            self._correlated_catalogue = catalogue
        return catalogue

    def available_color_correlations(self) -> tuple[_pyamplicol.ColorCorrelator, ...]:
        """List this process's resolved operators, including ``"born"``.

        Reads the declared catalogue without loading an amplitude executor.
        Each returned record contains its runtime ID and ordered bra/ket steps.
        """
        if self.contribution_kind == "virtual":
            records = self._virtual_correlation_operation(
                "available_color_correlations"
            )()
            if not isinstance(records, tuple) or any(
                not isinstance(item, _pyamplicol.ColorCorrelator) for item in records
            ):
                raise EvaluationError(
                    "virtual backend returned an invalid correlation catalogue"
                )
            return records
        return self._correlator_catalogue().declarations.color_requests

    def _virtual_correlation_operation(self, name: str) -> Callable[..., Any]:
        operation = getattr(self._backend, name, None)
        if not callable(operation):
            raise CompatibilityError(
                "virtual backend has no saved directed correlation operation"
            )
        return cast(Callable[..., Any], operation)

    def _correlator_evaluator(self) -> CorrelatorEvaluator:
        evaluator: CorrelatorEvaluator | None = getattr(
            self, "_correlated_evaluator", None
        )
        if evaluator is None:
            from pyamplicol.runtime.correlations import CorrelatorEvaluator

            evaluator = CorrelatorEvaluator(
                self._backend, catalogue=self._correlator_catalogue()
            )
            self._correlated_evaluator = evaluator
        return evaluator

    def set_spin_correlation_vectors(
        self, vectors: Mapping[int, object] | None
    ) -> None:
        """Replace declared spin-1 sources by literal complex four-vectors.

        Keys are public one-based leg labels. A value is either ``(E,x,y,z)``
        or one such vector per phase-space point. The set of keys must match a
        spin class declared at generation. ``None`` or ``{}`` restores ordinary
        helicity states. Vectors are neither normalized nor projected.
        Real components accept Decimal; a ``(real, imaginary)`` Decimal pair
        represents a complex component without a binary64 conversion.
        This affects the correlated evaluation methods, not :meth:`evaluate`.
        """
        if self.contribution_kind == "virtual":
            self._virtual_correlation_operation("set_spin_correlation_vectors")(vectors)
            return
        self._correlator_evaluator().set_spin_correlation_vectors(vectors)

    def evaluate_correlated(
        self,
        momenta: Momenta,
        *,
        color_correlation: str = "born",
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
    ) -> tuple[_pyamplicol.CorrelatedValue, ...] | VirtualEvaluation:
        """Evaluate a generation-time colour ID with the current spin vectors.

        Tree results contain one complex decimal value per point. Virtual
        results are correlated :class:`VirtualEvaluation` values retaining
        every squared order and all three Laurent coefficients; ``mu_r2`` is
        required. Virtual arithmetic inherits the loaded runtime default.
        ``"born"`` selects the Born metric at the declared colour accuracy.
        All ordinary normalization factors remain in place; a replaced spin
        leg is counted once in the helicity sum.

        ``arithmetic="double-double"`` selects genuine DoubleFloat arithmetic
        (at most 31 output digits); tree defaults use arbitrary precision.
        Decimal spin inputs (including complex-component pairs) retain their
        digits until conversion into the requested arithmetic.
        ``on_failure`` follows :meth:`evaluate`, preserving complex successful
        values and unavailable whole-point entries. ``poles`` and
        ``stability_checks`` follow :meth:`evaluate` (v2 virtual runtimes).
        """
        options = self._evaluation_options(
            arithmetic, mu_r2, r2_strategy, stability, on_failure
        )
        options.update(self._v2_options(poles, stability_checks))
        if self.contribution_kind == "virtual":
            operation = self._virtual_correlation_operation("evaluate_correlated")
            self._require_evaluation_options(operation, options)
            value = operation(
                momenta,
                color_correlation=color_correlation,
                helicities=_selector_ids(
                    helicities, expected_type=HelicityConfiguration, name="helicity"
                ),
                precision=_validate_precision(
                    getattr(self._backend, "default_precision", 16)
                    if precision is None
                    else precision
                ),
                **options,
            )
            if not isinstance(value, VirtualEvaluation) or not value.correlated:
                raise EvaluationError(
                    "virtual correlation backend returned an invalid result"
                )
            return value
        return self._correlator_evaluator().evaluate(
            momenta,
            color_correlation=color_correlation,
            helicities=_selector_ids(
                helicities, expected_type=HelicityConfiguration, name="helicity"
            ),
            precision=_validate_precision(16 if precision is None else precision),
            arithmetic="arbitrary" if arithmetic is None else arithmetic,
        )

    def evaluate_correlated_graded(
        self,
        momenta: Momenta,
        *,
        color_correlation: str = "born",
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
    ) -> _pyamplicol.GradedCorrelatedEvaluation:
        """:meth:`evaluate_correlated` split by power of N_c (tree artifacts
        graded by N_c: ``color.accuracy = "strict-lc"`` or ``"strict-nlc"``,
        or full colour with ``color.graded``). ``values[point][degree]`` is
        the N_c^degree term at N_c = 3; the degrees sum to ``total``."""

        if self.contribution_kind != "tree":
            raise CompatibilityError("graded correlations are implemented for trees")
        return self._correlator_evaluator().evaluate_graded(
            momenta,
            color_correlation=color_correlation,
            helicities=_selector_ids(
                helicities, expected_type=HelicityConfiguration, name="helicity"
            ),
            precision=_validate_precision(16 if precision is None else precision),
            arithmetic="arbitrary" if arithmetic is None else arithmetic,
        )

    def evaluate_correlated_many(
        self,
        momenta: Momenta,
        requests: Mapping[str, _pyamplicol.CorrelatedRequest],
        *,
        helicities: Sequence[str | HelicityConfiguration] | None = None,
        precision: int | None = None,
        arithmetic: str | None = None,
        mu_r2: ScalarInput | Sequence[ScalarInput] | None = None,
        r2_strategy: R2Strategy | str | None = None,
        stability: VirtualStabilityPolicy | None = None,
        on_failure: Literal["raise", "return"] = "raise",
        poles: PoleSelection | str | None = None,
        stability_checks: VirtualStabilityChecks | None = None,
    ) -> (
        dict[str, tuple[_pyamplicol.CorrelatedValue, ...]]
        | dict[str, VirtualEvaluation]
    ):
        """Evaluate labelled colour/spin combinations over a point batch.

        Equal spin assignments share amplitudes across colour IDs. Different
        assignments may reuse stages with identical exact inputs. Numerical
        reuse is local to this call; requests never change the spin setter.
        ``None`` vectors inherit its initial state and ``{}`` selects physical
        helicities. Each result tuple follows the input point order.
        Virtual requests select declared full-colour histories from one native
        batch per distinct source assignment; values retain complex coupling-order
        and Laurent axes. Saved joint massless-gluon spin classes compose with
        those histories. Resolved-history output remains unavailable.
        Arithmetic and Decimal spin inputs follow :meth:`evaluate_correlated`.
        With ``on_failure="return"``, distinct source assignments can have
        different failure masks; every labelled result retains input positions.
        ``poles`` and ``stability_checks`` follow :meth:`evaluate` (v2 virtual
        runtimes).
        """
        options = self._evaluation_options(
            arithmetic, mu_r2, r2_strategy, stability, on_failure
        )
        options.update(self._v2_options(poles, stability_checks))
        if self.contribution_kind == "virtual":
            operation = self._virtual_correlation_operation("evaluate_correlated_many")
            self._require_evaluation_options(operation, options)
            values = operation(
                momenta,
                requests,
                helicities=_selector_ids(
                    helicities, expected_type=HelicityConfiguration, name="helicity"
                ),
                precision=_validate_precision(
                    getattr(self._backend, "default_precision", 16)
                    if precision is None
                    else precision
                ),
                **options,
            )
            if (
                not isinstance(values, dict)
                or list(values) != list(requests)
                or any(
                    not isinstance(value, VirtualEvaluation) or not value.correlated
                    for value in values.values()
                )
            ):
                raise EvaluationError(
                    "virtual correlation backend returned invalid grouped results"
                )
            return cast(dict[str, VirtualEvaluation], values)
        return self._correlator_evaluator().evaluate_many(
            momenta,
            requests,
            helicities=_selector_ids(
                helicities, expected_type=HelicityConfiguration, name="helicity"
            ),
            precision=_validate_precision(16 if precision is None else precision),
            arithmetic="arbitrary" if arithmetic is None else arithmetic,
        )

    def clear(self) -> None:
        """Drop warmed execution state while keeping this artifact loaded.

        An already materialized :attr:`physics` compatibility view remains
        cached until the runtime itself is released.
        """

        self._backend.clear()
        correlated = getattr(self, "_correlated_evaluator", None)
        if correlated is not None:
            correlated.clear()

    @property
    def representative_process_key(self) -> str:
        """Stable artifact process implementing the selected public ordering."""

        value = getattr(self._backend, "representative_process_key", None)
        return value if isinstance(value, str) and value else self.physics.process_id

    @property
    def external_permutation(self) -> tuple[int, ...]:
        """Representative-index to public-index external-leg permutation."""

        value = getattr(self._backend, "external_permutation", None)
        count_value = getattr(self._backend, "external_count", None)
        count: int | None
        if count_value is None:
            count = None
        elif (
            isinstance(count_value, bool)
            or not isinstance(count_value, int)
            or count_value < 1
        ):
            raise CompatibilityError("runtime external process count is invalid")
        else:
            count = count_value
        if value is None:
            if count is None:
                # Preserve the original minimum RuntimeBackend contract. The
                # built-in backend always supplies authenticated compact
                # metadata, while older injected backends may expose only
                # ProcessPhysics.
                return tuple(range(len(self.physics.external_particles)))
            return tuple(range(count))
        try:
            permutation = tuple(value)
        except TypeError as exc:
            raise CompatibilityError(
                "runtime external permutation must be an iterable of integers"
            ) from exc
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in permutation
        ):
            raise CompatibilityError(
                "runtime external permutation must contain integers"
            )
        expected_count = (
            len(self.physics.external_particles) if count is None else count
        )
        if len(permutation) != expected_count or sorted(permutation) != list(
            range(expected_count)
        ):
            raise CompatibilityError("runtime external permutation is invalid")
        return permutation

    @property
    def process_group_factor(self) -> Fraction:
        """Exact factor relating the selected process to its representative.

        A process selected through the artifact's process-group map evaluates
        ``factor * ME_representative`` at the permuted momenta; every other
        selection reports one.
        """

        value = getattr(self._backend, "process_group_factor", None)
        if value is None:
            return Fraction(1)
        if not isinstance(value, Fraction) or value <= 0:
            raise CompatibilityError("runtime process-group factor is invalid")
        return value

    @property
    def charge_conjugated(self) -> bool:
        """Whether the selection is the charge conjugate of its representative.

        Its LC color-flow words are then the reversed representative words.
        """

        return getattr(self._backend, "charge_conjugated", False) is True

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
        *,
        _chunk_guard: Callable[[float | None, str], None] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, (BenchmarkConfig, RunConfig)):
            raise TypeError(
                "BenchmarkRunner config must be BenchmarkConfig, RunConfig, or null"
            )
        _validate_progress(progress)
        if _chunk_guard is not None and not callable(_chunk_guard):
            raise TypeError("BenchmarkRunner chunk guard must be callable")
        self._config = config
        self._progress = progress
        self._chunk_guard = _chunk_guard
        self._backend: BenchmarkBackend | None = None

    def _implementation(self) -> BenchmarkBackend:
        if self._backend is None:
            self._backend = _get_benchmark_factory()(self._config, self._progress)
            if self._chunk_guard is not None:
                setter = getattr(self._backend, "set_chunk_guard", None)
                if not callable(setter):
                    raise EvaluationError(
                        "benchmark backend does not support a profiling chunk guard"
                    )
                setter(self._chunk_guard)
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
    correlators: _pyamplicol.CorrelatorConfig | None = None,
) -> GenerationResult:
    """Generate a process artifact using a one-shot convenience function."""

    return Generator(config=config, progress=progress).generate(
        processes, output, model=model, mode=mode, correlators=correlators
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
