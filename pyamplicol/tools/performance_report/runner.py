"""Direct Python-API generation, profiling, and validation for report cells."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .models import (
    Accuracy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)

RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-15
GENERATION_VALIDATION_SEED = 12345


class RunnerError(RuntimeError):
    """Raised when a cell cannot satisfy the report measurement contract."""


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    target_runtime_seconds: float = 20.0
    batch_size: int = 128
    worker_cores: int = 1
    model_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.target_runtime_seconds <= 0.0:
            raise ValueError("target_runtime_seconds must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.worker_cores < 1:
            raise ValueError("worker_cores must be positive")


@dataclass(frozen=True, slots=True)
class SelectorContract:
    selected_color_flow_ids: tuple[str, ...]
    selected_color_words: tuple[tuple[int, ...], ...]
    all_flow_helicity_ids: tuple[str, ...]
    all_flow_source_helicities: tuple[tuple[int, int], ...]
    point_digest: str

    def __post_init__(self) -> None:
        if not self.selected_color_flow_ids:
            raise ValueError("selector contract requires a selected color flow")
        if len(self.selected_color_flow_ids) != len(self.selected_color_words):
            raise ValueError("color-flow IDs and words must have equal length")
        if len(set(self.selected_color_flow_ids)) != len(
            self.selected_color_flow_ids
        ):
            raise ValueError("selector contract color-flow IDs must be unique")
        if len(self.all_flow_helicity_ids) != 1:
            raise ValueError("selector contract requires one fixed helicity")
        if not self.all_flow_source_helicities:
            raise ValueError("selector contract requires source helicities")
        if len(self.point_digest) != 64:
            raise ValueError("selector contract point digest must be SHA-256")

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_color_flow_ids": list(self.selected_color_flow_ids),
            "selected_color_words": [
                list(word) for word in self.selected_color_words
            ],
            "all_flow_helicity_ids": list(self.all_flow_helicity_ids),
            "all_flow_source_helicities": {
                str(label): value
                for label, value in self.all_flow_source_helicities
            },
            "point_digest": self.point_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectorContract:
        raw_ids = value.get("selected_color_flow_ids")
        raw_words = value.get("selected_color_words")
        raw_helicities = value.get("all_flow_helicity_ids")
        raw_sources = value.get("all_flow_source_helicities")
        point_digest = value.get("point_digest")
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, (str, bytes))
            or not isinstance(raw_words, Sequence)
            or isinstance(raw_words, (str, bytes))
            or not isinstance(raw_helicities, Sequence)
            or isinstance(raw_helicities, (str, bytes))
            or not isinstance(raw_sources, Mapping)
            or not isinstance(point_digest, str)
        ):
            raise ValueError("selector contract has an invalid shape")
        return cls(
            selected_color_flow_ids=tuple(str(item) for item in raw_ids),
            selected_color_words=tuple(
                tuple(int(label) for label in word)  # type: ignore[arg-type]
                for word in raw_words
            ),
            all_flow_helicity_ids=tuple(str(item) for item in raw_helicities),
            all_flow_source_helicities=tuple(
                sorted((int(label), int(state)) for label, state in raw_sources.items())
            ),
            point_digest=point_digest,
        )


@dataclass(frozen=True, slots=True)
class GeneratedArtifact:
    path: Path
    process_id: str
    generation_seconds: float
    model_preparation_seconds: float
    model_preparation_reused: bool
    requested_config: Mapping[str, object]
    effective_config: Mapping[str, object]


class RuntimeLike(Protocol):
    @property
    def physics(self) -> object: ...

    def evaluate(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> Sequence[object]: ...

    def evaluate_resolved(
        self,
        momenta: object,
        *,
        helicities: Sequence[str] | None = None,
        color_flows: Sequence[str] | None = None,
        precision: int = 16,
    ) -> object: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        default=lambda item: item.tolist(),
    ).encode("ascii")


def point_digest(points: object) -> str:
    return hashlib.sha256(_canonical_json(points)).hexdigest()


def _real_nonnegative(value: object) -> float:
    number = complex(value)
    if abs(number.imag) > 1.0e-9 * max(abs(number.real), 1.0):
        raise RunnerError(
            f"matrix element has a non-negligible imaginary part: {value}"
        )
    result = abs(float(number.real))
    if not math.isfinite(result):
        raise RunnerError("matrix element is not finite")
    return result


def _model_source_path(repo_root: Path, model: ModelKey) -> Path | None:
    if model is ModelKey.BUILTIN_SM:
        return None
    if model is ModelKey.UFO_SM:
        return repo_root / "src/pyamplicol/assets/models/json/sm/sm.json"
    raise RunnerError(f"model {model.value!r} is not supported by process matrices")


def config_values(
    cell: CellSpec,
    settings: RunnerSettings,
    *,
    repo_root: Path,
) -> dict[str, object]:
    """Return complete generation/benchmark settings for one pyAmpliCol cell."""

    measurement = cell.measurement
    if measurement.execution_mode is ExecutionMode.AMPLICOL:
        raise RunnerError("original AmpliCol does not use the pyAmpliCol runner")
    if measurement.model is None:
        raise RunnerError("pyAmpliCol measurement requires an explicit model")
    model_path = _model_source_path(repo_root, measurement.model)
    layout = (
        "all-flow-union"
        if cell.workload is Workload.ALL_FLOW
        else "topology-replay"
    )
    values: dict[str, object] = {
        "model": {
            "source": "built-in-sm" if model_path is None else os.fspath(model_path),
            "cache": True,
            "cache_dir": (
                None
                if settings.model_cache_dir is None
                else os.fspath(settings.model_cache_dir)
            ),
        },
        "color": {
            "accuracy": measurement.accuracy.value,
            "lc_flow_layout": layout,
        },
        "generation": {
            "workers": settings.worker_cores,
            "emit_api_bundle": True,
            "validation": {
                "enabled": True,
                "samples": 10,
                "seed": GENERATION_VALIDATION_SEED,
                "relative_tolerance": RELATIVE_TOLERANCE,
                "absolute_tolerance": 1.0e-300,
                "post_build_validation": True,
            },
        },
        "evaluator": {
            "backend": measurement.backend,
            "execution_mode": measurement.execution_mode.value,
            "batch_size": settings.batch_size,
            "output_chunk_size": 512,
            "optimization": {
                "horner_iterations": 10,
                "cpe_iterations": None,
                "cores": settings.worker_cores,
                "max_horner_variables": 1000,
                "max_common_pair_cache_entries": 5_000_000,
                "max_common_pair_distance": 1000,
            },
            "jit": {
                "optimization_level": measurement.jit_optimization_level or 2,
            },
            "cpp": {"optimization": "O3"},
        },
        "benchmark": {
            "target_runtime": settings.target_runtime_seconds,
            "batch_size": settings.batch_size,
            "warmup_runs": 2,
            "minimum_samples": 5,
        },
        "output": {"format": "json", "progress": "off"},
    }
    return values


def _physics_ids(physics: object, name: str) -> tuple[str, ...]:
    return tuple(str(item.id) for item in getattr(physics, name, ()))


def validate_runtime_contract(cell: CellSpec, runtime: RuntimeLike) -> None:
    physics = runtime.physics
    accuracy = str(getattr(physics, "color_accuracy", ""))
    if accuracy != cell.measurement.accuracy.value:
        raise RunnerError(
            f"artifact color accuracy {accuracy!r} does not match "
            f"{cell.measurement.accuracy.value!r}"
        )
    capabilities = set(getattr(physics, "selector_capabilities", ()))
    helicity_ids = _physics_ids(physics, "helicities")
    if not helicity_ids:
        raise RunnerError("artifact exposes no physical helicities")
    if cell.measurement.accuracy is Accuracy.LC:
        color_ids = _physics_ids(physics, "color_flows")
        if not color_ids:
            raise RunnerError("LC artifact exposes no physical color flows")
        missing = {"helicity", "color_flow"} - capabilities
        if missing:
            raise RunnerError(
                "LC artifact does not retain complete runtime selectors: "
                + ", ".join(sorted(missing))
            )
    elif cell.workload is not Workload.CONTRACTED:
        raise RunnerError("NLC/full measurements must use the contracted workload")


def validate_artifact_contract(cell: CellSpec, artifact_path: Path) -> None:
    from pyamplicol.artifacts import inspect_artifact

    inspection = inspect_artifact(artifact_path)
    if len(inspection.processes) != 1:
        raise RunnerError("report artifacts must contain exactly one process")
    process = inspection.processes[0]
    if process.execution_mode != cell.measurement.execution_mode.value:
        raise RunnerError(
            f"artifact execution mode {process.execution_mode!r} does not match "
            f"{cell.measurement.execution_mode.value!r}"
        )
    if process.generation_specialized_axes:
        raise RunnerError(
            "report artifacts must retain complete runtime coverage; specialized "
            f"axes: {process.generation_specialized_axes}"
        )
    if process.selected_source_helicities or process.selected_color_sector_ids:
        raise RunnerError("report artifact contains forbidden generation selectors")
    if cell.measurement.accuracy is Accuracy.LC:
        expected_layout = (
            "all-flow-union"
            if cell.workload is Workload.ALL_FLOW
            else "topology-replay"
        )
        if process.lc_flow_layout != expected_layout:
            raise RunnerError(
                f"artifact LC layout {process.lc_flow_layout!r} does not match "
                f"{expected_layout!r}"
            )


def _selector_kwargs(
    cell: CellSpec,
    contract: SelectorContract | None,
) -> dict[str, tuple[str, ...] | None]:
    if cell.measurement.accuracy is not Accuracy.LC:
        return {"helicities": None, "color_flows": None}
    if contract is None:
        raise RunnerError("LC measurement requires a selector contract")
    if cell.workload is Workload.SELECTED_FLOW:
        return {
            "helicities": None,
            "color_flows": contract.selected_color_flow_ids,
        }
    if cell.workload is Workload.ALL_FLOW:
        return {
            "helicities": contract.all_flow_helicity_ids,
            "color_flows": None,
        }
    raise RunnerError("LC measurement has an invalid workload")


def derive_selector_contract(
    runtime: RuntimeLike,
    points: object,
) -> SelectorContract:
    """Select one deterministic flow and one nonzero fixed helicity."""

    physics = runtime.physics
    color_flows = tuple(getattr(physics, "color_flows", ()))
    helicities = tuple(getattr(physics, "helicities", ()))
    particles = tuple(getattr(physics, "external_particles", ()))
    if not color_flows or not helicities or not particles:
        raise RunnerError("LC selector derivation requires complete physical axes")

    selected_flow = color_flows[0]
    resolved = runtime.evaluate_resolved(points, color_flows=(str(selected_flow.id),))
    values = getattr(resolved, "values", ())
    chosen_index: int | None = None
    for helicity_index, _helicity in enumerate(helicities):
        magnitude = sum(
            abs(complex(point[helicity_index][color_index]))
            for point in values
            for color_index in range(len(point[helicity_index]))
        )
        if magnitude > ABSOLUTE_TOLERANCE:
            chosen_index = helicity_index
            break
    if chosen_index is None:
        raise RunnerError("no nonzero fixed-helicity selector exists at report point")
    helicity = helicities[chosen_index]
    labels = tuple(int(particle.label) for particle in particles)
    states = tuple(int(value) for value in helicity.values)
    if len(labels) != len(states):
        raise RunnerError(
            "helicity source-state axis does not match external particles"
        )
    return SelectorContract(
        selected_color_flow_ids=(str(selected_flow.id),),
        selected_color_words=(
            tuple(int(label) for label in selected_flow.word),
        ),
        all_flow_helicity_ids=(str(helicity.id),),
        all_flow_source_helicities=tuple(zip(labels, states, strict=True)),
        point_digest=point_digest(points),
    )


def validate_selector_contract(
    runtime: RuntimeLike,
    contract: SelectorContract,
    points: object,
) -> None:
    if point_digest(points) != contract.point_digest:
        raise RunnerError("selector contract and measurement point differ")
    physics = runtime.physics
    colors = {
        str(flow.id): tuple(int(label) for label in flow.word)
        for flow in getattr(physics, "color_flows", ())
    }
    for identifier, word in zip(
        contract.selected_color_flow_ids,
        contract.selected_color_words,
        strict=True,
    ):
        if colors.get(identifier) != word:
            raise RunnerError(
                f"artifact does not expose selected physical flow {identifier!r}"
            )
    helicities = {
        str(item.id): tuple(int(value) for value in item.values)
        for item in getattr(physics, "helicities", ())
    }
    labels = tuple(
        int(particle.label)
        for particle in getattr(physics, "external_particles", ())
    )
    expected_states = dict(contract.all_flow_source_helicities)
    expected = tuple(expected_states[label] for label in labels)
    for identifier in contract.all_flow_helicity_ids:
        if helicities.get(identifier) != expected:
            raise RunnerError(
                f"artifact does not expose selected physical helicity {identifier!r}"
            )


def resolved_sum_validation(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    selector_contract: SelectorContract | None,
) -> dict[str, object]:
    selectors = _selector_kwargs(cell, selector_contract)
    optimized = runtime.evaluate(points, **selectors)
    resolved = runtime.evaluate_resolved(points, **selectors)
    totals = tuple(resolved.total())
    if len(optimized) != len(totals):
        raise RunnerError("optimized and resolved evaluations have different lengths")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for optimized_value, resolved_value in zip(optimized, totals, strict=True):
        absolute = abs(complex(optimized_value) - complex(resolved_value))
        relative = absolute / max(abs(complex(optimized_value)), 1.0e-300)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    passed = (
        maximum_absolute <= ABSOLUTE_TOLERANCE
        or maximum_relative <= RELATIVE_TOLERANCE
    )
    return {
        "status": (
            ResultStatus.OK.value
            if passed
            else ResultStatus.VALIDATION_FAILED.value
        ),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
    }


def pointwise_validation(candidate: float, baseline: float) -> dict[str, object]:
    absolute = abs(candidate - baseline)
    relative = absolute / max(abs(baseline), 1.0e-300)
    passed = absolute <= ABSOLUTE_TOLERANCE or relative <= RELATIVE_TOLERANCE
    return {
        "status": (
            ResultStatus.OK.value
            if passed
            else ResultStatus.VALIDATION_FAILED.value
        ),
        "candidate": candidate,
        "baseline": baseline,
        "absolute_difference": absolute,
        "relative_difference": relative,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
    }


def _benchmark_measurement(
    benchmark: object,
    *,
    matrix_element: float,
) -> dict[str, object]:
    uncertainty = benchmark.uncertainty
    return {
        "status": ResultStatus.OK.value,
        "wall_seconds_per_point": float(benchmark.wall_time_per_point),
        "execution_seconds_per_point": (
            None
            if benchmark.evaluator_time_per_point is None
            else float(benchmark.evaluator_time_per_point)
        ),
        "matrix_element": matrix_element,
        "sample_count": int(benchmark.sample_count),
        "standard_error_seconds_per_point": float(uncertainty.standard_error),
        "relative_standard_error": float(uncertainty.relative_standard_error),
    }


def profile_runtime(
    runtime: RuntimeLike,
    points: object,
    *,
    cell: CellSpec,
    benchmark_config: object,
    selector_contract: SelectorContract | None,
) -> dict[str, object]:
    from pyamplicol.api import BenchmarkRunner

    validate_runtime_contract(cell, runtime)
    if selector_contract is not None:
        validate_selector_contract(runtime, selector_contract, points)
    selectors = _selector_kwargs(cell, selector_contract)
    values = runtime.evaluate(points, **selectors)
    if not values:
        raise RunnerError("runtime returned no matrix elements")
    selected_config = replace(
        benchmark_config,
        helicity_ids=tuple(selectors["helicities"] or ()),
        color_flow_ids=tuple(selectors["color_flows"] or ()),
    )
    benchmark = BenchmarkRunner(selected_config).run(runtime, points=points)
    result = _benchmark_measurement(
        benchmark,
        matrix_element=_real_nonnegative(values[0]),
    )
    result["resolved_sum_validation"] = resolved_sum_validation(
        runtime,
        points,
        cell=cell,
        selector_contract=selector_contract,
    )
    if result["resolved_sum_validation"]["status"] != ResultStatus.OK.value:
        result["status"] = ResultStatus.VALIDATION_FAILED.value
    return result


def runtime_validation_points(runtime: object) -> object:
    backend = getattr(runtime, "_backend", None)
    operation = getattr(backend, "validation_momenta", None)
    if not callable(operation):
        raise RunnerError("artifact does not retain deterministic validation momenta")
    points = operation()
    if points is None:
        raise RunnerError("artifact validation momenta are unavailable")
    return points


def _single_process_id(artifact_path: Path, fallback: str) -> str:
    manifest_path = artifact_path / "artifact.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        processes = manifest["processes"]
        if isinstance(processes, list) and len(processes) == 1:
            identifier = processes[0]["id"]
            if isinstance(identifier, str) and identifier:
                return identifier
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        pass
    return fallback


def generate_artifact(
    cell: CellSpec,
    destination: Path,
    *,
    settings: RunnerSettings,
    repo_root: Path,
    prepared_model_path: Path | None = None,
) -> GeneratedArtifact:
    """Generate one complete-coverage artifact and time process generation only."""

    from pyamplicol.api import Generator, ModelSource
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import config_to_dict, resolve_config

    values = config_values(cell, settings, repo_root=repo_root)
    resolution = resolve_config(values, action=Action.GENERATE, base_dir=repo_root)
    assert cell.measurement.model is not None
    source_path = _model_source_path(repo_root, cell.measurement.model)
    source = (
        ModelSource.built_in_sm()
        if source_path is None
        else ModelSource.from_path(source_path)
    )
    model_started = time.perf_counter()
    prepared_execution = cell.measurement.execution_mode in {
        ExecutionMode.EAGER,
        ExecutionMode.RECURRENCE,
    }
    if prepared_model_path is not None:
        if not prepared_model_path.is_file():
            raise RunnerError(f"prepared model does not exist: {prepared_model_path}")
        model = ModelSource.from_path(prepared_model_path)
        preparation_reused = True
    elif prepared_execution and cell.measurement.model is ModelKey.BUILTIN_SM:
        # Omitting the explicit model lets the generation service select the
        # validated wheel-owned built-in-SM JIT O2 prepared pack.
        model = None
        preparation_reused = True
    elif prepared_execution:
        raise RunnerError(
            f"{cell.measurement.model.value} {cell.measurement.execution_mode.value} "
            "generation requires a prepared model path"
        )
    else:
        model = source.compile(
            cache_dir=settings.model_cache_dir,
            use_cache=True,
            require_supported=True,
        )
        preparation_reused = False
    model_seconds = time.perf_counter() - model_started
    generation_started = time.perf_counter()
    Generator(resolution).generate(
        cell.process,
        destination,
        model=model,
        mode="replace",
    )
    generation_seconds = time.perf_counter() - generation_started
    return GeneratedArtifact(
        path=destination,
        process_id=_single_process_id(destination, cell.process),
        generation_seconds=generation_seconds,
        model_preparation_seconds=model_seconds,
        model_preparation_reused=preparation_reused,
        requested_config=config_to_dict(resolution.requested),
        effective_config=config_to_dict(resolution.effective),
    )


def provenance_payload() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


__all__ = [
    "ABSOLUTE_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "GeneratedArtifact",
    "RunnerError",
    "RunnerSettings",
    "SelectorContract",
    "config_values",
    "derive_selector_contract",
    "generate_artifact",
    "point_digest",
    "pointwise_validation",
    "profile_runtime",
    "provenance_payload",
    "resolved_sum_validation",
    "runtime_validation_points",
    "validate_artifact_contract",
    "validate_runtime_contract",
    "validate_selector_contract",
]
