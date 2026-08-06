#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Developer-only first real LC gate for the on-the-fly evaluator.

One topology-replay ``d d~ > t t~ g g`` artifact must serve both intended
workloads: selected-flow/helicity-sum and all-flow/single-helicity.  Dense
correctness against the production :class:`pyamplicol.Runtime` is mandatory;
the timing comparison that follows is diagnostic only.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyamplicol.generation.service as generation_service  # noqa: E402
from pyamplicol import (  # noqa: E402
    BenchmarkRunner,
    CompiledModel,
    Generator,
    ModelSource,
    ProcessRequest,
    Runtime,
)
from pyamplicol._internal.versions import (  # noqa: E402
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
)
from pyamplicol.api.errors import ArtifactError  # noqa: E402
from pyamplicol.artifacts import confined_path, load_manifest  # noqa: E402
from pyamplicol.config import (  # noqa: E402
    BenchmarkConfig,
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation.evaluator_container import (  # noqa: E402
    PacbinError,
    PacbinMemberKind,
    PacbinReader,
)
from pyamplicol.models.builtin.validation import generic_validation_point  # noqa: E402
from pyamplicol.processes import canonical_process_key  # noqa: E402
from tools.ci.memory_watchdog import (  # noqa: E402
    GIB,
    DarwinPhysicalFootprintProbe,
    PhysicalFootprintProbe,
    run_guarded,
)
from tools.performance_report.runner import (  # noqa: E402
    point_digest,
    pointwise_validation,
)

PROCESS = "d d~ > t t~ g g"
PROCESS_KEY = canonical_process_key(PROCESS)
PROCESS_ID = "otf_dd_tt_gg"
EXTERNAL_PDGS = (1, -1, 6, -6, 21, 21)
ON_THE_FLY_CAPABILITIES = tuple(
    sorted(
        (
            ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
            ON_THE_FLY_RUNTIME_CAPABILITY,
        )
    )
)
ON_THE_FLY_SEED_MEMBER_PATH = "on-the-fly/process-seed-v1.bin"
FLOW_ID = "flow:2,5,6,4,3,1"
FLOW_WORD = (2, 5, 6, 4, 3, 1)
HELICITY_ID = "h:-1,+1,-1,+1,-1,+1"
HELICITY_VALUES = (-1, 1, -1, 1, -1, 1)
AMPICOL_CELL_ID = "reference-amplicol-lc-n4-dd-tt-jets-selected-flow"
SEEDS = (101, 211, 307, 401, 503, 607, 709, 811)
REL_TOL = 1.0e-12
AMPICOL_REL_TOL = 1.0e-8
WATCHDOG_BYTES = 30 * GIB
WARMUPS = 2
MINIMUM_SAMPLES = 5
WORK_CENSUS_BASIS = "fully-resident-query-local-trace-v1"
FAMILY_WORK_CENSUS_BASIS = "shared-query-family-union-v1"
OPERATION_COUNT_FIELDS = (
    "source_operation_count",
    "contribution_operation_count",
    "finalization_operation_count",
    "closure_operation_count",
)
WORK_CENSUS_COUNT_FIELDS = (
    "logical_current_count",
    "resident_current_count",
    "resident_current_component_count",
    *OPERATION_COUNT_FIELDS,
    "total_kernel_application_count",
    "semantic_executor_binding_count",
    "distinct_prepared_executor_count",
)
PUBLIC_PROFILE_WORK_FIELDS = tuple(
    f"recurrence_{role}_{unit}_count"
    for role in ("source", "contribution", "finalization", "closure")
    for unit in ("call", "row")
)
FAMILY_CENSUS_COUNT_FIELDS = (
    "query_count",
    "source_frame_partition_count",
    "projection_applied_query_count",
    "projection_pre_current_count",
    "projection_pre_contribution_count",
    "projection_pre_closure_count",
    "projection_post_current_count",
    "projection_post_contribution_count",
    "projection_post_closure_count",
    "dynamic_current_occurrence_count",
    "dynamic_current_component_occurrence_count",
    "dynamic_source_rows",
    "dynamic_contribution_rows",
    "dynamic_finalization_rows",
    "dynamic_closure_rows",
    "dynamic_source_calls",
    "dynamic_contribution_calls",
    "dynamic_finalization_calls",
    "dynamic_closure_calls",
    "union_unique_current_count",
    "union_unique_current_component_count",
    "union_source_rows",
    "union_contribution_rows",
    "union_finalization_rows",
    "union_closure_rows",
    "union_amplitude_destination_count",
    "union_source_executor_call_groups",
    "union_contribution_executor_call_groups",
    "union_finalization_executor_call_groups",
    "union_closure_executor_call_groups",
)
MATERIALIZED_PROCESS_LANE_SYMBOLS = (
    "GenerationBackend._prepare_process_construction",
    "build_color_plan",
    "compile_generic_dag",
    "_invoke_rust_eager_lowering_v1",
    "_invoke_rust_recurrence_lowering_v2",
    "run_generic_dag_numerical_current_warmup",
    "run_recurrence_numerical_current_warmup",
)
_FORBIDDEN_MATERIALIZATION_FIELDS = frozenset(
    {
        "color_components",
        "color_plan",
        "compiled",
        "compiled_execution",
        "compiled_plan",
        "dag",
        "dag_summary",
        "direct_plan",
        "direct_recurrence_plan",
        "eager",
        "eager_plan",
        "eager_runtime",
        "generic_dag",
        "helicity_components",
        "lc_topology_replay",
        "recurrence",
        "recurrence_plan",
        "recurrence_runtime",
        "runtime_schema",
        "stages",
        "structural",
        "structural_proof",
        "structural_source_proof",
    }
)
Point = tuple[tuple[float, ...], ...]
Points = tuple[Point, ...]


class GateError(RuntimeError):
    """The fixed gate could not produce valid evidence."""


@dataclass(frozen=True, slots=True)
class RetainedInputs:
    builder: object
    template: object
    direct_json: bytes
    pack_digest: str


@dataclass(frozen=True, slots=True)
class Query:
    flow_id: str
    flow_index: int
    helicity_id: str
    helicities: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.flow_id}|{self.helicity_id}"


@dataclass(frozen=True, slots=True)
class PublicEvaluation:
    total: tuple[object, ...]
    resolved: object


@dataclass(frozen=True, slots=True)
class DualAuthorityCorrectness:
    recurrence_selected: PublicEvaluation
    recurrence_all_flow: PublicEvaluation
    compiled_selected: PublicEvaluation
    compiled_all_flow: PublicEvaluation
    on_the_fly_selected: PublicEvaluation
    on_the_fly_all_flow: PublicEvaluation
    public_correctness: dict[str, object]
    clear_checks: dict[str, object]


@dataclass(frozen=True, slots=True)
class PublicGatePhase:
    runtime: object
    selected_recurrence_runtime: object
    all_flow_recurrence_runtime: object
    compiled_selected_runtime: object
    compiled_all_flow_runtime: object
    comparator_selectors: tuple[tuple[str, tuple[object, object]], ...]
    points: Points
    timing_points: Points
    correctness: DualAuthorityCorrectness
    artifact_contract: dict[str, object]
    load_seconds: float
    public_profiles: dict[str, object]
    timings: dict[str, dict[str, dict[str, object]]]


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument(
        "--prepared-model",
        type=Path,
        help="explicit prepared built-in-SM .pyamplicol-model bundle",
    )
    candidate_source.add_argument(
        "--candidate-artifact",
        type=Path,
        help=(
            "reuse an existing compact on-the-fly artifact; valid only with "
            "--public-correctness-only"
        ),
    )
    parser.add_argument(
        "--amplicol-result",
        type=Path,
        help="stored successful AmpliCol worker-result.json (optional sanity point)",
    )
    parser.add_argument(
        "--recurrence-selected-artifact",
        required=True,
        type=Path,
        help=(
            "explicit LC topology-replay recurrence comparator for the "
            "selected-flow/helicity-sum workload"
        ),
    )
    parser.add_argument(
        "--recurrence-all-flow-artifact",
        required=True,
        type=Path,
        help=(
            "explicit LC all-flow-union recurrence comparator for the "
            "all-flow/single-helicity workload"
        ),
    )
    parser.add_argument(
        "--compiled-selected-artifact",
        required=True,
        type=Path,
        help="explicit LC compiled selected-flow/helicity-sum comparator",
    )
    parser.add_argument(
        "--compiled-all-flow-artifact",
        required=True,
        type=Path,
        help="explicit LC compiled all-flow/single-helicity comparator",
    )
    parser.add_argument("--target-runtime", type=_positive_float, default=1.0)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument(
        "--public-correctness-only",
        action="store_true",
        help=(
            "provisional developer lane: public dual-authority correctness, "
            "profiles, and BenchmarkRunner timings only; never creates a private "
            "probe carrier or private census"
        ),
    )
    parser.add_argument(
        "--bypass-color-projection",
        action="store_true",
        help=(
            "diagnostic: execute the selected graph before late color-alias projection"
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _on_the_fly_config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=True, post_build_validation=True
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode="on-the-fly",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _recurrence_probe_config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout="topology-replay"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _points() -> Points:
    return tuple(
        tuple(tuple(map(float, particle.momentum)) for particle in point)
        for point in (
            generic_validation_point(PROCESS, sqrt_s=1000.0, seed=seed)
            for seed in SEEDS
        )
    )


def _repeat(points: Points, count: int) -> Points:
    if not points or count <= 0:
        raise GateError("point repetition needs non-empty points and a positive count")
    return tuple(points[index % len(points)] for index in range(count))


def _flatten(points: Points) -> list[float]:
    return [x for point in points for momentum in point for x in momentum]


def _real(value: object, label: str) -> float:
    try:
        result = complex(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise GateError(f"{label} is not numeric") from error
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise GateError(f"{label} is not finite")
    if result.imag != 0.0:
        raise GateError(f"{label} is not real")
    return result.real


def _series(
    candidate: Sequence[object],
    reference: Sequence[object],
    label: str,
    tolerance: float = REL_TOL,
) -> dict[str, object]:
    if not candidate or len(candidate) != len(reference):
        raise GateError(f"{label} has incompatible or empty point axes")
    rows = tuple(
        pointwise_validation(
            _real(left, "candidate"),
            _real(right, "reference"),
            relative_tolerance=tolerance,
        )
        for left, right in zip(candidate, reference, strict=True)
    )
    failed = next(
        (index for index, row in enumerate(rows) if row["status"] != "ok"), None
    )
    if failed is not None:
        raise GateError(
            f"{label} disagrees at point {failed}: "
            f"residual={rows[failed]['conditioned_residual']!r}"
        )
    worst = max(
        range(len(rows)), key=lambda index: float(rows[index]["conditioned_residual"])
    )
    return {
        "checks": len(rows),
        "maximum_conditioned_residual": rows[worst]["conditioned_residual"],
        "maximum_absolute_difference": max(
            float(row["absolute_difference"]) for row in rows
        ),
        "worst_point_index": worst,
        "worst": rows[worst],
    }


def _summaries(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise GateError("component comparison set is empty")
    worst = max(
        range(len(rows)),
        key=lambda index: float(rows[index]["maximum_conditioned_residual"]),
    )
    return {
        "component_count": len(rows),
        "checks": sum(int(row["checks"]) for row in rows),
        "maximum_conditioned_residual": rows[worst]["maximum_conditioned_residual"],
        "maximum_absolute_difference": max(
            float(row["maximum_absolute_difference"]) for row in rows
        ),
        "worst_component_index": worst,
        "worst": rows[worst],
    }


def _sum(rows: Sequence[Sequence[object]], point_count: int) -> tuple[float, ...]:
    parsed = tuple(tuple(_real(value, "component") for value in row) for row in rows)
    if not parsed or point_count <= 0 or any(len(row) != point_count for row in parsed):
        raise GateError("component rows do not match the point count")
    return tuple(
        math.fsum(row[point] for row in parsed) for point in range(point_count)
    )


def _calibrate(pilot_seconds: float, target_seconds: float) -> int:
    if (
        not math.isfinite(pilot_seconds)
        or pilot_seconds <= 0.0
        or not math.isfinite(target_seconds)
        or target_seconds <= 0.0
    ):
        raise GateError("benchmark pilot and target must be positive and finite")
    repetitions = max(1, math.ceil(target_seconds / pilot_seconds))
    if repetitions >= 1 << 32:
        raise GateError("benchmark repetitions exceed the native u32 domain")
    return repetitions


def _canonical_direct(value: object) -> bytes:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise GateError("captured direct catalog has no mapping")
    return json.dumps(
        to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _prepared_model_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise GateError(f"prepared model does not exist: {path}") from error
    if not resolved.is_file():
        raise GateError(f"prepared model is not a regular file: {resolved}")
    if not resolved.name.lower().endswith(".pyamplicol-model"):
        raise GateError("prepared model must end with '.pyamplicol-model'")
    return resolved


def _candidate_artifact_path(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise GateError(f"candidate artifact does not exist: {path}") from error
    if not resolved.is_dir():
        raise GateError(f"candidate artifact is not a directory: {resolved}")
    return resolved


def _validate_mode_arguments(arguments: argparse.Namespace) -> None:
    prepared_model = getattr(arguments, "prepared_model", None)
    candidate_artifact = getattr(arguments, "candidate_artifact", None)
    public_only = bool(getattr(arguments, "public_correctness_only", False))
    if public_only:
        if getattr(arguments, "bypass_color_projection", False):
            raise GateError(
                "--bypass-color-projection is unavailable with "
                "--public-correctness-only"
            )
        if getattr(arguments, "amplicol_result", None) is not None:
            raise GateError(
                "--amplicol-result needs the full private gate and is unavailable "
                "with --public-correctness-only"
            )
        return
    if prepared_model is None:
        raise GateError("the full gate requires --prepared-model")
    if candidate_artifact is not None:
        raise GateError(
            "--candidate-artifact is valid only with --public-correctness-only"
        )


def _reject_materialization_fields(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_name, nested in value.items():
            name = str(raw_name).lower().replace("-", "_")
            path = f"{label}.{raw_name}"
            if name in _FORBIDDEN_MATERIALIZATION_FIELDS:
                raise GateError(
                    f"{label} contains forbidden materialization field {path!r}"
                )
            _reject_materialization_fields(nested, path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_materialization_fields(nested, f"{label}[{index}]")


def _on_the_fly_source_projections(
    args: tuple[object, ...], kwargs: Mapping[str, object]
) -> tuple[Mapping[str, object], ...]:
    raw = args[0] if args else kwargs.get("ordered_source_projection_jsons")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise GateError("on-the-fly batch seed binding omitted its source projections")
    result: list[Mapping[str, object]] = []
    for index, encoded in enumerate(raw):
        if not isinstance(encoded, bytes):
            raise GateError(
                f"on-the-fly source projection {index} is not encoded bytes"
            )
        try:
            projection = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GateError(
                f"on-the-fly source projection {index} is not valid JSON"
            ) from error
        record = _mapping(projection, f"on-the-fly source projection {index}")
        _reject_materialization_fields(record, f"on-the-fly source projection {index}")
        result.append(record)
    return tuple(result)


def _materialized_process_lane_patch_targets() -> tuple[tuple[str, object, str], ...]:
    return (
        (
            "GenerationBackend._prepare_process_construction",
            generation_service.GenerationBackend,
            "_prepare_process_construction",
        ),
        ("build_color_plan", generation_service, "build_color_plan"),
        ("compile_generic_dag", generation_service, "compile_generic_dag"),
        (
            "_invoke_rust_eager_lowering_v1",
            generation_service,
            "_invoke_rust_eager_lowering_v1",
        ),
        (
            "_invoke_rust_recurrence_lowering_v2",
            generation_service,
            "_invoke_rust_recurrence_lowering_v2",
        ),
        (
            "run_generic_dag_numerical_current_warmup",
            generation_service,
            "run_generic_dag_numerical_current_warmup",
        ),
        (
            "run_recurrence_numerical_current_warmup",
            generation_service,
            "run_recurrence_numerical_current_warmup",
        ),
    )


def _generate_on_the_fly(
    artifact: Path, model: CompiledModel
) -> tuple[float, dict[str, int]]:
    seed_batch_binding_call_count = 0
    seed_build_count = 0
    original_seed_builder = (
        generation_service._invoke_rust_on_the_fly_seed_batch_builder_v1
    )

    def capture_seed_batch(*args: object, **kwargs: object) -> object:
        nonlocal seed_batch_binding_call_count, seed_build_count
        seed_batch_binding_call_count += 1
        projections = _on_the_fly_source_projections(args, kwargs)
        result = original_seed_builder(*args, **kwargs)
        if not isinstance(result, tuple):
            raise GateError("on-the-fly batch seed binding returned a non-tuple")
        if len(result) != len(projections):
            raise GateError(
                "on-the-fly batch seed binding changed the process count: "
                f"{len(projections)} projection(s), {len(result)} seed(s)"
            )
        seed_build_count += len(result)
        return result

    materialized_lane_call_counts = {
        name: 0 for name in MATERIALIZED_PROCESS_LANE_SYMBOLS
    }

    def reject_materialized_lane(
        name: str,
    ) -> Callable[..., object]:
        def reject(*_args: object, **_kwargs: object) -> object:
            materialized_lane_call_counts[name] += 1
            raise GateError(
                f"on-the-fly generation entered materialized process lane {name}"
            )

        return reject

    started = time.perf_counter()
    with ExitStack() as patches:
        patches.enter_context(
            mock.patch.object(
                generation_service,
                "_invoke_rust_on_the_fly_seed_batch_builder_v1",
                capture_seed_batch,
            )
        )
        for name, owner, attribute in _materialized_process_lane_patch_targets():
            patches.enter_context(
                mock.patch.object(
                    owner,
                    attribute,
                    reject_materialized_lane(name),
                )
            )
        Generator(_on_the_fly_config()).generate(
            ProcessRequest.parse(PROCESS, name=PROCESS_ID), artifact, model=model
        )
    if seed_batch_binding_call_count != 1 or seed_build_count != 1:
        raise GateError(
            "expected one on-the-fly batch binding call producing one seed, "
            f"observed {seed_batch_binding_call_count} call(s) and "
            f"{seed_build_count} seed(s)"
        )
    return time.perf_counter() - started, {
        "expanded_process_count": 1,
        "on_the_fly_seed_batch_binding_call_count": (seed_batch_binding_call_count),
        "on_the_fly_seed_build_count": seed_build_count,
        "recurrence_lowering_call_count": materialized_lane_call_counts[
            "_invoke_rust_recurrence_lowering_v2"
        ],
        "materialized_process_lane_count": sum(materialized_lane_call_counts.values()),
    }


def _generate_probe_carrier(artifact: Path, model: CompiledModel) -> RetainedInputs:
    captured: list[RetainedInputs] = []
    original = generation_service._invoke_rust_recurrence_lowering_v2

    def capture(*args: object, **kwargs: object) -> object:
        if len(args) < 4:
            raise GateError("lowering call omitted its canonical inputs")
        direct = _canonical_direct(args[2])
        supplied = kwargs.get("direct_template_catalog_json")
        if supplied is not None and supplied != direct:
            raise GateError("lowering direct-catalog bytes are not canonical")
        captured.append(RetainedInputs(args[0], args[1], direct, str(args[3])))
        return original(*args, **kwargs)

    with mock.patch.object(
        generation_service, "_invoke_rust_recurrence_lowering_v2", capture
    ):
        Generator(_recurrence_probe_config()).generate(
            ProcessRequest.parse(PROCESS, name=PROCESS_ID), artifact, model=model
        )
    if len(captured) != 1:
        raise GateError(f"expected one lowering call, observed {len(captured)}")
    return captured[0]


def _generate_candidate_with_prepared_model(
    artifact: Path, prepared_model: Path
) -> tuple[float, dict[str, int], CompiledModel, Path, float]:
    resolved = _prepared_model_path(prepared_model)
    started = time.perf_counter()
    model = ModelSource.from_path(resolved).compile()
    load_seconds = time.perf_counter() - started
    generation_seconds, generation_census = _generate_on_the_fly(artifact, model)
    return generation_seconds, generation_census, model, resolved, load_seconds


def _selectors(physics: object) -> tuple[object, object]:
    flows = tuple(
        flow
        for flow in getattr(physics, "color_flows", ())
        if flow.id == FLOW_ID and tuple(flow.word) == FLOW_WORD
    )
    helicities = tuple(
        helicity
        for helicity in getattr(physics, "helicities", ())
        if helicity.id == HELICITY_ID and tuple(helicity.values) == HELICITY_VALUES
    )
    if len(flows) != 1 or len(helicities) != 1:
        raise GateError("artifact does not expose the exact fixed flow and helicity")
    if helicities[0].structural_zero:
        raise GateError("fixed all-flow helicity is structurally zero")
    return flows[0], helicities[0]


def _query(flow: object, helicity: object) -> Query:
    index = flow.index
    values = tuple(helicity.values)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise GateError("color-flow native index is invalid")
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise GateError("helicity values are invalid")
    return Query(flow.id, index, helicity.id, values)


def _probe() -> Callable[..., dict[str, Any]]:
    candidate = getattr(
        importlib.import_module("pyamplicol._rusticol"),
        "_on_the_fly_artifact_probe_v1",
        None,
    )
    if not callable(candidate):
        raise GateError("native extension lacks on-the-fly benchmark test support")
    return candidate


def _family_probe() -> Callable[..., dict[str, Any]]:
    candidate = getattr(
        importlib.import_module("pyamplicol._rusticol"),
        "_on_the_fly_query_family_census_v1",
        None,
    )
    if not callable(candidate):
        raise GateError("native extension lacks on-the-fly family-census support")
    return candidate


def _invoke(
    probe: Callable[..., dict[str, Any]],
    artifact: Path,
    retained: RetainedInputs,
    query: Query,
    points: Points,
    *,
    repetitions: int = 0,
    enable_color_projection: bool = True,
    query_family: Sequence[Query] | None = None,
) -> dict[str, Any]:
    benchmark = repetitions > 0
    return probe(
        str(artifact),
        PROCESS_ID,
        retained.builder,
        retained.template,
        retained.direct_json,
        retained.pack_digest,
        query.flow_index,
        list(query.helicities),
        _flatten(points),
        len(points),
        benchmark=benchmark,
        benchmark_warmup_repetitions=WARMUPS if benchmark else 0,
        benchmark_repetitions=repetitions,
        collect_current_diagnostics=False,
        enable_color_projection=enable_color_projection,
        query_family=(
            None
            if query_family is None
            else [
                (member.flow_index, list(member.helicities)) for member in query_family
            ]
        ),
    )


def _probe_values(
    report: Mapping[str, object], point_count: int, repetitions: int = 0
) -> tuple[float, ...]:
    poison = (
        "direct_plan_load_attempts",
        "direct_plan_decode_attempts",
        "direct_plan_materialization_attempts",
        "established_builder_attempts",
    )
    benchmark = repetitions > 0
    cycles = WARMUPS + repetitions if benchmark else 1
    hits = cycles if benchmark else 0
    if (
        report.get("process_id") != PROCESS_ID
        or report.get("point_count") != point_count
        or report.get("trace_build_count") != 1
        or report.get("trace_cache_hit_count") != hits
        or report.get("momentum_fill_count") != cycles
        or report.get("currents") != []
        or any(report.get(name) != 0 for name in poison)
    ):
        raise GateError("hidden query violated its build/cache/fill/poison contract")
    _work_census(report)
    values = report.get("normalized_values")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise GateError("hidden query has no normalized values")
    parsed = tuple(_real(value, "hidden value") for value in values)
    if len(parsed) != point_count:
        raise GateError("hidden query returned the wrong point count")
    elapsed = report.get("benchmark_elapsed_seconds")
    per_point = report.get("benchmark_seconds_per_point")
    if benchmark:
        if not isinstance(elapsed, (int, float)) or not isinstance(
            per_point, (int, float)
        ):
            raise GateError("hidden benchmark omitted timings")
        expected = float(elapsed) / (repetitions * point_count)
        valid_timing = (
            math.isfinite(float(elapsed))
            and float(elapsed) >= 0.0
            and math.isclose(float(per_point), expected, rel_tol=1.0e-15, abs_tol=0.0)
        )
        if not valid_timing:
            raise GateError("hidden benchmark timings are inconsistent")
    elif elapsed is not None or per_point is not None:
        raise GateError("untimed hidden query returned timings")
    return parsed


def _work_census(report: Mapping[str, object]) -> dict[str, object]:
    if report.get("work_census_basis") != WORK_CENSUS_BASIS:
        raise GateError("hidden query has an unknown work-census basis")
    counts: dict[str, int] = {}
    for name in WORK_CENSUS_COUNT_FIELDS:
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"hidden query work census has invalid {name}")
        counts[name] = value
    if counts["resident_current_count"] != counts["logical_current_count"]:
        raise GateError("hidden query does not use a fully resident current arena")
    if counts["resident_current_component_count"] < counts["resident_current_count"]:
        raise GateError("hidden query resident current components are inconsistent")
    if counts["total_kernel_application_count"] != sum(
        counts[name] for name in OPERATION_COUNT_FIELDS
    ):
        raise GateError("hidden query kernel and operation counts disagree")
    semantic = counts["semantic_executor_binding_count"]
    prepared = counts["distinct_prepared_executor_count"]
    total = counts["total_kernel_application_count"]
    if not 0 < prepared <= semantic <= total:
        raise GateError("hidden query prepared-executor counts are inconsistent")
    return {"work_census_basis": WORK_CENSUS_BASIS, **counts}


def _family_probe_result(
    report: Mapping[str, object],
    queries: Sequence[Query],
    point_count: int,
    repetitions: int = 0,
) -> dict[str, object]:
    if not queries:
        raise GateError("hidden family report has no requested queries")
    benchmark = repetitions > 0
    cycles = WARMUPS + repetitions if benchmark else 1
    poison = (
        "direct_plan_load_attempts",
        "direct_plan_decode_attempts",
        "direct_plan_materialization_attempts",
        "established_builder_attempts",
    )
    if (
        report.get("process_id") != PROCESS_ID
        or report.get("point_count") != point_count
        or report.get("work_census_basis") != FAMILY_WORK_CENSUS_BASIS
        or report.get("trace_build_count") != len(queries)
        or report.get("trace_cache_hit_count") != 0
        or report.get("momentum_fill_count") != cycles
        or report.get("currents") != []
        or any(report.get(name) != 0 for name in poison)
    ):
        raise GateError("hidden family violated its build/cache/fill/poison contract")
    family = report.get("query_family")
    if not isinstance(family, Mapping):
        raise GateError("hidden family report is absent")
    raw_census = family.get("census")
    if not isinstance(raw_census, Mapping):
        raise GateError("hidden family census is absent")
    census: dict[str, int] = {}
    for name in FAMILY_CENSUS_COUNT_FIELDS:
        value = raw_census.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"hidden family census has invalid {name}")
        census[name] = value
    if (
        census["query_count"] != len(queries)
        or census["source_frame_partition_count"] != 1
        or census["union_amplitude_destination_count"] != len(queries)
    ):
        raise GateError("hidden family census has the wrong query/source shape")
    for role in ("source", "contribution", "finalization", "closure"):
        if (
            family.get(f"execution_{role}_rows") != census[f"union_{role}_rows"]
            or family.get(f"execution_{role}_calls")
            != census[f"union_{role}_executor_call_groups"]
        ):
            raise GateError(f"hidden family execution disagrees for {role}")
    union_total = sum(
        census[f"union_{role}_rows"]
        for role in ("source", "contribution", "finalization", "closure")
    )
    if (
        report.get("source_operation_count") != census["union_source_rows"]
        or report.get("contribution_operation_count")
        != census["union_contribution_rows"]
        or report.get("finalization_operation_count")
        != census["union_finalization_rows"]
        or report.get("closure_operation_count") != census["union_closure_rows"]
        or report.get("total_kernel_application_count") != union_total
        or family.get("execution_cache_hit") is not benchmark
        or family.get("private_timing_excludes_source_crossing") is not True
    ):
        raise GateError("hidden family top-level execution census is inconsistent")
    cold_prepare = family.get("cold_prepare_seconds")
    if (
        isinstance(cold_prepare, bool)
        or not isinstance(cold_prepare, (int, float))
        or not math.isfinite(float(cold_prepare))
        or float(cold_prepare) < 0.0
    ):
        raise GateError("hidden family cold preparation timing is invalid")

    raw_queries = family.get("queries")
    if not isinstance(raw_queries, Sequence) or isinstance(raw_queries, (str, bytes)):
        raise GateError("hidden family query outputs are absent")
    if len(raw_queries) != len(queries):
        raise GateError("hidden family returned the wrong query count")
    rows: list[dict[str, object]] = []
    for expected, raw in zip(queries, raw_queries, strict=True):
        if not isinstance(raw, Mapping):
            raise GateError("hidden family query output is not a mapping")
        if (
            raw.get("selected_public_flow_id") != expected.flow_index
            or tuple(raw.get("public_helicities", ())) != expected.helicities
            or not isinstance(raw.get("query_digest"), str)
        ):
            raise GateError("hidden family query identity/order changed")
        raw_amplitudes = raw.get("raw_amplitudes")
        normalized_values = raw.get("normalized_values")
        if (
            not isinstance(raw_amplitudes, Sequence)
            or isinstance(raw_amplitudes, (str, bytes))
            or not isinstance(normalized_values, Sequence)
            or isinstance(normalized_values, (str, bytes))
            or len(raw_amplitudes) != point_count
            or len(normalized_values) != point_count
        ):
            raise GateError("hidden family query returned the wrong point axis")
        parsed_raw = tuple(
            complex(
                _real(pair[0], "family raw real"),
                _real(pair[1], "family raw imag"),
            )
            for pair in raw_amplitudes
            if isinstance(pair, Sequence)
            and not isinstance(pair, (str, bytes))
            and len(pair) == 2
        )
        if len(parsed_raw) != point_count:
            raise GateError("hidden family query has malformed raw amplitudes")
        parsed_normalized = tuple(
            _real(value, "family normalized value") for value in normalized_values
        )
        rows.append(
            {
                "query": expected.label,
                "query_digest": raw["query_digest"],
                "raw_amplitudes": parsed_raw,
                "normalized_values": parsed_normalized,
            }
        )

    elapsed = family.get("private_warmed_elapsed_seconds")
    per_point = family.get("private_warmed_seconds_per_point")
    if benchmark:
        if not isinstance(elapsed, (int, float)) or not isinstance(
            per_point, (int, float)
        ):
            raise GateError("hidden family benchmark omitted timings")
        expected = float(elapsed) / (repetitions * point_count)
        if (
            not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
            or not math.isclose(float(per_point), expected, rel_tol=1.0e-15)
        ):
            raise GateError("hidden family benchmark timings are inconsistent")
    elif elapsed is not None or per_point is not None:
        raise GateError("untimed hidden family returned timings")
    return {
        "queries": rows,
        "census": census,
        "union_total_kernel_application_count": union_total,
        "cold_prepare_seconds": float(cold_prepare),
        "private_warmed_elapsed_seconds": elapsed,
        "private_warmed_seconds_per_point": per_point,
        "private_timing_excludes_source_crossing": True,
    }


def _axes(resolved: object) -> tuple[dict[str, int], dict[str, int]]:
    helicities = tuple(resolved.helicity_ids)
    colors = tuple(resolved.color_ids)
    if (
        not helicities
        or not colors
        or len(set(helicities)) != len(helicities)
        or len(set(colors)) != len(colors)
    ):
        raise GateError("resolved axes are empty or ambiguous")
    return (
        {value: index for index, value in enumerate(helicities)},
        {value: index for index, value in enumerate(colors)},
    )


def _resolved_component_checks(
    candidate: object, reference: object, label: str
) -> dict[str, object]:
    candidate_helicities = tuple(candidate.helicity_ids)
    candidate_colors = tuple(candidate.color_ids)
    reference_helicities = tuple(reference.helicity_ids)
    reference_colors = tuple(reference.color_ids)
    candidate_helicity_axis, candidate_color_axis = _axes(candidate)
    reference_helicity_axis, reference_color_axis = _axes(reference)
    if set(candidate_helicities) != set(reference_helicities) or set(
        candidate_colors
    ) != set(reference_colors):
        raise GateError(f"{label} resolved selector axes differ")
    candidate_values = tuple(candidate.values)
    reference_values = tuple(reference.values)
    if not candidate_values or len(candidate_values) != len(reference_values):
        raise GateError(f"{label} resolved point axes differ or are empty")
    rows: list[dict[str, object]] = []
    try:
        for helicity_id in candidate_helicities:
            candidate_helicity_index = candidate_helicity_axis[helicity_id]
            reference_helicity_index = reference_helicity_axis[helicity_id]
            for color_id in candidate_colors:
                candidate_color_index = candidate_color_axis[color_id]
                reference_color_index = reference_color_axis[color_id]
                rows.append(
                    _series(
                        tuple(
                            point[candidate_helicity_index][candidate_color_index]
                            for point in candidate_values
                        ),
                        tuple(
                            point[reference_helicity_index][reference_color_index]
                            for point in reference_values
                        ),
                        f"{label} {helicity_id}/{color_id}",
                    )
                )
    except (IndexError, TypeError) as error:
        raise GateError(f"{label} resolved values have an invalid shape") from error
    return _summaries(rows)


def _public_evaluation(
    runtime: object,
    points: Points,
    workload: str,
    label: str,
) -> PublicEvaluation:
    if workload == "selected_flow_helicity_sum":
        selectors = {"color_flows": (FLOW_ID,)}
        expected_axis = "color_ids"
        expected_ids = (FLOW_ID,)
    elif workload == "all_flow_single_helicity":
        selectors = {"helicities": (HELICITY_ID,)}
        expected_axis = "helicity_ids"
        expected_ids = (HELICITY_ID,)
    else:
        raise GateError(f"unknown public correctness workload: {workload}")
    total = tuple(runtime.evaluate(points, **selectors))
    resolved = runtime.evaluate_resolved(points, **selectors)
    if tuple(getattr(resolved, expected_axis, ())) != expected_ids:
        raise GateError(f"{label} did not resolve the exact public selector ID")
    if len(total) != len(points):
        raise GateError(f"{label} returned the wrong point count")
    return PublicEvaluation(total=total, resolved=resolved)


def _dual_authority_correctness(
    runtime: object,
    selected_recurrence_runtime: object,
    all_flow_recurrence_runtime: object,
    compiled_selected_runtime: object,
    compiled_all_flow_runtime: object,
    points: Points,
) -> DualAuthorityCorrectness:
    recurrence_selected = _public_evaluation(
        selected_recurrence_runtime,
        points,
        "selected_flow_helicity_sum",
        "selected recurrence authority",
    )
    recurrence_all_flow = _public_evaluation(
        all_flow_recurrence_runtime,
        points,
        "all_flow_single_helicity",
        "all-flow recurrence authority",
    )
    compiled_selected = _public_evaluation(
        compiled_selected_runtime,
        points,
        "selected_flow_helicity_sum",
        "selected compiled authority",
    )
    compiled_all_flow = _public_evaluation(
        compiled_all_flow_runtime,
        points,
        "all_flow_single_helicity",
        "all-flow compiled authority",
    )

    authority_agreement = {
        "selected_flow_helicity_sum": {
            "total": _series(
                recurrence_selected.total,
                compiled_selected.total,
                "recurrence/compiled selected total",
            ),
            "resolved": _resolved_component_checks(
                recurrence_selected.resolved,
                compiled_selected.resolved,
                "recurrence/compiled selected resolved",
            ),
        },
        "all_flow_single_helicity": {
            "total": _series(
                recurrence_all_flow.total,
                compiled_all_flow.total,
                "recurrence/compiled all-flow total",
            ),
            "resolved": _resolved_component_checks(
                recurrence_all_flow.resolved,
                compiled_all_flow.resolved,
                "recurrence/compiled all-flow resolved",
            ),
        },
    }

    on_the_fly_selected = _public_evaluation(
        runtime,
        points,
        "selected_flow_helicity_sum",
        "on-the-fly selected candidate",
    )
    on_the_fly_all_flow = _public_evaluation(
        runtime,
        points,
        "all_flow_single_helicity",
        "on-the-fly all-flow candidate",
    )
    public_correctness = {
        "selected_flow_helicity_sum": {
            "recurrence_compiled_authority": authority_agreement[
                "selected_flow_helicity_sum"
            ],
            "total_authority": _series(
                on_the_fly_selected.total,
                recurrence_selected.total,
                "on-the-fly selected total/recurrence authority",
            ),
            "resolved_authority": _resolved_component_checks(
                on_the_fly_selected.resolved,
                recurrence_selected.resolved,
                "on-the-fly selected resolved/recurrence authority",
            ),
            "total_compiled_authority": _series(
                on_the_fly_selected.total,
                compiled_selected.total,
                "on-the-fly selected total/compiled authority",
            ),
            "resolved_compiled_authority": _resolved_component_checks(
                on_the_fly_selected.resolved,
                compiled_selected.resolved,
                "on-the-fly selected resolved/compiled authority",
            ),
            "total_resolved_identity": _series(
                on_the_fly_selected.total,
                on_the_fly_selected.resolved.total(),
                "on-the-fly selected total/resolved",
            ),
        },
        "all_flow_single_helicity": {
            "recurrence_compiled_authority": authority_agreement[
                "all_flow_single_helicity"
            ],
            "total_authority": _series(
                on_the_fly_all_flow.total,
                recurrence_all_flow.total,
                "on-the-fly all-flow total/recurrence authority",
            ),
            "resolved_authority": _resolved_component_checks(
                on_the_fly_all_flow.resolved,
                recurrence_all_flow.resolved,
                "on-the-fly all-flow resolved/recurrence authority",
            ),
            "total_compiled_authority": _series(
                on_the_fly_all_flow.total,
                compiled_all_flow.total,
                "on-the-fly all-flow total/compiled authority",
            ),
            "resolved_compiled_authority": _resolved_component_checks(
                on_the_fly_all_flow.resolved,
                compiled_all_flow.resolved,
                "on-the-fly all-flow resolved/compiled authority",
            ),
            "total_resolved_identity": _series(
                on_the_fly_all_flow.total,
                on_the_fly_all_flow.resolved.total(),
                "on-the-fly all-flow total/resolved",
            ),
        },
    }

    runtime.clear()
    selected_after_clear = _public_evaluation(
        runtime,
        points,
        "selected_flow_helicity_sum",
        "on-the-fly selected candidate after clear",
    )
    all_flow_after_clear = _public_evaluation(
        runtime,
        points,
        "all_flow_single_helicity",
        "on-the-fly all-flow candidate after clear",
    )
    clear_checks = {
        "selected_total": _series(
            selected_after_clear.total,
            recurrence_selected.total,
            "on-the-fly selected total/recurrence authority after clear",
        ),
        "selected_resolved": _resolved_component_checks(
            selected_after_clear.resolved,
            recurrence_selected.resolved,
            "on-the-fly selected resolved/recurrence authority after clear",
        ),
        "selected_total_compiled": _series(
            selected_after_clear.total,
            compiled_selected.total,
            "on-the-fly selected total/compiled authority after clear",
        ),
        "selected_resolved_compiled": _resolved_component_checks(
            selected_after_clear.resolved,
            compiled_selected.resolved,
            "on-the-fly selected resolved/compiled authority after clear",
        ),
        "all_flow_total": _series(
            all_flow_after_clear.total,
            recurrence_all_flow.total,
            "on-the-fly all-flow total/recurrence authority after clear",
        ),
        "all_flow_resolved": _resolved_component_checks(
            all_flow_after_clear.resolved,
            recurrence_all_flow.resolved,
            "on-the-fly all-flow resolved/recurrence authority after clear",
        ),
        "all_flow_total_compiled": _series(
            all_flow_after_clear.total,
            compiled_all_flow.total,
            "on-the-fly all-flow total/compiled authority after clear",
        ),
        "all_flow_resolved_compiled": _resolved_component_checks(
            all_flow_after_clear.resolved,
            compiled_all_flow.resolved,
            "on-the-fly all-flow resolved/compiled authority after clear",
        ),
    }
    return DualAuthorityCorrectness(
        recurrence_selected=recurrence_selected,
        recurrence_all_flow=recurrence_all_flow,
        compiled_selected=compiled_selected,
        compiled_all_flow=compiled_all_flow,
        on_the_fly_selected=on_the_fly_selected,
        on_the_fly_all_flow=on_the_fly_all_flow,
        public_correctness=public_correctness,
        clear_checks=clear_checks,
    )


def _dense_authority(
    selected_runtime: object,
    all_flow_runtime: object,
    point_count: int,
    *,
    compiled_selected_runtime: object | None = None,
    compiled_all_flow_runtime: object | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "authority_kind": "validated_production_pyamplicol",
        "runtime_api": "Runtime.evaluate_resolved",
        "selected_flow_artifact_id": selected_runtime.artifact_id,
        "all_flow_artifact_id": all_flow_runtime.artifact_id,
        "point_count": point_count,
        "certifies": (
            "selected_flow_helicity_sum",
            "all_flow_single_helicity",
        ),
    }
    if (compiled_selected_runtime is None) != (compiled_all_flow_runtime is None):
        raise GateError("dense compiled authority pair is incomplete")
    if compiled_selected_runtime is not None and compiled_all_flow_runtime is not None:
        result.update(
            {
                "authority_execution_modes": ("recurrence", "compiled"),
                "compiled_selected_flow_artifact_id": (
                    compiled_selected_runtime.artifact_id
                ),
                "compiled_all_flow_artifact_id": compiled_all_flow_runtime.artifact_id,
                "authority_cross_check": "componentwise_public_id_alignment",
            }
        )
    return result


def _benchmark_config(
    target: float,
    batch: int,
    *,
    helicities: tuple[str, ...] = (),
    flows: tuple[str, ...] = (),
) -> BenchmarkConfig:
    return BenchmarkConfig(
        target_runtime=target,
        batch_size=batch,
        precision=16,
        warmup_runs=WARMUPS,
        minimum_samples=MINIMUM_SAMPLES,
        helicity_ids=helicities,
        color_flow_ids=flows,
    )


def _public_recurrence_work(result: object) -> dict[str, object] | None:
    breakdown = result.timing_breakdown
    counters = None if breakdown is None else breakdown.counters
    if counters is None:
        return None
    role_counts = {
        "source_calls_per_runtime_call": counters.recurrence_source_calls_per_call,
        "source_rows_per_runtime_call": counters.recurrence_source_rows_per_call,
        "contribution_calls_per_runtime_call": (
            counters.recurrence_contribution_calls_per_call
        ),
        "contribution_rows_per_runtime_call": (
            counters.recurrence_contribution_rows_per_call
        ),
        "finalization_calls_per_runtime_call": (
            counters.recurrence_finalization_calls_per_call
        ),
        "finalization_rows_per_runtime_call": (
            counters.recurrence_finalization_rows_per_call
        ),
        "closure_calls_per_runtime_call": counters.recurrence_closure_calls_per_call,
        "closure_rows_per_runtime_call": counters.recurrence_closure_rows_per_call,
    }
    if not any(value is not None for value in role_counts.values()):
        return None
    return {
        "basis": counters.normalization,
        "semantics": (
            "rows are logical DirectPlan applications; calls are grouped "
            "prepared-backend invocations"
        ),
        **role_counts,
    }


def _public_timing(result: object, execution_mode: str) -> dict[str, object]:
    breakdown = result.timing_breakdown
    if breakdown is None or breakdown.execution_mode != execution_mode:
        raise GateError(
            f"public {execution_mode} benchmark has no matching timing breakdown"
        )
    evaluator_label = (
        "recurrence core"
        if execution_mode == "recurrence"
        else "warmed evaluator envelope"
    )
    return {
        "execution_mode": execution_mode,
        "sample_count": result.sample_count,
        "repetitions_per_sample": result.repetitions_per_sample,
        "wall_seconds_per_point": result.wall_time_per_point,
        "evaluator_seconds_per_point": result.evaluator_time_per_point,
        "evaluator_total_seconds_per_point": result.evaluator_total_time_per_point,
        "clock_attribution": {
            "wall_seconds_per_point": "BenchmarkRunner outer wall clock",
            "evaluator_seconds_per_point": evaluator_label,
            "evaluator_total_seconds_per_point": (
                "independently accumulated warmed evaluator-total clock"
            ),
            "relationship": (
                "all reported clocks are independent; equality is neither assumed "
                "nor asserted"
            ),
        },
        "interrupted": result.interrupted,
        "effective_config": dataclasses.asdict(result.effective_config),
        "uncertainty": dataclasses.asdict(result.uncertainty),
        "recurrence_runtime_work": (
            None if execution_mode == "compiled" else _public_recurrence_work(result)
        ),
    }


def _exact_on_the_fly_capabilities(value: object, label: str) -> None:
    capabilities = _items(value, label)
    if capabilities != ON_THE_FLY_CAPABILITIES:
        raise GateError(
            f"{label} must contain exactly the two LC on-the-fly capabilities"
        )


def _manifest_payload_record(
    manifest: Any,
    relative: str,
    *,
    role: str,
    process_id: str | None,
) -> object:
    records = tuple(
        record
        for record in getattr(manifest, "payloads", ())
        if record.path == relative
    )
    if len(records) != 1:
        raise GateError(
            f"on-the-fly artifact must authenticate exactly one {relative!r} payload"
        )
    record = records[0]
    if record.role != role or record.process_id != process_id or record.executable:
        raise GateError(
            f"authenticated on-the-fly payload {relative!r} has wrong ownership, "
            "role, or executable state"
        )
    return record


def _authenticated_json_payload(
    manifest: Any,
    relative: str,
    *,
    role: str,
    process_id: str | None,
    label: str,
) -> Mapping[str, object]:
    _manifest_payload_record(
        manifest,
        relative,
        role=role,
        process_id=process_id,
    )
    try:
        path = confined_path(manifest.root, relative)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ArtifactError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read authenticated {label}: {relative}") from error
    return _mapping(payload, label)


def _resolved_artifact_relative(
    manifest: Any, base: Path, relative: object, label: str
) -> str:
    if not isinstance(relative, str) or not relative:
        raise GateError(f"{label} is not a non-empty relative path")
    joined = (base / relative).as_posix()
    try:
        path = confined_path(manifest.root, joined)
    except ArtifactError as error:
        raise GateError(f"{label} is not confined to the artifact") from error
    return path.relative_to(manifest.root).as_posix()


def _on_the_fly_artifact_contract(artifact: Path, runtime: object) -> dict[str, object]:
    try:
        manifest = load_manifest(artifact, verify_payloads=True)
    except (ArtifactError, OSError) as error:
        raise GateError(
            f"cannot authenticate generated on-the-fly artifact: {artifact}"
        ) from error

    _exact_on_the_fly_capabilities(
        manifest.runtime.get("required_runtime_capabilities"),
        "artifact runtime capabilities",
    )
    processes = tuple(manifest.processes)
    if len(processes) != 1 or processes[0].get("id") != PROCESS_ID:
        raise GateError(
            "generated on-the-fly artifact must contain exactly the fixed process"
        )
    process = processes[0]
    expression = process.get("expression")
    if (
        not isinstance(expression, str)
        or canonical_process_key(expression) != PROCESS_KEY
        or process.get("color_accuracy") != "lc"
    ):
        raise GateError("generated on-the-fly artifact has wrong process identity")
    _exact_on_the_fly_capabilities(
        process.get("required_runtime_capabilities"),
        "process runtime capabilities",
    )

    evaluator_relative = str(manifest.runtime["evaluator_manifest_path"])
    evaluator_set = _authenticated_json_payload(
        manifest,
        evaluator_relative,
        role="evaluator-manifest",
        process_id=None,
        label="on-the-fly evaluator set",
    )
    if set(evaluator_set) != {
        "schema_version",
        "kind",
        "required_runtime_capabilities",
        "processes",
    } or (
        evaluator_set.get("schema_version") != 3
        or evaluator_set.get("kind") != "pyamplicol-runtime-execution-set"
    ):
        raise GateError("generated on-the-fly evaluator set is not compact v3")
    _exact_on_the_fly_capabilities(
        evaluator_set.get("required_runtime_capabilities"),
        "evaluator-set runtime capabilities",
    )
    entries = _records(evaluator_set.get("processes"), "evaluator-set processes")
    if len(entries) != 1:
        raise GateError("on-the-fly evaluator set must select exactly one process")
    entry = entries[0]
    if (
        set(entry)
        != {
            "process_id",
            "manifest_path",
            "required_runtime_capabilities",
        }
        or entry.get("process_id") != PROCESS_ID
    ):
        raise GateError("on-the-fly evaluator-set process entry is invalid")
    _exact_on_the_fly_capabilities(
        entry.get("required_runtime_capabilities"),
        "evaluator-set process capabilities",
    )
    execution_relative = _resolved_artifact_relative(
        manifest,
        Path(evaluator_relative).parent,
        entry.get("manifest_path"),
        "on-the-fly execution path",
    )
    record = _authenticated_json_payload(
        manifest,
        execution_relative,
        role="evaluator-manifest",
        process_id=PROCESS_ID,
        label="on-the-fly execution",
    )
    _reject_materialization_fields(record, "on-the-fly execution")
    if set(record) != {
        "schema_version",
        "kind",
        "required_runtime_capabilities",
        "process",
        "key",
        "color_accuracy",
        "external_pdg_order",
        "kernel_pack",
        "runtime_options",
        "selector_policy",
        "runtime_metadata",
        "runtime_container",
    }:
        raise GateError("generated on-the-fly execution has non-compact fields")
    _exact_on_the_fly_capabilities(
        record.get("required_runtime_capabilities"),
        "execution runtime capabilities",
    )
    container = _mapping(
        record.get("runtime_container"), "on-the-fly runtime container"
    )
    if (
        record.get("schema_version") != 3
        or record.get("kind") != "pyamplicol-runtime-on-the-fly-execution"
        or record.get("process") != expression
        or record.get("key") != PROCESS_ID
        or record.get("color_accuracy") != "lc"
        or tuple(_items(record.get("external_pdg_order"), "external PDG order"))
        != tuple(process["external_pdgs"])
        or set(container)
        != {
            "kind",
            "schema_version",
            "storage_abi",
            "path",
            "seed_member_path",
        }
        or container.get("kind") != "pyamplicol-on-the-fly-runtime-container"
        or container.get("schema_version") != 1
        or container.get("storage_abi") != "pacbin-v1"
        or container.get("path") != "on-the-fly-runtime.pacbin"
        or container.get("seed_member_path") != ON_THE_FLY_SEED_MEMBER_PATH
    ):
        raise GateError("generated on-the-fly artifact has the wrong compact contract")
    kernel_pack = _mapping(record.get("kernel_pack"), "on-the-fly kernel pack")
    if kernel_pack != {
        "manifest_path": "model/eager-kernel-pack.json",
        "payload_root": "model/eager-kernels",
    }:
        raise GateError("generated on-the-fly kernel-pack paths are not canonical")

    physics_relative = str(process["physics_path"])
    public = _authenticated_json_payload(
        manifest,
        physics_relative,
        role="runtime-physics",
        process_id=PROCESS_ID,
        label="on-the-fly public metadata",
    )
    _reject_materialization_fields(public, "on-the-fly public metadata")
    if set(public) != {
        "schema_version",
        "kind",
        "process_id",
        "process",
        "color_accuracy",
        "external_particles",
        "model_parameters",
    } or (
        public.get("schema_version") != 1
        or public.get("kind") != "pyamplicol-on-the-fly-public-metadata"
        or public.get("process_id") != PROCESS_ID
        or public.get("process") != expression
        or public.get("color_accuracy") != "lc"
    ):
        raise GateError("generated on-the-fly public metadata is not compact LC")

    validation_relative = f"processes/{PROCESS_ID}/validation-momenta.json"
    _manifest_payload_record(
        manifest,
        validation_relative,
        role="validation-momenta",
        process_id=PROCESS_ID,
    )
    runtime_relative = _resolved_artifact_relative(
        manifest,
        Path(execution_relative).parent,
        container.get("path"),
        "on-the-fly runtime container path",
    )
    runtime_record = _manifest_payload_record(
        manifest,
        runtime_relative,
        role="evaluator-state",
        process_id=PROCESS_ID,
    )
    process_evaluator_state = tuple(
        payload
        for payload in manifest.payloads
        if payload.role == "evaluator-state" and payload.process_id == PROCESS_ID
    )
    if process_evaluator_state != (runtime_record,):
        raise GateError(
            "runtime PACBIN must be the sole process-owned evaluator-state payload"
        )

    expected_inventory = {
        execution_relative,
        physics_relative,
        runtime_relative,
        validation_relative,
    }
    process_prefix = f"processes/{PROCESS_ID}/"
    authenticated_inventory = {
        payload.path
        for payload in manifest.payloads
        if payload.process_id == PROCESS_ID or payload.path.startswith(process_prefix)
    }
    process_root = manifest.root / "processes" / PROCESS_ID
    try:
        physical_inventory = {
            path.relative_to(manifest.root).as_posix()
            for path in process_root.rglob("*")
            if path.is_file()
        }
    except OSError as error:
        raise GateError(
            "cannot inventory generated on-the-fly process files"
        ) from error
    if (
        authenticated_inventory != expected_inventory
        or physical_inventory != expected_inventory
    ):
        unexpected = sorted(
            (authenticated_inventory | physical_inventory) - expected_inventory
        )
        missing = sorted(
            expected_inventory - (authenticated_inventory & physical_inventory)
        )
        raise GateError(
            "generated on-the-fly process inventory contains materialized sidecars "
            f"or omissions; unexpected={unexpected!r}, missing={missing!r}"
        )

    try:
        runtime_path = confined_path(manifest.root, runtime_relative)
        with PacbinReader.open(runtime_path, verify_payloads=True) as reader:
            members = reader.members
    except (ArtifactError, OSError, PacbinError) as error:
        raise GateError("cannot authenticate on-the-fly runtime PACBIN") from error
    if (
        len(members) != 1
        or members[0].logical_path != ON_THE_FLY_SEED_MEMBER_PATH
        or members[0].kind is not PacbinMemberKind.ON_THE_FLY_PROCESS_SEED
        or members[0].length <= 0
    ):
        raise GateError(
            "on-the-fly runtime PACBIN must contain exactly one process seed"
        )
    if (
        runtime.execution_mode != "on-the-fly"
        or runtime.representative_process_key != PROCESS_ID
        or runtime.artifact_id != manifest.artifact_id
    ):
        raise GateError("Runtime.load did not select the generated on-the-fly process")
    return {
        "artifact": str(manifest.root),
        "artifact_id": runtime.artifact_id,
        "execution_mode": runtime.execution_mode,
        "representative_process_key": runtime.representative_process_key,
        "execution_kind": record["kind"],
        "required_runtime_capabilities": ON_THE_FLY_CAPABILITIES,
        "process_file_inventory": tuple(sorted(expected_inventory)),
        "process_evaluator_state_payload_count": len(process_evaluator_state),
        "runtime_container_kind": container["kind"],
        "runtime_container_authenticated": True,
        "runtime_container_member_count": len(members),
        "seed_member_path": container["seed_member_path"],
        "seed_member_kind": members[0].kind.name,
        "dense_physics_accessed": False,
        "validation_boundary": (
            "authenticated manifest/public/execution metadata and PACBIN followed "
            "by compact post-build Runtime.load; candidate dense physics metadata "
            "is never opened"
        ),
    }


def _on_the_fly_public_profile(
    runtime: object,
    points: Points,
    expected: Sequence[object],
    *,
    helicities: tuple[str, ...] = (),
    flows: tuple[str, ...] = (),
) -> dict[str, object]:
    backend = getattr(runtime, "_backend", None)
    profiler = getattr(backend, "profile", None)
    if not callable(profiler):
        raise GateError("public Runtime backend has no existing profile API")
    payload = profiler(
        points,
        helicities=helicities or None,
        color_flows=flows or None,
        precision=16,
        include_values=True,
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("execution_mode") != "on-the-fly"
    ):
        raise GateError("on-the-fly profile did not report its execution lane")
    values = payload.get("values")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise GateError("on-the-fly profile omitted requested values")
    value_check = _series(tuple(values), expected, "on-the-fly profile values")
    counts: dict[str, int] = {}
    for name in PUBLIC_PROFILE_WORK_FIELDS:
        counts[name] = _count(payload.get(name), f"on-the-fly profile {name}")
    if any(
        counts[f"recurrence_{role}_call_count"] > counts[f"recurrence_{role}_row_count"]
        for role in ("source", "contribution", "finalization", "closure")
    ):
        raise GateError("on-the-fly profile has more grouped calls than rows")
    wall = payload.get("wall_time_s")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
    ):
        raise GateError("on-the-fly profile wall clock is invalid")
    return {
        "api": "existing Runtime backend profile",
        "execution_mode": "on-the-fly",
        "point_count": len(points),
        "selectors": {"helicities": helicities, "color_flows": flows},
        "value_check": value_check,
        "wall_seconds": float(wall),
        "work": {
            "basis": "one warmed public profile call",
            "semantics": (
                "rows are executed recurrence-style transitions; calls are grouped "
                "prepared-executor invocations and are not current counts"
            ),
            **counts,
        },
    }


def _workload_operation_census(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sums = {name: 0 for name in OPERATION_COUNT_FIELDS}
    total = 0
    for row in rows:
        census = row.get("work_census")
        if (
            not isinstance(census, Mapping)
            or census.get("work_census_basis") != WORK_CENSUS_BASIS
        ):
            raise GateError("timed query has no valid work census")
        for name in OPERATION_COUNT_FIELDS:
            value = census.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GateError("timed query operation census is invalid")
            sums[name] += value
        kernel_count = census.get("total_kernel_application_count")
        if (
            isinstance(kernel_count, bool)
            or not isinstance(kernel_count, int)
            or kernel_count < 0
        ):
            raise GateError("timed query kernel-application census is invalid")
        total += kernel_count
    if total != sum(sums.values()):
        raise GateError("workload kernel and operation sums disagree")
    return {
        "aggregation_basis": "sum-one-execution-per-serialized-query-v1",
        "query_census_basis": WORK_CENSUS_BASIS,
        "query_count": len(rows),
        **sums,
        "total_kernel_application_count": total,
    }


def _query_family_census(
    probe: Callable[..., dict[str, Any]],
    retained: RetainedInputs,
    queries: Sequence[Query],
    hidden_timing: Mapping[str, object],
    enable_color_projection: bool,
) -> dict[str, object]:
    if not queries:
        raise GateError("query-family census needs at least one query")
    raw = probe(
        retained.builder,
        retained.template,
        retained.direct_json,
        retained.pack_digest,
        [query.flow_index for query in queries],
        [list(query.helicities) for query in queries],
        enable_color_projection=enable_color_projection,
    )
    if not isinstance(raw, Mapping):
        raise GateError("native query-family census is not a mapping")
    counts: dict[str, int] = {}
    for name in FAMILY_CENSUS_COUNT_FIELDS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"native query-family census has invalid {name}")
        counts[name] = value
    if counts["query_count"] != len(queries):
        raise GateError("native query-family census has the wrong query count")
    if not 0 < counts["source_frame_partition_count"] <= len(queries):
        raise GateError(
            "native query-family census has invalid source-frame partitions"
        )

    rows = hidden_timing.get("queries")
    operation_census = hidden_timing.get("workload_operation_census")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not isinstance(operation_census, Mapping)
    ):
        raise GateError("timed workload has no query-local census")
    current_occurrences = 0
    component_occurrences = 0
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("work_census"), Mapping
        ):
            raise GateError("timed workload query has no work census")
        work = row["work_census"]
        for name, target in (
            ("logical_current_count", "current"),
            ("resident_current_component_count", "component"),
        ):
            value = work.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GateError(f"timed workload has invalid {target} census")
            if name == "logical_current_count":
                current_occurrences += value
            else:
                component_occurrences += value
    expected_rows = {
        "dynamic_source_rows": "source_operation_count",
        "dynamic_contribution_rows": "contribution_operation_count",
        "dynamic_finalization_rows": "finalization_operation_count",
        "dynamic_closure_rows": "closure_operation_count",
    }
    if (
        counts["dynamic_current_occurrence_count"] != current_occurrences
        or counts["dynamic_current_component_occurrence_count"] != component_occurrences
        or any(
            counts[native] != operation_census.get(query_local)
            for native, query_local in expected_rows.items()
        )
        or counts["projection_post_current_count"] != current_occurrences
        or counts["projection_post_contribution_count"]
        != counts["dynamic_contribution_rows"]
        or counts["projection_post_closure_count"] != counts["dynamic_closure_rows"]
    ):
        raise GateError(
            "native query-family census disagrees with independently timed query traces"
        )
    for role in ("source", "contribution", "finalization", "closure"):
        if counts[f"dynamic_{role}_calls"] != counts[f"dynamic_{role}_rows"]:
            raise GateError("query-local dynamic call census is inconsistent")
        if counts[f"union_{role}_executor_call_groups"] > counts[f"union_{role}_rows"]:
            raise GateError("query-family executor groups outnumber union rows")
    if (
        counts["union_amplitude_destination_count"] != len(queries)
        or counts["union_unique_current_count"] > current_occurrences
        or counts["union_unique_current_component_count"] > component_occurrences
        or any(
            counts[f"union_{role}_rows"] > counts[f"dynamic_{role}_rows"]
            for role in ("source", "contribution", "finalization", "closure")
        )
    ):
        raise GateError("query-family union census does not bound query-local work")
    return {
        "basis": "exact-current-core-key-query-family-union-v1",
        "semantics": (
            "dynamic counts sum today's independently executed query-local traces; "
            "union counts intern exact semantic currents/interactions within each "
            "authenticated source-frame partition while retaining one amplitude "
            "destination per requested query"
        ),
        **counts,
    }


def _assert_executable_family_matches_structural_census(
    executable: Mapping[str, object], structural: Mapping[str, object]
) -> None:
    for name in FAMILY_CENSUS_COUNT_FIELDS:
        if executable.get(name) != structural.get(name):
            raise GateError(
                f"executable family census disagrees with the independent structural "
                f"census for {name}"
            )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GateError(f"{label} is not a mapping")
    return value


def _items(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GateError(f"{label} is not a sequence")
    return tuple(value)


def _records(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, f"{label}[{index}]")
        for index, item in enumerate(_items(value, label))
    )


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{label} is not a non-negative integer")
    return value


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _artifact_execution(
    artifact: Path,
    execution_mode: str,
    runtime: object | None = None,
) -> tuple[object, Mapping[str, object], Path]:
    try:
        root = artifact.expanduser().resolve(strict=True)
    except OSError as error:
        raise GateError(f"comparator artifact does not exist: {artifact}") from error
    if not root.is_dir():
        raise GateError(f"comparator artifact is not a directory: {root}")
    if runtime is None:
        runtime = Runtime.load(root)
    physics = _validate_comparator_physics(runtime, execution_mode, str(root))
    process_id = physics.process_id
    path = root / "processes" / process_id / "execution.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise GateError(f"cannot read comparator execution record: {path}") from error
    return runtime, _mapping(payload, "comparator execution record"), root


def _validate_comparator_physics(
    runtime: object, execution_mode: str, label: str
) -> object:
    physics = runtime.physics
    process = getattr(physics, "process", None)
    process_id = getattr(physics, "process_id", None)
    color_accuracy = getattr(physics, "color_accuracy", None)
    external_pdgs: list[int] = []
    for index, particle in enumerate(getattr(physics, "external_particles", ())):
        pdg_id = _field(particle, "pdg_id")
        if isinstance(pdg_id, bool) or not isinstance(pdg_id, int):
            raise GateError(
                f"{label} external particle {index} has an invalid PDG ID: {pdg_id!r}"
            )
        external_pdgs.append(pdg_id)
    if runtime.execution_mode != execution_mode or color_accuracy != "lc":
        raise GateError(f"{label} is not an LC {execution_mode} authority")
    if (
        not isinstance(process, str)
        or process != PROCESS
        or canonical_process_key(process) != PROCESS_KEY
        or process_id != PROCESS_ID
    ):
        raise GateError(
            f"{label} has the wrong canonical process identity: {process!r}"
        )
    if tuple(external_pdgs) != EXTERNAL_PDGS:
        raise GateError(
            f"{label} has the wrong external PDG ordering: {tuple(external_pdgs)!r}"
        )
    return physics


def _recurrence_counts(
    record: Mapping[str, object],
    *,
    components: object | None = None,
    destinations: object | None = None,
    certificate: bool = False,
) -> dict[str, int]:
    closure_key = "closure_count" if certificate else "closure_term_count"
    result = {
        "source_row_count": _count(
            record.get("source_row_count"), "recurrence source-row count"
        ),
        "current_count": _count(
            record.get("current_count"), "recurrence current count"
        ),
        "semantic_current_component_count": _count(
            (
                record.get("semantic_component_count")
                if components is None
                else components
            ),
            "recurrence semantic-component count",
        ),
        "contribution_count": _count(
            record.get("contribution_count"), "recurrence contribution count"
        ),
        "finalization_count": _count(
            record.get("finalization_count"), "recurrence finalization count"
        ),
        "closure_count": _count(record.get(closure_key), "recurrence closure count"),
        "amplitude_destination_count": _count(
            (
                record.get("amplitude_destination_count")
                if destinations is None
                else destinations
            ),
            "recurrence amplitude-destination count",
        ),
    }
    if result["semantic_current_component_count"] < result["current_count"]:
        raise GateError("recurrence has fewer components than currents")
    result["kernel_row_count"] = sum(
        result[name]
        for name in (
            "source_row_count",
            "contribution_count",
            "finalization_count",
            "closure_count",
        )
    )
    if certificate and result["kernel_row_count"] != _count(
        record.get("row_count"), "recurrence certificate row count"
    ):
        raise GateError("recurrence certificate row count is inconsistent")
    return result


def _exact_public_index(records: object, selector_id: str, label: str) -> int:
    matches = [
        record for record in records if getattr(record, "id", None) == selector_id
    ]
    if len(matches) != 1:
        raise GateError(f"{label} selector {selector_id!r} is absent or ambiguous")
    index = getattr(matches[0], "index", None)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise GateError(f"{label} selector {selector_id!r} has an invalid public index")
    return index


def _recurrence_artifact_census(
    artifact: Path, *, layout: str, runtime: object | None = None
) -> dict[str, object]:
    runtime, payload, root = _artifact_execution(artifact, "recurrence", runtime)
    if payload.get("kind") != "pyamplicol-runtime-recurrence-execution":
        raise GateError("recurrence comparator has the wrong execution kind")
    summary = _mapping(payload.get("recurrence_summary"), "recurrence summary")
    plan = _mapping(payload.get("plan"), "recurrence plan")
    inspection = _mapping(plan.get("inspection_summary"), "recurrence inspection")
    if inspection.get("lc_flow_layout") != layout:
        raise GateError(f"recurrence comparator is not {layout!r}")
    schedule = _mapping(inspection.get("schedule"), "recurrence schedule")
    arena = _mapping(inspection.get("direct_arena"), "recurrence direct arena")
    whole = _recurrence_counts(
        schedule, components=arena.get("semantic_component_count")
    )
    if (
        summary.get("current_count") != whole["current_count"]
        or summary.get("contribution_count") != whole["contribution_count"]
        or summary.get("closure_term_count") != whole["closure_count"]
    ):
        raise GateError("recurrence schedule summaries disagree")
    certificate = _mapping(
        inspection.get("selector_work_certificate"),
        "recurrence selector-work certificate",
    )
    union = _recurrence_counts(
        _mapping(certificate.get("persisted_union"), "recurrence persisted union"),
        destinations=whole["amplitude_destination_count"],
        certificate=True,
    )
    if union != whole:
        raise GateError("recurrence whole schedule and persisted union disagree")

    physics = runtime.physics
    if layout == "topology-replay":
        public_index = _exact_public_index(
            physics.color_flows, FLOW_ID, "recurrence color-flow"
        )
        metadata = _mapping(payload.get("runtime_metadata"), "recurrence metadata")
        public_flows = _records(
            metadata.get("public_color_flows"), "recurrence public color flows"
        )
        bindings = [
            record for record in public_flows if record.get("public_id") == FLOW_ID
        ]
        if len(bindings) != 1:
            raise GateError("recurrence color-flow binding is absent or ambiguous")
        target_sector = _count(
            bindings[0].get("target_sector_id"), "recurrence target sector"
        )
        if target_sector != public_index:
            raise GateError(
                "recurrence public color-flow index and target sector differ"
            )
        representatives = _records(
            certificate.get("representatives"), "recurrence representatives"
        )
        selected = [
            record
            for record in representatives
            if record.get("representative_sector_id") == target_sector
        ]
        if len(selected) != 1:
            raise GateError(
                "recurrence selected-flow certificate is absent or ambiguous"
            )
        live = _recurrence_counts(selected[0], certificate=True)
        selector = {
            "kind": "exact_public_color_flow_index",
            "id": FLOW_ID,
            "public_index": public_index,
            "representative_sector_id": target_sector,
        }
    else:
        public_index = _exact_public_index(
            physics.helicities, HELICITY_ID, "recurrence helicity"
        )
        representatives = _records(
            certificate.get("representatives"), "recurrence representatives"
        )
        if representatives:
            raise GateError("all-flow-union recurrence certificate has alternatives")
        live = union
        selector = {
            "kind": "exact_public_helicity_index",
            "id": HELICITY_ID,
            "public_index": public_index,
        }
    return {
        "artifact": str(root),
        "artifact_id": runtime.artifact_id,
        "layout": layout,
        "basis": "authenticated-persisted-recurrence-selector-certificate-v1",
        "selector": selector,
        "whole": whole,
        "selector_live": live,
        "semantics": (
            "whole is the persisted recurrence DirectPlan; selector_live is the "
            "authenticated exact-selector certificate and is not inferred from "
            "runtime timing counters"
        ),
    }


def _compiled_execution_counts(
    execution: Mapping[str, object], label: str
) -> dict[str, int]:
    if execution.get("kind") != "pyamplicol-runtime-execution":
        raise GateError(f"{label} has the wrong execution kind")
    summary = _mapping(execution.get("dag_summary"), f"{label} DAG summary")
    schema = _mapping(execution.get("runtime_schema"), f"{label} runtime schema")
    storage = _mapping(schema.get("current_storage"), f"{label} current storage")
    result = {
        "source_count": _count(summary.get("source_count"), f"{label} source count"),
        "current_count": _count(summary.get("current_count"), f"{label} current count"),
        "current_component_count": _count(
            storage.get("component_count"), f"{label} component count"
        ),
        "interaction_attachment_count": _count(
            summary.get("interaction_count"), f"{label} interaction count"
        ),
        "interaction_evaluation_count": _count(
            summary.get("interaction_evaluation_count"),
            f"{label} interaction-evaluation count",
        ),
        "amplitude_root_count": _count(
            summary.get("amplitude_root_count"), f"{label} amplitude-root count"
        ),
    }
    if result["current_component_count"] < result["current_count"]:
        raise GateError(f"{label} has fewer components than currents")
    if result["interaction_evaluation_count"] > result["interaction_attachment_count"]:
        raise GateError(f"{label} evaluates more interactions than it attaches")
    return result


def _compiled_artifact_census(
    artifact: Path, *, workload: str, runtime: object | None = None
) -> dict[str, object]:
    runtime, payload, root = _artifact_execution(artifact, "compiled", runtime)
    primary = _compiled_execution_counts(payload, "compiled primary")
    physics = runtime.physics
    if workload == "selected_flow_helicity_sum":
        public_index = _exact_public_index(
            physics.color_flows, FLOW_ID, "compiled color-flow"
        )
        compiled = _mapping(payload.get("compiled"), "compiled metadata")
        topology = _mapping(
            compiled.get("lc_topology_replay"), "compiled topology replay"
        )
        matches = [
            record
            for record in _records(topology.get("groups"), "compiled topology groups")
            if public_index
            in tuple(
                _count(value, "compiled active sector")
                for value in _items(record.get("active_sector_ids"), "active sectors")
            )
        ]
        if len(matches) != 1:
            raise GateError("compiled color-flow topology group is absent or ambiguous")
        materialized_sector = _count(
            matches[0].get("materialized_sector_id"),
            "compiled materialized sector",
        )
        program_record = _mapping(
            payload.get("helicity_sum_execution"), "compiled helicity-sum program"
        )
        reductions = _mapping(
            program_record.get("physics_reduction"),
            "compiled helicity-sum physics reduction",
        )
        reduction_groups = _records(
            reductions.get("groups"), "compiled physics-reduction groups"
        )
        if not any(
            FLOW_ID in tuple(record.get("physical_color_ids", ()))
            for record in reduction_groups
        ):
            raise GateError(
                "compiled helicity-sum program does not bind the exact flow"
            )
        children = _records(
            program_record.get("color_selector_executions"),
            "compiled color-selector executions",
        )
        selected = [
            record
            for record in children
            if record.get("materialized_sector_id") == materialized_sector
        ]
        if len(selected) != 1:
            raise GateError("compiled selected-flow child is absent or ambiguous")
        leaf_record = _mapping(selected[0].get("execution"), "compiled selected leaf")
        levels: dict[str, object] = {
            "primary": primary,
            "program": _compiled_execution_counts(
                program_record, "compiled helicity-sum program"
            ),
            "executed_leaf": _compiled_execution_counts(
                leaf_record, "compiled selected-flow leaf"
            ),
        }
        selector = {
            "kind": "exact_public_color_flow_index",
            "id": FLOW_ID,
            "public_index": public_index,
            "materialized_sector_id": materialized_sector,
        }
    elif workload == "all_flow_single_helicity":
        public_index = _exact_public_index(
            physics.helicities, HELICITY_ID, "compiled helicity"
        )
        current = payload
        depth = 0
        while "helicity_selector_executions" in current:
            children = _records(
                current.get("helicity_selector_executions"),
                "compiled helicity-selector executions",
            )
            matching = [
                record
                for record in children
                if public_index
                in tuple(
                    _count(value, "compiled selector-domain index")
                    for value in _items(
                        record.get("selector_domain_ids"),
                        "compiled selector-domain indices",
                    )
                )
            ]
            if len(matching) != 1:
                raise GateError(
                    "compiled helicity-selector child is absent or ambiguous"
                )
            current = _mapping(
                matching[0].get("execution"), "compiled helicity-selector child"
            )
            depth += 1
        if depth == 0:
            raise GateError("compiled all-flow artifact has no selector execution")
        levels = {
            "primary": primary,
            "executed_leaf": _compiled_execution_counts(
                current, "compiled all-flow executed leaf"
            ),
        }
        selector = {
            "kind": "exact_public_helicity_index",
            "id": HELICITY_ID,
            "public_index": public_index,
            "selector_depth": depth,
        }
    else:
        raise GateError(f"unknown compiled census workload: {workload}")
    return {
        "artifact": str(root),
        "artifact_id": runtime.artifact_id,
        "basis": "authenticated-exact-child-compiled-generic-dag-v1",
        "workload": workload,
        "selector": selector,
        "levels": levels,
        "semantics": (
            "GenericDAG counts are reported independently for the outer primary, "
            "an enclosing program where present, and the one exact executed leaf; "
            "alternative compiled nodes are never summed, and these attachment/"
            "evaluation counts are not asserted equal to recurrence or on-the-fly rows"
        ),
    }


def _hidden_timing(
    probe: Callable[..., dict[str, Any]],
    artifact: Path,
    retained: RetainedInputs,
    queries: Sequence[Query],
    points: Points,
    target: float,
    enable_color_projection: bool = True,
) -> dict[str, object]:
    if not queries:
        raise GateError("workload has no physical queries")
    invoke = partial(
        _invoke,
        probe,
        artifact,
        retained,
        points=points,
        enable_color_projection=enable_color_projection,
    )
    pilot_seconds = 0.0
    pilot_census: dict[Query, dict[str, object]] = {}
    for query in queries:
        report = invoke(query, repetitions=1)
        _probe_values(report, len(points), 1)
        pilot_census[query] = _work_census(report)
        pilot_seconds += float(report["benchmark_elapsed_seconds"])
    repetitions = _calibrate(pilot_seconds, target)
    rows = []
    for query in queries:
        report = invoke(query, repetitions=repetitions)
        _probe_values(report, len(points), repetitions)
        work_census = _work_census(report)
        if work_census != pilot_census[query]:
            raise GateError("query work census changed between benchmark passes")
        rows.append(
            {
                "query": query.label,
                "trace_digest": report.get("trace_digest"),
                "elapsed_seconds": report["benchmark_elapsed_seconds"],
                "seconds_per_point": report["benchmark_seconds_per_point"],
                "trace_build_count": 1,
                "trace_cache_hit_count": WARMUPS + repetitions,
                "momentum_fill_count": WARMUPS + repetitions,
                "work_census": work_census,
            }
        )
    return {
        "query_count": len(rows),
        "pilot_aggregate_seconds": pilot_seconds,
        "warmups": WARMUPS,
        "repetitions": repetitions,
        "aggregate_elapsed_seconds": math.fsum(
            float(row["elapsed_seconds"]) for row in rows
        ),
        "additive_seconds_per_point": math.fsum(
            float(row["seconds_per_point"]) for row in rows
        ),
        "queries": rows,
        "workload_operation_census": _workload_operation_census(rows),
        "timer_includes": (
            "trace-cache lookup",
            "momentum fill",
            "structural-trace execution",
        ),
        "timer_excludes": (
            "artifact/pack load",
            "trace construction",
            "parameter/workspace setup",
            "source-major conversion",
            "normalization/output/Python conversion",
        ),
        "acceptance_gate": False,
    }


def _hidden_family_timing(
    probe: Callable[..., dict[str, Any]],
    artifact: Path,
    retained: RetainedInputs,
    queries: Sequence[Query],
    points: Points,
    target: float,
    enable_color_projection: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    if not queries:
        raise GateError("family workload has no physical queries")
    invoke = partial(
        _invoke,
        probe,
        artifact,
        retained,
        queries[0],
        points,
        enable_color_projection=enable_color_projection,
        query_family=queries,
    )
    pilot = _family_probe_result(invoke(repetitions=1), queries, len(points), 1)
    pilot_elapsed = pilot.get("private_warmed_elapsed_seconds")
    if not isinstance(pilot_elapsed, (int, float)):
        raise GateError("family pilot omitted its warmed elapsed time")
    repetitions = _calibrate(float(pilot_elapsed), target)
    measured = _family_probe_result(
        invoke(repetitions=repetitions), queries, len(points), repetitions
    )
    if measured["census"] != pilot["census"]:
        raise GateError("family union census changed between benchmark passes")
    return measured, {
        "query_count": len(queries),
        "pilot_seconds": float(pilot_elapsed),
        "warmups": WARMUPS,
        "repetitions": repetitions,
        "cold_prepare_seconds": measured["cold_prepare_seconds"],
        "private_warmed_elapsed_seconds": measured["private_warmed_elapsed_seconds"],
        "private_warmed_seconds_per_point": measured[
            "private_warmed_seconds_per_point"
        ],
        "clock_attribution": {
            "private_warmed_seconds_per_point": (
                "on-the-fly private query-family execution clock"
            ),
            "relationship": (
                "independent of recurrence core, recurrence wall, and AmpliCol wall"
            ),
        },
        "private_timing_excludes_source_crossing": True,
        "work_census_basis": FAMILY_WORK_CENSUS_BASIS,
        "work_census": measured["census"],
        "total_kernel_application_count": measured[
            "union_total_kernel_application_count"
        ],
        "timer_includes": (
            "shared source fill",
            "one grouped union execution",
            "flat caller-output write",
        ),
        "timer_excludes": (
            "artifact/pack load",
            "query construction",
            "family preparation",
            "parameter/workspace setup",
            "source-major crossing",
            "normalization/Python conversion",
        ),
        "timing_status": "private-provisional",
        "acceptance_gate": False,
    }


def _hidden_family_correctness(
    probe: Callable[..., dict[str, Any]],
    artifact: Path,
    retained: RetainedInputs,
    queries: Sequence[Query],
    points: Points,
    normalized_authority: Sequence[Sequence[object]],
    query_local_reports: Sequence[Mapping[str, object]],
    enable_color_projection: bool = True,
) -> tuple[tuple[tuple[float, ...], ...], dict[str, object]]:
    if (
        not queries
        or len(queries) != len(normalized_authority)
        or len(queries) != len(query_local_reports)
    ):
        raise GateError("family correctness inputs have incompatible query axes")
    report = _invoke(
        probe,
        artifact,
        retained,
        queries[0],
        points,
        enable_color_projection=enable_color_projection,
        query_family=queries,
    )
    parsed = _family_probe_result(report, queries, len(points))
    parsed_rows = parsed["queries"]
    if not isinstance(parsed_rows, Sequence):
        raise GateError("family correctness report lost its query rows")
    normalized_checks: list[dict[str, object]] = []
    raw_real_checks: list[dict[str, object]] = []
    raw_imag_checks: list[dict[str, object]] = []
    normalized_rows: list[tuple[float, ...]] = []
    for query, row, authority, query_local in zip(
        queries,
        parsed_rows,
        normalized_authority,
        query_local_reports,
        strict=True,
    ):
        if not isinstance(row, Mapping):
            raise GateError("family correctness row is malformed")
        normalized = row.get("normalized_values")
        raw = row.get("raw_amplitudes")
        local_raw = query_local.get("raw_amplitudes")
        if (
            not isinstance(normalized, Sequence)
            or not isinstance(raw, Sequence)
            or not isinstance(local_raw, Sequence)
            or len(raw) != len(local_raw)
        ):
            raise GateError("family correctness row is incomplete")
        normalized_tuple = tuple(_real(value, "family value") for value in normalized)
        normalized_rows.append(normalized_tuple)
        normalized_checks.append(
            _series(normalized_tuple, authority, f"{query.label} family/public")
        )
        family_complex = tuple(complex(value) for value in raw)
        local_complex = tuple(
            complex(
                _real(value[0], "local raw real"),
                _real(value[1], "local raw imag"),
            )
            for value in local_raw
            if isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 2
        )
        if len(local_complex) != len(family_complex):
            raise GateError("query-local raw amplitude row is malformed")
        raw_real_checks.append(
            _series(
                tuple(value.real for value in family_complex),
                tuple(value.real for value in local_complex),
                f"{query.label} family/query-local raw real",
            )
        )
        raw_imag_checks.append(
            _series(
                tuple(value.imag for value in family_complex),
                tuple(value.imag for value in local_complex),
                f"{query.label} family/query-local raw imaginary",
            )
        )
    return tuple(normalized_rows), {
        "normalized_component_checks": _summaries(normalized_checks),
        "raw_real_component_checks": _summaries(raw_real_checks),
        "raw_imaginary_component_checks": _summaries(raw_imag_checks),
        "work_census_basis": FAMILY_WORK_CENSUS_BASIS,
        "work_census": parsed["census"],
        "total_kernel_application_count": parsed[
            "union_total_kernel_application_count"
        ],
    }


def _anchor(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"available": False, "comparison_performed": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["validation"]
        selector = payload["selector_contract"]
        common = validation["lc_common_component"]
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise GateError(f"cannot read AmpliCol context record: {path}") from error
    if payload.get("status") != "ok":
        raise GateError("AmpliCol context is not a successful record")
    try:
        selector_matches = (
            tuple(selector.get("selected_color_flow_ids", ())) == (FLOW_ID,)
            and tuple(map(tuple, selector.get("selected_color_words", ())))
            == (FLOW_WORD,)
            and tuple(common.get("helicity_ids", ())) == (HELICITY_ID,)
            and tuple(common.get("color_flow_ids", ())) == (FLOW_ID,)
        )
    except TypeError:
        selector_matches = False
    return {
        "available": True,
        "path": str(path.resolve(strict=True)),
        "cell_id": common.get("cell_id"),
        "point_digest": common.get("point_digest"),
        "component": common.get("value"),
        "selected_flow_sum": payload.get("matrix_element"),
        "execution_seconds_per_point": payload.get("execution_seconds_per_point"),
        "wall_seconds_per_point": payload.get("wall_seconds_per_point"),
        "clock_attribution": {
            "execution_seconds_per_point": (
                "stored original-AmpliCol direct execution clock"
            ),
            "wall_seconds_per_point": "stored original-AmpliCol worker wall clock",
            "relationship": (
                "independent clocks; equality is neither assumed nor asserted, and "
                "both are independent of pyAmpliCol/on-the-fly clocks"
            ),
        },
        "selector_matches": selector_matches,
        "comparison_performed": False,
    }


def _anchor_checks(
    context: Mapping[str, object],
    digest: str,
    *,
    public_component: object,
    hidden_component: object,
    public_sum: object,
    hidden_sum: object,
) -> dict[str, object]:
    result = dict(context)
    result["generated_point_digest"] = digest
    if not context.get("available"):
        return result
    if context.get("cell_id") != AMPICOL_CELL_ID:
        result["reason"] = "cell identity differs; historical result is context only"
        return result
    if context.get("selector_matches") is not True:
        result["reason"] = (
            "selector identity differs; historical result is context only"
        )
        return result
    if context.get("point_digest") != digest:
        result["reason"] = "point digest differs; historical result is context only"
        return result
    result.update(
        {
            "comparison_performed": True,
            "public_component": _series(
                (public_component,),
                (context["component"],),
                "public/AmpliCol component",
                AMPICOL_REL_TOL,
            ),
            "hidden_component": _series(
                (hidden_component,),
                (context["component"],),
                "on-the-fly/AmpliCol component",
                AMPICOL_REL_TOL,
            ),
            "public_selected_flow_sum": _series(
                (public_sum,),
                (context["selected_flow_sum"],),
                "public/AmpliCol selected-flow sum",
                AMPICOL_REL_TOL,
            ),
            "hidden_selected_flow_sum": _series(
                (hidden_sum,),
                (context["selected_flow_sum"],),
                "on-the-fly/AmpliCol selected-flow sum",
                AMPICOL_REL_TOL,
            ),
        }
    )
    return result


def _benchmark_runtime(
    runtime: object,
    execution_mode: str,
    points: Points,
    target: float,
    batch: int,
    *,
    helicities: tuple[str, ...] = (),
    flows: tuple[str, ...] = (),
) -> dict[str, object]:
    result = BenchmarkRunner(
        _benchmark_config(target, batch, helicities=helicities, flows=flows)
    ).run(runtime, points=points)
    timing = _public_timing(result, execution_mode)
    wall = float(timing["wall_seconds_per_point"])
    if not math.isfinite(wall) or wall <= 0.0:
        raise GateError(f"public {execution_mode} benchmark has non-positive wall time")
    return timing


def _run_public_gate_phase(
    artifact: Path,
    recurrence_selected_artifact: Path,
    recurrence_all_flow_artifact: Path,
    compiled_selected_artifact: Path,
    compiled_all_flow_artifact: Path,
    target: float,
    batch: int,
) -> PublicGatePhase:
    started = time.perf_counter()
    runtime = Runtime.load(artifact, process=PROCESS_ID)
    load_seconds = time.perf_counter() - started
    artifact_contract = _on_the_fly_artifact_contract(artifact, runtime)

    selected_recurrence_runtime = Runtime.load(recurrence_selected_artifact)
    all_flow_recurrence_runtime = Runtime.load(recurrence_all_flow_artifact)
    compiled_selected_runtime = Runtime.load(compiled_selected_artifact)
    compiled_all_flow_runtime = Runtime.load(compiled_all_flow_artifact)
    authorities = (
        ("selected recurrence", selected_recurrence_runtime, "recurrence"),
        ("all-flow recurrence", all_flow_recurrence_runtime, "recurrence"),
        ("selected compiled", compiled_selected_runtime, "compiled"),
        ("all-flow compiled", compiled_all_flow_runtime, "compiled"),
    )
    comparator_selectors: list[tuple[str, tuple[object, object]]] = []
    for label, authority, expected_mode in authorities:
        physics = _validate_comparator_physics(authority, expected_mode, label)
        comparator_selectors.append((label, _selectors(physics)))

    points = _points()
    correctness = _dual_authority_correctness(
        runtime,
        selected_recurrence_runtime,
        all_flow_recurrence_runtime,
        compiled_selected_runtime,
        compiled_all_flow_runtime,
        points,
    )

    # This block is deliberately after the complete before/after-clear public
    # authority gate.  A mismatch must not produce profile or timing evidence.
    public_profiles = {
        "selected_flow_helicity_sum": _on_the_fly_public_profile(
            runtime,
            points,
            correctness.on_the_fly_selected.total,
            flows=(FLOW_ID,),
        ),
        "all_flow_single_helicity": _on_the_fly_public_profile(
            runtime,
            points,
            correctness.on_the_fly_all_flow.total,
            helicities=(HELICITY_ID,),
        ),
    }
    timing_points = _repeat(points, batch)
    timings = {
        "on_the_fly": {
            "selected_flow_helicity_sum": _benchmark_runtime(
                runtime,
                "on-the-fly",
                timing_points,
                target,
                batch,
                flows=(FLOW_ID,),
            ),
            "all_flow_single_helicity": _benchmark_runtime(
                runtime,
                "on-the-fly",
                timing_points,
                target,
                batch,
                helicities=(HELICITY_ID,),
            ),
        },
        "recurrence": {
            "selected_flow_helicity_sum": _benchmark_runtime(
                selected_recurrence_runtime,
                "recurrence",
                timing_points,
                target,
                batch,
                flows=(FLOW_ID,),
            ),
            "all_flow_single_helicity": _benchmark_runtime(
                all_flow_recurrence_runtime,
                "recurrence",
                timing_points,
                target,
                batch,
                helicities=(HELICITY_ID,),
            ),
        },
        "compiled": {
            "selected_flow_helicity_sum": _benchmark_runtime(
                compiled_selected_runtime,
                "compiled",
                timing_points,
                target,
                batch,
                flows=(FLOW_ID,),
            ),
            "all_flow_single_helicity": _benchmark_runtime(
                compiled_all_flow_runtime,
                "compiled",
                timing_points,
                target,
                batch,
                helicities=(HELICITY_ID,),
            ),
        },
    }
    return PublicGatePhase(
        runtime=runtime,
        selected_recurrence_runtime=selected_recurrence_runtime,
        all_flow_recurrence_runtime=all_flow_recurrence_runtime,
        compiled_selected_runtime=compiled_selected_runtime,
        compiled_all_flow_runtime=compiled_all_flow_runtime,
        comparator_selectors=tuple(comparator_selectors),
        points=points,
        timing_points=timing_points,
        correctness=correctness,
        artifact_contract=artifact_contract,
        load_seconds=load_seconds,
        public_profiles=public_profiles,
        timings=timings,
    )


def _run(
    output: Path,
    prepared_model: Path,
    amplicol: Path | None,
    recurrence_selected_artifact: Path,
    recurrence_all_flow_artifact: Path,
    compiled_selected_artifact: Path,
    compiled_all_flow_artifact: Path,
    target: float,
    batch: int,
    enable_color_projection: bool = True,
) -> dict[str, object]:
    artifact = output / "artifact"
    with tempfile.TemporaryDirectory(
        prefix=".on-the-fly-private-probe-", dir=output
    ) as probe_directory:
        probe_carrier = Path(probe_directory) / "recurrence-artifact"
        (
            generation_seconds,
            generation_census,
            model,
            prepared_path,
            prepared_load_seconds,
        ) = _generate_candidate_with_prepared_model(artifact, prepared_model)
        public = _run_public_gate_phase(
            artifact,
            recurrence_selected_artifact,
            recurrence_all_flow_artifact,
            compiled_selected_artifact,
            compiled_all_flow_artifact,
            target,
            batch,
        )
        runtime = public.runtime
        selected_recurrence_runtime = public.selected_recurrence_runtime
        all_flow_recurrence_runtime = public.all_flow_recurrence_runtime
        compiled_selected_runtime = public.compiled_selected_runtime
        compiled_all_flow_runtime = public.compiled_all_flow_runtime
        points = public.points
        timing_points = public.timing_points
        correctness = public.correctness
        artifact_contract = public.artifact_contract
        load_seconds = public.load_seconds
        public_profiles = public.public_profiles
        timings = public.timings

        # The default lane creates its private carrier only after the complete
        # public correctness/profile/timing phase has passed.  The provisional
        # public-only lane never reaches this call.
        retained = _generate_probe_carrier(probe_carrier, model)
        probe_runtime = Runtime.load(probe_carrier, process=PROCESS_ID)
        if probe_runtime.execution_mode != "recurrence":
            raise GateError("private probe carrier is not a recurrence artifact")
        fixed_flow, fixed_helicity = _selectors(probe_runtime.physics)
        for label, (flow, helicity) in public.comparator_selectors:
            if flow.id != fixed_flow.id or helicity.id != fixed_helicity.id:
                raise GateError(
                    f"probe carrier and {label} authority disagree on public IDs"
                )
        selected_authority = correctness.recurrence_selected.total
        all_authority = correctness.recurrence_all_flow.total
        selected_resolved = correctness.recurrence_selected.resolved
        all_resolved = correctness.recurrence_all_flow.resolved
        selected_public = correctness.on_the_fly_selected.total
        selected_public_resolved = correctness.on_the_fly_selected.resolved
        public_correctness = correctness.public_correctness
        clear_checks = correctness.clear_checks

        selected_helicity_axis, selected_color_axis = _axes(selected_resolved)
        selected_fixed_c = selected_color_axis[FLOW_ID]
        all_helicity_axis, all_color_axis = _axes(all_resolved)
        all_fixed_h = all_helicity_axis[HELICITY_ID]

        recurrence_selected_census = _recurrence_artifact_census(
            recurrence_selected_artifact,
            layout="topology-replay",
            runtime=selected_recurrence_runtime,
        )
        recurrence_all_flow_census = _recurrence_artifact_census(
            recurrence_all_flow_artifact,
            layout="all-flow-union",
            runtime=all_flow_recurrence_runtime,
        )
        compiled_selected_census = _compiled_artifact_census(
            compiled_selected_artifact,
            workload="selected_flow_helicity_sum",
            runtime=compiled_selected_runtime,
        )
        compiled_all_flow_census = _compiled_artifact_census(
            compiled_all_flow_artifact,
            workload="all_flow_single_helicity",
            runtime=compiled_all_flow_runtime,
        )
        probe = _probe()
        family_probe = _family_probe()
        cache: dict[Query, tuple[tuple[float, ...], float, dict[str, Any]]] = {}

        def hidden(flow: object, helicity: object) -> tuple[float, ...]:
            query = _query(flow, helicity)
            if query not in cache:
                before = time.perf_counter()
                report = _invoke(
                    probe,
                    probe_carrier,
                    retained,
                    query,
                    points,
                    enable_color_projection=enable_color_projection,
                )
                cache[query] = (
                    _probe_values(report, len(points)),
                    time.perf_counter() - before,
                    report,
                )
            return cache[query][0]

        selected_hidden: list[tuple[float, ...]] = []
        selected_reference: list[tuple[complex, ...]] = []
        selected_queries: list[Query] = []
        selected_checks: list[dict[str, object]] = []
        for helicity in probe_runtime.physics.helicities:
            if helicity.id not in selected_helicity_axis:
                raise GateError("probe carrier helicity is absent from authority")
            row = tuple(
                complex(point[selected_helicity_axis[helicity.id]][selected_fixed_c])
                for point in selected_resolved.values
            )
            if helicity.structural_zero:
                selected_checks.append(_series((0.0,) * len(points), row, helicity.id))
                continue
            candidate = hidden(fixed_flow, helicity)
            selected_hidden.append(candidate)
            selected_reference.append(row)
            selected_queries.append(_query(fixed_flow, helicity))
            selected_checks.append(_series(candidate, row, helicity.id))

        flow_hidden: list[tuple[float, ...]] = []
        flow_reference: list[tuple[complex, ...]] = []
        flow_queries: list[Query] = []
        flow_checks: list[dict[str, object]] = []
        for flow in probe_runtime.physics.color_flows:
            if flow.id not in all_color_axis:
                raise GateError("probe carrier color flow is absent from authority")
            row = tuple(
                complex(point[all_fixed_h][all_color_axis[flow.id]])
                for point in all_resolved.values
            )
            try:
                candidate = hidden(flow, fixed_helicity)
            except Exception as error:
                raise GateError(
                    "the private topology carrier cannot serve the all-flow "
                    f"single-helicity query {flow.id}; this is a design finding"
                ) from error
            flow_hidden.append(candidate)
            flow_reference.append(row)
            flow_queries.append(_query(flow, fixed_helicity))
            flow_checks.append(_series(candidate, row, flow.id))

        selected_hidden_sum = _sum(selected_hidden, len(points))
        flow_hidden_sum = _sum(flow_hidden, len(points))
        selected_family_rows, selected_family_correctness = _hidden_family_correctness(
            probe,
            probe_carrier,
            retained,
            selected_queries,
            points,
            selected_reference,
            [cache[query][2] for query in selected_queries],
            enable_color_projection,
        )
        flow_family_rows, flow_family_correctness = _hidden_family_correctness(
            probe,
            probe_carrier,
            retained,
            flow_queries,
            points,
            flow_reference,
            [cache[query][2] for query in flow_queries],
            enable_color_projection,
        )
        selected_family_sum = _sum(selected_family_rows, len(points))
        flow_family_sum = _sum(flow_family_rows, len(points))

        fixed_hidden = hidden(fixed_flow, fixed_helicity)
        public_helicity_axis, public_color_axis = _axes(selected_public_resolved)
        fixed_public = tuple(
            complex(
                point[public_helicity_axis[HELICITY_ID]][public_color_axis[FLOW_ID]]
            )
            for point in selected_public_resolved.values
        )
        digest = point_digest((points[0],))
        sanity = _anchor_checks(
            _anchor(amplicol),
            digest,
            public_component=fixed_public[0],
            hidden_component=fixed_hidden[0],
            public_sum=selected_public[0],
            hidden_sum=selected_hidden_sum[0],
        )

        selected_hidden_timing = _hidden_timing(
            probe,
            probe_carrier,
            retained,
            selected_queries,
            timing_points,
            target,
            enable_color_projection,
        )
        flow_hidden_timing = _hidden_timing(
            probe,
            probe_carrier,
            retained,
            flow_queries,
            timing_points,
            target,
            enable_color_projection,
        )
        selected_structural = _query_family_census(
            family_probe,
            retained,
            selected_queries,
            selected_hidden_timing,
            enable_color_projection,
        )
        flow_structural = _query_family_census(
            family_probe,
            retained,
            flow_queries,
            flow_hidden_timing,
            enable_color_projection,
        )
        _, selected_family_timing = _hidden_family_timing(
            probe,
            probe_carrier,
            retained,
            selected_queries,
            timing_points,
            target,
            enable_color_projection,
        )
        _, flow_family_timing = _hidden_family_timing(
            probe,
            probe_carrier,
            retained,
            flow_queries,
            timing_points,
            target,
            enable_color_projection,
        )
        selected_family_census = _mapping(
            selected_family_timing.get("work_census"),
            "selected on-the-fly family census",
        )
        flow_family_census = _mapping(
            flow_family_timing.get("work_census"),
            "all-flow on-the-fly family census",
        )
        _assert_executable_family_matches_structural_census(
            selected_family_census, selected_structural
        )
        _assert_executable_family_matches_structural_census(
            flow_family_census, flow_structural
        )

        for workload, private in (
            ("selected_flow_helicity_sum", selected_family_timing),
            ("all_flow_single_helicity", flow_family_timing),
        ):
            private["diagnostic_ratio_to_recurrence_outer_wall"] = float(
                private["private_warmed_seconds_per_point"]
            ) / float(timings["recurrence"][workload]["wall_seconds_per_point"])
            private["ratio_is_acceptance_gate"] = False

        def comparison(
            workload: str,
            family_census: Mapping[str, object],
            recurrence_census: Mapping[str, object],
            compiled_census: Mapping[str, object],
        ) -> dict[str, object]:
            recurrence_live = _mapping(
                recurrence_census.get("selector_live"),
                "recurrence selector-live census",
            )
            compiled_levels = _mapping(
                compiled_census.get("levels"), "compiled DAG levels"
            )
            compiled_leaf = _mapping(
                compiled_levels.get("executed_leaf"), "compiled executed-leaf census"
            )
            return {
                "logical_current_counts": {
                    "on_the_fly_warmed_family_unique": family_census[
                        "union_unique_current_count"
                    ],
                    "recurrence_selector_live": recurrence_live["current_count"],
                    "compiled_executed_leaf": compiled_leaf["current_count"],
                },
                "current_component_counts": {
                    "on_the_fly_warmed_family_unique": family_census[
                        "union_unique_current_component_count"
                    ],
                    "recurrence_selector_live": recurrence_live[
                        "semantic_current_component_count"
                    ],
                    "compiled_executed_leaf": compiled_leaf["current_component_count"],
                },
                "role_native_operation_counts": {
                    "on_the_fly_transition_rows": {
                        role: family_census[f"union_{role}_rows"]
                        for role in (
                            "source",
                            "contribution",
                            "finalization",
                            "closure",
                        )
                    },
                    "recurrence_selector_live_rows": {
                        "source": recurrence_live["source_row_count"],
                        "contribution": recurrence_live["contribution_count"],
                        "finalization": recurrence_live["finalization_count"],
                        "closure": recurrence_live["closure_count"],
                    },
                    "compiled_executed_leaf": {
                        "interaction_attachments": compiled_leaf[
                            "interaction_attachment_count"
                        ],
                        "interaction_evaluations": compiled_leaf[
                            "interaction_evaluation_count"
                        ],
                    },
                },
                "on_the_fly_private_warmed_unique_currents_and_rows": family_census,
                "on_the_fly_public_warmed_rows_and_grouped_calls": (
                    public_profiles[workload]["work"]
                ),
                "recurrence_public_warmed_rows_and_grouped_calls": timings[
                    "recurrence"
                ][workload]["recurrence_runtime_work"],
                "recurrence_persisted_and_selector_live_plan": recurrence_census,
                "compiled_primary_program_and_executed_leaf_dag": compiled_census,
                "counter_warning": (
                    "on-the-fly/recurrence rows are transitions and grouped executor "
                    "calls; compiled counts are currents/components/interaction "
                    "attachments/evaluations. Unlike counters are not equated."
                ),
            }

        cold = tuple(value[1] for value in cache.values())
        return {
            "kind": "pyamplicol-on-the-fly-lc-gate",
            "status": "passed",
            "process": PROCESS,
            "process_id": PROCESS_ID,
            "artifact": str(artifact),
            "artifact_id": runtime.artifact_id,
            "on_the_fly_artifact": artifact_contract,
            "private_probe_carrier": {
                "execution_mode": "recurrence",
                "artifact_id": probe_runtime.artifact_id,
                "role": (
                    "ephemeral topology carrier for private Rust diagnostics only; "
                    "not the candidate, not an authority, and excluded from candidate "
                    "generation/load timing"
                ),
                "retained_after_gate": False,
            },
            "layout": "topology-replay",
            "color_projection": {
                "query": _query(fixed_flow, fixed_helicity).label,
                **{
                    name: value
                    for name, value in cache[_query(fixed_flow, fixed_helicity)][
                        2
                    ].items()
                    if "projection_" in name
                },
            },
            "public_correctness": public_correctness,
            "public_profiles": public_profiles,
            "clear_and_rebuild": clear_checks,
            "dense_correctness_authority": _dense_authority(
                selected_recurrence_runtime,
                all_flow_recurrence_runtime,
                len(points),
                compiled_selected_runtime=compiled_selected_runtime,
                compiled_all_flow_runtime=compiled_all_flow_runtime,
            ),
            "prepared_model": {
                "path": str(prepared_path),
                "load_seconds": prepared_load_seconds,
            },
            "cold_boundaries": {
                "on_the_fly_generation_seconds": generation_seconds,
                "on_the_fly_runtime_load_seconds": load_seconds,
                "generation_census": generation_census,
                "semantics": (
                    "cold service generation and Runtime.load/post-build work are not "
                    "warmed work; the recurrence probe carrier is excluded"
                ),
            },
            "explicit_artifact_roots": {
                "on_the_fly_candidate": str(artifact),
                "recurrence_selected": str(recurrence_selected_artifact),
                "recurrence_all_flow": str(recurrence_all_flow_artifact),
                "compiled_selected": str(compiled_selected_artifact),
                "compiled_all_flow": str(compiled_all_flow_artifact),
            },
            "comparator_artifacts": {
                "recurrence_selected_flow_helicity_sum": (recurrence_selected_census),
                "recurrence_all_flow_single_helicity": recurrence_all_flow_census,
                "compiled_selected_flow_helicity_sum": compiled_selected_census,
                "compiled_all_flow_single_helicity": compiled_all_flow_census,
            },
            "points": {
                "seeds": SEEDS,
                "seed_101_digest": digest,
                "dense": len(points),
            },
            "cold_private_query_outer_seconds": {
                "queries": len(cold),
                "minimum": min(cold),
                "median": statistics.median(cold),
                "maximum": max(cold),
                "total": math.fsum(cold),
                "not_warmed_work": True,
            },
            "selected_flow_helicity_sum": {
                "private_component_checks": _summaries(selected_checks),
                "private_probe_aggregate": _series(
                    selected_hidden_sum,
                    selected_authority,
                    "selected-flow private probe/recurrence authority",
                ),
                "private_family_correctness": selected_family_correctness,
                "private_family_aggregate": _series(
                    selected_family_sum,
                    selected_authority,
                    "selected-flow private family/recurrence authority",
                ),
                "private_query_local_warm": selected_hidden_timing,
                "private_family_warm": selected_family_timing,
                "public_timings": {
                    lane: value["selected_flow_helicity_sum"]
                    for lane, value in timings.items()
                },
                "descriptive_work_comparison": comparison(
                    "selected_flow_helicity_sum",
                    selected_family_census,
                    recurrence_selected_census,
                    compiled_selected_census,
                ),
            },
            "all_flow_single_helicity": {
                "private_component_checks": _summaries(flow_checks),
                "private_probe_aggregate": _series(
                    flow_hidden_sum,
                    all_authority,
                    "all-flow private probe/recurrence authority",
                ),
                "private_family_correctness": flow_family_correctness,
                "private_family_aggregate": _series(
                    flow_family_sum,
                    all_authority,
                    "all-flow private family/recurrence authority",
                ),
                "private_query_local_warm": flow_hidden_timing,
                "private_family_warm": flow_family_timing,
                "public_timings": {
                    lane: value["all_flow_single_helicity"]
                    for lane, value in timings.items()
                },
                "descriptive_work_comparison": comparison(
                    "all_flow_single_helicity",
                    flow_family_census,
                    recurrence_all_flow_census,
                    compiled_all_flow_census,
                ),
            },
            "amplicol_sanity": sanity,
            "clock_contract": {
                "on_the_fly_private_warmed": (
                    "private Rust family clock excluding source crossing and Python"
                ),
                "on_the_fly_public": (
                    "BenchmarkRunner evaluator/evaluator-total and outer wall clocks"
                ),
                "recurrence": ("recurrence core/evaluator-total and outer wall clocks"),
                "compiled": (
                    "compiled evaluator/evaluator-total and outer wall clocks"
                ),
                "amplicol": ("direct execution and worker wall clocks when supplied"),
                "relationship": (
                    "all scopes are independent; no equality or interchangeability "
                    "is assumed"
                ),
            },
            "integration_contract": {
                "python_api": (
                    "existing Generator, Runtime.load/evaluate/evaluate_resolved/"
                    "clear, existing runtime profile backend, and BenchmarkRunner"
                ),
                "rust_lane": "separate native on-the-fly execution lane",
                "new_public_evaluation_api": False,
            },
            "performance_is_acceptance_gate": False,
        }


def _run_public_correctness_only(
    output: Path,
    prepared_model: Path | None,
    candidate_artifact: Path | None,
    recurrence_selected_artifact: Path,
    recurrence_all_flow_artifact: Path,
    compiled_selected_artifact: Path,
    compiled_all_flow_artifact: Path,
    target: float,
    batch: int,
) -> dict[str, object]:
    if (prepared_model is None) == (candidate_artifact is None):
        raise GateError(
            "public correctness-only mode needs exactly one candidate source"
        )

    generation_seconds: float | None
    generation_census: dict[str, int] | None
    prepared_record: dict[str, object]
    if candidate_artifact is None:
        if prepared_model is None:  # guarded by the exclusive candidate-source check
            raise GateError("public correctness-only generation needs a prepared model")
        artifact = output / "artifact"
        (
            generation_seconds,
            generation_census,
            _model,
            prepared_path,
            prepared_load_seconds,
        ) = _generate_candidate_with_prepared_model(artifact, prepared_model)
        candidate_source = "generated-in-guarded-worker"
        prepared_record = {
            "status": "loaded",
            "path": str(prepared_path),
            "load_seconds": prepared_load_seconds,
        }
    else:
        artifact = _candidate_artifact_path(candidate_artifact)
        generation_seconds = None
        generation_census = None
        candidate_source = "reused-existing-artifact"
        prepared_record = {
            "status": "not-loaded",
            "path": None,
            "load_seconds": None,
        }

    public = _run_public_gate_phase(
        artifact,
        recurrence_selected_artifact,
        recurrence_all_flow_artifact,
        compiled_selected_artifact,
        compiled_all_flow_artifact,
        target,
        batch,
    )
    correctness = public.correctness
    public_timings = {
        workload: {
            lane: public.timings[lane][workload]
            for lane in ("on_the_fly", "recurrence", "compiled")
        }
        for workload in (
            "selected_flow_helicity_sum",
            "all_flow_single_helicity",
        )
    }

    def comparator(path: Path, runtime: object, role: str) -> dict[str, object]:
        return {
            "artifact": str(Path(os.path.abspath(path.expanduser()))),
            "artifact_id": runtime.artifact_id,
            "execution_mode": runtime.execution_mode,
            "role": role,
        }

    skipped_reason = (
        "the explicit --public-correctness-only developer lane forbids private "
        "carrier generation, hidden probes, private census, and private timing"
    )
    return {
        "kind": "pyamplicol-on-the-fly-lc-public-correctness-only",
        "status": "passed",
        "scope": "provisional-public-correctness-only",
        "provisional": True,
        "public_only": True,
        "full_gate_status": "not-run",
        "source_bound": False,
        "source_binding": "not-asserted",
        "process": PROCESS,
        "process_id": PROCESS_ID,
        "artifact": str(artifact),
        "artifact_id": public.runtime.artifact_id,
        "candidate_source": candidate_source,
        "on_the_fly_artifact": public.artifact_contract,
        "layout": "topology-replay",
        "public_correctness": correctness.public_correctness,
        "clear_and_rebuild": correctness.clear_checks,
        "public_profiles": public.public_profiles,
        "public_timings": public_timings,
        "dense_correctness_authority": _dense_authority(
            public.selected_recurrence_runtime,
            public.all_flow_recurrence_runtime,
            len(public.points),
            compiled_selected_runtime=public.compiled_selected_runtime,
            compiled_all_flow_runtime=public.compiled_all_flow_runtime,
        ),
        "prepared_model": prepared_record,
        "cold_boundaries": {
            "on_the_fly_generation_seconds": generation_seconds,
            "on_the_fly_runtime_load_seconds": public.load_seconds,
            "generation_census": generation_census,
            "generation_reused": candidate_artifact is not None,
        },
        "explicit_artifact_roots": {
            "on_the_fly_candidate": str(artifact),
            "recurrence_selected": str(recurrence_selected_artifact),
            "recurrence_all_flow": str(recurrence_all_flow_artifact),
            "compiled_selected": str(compiled_selected_artifact),
            "compiled_all_flow": str(compiled_all_flow_artifact),
        },
        "comparator_artifacts": {
            "recurrence_selected_flow_helicity_sum": comparator(
                recurrence_selected_artifact,
                public.selected_recurrence_runtime,
                "selected-flow/helicity-sum recurrence authority",
            ),
            "recurrence_all_flow_single_helicity": comparator(
                recurrence_all_flow_artifact,
                public.all_flow_recurrence_runtime,
                "all-flow/single-helicity recurrence authority",
            ),
            "compiled_selected_flow_helicity_sum": comparator(
                compiled_selected_artifact,
                public.compiled_selected_runtime,
                "selected-flow/helicity-sum compiled authority",
            ),
            "compiled_all_flow_single_helicity": comparator(
                compiled_all_flow_artifact,
                public.compiled_all_flow_runtime,
                "all-flow/single-helicity compiled authority",
            ),
        },
        "points": {
            "seeds": SEEDS,
            "seed_101_digest": point_digest((public.points[0],)),
            "dense": len(public.points),
        },
        "private_probe_carrier": {
            "status": "skipped",
            "available": False,
            "reason": skipped_reason,
            "retained_after_gate": False,
        },
        "private_census": {
            "status": "unavailable",
            "query_local": "unavailable",
            "query_family": "unavailable",
            "reason": skipped_reason,
        },
        "private_timing": {
            "status": "unavailable",
            "query_local": "unavailable",
            "query_family": "unavailable",
            "reason": skipped_reason,
        },
        "clock_contract": {
            "on_the_fly_public": (
                "BenchmarkRunner evaluator/evaluator-total and outer wall clocks"
            ),
            "recurrence": "recurrence core/evaluator-total and outer wall clocks",
            "compiled": "compiled evaluator/evaluator-total and outer wall clocks",
            "private": "unavailable in provisional public-only mode",
            "relationship": (
                "all public clock scopes are independent; no equality or "
                "interchangeability is assumed"
            ),
        },
        "integration_contract": {
            "python_api": (
                "existing Runtime.load/evaluate/evaluate_resolved/clear, existing "
                "runtime profile backend, and BenchmarkRunner"
            ),
            "new_public_evaluation_api": False,
            "private_native_diagnostics_invoked": False,
        },
        "performance_is_acceptance_gate": False,
    }


def _write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to replace evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise GateError(f"cannot write evidence: {path}") from error


def _read(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read evidence: {path}") from error
    if not isinstance(result, dict):
        raise GateError(f"evidence is not an object: {path}")
    return result


def _watchdog_summary(report: Mapping[str, object]) -> dict[str, object]:
    execution = report.get("execution")
    enforcement = report.get("enforcement")
    if not isinstance(execution, Mapping) or not isinstance(enforcement, Mapping):
        raise GateError("watchdog report is incomplete")
    return {
        "passes": report.get("passes"),
        "outcome": execution.get("outcome"),
        "reason": execution.get("reason"),
        "elapsed_wall_seconds": execution.get("elapsed_wall_seconds"),
        "limit_bytes": enforcement.get("limit_bytes"),
        "peak_rss_bytes": enforcement.get("peak_rss_bytes"),
        "peak_physical_footprint_bytes": enforcement.get(
            "peak_physical_footprint_bytes"
        ),
        "peak_guard_bytes": enforcement.get("peak_guard_bytes"),
        "peak_processes": enforcement.get("peak_processes"),
    }


def _physical_footprint_probe() -> PhysicalFootprintProbe | None:
    if platform.system() != "Darwin":
        return None
    try:
        return DarwinPhysicalFootprintProbe()
    except Exception as error:
        raise GateError("Darwin physical-footprint guard is unavailable") from error


def _worker_command(arguments: argparse.Namespace, output: Path) -> list[str]:
    _validate_mode_arguments(arguments)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--target-runtime",
        str(arguments.target_runtime),
        "--batch-size",
        str(arguments.batch_size),
    ]
    if arguments.prepared_model is not None:
        command.extend(
            ("--prepared-model", str(_prepared_model_path(arguments.prepared_model)))
        )
    else:
        command.extend(
            (
                "--candidate-artifact",
                str(_candidate_artifact_path(arguments.candidate_artifact)),
            )
        )
    if arguments.public_correctness_only:
        command.append("--public-correctness-only")
    if arguments.amplicol_result is not None:
        amplicol = Path(os.path.abspath(arguments.amplicol_result.expanduser()))
        command.extend(("--amplicol-result", str(amplicol)))
    for option, value in (
        ("--recurrence-selected-artifact", arguments.recurrence_selected_artifact),
        ("--recurrence-all-flow-artifact", arguments.recurrence_all_flow_artifact),
        ("--compiled-selected-artifact", arguments.compiled_selected_artifact),
        ("--compiled-all-flow-artifact", arguments.compiled_all_flow_artifact),
    ):
        command.extend((option, os.path.abspath(value.expanduser())))
    if arguments.bypass_color_projection:
        command.append("--bypass-color-projection")
    return command


def _worker_main(arguments: argparse.Namespace) -> int:
    _validate_mode_arguments(arguments)
    destination = arguments.output / "worker.json"
    try:
        if arguments.public_correctness_only:
            result = _run_public_correctness_only(
                arguments.output,
                arguments.prepared_model,
                arguments.candidate_artifact,
                arguments.recurrence_selected_artifact,
                arguments.recurrence_all_flow_artifact,
                arguments.compiled_selected_artifact,
                arguments.compiled_all_flow_artifact,
                arguments.target_runtime,
                arguments.batch_size,
            )
        else:
            result = _run(
                arguments.output,
                arguments.prepared_model,
                arguments.amplicol_result,
                arguments.recurrence_selected_artifact,
                arguments.recurrence_all_flow_artifact,
                arguments.compiled_selected_artifact,
                arguments.compiled_all_flow_artifact,
                arguments.target_runtime,
                arguments.batch_size,
                not arguments.bypass_color_projection,
            )
    except Exception as error:
        _write(
            destination,
            {
                "kind": (
                    "pyamplicol-on-the-fly-lc-public-correctness-only"
                    if arguments.public_correctness_only
                    else "pyamplicol-on-the-fly-lc-gate"
                ),
                "status": "failed",
                "scope": (
                    "provisional-public-correctness-only"
                    if arguments.public_correctness_only
                    else "full"
                ),
                "error": str(error),
            },
        )
        print(f"on-the-fly LC gate failed: {error}", file=sys.stderr)
        return 1
    _write(destination, result)
    return 0


def _driver_report(
    worker: Mapping[str, object], watchdog: Mapping[str, object]
) -> dict[str, object]:
    public_only = worker.get("public_only") is True
    result: dict[str, object] = {
        "kind": (
            "pyamplicol-on-the-fly-lc-public-correctness-only-run"
            if public_only
            else "pyamplicol-on-the-fly-lc-gate-run"
        ),
        "status": "passed",
        "scope": ("provisional-public-correctness-only" if public_only else "full"),
        "provisional": public_only,
        "worker": worker,
        "watchdog": watchdog,
    }
    if public_only:
        result.update({"source_bound": False, "source_binding": "not-asserted"})
    return result


def _driver_main(arguments: argparse.Namespace) -> int:
    output = Path(os.path.abspath(arguments.output.expanduser()))
    if output.exists() or output.is_symlink():
        raise GateError(f"output already exists: {output}")
    output.mkdir(parents=True)
    watchdog_path = output / "watchdog.json"
    exit_code = run_guarded(
        _worker_command(arguments, output),
        limit_bytes=WATCHDOG_BYTES,
        report_path=watchdog_path,
        physical_footprint_probe=_physical_footprint_probe(),
    )
    watchdog = _watchdog_summary(_read(watchdog_path))
    worker_path = output / "worker.json"
    worker = _read(worker_path) if worker_path.is_file() else None
    if exit_code != 0 or watchdog["passes"] is not True:
        detail = None if worker is None else worker.get("error")
        raise GateError(
            f"guarded worker failed: {watchdog['outcome']!r}, {detail!r}; {output}"
        )
    if worker is None or worker.get("status") != "passed":
        raise GateError("guarded worker omitted its successful result")
    public_only = worker.get("public_only") is True
    destination = output / "report.json"
    _write(destination, _driver_report(worker, watchdog))
    label = "public correctness-only lane" if public_only else "LC gate"
    print(f"On-the-fly {label} passed: {destination}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return _worker_main(arguments) if arguments.worker else _driver_main(arguments)
    except GateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
