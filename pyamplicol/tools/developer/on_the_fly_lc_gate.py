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
import time
from collections.abc import Callable, Mapping, Sequence
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
from pyamplicol.models.builtin.validation import generic_validation_point  # noqa: E402
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
PROCESS_ID = "otf_dd_tt_gg"
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
    parser.add_argument(
        "--prepared-model",
        required=True,
        type=Path,
        help="explicit prepared built-in-SM .pyamplicol-model bundle",
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
        "--bypass-color-projection",
        action="store_true",
        help=(
            "diagnostic: execute the selected graph before late color-alias "
            "projection"
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _config() -> RunConfig:
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
        "maximum_conditioned_residual": rows[worst][
            "maximum_conditioned_residual"
        ],
        "maximum_absolute_difference": max(
            float(row["maximum_absolute_difference"]) for row in rows
        ),
        "worst_component_index": worst,
        "worst": rows[worst],
    }


def _sum(rows: Sequence[Sequence[object]], point_count: int) -> tuple[float, ...]:
    parsed = tuple(tuple(_real(value, "component") for value in row) for row in rows)
    if (
        not parsed
        or point_count <= 0
        or any(len(row) != point_count for row in parsed)
    ):
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


def _generate(
    artifact: Path, model: CompiledModel
) -> tuple[float, RetainedInputs]:
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

    started = time.perf_counter()
    with mock.patch.object(
        generation_service, "_invoke_rust_recurrence_lowering_v2", capture
    ):
        Generator(_config()).generate(
            ProcessRequest.parse(PROCESS, name=PROCESS_ID), artifact, model=model
        )
    elapsed = time.perf_counter() - started
    if len(captured) != 1:
        raise GateError(f"expected one lowering call, observed {len(captured)}")
    return elapsed, captured[0]


def _generate_with_prepared_model(
    artifact: Path, prepared_model: Path
) -> tuple[float, RetainedInputs, Path, float]:
    resolved = _prepared_model_path(prepared_model)
    started = time.perf_counter()
    model = ModelSource.from_path(resolved).compile()
    load_seconds = time.perf_counter() - started
    generation_seconds, retained = _generate(artifact, model)
    return generation_seconds, retained, resolved, load_seconds


def _selectors(physics: object) -> tuple[object, object]:
    flows = tuple(
        flow
        for flow in getattr(physics, "color_flows", ())
        if flow.id == FLOW_ID and tuple(flow.word) == FLOW_WORD
    )
    helicities = tuple(
        helicity
        for helicity in getattr(physics, "helicities", ())
        if helicity.id == HELICITY_ID
        and tuple(helicity.values) == HELICITY_VALUES
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
                (member.flow_index, list(member.helicities))
                for member in query_family
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
            and math.isclose(
                float(per_point), expected, rel_tol=1.0e-15, abs_tol=0.0
            )
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


def _dense_authority(runtime: object, point_count: int) -> dict[str, object]:
    return {
        "authority_kind": "validated_production_pyamplicol",
        "runtime_api": "Runtime.evaluate_resolved",
        "artifact_id": runtime.artifact_id,
        "point_count": point_count,
        "certifies": (
            "selected_flow_helicity_sum",
            "all_flow_single_helicity",
        ),
    }


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


def _public_timing(result: object) -> dict[str, object]:
    return {
        "sample_count": result.sample_count,
        "repetitions_per_sample": result.repetitions_per_sample,
        "wall_seconds_per_point": result.wall_time_per_point,
        "recurrence_core_seconds_per_point": result.evaluator_time_per_point,
        "evaluator_seconds_per_point": result.evaluator_time_per_point,
        "evaluator_total_seconds_per_point": result.evaluator_total_time_per_point,
        "clock_attribution": {
            "wall_seconds_per_point": "BenchmarkRunner outer wall clock",
            "recurrence_core_seconds_per_point": (
                "recurrence runtime core clock reported by the comparator"
            ),
            "relationship": (
                "independent clocks; equality is neither assumed nor asserted"
            ),
        },
        "interrupted": result.interrupted,
        "effective_config": dataclasses.asdict(result.effective_config),
        "uncertainty": dataclasses.asdict(result.uncertainty),
        "recurrence_runtime_work": _public_recurrence_work(result),
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
        or counts["dynamic_current_component_occurrence_count"]
        != component_occurrences
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
        if counts[f"union_{role}_executor_call_groups"] > counts[
            f"union_{role}_rows"
        ]:
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


def _artifact_execution(
    artifact: Path, execution_mode: str
) -> tuple[object, Mapping[str, object], Path]:
    try:
        root = artifact.expanduser().resolve(strict=True)
    except OSError as error:
        raise GateError(f"comparator artifact does not exist: {artifact}") from error
    if not root.is_dir():
        raise GateError(f"comparator artifact is not a directory: {root}")
    runtime = Runtime.load(root)
    physics = runtime.physics
    if runtime.execution_mode != execution_mode or physics.color_accuracy != "lc":
        raise GateError(
            f"comparator is not an LC {execution_mode} artifact: {root}"
        )
    process_id = physics.process_id
    if not isinstance(process_id, str) or not process_id:
        raise GateError("comparator runtime has no process identity")
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
    artifact: Path, *, layout: str
) -> dict[str, object]:
    runtime, payload, root = _artifact_execution(artifact, "recurrence")
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
        "current_count": _count(
            summary.get("current_count"), f"{label} current count"
        ),
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
    artifact: Path, *, workload: str
) -> dict[str, object]:
    runtime, payload, root = _artifact_execution(artifact, "compiled")
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
            if public_index in tuple(
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
                if public_index in tuple(
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
        "private_warmed_elapsed_seconds": measured[
            "private_warmed_elapsed_seconds"
        ],
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
        "wall_seconds_per_point": payload.get("wall_seconds_per_point"),
        "clock_attribution": (
            "stored original-AmpliCol worker wall clock; independent of pyAmpliCol "
            "and on-the-fly clocks"
        ),
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
    generation_seconds, retained, prepared_path, prepared_load_seconds = (
        _generate_with_prepared_model(artifact, prepared_model)
    )
    started = time.perf_counter()
    runtime = Runtime.load(artifact, process=PROCESS_ID)
    load_seconds = time.perf_counter() - started
    if runtime.execution_mode != "recurrence" or runtime.physics.color_accuracy != "lc":
        raise GateError("generated artifact is not an LC recurrence artifact")
    fixed_flow, fixed_helicity = _selectors(runtime.physics)
    points = _points()
    resolved = runtime.evaluate_resolved(points)
    helicity_axis, color_axis = _axes(resolved)
    fixed_h = helicity_axis[HELICITY_ID]
    fixed_c = color_axis[FLOW_ID]
    probe = _probe()
    family_probe = _family_probe()
    cache: dict[Query, tuple[tuple[float, ...], float, dict[str, Any]]] = {}

    def hidden(flow: object, helicity: object) -> tuple[float, ...]:
        query = _query(flow, helicity)
        if query not in cache:
            before = time.perf_counter()
            report = _invoke(
                probe,
                artifact,
                retained,
                query,
                points,
                enable_color_projection=enable_color_projection,
            )
            values = _probe_values(report, len(points))
            cache[query] = (
                values,
                time.perf_counter() - before,
                report,
            )
        return cache[query][0]

    selected_hidden, selected_reference = [], []
    selected_queries, selected_checks = [], []
    for helicity in runtime.physics.helicities:
        row = tuple(
            complex(point[helicity_axis[helicity.id]][fixed_c])
            for point in resolved.values
        )
        if helicity.structural_zero:
            selected_checks.append(_series((0.0,) * len(points), row, helicity.id))
            continue
        candidate = hidden(fixed_flow, helicity)
        selected_hidden.append(candidate)
        selected_reference.append(row)
        selected_queries.append(_query(fixed_flow, helicity))
        selected_checks.append(_series(candidate, row, helicity.id))
    selected_hidden_sum = _sum(selected_hidden, len(points))
    selected_resolved_sum = _sum(selected_reference, len(points))
    selected_public_sum = runtime.evaluate(points, color_flows=(FLOW_ID,))

    flow_hidden, flow_reference, flow_queries, flow_checks = [], [], [], []
    for flow in runtime.physics.color_flows:
        row = tuple(
            complex(point[fixed_h][color_axis[flow.id]]) for point in resolved.values
        )
        try:
            candidate = hidden(flow, fixed_helicity)
        except Exception as error:
            raise GateError(
                "the single topology-replay artifact cannot serve the all-flow "
                f"single-helicity query {flow.id}; this is a design finding"
            ) from error
        flow_hidden.append(candidate)
        flow_reference.append(row)
        flow_queries.append(_query(flow, fixed_helicity))
        flow_checks.append(_series(candidate, row, flow.id))
    flow_hidden_sum = _sum(flow_hidden, len(points))
    flow_resolved_sum = _sum(flow_reference, len(points))
    flow_public_sum = runtime.evaluate(points, helicities=(HELICITY_ID,))

    selected_family_rows, selected_family_correctness = _hidden_family_correctness(
        probe,
        artifact,
        retained,
        selected_queries,
        points,
        selected_reference,
        [cache[query][2] for query in selected_queries],
        enable_color_projection,
    )
    flow_family_rows, flow_family_correctness = _hidden_family_correctness(
        probe,
        artifact,
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
    fixed_public = tuple(complex(point[fixed_h][fixed_c]) for point in resolved.values)
    digest = point_digest((points[0],))
    sanity = _anchor_checks(
        _anchor(amplicol),
        digest,
        public_component=fixed_public[0],
        hidden_component=fixed_hidden[0],
        public_sum=selected_public_sum[0],
        hidden_sum=selected_hidden_sum[0],
    )

    timing_points = _repeat(points, batch)
    selected_hidden_timing = _hidden_timing(
        probe,
        artifact,
        retained,
        selected_queries,
        timing_points,
        target,
        enable_color_projection,
    )
    flow_hidden_timing = _hidden_timing(
        probe,
        artifact,
        retained,
        flow_queries,
        timing_points,
        target,
        enable_color_projection,
    )
    selected_structural_census = _query_family_census(
        family_probe,
        retained,
        selected_queries,
        selected_hidden_timing,
        enable_color_projection,
    )
    selected_hidden_timing["structural_family_census"] = selected_structural_census
    flow_structural_census = _query_family_census(
        family_probe,
        retained,
        flow_queries,
        flow_hidden_timing,
        enable_color_projection,
    )
    flow_hidden_timing["structural_family_census"] = flow_structural_census
    _, selected_family_timing = _hidden_family_timing(
        probe,
        artifact,
        retained,
        selected_queries,
        timing_points,
        target,
        enable_color_projection,
    )
    _, flow_family_timing = _hidden_family_timing(
        probe,
        artifact,
        retained,
        flow_queries,
        timing_points,
        target,
        enable_color_projection,
    )
    selected_family_census = selected_family_timing["work_census"]
    flow_family_census = flow_family_timing["work_census"]
    if not isinstance(selected_family_census, Mapping) or not isinstance(
        flow_family_census, Mapping
    ):
        raise GateError("executable family timings lost their work census")
    _assert_executable_family_matches_structural_census(
        selected_family_census, selected_structural_census
    )
    _assert_executable_family_matches_structural_census(
        flow_family_census, flow_structural_census
    )
    recurrence_selected_census = _recurrence_artifact_census(
        recurrence_selected_artifact, layout="topology-replay"
    )
    recurrence_all_flow_census = _recurrence_artifact_census(
        recurrence_all_flow_artifact, layout="all-flow-union"
    )
    compiled_selected_census = _compiled_artifact_census(
        compiled_selected_artifact, workload="selected_flow_helicity_sum"
    )
    compiled_all_flow_census = _compiled_artifact_census(
        compiled_all_flow_artifact, workload="all_flow_single_helicity"
    )
    selected_recurrence_runtime = Runtime.load(recurrence_selected_artifact)
    all_flow_recurrence_runtime = Runtime.load(recurrence_all_flow_artifact)
    selected_public_timing = _public_timing(
        BenchmarkRunner(_benchmark_config(target, batch, flows=(FLOW_ID,))).run(
            selected_recurrence_runtime, points=timing_points
        )
    )
    flow_public_timing = _public_timing(
        BenchmarkRunner(
            _benchmark_config(target, batch, helicities=(HELICITY_ID,))
        ).run(all_flow_recurrence_runtime, points=timing_points)
    )
    for hidden_timing, family_timing, public_timing in (
        (selected_hidden_timing, selected_family_timing, selected_public_timing),
        (flow_hidden_timing, flow_family_timing, flow_public_timing),
    ):
        wall = float(public_timing["wall_seconds_per_point"])
        if not math.isfinite(wall) or wall <= 0.0:
            raise GateError("public benchmark returned a non-positive wall time")
        hidden_timing["provisional_wall_ratio"] = (
            float(hidden_timing["additive_seconds_per_point"]) / wall
        )
        family_timing["provisional_wall_ratio"] = (
            float(family_timing["private_warmed_seconds_per_point"]) / wall
        )

    cold = tuple(value[1] for value in cache.values())
    return {
        "kind": "pyamplicol-on-the-fly-lc-gate",
        "status": "passed",
        "process": PROCESS,
        "process_id": PROCESS_ID,
        "artifact": str(artifact),
        "artifact_id": runtime.artifact_id,
        "on_the_fly_topology_artifact": {
            "artifact": str(artifact),
            "artifact_id": runtime.artifact_id,
            "role": "one topology source for both on-the-fly query families",
        },
        "layout": "topology-replay",
        "color_projection": {
            "query": _query(fixed_flow, fixed_helicity).label,
            **{
                name: value
                for name, value in cache[_query(fixed_flow, fixed_helicity)][2].items()
                if "projection_" in name
            },
        },
        "dense_correctness_authority": _dense_authority(runtime, len(points)),
        "prepared_model": {
            "path": str(prepared_path),
            "load_seconds": prepared_load_seconds,
        },
        "generation_seconds": generation_seconds,
        "runtime_load_seconds": load_seconds,
        "comparator_artifacts": {
            "recurrence_selected_flow_helicity_sum": recurrence_selected_census,
            "recurrence_all_flow_single_helicity": recurrence_all_flow_census,
            "compiled_selected_flow_helicity_sum": compiled_selected_census,
            "compiled_all_flow_single_helicity": compiled_all_flow_census,
        },
        "points": {"seeds": SEEDS, "seed_101_digest": digest, "dense": len(points)},
        "cold_query_outer_seconds": {
            "queries": len(cold),
            "minimum": min(cold),
            "median": statistics.median(cold),
            "maximum": max(cold),
            "total": math.fsum(cold),
        },
        "selected_flow_helicity_sum": {
            "component_checks": _summaries(selected_checks),
            "hidden_aggregate": _series(
                selected_hidden_sum, selected_public_sum, "selected-flow aggregate"
            ),
            "resolved_aggregate": _series(
                selected_resolved_sum,
                selected_public_sum,
                "selected-flow resolved aggregate",
            ),
            "family_correctness": selected_family_correctness,
            "family_aggregate": _series(
                selected_family_sum,
                selected_public_sum,
                "selected-flow family aggregate",
            ),
            "hidden_warm": selected_hidden_timing,
            "family_warm": selected_family_timing,
            "public_warm": selected_public_timing,
            "descriptive_work_comparison": {
                "on_the_fly_selector_workload": selected_family_census,
                "public_recurrence_selector_workload": selected_public_timing[
                    "recurrence_runtime_work"
                ],
                "recurrence_comparator": recurrence_selected_census,
                "compiled_comparator": compiled_selected_census,
            },
        },
        "all_flow_single_helicity": {
            "component_checks": _summaries(flow_checks),
            "hidden_aggregate": _series(
                flow_hidden_sum, flow_public_sum, "all-flow aggregate"
            ),
            "resolved_aggregate": _series(
                flow_resolved_sum, flow_public_sum, "all-flow resolved aggregate"
            ),
            "family_correctness": flow_family_correctness,
            "family_aggregate": _series(
                flow_family_sum, flow_public_sum, "all-flow family aggregate"
            ),
            "hidden_warm": flow_hidden_timing,
            "family_warm": flow_family_timing,
            "public_warm": flow_public_timing,
            "descriptive_work_comparison": {
                "on_the_fly_selector_workload": flow_family_census,
                "public_recurrence_selector_workload": flow_public_timing[
                    "recurrence_runtime_work"
                ],
                "recurrence_comparator": recurrence_all_flow_census,
                "compiled_comparator": compiled_all_flow_census,
            },
        },
        "amplicol_sanity": sanity,
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
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--prepared-model",
        str(_prepared_model_path(arguments.prepared_model)),
        "--target-runtime",
        str(arguments.target_runtime),
        "--batch-size",
        str(arguments.batch_size),
    ]
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
    destination = arguments.output / "worker.json"
    try:
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
        _write(destination, {"status": "failed", "error": str(error)})
        print(f"on-the-fly LC gate failed: {error}", file=sys.stderr)
        return 1
    _write(destination, result)
    return 0


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
    destination = output / "report.json"
    _write(
        destination,
        {
            "kind": "pyamplicol-on-the-fly-lc-gate-run",
            "status": "passed",
            "worker": worker,
            "watchdog": watchdog,
        },
    )
    print(f"On-the-fly LC gate passed: {destination}")
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
