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


def _invoke(
    probe: Callable[..., dict[str, Any]],
    artifact: Path,
    retained: RetainedInputs,
    query: Query,
    points: Points,
    *,
    repetitions: int = 0,
    enable_color_projection: bool = True,
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
        "evaluator_seconds_per_point": result.evaluator_time_per_point,
        "evaluator_total_seconds_per_point": result.evaluator_total_time_per_point,
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
    selected_public_timing = _public_timing(
        BenchmarkRunner(_benchmark_config(target, batch, flows=(FLOW_ID,))).run(
            runtime, points=timing_points
        )
    )
    flow_public_timing = _public_timing(
        BenchmarkRunner(
            _benchmark_config(target, batch, helicities=(HELICITY_ID,))
        ).run(runtime, points=timing_points)
    )
    for hidden_timing, public_timing in (
        (selected_hidden_timing, selected_public_timing),
        (flow_hidden_timing, flow_public_timing),
    ):
        wall = float(public_timing["wall_seconds_per_point"])
        if not math.isfinite(wall) or wall <= 0.0:
            raise GateError("public benchmark returned a non-positive wall time")
        hidden_timing["provisional_wall_ratio"] = (
            float(hidden_timing["additive_seconds_per_point"]) / wall
        )

    cold = tuple(value[1] for value in cache.values())
    return {
        "kind": "pyamplicol-on-the-fly-lc-gate",
        "status": "passed",
        "process": PROCESS,
        "process_id": PROCESS_ID,
        "artifact": str(artifact),
        "artifact_id": runtime.artifact_id,
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
            "hidden_warm": selected_hidden_timing,
            "public_warm": selected_public_timing,
        },
        "all_flow_single_helicity": {
            "component_checks": _summaries(flow_checks),
            "hidden_aggregate": _series(
                flow_hidden_sum, flow_public_sum, "all-flow aggregate"
            ),
            "resolved_aggregate": _series(
                flow_resolved_sum, flow_public_sum, "all-flow resolved aggregate"
            ),
            "hidden_warm": flow_hidden_timing,
            "public_warm": flow_public_timing,
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
