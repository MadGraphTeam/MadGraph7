#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Guarded, process-parametric high-multiplicity LC study for on-the-fly mode.

Run ``selected`` first: it creates the multiplicity's sole fresh OTF artifact.
Run ``all-flow`` separately with ``--candidate-artifact`` pointing at that
artifact.  n=5/6 get fresh recurrence and compiled authorities for only the
requested workload; n=8/9/10 are intentionally OTF-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyamplicol.generation.service as generation_service  # noqa: E402
from pyamplicol import (  # noqa: E402
    BenchmarkRunner,
    Generator,
    ModelSource,
    ProcessRequest,
    Runtime,
)
from pyamplicol.api.errors import ArtifactError  # noqa: E402
from pyamplicol.artifacts import load_manifest  # noqa: E402
from pyamplicol.config import (  # noqa: E402
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
from tools.ci.memory_watchdog import GIB, run_guarded  # noqa: E402
from tools.developer import on_the_fly_lc_gate as n4  # noqa: E402
from tools.performance_report.runner import (  # noqa: E402
    RunnerError,
    SelectorContract,
    derive_selector_contract,
    point_digest,
    validate_selector_contract,
)
from tools.performance_report.selector_policy import (  # noqa: E402
    SelectorPolicyError,
    canonical_lc_flow_word,
    fixed_selector_helicity,
    selector_color_flow_id,
    selector_helicity_id,
)

KIND = "pyamplicol-on-the-fly-high-multiplicity-study"
SCHEMA_VERSION = 1
SUPPORTED_MULTIPLICITIES = (5, 6, 8, 9, 10)
DUAL_AUTHORITY = frozenset({5, 6})
SEEDS = n4.SEEDS
BATCH_SIZE = 128
WARMUP_RUNS = 2
MINIMUM_SAMPLES = 5
WATCHDOG_BYTES = 30 * GIB
STATE_KIND = "rusticol-on-the-fly-runtime-state-census-v1"
STATE_METHOD = "_on_the_fly_runtime_state_census_json"
STATE_COUNTS = (
    "process_preparation_count",
    "retained_family_count",
    "pending_family_count",
    "retained_selection_count",
    "retained_request_count",
    "retained_amplitude_destination_count",
    "retained_executor_handle_count",
    "retained_query_local_trace_count",
    "retained_embedded_lookup_key_count",
    "semantic_executor_binding_count",
)
COMPACT_ZERO_COUNTS = (
    "pending_family_count",
    "retained_query_local_trace_count",
    "retained_embedded_lookup_key_count",
)
ACTIVE_COUNT_FIELDS = (
    "query_count",
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
OPERATION_ROLES = ("source", "contribution", "finalization", "closure")

Point = tuple[tuple[float, ...], ...]
Points = tuple[Point, ...]


class StudyError(RuntimeError):
    """The bounded study contract was not satisfied."""


@dataclass(frozen=True, slots=True)
class Case:
    multiplicity: int
    process: str
    process_id: str
    pdgs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Selector:
    flow_word: tuple[int, ...]
    flow_id: str
    helicities: tuple[int, ...]
    helicity_id: str

    def mapping(self, points: Points) -> dict[str, object]:
        return {
            "selected_color_flow_ids": [self.flow_id],
            "selected_color_words": [list(self.flow_word)],
            "all_flow_helicity_ids": [self.helicity_id],
            "all_flow_source_helicities": {
                str(index): value
                for index, value in enumerate(self.helicities, start=1)
            },
            "point_digest": point_digest(points),
        }


@dataclass(frozen=True, slots=True)
class Evaluation:
    total: tuple[object, ...]
    resolved: object | None
    total_vs_resolved: dict[str, object] | None = None


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--multiplicity", required=True, type=int, choices=SUPPORTED_MULTIPLICITIES
    )
    parser.add_argument(
        "--workload", choices=("selected", "all-flow"), default="selected"
    )
    parser.add_argument("--prepared-model", type=Path)
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        help="sole OTF artifact created by the selected invocation",
    )
    parser.add_argument(
        "--selected-report",
        type=Path,
        help="passed selected-workload report that created the OTF candidate",
    )
    parser.add_argument("--target-runtime", type=_positive_float, default=1.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _case(multiplicity: int) -> Case:
    if multiplicity not in SUPPORTED_MULTIPLICITIES:
        raise StudyError(f"unsupported multiplicity n={multiplicity}")
    gluons = multiplicity - 2
    return Case(
        multiplicity,
        "d d~ > " + " ".join(("t", "t~", *("g" for _ in range(gluons)))),
        f"otf_dd_tt_{gluons}g",
        (1, -1, 6, -6, *(21 for _ in range(gluons))),
    )


def _validate_arguments(args: argparse.Namespace) -> None:
    _case(args.multiplicity)
    if args.workload == "selected":
        if args.prepared_model is None:
            raise StudyError("selected requires --prepared-model")
        if args.candidate_artifact is not None:
            raise StudyError("selected creates the candidate artifact")
        if args.selected_report is not None:
            raise StudyError("selected creates, rather than consumes, its report")
    elif args.candidate_artifact is None:
        raise StudyError("all-flow requires --candidate-artifact")
    elif args.selected_report is None:
        raise StudyError("all-flow requires --selected-report")
    elif args.multiplicity in DUAL_AUTHORITY and args.prepared_model is None:
        raise StudyError("n=5/6 all-flow requires --prepared-model for authorities")
    elif args.multiplicity not in DUAL_AUTHORITY and args.prepared_model is not None:
        raise StudyError("n>=8 all-flow reuses the candidate without --prepared-model")


def _existing(path: Path, *, directory: bool, label: str) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if not (result.is_dir() if directory else result.is_file()):
        raise StudyError(f"{label} does not exist: {result}")
    return result


def _points(case: Case) -> Points:
    return tuple(
        tuple(tuple(map(float, particle.momentum)) for particle in point)
        for point in (
            generic_validation_point(case.process, sqrt_s=1000.0, seed=seed)
            for seed in SEEDS
        )
    )


def _timing_points(points: Points) -> Points:
    return tuple(points[index % len(points)] for index in range(BATCH_SIZE))


def _selector_from_contract(case: Case, contract: SelectorContract) -> Selector:
    expected_helicities = fixed_selector_helicity(case.pdgs)
    expected_helicity_id = selector_helicity_id(expected_helicities)
    expected_sources = tuple(enumerate(expected_helicities, start=1))
    if (
        len(contract.selected_color_words) != 1
        or len(contract.selected_color_flow_ids) != 1
        or contract.runtime_all_flow_helicity_ids != (expected_helicity_id,)
        or contract.all_flow_source_helicities != expected_sources
    ):
        raise StudyError("recurrence selector contract has the wrong process axis")
    word = contract.selected_color_words[0]
    flow_id = contract.selected_color_flow_ids[0]
    if len(word) != len(case.pdgs) or selector_color_flow_id(word) != flow_id:
        raise StudyError("recurrence selector flow does not round-trip")
    return Selector(word, flow_id, expected_helicities, expected_helicity_id)


def _compact_selector_context(
    runtime: object,
    case: Case,
    requested: tuple[str, ...],
) -> dict[str, object]:
    backend = getattr(runtime, "_backend", None)
    operation = getattr(backend, "_on_the_fly_benchmark_context", None)
    if not callable(operation):
        raise StudyError("candidate has no compact on-the-fly selector context")
    raw = operation(requested)
    if not isinstance(raw, Mapping):
        raise StudyError("candidate compact selector context is unavailable")
    selected = raw.get("selected_color_ids")
    if (
        raw.get("process_id") != case.process_id
        or raw.get("process_expression") != case.process
        or raw.get("color_accuracy") != "lc"
        or isinstance(raw.get("helicity_count"), bool)
        or not isinstance(raw.get("helicity_count"), int)
        or int(raw["helicity_count"]) < 1
        or isinstance(raw.get("color_count"), bool)
        or not isinstance(raw.get("color_count"), int)
        or int(raw["color_count"]) < 1
        or not isinstance(selected, Sequence)
        or isinstance(selected, (str, bytes))
        or len(selected) != len(requested)
        or any(not isinstance(value, str) or not value for value in selected)
    ):
        raise StudyError("candidate compact selector context has the wrong identity")
    return {
        "process_id": case.process_id,
        "process_expression": case.process,
        "color_accuracy": "lc",
        "helicity_count": raw["helicity_count"],
        "color_count": raw["color_count"],
        "requested_color_ids": list(requested),
        "selected_color_ids": list(selected),
    }


def _flow_word(flow_id: str) -> tuple[int, ...]:
    if not flow_id.startswith("flow:"):
        raise StudyError("compact selector did not return a semantic flow ID")
    try:
        word = canonical_lc_flow_word(
            tuple(int(token) for token in flow_id.removeprefix("flow:").split(","))
        )
    except (SelectorPolicyError, ValueError) as error:
        raise StudyError(
            "compact selector returned an invalid semantic flow ID"
        ) from error
    if selector_color_flow_id(word) != flow_id:
        raise StudyError("compact selector flow ID does not round-trip")
    return word


def _compact_reference_selector(
    runtime: object,
    case: Case,
) -> tuple[Selector, dict[str, object]]:
    context = _compact_selector_context(runtime, case, ("1",))
    flow_id = str(context["selected_color_ids"][0])  # type: ignore[index]
    word = _flow_word(flow_id)
    if len(word) != len(case.pdgs):
        raise StudyError("compact reference flow has the wrong external arity")
    helicities = fixed_selector_helicity(case.pdgs)
    return (
        Selector(
            word,
            flow_id,
            helicities,
            selector_helicity_id(helicities),
        ),
        context,
    )


def _lower_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StudyError(f"{label} is not a lowercase hexadecimal identity")
    return value


def _artifact_identity(
    path: Path,
    runtime: object,
    case: Case,
    expected_mode: str,
) -> dict[str, object]:
    try:
        manifest = load_manifest(path, verify_payloads=False)
    except (ArtifactError, OSError) as error:
        raise StudyError(f"cannot load artifact manifest: {path}") from error
    records = tuple(manifest.processes)
    if len(records) != 1:
        raise StudyError("study artifacts must contain exactly one process")
    process = records[0]
    if (
        process.get("id") != case.process_id
        or process.get("expression") != case.process
        or tuple(process.get("external_pdgs", ())) != case.pdgs
        or process.get("color_accuracy") != "lc"
    ):
        raise StudyError("artifact manifest has the wrong process identity")

    artifact_id = _lower_hex(manifest.artifact_id, 64, "manifest artifact ID")
    if (
        getattr(runtime, "artifact_id", None) != artifact_id
        or getattr(runtime, "execution_mode", None) != expected_mode
        or getattr(runtime, "representative_process_key", None) != case.process_id
    ):
        raise StudyError("runtime identity differs from its artifact manifest")
    backend = getattr(runtime, "_backend", None)
    native = getattr(backend, "_runtime", None)
    native_artifact_id = getattr(native, "artifact_id", None)
    metadata = getattr(backend, "_native_metadata", None)
    if native_artifact_id != artifact_id or not isinstance(metadata, Mapping):
        raise StudyError("native runtime omitted its authenticated artifact identity")
    if (
        metadata.get("execution_mode") != expected_mode
        or metadata.get("process") != case.process
        or metadata.get("process_key") != case.process_id
        or metadata.get("representative_process") != case.process
        or metadata.get("representative_process_key") != case.process_id
        or metadata.get("color_accuracy") != "lc"
        or tuple(metadata.get("external_pdg_order", ())) != case.pdgs
        or metadata.get("external_count") != len(case.pdgs)
    ):
        raise StudyError("native runtime metadata has the wrong process identity")

    producer = manifest.producer
    source_revision = _lower_hex(
        producer.get("git_revision"), 40, "producer source revision"
    )
    native_digest = _lower_hex(
        producer.get("native_build_inputs_sha256"),
        64,
        "producer native-input digest",
    )
    model_identity = dict(manifest.model)
    return {
        "path": str(manifest.root),
        "artifact_id": artifact_id,
        "native_authenticated_artifact_id": native_artifact_id,
        "execution_mode": expected_mode,
        "process_id": case.process_id,
        "process_expression": case.process,
        "external_pdgs": list(case.pdgs),
        "producer_identity": {
            "source_revision": source_revision,
            "native_build_inputs_sha256": native_digest,
        },
        "model_identity": model_identity,
    }


def _common_producer_identity(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    identities = tuple(record.get("producer_identity") for record in artifacts.values())
    if not identities or any(not isinstance(value, Mapping) for value in identities):
        raise StudyError("artifact producer identity evidence is incomplete")
    first = dict(identities[0])  # type: ignore[arg-type]
    if any(dict(value) != first for value in identities[1:]):  # type: ignore[arg-type]
        raise StudyError("study artifacts were produced by different native sources")
    return first


def _common_model_identity(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    identities = tuple(record.get("model_identity") for record in artifacts.values())
    if not identities or any(not isinstance(value, Mapping) for value in identities):
        raise StudyError("artifact model identity evidence is incomplete")
    first = dict(identities[0])  # type: ignore[arg-type]
    if any(dict(value) != first for value in identities[1:]):  # type: ignore[arg-type]
        raise StudyError("study artifacts were generated from different models")
    return first


def _config(mode: str, layout: str) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout=layout),
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
            execution_mode=mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _generate_otf(case: Case, path: Path, model: object) -> dict[str, object]:
    original = generation_service._invoke_rust_on_the_fly_seed_batch_builder_v1
    seed_calls = seed_count = materialized_calls = 0

    def capture(*args: object, **kwargs: object) -> object:
        nonlocal seed_calls, seed_count
        seed_calls += 1
        projections = n4._on_the_fly_source_projections(args, kwargs)
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != len(projections):
            raise StudyError("OTF seed binding changed the process batch")
        seed_count += len(result)
        return result

    def reject(name: str) -> Callable[..., object]:
        def operation(*_args: object, **_kwargs: object) -> object:
            nonlocal materialized_calls
            materialized_calls += 1
            raise StudyError(f"OTF generation entered materialized lane {name}")

        return operation

    started = time.perf_counter()
    with ExitStack() as patches:
        patches.enter_context(
            mock.patch.object(
                generation_service,
                "_invoke_rust_on_the_fly_seed_batch_builder_v1",
                capture,
            )
        )
        for name, owner, attribute in n4._materialized_process_lane_patch_targets():
            patches.enter_context(mock.patch.object(owner, attribute, reject(name)))
        Generator(_config("on-the-fly", "topology-replay")).generate(
            ProcessRequest.parse(case.process, name=case.process_id), path, model=model
        )
    if (seed_calls, seed_count, materialized_calls) != (1, 1, 0):
        raise StudyError("OTF generation did not create exactly one compact seed")
    return {
        "seconds": time.perf_counter() - started,
        "seed_binding_calls": seed_calls,
        "seed_count": seed_count,
        "materialized_lane_calls": materialized_calls,
    }


def _generate_authority(
    case: Case, path: Path, model: object, mode: str, layout: str
) -> dict[str, object]:
    started = time.perf_counter()
    Generator(_config(mode, layout)).generate(
        ProcessRequest.parse(case.process, name=case.process_id), path, model=model
    )
    return {
        "execution_mode": mode,
        "layout": layout,
        "seconds": time.perf_counter() - started,
    }


def _census(runtime: object, process_id: str) -> dict[str, Any]:
    native = getattr(getattr(runtime, "_backend", None), "_runtime", None)
    operation = getattr(native, STATE_METHOD, None)
    if not callable(operation):
        raise StudyError(f"OTF runtime does not expose {STATE_METHOD}")
    raw = operation()
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError as error:
        raise StudyError("OTF runtime census is invalid JSON") from error
    if not isinstance(value, dict):
        raise StudyError("OTF runtime census is unavailable")
    if value.get("kind") != STATE_KIND or value.get("process_id") != process_id:
        raise StudyError("OTF runtime census has the wrong identity")
    for field in STATE_COUNTS:
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StudyError(f"OTF runtime census has invalid {field}")
    active = value.get("active_family_union_census")
    if active is not None and not isinstance(active, Mapping):
        raise StudyError("OTF active-family census is invalid")
    return value


def _assert_cold(value: Mapping[str, object], label: str) -> None:
    if (
        any(value[field] != 0 for field in STATE_COUNTS)
        or value.get("active_family_union_census") is not None
    ):
        raise StudyError(f"{label} retained mutable OTF state")


def _active_family_census(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise StudyError(f"{label} has no active family-union census")
    if (
        value.get("basis") != "shared-query-family-union-v1"
        or value.get("scope") != "active-family-union"
    ):
        raise StudyError(f"{label} has the wrong active family-union identity")
    result = dict(value)
    for field in ACTIVE_COUNT_FIELDS:
        count = result.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StudyError(f"{label} has invalid active-family {field}")
    if (
        result["query_count"] < 1
        or result["union_unique_current_count"]
        > result["union_unique_current_component_count"]
        or result["union_amplitude_destination_count"] < 1
        or result["union_amplitude_destination_count"] > result["query_count"]
    ):
        raise StudyError(f"{label} has inconsistent active-family dimensions")
    for role in OPERATION_ROLES:
        if result[f"union_{role}_executor_call_groups"] > result[f"union_{role}_rows"]:
            raise StudyError(f"{label} has more {role} call groups than rows")
    return result


def _assert_family_state(
    value: Mapping[str, object],
    label: str,
    *,
    families: int,
    selections: int,
    handles: int,
    minimum_bindings: int,
) -> dict[str, object]:
    if (
        value["process_preparation_count"] != 1
        or value["retained_family_count"] != families
        or value["retained_selection_count"] != selections
        or value["retained_executor_handle_count"] != handles
        or value["semantic_executor_binding_count"] < minimum_bindings
        or any(value[field] != 0 for field in COMPACT_ZERO_COUNTS)
        or value["retained_request_count"] < 1
        or value["retained_amplitude_destination_count"] < 1
        or value["retained_amplitude_destination_count"]
        > value["retained_request_count"]
    ):
        raise StudyError(f"{label} has an invalid compact retained-state census")
    active = _active_family_census(value.get("active_family_union_census"), label)
    if (
        active["query_count"] > value["retained_request_count"]
        or active["union_amplitude_destination_count"]
        > value["retained_amplitude_destination_count"]
    ):
        raise StudyError(f"{label} active family exceeds retained requests")
    return active


def _same_counts(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return all(left[field] == right[field] for field in STATE_COUNTS)


def _selector_kwargs(workload: str, selector: Selector) -> dict[str, tuple[str, ...]]:
    if workload == "selected":
        return {"color_flows": (selector.flow_id,)}
    if workload == "all-flow":
        return {"helicities": (selector.helicity_id,)}
    if workload == "exact":
        return {
            "color_flows": (selector.flow_id,),
            "helicities": (selector.helicity_id,),
        }
    raise StudyError(f"unknown study workload {workload!r}")


def _evaluate(
    runtime: object,
    points: Points,
    workload: str,
    selector: Selector,
    *,
    resolved: bool,
) -> Evaluation:
    kwargs = _selector_kwargs(workload, selector)
    total = tuple(runtime.evaluate(points, **kwargs))
    detail = runtime.evaluate_resolved(points, **kwargs) if resolved else None
    total_vs_resolved = None
    if len(total) != len(points):
        raise StudyError("runtime returned the wrong point count")
    if detail is not None:
        axes = (
            (("color_ids", (selector.flow_id,)),)
            if workload == "selected"
            else (("helicity_ids", (selector.helicity_id,)),)
            if workload == "all-flow"
            else (
                ("color_ids", (selector.flow_id,)),
                ("helicity_ids", (selector.helicity_id,)),
            )
        )
        if any(tuple(getattr(detail, axis, ())) != expected for axis, expected in axes):
            raise StudyError("resolved output changed the public selector ID")
        resolved_total = getattr(detail, "total", None)
        if not callable(resolved_total):
            raise StudyError("resolved output omitted its public total")
        try:
            total_vs_resolved = n4._series(
                total,
                tuple(resolved_total()),
                f"{workload} public total/resolved total",
            )
        except n4.GateError as error:
            raise StudyError(str(error)) from error
    return Evaluation(total, detail, total_vs_resolved)


def _compare(left: Evaluation, right: Evaluation, label: str) -> dict[str, object]:
    try:
        return {
            "total": n4._series(left.total, right.total, f"{label} total"),
            "resolved": (
                None
                if left.resolved is None or right.resolved is None
                else n4._resolved_component_checks(
                    left.resolved, right.resolved, f"{label} resolved"
                )
            ),
        }
    except n4.GateError as error:
        raise StudyError(str(error)) from error


def _lifecycle(
    runtime: object,
    case: Case,
    selector: Selector,
    points: Points,
    workload: str,
    resolved: bool,
) -> tuple[Evaluation, Evaluation, dict[str, object]]:
    cold = _census(runtime, case.process_id)
    _assert_cold(cold, "cold runtime")
    selected_a = _evaluate(
        runtime,
        points,
        "selected",
        selector,
        resolved=resolved and workload == "selected",
    )
    census_a = _census(runtime, case.process_id)
    active_a = _assert_family_state(
        census_a,
        "selected A",
        families=1,
        selections=1,
        handles=1,
        minimum_bindings=1,
    )
    if workload == "selected":
        exact_c = _evaluate(runtime, points, "exact", selector, resolved=False)
        census_c = _census(runtime, case.process_id)
        _assert_family_state(
            census_c,
            "exact-selector C",
            families=2,
            selections=2,
            handles=2,
            minimum_bindings=int(census_a["semantic_executor_binding_count"]),
        )
        repeated = _evaluate(runtime, points, "exact", selector, resolved=False)
        repeated_census = _census(runtime, case.process_id)
        if repeated_census != census_c:
            raise StudyError("repeated exact-selector C did not plateau")
        _compare(repeated, exact_c, "exact-selector C repeat")
        selected_revisit = _evaluate(
            runtime, points, "selected", selector, resolved=False
        )
        revisit = _census(runtime, case.process_id)
        _assert_family_state(
            revisit,
            "selected A revisit",
            families=2,
            selections=2,
            handles=2,
            minimum_bindings=int(census_c["semantic_executor_binding_count"]),
        )
        if (
            not _same_counts(revisit, census_c)
            or revisit["active_family_union_census"] != active_a
        ):
            raise StudyError("A -> C -> C -> A did not retain both families")
        _compare(selected_revisit, selected_a, "selected A revisit")

        runtime.clear()
        cleared = _census(runtime, case.process_id)
        _assert_cold(cleared, "Runtime.clear()")
        rebuilt = _evaluate(runtime, points, "selected", selector, resolved=resolved)
        rebuilt_a_census = _census(runtime, case.process_id)
        if rebuilt_a_census != census_a:
            raise StudyError("post-clear selected A did not rebuild exactly")
        _evaluate(runtime, points, "exact", selector, resolved=False)
        rebuilt_c_census = _census(runtime, case.process_id)
        if rebuilt_c_census != census_c:
            raise StudyError("post-clear exact-selector C did not rebuild exactly")
        rebuilt_revisit = _evaluate(
            runtime, points, "selected", selector, resolved=False
        )
        rebuilt_census = _census(runtime, case.process_id)
        if rebuilt_census != revisit:
            raise StudyError("post-clear A/C/A retention did not rebuild exactly")
        _compare(rebuilt_revisit, rebuilt, "post-clear selected A revisit")
        _compare(rebuilt, selected_a, "post-clear selected A headline")
        return (
            selected_a,
            rebuilt,
            {
                "cold": cold,
                "selected_a": census_a,
                "requested_family": census_c,
                "requested_repeat": repeated_census,
                "selected_a_revisit": revisit,
                "after_clear": cleared,
                "after_rebuild_selected_a": rebuilt_a_census,
                "after_rebuild_exact_c": rebuilt_c_census,
                "after_rebuild": rebuilt_census,
                "sequence": "A,C,C,A; clear; A,C,A",
            },
        )

    first = _evaluate(runtime, points, "all-flow", selector, resolved=resolved)
    target_census = _census(runtime, case.process_id)
    _assert_family_state(
        target_census,
        "all-flow B",
        families=2,
        selections=2,
        handles=2,
        minimum_bindings=int(census_a["semantic_executor_binding_count"]),
    )
    repeated = _evaluate(runtime, points, "all-flow", selector, resolved=resolved)
    repeated_census = _census(runtime, case.process_id)
    if repeated_census != target_census:
        raise StudyError("repeated all-flow B did not plateau")
    _compare(repeated, first, "all-flow B repeat")
    _evaluate(runtime, points, "selected", selector, resolved=False)
    revisit = _census(runtime, case.process_id)
    _assert_family_state(
        revisit,
        "selected A revisit",
        families=2,
        selections=2,
        handles=2,
        minimum_bindings=int(target_census["semantic_executor_binding_count"]),
    )
    if (
        not _same_counts(revisit, target_census)
        or revisit["active_family_union_census"] != active_a
    ):
        raise StudyError("A -> B -> B -> A did not retain both families")

    runtime.clear()
    cleared = _census(runtime, case.process_id)
    _assert_cold(cleared, "Runtime.clear()")
    _evaluate(runtime, points, "selected", selector, resolved=False)
    rebuilt = _evaluate(runtime, points, "all-flow", selector, resolved=resolved)
    rebuilt_census = _census(runtime, case.process_id)
    if rebuilt_census != target_census:
        raise StudyError("post-clear all-flow B did not rebuild exactly")
    _compare(rebuilt, first, "post-clear all-flow B")
    return (
        first,
        rebuilt,
        {
            "cold": cold,
            "selected_a": census_a,
            "requested_family": target_census,
            "requested_repeat": repeated_census,
            "selected_a_revisit": revisit,
            "after_clear": cleared,
            "after_rebuild": rebuilt_census,
            "sequence": "A,B,B,A; clear; A,B",
        },
    )


def _authority_contract(
    case: Case,
    recurrence: object,
    compiled: object,
    points: Points,
) -> tuple[Selector, dict[str, object]]:
    try:
        contract = derive_selector_contract(recurrence, points)
        validate_selector_contract(recurrence, contract, points)
        validate_selector_contract(compiled, contract, points)
    except RunnerError as error:
        raise StudyError(str(error)) from error
    if (
        recurrence.execution_mode != "recurrence"
        or compiled.execution_mode != "compiled"
    ):
        raise StudyError("authority execution modes differ")
    return _selector_from_contract(case, contract), contract.as_dict()


def _cross_check_compact_selector(
    runtime: object,
    case: Case,
    selector: Selector,
) -> dict[str, object]:
    context = _compact_selector_context(runtime, case, (selector.flow_id,))
    selected = context["selected_color_ids"]
    if (
        selected != [selector.flow_id]
        or _flow_word(selector.flow_id) != selector.flow_word
    ):
        raise StudyError("candidate compact selector differs from recurrence policy")
    return context


def _benchmark(
    runtime: object,
    mode: str,
    points: Points,
    workload: str,
    selector: Selector,
    target: float,
) -> dict[str, object]:
    kwargs = _selector_kwargs(workload, selector)
    config = n4._benchmark_config(
        target,
        BATCH_SIZE,
        helicities=kwargs.get("helicities", ()),
        flows=kwargs.get("color_flows", ()),
    )
    result = BenchmarkRunner(config).run(runtime, points=_timing_points(points))
    effective = result.effective_config
    if result.interrupted:
        raise StudyError("benchmark was interrupted")
    if (
        effective.batch_size != BATCH_SIZE
        or effective.warmup_runs != WARMUP_RUNS
        or effective.minimum_samples < MINIMUM_SAMPLES
        or result.sample_count < MINIMUM_SAMPLES
    ):
        raise StudyError("benchmark did not satisfy the effective timing contract")
    try:
        timing = n4._public_timing(result, mode)
    except n4.GateError as error:
        raise StudyError(str(error)) from error
    timing["simd"] = {
        "status": "not-proven",
        "reason": "native executed-block telemetry is absent",
    }
    return timing


def _selected_report_lineage(
    path: Path,
    case: Case,
    candidate: Mapping[str, object],
) -> dict[str, object]:
    outer = _read(path)
    selected = outer.get("study")
    if (
        outer.get("kind") != f"{KIND}-run"
        or outer.get("schema_version") != SCHEMA_VERSION
        or outer.get("status") != "passed"
        or not isinstance(selected, Mapping)
        or selected.get("kind") != KIND
        or selected.get("schema_version") != SCHEMA_VERSION
        or selected.get("status") != "passed"
        or selected.get("workload") != "selected"
        or selected.get("process") != dataclass_payload(case)
    ):
        raise StudyError("selected report has the wrong study identity")
    generation = selected.get("generation")
    on_the_fly = (
        generation.get("on_the_fly") if isinstance(generation, Mapping) else None
    )
    if (
        not isinstance(on_the_fly, Mapping)
        or on_the_fly.get("seed_binding_calls") != 1
        or on_the_fly.get("seed_count") != 1
        or on_the_fly.get("materialized_lane_calls") != 0
    ):
        raise StudyError("selected report does not prove one compact OTF seed")
    artifacts = selected.get("artifacts")
    recorded = artifacts.get("candidate") if isinstance(artifacts, Mapping) else None
    recorded_path = recorded.get("path") if isinstance(recorded, Mapping) else None
    current_path = candidate.get("path")
    if not isinstance(recorded_path, str) or not isinstance(current_path, str):
        raise StudyError("selected report omitted the candidate artifact path")
    try:
        canonical = str(Path(recorded_path).expanduser().resolve(strict=True))
    except OSError as error:
        raise StudyError("selected report candidate artifact is unavailable") from error
    if (
        recorded_path != canonical
        or current_path != canonical
        or recorded.get("artifact_id") != candidate.get("artifact_id")
    ):
        raise StudyError("selected report and --candidate-artifact disagree")
    return {
        "path": str(path.expanduser().resolve(strict=True)),
        "kind": outer["kind"],
        "status": "passed",
        "candidate_path": canonical,
        "candidate_artifact_id": candidate["artifact_id"],
        "seed_binding_calls": 1,
        "seed_count": 1,
    }


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    _validate_arguments(args)
    case = _case(args.multiplicity)
    points = _points(case)
    output = Path(os.path.abspath(args.output.expanduser()))
    generation: dict[str, object] = {}
    model = None
    if args.workload == "selected":
        prepared = _existing(
            args.prepared_model, directory=False, label="prepared model"
        )
        model = ModelSource.from_path(prepared).compile()
        candidate_path = output / "candidate-artifact"
        generation["on_the_fly"] = _generate_otf(case, candidate_path, model)
    else:
        candidate_path = _existing(
            args.candidate_artifact, directory=True, label="candidate artifact"
        )
    candidate = Runtime.load(candidate_path, process=case.process_id)
    artifacts: dict[str, dict[str, object]] = {
        "candidate": _artifact_identity(candidate_path, candidate, case, "on-the-fly")
    }
    selected_report_lineage = None
    if args.workload == "all-flow":
        selected_report_path = _existing(
            args.selected_report, directory=False, label="selected report"
        )
        selected_report_lineage = _selected_report_lineage(
            selected_report_path, case, artifacts["candidate"]
        )

    recurrence = compiled = None
    selector: Selector
    contract: dict[str, object]
    compact_selector_context: dict[str, object]
    if case.multiplicity in DUAL_AUTHORITY:
        if model is None:
            prepared = _existing(
                args.prepared_model, directory=False, label="prepared model"
            )
            model = ModelSource.from_path(prepared).compile()
        layout = "topology-replay" if args.workload == "selected" else "all-flow-union"
        for mode in ("recurrence", "compiled"):
            generation[mode] = _generate_authority(
                case, output / f"{mode}-authority", model, mode, layout
            )
        recurrence = Runtime.load(
            output / "recurrence-authority", process=case.process_id
        )
        compiled = Runtime.load(output / "compiled-authority", process=case.process_id)
        artifacts["recurrence_authority"] = _artifact_identity(
            output / "recurrence-authority", recurrence, case, "recurrence"
        )
        artifacts["compiled_authority"] = _artifact_identity(
            output / "compiled-authority", compiled, case, "compiled"
        )
        selector, contract = _authority_contract(case, recurrence, compiled, points)
        compact_selector_context = _cross_check_compact_selector(
            candidate, case, selector
        )
    else:
        selector, compact_selector_context = _compact_reference_selector(
            candidate, case
        )
        contract = selector.mapping(points)
    producer_identity = _common_producer_identity(artifacts)
    model_identity = _common_model_identity(artifacts)

    before, after, lifecycle = _lifecycle(
        candidate,
        case,
        selector,
        points,
        args.workload,
        case.multiplicity in DUAL_AUTHORITY,
    )
    if recurrence is not None and compiled is not None:
        rec = _evaluate(recurrence, points, args.workload, selector, resolved=True)
        comp = _evaluate(compiled, points, args.workload, selector, resolved=True)
        correctness = {
            "status": "passed",
            "within_runtime_total_vs_resolved": {
                "on_the_fly_before": before.total_vs_resolved,
                "on_the_fly_after": after.total_vs_resolved,
                "recurrence": rec.total_vs_resolved,
                "compiled": comp.total_vs_resolved,
            },
            "recurrence_vs_compiled": _compare(rec, comp, "recurrence/compiled"),
            "otf_before_vs_recurrence": _compare(before, rec, "OTF before/recurrence"),
            "otf_after_vs_recurrence": _compare(after, rec, "OTF after/recurrence"),
            "otf_before_vs_compiled": _compare(before, comp, "OTF before/compiled"),
            "otf_after_vs_compiled": _compare(after, comp, "OTF after/compiled"),
        }
    else:
        correctness = {
            "status": "not-claimed",
            "reason": "n>=8 is OTF-only",
            "pre_post_clear_self_consistency": _compare(before, after, "OTF pre/post"),
        }

    timings = {
        "on_the_fly": _benchmark(
            candidate,
            "on-the-fly",
            points,
            args.workload,
            selector,
            args.target_runtime,
        )
    }
    if _census(candidate, case.process_id) != lifecycle["after_rebuild"]:
        raise StudyError("timed requested family did not plateau")
    if recurrence is not None and compiled is not None:
        timings["recurrence"] = _benchmark(
            recurrence,
            "recurrence",
            points,
            args.workload,
            selector,
            args.target_runtime,
        )
        timings["compiled"] = _benchmark(
            compiled, "compiled", points, args.workload, selector, args.target_runtime
        )
    candidate_wall = float(timings["on_the_fly"]["wall_seconds_per_point"])
    ratios = {
        f"on_the_fly_over_{mode}": candidate_wall
        / float(value["wall_seconds_per_point"])
        for mode, value in timings.items()
        if mode != "on_the_fly"
    }
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "scope": "dual-authority" if recurrence is not None else "otf-only-feasibility",
        "process": dataclass_payload(case),
        "workload": args.workload,
        "artifacts": artifacts,
        "producer_identity": producer_identity,
        "model_identity": model_identity,
        "selected_report_lineage": selected_report_lineage,
        "candidate_reuse": (
            {
                "status": "contract",
                "selected_created_candidate": True,
                "all_flow_must_reuse_this_artifact": True,
            }
            if args.workload == "selected"
            else {
                "status": "actual-reuse",
                "selected_created_candidate": False,
                "all_flow_reused_candidate": True,
            }
        ),
        "selector_contract": contract,
        "compact_selector_context": compact_selector_context,
        "points": {
            "seeds": list(SEEDS),
            "sqrt_s": 1000.0,
            "count": len(points),
            "digest": point_digest(points),
        },
        "generation": generation,
        "correctness": correctness,
        "cache_lifecycle": lifecycle,
        "timing_contract": {
            "batch_size": BATCH_SIZE,
            "warmup_runs": WARMUP_RUNS,
            "minimum_samples": MINIMUM_SAMPLES,
            "performance_is_acceptance_gate": False,
        },
        "timings": timings,
        "descriptive_ratios": ratios,
        "n8_plus_forbidden_paths": (
            None
            if case.multiplicity in DUAL_AUTHORITY
            else {
                "comparators": "not-created",
                "physics_enumeration": "not-called",
                "resolved_output": "not-called",
                "recurrence_probe": "not-called",
                "prepared_model_on_all_flow": "forbidden",
                "materialized_process_lanes": "generation-poisoned",
            }
        ),
    }


def dataclass_payload(case: Case) -> dict[str, object]:
    return {
        "multiplicity": case.multiplicity,
        "expression": case.process,
        "process_id": case.process_id,
        "external_pdgs": list(case.pdgs),
    }


def _write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise StudyError(f"refusing to replace evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise StudyError(f"cannot write evidence: {path}") from error


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read evidence: {path}") from error
    if not isinstance(value, dict):
        raise StudyError(f"evidence is not an object: {path}")
    return value


def _worker_command(args: argparse.Namespace, output: Path) -> list[str]:
    _validate_arguments(args)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--multiplicity",
        str(args.multiplicity),
        "--workload",
        args.workload,
        "--target-runtime",
        str(args.target_runtime),
    ]
    for option, value, directory, label in (
        ("--prepared-model", args.prepared_model, False, "prepared model"),
        ("--candidate-artifact", args.candidate_artifact, True, "candidate artifact"),
        ("--selected-report", args.selected_report, False, "selected report"),
    ):
        if value is not None:
            command.extend(
                (option, str(_existing(value, directory=directory, label=label)))
            )
    return command


def _worker_main(args: argparse.Namespace) -> int:
    try:
        result = _run_worker(args)
    except Exception as error:
        _write(
            args.output / "worker.json",
            {
                "kind": KIND,
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error": str(error),
            },
        )
        return 1
    _write(args.output / "worker.json", result)
    return 0


def _driver_main(args: argparse.Namespace) -> int:
    _validate_arguments(args)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists() or output.is_symlink():
        raise StudyError(f"output already exists: {output}")
    output.mkdir(parents=True)
    watchdog_path = output / "watchdog.json"
    exit_code = run_guarded(
        _worker_command(args, output),
        limit_bytes=WATCHDOG_BYTES,
        report_path=watchdog_path,
        physical_footprint_probe=n4._physical_footprint_probe(),
    )
    try:
        watchdog = n4._watchdog_summary(_read(watchdog_path))
    except n4.GateError as error:
        raise StudyError(str(error)) from error
    worker_path = output / "worker.json"
    worker = _read(worker_path) if worker_path.is_file() else None
    if (
        exit_code
        or watchdog["passes"] is not True
        or not worker
        or worker.get("status") != "passed"
    ):
        detail = None if worker is None else worker.get("error")
        raise StudyError(f"guarded worker failed: {watchdog['outcome']!r}, {detail!r}")
    report = {
        "kind": f"{KIND}-run",
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "study": worker,
        "watchdog": watchdog,
    }
    _write(output / "report.json", report)
    print(f"On-the-fly high-multiplicity study passed: {output / 'report.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _worker_main(args) if args.worker else _driver_main(args)
    except StudyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
