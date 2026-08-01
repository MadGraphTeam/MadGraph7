#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Render the authenticated final qq -> Z+6g Arena comparison as Markdown.

The tool accepts only a content-addressed M0 request and its accepted combined
decision.  It re-runs the strict M0 validators over all four pyAmpliCol
captures and both original-AmpliCol workloads, checks that the stored decision
is exactly the recomputed decision apart from its creation timestamp/digest,
and then renders deterministic Markdown from the validated evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

SCRIPT = Path(__file__).resolve()
DEFAULT_OUTPUT = Path(
    ".artifacts/developer/qq-z6g-arena-comparison.md"
)
PRE_ARENA_REQUEST_KIND = "pyamplicol-qq-z6g-pre-arena-evidence-request"
PRE_ARENA_REQUEST_SCHEMA = 1
_PRE_ARENA_REQUEST_KEYS = {
    "kind",
    "schema_version",
    "matrix_aggregate",
    "primary_results",
}
RATIO_PAIRS = (
    ("compiled", "eager"),
    ("compiled", "recurrence"),
    ("eager", "recurrence"),
)


class ComparisonError(RuntimeError):
    """The accepted evidence cannot support the requested comparison."""


@dataclass(frozen=True)
class Evidence:
    """Strictly revalidated M0 inputs used to render one document."""

    m0: ModuleType
    expected: dict[str, Any]
    captures: dict[tuple[str, str], Any]
    amplicol: dict[str, Any]
    validation: dict[str, Any]
    acceptance: dict[str, Any]
    request_raw_sha256: str
    request_canonical_sha256: str
    acceptance_raw_sha256: str
    acceptance_content_sha256: str
    pre_arena: PreArenaEvidence


@dataclass(frozen=True)
class PreArenaEvidence:
    """Authenticated frozen AArch64 matrix evidence for the primary workload."""

    matrix: ModuleType
    manifest: Any
    aggregate: Any
    results: dict[str, Any]
    timings: dict[tuple[str, str, str, int], dict[str, Any]]
    baseline_build: dict[str, Any]
    current_build: dict[str, Any]
    baseline_runtime_sha256: str
    platform: dict[str, Any]


def _die(message: str) -> NoReturn:
    raise ComparisonError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_m0() -> ModuleType:
    path = SCRIPT.with_name("eager_compiled_arena_m0.py")
    spec = importlib.util.spec_from_file_location("_qq_z6g_comparison_m0", path)
    if spec is None or spec.loader is None:
        _die(f"cannot load M0 validator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ComparisonError(f"cannot load M0 validator: {error}") from error
    return module


def _load_matrix() -> ModuleType:
    path = SCRIPT.with_name("compiled_mode_matrix.py")
    spec = importlib.util.spec_from_file_location("_qq_z6g_comparison_matrix", path)
    if spec is None or spec.loader is None:
        _die(f"cannot load frozen matrix contract from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ComparisonError(f"cannot load frozen matrix contract: {error}") from error
    return module


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _die(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        _die(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _die(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _die(f"{label} must be an integer >= {minimum}")
    return value


def _number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        _die(f"{label} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        _die(f"{label} must be positive")
    if nonnegative and result < 0.0:
        _die(f"{label} must be nonnegative")
    return result


def _sha256(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _die(f"{label} must be a lowercase SHA-256")
    return result


def _revision(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _die(f"{label} must be a full lowercase Git revision")
    return result


def _load_json_ref(
    m0: ModuleType,
    path: Path,
    expected_sha256: str,
    label: str,
) -> Any:
    m0._require_sha256(expected_sha256, f"{label} raw SHA-256")
    resolved = path.resolve(strict=True)
    reference = m0.FileRef(
        path=resolved,
        stated_path=str(path),
        size_bytes=resolved.stat().st_size,
        sha256=expected_sha256,
    )
    return m0._load_json_ref(reference, label)


def _decision_semantics(
    *,
    acceptance: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    m0: ModuleType,
) -> dict[str, Any]:
    expected_keys = set(recomputed) | {"content_sha256"}
    if set(acceptance) != expected_keys:
        _die("M0 acceptance has unknown or missing root keys")
    without_digest = dict(acceptance)
    content_sha256 = without_digest.pop("content_sha256")
    if _sha256(content_sha256, "M0 acceptance content digest") != _canonical_sha256(
        without_digest
    ):
        _die("M0 acceptance content digest is stale")
    m0._utc(acceptance.get("created_at_utc"), "M0 acceptance creation time")
    accepted_semantics = dict(without_digest)
    recomputed_semantics = dict(recomputed)
    accepted_semantics.pop("created_at_utc")
    recomputed_semantics.pop("created_at_utc")
    if accepted_semantics != recomputed_semantics:
        _die("stored M0 acceptance differs from a fresh strict recomputation")
    return accepted_semantics


def _manifest_ref(
    *,
    m0: ModuleType,
    value: object,
    base: Path,
    label: str,
) -> Any:
    try:
        reference = m0._file_ref(value, base=base, label=label)
        return m0._load_json_ref(reference, label)
    except Exception as error:
        raise ComparisonError(f"{label} is invalid: {error}") from error


def _matrix_primary_cells(matrix: ModuleType) -> dict[str, Any]:
    primary = {
        cell.cell_id: cell
        for cell in matrix.CANONICAL_CELLS
        if cell.category == "primary"
    }
    if len(primary) != 24:
        _die("frozen matrix no longer contains exactly 24 primary qq_Z6g cells")
    expected_axes = {
        (model, mode, workload, batch)
        for model in ("built-in", "ufo-sm")
        for mode in ("eager", "compiled")
        for workload in ("lc-topology", "lc-union")
        for batch in (1, 128, 1024)
    }
    observed_axes = {
        (
            cell.model_kind,
            cell.execution_mode,
            cell.workload_key,
            cell.batch_size,
        )
        for cell in primary.values()
    }
    if observed_axes != expected_axes:
        _die("frozen matrix primary qq_Z6g axes have drifted")
    return primary


def _validate_matrix_aggregate(
    *,
    matrix: ModuleType,
    loaded: Any,
    expected_pre_arena_source_revision: str,
    expected_pre_arena_build_sha256: str,
    expected_pre_arena_runtime_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = loaded.payload
    if (
        payload.get("kind") != matrix.RESULT_KIND
        or payload.get("schema_version") != matrix.SCHEMA_VERSION
        or payload.get("matrix_contract") != matrix.MATRIX_CONTRACT
        or payload.get("complete") is not True
        or payload.get("run_complete") is not True
        or payload.get("passes") is not True
    ):
        _die("pre-Arena 168-cell matrix aggregate is not a complete acceptance")

    definition = _mapping(payload.get("matrix_definition"), "matrix definition")
    canonical_definition = [
        asdict(cell) | {"cell_id": cell.cell_id} for cell in matrix.CANONICAL_CELLS
    ]
    if definition.get("sha256") != matrix._canonical_sha256(canonical_definition):
        _die("pre-Arena matrix definition digest differs from the frozen 168 cells")
    coverage = _mapping(payload.get("coverage"), "matrix coverage")
    if (
        coverage.get("expected") != 168
        or coverage.get("observed") != 168
        or coverage.get("missing") != []
        or coverage.get("unexpected") != []
        or coverage.get("passes") is not True
    ):
        _die("pre-Arena matrix does not cover exactly 168 cells")
    for gate_name in (
        "cell_gate",
        "identity_gate",
        "gain_gate",
        "generation_gate",
        "outer_provenance_gate",
    ):
        gate = _mapping(payload.get(gate_name), f"matrix {gate_name}")
        if gate.get("passes") is not True:
            _die(f"pre-Arena matrix {gate_name} did not pass")

    expected_builds = _mapping(payload.get("expected_builds"), "matrix builds")
    if set(expected_builds) != {"baseline", "current"}:
        _die("pre-Arena matrix build inventory is incomplete")
    baseline_build = _mapping(expected_builds["baseline"], "baseline build")
    current_build = _mapping(expected_builds["current"], "matrix current build")
    for lane, build in (
        ("baseline", baseline_build),
        ("current", current_build),
    ):
        if set(build) != {
            "source_revision",
            "native_build_inputs_sha256",
            "distribution_content_sha256",
            "native_module_sha256",
        }:
            _die(f"matrix {lane} build identity has unknown or missing fields")
        _revision(build["source_revision"], f"matrix {lane} source revision")
        for field in (
            "native_build_inputs_sha256",
            "distribution_content_sha256",
            "native_module_sha256",
        ):
            _sha256(build[field], f"matrix {lane} {field}")
    if (
        baseline_build["source_revision"] != expected_pre_arena_source_revision
        or baseline_build["source_revision"] != matrix.FROZEN_BASELINE_SOURCE_REVISION
    ):
        _die("matrix baseline is not the explicitly pinned frozen pre-Arena source")
    if _canonical_sha256(baseline_build) != expected_pre_arena_build_sha256:
        _die("matrix baseline build identity differs from its explicit digest pin")

    identity_gate = _mapping(payload["identity_gate"], "matrix identity gate")
    distinct = _mapping(
        identity_gate.get("distinct_sha256"),
        "matrix distinct identities",
    )
    baseline_runtimes = _list(
        distinct.get("runtime:baseline"),
        "matrix baseline runtime identities",
    )
    if baseline_runtimes != [expected_pre_arena_runtime_sha256]:
        _die("matrix baseline runtime identity differs from its explicit digest pin")
    _sha256(baseline_runtimes[0], "matrix baseline runtime identity")

    provenance = _mapping(payload.get("provenance"), "matrix provenance")
    preflight = _mapping(provenance.get("preflight"), "matrix preflight")
    postflight = _mapping(provenance.get("postflight"), "matrix postflight")
    if preflight != postflight:
        _die("matrix host/runtime state changed between preflight and postflight")
    if preflight.get("system") != "Darwin" or preflight.get("machine") != "arm64":
        _die("pre-Arena matrix is not the required Darwin AArch64 campaign")
    repository = _mapping(preflight.get("repository"), "matrix repository")
    if (
        repository.get("clean") is not True
        or repository.get("head_revision") != current_build["source_revision"]
    ):
        _die("matrix current source was not its exact clean measured revision")
    platform_record = {
        key: preflight.get(key) for key in ("platform", "system", "machine")
    }

    cells = _list(payload.get("cells"), "matrix cell audit")
    if len(cells) != 168:
        _die("matrix aggregate does not retain all 168 cell audit records")
    audits: dict[str, Any] = {}
    for raw in cells:
        audit = _mapping(raw, "matrix cell audit record")
        cell_id = _text(audit.get("cell_id"), "matrix cell id")
        if cell_id in audits:
            _die(f"matrix aggregate repeats cell {cell_id}")
        if audit.get("passes") is not True or audit.get("errors") != []:
            _die(f"matrix aggregate cell {cell_id} did not pass")
        audits[cell_id] = audit
    expected_cell_ids = {cell.cell_id for cell in matrix.CANONICAL_CELLS}
    if set(audits) != expected_cell_ids:
        _die("matrix aggregate cell audit inventory differs from the frozen matrix")
    return audits, baseline_build, current_build, platform_record


def _matrix_result_distributions(
    *,
    matrix: ModuleType,
    cell: Any,
    result: Mapping[str, Any],
    audit: Mapping[str, Any],
    baseline_runtime_sha256: str,
) -> dict[str, Any]:
    regression = matrix.regression
    if (
        result.get("kind") != regression.RESULT_KIND
        or result.get("schema_version") != regression.SCHEMA_VERSION
        or result.get("complete") is not True
        or result.get("performance_result_authoritative") is not True
        or result.get("passes") is not True
    ):
        _die(f"pre-Arena primary result {cell.cell_id} is not authoritative")
    if matrix._canonical_sha256(result) != audit.get("result_content_sha256"):
        _die(f"pre-Arena primary result {cell.cell_id} differs from its aggregate")
    if matrix._canonical_sha256(audit.get("configuration")) != matrix._canonical_sha256(
        asdict(cell)
    ):
        _die(f"pre-Arena aggregate cell {cell.cell_id} configuration drifted")
    for gate_name in (
        "gate",
        "correctness_gate",
        "arena_profile_gate",
        "resource_gate",
    ):
        gate = _mapping(result.get(gate_name), f"{cell.cell_id} {gate_name}")
        if gate.get("passes") is not True:
            _die(f"pre-Arena primary result {cell.cell_id} {gate_name} failed")

    configuration = _mapping(
        result.get("configuration"),
        f"{cell.cell_id} configuration",
    )
    expected_fields = {
        "process": cell.process,
        "model_label": cell.model_kind,
        "execution_mode": cell.execution_mode,
        "workload": cell.workload,
        "jit_optimization_level": cell.jit_optimization_level,
        "color_accuracy": cell.color_accuracy,
        "lc_flow_layout": cell.lc_flow_layout,
        "batch_size": cell.batch_size,
        "helicities": list(cell.helicities),
        "color_flows": list(cell.color_flows),
        "native_wall_time_source": regression.NATIVE_WALL_TIME_SOURCE,
        "native_wall_time_sample_pass": regression.NATIVE_WALL_TIME_SAMPLE_PASS,
        "timing_sample_contract": regression.PAIRED_TIMING_SAMPLE_CONTRACT,
    }
    for field, expected in expected_fields.items():
        if configuration.get(field) != expected:
            _die(f"pre-Arena primary {cell.cell_id} configuration.{field} drifted")
    sample_count = _integer(
        configuration.get("independent_samples_per_lane"),
        f"{cell.cell_id} sample count",
        minimum=matrix.ACCEPTANCE_SAMPLE_COUNT,
    )
    measurements = _list(
        result.get("measurements"),
        f"{cell.cell_id} measurements",
    )
    if len(measurements) != sample_count * 2:
        _die(f"pre-Arena primary result {cell.cell_id} sample inventory is incomplete")

    values: dict[str, list[float]] = {"baseline": [], "current": []}
    runtime_digests: dict[str, set[str]] = {
        "baseline": set(),
        "current": set(),
    }
    observed_orders: dict[int, list[str]] = {}
    for raw in measurements:
        measurement = _mapping(raw, f"{cell.cell_id} measurement")
        lane = measurement.get("lane")
        if lane not in values:
            _die(f"pre-Arena primary result {cell.cell_id} has an unknown lane")
        pair_index = _integer(
            measurement.get("pair_index"),
            f"{cell.cell_id} pair index",
            minimum=1,
        )
        order = _integer(
            measurement.get("measurement_order"),
            f"{cell.cell_id} measurement order",
            minimum=1,
        )
        if order not in (1, 2):
            _die(f"pre-Arena primary result {cell.cell_id} has invalid pair order")
        pair = observed_orders.setdefault(pair_index, ["", ""])
        if pair[order - 1]:
            _die(f"pre-Arena primary result {cell.cell_id} repeats a pair position")
        pair[order - 1] = str(lane)
        values[str(lane)].append(
            _number(
                measurement.get("wall_seconds_per_point"),
                f"{cell.cell_id}/{lane} wall time",
                positive=True,
            )
        )
        runtime = _mapping(
            measurement.get("runtime_identity"),
            f"{cell.cell_id}/{lane} runtime identity",
        )
        runtime_digests[str(lane)].add(
            matrix._canonical_sha256(matrix._stable_runtime_identity_value(runtime))
        )
    expected_orders = {
        pair_index: (
            ["baseline", "current"]
            if (pair_index - 1) % 2 == 0
            else ["current", "baseline"]
        )
        for pair_index in range(1, sample_count + 1)
    }
    if observed_orders != expected_orders or result.get("pair_orders") != list(
        expected_orders.values()
    ):
        _die(f"pre-Arena primary result {cell.cell_id} is not paired/interleaved")
    if runtime_digests["baseline"] != {baseline_runtime_sha256}:
        _die(f"pre-Arena primary result {cell.cell_id} baseline runtime drifted")
    aggregate_runtime_digests = _mapping(
        audit.get("runtime_identity_sha256_by_lane"),
        f"{cell.cell_id} aggregate runtime identities",
    )
    if {
        lane: next(iter(digests)) for lane, digests in runtime_digests.items()
    } != aggregate_runtime_digests:
        _die(f"pre-Arena primary result {cell.cell_id} runtime audit is stale")

    recomputed = {
        lane: regression._distribution(lane_values)
        for lane, lane_values in values.items()
    }
    if result.get("distributions") != recomputed:
        _die(f"pre-Arena primary result {cell.cell_id} distributions are stale")
    paired = regression._paired_distribution(measurements)
    if result.get("paired_distribution") != paired:
        _die(f"pre-Arena primary result {cell.cell_id} paired statistics are stale")
    correctness = regression._correctness_gate(measurements)
    if result.get("correctness_gate") != correctness:
        _die(f"pre-Arena primary result {cell.cell_id} correctness gate is stale")
    return recomputed


def validate_pre_arena_evidence(
    *,
    m0: ModuleType,
    manifest_path: Path,
    manifest_sha256: str,
    expected_pre_arena_source_revision: str,
    expected_pre_arena_build_sha256: str,
    expected_pre_arena_runtime_sha256: str,
) -> PreArenaEvidence:
    """Authenticate the frozen AArch64 matrix and all 24 primary result files."""

    matrix = _load_matrix()
    manifest_loaded = _load_json_ref(
        m0,
        manifest_path,
        manifest_sha256,
        "pre-Arena evidence request",
    )
    manifest = manifest_loaded.payload
    if set(manifest) != _PRE_ARENA_REQUEST_KEYS:
        _die("pre-Arena evidence request has unknown or missing root keys")
    if (
        manifest.get("kind") != PRE_ARENA_REQUEST_KIND
        or manifest.get("schema_version") != PRE_ARENA_REQUEST_SCHEMA
    ):
        _die("pre-Arena evidence request kind/schema is unsupported")
    base = manifest_loaded.ref.path.parent
    aggregate_loaded = _manifest_ref(
        m0=m0,
        value=manifest.get("matrix_aggregate"),
        base=base,
        label="pre-Arena matrix aggregate",
    )
    audits, baseline_build, current_build, platform_record = _validate_matrix_aggregate(
        matrix=matrix,
        loaded=aggregate_loaded,
        expected_pre_arena_source_revision=expected_pre_arena_source_revision,
        expected_pre_arena_build_sha256=expected_pre_arena_build_sha256,
        expected_pre_arena_runtime_sha256=expected_pre_arena_runtime_sha256,
    )
    primary = _matrix_primary_cells(matrix)
    refs = _list(manifest.get("primary_results"), "pre-Arena primary results")
    if len(refs) != len(primary):
        _die("pre-Arena evidence request must contain exactly 24 primary results")
    results: dict[str, Any] = {}
    timings: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for index, value in enumerate(refs):
        record = _mapping(value, f"pre-Arena primary result ref {index}")
        if set(record) != {"cell_id", "path", "size_bytes", "sha256"}:
            _die(f"pre-Arena primary result ref {index} has unknown/missing fields")
        cell_id = _text(record["cell_id"], f"pre-Arena primary result {index} id")
        cell = primary.get(cell_id)
        if cell is None or cell_id in results:
            _die(f"pre-Arena primary result inventory has invalid cell {cell_id}")
        loaded = _manifest_ref(
            m0=m0,
            value={key: record[key] for key in ("path", "size_bytes", "sha256")},
            base=base,
            label=f"pre-Arena primary result {cell_id}",
        )
        distributions = _matrix_result_distributions(
            matrix=matrix,
            cell=cell,
            result=loaded.payload,
            audit=audits[cell_id],
            baseline_runtime_sha256=expected_pre_arena_runtime_sha256,
        )
        results[cell_id] = loaded
        model = "built-in-sm" if cell.model_kind == "built-in" else "ufo-sm"
        layout = (
            "topology-replay"
            if cell.workload_key == "lc-topology"
            else "all-flow-union"
        )
        key = (model, layout, cell.execution_mode, cell.batch_size)
        if key in timings:
            _die(f"pre-Arena primary timing key is duplicated: {key}")
        timings[key] = distributions
    if set(results) != set(primary) or len(timings) != 24:
        _die("pre-Arena primary result inventory is incomplete")
    return PreArenaEvidence(
        matrix=matrix,
        manifest=manifest_loaded,
        aggregate=aggregate_loaded,
        results=results,
        timings=timings,
        baseline_build=baseline_build,
        current_build=current_build,
        baseline_runtime_sha256=expected_pre_arena_runtime_sha256,
        platform=platform_record,
    )


def _bind_matrix_current_to_final(
    *,
    pre_arena: PreArenaEvidence,
    final_capture: Any,
    expected_source_revision: str,
) -> None:
    current = pre_arena.current_build
    runtime = _mapping(final_capture.runtime_identity, "final Arena runtime identity")
    native = _mapping(runtime.get("native_extension"), "final native extension")
    distribution = _mapping(
        runtime.get("installed_distribution"),
        "final installed distribution",
    )
    distribution_content = _mapping(
        distribution.get("distribution_content"),
        "final distribution content",
    )
    expected = {
        "source_revision": expected_source_revision,
        "native_build_inputs_sha256": _sha256(
            native.get("build_inputs_sha256"),
            "final native build-input digest",
        ),
        "distribution_content_sha256": _sha256(
            distribution_content.get("sha256"),
            "final distribution-content digest",
        ),
        "native_module_sha256": _sha256(
            native.get("sha256"),
            "final native-module digest",
        ),
    }
    if current != expected:
        _die(
            "168-cell matrix current build does not equal the final M0 "
            "source/native runtime"
        )


def validate_evidence(
    *,
    request_path: Path,
    request_sha256: str,
    acceptance_path: Path,
    acceptance_sha256: str,
    expected_source_revision: str,
    expected_runtime_provenance_sha256: str,
    pre_arena_manifest_path: Path,
    pre_arena_manifest_sha256: str,
    expected_pre_arena_source_revision: str,
    expected_pre_arena_build_sha256: str,
    expected_pre_arena_runtime_sha256: str,
) -> Evidence:
    """Re-run the complete M0 decision and bind it to explicit final identities."""

    m0 = _load_m0()
    expected_source_revision = _revision(
        expected_source_revision,
        "--expected-source-revision",
    )
    expected_runtime_provenance_sha256 = _sha256(
        expected_runtime_provenance_sha256,
        "--expected-runtime-provenance-sha256",
    )
    expected_pre_arena_source_revision = _revision(
        expected_pre_arena_source_revision,
        "--expected-pre-arena-source-revision",
    )
    expected_pre_arena_build_sha256 = _sha256(
        expected_pre_arena_build_sha256,
        "--expected-pre-arena-build-sha256",
    )
    expected_pre_arena_runtime_sha256 = _sha256(
        expected_pre_arena_runtime_sha256,
        "--expected-pre-arena-runtime-sha256",
    )
    request_loaded = _load_json_ref(
        m0,
        request_path,
        request_sha256,
        "M0 request",
    )
    request = request_loaded.payload
    m0._require_exact_keys(request, m0._REQUEST_KEYS, "M0 request")
    if (
        request.get("kind") != m0.REQUEST_KIND
        or request.get("schema_version") != m0.REQUEST_SCHEMA
    ):
        _die("M0 request kind/schema is unsupported")
    expected = m0._validate_expected(request.get("expected"))
    if expected["pyamplicol_source_revision"] != expected_source_revision:
        _die("M0 request does not pin the explicitly requested final source revision")
    if expected["runtime_provenance_sha256"] != expected_runtime_provenance_sha256:
        _die("M0 request does not pin the explicitly requested final runtime identity")

    capture_refs, amplicol_refs = m0._request_refs(
        request,
        request_loaded.ref.path.parent,
    )
    benchmark = m0._load_benchmark_module()
    captures = {
        key: m0._validate_capture(
            m0._load_json_ref(ref, f"{key[0]}/{key[1]} capture"),
            model=key[0],
            layout=key[1],
            expected=expected,
            benchmark=benchmark,
        )
        for key, ref in capture_refs.items()
    }
    amplicol = {
        role: m0._validate_amplicol(
            m0._load_json_ref(ref, f"amplicol/{role}"),
            role=role,
            expected=expected,
        )
        for role, ref in amplicol_refs.items()
    }
    validation = m0._cross_validate(captures, amplicol)
    recomputed = m0._accepted_manifest(
        request_loaded=request_loaded,
        captures=captures,
        amplicol=amplicol,
        validation=validation,
        expected=expected,
    )

    acceptance_loaded = _load_json_ref(
        m0,
        acceptance_path,
        acceptance_sha256,
        "M0 acceptance",
    )
    acceptance = acceptance_loaded.payload
    _decision_semantics(
        acceptance=acceptance,
        recomputed=recomputed,
        m0=m0,
    )
    if (
        acceptance.get("accepted") is not True
        or acceptance.get("status") != "accepted"
        or acceptance.get("errors") != []
    ):
        _die("M0 decision is not an error-free acceptance")
    common = _mapping(acceptance.get("common_contract"), "acceptance.common_contract")
    source = _mapping(common.get("source"), "acceptance.common_contract.source")
    if source.get("revision") != expected_source_revision:
        _die("accepted source identity differs from the explicit final revision")
    if common.get("runtime_provenance_sha256") != expected_runtime_provenance_sha256:
        _die("accepted runtime identity differs from the explicit final digest")
    pre_arena = validate_pre_arena_evidence(
        m0=m0,
        manifest_path=pre_arena_manifest_path,
        manifest_sha256=pre_arena_manifest_sha256,
        expected_pre_arena_source_revision=expected_pre_arena_source_revision,
        expected_pre_arena_build_sha256=expected_pre_arena_build_sha256,
        expected_pre_arena_runtime_sha256=expected_pre_arena_runtime_sha256,
    )
    primary = _matrix_primary_cells(pre_arena.matrix)
    topology_cell = next(
        cell for cell in primary.values() if cell.workload_key == "lc-topology"
    )
    union_cell = next(
        cell for cell in primary.values() if cell.workload_key == "lc-union"
    )
    if tuple(topology_cell.color_flows) != (expected["color_flow"]["id"],) or tuple(
        union_cell.helicities
    ) != (expected["helicity"]["id"],):
        _die("pre-Arena and final Arena captures use different runtime selectors")
    first_capture = captures[("built-in-sm", "topology-replay")]
    if (
        pre_arena.platform["system"] != first_capture.host["system"]
        or pre_arena.platform["machine"] != first_capture.host["machine"]
    ):
        _die("pre-Arena and final Arena evidence are not from one architecture")
    _bind_matrix_current_to_final(
        pre_arena=pre_arena,
        final_capture=first_capture,
        expected_source_revision=expected_source_revision,
    )
    return Evidence(
        m0=m0,
        expected=expected,
        captures=captures,
        amplicol=amplicol,
        validation=validation,
        acceptance=acceptance,
        request_raw_sha256=request_sha256,
        request_canonical_sha256=request_loaded.canonical_sha256,
        acceptance_raw_sha256=acceptance_sha256,
        acceptance_content_sha256=_sha256(
            acceptance["content_sha256"],
            "M0 acceptance content digest",
        ),
        pre_arena=pre_arena,
    )


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    if any(len(row) != len(headers) for row in rows):
        _die("internal Markdown table width mismatch")
    result = [
        "| " + " | ".join(_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    result.extend(
        "| " + " | ".join(_escape(value) for value in row) + " |" for row in rows
    )
    return result


def _model_label(model: str) -> str:
    return {"built-in-sm": "Built-in SM", "ufo-sm": "UFO SM"}[model]


def _layout_label(layout: str) -> str:
    return {
        "topology-replay": "Selected LC flow / helicity sum",
        "all-flow-union": "All LC flows / one helicity",
    }[layout]


def _mode_label(mode: str) -> str:
    return {
        "compiled": "Compiled DAG (JIT O3)",
        "eager": "Eager DAG",
        "recurrence": "Recurrence",
    }[mode]


def _seconds(value: float) -> str:
    return f"{value:.9g} s"


def _microseconds(value: float) -> str:
    return f"{value * 1.0e6:.9g} µs/point"


def _ratio(value: float) -> str:
    return f"{value:.9g}x"


def _digest(value: str) -> str:
    return f"`{value}`"


def _bytes(value: int) -> str:
    return f"{value:,} B ({value / (1024 * 1024):.3f} MiB)"


def _complex(value: complex) -> str:
    return f"`{value.real:.17g}{value.imag:+.17g}i`"


def _comparison_stats(
    left: Sequence[complex],
    right: Sequence[complex],
    *,
    absolute_floor: float,
) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        _die("numerical comparison has mismatched or empty point inventories")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for lhs, rhs in zip(left, right, strict=True):
        difference = abs(lhs - rhs)
        relative = difference / max(abs(rhs), absolute_floor)
        maximum_absolute = max(maximum_absolute, difference)
        maximum_relative = max(maximum_relative, relative)
    return maximum_absolute, maximum_relative


def _capture_input_rows(evidence: Evidence) -> list[list[object]]:
    rows: list[list[object]] = [
        [
            "Combined request",
            _digest(evidence.request_raw_sha256),
            _digest(evidence.request_canonical_sha256),
        ],
        [
            "Combined acceptance",
            _digest(evidence.acceptance_raw_sha256),
            _digest(evidence.acceptance_content_sha256),
        ],
    ]
    for model in evidence.m0.MODELS:
        for layout in evidence.m0.LAYOUTS:
            loaded = evidence.captures[(model, layout)].loaded
            rows.append(
                [
                    f"{_model_label(model)} — {_layout_label(layout)}",
                    _digest(loaded.ref.sha256),
                    _digest(loaded.canonical_sha256),
                ]
            )
    for role in (
        evidence.m0.AMPLICOL_SELECTED_ROLE,
        evidence.m0.AMPLICOL_UNION_ROLE,
    ):
        loaded = evidence.amplicol[role].loaded
        rows.append(
            [
                f"Original AmpliCol — {role}",
                _digest(loaded.ref.sha256),
                _digest(loaded.canonical_sha256),
            ]
        )
    for label, loaded in (
        ("Pre-Arena evidence request", evidence.pre_arena.manifest),
        ("Pre-Arena 168-cell aggregate", evidence.pre_arena.aggregate),
    ):
        rows.append(
            [
                label,
                _digest(loaded.ref.sha256),
                _digest(loaded.canonical_sha256),
            ]
        )
    for cell_id, loaded in sorted(evidence.pre_arena.results.items()):
        rows.append(
            [
                f"Pre-Arena primary — {cell_id}",
                _digest(loaded.ref.sha256),
                _digest(loaded.canonical_sha256),
            ]
        )
    return rows


def _runtime_identity_rows(evidence: Evidence) -> list[list[object]]:
    first = evidence.captures[("built-in-sm", "topology-replay")]
    runtime = _mapping(first.runtime_identity, "runtime identity")
    interpreter = _mapping(runtime.get("interpreter"), "runtime interpreter")
    native = _mapping(runtime.get("native_extension"), "native extension")
    distribution = _mapping(
        runtime.get("installed_distribution"),
        "installed distribution",
    )
    distribution_content = _mapping(
        distribution.get("distribution_content"),
        "distribution content",
    )
    build_info = _mapping(runtime.get("active_build_info"), "active build info")
    build_payload = _mapping(build_info.get("payload"), "active build payload")
    source_revision = evidence.expected["pyamplicol_source_revision"]
    if build_payload.get("source_revision") != source_revision:
        _die("active build info source revision is stale")
    distribution_file_count = _integer(
        distribution_content.get("file_count"),
        "distribution file count",
        minimum=1,
    )
    distribution_size = _integer(
        distribution_content.get("size_bytes"),
        "distribution size",
        minimum=1,
    )
    distribution_sha256 = _sha256(
        distribution_content.get("sha256"),
        "distribution digest",
    )
    return [
        ["pyAmpliCol source revision", _digest(source_revision)],
        [
            "Runtime-provenance digest",
            _digest(evidence.expected["runtime_provenance_sha256"]),
        ],
        [
            "Candidate/package version",
            _text(distribution.get("package_version"), "distribution package version"),
        ],
        [
            "Candidate fingerprint",
            _text(build_payload.get("candidate_fingerprint"), "candidate fingerprint"),
        ],
        [
            "Native extension SHA-256",
            _digest(_sha256(native.get("sha256"), "native extension digest")),
        ],
        [
            "Native build-input SHA-256",
            _digest(
                _sha256(
                    native.get("build_inputs_sha256"),
                    "native build-input digest",
                )
            ),
        ],
        [
            "Installed distribution",
            (
                f"{distribution_file_count} files, "
                f"{_bytes(distribution_size)}; "
                f"{_digest(distribution_sha256)}"
            ),
        ],
        [
            "Python interpreter",
            (
                f"{_text(interpreter.get('implementation'), 'interpreter')} "
                f"{_text(interpreter.get('python_version'), 'Python version')}; "
                f"{_digest(_sha256(interpreter.get('sha256'), 'interpreter digest'))}"
            ),
        ],
    ]


def _original_identity_rows(evidence: Evidence) -> list[list[object]]:
    selected = evidence.amplicol[evidence.m0.AMPLICOL_SELECTED_ROLE]
    source = _mapping(selected.source_identity, "original AmpliCol source")
    compiler = _mapping(source.get("compiler"), "original AmpliCol compiler")
    return [
        [
            "Original AmpliCol revision",
            _digest(_revision(source.get("revision"), "AmpliCol revision")),
        ],
        [
            "Tracked source-tree SHA-256",
            _digest(_sha256(source.get("source_tree_sha256"), "AmpliCol source tree")),
        ],
        [
            "Fortran compiler",
            (
                f"{_text(compiler.get('id'), 'compiler id')} — "
                f"{_text(compiler.get('version'), 'compiler version')}"
            ),
        ],
        ["Compiler target", _text(compiler.get("target"), "compiler target")],
        [
            "Compiler-flags SHA-256",
            _digest(_sha256(compiler.get("flags_sha256"), "compiler flags")),
        ],
    ]


def _pre_arena_identity_rows(evidence: Evidence) -> list[list[object]]:
    baseline = evidence.pre_arena.baseline_build
    current = evidence.pre_arena.current_build
    platform_record = evidence.pre_arena.platform
    return [
        [
            "Frozen pre-Arena source revision",
            _digest(
                _revision(
                    baseline["source_revision"],
                    "pre-Arena source revision",
                )
            ),
        ],
        [
            "Frozen pre-Arena build identity SHA-256",
            _digest(_canonical_sha256(baseline)),
        ],
        [
            "Frozen pre-Arena runtime identity SHA-256",
            _digest(evidence.pre_arena.baseline_runtime_sha256),
        ],
        [
            "Frozen pre-Arena native build-input SHA-256",
            _digest(baseline["native_build_inputs_sha256"]),
        ],
        [
            "Frozen pre-Arena distribution SHA-256",
            _digest(baseline["distribution_content_sha256"]),
        ],
        [
            "Frozen pre-Arena native module SHA-256",
            _digest(baseline["native_module_sha256"]),
        ],
        [
            "168-cell matrix Arena source revision",
            _digest(
                _revision(
                    current["source_revision"],
                    "matrix Arena source revision",
                )
            ),
        ],
        [
            "168-cell matrix Arena native module SHA-256",
            _digest(current["native_module_sha256"]),
        ],
        [
            "Matrix platform",
            (
                f"{_text(platform_record['system'], 'matrix system')} "
                f"{_text(platform_record['machine'], 'matrix machine')} — "
                f"{_text(platform_record['platform'], 'matrix platform')}"
            ),
        ],
    ]


def _numerical_sections(evidence: Evidence) -> list[str]:
    lines: list[str] = [
        "## Numerical agreement",
        "",
        (
            "The combined M0 validator re-evaluated every retained point, required "
            "selected totals to close against resolved sums, required all three "
            "pyAmpliCol lanes to agree, required built-in and UFO models to agree, "
            "and required both original-AmpliCol workloads to agree with their "
            "built-in pyAmpliCol counterparts."
        ),
        "",
        "### Pointwise values across model implementations and original AmpliCol",
        "",
    ]
    point_rows: list[list[object]] = []
    for layout in evidence.m0.LAYOUTS:
        builtin = evidence.captures[("built-in-sm", layout)].validation_values
        ufo = evidence.captures[("ufo-sm", layout)].validation_values
        role = (
            evidence.m0.AMPLICOL_UNION_ROLE
            if layout == "all-flow-union"
            else evidence.m0.AMPLICOL_SELECTED_ROLE
        )
        original = evidence.amplicol[role].values
        if len(builtin) != len(ufo) or len(builtin) != len(original):
            _die(f"{layout} pointwise inventories differ after M0 validation")
        for index, (built, external, independent) in enumerate(
            zip(builtin, ufo, original, strict=True)
        ):
            maximum_absolute, maximum_relative = _comparison_stats(
                (built, built),
                (external, independent),
                absolute_floor=evidence.m0.ATOL,
            )
            point_rows.append(
                [
                    _layout_label(layout),
                    index,
                    _complex(built),
                    _complex(external),
                    _complex(independent),
                    f"{maximum_absolute:.9g}",
                    f"{maximum_relative:.9g}",
                    "yes",
                ]
            )
    lines.extend(
        _table(
            (
                "Workload",
                "Point",
                "Built-in SM",
                "UFO SM",
                "Original AmpliCol",
                "Max |Δ|",
                "Max relative Δ",
                "Pass",
            ),
            point_rows,
        )
    )
    lines.extend(["", "### Selected-total versus resolved-sum closure", ""])
    closure_rows: list[list[object]] = []
    lane_rows: list[list[object]] = []
    component_rows: list[list[object]] = []
    for model in evidence.m0.MODELS:
        for layout in evidence.m0.LAYOUTS:
            capture = evidence.captures[(model, layout)]
            payload = capture.loaded.payload
            profiles = _mapping(payload.get("profiles"), "capture profiles")
            for mode in evidence.m0.MODES:
                profile = _mapping(profiles.get(mode), f"{model}/{layout}/{mode}")
                validation = _mapping(
                    profile.get("validation"),
                    f"{model}/{layout}/{mode} validation",
                )
                if validation.get("passes") is not True:
                    _die(f"{model}/{layout}/{mode} validation is not passing")
                closure_absolute = _number(
                    validation.get("maximum_absolute_difference"),
                    "closure max abs",
                    nonnegative=True,
                )
                closure_relative = _number(
                    validation.get("maximum_relative_difference"),
                    "closure max rel",
                    nonnegative=True,
                )
                closure_rows.append(
                    [
                        _model_label(model),
                        _layout_label(layout),
                        _mode_label(mode),
                        f"{closure_absolute:.9g}",
                        f"{closure_relative:.9g}",
                        "yes",
                    ]
                )
            comparisons = _mapping(
                payload.get("lane_comparisons"),
                f"{model}/{layout} lane comparisons",
            )
            summary = _mapping(
                payload.get("validation_summary"),
                f"{model}/{layout} validation summary",
            )
            resolved = _mapping(
                summary.get("resolved_component_comparisons"),
                f"{model}/{layout} component comparisons",
            )
            for left, right in RATIO_PAIRS:
                key = f"{left}__{right}"
                comparison = _mapping(comparisons.get(key), f"{model}/{layout}/{key}")
                if comparison.get("passes") is not True:
                    _die(f"{model}/{layout}/{key} total comparison failed")
                lane_absolute = _number(
                    comparison.get("maximum_absolute_difference"),
                    "lane max abs",
                    nonnegative=True,
                )
                lane_relative = _number(
                    comparison.get("maximum_relative_difference"),
                    "lane max rel",
                    nonnegative=True,
                )
                lane_rows.append(
                    [
                        _model_label(model),
                        _layout_label(layout),
                        f"{left} vs {right}",
                        f"{lane_absolute:.9g}",
                        f"{lane_relative:.9g}",
                        "yes",
                    ]
                )
                component = _mapping(
                    resolved.get(key),
                    f"{model}/{layout}/{key} component comparison",
                )
                if component.get("passes") is not True:
                    _die(f"{model}/{layout}/{key} component comparison failed")
                component_absolute = _number(
                    component.get("maximum_absolute_difference"),
                    "component max abs",
                    nonnegative=True,
                )
                component_relative = _number(
                    component.get("maximum_relative_difference"),
                    "component max rel",
                    nonnegative=True,
                )
                component_rows.append(
                    [
                        _model_label(model),
                        _layout_label(layout),
                        f"{left} vs {right}",
                        _integer(
                            component.get("compared_component_count"),
                            "compared component count",
                            minimum=1,
                        ),
                        f"{component_absolute:.9g}",
                        f"{component_relative:.9g}",
                        "yes",
                    ]
                )
    lines.extend(
        _table(
            (
                "Model",
                "Workload",
                "Mode",
                "Max |selected - resolved|",
                "Max relative Δ",
                "Pass",
            ),
            closure_rows,
        )
    )
    lines.extend(["", "### Cross-lane total agreement", ""])
    lines.extend(
        _table(
            ("Model", "Workload", "Lane pair", "Max |Δ|", "Max relative Δ", "Pass"),
            lane_rows,
        )
    )
    lines.extend(["", "### Cross-lane resolved-component agreement", ""])
    lines.extend(
        _table(
            (
                "Model",
                "Workload",
                "Lane pair",
                "Components",
                "Max |Δ|",
                "Max relative Δ",
                "Pass",
            ),
            component_rows,
        )
    )
    return lines


def _generation_sections(evidence: Evidence) -> list[str]:
    lines = [
        "## Generation, artifact, and memory evidence",
        "",
        (
            "Generation wall time and RSS are single worker observations, not "
            "multi-sample runtime statistics. RSS is the retained high-water-mark "
            "lower bound described by each capture."
        ),
        "",
    ]
    rows: list[list[object]] = []
    phase_rows: list[list[object]] = []
    for model in evidence.m0.MODELS:
        for layout in evidence.m0.LAYOUTS:
            payload = evidence.captures[(model, layout)].loaded.payload
            generation = _mapping(
                payload.get("generation"),
                f"{model}/{layout} generation",
            )
            if set(generation) != set(evidence.m0.MODES):
                _die(f"{model}/{layout} generation inventory is incomplete")
            for mode in evidence.m0.MODES:
                record = _mapping(
                    generation[mode],
                    f"{model}/{layout}/{mode} generation",
                )
                reused = record.get("generation_reused")
                if not isinstance(reused, bool):
                    _die(f"{model}/{layout}/{mode} reuse flag is invalid")
                raw_wall = record.get("generation_wall_seconds")
                wall = (
                    "Unavailable (artifact reused)"
                    if raw_wall is None and reused
                    else _seconds(
                        _number(
                            raw_wall,
                            f"{model}/{layout}/{mode} generation wall",
                            positive=True,
                        )
                    )
                )
                stats = _mapping(
                    record.get("artifact_stats"),
                    f"{model}/{layout}/{mode} artifact stats",
                )
                files = _integer(
                    stats.get("file_count"),
                    "artifact file count",
                    minimum=1,
                )
                size = _integer(
                    stats.get("size_bytes"),
                    "artifact size",
                    minimum=1,
                )
                peak = record.get("peak_rss")
                if peak is None and reused:
                    rss = "Unavailable (artifact reused)"
                else:
                    peak_map = _mapping(peak, "generation peak RSS")
                    rss = _bytes(
                        _integer(
                            peak_map.get("observed_lower_bound_bytes"),
                            "generation RSS lower bound",
                            minimum=1,
                        )
                    )
                phases = _mapping(
                    record.get("phase_timings_seconds"),
                    f"{model}/{layout}/{mode} phase timings",
                )
                phase_values = {
                    str(name): _number(
                        value,
                        f"{model}/{layout}/{mode}/{name}",
                        nonnegative=True,
                    )
                    for name, value in phases.items()
                }
                if not phase_values:
                    _die(f"{model}/{layout}/{mode} has no phase timings")
                model_load = (
                    _seconds(phase_values["model-loading"])
                    if "model-loading" in phase_values
                    else "Unavailable"
                )
                semantic_sha = _sha256(
                    record.get("artifact_semantic_identity_sha256"),
                    "artifact semantic identity",
                )
                rows.append(
                    [
                        _model_label(model),
                        _layout_label(layout),
                        _mode_label(mode),
                        wall,
                        model_load,
                        files,
                        _bytes(size),
                        rss,
                        _digest(semantic_sha),
                    ]
                )
                for phase_name in sorted(phase_values):
                    phase_rows.append(
                        [
                            _model_label(model),
                            _layout_label(layout),
                            _mode_label(mode),
                            phase_name,
                            _seconds(phase_values[phase_name]),
                        ]
                    )
    lines.extend(
        _table(
            (
                "Model",
                "Workload",
                "Mode",
                "Generation wall",
                "Model-load phase",
                "Files",
                "Artifact bytes",
                "Generation peak RSS lower bound",
                "Artifact semantic SHA-256",
            ),
            rows,
        )
    )
    lines.extend(["", "### Retained generation phase breakdown", ""])
    lines.extend(
        _table(
            ("Model", "Workload", "Mode", "Phase", "Seconds"),
            phase_rows,
        )
    )
    lines.extend(
        [
            "",
            "### Metrics not retained by the authenticated capture",
            "",
        ]
    )
    lines.extend(
        _table(
            ("Metric", "Status", "Reason"),
            (
                (
                    "Cold artifact load time",
                    "Unavailable",
                    "Not retained in the final schema-6 aggregate.",
                ),
                (
                    "Profile-worker RSS",
                    "Unavailable",
                    (
                        "Not retained in the final schema-6 aggregate; only "
                        "generation RSS is reported above."
                    ),
                ),
                (
                    "Memory traffic",
                    "Unavailable",
                    "No authenticated byte/counter measurement is present.",
                ),
                (
                    "Allocation counts/bytes",
                    "Unavailable",
                    "No authenticated allocator measurement is present.",
                ),
            ),
        )
    )
    return lines


def _raw_round_values(
    capture: Any,
    *,
    mode: str,
    batch: int,
) -> dict[int, float]:
    payload = capture.loaded.payload
    profiles = _mapping(payload.get("profiles"), "capture profiles")
    profile = _mapping(profiles.get(mode), f"profile {mode}")
    measurements = _list(profile.get("profiles"), f"profile {mode} measurements")
    matches = [
        _mapping(row, f"{mode}/{batch} measurement")
        for row in measurements
        if isinstance(row, dict) and row.get("batch_size") == batch
    ]
    if len(matches) != 1:
        _die(f"{mode}/batch-{batch} does not have exactly one measurement")
    samples = _list(
        matches[0].get("subprocess_samples"),
        f"{mode}/batch-{batch} subprocess samples",
    )
    result: dict[int, float] = {}
    ordered: list[float] = []
    for sample in samples:
        row = _mapping(sample, f"{mode}/batch-{batch} sample")
        round_index = _integer(row.get("round"), "schedule round")
        if round_index in result:
            _die(f"{mode}/batch-{batch} repeats schedule round {round_index}")
        value = _number(
            row.get("wall_seconds_per_point"),
            f"{mode}/batch-{batch}/round-{round_index} wall time",
            positive=True,
        )
        if row.get("interrupted") is not False:
            _die(f"{mode}/batch-{batch}/round-{round_index} was interrupted")
        result[round_index] = value
        ordered.append(value)
    retained = capture.timings[mode][str(batch)]
    if ordered != retained["raw_seconds_per_point"]:
        _die(f"{mode}/batch-{batch} raw samples differ from validated M0 timings")
    return result


def _runtime_sections(evidence: Evidence) -> list[str]:
    lines = [
        "## Runtime performance",
        "",
        (
            "Values are subprocess medians with raw MAD, normalized to one physical "
            "phase-space point. The headline source is "
            "`runtime_core_repeated_wall_time`."
        ),
        "",
        "### Median ± raw MAD",
        "",
    ]
    timing_rows: list[list[object]] = []
    ratio_rows: list[list[object]] = []
    for model in evidence.m0.MODELS:
        for layout in evidence.m0.LAYOUTS:
            capture = evidence.captures[(model, layout)]
            for mode in evidence.m0.MODES:
                for batch in evidence.m0.BATCHES:
                    timing = _mapping(
                        capture.timings[mode][str(batch)],
                        f"{model}/{layout}/{mode}/{batch} timing",
                    )
                    median = _number(
                        timing.get("median_seconds_per_point"),
                        "runtime median",
                        positive=True,
                    )
                    mad = _number(
                        timing.get("mad_seconds_per_point"),
                        "runtime MAD",
                        nonnegative=True,
                    )
                    count = _integer(
                        timing.get("sample_count"),
                        "runtime sample count",
                        minimum=evidence.m0.MIN_SAMPLES,
                    )
                    timing_rows.append(
                        [
                            _model_label(model),
                            _layout_label(layout),
                            _mode_label(mode),
                            batch,
                            _microseconds(median),
                            _microseconds(mad),
                            count,
                        ]
                    )
            for batch in evidence.m0.BATCHES:
                by_mode = {
                    mode: _raw_round_values(capture, mode=mode, batch=batch)
                    for mode in evidence.m0.MODES
                }
                round_sets = {tuple(sorted(values)) for values in by_mode.values()}
                if len(round_sets) != 1:
                    _die(f"{model}/{layout}/batch-{batch} round sets differ by mode")
                rounds = next(iter(round_sets))
                if len(rounds) < evidence.m0.MIN_SAMPLES:
                    _die(f"{model}/{layout}/batch-{batch} lacks seven paired rounds")
                ratios: list[str] = []
                for left, right in RATIO_PAIRS:
                    values = [
                        by_mode[left][round_index] / by_mode[right][round_index]
                        for round_index in rounds
                    ]
                    median = statistics.median(values)
                    mad = statistics.median(abs(value - median) for value in values)
                    ratios.append(f"{_ratio(median)} ± {_ratio(mad)}")
                ratio_rows.append(
                    [
                        _model_label(model),
                        _layout_label(layout),
                        batch,
                        len(rounds),
                        *ratios,
                    ]
                )
    lines.extend(
        _table(
            ("Model", "Workload", "Mode", "Batch", "Median", "Raw MAD", "Subprocesses"),
            timing_rows,
        )
    )
    lines.extend(["", "### Same-round runtime ratios", ""])
    lines.append(
        "Each ratio is formed within the same scheduler round before taking "
        "its median and raw MAD."
    )
    lines.append("")
    lines.extend(
        _table(
            (
                "Model",
                "Workload",
                "Batch",
                "Paired rounds",
                "Compiled / eager",
                "Compiled / recurrence",
                "Eager / recurrence",
            ),
            ratio_rows,
        )
    )
    return lines


def _pre_arena_comparison_sections(evidence: Evidence) -> list[str]:
    lines = [
        "## Frozen pre-Arena comparison",
        "",
        (
            "The baseline and matrix-Arena columns come from the authenticated "
            "AArch64 168-cell campaign. The final-Arena column comes from the "
            "strict M0 campaign. All use the same physical process, model family, "
            "LC selector, execution mode, batch, and runtime-core wall-time "
            "boundary. These campaigns were not scheduler-round paired, so the "
            "cross-campaign ratios below are ratios of independently computed "
            "medians; no paired uncertainty is claimed."
        ),
        "",
    ]
    rows: list[list[object]] = []
    for model in evidence.m0.MODELS:
        for layout in evidence.m0.LAYOUTS:
            capture = evidence.captures[(model, layout)]
            for mode in ("compiled", "eager"):
                for batch in evidence.m0.BATCHES:
                    distributions = evidence.pre_arena.timings[
                        (model, layout, mode, batch)
                    ]
                    baseline = _mapping(
                        distributions.get("baseline"),
                        "pre-Arena timing distribution",
                    )
                    matrix_current = _mapping(
                        distributions.get("current"),
                        "matrix Arena timing distribution",
                    )
                    final_current = _mapping(
                        capture.timings[mode][str(batch)],
                        "final Arena timing distribution",
                    )
                    baseline_median = _number(
                        baseline.get("median_seconds_per_point"),
                        "pre-Arena median",
                        positive=True,
                    )
                    baseline_mad = _number(
                        baseline.get("mad_seconds_per_point"),
                        "pre-Arena MAD",
                        nonnegative=True,
                    )
                    matrix_median = _number(
                        matrix_current.get("median_seconds_per_point"),
                        "matrix Arena median",
                        positive=True,
                    )
                    matrix_mad = _number(
                        matrix_current.get("mad_seconds_per_point"),
                        "matrix Arena MAD",
                        nonnegative=True,
                    )
                    final_median = _number(
                        final_current.get("median_seconds_per_point"),
                        "final Arena median",
                        positive=True,
                    )
                    final_mad = _number(
                        final_current.get("mad_seconds_per_point"),
                        "final Arena MAD",
                        nonnegative=True,
                    )
                    final_ratio = final_median / baseline_median
                    rows.append(
                        [
                            _model_label(model),
                            _layout_label(layout),
                            _mode_label(mode),
                            batch,
                            (
                                f"{_microseconds(baseline_median)} ± "
                                f"{_microseconds(baseline_mad)}"
                            ),
                            (
                                f"{_microseconds(matrix_median)} ± "
                                f"{_microseconds(matrix_mad)}"
                            ),
                            (
                                f"{_microseconds(final_median)} ± "
                                f"{_microseconds(final_mad)}"
                            ),
                            _ratio(matrix_median / baseline_median),
                            _ratio(final_ratio),
                            f"{(1.0 - final_ratio) * 100.0:.6g}%",
                        ]
                    )
    lines.extend(
        _table(
            (
                "Model",
                "Workload",
                "Mode",
                "Batch",
                "Frozen pre-Arena median ± MAD",
                "168-cell Arena median ± MAD",
                "Final M0 Arena median ± MAD",
                "Matrix Arena / pre-Arena",
                "Final Arena / pre-Arena",
                "Final relative gain",
            ),
            rows,
        )
    )
    lines.extend(
        [
            "",
            (
                "A final relative gain is positive when the final Arena runtime is "
                "faster than the frozen pre-Arena baseline."
            ),
        ]
    )
    return lines


def _amplicol_round_values(evidence: Any) -> dict[int, float]:
    timing = _mapping(evidence.timing, "AmpliCol timing")
    values = _list(
        timing.get("raw_seconds_per_point"),
        "AmpliCol raw seconds per point",
    )
    records = list(evidence.interleave_records)
    if len(values) != len(records):
        _die("AmpliCol timing values and interleave records differ in length")
    result: dict[int, float] = {}
    for value, record in zip(values, records, strict=True):
        round_index = _integer(record.get("round"), "AmpliCol interleave round")
        if round_index in result:
            _die(f"AmpliCol workload repeats round {round_index}")
        result[round_index] = _number(
            value,
            f"AmpliCol round {round_index} timing",
            positive=True,
        )
    return result


def _amplicol_sections(evidence: Evidence) -> list[str]:
    lines = [
        "## Original AmpliCol comparison",
        "",
        (
            "The original-AmpliCol selected-flow and all-flow measurements were "
            "captured as seven chronological selected-then-union subprocess pairs. "
            "They use different internal timing boundaries from pyAmpliCol, shown "
            "explicitly below."
        ),
        "",
    ]
    external_rows: list[list[object]] = []
    role_order = (
        evidence.m0.AMPLICOL_SELECTED_ROLE,
        evidence.m0.AMPLICOL_UNION_ROLE,
    )
    for role in role_order:
        timing = _mapping(evidence.amplicol[role].timing, f"AmpliCol {role} timing")
        external_rows.append(
            [
                role,
                timing.get("timing_boundary"),
                _microseconds(
                    _number(
                        timing.get("median_seconds_per_point"),
                        f"AmpliCol {role} median",
                        positive=True,
                    )
                ),
                _microseconds(
                    _number(
                        timing.get("mad_seconds_per_point"),
                        f"AmpliCol {role} MAD",
                        nonnegative=True,
                    )
                ),
                _integer(
                    timing.get("sample_count"),
                    f"AmpliCol {role} samples",
                    minimum=evidence.m0.MIN_SAMPLES,
                ),
            ]
        )
    lines.extend(
        _table(
            ("Workload", "Timing boundary", "Median", "Raw MAD", "Subprocesses"),
            external_rows,
        )
    )
    selected_rounds = _amplicol_round_values(evidence.amplicol[role_order[0]])
    union_rounds = _amplicol_round_values(evidence.amplicol[role_order[1]])
    if set(selected_rounds) != set(union_rounds):
        _die("original-AmpliCol selected/union paired rounds differ")
    selected_over_union = [
        selected_rounds[index] / union_rounds[index]
        for index in sorted(selected_rounds)
    ]
    ratio_median = statistics.median(selected_over_union)
    ratio_mad = statistics.median(
        abs(value - ratio_median) for value in selected_over_union
    )
    lines.extend(
        [
            "",
            (
                f"Same-round original selected/union ratio: "
                f"**{_ratio(ratio_median)} ± {_ratio(ratio_mad)} raw MAD** "
                f"across {len(selected_over_union)} paired rounds."
            ),
            "",
            "### pyAmpliCol median / original-AmpliCol median",
            "",
        ]
    )
    comparison_rows: list[list[object]] = []
    for layout, role in (
        ("topology-replay", role_order[0]),
        ("all-flow-union", role_order[1]),
    ):
        amplicol_median = _number(
            evidence.amplicol[role].timing["median_seconds_per_point"],
            "AmpliCol comparison median",
            positive=True,
        )
        for model in evidence.m0.MODELS:
            capture = evidence.captures[(model, layout)]
            for mode in evidence.m0.MODES:
                for batch in evidence.m0.BATCHES:
                    py_median = _number(
                        capture.timings[mode][str(batch)]["median_seconds_per_point"],
                        "pyAmpliCol comparison median",
                        positive=True,
                    )
                    comparison_rows.append(
                        [
                            _layout_label(layout),
                            _model_label(model),
                            _mode_label(mode),
                            batch,
                            _ratio(py_median / amplicol_median),
                            "runtime-core wall / "
                            + evidence.amplicol[role].timing["timing_boundary"],
                        ]
                    )
    lines.extend(
        _table(
            (
                "Workload",
                "Model",
                "Mode",
                "Batch",
                "pyAmpliCol / AmpliCol",
                "Boundary pairing",
            ),
            comparison_rows,
        )
    )
    return lines


def render_markdown(evidence: Evidence) -> str:
    """Render deterministic Markdown from already revalidated evidence."""

    first = evidence.captures[("built-in-sm", "topology-replay")]
    host = _mapping(first.host, "capture host")
    color_axis = first.color_axis
    helicity_axis = first.helicity_axis
    lines: list[str] = [
        (
            "<!-- Generated by tools/developer/qq_z6g_final_comparison.py; "
            "do not edit. -->"
        ),
        "",
        "# Final qq → Z + 6g Arena comparison",
        "",
        (
            "Status: **accepted** by the strict combined M0 gate. This document "
            "contains no benchmark values that were not present in the authenticated "
            "capture set or deterministically recomputed from its raw samples."
        ),
        "",
        "## Scope and workload semantics",
        "",
        f"- Process: `{evidence.m0.PROCESS}`.",
        (
            f"- Physical axes retained in every artifact: "
            f"{_integer(color_axis.get('count'), 'color-flow count', minimum=1)} "
            f"LC flows and "
            f"{_integer(helicity_axis.get('count'), 'helicity count', minimum=1)} "
            "helicity configurations."
        ),
        (
            f"- Selected-flow workload: runtime flow "
            f"`{evidence.expected['color_flow']['id']}` with a complete helicity sum."
        ),
        (
            f"- Union workload: all physical LC flows with runtime helicity "
            f"`{evidence.expected['helicity']['id']}`."
        ),
        (
            "- All four captures retain complete flow/helicity axes and have no "
            "generation-specialized axes."
        ),
        (
            "- Compiled means the compiled DAG with effective JIT O3. Eager and "
            "recurrence are separate runtime engines over the same authenticated "
            "physics/model contracts."
        ),
        "",
        "### LC/NLC/full-color boundary",
        "",
        (
            "**The all-flow union evidence here is LC-only.** It sums all physical "
            "LC flows for one runtime-selected helicity. NLC and full-color modes use "
            "contracted color evaluation and are outside this M0 capture set; no "
            "NLC/full runtime number or union equivalence is inferred from these LC "
            "measurements."
        ),
        "",
        "## Evidence and identities",
        "",
        "### Content-addressed inputs",
        "",
    ]
    lines.extend(
        _table(
            ("Logical evidence", "Raw file SHA-256", "Canonical/content SHA-256"),
            _capture_input_rows(evidence),
        )
    )
    lines.extend(["", "### pyAmpliCol source and runtime", ""])
    lines.extend(_table(("Identity", "Value"), _runtime_identity_rows(evidence)))
    lines.extend(["", "### Frozen pre-Arena matrix and runtime", ""])
    lines.extend(_table(("Identity", "Value"), _pre_arena_identity_rows(evidence)))
    lines.extend(["", "### Original AmpliCol source and compiler", ""])
    lines.extend(_table(("Identity", "Value"), _original_identity_rows(evidence)))
    lines.extend(["", "### Host", ""])
    lines.extend(
        _table(
            ("Field", "Value"),
            (
                ("Platform", host.get("platform")),
                ("System/release", f"{host.get('system')} {host.get('release')}"),
                ("Machine", host.get("machine")),
                ("CPU model", host.get("cpu_model")),
                ("Logical CPUs", host.get("logical_cpu_count")),
            ),
        )
    )
    lines.extend(["", "### Model identities", ""])
    lines.extend(
        _table(
            ("Model", "Common physics SHA-256", "Generation-model SHA-256"),
            [
                [
                    _model_label(model),
                    _digest(
                        evidence.expected["model_common_physics_identity_sha256"][model]
                    ),
                    _digest(
                        evidence.expected["generation_model_identities_sha256"][model]
                    ),
                ]
                for model in evidence.m0.MODELS
            ],
        )
    )
    lines.extend(["", *_numerical_sections(evidence)])
    lines.extend(["", *_generation_sections(evidence)])
    lines.extend(["", *_runtime_sections(evidence)])
    lines.extend(["", *_pre_arena_comparison_sections(evidence)])
    lines.extend(["", *_amplicol_sections(evidence)])
    lines.extend(
        [
            "",
            "## Methodology and interpretation caveats",
            "",
            (
                "- Runtime cells use at least seven independent worker subprocesses. "
                "Each worker retains native wall blocks; the headline statistic is "
                "the median of subprocess wall-seconds-per-point values and its raw "
                "MAD."
            ),
            (
                "- Same-round ratios divide paired subprocess measurements before "
                "taking the ratio median/MAD. A ratio below one favors the numerator."
            ),
            (
                "- Frozen pre-Arena comparisons are cross-campaign median ratios, "
                "not same-round ratios; the table retains the raw MAD for every "
                "campaign instead of constructing a synthetic paired uncertainty."
            ),
            (
                "- Batch sizes 1, 128, and 1024 are API/runtime batching choices; all "
                "reported values remain normalized per evaluated physical point."
            ),
            (
                "- Generation timings, artifact sizes, and generation RSS are not "
                "steady-state runtime measurements. Generation RSS is a high-water "
                "lower bound, not an aggregate process-tree sample."
            ),
            (
                "- Original-AmpliCol and pyAmpliCol timing boundaries differ. Their "
                "ratios are reported for transparency, not presented as identical "
                "microbenchmark boundaries."
            ),
            (
                "- Zero or unavailable evaluator-attribution, cold-load, traffic, or "
                "allocation fields are not substituted with estimates. The headline "
                "runtime tables use retained wall-time evidence only."
            ),
            (
                f"- Numerical acceptance used relative tolerance "
                f"`{evidence.m0.RTOL:.17g}` and absolute tolerance "
                f"`{evidence.m0.ATOL:.17g}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def generate(
    *,
    request_path: Path,
    request_sha256: str,
    acceptance_path: Path,
    acceptance_sha256: str,
    expected_source_revision: str,
    expected_runtime_provenance_sha256: str,
    pre_arena_manifest_path: Path,
    pre_arena_manifest_sha256: str,
    expected_pre_arena_source_revision: str,
    expected_pre_arena_build_sha256: str,
    expected_pre_arena_runtime_sha256: str,
    output_path: Path,
) -> str:
    evidence = validate_evidence(
        request_path=request_path,
        request_sha256=request_sha256,
        acceptance_path=acceptance_path,
        acceptance_sha256=acceptance_sha256,
        expected_source_revision=expected_source_revision,
        expected_runtime_provenance_sha256=expected_runtime_provenance_sha256,
        pre_arena_manifest_path=pre_arena_manifest_path,
        pre_arena_manifest_sha256=pre_arena_manifest_sha256,
        expected_pre_arena_source_revision=expected_pre_arena_source_revision,
        expected_pre_arena_build_sha256=expected_pre_arena_build_sha256,
        expected_pre_arena_runtime_sha256=expected_pre_arena_runtime_sha256,
    )
    markdown = render_markdown(evidence)
    _write_atomic(output_path, markdown)
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--request", type=Path, required=True)
    result.add_argument("--request-sha256", required=True)
    result.add_argument("--acceptance", type=Path, required=True)
    result.add_argument("--acceptance-sha256", required=True)
    result.add_argument("--expected-source-revision", required=True)
    result.add_argument("--expected-runtime-provenance-sha256", required=True)
    result.add_argument("--pre-arena-manifest", type=Path, required=True)
    result.add_argument("--pre-arena-manifest-sha256", required=True)
    result.add_argument("--expected-pre-arena-source-revision", required=True)
    result.add_argument("--expected-pre-arena-build-sha256", required=True)
    result.add_argument("--expected-pre-arena-runtime-sha256", required=True)
    result.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown output (default: {DEFAULT_OUTPUT})",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        markdown_sha256 = generate(
            request_path=arguments.request,
            request_sha256=arguments.request_sha256,
            acceptance_path=arguments.acceptance,
            acceptance_sha256=arguments.acceptance_sha256,
            expected_source_revision=arguments.expected_source_revision,
            expected_runtime_provenance_sha256=(
                arguments.expected_runtime_provenance_sha256
            ),
            pre_arena_manifest_path=arguments.pre_arena_manifest,
            pre_arena_manifest_sha256=arguments.pre_arena_manifest_sha256,
            expected_pre_arena_source_revision=(
                arguments.expected_pre_arena_source_revision
            ),
            expected_pre_arena_build_sha256=(arguments.expected_pre_arena_build_sha256),
            expected_pre_arena_runtime_sha256=(
                arguments.expected_pre_arena_runtime_sha256
            ),
            output_path=arguments.output,
        )
    except Exception as error:
        print(f"qq_Z6g comparison rejected: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "markdown_sha256": markdown_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
