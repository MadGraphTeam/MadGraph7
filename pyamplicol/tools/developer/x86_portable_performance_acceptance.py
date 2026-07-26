#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Combine x86 correctness, 168-cell matrix, and ``qq -> Z+6g`` evidence.

The inputs are independently content-addressed artifacts from the same
dispatched source revision.  The output intentionally contains no host-local
paths so it can be retained as the final portable candidate verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import arena_native_x86_acceptance as native  # noqa: E402
from tools.developer import compiled_mode_matrix as matrix  # noqa: E402
from tools.developer import compiled_mode_matrix_x86 as matrix_x86  # noqa: E402
from tools.developer import x86_qq_recurrence_acceptance as qq  # noqa: E402

RESULT_KIND = "pyamplicol-x86-portable-candidate-acceptance"
SCHEMA_VERSION = 1
CONTENT_IDENTITY_ALGORITHM = "sha256-canonical-json-body-v1"
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class PortableAcceptanceError(RuntimeError):
    """Raised when independently produced x86 evidence does not compose."""


def _canonical_sha256(value: object) -> str:
    return matrix_x86._canonical_sha256(value)


def _attach_content_identity(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["content_identity"] = {
        "algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sha256": _canonical_sha256(body),
    }
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_content_identity(
    payload: Mapping[str, object],
    *,
    label: str,
) -> str:
    identity = payload.get("content_identity")
    body = dict(payload)
    body.pop("content_identity", None)
    if (
        not isinstance(identity, Mapping)
        or identity.get("algorithm") != CONTENT_IDENTITY_ALGORITHM
        or identity.get("sha256") != _canonical_sha256(body)
    ):
        raise PortableAcceptanceError(f"{label} content identity is invalid")
    return str(identity["sha256"])


def _portable_file_identity(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        key: identity[key]
        for key in ("size_bytes", "sha256", "canonical_sha256")
        if key in identity
    }


def _checked_input(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, object], str]:
    try:
        payload, identity = matrix_x86._checked_json(path, label=label)
    except matrix_x86.ShardError as error:
        raise PortableAcceptanceError(str(error)) from error
    digest = _require_content_identity(payload, label=label)
    return payload, _portable_file_identity(identity), digest


def _audit_native(
    payload: Mapping[str, Any],
    *,
    expected_revision: str,
) -> dict[str, object]:
    request = payload.get("request")
    source = payload.get("source_identity")
    runtime = payload.get("runtime_identity")
    evidence = payload.get("evidence")
    validation = payload.get("validation")
    if (
        payload.get("kind") != native.ACCEPTANCE_KIND
        or payload.get("schema_version") != native.SCHEMA_VERSION
        or payload.get("status") != "ok"
        or payload.get("passes") is not True
        or not isinstance(request, Mapping)
        or request.get("expected_revision") != expected_revision
        or request.get("expected_target") != native.EXPECTED_TARGET
        or request.get("point_count") != native.EXPECTED_POINT_COUNT
        or not isinstance(source, Mapping)
        or source.get("all_match") is not True
        or not isinstance(runtime, Mapping)
        or runtime.get("all_match") is not True
        or not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "runtime_preflight",
            "compiled_all_jit",
            "four_quark",
            "eager_compiled_color",
        }
        or not isinstance(validation, Mapping)
        or not validation
        or any(value is not True for value in validation.values())
    ):
        raise PortableAcceptanceError("native correctness acceptance is incomplete")
    for name, record in evidence.items():
        semantic = (
            record.get("semantic_validation")
            if isinstance(record, Mapping)
            else None
        )
        if not isinstance(semantic, Mapping) or semantic.get("passes") is not True:
            raise PortableAcceptanceError(
                f"native correctness evidence {name} did not pass"
            )
    audit_source = source.get("audit")
    audit_runtime = runtime.get("audit")
    active_build = (
        audit_runtime.get("active_build_info")
        if isinstance(audit_runtime, Mapping)
        else None
    )
    build_payload = (
        active_build.get("payload")
        if isinstance(active_build, Mapping)
        else None
    )
    native_extension = (
        audit_runtime.get("native_extension")
        if isinstance(audit_runtime, Mapping)
        else None
    )
    if (
        not isinstance(audit_source, Mapping)
        or audit_source.get("revision") != expected_revision
        or audit_source.get("dirty") is not False
        or not isinstance(build_payload, Mapping)
        or build_payload.get("source_revision") != expected_revision
        or build_payload.get("publishable") is not False
        or not isinstance(native_extension, Mapping)
        or native_extension.get("target") != native.EXPECTED_TARGET
        or native_extension.get("build_inputs_sha256")
        != build_payload.get("native_build_inputs_sha256")
    ):
        raise PortableAcceptanceError("native correctness runtime/source drifted")
    return {
        "target": native.EXPECTED_TARGET,
        "point_count": native.EXPECTED_POINT_COUNT,
        "native_build_inputs_sha256": build_payload["native_build_inputs_sha256"],
        "native_module_sha256": native_extension.get("sha256"),
        "required_evidence": sorted(evidence),
        "passes": True,
    }


def _audit_matrix(
    payload: Mapping[str, Any],
    *,
    workflow_run_id: str,
    expected_revision: str,
) -> dict[str, object]:
    shard_gate = payload.get("shard_gate")
    audit = payload.get("matrix_audit")
    if (
        payload.get("kind") != matrix_x86.AGGREGATE_KIND
        or payload.get("schema_version") != matrix_x86.SCHEMA_VERSION
        or payload.get("matrix_contract") != matrix.MATRIX_CONTRACT
        or payload.get("workflow_run_id") != workflow_run_id
        or payload.get("expected_current_source_revision") != expected_revision
        or payload.get("complete") is not True
        or payload.get("passes") is not True
        or not isinstance(shard_gate, Mapping)
        or shard_gate.get("passes") is not True
        or shard_gate.get("errors") != []
        or not isinstance(audit, Mapping)
        or audit.get("complete") is not True
        or audit.get("passes") is not True
    ):
        raise PortableAcceptanceError("168-cell matrix acceptance is incomplete")
    coverage = audit.get("coverage")
    cell_gate = audit.get("cell_gate")
    identity_gate = audit.get("identity_gate")
    gain_gate = audit.get("gain_gate")
    generation_gate = audit.get("generation_gate")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("expected") != 168
        or coverage.get("observed") != 168
        or coverage.get("missing") != []
        or coverage.get("unexpected") != []
        or coverage.get("passes") is not True
        or not all(
            isinstance(gate, Mapping) and gate.get("passes") is True
            for gate in (cell_gate, identity_gate, gain_gate, generation_gate)
        )
    ):
        raise PortableAcceptanceError("168-cell matrix gates are incomplete")
    expected_builds = payload.get("expected_builds")
    current = (
        expected_builds.get("current")
        if isinstance(expected_builds, Mapping)
        else None
    )
    if (
        not isinstance(current, Mapping)
        or current.get("source_revision") != expected_revision
    ):
        raise PortableAcceptanceError("matrix current build identity drifted")
    return {
        "matrix_contract": matrix.MATRIX_CONTRACT,
        "cell_count": 168,
        "shard_count": matrix_x86.SHARD_COUNT,
        "runtime_bundle_sha256": payload.get("runtime_bundle_sha256"),
        "current_native_build_inputs_sha256": current.get(
            "native_build_inputs_sha256"
        ),
        "current_native_module_sha256": current.get("native_module_sha256"),
        "gain_gate": {
            "required_relative_gain": gain_gate.get("required_relative_gain"),
            "passes": True,
        },
        "generation_gate": {
            "maximum_geometric_mean": generation_gate.get(
                "maximum_geometric_mean"
            ),
            "observed_geometric_mean": generation_gate.get(
                "geometric_mean_current_over_baseline"
            ),
            "passes": True,
        },
        "passes": True,
    }


def _audit_qq(
    payload: Mapping[str, Any],
    *,
    workflow_run_id: str,
    expected_revision: str,
) -> dict[str, object]:
    captures = payload.get("captures")
    policy = payload.get("policy")
    if (
        payload.get("kind") != qq.RESULT_KIND
        or payload.get("schema_version") != qq.SCHEMA_VERSION
        or payload.get("workflow_run_id") != workflow_run_id
        or payload.get("target") != native.EXPECTED_TARGET
        or payload.get("expected_current_revision") != expected_revision
        or payload.get("performance_cell_count") != 8
        or payload.get("passes") is not True
        or not isinstance(captures, Mapping)
        or set(captures) != set(qq.CAPTURE_CONTRACTS)
        or not isinstance(policy, Mapping)
        or policy.get("diagnostic_shortcuts_allowed") is not False
        or policy.get("compiled_over_recurrence_ratio_ceiling")
        != qq.COMPILED_RECURRENCE_RATIO_CEILING
    ):
        raise PortableAcceptanceError("qq Z+6g acceptance is incomplete")
    cell_count = 0
    maximum_upper = 0.0
    for role, capture in captures.items():
        cells = (
            capture.get("performance_cells")
            if isinstance(capture, Mapping)
            else None
        )
        numerical = (
            capture.get("numerical_validation")
            if isinstance(capture, Mapping)
            else None
        )
        if (
            not isinstance(capture, Mapping)
            or capture.get("passes") is not True
            or not isinstance(numerical, Mapping)
            or numerical.get("passes") is not True
            or not isinstance(cells, list)
            or len(cells) != 2
        ):
            raise PortableAcceptanceError(f"qq Z+6g capture {role} is incomplete")
        for cell in cells:
            statistics_record = (
                cell.get("ratio_statistics")
                if isinstance(cell, Mapping)
                else None
            )
            upper = (
                statistics_record.get("upper_three_raw_mad")
                if isinstance(statistics_record, Mapping)
                else None
            )
            if (
                not isinstance(cell, Mapping)
                or cell.get("passes") is not True
                or cell.get("sample_count") != 7
                or cell.get("batch_size") not in qq.PERFORMANCE_BATCH_SIZES
                or not isinstance(upper, (int, float))
                or isinstance(upper, bool)
                or float(upper) > qq.COMPILED_RECURRENCE_RATIO_CEILING
            ):
                raise PortableAcceptanceError(
                    f"qq Z+6g capture {role} performance gate failed"
                )
            cell_count += 1
            maximum_upper = max(maximum_upper, float(upper))
    runtime_bundle = payload.get("runtime_bundle")
    if not isinstance(runtime_bundle, Mapping):
        raise PortableAcceptanceError("qq Z+6g runtime bundle identity is absent")
    runtime_bundle_sha256 = runtime_bundle.get("content_sha256")
    if not isinstance(runtime_bundle_sha256, str):
        raise PortableAcceptanceError("qq Z+6g runtime bundle digest is absent")
    return {
        "capture_count": len(captures),
        "performance_cell_count": cell_count,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "compiled_over_recurrence_ratio_ceiling": (
            qq.COMPILED_RECURRENCE_RATIO_CEILING
        ),
        "maximum_observed_upper_three_raw_mad": maximum_upper,
        "numerical_validation": True,
        "passes": True,
    }


def audit(
    *,
    native_acceptance: Path,
    matrix_acceptance: Path,
    qq_acceptance: Path,
    workflow_run_id: str,
    expected_revision: str,
) -> dict[str, object]:
    native_payload, native_file, native_digest = _checked_input(
        native_acceptance,
        label="native x86 correctness acceptance",
    )
    matrix_payload, matrix_file, matrix_digest = _checked_input(
        matrix_acceptance,
        label="x86 168-cell matrix acceptance",
    )
    qq_payload, qq_file, qq_digest = _checked_input(
        qq_acceptance,
        label="x86 qq Z+6g acceptance",
    )
    native_summary = _audit_native(
        native_payload,
        expected_revision=expected_revision,
    )
    matrix_summary = _audit_matrix(
        matrix_payload,
        workflow_run_id=workflow_run_id,
        expected_revision=expected_revision,
    )
    qq_summary = _audit_qq(
        qq_payload,
        workflow_run_id=workflow_run_id,
        expected_revision=expected_revision,
    )
    if (
        native_summary["native_build_inputs_sha256"]
        != matrix_summary["current_native_build_inputs_sha256"]
    ):
        raise PortableAcceptanceError(
            "correctness and performance used different native build inputs"
        )
    if (
        native_summary["native_module_sha256"]
        != matrix_summary["current_native_module_sha256"]
    ):
        raise PortableAcceptanceError(
            "correctness and performance used different native modules"
        )
    if (
        matrix_summary["runtime_bundle_sha256"]
        != qq_summary["runtime_bundle_sha256"]
    ):
        raise PortableAcceptanceError(
            "matrix and qq evidence used different runtime bundles"
        )
    return _attach_content_identity(
        {
            "kind": RESULT_KIND,
            "schema_version": SCHEMA_VERSION,
            "workflow_run_id": workflow_run_id,
            "source_revision": expected_revision,
            "target": native.EXPECTED_TARGET,
            "inputs": {
                "native_correctness": {
                    "file": native_file,
                    "content_sha256": native_digest,
                },
                "compiled_eager_matrix": {
                    "file": matrix_file,
                    "content_sha256": matrix_digest,
                },
                "qq_z6g_recurrence": {
                    "file": qq_file,
                    "content_sha256": qq_digest,
                },
            },
            "correctness": native_summary,
            "compiled_eager_matrix": matrix_summary,
            "qq_z6g_recurrence": qq_summary,
            "bindings": {
                "same_source_revision": True,
                "same_native_build_inputs": True,
                "same_native_module": True,
                "same_performance_runtime_bundle": True,
            },
            "passes": True,
        }
    )


def _git_sha(value: str) -> str:
    if _GIT_SHA.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase 40-character Git SHA")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--native-acceptance", type=Path, required=True)
    result.add_argument("--matrix-acceptance", type=Path, required=True)
    result.add_argument("--qq-acceptance", type=Path, required=True)
    result.add_argument("--workflow-run-id", required=True)
    result.add_argument("--expected-revision", type=_git_sha, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        result = audit(
            native_acceptance=arguments.native_acceptance,
            matrix_acceptance=arguments.matrix_acceptance,
            qq_acceptance=arguments.qq_acceptance,
            workflow_run_id=arguments.workflow_run_id,
            expected_revision=arguments.expected_revision,
        )
        _write_json_atomic(arguments.output, result)
    except (OSError, PortableAcceptanceError) as error:
        print(f"x86-portable-performance-acceptance: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
