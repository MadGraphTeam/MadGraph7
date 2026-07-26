# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from tools.developer import arena_native_x86_acceptance as native
from tools.developer import compiled_mode_matrix as matrix
from tools.developer import compiled_mode_matrix_x86 as matrix_x86
from tools.developer import x86_portable_performance_acceptance as portable
from tools.developer import x86_qq_recurrence_acceptance as qq

REVISION = "a" * 40
WORKFLOW_RUN_ID = "987654321"
NATIVE_INPUTS = "b" * 64
RUNTIME_BUNDLE = "c" * 64
NATIVE_MODULE = "d" * 64


def _write(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _native() -> dict[str, object]:
    evidence = {
        name: {"semantic_validation": {"passes": True}}
        for name in (
            "runtime_preflight",
            "compiled_all_jit",
            "four_quark",
            "eager_compiled_color",
        )
    }
    return native._attach_content_identity(
        {
            "kind": native.ACCEPTANCE_KIND,
            "schema_version": native.SCHEMA_VERSION,
            "status": "ok",
            "passes": True,
            "request": {
                "expected_revision": REVISION,
                "expected_target": native.EXPECTED_TARGET,
                "point_count": native.EXPECTED_POINT_COUNT,
            },
            "source_identity": {
                "all_match": True,
                "audit": {"revision": REVISION, "dirty": False},
            },
            "runtime_identity": {
                "all_match": True,
                "audit": {
                    "active_build_info": {
                        "payload": {
                            "source_revision": REVISION,
                            "native_build_inputs_sha256": NATIVE_INPUTS,
                            "publishable": False,
                        }
                    },
                    "native_extension": {
                        "target": native.EXPECTED_TARGET,
                        "build_inputs_sha256": NATIVE_INPUTS,
                        "sha256": NATIVE_MODULE,
                    },
                },
            },
            "evidence": evidence,
            "validation": {
                "exact_source_revision": True,
                "source_clean_preflight_and_postflight": True,
                "source_only_python_runtime": True,
                "native_linux_x86_64_target": True,
            },
        }
    )


def _matrix() -> dict[str, object]:
    passing_gate = {"passes": True}
    return matrix_x86._attach_content_identity(
        {
            "kind": matrix_x86.AGGREGATE_KIND,
            "schema_version": matrix_x86.SCHEMA_VERSION,
            "matrix_contract": matrix.MATRIX_CONTRACT,
            "workflow_run_id": WORKFLOW_RUN_ID,
            "runtime_bundle_sha256": RUNTIME_BUNDLE,
            "expected_current_source_revision": REVISION,
            "expected_builds": {
                "current": {
                    "source_revision": REVISION,
                    "native_build_inputs_sha256": NATIVE_INPUTS,
                    "native_module_sha256": NATIVE_MODULE,
                }
            },
            "shard_gate": {"errors": [], "passes": True},
            "matrix_audit": {
                "complete": True,
                "passes": True,
                "coverage": {
                    "expected": 168,
                    "observed": 168,
                    "missing": [],
                    "unexpected": [],
                    "passes": True,
                },
                "cell_gate": dict(passing_gate),
                "identity_gate": dict(passing_gate),
                "gain_gate": {
                    "required_relative_gain": 0.1,
                    "passes": True,
                },
                "generation_gate": {
                    "maximum_geometric_mean": 1.05,
                    "geometric_mean_current_over_baseline": 1.01,
                    "passes": True,
                },
            },
            "complete": True,
            "passes": True,
        }
    )


def _qq() -> dict[str, object]:
    captures = {}
    for role in qq.CAPTURE_CONTRACTS:
        cells = []
        for batch_size in qq.PERFORMANCE_BATCH_SIZES:
            recurrence = [0.001 for _ in range(7)]
            ratios = [1.0 + index * 0.01 for index in range(7)]
            compiled = [
                denominator * ratio
                for denominator, ratio in zip(
                    recurrence,
                    ratios,
                    strict=True,
                )
            ]
            median = statistics.median(ratios)
            raw_mad = statistics.median(
                abs(value - median) for value in ratios
            )
            cells.append(
                {
                    "batch_size": batch_size,
                    "pairing": "same-interleaved-schedule-round-v1",
                    "sample_count": 7,
                    "compiled_wall_seconds_per_point_by_round": compiled,
                    "recurrence_wall_seconds_per_point_by_round": recurrence,
                    "compiled_over_recurrence_ratios_by_round": ratios,
                    "ratio_statistics": {
                        "contract": "median-and-raw-mad-v1",
                        "median": median,
                        "raw_mad": raw_mad,
                        "upper_three_raw_mad": median + 3.0 * raw_mad,
                    },
                    "ceiling": 1.15,
                    "passes": True,
                }
            )
        contract = qq.CAPTURE_CONTRACTS[role]
        captures[role] = {
            "source_revision": REVISION,
            "runtime_provenance_sha256": "e" * 64,
            "model": {"kind": contract["model"]},
            "layout": contract["layout"],
            "workload": contract["workload"],
            "numerical_validation": {
                "contract": "recomputed-recurrence-z6g-pairwise-and-resolved-v1",
                "passes": True,
            },
            "performance_cells": cells,
            "passes": True,
        }
    return qq._attach_content_identity(
        {
            "kind": qq.RESULT_KIND,
            "schema_version": qq.SCHEMA_VERSION,
            "workflow_run_id": WORKFLOW_RUN_ID,
            "target": native.EXPECTED_TARGET,
            "expected_current_revision": REVISION,
            "runtime_bundle": {"content_sha256": RUNTIME_BUNDLE},
            "policy": {
                "required_capture_roles": list(qq.CAPTURE_CONTRACTS),
                "required_modes": list(qq.REQUIRED_MODES),
                "required_batch_sizes": list(qq.REQUIRED_BATCH_SIZES),
                "performance_batch_sizes": list(qq.PERFORMANCE_BATCH_SIZES),
                "minimum_target_runtime_seconds_per_worker": 5.0,
                "subprocess_samples_per_cell": 7,
                "native_wall_blocks_per_worker": 7,
                "warmup_runs": 2,
                "numerical_validation_samples": 10,
                "diagnostic_shortcuts_allowed": False,
                "compiled_over_recurrence_ratio_ceiling": 1.15,
                "ratio_gate": (
                    "median-plus-three-raw-mad-at-or-below-ceiling-v1"
                ),
            },
            "captures": captures,
            "performance_cell_count": 8,
            "passes": True,
        }
    )


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        _write(tmp_path / "native.json", _native()),
        _write(tmp_path / "matrix.json", _matrix()),
        _write(tmp_path / "qq.json", _qq()),
    )


def _audit(inputs: tuple[Path, Path, Path]) -> dict[str, object]:
    native_path, matrix_path, qq_path = inputs
    return portable.audit(
        native_acceptance=native_path,
        matrix_acceptance=matrix_path,
        qq_acceptance=qq_path,
        workflow_run_id=WORKFLOW_RUN_ID,
        expected_revision=REVISION,
    )


def test_combines_exact_content_addressed_inputs_portably(
    inputs: tuple[Path, Path, Path],
) -> None:
    result = _audit(inputs)
    assert result["passes"] is True
    assert result["bindings"] == {
        "same_source_revision": True,
        "same_native_build_inputs": True,
        "same_native_module": True,
        "same_performance_runtime_bundle": True,
    }
    assert str(inputs[0].parent) not in json.dumps(result)
    identity = result["content_identity"]
    body = dict(result)
    body.pop("content_identity")
    assert identity["sha256"] == portable._canonical_sha256(body)


def test_rejects_different_native_build_inputs(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _matrix()
    body = dict(payload)
    body.pop("content_identity")
    body["expected_builds"]["current"]["native_build_inputs_sha256"] = "0" * 64
    _write(matrix_path, matrix_x86._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="different native build inputs",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_missing_native_identity_digests(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _native()
    body = dict(payload)
    body.pop("content_identity")
    runtime = body["runtime_identity"]["audit"]
    runtime["active_build_info"]["payload"].pop("native_build_inputs_sha256")
    runtime["native_extension"].pop("build_inputs_sha256")
    runtime["native_extension"].pop("sha256")
    _write(native_path, native._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="identity digests are invalid",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_missing_matrix_identity_digests(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _matrix()
    body = dict(payload)
    body.pop("content_identity")
    current = body["expected_builds"]["current"]
    current.pop("native_build_inputs_sha256")
    current.pop("native_module_sha256")
    body.pop("runtime_bundle_sha256")
    _write(matrix_path, matrix_x86._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="matrix current build identity drifted",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_non_lowercase_identity_digest(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _native()
    body = dict(payload)
    body.pop("content_identity")
    runtime = body["runtime_identity"]["audit"]
    runtime["native_extension"]["sha256"] = "A" * 64
    _write(native_path, native._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="identity digests are invalid",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_different_performance_runtime_bundles(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _qq()
    body = dict(payload)
    body.pop("content_identity")
    body["runtime_bundle"]["content_sha256"] = "0" * 64
    _write(qq_path, qq._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="different runtime bundles",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_missing_qq_runtime_bundle_digest(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _qq()
    body = dict(payload)
    body.pop("content_identity")
    body["runtime_bundle"].pop("content_sha256")
    _write(qq_path, qq._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="runtime bundle digest is absent",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_duplicate_qq_batch_and_missing_1024(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _qq()
    body = dict(payload)
    body.pop("content_identity")
    cells = body["captures"]["builtin-topology"]["performance_cells"]
    cells[1] = dict(cells[0])
    _write(qq_path, qq._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="batch coverage is incomplete",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_malformed_qq_runtime_bundle(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _qq()
    body = dict(payload)
    body.pop("content_identity")
    body["runtime_bundle"] = []
    _write(qq_path, qq._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="runtime bundle identity is absent",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_different_native_module(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _matrix()
    body = dict(payload)
    body.pop("content_identity")
    body["expected_builds"]["current"]["native_module_sha256"] = "0" * 64
    _write(matrix_path, matrix_x86._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="different native modules",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_failed_or_incomplete_matrix_gate(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = _matrix()
    body = dict(payload)
    body.pop("content_identity")
    body["matrix_audit"]["coverage"]["observed"] = 167
    _write(matrix_path, matrix_x86._attach_content_identity(body))
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="matrix gates are incomplete",
    ):
        _audit((native_path, matrix_path, qq_path))


def test_rejects_content_tampering(
    inputs: tuple[Path, Path, Path],
) -> None:
    native_path, matrix_path, qq_path = inputs
    payload = json.loads(qq_path.read_text(encoding="utf-8"))
    payload["performance_cell_count"] = 7
    _write(qq_path, payload)
    with pytest.raises(
        portable.PortableAcceptanceError,
        match="content identity is invalid",
    ):
        _audit((native_path, matrix_path, qq_path))
