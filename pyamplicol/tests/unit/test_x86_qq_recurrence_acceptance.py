# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from tools.developer import x86_performance_runtime_bundle as bundle_tool
from tools.developer import x86_qq_recurrence_acceptance as acceptance

REVISION = "a" * 40
WORKFLOW_RUN_ID = "123456789"
BUILD_INFO_SHA256 = "b" * 64
NATIVE_SHA256 = "c" * 64
NATIVE_INPUTS_SHA256 = "d" * 64
PREPARED_SHA256 = "e" * 64


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _current_installation() -> dict[str, object]:
    return {
        "package_version": "0.1.0.dev0+candidate.test",
        "build_info": {
            "schema_version": 1,
            "version": "0.1.0.dev0+candidate.test",
            "candidate_fingerprint": "0123456789ab",
            "source_revision": REVISION,
            "source_checkout": "/workspace/source",
            "native_build_inputs_sha256": NATIVE_INPUTS_SHA256,
            "publishable": False,
        },
        "build_info_sha256": BUILD_INFO_SHA256,
        "distribution_content": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": "f" * 64,
            "file_count": 12,
            "size_bytes": 3456,
        },
        "native_module": {
            "relative_path": "pyamplicol/_rusticol.abi3.so",
            "sha256": NATIVE_SHA256,
            "size_bytes": 2345,
        },
    }


def _bundle() -> dict[str, object]:
    return bundle_tool._attach_content_identity(
        {
            "kind": bundle_tool.BUNDLE_KIND,
            "schema_version": bundle_tool.SCHEMA_VERSION,
            "target": "x86_64-unknown-linux-gnu",
            "workflow_run_id": WORKFLOW_RUN_ID,
            "expected_current_revision": REVISION,
            "installations": {
                "baseline": {"test": True},
                "current": _current_installation(),
            },
            "dependency_site": {"test": True},
            "wheels": {"test": True},
            "prepared_models": {
                "ufo-sm": {
                    "relative_path": (
                        "prepared-models/ufo-sm-jit-o2.pyamplicol-model"
                    ),
                    "sha256": PREPARED_SHA256,
                    "size_bytes": 4567,
                }
            },
            "frozen_baseline": {"test": True},
            "passes": True,
        }
    )


def _profile(batch_size: int, *, mode: str, slow: bool = False) -> dict[str, object]:
    recurrence = [0.001 * (1.0 + index * 0.01) for index in range(7)]
    ratios = (
        [1.20 + index * 0.01 for index in range(7)]
        if slow
        else [1.00 + index * 0.01 for index in range(7)]
    )
    values = (
        recurrence
        if mode == "recurrence"
        else [wall * ratio for wall, ratio in zip(recurrence, ratios, strict=True)]
    )
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "batch_size": batch_size,
        "interrupted": False,
        "sample_count": 7,
        "subprocess_sample_count": 7,
        "statistics_contract": "subprocess-median-and-raw-mad-v1",
        "wall_seconds_per_point": median,
        "wall_seconds_per_point_median": median,
        "wall_seconds_per_point_mad": mad,
        "subprocess_samples": [
            {
                "round": round_index,
                "wall_seconds_per_point": value,
            }
            for round_index, value in enumerate(values)
        ],
    }


def _models(role: str) -> dict[str, object]:
    if role.startswith("builtin-"):
        packaged = {
            "kind": "packaged-prepared-model",
            "resource_id": "built-in-sm-jit-o2",
            "size_bytes": 1234,
            "sha256": "1" * 64,
            "compile_excluded_from_generation": True,
        }
        return {
            "compiled": {
                "kind": "built-in-sm-source",
                "resource_id": None,
                "source_revision": REVISION,
                "compile_excluded_from_generation": False,
            },
            "eager": dict(packaged),
            "recurrence": dict(packaged),
        }
    explicit = {
        "kind": "explicit-prepared-model",
        "resource_id": None,
        "file": {
            "path": "/runtime/ufo.pyamplicol-model",
            "resolved_path": "/runtime/ufo.pyamplicol-model",
            "sha256": PREPARED_SHA256,
            "size_bytes": 4567,
        },
        "compile_excluded_from_generation": True,
    }
    return {mode: copy.deepcopy(explicit) for mode in acceptance.REQUIRED_MODES}


def _runtime() -> dict[str, object]:
    current = _current_installation()
    return {
        "active_build_info": {
            "path": "/runtime/pyamplicol/_build_info.json",
            "resolved_path": "/runtime/pyamplicol/_build_info.json",
            "sha256": BUILD_INFO_SHA256,
            "size_bytes": 400,
            "payload": copy.deepcopy(current["build_info"]),
        },
        "installed_distribution": {
            "package_version": current["package_version"],
            "build_info_files": [
                {
                    "relative_path": "pyamplicol/_build_info.json",
                    "sha256": BUILD_INFO_SHA256,
                    "size_bytes": 400,
                }
            ],
            "native_modules": [copy.deepcopy(current["native_module"])],
            "distribution_content": copy.deepcopy(current["distribution_content"]),
        },
        "native_extension": {
            "path": "/runtime/pyamplicol/_rusticol.abi3.so",
            "resolved_path": "/runtime/pyamplicol/_rusticol.abi3.so",
            "sha256": NATIVE_SHA256,
            "size_bytes": 2345,
            "build_inputs_sha256": NATIVE_INPUTS_SHA256,
            "package_version": "0.1.0-dev.0+candidate.test",
        },
        "interpreter": {"test": True},
        "dependencies": {"test": True},
    }


def _capture(role: str, *, slow: bool = False) -> dict[str, Any]:
    contract = acceptance.CAPTURE_CONTRACTS[role]
    return {
        "kind": "pyamplicol-recurrence-z6g-benchmark",
        "schema_version": 6,
        "complete": True,
        "passes": True,
        "capture_acceptance": {"test": "capture"},
        "milestone0_acceptance": {"test": "milestone"},
        "source": {
            "checkout": "/workspace/source",
            "revision": REVISION,
            "dirty": False,
            "untracked_files_checked": True,
        },
        "runtime_provenance": _runtime(),
        "provenance": {
            "host": {
                "system": "Linux",
                "machine": "x86_64",
            }
        },
        "process": "u u~ > Z g g g g g g",
        "process_name": "uubar_Z_6g",
        "workload": contract["workload"],
        "configuration": {
            "batch_sizes": [1, 128, 1024],
            "target_runtime_seconds": 5.0,
            "minimum_samples": 7,
            "subprocess_samples": 7,
            "warmup_runs": 2,
            "generation_timeout_seconds": 900.0,
            "profile_timeout_seconds": 300.0,
            "color_flow_request": contract["color_flow"],
            "helicity_request": contract["helicity"],
            "lc_flow_layout": contract["layout"],
            "gluon_count": 6,
            "validation_samples": 10,
            "point_tile_size": 1024,
            "jit_optimization_level": 3,
            "generation_only": False,
            "allow_diagnostic_incomplete_success": False,
            "modes": ["compiled", "eager", "recurrence"],
            "prepared_model_path": (
                None
                if role.startswith("builtin-")
                else "/runtime/ufo.pyamplicol-model"
            ),
            "model_identities": _models(role),
            "specialize_flow_at_generation": False,
            "external_watchdog_required_for_long_runs": True,
        },
        "profile_schedule": {"test": True},
        "profiles": {
            "compiled": {
                "profiles": [
                    _profile(batch_size, mode="compiled", slow=slow)
                    for batch_size in (1, 128, 1024)
                ]
            },
            "eager": {"profiles": []},
            "recurrence": {
                "profiles": [
                    _profile(batch_size, mode="recurrence")
                    for batch_size in (1, 128, 1024)
                ]
            },
        },
        "validation_summary": {"test": "validation"},
    }


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    manifest = _write(tmp_path / "runtime-bundle.json", _bundle())
    captures = {
        role: _write(tmp_path / f"{role}.json", _capture(role))
        for role in acceptance.CAPTURE_CONTRACTS
    }
    return manifest, captures


def _audit(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    manifest, captures = evidence
    monkeypatch.setattr(acceptance, "_recompute_harness_contracts", lambda _p: None)
    return acceptance.audit(
        capture_paths=captures,
        runtime_bundle_manifest=manifest,
        workflow_run_id=WORKFLOW_RUN_ID,
        expected_current_revision=REVISION,
    )


def test_accepts_exact_four_capture_contract_and_is_portable(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _audit(evidence, monkeypatch)
    assert result["passes"] is True
    assert result["performance_cell_count"] == 8
    assert set(result["captures"]) == set(acceptance.CAPTURE_CONTRACTS)
    assert str(evidence[0].parent) not in json.dumps(result)
    identity = result["content_identity"]
    body = dict(result)
    body.pop("content_identity")
    assert identity["sha256"] == acceptance._canonical_sha256(body)


def test_slow_compiled_capture_fails_performance_gate(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, captures = evidence
    _write(captures["ufo-union"], _capture("ufo-union", slow=True))
    result = _audit((manifest, captures), monkeypatch)
    assert result["passes"] is False
    failed = result["captures"]["ufo-union"]
    assert all(cell["passes"] is False for cell in failed["performance_cells"])


def test_zero_paired_ratio_uncertainty_is_accepted_when_below_ceiling(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, captures = evidence
    payload = _capture("builtin-topology")
    for batch_size in (128, 1024):
        recurrence = acceptance._profile_for_batch(
            payload, mode="recurrence", batch_size=batch_size
        )
        compiled = acceptance._profile_for_batch(
            payload, mode="compiled", batch_size=batch_size
        )
        compiled["subprocess_samples"] = copy.deepcopy(
            recurrence["subprocess_samples"]
        )
        compiled["wall_seconds_per_point"] = recurrence["wall_seconds_per_point"]
        compiled["wall_seconds_per_point_median"] = recurrence[
            "wall_seconds_per_point_median"
        ]
        compiled["wall_seconds_per_point_mad"] = recurrence[
            "wall_seconds_per_point_mad"
        ]
    _write(captures["builtin-topology"], payload)
    result = _audit((manifest, captures), monkeypatch)
    cells = result["captures"]["builtin-topology"]["performance_cells"]
    assert all(cell["ratio_statistics"]["raw_mad"] == 0.0 for cell in cells)
    assert all(cell["ratio_statistics"]["median"] == 1.0 for cell in cells)
    assert all(
        cell["ratio_statistics"]["upper_three_raw_mad"] == 1.0 for cell in cells
    )
    assert all(cell["passes"] is True for cell in cells)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["configuration"].__setitem__(
                "target_runtime_seconds", 4.999
            ),
            "authoritative benchmark policy",
        ),
        (
            lambda payload: payload["configuration"].__setitem__(
                "allow_diagnostic_incomplete_success", True
            ),
            "authoritative benchmark policy",
        ),
        (
            lambda payload: payload["configuration"].__setitem__(
                "specialize_flow_at_generation", True
            ),
            "authoritative benchmark policy",
        ),
        (
            lambda payload: payload["configuration"].__setitem__(
                "helicity_request", "1"
            ),
            "authoritative benchmark policy",
        ),
        (
            lambda payload: payload["runtime_provenance"][
                "native_extension"
            ].__setitem__("sha256", "0" * 64),
            "runtime does not match",
        ),
        (
            lambda payload: payload.__setitem__("provenance", []),
            "not measured from exact clean x86 source",
        ),
    ],
)
def test_rejects_policy_and_runtime_shortcuts(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    manifest, captures = evidence
    payload = _capture("builtin-union")
    mutation(payload)
    _write(captures["builtin-union"], payload)
    with pytest.raises(acceptance.AcceptanceError, match=message):
        _audit((manifest, captures), monkeypatch)


def test_rejects_ufo_model_not_bound_to_bundle(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, captures = evidence
    payload = _capture("ufo-topology")
    payload["configuration"]["model_identities"]["compiled"]["file"]["sha256"] = (
        "0" * 64
    )
    _write(captures["ufo-topology"], payload)
    with pytest.raises(acceptance.AcceptanceError, match="UFO model identity"):
        _audit((manifest, captures), monkeypatch)


def test_rejects_missing_or_duplicate_paired_round() -> None:
    payload = _capture("builtin-topology")
    profile = acceptance._profile_for_batch(
        payload,
        mode="compiled",
        batch_size=128,
    )
    profile["subprocess_samples"][6]["round"] = 5
    with pytest.raises(acceptance.AcceptanceError, match="invalid paired samples"):
        acceptance._paired_cell_evidence(
            payload,
            batch_size=128,
            ceiling=1.15,
        )


def test_runtime_bundle_content_tampering_is_rejected(
    evidence: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, captures = evidence
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["prepared_models"]["ufo-sm"]["size_bytes"] += 1
    _write(manifest, payload)
    with pytest.raises(acceptance.AcceptanceError, match="content identity"):
        _audit((manifest, captures), monkeypatch)


def test_recompute_uses_authoritative_harness_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = {"passes": True}
    capture = {
        "complete": True,
        "evidence_complete": True,
        "passes": True,
        "authoritative_eligible": True,
        "authoritative_ineligibility_reasons": [],
        "generation_specialized_axes_by_mode": {},
        "incomplete_physical_axes": [],
    }
    milestone = {"accepted": False}
    payload = _capture("builtin-topology")
    payload["validation_summary"] = validation
    payload["capture_acceptance"] = capture
    payload["milestone0_acceptance"] = milestone
    monkeypatch.setattr(
        acceptance.benchmark,
        "_pairwise_profile_validation",
        lambda _profiles: validation,
    )
    monkeypatch.setattr(
        acceptance.benchmark,
        "_capture_acceptance",
        lambda *_args, **_kwargs: capture,
    )
    monkeypatch.setattr(
        acceptance.benchmark,
        "_milestone0_acceptance_manifest",
        lambda *_args, **_kwargs: milestone,
    )
    acceptance._recompute_harness_contracts(payload)
    payload["validation_summary"] = {"passes": False}
    with pytest.raises(acceptance.AcceptanceError, match="not reproducible"):
        acceptance._recompute_harness_contracts(payload)
