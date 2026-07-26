# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "qq_z6g_final_comparison.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


comparison = _load("_test_qq_z6g_final_comparison", SCRIPT)


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _raw_sha256(path),
    }


def _matrix_aggregate_fixture(matrix: ModuleType) -> dict[str, Any]:
    baseline = {
        "source_revision": matrix.FROZEN_BASELINE_SOURCE_REVISION,
        "native_build_inputs_sha256": "1" * 64,
        "distribution_content_sha256": "2" * 64,
        "native_module_sha256": "3" * 64,
    }
    current = {
        "source_revision": "4" * 40,
        "native_build_inputs_sha256": "5" * 64,
        "distribution_content_sha256": "6" * 64,
        "native_module_sha256": "7" * 64,
    }
    platform_record = {
        "platform": "macOS-test-arm64",
        "system": "Darwin",
        "machine": "arm64",
        "repository": {
            "clean": True,
            "head_revision": current["source_revision"],
        },
    }
    cells = [
        {
            "cell_id": cell.cell_id,
            "configuration": asdict(cell),
            "result_content_sha256": "8" * 64,
            "errors": [],
            "passes": True,
        }
        for cell in matrix.CANONICAL_CELLS
    ]
    return {
        "kind": matrix.RESULT_KIND,
        "schema_version": matrix.SCHEMA_VERSION,
        "matrix_contract": matrix.MATRIX_CONTRACT,
        "matrix_definition": {
            "sha256": matrix._canonical_sha256(
                [
                    asdict(cell) | {"cell_id": cell.cell_id}
                    for cell in matrix.CANONICAL_CELLS
                ]
            )
        },
        "expected_builds": {"baseline": baseline, "current": current},
        "complete": True,
        "run_complete": True,
        "passes": True,
        "coverage": {
            "expected": 168,
            "observed": 168,
            "missing": [],
            "unexpected": [],
            "passes": True,
        },
        "cell_gate": {"passes": True},
        "identity_gate": {
            "passes": True,
            "distinct_sha256": {"runtime:baseline": ["9" * 64]},
        },
        "gain_gate": {"passes": True},
        "generation_gate": {"passes": True},
        "outer_provenance_gate": {"passes": True},
        "provenance": {
            "preflight": platform_record,
            "postflight": copy.deepcopy(platform_record),
        },
        "cells": cells,
    }


def _primary_result_fixture(
    matrix: ModuleType,
    cell: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    baseline_runtime = {
        "lane": "baseline",
        "native_module": {"sha256": "3" * 64},
    }
    current_runtime = {
        "lane": "current",
        "native_module": {"sha256": "7" * 64},
    }
    measurements: list[dict[str, Any]] = []
    values: dict[str, list[float]] = {"baseline": [], "current": []}
    for pair_index in range(1, 8):
        order = (
            ("baseline", "current")
            if (pair_index - 1) % 2 == 0
            else ("current", "baseline")
        )
        for order_index, lane in enumerate(order, start=1):
            wall = (
                2.0e-6 + pair_index * 1.0e-8
                if lane == "baseline"
                else 1.0e-6 + pair_index * 1.0e-8
            )
            values[lane].append(wall)
            measurements.append(
                {
                    "pair_index": pair_index,
                    "measurement_order": order_index,
                    "lane": lane,
                    "wall_seconds_per_point": wall,
                    "runtime_identity": (
                        baseline_runtime if lane == "baseline" else current_runtime
                    ),
                }
            )

    def distribution(raw: list[float]) -> dict[str, Any]:
        median = statistics.median(raw)
        return {
            "sample_count": len(raw),
            "samples_seconds_per_point": raw,
            "median_seconds_per_point": median,
            "mad_seconds_per_point": statistics.median(
                abs(value - median) for value in raw
            ),
            "minimum_seconds_per_point": min(raw),
            "maximum_seconds_per_point": max(raw),
        }

    distributions = {lane: distribution(raw) for lane, raw in values.items()}
    paired = matrix.regression._paired_distribution(measurements)
    configuration = {
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
        "native_wall_time_source": matrix.regression.NATIVE_WALL_TIME_SOURCE,
        "native_wall_time_sample_pass": (
            matrix.regression.NATIVE_WALL_TIME_SAMPLE_PASS
        ),
        "timing_sample_contract": matrix.regression.PAIRED_TIMING_SAMPLE_CONTRACT,
        "independent_samples_per_lane": 7,
    }
    correctness = {"passes": True, "fixture": "recomputed"}
    result = {
        "kind": matrix.regression.RESULT_KIND,
        "schema_version": matrix.regression.SCHEMA_VERSION,
        "complete": True,
        "performance_result_authoritative": True,
        "passes": True,
        "configuration": configuration,
        "gate": {"passes": True},
        "correctness_gate": correctness,
        "arena_profile_gate": {"passes": True},
        "resource_gate": {"passes": True},
        "measurements": measurements,
        "pair_orders": [
            ["baseline", "current"] if index % 2 == 0 else ["current", "baseline"]
            for index in range(7)
        ],
        "distributions": distributions,
        "paired_distribution": paired,
    }
    baseline_runtime_sha256 = matrix._canonical_sha256(
        matrix._stable_runtime_identity_value(baseline_runtime)
    )
    audit = {
        "configuration": asdict(cell),
        "result_content_sha256": matrix._canonical_sha256(result),
        "runtime_identity_sha256_by_lane": {
            "baseline": baseline_runtime_sha256,
            "current": matrix._canonical_sha256(
                matrix._stable_runtime_identity_value(current_runtime)
            ),
        },
    }
    return result, audit, baseline_runtime_sha256


def _section_evidence() -> Any:
    modes = ("compiled", "eager", "recurrence")
    layouts = ("topology-replay", "all-flow-union")
    source_revision = "a" * 40
    runtime = {
        "interpreter": {
            "implementation": "CPython",
            "python_version": "3.12.6",
            "sha256": "1" * 64,
        },
        "native_extension": {
            "sha256": "2" * 64,
            "build_inputs_sha256": "3" * 64,
        },
        "installed_distribution": {
            "package_version": "0.1.0",
            "distribution_content": {
                "file_count": 10,
                "size_bytes": 1024,
                "sha256": "4" * 64,
            },
        },
        "active_build_info": {
            "payload": {
                "source_revision": source_revision,
                "candidate_fingerprint": "candidate-test",
            }
        },
    }
    captures: dict[tuple[str, str], Any] = {}
    for layout_index, layout in enumerate(layouts):
        profiles: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        for mode_index, mode in enumerate(modes, start=1):
            raw = [
                float(mode_index + layout_index) * 1.0e-6 + index * 1.0e-9
                for index in range(7)
            ]
            median = statistics.median(raw)
            mad = statistics.median(abs(value - median) for value in raw)
            profiles[mode] = {
                "validation": {
                    "passes": True,
                    "maximum_absolute_difference": 1.0e-15,
                    "maximum_relative_difference": 1.0e-16,
                },
                "profiles": [
                    {
                        "batch_size": 1,
                        "subprocess_samples": [
                            {
                                "round": index,
                                "wall_seconds_per_point": value,
                                "interrupted": False,
                            }
                            for index, value in enumerate(raw)
                        ],
                    }
                ],
            }
            timings[mode] = {
                "1": {
                    "sample_count": 7,
                    "median_seconds_per_point": median,
                    "mad_seconds_per_point": mad,
                    "raw_seconds_per_point": raw,
                }
            }
        comparisons = {
            f"{left}__{right}": {
                "passes": True,
                "maximum_absolute_difference": 1.0e-15,
                "maximum_relative_difference": 1.0e-16,
            }
            for left, right in comparison.RATIO_PAIRS
        }
        components = {
            key: {
                "passes": True,
                "compared_component_count": 3,
                "maximum_absolute_difference": 1.0e-15,
                "maximum_relative_difference": 1.0e-16,
            }
            for key in comparisons
        }
        generation = {
            mode: {
                "generation_reused": False,
                "generation_wall_seconds": 0.1 + mode_index * 0.01,
                "artifact_stats": {
                    "file_count": 2,
                    "size_bytes": 2048,
                },
                "peak_rss": {"observed_lower_bound_bytes": 4096},
                "phase_timings_seconds": {
                    "model-loading": 0.01,
                    "emission": 0.02,
                },
                "artifact_semantic_identity_sha256": (str(mode_index + 4) * 64),
            }
            for mode_index, mode in enumerate(modes)
        }
        payload = {
            "profiles": profiles,
            "lane_comparisons": comparisons,
            "validation_summary": {"resolved_component_comparisons": components},
            "generation": generation,
        }
        capture = SimpleNamespace(
            loaded=SimpleNamespace(payload=payload),
            validation_values=(1.0 + 0.0j, 2.0 + 0.0j),
            timings=timings,
            runtime_identity=runtime,
            host={
                "platform": "macOS-test",
                "system": "Darwin",
                "release": "test",
                "machine": "arm64",
                "cpu_model": "test",
                "logical_cpu_count": 8,
            },
            color_axis={"count": 720},
            helicity_axis={"count": 512},
        )
        captures[("built-in-sm", layout)] = capture
        captures[("ufo-sm", layout)] = copy.deepcopy(capture)

    amplicol: dict[str, Any] = {}
    for role_index, role in enumerate(("selected-flow", "all-flow"), start=1):
        raw = [role_index * 5.0e-6 + index * 1.0e-9 for index in range(7)]
        median = statistics.median(raw)
        amplicol[role] = SimpleNamespace(
            values=(1.0 + 0.0j, 2.0 + 0.0j),
            timing={
                "sample_count": 7,
                "median_seconds_per_point": median,
                "mad_seconds_per_point": statistics.median(
                    abs(value - median) for value in raw
                ),
                "raw_seconds_per_point": raw,
                "timing_boundary": (
                    "amplitude-evaluation"
                    if role == "selected-flow"
                    else "direct-library-total"
                ),
            },
            interleave_records=tuple({"round": index} for index in range(7)),
            source_identity={
                "revision": "b" * 40,
                "source_tree_sha256": "c" * 64,
                "compiler": {
                    "id": "gfortran",
                    "version": "14",
                    "target": "arm64-apple-darwin",
                    "flags_sha256": "d" * 64,
                },
            },
        )
    m0 = SimpleNamespace(
        MODELS=("built-in-sm", "ufo-sm"),
        LAYOUTS=layouts,
        MODES=modes,
        BATCHES=(1,),
        MIN_SAMPLES=7,
        AMPLICOL_SELECTED_ROLE="selected-flow",
        AMPLICOL_UNION_ROLE="all-flow",
        ATOL=1.0e-15,
        RTOL=1.0e-12,
        PROCESS="u u~ > z g g g g g g",
    )
    return SimpleNamespace(
        m0=m0,
        captures=captures,
        amplicol=amplicol,
        expected={
            "pyamplicol_source_revision": source_revision,
            "runtime_provenance_sha256": "e" * 64,
        },
    )


def test_default_output_and_parser_require_all_evidence_pins() -> None:
    assert Path("docs/QQ_Z6G_ARENA_COMPARISON.md") == comparison.DEFAULT_OUTPUT
    option_destinations = {
        action.dest for action in comparison.parser()._actions if action.required
    }
    assert {
        "request",
        "request_sha256",
        "acceptance",
        "acceptance_sha256",
        "expected_source_revision",
        "expected_runtime_provenance_sha256",
        "pre_arena_manifest",
        "pre_arena_manifest_sha256",
        "expected_pre_arena_source_revision",
        "expected_pre_arena_build_sha256",
        "expected_pre_arena_runtime_sha256",
    } <= option_destinations


def test_decision_semantics_rejects_unknown_and_stale_content() -> None:
    recomputed = {
        "accepted": True,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "status": "accepted",
    }
    body = copy.deepcopy(recomputed)
    acceptance = body | {"content_sha256": comparison._canonical_sha256(body)}
    fake_m0 = SimpleNamespace(_utc=lambda value, label: value)

    assert comparison._decision_semantics(
        acceptance=acceptance,
        recomputed=recomputed,
        m0=fake_m0,
    ) == {"accepted": True, "status": "accepted"}

    unknown = acceptance | {"unexpected": True}
    with pytest.raises(comparison.ComparisonError, match="unknown or missing"):
        comparison._decision_semantics(
            acceptance=unknown,
            recomputed=recomputed,
            m0=fake_m0,
        )
    stale = acceptance | {"accepted": False}
    with pytest.raises(comparison.ComparisonError, match="digest is stale"):
        comparison._decision_semantics(
            acceptance=stale,
            recomputed=recomputed,
            m0=fake_m0,
        )


def test_validate_evidence_rejects_wrong_final_source_and_runtime_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "kind": "request",
        "schema_version": 1,
        "expected": {
            "pyamplicol_source_revision": "a" * 40,
            "runtime_provenance_sha256": "b" * 64,
        },
    }
    fake_m0 = SimpleNamespace(
        _REQUEST_KEYS=set(request),
        REQUEST_KIND="request",
        REQUEST_SCHEMA=1,
        _require_exact_keys=lambda value, keys, label: None,
        _validate_expected=lambda value: value,
    )
    loaded = SimpleNamespace(
        payload=request,
        ref=SimpleNamespace(path=tmp_path / "request.json"),
        canonical_sha256="c" * 64,
    )
    monkeypatch.setattr(comparison, "_load_m0", lambda: fake_m0)
    monkeypatch.setattr(comparison, "_load_json_ref", lambda *args, **kwargs: loaded)
    common = {
        "request_path": tmp_path / "request.json",
        "request_sha256": "d" * 64,
        "acceptance_path": tmp_path / "acceptance.json",
        "acceptance_sha256": "e" * 64,
        "expected_source_revision": "a" * 40,
        "expected_runtime_provenance_sha256": "b" * 64,
        "pre_arena_manifest_path": tmp_path / "pre-arena.json",
        "pre_arena_manifest_sha256": "f" * 64,
        "expected_pre_arena_source_revision": "1" * 40,
        "expected_pre_arena_build_sha256": "2" * 64,
        "expected_pre_arena_runtime_sha256": "3" * 64,
    }

    with pytest.raises(comparison.ComparisonError, match="final source"):
        comparison.validate_evidence(
            **(common | {"expected_source_revision": "4" * 40})
        )
    with pytest.raises(comparison.ComparisonError, match="runtime identity"):
        comparison.validate_evidence(
            **(common | {"expected_runtime_provenance_sha256": "5" * 64})
        )


def test_manifest_reference_is_content_addressed_and_exact(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"value":1}\n', encoding="utf-8")
    m0 = comparison._load_m0()
    reference = {
        "path": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": _raw_sha256(target),
    }
    loaded = comparison._manifest_ref(
        m0=m0,
        value=reference,
        base=tmp_path,
        label="fixture",
    )
    assert loaded.payload == {"value": 1}

    with pytest.raises(comparison.ComparisonError, match="unknown"):
        comparison._manifest_ref(
            m0=m0,
            value=reference | {"canonical_sha256": "0" * 64},
            base=tmp_path,
            label="fixture",
        )
    with pytest.raises(comparison.ComparisonError, match="content hash drifted"):
        comparison._manifest_ref(
            m0=m0,
            value=reference | {"sha256": "0" * 64},
            base=tmp_path,
            label="fixture",
        )


def test_matrix_aggregate_requires_exact_aarch64_identity_pins() -> None:
    matrix = comparison._load_matrix()
    payload = _matrix_aggregate_fixture(matrix)
    baseline = payload["expected_builds"]["baseline"]
    assert isinstance(baseline, dict)
    build_sha256 = comparison._canonical_sha256(baseline)
    loaded = SimpleNamespace(payload=payload)

    audits, observed_baseline, current, platform_record = (
        comparison._validate_matrix_aggregate(
            matrix=matrix,
            loaded=loaded,
            expected_pre_arena_source_revision=(matrix.FROZEN_BASELINE_SOURCE_REVISION),
            expected_pre_arena_build_sha256=build_sha256,
            expected_pre_arena_runtime_sha256="9" * 64,
        )
    )
    assert len(audits) == 168
    assert observed_baseline == baseline
    assert current["source_revision"] == "4" * 40
    assert platform_record["machine"] == "arm64"

    tampered = copy.deepcopy(payload)
    tampered["provenance"]["postflight"]["machine"] = "x86_64"
    with pytest.raises(comparison.ComparisonError, match="changed"):
        comparison._validate_matrix_aggregate(
            matrix=matrix,
            loaded=SimpleNamespace(payload=tampered),
            expected_pre_arena_source_revision=(matrix.FROZEN_BASELINE_SOURCE_REVISION),
            expected_pre_arena_build_sha256=build_sha256,
            expected_pre_arena_runtime_sha256="9" * 64,
        )
    for overrides, message in (
        (
            {"expected_pre_arena_source_revision": "a" * 40},
            "frozen pre-Arena source",
        ),
        (
            {"expected_pre_arena_build_sha256": "a" * 64},
            "build identity",
        ),
        (
            {"expected_pre_arena_runtime_sha256": "a" * 64},
            "runtime identity",
        ),
    ):
        arguments = {
            "matrix": matrix,
            "loaded": loaded,
            "expected_pre_arena_source_revision": (
                matrix.FROZEN_BASELINE_SOURCE_REVISION
            ),
            "expected_pre_arena_build_sha256": build_sha256,
            "expected_pre_arena_runtime_sha256": "9" * 64,
        }
        arguments.update(overrides)
        with pytest.raises(comparison.ComparisonError, match=message):
            comparison._validate_matrix_aggregate(**arguments)


def test_matrix_current_build_is_identical_to_final_m0_runtime() -> None:
    source_revision = "a" * 40
    current = {
        "source_revision": source_revision,
        "native_build_inputs_sha256": "1" * 64,
        "distribution_content_sha256": "2" * 64,
        "native_module_sha256": "3" * 64,
    }
    pre_arena = SimpleNamespace(current_build=current)
    capture = SimpleNamespace(
        runtime_identity={
            "native_extension": {
                "build_inputs_sha256": "1" * 64,
                "sha256": "3" * 64,
            },
            "installed_distribution": {"distribution_content": {"sha256": "2" * 64}},
        }
    )
    comparison._bind_matrix_current_to_final(
        pre_arena=pre_arena,
        final_capture=capture,
        expected_source_revision=source_revision,
    )

    mismatched = SimpleNamespace(current_build=current | {"source_revision": "b" * 40})
    with pytest.raises(comparison.ComparisonError, match="does not equal"):
        comparison._bind_matrix_current_to_final(
            pre_arena=mismatched,
            final_capture=capture,
            expected_source_revision=source_revision,
        )


def test_validate_pre_arena_evidence_authenticates_all_24_primary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = comparison._load_matrix()
    monkeypatch.setattr(comparison, "_load_matrix", lambda: matrix)
    monkeypatch.setattr(
        matrix.regression,
        "_correctness_gate",
        lambda measurements: {"passes": True, "fixture": "recomputed"},
    )
    aggregate = _matrix_aggregate_fixture(matrix)
    primary_cells = {
        cell.cell_id: cell
        for cell in matrix.CANONICAL_CELLS
        if cell.category == "primary"
    }
    audit_by_id = {record["cell_id"]: record for record in aggregate["cells"]}
    refs: list[dict[str, Any]] = []
    runtime_sha256: str | None = None
    first_result_path: Path | None = None
    for cell_id, cell in sorted(primary_cells.items()):
        result, audit, observed_runtime_sha256 = _primary_result_fixture(
            matrix,
            cell,
        )
        if runtime_sha256 is None:
            runtime_sha256 = observed_runtime_sha256
        assert runtime_sha256 == observed_runtime_sha256
        audit_by_id[cell_id].update(audit)
        path = tmp_path / "primary" / cell_id / "result.json"
        identity = _write_json(path, result)
        refs.append({"cell_id": cell_id, **identity})
        first_result_path = first_result_path or path
    assert runtime_sha256 is not None
    aggregate["identity_gate"]["distinct_sha256"]["runtime:baseline"] = [runtime_sha256]
    aggregate_identity = _write_json(
        tmp_path / "matrix-result.json",
        aggregate,
    )
    manifest = {
        "kind": comparison.PRE_ARENA_REQUEST_KIND,
        "schema_version": comparison.PRE_ARENA_REQUEST_SCHEMA,
        "matrix_aggregate": aggregate_identity,
        "primary_results": refs,
    }
    manifest_path = tmp_path / "pre-arena-request.json"
    manifest_identity = _write_json(manifest_path, manifest)
    baseline = aggregate["expected_builds"]["baseline"]
    assert isinstance(baseline, dict)
    m0 = comparison._load_m0()

    evidence = comparison.validate_pre_arena_evidence(
        m0=m0,
        manifest_path=manifest_path,
        manifest_sha256=manifest_identity["sha256"],
        expected_pre_arena_source_revision=(matrix.FROZEN_BASELINE_SOURCE_REVISION),
        expected_pre_arena_build_sha256=comparison._canonical_sha256(baseline),
        expected_pre_arena_runtime_sha256=runtime_sha256,
    )
    assert len(evidence.results) == 24
    assert len(evidence.timings) == 24

    unknown_manifest = manifest | {"unexpected": True}
    unknown_path = tmp_path / "unknown-request.json"
    unknown_identity = _write_json(unknown_path, unknown_manifest)
    with pytest.raises(comparison.ComparisonError, match="unknown or missing"):
        comparison.validate_pre_arena_evidence(
            m0=m0,
            manifest_path=unknown_path,
            manifest_sha256=unknown_identity["sha256"],
            expected_pre_arena_source_revision=(matrix.FROZEN_BASELINE_SOURCE_REVISION),
            expected_pre_arena_build_sha256=comparison._canonical_sha256(baseline),
            expected_pre_arena_runtime_sha256=runtime_sha256,
        )
    missing_manifest = {
        key: value for key, value in manifest.items() if key != "primary_results"
    }
    missing_path = tmp_path / "missing-request.json"
    missing_identity = _write_json(missing_path, missing_manifest)
    with pytest.raises(comparison.ComparisonError, match="unknown or missing"):
        comparison.validate_pre_arena_evidence(
            m0=m0,
            manifest_path=missing_path,
            manifest_sha256=missing_identity["sha256"],
            expected_pre_arena_source_revision=(matrix.FROZEN_BASELINE_SOURCE_REVISION),
            expected_pre_arena_build_sha256=comparison._canonical_sha256(baseline),
            expected_pre_arena_runtime_sha256=runtime_sha256,
        )

    assert first_result_path is not None
    first_result_path.write_text(
        first_result_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(comparison.ComparisonError, match="drifted"):
        comparison.validate_pre_arena_evidence(
            m0=m0,
            manifest_path=manifest_path,
            manifest_sha256=manifest_identity["sha256"],
            expected_pre_arena_source_revision=(matrix.FROZEN_BASELINE_SOURCE_REVISION),
            expected_pre_arena_build_sha256=comparison._canonical_sha256(baseline),
            expected_pre_arena_runtime_sha256=runtime_sha256,
        )


def test_primary_raw_medians_and_correctness_are_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = comparison._load_matrix()
    cell = next(cell for cell in matrix.CANONICAL_CELLS if cell.category == "primary")
    result, audit, runtime_sha256 = _primary_result_fixture(matrix, cell)
    monkeypatch.setattr(
        matrix.regression,
        "_correctness_gate",
        lambda measurements: {"passes": True, "fixture": "recomputed"},
    )

    distributions = comparison._matrix_result_distributions(
        matrix=matrix,
        cell=cell,
        result=result,
        audit=audit,
        baseline_runtime_sha256=runtime_sha256,
    )
    assert (
        distributions["baseline"]["median_seconds_per_point"]
        > (distributions["current"]["median_seconds_per_point"])
    )

    stale = copy.deepcopy(result)
    stale["distributions"]["baseline"]["median_seconds_per_point"] *= 2.0
    stale_audit = copy.deepcopy(audit)
    stale_audit["result_content_sha256"] = matrix._canonical_sha256(stale)
    with pytest.raises(comparison.ComparisonError, match="distributions are stale"):
        comparison._matrix_result_distributions(
            matrix=matrix,
            cell=cell,
            result=stale,
            audit=stale_audit,
            baseline_runtime_sha256=runtime_sha256,
        )


def test_runtime_numerical_generation_and_amplicol_sections_use_strict_shapes() -> None:
    evidence = _section_evidence()

    runtime_rows = comparison._runtime_identity_rows(evidence)
    numerical = "\n".join(comparison._numerical_sections(evidence))
    generation = "\n".join(comparison._generation_sections(evidence))
    runtime = "\n".join(comparison._runtime_sections(evidence))
    original_rows = comparison._original_identity_rows(evidence)
    amplicol = "\n".join(comparison._amplicol_sections(evidence))

    assert any(row[0] == "Native extension SHA-256" for row in runtime_rows)
    assert "Selected-total versus resolved-sum closure" in numerical
    assert "Cross-lane resolved-component agreement" in numerical
    assert "Generation peak RSS lower bound" in generation
    assert "Cold artifact load time | Unavailable" in generation
    assert "Same-round runtime ratios" in runtime
    assert any(row[0] == "Original AmpliCol revision" for row in original_rows)
    assert "Same-round original selected/union ratio" in amplicol

    broken = copy.deepcopy(
        evidence.captures[("built-in-sm", "topology-replay")].loaded.payload
    )
    broken["generation"]["compiled"]["artifact_stats"]["size_bytes"] = None
    evidence.captures[("built-in-sm", "topology-replay")].loaded.payload = broken
    with pytest.raises(comparison.ComparisonError, match="artifact size"):
        comparison._generation_sections(evidence)


def test_generate_is_atomic_and_cli_failure_leaves_no_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        comparison,
        "validate_evidence",
        lambda **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        comparison,
        "render_markdown",
        lambda evidence: "# deterministic\n",
    )
    output = tmp_path / "result.md"
    arguments = {
        "request_path": tmp_path / "request.json",
        "request_sha256": "1" * 64,
        "acceptance_path": tmp_path / "acceptance.json",
        "acceptance_sha256": "2" * 64,
        "expected_source_revision": "3" * 40,
        "expected_runtime_provenance_sha256": "4" * 64,
        "pre_arena_manifest_path": tmp_path / "pre-arena.json",
        "pre_arena_manifest_sha256": "5" * 64,
        "expected_pre_arena_source_revision": "6" * 40,
        "expected_pre_arena_build_sha256": "7" * 64,
        "expected_pre_arena_runtime_sha256": "8" * 64,
        "output_path": output,
    }
    digest = comparison.generate(**arguments)
    assert output.read_text(encoding="utf-8") == "# deterministic\n"
    assert digest == hashlib.sha256(b"# deterministic\n").hexdigest()
    assert not list(tmp_path.glob(".result.md.*.tmp"))

    output.write_text("complete previous output\n", encoding="utf-8")
    monkeypatch.setattr(
        comparison,
        "validate_evidence",
        lambda **kwargs: comparison._die("rejected fixture"),
    )
    cli = [
        "--request",
        str(arguments["request_path"]),
        "--request-sha256",
        str(arguments["request_sha256"]),
        "--acceptance",
        str(arguments["acceptance_path"]),
        "--acceptance-sha256",
        str(arguments["acceptance_sha256"]),
        "--expected-source-revision",
        str(arguments["expected_source_revision"]),
        "--expected-runtime-provenance-sha256",
        str(arguments["expected_runtime_provenance_sha256"]),
        "--pre-arena-manifest",
        str(arguments["pre_arena_manifest_path"]),
        "--pre-arena-manifest-sha256",
        str(arguments["pre_arena_manifest_sha256"]),
        "--expected-pre-arena-source-revision",
        str(arguments["expected_pre_arena_source_revision"]),
        "--expected-pre-arena-build-sha256",
        str(arguments["expected_pre_arena_build_sha256"]),
        "--expected-pre-arena-runtime-sha256",
        str(arguments["expected_pre_arena_runtime_sha256"]),
        "--output",
        str(output),
    ]
    assert comparison.main(cli) == 2
    assert output.read_text(encoding="utf-8") == "complete previous output\n"
    assert "qq_Z6g comparison rejected" in capsys.readouterr().err


def test_pre_arena_table_and_full_render_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timing = {
        "median_seconds_per_point": 1.0e-6,
        "mad_seconds_per_point": 1.0e-8,
        "sample_count": 7,
        "raw_seconds_per_point": [1.0e-6] * 7,
    }
    capture = SimpleNamespace(
        host={
            "platform": "macOS-test",
            "system": "Darwin",
            "release": "test",
            "machine": "arm64",
            "cpu_model": "test",
            "logical_cpu_count": 8,
        },
        color_axis={"count": 720},
        helicity_axis={"count": 512},
        timings={
            "compiled": {"1": timing},
            "eager": {"1": timing | {"median_seconds_per_point": 2.0e-6}},
        },
    )
    pre_arena_timings = {
        ("built-in-sm", "topology-replay", "compiled", 1): {
            "baseline": {
                "median_seconds_per_point": 2.0e-6,
                "mad_seconds_per_point": 2.0e-8,
            },
            "current": timing,
        },
        ("built-in-sm", "topology-replay", "eager", 1): {
            "baseline": {
                "median_seconds_per_point": 4.0e-6,
                "mad_seconds_per_point": 4.0e-8,
            },
            "current": timing | {"median_seconds_per_point": 2.0e-6},
        },
    }
    m0 = SimpleNamespace(
        MODELS=("built-in-sm",),
        LAYOUTS=("topology-replay",),
        MODES=("compiled", "eager", "recurrence"),
        BATCHES=(1,),
        PROCESS="u u~ > z g g g g g g",
        RTOL=1.0e-11,
        ATOL=1.0e-300,
    )
    evidence = SimpleNamespace(
        m0=m0,
        captures={("built-in-sm", "topology-replay"): capture},
        expected={
            "color_flow": {"id": "flow:test"},
            "helicity": {"id": "h:test"},
            "model_common_physics_identity_sha256": {"built-in-sm": "a" * 64},
            "generation_model_identities_sha256": {"built-in-sm": "b" * 64},
        },
        pre_arena=SimpleNamespace(timings=pre_arena_timings),
    )
    baseline_section = comparison._pre_arena_comparison_sections(evidence)
    assert "Final Arena / pre-Arena" in "\n".join(baseline_section)
    assert "0.5x" in "\n".join(baseline_section)

    monkeypatch.setattr(
        comparison, "_capture_input_rows", lambda evidence: [["x", "y", "z"]]
    )
    monkeypatch.setattr(
        comparison, "_runtime_identity_rows", lambda evidence: [["x", "y"]]
    )
    monkeypatch.setattr(
        comparison, "_pre_arena_identity_rows", lambda evidence: [["x", "y"]]
    )
    monkeypatch.setattr(
        comparison, "_original_identity_rows", lambda evidence: [["x", "y"]]
    )
    monkeypatch.setattr(
        comparison, "_numerical_sections", lambda evidence: ["## Numeric"]
    )
    monkeypatch.setattr(
        comparison, "_generation_sections", lambda evidence: ["## Generation"]
    )
    monkeypatch.setattr(
        comparison, "_runtime_sections", lambda evidence: ["## Runtime"]
    )
    monkeypatch.setattr(
        comparison,
        "_pre_arena_comparison_sections",
        lambda evidence: baseline_section,
    )
    monkeypatch.setattr(
        comparison, "_amplicol_sections", lambda evidence: ["## AmpliCol"]
    )
    first = comparison.render_markdown(evidence)
    second = comparison.render_markdown(evidence)
    assert first == second
    assert "# Final qq → Z + 6g Arena comparison" in first
    assert "**The all-flow union evidence here is LC-only.**" in first
    assert (
        hashlib.sha256(first.encode()).digest()
        == hashlib.sha256(second.encode()).digest()
    )
