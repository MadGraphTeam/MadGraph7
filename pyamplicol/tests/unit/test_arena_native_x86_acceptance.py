# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tools.developer import arena_native_x86_acceptance as acceptance

REVISION = "1" * 40
WORKSPACE = acceptance.ROOT
SHA256 = "2" * 64


def _source() -> dict[str, object]:
    return {
        "checkout": str(WORKSPACE),
        "revision": REVISION,
        "dirty": False,
        "untracked_files_checked": True,
    }


def _runtime() -> dict[str, object]:
    build_info = {"source_revision": REVISION}
    return {
        "interpreter": {
            "path": "/python",
            "size_bytes": 10,
            "sha256": "3" * 64,
            "python_version": "3.11.9",
            "implementation": "CPython",
        },
        "package": {"version": "0.1.0"},
        "python_package_tree": {
            "kind": "pyamplicol-python-package-tree-v2",
            "sha256": "4" * 64,
        },
        "native_extension": {
            "path": "/pyamplicol/_rusticol.so",
            "size_bytes": 20,
            "sha256": "5" * 64,
            "build_inputs_sha256": "6" * 64,
        },
        "active_build_info": {
            "payload": build_info,
            "canonical_sha256": acceptance._canonical_sha256(build_info),
        },
        "loaded_module_origin_policy": {
            "kind": "pyamplicol-loaded-module-origin-policy-v1",
            "all_loaded_origins_authenticated": True,
            "observed_module_count": 1,
            "observations": [{"module": "pyamplicol"}],
            "observations_sha256": SHA256,
        },
    }


def _gate_common(kind: str) -> dict[str, Any]:
    source = _source()
    runtime = _runtime()
    script = {"path": "/gate.py", "size_bytes": 1, "sha256": SHA256}
    return {
        "kind": kind,
        "schema_version": 1,
        "status": "ok",
        "passes": True,
        "source_identity": source,
        "source_identity_postflight": copy.deepcopy(source),
        "source_identity_match": True,
        "runtime_identity": runtime,
        "runtime_identity_postflight": copy.deepcopy(runtime),
        "runtime_identity_match": True,
        "gate_script_identity": script,
        "gate_script_identity_postflight": copy.deepcopy(script),
        "gate_script_identity_match": True,
        "failures": [],
    }


def _comparison() -> dict[str, object]:
    return {
        "value_count": 2,
        "failing_value_count": 0,
        "passes": True,
    }


def _comparison_mapping(names: set[str]) -> dict[str, object]:
    return {name: _comparison() for name in names}


def _all_jit_payload() -> dict[str, Any]:
    payload = _gate_common(acceptance.ALL_JIT_KIND)
    payload["request"] = {
        "process": "g g > g g",
        "jit_optimization_levels": [0, 1, 2, 3],
        "point_count": 3,
        "point_seeds": list(acceptance._ALL_JIT_POINT_SEEDS),
        "momenta_identity": {
            "algorithm": "sha256-float-hex-momenta-v1",
            "sha256": SHA256,
            "point_count": 3,
            "external_particle_count": 4,
        },
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-15,
        "mandatory_watchdog_command": "memory-watchdog -- gate",
        "memory_limit_gib": 30,
    }
    levels = {}
    for name, level in acceptance._ALL_JIT_LEVELS.items():
        levels[name] = {
            "configuration": {
                "process": "g g > g g",
                "color_accuracy": "lc",
                "lc_flow_layout": "topology-replay",
                "backend": "jit",
                "execution_mode": "compiled",
                "jit_optimization_level": level,
            },
            "generation": {
                "worker_provenance": {
                    "source_revision": REVISION,
                    "postflight_identity_match": True,
                }
            },
            "artifact_identity": {
                "payload_count": 1,
                "direct_arena_audit": {"passes": True},
            },
            "numerical_validation": {
                "execution_mode": "compiled",
                "point_count": 3,
                "point_seeds": list(acceptance._ALL_JIT_POINT_SEEDS),
                "comparisons": _comparison_mapping(acceptance._ALL_JIT_COMPARISONS),
                "native_profile": {
                    "passes": True,
                    "execution_mode": "compiled",
                    "direct_arena_engine_counter": {"value": 1},
                    "direct_arena_call_counter": {"value": 2},
                },
                "passes": True,
            },
            "passes": True,
        }
    payload["levels"] = levels
    payload["cross_level_numerical_parity"] = {
        "baseline_optimization_level": 3,
        "comparisons": {
            name: _comparison_mapping(acceptance._CROSS_LEVEL_COMPARISONS)
            for name in ("o0_vs_o3", "o1_vs_o3", "o2_vs_o3")
        },
        "passes": True,
    }
    return acceptance._attach_content_identity(payload)


def _four_quark_payload() -> dict[str, Any]:
    payload = _gate_common(acceptance.FOUR_QUARK_KIND)
    payload["request"] = {
        "process": "d d~ > u u~ s s~ c c~",
        "independent_quark_line_count": 4,
        "expected_external_pdgs": [-4, -3, -2, -1, 1, 2, 3, 4],
        "max_quark_lines": 4,
        "point_count": 3,
        "point_seeds": list(acceptance._FOUR_QUARK_POINT_SEEDS),
        "momenta_identity": {
            "algorithm": "sha256-float-hex-momenta-v1",
            "sha256": SHA256,
            "point_count": 3,
            "external_particle_count": 8,
        },
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-300,
        "mandatory_watchdog_command": "memory-watchdog -- gate",
        "memory_limit_gib": 30,
    }
    payload["invalid_union_configurations"] = [
        {
            "color_accuracy": color,
            "lc_flow_layout": "all-flow-union",
            "rejected": True,
            "error_type": "ConfigurationError",
            "message": f"all-flow-union requires LC, not {color}",
        }
        for color in ("nlc", "full")
    ]
    lanes = {}
    for name, (color, layout, contracted) in acceptance._FOUR_QUARK_LANES.items():
        comparisons = set(acceptance._FOUR_QUARK_BASE_COMPARISONS)
        profiles = {"complete", "helicity_selector"}
        if not contracted:
            comparisons.add("color_selector_evaluate_vs_resolved_total")
            profiles.update({"color_selector", "combined_selector"})
        lanes[name] = {
            "configuration": {
                "process": "d d~ > u u~ s s~ c c~",
                "max_quark_lines": 4,
                "color_accuracy": color,
                "lc_flow_layout": layout,
                "contracted_color": contracted,
                "backend": "jit",
                "execution_mode": "compiled",
                "jit_optimization_level": 3,
            },
            "generation": {
                "worker_provenance": {
                    "source_revision": REVISION,
                    "postflight_identity_match": True,
                }
            },
            "artifact_identity": {
                "payload_count": 1,
                "direct_arena_audit": {"passes": True},
            },
            "numerical_validation": {
                "point_count": 3,
                "point_seeds": list(acceptance._FOUR_QUARK_POINT_SEEDS),
                "comparisons": _comparison_mapping(comparisons),
                "runtime_execution": {
                    "execution_mode": "compiled",
                    "direct_arena_profiles": {
                        profile: {
                            "compiled_direct_arena_engine_count": 1,
                            "compiled_direct_arena_call_count": 2,
                            "legacy_boundary_component_total": 0,
                            "passes": True,
                        }
                        for profile in profiles
                    },
                },
                "passes": True,
            },
            "passes": True,
        }
    payload["lanes"] = lanes
    payload["lc_cross_layout_parity"] = {
        "physical_axes_match": True,
        "comparisons": _comparison_mapping(acceptance._CROSS_LC_COMPARISONS),
        "passes": True,
    }
    return acceptance._attach_content_identity(payload)


def _resolved() -> dict[str, object]:
    return {
        "helicity_ids_match": True,
        "color_ids_match": True,
        "shape_matches": True,
        "point_count": 2,
        "component_count": 4,
        "passes": True,
    }


def _correctness(*, specialized: bool) -> dict[str, Any]:
    result = {
        "total": [{"passes": True}],
        "resolved_f64": _resolved(),
        "resolved_precision32": _resolved(),
        "eager_resolved_sum": [{"passes": True}],
        "compiled_resolved_sum": [{"passes": True}],
        "passes": True,
    }
    if specialized:
        result["specialized_compiled"] = _correctness(specialized=False)
    return result


def _color_matrix_payload() -> dict[str, Any]:
    records = []
    for case_key, (process, n_final) in acceptance._SMOKE_CASES.items():
        for color in acceptance._COLORS:
            lc = color == "lc"
            workload_names = (
                ("single-flow-helicity-sum", "all-flow-single-helicity")
                if lc
                else ("summed",)
            )
            workloads = []
            for name in workload_names:
                modes = {"compiled_complete", "eager_complete"}
                if lc:
                    modes.add("compiled_specialized")
                workloads.append(
                    {
                        "name": name,
                        "correctness": _correctness(specialized=lc),
                        "profiles": [
                            {
                                "batch_size": batch_size,
                                **{
                                    mode: {
                                        "result": {
                                            "sample_count": 5,
                                            "wall_time_per_point": 1.0e-6,
                                            "interrupted": False,
                                            "effective_config": {
                                                "batch_size": batch_size
                                            },
                                        }
                                    }
                                    for mode in modes
                                },
                            }
                            for batch_size in (128, 1024)
                        ],
                    }
                )
            records.append(
                {
                    "case": {
                        "key": case_key,
                        "process": process,
                        "n_final": n_final,
                        "smoke": True,
                    },
                    "model": "built-in",
                    "color": color,
                    "process_id": f"{case_key}-{color}",
                    "compiled_generation_under_hard_limit": True,
                    "generation": {
                        mode: {"core_phase_seconds": 1.0}
                        for mode in ("compiled", "eager")
                    },
                    "workloads": workloads,
                    "selector_pattern_profiles": (
                        {
                            mode: {"result": {"passes": True}}
                            for mode in ("compiled", "eager")
                        }
                        if lc
                        else None
                    ),
                }
            )
    return {
        "kind": acceptance.COLOR_MATRIX_KIND,
        "schema_version": acceptance.COLOR_MATRIX_SCHEMA_VERSION,
        "complete": True,
        "source_revision": REVISION,
        "suite": "smoke",
        "configuration": {
            "batch_sizes": [128, 1024],
            "colors": ["lc", "nlc", "full"],
            "generation_timeout": 300.0,
            "memory_limit_gib": 30.0,
            "minimum_samples": 5,
            "target_runtime": 5.0,
            "selector_target_runtime": 1.0,
            "selector_seed": 0xC0FFEE,
        },
        "gates": {name: True for name in acceptance._COLOR_MATRIX_GATES},
        "passes": True,
        "records": records,
    }


def _reattach(payload: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(payload)
    body.pop("content_identity", None)
    return acceptance._attach_content_identity(body)


def test_acceptance_validators_accept_complete_synthetic_evidence() -> None:
    assert acceptance._validate_all_jit(
        _all_jit_payload(),
        expected_revision=REVISION,
        expected_workspace=WORKSPACE,
        point_count=3,
    )["numerical_validation"]
    assert acceptance._validate_four_quark(
        _four_quark_payload(),
        expected_revision=REVISION,
        expected_workspace=WORKSPACE,
        point_count=3,
    )["numerical_validation"]
    color = acceptance._validate_color_matrix(
        _color_matrix_payload(),
        expected_revision=REVISION,
    )
    assert color["cell_count"] == 6
    assert color["workload_count"] == 8


def test_all_jit_rejects_tampering_and_vacuous_numerical_comparisons() -> None:
    tampered = _all_jit_payload()
    tampered["status"] = "validation_failed"
    with pytest.raises(acceptance.AcceptanceError, match="content identity"):
        acceptance._validate_all_jit(
            tampered,
            expected_revision=REVISION,
            expected_workspace=WORKSPACE,
            point_count=3,
        )

    empty = _all_jit_payload()
    comparison = empty["levels"]["jit-o2"]["numerical_validation"]["comparisons"][
        "f64_vs_precision32_total"
    ]
    comparison["value_count"] = 0
    with pytest.raises(acceptance.AcceptanceError, match="comparison is empty"):
        acceptance._validate_all_jit(
            _reattach(empty),
            expected_revision=REVISION,
            expected_workspace=WORKSPACE,
            point_count=3,
        )


def test_gate_runtime_must_match_the_preflight_candidate_bytes() -> None:
    expected_runtime = _runtime()
    package_tree = expected_runtime.pop("python_package_tree")
    expected_runtime["preimport_runtime_identity"] = {
        "python_package_tree": package_tree
    }
    acceptance._validate_all_jit(
        _all_jit_payload(),
        expected_revision=REVISION,
        expected_workspace=WORKSPACE,
        point_count=3,
        expected_runtime=expected_runtime,
    )

    expected_runtime["native_extension"]["sha256"] = "7" * 64
    with pytest.raises(acceptance.AcceptanceError, match="preflight candidate runtime"):
        acceptance._validate_all_jit(
            _all_jit_payload(),
            expected_revision=REVISION,
            expected_workspace=WORKSPACE,
            point_count=3,
            expected_runtime=expected_runtime,
        )


def test_four_quark_rejects_missing_lane_and_invalid_union_coverage() -> None:
    missing_lane = _four_quark_payload()
    del missing_lane["lanes"]["lc-all-flow-union"]
    with pytest.raises(acceptance.AcceptanceError, match="lane coverage"):
        acceptance._validate_four_quark(
            _reattach(missing_lane),
            expected_revision=REVISION,
            expected_workspace=WORKSPACE,
            point_count=3,
        )

    missing_rejection = _four_quark_payload()
    missing_rejection["invalid_union_configurations"].pop()
    with pytest.raises(acceptance.AcceptanceError, match="reject both"):
        acceptance._validate_four_quark(
            _reattach(missing_rejection),
            expected_revision=REVISION,
            expected_workspace=WORKSPACE,
            point_count=3,
        )


def test_color_matrix_rejects_partial_scope_and_precision32_failure() -> None:
    partial = _color_matrix_payload()
    partial["records"].pop()
    with pytest.raises(acceptance.AcceptanceError, match="cell scope"):
        acceptance._validate_color_matrix(partial, expected_revision=REVISION)

    failed = _color_matrix_payload()
    failed["records"][0]["workloads"][0]["correctness"]["resolved_precision32"][
        "passes"
    ] = False
    with pytest.raises(acceptance.AcceptanceError, match="resolved_precision32"):
        acceptance._validate_color_matrix(failed, expected_revision=REVISION)


def test_color_matrix_rejects_vacuous_reduction_and_false_hard_gate() -> None:
    vacuous = _color_matrix_payload()
    vacuous["records"][0]["workloads"][0]["correctness"]["eager_resolved_sum"] = []
    with pytest.raises(acceptance.AcceptanceError, match="must not be empty"):
        acceptance._validate_color_matrix(vacuous, expected_revision=REVISION)

    failed_gate = _color_matrix_payload()
    failed_gate["gates"]["correctness"] = False
    with pytest.raises(acceptance.AcceptanceError, match=r"gates\.correctness"):
        acceptance._validate_color_matrix(failed_gate, expected_revision=REVISION)


def test_checked_json_binds_exact_bytes_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"passes":true}\n', encoding="utf-8")
    payload, identity = acceptance._checked_json(evidence, label="fixture")
    assert payload == {"passes": True}
    assert (
        identity["sha256"]
        == acceptance.hashlib.sha256(evidence.read_bytes()).hexdigest()
    )

    link = tmp_path / "link.json"
    link.symlink_to(evidence)
    with pytest.raises(acceptance.AcceptanceError, match="regular file"):
        acceptance._checked_json(link, label="fixture link")


def test_content_identity_covers_the_entire_manifest_body() -> None:
    payload = acceptance._attach_content_identity(
        {
            "kind": acceptance.ACCEPTANCE_KIND,
            "schema_version": 1,
            "evidence": {
                "all_jit": {"file_identity": {"sha256": SHA256}},
                "four_quark": {"file_identity": {"sha256": "3" * 64}},
                "color_matrix": {"file_identity": {"sha256": "4" * 64}},
                "preflight": {"file_identity": {"sha256": "5" * 64}},
            },
            "passes": True,
        }
    )
    acceptance._require_content_identity(payload, label="manifest")
    payload["evidence"]["all_jit"]["file_identity"]["sha256"] = "6" * 64
    with pytest.raises(acceptance.AcceptanceError, match="content identity"):
        acceptance._require_content_identity(payload, label="manifest")


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.json"
    path.write_text('{"passes":true,"passes":true}\n', encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="duplicate key"):
        acceptance._checked_json(path, label="duplicate fixture")
