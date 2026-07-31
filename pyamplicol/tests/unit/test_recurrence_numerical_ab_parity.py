# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.developer import recurrence_numerical_ab_parity as parity

PROCESS_KEY, PROCESS_EXPRESSION = parity.PROCESS_CASES[0]


def test_file_identity_matches_harness_explicit_model_identity(
    tmp_path: Path,
) -> None:
    prepared_model = tmp_path / "model.pack"
    prepared_model.write_bytes(b"prepared-model")

    assert parity._path_identity(
        prepared_model.resolve()
    ) == parity.harness._path_identity(prepared_model.resolve())


def _selector_contract(layout: str) -> dict[str, object]:
    if layout == "topology-replay":
        body: dict[str, object] = {
            "workload": "single-runtime-selected-flow/helicity-sum",
            "physical_color_ids": ["c:1"],
            "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
            "structural_zero_helicity_ids": [],
            "selected_color_flow_ids": ["c:1"],
            "selected_helicity_ids": [],
        }
    else:
        body = {
            "workload": "all-flows/runtime-selected-single-helicity",
            "physical_color_ids": ["c:1"],
            "physical_helicity_ids": ["h:-1,+1", "h:+1,-1"],
            "structural_zero_helicity_ids": [],
            "selected_color_flow_ids": [],
            "selected_helicity_ids": ["h:-1,+1"],
        }
    return {
        **body,
        "contract_sha256": parity._canonical_sha256(body),
    }


def _capture(
    variant: str,
    *,
    layout: str = "topology-replay",
    process_key: str = PROCESS_KEY,
    process_expression: str = PROCESS_EXPRESSION,
    checkout: str | None = None,
    prepared_model_identity: dict[str, object] | None = None,
    source_revision: str | None = None,
) -> dict[str, object]:
    points = [
        [
            [50.0, 0.0, 0.0, 50.0],
            [50.0, 0.0, 0.0, -50.0],
        ]
    ]
    semantic = {
        "coverage": {
            "color": "complete",
            "helicities": "complete",
            "complete_physical_axes": True,
        },
        "physical_color_flows": {
            "count": 1,
            "ordered_entries": [{"id": "c:1", "index": 0}],
            "ordered_ids": ["c:1"],
        },
        "physical_helicities": {
            "count": 2,
            "ordered_entries": [
                {
                    "id": "h:-1,+1",
                    "index": 0,
                    "structural_zero": False,
                },
                {
                    "id": "h:+1,-1",
                    "index": 1,
                    "structural_zero": False,
                },
            ],
            "ordered_ids": ["h:-1,+1", "h:+1,-1"],
        },
        "normalization": {"kind": "squared-matrix-element"},
        "manifest_model_identity": {"name": "built-in-sm", "sha256": "a" * 64},
        "runtime_selector_semantics": {
            "color": "sum",
            "helicity": "sum",
            "generation_specialized_axes": [],
        },
        "reduction_ordering": {
            "axes": ["helicity", "color"],
            "component_order": "helicity-major/color-minor",
        },
        "execution_reduction_identity": {"kind": "ordered-complex-sum"},
        "execution_reduction_coverage": {
            "complete": True,
            "errors": [],
        },
        "generation_specialized_axes": [],
    }
    parameter_payload = {"alpha_s": 0.118, "mu_r": 91.1876}
    prepared_model = prepared_model_identity or {
        "path": f"/source/{variant}/model.pack",
        "size_bytes": 10,
        "sha256": "a" * 64,
    }
    artifact_id = "c" * 64
    source_checkout = checkout or f"/source/{variant}"
    revision = source_revision or parity.EXPECTED_SOURCE_REVISIONS[variant]
    artifact_root = f"/artifact/{variant}"
    native_identity = {
        "path": f"/runtime/{variant}/_rusticol.so",
        "size_bytes": 100,
        "sha256": "d" * 64,
    }
    build_info_identity = {
        "path": f"/runtime/{variant}/_build_info.json",
        "size_bytes": 100,
        "sha256": "e" * 64,
    }
    model_manifest = {
        "name": "built-in-sm",
        "content_sha256": "a" * 64,
    }
    common_model_identity = {
        "name": "built-in-sm",
        "content_sha256": "a" * 64,
    }

    def loaded_verification(phase: str) -> dict[str, object]:
        return {
            "kind": "pyamplicol-loaded-runtime-artifact-verification",
            "schema_version": 1,
            "phase": phase,
            "checked_at_utc": "2026-07-31T00:00:00Z",
            "expected_artifact_id": artifact_id,
            "loaded_artifact_id": artifact_id,
            "passes": True,
        }

    resolved_helicity_ids = (
        ["h:-1,+1", "h:+1,-1"] if layout == "topology-replay" else ["h:-1,+1"]
    )
    resolved_components = (
        [[[1.0, 0.0], [2.0, 0.0]]] if layout == "topology-replay" else [[[3.0, 0.0]]]
    )
    validation = {
        "fixture": {
            "file": {
                "path": f"{artifact_root}/processes/process/validation-momenta.json",
                "size_bytes": 100,
                "sha256": "b" * 64,
            },
            "point_count": 1,
            "points_sha256": parity._canonical_sha256(points),
            "points": points,
        },
        "selector_contract": _selector_contract(layout),
        "selected_totals": [[3.0, 0.0]],
        "resolved_sums": [[3.0, 0.0]],
        "resolved_helicity_ids": resolved_helicity_ids,
        "resolved_color_ids": ["c:1"],
        "resolved_components": resolved_components,
        "point_comparisons": [
            {
                "point_index": 0,
                "selected_total": [3.0, 0.0],
                "resolved_sum": [3.0, 0.0],
                "absolute_difference": 0.0,
                "relative_difference": 0.0,
                "passes": True,
            }
        ],
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
        "passes": True,
    }
    return parity._content_address(
        {
            "kind": parity.CAPTURE_KIND,
            "schema_version": parity.CAPTURE_SCHEMA,
            "variant": variant,
            "process_key": process_key,
            "requested_process": process_expression,
            "observed_process": process_expression.lower(),
            "layout": layout,
            "generation_request": parity._generation_request(
                validation_samples=1,
                point_tile_size=1024,
                jit_optimization_level=2,
            ),
            "source": {
                "checkout": source_checkout,
                "revision": revision,
                "dirty": False,
                "untracked_files_checked": True,
            },
            "runtime_provenance": {
                "interpreter": {
                    "path": "/runtime/python",
                    "size_bytes": 100,
                    "sha256": "2" * 64,
                    "python_version": "3.12.6",
                    "implementation": "CPython",
                },
                "installed_distribution": {
                    "package_version": "0.1.0",
                    "distribution_content": {
                        "algorithm": "sha256-relative-path-size-content-v1",
                        "file_count": 2,
                        "size_bytes": 200,
                        "sha256": "3" * 64,
                    },
                    "native_modules": [native_identity],
                    "build_info_files": [build_info_identity],
                },
                "active_build_info": {
                    **build_info_identity,
                    "payload": {
                        "source_checkout": source_checkout,
                        "source_revision": revision,
                        "native_build_inputs_sha256": "4" * 64,
                    },
                },
                "native_extension": {
                    **native_identity,
                    "package_version": "0.1.0",
                    "build_inputs_sha256": "4" * 64,
                },
                "dependencies": {
                    "Cargo.lock": {
                        "present": True,
                        "path": f"{source_checkout}/Cargo.lock",
                        "resolved_path": f"{source_checkout}/Cargo.lock",
                        "size_bytes": 100,
                        "sha256": "5" * 64,
                    }
                },
            },
            "prepared_model": prepared_model,
            "generation": {
                "mode": "recurrence",
                "generation_reused": False,
                "model_source": {
                    "kind": "explicit-prepared-model",
                    "file": prepared_model,
                    "compile_excluded_from_generation": True,
                },
            },
            "artifact": {
                "path": artifact_root,
                "artifact_id": artifact_id,
                "manifest": {
                    "path": f"{artifact_root}/artifact.json",
                    "size_bytes": 100,
                    "sha256": "6" * 64,
                },
                "tree": {
                    "algorithm": "sha256-relative-path-size-content-v1",
                    "file_count": 4,
                    "size_bytes": 400,
                    "sha256": "7" * 64,
                },
                "payloads": [
                    {
                        "path": "config/effective.toml",
                        "role": "configuration-effective",
                        "process_id": None,
                        "size_bytes": 100,
                        "sha256": "8" * 64,
                    },
                    {
                        "path": "model/parameters.json",
                        "role": "model-parameters",
                        "process_id": None,
                        "size_bytes": 100,
                        "sha256": "9" * 64,
                    },
                    {
                        "path": "processes/process/validation-momenta.json",
                        "role": "validation-momenta",
                        "process_id": "process",
                        "size_bytes": 100,
                        "sha256": "b" * 64,
                    },
                ],
                "process_id": "process",
                "process_expression": process_expression,
                "color_accuracy": "lc",
                "producer": {"version": "0.1.0"},
                "model_identity": {
                    "manifest": model_manifest,
                    "manifest_sha256": parity._canonical_sha256(model_manifest),
                    "common_physics_identity": common_model_identity,
                    "common_physics_identity_sha256": parity._canonical_sha256(
                        common_model_identity
                    ),
                },
                "semantic_identity": semantic,
                "semantic_identity_sha256": parity._canonical_sha256(semantic),
            },
            "effective_contract": {
                "execution_mode": "recurrence",
                "backend": "jit",
                "color_accuracy": "lc",
                "lc_flow_layout": layout,
                "jit_optimization_level": 2,
            },
            "parameters": {
                "file": {
                    "path": f"{artifact_root}/model/parameters.json",
                    "size_bytes": 100,
                    "sha256": "9" * 64,
                },
                "payload": parameter_payload,
                "payload_sha256": parity._canonical_sha256(parameter_payload),
            },
            "loaded_artifact_verification": {
                "before_evaluation": loaded_verification(
                    "before-numerical-parity-evaluation"
                ),
                "after_evaluation": loaded_verification(
                    "after-numerical-parity-evaluation"
                ),
            },
            "validation": validation,
            "tolerances": {
                "absolute": parity.ABSOLUTE_TOLERANCE,
                "relative": parity.RELATIVE_TOLERANCE,
            },
        }
    )


def _readdress(capture: dict[str, object]) -> dict[str, object]:
    capture.pop("content_sha256", None)
    return parity._content_address(capture)


def _compare(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    layout: str = "topology-replay",
) -> dict[str, object]:
    return parity._compare_captures(
        baseline,
        candidate,
        process_key=PROCESS_KEY,
        process_expression=PROCESS_EXPRESSION,
        layout=layout,
    )


def test_acceptance_matrix_and_defaults_are_fixed() -> None:
    assert parity.PROCESS_CASES == (
        ("dd_z_4g", "d d~ > Z g g g g"),
        ("dd_tt_3g", "d d~ > t t~ g g g"),
        ("gg_4g", "g g > g g g g"),
    )
    assert parity.LAYOUTS == ("topology-replay", "all-flow-union")
    assert parity.EXPECTED_SOURCE_REVISIONS == {
        "baseline": "172e58fd33a3c65563866c50cfbb5e1ddcd7b302",
        "candidate": "4e2b1e02dddde2d55b7250cbd52a93001f09b2c2",
    }
    assert parity.WATCHDOG_LIMIT_GIB == 30.0
    assert parity.ABSOLUTE_TOLERANCE == 1.0e-15
    assert parity.RELATIVE_TOLERANCE == 1.0e-12
    assert parity.DEFAULT_VALIDATION_SAMPLES == 10
    assert (
        parity._generation_request(
            validation_samples=10,
            point_tile_size=1024,
            jit_optimization_level=2,
        )["specialize_flow_at_generation"]
        is False
    )


def test_value_comparison_uses_strict_absolute_or_relative_tolerance() -> None:
    exact = parity._value_comparison([2.0, -1.0], [2.0, -1.0])
    relative = parity._value_comparison([1.0e6, 0.0], [1.0e6 + 1.0e-7, 0.0])
    failure = parity._value_comparison([1.0, 0.0], [1.0 + 1.0e-8, 0.0])

    assert exact["passes"] is True
    assert relative["absolute_difference"] > parity.ABSOLUTE_TOLERANCE
    assert relative["passes"] is True
    assert failure["passes"] is False


def test_workload_selection_is_deterministic_and_skips_structural_zeros() -> None:
    physics = SimpleNamespace(
        color_flows=(
            SimpleNamespace(id="c:first"),
            SimpleNamespace(id="c:second"),
        ),
        helicities=(
            SimpleNamespace(id="h:zero", structural_zero=True),
            SimpleNamespace(id="h:first-live", structural_zero=False),
            SimpleNamespace(id="h:second-live", structural_zero=False),
        ),
    )

    topology_selectors, topology_contract = parity._select_workload(
        physics,
        layout="topology-replay",
    )
    union_selectors, union_contract = parity._select_workload(
        physics,
        layout="all-flow-union",
    )

    assert topology_selectors == {"color_flows": ("c:first",)}
    assert topology_contract["selected_color_flow_ids"] == ["c:first"]
    assert union_selectors == {"helicities": ("h:first-live",)}
    assert union_contract["selected_helicity_ids"] == ["h:first-live"]
    assert union_contract["structural_zero_helicity_ids"] == ["h:zero"]
    for contract in (topology_contract, union_contract):
        body = {
            key: value for key, value in contract.items() if key != "contract_sha256"
        }
        assert contract["contract_sha256"] == parity._canonical_sha256(body)


@pytest.mark.parametrize(
    ("layout", "component_count"),
    (("topology-replay", 2), ("all-flow-union", 1)),
)
def test_matching_captures_emit_content_addressed_point_component_and_selector_evidence(
    layout: str,
    component_count: int,
) -> None:
    comparison = _compare(
        _capture("baseline", layout=layout),
        _capture("candidate", layout=layout),
        layout=layout,
    )

    assert comparison["passes"] is True
    parity._validate_content_address(
        comparison,
        kind=parity.COMPARISON_KIND,
        schema_version=parity.COMPARISON_SCHEMA,
        label="comparison",
    )
    assert comparison["summary"] == {
        "point_count": 1,
        "resolved_component_count": component_count,
        "maximum_absolute_point_difference": 0.0,
        "maximum_relative_point_difference": 0.0,
        "maximum_absolute_component_difference": 0.0,
        "maximum_relative_component_difference": 0.0,
    }
    records = [
        comparison["fixture_comparison"],
        comparison["selector_comparison"],
        comparison["parameter_comparison"],
        *comparison["pointwise_comparisons"],
        *comparison["resolved_component_comparisons"],
    ]
    assert all(
        record["content_sha256"]
        == parity._canonical_sha256(
            {key: value for key, value in record.items() if key != "content_sha256"}
        )
        for record in records
    )
    observed_components = [
        (entry["point_index"], entry["helicity_id"], entry["color_id"])
        for entry in comparison["resolved_component_comparisons"]
    ]
    expected_components = [(0, "h:-1,+1", "c:1")]
    if layout == "topology-replay":
        expected_components.append((0, "h:+1,-1", "c:1"))
    assert observed_components == expected_components


def test_point_and_resolved_component_drift_fail_independently() -> None:
    baseline = _capture("baseline")
    point_candidate = copy.deepcopy(_capture("candidate"))
    point_validation = point_candidate["validation"]
    point_validation["selected_totals"][0] = [3.0001, 0.0]
    point_validation["resolved_sums"][0] = [3.0001, 0.0]
    point_validation["point_comparisons"][0]["selected_total"] = [3.0001, 0.0]
    point_validation["point_comparisons"][0]["resolved_sum"] = [3.0001, 0.0]
    point_candidate = _readdress(point_candidate)

    point_comparison = _compare(baseline, point_candidate)
    assert point_comparison["passes"] is False
    assert point_comparison["pointwise_comparisons"][0]["passes"] is False
    assert all(
        entry["passes"] for entry in point_comparison["resolved_component_comparisons"]
    )

    component_candidate = copy.deepcopy(_capture("candidate"))
    component_candidate["validation"]["resolved_components"][0][1] = [2.0001, 0.0]
    component_candidate = _readdress(component_candidate)
    component_comparison = _compare(baseline, component_candidate)
    assert component_comparison["passes"] is False
    assert component_comparison["pointwise_comparisons"][0]["passes"] is True
    assert [
        entry["passes"]
        for entry in component_comparison["resolved_component_comparisons"]
    ] == [True, False]


def test_fixture_or_selector_drift_prevents_semantically_invalid_pairing() -> None:
    baseline = _capture("baseline")
    fixture_candidate = copy.deepcopy(_capture("candidate"))
    fixture = fixture_candidate["validation"]["fixture"]
    fixture["points"][0][0][0] = 51.0
    fixture["points_sha256"] = parity._canonical_sha256(fixture["points"])
    fixture_candidate = _readdress(fixture_candidate)

    fixture_comparison = _compare(baseline, fixture_candidate)
    assert fixture_comparison["passes"] is False
    assert fixture_comparison["pointwise_comparisons"] == []
    assert fixture_comparison["resolved_component_comparisons"] == []
    assert "deterministic phase-space fixtures differ" in fixture_comparison["errors"]

    selector_candidate = copy.deepcopy(_capture("candidate"))
    semantic = selector_candidate["artifact"]["semantic_identity"]
    semantic["physical_color_flows"]["ordered_ids"] = ["c:other"]
    semantic["physical_color_flows"]["ordered_entries"][0]["id"] = "c:other"
    selector_candidate["artifact"]["semantic_identity_sha256"] = (
        parity._canonical_sha256(semantic)
    )
    selector = selector_candidate["validation"]["selector_contract"]
    selector["physical_color_ids"] = ["c:other"]
    selector["selected_color_flow_ids"] = ["c:other"]
    selector_candidate["validation"]["resolved_color_ids"] = ["c:other"]
    selector["contract_sha256"] = parity._canonical_sha256(
        {key: value for key, value in selector.items() if key != "contract_sha256"}
    )
    selector_candidate = _readdress(selector_candidate)
    selector_comparison = _compare(baseline, selector_candidate)
    assert selector_comparison["passes"] is False
    assert selector_comparison["pointwise_comparisons"] == []
    assert selector_comparison["resolved_component_comparisons"] == []
    assert selector_comparison["selector_comparison"]["passes"] is False


def test_capture_digest_tampering_and_incomplete_semantics_fail_closed() -> None:
    tampered = _capture("baseline")
    tampered["prepared_model"]["sha256"] = "f" * 64
    with pytest.raises(parity.ParityError, match="content-address"):
        parity._validate_capture(
            tampered,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )

    incomplete = copy.deepcopy(_capture("baseline"))
    semantic = incomplete["artifact"]["semantic_identity"]
    semantic.pop("reduction_ordering")
    incomplete["artifact"]["semantic_identity_sha256"] = parity._canonical_sha256(
        semantic
    )
    incomplete = _readdress(incomplete)
    with pytest.raises(parity.ParityError, match="selector semantics"):
        parity._validate_capture(
            incomplete,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )


def test_capture_rejects_partial_sample_inventory_and_truncated_provenance() -> None:
    partial = _capture("baseline")
    partial["generation_request"]["validation_samples"] = 2
    partial = _readdress(partial)
    with pytest.raises(parity.ParityError, match="fixture or selector"):
        parity._validate_capture(
            partial,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )

    truncated = _capture("baseline")
    truncated["runtime_provenance"].pop("native_extension")
    truncated = _readdress(truncated)
    with pytest.raises(parity.ParityError, match="runtime provenance"):
        parity._validate_capture(
            truncated,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )


def test_capture_rejects_flow_specialization_in_request_or_semantics() -> None:
    specialized_request = _capture("baseline")
    specialized_request["generation_request"]["specialize_flow_at_generation"] = True
    specialized_request = _readdress(specialized_request)
    with pytest.raises(parity.ParityError, match="capture is inconsistent"):
        parity._validate_capture(
            specialized_request,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )

    specialized_semantics = _capture("baseline")
    semantic = specialized_semantics["artifact"]["semantic_identity"]
    semantic["generation_specialized_axes"] = ["color"]
    specialized_semantics["artifact"]["semantic_identity_sha256"] = (
        parity._canonical_sha256(semantic)
    )
    specialized_semantics = _readdress(specialized_semantics)
    with pytest.raises(parity.ParityError, match="capture is inconsistent"):
        parity._validate_capture(
            specialized_semantics,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )

    specialized_nested_semantics = _capture("baseline")
    semantic = specialized_nested_semantics["artifact"]["semantic_identity"]
    semantic["runtime_selector_semantics"]["generation_specialized_axes"] = ["color"]
    specialized_nested_semantics["artifact"]["semantic_identity_sha256"] = (
        parity._canonical_sha256(semantic)
    )
    specialized_nested_semantics = _readdress(specialized_nested_semantics)
    with pytest.raises(parity.ParityError, match="capture is inconsistent"):
        parity._validate_capture(
            specialized_nested_semantics,
            expected_variant="baseline",
            expected_process_key=PROCESS_KEY,
            expected_process=PROCESS_EXPRESSION,
            expected_layout="topology-replay",
        )


def test_artifact_manifest_parameter_and_fixture_links_fail_closed() -> None:
    invalid_manifest = _capture("baseline")
    invalid_manifest["artifact"]["manifest"]["path"] = "/outside/artifact.json"

    invalid_parameters = _capture("baseline")
    invalid_parameters["parameters"]["file"]["sha256"] = "f" * 64

    invalid_fixture = _capture("baseline")
    invalid_fixture["validation"]["fixture"]["file"]["path"] = (
        "/outside/validation-momenta.json"
    )

    for capture, message in (
        (invalid_manifest, "manifest path"),
        (invalid_parameters, "parameters is not bound"),
        (invalid_fixture, "fixture is not bound"),
    ):
        with pytest.raises(parity.ParityError, match=message):
            parity._validate_capture(
                _readdress(capture),
                expected_variant="baseline",
                expected_process_key=PROCESS_KEY,
                expected_process=PROCESS_EXPRESSION,
                expected_layout="topology-replay",
            )


def test_canonical_parameter_identity_rejects_int_float_type_drift() -> None:
    baseline = _capture("baseline")
    candidate = _capture("candidate")
    baseline["parameters"]["payload"] = {"x": 1}
    baseline["parameters"]["payload_sha256"] = parity._canonical_sha256({"x": 1})
    candidate["parameters"]["payload"] = {"x": 1.0}
    candidate["parameters"]["payload_sha256"] = parity._canonical_sha256({"x": 1.0})

    comparison = _compare(
        _readdress(baseline),
        _readdress(candidate),
    )

    assert comparison["passes"] is False
    assert comparison["parameter_comparison"]["passes"] is False
    assert "default model parameters differ" in comparison["errors"]


def test_source_preflight_rejects_any_revision_other_than_the_pinned_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "baseline"
    checkout.mkdir()
    variant = parity.Variant(
        name="baseline",
        python=tmp_path / "python",
        checkout=checkout,
        pythonpath=tmp_path / "site-packages",
        prepared_model=tmp_path / "model.pack",
    )

    def git_stdout(
        _checkout: Path,
        *arguments: str,
        label: str,
    ) -> str:
        del label
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(checkout)
        if arguments == ("rev-parse", "--verify", "HEAD"):
            return parity.EXPECTED_SOURCE_REVISIONS["candidate"]
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(parity, "_git_stdout", git_stdout)
    with pytest.raises(parity.ParityError, match="expected 172e58f"):
        parity._preflight_variant_source(variant)


def test_variant_binding_rejects_mixed_runtime_provenance_across_workers() -> None:
    capture = _capture("baseline")
    variant = parity.Variant(
        name="baseline",
        python=Path("/runtime/python"),
        checkout=Path("/source/baseline"),
        pythonpath=Path("/runtime/site-packages"),
        prepared_model=Path("/source/baseline/model.pack"),
    )
    bindings: dict[str, dict[str, object]] = {}
    parity._pin_variant_binding(
        bindings,
        variant=variant,
        capture=capture,
        expected_model=capture["prepared_model"],
        expected_generation_request=capture["generation_request"],
        expected_source=capture["source"],
    )

    drifted = copy.deepcopy(capture)
    drifted["runtime_provenance"]["native_extension"]["package_version"] = "0.2.0"
    with pytest.raises(parity.ParityError, match="changed during the campaign"):
        parity._pin_variant_binding(
            bindings,
            variant=variant,
            capture=drifted,
            expected_model=drifted["prepared_model"],
            expected_generation_request=drifted["generation_request"],
            expected_source=drifted["source"],
        )


def test_workspace_python_symlink_is_retained_and_output_scope_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity, "DRIVER_ROOT", tmp_path.resolve())
    interpreter = tmp_path / "python"
    interpreter.symlink_to(Path(sys.executable))
    assert (
        parity._resolve_python_inside_workspace(interpreter, label="test Python")
        == interpreter.absolute()
    )

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(parity, "ALLOWED_OUTPUT_PARENT", allowed.resolve())
    requested = allowed / "campaign"
    assert parity._resolve_output_root(requested) == requested.resolve()
    with pytest.raises(parity.ParityError, match="below"):
        parity._resolve_output_root(tmp_path / "escaped")
    requested.mkdir()
    with pytest.raises(parity.ParityError, match="already exists"):
        parity._resolve_output_root(requested)


def test_worker_command_and_environment_isolate_all_writes_under_watchdog(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    pythonpath = tmp_path / "site-packages"
    prepared_model = tmp_path / "model.pack"
    python = tmp_path / "python"
    for directory in (checkout, pythonpath):
        directory.mkdir()
    for file_path in (prepared_model, python):
        file_path.write_bytes(b"x")
    variant = parity.Variant(
        name="baseline",
        python=python,
        checkout=checkout,
        pythonpath=pythonpath,
        prepared_model=prepared_model,
    )
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    command = parity._worker_command(
        variant,
        capture_root=capture_root,
        process_key=PROCESS_KEY,
        process_expression=PROCESS_EXPRESSION,
        layout="all-flow-union",
        validation_samples=10,
        point_tile_size=1024,
        jit_optimization_level=2,
    )
    assert command[:6] == [
        str(python),
        str(parity.WATCHDOG_PATH),
        "--limit-gib",
        "30",
        "--",
        str(python),
    ]
    assert command[command.index("--artifact") + 1] == str(capture_root / "artifact")
    assert command[command.index("--capture-json") + 1] == str(
        capture_root / "capture.json"
    )

    environment_root = tmp_path / "environment"
    environment_root.mkdir()
    environment, overrides = parity._worker_environment(
        variant,
        environment_root,
    )
    assert overrides[parity.SOURCE_CHECKOUT_ENV] == str(checkout)
    assert overrides["PYTHONPATH"] == str(pythonpath)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX", "MPLCONFIGDIR"):
        assert Path(overrides[key]).resolve().is_relative_to(environment_root.resolve())
        assert environment[key] == overrides[key]
    assert os.environ.get(parity.SOURCE_CHECKOUT_ENV) != str(checkout)


def test_orchestrator_runs_the_complete_matrix_with_balanced_pair_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parity, "ALLOWED_OUTPUT_PARENT", tmp_path.resolve())
    shared_python = tmp_path / "python"
    shared_python.write_bytes(b"python")
    variants: dict[str, parity.Variant] = {}
    for name in ("baseline", "candidate"):
        checkout = tmp_path / name / "checkout"
        pythonpath = tmp_path / name / "site-packages"
        prepared_model = tmp_path / name / "model.pack"
        checkout.mkdir(parents=True)
        pythonpath.mkdir(parents=True)
        prepared_model.write_bytes(b"identical-model")
        variants[name] = parity.Variant(
            name=name,
            python=shared_python,
            checkout=checkout,
            pythonpath=pythonpath,
            prepared_model=prepared_model,
        )

    monkeypatch.setattr(
        parity,
        "_variant",
        lambda name, **_values: variants[name],
    )
    preflight_calls: list[str] = []

    def preflight_source(variant: parity.Variant) -> dict[str, object]:
        preflight_calls.append(variant.name)
        return {
            "checkout": str(variant.checkout.resolve()),
            "revision": parity.EXPECTED_SOURCE_REVISIONS[variant.name],
            "dirty": False,
            "untracked_files_checked": True,
        }

    monkeypatch.setattr(parity, "_preflight_variant_source", preflight_source)
    calls: list[tuple[str, str, str]] = []

    def capture_variant(
        variant: parity.Variant,
        *,
        process_key: str,
        process_expression: str,
        layout: str,
        **_values: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert preflight_calls[:2] == ["baseline", "candidate"]
        calls.append((variant.name, process_key, layout))
        capture = _capture(
            variant.name,
            layout=layout,
            process_key=process_key,
            process_expression=process_expression,
            checkout=str(variant.checkout.resolve()),
            prepared_model_identity=parity._path_identity(variant.prepared_model),
        )
        invocation = parity._content_address(
            {
                "kind": "test-invocation",
                "schema_version": 1,
                "variant": variant.name,
                "process_key": process_key,
                "layout": layout,
            }
        )
        return capture, invocation

    monkeypatch.setattr(parity, "_capture_variant", capture_variant)
    arguments = SimpleNamespace(
        baseline_python=shared_python,
        candidate_python=shared_python,
        baseline_checkout=variants["baseline"].checkout,
        candidate_checkout=variants["candidate"].checkout,
        baseline_pythonpath=variants["baseline"].pythonpath,
        candidate_pythonpath=variants["candidate"].pythonpath,
        baseline_prepared_model=variants["baseline"].prepared_model,
        candidate_prepared_model=variants["candidate"].prepared_model,
        output_root=tmp_path / "campaign",
        validation_samples=1,
        point_tile_size=1024,
        jit_optimization_level=2,
        worker_timeout=60.0,
    )

    result = parity.run(arguments)

    assert result["passes"] is True
    assert result["comparison_count"] == 6
    assert result["passing_comparison_count"] == 6
    assert len(calls) == 12
    assert preflight_calls == ["baseline", "candidate", "baseline", "candidate"]
    assert [name for name, _process, _layout in calls] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert {(process_key, layout) for _name, process_key, layout in calls} == {
        (process_key, layout)
        for process_key, _expression in parity.PROCESS_CASES
        for layout in parity.LAYOUTS
    }
    result_path = arguments.output_root / "result.json"
    assert result_path.is_file()
    parity._validate_content_address(
        parity._json_object(result_path, label="result"),
        kind=parity.RESULT_KIND,
        schema_version=parity.RESULT_SCHEMA,
        label="result",
    )
