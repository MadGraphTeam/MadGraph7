# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.developer import four_quark_compiled_gate as gate


def test_checked_file_identity_rejects_a_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW is unavailable")
    target = tmp_path / "target.bin"
    target.write_bytes(b"authenticated")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(gate.GateError, match="checked fd"):
        gate._regular_file_identity(link)


def _source_leaf() -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": gate.SYMJIT_RUNTIME_CAPABILITY,
        "input_len": 1,
        "output_len": 1,
        "application_path": "evaluators/stage.symjit",
        "application_abi": gate.SYMJIT_APPLICATION_ABI,
        "optimization_level": 3,
    }


def _stage() -> dict[str, object]:
    return {
        "stage_kind": "amplitude-roots",
        "output_length": 1,
        "evaluator": _source_leaf(),
        "compiled_plane_arena": {
            "kind": "compiled-plane-arena-stage",
            "schema_version": 1,
            "application_abi": gate.COMPILED_PLANE_DIRECT_APPLICATION_ABI,
            "source_application_abi": gate.SYMJIT_APPLICATION_ABI,
            "element_layout": "split-complex-component-major",
            "input_output_aliasing": "forbidden",
            "output_output_aliasing": "forbidden",
            "output_operation": "overwrite",
            "output_factor": "identity",
            "input_bindings": [
                {
                    "kind": "value",
                    "parameter_index": 0,
                    "source_id": 0,
                    "component": 0,
                }
            ],
            "output_bindings": [
                {"arena": "amplitude", "component": 0, "output_index": 0}
            ],
            "leaves": [
                {
                    "application_path": "evaluators/stage.symjit",
                    "source_application_abi": gate.SYMJIT_APPLICATION_ABI,
                    "optimization_level": 3,
                    "direct_codegen_optimization_level": 3,
                    "input_indices": [0],
                    "input_len": 1,
                    "output_len": 1,
                    "output_start": 0,
                    "output_stop": 1,
                }
            ],
        },
    }


def _execution_payload() -> dict[str, object]:
    return {
        "required_runtime_capabilities": [
            gate.COMPILED_PLANE_ARENA_CAPABILITY,
            gate.SYMJIT_RUNTIME_CAPABILITY,
        ],
        "compiled": {
            "model_parameter_evaluator": {
                "kind": "model-parameter-evaluator",
                "evaluator": _source_leaf(),
            },
            "stage_evaluators": {
                "kind": "generic-dag-stage-evaluator-artifacts",
                "runtime_available": True,
                "required_runtime_capabilities": [
                    gate.COMPILED_PLANE_ARENA_CAPABILITY,
                    gate.SYMJIT_RUNTIME_CAPABILITY,
                ],
                "stage_count": 1,
                "stages": [],
                "amplitude_stage": _stage(),
            },
        },
    }


def _capture(offset: float = 0.0) -> gate.EvaluationCapture:
    return gate.EvaluationCapture(
        physics_axes={
            "process": gate.PROCESS,
            "helicities": ["h:one"],
            "color_flows": ["flow:one"],
        },
        f64_total=(complex(1.0 + offset), complex(2.0 + offset)),
        exact_total=(complex(1.0 + offset), complex(2.0 + offset)),
        f64_components=(complex(0.25 + offset), complex(0.75)),
        exact_components=(complex(0.25 + offset), complex(0.75)),
        color_probe=(complex(0.5 + offset), complex(1.0)),
        helicity_probe=(complex(0.4 + offset), complex(0.8)),
    )


def test_recursive_direct_arena_audit_accepts_complete_o3_stage() -> None:
    result = gate._audit_execution_payload(_execution_payload())

    assert result == {
        "capability": gate.COMPILED_PLANE_ARENA_CAPABILITY,
        "capability_declaration_count": 2,
        "stage_set_count": 1,
        "stage_count": 1,
        "descriptor_count": 1,
        "leaf_count": 1,
        "model_parameter_slot_count": 1,
        "model_parameter_evaluator_count": 1,
        "model_parameter_direct_application_count": 0,
        "passes": True,
    }


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda payload: payload["compiled"]["stage_evaluators"][
                "amplitude_stage"
            ].pop("compiled_plane_arena"),
            "compiled_plane_arena",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["leaves"][0].update({"optimization_level": 2}),
            "not JIT O3",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["leaves"][0].update({"direct_codegen_optimization_level": 2}),
            "fixed O3",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["leaves"][0].update({"output_stop": 2}),
            "outputs are invalid",
        ),
        (
            lambda payload: payload["compiled"]["model_parameter_evaluator"].update(
                {"compiled_plane_arena": {"kind": "compiled-plane-arena-stage"}}
            ),
            "model_parameter_evaluator illegally",
        ),
        (
            lambda payload: payload["compiled"]["model_parameter_evaluator"].update(
                {
                    "nested": {
                        "application_abi": (gate.NATIVE_COMPILED_DIRECT_APPLICATION_ABI)
                    }
                }
            ),
            "direct application ABI",
        ),
    ],
)
def test_recursive_direct_arena_audit_rejects_descriptor_drift(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    payload = deepcopy(_execution_payload())
    mutation(payload)

    with pytest.raises(gate.GateError, match=message):
        gate._audit_execution_payload(payload)


def test_lc_cross_layout_audit_requires_exact_axes_and_pointwise_parity() -> None:
    result = gate._cross_lc_parity(_capture(), _capture())

    assert result["passes"] is True
    assert result["physical_axes_match"] is True
    assert set(result["comparisons"]) == {
        "f64_totals",
        "precision32_totals",
        "f64_resolved_components",
        "precision32_resolved_components",
        "color_selector",
        "helicity_selector",
    }

    with pytest.raises(gate.GateError, match="physical axes differ"):
        drift = _capture()
        drift.physics_axes["color_flows"] = ["flow:other"]
        gate._cross_lc_parity(_capture(), drift)

    with pytest.raises(gate.GateError, match="f64 totals failed"):
        gate._cross_lc_parity(_capture(), _capture(offset=1.0e-3))


def test_union_rejection_is_checked_through_public_color_config() -> None:
    result = gate._assert_union_rejections()

    assert [entry["color_accuracy"] for entry in result] == ["nlc", "full"]
    assert all(entry["rejected"] is True for entry in result)
    assert all("all-flow-union" in entry["message"] for entry in result)


def test_content_identity_addresses_the_complete_evidence_body() -> None:
    body = {"kind": gate.RESULT_KIND, "passes": True, "lanes": {"lc": [1, 2]}}

    addressed = gate._attach_content_identity(body)

    assert addressed["content_identity"]["sha256"] == gate._canonical_sha256(body)
    assert gate._attach_content_identity(body) == addressed
    assert gate._attach_content_identity({**body, "passes": False}) != addressed


def test_compiled_profile_requires_direct_calls_and_zero_boundary_traffic() -> None:
    profile = {
        "execution_mode": "compiled",
        "evaluator_backend_call_count": 7,
        "compiled_direct_arena_engine_count": 2,
        "compiled_direct_arena_call_count": 7,
    }
    profile.update(dict.fromkeys(gate._ZERO_PROFILE_COUNTERS, 0))
    profile.update(dict.fromkeys(gate._ZERO_PROFILE_TIMES, 0.0))

    evidence = gate._audit_compiled_direct_profile(profile)

    assert evidence["compiled_direct_arena_call_count"] == 7
    assert evidence["compiled_direct_arena_engine_count"] == 2
    assert evidence["legacy_boundary_component_total"] == 0
    with pytest.raises(gate.GateError, match="no backend calls"):
        gate._audit_compiled_direct_profile(
            {**profile, "compiled_direct_arena_call_count": 0}
        )
    with pytest.raises(gate.GateError, match="legacy boundary traffic"):
        gate._audit_compiled_direct_profile(
            {**profile, "stage_input_copy_component_count": 1}
        )
    with pytest.raises(gate.GateError, match="every evaluator backend call"):
        gate._audit_compiled_direct_profile(
            {**profile, "compiled_direct_arena_call_count": 6}
        )


def test_helicity_probe_is_computed_nonzero_and_self_representing() -> None:
    def helicity(
        identifier: str,
        *,
        computed: bool,
        structural_zero: bool,
        coefficient: float,
        representative_id: str,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=identifier,
            computed=computed,
            structural_zero=structural_zero,
            coefficient=coefficient,
            representative_id=representative_id,
        )

    physics = SimpleNamespace(
        helicities=(
            helicity(
                "zero",
                computed=False,
                structural_zero=True,
                coefficient=0.0,
                representative_id="zero",
            ),
            helicity(
                "first",
                computed=True,
                structural_zero=False,
                coefficient=1.0,
                representative_id="first",
            ),
            helicity(
                "alias",
                computed=False,
                structural_zero=False,
                coefficient=1.0,
                representative_id="first",
            ),
            helicity(
                "second",
                computed=True,
                structural_zero=False,
                coefficient=1.0,
                representative_id="second",
            ),
        )
    )

    selected, count = gate._select_helicity_probe(physics, lane_name="test")

    assert selected.id == "second"
    assert count == 2

    with pytest.raises(gate.GateError, match="no executable helicity"):
        gate._select_helicity_probe(
            SimpleNamespace(helicities=(physics.helicities[0],)),
            lane_name="test",
        )


def test_orchestrator_uses_exactly_four_isolated_generation_lanes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated: list[str] = []
    capture = _capture()
    monkeypatch.setattr(gate, "_source_identity", lambda: {"revision": "a" * 40})
    monkeypatch.setattr(gate, "_runtime_identity", lambda source: {"bound": source})
    monkeypatch.setattr(
        gate,
        "_file_identity",
        lambda path: {"path": str(path), "sha256": "c" * 64},
    )
    monkeypatch.setattr(
        gate,
        "_assert_union_rejections",
        lambda: [{"color_accuracy": "nlc"}, {"color_accuracy": "full"}],
    )
    monkeypatch.setattr(
        gate,
        "_prepare_output_root",
        lambda output_root, replace: tmp_path,
    )
    monkeypatch.setattr(gate, "_deterministic_points", lambda count: (((),),) * count)
    monkeypatch.setattr(
        gate,
        "_points_identity",
        lambda points: {"sha256": "b" * 64, "point_count": len(points)},
    )

    def generate(
        artifact: Path,
        lane: gate.Lane,
        *,
        timeout: float,
        expected_revision: str,
        expected_runtime_sha256: str,
        expected_script_sha256: str,
    ) -> dict[str, object]:
        generated.append(lane.name)
        return {
            "lane": lane.name,
            "timeout": timeout,
            "artifact": str(artifact),
            "expected_revision": expected_revision,
            "expected_runtime_sha256": expected_runtime_sha256,
            "expected_script_sha256": expected_script_sha256,
        }

    monkeypatch.setattr(gate, "_invoke_generation_worker", generate)
    monkeypatch.setattr(
        gate,
        "_artifact_identity_and_audit",
        lambda artifact, lane: ({"artifact_id": lane.name}, "process"),
    )
    monkeypatch.setattr(
        gate,
        "_evaluate_artifact",
        lambda artifact, process_id, lane, points: ({"passes": True}, capture),
    )
    monkeypatch.setattr(
        gate,
        "_cross_lc_parity",
        lambda topology, union: {"passes": True},
    )
    monkeypatch.setattr(gate, "_peak_rss_gib", lambda: 1.25)
    arguments = argparse.Namespace(
        output_root=tmp_path,
        replace=False,
        points=2,
        generation_timeout=123.0,
    )

    result = gate.run_gate(arguments)

    assert generated == [lane.name for lane in gate.LANES]
    assert set(result["lanes"]) == {lane.name for lane in gate.LANES}
    assert result["passes"] is True
    assert result["status"] == "ok"
    assert result["source_identity_match"] is True
    assert result["runtime_identity_match"] is True
    assert result["gate_script_identity_match"] is True
    assert result["content_identity"]["sha256"] == gate._canonical_sha256(
        {key: value for key, value in result.items() if key != "content_identity"}
    )


def test_parser_documents_mandatory_watchdog_and_two_point_default() -> None:
    parsed = gate.parser().parse_args(["--output-root", "out"])

    assert parsed.points == 2
    assert parsed.generation_timeout == 10_800.0
    assert "--limit-gib 30" in gate.WATCHDOG_INVOCATION
    assert "--output-root" in gate.WATCHDOG_INVOCATION
