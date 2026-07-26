# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import pytest

from tools.developer import compiled_all_jit_arena_gate as gate


def test_checked_file_identity_rejects_a_symlink(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW is unavailable")
    target = tmp_path / "target.bin"
    target.write_bytes(b"authenticated")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(gate.GateError, match="checked fd"):
        gate._regular_file_identity(link)


def _source_leaf(optimization_level: int) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": gate.SYMJIT_RUNTIME_CAPABILITY,
        "input_len": 1,
        "output_len": 1,
        "application_path": f"evaluators/stage-o{optimization_level}.symjit",
        "application_abi": gate.SYMJIT_APPLICATION_ABI,
        "optimization_level": optimization_level,
    }


def _stage(optimization_level: int) -> dict[str, object]:
    path = f"evaluators/stage-o{optimization_level}.symjit"
    return {
        "stage_kind": "amplitude-roots",
        "output_length": 1,
        "evaluator": _source_leaf(optimization_level),
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
                    "application_path": path,
                    "source_application_abi": gate.SYMJIT_APPLICATION_ABI,
                    "optimization_level": optimization_level,
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


def _stage_set(optimization_level: int) -> dict[str, object]:
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "runtime_available": True,
        "required_runtime_capabilities": [
            gate.COMPILED_PLANE_ARENA_CAPABILITY,
            gate.SYMJIT_RUNTIME_CAPABILITY,
        ],
        "stage_count": 1,
        "stages": [],
        "amplitude_stage": _stage(optimization_level),
    }


def _execution_payload(optimization_level: int) -> dict[str, object]:
    return {
        "required_runtime_capabilities": [
            gate.COMPILED_PLANE_ARENA_CAPABILITY,
            gate.SYMJIT_RUNTIME_CAPABILITY,
        ],
        "compiled": {
            "model_parameter_evaluator": {
                "kind": "model-parameter-evaluator",
                "evaluator": _source_leaf(optimization_level),
            },
            "stage_evaluators": _stage_set(optimization_level),
        },
        "helicity_sum_execution": {
            "required_runtime_capabilities": [
                gate.COMPILED_PLANE_ARENA_CAPABILITY,
            ],
            "runtime_schema": {
                "model_parameter_evaluator": None,
                "stage_evaluators": _stage_set(optimization_level),
            },
        },
    }


def _profile() -> dict[str, object]:
    result: dict[str, object] = {
        "execution_mode": "compiled",
        "evaluator_backend_call_count": 7,
        "compiled_direct_arena_engine_count": 2,
        "compiled_direct_arena_call_count": 7,
    }
    result.update(dict.fromkeys(gate._ZERO_PROFILE_COUNTERS, 0))
    result.update(dict.fromkeys(gate._ZERO_PROFILE_TIMES, 0.0))
    return result


def _capture(offset: float = 0.0) -> gate.EvaluationCapture:
    values = (complex(1.0 + offset), complex(2.0 + offset))
    components = (complex(0.25 + offset), complex(0.75))
    return gate.EvaluationCapture(
        physics_axes={
            "process": gate.PROCESS,
            "helicities": ["h:one"],
            "color_flows": ["flow:one"],
        },
        f64_total=values,
        precision32_total=values,
        f64_components=components,
        precision32_components=components,
    )


@pytest.mark.parametrize("optimization_level", gate.JIT_LEVELS)
def test_recursive_audit_accepts_every_jit_level(
    optimization_level: int,
) -> None:
    result = gate._audit_execution_payload(
        _execution_payload(optimization_level),
        optimization_level,
    )

    assert result["source_optimization_level"] == optimization_level
    assert result["stage_set_count"] == 2
    assert result["descriptor_count"] == 2
    assert result["leaf_count"] == 2
    assert result["model_parameter_direct_application_count"] == 0
    assert result["passes"] is True


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
            ]["leaves"][0].update({"optimization_level": 3}),
            "does not retain JIT O1",
        ),
        (
            lambda payload: payload["compiled"]["stage_evaluators"]["amplitude_stage"][
                "compiled_plane_arena"
            ]["leaves"][0].update({"direct_codegen_optimization_level": 2}),
            "fixed O3",
        ),
        (
            lambda payload: payload["helicity_sum_execution"]["runtime_schema"][
                "stage_evaluators"
            ]["amplitude_stage"]["evaluator"].update({"optimization_level": 2}),
            "does not retain JIT O1",
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
def test_recursive_audit_rejects_level_or_direct_application_drift(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    payload = deepcopy(_execution_payload(1))
    mutation(payload)

    with pytest.raises(gate.GateError, match=message):
        gate._audit_execution_payload(payload, 1)


def test_profile_audit_requires_direct_calls_and_zero_legacy_traffic() -> None:
    result = gate._audit_profile(_profile())

    assert result["direct_arena_engine_counter"] == {
        "name": "compiled_direct_arena_engine_count",
        "value": 2,
    }
    assert result["direct_arena_call_counter"] == {
        "name": "compiled_direct_arena_call_count",
        "value": 7,
    }
    assert result["passes"] is True

    traffic = _profile()
    traffic["amplitude_output_remap_component_count"] = 1
    with pytest.raises(gate.GateError, match=r"output_remap.*=1"):
        gate._audit_profile(traffic)

    no_calls = _profile()
    no_calls["compiled_direct_arena_call_count"] = 0
    with pytest.raises(gate.GateError, match="no compiled Direct-Arena calls"):
        gate._audit_profile(no_calls)

    incomplete_calls = _profile()
    incomplete_calls["compiled_direct_arena_call_count"] = 6
    with pytest.raises(gate.GateError, match="every evaluator backend call"):
        gate._audit_profile(incomplete_calls)


def test_profile_audit_requires_explicit_direct_engine_counter() -> None:
    profile = _profile()

    result = gate._audit_profile(profile)

    assert result["direct_arena_engine_counter"] == {
        "name": "compiled_direct_arena_engine_count",
        "value": 2,
    }

    profile["compiled_direct_arena_engine_count"] = 0
    with pytest.raises(gate.GateError, match="no compiled Direct-Arena engines"):
        gate._audit_profile(profile)


def test_runtime_profile_fails_closed_when_api_has_no_profiler() -> None:
    class RuntimeWithoutProfile:
        pass

    with pytest.raises(gate.GateError, match="activity cannot be proven"):
        gate._profile_runtime(RuntimeWithoutProfile(), (((),),))


def test_cross_level_parity_covers_totals_and_resolved_components() -> None:
    captures = {level: _capture() for level in gate.JIT_LEVELS}

    result = gate._cross_level_parity(captures)

    assert set(result["comparisons"]) == {"o0_vs_o3", "o1_vs_o3", "o2_vs_o3"}
    assert all(
        set(comparisons)
        == {
            "f64_totals",
            "precision32_totals",
            "f64_resolved_components",
            "precision32_resolved_components",
        }
        for comparisons in result["comparisons"].values()
    )

    captures[1] = _capture(offset=1.0e-3)
    with pytest.raises(gate.GateError, match="O1/O3 f64 totals"):
        gate._cross_level_parity(captures)


def test_orchestrator_generates_and_checks_exactly_o0_through_o3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated: list[int] = []
    source = {"revision": "a" * 40}
    runtime = {"identity": "candidate"}
    runtime_sha = gate._canonical_sha256(runtime)
    monkeypatch.setattr(gate, "_source_identity", lambda: source)
    monkeypatch.setattr(gate, "_runtime_identity", lambda observed: runtime)
    monkeypatch.setattr(
        gate,
        "_file_identity",
        lambda path: {"path": str(path), "sha256": "c" * 64},
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
        optimization_level: int,
        expected_revision: str,
        expected_script_sha256: str,
        *,
        timeout: float,
    ) -> dict[str, object]:
        generated.append(optimization_level)
        return {
            "optimization_level": optimization_level,
            "artifact": str(artifact),
            "timeout": timeout,
            "worker_provenance": {
                "source_revision": expected_revision,
                "runtime_identity_sha256": runtime_sha,
                "gate_script_sha256": expected_script_sha256,
                "postflight_identity_match": True,
            },
        }

    monkeypatch.setattr(gate, "_invoke_generation_worker", generate)
    monkeypatch.setattr(
        gate,
        "_artifact_identity_and_audit",
        lambda artifact, level: ({"artifact_id": f"o{level}"}, "process"),
    )
    monkeypatch.setattr(
        gate,
        "_evaluate_artifact",
        lambda artifact, process_id, optimization_level, points: (
            {"passes": True},
            _capture(),
        ),
    )
    monkeypatch.setattr(gate, "_peak_rss_gib", lambda: 1.0)
    arguments = argparse.Namespace(
        output_root=tmp_path,
        replace=False,
        points=2,
        generation_timeout=123.0,
    )

    result = gate.run_gate(arguments)

    assert generated == [0, 1, 2, 3]
    assert set(result["levels"]) == {"jit-o0", "jit-o1", "jit-o2", "jit-o3"}
    assert result["passes"] is True
    assert result["status"] == "ok"
    assert result["source_identity_match"] is True
    assert result["runtime_identity_match"] is True
    assert result["gate_script_identity_match"] is True
    assert result["content_identity"]["sha256"] == gate._canonical_sha256(
        {key: value for key, value in result.items() if key != "content_identity"}
    )


def test_parser_documents_mandatory_watchdog_and_defaults() -> None:
    parsed = gate.parser().parse_args(["--output-root", "out"])

    assert parsed.points == 2
    assert parsed.generation_timeout == 3_600.0
    assert "--limit-gib 30" in gate.WATCHDOG_INVOCATION
    assert "compiled_all_jit_arena_gate.py" in gate.WATCHDOG_INVOCATION


def test_output_root_must_not_dirty_the_authenticated_checkout() -> None:
    with pytest.raises(gate.GateError, match="outside the source checkout"):
        gate._prepare_output_root(gate.ROOT / "generated-gate", replace=False)
