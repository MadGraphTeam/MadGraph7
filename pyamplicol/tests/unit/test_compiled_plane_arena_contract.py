# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from copy import deepcopy

import pytest

from pyamplicol._internal.versions import (
    COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
    COMPILED_PLANE_DIRECT_APPLICATION_ABI,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.generation.artifact_writer import (
    _prefix_evaluator_payload_paths,
    _stage_evaluator_set,
)
from pyamplicol.generation.stage_artifacts import (
    _compiled_plane_arena_stage,
)


def _leaf(path: str, input_len: int, output_len: int) -> dict[str, object]:
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
        "input_len": input_len,
        "output_len": output_len,
        "application_path": path,
        "application_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 3,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": None,
        "evaluator_state_runtime_capability": (
            SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
        ),
    }


def _stage(*, amplitude: bool = False) -> dict[str, object]:
    evaluator = {
        "kind": "chunked-symbolica-evaluator",
        "input_len": 3,
        "chunk_input_indices": [[0, 2], [1]],
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
        "chunks": [
            _leaf("evaluators/left.symjit", 2, 1),
            _leaf("evaluators/right.symjit", 1, 1),
        ],
    }
    return {
        "stage_index": 2,
        "stage_kind": "amplitude-roots" if amplitude else "current-combine",
        "subset_size": None if amplitude else 3,
        "evaluator_label": "amplitude" if amplitude else "stage-2",
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 2,
        "output_slots": [
            {
                "value_slot_id": -1 if amplitude else 7,
                "current_id": -1 if amplitude else 7,
                "variant": "amplitude-root" if amplitude else "propagated",
                "component_start": 0 if amplitude else 12,
                "component_stop": 2 if amplitude else 14,
                "output_start": 0,
                "output_stop": 2,
                "color_selector_domain_ids": [0],
            }
        ],
        "input_value_slot_ids": [0],
        "output_value_slot_ids": [] if amplitude else [7],
        "interaction_ids": [4],
        "input_components": [
            {
                "kind": "value",
                "source_id": 0,
                "component": 0,
                "global_component": 0,
                "parameter_index": 0,
                "real_valued": False,
            },
            {
                "kind": "momentum",
                "source_id": 3,
                "component": 2,
                "global_component": 22,
                "parameter_index": 1,
                "real_valued": True,
            },
            {
                "kind": "model_parameter",
                "source_id": 1,
                "component": 0,
                "global_component": 31,
                "parameter_index": 2,
                "real_valued": False,
            },
        ],
        "parameter_count": 3,
        "value_parameter_count": 1,
        "momentum_parameter_count": 1,
        "model_parameter_count": 1,
        "real_valued_inputs": [1],
        "expression_ready": True,
        "blockers": [],
        "evaluator": evaluator,
    }


def _set() -> dict[str, object]:
    stage = _stage()
    amplitude = _stage(amplitude=True)
    stage["compiled_plane_arena"] = _compiled_plane_arena_stage(stage)
    amplitude["compiled_plane_arena"] = _compiled_plane_arena_stage(amplitude)
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "required_runtime_capabilities": [
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            SYMJIT_F64_RUNTIME_CAPABILITY,
        ],
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 2,
        "stages": [stage],
        "amplitude_stage": amplitude,
    }


def test_compiled_plane_contract_freezes_leaf_and_plane_bindings() -> None:
    stage = _stage()
    direct = _compiled_plane_arena_stage(stage)

    assert direct is not None
    assert direct["application_abi"] == COMPILED_PLANE_DIRECT_APPLICATION_ABI
    assert [leaf["input_indices"] for leaf in direct["leaves"]] == [[0, 2], [1]]
    assert [leaf["output_start"] for leaf in direct["leaves"]] == [0, 1]
    assert [binding["component"] for binding in direct["output_bindings"]] == [
        12,
        13,
    ]


def test_compiled_plane_contract_rejects_non_o3_generation() -> None:
    stage = _stage()
    stage["evaluator"]["chunks"][0]["optimization_level"] = 2

    with pytest.raises(
        ValueError,
        match="requires compiled JIT optimization level 3",
    ):
        _compiled_plane_arena_stage(stage)


def test_stage_set_requires_complete_capability_bound_metadata() -> None:
    serialized = _stage_evaluator_set(_set())
    assert (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
        in serialized["required_runtime_capabilities"]
    )
    assert all(
        "compiled_plane_arena" in stage
        for stage in (*serialized["stages"], serialized["amplitude_stage"])
    )

    missing = deepcopy(_set())
    del missing["stages"][0]["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled-plane-arena-v1"):
        _stage_evaluator_set(missing)

    legacy = deepcopy(_set())
    legacy["required_runtime_capabilities"].remove(
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
    )
    del legacy["stages"][0]["compiled_plane_arena"]
    del legacy["amplitude_stage"]["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled SymJIT artifacts require"):
        _stage_evaluator_set(legacy)

    drift = deepcopy(_set())
    drift["amplitude_stage"]["compiled_plane_arena"]["leaves"][1][
        "output_start"
    ] = 0
    with pytest.raises(ValueError, match="leaf bindings"):
        _stage_evaluator_set(drift)


def test_direct_source_paths_follow_nested_lane_prefixes() -> None:
    serialized = _stage_evaluator_set(_set())
    prefixed = _prefix_evaluator_payload_paths(serialized, "lane-7")
    stage = prefixed["stages"][0]
    assert stage["evaluator"]["chunks"][0]["application_path"] == (
        "lane-7/evaluators/left.symjit"
    )
    assert stage["compiled_plane_arena"]["leaves"][0]["application_path"] == (
        "lane-7/evaluators/left.symjit"
    )
