# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyamplicol._internal.physics.types import NativeEvaluationError
from pyamplicol._internal.versions import (
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
)
from pyamplicol.evaluators.execution_schema import evaluator_runtime_capabilities
from pyamplicol.evaluators.symbolica_adapters import (
    _ChunkedSymbolicaEvaluator,
    _compiled_runtime_capability,
    _JITSymbolicaEvaluatorAdapter,
)
from pyamplicol.evaluators.symbolica_compile import _chunk_parameter_indices
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from pyamplicol.generation.artifact_writer import _evaluator, _stage_evaluator_set

ROOT = Path(__file__).resolve().parents[2]


class _FakeJITEvaluator:
    def __init__(self, *, export_error: Exception | None = None) -> None:
        self.export_error = export_error
        self.evaluation_count = 0

    def evaluate_complex(self, rows: Any) -> np.ndarray:
        self.evaluation_count += 1
        return np.zeros((len(rows), 2), dtype=np.complex128)

    def export_symjit(self, *, complex: bool = False) -> bytes:
        assert complex is True
        if self.export_error is not None:
            raise self.export_error
        return b"symjit-application-v3"

    def save(self) -> bytes:
        return b"symbolica-evaluator-state"


class _MappedEvaluator:
    def __init__(self, input_len: int) -> None:
        self.input_len = input_len

    def evaluate_complex(self, rows: Any) -> np.ndarray:
        values = np.asarray(rows, dtype=np.complex128)
        return values.sum(axis=1, keepdims=True)


def _jit_adapter(
    evaluator: object | None = None,
    *,
    direct_translation: bool = False,
) -> _JITSymbolicaEvaluatorAdapter:
    return _JITSymbolicaEvaluatorAdapter(
        _FakeJITEvaluator() if evaluator is None else evaluator,
        SymbolicaEvaluatorSettings(
            backend="jit",
            jit_direct_translation=direct_translation,
            jit_optimization_level=3,
            n_cores=1,
        ),
        "test stage",
        input_len=3,
        output_len=2,
    )


def test_jit_artifact_persists_direct_application_and_precision_fallback(
    tmp_path: Path,
) -> None:
    source = _FakeJITEvaluator()
    manifest = _jit_adapter(source).artifact_manifest(tmp_path)

    assert source.evaluation_count == 1
    assert manifest["kind"] == "symjit-application-evaluator"
    assert manifest["runtime_capability"] == SYMJIT_F64_RUNTIME_CAPABILITY
    assert manifest["application_abi"] == SYMJIT_APPLICATION_ABI
    assert manifest["batch_layout"] == "row-major"
    assert manifest["compiler_type"] == "native"
    assert manifest["translation_mode"] == "indirect"
    assert manifest["optimization_level"] == 3
    assert manifest["word_bits"] == 64
    assert manifest["endianness"] == "little"
    assert manifest["required_defuns"] == []
    assert manifest["evaluator_state_runtime_capability"] == (
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
    )
    assert (tmp_path / str(manifest["application_path"])).read_bytes() == (
        b"symjit-application-v3"
    )
    assert (tmp_path / str(manifest["evaluator_state_path"])).read_bytes() == (
        b"symbolica-evaluator-state"
    )
    assert evaluator_runtime_capabilities(manifest) == (SYMJIT_F64_RUNTIME_CAPABILITY,)


def test_jit_artifact_rejects_non_self_contained_export(tmp_path: Path) -> None:
    adapter = _jit_adapter(
        _FakeJITEvaluator(export_error=ValueError("external defuns present"))
    )
    with pytest.raises(NativeEvaluationError, match="external evaluator functions"):
        adapter.artifact_manifest(tmp_path)


def test_jit_artifact_rejects_direct_translation(tmp_path: Path) -> None:
    adapter = _jit_adapter(direct_translation=True)
    with pytest.raises(NativeEvaluationError, match="cannot be persisted"):
        adapter.artifact_manifest(tmp_path)


def test_chunked_evaluator_aggregates_primary_capabilities(tmp_path: Path) -> None:
    manifest = _ChunkedSymbolicaEvaluator(
        (_jit_adapter(), _jit_adapter())
    ).artifact_manifest(tmp_path)

    assert manifest["required_runtime_capabilities"] == [SYMJIT_F64_RUNTIME_CAPABILITY]
    assert manifest["input_len"] == 3
    assert manifest["chunk_input_indices"] == [[0, 1, 2], [0, 1, 2]]
    assert evaluator_runtime_capabilities(manifest) == (SYMJIT_F64_RUNTIME_CAPABILITY,)
    serialized = _evaluator(manifest)
    assert serialized["input_len"] == 3
    assert serialized["chunk_input_indices"] == [[0, 1, 2], [0, 1, 2]]


def test_chunked_evaluator_selects_parent_inputs_per_chunk() -> None:
    evaluator = _ChunkedSymbolicaEvaluator(
        (_MappedEvaluator(2), _MappedEvaluator(1)),
        input_len=3,
        chunk_input_indices=((0, 2), (1,)),
    )

    values = evaluator.evaluate_complex(
        np.asarray([[1.0, 10.0, 3.0], [2.0, 20.0, 5.0]], dtype=np.complex128)
    )

    assert values.tolist() == [[4.0 + 0.0j, 10.0 + 0.0j], [7.0 + 0.0j, 20.0 + 0.0j]]


def test_symbolica_chunk_dependencies_preserve_parent_parameter_order() -> None:
    from symbolica import S

    x, y, z, closed, argument = S("chunk_x", "chunk_y", "chunk_z", "closed", "arg")
    function = S("chunk_function")

    assert _chunk_parameter_indices(
        (z + function(x),),
        (x, y, z, closed),
        functions={(function, (argument,)): argument + closed},
    ) == (0, 2, 3)


def test_compiled_capability_distinguishes_cpp_and_asm() -> None:
    assert _compiled_runtime_capability({"compiled_inline_asm": "none"}) == (
        SYMBOLICA_CPP_RUNTIME_CAPABILITY
    )
    assert _compiled_runtime_capability({"compiled_inline_asm": "default"}) == (
        SYMBOLICA_ASM_RUNTIME_CAPABILITY
    )


def test_artifact_writer_preserves_direct_symjit_contract(tmp_path: Path) -> None:
    generated = _jit_adapter().artifact_manifest(tmp_path)
    serialized = _evaluator(generated)

    assert set(serialized) == {
        "application_abi",
        "application_path",
        "batch_layout",
        "compiler_type",
        "element_layout",
        "endianness",
        "evaluator_state_path",
        "evaluator_state_runtime_capability",
        "input_len",
        "kind",
        "optimization_level",
        "output_len",
        "required_defuns",
        "runtime_capability",
        "translation_mode",
        "word_bits",
    }
    assert serialized["application_path"] == generated["application_path"]
    assert serialized["runtime_capability"] == SYMJIT_F64_RUNTIME_CAPABILITY


def test_stage_manifest_verifies_aggregated_runtime_capabilities(
    tmp_path: Path,
) -> None:
    generated = _jit_adapter().artifact_manifest(tmp_path)
    amplitude_stage = {
        "stage_index": 0,
        "stage_kind": "amplitude",
        "subset_size": None,
        "evaluator_label": "amplitude",
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 2,
        "output_slots": [],
        "input_value_slot_ids": [],
        "output_value_slot_ids": [],
        "interaction_ids": [],
        "input_components": [],
        "parameter_count": 3,
        "value_parameter_count": 0,
        "momentum_parameter_count": 3,
        "model_parameter_count": 0,
        "real_valued_inputs": [0, 1, 2],
        "expression_ready": True,
        "blockers": [],
        "evaluator": generated,
    }
    stage_set = {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 1,
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
        "stages": [],
        "amplitude_stage": amplitude_stage,
    }

    serialized = _stage_evaluator_set(stage_set)
    assert serialized["required_runtime_capabilities"] == [
        SYMJIT_F64_RUNTIME_CAPABILITY
    ]

    stage_set["required_runtime_capabilities"] = [SYMBOLICA_CPP_RUNTIME_CAPABILITY]
    with pytest.raises(ValueError, match="do not match evaluator payloads"):
        _stage_evaluator_set(stage_set)


def test_symjit_application_abi_matches_contributor_contract() -> None:
    lock = tomllib.loads(
        (ROOT / "dependencies" / "contributor-lock.toml").read_text(
            encoding="utf-8"
        )
    )
    assert lock["abis"]["symjit_application"] == SYMJIT_APPLICATION_ABI
