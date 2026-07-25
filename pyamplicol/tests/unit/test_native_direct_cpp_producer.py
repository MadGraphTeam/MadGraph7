# SPDX-License-Identifier: 0BSD
"""Fail-closed contracts for the native compiled DirectApplication producer."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyamplicol._internal.physics.types import NativeEvaluationError
from pyamplicol._internal.versions import (
    NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
)
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppArtifact,
    NativeDirectCppSpec,
    compiler_from_symbolica_settings,
    render_native_direct_cpp,
)
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppParameterKind as ParameterKind,
)
from pyamplicol.evaluators.symbolica_adapters import (
    _CompiledComplexEvaluatorAdapter,
    _native_direct_application_manifest,
)
from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings
from pyamplicol.generation.artifact_writer import _evaluator


class _Evaluator:
    def __init__(self, instructions: list[tuple[object, ...]]) -> None:
        self.instructions = instructions

    def get_instructions(self) -> tuple[list[tuple[object, ...]], int, list[complex]]:
        return self.instructions, 2, [1j]

    def save(self) -> bytes:
        return b"retained-real-evaluator-state"


def _spec(**changes: object) -> NativeDirectCppSpec:
    values: dict[str, object] = {
        "function_name": "retained_leaf",
        "parameter_kinds": (
            ParameterKind.COMPLEX_PLANE,
            ParameterKind.REAL_SCALAR,
        ),
        "output_count": 1,
        "target_triple": "aarch64-apple-darwin",
        "cpu_features": ("neon",),
        "simd_lane_width": 2,
    }
    values.update(changes)
    return NativeDirectCppSpec(**values)  # type: ignore[arg-type]


def test_real_instruction_stream_renders_plane_native_simd_and_scalar_tail() -> None:
    evaluator = _Evaluator(
        [
            ("mul", ("temp", 0), [("param", 0), ("param", 0)], 0),
            ("mul", ("temp", 1), [("param", 1), ("param", 1)], 2),
            ("pow", ("temp", 1), ("temp", 1), -1, True),
            ("add", ("out", 0), [("temp", 0), ("temp", 1)], 0),
        ]
    )

    rendered = render_native_direct_cpp(evaluator, _spec())

    assert rendered.instruction_count == 4
    assert rendered.temporary_count == 2
    assert rendered.input_plane_count == 2
    assert rendered.scalar_input_count == 1
    assert rendered.output_plane_count == 2
    assert rendered.logical_stack_bytes == 96
    assert rendered.target_triple == "aarch64-apple-darwin"
    assert rendered.cpu_features == ("neon",)
    assert rendered.simd_lane_width == 2
    assert "_complexf64(" not in rendered.source
    assert "using DirectVector = double __attribute__((vector_size(16)))" in (
        rendered.source
    )
    assert "evaluate_direct_bundle<DirectVector>" in rendered.source
    assert "evaluate_direct_bundle<double>" in rendered.source
    assert "direct_real_scalar<Lane>(*scalars[0].value)" in rendered.source
    assert 'return "aarch64-apple-darwin";' in rendered.source
    assert 'return "neon";' in rendered.source


def test_unknown_symbolica_operation_fails_without_dense_fallback() -> None:
    evaluator = _Evaluator(
        [
            (
                "fun",
                ("out", 0),
                "sqrt",
                [],
                [("param", 0)],
                False,
            )
        ]
    )

    with pytest.raises(
        NativeEvaluationError,
        match=r"cannot lower Symbolica operation.*refusing a dense-row fallback",
    ):
        render_native_direct_cpp(evaluator, _spec())


def test_real_argument_and_stack_contracts_fail_closed() -> None:
    inconsistent = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 2),
        ]
    )
    with pytest.raises(NativeEvaluationError, match="metadata is inconsistent"):
        render_native_direct_cpp(inconsistent, _spec())

    bounded = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )
    with pytest.raises(NativeEvaluationError, match="bounded stack contract"):
        render_native_direct_cpp(
            bounded,
            _spec(max_logical_stack_bytes=1),
        )


def test_compiler_settings_retain_cpp_and_asm_direct_flags() -> None:
    compiler = compiler_from_symbolica_settings(
        {
            "compiled_inline_asm": "none",
            "compiled_optimization_level": 3,
            "compiled_native": True,
            "compiler_path": "/usr/bin/c++",
            "effective_compiler_flags": ["-fno-math-errno"],
        }
    )
    assert compiler.executable == "/usr/bin/c++"
    assert compiler.optimization_level == 3
    assert compiler.native_arch is True
    assert compiler.extra_flags == ("-fno-math-errno",)

    asm_compiler = compiler_from_symbolica_settings(
        {
            "compiled_inline_asm": "default",
            "compiled_optimization_level": 3,
            "compiled_native": False,
            "compiler_flags": ["-fno-math-errno"],
        }
    )
    assert asm_compiler.optimization_level == 3
    assert asm_compiler.native_arch is False
    assert asm_compiler.extra_flags == ("-fno-math-errno",)


def test_direct_only_companion_identity_survives_evaluator_manifest(
    tmp_path: Path,
) -> None:
    rendered = render_native_direct_cpp(
        _Evaluator(
            [
                ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
            ]
        ),
        _spec(),
    )
    source_path = tmp_path / "retained_leaf.direct.cpp"
    library_path = tmp_path / "libretained_leaf.direct"
    source_path.write_text(rendered.source, encoding="utf-8")
    library_path.write_bytes(b"direct-only-test-library")
    companion = NativeDirectCppArtifact(
        source_path=source_path,
        library_path=library_path,
        source=rendered,
        compiler_command=("c++",),
        compile_seconds=0.125,
    )

    identity = _native_direct_application_manifest(
        companion,
        tmp_path,
        expected_function_name="retained_leaf",
    )
    evaluator = _evaluator(
        {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": "symbolica.compiled-cpp.complex-f64.v1",
            "function_name": "retained_leaf",
            "input_len": 2,
            "output_len": 1,
            "library_path": "libretained_leaf.direct",
            "evaluator_state_path": "retained_leaf.evaluator.bin",
            "number_type": "complex",
            "native_direct_application": identity,
        }
    )

    assert identity["application_abi"] == NATIVE_COMPILED_DIRECT_APPLICATION_ABI
    assert identity["target"] == {
        "triple": "aarch64-apple-darwin",
        "cpu_features": ["neon"],
    }
    assert identity["source_path"] == "retained_leaf.direct.cpp"
    assert identity["library_path"] == "libretained_leaf.direct"
    assert evaluator["native_direct_application"] == identity

    asm_evaluator = _evaluator(
        {
            "kind": "compiled-complex-evaluator",
            "runtime_capability": "symbolica.compiled-asm.complex-f64.v1",
            "function_name": "retained_leaf",
            "input_len": 2,
            "output_len": 1,
            "library_path": "libretained_leaf.direct",
            "evaluator_state_path": "retained_leaf.evaluator.bin",
            "number_type": "complex",
            "native_direct_application": identity,
        }
    )
    assert asm_evaluator["native_direct_application"] == identity


def test_process_stage_adapter_serializes_no_dense_library(tmp_path: Path) -> None:
    source = _Evaluator(
        [
            ("mul", ("out", 0), [("param", 0), ("param", 0)], 0),
        ]
    )
    adapter = _CompiledComplexEvaluatorAdapter(
        source,
        SymbolicaEvaluatorSettings(
            backend="compiled-complex",
            compiled_inline_asm="none",
            compiled_output_dir=str(tmp_path),
        ),
        "direct only stage",
        input_len=1,
        output_len=1,
        compile_dense=False,
    )
    rendered = render_native_direct_cpp(
        source,
        _spec(
            function_name=adapter.function_name,
            parameter_kinds=(ParameterKind.COMPLEX_PLANE,),
        ),
    )
    direct_source = tmp_path / f"{adapter.function_name}.direct.cpp"
    direct_library = tmp_path / f"lib{adapter.function_name}.direct"
    direct_source.write_text(rendered.source, encoding="utf-8")
    direct_library.write_bytes(b"direct-only-test-library")
    adapter.native_direct_application = NativeDirectCppArtifact(
        source_path=direct_source,
        library_path=direct_library,
        source=rendered,
        compiler_command=("c++",),
        compile_seconds=0.125,
    )

    manifest = adapter.artifact_manifest(tmp_path)

    assert manifest["library_path"] == direct_library.name
    assert manifest["source_path"] == direct_source.name
    assert not adapter.library_path.exists()
    assert not adapter.source_path.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [
            direct_source.name,
            direct_library.name,
            adapter.evaluator_state_path.name,
        ]
    )
