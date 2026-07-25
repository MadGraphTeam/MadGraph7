# SPDX-License-Identifier: 0BSD
"""Fail-closed contracts for the native compiled DirectApplication producer."""

from __future__ import annotations

import pytest

from pyamplicol._internal.physics.types import NativeEvaluationError
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppParameterKind as ParameterKind,
)
from pyamplicol.evaluators.native_direct_cpp import (
    NativeDirectCppSpec,
    compiler_from_symbolica_settings,
    render_native_direct_cpp,
)


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


def test_compiler_settings_reject_inline_asm_and_retain_cpp_flags() -> None:
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

    with pytest.raises(NativeEvaluationError, match="cannot reuse an inline-ASM"):
        compiler_from_symbolica_settings(
            {
                "compiled_inline_asm": "default",
            }
        )
