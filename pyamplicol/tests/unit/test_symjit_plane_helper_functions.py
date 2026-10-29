# SPDX-License-Identifier: 0BSD
"""SymJIT plane applications of programs with a model's helper functions.

A model helper function reaches the SymJIT exporter as a Symbolica
sub-evaluator built on pyAmpliCol's scalar primitives
(``pyamplicol_scalar_v1::*``) and ``symbolica::if``. The exporter inlines it;
the plane compiler and loader bind the primitives by name, so the stored
application evaluates without Symbolica state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyamplicol import _rusticol
from pyamplicol._internal.physics.symbols import symbols
from pyamplicol._internal.physics.types import NativeEvaluationError
from pyamplicol.evaluators.symbolica_helpers import _symbolica_instruction_program_repr
from pyamplicol.loops.scalar_frontend import _producer_predicate


def _primitive(name: str) -> Any:
    return symbols.qualified(
        f"pyamplicol_scalar_v1::{name}", eval={"complex": _producer_predicate(name)}
    )


def _nested_helpers_evaluator() -> tuple[Any, Any]:
    """Three helpers, each calling the previous one, none inlined by
    Symbolica: the exporters inline bodies nested three deep."""
    from symbolica import Expression, FunctionDefinition, S

    equal = _primitive("equal")
    choose = symbols.qualified("symbolica::if")
    x, y, a, b = (S(f"symjit_nested_{name}") for name in "xyab")
    inner, middle, outer = (
        symbols.qualified(f"pyamplicol_symjit_helper_test::nested_{name}")
        for name in ("inner", "middle", "outer")
    )
    functions = [
        FunctionDefinition(
            inner, [a, b], choose(equal(a, b), a, 3 * b), inlining="never"
        ),
        FunctionDefinition(
            middle, [a, b], inner(a, b) * a + inner(b, a), inlining="never"
        ),
        FunctionDefinition(
            outer, [a, b], middle(a, b) - 2 * middle(b, b), inlining="never"
        ),
    ]
    evaluator = Expression.evaluator_multiple(
        [outer(x, y) + y, middle(x, x) * outer(y, x)],
        [x, y],
        functions=functions,
        jit_compile=False,
        n_cores=1,
    )
    rows = np.array(
        [[1.5, 1.5], [1.5, 2.5], [2.5, -1.5], [1 + 2j, 1 + 2j], [0.25, 8.0]],
        dtype=np.complex128,
    )
    return evaluator, rows


def _depth(exported: Any) -> int:
    return 1 + max(
        (_depth(sub.evaluator) for sub in exported.sub_evaluators), default=0
    )


def test_nested_helper_bodies_inline_in_both_exporters() -> None:
    """Dyadic inputs keep every value exact, so the comparisons are bitwise:
    the Python exporter (tree adapter) and the Rust one (loop plane
    applications, from the saved state) inline the nested bodies."""
    from pyamplicol.loops.native import load_loop_native

    evaluator, rows = _nested_helpers_evaluator()
    exported = evaluator.get_instructions()
    assert _depth(exported) >= 3
    expected = np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)
    assert np.isfinite(expected).all()
    flat_rows = rows.view(np.float64).ravel().tolist()

    program = _symbolica_instruction_program_repr(exported)
    python_application = _rusticol._compile_symjit_plane_application_v2(
        program, 2, 2, 2, False
    )
    rust_application, inputs, outputs = load_loop_native().compile_v2_plane(
        bytes(evaluator.save()), 2, False
    )
    assert (inputs, outputs) == (2, 2)
    for application in (python_application, bytes(rust_application)):
        flat = _rusticol._evaluate_symjit_plane_application_v2(
            application, 2, 2, flat_rows, len(rows)
        )
        values = np.asarray(flat, dtype=np.float64).view(np.complex128)
        assert values.reshape(-1, 2).tobytes() == expected.tobytes()


def test_unbound_functions_are_refused_by_the_plane_compiler() -> None:
    from symbolica import Expression, FunctionDefinition, S

    x, a = S("symjit_helper_unbound_x"), S("symjit_helper_unbound_a")
    opaque = symbols.qualified(
        "pyamplicol_symjit_helper_test::opaque", eval={"complex": lambda *_: 1}
    )
    helper = symbols.qualified("pyamplicol_symjit_helper_test::outer")
    evaluator = Expression.evaluator_multiple(
        [helper(x) + x],
        [x],
        functions=[FunctionDefinition(helper, [a], opaque(a) * a, inlining="never")],
        jit_compile=False,
        n_cores=1,
    )
    program = _symbolica_instruction_program_repr(evaluator.get_instructions())
    assert "'pyamplicol_symjit_helper_test::opaque'" in program
    with pytest.raises(Exception, match="opaque"):
        _rusticol._compile_symjit_plane_application_v2(program, 1, 1, 2, False)


def _plane_adapter(outputs: list[Any], parameters: list[Any]) -> Any:
    from symbolica import Expression

    from pyamplicol.evaluators.symbolica_adapters import _JITSymbolicaEvaluatorAdapter
    from pyamplicol.evaluators.symbolica_settings import SymbolicaEvaluatorSettings

    evaluator = Expression.evaluator_multiple(
        outputs, parameters, jit_compile=False, n_cores=1
    )
    settings = SymbolicaEvaluatorSettings(
        backend="jit", jit_optimization_level=2, n_cores=1
    )
    return _JITSymbolicaEvaluatorAdapter(
        evaluator,
        settings,
        "plane constants",
        input_len=len(parameters),
        output_len=len(outputs),
    )


def test_registered_constants_read_as_zero_are_refused() -> None:
    """Symbolica's Python export writes pi as the constant 0; the plane
    exporter refuses the program instead of storing wrong values."""
    from symbolica import E, S

    x = S("symjit_constant_x")
    adapter = _plane_adapter([E("symbolica::pi") * x + 1, x * x], [x])
    assert [repr(c) for c in adapter._source_evaluator.get_instructions().constants]
    with pytest.raises(NativeEvaluationError, match="registered constants"):
        adapter._export_symjit_plane_application(optimization_level=2)


def test_true_zero_constants_export() -> None:
    from symbolica import E, S

    x, y = S("symjit_zero_x"), S("symjit_zero_y")
    adapter = _plane_adapter([x * y + 2, E("0"), x], [x, y])
    constants = adapter._source_evaluator.get_instructions().constants
    assert "0" in [repr(c) for c in constants]
    application, _digest, _target = adapter._export_symjit_plane_application(
        optimization_level=2
    )
    assert application


SYMJIT_RANDOM_CALL_TAG = (
    "SymJIT 2.27.3 tags every external call with a rand::random() constant "
    "(compiler.rs translate_external_fun, against CSE), so the stored IR of a "
    "program calling the scalar primitives differs between compilations; the "
    "values do not, and a stored application recompiles to the same code at "
    "every load (still in SymJIT 2.28.1 and v228 d37160b8; deterministic-tag "
    "patch prepared in workspace/reports/engine-v2/upstream/"
    "symjit-reproducible-external-calls, not filed yet)"
)


@pytest.mark.xfail(strict=True, reason=SYMJIT_RANDOM_CALL_TAG)
def test_plane_applications_calling_primitives_are_byte_reproducible() -> None:
    from symbolica import Expression, S

    x, y = S("symjit_reproducible_x"), S("symjit_reproducible_y")
    evaluator = Expression.evaluator_multiple(
        [_primitive("equal")(x, y) + x], [x, y], jit_compile=False, n_cores=1
    )
    program = _symbolica_instruction_program_repr(evaluator.get_instructions())
    first = _rusticol._compile_symjit_plane_application_v2(program, 2, 1, 2, False)
    assert first == _rusticol._compile_symjit_plane_application_v2(
        program, 2, 1, 2, False
    )
