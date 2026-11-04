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
from pyamplicol.evaluators.symbolica_helpers import _symbolica_instruction_program_repr
from pyamplicol.loops.scalar_frontend import _producer_predicate


def _primitive(name: str) -> Any:
    return symbols.qualified(
        f"pyamplicol_scalar_v1::{name}", eval={"complex": _producer_predicate(name)}
    )


def test_helper_bodies_inline_and_evaluate_as_symbolica_does() -> None:
    """Dyadic inputs keep every value exact, so the comparison is bitwise."""
    from symbolica import Expression, FunctionDefinition, S

    equal, truth, less = (_primitive(name) for name in ("equal", "truth", "less"))
    choose = symbols.qualified("symbolica::if")
    x, y, a, b = (S(f"symjit_helper_{name}") for name in "xyab")
    helper = symbols.qualified("pyamplicol_symjit_helper_test::helper")
    body = choose(equal(a, b), a, 3 * b) + choose(truth(less(a, b)), b * b, a)
    evaluator = Expression.evaluator_multiple(
        [
            helper(x, y) + 2 * helper(y, y),
            _primitive("real_part")(x) * helper(x, x + y),
        ],
        [x, y],
        functions=[FunctionDefinition(helper, [a, b], body, inlining="never")],
        jit_compile=False,
        n_cores=1,
    )
    exported = evaluator.get_instructions()
    assert exported.sub_evaluators
    program = _symbolica_instruction_program_repr(exported)
    assert "'pyamplicol_scalar_v1::equal'" in program

    application = _rusticol._compile_symjit_plane_application_v2(
        program, 2, 2, 2, False
    )
    rows = np.array(
        [
            [1.5, 1.5],
            [1.5, 2.5],
            [2.5, 1.5],
            [1 + 2j, 1 + 2j],
            [1 + 2j, 1 - 2j],
            [0.0, -0.0],
            [3j, 3.0],
            [-4.0, -4.0],
            [0.25, 8.0],
        ],
        dtype=np.complex128,
    )
    expected = np.asarray(evaluator.evaluate_complex(rows), dtype=np.complex128)
    flat = _rusticol._evaluate_symjit_plane_application_v2(
        application, 2, 2, rows.view(np.float64).ravel().tolist(), len(rows)
    )
    values = np.asarray(flat, dtype=np.float64).view(np.complex128).reshape(-1, 2)
    assert np.isfinite(expected).all()
    assert values.tobytes() == expected.tobytes()


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
    # The helpers select on a predicate: the plane check certifies their
    # real-axis branches at real points of the program (the real rows).
    real = [[z.real for z in row] for row in rows if not any(z.imag for z in row)]
    rust_application, inputs, outputs = load_loop_native().compile_v2_plane(
        bytes(evaluator.save()), 2, False, real
    )
    assert (inputs, outputs) == (2, 2)
    for application in (python_application, bytes(rust_application)):
        flat = _rusticol._evaluate_symjit_plane_application_v2(
            application, 2, 2, flat_rows, len(rows)
        )
        values = np.asarray(flat, dtype=np.float64).view(np.complex128)
        assert values.reshape(-1, 2).tobytes() == expected.tobytes()


def test_untaken_branches_define_no_shared_values() -> None:
    """A value computed in one branch of a helper's conditional and twice in
    straight-line code (other helpers' bodies): SymJIT's common-subexpression
    elimination (2.27.3 to 2.28.2) assigned the shared value at its first
    occurrence, inside the branch, so where the branch is not taken the
    straight-line uses read an unset temporary (found while localising
    CM01's zero-width defect, M-SM MS-MASTERS;
    ``upstream/symjit-cse-across-branches``, fixed in 2.28.3 by
    siravan/symjit#31). Both translations keep the jumps, and the planes
    equal Symbolica on either side of the condition, at the parameter
    program's level 0 and at level 2, with the plane check given only the
    taken side."""
    from symbolica import Evaluator, Expression, FunctionDefinition, S

    from pyamplicol.loops.native import load_loop_native

    truth = _primitive("truth")
    choose = symbols.qualified("symbolica::if")
    c, x, y, z = (S(f"symjit_cse_{name}") for name in "cxyz")
    k, a, b, d = (S(f"symjit_cse_arg_{name}") for name in "kabd")
    f, g, h = (
        symbols.qualified(f"pyamplicol_symjit_helper_test::cse_{name}")
        for name in "fgh"
    )
    shared = (a * b + d) * a
    evaluator = Expression.evaluator_multiple(
        [f(c, x, y, z), g(x, y, z), h(x, y, z)],
        [c, x, y, z],
        functions=[
            FunctionDefinition(
                f, [k, a, b, d], choose(truth(k), shared, b), inlining="never"
            ),
            FunctionDefinition(g, [a, b, d], shared + b, inlining="never"),
            FunctionDefinition(h, [a, b, d], shared + d, inlining="never"),
        ],
        jit_compile=False,
        n_cores=1,
    )
    state = bytes(evaluator.save())
    rows = np.array(
        [[1, 2, 3, 5], [0, 2, 3, 5], [0, 1.5, -2, 0.25]], dtype=np.complex128
    )
    expected = np.asarray(Evaluator.load(state).evaluate_complex(rows))
    assert expected[1].tolist() == [3, 25, 27]
    program = _symbolica_instruction_program_repr(evaluator.get_instructions())
    assert "if_else" in program and "join" in program
    native = load_loop_native()
    for level in (0, 2):
        rust_application, inputs, outputs = native.compile_v2_plane(
            state, level, False, [[1.0, 2.0, 3.0, 5.0]]
        )
        assert (inputs, outputs) == (4, 3)
        python_application = _rusticol._compile_symjit_plane_application_v2(
            program, 4, 3, level, False
        )
        for application in (bytes(rust_application), python_application):
            flat = _rusticol._evaluate_symjit_plane_application_v2(
                application, 4, 3, rows.view(np.float64).ravel().tolist(), len(rows)
            )
            values = np.asarray(flat, dtype=np.float64).view(np.complex128)
            assert values.reshape(-1, 3).tobytes() == expected.tobytes(), (
                level,
                values,
            )


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


def test_registered_constants_export_their_binary64_values() -> None:
    """Symbolica's instruction export holds 0 in the slot of a registered
    constant (pi) and names the slot in ``constant_functions``; the plane
    exporter writes the binary64 value there, so the stored application
    equals Symbolica's evaluator."""
    import numpy as np
    from symbolica import E, S

    from pyamplicol import _rusticol

    x = S("symjit_constant_x")
    adapter = _plane_adapter([E("symbolica::pi") * x + 1, x * x], [x])
    exported = adapter._source_evaluator.get_instructions()
    (constant,) = exported.constant_functions
    assert repr(exported.constants[constant.index]) == "0"
    application, _digest, _target = adapter._export_symjit_plane_application(
        optimization_level=2
    )
    rows = np.ascontiguousarray([[0.5 + 0.25j], [-1.75 + 3j]], dtype=np.complex128)
    expected = np.asarray(
        adapter._source_evaluator.evaluate_complex(rows), dtype=np.complex128
    )
    values = np.asarray(
        _rusticol._evaluate_symjit_plane_application_v2(
            application, 1, 2, rows.view(np.float64).ravel().tolist(), len(rows)
        )
    ).view(np.complex128)
    np.testing.assert_allclose(values, expected.reshape(values.shape), rtol=1e-15)


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


def test_plane_applications_calling_primitives_are_byte_reproducible() -> None:
    """Two compilations of a program calling a scalar primitive save the same
    bytes: SymJIT 2.28.1 and later tag external calls deterministically and
    write their sets and subroutines in a fixed order (siravan/symjit#26;
    2.27.3 tagged every call with ``rand::random()``)."""
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


def test_only_whole_state_saves_are_reencoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``saved_evaluator`` hands a save whose state header says "whole state"
    (a Python Symbolica without symbolica-dev#163, PyPI 3.0.3) to the native
    and returns its state-minimal form; any other save, the installed
    Symbolica's own included, is returned as it is, without the native."""

    from symbolica import Expression, S

    from pyamplicol.evaluators import symbolica_helpers as helpers

    calls: list[bytes] = []

    class Native:
        @staticmethod
        def state_minimal_saved_evaluator(saved: bytes) -> bytes:
            calls.append(saved)
            return b"state-minimal"

    class Saved:
        def __init__(self, data: bytes) -> None:
            self.data = data

        def save(self) -> bytes:
            return self.data

    monkeypatch.setattr("pyamplicol.loops.native.load_loop_native", lambda: Native)
    header = helpers._SAVED_EVALUATOR_MAGIC + b"\x01" + b"\x00" * 6
    whole, partial = header + b"\x01payload", header + b"\xfepayload"
    assert helpers.saved_evaluator(Saved(partial)) == partial
    assert helpers.saved_evaluator(Saved(b"raw bincode")) == b"raw bincode"
    x = S("pyamplicol_saved_evaluator_test::x")
    installed = Expression.evaluator_multiple([x + 1], [x], jit_compile=False)
    saved = bytes(installed.save())
    if saved[helpers._WHOLE_STATE_FLAG] != 1:
        assert helpers.saved_evaluator(installed) == saved
    assert not calls
    assert helpers.saved_evaluator(Saved(whole)) == b"state-minimal"
    assert calls == [whole]
