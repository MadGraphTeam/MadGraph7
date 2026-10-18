# SPDX-License-Identifier: 0BSD
"""Dependency JIT conformance at the typed native coefficient-schedule boundary."""

import cmath

import pytest
from _loop_reduction import CoefficientPlan, NativeBatch
from symbolica import E, Expression, S


def saved(expressions, parameters):
    return Expression.evaluator_multiple(
        expressions, parameters, n_cores=1, jit_compile=False
    ).save()


def test_jit_complex_schedule_matches_interpreter_with_changing_tails_and_constants():
    x, y = S("jit_schedule_x", "jit_schedule_y")
    first = saved([x * y + E("2i"), x.conj() / y + E("pi")], [x, y])
    second = saved([x * x + y, (x + E("1")).sqrt()], [x, y])
    args = ([first, second], [[0, 1], [2, 3]], 2, [2, 3, 4, 5])
    oracle = CoefficientPlan(*args)
    jit = CoefficientPlan(*args, engine="jit")
    assert oracle.engine == "interpreted" and jit.engine == "jit"
    for rows in (1, 4, 5, 33, 128, 0, 1):
        values = [
            [str(component.real), str(component.imag)]
            for row in range(rows)
            for component in [complex(1 + row / 100, 0.3), complex(2, -0.7)]
        ]
        batch = NativeBatch.from_decimal(
            values, rows, 2, row_ids=[501 - i for i in range(rows)]
        )
        actual = jit.evaluate_batch(batch)
        reference = oracle.evaluate_batch(batch)
        assert actual.row_ids == reference.row_ids == batch.row_ids
        assert actual.precision_bits == 53
        for pair, target in zip(
            actual.to_decimal(), reference.to_decimal(), strict=True
        ):
            value = complex(*map(float, pair))
            expected = complex(*map(float, target))
            assert cmath.isclose(value, expected, rel_tol=2e-14, abs_tol=2e-14)


def test_jit_constant_nodes_empty_batches_and_empty_program():
    state = saved([E("2i/3"), E("pi")], [])
    jit = CoefficientPlan([state], [[]], 0, [0, 1], engine="jit")
    for rows in (3, 1, 0, 5):
        batch = NativeBatch.from_decimal([], rows, 0)
        result = jit.evaluate_batch(batch)
        assert result.rows == rows and result.columns == 2
        for row in range(rows):
            values = result.to_decimal()[2 * row : 2 * row + 2]
            assert complex(*map(float, values[0])) == complex(0, 2 / 3)
            assert float(values[1][0]) == pytest.approx(cmath.pi, rel=1e-15)
    passthrough = CoefficientPlan([], [], 1, [0], engine="jit")
    batch = NativeBatch.from_decimal([["1", "2"]], 1, 1, row_ids=[8])
    assert passthrough.evaluate_batch(batch).to_decimal() == batch.to_decimal()


def test_jit_rejects_precision_demotion_and_invalid_schedule():
    x = S("jit_shape_x")
    state = saved([x], [x])
    for arithmetic, precision in [("double-double", None), ("arbitrary", 80)]:
        with pytest.raises(ValueError, match="JIT requires double"):
            CoefficientPlan([state], [[0]], 1, [1], arithmetic, precision, engine="jit")
    with pytest.raises(ValueError, match="engine"):
        CoefficientPlan([state], [[0]], 1, [1], engine="unknown")
    with pytest.raises(ValueError, match="forward reference"):
        CoefficientPlan([state], [[1]], 1, [1], engine="jit")
    jit = CoefficientPlan([state], [[0]], 1, [1], engine="jit")
    dd = NativeBatch.from_decimal([["1", "0"]], 1, 1, "double-double")
    with pytest.raises(ValueError, match="precision"):
        jit.evaluate_batch(dd)
