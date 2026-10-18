# SPDX-License-Identifier: 0BSD
"""Typed native schedule/transport/reducer regressions without decimal intermediates."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import (
    CoefficientEvaluator,
    CoefficientPlan,
    NativeBatch,
    PolynomialPlan,
    massless_spinors_batch,
    massless_spinors_decimal,
)
from symbolica import E, S

MODES = [("double", None), ("double-double", None), ("arbitrary", 80)]


def schedule(arithmetic, precision):
    x, y = S("native_batch_x", "native_batch_y")
    first = (x - E("1")).evaluator([x], n_cores=1, jit_compile=False).save()
    second = (x * y).evaluator([x, y], n_cores=1, jit_compile=False).save()
    return CoefficientPlan(
        [first, second], [[0], [2, 1]], 2, [3], arithmetic, precision
    )


@pytest.mark.parametrize("arithmetic,precision", MODES)
def test_native_schedule_to_reducer_preserves_bits_ids_and_changing_batches(
    arithmetic, precision
):
    dag = schedule(arithmetic, precision)
    reducer = PolynomialPlan(1, ["c", "2", "0"], ["c"], arithmetic, precision)

    def batch(values, rows, columns, ids):
        return NativeBatch.from_decimal(
            values, rows, columns, arithmetic, precision, ids
        )

    with localcontext() as context:
        context.prec = 180
        tiny = (
            Decimal("0.25")
            if arithmetic == "double"
            else Decimal(2) ** -(80 if arithmetic == "double-double" else 180)
        )
        for rows in [1, 33, 0, 8, 1]:
            ids = [1001 - i * 7 for i in range(rows)]
            inputs = batch(
                [
                    value
                    for i in range(rows)
                    for value in [[str(1 + tiny), "0"], [str(i + 1), "0"]]
                ],
                rows,
                2,
                ids,
            )
            coefficients = dag.evaluate_batch(inputs)
            assert coefficients.row_ids == ids
            reduced = reducer.evaluate_batch(
                batch([["0", "0"]] * (4 * rows), rows, 4, ids),
                batch([["1", "0"]] * rows, rows, 1, ids),
                batch([["1", "0"]] * rows, rows, 1, ids),
                coefficients,
                s_matrix=batch([["0", "0"]] * rows, rows, 1, ids),
            )
            assert reduced.statuses == [(i, "success") for i in ids]
            values = reduced.values
            assert values.row_ids == ids
            for i, (real, imag) in enumerate(values.take_columns([1]).to_decimal(), 1):
                assert Decimal(imag) == 0
                assert abs(Decimal(real) / (tiny * i) - 1) < Decimal("0.00001")
            assert reduced.successful_values().to_decimal() == values.to_decimal()


@pytest.mark.parametrize("arithmetic,precision", MODES)
def test_native_row_column_transport_and_spinors(arithmetic, precision):
    values = [[str(i), "0"] for i in range(12)]
    batch = NativeBatch.from_decimal(values, 3, 4, arithmetic, precision, [91, 7, 42])
    selected = batch.take_rows([2, 0]).take_columns([3, 1])
    assert selected.row_ids == [42, 91]
    assert [Decimal(x[0]) for x in selected.to_decimal()] == [11, 9, 3, 1]
    joined = NativeBatch.concat_columns([selected, selected.take_columns([1])])
    assert joined.row_ids == [42, 91]
    assert [Decimal(x[0]) for x in joined.to_decimal()] == [11, 9, 9, 3, 1, 1]
    reshaped = joined.reshape_rows(1, 6, [88])
    assert reshaped.row_ids == [88]
    assert reshaped.to_decimal() == joined.to_decimal()
    repeated = selected.take_rows([1, 1, 0], row_ids=[1, 2, 3])
    assert repeated.row_ids == [1, 2, 3]
    assert repeated.rows == 3
    x = S("native_batch_identity_x")
    evaluator = CoefficientEvaluator(
        (x * x).evaluator([x], n_cores=1, jit_compile=False).save(),
        arithmetic,
        precision,
    )
    assert [
        Decimal(v[0])
        for v in evaluator.evaluate_batch(selected.take_columns([1])).to_decimal()
    ] == [81, 1]
    momenta = [["13", "3", "4", "-12"], ["5", "0", "0", "-5"]]
    states = NativeBatch.from_decimal(
        [[v, "0"] for p in momenta for v in p], 2, 4, arithmetic, precision, [8, 2]
    )
    result = massless_spinors_batch(states, [1, -1], ["u", "v_bar"])
    expected = massless_spinors_decimal(
        momenta, [1, -1], ["u", "v_bar"], arithmetic, precision
    )
    assert result.row_ids == [8, 2]
    assert result.to_decimal() == [v for p in expected for v in p]


def test_native_schedule_and_buffer_invariants_fail_before_execution():
    x = S("native_batch_shape_x")
    state = x.evaluator([x], n_cores=1, jit_compile=False).save()
    for inputs, outputs in [([[1]], [1]), ([[]], [1]), ([[0]], [2])]:
        with pytest.raises(ValueError):
            CoefficientPlan([state], inputs, 1, outputs)
    with pytest.raises(ValueError):
        CoefficientPlan([state], [], 1, [1])
    batch = NativeBatch.from_decimal([["1", "0"]], 1, 1, row_ids=[9])
    different_ids = NativeBatch.from_decimal([["1", "0"]], 1, 1, row_ids=[8])
    dd = NativeBatch.from_decimal([["1", "0"]], 1, 1, "double-double", row_ids=[9])
    for other in [different_ids, dd]:
        with pytest.raises(ValueError):
            NativeBatch.concat_columns([batch, other])
    for operation in [
        lambda: batch.take_rows([1]),
        lambda: batch.take_columns([1]),
        lambda: batch.reshape_rows(2, 1, [1, 2]),
        lambda: batch.take_rows([0], row_ids=[]),
    ]:
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ValueError):
        NativeBatch.from_decimal([], 1, 1)
    plan = CoefficientPlan([state], [[0]], 1, [1])
    with pytest.raises(ValueError):
        plan.evaluate_batch(dd)
    with pytest.raises(ValueError):
        CoefficientEvaluator(state).evaluate_batch(dd)
    reducer = PolynomialPlan(1, ["c", "0", "0"], ["c"])
    shifts = NativeBatch.from_decimal([["0", "0"]] * 4, 1, 4, row_ids=[9])
    with pytest.raises(ValueError, match="identities"):
        reducer.evaluate_batch(shifts, batch, different_ids, batch)
    with pytest.raises(ValueError, match="dimensions"):
        reducer.evaluate_batch(shifts, batch, batch, batch, s_matrix=shifts)
    with pytest.raises(ValueError, match="arithmetic"):
        reducer.evaluate_batch(shifts, batch, batch, dd)


@pytest.mark.parametrize("arithmetic,precision", MODES)
def test_native_nonfinite_scan_preserves_row_ids_without_numeric_conversion(
    arithmetic, precision
):
    batch = NativeBatch.from_decimal(
        [["1", "0"], ["2", "0"], ["NaN", "0"], ["0", "0"], ["1", "inf"], ["-inf", "0"]],
        3,
        2,
        arithmetic,
        precision,
        [91, 7, 42],
    )
    assert batch.nonfinite_rows() == [7, 42]
    assert batch.take_rows([2, 0, 1]).nonfinite_rows() == [42, 7]
    assert batch.take_rows([0]).nonfinite_rows() == []
    assert batch.take_rows([]).nonfinite_rows() == []
    assert batch.take_columns([]).nonfinite_rows() == []


@pytest.mark.parametrize("engine", ["generic", "fast"])
def test_failed_reduction_lanes_are_hidden_and_successful_compaction_retains_ids(
    engine,
):
    reducer = PolynomialPlan(1, ["c", "0", "0"], ["c"], engine=engine)

    def batch(values, columns):
        return NativeBatch.from_decimal(values, 2, columns, row_ids=[9, 3])

    shifts = batch([["0", "0"]] * 8, 4)
    masses = batch([["1", "0"]] * 2, 1)
    good = batch([["1", "0"], ["2", "0"]], 1)
    assert reducer.evaluate_batch(shifts, masses, masses, good).values.row_ids == [9, 3]
    failed = reducer.evaluate_batch(
        shifts, masses, masses, batch([["1", "0"], ["NaN", "0"]], 1)
    )
    assert failed.statuses[0] == (9, "success")
    assert failed.statuses[1][0] == 3
    assert "NonFinite" in failed.statuses[1][1]
    assert failed.successful_values().row_ids == [9]
    with pytest.raises(ValueError, match="failed lanes"):
        _ = failed.values
    retry = reducer.evaluate_batch(
        shifts.take_rows([1]),
        masses.take_rows([1]),
        masses.take_rows([1]),
        good.take_rows([1]),
    )
    assert retry.statuses == [(3, "success")]
    assert retry.values.row_ids == [3]
    assert [Decimal(v[0]) for v in retry.values.to_decimal()] == [2, 2, 0]
