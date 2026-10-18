# SPDX-License-Identifier: 0BSD
"""Analytic mu^4 box anchors and automatic sampling-scale lane handling."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import NativeBatch, PolynomialPlan


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,tiny_power,engine",
    [
        ("double", None, "1e-10", 20, "generic"),
        ("double", None, "1e-10", 20, "fast-tensor"),
        ("double-double", None, "1e-27", 80, "generic"),
        ("arbitrary", 80, "1e-70", 180, "generic"),
    ],
)
def test_mu4_box_automatic_sampling_three_precisions(
    arithmetic, precision, tolerance, tiny_power, engine
):
    reducer = PolynomialPlan(
        4, ["c*mu2^2", "0", "0"], ["c"], arithmetic, precision, engine
    )

    def batch(values, rows, columns, ids):
        return NativeBatch.from_decimal(
            values, rows, columns, arithmetic, precision, ids
        )

    with localcontext() as context:
        context.prec = 160
        coefficient = 1 + Decimal(2) ** -tiny_power
        for rows in (3, 1, 0, 5, 1):
            ids = [91 + 7 * i for i in range(rows)]
            shifts = batch(
                [
                    [str(component), "0"]
                    for _ in range(rows)
                    for vector in (
                        (0, 0, 0, 0),
                        (2, 1, 0, 0),
                        (3, 0, 1, 0),
                        (4, 0, 0, 1),
                    )
                    for component in vector
                ],
                rows,
                16,
                ids,
            )
            masses = batch([["1", "-0.1"]] * (4 * rows), rows, 4, ids)
            scales = batch([["1", "0"]] * rows, rows, 1, ids)
            parameters = batch([[str(coefficient), "0"]] * rows, rows, 1, ids)
            result = reducer.evaluate_batch(shifts, masses, scales, parameters)
            assert result.statuses == [(identity, "success") for identity in ids]
            assert result.values.row_ids == ids
            values = result.values.to_decimal()
            for row in range(rows):
                # I4[mu^4] = -1/6 in the documented raw Ninja normalization.
                real, imaginary = values[3 * row]
                assert abs(Decimal(real) + coefficient / 6) < Decimal(tolerance)
                assert abs(Decimal(imaginary)) < Decimal(tolerance)
                for pair in values[3 * row + 1 : 3 * row + 3]:
                    assert all(
                        abs(Decimal(value)) < Decimal(tolerance) for value in pair
                    )


def test_invalid_box_scale_lane_is_hidden_after_previous_success():
    reducer = PolynomialPlan(4, ["mu2^2", "0", "0"], [])

    def batch(values, columns):
        return NativeBatch.from_decimal(values, 2, columns, row_ids=[7, 3])

    shifts = batch(
        [
            [str(component), "0"]
            for _ in range(2)
            for vector in ((0, 0, 0, 0), (2, 1, 0, 0), (3, 0, 1, 0), (4, 0, 0, 1))
            for component in vector
        ],
        16,
    )
    masses = batch([["1", "0"]] * 8, 4)
    scales = batch([["1", "0"]] * 2, 1)
    parameters = batch([], 0)
    assert reducer.evaluate_batch(
        shifts, masses, scales, parameters
    ).values.row_ids == [7, 3]
    invalid = batch([["1", "0"]] * 4 + [["NaN", "0"]] * 4, 4)
    result = reducer.evaluate_batch(shifts, invalid, scales, parameters)
    assert result.statuses[0] == (7, "success")
    assert result.statuses[1] == (3, "sampling scale: NonFiniteInput")
    assert result.successful_values().row_ids == [7]
    with pytest.raises(ValueError, match="failed lanes"):
        _ = result.values
    assert reducer.evaluate_batch(
        shifts, masses, scales, parameters
    ).values.row_ids == [7, 3]
