# SPDX-License-Identifier: 0BSD
"""Native row tiling, live scratch reuse and exact-carrier conformance."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import CoefficientPlan, NativeBatch
from symbolica import E, Expression, S


def saved(expressions, parameters):
    return Expression.evaluator_multiple(
        expressions, parameters, n_cores=1, jit_compile=False
    ).save()


def schedule(arithmetic, precision, engine, tile_rows):
    x, y, z = S("tile_x", "tile_y", "tile_z")
    return CoefficientPlan(
        [
            saved([x - 1, y.conj() + E("1i"), E("if")(z, z**2, 0)], [x, y, z]),
            saved([E("0"), E("7/3+2i/5")], []),
            saved([x * y + z, x + z, E("0")], [x, y, z]),
        ],
        [[0, 1, 2], [], [3, 4, 7]],
        3,
        [8, 6, 5, 9, 3, 8, 1],
        arithmetic,
        precision,
        engine,
        tile_rows,
    )


@pytest.mark.parametrize("tile_rows", [16, 64, 128])
@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "interpreted", Decimal("2e-14")),
        ("double", None, "jit", Decimal("2e-14")),
        ("double-double", None, "interpreted", Decimal("2e-29")),
        ("arbitrary", 80, "interpreted", Decimal("2e-75")),
    ],
)
def test_tiled_scratch_reuse_preserves_values_ids_and_changing_tails(
    tile_rows, arithmetic, precision, engine, tolerance
):
    program = schedule(arithmetic, precision, engine, tile_rows)
    unbounded = schedule(arithmetic, precision, engine, 8192)
    assert program.tile_rows == tile_rows
    assert program.workspace_capacity == (0, 0, 0)
    # One complete tile establishes the maximum schedule-owned storage.
    warm = NativeBatch.from_decimal(
        [["1", "0"], ["1", "0"], ["0", "0"]] * tile_rows,
        tile_rows,
        3,
        arithmetic,
        precision,
    )
    program.evaluate_batch(warm)
    capacity = program.workspace_capacity
    with localcontext() as context:
        context.prec = 160
        tiny = Decimal(2) ** -(
            3
            if arithmetic == "double"
            else 80
            if arithmetic == "double-double"
            else 180
        )
        for iteration, rows in enumerate((1, 65, 0, 129, 7, 257, 1)):
            ids = [2003 - 7 * i for i in range(rows)]
            values = [
                v
                for i in range(rows)
                for v in (
                    [str(1 + tiny * (i + iteration + 1)), "0"],
                    [str(i + 2), str(iteration - i)],
                    [str(i % 2), "0"],
                )
            ]
            inputs = NativeBatch.from_decimal(
                values, rows, 3, arithmetic, precision, ids
            )
            actual = program.evaluate_batch(inputs)
            expected = unbounded.evaluate_batch(inputs)
            assert actual.row_ids == expected.row_ids == ids
            assert actual.rows == rows and actual.columns == 7
            assert actual.precision_bits == inputs.precision_bits
            assert program.workspace_capacity == capacity
            observed = actual.to_decimal()
            for a, b in zip(observed, expected.to_decimal(), strict=True):
                for left, right in zip(a, b, strict=True):
                    left, right = Decimal(left), Decimal(right)
                    assert abs(left - right) < tolerance * max(1, abs(right))
            for i in range(rows):
                offset = 7 * i
                assert all(Decimal(v) == 0 for v in observed[offset + 1])
                assert Decimal(observed[offset + 2][0]) == i % 2
                delta = Decimal(observed[offset + 4][0])
                assert delta != 0
                assert abs(delta / (tiny * (i + iteration + 1)) - 1) < Decimal("1e-8")
                assert observed[offset] == observed[offset + 5]


@pytest.mark.parametrize(
    "arithmetic,precision",
    [("double", None), ("double-double", None), ("arbitrary", 80)],
)
def test_tiled_empty_constant_and_passthrough_schedules(arithmetic, precision):
    constant = CoefficientPlan(
        [saved([E("2/3"), E("0")], [])],
        [[]],
        0,
        [1, 0],
        arithmetic,
        precision,
        tile_rows=16,
    )
    passthrough = CoefficientPlan(
        [], [], 1, [0, 0], arithmetic, precision, tile_rows=16
    )
    no_outputs = CoefficientPlan([], [], 0, [], arithmetic, precision)
    assert no_outputs.tile_rows == 16
    for rows in (33, 0, 1, 129):
        ids = [900 - 2 * i for i in range(rows)]
        inputs = NativeBatch.from_decimal([], rows, 0, arithmetic, precision, ids)
        empty = no_outputs.evaluate_batch(inputs)
        assert empty.row_ids == ids and empty.rows == rows and empty.columns == 0
        assert empty.to_decimal() == []
        value = constant.evaluate_batch(inputs)
        assert value.row_ids == ids and value.rows == rows and value.columns == 2
        for i, pair in enumerate(value.to_decimal()):
            assert Decimal(pair[1]) == 0
            assert (
                Decimal(pair[0]) == 0
                if i % 2 == 0
                else abs(Decimal(pair[0]) - Decimal(2) / 3) < Decimal("1e-16")
            )
        source = NativeBatch.from_decimal(
            [[str(i), "1"] for i in range(rows)], rows, 1, arithmetic, precision, ids
        )
        output = passthrough.evaluate_batch(source)
        assert output.row_ids == ids
        assert output.to_decimal() == [
            pair for pair in source.to_decimal() for _ in range(2)
        ]
    with pytest.raises(ValueError, match="tile size"):
        CoefficientPlan([], [], 0, [], arithmetic, precision, tile_rows=0)
