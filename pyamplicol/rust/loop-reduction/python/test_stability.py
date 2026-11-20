# SPDX-License-Identifier: 0BSD
"""Carrier conversion and lane transport of native batches."""

from decimal import Decimal, localcontext

import _loop_reduction as backend
import pytest


def batch(values, ids, *, arithmetic="arbitrary", precision=90, columns=1):
    return backend.NativeBatch.from_decimal(
        values,
        len(ids),
        columns,
        row_ids=ids,
        arithmetic=arithmetic,
        precision=precision if arithmetic == "arbitrary" else None,
    )


def decimals(value):
    return [tuple(map(Decimal, row)) for row in value.to_decimal()]


def test_direct_conversion_preserves_available_bits_and_does_not_invent_digits():
    with localcontext() as context:
        context.prec = 130
        original = Decimal(1) + Decimal(2) ** -90
        value = batch([(str(original), str(-original))], [701])
        double = value.convert_arithmetic("double")
        dd = value.convert_arithmetic("double-double")
        assert double.row_ids == dd.row_ids == [701]
        assert double.precision_bits == 53 and dd.precision_bits == 106
        assert tuple(map(Decimal, double.to_decimal()[0])) == (1, -1)
        assert abs(Decimal(dd.to_decimal()[0][0]) - original) < Decimal("1e-32")
        promoted = double.convert_arithmetic("arbitrary", precision=110)
        assert tuple(map(Decimal, promoted.to_decimal()[0])) == (1, -1)
        empty = value.take_rows([]).convert_arithmetic("arbitrary", precision=16)
        assert empty.rows == 0 and empty.columns == 1 and empty.row_ids == []


def test_sparse_native_double_double_low_word_survives_conversion():
    from symbolica import E, S

    x = S("native_stability_sparse_x")
    evaluator = backend.CoefficientEvaluator(
        (E("1") + x).evaluator([x], n_cores=1, jit_compile=False).save(),
        arithmetic="double-double",
    )
    with localcontext() as context:
        context.prec = 200
        tiny = Decimal(2) ** -200
        low = batch([(str(tiny), "0")], [93], arithmetic="double-double")
        value = evaluator.evaluate_batch(low)
        # DD metadata promises 106 bits, but a sparse low word can retain more;
        # widening must not first round it to a dense 106-bit Float.
        copied = value.convert_arithmetic("double-double")
        for kept in (value, copied):
            widened = kept.convert_arithmetic("arbitrary", precision=130)
            assert abs(Decimal(widened.to_decimal()[0][0]) - 1 - tiny) < Decimal(
                "1e-130"
            )


def test_native_double_double_midpoint_conversion_rounds_only_once():
    from symbolica import S

    x, y = S("native_stability_tie_x", "native_stability_tie_y")
    evaluator = backend.CoefficientEvaluator(
        (x + y).evaluator([x, y], n_cores=1, jit_compile=False).save(),
        arithmetic="double-double",
    )
    with localcontext() as context:
        context.prec = 180
        hi = 1 + Decimal(2) ** -52
        lo = -(Decimal(2) ** -53 - Decimal(2) ** -106)
        value = evaluator.evaluate_batch(
            batch(
                [(str(hi), "0"), (str(lo), "0")],
                [51],
                arithmetic="double-double",
                columns=2,
            )
        )
        narrowed = value.convert_arithmetic("double")
        # hi is a binary64 number: the narrowed value is exactly it.
        assert tuple(map(float, narrowed.to_decimal()[0])) == (float(hi), 0.0)
        # Also exercise target precision below one binary64 word. Compare the
        # dependency's once-rounded exact rational evaluator at the same target.
        target = value.convert_arithmetic("arbitrary", precision=5)
        expected = batch([(str(hi + lo), "0")], [51], precision=5)
        assert decimals(target) == decimals(expected)
        bits = target.precision_bits
        hi = 1 + Decimal(2) ** -bits
        lo = Decimal(2) ** -(bits + 100)
        narrow_tie = evaluator.evaluate_batch(
            batch(
                [(str(hi), "0"), (str(lo), "0")],
                [51],
                arithmetic="double-double",
                columns=2,
            )
        ).convert_arithmetic("arbitrary", precision=5)
        expected = batch([(str(1 + Decimal(2) ** (1 - bits)), "0")], [51], precision=5)
        assert decimals(narrow_tie) == decimals(expected)


def test_an_overflowing_conversion_is_a_nonfinite_row():
    overflow = batch([("1e1000", "0")], [91]).convert_arithmetic("double")
    assert overflow.nonfinite_rows() == [91]


@pytest.mark.parametrize("arithmetic", ["double", "double-double", "arbitrary"])
def test_compact_scatter_retains_missing_lanes_without_placeholders(arithmetic):
    a = batch([("4", "5"), ("6", "7")], [8, 2], arithmetic=arithmetic)
    b = batch([("11", "12")], [17], arithmetic=arithmetic)
    result = backend.NativeBatch.merge_rows([17, 99, 2, 8], [a, b])
    assert result.row_ids == [17, 2, 8]
    assert [tuple(map(Decimal, value)) for value in result.to_decimal()] == [
        (11, 12),
        (6, 7),
        (4, 5),
    ]
    empty = backend.NativeBatch.merge_rows([17, 99, 2, 8], [a.take_rows([])])
    assert empty.row_ids == [] and empty.columns == 1
    for ids, groups in (([8, 8], [a]), ([8, 2], [a, a]), ([8], [a])):
        with pytest.raises(ValueError, match="identities"):
            backend.NativeBatch.merge_rows(ids, groups)
    foreign = "double-double" if arithmetic == "double" else "double"
    with pytest.raises(ValueError, match="arithmetic or precision"):
        backend.NativeBatch.merge_rows([8, 2, 17], [a, b.convert_arithmetic(foreign)])
