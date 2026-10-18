# SPDX-License-Identifier: 0BSD
"""Numerical agreement is evidence, not a reducer accuracy certificate."""

from decimal import Decimal, localcontext

import pytest

backend = pytest.importorskip("_loop_reduction")


def batch(values, ids, *, arithmetic="arbitrary", precision=90, columns=1):
    return backend.NativeBatch.from_decimal(
        values,
        len(ids),
        columns,
        row_ids=ids,
        arithmetic=arithmetic,
        precision=precision if arithmetic == "arbitrary" else None,
    )


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
        assert promoted.agreement_rows(value, "1e-30", "0") == [(701, False)]
        assert dd.agreement_rows(value, "1e-30", "0") == [(701, True)]
        empty = value.take_rows([]).convert_arithmetic("arbitrary", precision=16)
        assert empty.rows == 0 and empty.columns == 1 and empty.row_ids == []


def test_sparse_native_double_double_low_word_survives_conversion_and_comparison():
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
        one = batch([("1", "0")], [93], arithmetic="double-double")
        reference = batch([(str(1 + tiny), "0")], [93], precision=130)
        # DD metadata promises106 bits, but a sparse low word can retain more.
        # Comparison must not first round it to a dense106-bit Float.
        assert value.agreement_rows(one, "1e-70", "0") == [(93, False)]
        assert value.agreement_rows(reference, "1e-100", "0") == [(93, True)]
        copied = value.convert_arithmetic("double-double")
        assert copied.agreement_rows(reference, "1e-100", "0") == [(93, True)]
        widened = value.convert_arithmetic("arbitrary", precision=130)
        assert widened.agreement_rows(reference, "1e-100", "0") == [(93, True)]
        assert abs(Decimal(widened.to_decimal()[0][0]) - 1 - tiny) < Decimal("1e-130")


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
        reference = batch([(str(hi), "0")], [51], arithmetic="double")
        assert narrowed.agreement_rows(reference, "1e-30", "0") == [(51, True)]
        # Also exercise target precision below one binary64 word. Compare the
        # dependency's once-rounded exact rational evaluator at the same target.
        target = value.convert_arithmetic("arbitrary", precision=5)
        expected = batch([(str(hi + lo), "0")], [51], precision=5)
        assert target.agreement_rows(expected, "1e-30", "0") == [(51, True)]
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
        assert narrow_tie.agreement_rows(expected, "1e-30", "0") == [(51, True)]


@pytest.mark.parametrize("left", ["double", "double-double", "arbitrary"])
@pytest.mark.parametrize("right", ["double", "double-double", "arbitrary"])
def test_all_columns_complex_extremes_and_original_lane_ids(left, right):
    # Last imaginary column represents a pole or a separate component: it must
    # not be ignored by a finite-only or real-only comparison.
    values = [("1e200", "1e-200"), ("0", "-1e200"), ("2", "0")]
    a = batch(values * 2, [91, 4], arithmetic=left, columns=3)
    b = batch(
        values + values[:-1] + [("2", "1e-8")], [91, 4], arithmetic=right, columns=3
    )
    assert a.agreement_rows(b, "1e-14", "1e-20") == [(91, True), (4, False)]
    assert b.agreement_rows(a, "1e-14", "1e-20") == [(91, True), (4, False)]
    assert a.agreement_rows(a, "1e-30", "0") == [(91, True), (4, True)]


def test_nonfinite_and_invalid_tolerances_never_establish_agreement():
    ordinary = batch([("1", "0")], [8])
    for value in ("nan", "inf", "-inf"):
        bad = batch([(value, "0")], [8], arithmetic="double")
        assert bad.agreement_rows(bad, "1e-10", "0") == [(8, False)]
        assert ordinary.agreement_rows(bad, "1e-10", "0") == [(8, False)]
    for relative, absolute in (("0", "0"), ("-1", "0"), ("nan", "0"), ("0", "inf")):
        with pytest.raises(ValueError, match="tolerances"):
            ordinary.agreement_rows(ordinary, relative, absolute)
    with pytest.raises(ValueError, match="identities and columns"):
        ordinary.agreement_rows(ordinary.take_rows([0], row_ids=[9]), "1e-8", "0")
    overflow = batch([("1e1000", "0")], [91]).convert_arithmetic("double")
    assert overflow.nonfinite_rows() == [91]
    assert overflow.agreement_rows(overflow, "1e-8", "0") == [(91, False)]
    tiny = batch([("1e-300", "0")], [91], arithmetic="double")
    changed = batch([("1.1e-300", "0")], [91])
    assert tiny.agreement_rows(changed, "1e-8", "0") == [(91, False)]


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
