# SPDX-License-Identifier: 0BSD
"""Real Python -> shared decoder -> native batch -> NinjaRS conformance."""

from decimal import Decimal, localcontext

import pytest
from _loop_reduction import (
    CoefficientEvaluator,
    PolynomialPlan,
    massless_spinors_decimal,
)
from symbolica import E, S


def saved_coefficient():
    x = S("loop_native_coefficient_x")
    return (x - E("1")).evaluator([x], n_cores=1, jit_compile=False).save()


@pytest.mark.parametrize(
    "arithmetic,precision,bits",
    [
        ("double", None, 53),
        ("double-double", None, 106),
        ("arbitrary", 80, 266),
    ],
)
def test_saved_coefficients_reduce_in_native_batches(arithmetic, precision, bits):
    coefficients = CoefficientEvaluator(saved_coefficient(), arithmetic, precision)
    plan = PolynomialPlan(1, ["c", "2", "0"], ["c"], arithmetic, precision)
    digits = precision or (16 if arithmetic == "double" else 32)
    assert coefficients.precision_bits == plan.precision_bits == bits
    with localcontext() as context:
        context.prec = 180
        tiny = Decimal(2) ** -(80 if arithmetic == "double-double" else 180)
        tolerance = Decimal(10) ** -(12 if arithmetic == "double" else digits - 5)
        for size in [1, 33, 8, 0, 1]:
            expected = [
                Decimal(row + 1) / 4 if arithmetic == "double" else tiny * (row + 1)
                for row in range(size)
            ]
            inputs = [[str(1 + value), "0"] for value in expected]
            values = coefficients.evaluate_decimal(inputs, size)
            ids = [700 - row * 3 for row in range(size)]
            result = plan.evaluate_decimal(
                ids,
                [[["0", "0"] for _ in range(4)] for _ in ids],
                [["1", "0"] for _ in ids],
                [["1", "0"] for _ in ids],
                values,
            )
            assert [row[0] for row in result] == ids
            for (_, status, poles), value in zip(result, expected, strict=True):
                assert status == "success"
                assert poles is not None
                # A0(1) = 1 + 1/epsilon, hence N0=c,N1=2 gives
                # finite=c+2, single pole=c. No physical prefactors are implied.
                finite, single, double = [Decimal(pair[0]) for pair in poles]
                assert abs(finite - value - 2) < tolerance
                assert abs(single - value) < tolerance
                if arithmetic != "double":
                    # Detect any binary64 conversion of 1+tiny at any boundary.
                    assert single != 0
                    assert abs(single / value - 1) < Decimal("0.00001")
                assert double == 0
                assert all(Decimal(pair[1]) == 0 for pair in poles)


@pytest.mark.parametrize(
    "arithmetic,precision,digits",
    [
        ("double", None, 16),
        ("double-double", None, 32),
        ("arbitrary", 80, 80),
    ],
)
def test_explicit_mu2_triangle_crosses_python_boundary(arithmetic, precision, digits):
    plan = PolynomialPlan(3, ["2*mu2", "0", "0"], [], arithmetic, precision)
    shifts = [
        [["0", "0"] for _ in range(4)],
        [["-0.5", "0"], ["0", "0"], ["0", "0"], ["-0.5", "0"]],
        [["-1", "0"], ["0", "0"], ["0", "0"], ["0", "0"]],
    ]
    result = plan.evaluate_decimal(
        [91, 13], shifts * 2, [["4", "0"]] * 6, [["1", "0"]] * 2, []
    )
    for lane_id, status, values in result:
        assert lane_id in [91, 13]
        assert status == "success"
        tolerance = Decimal(10) ** -(digits - 4)
        for actual, expected in zip(values, [[1, 0], [0, 0], [0, 0]], strict=True):
            assert all(
                abs(Decimal(part) - target) < tolerance
                for part, target in zip(actual, expected, strict=True)
            )


def test_invalid_shapes_serialization_and_precision_fail_at_boundary():
    state = saved_coefficient()
    with pytest.raises(ValueError, match="trailing"):
        CoefficientEvaluator(state + b"\0")
    with pytest.raises(ValueError, match="precision"):
        CoefficientEvaluator(state, "double", 20)
    with pytest.raises(ValueError, match="higher-precision"):
        CoefficientEvaluator(state, "double-double").evaluate_f64([[2, 0]], 1)
    evaluator = CoefficientEvaluator(state)
    assert evaluator.evaluate_f64([[2, 0], [4, 1]], 2) == [[1, 0], [3, 1]]
    with pytest.raises(ValueError):
        evaluator.evaluate_decimal([], 2)
    with pytest.raises(ValueError, match="propagator"):
        PolynomialPlan(0, ["1", "0", "0"], [])
    plan = PolynomialPlan(1, ["c", "0", "0"], ["c"])
    with pytest.raises(ValueError):
        plan.evaluate_decimal([10], [[["0", "0"]] * 4], [["1", "0"]], [["1", "0"]], [])


@pytest.mark.parametrize("digits,bits", [(16, 54), (32, 107)])
def test_arbitrary_arithmetic_is_independent_of_decimal_precision(digits, bits):
    evaluator = CoefficientEvaluator(saved_coefficient(), "arbitrary", digits)
    plan = PolynomialPlan(1, ["1", "0", "0"], [], "arbitrary", digits)
    assert evaluator.precision_bits == plan.precision_bits == bits
    with pytest.raises(ValueError, match="higher-precision"):
        evaluator.evaluate_f64([[2, 0]], 1)
    assert Decimal(evaluator.evaluate_decimal([["2", "0"]], 1)[0][0]) == 1
    for arithmetic, precision in [
        ("arbitrary", None),
        ("arbitrary", 0),
        ("quad", None),
    ]:
        with pytest.raises(ValueError):
            CoefficientEvaluator(saved_coefficient(), arithmetic, precision)


@pytest.mark.parametrize(
    "arithmetic,precision,scale,tolerance",
    [
        ("double", None, "1e-100", "1e-13"),
        ("double-double", None, "1e-100", "1e-28"),
        ("arbitrary", 80, "1e-400", "1e-74"),
    ],
)
def test_native_massless_states_preserve_homogeneity_below_binary64_range(
    arithmetic, precision, scale, tolerance
):
    # Off-axis with negative pz distinguishes true zero tests from underflow.
    points = [["13", "3", "4", "-12"], ["5", "0", "0", "-5"], ["5", "0", "0", "5"]]
    rows = [
        (p, h, k) for p in points for h in [-1, 1] for k in ["u", "v", "u_bar", "v_bar"]
    ]
    momenta, helicities, kinds = map(list, zip(*rows, strict=True))
    with localcontext() as context:
        context.prec = 100
        factor = Decimal(scale)
        small = [[str(Decimal(x) * factor) for x in p] for p in momenta]
        baseline = massless_spinors_decimal(
            momenta, helicities, kinds, arithmetic, precision
        )
        actual = massless_spinors_decimal(
            small, helicities, kinds, arithmetic, precision
        )
        for row, reference in zip(actual, baseline, strict=True):
            for component, expected in zip(row, reference, strict=True):
                for value, target in zip(component, expected, strict=True):
                    assert abs(
                        Decimal(value) / factor.sqrt() - Decimal(target)
                    ) < Decimal(tolerance)
    assert massless_spinors_decimal([], [], [], arithmetic, precision) == []


def test_native_massless_states_validate_shapes_and_physical_state_labels():
    for momenta, helicities, kinds in [
        ([["1", "0", "0", "1"]], [], ["u"]),
        ([["1", "0", "0", "1"]], [0], ["u"]),
        ([["1", "0", "0", "1"]], [1], ["x"]),
        ([["-1", "0", "0", "-1"]], [1], ["u"]),
        ([["1", "nan", "0", "1"]], [1], ["u"]),
    ]:
        with pytest.raises(ValueError):
            massless_spinors_decimal(momenta, helicities, kinds)


@pytest.mark.parametrize(
    "arithmetic,precision,digits",
    [
        ("double", None, 16),
        ("double-double", None, 32),
        ("arbitrary", 80, 80),
    ],
)
def test_mathematical_constants_use_requested_working_precision(
    arithmetic, precision, digits
):
    x = S("loop_native_constant_x")
    state = (
        E("sqrt(pi)*loop_native_constant_x")
        .evaluator([x], n_cores=1, jit_compile=False)
        .save()
    )
    evaluator = CoefficientEvaluator(state, arithmetic, precision)
    values = evaluator.evaluate_decimal([["1", "0"], ["2", "0"]], 2)
    # Independent decimal digits of sqrt(pi), preserving more than binary64.
    expected = Decimal(
        "1.7724538509055160272981674833411451827975494561223871282138077898529112845910321813749506567"
    )
    with localcontext() as context:
        context.prec = 100
        for i, (real, imag) in enumerate(values, 1):
            assert abs(Decimal(real) / i - expected) < Decimal(10) ** -(digits - 3)
            assert Decimal(imag) == 0
