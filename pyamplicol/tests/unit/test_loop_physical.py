# SPDX-License-Identifier: 0BSD
"""Physical bare DY control; analytic/reference formulae live only in tests."""

from __future__ import annotations

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical import DrellYanBareProgram


@pytest.fixture(scope="module")
def native():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires a licensed Symbolica runtime")
    return pytest.importorskip("_loop_reduction")


@pytest.fixture(scope="module")
def points():
    fixture = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/drell_yan_qcd.json"
    )
    return np.asarray(json.loads(fixture.read_text())["points"], dtype=str)


def _expected(point, scale, alpha_s="0.118", inverse_alpha="132.507"):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((x**2 for x in vector[1:]), E("0"))

    s = square([a + b for a, b in zip(p[0], p[1], strict=True)])
    t = square([a - b for a, b in zip(p[0], p[2], strict=True)])
    u = square([a - b for a, b in zip(p[0], p[3], strict=True)])
    born = (
        E("8/27")
        * (4 * E("pi") / parse_exact_ufo(inverse_alpha)) ** 2
        * (t * t + u * u)
        / (s * s)
    )
    logarithm = (parse_exact_ufo(scale) / s).log()
    ratio = parse_exact_ufo(alpha_s) * E("4/3") / (2 * E("pi"))
    # Model R2 contributes -2*ratio*Born; this control deliberately excludes it.
    values = [
        born,
        born * ratio * (E("pi") ** 2 - 6 - 3 * logarithm - logarithm**2),
        born * ratio * (-3 - 2 * logarithm),
        -2 * born * ratio,
    ]
    evaluator = Expression.evaluator_multiple(values, [], n_cores=1, jit_compile=False)
    return [
        value.to_decimal_tuple()[0]
        for value in evaluator.evaluate_complex_with_prec([], 100)
    ]


def _compare(result, points, scales, tolerance, **parameters):
    actual = np.concatenate([result.born[:, None, :], result.virtual], axis=1)
    with localcontext() as context:
        context.prec = 110
        for row, point, scale in zip(actual, points, scales, strict=True):
            expected = _expected(point, scale, **parameters)
            for pair, target in zip(row, expected, strict=True):
                assert abs(Decimal(pair[0]) - target) < tolerance * abs(target)
                assert abs(Decimal(pair[1])) < tolerance * abs(target)
    assert all(status == "success" for row in result.statuses for status in row)


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, Decimal("2e-11")),
        ("double-double", None, Decimal("2e-27")),
        ("arbitrary", 70, Decimal("2e-64")),
    ],
)
def test_native_physical_born_and_bare_loop_all_precisions(
    native, points, arithmetic, precision, tolerance
):
    program = DrellYanBareProgram(
        backend=native, arithmetic=arithmetic, precision=precision
    )
    assert str(program.born_weight) == "1/12"
    assert str(program.interference_weight) == "1/9"
    scales = ["1000000", "2000000", "700000"]
    _compare(program.evaluate_decimal(points, scales), points, scales, tolerance)
    # Rebind couplings and scale on the same native plans, then shrink and empty.
    subset = points[[2]]
    _compare(
        program.evaluate_decimal(subset, ["900000"], alpha_s="0.1"),
        subset,
        ["900000"],
        tolerance,
        alpha_s="0.1",
    )
    empty = program.evaluate_decimal(points[:0], [])
    assert empty.born.shape == (0, 2) and empty.virtual.shape == (0, 3, 2)
    assert empty.statuses == ()
    with pytest.raises(TypeError, match="decimal strings"):
        program.evaluate_decimal(points.astype(float), scales)


def test_native_physical_input_bits_beyond_binary64_survive_all_stages(native, points):
    program = DrellYanBareProgram(backend=native, arithmetic="arbitrary", precision=80)
    with localcontext() as context:
        context.prec = 220
        factor = 1 + Decimal(2) ** -180
        original = points[:1]
        perturbed = np.asarray(
            [[[str(Decimal(value) * factor) for value in leg] for leg in original[0]]],
            dtype=object,
        )
        result = program.evaluate_decimal(
            np.concatenate([original, perturbed]), ["1000000"] * 2
        )
        delta = Decimal(result.virtual[1, 1, 0]) - Decimal(result.virtual[0, 1, 0])
        assert delta != 0
        expected_delta = (
            _expected(perturbed[0], "1000000")[2] - _expected(original[0], "1000000")[2]
        )
        assert abs(delta / expected_delta - 1) < Decimal("1e-20")
        _compare(
            result,
            np.concatenate([original, perturbed]),
            ["1000000"] * 2,
            Decimal("2e-73"),
        )


def test_bare_result_against_actual_independent_madloop(native, points):
    path = os.environ.get("PYAMPLICOL_DRELL_YAN_REFERENCE")
    if not path:
        pytest.skip(
            "set PYAMPLICOL_DRELL_YAN_REFERENCE to the independent MadLoop report"
        )
    reference = json.loads(Path(path).read_text())
    assert reference["actual_poles"] and reference["provider"].startswith("MadLoop")
    program = DrellYanBareProgram(backend=native)
    actual = program.evaluate_decimal(points, ["1000000"] * 3)
    for index, row in enumerate(reference["results"]):
        born = float(actual.born[index, 0])
        assert born == pytest.approx(row["born"], rel=2e-14)
        # Verified loop_sm R2_UUA rule, removed from the complete reference.
        import math

        r2 = -2 * (0.118 / (2 * math.pi)) * (4 / 3) * row["born"]
        expected = [row["finite"] - r2, row["single_pole"], row["double_pole"]]
        np.testing.assert_allclose(
            actual.virtual[index, :, 0].astype(float), expected, rtol=5e-12, atol=1e-16
        )
