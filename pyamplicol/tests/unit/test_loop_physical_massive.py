# SPDX-License-Identifier: 0BSD
"""Massive physical states in the same native graph/family assembler."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_process import top_pair_qcd_program

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def native(loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires licensed Symbolica")
    return loop_native


@pytest.fixture(scope="module")
def points():
    path = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/top_pair_qcd.json"
    )
    return np.asarray(
        json.loads(path.read_text(), parse_float=str, parse_int=str)["points"],
        dtype=str,
    )


def _expected(point):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((v**2 for v in vector[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    mass2, alpha = E("173^2"), E("59/500")
    born = (
        E("4/9")
        * (4 * E("pi") * alpha) ** 2
        * ((t - mass2) ** 2 + (u - mass2) ** 2 + 2 * mass2 * s)
        / s**2
    )
    # Only the two massless initial quarks produce collinear double poles.
    double = -2 * E("4/3") * alpha / (2 * E("pi")) * born
    evaluator = Expression.evaluator_multiple(
        [born, double], [], n_cores=1, jit_compile=False
    )
    return [
        v.to_decimal_tuple()[0] for v in evaluator.evaluate_complex_with_prec([], 110)
    ]


def test_original_decimal_fixture_is_exactly_on_shell(points):
    with localcontext() as context:
        context.prec = 100
        for point in points:
            for index, leg in enumerate(point):
                p = [Decimal(component) for component in leg]
                assert p[0] ** 2 - sum(x * x for x in p[1:]) == (
                    0 if index < 2 else Decimal(173) ** 2
                )
            for component in range(4):
                assert Decimal(point[0, component]) + Decimal(
                    point[1, component]
                ) == Decimal(point[2, component]) + Decimal(point[3, component])


@pytest.mark.parametrize(
    "arithmetic,precision,engine,coefficient_engine,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-11")),
        ("double", None, "generic", "interpreted", Decimal("3e-11")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-26")),
        ("arbitrary", 60, "generic", "interpreted", Decimal("3e-52")),
    ],
)
def test_massive_bare_native_pipeline(
    loop_reference_path,
    native,
    points,
    arithmetic,
    precision,
    engine,
    coefficient_engine,
    tolerance,
):
    program = top_pair_qcd_program(
        backend=native,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
        coefficient_engine=coefficient_engine,
    )
    assert len(program.recurrences) == 14 and len(program.families) == 9
    assert program.external_masses[0] == program.external_masses[1] == 0
    assert program.external_masses[2] == program.external_masses[3] != 0
    result = program.evaluate_decimal(
        points, ["1000000"] * len(points), model_values=["0.118", "173"]
    )
    with localcontext() as context:
        context.prec = 120
        for index, point in enumerate(points):
            for actual, target in zip(
                [result.born[index, 0], result.virtual[index, 2, 0]],
                _expected(point),
                strict=True,
            ):
                assert abs(Decimal(actual) - target) < abs(target) * tolerance
            assert all(Decimal(value) == 0 for value in result.virtual[index, :, 1])
    path = str(loop_reference_path("top-pair-component-control"))
    reference = json.loads(Path(path).read_text())
    for index, row in enumerate(reference["rows"]):
        np.testing.assert_allclose(
            result.virtual[index, :, 0].astype(float),
            [row["bare"][key] for key in ("finite", "single_pole", "double_pole")],
            rtol=1e-10,
            atol=1e-13,
        )
    # The MP finite term is only compared with a binary64 external reference;
    # exact Born and the universal double pole supply stronger precision tests.
    assert all(status == "success" for row in result.statuses for _, status in row)
    empty = program.evaluate_decimal(points[:0], [], model_values=["0.118", "173"])
    assert empty.virtual.shape == (0, 3, 2)
