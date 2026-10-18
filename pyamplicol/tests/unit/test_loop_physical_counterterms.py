# SPDX-License-Identifier: 0BSD
"""Source-derived model R2 completes the physical Drell--Yan control."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_counterterms import (
    NativeModelVirtualProgram,
    drell_yan_model_virtual_program,
)


@pytest.fixture(scope="module")
def source():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to the original loop_sm model")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    return build_loop_model(load_raw_loop_model(path))


@pytest.fixture(scope="module")
def points():
    path = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/drell_yan_qcd.json"
    )
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


def _expected(point, scale, alpha_s, inverse_alpha):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((v**2 for v in vector[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    born = (
        E("8/27")
        * (4 * E("pi") / parse_exact_ufo(inverse_alpha)) ** 2
        * (t * t + u * u)
        / s**2
    )
    ratio = parse_exact_ufo(alpha_s) * E("4/3") / (2 * E("pi"))
    logarithm = (parse_exact_ufo(scale) / s).log()
    complete = [
        born * ratio * (E("pi") ** 2 - 8 - 3 * logarithm - logarithm**2),
        born * ratio * (-3 - 2 * logarithm),
        -2 * born * ratio,
    ]
    evaluator = Expression.evaluator_multiple(
        [born, *complete, -2 * born * ratio], [], n_cores=1, jit_compile=False
    )
    return [
        value.to_decimal_tuple()[0]
        for value in evaluator.evaluate_complex_with_prec([], 110)
    ]


def _compare(result, points, scales, alpha_s, inverse_alpha, tolerance):
    assert result.born.shape == (len(points), 2)
    assert (
        result.virtual.shape
        == result.bare.shape
        == result.uv.shape
        == result.r2.shape
        == (len(points), 3, 2)
    )
    with localcontext() as context:
        context.prec = 120
        for index, (point, scale) in enumerate(zip(points, scales, strict=True)):
            expected = _expected(point, scale, alpha_s, inverse_alpha)
            actual = [result.born[index], *result.virtual[index], result.r2[index, 0]]
            for pair, target in zip(actual, expected, strict=True):
                assert abs(Decimal(pair[0]) - target) < abs(target) * tolerance
                assert Decimal(pair[1]) == 0
            assert all(Decimal(value) == 0 for value in result.uv[index].flat)
            assert all(Decimal(value) == 0 for value in result.r2[index, 1:].flat)
            for pole in range(3):
                reconstructed = Decimal(result.bare[index, pole, 0]) + Decimal(
                    result.r2[index, pole, 0]
                )
                assert (
                    abs(reconstructed - Decimal(result.virtual[index, pole, 0]))
                    < tolerance
                )
    assert all(status == "success" for row in result.statuses for _, status in row)


def _replace_ct_scalar(program, expression, *, kind="UVloop"):
    from pyamplicol.loops.model_expressions import LaurentExpression

    insertion = program.insertions[0]
    assert len(insertion.rule.couplings) == 1
    name, entry = insertion.rule.couplings[0]
    entry = replace(
        entry, coupling=replace(entry.coupling, value=LaurentExpression(expression))
    )
    return replace(
        insertion, rule=replace(insertion.rule, kind=kind, couplings=((name, entry),))
    )


def test_local_uv_scale_is_per_point_and_laurent_domain_is_explicit(source, points):
    from symbolica import E, Expression, S

    from pyamplicol.loops.model_expressions import epsilon_symbol, parse_exact_ufo

    base = drell_yan_model_virtual_program(source, arithmetic="arbitrary", precision=70)
    original = base.insertions[0].rule.couplings[0][1].coupling.value.expression
    scale = S("ct_scale_test::mu_r2")
    modified = _replace_ct_scalar(
        base, original * ((scale / E("1000000")).log() + 1 / epsilon_symbol())
    )
    program = NativeModelVirtualProgram(
        base.bare,
        [modified],
        source_bindings=(
            *base.source_bindings,
            (scale, base.bare.scale_squared_symbol),
        ),
    )
    with localcontext() as context:
        context.prec = 160
        scales = [
            "1000000",
            "4000000",
            str(Decimal("1000000") * (1 + Decimal(2) ** -140)),
        ]
        repeated = np.repeat(points[:1], len(scales), axis=0)
        result = program.evaluate_decimal(
            repeated, scales, model_values=["0.118", "132.507"]
        )
        r2 = _expected(points[0], "1000000", "0.118", "132.507")[-1]
        logs = Expression.evaluator_multiple(
            [(parse_exact_ufo(value) / E("1000000")).log() for value in scales],
            [],
            n_cores=1,
            jit_compile=False,
        ).evaluate_complex_with_prec([], 110)
        for row, logarithm in zip(result.uv, logs, strict=True):
            expected = r2 * logarithm.to_decimal_tuple()[0]
            assert abs(Decimal(row[0, 0]) - expected) < Decimal("1e-66")
            assert abs(Decimal(row[1, 0]) - r2) < Decimal("1e-66")
        assert Decimal(result.uv[2, 0, 0]) != 0
        assert all(Decimal(value) == 0 for value in result.r2.flat)
    unsupported = _replace_ct_scalar(base, original / epsilon_symbol() ** 3)
    with pytest.raises(NotImplementedError, match="Laurent powers"):
        NativeModelVirtualProgram(
            base.bare, [unsupported], source_bindings=base.source_bindings
        )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, Decimal("3e-11")),
        ("double-double", None, Decimal("3e-27")),
        ("arbitrary", 70, Decimal("3e-63")),
    ],
)
def test_model_r2_completes_native_bare_in_all_precisions(
    source, points, arithmetic, precision, tolerance
):
    program = drell_yan_model_virtual_program(
        source, arithmetic=arithmetic, precision=precision
    )
    assert len(program.insertions) == len(program.currents) == 1
    assert program.currents[0].insertion.rule.kind == "R2"
    scales = ["1000000", "2000000", "700000"]
    result = program.evaluate_decimal(points, scales, model_values=["0.118", "132.507"])
    _compare(result, points, scales, "0.118", "132.507", tolerance)
    # Source couplings share the same precise global model input binding as Born.
    alpha_s = "0.100000000000000000000000000123456789"
    rebound = program.evaluate_decimal(
        points[1:2], ["900000"], model_values=[alpha_s, "137.035999084"]
    )
    _compare(rebound, points[1:2], ["900000"], alpha_s, "137.035999084", tolerance)
    empty = program.evaluate_decimal(points[:0], [], model_values=["0.118", "132.507"])
    assert empty.virtual.shape == (0, 3, 2) and empty.statuses == ()


@pytest.mark.parametrize("coefficient_engine", ["interpreted", "jit"])
def test_complete_result_against_actual_madloop_and_uncalibrated_basis_rejected(
    source, points, coefficient_engine
):
    path = os.environ.get("PYAMPLICOL_DRELL_YAN_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_DRELL_YAN_REFERENCE to the MadLoop report")
    reference = json.loads(Path(path).read_text())
    program = drell_yan_model_virtual_program(
        source, coefficient_engine=coefficient_engine
    )
    result = program.evaluate_decimal(
        points, ["1000000"] * len(points), model_values=["0.118", "132.507"]
    )
    assert result.coefficient_engine == coefficient_engine
    assert program.currents[0].executor.coefficient_engine == coefficient_engine
    for index, row in enumerate(reference["results"]):
        np.testing.assert_allclose(
            result.virtual[index, :, 0].astype(float),
            [row[key] for key in ("finite", "single_pole", "double_pole")],
            rtol=5e-12,
            atol=1e-16,
        )
    insertion = program.insertions[0]
    uncertified = replace(
        insertion, rule=replace(insertion.rule, basis_correspondences=())
    )
    with pytest.raises(NotImplementedError, match="calibrated tensor correspondence"):
        NativeModelVirtualProgram(program.bare, [uncertified], source_bindings=[])
