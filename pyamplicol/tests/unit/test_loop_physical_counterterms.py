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
    distinct_flavour_model_virtual_program,
    drell_yan_model_virtual_program,
    top_pair_model_virtual_program,
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
                assert abs(Decimal(pair[1])) < abs(target) * tolerance
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


def test_singular_counterterm_reports_original_point_before_result(source, points):
    from symbolica import E, S

    base = drell_yan_model_virtual_program(source)
    original = base.insertions[0].rule.couplings[0][1].coupling.value.expression
    scale = S("ct_finiteness_test::scale")
    insertion = _replace_ct_scalar(base, original / (scale - E("4000000")))
    program = NativeModelVirtualProgram(
        base.bare,
        [insertion],
        source_bindings=(
            *base.source_bindings,
            (scale, base.bare.scale_squared_symbol),
        ),
    )
    with pytest.raises(
        ArithmeticError,
        match=r"nonfinite counterterm scalar coefficients.*\[\(1, 1\)\]",
    ):
        program.evaluate_decimal(
            points[:2], ["1000000", "4000000"], model_values=["0.118", "132.507"]
        )


def _uudd_born_uv(point, scale, alpha_s="0.118", mass="173"):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((v**2 for v in vector[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    alpha = parse_exact_ufo(alpha_s)
    born = E("4/9") * (4 * E("pi") * alpha) ** 2 * (t * t + u * u) / s**2
    # Six active internal quark species give beta0=7 in the bare UV pole;
    # the model's finite top decoupling term implements five-flavour alpha_s.
    finite = (
        born
        * alpha
        / (3 * E("pi"))
        * (parse_exact_ufo(scale) / parse_exact_ufo(mass) ** 2).log()
    )
    single = -7 * born * alpha / (2 * E("pi"))
    evaluator = Expression.evaluator_multiple(
        [born, finite, single], [], n_cores=1, jit_compile=False
    )
    return [
        value.to_decimal_tuple()[0]
        for value in evaluator.evaluate_complex_with_prec([], 110)
    ]


def _top_uv_expected(point, scale, alpha_s):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((v**2 for v in vector[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    alpha, mass2 = parse_exact_ufo(alpha_s), E("173^2")
    born = (
        E("4/9")
        * (4 * E("pi") * alpha) ** 2
        * ((t - mass2) ** 2 + (u - mass2) ** 2 + 2 * mass2 * s)
        / s**2
    )
    logarithm = (parse_exact_ufo(scale) / mass2).log()
    ratio = alpha / (2 * E("pi"))
    # On-shell heavy-quark field renormalization, including both external legs.
    values = [
        born * ratio * E("2/3") * logarithm,
        -7 * born * ratio,
        -born * ratio * E("4/3") * (4 + 3 * logarithm),
        -3 * born * ratio * E("4/3"),
    ]
    evaluator = Expression.evaluator_multiple(values, [], n_cores=1, jit_compile=False)
    return [
        v.to_decimal_tuple()[0] for v in evaluator.evaluate_complex_with_prec([], 110)
    ]


@pytest.mark.parametrize(
    "arithmetic,precision,engine,coefficient_engine,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-11")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-26")),
        ("arbitrary", 60, "generic", "interpreted", Decimal("3e-53")),
    ],
)
def test_complete_top_pair_includes_actual_external_field_factors(
    source, arithmetic, precision, engine, coefficient_engine, tolerance
):
    fixture = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/top_pair_qcd.json"
    )
    points = np.asarray(
        json.loads(fixture.read_text(), parse_float=str, parse_int=str)["points"],
        dtype=str,
    )
    program = top_pair_model_virtual_program(
        source,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
        coefficient_engine=coefficient_engine,
    )
    assert len(program.external_fields) == 2 and len(program.field_currents) == 4
    result = program.evaluate_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    reference_path = os.environ.get("PYAMPLICOL_TOP_PAIR_COMPONENT_REFERENCE")
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        for index, row in enumerate(reference["rows"]):
            for name, key in (
                ("bare", "bare"),
                ("uv", "uv"),
                ("r2", "r2"),
                ("virtual", "full"),
                ("uv_local", "uv_vertex"),
                ("uv_external_fields", "wf"),
            ):
                np.testing.assert_allclose(
                    getattr(result, name)[index, :, 0].astype(float),
                    [
                        row[key][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ],
                    rtol=1e-10,
                    atol=1e-13,
                )
    scales, alpha = ["900000", "2100000"], "0.100000000000000000000000001"
    rebound = program.evaluate_decimal(points[1:], scales, model_values=[alpha, "173"])
    with localcontext() as context:
        context.prec = 120
        for index, (point, scale) in enumerate(zip(points[1:], scales, strict=True)):
            expected = _top_uv_expected(point, scale, alpha)
            actual = [
                rebound.uv_local[index, 0, 0],
                rebound.uv_local[index, 1, 0],
                rebound.uv_external_fields[index, 0, 0],
                rebound.uv_external_fields[index, 1, 0],
            ]
            for value, target in zip(actual, expected, strict=True):
                assert abs(Decimal(value) - target) < abs(target) * tolerance
            for pole in range(3):
                assert (
                    abs(
                        Decimal(rebound.uv[index, pole, 0])
                        - Decimal(rebound.uv_local[index, pole, 0])
                        - Decimal(rebound.uv_external_fields[index, pole, 0])
                    )
                    < tolerance
                )
    empty = program.evaluate_decimal(points[:0], [], model_values=["0.118", "173"])
    assert empty.uv_external_fields.shape == (0, 3, 2)


@pytest.mark.parametrize(
    "arithmetic,precision,engine,coefficient_engine,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-11")),
        ("double", None, "generic", "interpreted", Decimal("3e-11")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-27")),
        ("arbitrary", 60, "generic", "interpreted", Decimal("3e-53")),
    ],
)
def test_complete_uudd_components_precisions_and_scale_binding(
    source, arithmetic, precision, engine, coefficient_engine, tolerance
):
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(json.loads(fixture.read_text())["points"], dtype=str)
    program = distinct_flavour_model_virtual_program(
        source,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
        coefficient_engine=coefficient_engine,
    )
    assert len(program.insertions) == 23 and len(program.currents) == 25
    assert all(
        insertion.rule.has_validated_tensor_basis for insertion in program.insertions
    )
    result = program.evaluate_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    reference_path = os.environ.get("PYAMPLICOL_UUDD_COMPONENT_REFERENCE")
    if reference_path:
        reference = json.loads(Path(reference_path).read_text())
        for index, row in enumerate(reference["rows"]):
            for name, key in (
                ("bare", "bare"),
                ("uv", "uv"),
                ("r2", "r2"),
                ("virtual", "full"),
            ):
                np.testing.assert_allclose(
                    getattr(result, name)[index, :, 0].astype(float),
                    [
                        row[key][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ],
                    rtol=4e-11,
                    atol=1e-14,
                )
    # Exact Born/UV controls provide precision evidence independently of the
    # binary64 complete reference, whose digits cannot certify MP finite terms.
    scales = ["900000", "2100000"]
    rebound = program.evaluate_decimal(
        points[1:], scales, model_values=["0.100000000000000000000000001", "190"]
    )
    with localcontext() as context:
        context.prec = 120
        for index, (point, scale) in enumerate(zip(points[1:], scales, strict=True)):
            expected = _uudd_born_uv(
                point, scale, "0.100000000000000000000000001", "190"
            )
            for actual, target in zip(
                [
                    rebound.born[index, 0],
                    rebound.uv[index, 0, 0],
                    rebound.uv[index, 1, 0],
                ],
                expected,
                strict=True,
            ):
                assert abs(Decimal(actual) - target) < abs(target) * tolerance
            assert all(Decimal(value) == 0 for value in rebound.r2[index, 1:].flat)
            assert all(Decimal(value) == 0 for value in rebound.uv[index, 2:].flat)
    if arithmetic == "double" and engine == "fast":
        # The same compiled native conditional safely selects reglog(0)=0.
        massless = program.evaluate_decimal(
            points[:1], ["1000000"], model_values=["0.118", "0"]
        )
        assert Decimal(massless.uv[0, 0, 0]) == 0


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
    native = program.evaluate_ordered_native(
        points, scales, model_values=["0.118", "132.507"]
    )
    assert native.batch.point_ids == native.born.row_ids == [0, 1, 2]
    assert len(native.component_batches) == 6
    assert all(value.columns == 3 for value in native.component_batches)
    assert native.totals.columns == 19
    from types import SimpleNamespace

    result = SimpleNamespace(
        statuses=native.statuses,
        **{
            name: np.asarray(getattr(native, name).to_decimal(), dtype=object).reshape(
                (len(points), 2) if name == "born" else (len(points), 3, 2)
            )
            for name in ("born", "virtual", "bare", "uv", "r2")
        }
    )
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
