# SPDX-License-Identifier: 0BSD
"""Actual vector states and local colour alternatives in a complete bare control."""

from __future__ import annotations

import json
import os
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_process import gluon_scattering_qcd_program


def _expected(point, alpha_s):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((value**2 for value in vector[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    alpha = parse_exact_ufo(alpha_s)
    # Spin/colour averaged gg Born, with the declared final identical factor 1/2.
    born = (
        E("9/4")
        * (4 * E("pi") * alpha) ** 2
        * (3 - t * u / s**2 - u * s / t**2 - s * t / u**2)
    )
    double = -12 * alpha / (2 * E("pi")) * born
    evaluator = Expression.evaluator_multiple(
        [born, double], [], n_cores=1, jit_compile=False
    )
    return [
        value.to_decimal_tuple()[0]
        for value in evaluator.evaluate_complex_with_prec([], 110)
    ]


@pytest.mark.parametrize(
    "arithmetic,precision,reducer,coefficients,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-11")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-25")),
        ("arbitrary", 50, "generic", "interpreted", Decimal("3e-42")),
    ],
)
def test_gluon_bare_batch_color_terms_and_ward_identity(
    arithmetic, precision, reducer, coefficients, tolerance
):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native gluon control requires licensed Symbolica")
    native = pytest.importorskip("_loop_reduction")
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = np.asarray(
        json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )
    program = gluon_scattering_qcd_program(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=reducer,
        coefficient_engine=coefficients,
    )
    assert len(program.born_dags) == 4
    assert len(program.born_colors) == 6
    assert sorted(Counter(program.born_term_graphs).values()) == [1, 1, 1, 3]
    assert len(program.loop_graphs) == 186
    assert len(program.loop_color_tensors) == 258
    alpha_s = "0.1180000000000000000000001"
    result = program.evaluate_native(
        points, ["1000000"] * len(points), model_values=[alpha_s, "173"]
    )
    born = np.asarray(result.born.to_decimal(), dtype=object)
    poles = np.asarray(result.virtual.to_decimal(), dtype=object).reshape(3, 3, 2)
    with localcontext() as context:
        context.prec = 120
        for point, born_row, pole_row in zip(points, born, poles, strict=True):
            expected = _expected(point, alpha_s)
            for value, target in zip((born_row, pole_row[2]), expected, strict=True):
                assert abs(Decimal(value[0]) - target) < abs(target) * tolerance
                assert abs(Decimal(value[1])) < abs(target) * tolerance
    assert all(status == "success" for row in result.statuses for _, status in row)
    path = os.environ.get("PYAMPLICOL_GG_COMPONENT_REFERENCE")
    if path:
        reference = json.loads(Path(path).read_text())
        np.testing.assert_allclose(
            poles[:, :, 0].astype(float),
            [
                [row["bare"][key] for key in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"]
            ],
            rtol=5e-11,
            atol=1e-10,
        )

    # Replace one external polarization by its actual momentum, using only
    # native column gathers and the same complete Born coefficient programs.
    batch = result.batch
    columns = list(range(batch.bound.columns))
    leg, state_symbols = program.external_states[0]
    assert leg == 0
    for component, symbol in enumerate(state_symbols):
        columns[batch.positions[symbol.get_name()]] = batch.positions[
            program.momenta[0][component].get_name()
        ]
    longitudinal = batch.bound.take_columns(columns)
    amplitudes = native.NativeBatch.concat_columns(
        [
            executor.evaluate_batch(
                longitudinal.take_columns(
                    [batch.positions[p.get_name()] for p in dag.parameters]
                )
            )
            for dag, executor in zip(
                program.born_dags, program.born_executors, strict=True
            )
        ]
    )
    ward = program.born_projector.evaluate_batch(
        amplitudes.reshape_rows(
            batch.size,
            len(program.helicities) * amplitudes.columns,
            batch.point_ids,
        )
    )
    with localcontext() as context:
        context.prec = 120
        for value, ordinary, point in zip(ward.to_decimal(), born, points, strict=True):
            scale = abs(Decimal(ordinary[0])) * Decimal(point[0, 0]) ** 2
            assert all(abs(Decimal(x)) < scale * tolerance for x in value)
    empty = program.evaluate_decimal(points[:0], [], model_values=[alpha_s, "173"])
    assert empty.virtual.shape == (0, 3, 2) and empty.statuses == ()


@pytest.mark.parametrize(
    "arithmetic,precision,reducer,coefficients,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-11")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-27")),
        ("arbitrary", 50, "generic", "interpreted", Decimal("3e-45")),
    ],
)
def test_gluon_model_components_and_per_point_uv_scales(
    arithmetic, precision, reducer, coefficients, tolerance
):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native gluon control requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    source_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not source_path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from symbolica import E, Expression

    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_expressions import parse_exact_ufo
    from pyamplicol.loops.physical_counterterms import (
        gluon_scattering_model_virtual_program,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(source_path))
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = np.asarray(
        json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )
    program = gluon_scattering_model_virtual_program(
        source,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=reducer,
        coefficient_engine=coefficients,
    )
    assert len(program.insertions) == 119 and len(program.external_fields) == 48
    assert any(current.term_count > 1 for current in program.currents)
    assert any(len(current.born_indices) == 3 for current in program.field_currents)
    result = program.evaluate_decimal(
        points, ["1000000"] * len(points), model_values=["0.118", "173"]
    )
    path = os.environ.get("PYAMPLICOL_GG_COMPONENT_REFERENCE")
    if path:
        reference = json.loads(Path(path).read_text())
        for name, key in (
            ("bare", "bare"),
            ("uv", "uv"),
            ("r2", "r2"),
            ("virtual", "full"),
            ("uv_local", "uv_vertex"),
            ("uv_external_fields", "wf"),
        ):
            np.testing.assert_allclose(
                getattr(result, name)[:, :, 0].astype(float),
                [
                    [
                        row[key][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ]
                    for row in reference["rows"]
                ],
                rtol=5e-11,
                atol=1e-10,
            )

    # Independent heavy-quark decoupling/field formulae, evaluated beyond the
    # reference's binary64 precision. Model parameters remain global; muR is
    # distinct per lane, and its original decimal digits reach the evaluator.
    alpha = "0.100000000000000000000000001"
    mass = "190.000000000000000000000001"
    scales = ["900000.000000000000000000001", "2100000"]
    rebound = program.evaluate_decimal(points[1:], scales, model_values=[alpha, mass])
    with localcontext() as context:
        context.prec = 120
        for index, (point, scale) in enumerate(zip(points[1:], scales, strict=True)):
            born = _expected(point, alpha)[0]
            coupling = parse_exact_ufo(alpha)
            logarithm = (parse_exact_ufo(scale) / parse_exact_ufo(mass) ** 2).log()
            local_finite = coupling / (3 * E("pi")) * logarithm
            ratios = [
                local_finite,
                -7 * coupling / (2 * E("pi")),
                -2 * local_finite,
                -2 * coupling / (3 * E("pi")),
            ]
            targets = Expression.evaluator_multiple(
                ratios, [], n_cores=1, jit_compile=False
            ).evaluate_complex_with_prec([], 110)
            actual = [
                rebound.uv_local[index, 0],
                rebound.uv_local[index, 1],
                rebound.uv_external_fields[index, 0],
                rebound.uv_external_fields[index, 1],
            ]
            for pair, target in zip(actual, targets, strict=True):
                value = born * target.to_decimal_tuple()[0]
                assert abs(Decimal(pair[0]) - value) < abs(value) * tolerance
                assert abs(Decimal(pair[1])) < abs(value) * tolerance
    assert all(status == "success" for row in rebound.statuses for _, status in row)
    empty = program.evaluate_decimal(points[:0], [], model_values=[alpha, mass])
    assert empty.virtual.shape == (0, 3, 2) and empty.statuses == ()
    with pytest.raises(NotImplementedError, match="SU\\(Nc\\) continuation"):
        gluon_scattering_model_virtual_program(source, color_accuracy="lc")
