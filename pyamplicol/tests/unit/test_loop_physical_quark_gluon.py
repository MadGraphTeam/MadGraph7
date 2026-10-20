# SPDX-License-Identifier: 0BSD
"""Mixed Dirac/vector control with isolated calibrated fermion R2 insertion."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.loop_physics


def _exact_values(point, scale, alpha, mass):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    p = [[parse_exact_ufo(value) for value in leg] for leg in point]

    def square(vector):
        return vector[0] ** 2 - sum((value**2 for value in vector[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    coupling = parse_exact_ufo(alpha)
    born = (4 * E("pi") * coupling) ** 2 * (
        E("16/27") * (t / u + u / t) - E("4/3") * (t**2 + u**2) / s**2
    )
    double = -E("26/3") * coupling / (2 * E("pi")) * born
    local_finite = (
        born
        * coupling
        / (3 * E("pi"))
        * (parse_exact_ufo(scale) / parse_exact_ufo(mass) ** 2).log()
    )
    local_pole = -7 * born * coupling / (2 * E("pi"))
    field_pole = -born * coupling / (3 * E("pi"))
    evaluator = Expression.evaluator_multiple(
        [born, double, local_finite, local_pole, -local_finite, field_pole],
        [],
        n_cores=1,
        jit_compile=False,
    )
    return [
        value.to_decimal_tuple()[0]
        for value in evaluator.evaluate_complex_with_prec([], 110)
    ]


@pytest.mark.parametrize(
    "arithmetic,precision,reducer,coefficients,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-11")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-26")),
        ("arbitrary", 60, "generic", "interpreted", Decimal("3e-51")),
    ],
)
def test_quark_gluon_isolated_r2_complete_components_and_precise_inputs(
    loop_reference_path,
    loop_sm_ufo,
    loop_native,
    arithmetic,
    precision,
    reducer,
    coefficients,
    tolerance,
):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native mixed-state control requires licensed Symbolica")
    source_path = str(loop_sm_ufo)
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        NativeModelVirtualProgram,
        quark_gluon_model_virtual_program,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(source_path))
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/quark_gluon_qcd.json"
    )
    points = np.asarray(
        json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )
    program = quark_gluon_model_virtual_program(
        source,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=reducer,
        coefficient_engine=coefficients,
    )
    assert len(program.bare.born_dags) == 3 and len(program.bare.loop_graphs) == 45
    assert len(program.insertions) == 63 and len(program.external_fields) == 18
    selected = [
        item
        for item in program.insertions
        if any(entry.vertex_name == "V_R2UU" for entry in item.rule.entries)
    ]
    assert len(selected) == 2
    assert all(item.rule.tensor_representatives for item in selected)
    isolated = NativeModelVirtualProgram(
        program.bare, selected, source_bindings=program.source_bindings
    ).evaluate_decimal(points, ["1000000"] * 3, model_values=["0.118", "173"])
    result = program.evaluate_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    path = str(loop_reference_path("quark-gluon-component-control"))
    reference = json.loads(Path(path).read_text())
    for actual, key in (
        (isolated.r2, "r2_quark_two_point"),
        (result.bare, "bare"),
        (result.uv, "uv"),
        (result.r2, "r2"),
        (result.virtual, "full"),
        (result.uv_local, "uv_vertex"),
        (result.uv_external_fields, "wf"),
    ):
        np.testing.assert_allclose(
            actual[:, :, 0].astype(float),
            [
                [row[key][pole] for pole in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"]
            ],
            rtol=5e-11,
            atol=1e-11,
        )

    alpha, mass = "0.100000000000000000000000001", "190.000000000000000000000001"
    scales = ["900000.000000000000000000001", "2100000"]
    rebound = program.evaluate_decimal(points[1:], scales, model_values=[alpha, mass])
    with localcontext() as context:
        context.prec = 120
        for index, (point, scale) in enumerate(zip(points[1:], scales, strict=True)):
            targets = _exact_values(point, scale, alpha, mass)
            actual = [
                rebound.born[index],
                rebound.virtual[index, 2],
                rebound.uv_local[index, 0],
                rebound.uv_local[index, 1],
                rebound.uv_external_fields[index, 0],
                rebound.uv_external_fields[index, 1],
            ]
            for pair, target in zip(actual, targets, strict=True):
                assert abs(Decimal(pair[0]) - target) < abs(target) * tolerance
                assert abs(Decimal(pair[1])) < abs(target) * tolerance
            assert abs(Decimal(rebound.uv[index, 0, 0])) < tolerance
    assert all(status == "success" for row in rebound.statuses for _, status in row)
    assert all(Decimal(x) == 0 for x in isolated.r2[:, 1:].flat)
    empty = program.evaluate_decimal(points[:0], [], model_values=[alpha, mass])
    assert empty.virtual.shape == (0, 3, 2) and empty.statuses == ()
