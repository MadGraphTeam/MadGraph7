# SPDX-License-Identifier: 0BSD
"""Closed dimensional sectors through native physical family reduction."""

from __future__ import annotations

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_process import _reduction_numerators


def test_closed_sector_layout_keeps_epsilon_and_mu2_distinct():
    powers = (
        (2, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 1, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 2),
    )
    names = ("c0", "c1", "c2", "c3")
    assert _reduction_numerators(powers, names) == ["c0*q0^2", "0", "0"]
    assert _reduction_numerators(powers, names, rational=True) == [
        "c1*mu2^1",
        "c2*q1^1",
        "c3",
    ]
    with pytest.raises(NotImplementedError, match="epsilon power"):
        _reduction_numerators([(0, 0, 0, 0, 0, 3)], ["c"])


@pytest.mark.parametrize(
    "arithmetic,precision,reducer,coefficients,pole_tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-10")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-23")),
        ("arbitrary", 50, "generic", "interpreted", Decimal("3e-40")),
    ],
)
def test_complete_gluon_direct_sectors_and_uv(
    arithmetic, precision, reducer, coefficients, pole_tolerance
):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native dimensional control requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    source_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    reference_path = os.environ.get("PYAMPLICOL_GG_COMPONENT_REFERENCE")
    if not source_path or not reference_path:
        pytest.skip("set the loop_sm and independent gg component references")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        NativeModelVirtualProgram,
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
        r2_strategy="direct",
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=reducer,
        coefficient_engine=coefficients,
    )
    assert len(program.bare.recurrences) == 186
    assert all(r.kernel_provider is not None for r in program.bare.recurrences)
    assert all(not r.dag.direct_r2_certified for r in program.bare.recurrences)
    assert all(i.rule.kind != "R2" for i in program.insertions)
    assert program.insertions and program.field_currents
    result = program.evaluate_decimal(
        points, ["1000000"] * len(points), model_values=["0.118", "173"]
    )
    assert result.r2_strategy == "direct"
    # Reuse only the ordinary states/graph program. This wrapper independently
    # evaluates the source R2 trees in the same carrier, without inserting them
    # into the direct result or subtracting two integrated amplitudes.
    model = NativeModelVirtualProgram(
        program.bare,
        program.matched_insertions,
        source_bindings=program.source_bindings,
        external_fields=program.external_fields,
    )
    source_result = model.evaluate_decimal(
        points, ["1000000"] * len(points), model_values=["0.118", "173"]
    )
    reference = json.loads(Path(reference_path).read_text())
    for name, key in (
        ("bare", "bare"),
        ("r2", "r2"),
        ("uv", "uv"),
        ("virtual", "full"),
        ("uv_local", "uv_vertex"),
        ("uv_external_fields", "wf"),
    ):
        np.testing.assert_allclose(
            getattr(result, name)[:, :, 0].astype(float),
            [
                [row[key][p] for p in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"]
            ],
            rtol=5e-11,
            atol=1e-10,
        )
    # The complete rational remainder is finite; do not force individual
    # graph/family poles to zero before the coherent native sum.
    with localcontext() as context:
        context.prec = 120
        for born, remainder in zip(result.born, result.r2, strict=True):
            scale = abs(Decimal(born[0]))
            for pair in remainder[1:]:
                assert all(abs(Decimal(v)) < scale * pole_tolerance for v in pair)
        for born, direct, source_value in zip(
            result.born, result.r2, source_result.r2, strict=True
        ):
            scale = abs(Decimal(born[0]))
            for actual, expected in zip(direct.flat, source_value.flat, strict=True):
                assert abs(Decimal(actual) - Decimal(expected)) < scale * pole_tolerance
    assert all(status == "success" for row in result.statuses for _, status in row)
    empty = program.evaluate_decimal(points[:0], [], model_values=["0.118", "173"])
    assert empty.r2.shape == (0, 3, 2) and empty.statuses == ()
