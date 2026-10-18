# SPDX-License-Identifier: 0BSD
"""Complete five-leg QCD interference with independent component and IR controls."""

from __future__ import annotations

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def control():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed native generation required")
    pytest.importorskip("_loop_reduction")
    source_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not source_path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        quark_three_gluon_model_virtual_program,
    )
    from pyamplicol.models.loop_import import load_raw_loop_model

    program = quark_three_gluon_model_virtual_program(
        build_loop_model(load_raw_loop_model(source_path)),
        arithmetic="double",
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/quark_three_gluon_qcd.json"
    )
    points = np.asarray(
        json.loads(fixture.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )
    return program, points


def _components(result):
    return tuple(
        getattr(result, name)
        for name in ("bare", "uv", "r2", "virtual", "uv_local", "uv_external_fields")
    )


def _check_reference(result):
    path = os.environ.get("PYAMPLICOL_QUARK_THREE_GLUON_COMPONENT_REFERENCE")
    if not path:
        return
    reference = json.loads(Path(path).read_text())
    np.testing.assert_allclose(
        result.born[:, 0].astype(float),
        [row["born"] for row in reference["rows"]],
        rtol=2e-11,
        atol=2e-15,
    )
    for actual, key in zip(
        _components(result),
        ("bare", "uv", "r2", "full", "uv_vertex", "wf"),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual[:, :, 0].astype(float),
            [
                [row[key][pole] for pole in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"]
            ],
            rtol=2e-9,
            atol=2e-14,
            err_msg=key,
        )


@pytest.mark.parametrize(
    "arithmetic,precision,bit,tolerance",
    [
        ("double", None, 0, "2e-9"),
        ("double-double", None, 70, "3e-25"),
        ("arbitrary", 80, 180, "1e-60"),
    ],
)
def test_five_leg_components_poles_and_exact_coupling(
    control, arithmetic, precision, bit, tolerance
):
    from symbolica import E, Expression

    original, points = control
    program = (
        original
        if arithmetic == "double"
        else original.rebind(arithmetic=arithmetic, precision=precision)
    )
    assert str(program.bare.initial_average) == "216"
    assert program.bare.order_layout.is_single
    assert dict(program.bare.order_layout.born[0].order) == {"QCD": 6}
    assert dict(program.bare.order_layout.virtual[0].order) == {"QCD": 8}
    result = program.evaluate_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "173"]
    )
    assert all(status == "success" for row in result.statuses for _, status in row)
    _check_reference(result)
    factor = (
        Expression.evaluator_multiple(
            [-E("35/3") * E("118/1000") / (2 * E("pi"))],
            [],
            n_cores=1,
            jit_compile=False,
        )
        .evaluate_complex_with_prec([], 110)[0]
        .to_decimal_tuple()[0]
    )
    with localcontext() as context:
        context.prec = 110
        tolerance = Decimal(tolerance)
        for index in range(3):
            expected = Decimal(result.born[index, 0]) * factor
            assert (
                abs(Decimal(result.virtual[index, 2, 0]) - expected)
                < abs(expected) * tolerance
            )
        alpha_factor = Decimal(2) if not bit else 1 + Decimal(2) ** -bit
        changed = program.evaluate_decimal(
            points[:1],
            ["1000000"],
            model_values=[str(Decimal("0.118") * alpha_factor), "173"],
        )
        born = Decimal(result.born[0, 0])
        assert Decimal(changed.born[0, 0]) != born
        assert (
            abs(Decimal(changed.born[0, 0]) - born * alpha_factor**3)
            < abs(born) * tolerance
        )
        # All physical components carry four powers of alpha_s. Local and
        # external UV are checked separately despite their finite cancellation.
        for before, after in zip(
            _components(result), _components(changed), strict=True
        ):
            for pole in range(3):
                value = Decimal(before[0, pole, 0])
                expected = value * alpha_factor**4
                assert (
                    abs(Decimal(after[0, pole, 0]) - expected) < abs(born) * tolerance
                )
        for name in ("bare", "r2", "virtual", "uv_local", "uv_external_fields"):
            assert getattr(changed, name)[0, 0, 0] != getattr(result, name)[0, 0, 0]
    empty = program.evaluate_decimal(points[:0], [], model_values=["0.118", "173"])
    assert empty.virtual.shape == (0, 3, 2)
    # Original output owns its values across parameter updates and empty calls.
    permuted = program.evaluate_decimal(
        points[[2, 0]], ["1000000"] * 2, model_values=["0.118", "173"]
    )
    for expected, actual in zip(
        _components(result), _components(permuted), strict=True
    ):
        with localcontext() as context:
            context.prec = 110
            for before, after in zip(expected[[2, 0]].flat, actual.flat, strict=True):
                assert abs(Decimal(before) - Decimal(after)) < abs(
                    Decimal(result.born[0, 0])
                ) * tolerance
