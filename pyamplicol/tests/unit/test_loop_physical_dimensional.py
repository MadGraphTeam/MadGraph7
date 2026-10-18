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
        ("double", None, "fast-tensor", "jit", Decimal("3e-10")),
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


def test_direct_gluon_partial_and_fixed_strategy_rescue():
    """Bounded direct-only retries and actual native cut-failure compaction."""
    symbolica = pytest.importorskip("symbolica")
    backend = pytest.importorskip("_loop_reduction")
    source_path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not symbolica.is_licensed() or not source_path:
        pytest.skip("licensed Symbolica and the loop_sm reference are required")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.physical_counterterms import (
        NativeModelVirtualProgram,
        gluon_scattering_model_virtual_program,
    )
    from pyamplicol.loops.physical_process import NativeBareProcessProgram
    from pyamplicol.loops.stability import ArithmeticSpec, NativeModelStabilityProgram
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(source_path))
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    point = np.asarray(json.loads(fixture.read_text())["points"][:1], dtype=str)
    model_values = ["0.118", "173"]
    direct = gluon_scattering_model_virtual_program(
        source, r2_strategy="direct", reduction_engine="fast", coefficient_engine="jit"
    )
    ordinary = direct.evaluate_ordered_native(
        point, ["1000000"], model_values=model_values
    )
    # Independent N4-only currents and source R2 trees; no dimensional program
    # or Delta value is passed into this model-rule oracle.
    bare = direct.bare
    n4 = NativeBareProcessProgram(
        bare.process,
        state_kinds=bare.state_kinds,
        initial_average=bare.initial_average,
        model_expressions=bare.model_expressions,
        model_inputs=bare.model_inputs,
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    model = NativeModelVirtualProgram(
        n4,
        direct.matched_insertions,
        source_bindings=direct.source_bindings,
        external_fields=direct.external_fields,
    )
    assert not n4.dimensional_numerators
    expected = model.evaluate_ordered_native(
        point, ["1000000"], model_values=model_values
    )
    for actual, oracle in zip(
        [ordinary.born, *ordinary.component_batches, ordinary.totals],
        [expected.born, *expected.component_batches, expected.totals],
        strict=True,
    ):
        assert actual.agreement_rows(oracle, "3e-10", "1e-10") == [(0, True)]

    data, ids = np.repeat(point, 2, axis=0), [17, 91]
    family = next(
        f
        for f in bare.families
        if f.kinematics.propagators == 4
        and f.rational_reducer is not None
        and any(p.get_name() == "UFO::MT" for p in f.mass_parameters)
    )
    assert family.index > bare.families[0].index

    class DegenerateOneLane:
        def __init__(self, reducer):
            self.reducer = reducer

        def evaluate_batch(self, shifts, masses, scales, coefficients, *, s_matrix):
            good = [i for i, row in enumerate(shifts.row_ids) if row != 91]
            bad = [row for row in shifts.row_ids if row == 91]

            def degenerate(values, entries):
                replacements = backend.NativeBatch.from_decimal(
                    [(str(value), "0") for value in entries] * len(bad),
                    len(bad),
                    values.columns,
                    row_ids=bad,
                    **bare.options,
                )
                return backend.NativeBatch.merge_rows(
                    values.row_ids, [values.take_rows(good), replacements]
                )

            # Distinct collinear timelike shifts avoid repeated denominators,
            # but give a degenerate Gram system. Ninja supplies the numerical
            # failure status; the test never synthesizes statuses or integrals.
            distinct_shifts = [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]
            invariants = [0, 1, 4, 9, 1, 0, 1, 4, 4, 1, 0, 1, 9, 4, 1, 0]
            return self.reducer.evaluate_batch(
                degenerate(shifts, distinct_shifts),
                masses,
                scales,
                coefficients,
                s_matrix=degenerate(s_matrix, invariants),
            )

    for attribute, stage in (
        ("reducer", "loop reduction"),
        ("rational_reducer", "dimensional remainder reduction"),
    ):
        original = getattr(family, attribute)
        setattr(family, attribute, DegenerateOneLane(original))
        try:
            sparse = direct.evaluate_ordered_partial_native(
                data,
                ["1000000"] * 2,
                model_values=model_values,
                point_ids=ids,
            )
            assert sparse.batch.point_ids == [17]
            assert {failure.point_id for failure in sparse.failures} == {91}
            assert any(
                failure.stage == stage and failure.family_index == family.index
                for failure in sparse.failures
            )
            assert sparse.totals.agreement_rows(
                ordinary.totals.take_rows([0], row_ids=[17]), "3e-10", "1e-10"
            ) == [(17, True)]
        finally:
            setattr(family, attribute, original)

    stability = NativeModelStabilityProgram(
        direct,
        [ArithmeticSpec("arbitrary", 50), ArithmeticSpec("arbitrary", 80)],
        relative_tolerance="1e-8",
        absolute_tolerance="1e-8",
    )
    # A genuine finite original scale overflows the requested carrier. Every
    # retry reconstructs the complete dimensional numerator and epsilon sectors.
    result = stability.evaluate_native(
        data,
        ["1000000", "1e400"],
        model_values=model_values,
        point_ids=ids,
        verify=True,
    )
    assert result.point_ids == (17, 91)
    assert result.r2_strategy == "direct"
    assert result.diagnostics[0].status == "success"
    assert result.diagnostics[1].status == "rescued"
    assert result.diagnostics[1].verification_pair == (
        ArithmeticSpec("arbitrary", 50),
        ArithmeticSpec("arbitrary", 80),
    )
    assert all(
        view.r2_strategy == "direct" and view.bare.dimensional_numerators
        for view in stability._views.values()
    )
    assert all(
        all(item.rule.kind != "R2" for item in view.insertions)
        for view in stability._views.values()
    )
    assert all(
        value.row_ids == ids and value.precision_bits == 53
        for value in [result.born, *result.component_batches, result.totals]
    )
    assert result.totals.take_rows([0]).agreement_rows(
        ordinary.totals.take_rows([0], row_ids=[17]), "3e-10", "1e-10"
    ) == [(17, True)]
    empty = stability.evaluate_native(
        point[:0], [], model_values=model_values, verify=True
    )
    assert empty.point_ids == () and empty.totals.rows == 0
    direct.r2_strategy = "model"
    try:
        with pytest.raises(ValueError, match="strategy or colour scope changed"):
            stability.evaluate_native(point[:0], [], model_values=model_values)
    finally:
        direct.r2_strategy = "direct"


@pytest.mark.parametrize(
    "arithmetic,precision,engine",
    [
        ("double", None, "fast"),
        ("double-double", None, "generic"),
        ("arbitrary", 50, "generic"),
    ],
)
def test_dimensional_family_failure_status_keeps_native_lane_identity(
    arithmetic, precision, engine
):
    backend = pytest.importorskip("_loop_reduction")
    options = dict(arithmetic=arithmetic, precision=precision)
    plan = backend.PolynomialPlan(
        4, ["mu2^2", "mu2", "0"], [], engine=engine, **options
    )
    shifts = [0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 1, 0, 4, 0, 0, 1]
    degenerate = [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]

    def native(values, columns):
        return backend.NativeBatch.from_decimal(
            [(str(value), "0") for value in values],
            2,
            columns,
            row_ids=[17, 91],
            **options,
        )

    inputs = [
        native(shifts + degenerate, 16),
        native([29929] * 8, 4),
        native([1000000] * 2, 1),
        native([], 0),
    ]
    result = plan.evaluate_batch(*inputs)
    assert result.statuses[0] == (17, "success")
    assert result.statuses[1][0] == 91 and result.statuses[1][1] != "success"
    values = result.successful_values()
    assert values.row_ids == [17] and not values.nonfinite_rows()
    again = plan.evaluate_batch(*(value.take_rows([0]) for value in inputs))
    assert again.values.to_decimal() == values.to_decimal()
