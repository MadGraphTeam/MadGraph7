# SPDX-License-Identifier: 0BSD
"""Retained directed instructions bind the actual source scalar/current owner."""

import json
import os
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .test_loop_physical_correlations import (
    _close,
    _conjugate,
    _decimals,
    _histories,
)
from .test_loop_physical_counterterms import _uudd_born_uv


@pytest.fixture(scope="module")
def source_directed():
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed source generation and loop_sm reference required")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram
    from pyamplicol.models.loop_import import load_raw_loop_model

    prepared = prepare_loop_request_model(
        load_raw_loop_model(path),
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )
    source = prepared.compile(
        (2, -2),
        (1, -1),
        selected_corrections=["QCD"],
        parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        coefficient_engine="jit",
        reduction_engine="fast",
    )
    directed = NativeDirectedModelProgram(source, _histories(source.virtual.bare))
    fixture = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    points = np.asarray(json.loads(fixture.read_text())["points"][:2], dtype=str)
    return source, directed.prepared, points


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance,bit",
    [
        ("double", None, "fast", "4e-11", 20),
        ("double-double", None, "generic", "4e-25", 70),
        ("arbitrary", 70, "generic", "4e-57", 170),
    ],
)
def test_source_directed_retained_terms_components_and_exact_updates(
    source_directed, monkeypatch, arithmetic, precision, engine, tolerance, bit
):
    from pyamplicol.loops import prepared_correlations
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    source, prepared, points = source_directed

    def forbidden(*args, **kwargs):
        raise AssertionError("retained directed binding repeated algebra preparation")

    # A new carrier binds saved dependency instructions and the existing owner;
    # it must not repeat colour/CT projector construction or expression lowering.
    for name in ("prepare_directed_bare", "prepare_directed_model", "_saved"):
        monkeypatch.setattr(prepared_correlations, name, forbidden)
    for name in ("_born_projection_state", "_family_projection_state"):
        monkeypatch.setattr(NativeBareProcessProgram, name, forbidden)
    rebound = source.rebind(
        arithmetic=arithmetic, precision=precision, reduction_engine=engine
    )
    directed = prepared.bind(rebound)
    assert directed.model is rebound.virtual
    assert directed.bare.model_state is rebound.source_bare.state
    assert directed.prepared is prepared
    assert directed.model.prepared is prepared.plan
    scales = ["1000000", "900000"]
    baseline = rebound.evaluate_ordered_decimal(points, scales)
    result = directed.evaluate_decimal(
        points, scales, point_ids=[np.iinfo(np.uintp).max, 7]
    )
    tolerance = Decimal(tolerance)
    with localcontext() as context:
        context.prec = 120
        for name in (
            "born",
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
        ):
            values = _decimals(getattr(result, name))
            # Inclusive identity is transport consistency, not an independent
            # physics oracle: the source currents are intentionally shared.
            _close(values[:, 0], _decimals(getattr(baseline, name)), tolerance)
            _close(values[:, 1], values[:, 0] * Decimal(4) / 3, tolerance)
            _close(values[:, 3], _conjugate(values[:, 2]), tolerance)
        _close(
            _decimals(result.virtual),
            _decimals(result.first) + _decimals(result.second),
            tolerance,
        )
        _close(
            _decimals(result.first[:, 2]),
            _conjugate(_decimals(result.second[:, 3])),
            tolerance,
        )
        _close(
            _decimals(result.virtual),
            _decimals(result.bare) + _decimals(result.uv) + _decimals(result.r2),
            tolerance,
        )
        assert abs(Decimal(result.virtual[0, 2, 0, 0, 1])) > Decimal("1e-4")
        # Independent analytic Born and decoupling UV expressions do not use the
        # prepared directed or ordinary projector instructions.
        for index, (point, scale) in enumerate(zip(points, scales, strict=True)):
            expected = _uudd_born_uv(point, scale)
            actual = (
                result.born[index, 0, 0, 0],
                result.uv[index, 0, 0, 0, 0],
                result.uv[index, 0, 0, 1, 0],
            )
            for value, target in zip(actual, expected, strict=True):
                assert abs(Decimal(value) - target) < tolerance * abs(target)
        old = Decimal("0.118")
        alpha = old * (1 + Decimal(2) ** -bit)
        rebound.update_parameters({"aS": str(alpha)})
        try:
            updated = directed.evaluate_decimal(points[:1], scales[:1], point_ids=[42])
            for name, power in (
                ("born", 2),
                ("virtual", 3),
                ("bare", 3),
                ("uv", 3),
                ("r2", 3),
            ):
                expected = _decimals(getattr(result, name)[:1]) * (alpha / old) ** power
                _close(_decimals(getattr(updated, name)), expected, tolerance)
            assert Decimal(updated.born[0, 0, 0, 0]) != Decimal(result.born[0, 0, 0, 0])
        finally:
            rebound.update_parameters({"aS": str(old)})
    sectors = len(directed.bare.order_layout.virtual)
    assert (
        len(result.channels)
        == sum(
            directed.directions * len(family.sector_indices)
            for family in directed.bare.families
        )
        > 0
    )
    assert len(result.statuses) == 2
    for point, row in zip((np.iinfo(np.uintp).max, 7), result.statuses, strict=True):
        for (lane, status), _channel in zip(row, result.channels, strict=True):
            assert lane == point
            assert status == "success"
    empty = directed.evaluate_decimal(points[:0], [], point_ids=[])
    assert empty.virtual.shape == (0, 4, sectors, 3, 2)
    assert empty.statuses == ()


def test_directed_preparation_is_immutable_and_rejects_another_plan(source_directed):
    source, prepared, _ = source_directed
    with pytest.raises(FrozenInstanceError):
        prepared.assembly_state = b"changed"
    with pytest.raises(ValueError, match="different retained model plan"):
        replace(prepared, plan=object()).bind(source)
    with pytest.raises(ValueError, match="different retained bare plan"):
        replace(prepared, bare=replace(prepared.bare, plan=object())).bind(source)


def test_directed_reducer_failure_maps_dense_rows_to_original_point(source_directed):
    source, prepared, points = source_directed
    directed = prepared.bind(
        source.rebind(arithmetic="double", reduction_engine="fast")
    )
    family = directed.bare.families[0]
    original = family.reducer

    class FailedDirection:
        def evaluate_batch(self, shifts, masses, scales, coefficients, **kwargs):
            assert coefficients.row_ids == list(range(coefficients.rows))
            return SimpleNamespace(statuses=[(1, "injected reduction failure")])

    family.reducer = FailedDirection()
    point = np.iinfo(np.uintp).max
    channels = [
        (direction, sector)
        for direction in range(directed.directions)
        for sector in family.sector_indices
    ]
    try:
        with pytest.raises(ArithmeticError) as failure:
            directed.evaluate_native(points[:1], ["1000000"], point_ids=[point])
        expected = (point, *channels[1], family.index, "injected reduction failure")
        assert str(expected) in str(failure.value)
    finally:
        family.reducer = original
