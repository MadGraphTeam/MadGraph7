# SPDX-License-Identifier: 0BSD
"""Retained complete W source controls; setup never regenerates physical graphs."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from .test_loop_massive_vector_sources import _scale, _wavefunctions
from .test_loop_physical_correlations import _decimals

RETAINED_CONTROL = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_CONTROL is None:
        pytest.skip("run with actual retained formal W owners and prepared banks")
    return RETAINED_CONTROL


def _close_component(actual, expected, tolerance, *, scale=None):
    actual, expected = (
        np.asarray(actual, dtype=object),
        np.asarray(expected, dtype=object),
    )
    assert actual.shape == expected.shape
    norm = max(
        Decimal("1e-100"),
        *(abs(value) for value in (expected if scale is None else scale).flat),
    )
    assert all(
        abs(a - b) <= tolerance * norm
        for a, b in zip(actual.flat, expected.flat, strict=True)
    )


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "1e-8", 20),
        ("double-double", None, "1e-24", 70),
        ("arbitrary", 70, "1e-60", 170),
    ),
)
def test_complete_operator_components_sources_and_exact_rebinding(
    retained, monkeypatch, accuracy, arithmetic, precision, tolerance, bit
):
    from pyamplicol.loops import directed_color_projection, prepared_correlations
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="interpreted",
        projector_engine="interpreted",
    )
    model = retained["models"][accuracy].rebind(**options)
    sources = NativeSpinSourcePlan(model.bare, ((3,), (4,), (3, 4)))

    def forbidden(*args, **kwargs):
        raise AssertionError("retained directed bank repeated colour algebra")

    for module, names in (
        (
            directed_color_projection,
            ("prepare_directed_color_bank", "project_directed_color_columns"),
        ),
        (
            prepared_correlations,
            (
                "prepare_directed_complete",
                "prepare_directed_bare",
                "prepare_directed_model",
                "_saved",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    ordinary = retained["prepared"][accuracy].bind(model)
    classes = {
        key: value.bind(model)
        for key, value in retained["classes"][accuracy].items()
        if key
    }
    bank = ordinary.prepared.bare.color_bank
    assert all(view.prepared.bare.color_bank is bank for view in classes.values())
    point = retained["point"]
    scales, ids = ["1000000"], [2**64 - 1]
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        tiny = Decimal(2) ** -bit
        old = Decimal("0.118")
        model.bare.model_state.update({"aS": str(old * (1 + tiny))})
        snapshot = model.bare.model_state.snapshot()
        baseline = ordinary.evaluate_decimal([point], scales, point_ids=ids)
        expected = model.evaluate_ordered_decimal([point], scales)
        names = (
            "born",
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
        )
        for name in names:
            values, target = (
                _decimals(getattr(baseline, name)),
                _decimals(getattr(expected, name)),
            )
            _close_component(values[:, 0], target, tolerance, scale=values)
            # Rows1..9 are the entire three-coloured-leg dipole family. The
            # gluon self insertion is CA=Nc, and the final physical split is TR.
            for start in (1, 4, 7):
                _close_component(
                    values[:, start : start + 3].sum(axis=1),
                    0 * values[:, 0],
                    tolerance,
                    scale=values,
                )
            _close_component(values[:, 9], 3 * values[:, 0], tolerance, scale=values)
            _close_component(values[:, 10], values[:, 0] / 2, tolerance, scale=values)
        _close_component(
            _decimals(baseline.virtual),
            _decimals(baseline.first) + _decimals(baseline.second),
            tolerance,
        )
        _close_component(
            _decimals(baseline.virtual),
            _decimals(baseline.bare) + _decimals(baseline.uv) + _decimals(baseline.r2),
            tolerance,
        )
        assert all(
            lane == ids[0] and status == "success"
            for row in baseline.statuses
            for lane, status in row
        )
        vectors = _wavefunctions(
            model.bare.backend, point, 3, [-1, 0, 1], False, options
        )
        gluons = _wavefunctions(model.bare.backend, point, 4, [-1, 1], False, options)
        source_ids = [2**64 - 1, 7, 42, 0, 2**63, 9]
        selected = [
            *vectors,
            _scale(vectors[0], Decimal(2), Decimal(3)),
            [("0", "0")] * 4,
            _scale(vectors[0], 1 + tiny, Decimal(0)),
        ]
        singleton = classes[3,].evaluate_decimal(
            [point] * 6,
            scales * 6,
            point_ids=source_ids,
            spin_sources=sources.prepare_decimal({3: selected}, source_ids),
        )
        joint = classes[3, 4].evaluate_decimal(
            [point] * 6,
            scales * 6,
            point_ids=source_ids,
            spin_sources=sources.prepare_decimal(
                {
                    3: [v for v in vectors for _ in gluons],
                    4: [g for _ in vectors for g in gluons],
                },
                source_ids,
            ),
        )
        gluon_singleton = classes[4,].evaluate_decimal(
            [point] * 2,
            scales * 2,
            point_ids=source_ids[:2],
            spin_sources=sources.prepare_decimal({4: gluons}, source_ids[:2]),
        )
        for name in (*names, "first", "second"):
            values = _decimals(getattr(singleton, name))
            target = _decimals(getattr(baseline, name))[0]
            _close_component(values[:3].sum(axis=0), target, tolerance, scale=values)
            _close_component(values[3], 13 * values[0], tolerance, scale=values)
            _close_component(values[4], 0 * values[0], tolerance, scale=values)
            _close_component(
                _decimals(getattr(joint, name)).sum(axis=0),
                target,
                tolerance,
                scale=values,
            )
            _close_component(
                _decimals(getattr(gluon_singleton, name)).sum(axis=0),
                target,
                tolerance,
                scale=values,
            )
        born = _decimals(singleton.born)
        shift = born[5, 0, 0, 0] - born[0, 0, 0, 0]
        expected_shift = born[0, 0, 0, 0] * (2 * tiny + tiny * tiny)
        assert expected_shift != 0
        assert abs(shift - expected_shift) < abs(expected_shift) * Decimal("2e-4")
        assert model.bare.model_state.snapshot() is snapshot
        empty = classes[3, 4].evaluate_decimal(
            np.empty((0, len(point), 4), dtype=object),
            [],
            point_ids=[],
            spin_sources=sources.prepare_decimal({3: [], 4: []}, []),
        )
        assert empty.virtual.shape == (0, 11, 1, 3, 2) and empty.statuses == ()
        tail = classes[3,].evaluate_decimal(
            [point],
            scales,
            point_ids=[42],
            spin_sources=sources.prepare_decimal({3: [vectors[2]]}, [42]),
        )
        for name in (*names, "first", "second"):
            _close_component(
                _decimals(getattr(tail, name))[0],
                _decimals(getattr(singleton, name))[2],
                tolerance,
                scale=_decimals(getattr(singleton, name)),
            )


def test_bank_rejects_changed_actual_metadata_and_identity_rows(retained):
    from pyamplicol.loops.directed_color_projection import _require_identity_consistency

    model = retained["models"]["nlc"].rebind(
        arithmetic="double", reduction_engine="fast"
    )
    bank = retained["prepared"]["nlc"].bare.color_bank
    mutations = {
        "color_accuracy": "lc",
        "virtual_color_degrees": (123,),
        "born_color_degrees": (123,),
        "initial_average": model.bare.initial_average + 1,
        "born_color_tensors": (object(), *model.bare.born_color_tensors[1:]),
        "born_term_orders": (),
    }
    for name, changed in mutations.items():
        old = getattr(model.bare, name)
        setattr(model.bare, name, changed)
        try:
            with pytest.raises(ValueError):
                bank.require_owner(model)
        finally:
            setattr(model.bare, name, old)
    bank.require_owner(model)
    identity = bank.projection.history_families[0]
    families = list(bank.projection.families)
    families[identity] = replace(families[identity], virtual_degrees=(123,))
    with pytest.raises(ValueError, match="ordinary formal colour owner"):
        _require_identity_consistency(
            model, replace(bank.projection, families=tuple(families)), bank.histories
        )
    current = model.currents[0]
    weights = current.color_weights
    current.color_weights = tuple(tuple(value + 1 for value in row) for row in weights)
    try:
        with pytest.raises(ValueError, match="ordinary effective colour weights"):
            _require_identity_consistency(model, bank.projection, bank.histories)
    finally:
        current.color_weights = weights
    _require_identity_consistency(model, bank.projection, bank.histories)
