# SPDX-License-Identifier: 0BSD
"""Real-mass EW colour identities over one independently validated source owner.

The retained fixture is supplied only after the complete ordinary MadLoop/
Recola gate. These tests neither generate graphs nor relax directed admission.
They do not certify EW vector sources, formal projection, complex masses or
non-Hermitian ordered histories.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _conjugate, _decimals

RETAINED_EW = None
COMPONENTS = ("born", "virtual", "bare", "uv", "r2", "uv_local", "uv_external_fields")
TERMS = ("first", "second")


def prepare_ew_colour_control(model):
    """Derive the fixture's physical colour factors with dependency operations."""
    from symbolica import E

    from pyamplicol.color.connections import ColorConnection, EmitGluon
    from pyamplicol.loops.color_bilinears import PhysicalColorSpace
    from pyamplicol.loops.color_connections import apply_native_color_connection
    from pyamplicol.loops.physical_correlations import _physical_color_ports

    legs, born, _ = _physical_color_ports(model.bare)
    charged = tuple(leg for leg in legs if leg.representation != 1)
    # These are the selected physical fixture's port expectations, never an
    # admission rule or a limit on how many colour legs production may carry.
    assert len(charged) == 2
    assert {leg.representation for leg in charged} == {3, -3}
    assert all(not model.bare.state_layout.legs[leg.label].incoming for leg in charged)
    identity = ColorConnection(legs)
    emissions = tuple(
        ColorConnection(legs, (EmitGluon(leg.label, -1),)) for leg in charged
    )
    histories = ((identity, identity), *((a, b) for a in emissions for b in emissions))
    space = PhysicalColorSpace()
    tensor, norm = next(
        (value, norm) for value in born if (norm := space.overlap(value, value)) != 0
    )
    images = tuple(
        apply_native_color_connection(tensor, history) for history in emissions
    )
    factors = (E("1"), *(space.overlap(a, b) / norm for a in images for b in images))
    factors = tuple(value.together() for value in factors)
    assert factors[1] == factors[4] == E("4/3")
    assert factors[2] == factors[3] == -factors[1]
    assert factors[1] + factors[2] == factors[3] + factors[4] == 0
    return histories, factors


@pytest.fixture(scope="module")
def retained():
    if RETAINED_EW is None:
        pytest.skip("independently validated retained real-mass EW owner required")
    return RETAINED_EW


def _factor(value):
    rational = Fraction(str(value))
    return Decimal(rational.numerator) / Decimal(rational.denominator)


def _check_complete(result, ordinary, factors, tolerance):
    values = {name: _decimals(getattr(result, name)) for name in (*COMPONENTS, *TERMS)}
    for name in COMPONENTS:
        _close_component(
            values[name][:, 0],
            _decimals(getattr(ordinary, name)),
            tolerance,
            scale=values[name],
        )
    for value in values.values():
        # Compare each integrated term independently: equality of their sum
        # alone would not establish the directed assembly contract.
        for index, factor in enumerate(factors):
            _close_component(
                value[:, index], _factor(factor) * value[:, 0], tolerance, scale=value
            )
        _close_component(
            value[:, 1] + value[:, 2], 0 * value[:, 0], tolerance, scale=value
        )
        _close_component(
            value[:, 3] + value[:, 4], 0 * value[:, 0], tolerance, scale=value
        )
    for name in COMPONENTS:
        _close_component(
            values[name][:, 2],
            _conjugate(values[name][:, 3]),
            tolerance,
            scale=values[name],
        )
    _close_component(
        values["second"][:, 2],
        _conjugate(values["first"][:, 3]),
        tolerance,
        scale=values["first"],
    )
    _close_component(values["virtual"], values["first"] + values["second"], tolerance)
    _close_component(
        values["virtual"], values["bare"] + values["uv"] + values["r2"], tolerance
    )
    _close_component(
        values["uv"], values["uv_local"] + values["uv_external_fields"], tolerance
    )
    return values


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "3e-9", 20),
        ("double-double", None, "3e-24", 70),
        ("arbitrary", 70, "3e-54", 170),
    ),
)
def test_ew_complete_components_identity_charges_and_both_terms(
    retained,
    monkeypatch,
    arithmetic,
    precision,
    tolerance,
    bit,
):
    from pyamplicol.loops import prepared_correlations

    original = retained["source"].virtual
    original_snapshot = original.bare.model_state.snapshot()
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="interpreted",
        projector_engine="interpreted",
    )
    model = original.rebind(**options)
    prepared = retained["directed"]
    prepared.require_owner(model)
    assert model.bare.order_layout.correction_names == ("QED",)
    assert model.r2_strategy == "model" and model.color_accuracy == "full"
    assert model.color_projection is None
    assert prepared.bare.histories == retained["histories"]
    assert prepared.bare.spin_layout is None

    def forbidden(*args, **kwargs):
        raise AssertionError("EW arithmetic rebind repeated directed preparation")

    for name in (
        "prepare_directed_complete",
        "prepare_directed_bare",
        "prepare_directed_model",
        "_saved",
    ):
        monkeypatch.setattr(prepared_correlations, name, forbidden)
    directed = prepared.bind(model)
    points = np.asarray(retained["points"], dtype=str)
    ids = [2**64 - 1, 7, 42]
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        point = points[:1]
        scales = ["1000000"]
        before = directed.evaluate_decimal(point, scales, point_ids=ids[:1])
        ordinary = model.evaluate_ordered_decimal(point, scales)
        baseline = _check_complete(before, ordinary, retained["factors"], tolerance)
        assert all(
            lane == ids[0] and status == "success"
            for row in before.statuses
            for lane, status in row
        )

        # Preserve the original exact input and change only the rebound model.
        gf = Decimal(retained["parameters"]["Gf"])
        shifted = str(gf * (1 + Decimal(2) ** -bit))
        model.bare.model_state.update({"Gf": shifted})
        updated = model.bare.model_state.snapshot()
        assert updated.overrides["Gf"] == (shifted, "0")
        after = directed.evaluate_decimal(point, scales, point_ids=ids[:1])
        changed_ordinary = model.evaluate_ordered_decimal(point, scales)
        changed = _check_complete(
            after, changed_ordinary, retained["factors"], tolerance
        )
        assert np.any(changed["born"] != baseline["born"])

        # Scale and source-state updates are distinct inputs. This first scale
        # is the one independently checked in the ordinary provider gate.
        selected = points[[1, 0, 1]]
        selected_scales = ["2250000", "1000000", "2250000"]
        varied = directed.evaluate_decimal(selected, selected_scales, point_ids=ids)
        expected = model.evaluate_ordered_decimal(selected, selected_scales)
        values = _check_complete(varied, expected, retained["factors"], tolerance)
        assert np.array_equal(values["virtual"][0], values["virtual"][2])
        assert all(
            lane == label and status == "success"
            for label, statuses in zip(ids, varied.statuses, strict=True)
            for lane, status in statuses
        )
        # The middle row independently repeats the updated single-row input.
        for name in (*COMPONENTS, *TERMS):
            _close_component(values[name][1:2], changed[name], tolerance)
        if arithmetic == "double":
            empty = directed.evaluate_decimal(
                np.empty((0, 4, 4), dtype=str), [], point_ids=[]
            )
            assert empty.virtual.shape == (
                0,
                len(retained["histories"]),
                len(model.bare.order_layout.virtual),
                3,
                2,
            )
            assert empty.statuses == ()
            again = directed.evaluate_decimal(point, scales, point_ids=[91])
            for name in (*COMPONENTS, *TERMS):
                _close_component(
                    _decimals(getattr(again, name)), changed[name], tolerance
                )
        assert model.bare.model_state.snapshot() is updated
        assert original.bare.model_state.snapshot() is original_snapshot
        print(
            "EW_DIRECTED_COMPLETE_COLOUR_CONTROL",
            arithmetic,
            "histories",
            len(retained["histories"]),
            "components",
            len(COMPONENTS),
            "both_terms",
            True,
            flush=True,
        )
