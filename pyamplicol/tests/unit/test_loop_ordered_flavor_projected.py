# SPDX-License-Identifier: 0BSD
"""Independent numerical currents remain distinct from exact flavour admission.

The two owners may share proved colour rectangles. Their numerical comparison
does not constitute an independent colour-projection calculation.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from .test_loop_generation_spin_flavor_runtime import _vectors
from .test_loop_ordered_flavor_physical import NAMES, _no_new_algebra, _points
from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _conjugate, _decimals

RETAINED_PROJECTED = None


@pytest.fixture(scope="module")
def pair():
    if RETAINED_PROJECTED is None:
        pytest.skip("independent projected complete-family owners required")
    return RETAINED_PROJECTED


def _actual_family_pair(pair, accuracy):
    keys = (("ddgg", accuracy), ("ssgg", accuracy))
    if any(key not in pair["prepared"] for key in keys):
        pytest.skip("both actual projected complete-family banks are required")
    models = tuple(pair["models"][key] for key in keys)
    assert models[0].bare.process is not models[1].bare.process
    assert models[0].bare.model_state.plan is not models[1].bare.model_state.plan
    plans = tuple(pair["prepared"][key] for key in keys)
    banks = tuple(plan.bare.color_bank for plan in plans)
    for key, model, plan, bank in zip(keys, models, plans, banks, strict=True):
        plan.require_owner(model)
        bank.require_owner(model)
        assert bank.projection.accuracy == accuracy
        assert model.source_color is pair["declaration"]
        assert dict(model.prepared.metadata)["source_color"] is pair["declaration"]
        for joint, source in pair["classes"][key].items():
            if joint:
                source.require_owner(model)
                assert source.bare.color_bank is bank
    left, right = (bank.projection for bank in banks)
    full = pair["prepared"]["ddgg", "full"].bare.color_bank.projection
    assert left.history_families == right.history_families == full.history_families
    assert plans[0].bare.histories == plans[1].bare.histories
    assert len(left.families) == len(right.families)
    for a, b, unprojected in zip(
        left.families, right.families, full.families, strict=True
    ):
        # Raw Born/loop columns may be enumerated differently. These are actual
        # complete operator families and common sector degrees, not matrix
        # equality or a substitute for the source-rule/continuation proof.
        assert a.histories == b.histories == unprojected.histories
        assert a.order == b.order == unprojected.order
        assert a.output_legs == b.output_legs == unprojected.output_legs
        assert a.born_degrees == b.born_degrees == unprojected.born_degrees
        assert a.virtual_degrees == b.virtual_degrees == unprojected.virtual_degrees
    assert [len(f.histories) for f in left.families] == [1, 4, 28]
    return keys, models, plans


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "3e-9", 20),
        ("double-double", None, "3e-24", 70),
        ("arbitrary", 70, "3e-54", 170),
    ),
)
def test_independent_projected_ordered_members_keep_components_sources_and_directions(
    pair,
    monkeypatch,
    accuracy,
    arithmetic,
    precision,
    tolerance,
    bit,
):
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    keys, original_models, plans = _actual_family_pair(pair, accuracy)
    _no_new_algebra(monkeypatch)
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="interpreted",
        projector_engine="interpreted",
    )
    models = tuple(model.rebind(**options) for model in original_models)
    source_plans = tuple(
        NativeSpinSourcePlan(model.bare, ((3,), (4,), (3, 4))) for model in models
    )
    points = _points()
    points = [points[1], points[0], points[1]]
    ids, scales = [2**64 - 1, 7, 42], ["900000", "1000000", "900000"]
    a, b = _vectors()
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        for model in models:
            model.bare.model_state.update(
                {"MB": "0", "ymb": "0", "aS": alpha, "MT": "173", "MU_R": "1000"}
            )
        snapshots = tuple(model.bare.model_state.snapshot() for model in models)
        for joint in ((), (3,), (4,), (3, 4)):
            vectors = {leg: ([a, b, a] if leg == 3 else b) for leg in joint}
            actual = []
            for key, model, ordinary, sources in zip(
                keys, models, plans, source_plans, strict=True
            ):
                prepared = ordinary if not joint else pair["classes"][key][joint]
                result = prepared.bind(model).evaluate_decimal(
                    points,
                    scales,
                    point_ids=ids,
                    spin_sources=sources.prepare_decimal(vectors, ids),
                )
                assert all(
                    len(statuses) == len(result.channels)
                    and all(
                        lane == identity and status == "success"
                        for lane, status in statuses
                    )
                    for identity, statuses in zip(ids, result.statuses, strict=True)
                )
                values = {name: _decimals(getattr(result, name)) for name in NAMES}
                for name in NAMES[:-2]:
                    _close_component(
                        values[name][:, 2],
                        _conjugate(values[name][:, 3]),
                        tolerance,
                        scale=values[name],
                    )
                _close_component(
                    values["virtual"], values["first"] + values["second"], tolerance
                )
                _close_component(
                    values["virtual"],
                    values["bare"] + values["uv"] + values["r2"],
                    tolerance,
                )
                _close_component(
                    values["second"][:, 2],
                    _conjugate(values["first"][:, 3]),
                    tolerance,
                    scale=values["first"],
                )
                assert np.array_equal(values["virtual"][0], values["virtual"][2])
                actual.append(values)
            for name in NAMES:
                _close_component(actual[0][name], actual[1][name], tolerance)
            # LC may remove this imaginary part. Reporting it is not an
            # assertion that a projected operator must remain non-Hermitian.
            print(
                "PROJECTED_INDEPENDENT_NONIDENTITY",
                accuracy,
                arithmetic,
                joint,
                "max_imaginary",
                max(abs(x) for x in actual[0]["virtual"][:, 2, :, :, 1].flat),
                flush=True,
            )
        assert all(
            model.bare.model_state.snapshot() is snapshot
            for model, snapshot in zip(models, snapshots, strict=True)
        )
        if arithmetic == "double":
            for key, model, sources in zip(keys, models, source_plans, strict=True):
                empty = (
                    pair["classes"][key][3, 4]
                    .bind(model)
                    .evaluate_decimal(
                        np.empty((0, 4, 4), dtype=str),
                        [],
                        point_ids=[],
                        spin_sources=sources.prepare_decimal({3: a, 4: b}, []),
                    )
                )
                assert empty.statuses == ()
                assert empty.virtual.shape == (
                    0,
                    4,
                    len(model.bare.order_layout.virtual),
                    3,
                    2,
                )
