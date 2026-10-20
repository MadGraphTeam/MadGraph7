# SPDX-License-Identifier: 0BSD
"""Physical witnesses for a named owner's full-bank LC/NLC reprojection.

The independent projection oracle is test_loop_directed_color_reuse. These
retained controls compare ordinary identity amplitudes and physical source sums;
reused directed executors are not independent colour calculations.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from .test_loop_massive_vector_sources import _scale, _wavefunctions
from .test_loop_ordered_flavor_physical import NAMES, _no_new_algebra, _points
from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _conjugate, _decimals

RETAINED_REUSE = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_REUSE is None:
        pytest.skip("actual same-process full/LC/NLC owners and reused banks required")
    return RETAINED_REUSE


def _values(result, ids):
    assert len(result.statuses) == len(ids)
    assert all(
        len(row) == len(result.channels)
        and all(lane == identity and status == "success" for lane, status in row)
        for identity, row in zip(ids, result.statuses, strict=True)
    )
    return {name: _decimals(getattr(result, name)) for name in NAMES}


def _identities(values, tolerance):
    for name in NAMES[:-2]:
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
        values["uv"],
        values["uv_local"] + values["uv_external_fields"],
        tolerance,
    )


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "3e-9", 20),
        ("double-double", None, "3e-24", 70),
        ("arbitrary", 70, "3e-54", 170),
    ),
)
def test_reused_actual_bank_keeps_complete_projected_sources_and_directions(
    retained, monkeypatch, accuracy, arithmetic, precision, tolerance, bit
):
    from pyamplicol.loops import directed_color_projection
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    process_name = retained.get("process_name", "ddgg")
    key = (process_name, accuracy)
    prepared = retained["prepared"][key]
    bank = prepared.bare.color_bank
    original = retained["models"][key]
    bank.require_owner(original)
    full = retained["full"].bare.color_bank
    assert bank is not full and bank.plan is original.prepared
    assert bank.projection.families is full.projection.families
    assert bank.projection.history_families is full.projection.history_families
    assert bank.histories == full.histories
    assert bank.projection.accuracy == accuracy
    assert [len(f.histories) for f in bank.projection.families] == [1, 4, 28]
    assert [f.order for f in bank.projection.families] == [0, 1, 2]
    formal = original.bare.formal_color_metrics
    assert bank.projection.families[0].born_degrees == formal.born_degrees
    assert bank.projection.families[0].virtual_degrees == formal.virtual_degrees
    for source in retained["classes"][key].values():
        source.require_owner(original)
        assert source.bare.color_bank is bank

    _no_new_algebra(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("physical source/rebind repeated exact reprojection")

    monkeypatch.setattr(
        directed_color_projection, "reproject_directed_color_bank", forbidden
    )
    monkeypatch.setattr(
        directed_color_projection, "reproject_directed_color_columns", forbidden
    )
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="interpreted",
        projector_engine="interpreted",
    )
    model = original.rebind(**options)
    sources = NativeSpinSourcePlan(model.bare, ((3,), (4,), (3, 4)))
    directed = prepared.bind(model)
    classes = {
        joint: source.bind(model)
        for joint, source in retained["classes"][key].items()
        if joint
    }
    point, scales = _points()[0], ["1000000"]
    ids = [2**64 - 1]
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        tiny = Decimal(2) ** -bit
        model.bare.model_state.update(
            {
                "MB": "0",
                "ymb": "0",
                "aS": str(Decimal("0.118") * (1 + tiny)),
                "MT": "173",
                "MU_R": "1000",
            }
        )
        snapshot = model.bare.model_state.snapshot()
        baseline = _values(
            directed.evaluate_decimal([point], scales, point_ids=ids), ids
        )
        _identities(baseline, tolerance)
        ordinary = model.evaluate_ordered_decimal([point], scales)
        for name in NAMES[:-2]:
            _close_component(
                baseline[name][:, 0],
                _decimals(getattr(ordinary, name)),
                tolerance,
                scale=baseline[name],
            )

        a = _wavefunctions(model.bare.backend, point, 3, [-1, 1], False, options)
        b = _wavefunctions(model.bare.backend, point, 4, [-1, 1], False, options)
        source_ids = [2**64 - 1, 7, 42, 0, 2**63]
        first_sources = [
            *a,
            _scale(a[0], Decimal(2), Decimal(3)),
            [("0", "0")] * 4,
            _scale(a[0], 1 + tiny, Decimal(0)),
        ]
        singleton = _values(
            classes[3,].evaluate_decimal(
                [point] * len(source_ids),
                scales * len(source_ids),
                point_ids=source_ids,
                spin_sources=sources.prepare_decimal({3: first_sources}, source_ids),
            ),
            source_ids,
        )
        other = _values(
            classes[4,].evaluate_decimal(
                [point] * 2,
                scales * 2,
                point_ids=source_ids[:2],
                spin_sources=sources.prepare_decimal({4: b}, source_ids[:2]),
            ),
            source_ids[:2],
        )
        joint = _values(
            classes[3, 4].evaluate_decimal(
                [point] * 4,
                scales * 4,
                point_ids=source_ids[:4],
                spin_sources=sources.prepare_decimal(
                    {3: [v for v in a for _ in b], 4: [v for _ in a for v in b]},
                    source_ids[:4],
                ),
            ),
            source_ids[:4],
        )
        for values in (singleton, other, joint):
            _identities(values, tolerance)
        for name in NAMES:
            reference = baseline[name][0]
            values = singleton[name]
            _close_component(values[:2].sum(axis=0), reference, tolerance, scale=values)
            _close_component(values[2], 13 * values[0], tolerance, scale=values)
            _close_component(values[3], 0 * values[0], tolerance, scale=values)
            _close_component(
                other[name].sum(axis=0), reference, tolerance, scale=other[name]
            )
            _close_component(
                joint[name].sum(axis=0), reference, tolerance, scale=joint[name]
            )
        # Existing exact source ingress must survive reuse. This is an internal
        # carrier check, not an independent high-precision process reference.
        born = singleton["born"]
        shift = born[4, 0, 0, 0] - born[0, 0, 0, 0]
        expected_shift = born[0, 0, 0, 0] * (2 * tiny + tiny * tiny)
        assert expected_shift != 0
        assert abs(shift - expected_shift) <= abs(expected_shift) * Decimal("2e-4")
        assert model.bare.model_state.snapshot() is snapshot
        # A projected operator may lose its imaginary part; report it without
        # treating survival as a physical requirement of LC or NLC.
        print(
            "SAME_PROCESS_PROJECTED_ORDERED_PHYSICAL",
            process_name,
            accuracy,
            arithmetic,
            "max_imaginary",
            max(abs(x) for x in baseline["virtual"][:, 2, :, :, 1].flat),
            flush=True,
        )
        if arithmetic == "double":
            empty = classes[3, 4].evaluate_decimal(
                np.empty((0, 4, 4), dtype=str),
                [],
                point_ids=[],
                spin_sources=sources.prepare_decimal({3: a[0], 4: b[0]}, []),
            )
            assert empty.virtual.shape == (
                0,
                4,
                len(model.bare.order_layout.virtual),
                3,
                2,
            )
            assert empty.statuses == ()
            tail = _values(
                classes[3,].evaluate_decimal(
                    [point],
                    scales,
                    point_ids=[42],
                    spin_sources=sources.prepare_decimal({3: [a[1]]}, [42]),
                ),
                [42],
            )
            for name in NAMES:
                _close_component(
                    tail[name][0], singleton[name][1], tolerance, scale=singleton[name]
                )
        retained["records"][accuracy, arithmetic] = {
            "ordinary": baseline,
            "source3": singleton,
            "source4": other,
            "joint": joint,
        }
