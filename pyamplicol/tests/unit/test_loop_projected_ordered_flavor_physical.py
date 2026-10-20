# SPDX-License-Identifier: 0BSD
"""Mapped projected outputs against independently prepared member currents."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from .test_loop_generation_spin_flavor_runtime import _vectors
from .test_loop_ordered_flavor_physical import NAMES, _no_new_algebra, _points
from .test_loop_ordered_flavor_transport import _admit, _pair
from .test_loop_ordered_projection_physical import _close_component
from .test_loop_physical_correlations import _conjugate, _decimals

RETAINED_ADMISSION = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_ADMISSION is None:
        pytest.skip("actual retained independent complete-family owners required")
    return RETAINED_ADMISSION


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    (
        ("double", None, "3e-9", 20),
        ("double-double", None, "3e-24", 70),
        ("arbitrary", 70, "3e-54", 170),
    ),
)
def test_mapped_projected_members_preserve_complete_terms_and_sources(
    retained, monkeypatch, accuracy, arithmetic, precision, tolerance, bit
):
    from pyamplicol.loops.spin_sources import NativeSpinSourcePlan

    relation = _admit(retained, accuracy)
    originals, ordinary, classes, _ = _pair(retained, accuracy)
    _no_new_algebra(monkeypatch)
    options = dict(
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="interpreted",
        projector_engine="interpreted",
    )
    programs = tuple(source.rebind(**options) for source in originals)
    source_plans = tuple(
        NativeSpinSourcePlan(source.virtual.bare, ((3,), (4,), (3, 4)))
        for source in programs
    )
    base = _points()
    points = [base[1], base[0], base[1]]
    ids, scales = [2**64 - 1, 7, 42], ["900000", "1000000", "900000"]
    vector_a, vector_b = _vectors()
    with localcontext() as context:
        context.prec = 120
        tolerance = Decimal(tolerance)
        alpha = str(Decimal("0.118") * (1 + Decimal(2) ** -bit))
        for source in programs:
            source.update_parameters(
                {"MB": "0", "ymb": "0", "aS": alpha, "MT": "173", "MU_R": "1000"}
            )
        snapshots = tuple(source.source_bare.state.snapshot() for source in programs)
        for left, right in zip(
            (ordinary[0], *classes[0]), (ordinary[1], *classes[1]), strict=True
        ):
            key = () if left.bare.spin_layout is None else left.bare.spin_layout.legs
            vectors = {
                leg: [vector_a, vector_b, vector_a] if leg == 3 else vector_b
                for leg in key
            }
            sources = tuple(plan.prepare_decimal(vectors, ids) for plan in source_plans)
            mapped = relation.bind(programs[0], directed=left)
            actual = mapped.evaluate_directed_decimal(
                points, scales, point_ids=ids, spin_sources=sources[0]
            )
            expected = right.bind(programs[1]).evaluate_decimal(
                points, scales, point_ids=ids, spin_sources=sources[1]
            )
            values = {name: _decimals(getattr(actual, name)) for name in NAMES}
            for name in NAMES:
                _close_component(
                    values[name], _decimals(getattr(expected, name)), tolerance
                )
            assert actual.orders == expected.orders == relation.orders
            assert actual.histories == expected.histories
            assert actual.channels == expected.channels
            assert actual.statuses == expected.statuses
            assert all(
                len(statuses) == len(actual.channels)
                and all(
                    lane == identity and status == "success"
                    for lane, status in statuses
                )
                for identity, statuses in zip(ids, actual.statuses, strict=True)
            )
            _close_component(
                values["virtual"], values["first"] + values["second"], tolerance
            )
            _close_component(
                values["virtual"][:, 2],
                _conjugate(values["virtual"][:, 3]),
                tolerance,
            )
            if accuracy == "nlc":
                assert any(
                    abs(value) > Decimal("1e-6")
                    for value in values["virtual"][:, 2, :, :, 1].flat
                )
            assert np.array_equal(actual.virtual[0], actual.virtual[2])
            empty = mapped.evaluate_directed_decimal(
                np.empty((0, 4, 4), dtype=str),
                [],
                point_ids=[],
                spin_sources=source_plans[0].prepare_decimal(
                    {leg: vector_a for leg in key}, []
                ),
            )
            assert empty.virtual.shape == (0, 4, len(relation.orders.virtual), 3, 2)
            assert empty.statuses == ()
        assert all(
            source.source_bare.state.snapshot() is snapshot
            for source, snapshot in zip(programs, snapshots, strict=True)
        )
    print("PROJECTED_ORDERED_MAPPED_MEMBER_PASS", accuracy, arithmetic, flush=True)
