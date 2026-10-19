# SPDX-License-Identifier: 0BSD
"""Shared exact weights agree with the retained independent bilinear schedule."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "2e-13"),
        ("double-double", None, "2e-28"),
        ("arbitrary", 80, "1e-70"),
    ],
)
@pytest.mark.parametrize("directions", [1, 2, 3])
def test_shared_weights_keep_orders_directions_families_and_native_rows(
    arithmetic, precision, tolerance, directions
):
    native = pytest.importorskip("_loop_reduction")
    from symbolica import E

    from pyamplicol.loops.dual_weights import prepare_dual_born_weights
    from pyamplicol.loops.orders import BornLoopPair, VirtualOrderSector
    from pyamplicol.loops.physical_orders import PhysicalOrderLayout
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    bare = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
    bare.backend = native
    bare.options = dict(arithmetic=arithmetic, precision=precision)
    bare.projector_engine = "interpreted"
    bare._evaluator_states = {}
    bare.helicities, bare.born_colors = ((-1,), (1,)), (None,) * 3
    q2, e2 = (("QCD", 2),), (("QED", 2),)
    q4, qe, e4 = (("QCD", 4),), (("QCD", 2), ("QED", 2)), (("QED", 4),)
    bare.born_term_orders = (q2, e2, q2)
    bare.loop_graphs = tuple(SimpleNamespace(order=order) for order in (q4, qe, e4))
    bare.order_layout = PhysicalOrderLayout(
        (),
        tuple(
            VirtualOrderSector(order, tuple(BornLoopPair(a, b) for a, b in pairs))
            for order, pairs in (
                ((("QCD", 6),), ((q2, q4),)),
                ((("QCD", 4), ("QED", 2)), ((q2, qe), (e2, q4))),
                ((("QCD", 2), ("QED", 4)), ((q2, e4), (e2, qe))),
                ((("QED", 6),), ((e2, e4),)),
            )
        ),
        ("QCD", "QED"),
    )
    zero, linear = (0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0)
    bare.loop_term_offsets = (0, 2, 3)
    bare.loop_reduction_powers = (
        ((zero, linear), (zero,)),
        ((linear, None),),
        ((zero,), (linear,)),
    )
    metrics = tuple(
        tuple(
            tuple(
                E("0")
                if (d == 2 and b != 1) or j == 4
                else E(f"({1 + b + 3 * j + d}/7)+({b - j + 2 * d})*1i")
                for j in range(5)
            )
            for b in range(3)
        )
        for d in range(directions)
    )
    if directions == 3:
        from pyamplicol.loops.color_components import BornColorComponents

        metrics = BornColorComponents(("a", "b", "c"), ((0,), (1,), (2,))).mask_bra(
            metrics[0]
        )
    weights = prepare_dual_born_weights(bare, metrics)
    weight_evaluator = weights.bind(bare)
    # The identically zero last colour column stays available in every admitted bank.
    assert sum(term == 4 for _, _, term in weights.columns) == 2 * directions
    families = []
    for indices, sectors in (((0, 2), (0, 1, 2, 3)), ((1,), (1, 2))):
        family = SimpleNamespace(
            graph_indices=indices,
            sector_indices=sectors,
            weight_columns=weights.family_columns(bare, indices, sectors),
        )
        old = bare._compile_family_projection(
            indices, (zero, linear), sectors, metrics=metrics
        )
        new = bare._compile_weighted_family_projection(
            indices, (zero, linear), sectors, weights=weights
        )
        families.append((family, old, new))
    for size in (0, 1, 5, 3, 1):
        ids = [91, 7, 30, 150, 2][:size]
        lane_ids = [2 * point + h for point in ids for h in range(2)]
        rows = 2 * size

        def batch(width, offset, rows=rows, lane_ids=lane_ids):
            return native.NativeBatch.from_decimal(
                [
                    (
                        f"{offset + row + column}."
                        + "123456789012345678901234567890123456789",
                        str(row - 2 * column),
                    )
                    for row in range(rows)
                    for column in range(width)
                ],
                rows,
                width,
                row_ids=lane_ids,
                **bare.options,
            )

        currents = SimpleNamespace(
            batch=SimpleNamespace(size=size, helicity_count=2, point_ids=ids),
            born=batch(3, 1),
            loops=(batch(3, 5), batch(2, 13), batch(2, 29)),
        )
        shared = weight_evaluator.evaluate_batch(currents.born)
        for family, old, new in families:
            reference = old.evaluate_batch(
                bare.family_projection_arguments(currents, family)
            )
            actual = new.evaluate_batch(
                bare.family_projection_arguments(currents, family, weights=shared)
            )
            assert actual.row_ids == ids
            assert actual.columns == 2 * directions * len(family.sector_indices)
            assert all(
                ok for _, ok in actual.agreement_rows(reference, tolerance, tolerance)
            )
