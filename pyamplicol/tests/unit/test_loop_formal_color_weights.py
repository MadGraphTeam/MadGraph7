# SPDX-License-Identifier: 0BSD
"""Formal producer ownership and exact dual-weight certificate controls."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.loops.color_components import prepare_born_color_components
from pyamplicol.loops.dual_weights import (
    prepare_certified_dual_born_weights,
    prepare_dual_born_weights,
    select_dual_born_weights,
)
from pyamplicol.loops.physical_color import project_physical_color
from pyamplicol.loops.physical_process import NativeBareProcessProgram

from .test_loop_color_orbits import _expanded_metric
from .test_loop_color_orbits import algebra as algebra

pytestmark = pytest.mark.loop_fast


@pytest.fixture
def formal_owner(algebra):
    s, _, (x, y, z), labels = algebra
    q2, e2 = (("QCD", 2),), (("QED", 2),)
    q4, qe = (("QCD", 4),), (("QCD", 2), ("QED", 2))
    bare = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
    bare.born_color_tensors = (x, y, z, x, y)
    bare.born_colors = bare.born_color_tensors
    bare.loop_color_tensors = (3 * x, 3 * y, z, s.E("2i") * x, s.E("2i") * y, 3 * x)
    bare.born_term_orders = (q2, q2, q2, e2, e2)
    bare.born_color_components = prepare_born_color_components(bare.born_colors)
    bare.loop_graphs = tuple(SimpleNamespace(order=o) for o in (q4, qe, q4))
    bare.loop_finals = ((0, 1, 2), (3, 4), (5,))
    bare.loop_reduction_powers = (((0,),) * 3, ((0,),) * 2, ((0,),))
    bare.loop_term_offsets = (0, 3, 5)
    bare.initial_average = s.E("36")
    bare.order_layout = SimpleNamespace(
        born=tuple(
            SimpleNamespace(pairs=pairs)
            for pairs in (((q2, q2),), ((q2, e2), (e2, q2)), ((e2, e2),))
        ),
        virtual=tuple(
            SimpleNamespace(
                pairs=tuple(SimpleNamespace(born=a, loop=b) for a, b in pairs)
            )
            for pairs in (((q2, q4),), ((q2, qe), (e2, q4)), ((e2, qe),))
        ),
    )
    # The real setter exercises invalidation and preflight. No graph program is
    # needed for this cold control; exact saved weights are compiled below.
    bare._compile_born_projection = lambda: None
    bare._compile_dual_born_weights = lambda: None
    bare.families = ()
    bare.formal_color_metrics = None
    return bare, s.S("formal_weight_test::Nc", is_real=True), labels


@pytest.mark.parametrize("accuracy", ["full", "lc", "nlc"])
@pytest.mark.parametrize("resolved", [False, True])
def test_actual_order_masked_rectangles_match_exact_saved_weights(
    formal_owner, accuracy, resolved
):
    bare, nc, labels = formal_owner
    # A counterterm row fixes a higher structural degree in only one sector;
    # no source/helicity or numerical cancellation may choose a replacement P.
    q2, qe = bare.born_term_orders[0], bare.loop_graphs[1].order
    projection = project_physical_color(
        bare, nc, ((nc**5,),), accuracy, additional_pairs=((q2, qe),)
    )
    weights = prepare_certified_dual_born_weights(bare, labels, resolved=resolved)
    owner = weights.color_certificate
    owner.require_owner(bare)
    assert owner.formal_metrics is projection.formal_metrics
    assert owner.formal_metrics.virtual_degrees == (4, 5, 4)
    rectangle = owner.rectangle
    assert len(rectangle.born.orbits) == 2
    assert len(rectangle.loop.orbits) == 2
    assert any(len(alias.columns) > 1 for alias in rectangle.loop.aliases)
    metrics = (
        bare.born_color_components.mask_bra(projection.formal_metrics.loop)
        if resolved
        else (projection.formal_metrics.loop,)
    )
    for direction, sector, bank in owner.banks:
        expected = tuple(
            tuple(
                value
                if (bare.born_term_orders[i], projection.formal_metrics.loop_orders[j])
                in projection.formal_metrics.virtual_pairs[sector]
                else 0
                for j, value in enumerate(row)
            )
            for i, row in enumerate(metrics[direction])
        )
        assert _expanded_metric(rectangle, bank) == expected
    # Exact evaluator instructions still use the original direct schedule,
    # after the declared Nc substitution and external averaging.
    numerical = (
        bare.born_color_components.mask_bra(bare.loop_metric)
        if resolved
        else (bare.loop_metric,)
    )
    original = prepare_dual_born_weights(bare, numerical)
    assert weights.state == original.state
    assert weights.columns == original.columns
    assert weights.directions == original.directions
    for indices, sectors in (((0, 2), (0, 1)), ((1,), (1, 2))):
        assert weights.family_columns(
            bare, indices, sectors
        ) == original.family_columns(bare, indices, sectors)


def test_owner_binding_and_unproved_replacement_invalidate_certificate(formal_owner):
    bare, nc, labels = formal_owner
    project_physical_color(bare, nc, (), "nlc")
    weights = prepare_certified_dual_born_weights(bare, labels, resolved=True)
    certificate = weights.color_certificate
    formal = certificate.formal_metrics
    # Prepared views retain the immutable tensor definitions and same record;
    # binding numerical workspaces does not establish a new projection.
    view = SimpleNamespace(**vars(bare))
    certificate.require_owner(view)
    for name, changed in (
        ("_color_metrics_revision", bare._color_metrics_revision + 1),
        ("virtual_color_degrees", (99,) * 3),
        ("initial_average", bare.initial_average + 1),
        ("born_term_orders", tuple(reversed(bare.born_term_orders))),
        ("loop_color_tensors", tuple(reversed(bare.loop_color_tensors))),
        ("born_color_components", replace(bare.born_color_components)),
    ):
        with pytest.raises(ValueError):
            certificate.require_owner(SimpleNamespace(**(vars(bare) | {name: changed})))
    bad_loop = tuple(tuple(value + nc - 3 for value in row) for row in formal.loop)
    # Numeric equality at Nc=3 alone cannot transfer formal proof ownership.
    forged = replace(formal, loop=bad_loop)
    with pytest.raises(ValueError, match="projection owner"):
        forged.require_owner(bare)
    forged = replace(forged, revision=formal.revision + 1)
    assert forged.bound() == formal.bound()
    # The supported numerical setter cannot admit any supplied formal proof,
    # even when all its Nc=3 values and next-revision metadata agree.
    with pytest.raises(TypeError, match="formal_metrics"):
        bare.set_color_metrics(*forged.bound(), formal_metrics=forged)
    assert bare.formal_color_metrics is formal
    # Even an exactly equal unproved numerical replacement drops the record.
    bare.set_color_metrics(bare.born_metric, bare.loop_metric)
    assert bare.formal_color_metrics is None
    with pytest.raises(ValueError, match="projection owner"):
        certificate.require_owner(bare)
    with pytest.raises(ValueError, match="retained formal"):
        prepare_certified_dual_born_weights(bare, labels)


def test_numerical_replacement_preflight_is_atomic(formal_owner):
    bare, nc, _ = formal_owner
    projection = project_physical_color(bare, nc, (), "full")
    formal = projection.formal_metrics
    for born, loop in (
        (bare.born_metric[:-1], bare.loop_metric),
        (bare.born_metric, tuple(row[:-1] for row in bare.loop_metric)),
    ):
        with pytest.raises(ValueError):
            bare.set_color_metrics(born, loop)
        assert bare.formal_color_metrics is formal
        formal.require_owner(bare)


def test_actual_action_selection_serializes_exact_ordered_domains(formal_owner):
    bare, nc, labels = formal_owner
    project_physical_color(bare, nc, (), "nlc")
    selected = select_dual_born_weights(bare, "auto")
    assert selected.requested_contraction == "auto"
    assert selected.color_certificate is not None
    descriptor, constants = selected.runtime_rectangle(bare)
    assert constants and descriptor["degree"] == 2
    assert sum(len(a["columns"]) for a in descriptor["born_aliases"]) == 5
    assert sum(len(a["columns"]) for a in descriptor["loop_aliases"]) == 6
    assert descriptor["banks"] == [(0, 0), (0, 1), (0, 2)]
    assert descriptor["kernels"] and descriptor["residuals"]
    assert selected.state == prepare_dual_born_weights(bare).state
    assert select_dual_born_weights(bare, "auto", labels=()).color_certificate is None
    with pytest.raises(ValueError, match="no nonzero regular rectangle"):
        select_dual_born_weights(bare, "symmetric-group-fft", labels=())
    bare._color_metrics_revision += 1
    with pytest.raises(ValueError, match="projection owner"):
        select_dual_born_weights(bare, "auto", labels=labels)
    bare.formal_color_metrics = None
    assert "no retained formal" in select_dual_born_weights(bare, "auto").direct_reason
    with pytest.raises(ValueError, match="no retained formal"):
        select_dual_born_weights(bare, "symmetric-group-fft")


def test_lc_auto_keeps_requested_policy_and_direct_public_selection(formal_owner):
    bare, nc, _ = formal_owner
    project_physical_color(bare, nc, (), "lc")
    selected = select_dual_born_weights(bare, "auto")
    assert selected.requested_contraction == "auto"
    assert selected.color_certificate is None
    assert "LC automatic" in selected.direct_reason
