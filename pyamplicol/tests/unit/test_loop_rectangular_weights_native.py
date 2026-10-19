# SPDX-License-Identifier: 0BSD
"""Native ordered rectangular operation against its independent exact state."""

import copy
import json
import os
from itertools import permutations
from types import SimpleNamespace

import pytest

from .test_loop_color_orbits import algebra as algebra
from .test_loop_formal_color_weights import formal_owner as formal_owner


@pytest.fixture(scope="module")
def native():
    if not os.environ.get("PYAMPLICOL_LOOP_FFT_NATIVE"):
        pytest.skip(
            "set PYAMPLICOL_LOOP_FFT_NATIVE for the normally staged native control"
        )
    from pyamplicol import _rusticol
    from pyamplicol._internal.versions import verify_native_module
    from pyamplicol.loops.native import load_loop_native

    assert load_loop_native() is _rusticol._loop
    verify_native_module(_rusticol)
    return _rusticol._loop


@pytest.fixture
def noncentral_owner(algebra):
    s, t, _, _ = algebra
    from pyamplicol.loops.color_components import BornColorComponents
    from pyamplicol.loops.physical_color import project_physical_color
    from pyamplicol.loops.physical_process import NativeBareProcessProgram

    labels = s.S(*(f"native_rectangle::{name}" for name in "xyz"))
    anchors = s.S(*(f"native_rectangle::{name}" for name in "abc"))
    metric = t.TensorExpression.g(t.Representation.coad(8), t.Representation.coad(8))
    seed = (
        metric(anchors[0], labels[0])
        * metric(anchors[1], labels[1])
        * metric(anchors[2], labels[2])
    )
    orbit = tuple(
        seed.rename_indices(dict(zip(labels, image, strict=True)))
        for image in permutations(labels)
    )
    loop_seed = s.E("2+3i") * orbit[0] + s.E("1-2i") * orbit[1] + 3 * orbit[2]
    loop_orbit = tuple(
        loop_seed.rename_indices(dict(zip(labels, image, strict=True)))
        for image in permutations(labels)
    )
    invariant = sum(orbit[1:], orbit[0])
    bare = NativeBareProcessProgram.__new__(NativeBareProcessProgram)
    bare.born_colors = bare.born_color_tensors = (*orbit, invariant, orbit[0])
    bare.loop_color_tensors = (*loop_orbit, invariant, 2 * invariant, loop_orbit[0])
    q2, q4 = (("QCD", 2),), (("QCD", 4),)
    bare.born_term_orders = (q2,) * len(bare.born_colors)
    bare.born_color_components = BornColorComponents(
        ("regular-orbit", "fixed"), ((0, 1, 2, 3, 4, 5, 7), (6,))
    )
    bare.loop_graphs = (SimpleNamespace(order=q4),)
    bare.loop_finals = (tuple(range(len(bare.loop_color_tensors))),)
    bare.loop_reduction_powers = (((0,),) * len(bare.loop_color_tensors),)
    bare.loop_term_offsets = (0,)
    bare.initial_average = s.E("36")
    bare.order_layout = SimpleNamespace(
        born=(SimpleNamespace(pairs=((q2, q2),)),),
        virtual=(SimpleNamespace(pairs=(SimpleNamespace(born=q2, loop=q4),)),),
    )
    bare._compile_born_projection = bare._compile_dual_born_weights = lambda: None
    bare.families = ()
    project_physical_color(bare, s.S("native_rectangle::Nc", is_real=True), (), "full")
    return bare, labels


@pytest.mark.parametrize("resolved", [False, True])
@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "3e-12"),
        ("double-double", None, "1e-29"),
        ("arbitrary", 70, "1e-60"),
    ],
)
def test_noncentral_rectangular_weights_residuals_aliases_and_native_batches(
    native, noncentral_owner, arithmetic, precision, tolerance, resolved
):
    from pyamplicol.loops.dual_weights import select_dual_born_weights

    bare, labels = noncentral_owner
    selected = select_dual_born_weights(
        bare, "symmetric-group-fft", labels=labels, resolved=resolved
    )
    rectangle = selected.color_certificate.rectangle
    assert rectangle.degree == 3
    assert any((k.values[1] - k.values[2]).together() != 0 for k in rectangle.kernels)
    assert rectangle.residual_entries
    descriptor, constants = selected.runtime_rectangle(bare)
    options = dict(arithmetic=arithmetic, precision=precision)
    actual = native.DualBornWeights(
        selected.state, json.dumps(descriptor), constants, selected.columns, **options
    )
    expected = native.CoefficientEvaluator(selected.state, **options)
    assert actual.algorithm == (
        "symmetric-group-fft" if arithmetic == "double" else "direct"
    )
    assert (actual.fallback_reason is not None) == (arithmetic != "double")
    for size in (0, 1, 8, 3, 1):
        ids = [91, 7, 30, 150, 2, 83, 900, 4][:size]
        values = [
            (
                f"{row + column}.12345678901234567890123456789",
                f"{2 * row - column}.98765432109876543210987654321",
            )
            for row in range(size)
            for column in range(len(bare.born_colors))
        ]
        batch = native.NativeBatch.from_decimal(
            values, size, len(bare.born_colors), row_ids=ids, **options
        )
        left, right = actual.evaluate_batch(batch), expected.evaluate_batch(batch)
        assert left.row_ids == right.row_ids == ids
        assert left.columns == right.columns == len(selected.columns)
        assert all(ok for _, ok in left.agreement_rows(right, tolerance, tolerance))


def test_native_rejects_missing_or_duplicate_rectangle_pairs(native, noncentral_owner):
    from pyamplicol.loops.dual_weights import select_dual_born_weights

    bare, labels = noncentral_owner
    selected = select_dual_born_weights(bare, "auto", labels=labels)
    descriptor, constants = selected.runtime_rectangle(bare)
    for duplicate in (False, True):
        changed = copy.deepcopy(descriptor)
        changed["residuals"].pop()
        if duplicate:
            changed["residuals"].append(changed["residuals"][0])
        with pytest.raises(ValueError, match="covered twice|coverage is incomplete"):
            native.DualBornWeights(
                selected.state, json.dumps(changed), constants, selected.columns
            )


def test_native_mixed_order_banks_keep_distinct_term_scatter(native, formal_owner):
    from pyamplicol.loops.dual_weights import select_dual_born_weights
    from pyamplicol.loops.physical_color import project_physical_color

    bare, nc, labels = formal_owner
    project_physical_color(bare, nc, (), "nlc")
    selected = select_dual_born_weights(bare, "auto", labels=labels)
    descriptor, constants = selected.runtime_rectangle(bare)
    assert descriptor["banks"] == [(0, 0), (0, 1), (0, 2)]
    actual = native.DualBornWeights(
        selected.state, json.dumps(descriptor), constants, selected.columns
    )
    direct = native.CoefficientEvaluator(selected.state)
    input = native.NativeBatch.from_decimal(
        [(str(i - 8), str(3 * i + 2)) for i in range(15)], 3, 5, row_ids=[71, 9, 302]
    )
    result = actual.evaluate_batch(input)
    assert result.row_ids == [71, 9, 302]
    assert all(
        ok
        for _, ok in result.agreement_rows(
            direct.evaluate_batch(input), "2e-12", "2e-12"
        )
    )
