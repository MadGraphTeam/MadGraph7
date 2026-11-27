# SPDX-License-Identifier: 0BSD
"""Native SU(N_c) overlaps of trace products by cycle counting (WP3.1 I10).

``loop_v2::generate::cycles`` counts a product of adjoint traces (every
label twice) as the subset sum over the labels taking the U(1) term of
their Fierz identity, each term N_c to its index loops. The trace algebra of
``color/contraction_trace.py`` is its oracle: equal on random products, and
``colour.contract_colour`` equals ``contract_connected_tensor_nc_terms``
on the Born metric and IR dipoles of a slice.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from pyamplicol.color.connections import contract_connected_tensor_nc_terms
from pyamplicol.color.contraction_trace import _simplify_trace_terms_nc_power
from pyamplicol.loops.v2.colour import contract_colour


def _products(count: int, seed: int) -> list[list[tuple[int, ...]]]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        labels = list(range(1, rng.randint(0, 8) + 1)) * 2
        rng.shuffle(labels)
        cuts = sorted(rng.randint(0, len(labels)) for _ in range(rng.randint(0, 4)))
        traces, previous = [], 0
        for cut in [*cuts, len(labels)]:
            traces.append(tuple(labels[previous:cut]))
            previous = cut
        out.append(traces)
    return out


@pytest.mark.loop_fast
def test_cycle_counting_equals_the_trace_algebra(loop_native):
    products = _products(3000, 20261023)
    counted = loop_native.trace_overlaps([[list(t) for t in p] for p in products])
    for traces, laurent in zip(products, counted, strict=True):
        expected = _simplify_trace_terms_nc_power(((Fraction(1), 0, tuple(traces)),))
        assert {p: Fraction(c) for p, c in laurent} == {
            p: c for p, c in expected.items() if c
        }, traces
    with pytest.raises(Exception, match="occurs once"):
        loop_native.trace_overlaps([[[1, 2], [2]]])


@pytest.mark.loop_physics
def test_contract_colour_equals_the_trace_algebra(loop_sm_compiled_model, loop_native):
    """The Born metric of u u~ > g g g (quark string and traces) cell by cell."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import load_compiled_model

    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "u u~ > g g g",
            {"QCD": 3, "QED": 0},
            {"QCD": 5, "QED": 0},
            fixed_parameters={"MB": "0"},
        ),
        loop_native,
    )
    tensors = slice_.basis.tensors
    for left in tensors:
        for right in tensors:
            assert contract_colour(left, right) == contract_connected_tensor_nc_terms(
                left, right
            )
