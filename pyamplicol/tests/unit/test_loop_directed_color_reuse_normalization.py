# SPDX-License-Identifier: 0BSD
"""Exact target-local averaging of unnormalized reused colour rectangles.

These are mathematical tensor inputs, not fabricated physical model owners.
The separate retained-owner tests reject inconsistent physical normalization.
"""

import pytest

from pyamplicol.loops.directed_color_projection import reproject_directed_color_columns

from .test_loop_directed_color_reuse import algebra as algebra
from .test_loop_directed_color_reuse import graded as graded


def _entries(rows):
    if isinstance(rows, tuple):
        for item in rows:
            yield from _entries(item)
    else:
        yield rows


@pytest.mark.parametrize("accuracy", ("full", "lc", "nlc"))
def test_reused_rectangles_bind_each_exact_target_average_once(graded, accuracy):
    nc, _tensor, _histories, reference_inputs, target_inputs, reference = graded
    # Distinct non-integer rational averages expose accidental reference
    # normalization, double averaging and binary64 conversion.
    from symbolica import E

    reference_average = E("18")
    target_averages = (E("125/7"), E("91/5"))
    reference_before = reference.bound(reference_average)
    retained_rows = (reference.born, reference.columns)
    target = reproject_directed_color_columns(
        reference, accuracy, reference_inputs, target_inputs
    )
    exact = tuple(
        value.replace(nc, 3).together()
        for value in _entries((target.born, target.columns))
    )
    assert any(value != 0 for value in exact)
    bound = []
    for average in target_averages:
        values = tuple(_entries(target.bound(average)))
        assert len(values) == len(exact)
        assert all(
            (actual * average - expected).together() == 0
            for actual, expected in zip(values, exact, strict=True)
        )
        bound.append(values)
    assert any(a != b for a, b in zip(*bound, strict=True))
    assert all(
        (left * target_averages[0] - right * target_averages[1]).together() == 0
        for left, right in zip(*bound, strict=True)
    )
    assert (reference.born, reference.columns) == retained_rows
    assert reference.bound(reference_average) == reference_before
    assert target.families == reference.families
