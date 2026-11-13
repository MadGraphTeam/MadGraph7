# SPDX-License-Identifier: 0BSD
"""Four-leg legacy O1 of the v2 observables beyond the default tier (WP2.5;
nightly).

The directed, spin and helicity-resolved outputs of the four-leg legacy
records (g g > g g and d d~ > g g with its member s s~ > g g; full colour, LC
and NLC) at 200 bits, and in binary64 at the points and records the default
tier leaves out (``o1_nightly``).
"""

from __future__ import annotations

import pytest

from .test_loop_v2_observables import (
    DIRECTED_DEFAULT,
    FOUR_LEG_RECORDS,
    RESOLVED_DEFAULT,
    RESOLVED_RECORDS,
    assert_directed_legacy,
    assert_helicity_resolved_legacy,
    o1_id,
    o1_nightly,
    observable_builder,
)

pytestmark = pytest.mark.loop_nightly


@pytest.fixture(scope="module")
def generate(loop_sm_compiled_model, loop_native):
    return observable_builder(loop_sm_compiled_model, loop_native)


@pytest.mark.parametrize(
    ("name", "carrier", "points"),
    o1_nightly(DIRECTED_DEFAULT, FOUR_LEG_RECORDS),
    ids=o1_id,
)
def test_four_leg_directed_and_spin_outputs_equal_legacy(
    generate, name, carrier, points
):
    assert_directed_legacy(generate, name, carrier, points)


@pytest.mark.parametrize(
    ("name", "carrier", "points"),
    o1_nightly(RESOLVED_DEFAULT, RESOLVED_RECORDS),
    ids=o1_id,
)
def test_four_leg_helicity_resolved_outputs_equal_legacy(
    generate, name, carrier, points
):
    assert_helicity_resolved_legacy(generate, name, carrier, points)
