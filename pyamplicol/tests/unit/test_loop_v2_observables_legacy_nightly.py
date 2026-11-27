# SPDX-License-Identifier: 0BSD
"""Five-leg legacy O1 of the v2 observables (WP2.5; nightly).

The directed and spin outputs of u u~ > g g g (full colour, spin sources, LC
and NLC) against the frozen legacy records, in binary64 (both executors,
under the binary64 rule of ``test_loop_v2_observables.py``) and at 200 bits,
at every point: the configurations the default tier leaves out
(``o1_nightly``). About two minutes each in the Python executor.
"""

from __future__ import annotations

import pytest

from .test_loop_v2_observables import (
    DIRECTED_DEFAULT,
    FIVE_LEG_RECORDS,
    assert_directed_legacy,
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
    o1_nightly(DIRECTED_DEFAULT, FIVE_LEG_RECORDS),
    ids=o1_id,
)
def test_five_leg_directed_and_spin_outputs_equal_legacy(
    generate, name, carrier, points
):
    assert_directed_legacy(generate, name, carrier, points)
