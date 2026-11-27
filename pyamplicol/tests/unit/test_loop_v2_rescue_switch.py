# SPDX-License-Identifier: 0BSD
"""NinjaRS's cancellation rescue follows pyAmpliCol's reduction-check switch,
end to end, on a known catastrophic point of g g > g g g (leading colour):
the checked mode (the default) rescues it, the unchecked mode
(``reduction_unchecked``, ``profile --runtime``'s optimistic baseline) is
bit for bit the reduction without the rescue (pyAmpliCol v1.0.3's value).

The guard against a silent change of NinjaRS's own default of the rescue
(``FastOptions::cancellation_rescue``): pyAmpliCol sets it explicitly in both
modes (``runtime::fast_options``), and this test fails if either mode stops
doing what it says.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.loop_physics

CARD = """
schema_version = 1
action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
[process]
entries = [{{ expression = "g g > g g g", name = "gg_ggg" }}]
loop_coupling_orders = ["QCD"]
max_coupling_orders = {{ QCD = 3, QED = 0 }}
[color]
accuracy = "lc"
[generation]
output = "{output}"
particle_veto = ["t"]
"""

# A flat phase-space point (sqrt(s) = 1 TeV) where a two-mass triangle cut is
# nearly degenerate: the binary64 reduction without the rescue loses every
# digit (its value is about 1e5 times the true one).
POINT = [
    [500.0, 0.0, 0.0, 500.0],
    [500.0, 0.0, 0.0, -500.0],
    [396.2841211238355, 382.23488712911774, 4.314451727306183, -104.49392911827043],
    [320.4782427306587, -224.43258053286556, -226.96666288556239, 28.67847272225751],
    [283.237636145506, -157.8023065962522, 222.6522111582562, 75.81545639601293],
]
MU_R2 = "8315.251344"
# The finite part in double-double arithmetic (the reference).
REFERENCE = -0.005366654929561328715628165898277900206149235833991228
# The binary64 finite part without the rescue (v1.0.3, and the unchecked mode).
UNRESCUED = 409.6791465718536


@pytest.fixture(scope="module")
def gg_ggg(loop_native, tmp_path_factory):
    root = tmp_path_factory.mktemp("gg-ggg-lc")
    card = root / "card.toml"
    card.write_text(CARD.format(output=root / "artifact"))
    subprocess.run(
        [sys.executable, "-m", "pyamplicol", "generate", "--card", str(card)],
        check=True,
        capture_output=True,
        cwd=root,
    )
    return root / "artifact"


def test_the_checks_switch_turns_the_rescue_on_and_off(gg_ggg):
    from pyamplicol import Runtime

    call = dict(mu_r2=MU_R2, poles="none", reduction="ninjars")
    unchecked = Runtime.load(gg_ggg, reduction_unchecked=True).evaluate([POINT], **call)
    value = complex(unchecked.values[0][0].finite)
    assert value.real == UNRESCUED, repr(value)
    assert unchecked.diagnostics[0].reduction_rescue is None

    checked = Runtime.load(gg_ggg).evaluate([POINT], **call)
    value = complex(checked.values[0][0].finite)
    assert abs(value.real - REFERENCE) <= 1e-8 * abs(REFERENCE), repr(value)
    rescue = checked.diagnostics[0].reduction_rescue
    assert rescue is not None and rescue.count > 0
    assert rescue.precision == "double-double"
