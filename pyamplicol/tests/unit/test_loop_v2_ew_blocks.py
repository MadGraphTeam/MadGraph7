# SPDX-License-Identifier: 0BSD
"""Electroweak loops of processes with two or more quark lines whose bosons
meet at a colour-singlet vertex (m-lit lit-ew-blocks, F-A).

In loop-request mode the colour engine built a merge of two colour singlets
that both carry coloured labels (WWA, WWZ or HZZ joining the bosons of two
quark lines, a four-point contact's partial or final merge) in both input
orders: two block words of one colour factor, so the loop seeds counted that
vertex's trees twice and the poles of the e^6 loop missed Catani's
prediction (u u~ > d d~ e+ e-: double pole -(alpha/2pi) sum Q^2
Re<A0|A0 + (9/14) T>, T the W+ W- -> gamma/Z -> e+ e- sub-amplitude). One
input order is kept now (``ColorEngine._loop_block_trace_orientation_allowed``
and its native mirror; the DAG-level tests are in
``test_loop_v2_native_dag.py`` and ``test_loop_v2_blocks.py``).

Here ``pyamplicol check`` decides on the regenerated requests (QED
corrections, QCD 0 QED 4, every width zero, massless b and tau): both poles
of every point against Catani's operator in binary64 and in double-double,
each carrier alone, for the W-paired lines (u u~ > d d~ e+ e-, the W-only
pairing u c~ > d s~ e+ e-, neutrinos u u~ > d d~ ve ve~), three lines
(u u~ > d d~ s s~) and a massive line coupling to the Higgs
(u u~ > s s~ t t~); and the Ward identity of both photons of
u u~ > d d~ a a (failed at 1e-5 - 2.5e-3 before). Each generates its request
in memory (about a minute and 8 GiB): nightly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_nightly

LINES = (
    "u u~ > d d~ e+ e-",
    "u c~ > d s~ e+ e-",
    "u u~ > d d~ ve ve~",
    "u u~ > d d~ s s~",
    "u u~ > s s~ t t~",
)


def _card(root: Path, expression: str, extra: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    card.write_text(
        f"""action = "check"
[model]
source = "loop_qcd_qed_sm_Gmu"
restriction = "default"
[process]
entries = [{{expression = {json.dumps(expression)}, name = "p"}}]
loop_coupling_orders = ["QED"]
coupling_order_policy = "explicit"
max_coupling_orders = {{ QCD = 0, QED = 4 }}
[loop.preparation]
fixed_parameters = {{MB = "0", MTA = "0", WT = "0", WZ = "0", WW = "0", WH = "0"}}
[check]
points = 2
{extra}"""
    )
    return card


@pytest.mark.parametrize("precision", ["double", "double-double"])
@pytest.mark.parametrize("expression", LINES)
def test_poles_equal_catani(tmp_path, expression, precision):
    from pyamplicol.api.services import check
    from pyamplicol.loops.v2 import checks

    report = check(_card(tmp_path, expression), precision=precision)
    (process,) = report.processes
    poles = [row for row in process.rows if row.check == "poles"]
    assert len(poles) == 2 * 2  # one QED^10 sector, two coefficients, two points
    assert all(row.verdict == checks.PASS for row in poles), poles
    assert all(row.carrier == precision for row in poles)
    assert report.exit_code == checks.EXIT_PASSED


def test_ward_identities_of_two_photons(tmp_path):
    from pyamplicol.api.services import check
    from pyamplicol.loops.v2 import checks

    # Untagged photons: the G_mu model has no adjustable 1/alpha(0) for
    # tagged ones (ew_policy.ExternalPhotonError).
    report = check(
        _card(tmp_path, "u u~ > d d~ a a", '[loop]\nexternal_photons = "none"\n')
    )
    (process,) = report.processes
    ward = [row for row in process.rows if row.check == "ward"]
    assert sorted({row.subject for row in ward}) == [
        "leg 5, c = 1",
        "leg 5, c = i",
        "leg 6, c = 1",
        "leg 6, c = i",
    ]
    assert all(row.verdict == checks.PASS for row in ward), ward
    assert report.exit_code == checks.EXIT_PASSED
