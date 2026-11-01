# SPDX-License-Identifier: 0BSD
"""One-loop decays (M-FINAL MF-DEC-2), default tier.

The default tier's row of ``tests/fixtures/loop/dec/manifest.json`` (XX01,
t > b w+ [QCD] on loop_sm, generated in seconds) at points 0 and 1 in
binary64 and double-double against MadLoop's quadruple-precision reference
under the campaign rules 1-4 of m-sm/PLAN.md §1.4 (``tools/loop_benchmark/
mqcd.py``, the M-QCD harness parametrised by the decay manifest); the other
rows and points run nightly (``test_loop_v2_dec_nightly.py``). With
``PYAMPLICOL_MQCD_RESULTS`` set every row writes its record for the report
data (``build_mqcd.py --manifest tests/fixtures/loop/dec/manifest.json
--output data/dec.json``).
"""

from __future__ import annotations

import pytest

from tests.unit.loop_mqcd_tiers import campaign_fixture
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "dec" / "manifest.json"
TIER = "default"
ROWS = sorted(
    name
    for name, row in mqcd.rows(mqcd.load_manifest(MANIFEST)).items()
    if row.tier == TIER and row.active
)

campaign = campaign_fixture(MANIFEST)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ROWS)
def test_row_meets_the_acceptance_rules(campaign, name):
    report = campaign.record(name, TIER)
    mqcd.write_result(report)
    assert report.checks, name
    assert report.passed, report.describe()
