# SPDX-License-Identifier: 0BSD
"""M-SM campaign, default tier (m-sm/PLAN.md §3.2 MS-VALIDATE item 3, §4.3).

The default tier's rows (``tests/fixtures/loop/msm/manifest.json``, tier
``default``: VJ02, u d~ > w+ g, and since MS-GEN TT01, g g > t t~, which
generates in under 60 s, both with QCD and QED corrections) at points 0
and 1 in binary64 and double-double under the acceptance rules of PLAN §1.4
(``tools/loop_benchmark/mqcd.py``, the M-QCD harness parametrised by the EW
manifest). A row whose reference has not landed (VJ02 waits for MS-REF) is
pending in the manifest (``test_loop_v2_msm_manifest``), never skipped
here: the module holds a row test only while the tier has an active row.
With ``PYAMPLICOL_MQCD_RESULTS`` set every row writes its record for the
report data (``build_mqcd.py --manifest tests/fixtures/loop/msm/manifest.json``).
"""

from __future__ import annotations

import pytest

from tests.unit.loop_mqcd_tiers import campaign_fixture
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "msm" / "manifest.json"
TIER = "default"
ROWS = sorted(
    name
    for name, row in mqcd.rows(mqcd.load_manifest(MANIFEST)).items()
    if row.tier == TIER and row.active
)

campaign = campaign_fixture(MANIFEST)


@pytest.mark.loop_fast
def test_default_tier_rows_are_active_or_wait_for_a_package():
    """Every default-tier row runs here or names the package it waits for."""

    for name, row in mqcd.rows(mqcd.load_manifest(MANIFEST)).items():
        if row.tier == TIER:
            assert row.active or row.pending["waits_for"], name


if ROWS:

    @pytest.mark.loop_physics
    @pytest.mark.parametrize("name", ROWS)
    def test_row_meets_the_acceptance_rules(campaign, name):
        report = campaign.record(name, TIER)
        mqcd.write_result(report)
        assert report.checks, name
        assert report.passed, report.describe()
