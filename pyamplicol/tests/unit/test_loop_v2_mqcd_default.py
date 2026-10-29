# SPDX-License-Identifier: 0BSD
"""M-QCD campaign, default tier (m-qcd/PLAN.md §4.3 "Tiers").

The default tier's rows (``tests/fixtures/loop/mqcd/manifest.json``, tier
``default``) whose reference exists, at points 0 and 1, in binary64 and
double-double, held to the acceptance rules 1-4 of PLAN §1.4
(``tools/loop_benchmark/mqcd.py``): double-double against the primary
provider within 1e-12 strictly relative per sector and coefficient (Born
included), binary64 against double-double under the user's O1 rule, the
stability estimate bracketing the true binary64 error tenfold, and the poles
against Catani's with the expected predicted and unpredicted sector sets.
Generation is shared per row; a row whose reference has not landed yet is
listed as pending in the manifest (``test_loop_v2_mqcd_manifest``), never
skipped here. With ``PYAMPLICOL_MQCD_RESULTS`` set every row writes its
record for the report data (``docs/performance_reports/loop/scripts/
build_mqcd.py``).
"""

from __future__ import annotations

import pytest

from tools.loop_benchmark import mqcd

TIER = "default"
ROWS = sorted(
    name for name, row in mqcd.rows().items() if row.tier == TIER and row.active
)


@pytest.fixture(scope="module")
def campaign(loop_sm_ufo, loop_native, tmp_path_factory):
    return mqcd.Campaign(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("mqcd"))


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ROWS)
def test_row_meets_the_acceptance_rules(campaign, name):
    report = campaign.report(name, TIER)
    mqcd.write_result(report)
    assert report.checks, name
    assert report.passed, report.describe()
