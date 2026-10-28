# SPDX-License-Identifier: 0BSD
"""M-QCD campaign, nightly tier (m-qcd/PLAN.md §4.3 "Tiers").

Every manifest row whose reference exists, at its nightly points (a default
row's points beyond 0 and 1), in binary64 and double-double under the
acceptance rules 1-4 of PLAN §1.4 (``tools/loop_benchmark/mqcd.py``), plus
the extras the harness checks: ``uv`` (rule 5, the bare loop's UV poles
against the counterterms and the Feynman-gauge field factors with n_f from
the request) and ``ap`` (rule 6, 40 digits against double-double and against
60 digits). One test id per row, so the nightly jobs split per row; the
manifest lists the pending rows and extras with the package each waits for.
"""

from __future__ import annotations

import pytest

from tools.loop_benchmark import mqcd

TIER = "nightly"
ROWS = sorted(
    name
    for name, row in mqcd.rows().items()
    if row.active
    and (row.tier_points(TIER) or set(row.extras) & mqcd.IMPLEMENTED_EXTRAS)
)


@pytest.fixture(scope="module")
def campaign(loop_sm_ufo, loop_native, tmp_path_factory):
    return mqcd.Campaign(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("mqcd"))


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ROWS)
def test_row_meets_the_acceptance_rules(campaign, name):
    report = campaign.report(name, TIER)
    mqcd.write_result(report)
    assert report.checks, name
    assert report.passed, report.describe()
