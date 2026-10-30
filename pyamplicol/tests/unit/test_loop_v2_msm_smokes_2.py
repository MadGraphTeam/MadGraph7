# SPDX-License-Identifier: 0BSD
"""M-SM restriction smokes, file job 2 (m-sm/PLAN.md §3.2 MS-VALIDATE item 8;
manifest ``smokes``): each shipped restriction of the G_mu and alpha(M_Z) UFOs
(``parallel_test*`` excepted, PLAN §1.1) generates its base row's request
(LL01 on G_mu, LL03 on alpha(M_Z), whose rows hold the default restrictions)
and equals the base row's generation through the native executor in
binary64 at the row's first point and parameters within 1e-14 per sector and
coefficient (``mqcd.smoke``). One test id per restriction, about 5 min each
(one EW generation)."""

from __future__ import annotations

import pytest

from tests.unit.loop_mqcd_tiers import campaign_fixture
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "msm" / "manifest.json"
SMOKES = mqcd.load_manifest(MANIFEST)["smokes"]
JOB = 2
NAMES = next(job["smokes"] for job in SMOKES["jobs"] if job["job"] == JOB)
SPECS = {entry["id"]: entry for entry in SMOKES["restrictions"]}

campaign = campaign_fixture(MANIFEST)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", NAMES)
def test_restriction_smoke_equals_its_base_row(campaign, name):
    spec = SPECS[name]
    checks = mqcd.smoke(campaign, spec["restriction"], spec)
    mqcd.write_result(mqcd.RowReport(f"smoke:{name}", "nightly", (), checks, {}))
    failures = [c for c in checks if not c.passed]
    assert not failures, failures
