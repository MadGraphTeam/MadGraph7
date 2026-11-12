# SPDX-License-Identifier: 0BSD
"""M-SM campaign, nightly tier, file job 13 (manifest ``nightly_jobs``;
m-sm/PLAN.md §3.2 and §4.3): the job's rows at their nightly points under
PLAN §1.4 (``tests/unit/test_loop_v2_msm_nightly.py`` describes the tier)."""

from __future__ import annotations

from tests.unit.loop_mqcd_tiers import nightly_module
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "msm" / "manifest.json"
campaign, test_row_meets_the_acceptance_rules = nightly_module(13, MANIFEST)
