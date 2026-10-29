# SPDX-License-Identifier: 0BSD
"""M-QCD campaign, nightly tier, file job 4 (manifest ``nightly_jobs``;
m-qcd/PLAN.md §3.2 and §4.3): the job's rows at their nightly points under
PLAN §1.4 (``tests/unit/test_loop_v2_mqcd_nightly.py`` describes the tier)."""

from __future__ import annotations

from tests.unit.loop_mqcd_tiers import nightly_module

campaign, test_row_meets_the_acceptance_rules = nightly_module(4)
