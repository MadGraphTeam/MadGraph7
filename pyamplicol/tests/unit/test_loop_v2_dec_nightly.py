# SPDX-License-Identifier: 0BSD
"""One-loop decays (M-FINAL MF-DEC-2), nightly tier, file job 1 (manifest
``nightly_jobs``): every row of ``tests/fixtures/loop/dec/manifest.json``
(XX01-XX08: 1 -> 2 and 1 -> 3, QCD and EW) at its nightly points in binary64
and double-double against MadLoop's quadruple-precision references under the
campaign rules 1-4 of m-sm/PLAN.md §1.4, plus the extras the harness checks
(``uv``: rule 5 with the incoming leg's field factor; ``ap``: rule 6;
``ward``: O10 on XX07's external gluon) and the public-route variants
(``XX01@public``, ``XX07@public``: ``Generator`` on the shipped loop_sm,
``Runtime.evaluate`` of the stored artifact with the O1 checks)."""

from __future__ import annotations

from tests.unit.loop_mqcd_tiers import nightly_module
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "dec" / "manifest.json"
campaign, test_row_meets_the_acceptance_rules = nightly_module(1, MANIFEST)
