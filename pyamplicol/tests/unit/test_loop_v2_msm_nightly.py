# SPDX-License-Identifier: 0BSD
"""M-SM campaign, nightly tier (m-sm/PLAN.md §3.2 MS-VALIDATE item 3, §4.3).

Every active M-SM manifest row (``tests/fixtures/loop/msm/manifest.json``)
at its nightly points in binary64 and double-double under the acceptance
rules of PLAN §1.4 (``tools/loop_benchmark/mqcd.py``: rule 1 against the
primary provider, MadLoop or, under the R2 rule, Recola or MadLoop plus its
omitted R2 component; rules 2-3 binary64 against double-double under the O1
rule; rule 4 the engine's QCD and QED operators with the expected predicted
and unpredicted sectors and typed reasons), plus the extras the harness
checks (``uv``: rule 5's EW component form against MadLoop's Born x CT
split; ``ap``; ``ward``). The tier is split into file jobs (manifest
``nightly_jobs``, each within PLAN §3.2's 30 minutes); this module is job 1,
test_loop_v2_msm_nightly_<job>.py the others; one test id per row. Pending
rows and extras name the package they wait for.
"""

from __future__ import annotations

from tests.unit.loop_mqcd_tiers import nightly_module
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "msm" / "manifest.json"
# Nightly file job 1 (manifest ``nightly_jobs``).
campaign, test_row_meets_the_acceptance_rules = nightly_module(1, MANIFEST)
