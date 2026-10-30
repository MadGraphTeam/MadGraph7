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

import pytest

from tests.unit.loop_mqcd_tiers import nightly_module
from tools.loop_benchmark import mqcd

MANIFEST = mqcd.ROOT / "tests" / "fixtures" / "loop" / "msm" / "manifest.json"
# Nightly file job 1 (manifest ``nightly_jobs``).
campaign, test_row_meets_the_acceptance_rules = nightly_module(1, MANIFEST)

MULTIPARTICLE = sorted(
    data["id"]
    for data in mqcd.load_manifest(MANIFEST)["rows"]
    if "madloop_subprocesses" in data
)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", MULTIPARTICLE)
def test_multiparticle_members_are_madloops(campaign, name):
    """MP01, MP02 (PLAN §4.2, Q2 (a)): the expansion of the multiparticle
    label (the photon in p through process.multiparticles, as MG5_aMC on a
    QED-perturbed model) is MadLoop's SubProcesses/P* up to the members
    without a Born, which the public route's dry run skips with the typed
    loop-induced record while planning every other member (each P reference
    runs as a row variant, ``MP0n@P<n>``)."""

    checks = mqcd.multiparticle_subprocesses(
        campaign, name
    ) + mqcd.multiparticle_members(campaign, name)
    mqcd.write_result(mqcd.RowReport(f"expansion:{name}", "nightly", (), checks, {}))
    assert all(c.passed for c in checks), checks
