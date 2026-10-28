# SPDX-License-Identifier: 0BSD
"""M-QCD campaign, nightly tier (m-qcd/PLAN.md §4.3 "Tiers").

Every manifest row whose reference exists, at its nightly points (a default
row's points beyond 0 and 1), in binary64 and double-double under the
acceptance rules 1-4 of PLAN §1.4 (``tools/loop_benchmark/mqcd.py``), plus
the extras the harness checks: ``hel`` (per-helicity double-double against
MadLoop's SLOOPMATRIXHEL rows within 1e-12), ``uv`` (rule 5, the bare loop's
UV poles against the counterterms and the Feynman-gauge field factors with
n_f from the request) and ``ap`` (rule 6, 40 digits against double-double and
against 60 digits). One test id per row, so the nightly jobs split per row; the
manifest lists the pending rows and extras with the package each waits for.
A row's variants (``<id>@<name>``: a runtime parameter change on the same
generation) are ids of their own; a non-gating variant (a diagnostic twin)
is listed in the manifest and left to the campaign CLI. Rows on the
native-input route (heavy 2->4/2->5 gluon rows) record rule 4 as not formed
(no native Catani prediction, WP3.1 I5c).
"""

from __future__ import annotations

import pytest

from tools.loop_benchmark import mqcd

TIER = "nightly"
ROWS = sorted(
    name
    for name, row in mqcd.rows().items()
    if row.active
    and row.gating
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


SMOKES = mqcd.load_manifest()["smokes"]["same_request"]


@pytest.mark.loop_nightly
@pytest.mark.parametrize("restriction", sorted(SMOKES))
def test_restriction_smoke_equals_its_base_row(campaign, restriction):
    """no_masses, no_tau_mass and no_widths: the same request as its base row
    (on that row's restriction) within 1e-14 (PLAN §3.2 item 5)."""

    checks = mqcd.smoke(campaign, restriction, SMOKES[restriction])
    mqcd.write_result(mqcd.RowReport(f"smoke:{restriction}", TIER, (), checks, {}))
    failures = [c for c in checks if not c.passed]
    assert not failures, failures


MULTIPARTICLE = sorted(
    data["id"]
    for data in mqcd.load_manifest()["rows"]
    if "madloop_subprocesses" in data
)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", MULTIPARTICLE)
def test_multiparticle_expansion_is_madloops(campaign, name):
    """M01: the public route's expansion of the multiparticle label is the
    set of processes of MadLoop's SubProcesses/P* (each P reference runs as
    a row variant, ``M01@P<n>``)."""

    checks = mqcd.multiparticle_subprocesses(campaign, name)
    mqcd.write_result(mqcd.RowReport(f"expansion:{name}", TIER, (), checks, {}))
    assert all(c.passed for c in checks), checks
