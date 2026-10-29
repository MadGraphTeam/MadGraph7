# SPDX-License-Identifier: 0BSD
"""M-QCD campaign, nightly tier (m-qcd/PLAN.md §4.3 "Tiers").

Every manifest row whose reference exists, at its nightly points (a default
row's points beyond 0 and 1), in binary64 and double-double under the
acceptance rules 1-4 of PLAN §1.4 (``tools/loop_benchmark/mqcd.py``), plus
the extras the harness checks: ``hel`` (per-helicity double-double against
MadLoop's SLOOPMATRIXHEL rows within 1e-12), ``uv`` (rule 5, the bare loop's
UV poles against the counterterms and the Feynman-gauge field factors with
n_f from the request), ``ap`` (rule 6, 40 digits against double-double and
against 60 digits), ``ward`` (O10, ``ward.ward_check`` on the row's Ward
slice), ``ff`` (D01's analytic quark form factor), ``sp`` (the special-point
variants) and the H rows' exact CKM relation (manifest ``ckm_relation``,
against the diagonal twin within 1e-24). The tier is split
into file jobs (manifest ``nightly_jobs``, rows packed by measured cost into
jobs within PLAN §3.2's 30 minutes): this module is job 1 plus the smokes
and the M01 expansion, test_loop_v2_mqcd_nightly_<job>.py the others; one
test id per row. The manifest lists the pending rows and extras with the
package each waits for.
A row's variants (``<id>@<name>``: a runtime parameter change on the same
generation) are ids of their own; a non-gating variant (a diagnostic twin)
is listed in the manifest and left to the campaign CLI. Rows on the
native-input route (heavy 2->4/2->5 rows; no native Catani prediction, WP3.1
I5c) take rule 4's "tree" prediction from the tree engine's committed
colour correlations (manifest ``tree_correlations``; checked here against
the executor's Catani on rows of the same colour structures) or record it
as not formed without them (status "partial"), as M01's public-route
variants do (their tools-route twins hold rule 4). M01 is composite: its
variants, its expansion and MQ-MODEL's bitwise test (manifest
``delegated``) make it up.
"""

from __future__ import annotations

import pytest

from tests.unit.loop_mqcd_tiers import nightly_module
from tools.loop_benchmark import mqcd

TIER = "nightly"
# Nightly file job 1 (manifest ``nightly_jobs``; the other jobs are
# test_loop_v2_mqcd_nightly_<job>.py), plus the smokes and the M01 expansion.
campaign, test_row_meets_the_acceptance_rules = nightly_module(1)


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


@pytest.mark.loop_nightly
def test_tree_catani_is_the_executor_s(campaign):
    """Rule 4's "tree" prediction of the native-input rows (the harness's
    massless Catani operator on the tree engine's colour-correlated Born)
    equals the Python-input executor's Catani on rows of the same colour
    structures (g g > g g, u u~ > g g g) within 1e-30, its Born ours."""

    checks = mqcd.tree_oracle(campaign)
    mqcd.write_result(mqcd.RowReport("oracle:tree-catani", TIER, (), checks, {}))
    assert checks and all(c.passed for c in checks), checks


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
