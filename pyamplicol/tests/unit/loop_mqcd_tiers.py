# SPDX-License-Identifier: 0BSD
"""Shared bodies of the campaign test modules, parametrised by manifest:
M-QCD (``tests/unit/test_loop_v2_mqcd_*.py``, ``tests/fixtures/loop/mqcd/
manifest.json``; m-qcd/PLAN.md §3.2 MQ-VALIDATE, §4.3 "Tiers") and M-SM
(``tests/unit/test_loop_v2_msm_*.py``, ``tests/fixtures/loop/msm/
manifest.json``; m-sm/PLAN.md §3.2 MS-VALIDATE, §4.3).

The nightly tier is split into file jobs (manifest ``nightly_jobs``: rows
packed by measured cost, each file job within PLAN §3.2's 30 minutes); each
``test_loop_v2_<campaign>_nightly*.py`` module runs one job through
``nightly_module``. Generation is module-scoped: a row's variants run in the
same job and reuse its generation. A manifest's ``models`` section names the
conftest fixture of each model's UFO other than loop_sm (the M-SM EW UFOs);
the campaign requests only those.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.loop_benchmark import mqcd


def campaign_fixture(manifest_path: Path = mqcd.MANIFEST):
    """The module-scoped ``campaign`` fixture of the manifest at
    ``manifest_path``."""

    @pytest.fixture(scope="module")
    def campaign(request, loop_sm_ufo, loop_native, tmp_path_factory):
        manifest = mqcd.load_manifest(manifest_path)
        ufos = {
            name: request.getfixturevalue(spec["fixture"])
            for name, spec in (manifest.get("models") or {}).items()
            if "fixture" in spec
        }
        return mqcd.Campaign(
            loop_sm_ufo,
            loop_native,
            tmp_path_factory.mktemp(manifest_path.parent.name),
            manifest=manifest,
            ufos=ufos,
        )

    return campaign


def nightly_module(job: int, manifest_path: Path = mqcd.MANIFEST):
    """(campaign fixture, row test) of nightly file job ``job`` of the
    manifest at ``manifest_path``: every row id of the job at its nightly
    points in binary64 and double-double under the acceptance rules of PLAN
    §1.4, plus the extras the harness checks."""

    names = mqcd.nightly_job(job, mqcd.load_manifest(manifest_path))

    @pytest.mark.loop_nightly
    @pytest.mark.parametrize("name", names)
    def test_row_meets_the_acceptance_rules(campaign, name):
        report = campaign.record(name, "nightly")
        mqcd.write_result(report)
        assert report.checks, name
        assert report.passed, report.describe()

    return campaign_fixture(manifest_path), test_row_meets_the_acceptance_rules
