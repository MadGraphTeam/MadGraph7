# SPDX-License-Identifier: 0BSD
"""Shared bodies of the M-QCD campaign test modules
(``tests/unit/test_loop_v2_mqcd_*.py``; m-qcd/PLAN.md §3.2 MQ-VALIDATE,
§4.3 "Tiers").

The nightly tier is split into file jobs (manifest ``nightly_jobs``: rows
packed by measured cost, each file job within PLAN §3.2's 30 minutes); each
``test_loop_v2_mqcd_nightly*.py`` module runs one job through
``nightly_module``. Generation is module-scoped: a row's variants run in the
same job and reuse its generation.
"""

from __future__ import annotations

import pytest

from tools.loop_benchmark import mqcd


def campaign_fixture():
    @pytest.fixture(scope="module")
    def campaign(loop_sm_ufo, loop_native, tmp_path_factory):
        return mqcd.Campaign(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("mqcd"))

    return campaign


def nightly_module(job: int):
    """(campaign fixture, row test) of nightly file job ``job``: every row id
    of the job at its nightly points in binary64 and double-double under the
    acceptance rules 1-4 of PLAN §1.4, plus the extras the harness checks."""

    names = mqcd.nightly_job(job)

    @pytest.mark.loop_nightly
    @pytest.mark.parametrize("name", names)
    def test_row_meets_the_acceptance_rules(campaign, name):
        report = campaign.report(name, "nightly")
        mqcd.write_result(report)
        assert report.checks, name
        assert report.passed, report.describe()

    return campaign_fixture(), test_row_meets_the_acceptance_rules
