# SPDX-License-Identifier: 0BSD
"""The loop-induced validation campaign (m-lit PLAN §3.4 LIT-VAL; §4.0 rules
on the §4.1 rows): every fixture run of ``tools/loop_benchmark/li_manifest.py``
generated through the public route and evaluated on both reduction backends,
binary64 and double-double (AP on each tier's AP run), judged by the §4.0
rules (``li_manifest.judge``). Local-only nightly ids (``loop_nightly``, run
at the M-LIT close, slice-status DECISIONS 2026-11-07 ~13:34): one test id
per run, the 6- and 7-leg rows take minutes to hours of generation.

A run passes when no check fails: estimate misses, TIRust's failed points
(partial, its typed reasons recorded) and the double-double / AP arms of L6
that wait on lit-tir-mp are recorded, not failures (m-lit rulings (5), (8));
a deviation below ~1e-5 relative is never a failure (precision lax).

With ``PYAMPLICOL_LI_CAMPAIGN_RECORDS=<dir>`` the records are kept there
(``li_manifest.py lit --records <dir>`` writes data/lit.json from them).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.loop_benchmark import li_manifest as lm

pytestmark = [pytest.mark.loop_physics, pytest.mark.loop_nightly]

ENTRIES = {e["fixture"]: e for e in lm.runs()}


def _work(tmp_path_factory) -> Path:
    configured = os.environ.get("PYAMPLICOL_LI_CAMPAIGN_RECORDS")
    return Path(configured) if configured else tmp_path_factory.mktemp("li-campaign")


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_campaign_run_passes_its_rules(name, tmp_path_factory, loop_native):
    entry = ENTRIES[name]
    work = _work(tmp_path_factory)
    path = work / name / "record.json"
    record = lm.evaluate(name, work, ap=bool(entry["ap"]))
    path.write_text(json.dumps(record, indent=1) + "\n")
    judged = lm.judge(record, entry)
    statuses = {c: judged[c]["status"] for c in lm.CHECKS if c in judged}
    assert "fail" not in statuses.values(), statuses
    for check in entry["checks"]:
        if check in lm.CHECKS and check != "L9":
            assert check in statuses, (check, statuses)


@pytest.mark.parametrize(
    "target", lm.limit_targets(lm.runs()), ids=lambda t: t["fixture"]
)
def test_heavy_top_limit(target, tmp_path_factory, loop_native):
    """LI-A7: the heavy-top scan decouples to its limit oracle (the
    low-energy theorem; the HEFT trees carry a recorded reference finding,
    ``tests/fixtures/loop/required/li/lit-val/records/delegated.json``)."""

    work = _work(tmp_path_factory)
    if not (work / target["fixture"] / "artifact" / "artifact.json").is_file():
        lm.evaluate(target["fixture"], work, ap=False, identity=False)
    result = lm.limits(target["fixture"], target["oracle"], work)
    findings = json.loads(lm.DELEGATED_RECORDS.read_text()).get("findings", {})
    if "L9" in findings.get(target["fixture"], {}):
        pytest.xfail(findings[target["fixture"]]["L9"][:200])
    assert result["status"] == "pass", result
