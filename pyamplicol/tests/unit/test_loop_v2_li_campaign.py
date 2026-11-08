# SPDX-License-Identifier: 0BSD
"""The loop-induced validation campaign (m-lit PLAN §3.4 LIT-VAL; §4.0 rules
on the §4.1 rows): every fixture run of ``tools/loop_benchmark/li_manifest.py``
generated through the public route and evaluated on both reduction backends,
binary64 and double-double (AP on each tier's AP run), judged by the §4.0
rules (``li_manifest.judge``). Local-only nightly ids (``loop_nightly``, run
at the M-LIT close, slice-status DECISIONS 2026-11-07 ~13:34): one test id
per run, the 6- and 7-leg rows take minutes to hours of generation.

A run passes when no check fails: estimate misses, TIRust's failed points
(partial, its typed reasons recorded) and TIRust arms of an extension before
lit-tir-mp (waiting on it) are recorded, not failures (m-lit rulings (5),
(8)); L6's verdict is the worst of its carriers, so a backend disagreement
in binary64, double-double or AP fails the run, as does a Ward probe (L4,
every fixture point and massless gauge leg) beyond ten times its estimate;
a deviation below ~1e-5 relative is never a failure (precision lax).

With ``PYAMPLICOL_LI_CAMPAIGN_RECORDS=<dir>`` the records are kept there
(``li_manifest.py lit --records <dir>`` writes data/lit.json from them).
"""

from __future__ import annotations

import json
import math
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
    skip, plain = lm.run_skips(name)
    record = lm.evaluate(
        name, work, ap=bool(entry["ap"]), skip=skip, high_checked=not plain
    )
    if record.get("transformed_points"):
        # LI-L1 on TIRust (``li_manifest.py identity-tirust``), folded into L5
        record["identity_tirust"] = lm.identity_tirust(name, work, record)
    path.write_text(json.dumps(record, indent=1) + "\n")
    judged = lm.judge(record, entry)
    statuses = {c: judged[c]["status"] for c in lm.CHECKS if c in judged}
    assert "fail" not in statuses.values(), statuses
    assert "effective_backend_switched" not in judged["L6"]["double"], judged["L6"]
    if record.get("transformed_points"):
        assert judged["L5"]["tirust"]["status"] != "not run", judged["L5"]
    for check in entry["checks"]:
        if check in lm.CHECKS and check != "L9":
            assert check in statuses, (check, statuses)


@pytest.mark.parametrize(
    "target", lm.limit_targets(lm.runs()), ids=lambda t: t["fixture"]
)
def test_heavy_top_limit(target, tmp_path_factory, loop_native):
    """LI-A7: the heavy-top scan decouples to its limit oracle (the
    low-energy theorem). The HEFT trees carry a recorded reference finding
    (``tests/fixtures/loop/required/li/lit-val/records/delegated.json``):
    there the scan must decouple to the finding's expected limit (8/9)
    within ten times its extrapolation error at every point, so a loop
    regression, or the tree's fix, shows here. LI-A1's runs (the HEFT ratio,
    reported) must only scan."""

    work = _work(tmp_path_factory)
    if not (work / target["fixture"] / "artifact" / "artifact.json").is_file():
        lm.evaluate(target["fixture"], work, ap=False, identity=False)
    result = lm.limits(target["fixture"], target["oracle"], work)
    entry = ENTRIES[target["fixture"]]
    if lm.LIMIT_ROW not in entry["ids"]:
        assert all(math.isfinite(p["ratios"][0]) for p in result["points"]), result
        return
    finding = (
        json.loads(lm.DELEGATED_RECORDS.read_text())
        .get("findings", {})
        .get(target["fixture"], {})
        .get("L9")
    )
    if isinstance(finding, dict) and "expected_limit" in finding:
        lm._apply_finding(result, finding)
        assert result["status"] == "reference defect", result
        return
    assert result["status"] == "pass", result


def test_crossing_identity(tmp_path_factory, loop_native):
    """C1: the crossing of LI-26's runs (``li_manifest.crossing``): every
    crossing passes or is recorded partial with its reason (crossed values
    exactly zero at a negative-energy fermion leg), none fails or misses."""

    work = _work(tmp_path_factory)
    records = {}
    for name, entry in ENTRIES.items():
        if lm.CROSSING_ROW not in entry["ids"]:
            continue
        path = work / name / "record.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        if not record.get("complete"):
            record = lm.evaluate(name, work, ap=False, identity=False)
        records[name] = record
    judged = lm.judge_crossing(lm.crossing(work, list(ENTRIES.values())), records)
    statuses = [r["status"] for r in judged["runs"].values()]
    assert "pass" in statuses, judged
    assert not {"fail", "estimate miss", "not run"} & set(statuses), judged


@pytest.mark.parametrize("check", sorted(i for i in lm.COLOUR if i.startswith("LI-")))
def test_colour_check(check, tmp_path_factory, loop_native):
    """LI-27's f^abc part of g g > a g (``li_manifest.py fabc``): the run's
    amplitude-basis anchor asymmetry within its bounds at every fixture
    point and carrier, the control g g > z g's beyond them somewhere (the
    check sees an f^abc part); never a theorem-based zero (Q13)."""

    work = _work(tmp_path_factory)
    spec = lm.COLOUR[check]
    for name in (spec["run"], spec["control"]):
        path = work / name / "record.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        if not record.get("complete"):
            record = lm.evaluate(name, work, ap=False, identity=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1) + "\n")
    result = lm.fabc_check(work, spec)
    assert result["status"] == "pass", result
