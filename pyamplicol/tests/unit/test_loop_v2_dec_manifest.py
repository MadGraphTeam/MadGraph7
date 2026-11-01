# SPDX-License-Identifier: 0BSD
"""The decay campaign manifest (M-FINAL MF-DEC-2, m-final/PLAN.md §2.2):
``tests/fixtures/loop/dec/manifest.json`` is what ``tools/loop_benchmark/
dec_manifest.py`` writes, holds the plan's rows XX01-XX08 against MF-DEC-1's
MadLoop references (``tests/fixtures/loop/required/dec/``), and its nightly
file jobs cover the tier. Fast: no generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.loop_benchmark import dec_manifest, mqcd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "fixtures" / "loop" / "dec" / "manifest.json"
PLAN_ROWS = {
    "XX01": ("t > b w+", ("QCD",), "loop_sm"),
    "XX02": ("t > b w+", ("QCD", "QED"), "loop_qcd_qed_sm_Gmu"),
    "XX03": ("z > e- e+", ("QED",), "loop_qcd_qed_sm"),
    "XX04": ("z > u u~", ("QCD", "QED"), "loop_qcd_qed_sm"),
    "XX05": ("w+ > e+ ve", ("QED",), "loop_qcd_qed_sm"),
    "XX06": ("h > b b~", ("QCD",), "loop_sm"),
    "XX07": ("t > b w+ g", ("QCD",), "loop_sm"),
    "XX08": ("z > u u~ g", ("QCD", "QED"), "loop_qcd_qed_sm"),
}


@pytest.fixture(scope="module")
def manifest():
    return mqcd.load_manifest(MANIFEST)


@pytest.mark.loop_fast
def test_manifest_is_the_generator_s(manifest):
    assert json.loads(json.dumps(dec_manifest.manifest())) == manifest


@pytest.mark.loop_fast
def test_rows_are_the_plan_rows(manifest):
    rows = mqcd.rows(manifest)
    core = {name: row for name, row in rows.items() if row.base is None}
    assert {
        name: (row.process, row.corrections, row.model) for name, row in core.items()
    } == PLAN_ROWS
    for name, row in rows.items():
        assert row.active and row.reference["format"] == "required", name
        code = manifest["restriction_codes"][row.restriction_code]
        assert (row.model, row.restriction) == (code["model"], code["restriction"])
        # The decaying particle's own width is zero in every row (user
        # ruling 2026-10-31 ~11:00): every width of the row is fixed to 0.
        widths = {k: v for k, v in row.fixed_parameters.items() if k.startswith("W")}
        assert widths and set(widths.values()) == {"0"}, name
        # Every sector is predicted (rule 4): joint requests without caps.
        assert set(row.predicted_sectors) == {
            mqcd.sector_key(s) for s in next(
                d for d in manifest["rows"] if d["id"] == (row.base or name)
            )["sectors"]
        } and not row.unpredicted_sectors, name
        if "QED" in row.corrections:
            assert 0.0075 < row.alphas["QED"] < 0.0078, name
    assert {n for n, r in core.items() if r.tier == "default"} == {"XX01"}
    assert {n for n, r in rows.items() if r.route == "public"} == {
        "XX01@public",
        "XX07@public",
    }


@pytest.mark.loop_fast
def test_references_load_at_their_points(manifest):
    for name, row in mqcd.rows(manifest).items():
        reference = mqcd.load_reference(row)
        indices = {i for points in row.points.values() for i in points}
        assert indices <= set(range(len(reference.points))), name
        assert len(reference.points[0]) == len(row.process.split()) - 1, name


@pytest.mark.loop_fast
def test_nightly_jobs_cover_the_tier_within_the_budget(manifest):
    ids = mqcd.nightly_ids(manifest)
    jobs = manifest["nightly_jobs"]["jobs"]
    assert sorted(n for job in jobs for n in job["rows"]) == sorted(ids)
    budget = manifest["nightly_jobs"]["budget_seconds"]
    assert budget <= 1800 and all(job["seconds"] <= budget for job in jobs)
    found = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests" / "unit").glob("test_loop_v2_dec_nightly*.py")
    }
    assert found == {job["file"] for job in jobs}
