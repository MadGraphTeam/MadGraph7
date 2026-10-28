# SPDX-License-Identifier: 0BSD
"""Integrity of the M-QCD reference campaign (m-qcd/PLAN.md §4, package MQ-REF).

Pure JSON and exact arithmetic over the committed files: the campaign table
(tools/loop_reference/mqcd_campaign.json), its reference manifest
(tests/fixtures/loop/required/mqcd_references.json) and every MadLoop
reference fixture it names. No provider, no pyAmpliCol generation.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from tools.loop_reference import required_spec

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
CAMPAIGN = ROOT / "tools" / "loop_reference" / "mqcd_campaign.json"
MANIFEST = FIXTURES / "required" / "mqcd_references.json"
LAURENT = ("finite", "single_pole", "double_pole")
# §4.2 of the plan: 76 core rows.
ROW_COUNTS = {
    "A": 22,
    "B": 7,
    "C": 9,
    "D": 23,
    "E": 4,
    "F": 4,
    "G": 2,
    "H": 3,
    "M": 1,
    "R": 1,
}

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def campaign() -> dict:
    return json.loads(CAMPAIGN.read_text())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _runs(campaign: dict) -> list[dict]:
    return campaign["runs"]


def _fixture(run_id: str) -> dict:
    return json.loads((FIXTURES / "required" / f"{run_id}.json").read_text())


def test_campaign_and_manifest_cover_every_row(campaign: dict, manifest: dict) -> None:
    rows = [row["row"] for row in campaign["rows"]]
    assert len(rows) == len(set(rows)) == sum(ROW_COUNTS.values())
    for letter, count in ROW_COUNTS.items():
        assert sum(r.startswith(letter) for r in rows) == count, letter
    assert [row["row"] for row in manifest["rows"]] == rows
    runs = {run["id"] for run in _runs(campaign)}
    for row in campaign["rows"]:
        assert set(row["runs"]) <= runs, row["row"]
        assert row["runs"] or row["existing_fixtures"], row["row"]
        assert row["restriction_code"] in campaign["restriction_codes"], row["row"]
    pending = [
        (row["row"], ref.get("run") or ref.get("fixture"))
        for row in manifest["rows"]
        for ref in row["references"] + row["existing_references"]
        if ref["status"] != "committed"
    ]
    assert not pending, pending
    for row in manifest["rows"]:
        for ref in row["existing_references"]:
            assert (FIXTURES / ref["fixture"]).is_file(), ref
        for ref in row["references"]:
            assert (FIXTURES / ref["fixture"]).is_file(), ref


def test_every_process_and_scheme_is_specified(campaign: dict) -> None:
    for run in _runs(campaign):
        spec = required_spec.load_process(required_spec.DEFAULT_SPEC, run["process"])
        scheme = spec["scheme_data"]
        assert scheme["restriction"], run["id"]
        assert spec["correction"] == "QCD"
    for code, route in campaign["restriction_codes"].items():
        assert (
            route["scheme"] in required_spec.load(required_spec.DEFAULT_SPEC)["schemes"]
        ), code


def _shell_and_conservation(
    point: list[list[str]], masses: list
) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = 80
        p = [[Decimal(x) for x in leg] for leg in point]
        scale = max(leg[0] for leg in p)
        # Relative to the event scale: a soft leg's own energy may be 1e-5 of it.
        shell = max(
            abs(
                e * e
                - x * x
                - y * y
                - z * z
                - Decimal(m.numerator) ** 2 / Decimal(m.denominator) ** 2
            )
            / (scale * scale)
            for (e, x, y, z), m in zip(p, masses, strict=True)
        )
        conservation = (
            max(
                abs(p[0][mu] + p[1][mu] - sum(leg[mu] for leg in p[2:]))
                for mu in range(4)
            )
            / scale
        )
    return shell, conservation


@pytest.mark.parametrize(
    "run_id", [r["id"] for r in json.loads(CAMPAIGN.read_text())["runs"]]
)
def test_reference_fixture_records_its_settings_points_and_sectors(
    campaign: dict, run_id: str
) -> None:
    run = {r["id"]: r for r in _runs(campaign)}[run_id]
    fixture = _fixture(run_id)
    text = json.dumps(fixture)
    # Relocatable: no absolute host paths, no old working root.
    assert not re.search(r"/shared/|/home/|pyLoopAmpliCol", text)
    assert fixture["mqcd_run"]["id"] == run_id
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, run["process"])
    if run.get("pdgs"):
        spec["pdgs"] = run["pdgs"]
    assert fixture["pdgs"] == spec["pdgs"]
    assert fixture["restriction"] == spec["scheme_data"]["restriction"]
    assert fixture["scheme"] == spec["scheme"]
    values = {**fixture["parameters"], **fixture["fixed_parameters"]}
    for key, value in run.get(
        "parameter_overrides", spec.get("parameter_overrides", {})
    ).items():
        assert values[key] == value
    # Provenance and MadLoop's accurate-mode settings.
    madloop = fixture["providers"]["madloop"]
    assert madloop["version"].startswith("MadLoop/MG5_aMC 3.7.0")
    requested = run.get(
        "madloop_accuracy", campaign["madloop_settings"]["requested_relative_accuracy"]
    )
    assert madloop["settings"]["requested_relative_accuracy"] == requested
    assert madloop["settings"]["stability_check"] == "forced on every call"
    assert madloop["generation"]["status"] == "compiled"
    # Points: on the scheme's mass shells, momentum conserving.
    masses = required_spec.masses(spec)
    assert len(fixture["rows"]) == len(fixture["points"]) == len(fixture["mu_r"])
    for point in fixture["points"]:
        shell, conservation = _shell_and_conservation(point, masses)
        assert shell < Decimal("1e-25") and conservation < Decimal("1e-25")
    # Every squared split order with MadLoop's own accuracy estimate.
    orders = {(o["QCD"], o["QED"]) for o in fixture["born_orders_all"]}
    step = {
        k: fixture["loop_orders"][k] - fixture["born_orders"][k] for k in ("QCD", "QED")
    }
    for row in fixture["rows"]:
        sectors = row["sectors"]
        born = {(s["QCD"], s["QED"]) for s in sectors["born"]}
        assert {(2 * q, 2 * e) for q, e in orders} <= born
        for sector in sectors["loop"]:
            assert sector["accuracy"] >= 0.0
            assert all(isinstance(sector[k], float) for k in LAURENT)
            if sector["born_sector"] is not None:
                assert (
                    sector["born_sector"]["QCD"],
                    sector["born_sector"]["QED"],
                ) in born
                assert sector["QCD"] - sector["born_sector"]["QCD"] == step["QCD"]
        primary = (
            2 * fixture["born_orders"]["QCD"] + step["QCD"],
            2 * fixture["born_orders"]["QED"] + step["QED"],
        )
        loop = {(s["QCD"], s["QED"]): s for s in sectors["loop"]}
        assert loop[primary]["finite"] == row["providers"]["madloop"]["finite"]
        assert row["madloop_accuracy"] == loop[primary]["accuracy"]


@pytest.mark.parametrize(
    "run_id",
    [r["id"] for r in json.loads(CAMPAIGN.read_text())["runs"] if r.get("helicities")],
)
def test_per_helicity_rows_sum_to_the_summed_value(campaign: dict, run_id: str) -> None:
    fixture = _fixture(run_id)
    per = fixture["per_helicity"]
    spins = 1
    for pdg in fixture["pdgs"][:2]:
        spins *= 2 if required_spec.particle(pdg)["spin"] in ("1/2", "1") else 1
    assert len(per["rows"]) == len(fixture["rows"])
    for summed, helicities in zip(fixture["rows"], per["rows"], strict=True):
        assert len(helicities) == len(per["helicities"])
        for key in ("born", *LAURENT):
            total = sum(h[key] for h in helicities)
            assert (
                abs(total - spins * summed[key])
                <= 1e-12 * spins * abs(summed[key]) + 1e-300
            ), key


def test_mixed_order_run_prints_every_squared_split_order() -> None:
    fixture = _fixture("uux_ttx_mixed")
    for row in fixture["rows"]:
        assert len(row["sectors"]["born"]) >= 3
        assert len(row["sectors"]["loop"]) >= 3


def test_multiparticle_runs_cover_madloops_subprocess_directories(
    campaign: dict,
) -> None:
    runs = [r for r in _runs(campaign) if r.get("subprocess")]
    generation = _fixture(runs[0]["id"])["providers"]["madloop"]["generation"]
    assert sorted(r["subprocess"] for r in runs) == generation["subprocess_directories"]
    assert len(generation["subprocesses"]) >= len(runs)


def test_madloop_born_split_orders_read_from_the_born_matrix(tmp_path: Path) -> None:
    from tools.loop_reference import references

    process = tmp_path / "P0_x"
    process.mkdir()
    (process / "born_matrix.f").write_text(
        "      INTEGER FUNCTION ML5_7_SOINDEX_FOR_SQUARED_ORDERS(ORDERS)\n"
        "      DATA (SQSPLITORDERS(  1,I),I=  1,  2) /    4,    0/\n"
        "      DATA (SQSPLITORDERS(  2,I),I=  1,  2) /    2,\n"
        "     $    2/\n"
        "      COMMON/ML5_7_SQPLITORDERS/SQPLITORDERS\n"
        "      END\n"
    )
    assert references.madloop_born_split_orders(process) == {1: (4, 0), 2: (2, 2)}
