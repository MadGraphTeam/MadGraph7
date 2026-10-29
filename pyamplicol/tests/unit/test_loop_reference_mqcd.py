# SPDX-License-Identifier: 0BSD
"""Integrity of the M-QCD reference campaign (m-qcd/PLAN.md §4, package MQ-REF).

Pure JSON and exact arithmetic over the committed files: the campaign table
(tools/loop_reference/mqcd_campaign.json), its reference manifest
(tests/fixtures/loop/required/mqcd_references.json) and every MadLoop
reference fixture it names. No provider, no pyAmpliCol generation.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_reference import required_spec

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
CAMPAIGN = ROOT / "tools" / "loop_reference" / "mqcd_campaign.json"
MANIFEST = FIXTURES / "required" / "mqcd_references.json"
LAURENT = ("finite", "single_pole", "double_pole")
# §4.2 of the plan: 76 core rows, plus D28 and D29 (q q~ > W+ W- g, mq-ewjoin).
ROW_COUNTS = {
    "A": 22,
    "B": 7,
    "C": 9,
    "D": 25,
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
    return json.loads((FIXTURES / "required" / "mqcd" / f"{run_id}.json").read_text())


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
    # Provenance and MadLoop's evaluation settings.
    madloop = fixture["providers"]["madloop"]
    settings = madloop["settings"]
    assert madloop["version"].startswith("MadLoop/MG5_aMC 3.7.0")
    assert madloop["generation"]["status"] == "compiled"
    assert settings["stability_check"] == "forced on every call"
    evaluation = campaign["madloop_settings"]["evaluation"]
    statuses = [row["madloop_status"] for row in fixture["rows"]]
    if run.get("madloop_evaluation", {}).get("mode") == "mlstabthres":
        assert run["madloop_evaluation"]["reason"]
        requested = run.get(
            "madloop_accuracy",
            campaign["madloop_settings"]["requested_relative_accuracy"],
        )
        assert settings["requested_relative_accuracy"] == requested
        assert "quadruple_precision_stage" not in settings
    else:
        # Every point in MadLoop's quadruple-precision stage, every helicity
        # configuration computed, no filter inherited from an earlier run.
        assert (
            settings["requested_relative_accuracy"]
            == (evaluation["requested_relative_accuracy"])
        )
        assert settings["quadruple_precision_stage"]
        assert settings["quadruple_precision_required"] is True
        assert settings["MadLoopParams_overrides"] == evaluation["madloop_params"]
        for key, value in evaluation["madloop_params"].items():
            assert settings["MadLoopParams"][key] == value
        assert all(status // 100 in (3, 4) for status in statuses), statuses
        # A point whose quadruple-precision stability test failed (4xx) is no
        # reference unless declared: status 'none' or a rule 1 outlier.
        for status, row in zip(statuses, fixture["rows"], strict=True):
            if status // 100 == 4:
                record = row["reference"]
                assert record["status"] == "none" or record["outlier"], status
        # The batch-2 evaluation it supersedes, where the run had one.
        secondary = madloop.get("secondary_runs", {}).get("mlstabthres")
        if secondary is not None:
            deviation = secondary["relative_deviation_from_reference_run"]
            assert len(deviation) == len(statuses)
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


HEL_RELATIVE_LIMIT = 1e-13


def _strict(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


@pytest.mark.parametrize(
    "run_id",
    [r["id"] for r in json.loads(CAMPAIGN.read_text())["runs"] if r.get("helicities")],
)
def test_per_helicity_entries_carry_reference_records(run_id: str) -> None:
    """Quadruple-precision helicity entries take their Born from the
    helicity-diagonal IR double pole. Only a vanishing helicity (<= 1e-12 of
    the summed Born) is an absolute entry (§1.4: absolute only for what
    vanishes identically); every other entry is strictly relative, and every
    committed entry is a reference: at points that are not MadLoop's own
    quadruple-precision points a printed binary64 Born more than 1e-13 off the
    quadruple-precision one (a point-sensitive helicity) would not be one."""
    fixture = _fixture(run_id)
    per = fixture["per_helicity"]
    factor = fixture["universal_double_pole_factor"]
    qp_point = bool(fixture["point_construction"].get("madloop_qp_point"))
    assert per["settings"]["MadLoopParams_overrides"] == {"HelicityFilterLevel": "0"}
    assert per["settings"]["quadruple_precision_stage"]
    for entries in per["rows"]:
        summed = sum(abs(e["born"]) for e in entries)
        for entry in entries:
            record = entry["reference"]
            assert entry["status"] // 100 == 3, entry["helicity"]
            mismatch = record["born_double_pole_relative_mismatch"]
            if factor:
                assert (
                    record["born_source"] == "madloop_double_pole_over_universal_factor"
                )
                assert entry["born"] == entry["double_pole"] / factor
                assert mismatch == _strict(entry["born_binary64_input"], entry["born"])
            else:
                assert record["born_source"] == "madloop" and mismatch is None
            vanishing = abs(entry["born"]) <= 1e-12 * summed
            assert record["comparison"] == ("absolute" if vanishing else "relative")
            assert record["absolute_scale"] == (summed if vanishing else None)
            assert bool(record["reason"]) == vanishing
            assert record["status"] == "ok", entry["helicity"]
            if not (qp_point or vanishing):
                assert mismatch is None or mismatch <= HEL_RELATIVE_LIMIT


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


# --------------------------------------------------------------------------
# Physics consistency of the committed references (pure JSON).

UNIVERSAL_POLE_TOLERANCE = 1e-10
# The worst residual over the campaign is 1.0e-13 of the scale below
# (u u~ > w+ w- finite): ten times headroom, and a reference error at rule 1's
# 1e-12 shows.
RG_TOLERANCE = 1e-12
RUN_IDS = [r["id"] for r in json.loads(CAMPAIGN.read_text())["runs"]]
# At a regular point MadLoop's printed Born (double precision at the binary64
# input) and the Born its quadruple-precision double pole implies are the same
# Born up to the binary64 rounding of the point: <= 1.1e-13 over the campaign
# (u u~ > w+ w-: gauge cancellations of the slightly off-shell binary64 W
# momenta), while a loop defect shows above it (C01 point 1 before
# HelicityFilterLevel 0: 4.8e-13). An ok point above the floor (or above ten
# times MadLoop's own accuracy, where that is larger) must be a declared
# outlier (reference.outlier).
REGULAR_IR_FLOOR = 2e-13
# Limit points (madloop_qp_point): the printed Born's binary64 condition, bounded
# by 1e3 binary64 epsilons times the family's observed sensitivity (soft E/sqrt(s)
# = v: 1/v; threshold beta: 1/beta^2; collinear theta: 1/theta^3; the
# radiation-zero family compares absolutely against its regular neighbour).
EPSILON = 2.0**-53
FAMILY_SENSITIVITY = {
    "soft": lambda v: 1 / v,
    "threshold": lambda v: 1 / v**2,
    "collinear": lambda v: 1 / v**3,
}


def _reference_born(sectors: dict) -> dict:
    return {(s["QCD"], s["QED"]): s["born"] for s in sectors["born"]}


def _printed(row: dict) -> dict:
    """The MadLoop row as printed (binary64 Borns where the reference Born is
    the double pole over the factor), the input of sector_pole_error."""
    sectors = json.loads(json.dumps(row["sectors"]))
    for born in sectors["born"]:
        born["born"] = born.get("born_binary64_input", born["born"])
    return {
        "born": row.get("born_madloop_binary64_input", row["born"]),
        "double_pole": row["providers"]["madloop"]["double_pole"],
        "sectors": sectors,
    }


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_every_loop_sector_has_the_universal_double_pole(run_id: str) -> None:
    """D_s = factor * Born(born sector of s) per loop squared split order at
    every reference point (a garbage Born sector, e.g. MG5's uninitialised
    JAMP, violates it at O(1)); the recorded MadLoop mismatch is reproduced
    and bounded: at regular ok points by REGULAR_IR_FLOOR (named outliers
    excepted), at limit points by their family's binary64 condition. At
    madloop_qp_point points the reference Born of every sector with a loop
    partner IS that sector's double pole over the factor, so the per-sector
    relation holds there by construction; the falsifiable quantity is the
    printed binary64 Born's mismatch."""
    from tools.loop_reference import references

    fixture = _fixture(run_id)
    factor = fixture["universal_double_pole_factor"]
    coupling = float(
        Decimal({**fixture["parameters"], **fixture["fixed_parameters"]}["aS"])
    )
    construction = fixture["point_construction"]
    family = construction.get("family")
    values = construction.get("family_values") or []
    for index, row in enumerate(fixture["rows"]):
        reference = row["reference"]
        sectors = row["sectors"]
        born = _reference_born(sectors)
        requested = (
            2 * fixture["born_orders"]["QCD"],
            2 * fixture["born_orders"]["QED"],
        )
        assert row["born"] == born[requested]
        assert abs(sum(born.values()) - sectors["total"]["born"]) <= 1e-14 * max(
            abs(b) for b in born.values()
        )
        if reference["status"] != "ok":
            assert reference["reason"]
            continue
        largest = max(abs(s["double_pole"]) for s in sectors["loop"])
        absolute = abs(factor) * (reference["absolute_scale"] or 0.0)
        for sector in sectors["loop"]:
            partner = sector["born_sector"]
            expected = factor * (
                born[(partner["QCD"], partner["QED"])] if partner else 0.0
            )
            scale = max(abs(sector["double_pole"]), abs(expected), largest, absolute)
            assert (
                abs(sector["double_pole"] - expected)
                <= UNIVERSAL_POLE_TOLERANCE * scale
            ), (
                index,
                sector["QCD"],
                sector["QED"],
            )
        if factor:
            printed = row.get("born_madloop_binary64_input", row["born"])
            implied = row["providers"]["madloop"]["double_pole"] / factor
            floor = reference["absolute_scale"] or 0.0
            mismatch = abs(printed - implied) / max(abs(printed), abs(implied), floor)
            assert mismatch == pytest.approx(
                reference["born_double_pole_relative_mismatch"], rel=1e-6, abs=1e-300
            )
        # The recorded violation is the one of every sector's printed values.
        error = references.sector_pole_error(
            _printed(row), factor, coupling, reference["absolute_scale"]
        )
        assert error == reference["max_sector_double_pole_relative_error"], index
        if construction.get("madloop_qp_point"):
            if family in FAMILY_SENSITIVITY:
                value = float(Fraction(values[index]))
                bound = 1e3 * EPSILON * FAMILY_SENSITIVITY[family](value)
            else:
                bound = 1e3 * EPSILON
            assert reference["born_double_pole_relative_mismatch"] <= bound, (
                index,
                bound,
            )
        elif not reference["outlier"]:
            accuracy = max(s["accuracy"] for s in sectors["loop"])
            assert error <= max(10 * accuracy, REGULAR_IR_FLOOR), (index, error)


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_scale_variation_obeys_the_renormalisation_group(run_id: str) -> None:
    """The mu_R = mu_0/2 copy of point 1, per loop sector (fixed alpha_s value,
    MSbar alpha_s renormalisation with the active flavours, heavy quarks
    decoupled): D' = D, S' = S + D L, F' = F + (S + p beta0 alpha_s/(2 pi) B) L
    + D L^2/2 with L = ln(mu'^2/mu^2) and p the Born sector's alpha_s power."""
    fixture = _fixture(run_id)
    variation = fixture["point_construction"].get("scale_variation")
    if variation is None:
        assert fixture["mqcd_run"]["points"].get("half_scale_point") is False
        return
    base, varied = variation["copy_of_row"], variation["row"]
    assert (base, varied) == (variation["copy_of_point"] - 1, variation["point"] - 1)
    assert fixture["points"][base] == fixture["points"][varied]
    rows = fixture["rows"]
    assert (
        rows[base]["reference"]["status"] == rows[varied]["reference"]["status"] == "ok"
    )
    ratio = Fraction(fixture["mu_r"][varied]) / Fraction(fixture["mu_r"][base])
    assert ratio == Fraction(1, 2)
    log = math.log(float(ratio) ** 2)
    alpha_s = float(
        Fraction({**fixture["parameters"], **fixture["fixed_parameters"]}["aS"])
    )
    beta0 = (33 - 2 * fixture["active_flavours"]) / 6
    born = _reference_born(rows[base]["sectors"])
    varied_loop = {(s["QCD"], s["QED"]): s for s in rows[varied]["sectors"]["loop"]}
    for sector in rows[base]["sectors"]["loop"]:
        partner = sector["born_sector"]
        b = born[(partner["QCD"], partner["QED"])] if partner else 0.0
        power = partner["QCD"] / 2 if partner else 0.0
        d, s, f = sector["double_pole"], sector["single_pole"], sector["finite"]
        expected = {
            "double_pole": d,
            "single_pole": s + d * log,
            "finite": f
            + (s + power * beta0 * alpha_s / (2 * math.pi) * b) * log
            + d * log * log / 2,
        }
        scale = max(abs(d), abs(s), abs(f), abs(b) * alpha_s / (2 * math.pi))
        other = varied_loop[(sector["QCD"], sector["QED"])]
        for key, value in expected.items():
            assert abs(other[key] - value) <= RG_TOLERANCE * scale, (key, sector["QCD"])


@pytest.mark.parametrize(
    "run_id",
    [
        r["id"]
        for r in json.loads(CAMPAIGN.read_text())["runs"]
        if r["points"]["generator"] == "supplied"
    ],
)
def test_supplied_points_are_published_at_thirty_digits(run_id: str) -> None:
    """Every component on the common quantum of 30 significant digits of
    sqrt(s); momentum conservation exact (or, for MadLoop's projected
    quadruple-precision points, to the quantum); the recorded defects are those
    of the published decimals."""
    fixture = _fixture(run_id)
    construction = fixture["point_construction"]
    precision = construction["precision"]
    assert precision["printed_digits"] == 30
    masses = required_spec.masses(
        required_spec.load_process(
            required_spec.DEFAULT_SPEC, fixture["mqcd_run"]["process"]
        )
    )
    projected = bool(construction.get("madloop_qp_point"))
    for index, point in enumerate(fixture["points"]):
        with localcontext() as context:
            context.prec = 120
            quantum = Decimal(1).scaleb(
                Decimal(construction["sqrt_s"][index]).adjusted() - 29
            )
            p = [[Decimal(x) for x in leg] for leg in point]
            assert all(x % quantum == 0 for leg in p for x in leg), index
            scale = max(leg[0] for leg in p)
            defect = max(
                abs(p[0][mu] + p[1][mu] - sum(leg[mu] for leg in p[2:]))
                for mu in range(4)
            )
            if projected:
                assert defect <= len(p) * quantum, index
            else:
                assert defect == 0, index
            shell = max(
                abs(
                    e * e
                    - x * x
                    - y * y
                    - z * z
                    - Decimal(m.numerator) ** 2 / Decimal(m.denominator) ** 2
                )
                / (e * e)
                for (e, x, y, z), m in zip(p, masses, strict=True)
            )
        recorded_shell = float(precision["max_relative_mass_shell_defect"][index])
        recorded_conservation = float(
            precision["max_relative_momentum_conservation_defect"][index]
        )
        assert float(shell) == pytest.approx(recorded_shell, rel=1e-2, abs=1e-300), (
            index
        )
        assert float(defect / scale) == pytest.approx(
            recorded_conservation, rel=1e-2, abs=1e-300
        ), index


def test_point_references_are_recorded_and_listed_in_the_manifest(
    campaign: dict, manifest: dict
) -> None:
    by_run = {ref["run"]: ref for row in manifest["rows"] for ref in row["references"]}
    assert set(by_run) == set(RUN_IDS)
    for run in _runs(campaign):
        fixture = _fixture(run["id"])
        records = [row["reference"] for row in fixture["rows"]]
        listed = by_run[run["id"]]
        declared = run.get("point_status", {})
        for index, record in enumerate(records):
            status = declared.get(str(index), {})
            assert record["status"] == status.get("status", "ok"), (run["id"], index)
            assert record["comparison"] == status.get("comparison", "relative")
            assert record["absolute_scale_row"] == status.get("scale_row")
            assert record["outlier"] == status.get("outlier")
            if record["comparison"] == "absolute":
                scale_row = fixture["rows"][record["absolute_scale_row"]]
                assert scale_row["reference"]["comparison"] == "relative"
                assert record["absolute_scale"] == abs(scale_row["born"])
                assert (
                    abs(fixture["rows"][index]["born"])
                    <= 1e-12 * record["absolute_scale"]
                )
        # The manifest names every rule 1 outlier and every binding row.
        assert listed["outliers"] == {
            str(i): r["outlier"] for i, r in enumerate(records) if r["outlier"]
        }
        assert listed["binding_rows"] == [
            i for i, r in enumerate(records) if r["status"] == "ok"
        ]
        codes = [row["madloop_status"] for row in fixture["rows"]]
        assert listed["madloop_return_codes"] == codes
        assert listed["evaluation_precision"] == [
            {3: "quadruple", 4: "quadruple (unstable)"}.get(c // 100, "double")
            for c in codes
        ]
        projected = bool(run["points"].get("madloop_qp_point"))
        assert bool(fixture["point_construction"].get("madloop_qp_point")) == projected
        assert listed["madloop_qp_point"] == projected
        for row in fixture["rows"]:
            source = row["reference"]["born_source"]
            if projected:
                assert source == "madloop_double_pole_over_universal_factor"
                assert row["born"] == (
                    row["providers"]["madloop"]["double_pole"]
                    / fixture["universal_double_pole_factor"]
                )
                assert (
                    row["born_madloop_binary64_input"]
                    == row["providers"]["madloop"]["born"]
                )
            else:
                assert source == "madloop"
                assert row["born"] == row["providers"]["madloop"]["born"]
        if run.get("second_providers"):
            assert run["second_provider_trigger"], run["id"]
            assert listed["second_provider_trigger"] == run["second_provider_trigger"]
        # Each second provider's outcome, per point and coefficient.
        outcome = listed["second_provider_outcome"]
        assert set(outcome) == set(run.get("second_providers", []))
        for provider, per_point in outcome.items():
            assert provider in fixture["providers"], (run["id"], provider)
            for row, recorded in zip(fixture["rows"], per_point, strict=True):
                theirs = row["providers"][provider]
                ours = row["providers"]["madloop"]
                assert recorded == {
                    k: _strict(ours[k], theirs[k]) for k in ("born", *LAURENT)
                }
        superseded = fixture["providers"]["madloop"].get("secondary_runs", {})
        if "mlstabthres" in superseded:
            assert run.get("madloop_evaluation", {}).get("mode") != "mlstabthres"
            assert superseded["mlstabthres"]["settings"][
                "requested_relative_accuracy"
            ] in (1e-13, 1e-20)
        assert (
            fixture["tolerance_scope"] == manifest["conventions"]["fixture_tolerance"]
        )


def _values(fixture: dict, index: int) -> dict:
    row = fixture["rows"][index]
    if "born" in row:
        return row
    return row["providers"][fixture.get("primary_provider", "madloop")]


def test_every_point_has_one_binding_reference(campaign: dict, manifest: dict) -> None:
    """An existing reference's point is superseded exactly where a campaign
    run of the row re-evaluates the same exact point and mu_R in MadLoop's
    quadruple-precision stage with an ok, non-outlier record (deviation
    recorded); every other point stays binding, and binding_references lists
    each reference's binding rows once."""
    runs = {run["id"]: run for run in _runs(campaign)}
    rows = {row["row"]: row for row in campaign["rows"]}
    superseded_any = False
    for entry in manifest["rows"]:
        row = rows[entry["row"]]
        expected = []
        for ref in entry["existing_references"]:
            fixture = json.loads((FIXTURES / ref["fixture"]).read_text())
            if ref.get("kind") != "provider reference":
                assert "point_binding" not in ref
                expected.append({"fixture": ref["fixture"], "rows": None})
                continue
            assert ref["points"] == len(fixture["points"]) == len(ref["point_binding"])
            codes = [
                r.get("madloop_status", r["providers"].get("madloop", {}).get("status"))
                for r in fixture["rows"]
            ]
            assert ref["madloop_return_codes"] == codes
            reruns = [
                r
                for r in row["runs"]
                if runs[r]["points"].get("generator") == "fixture"
                and runs[r]["points"]["fixture"] == ref["fixture"]
            ]
            for index, binding in enumerate(ref["point_binding"]):
                candidates = []
                for rerun in reruns:
                    other = _fixture(rerun)
                    point = [[str(x) for x in leg] for leg in fixture["points"][index]]
                    assert point == other["points"][index]
                    assert Fraction(str(fixture["mu_r"][index])) == Fraction(
                        other["mu_r"][index]
                    )
                    record = other["rows"][index]["reference"]
                    code = other["rows"][index]["madloop_status"]
                    if (
                        record["status"] == "ok"
                        and code // 100 == 3
                        and not record["outlier"]
                    ):
                        candidates.append((rerun, other))
                if not candidates:
                    assert binding == {"binding": True}, (ref["fixture"], index)
                    continue
                superseded_any = True
                rerun, other = candidates[0]
                assert binding["binding"] is False
                assert binding["superseded_by"] == {
                    "fixture": f"required/mqcd/{rerun}.json",
                    "row": index,
                }
                old, new = _values(fixture, index), other["rows"][index]
                assert binding["strict_relative_deviation"] == {
                    k: _strict(old[k], new[k]) for k in ("born", *LAURENT)
                }
                assert ref["provider_accuracy_per_sector"][index] is None
            chosen = [i for i, b in enumerate(ref["point_binding"]) if b["binding"]]
            if chosen:
                expected.append({"fixture": ref["fixture"], "rows": chosen})
        for ref in entry["references"]:
            if ref["binding_rows"]:
                expected.append(
                    {"fixture": ref["fixture"], "rows": ref["binding_rows"]}
                )
        assert entry["binding_references"] == expected, entry["row"]
        if row.get("helicity_reference"):
            assert entry["helicity_reference"] == (
                f"required/mqcd/{row['helicity_reference']}.json"
            )
            assert _fixture(row["helicity_reference"])["per_helicity"]
        else:
            assert "helicity_reference" not in entry
            assert not any(runs[r].get("helicities") for r in row["runs"])
    assert superseded_any
    # A06 point 1: the double-precision single pole of the existing fixture is
    # 1.2e-12 off; the quadruple-precision rerun binds there.
    a06 = next(e for e in manifest["rows"] if e["row"] == "A06")
    binding = a06["existing_references"][0]["point_binding"][1]
    assert binding["superseded_by"]["fixture"] == "required/mqcd/uux_uux_sectors.json"
    assert binding["strict_relative_deviation"]["single_pole"] > 1e-12


def test_no_required_fixture_names_a_host_path() -> None:
    offenders = [
        str(path.relative_to(FIXTURES))
        for path in sorted((FIXTURES / "required").rglob("*.json"))
        if re.search(r"/shared/|/home/|pyLoopAmpliCol", path.read_text())
    ]
    assert not offenders, offenders


# --------------------------------------------------------------------------
# Toolkit guards (synthetic inputs).


def _madloop_row(born_sectors: dict, double_poles: dict, factor: float) -> dict:
    loop = [
        {
            "QCD": q + 2,
            "QED": e,
            "finite": 0.0,
            "single_pole": 0.0,
            "double_pole": double_poles.get((q, e), factor * b),
            "accuracy": 0.0,
            "born_sector": {"QCD": q, "QED": e},
        }
        for (q, e), b in born_sectors.items()
    ]
    total_born = sum(born_sectors.values())
    total_pole = sum(s["double_pole"] for s in loop)
    return {
        "born": total_born,
        "finite": 0.0,
        "single_pole": 0.0,
        "double_pole": total_pole,
        "status": 216,
        "sectors": {
            "born": [
                {"QCD": q, "QED": e, "born": b} for (q, e), b in born_sectors.items()
            ],
            "loop": loop,
            "total": {"born": total_born, "double_pole": total_pole},
        },
    }


def test_merge_refuses_a_garbage_born_sector_unless_the_point_is_declared() -> None:
    from tools.loop_reference import references

    factor, coupling = -0.05, 0.118
    good = _madloop_row({(4, 0): 0.69, (0, 4): 0.0096}, {}, factor)
    garbage = _madloop_row(
        {(4, 0): 0.67, (0, 4): 3.58e9}, {(0, 4): factor * 0.0096}, factor
    )
    rows = [{"born": r["born"]} for r in (good, garbage)]
    with pytest.raises(SystemExit, match="point 2"):
        references.madloop_references(
            rows, [good, garbage], "madloop", factor, coupling, {}, False, 2e-8
        )
    declared = references.point_statuses(
        json.dumps({"1": {"status": "none", "reason": "synthetic"}}), 2
    )
    rows = [{"born": r["born"]} for r in (good, garbage)]
    references.madloop_references(
        rows, [good, garbage], "madloop", factor, coupling, declared, False, 2e-8
    )
    assert rows[0]["reference"]["max_sector_double_pole_relative_error"] < 1e-15
    assert rows[1]["reference"]["status"] == "none"
    with pytest.raises(SystemExit, match="reason"):
        references.point_statuses(json.dumps({"0": {"status": "none"}}), 2)
    with pytest.raises(SystemExit, match="scale_row"):
        references.point_statuses(
            json.dumps({"0": {"comparison": "absolute", "reason": "x"}}), 2
        )


def test_parse_rows_refuses_non_finite_sector_values() -> None:
    from tools.loop_reference import references

    stdout = (
        "SPLIT_ORDER_INDICES 0 0 1 1\n"
        "LOOP_SO 1 6 0\n"
        "ROW 1 216 0.1 1.0 2.0 3.0 -0.05 1e-15 1.0 2.0 3.0 -0.05\n"
        "SECTORS 1 x 1.0 2.0 3.0 -0.05 inf 2.0 3.0 -0.05 1e-15 1e-15\n"
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        references.parse_rows(stdout, "madloop", 1, {"QCD": 2, "QED": 0}, {1: (4, 0)})
    finite = stdout.replace("inf", "1.0")
    rows, _ = references.parse_rows(
        finite, "madloop", 1, {"QCD": 2, "QED": 0}, {1: (4, 0)}
    )
    assert rows[0]["sectors"]["born"][0]["born"] == 1.0


BORN_MATRIX = """      SUBROUTINE ML5_0_MATRIX(P,NHEL,IC,RES)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=5, NCOLOR=2)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=2, NSQAMPSO=3)
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)
C     ----------
C     BEGIN CODE
C     ----------
      CALL IXXXXX(P(0,1),ZERO,NHEL(1),+1*IC(1),W(1,1))
      JAMP(1,1) = AMP(1)
      JAMP(2,1) = AMP(1)
      JAMP(1,2) = AMP(2)
{last}
      RES(:) = 0.D0
      END
"""


def test_born_matrix_repair_zeroes_only_incompletely_assigned_jamps() -> None:
    from tools.loop_reference import references

    complete = BORN_MATRIX.format(last="      JAMP(2,2) = AMP(3)")
    assert references.fix_unassigned_jamps(complete) == (complete, [])
    text, repairs = references.fix_unassigned_jamps(BORN_MATRIX.format(last=""))
    assert repairs == [
        "born_matrix.f: zero-initialised JAMP in ML5_0_MATRIX "
        "(1 entries never assigned: [(2, 2)])"
    ]
    assert "C     ----------\n      JAMP(:,:) = (0D0,0D0)\n      CALL IXXXXX" in text


def test_supplied_points_refuse_unreachable_configurations() -> None:
    from fractions import Fraction as F

    from tools.loop_reference import supplied_points

    massive = [F(0), F(0), F(173), F(173)]
    for family in ("soft", "collinear"):
        with pytest.raises(ValueError, match="massless"):
            supplied_points.generate(massive, family, values=[F(1, 1000)])
    with pytest.raises(ValueError, match="distinct sqrt"):
        supplied_points.generate(massive, "regular", count=11)
    with pytest.raises(ValueError, match="no room"):
        supplied_points.generate([F(0), F(0), F(173), F(0)], "soft", values=[F(1, 2)])


def test_parse_rows_without_sectors_reads_the_new_driver_output() -> None:
    """The G1 benchmark's MadLoop runs call parse_rows without step/born_so on
    the driver's full output (LOOP_SO and SECTORS lines included)."""
    from tools.loop_reference import references

    stdout = (
        "SPLIT_ORDER_INDICES 1 1 1 1\n"
        "LOOP_SO 1 6 0\n"
        "INITIAL_CALL_SECONDS 0.1 1 216\n"
        "ROW 1 216 0.1 1.0 2.0 3.0 -0.05 1e-15 1.0 2.0 3.0 -0.05\n"
        "SECTORS 1 216 1.0 2.0 3.0 -0.05 1.0 2.0 3.0 -0.05 1e-15 1e-15\n"
    )
    rows, extra = references.parse_rows(stdout, "madloop", 1)
    assert rows[0]["finite"] == 2.0 and rows[0]["status"] == 216
    assert "sectors" not in rows[0] and "squared_split_orders" not in extra
    assert extra["initialization_calls"] == 1


LOOP_MATRIX = """        ELSEIF(.NOT.DOING_QP_EVALS)THEN
          CALL ML5_3_COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)
          IF(FPE_IN_DP_REDUCTION) THEN
            ACC(0)=1.0D0
          ENDIF
          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN
            I_LIB=I_LIB+1
            IF((I_LIB.GT.NLOOPLIB.OR.MLREDUCTIONLIB(I_LIB).EQ.0)
"""


def test_quadruple_stage_bypasses_only_the_double_precision_acceptance() -> None:
    from tools.loop_reference import references

    qp_test = (
        "          IF(MAXVAL(ACC).GE.MLSTABTHRES) THEN\n"
        "            I_QP_LIB=I_QP_LIB+1\n"
    )
    text = qp_test + LOOP_MATRIX
    forced = references.force_quadruple_stage(text, "ML5_3_")
    assert forced.startswith(qp_test)  # the quadruple-precision test is MadLoop's own
    assert "          IF(.TRUE.) THEN\n            I_LIB=NLOOPLIB+1\n" in forced
    assert forced.count("MAXVAL(ACC).GE.MLSTABTHRES") == 1
    with pytest.raises(RuntimeError, match="branch"):
        references.force_quadruple_stage(text, "ML5_0_")
    with pytest.raises(RuntimeError, match="acceptance"):
        references.force_quadruple_stage(
            text.replace("I_LIB=I_LIB+1", "I_LIB=1"), "ML5_3_"
        )


def test_per_helicity_records_compare_absolutely_only_vanishing_helicities() -> None:
    from tools.loop_reference import references

    factor = -0.25
    regular = {"status": 328, "born": 1.0, "double_pole": factor * (1.0 + 1e-16)}
    suppressed = {"status": 328, "born": 4e-11, "double_pole": factor * 4.0001e-11}
    vanishing = {"status": 328, "born": 1e-30, "double_pole": 0.0}
    double = {"status": 216, "born": 0.5, "double_pole": factor * 0.5}
    unstable = {"status": 428, "born": 0.25, "double_pole": factor * 0.25}
    cases = (regular, suppressed, vanishing, double, unstable)
    for qp_point, sensitive in ((False, "none"), (True, "ok")):
        entries = [dict(e) for e in cases]
        references.per_helicity_reference(entries, factor, qp_point)
        records = [e["reference"] for e in entries]
        assert [r["comparison"] for r in records] == [
            "relative",
            "relative",
            "absolute",
            "relative",
            "relative",
        ]
        # A point-sensitive helicity is no reference at an exact point, a
        # relative one at MadLoop's own quadruple-precision point; an unstable
        # quadruple-precision entry is never one.
        assert [r["status"] for r in records] == ["ok", sensitive, "ok", "ok", "none"]
        assert entries[0]["born"] == regular["double_pole"] / factor
        assert entries[0]["born_binary64_input"] == 1.0
        assert (
            records[3]["born_source"] == "madloop"
            and "born_binary64_input" not in entries[3]
        )
        summed = sum(abs(e["born"]) for e in entries)
        assert records[2]["absolute_scale"] == summed
        assert records[0]["absolute_scale"] is records[1]["absolute_scale"] is None
        assert "vanishing" in records[2]["reason"] and "428" in records[4]["reason"]
        assert bool(records[1]["reason"]) == (not qp_point)


def test_merge_refuses_an_undeclared_unstable_point_and_records_outliers() -> None:
    from tools.loop_reference import references

    factor, coupling = -0.05, 0.118
    good = _madloop_row({(4, 0): 0.69}, {}, factor)
    unstable = dict(_madloop_row({(4, 0): 0.71}, {}, factor), status=429)
    with pytest.raises(SystemExit, match="unstable"):
        references.madloop_references(
            [{"born": 0.69}, {"born": 0.71}],
            [good, unstable],
            "madloop",
            factor,
            coupling,
            {},
            False,
            2e-8,
        )
    outlier = {"rule": 1, "ruling": "pending", "conditions": {"i": "fails"}}
    declared = references.point_statuses(
        json.dumps({"1": {"outlier": outlier, "reason": "synthetic"}}), 2
    )
    rows = [{"born": 0.69}, {"born": 0.71}]
    references.madloop_references(
        rows, [good, unstable], "madloop", factor, coupling, declared, False, 2e-8
    )
    assert rows[0]["reference"]["outlier"] is None
    assert rows[1]["reference"]["outlier"] == outlier
    assert rows[1]["reference"]["status"] == "ok"
    for bad in (
        {"status": "none", "outlier": outlier, "reason": "x"},
        {"outlier": {"rule": 1}, "reason": "x"},
        {"outlier": {**outlier, "other": 1}, "reason": "x"},
    ):
        with pytest.raises(SystemExit, match="outlier"):
            references.point_statuses(json.dumps({"0": bad}), 2)
