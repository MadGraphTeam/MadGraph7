# SPDX-License-Identifier: 0BSD
"""Integrity of the M-SM reference campaign (m-sm/PLAN.md §4, package MS-REF).

Pure JSON and exact arithmetic over the committed files: the campaign table
(tools/loop_reference/msm_campaign.json), its reference manifest
(tests/fixtures/loop/required/msm_references.json) and every MadLoop reference
fixture it names, plus the toolkit pieces the M-SM campaign added (joint
QCD+QED sectors, the complex-mass G_mu alpha, explicit sqrt(s) points, R2
name lists). No provider, no pyAmpliCol generation.
"""

from __future__ import annotations

import cmath
import json
import math
import re
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_reference import references, required_spec

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
CAMPAIGN = ROOT / "tools" / "loop_reference" / "msm_campaign.json"
MANIFEST = FIXTURES / "required" / "msm_references.json"
LAURENT = ("finite", "single_pole", "double_pole")
# §4.2: 70 core rows (the decay rows XX01/XX02 wait for the user's Q3).
ROW_COUNTS = {"LL": 7, "DY": 7, "VJ": 3, "VV": 6, "V3": 1, "VH": 3, "TT": 6, "TV": 5, "ST": 4, "JJ": 4, "PI": 4,
              "CM": 9, "MP": 2, "SC": 9}
# Existing references that ew-validate (EWI-7) commits; pending here until it integrates.
EWI7_FIXTURES = {"required/ttx_ttx_ew.json", "required/udx_tbx_ew.json", "required/ttx_zh_ew.json"}
CMS_CONVENTIONS = ("CMSParam", "LogSwitch", "GfRedefinitionChoice")
UNIVERSAL_POLE_TOLERANCE = 1e-10
RUN_IDS = [r["id"] for r in json.loads(CAMPAIGN.read_text())["runs"]]

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def campaign() -> dict:
    return json.loads(CAMPAIGN.read_text())


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _runs(campaign: dict) -> dict:
    return {r["id"]: r for r in campaign["runs"]}


def _fixture(run_id: str) -> dict:
    return json.loads((FIXTURES / "required" / "msm" / f"{run_id}.json").read_text())


def _spec(run: dict) -> dict:
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, run["process"])
    if run.get("pdgs"):
        spec["pdgs"] = run["pdgs"]
    return spec


def test_campaign_and_manifest_cover_every_row(campaign: dict, manifest: dict) -> None:
    rows = [row["row"] for row in campaign["rows"]]
    assert len(rows) == len(set(rows)) == sum(ROW_COUNTS.values())
    for prefix, count in ROW_COUNTS.items():
        assert sum(r.startswith(prefix) for r in rows) == count, prefix
    assert [row["row"] for row in manifest["rows"]] == rows
    assert manifest["schema"] == "msm-references/1"
    runs = _runs(campaign)
    for row in campaign["rows"]:
        assert set(row["runs"]) <= set(runs), row["row"]
        assert row["runs"] or row["existing_fixtures"], row["row"]
        assert row["restriction_code"] in campaign["restriction_codes"], row["row"]
    for row in manifest["rows"]:
        for ref in row["references"]:
            assert ref["status"] == "committed" and (FIXTURES / ref["fixture"]).is_file(), ref["run"]
        for ref in row["existing_references"]:
            if ref["status"] != "committed":
                assert ref["fixture"] in EWI7_FIXTURES, ref
            else:
                assert (FIXTURES / ref["fixture"]).is_file(), ref
        # Oracle runs of another card (the technical epsIR runs) are never rule 1 references.
        oracles = {f"required/msm/{r['run']}.json" for r in row["references"] if r.get("binding") is False}
        assert not oracles & {b["fixture"] for b in row["binding_references"]}, row["row"]


def test_every_run_process_and_scheme_is_specified(campaign: dict) -> None:
    schemes = required_spec.load(required_spec.DEFAULT_SPEC)["schemes"]
    for run in campaign["runs"]:
        spec = _spec(run)
        assert spec["role"] == "msm" and spec["scheme_data"]["madloop_model"], run["id"]
        # Explicit squared orders on every job (m-sm/PLAN.md §3.2 MS-REF 2).
        process = spec["madloop"]["process"]
        assert re.search(r"(QCD\^2(==|<=)\d+ |\[virt=QED\] )QED\^2(==|<=)\d+$", process), run["id"]
    for code, route in campaign["restriction_codes"].items():
        assert route["scheme"] in schemes, code
        assert bool(route.get("mass_scheme") == "complex-mass") == bool(
            schemes[route["scheme"]].get("complex_mass_scheme")), code


def _shell_and_conservation(point: list[list[str]], masses: list) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = 80
        p = [[Decimal(x) for x in leg] for leg in point]
        scale = max(leg[0] for leg in p)
        shell = max(abs(e * e - x * x - y * y - z * z - Decimal(m.numerator) ** 2 / Decimal(m.denominator) ** 2)
                    / (scale * scale) for (e, x, y, z), m in zip(p, masses, strict=True))
        conservation = max(abs(p[0][mu] + p[1][mu] - sum(leg[mu] for leg in p[2:])) for mu in range(4)) / scale
    return shell, conservation


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_reference_fixture_records_its_settings_points_and_sectors(campaign: dict, run_id: str) -> None:
    run = _runs(campaign)[run_id]
    fixture = _fixture(run_id)
    assert not re.search(r"/shared/|/home/|pyLoopAmpliCol", json.dumps(fixture))
    assert fixture["mqcd_run"]["id"] == run_id
    spec = _spec(run)
    scheme = spec["scheme_data"]
    # Model, restriction, scheme, parameters and widths, CMS flag and conventions, photon tags.
    assert fixture["pdgs"] == spec["pdgs"]
    assert (fixture["scheme"], fixture["restriction"]) == (spec["scheme"], scheme.get("restriction"))
    assert fixture["madloop_model"] == Path(spec["madloop"]["model"]).name
    assert fixture["parameters"] == scheme["parameters"]
    assert fixture["fixed_parameters"] == scheme["fixed_parameters"]
    assert fixture["complex_mass_scheme"] is bool(scheme.get("complex_mass_scheme"))
    if fixture["complex_mass_scheme"] and "loop_qcd_qed_sm" in fixture["madloop_model"]:
        assert set(CMS_CONVENTIONS) <= set(fixture["scheme_conventions"])
    assert fixture.get("tagged_photon_legs") == spec.get("tagged_photon_legs")
    assert fixture.get("corrections", [spec["correction"]]) == spec["corrections"]
    # Provenance and MadLoop's quadruple-precision evaluation at every point.
    madloop = fixture["providers"]["madloop"]
    settings = madloop["settings"]
    evaluation = campaign["madloop_settings"]["evaluation"]
    assert madloop["version"].startswith("MadLoop/MG5_aMC 3.7.0")
    assert madloop["generation"]["status"] == "compiled"
    assert madloop["generation"]["command"] == spec["madloop"]["process"]
    assert settings["stability_check"] == "forced on every call"
    assert settings["requested_relative_accuracy"] == evaluation["requested_relative_accuracy"]
    assert settings["quadruple_precision_required"] is True
    assert settings["MadLoopParams_overrides"] == evaluation["madloop_params"]
    statuses = [row["madloop_status"] for row in fixture["rows"]]
    assert all(status // 100 == 3 for status in statuses), statuses
    # The R2 couplings the export calls (the rule-1 R2 correction's input).
    r2 = fixture["madloop_r2_couplings"]
    assert r2["called"] == sorted(set(r2["called"])) and len(r2["called"]) <= r2["ufo_r2_couplings"]
    # Points on the scheme's mass shells, momentum conserving.
    masses = required_spec.masses(spec)
    assert len(fixture["rows"]) == len(fixture["points"]) == len(fixture["mu_r"])
    for point in fixture["points"]:
        shell, conservation = _shell_and_conservation(point, masses)
        assert shell < Decimal("1e-25") and conservation < Decimal("1e-25")
    # Every squared split order with MadLoop's own accuracy; the ROW is the primary sector.
    primary = tuple(spec["born_orders"][k] + spec["loop_orders"][k] for k in ("QCD", "QED"))
    for row in fixture["rows"]:
        loop = {(s["QCD"], s["QED"]): s for s in row["sectors"]["loop"]}
        assert all(s["accuracy"] >= 0.0 and all(isinstance(s[k], float) for k in LAURENT) for s in loop.values())
        assert loop[primary]["finite"] == row["providers"]["madloop"]["finite"]
        if len(spec["corrections"]) > 1:
            assert all(set(s["born_sectors"]) == set(spec["corrections"]) for s in loop.values())


def _factors(fixture: dict, spec: dict) -> tuple[dict, dict]:
    """Universal double-pole factor and loop coupling per correction, recomputed
    (alpha_s; the scheme's alpha, the complex-mass G_mu one under CMS)."""
    scheme = spec["scheme_data"]
    alpha = float(Decimal(references.derived_parameters(scheme)["alpha"]))
    couplings = {c: float(Decimal(scheme["parameters"]["aS"])) if c == "QCD" else alpha for c in spec["corrections"]}
    return {c: required_spec.double_pole_factor(spec, couplings[c], c) for c in couplings}, couplings


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_every_loop_sector_has_the_universal_double_pole(campaign: dict, run_id: str) -> None:
    """D_s = sum over the request's corrections of factor_c * Born(s - step_c) per
    loop squared split order at every ok reference point (a joint QCD+QED sector
    gets both), with the factors recomputed here; the recorded violation is that
    of the printed sector values."""
    run = _runs(campaign)[run_id]
    fixture = _fixture(run_id)
    spec = _spec(run)
    factors, couplings = _factors(fixture, spec)
    assert fixture["universal_double_pole_factor"] == pytest.approx(factors[spec["correction"]], rel=1e-15)
    if len(factors) > 1:
        assert fixture["universal_double_pole_factors"] == pytest.approx(factors, rel=1e-15)
    for index, row in enumerate(fixture["rows"]):
        reference = row["reference"]
        if reference["status"] != "ok":
            assert reference["reason"]
            continue
        sectors = row["sectors"]
        born = {(s["QCD"], s["QED"]): s.get("born_binary64_input", s["born"]) for s in sectors["born"]}
        largest = max(abs(s["double_pole"]) for s in sectors["loop"])
        absolute = max(abs(f) for f in factors.values()) * (reference["absolute_scale"] or 0.0)
        for sector in sectors["loop"]:
            expected, unit = 0.0, 0.0
            for correction, factor in factors.items():
                step = required_spec.CORRECTION_STEP[correction]
                partner = born.get((sector["QCD"] - step["QCD"], sector["QED"] - step["QED"]), 0.0)
                expected += factor * partner
                # One natural loop unit: a double pole without massless charged legs vanishes.
                unit = max(unit, abs(partner) * couplings[correction] / (2 * math.pi))
            scale = max(abs(sector["double_pole"]), abs(expected), largest, absolute, unit)
            assert abs(sector["double_pole"] - expected) <= UNIVERSAL_POLE_TOLERANCE * scale, (
                index, sector["QCD"], sector["QED"])
        assert reference["max_sector_double_pole_relative_error"] <= 2e-8, index


# --------------------------------------------------------------------------
# Toolkit pieces of the M-SM campaign (synthetic inputs).


def test_joint_sectors_name_one_born_sector_per_correction() -> None:
    steps = {c: required_spec.CORRECTION_STEP[c] for c in ("QCD", "EW")}
    # ANS(0:3, 0:2) column-major: total, sector 1, sector 2; then ACC(0:2).
    values = [1.0, 0.0, 0.0, -0.06, 1.0, 0.1, 0.2, -0.05, 0.0, 0.3, 0.4, -0.01, 0.0, 0.0, 0.0]
    sectors = references.madloop_sectors(values, 2, 2, {1: (0, 4)}, {1: (2, 4), 2: (0, 6)}, steps)
    qcd, ew = sectors["loop"]
    assert qcd["born_sectors"] == {"QCD": {"QCD": 0, "QED": 4}, "EW": None}
    assert ew["born_sectors"] == {"QCD": None, "EW": {"QCD": 0, "QED": 4}}
    assert qcd["born_sector"] == ew["born_sector"] == {"QCD": 0, "QED": 4}
    row = {"born": 1.0, "double_pole": -0.06, "sectors": sectors}
    factors, couplings = {"QCD": -0.05, "EW": -0.01}, {"QCD": 0.118, "EW": 0.0075}
    assert references.sector_pole_error(row, factors, couplings) < 1e-15
    # One correction alone misses the other's sector at O(1).
    assert references.sector_pole_error(row, -0.05, 0.118) > 0.1
    sectors["loop"][1]["double_pole"] = -0.0101
    assert references.sector_pole_error(row, factors, couplings) == pytest.approx(1e-4 / 0.05)


def test_joint_mixed_sector_sums_both_corrections() -> None:
    """A loop sector reached by both corrections (s^4e^2 of a mixed Born) has the
    double pole factor_QCD * B(s^2e^2) + factor_EW * B(s^4)."""
    steps = {c: required_spec.CORRECTION_STEP[c] for c in ("QCD", "EW")}
    born_so = {1: (4, 0), 2: (2, 2)}
    loop_so = {1: (6, 0), 2: (4, 2)}
    fq, fe = -0.05, -0.01
    d6, d42 = fq * 2.0, fq * 0.5 + fe * 2.0
    values = [2.5, 0, 0, d6 + d42, 2.0, 0, 0, d6, 0.5, 0, 0, d42, 0, 0, 0]
    sectors = references.madloop_sectors(values, 2, 2, born_so, loop_so, steps)
    mixed = sectors["loop"][1]
    assert mixed["born_sectors"] == {"QCD": {"QCD": 2, "QED": 2}, "EW": {"QCD": 4, "QED": 0}}
    row = {"born": 2.5, "double_pole": d6 + d42, "sectors": sectors}
    assert references.sector_pole_error(row, {"QCD": fq, "EW": fe}, {"QCD": 0.118, "EW": 0.0075}) < 1e-15
    # Only single-partner sectors substitute the Born (MadLoop's quadruple-precision points).
    partners = references.double_pole_partners(sectors, {"QCD": fq, "EW": fe})
    assert set(partners) == {(4, 0)}


def test_complex_mass_gmu_alpha_is_the_ufo_formula() -> None:
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, "gg_ttxemepg_cms")
    scheme = spec["scheme_data"]
    values = {**scheme["parameters"], **scheme["fixed_parameters"]}
    mw, mz = float(values["MW"]), float(values["MZ"])
    w2, z2 = mw * mw - 1j * mw * float(values["WW"]), mz * mz - 1j * mz * float(values["WZ"])
    expected = math.sqrt(2) * float(values["Gf"]) * abs(w2 * (1 - w2 / z2)) / cmath.pi
    alpha = float(references.derived_parameters(scheme)["alpha"])
    assert alpha == pytest.approx(expected.real, rel=1e-14)
    real = required_spec.load_process(required_spec.DEFAULT_SPEC, "gg_ttxemepg_ew")["scheme_data"]
    assert abs(alpha / float(references.derived_parameters(real)["alpha"]) - 1) > 1e-4


def test_supplied_points_at_listed_energies() -> None:
    from tools.loop_reference import supplied_points

    masses = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
    with localcontext() as context:
        context.prec = supplied_points.CONSTRUCTION_DIGITS
        points, scales, _ = supplied_points.generate(masses, sqrt_s_values=[Fraction("91.188"), Fraction(4000),
                                                                            Fraction(4000)])
    assert scales == [Fraction("91.188"), Fraction(4000), Fraction(4000)]
    assert points[1] != points[2]
    with pytest.raises(ValueError, match="regular"):
        supplied_points.generate(masses, family="soft", values=[Fraction(1, 100)], sqrt_s_values=[Fraction(100)])


def test_r2_name_list_reads_call_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = tmp_path / "SubProcesses" / "P1_xx"
    process.mkdir(parents=True)
    (process / "helas_calls_ampb_1.f").write_text("      CALL FFV1_0(W(1,1),W(1,2),W(1,3),R2_G0TT,AMP(1))\n"
                                                    "C     R2_GOTT_B is not R2_G0tt_b\n")
    monkeypatch.setattr(references, "ufo_r2_coupling_names", lambda model: {"R2_G0tt", "R2_G0tt_b", "R2_AAU"})
    record = references.madloop_r2_couplings(tmp_path, "model")
    assert record["called"] == ["R2_G0tt"] and record["ufo_r2_couplings"] == 3
