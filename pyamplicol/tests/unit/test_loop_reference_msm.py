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
# §4.2: 70 core rows (the decay rows XX01/XX02 wait for the user's Q3) and
# SC10 (z z > h h, the R2_ZZHH row of MS-UFODB).
ROW_COUNTS = {
    "LL": 7,
    "DY": 7,
    "VJ": 3,
    "VV": 6,
    "V3": 1,
    "VH": 3,
    "TT": 6,
    "TV": 5,
    "ST": 4,
    "JJ": 4,
    "PI": 4,
    "CM": 9,
    "MP": 2,
    "SC": 10,
}
# ST04: ew-validate's workspace run committed by MS-REF (MadLoop double precision).
ST04_FIXTURE = "required/udx_tbx_ew_mb.json"
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
            assert (
                ref["status"] == "committed" and (FIXTURES / ref["fixture"]).is_file()
            ), ref["run"]
        for ref in row["existing_references"]:
            assert (
                ref["status"] == "committed" and (FIXTURES / ref["fixture"]).is_file()
            ), ref
        # Oracle runs of another card (technical epsIR runs) are no rule 1 references.
        oracles = {
            f"required/msm/{r['run']}.json"
            for r in row["references"]
            if r.get("binding") is False
        }
        assert not oracles & {b["fixture"] for b in row["binding_references"]}, row[
            "row"
        ]


def test_every_run_process_and_scheme_is_specified(campaign: dict) -> None:
    schemes = required_spec.load(required_spec.DEFAULT_SPEC)["schemes"]
    # A run may re-evaluate the process of one of its row's existing
    # references on the patched models (MS-UFO-PATCH): its role is that one's.
    existing = {}
    for row in campaign["rows"]:
        keys = {Path(path).stem for path in row["existing_fixtures"]}
        for exports in row.get("existing_r2_exports", {}).values():
            if not isinstance(exports, str):
                keys.update(Path(export).parent.name for export in exports)
        for run_id in row["runs"]:
            existing[run_id] = keys
    for run in campaign["runs"]:
        spec = _spec(run)
        assert spec["role"] == "msm" or run["process"] in existing[run["id"]], run["id"]
        assert spec["scheme_data"]["madloop_model"], run["id"]
        # Explicit squared orders on every job (m-sm/PLAN.md §3.2 MS-REF 2).
        process = spec["madloop"]["process"]
        assert re.search(
            r"(QCD\^2(==|<=)\d+ |\[virt=QED\] )QED\^2(==|<=)\d+$", process
        ), run["id"]
    for code, route in campaign["restriction_codes"].items():
        assert route["scheme"] in schemes, code
        assert bool(route.get("mass_scheme") == "complex-mass") == bool(
            schemes[route["scheme"]].get("complex_mass_scheme")
        ), code


def _check_points(fixture: dict, spec: dict) -> None:
    """Points on the scheme's mass shells, momentum conserving."""
    masses = required_spec.masses(spec)
    assert len(fixture["rows"]) == len(fixture["points"]) == len(fixture["mu_r"])
    for point in fixture["points"]:
        shell, conservation = _shell_and_conservation(point, masses)
        assert shell < Decimal("1e-25") and conservation < Decimal("1e-25")


def _shell_and_conservation(
    point: list[list[str]], masses: list
) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = 80
        p = [[Decimal(x) for x in leg] for leg in point]
        scale = max(leg[0] for leg in p)
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


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_reference_fixture_records_its_settings_points_and_sectors(
    campaign: dict, run_id: str
) -> None:
    run = _runs(campaign)[run_id]
    fixture = _fixture(run_id)
    assert not re.search(r"/shared/|/home/|pyLoopAmpliCol", json.dumps(fixture))
    assert fixture["mqcd_run"]["id"] == run_id
    spec = _spec(run)
    scheme = spec["scheme_data"]
    # Model, restriction, scheme, parameters, widths, CMS flag, conventions, photons.
    assert fixture["pdgs"] == spec["pdgs"]
    assert (fixture["scheme"], fixture["restriction"]) == (
        spec["scheme"],
        scheme.get("restriction"),
    )
    assert fixture["madloop_model"] == Path(spec["madloop"]["model"]).name
    assert fixture["parameters"] == scheme["parameters"]
    assert fixture["fixed_parameters"] == scheme["fixed_parameters"]
    assert fixture["complex_mass_scheme"] is bool(scheme.get("complex_mass_scheme"))
    if fixture["complex_mass_scheme"] and "loop_qcd_qed_sm" in fixture["madloop_model"]:
        assert set(CMS_CONVENTIONS) <= set(fixture["scheme_conventions"])
    assert fixture.get("tagged_photon_legs") == spec.get("tagged_photon_legs")
    if 22 in fixture["pdgs"]:
        # Every external photon is recorded tagged or untagged ([] = untagged).
        assert isinstance(fixture.get("tagged_photon_legs"), list)
    assert fixture.get("corrections", [spec["correction"]]) == spec["corrections"]
    # Provenance and MadLoop's quadruple-precision evaluation at every point.
    madloop = fixture["providers"]["madloop"]
    settings = madloop["settings"]
    evaluation = campaign["madloop_settings"]["evaluation"]
    assert madloop["version"].startswith("MadLoop/MG5_aMC 3.7.0")
    assert madloop["generation"]["status"] == "compiled"
    assert madloop["generation"]["command"] == spec["madloop"]["process"]
    assert settings["stability_check"] == "forced on every call"
    assert (
        settings["requested_relative_accuracy"]
        == evaluation["requested_relative_accuracy"]
    )
    assert settings["quadruple_precision_required"] is True
    # The campaign's MadLoop parameters, a run's own over them (SC02: CutTools).
    assert settings["MadLoopParams_overrides"] == {
        **evaluation["madloop_params"],
        **run.get("madloop_params", {}),
    }
    # MadLoop's return code HTU: H = 3 (stability rescue succeeded), T = 2
    # (quadruple precision used).
    statuses = [row["madloop_status"] for row in fixture["rows"]]
    assert all(s // 100 == 3 and s // 10 % 10 == 2 for s in statuses), statuses
    # The UFO R2 couplings the export inserts (the rule-1 R2 correction's input)
    # and the merged names MG5 calls them by (madloop_ct_couplings.py).
    r2 = fixture["madloop_r2_couplings"]
    assert (
        r2["called"] == sorted(set(r2["called"]))
        and len(r2["called"]) <= r2["ufo_r2_couplings"]
        and set(r2["called_as"]) <= set(r2["called"])
        and all(r2["called_as"][n].lstrip("-") != n for n in r2["called_as"])
    )
    _check_points(fixture, spec)
    # Every squared split order with MadLoop's accuracy; ROW = the primary sector.
    primary = tuple(
        spec["born_orders"][k] + spec["loop_orders"][k] for k in ("QCD", "QED")
    )
    for row in fixture["rows"]:
        loop = {(s["QCD"], s["QED"]): s for s in row["sectors"]["loop"]}
        assert all(
            s["accuracy"] >= 0.0 and all(isinstance(s[k], float) for k in LAURENT)
            for s in loop.values()
        )
        assert loop[primary]["finite"] == row["providers"]["madloop"]["finite"]
        if len(spec["corrections"]) > 1:
            assert all(
                set(s["born_sectors"]) == set(spec["corrections"])
                for s in loop.values()
            )


def _factors(fixture: dict, spec: dict) -> tuple[dict, dict]:
    """Universal double-pole factor and loop coupling per correction, recomputed
    (alpha_s; the scheme's alpha, the complex-mass G_mu one under CMS)."""
    scheme = spec["scheme_data"]
    alpha = float(Decimal(references.derived_parameters(scheme)["alpha"]))
    couplings = {
        c: float(Decimal(scheme["parameters"]["aS"])) if c == "QCD" else alpha
        for c in spec["corrections"]
    }
    return {
        c: required_spec.double_pole_factor(spec, couplings[c], c) for c in couplings
    }, couplings


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_every_loop_sector_has_the_universal_double_pole(
    campaign: dict, run_id: str
) -> None:
    """D_s = sum over the request's corrections of factor_c * Born(s - step_c) per
    loop squared split order at every ok reference point (a joint QCD+QED sector
    gets both), with the factors recomputed here; the recorded violation is that
    of the printed sector values."""
    _check_double_poles(_fixture(run_id), _spec(_runs(campaign)[run_id]))


def test_existing_fixtures_name_their_r2_lists(campaign: dict, manifest: dict) -> None:
    """existing_r2_couplings.json (mqcd_campaign.py existing-r2): every existing
    fixture of a row has the R2 name list of its MadLoop export(s) or a reason
    for none, and the manifest links it."""
    data = json.loads(
        (FIXTURES / "required" / "msm" / "existing_r2_couplings.json").read_text()
    )
    exports, without = data["exports"], data["without_r2_list"]
    existing = {f for row in campaign["rows"] for f in row["existing_fixtures"]}
    listed = {f for entry in exports.values() for f in entry["fixtures"]}
    assert listed | set(without) == existing and not listed & set(without)
    for key, entry in exports.items():
        assert entry["fixture"] == entry["fixtures"][0]
        assert entry["export"].startswith("${LOOP_REFERENCE_ROOT}/runs/")
        assert Path(entry["export"]).parent.name == key
        called = entry["called"]
        assert (
            called == sorted(set(called)) and len(called) <= entry["ufo_r2_couplings"]
        )
        assert set(entry["called_as"]) <= set(called)
    for row in manifest["rows"]:
        for ref in row["existing_references"]:
            link = ref["r2_couplings"]
            if isinstance(link, str):
                assert without[ref["fixture"]] == link
            else:
                keys = [item.split("#", 1)[1] for item in link]
                assert all(ref["fixture"] in exports[k]["fixtures"] for k in keys)


def test_st04_existing_reference_is_relocated_and_consistent() -> None:
    """ST04's committed fixture (ew-validate's run, MadLoop double precision):
    no host paths, the specified process and scheme, points on the mass shells,
    the universal double pole in every loop sector and the QED single pole."""
    fixture = json.loads((FIXTURES / ST04_FIXTURE).read_text())
    assert not re.search(r"/shared/|/home/|pyLoopAmpliCol", json.dumps(fixture))
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, fixture["name"])
    assert fixture["pdgs"] == spec["pdgs"] and fixture["scheme"] == spec["scheme"]
    assert fixture["madloop_model"] == Path(spec["madloop"]["model"]).name
    assert fixture["parameters"] == spec["scheme_data"]["parameters"]
    assert "recola" in fixture["excluded_providers"]
    _check_points(fixture, spec)
    _check_double_poles(fixture, spec)
    _check_ir_single_poles(fixture, spec, {})


def _check_double_poles(fixture: dict, spec: dict) -> None:
    factors, couplings = _factors(fixture, spec)
    assert fixture["universal_double_pole_factor"] == pytest.approx(
        factors[spec["correction"]], rel=1e-15
    )
    if len(factors) > 1:
        assert fixture["universal_double_pole_factors"] == pytest.approx(
            factors, rel=1e-15
        )
    for index, row in enumerate(fixture["rows"]):
        reference = row["reference"]
        if reference["status"] != "ok":
            assert reference["reason"]
            continue
        sectors = row["sectors"]
        born = {
            (s["QCD"], s["QED"]): s.get("born_binary64_input", s["born"])
            for s in sectors["born"]
        }
        largest = max(abs(s["double_pole"]) for s in sectors["loop"])
        absolute = max(abs(f) for f in factors.values()) * (
            reference["absolute_scale"] or 0.0
        )
        for sector in sectors["loop"]:
            expected, unit = 0.0, 0.0
            for correction, factor in factors.items():
                step = required_spec.CORRECTION_STEP[correction]
                partner = born.get(
                    (sector["QCD"] - step["QCD"], sector["QED"] - step["QED"]), 0.0
                )
                expected += factor * partner
                # One loop unit: no massless charged leg, no double pole.
                unit = max(unit, abs(partner) * couplings[correction] / (2 * math.pi))
            scale = max(
                abs(sector["double_pole"]), abs(expected), largest, absolute, unit
            )
            assert (
                abs(sector["double_pole"] - expected)
                <= UNIVERSAL_POLE_TOLERANCE * scale
            ), (index, sector["QCD"], sector["QED"])
        assert reference["max_sector_double_pole_relative_error"] <= 2e-8, index


# QED single pole per EW loop sector, in units of (alpha/2pi) Born: the G1
# tolerance; 1e-5 (the lax §1.4 precision) where the model writes a massive
# field's counterterms for the massless field (they also miss O(m^2/M_W^2)
# Yukawa terms: 2.4e-6 at SC02).
IR_SINGLE_POLE_TOLERANCE = 2e-8
MASSLESS_COUNTERTERM_TOLERANCE = 1e-5
# Massless fermions of the scheme close the photon's light-fermion loops.
FERMIONS = (1, 2, 3, 4, 5, 6, 11, 13, 15)
EW_RUN_IDS = [
    r["id"]
    for r in json.loads(CAMPAIGN.read_text())["runs"]
    if "EW" in _spec(r)["corrections"]
]


def _minkowski(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def _qed_single_pole(
    pdgs: list[int], masses: list, momenta: list, mu: float, light: Fraction
) -> float:
    """Single pole / ((alpha/2pi) Born) of a UV-renormalised one-loop amplitude
    from photon exchange: Re Gamma_0 / 2 of the one-loop IR anomalous dimension
    with massive legs (Becher-Neubert, arXiv:0904.1021 eq. 2) for QED charges
    (all outgoing, Q_i Q_j for T_i.T_j, gamma_cusp = 4; gamma = -3 Q^2 for a
    massless fermion, -2 Q^2 for a massive leg, +4/3 sum of N_c Q_f^2 over the
    massless fermions for a photon). Independent of the engine and of MadLoop."""
    charge, k = [], []
    for i, (pdg, p) in enumerate(zip(pdgs, momenta, strict=True)):
        sign = -1 if i < 2 else 1
        charge.append(float(required_spec.particle(pdg)["charge"]) * sign)
        k.append([sign * x for x in p])
    m = [float(x) for x in masses]
    gamma = 0.0
    for i, pdg in enumerate(pdgs):
        if abs(pdg) == 22:
            gamma += 4 / 3 * float(light)
            continue
        if not charge[i]:
            continue
        gamma += (-3.0 if m[i] == 0 else -2.0) * charge[i] ** 2
        for j in range(len(pdgs)):
            if j == i or not charge[j]:
                continue
            s = 2 * _minkowski(k[i], k[j])
            qq = charge[i] * charge[j]
            if m[i] == 0 and m[j] == 0:
                gamma += 2 * qq * math.log(mu * mu / abs(s))
            elif m[i] and m[j]:
                # gamma_cusp(beta) = 4 beta coth beta, cosh beta = -s/(2 m m) + i0.
                beta = cmath.acosh(complex(-s / (2 * m[i] * m[j]), 0.0))
                cusp = 1.0 if abs(beta) < 1e-12 else (beta / cmath.tanh(beta)).real
                gamma -= 2 * qq * cusp
            elif m[i]:
                gamma += 4 * qq * math.log(m[i] * mu / abs(s))
    return gamma / 2


def _ir_single_pole_residuals(fixture: dict, spec: dict) -> dict:
    """(row, QCD, QED) -> (single pole - QED IR prediction) / ((alpha/2pi) Born)
    for every EW loop sector at the lowest QCD order of the Borns (photon
    exchange on one Born sector, colour-diagonal; no gluon reaches it) at every
    ok point. Absolute rows scale by the regular neighbour's Born. A scheme's
    massless_field_counterterms add 3/2 Q^2 per such leg (the IR pole of the
    massive on-shell field renormalisation the model omits)."""
    scheme = spec["scheme_data"]
    unit = float(Decimal(references.derived_parameters(scheme)["alpha"])) / (
        2 * math.pi
    )
    masses = required_spec.masses(spec)
    light = sum(
        required_spec.particle(f)["charge"] ** 2 * (3 if f <= 6 else 1)
        for f, mass in zip(
            FERMIONS,
            required_spec.masses({"pdgs": FERMIONS, "scheme_data": scheme}),
            strict=True,
        )
        if not mass
    )
    offset = sum(
        1.5 * float(required_spec.particle(pdg)["charge"] ** 2)
        for pdg in fixture["pdgs"]
        if abs(pdg) in scheme.get("massless_field_counterterms", ())
    )
    out = {}
    for index, row in enumerate(fixture["rows"]):
        reference = row["reference"]
        if reference["status"] != "ok":
            continue
        born = {
            (s["QCD"], s["QED"]): s.get("born_binary64_input", s["born"])
            for s in row["sectors"]["born"]
        }
        lowest = min(q for q, _ in born)
        predicted = offset + _qed_single_pole(
            fixture["pdgs"],
            masses,
            [[float(x) for x in leg] for leg in fixture["points"][index]],
            float(fixture["mu_r"][index]),
            light,
        )
        for sector in row["sectors"]["loop"]:
            partner = born.get((sector["QCD"], sector["QED"] - 2))
            if sector["QCD"] != lowest or not partner:
                continue
            scale = max(abs(partner), reference.get("absolute_scale") or 0.0)
            out[(index, sector["QCD"], sector["QED"])] = (
                sector["single_pole"] - predicted * unit * partner
            ) / (unit * scale)
    return out


def _check_ir_single_poles(fixture: dict, spec: dict, run: dict) -> None:
    residuals = _ir_single_pole_residuals(fixture, spec)
    assert residuals
    scheme = spec["scheme_data"]
    worst = max(abs(r) for r in residuals.values())
    if scheme["parameters"].get("epsIR") == "0":
        # A technical epsIR = 0 run (MS-POLES oracle): no IR poles, no rule 1 reference.
        assert run.get("binding") is False
        assert min(abs(r) for r in residuals.values()) > 1e-3, residuals
    elif scheme.get("tau_mass_counterterms_omitted"):
        # The UFOmodeldb issue-1 patch's retained limitation: with a massive
        # tau the EW counterterms (which omit its mass) leave
        # MTA^2 / (2 sw^2 (s - MW^2)) x (alpha/2pi) Born in the single pole of
        # u d~ > e+ ve (the user's patch notes; the only process of the scheme).
        assert fixture["pdgs"] == [2, -1, -11, 12]
        p = {k: float(v) for k, v in scheme["parameters"].items()}
        sw2 = 1 - p["MW"] ** 2 / p["MZ"] ** 2
        for (index, _qcd, _qed), residual in residuals.items():
            legs = [[float(x) for x in leg] for leg in fixture["points"][index][:2]]
            total = [a + b for a, b in zip(*legs, strict=True)]
            s = _minkowski(total, total)
            expected = p["MTA"] ** 2 / (2 * sw2 * (s - p["MW"] ** 2))
            assert abs(residual - expected) <= IR_SINGLE_POLE_TOLERANCE, residuals
    elif scheme.get("ir_single_pole_remnant"):
        # The recorded model remnant (not UV finite) is real.
        assert min(abs(r) for r in residuals.values()) > 1e-6, residuals
    elif scheme.get("massless_field_counterterms"):
        assert worst <= MASSLESS_COUNTERTERM_TOLERANCE, residuals
    else:
        assert worst <= IR_SINGLE_POLE_TOLERANCE, residuals


@pytest.mark.parametrize("run_id", EW_RUN_IDS)
def test_every_ew_sector_has_the_qed_infrared_single_pole(
    campaign: dict, run_id: str
) -> None:
    """The single pole of every EW sector photon exchange alone reaches equals
    the QED IR prediction recomputed here (UV finiteness and a sound reduction:
    MadLoop's QP Ninja stage once passed its own test with this pole wrong)."""
    run = _runs(campaign)[run_id]
    _check_ir_single_poles(_fixture(run_id), _spec(run), run)


# --------------------------------------------------------------------------
# Toolkit pieces of the M-SM campaign (synthetic inputs).


def test_joint_sectors_name_one_born_sector_per_correction() -> None:
    steps = {c: required_spec.CORRECTION_STEP[c] for c in ("QCD", "EW")}
    # ANS(0:3, 0:2) column-major: total, sector 1, sector 2; then ACC(0:2).
    values = [
        1.0,
        0.0,
        0.0,
        -0.06,
        1.0,
        0.1,
        0.2,
        -0.05,
        0.0,
        0.3,
        0.4,
        -0.01,
        0.0,
        0.0,
        0.0,
    ]
    sectors = references.madloop_sectors(
        values, 2, 2, {1: (0, 4)}, {1: (2, 4), 2: (0, 6)}, steps
    )
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
    assert references.sector_pole_error(row, factors, couplings) == pytest.approx(
        1e-4 / 0.05
    )


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
    assert mixed["born_sectors"] == {
        "QCD": {"QCD": 2, "QED": 2},
        "EW": {"QCD": 4, "QED": 0},
    }
    row = {"born": 2.5, "double_pole": d6 + d42, "sectors": sectors}
    assert (
        references.sector_pole_error(
            row, {"QCD": fq, "EW": fe}, {"QCD": 0.118, "EW": 0.0075}
        )
        < 1e-15
    )
    # Only single-partner sectors substitute the Born (MadLoop's QP points).
    partners = references.double_pole_partners(sectors, {"QCD": fq, "EW": fe})
    assert set(partners) == {(4, 0)}


def test_complex_mass_gmu_alpha_is_the_ufo_formula() -> None:
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, "gg_ttxemepg_cms")
    scheme = spec["scheme_data"]
    values = {**scheme["parameters"], **scheme["fixed_parameters"]}
    mw, mz = float(values["MW"]), float(values["MZ"])
    w2, z2 = (
        mw * mw - 1j * mw * float(values["WW"]),
        mz * mz - 1j * mz * float(values["WZ"]),
    )
    expected = math.sqrt(2) * float(values["Gf"]) * abs(w2 * (1 - w2 / z2)) / cmath.pi
    alpha = float(references.derived_parameters(scheme)["alpha"])
    assert alpha == pytest.approx(expected.real, rel=1e-14)
    real = required_spec.load_process(required_spec.DEFAULT_SPEC, "gg_ttxemepg_ew")[
        "scheme_data"
    ]
    assert abs(alpha / float(references.derived_parameters(real)["alpha"]) - 1) > 1e-4


def test_cm09_records_the_exports_photon_conversion_parameter() -> None:
    """CM09 (u d~ > e+ ve a under CMS) records the export's complex
    eCoup_DGmua0_UV_EW finite part per mu_R (madloop_parameters.py): its real
    part runs with the parameter's 1/eps coefficient, e^2 (27 + 11 Nc) /
    (216 pi^2) per ln mu_R^2 at the complex-mass G_mu alpha; the imaginary part
    (the complex masses) does not run."""
    fixture = _fixture("udx_epvea_cms")
    record = fixture["export_parameters"]
    values = record["values"]["MDL_ECOUP_DGMUA0_UV_EW_FIN_"]
    assert record["mu_r"] == fixture["mu_r"] and len(values) == len(fixture["rows"])
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, "udx_epvea_cms")
    alpha = float(Decimal(references.derived_parameters(spec["scheme_data"])["alpha"]))
    slope = 4 * math.pi * alpha * (27 + 11 * 3) / (216 * math.pi**2)
    mu = [float(m) for m in record["mu_r"]]
    for i in range(1, len(mu)):
        assert values[i][0] - values[0][0] == pytest.approx(
            slope * math.log(mu[i] ** 2 / mu[0] ** 2), rel=1e-12
        )
        assert values[i][1] == pytest.approx(values[0][1], rel=1e-13)
    assert values[0][1] != 0.0


def test_supplied_points_at_listed_energies() -> None:
    from tools.loop_reference import supplied_points

    masses = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
    with localcontext() as context:
        context.prec = supplied_points.CONSTRUCTION_DIGITS
        points, scales, _ = supplied_points.generate(
            masses, sqrt_s_values=[Fraction("91.188"), Fraction(4000), Fraction(4000)]
        )
    assert scales == [Fraction("91.188"), Fraction(4000), Fraction(4000)]
    assert points[1] != points[2]
    with pytest.raises(ValueError, match="regular"):
        supplied_points.generate(
            masses,
            family="soft",
            values=[Fraction(1, 100)],
            sqrt_s_values=[Fraction(100)],
        )


def _synthetic_export(tmp_path: Path, record: dict) -> Path:
    export = tmp_path / "madloop-export"
    process = export / "SubProcesses" / "P1_xx"
    process.mkdir(parents=True)
    (process / "helas_calls_ampb_1.f").write_text(
        "      CALL FFV1_0(W(1,1),W(1,2),W(1,3),R2_CCA2CP,AMPL(1,1))\n"
        "      CALL VVV1_0(W(1,1),W(1,2),W(1,3),R2_AWWCS,AMPL(1,2))\n"
        "      CALL FFS1_0(W(1,1),W(1,2),W(1,3),R2_G0TT,AMPL(1,3))\n"
        "C     R2_G0TT_B is a comment\n"
    )
    (tmp_path / "madloop-export.ct_couplings.json").write_text(json.dumps(record))
    return export


R2_RECORD = {
    "subprocesses": {
        "xx": {
            "r2": ["R2_G0tt", "R2_uuA2Cp"],
            "r2_called_as": {"R2_uuA2Cp": "R2_ccA2Cp"},
            "ct_tokens": {
                "ampb": {
                    "R2_CCA2CP": ["R2_uuA2Cp"],
                    "R2_AWWCS": ["c_UVvexveA1EW_1eps"],
                    "R2_G0TT": ["R2_G0tt"],
                },
                "uvct": {},
            },
        }
    }
}


def test_r2_name_list_is_the_ufo_couplings_behind_the_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R2 list holds the UFO couplings MG5's merged call names stand for
    (R2_CCA2CP calls R2_uuA2Cp), not the call names; a recorded name absent
    from the sources is refused."""
    export = _synthetic_export(tmp_path, R2_RECORD)
    monkeypatch.setattr(
        references,
        "ufo_r2_coupling_names",
        lambda model: {"R2_G0tt", "R2_G0tt_b", "R2_uuA2Cp", "R2_ccA2Cp"},
    )
    record = references.madloop_r2_couplings(export, "model")
    assert record["called"] == ["R2_G0tt", "R2_uuA2Cp"]
    assert record["called_as"] == {"R2_uuA2Cp": "R2_ccA2Cp"}
    assert record["ufo_r2_couplings"] == 4
    stale = json.loads(json.dumps(R2_RECORD))
    stale["subprocesses"]["xx"]["r2"].append("R2_G0tt_b")
    _synthetic_export(tmp_path / "stale", stale)
    with pytest.raises(RuntimeError, match="absent"):
        references.madloop_r2_couplings(tmp_path / "stale" / "madloop-export", "model")


def test_ct_amplitudes_are_classified_by_their_ufo_couplings(tmp_path: Path) -> None:
    """madloop_ct_selection.py with the ct_couplings record: R2_AWWCS calls the
    UV coupling c_UVvexveA1EW_1eps here, so its amplitude is 'uv', not 'r2'."""
    from tools.loop_reference import madloop_ct_couplings, madloop_ct_selection

    export = _synthetic_export(tmp_path, R2_RECORD)
    process = export / "SubProcesses" / "P1_xx"
    (process / "loop_matrix.f").write_text("      PARAMETER (NCTAMPS=3)\n")
    groups = {"r2": re.compile("R2", re.I), "uv": re.compile("UV", re.I)}
    tokens = R2_RECORD["subprocesses"][madloop_ct_couplings.subprocess_key(process)]
    assert madloop_ct_selection.classify(process, groups) == {1: "r2", 2: "r2", 3: "r2"}
    assert madloop_ct_selection.classify(process, groups, tokens["ct_tokens"]) == {
        1: "r2",
        2: "uv",
        3: "r2",
    }


def test_coupling_renamings_compose_conversion_and_restriction() -> None:
    """madloop_ct_couplings.Recorder: the conversion's identical-expression
    renaming, then the restriction's merge (with its sign) or zero; the
    resolved names must be the final interaction's couplings."""
    from tools.loop_reference import madloop_ct_couplings

    recorder = madloop_ct_couplings.Recorder()
    recorder.original = {7: ["R2_a", "R2_b", "R2_c", "R2_d"]}
    recorder.renamed = {"R2_d": "R2_a"}
    recorder.merged = {"R2_b": ("GC_1", -1)}
    recorder.zero = {"R2_c"}
    interaction = {"id": 7, "couplings": {(0, 0): "R2_a", (0, 1): "-GC_1"}}
    assert recorder.resolve(interaction) == [
        ("R2_a", "R2_a"),
        ("R2_b", "-GC_1"),
        ("R2_d", "R2_a"),
    ]
    assert recorder.called("R2_b", particle=True) == "GC_1"
    with pytest.raises(RuntimeError, match="not the renamed"):
        recorder.resolve({"id": 7, "couplings": {(0, 0): "R2_a"}})


def test_every_uv_row_and_class_names_rule5_components(
    campaign: dict, manifest: dict
) -> None:
    """Rule 5 components on one row per class; every row tagged uv names the
    component files that serve it (its own runs', else its class's)."""
    files = {r["components"] for r in campaign["runs"] if r.get("components")}
    for row in campaign["rows"]:
        if "uv" in row["extras"]:
            assert row.get("components"), row["row"]
        assert set(row.get("components", ())) <= files, row["row"]
    classes = {row["class"] for row in campaign["rows"]}
    assert classes == {r["class"] for r in campaign["rows"] if r.get("components")}
    assert [r.get("components") for r in manifest["rows"]] == [
        r.get("components") for r in campaign["rows"]
    ]


@pytest.mark.parametrize(
    "run_id",
    [r["id"] for r in json.loads(CAMPAIGN.read_text())["runs"] if r.get("components")],
)
def test_components_add_up_to_the_reference(campaign: dict, run_id: str) -> None:
    """Rule 5 components (madloop_components.py): r2 + uv + rest = ct and bare +
    ct = the reference's sector value per coefficient; R2 is finite and the UV
    counterterms carry no double pole; every loop sector with a Born partner."""
    run = _runs(campaign)[run_id]
    record = json.loads((FIXTURES / run["components"]).read_text())
    fixture = _fixture(run_id)
    assert record["points"] == len(fixture["points"])
    partnered = {
        f"QCD^2={s['QCD']},QED^2={s['QED']}"
        for s in fixture["rows"][0]["sectors"]["loop"]
        if s["born_sector"] is not None
    }
    assert set(record["sectors"]) == partnered
    for key, sector in record["sectors"].items():
        assert sector["ct_amplitude_counts"]["rest"] == 0, key
        for index, row in enumerate(sector["rows"]):
            reference = next(
                s
                for s in fixture["rows"][index]["sectors"]["loop"]
                if f"QCD^2={s['QCD']},QED^2={s['QED']}" == key
            )
            scale = max(abs(reference[k]) for k in LAURENT)
            for k in LAURENT:
                assert row["total"][k] == reference[k]
                assert (
                    abs(row["r2"][k] + row["uv"][k] + row["rest"][k] - row["ct"][k])
                    <= 1e-12 * scale
                )
                assert (
                    abs(row["bare"][k] + row["ct"][k] - row["total"][k])
                    <= 1e-12 * scale
                )
            assert row["r2"]["single_pole"] == row["r2"]["double_pole"] == 0.0
            assert row["uv"]["double_pole"] == 0.0
