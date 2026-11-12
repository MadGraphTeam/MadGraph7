# SPDX-License-Identifier: 0BSD
"""The M-SM campaign manifest and the EW parts of the harness (m-sm/PLAN.md
§1.4, §3.2 MS-VALIDATE, §4): ``tests/fixtures/loop/msm/manifest.json``
(written by ``tools/loop_benchmark/msm_manifest.py``) holds PLAN §4.2's 70
core rows and MS-UFODB's SC10 in the M-QCD record type;
``tools/loop_benchmark/mqcd.py`` runs them with F_s per correction, rule 4's
typed unpredicted reasons, rule 5's EW component form, the R2 rule and the
EW reference formats. Fast: no generation."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_benchmark import mqcd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "fixtures" / "loop" / "msm" / "manifest.json"
PLAN_ROWS = {
    "LL": "01 02 03 05 08 09 10",
    "DY": "01 02 03 04 05 07 08",
    "VJ": "01 02 03",
    "VV": "01 02 03 05 06 07",
    "V3": "01",
    "VH": "01 02 03",
    "TT": "01 02 03 04 05 06",
    "TV": "01 02 03 06 07",
    "ST": "01 02 03 04",
    "JJ": "01 02 03 04",
    "PI": "01 02 03 04",
    "CM": "01 02 03 04 05 07 08 09 10",
    "MP": "01 02",
    "SC": "01 02 03 04 05 06 07 08 09",
}
# MS-UFODB's row (the R2_ZZHH vertex of UFOmodeldb 6ca4b120).
EXTRA_ROWS = {"SC10"}
# PLAN §4.3 tiers: VJ02 is the default tier, joined after MS-GEN by TT01
# (PLAN §3.2 item 3: generated in < 60 s; LL01 and DY01 are not); the 2->5
# target and CM07 run in the campaign tier.
DEFAULT_TIER = {"VJ02", "TT01"}
CAMPAIGN_TIER = {"TV07", "CM07"}
# PLAN §1.4: the known rows of MadLoop's R2 label matching (unpatched models).
R2_RULE_ROWS = {"LL10", "TT04", "SC09", "TT05"}


@pytest.fixture(scope="module")
def manifest():
    return mqcd.load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def rows(manifest):
    return mqcd.rows(manifest)


def _core(rows):
    return {name: row for name, row in rows.items() if row.base is None}


@pytest.mark.loop_fast
def test_rows_are_the_plan_rows(rows):
    expected = {f"{k}{n}" for k, ids in PLAN_ROWS.items() for n in ids.split()}
    assert len(expected) == 70
    core = _core(rows)
    assert set(core) == expected | EXTRA_ROWS
    assert {n for n, r in core.items() if r.tier == "default"} == DEFAULT_TIER
    assert {n for n, r in core.items() if r.tier == "campaign"} == CAMPAIGN_TIER


@pytest.mark.loop_fast
def test_rows_are_well_formed(manifest, rows):
    codes = manifest["restriction_codes"]
    models = {"loop_sm", *manifest["models"]}
    entries = {data["id"]: data for data in manifest["rows"]}
    for name, row in rows.items():
        assert row.tier in mqcd.TIERS, name
        assert row.status in ("active", "pending", "composite"), name
        if row.status == "composite":
            # A multiparticle row of its parts (MP01, MP02): MadLoop's P
            # directories, one subprocess variant per P reference.
            data = entries[name]
            assert row.reference is None and row.pending is None, name
            assert data["madloop_subprocesses"] and data["multiparticles"], name
            assert data["variants"] and all(v["subprocess"] for v in data["variants"])
            continue
        assert set(row.corrections) <= {"QCD", "QED"} and row.corrections, name
        code = codes[row.restriction_code]
        assert (row.model, row.restriction) == (code["model"], code["restriction"])
        assert row.model in models, name
        # The QED correction's loop coupling is the scheme's alpha.
        if "QED" in row.corrections:
            assert row.alphas["QED"] == code["alpha"], name
        if row.model != "loop_sm":
            assert row.compatibility == "gmu_particle_anti", name
            assert row.scale_parameter == "MU_R", name
        assert row.unpredicted_reasons is not None, name
        assert set(row.extras_pending) <= set(row.extras), name
        for extra in set(row.extras) - mqcd.IMPLEMENTED_EXTRAS - {"a0", "so"}:
            assert extra in row.extras_pending, (name, extra)
        assert row.points, name
        if row.status == "pending":
            waits = row.pending["waits_for"]
            waits = [waits] if isinstance(waits, str) else waits
            assert waits and set(waits) <= set(manifest["unblockers"]), name
        else:
            assert row.reference is not None and row.pending is None, name


@pytest.mark.loop_fast
def test_loop_orders_reach_the_row_sectors(manifest):
    """Every sector of a row is a Born amplitude order plus a loop amplitude
    order of its request, the loop orders two powers of a correction above
    a Born order; the expected predicted sectors are every sector the
    request forms (PLAN §4.2's list may name fewer: JJ04's QCD^2 QED^6)."""

    for data in manifest["rows"]:
        if not data["sectors"]:
            continue
        loops = data["loop_orders"]
        loops = [loops] if isinstance(loops, dict) else loops
        born = [mqcd.sector_key(b) for b in data["born_amplitude_orders"]]
        bound = mqcd.sector_key(data["born_orders"])
        for b in born:
            assert all(power <= dict(bound).get(k, 0) for k, power in b), data["id"]
        reached = set()
        for loop in loops:
            assert any(
                mqcd.born_sector(mqcd.sector_key(loop), c) in born
                for c in data["corrections"]
            ), data["id"]
            reached |= {
                mqcd.sector_key(
                    {k: dict(b).get(k, 0) + loop.get(k, 0) for k in ("QCD", "QED")}
                )
                for b in born
            }
        sectors = {mqcd.sector_key(s) for s in data["sectors"]}
        assert sectors <= reached, data["id"]
        predicted = {mqcd.sector_key(s) for s in data["predicted_sectors"]}
        unpredicted = {mqcd.sector_key(s) for s in data["unpredicted_sectors"]}
        assert predicted | unpredicted == reached, data["id"]


@pytest.mark.loop_fast
def test_joint_requests_predict_every_sector(rows):
    """PLAN §1.4: a joint QCD + QED request without amplitude-order caps has
    no unpredicted sector (one there is a finding, not a manifest entry)."""

    for name, row in rows.items():
        capped = "=" in row.process.replace("==", "")
        if set(row.corrections) == {"QCD", "QED"} and not capped:
            assert not row.unpredicted_sectors, name
            assert not row.unpredicted_reasons, name


@pytest.mark.loop_fast
def test_relaxed_tolerances_name_a_rule_and_stay_below_attention(rows):
    """Every relaxed tolerance names a rule the harness relaxes (rule 4 per
    carrier: ``"4"`` binary64, ``"4dd"`` double-double) and is at most
    ~1e-5 (the user's ruling of 2026-10-29 ~10:20)."""

    for name, row in rows.items():
        for rule, tolerance in row.relaxed.items():
            assert rule in ("2", "4", "4dd", "5", "6"), name
            assert 0 < tolerance <= mqcd.ATTENTION, name


@pytest.mark.loop_fast
def test_r2_rule_rows_compare_against_a_complete_insertion(rows):
    """PLAN §1.4 rule 1, the R2 rule: a row whose MadLoop export omits a UFO
    R2 coupling the engine inserts is compared against Recola (the primary
    provider) or against MadLoop plus the omitted component. The known rows
    of the unpatched models (R2_G0tt_b) are re-referenced on the shipped
    patched models (MS-UFO-PATCH), whose MadLoop exports call V_ttG02's
    split b-loop R2 couplings: none lists an omission, and the harness
    finds the set structurally on every row with a call list."""

    for name, row in _core(rows).items():
        omitted = row.madloop_omitted_r2
        if name in R2_RULE_ROWS and row.status == "active":
            assert not omitted, name
            called = set(row.madloop_r2_called or ())
            assert any(c.startswith("R2_G0tt_b_") for c in called), name
        if omitted:
            reference = mqcd.load_reference(row)
            assert reference.primary != "madloop" or row.reference.get("omitted_r2")


@pytest.mark.loop_fast
def test_r2_rule_finds_madloop_omissions_structurally(loop_gmu_ufodb_unpatched_ufo):
    """PLAN §1.4 rule 1, the R2 rule found structurally: MG5_aMC's export
    merges couplings of equal value under one name (V_R2AWW's R2_AWWud and
    R2_AWWtb under R2_AWWcs) and drops vanishing ones (R2_AAu3 ~ MU^2), so an
    inserted coupling counts as called when its value class is; R2_G0tt_b
    (V_ttG02's b-loop R2, whose propagator label MadLoop's matching never
    meets) is the omission; a row that lists as omitted a coupling MadLoop
    calls, or does not list one it omits, fails the structural check. On
    the unpatched G_mu UFO (UFOmodeldb 6ca4b120), which still has the
    single R2_G0tt_b coupling."""

    from types import SimpleNamespace as N

    names = ["R2_AWWcs", "R2_AWWud", "R2_AWWtb", "R2_AAu3", "R2_G0tt", "R2_G0tt_b"]
    classes = mqcd.r2_value_classes(loop_gmu_ufodb_unpatched_ufo, "default", names)
    assert classes["R2_AWWcs"] == classes["R2_AWWud"] == classes["R2_AWWtb"]
    assert classes["R2_AWWcs"] is not None and classes["R2_AAu3"] is None
    assert None not in (classes["R2_G0tt"], classes["R2_G0tt_b"])
    assert classes["R2_G0tt"] != classes["R2_G0tt_b"]

    def entry(vertex, coupling):
        return ("x", N(vertex_name=vertex, coupling=N(name=coupling)))

    rules = {
        "aww": N(kind="R2", couplings=[entry("V_R2AWW", "R2_AWWud"),
                                       entry("V_R2AWW", "R2_AWWtb")]),
        "aa": N(kind="R2", couplings=[entry("V_R2AA", "R2_AAu3")]),
        "g0": N(kind="R2", couplings=[entry("V_ttG0", "R2_G0tt"),
                                      entry("V_ttG02", "R2_G0tt_b")]),
    }  # fmt: skip
    admitted = dict.fromkeys(rules, True)
    recursion = N(rules=rules, admission=admitted, inert=set())
    slice_ = N(counterterm_program=N(recursion=recursion))
    reference = N(primary="recola")
    row = N(
        madloop_omitted_r2=("R2_G0tt_b",), madloop_r2_called=("R2_AWWcs", "R2_G0tt")
    )
    engine = mqcd.engine_r2(slice_, lambda rule: True)
    checks = mqcd.rule_r2(row, slice_, reference, classes, engine)
    assert [c.coefficient for c in checks] == ["inserted", "omitted"]
    assert all(c.passed for c in checks)
    # A listed coupling MadLoop calls is no omission.
    called = N(madloop_omitted_r2=("R2_G0tt",), madloop_r2_called=row.madloop_r2_called)
    inserted, omitted = mqcd.rule_r2(called, slice_, reference, classes, engine)
    assert inserted.passed and not omitted.passed and omitted.value == "R2_G0tt_b"
    # An unlisted omission fails: the set found is the listed set.
    unlisted = N(madloop_omitted_r2=(), madloop_r2_called=row.madloop_r2_called)
    (omitted,) = mqcd.rule_r2(unlisted, slice_, reference, classes, engine)
    assert not omitted.passed and omitted.value == "R2_G0tt_b"
    # A vertex no tree of the request can hold (``mqcd.r2_reachability``)
    # inserts nothing, so its couplings are no omission.
    unreached = mqcd.engine_r2(slice_, lambda rule: rule is not rules["g0"])
    (omitted,) = mqcd.rule_r2(unlisted, slice_, reference, classes, unreached)
    assert omitted.passed and omitted.value == "none"
    # An insertion listed as vanishing in the row's sectors is no omission;
    # a listed one the difference does not hold fails.
    zero = N(madloop_omitted_r2=(), madloop_r2_called=row.madloop_r2_called,
             r2_vanishing=("R2_G0tt_b",))  # fmt: skip
    (omitted,) = mqcd.rule_r2(zero, slice_, reference, classes, engine)
    assert omitted.passed and omitted.value == "none"
    (omitted,) = mqcd.rule_r2(zero, slice_, reference, classes, unreached)
    assert not omitted.passed


@pytest.mark.loop_fast
def test_active_references_load_at_their_points(rows):
    for name, row in rows.items():
        if row.status != "active":
            continue
        reference = mqcd.load_reference(row)
        indices = {i for points in row.points.values() for i in points}
        assert indices <= set(range(len(reference.points))), name
        assert len(reference.scales) == len(reference.points), name
        if "uv" in row.extras and "uv" not in row.extras_pending:
            assert all(reference.components), name


@pytest.mark.loop_fast
def test_delegated_columns_name_existing_tests(rows):
    """A delegated column (rule 5 after MS-POLES on DY01 and TT01: the
    counterterms' UV part at epsIR = 0 against MadLoop, MS-POLES's tests)
    names test functions that exist; build_mqcd.py reads their JUnit
    outcomes."""

    delegated = {name: row.delegated for name, row in rows.items() if row.delegated}
    assert set(delegated) >= {"DY01", "TT01"}
    for name, columns in delegated.items():
        for node in (node for ids in columns.values() for node in ids):
            path, function = node.split("::")
            source = (mqcd.ROOT / path).read_text()
            assert f"\ndef {function}(" in source, (name, node)


@pytest.mark.loop_fast
def test_tv07_is_the_target_on_native_inputs(rows):
    """EWI-9 (PLAN §3.2 item 4): TV07 at 3 points against gg_ttxemepg_ew with
    native inputs, rule 4 from the engine's operators; its joint twin's
    s^8e^4 sector against gg_ttxemepg_qcd at loop_sm's derived MW."""

    target, twin = rows["TV07"], rows["TV07@qcd"]
    assert target.inputs == "native" and target.catani == "engine"
    assert target.tier_points("campaign") == (0, 1, 2)
    assert {mqcd.sector_text(s) for s in target.predicted_sectors} == {
        "QCD^6 QED^6",
        "QCD^8 QED^4",
    }
    ours = mqcd.load_reference(target)
    assert set(ours.virtual[0]["madloop"]) == {mqcd.sector_key({"QCD": 6, "QED": 6})}
    theirs = mqcd.load_reference(twin)
    assert set(theirs.virtual[0]["madloop"]) == {mqcd.sector_key({"QCD": 8, "QED": 4})}
    assert twin.parameters["MW"].startswith("80.4190024457")


# ------------------------------------------------------------- the rules


def _row(**changes):
    data = {
        "id": "X01",
        "class": "W-DY",
        "process": "u u~ > e- e+ g",
        "born_orders": {"QCD": 1, "QED": 2},
        "loop_orders": [{"QCD": 3, "QED": 2}, {"QCD": 1, "QED": 4}],
        "corrections": ["QCD", "QED"],
        "restriction": "default",
        "restriction_code": "G0",
        "tier": "nightly",
        "status": "active",
        "reference": None,
        "points": {"nightly": [0]},
        "predicted_sectors": [],
        "unpredicted_sectors": [],
        "alphas": {"QED": 0.0075},
        "unpredicted_reasons": {},
        **changes,
    }
    return mqcd.Row.from_dict(data)


def _born(**sectors):
    return {mqcd.sector_key(dict(s)): Decimal(v) for s, v in sectors.items()}


@pytest.mark.loop_fast
def test_flux_takes_the_largest_correction_reaching_the_sector():
    """F_s = max_c |B_{s - 2c}| alpha_c / 2 pi (PLAN §1.4)."""

    born = {
        mqcd.sector_key({"QCD": 2, "QED": 4}): Decimal(3),
        mqcd.sector_key({"QCD": 0, "QED": 6}): Decimal(5),
    }
    point = mqcd.Evaluated("double-double", born, {}, None, None)
    couplings = {"QCD": 0.118, "QED": 0.0075}
    row = _row()
    # s^2e^6: QCD from s^0e^6 (5 x 0.118), QED from s^2e^4 (3 x 0.0075).
    sector = mqcd.sector_key({"QCD": 2, "QED": 6})
    flux = mqcd.sector_flux(row, point, sector, couplings)
    assert flux == pytest.approx(5 * 0.118 / (2 * math.pi))
    # A QED-only row: the QED Born and alpha.
    qed = _row(corrections=["QED"])
    assert mqcd.sector_flux(qed, point, sector, couplings) == pytest.approx(
        3 * 0.0075 / (2 * math.pi)
    )
    # The total: the summed Born times the largest alpha.
    assert mqcd.sector_flux(row, point, mqcd.TOTAL, couplings) == pytest.approx(
        8 * 0.118 / (2 * math.pi)
    )
    # An M-QCD row keeps |B_{s-2}| alpha_s / 2 pi.
    qcd = _row(corrections=["QCD"], alphas={})
    born = {mqcd.sector_key({"QCD": 4}): Decimal(2)}
    point = mqcd.Evaluated("double-double", born, {}, None, None)
    assert mqcd.sector_flux(
        qcd, point, mqcd.sector_key({"QCD": 6}), {"QCD": 0.118}
    ) == pytest.approx(2 * 0.118 / (2 * math.pi))


@pytest.mark.loop_fast
def test_rule4_asserts_the_typed_reasons():
    sector = mqcd.sector_key({"QCD": 4, "QED": 4})
    row = _row(unpredicted_reasons={"QCD^4 QED^4": "separated-exchange"})
    (check,) = mqcd.rule4_reasons(row, {sector: "separated-exchange"})
    assert check.passed
    (check,) = mqcd.rule4_reasons(row, {sector: "uncorrected-exchange"})
    assert not check.passed
    (check,) = mqcd.rule4_reasons(row, {})
    assert not check.passed
    # An unlisted reason fails even when the manifest names it.
    odd = _row(unpredicted_reasons={"QCD^4 QED^4": "no-reason"})
    (check,) = mqcd.rule4_reasons(odd, {sector: "no-reason"})
    assert not check.passed
    # M-QCD rows (no reasons in the manifest) check the sets only.
    assert mqcd.rule4_reasons(_row(unpredicted_reasons=None), {sector: "x"}) == []


@pytest.mark.loop_fast
def test_photon_conversion_is_mg5s_formula():
    """VJ03's reference: raw MadLoop G_mu x (alpha(0)/alpha_Gmu)^n, the
    virtual of the electromagnetic correction minus 2 n DGmua0 Born
    (``test_loop_v2_external_photons``'s formula, MadLoop's DGmua0 at MU_R =
    1000 GeV); with MS-REF's top-up the QCD sector is scaled only."""

    row = mqcd.rows(mqcd.load_manifest(MANIFEST))["VJ03"]
    reference = mqcd.load_reference(row)
    parts = row.reference.get("parts") or [row.reference]
    raw = json.loads((ROOT / parts[0]["path"]).read_text())["rows"][0]
    madloop = raw["providers"]["madloop"]
    values = json.loads(
        (ROOT / "tests/fixtures/loop/madloop-ct-parameters.json").read_text()
    )["models"]["gmu"]["values"]["eCoup_DGmua0_UV_EW"]
    pole, finite = float(values["-1"]["double"][0]), float(values["0"]["double"][0])
    alpha = float(Fraction("0.00754677226134035151919702202521"))
    r = 1 / 137.035999 / alpha
    born = sum(reference.born[0]["madloop"].values())
    virtual = reference.virtual[0]["madloop"]
    electroweak = mqcd.sector_key({"QCD": 2, "QED": 4})
    laurent = virtual.get(electroweak, virtual.get(mqcd.TOTAL))
    assert born == pytest.approx(r * madloop["born"], rel=1e-15)
    assert laurent[0] == pytest.approx(
        r * (madloop["finite"] - 2 * finite * madloop["born"]), rel=1e-14
    )
    assert laurent[1] == pytest.approx(
        r * (madloop["single_pole"] - 2 * pole * madloop["born"]), rel=1e-14
    )
    for part in parts[1:]:
        topup = json.loads((ROOT / part["path"]).read_text())
        values = topup["rows"][part["rows"][0]]["providers"]["madloop"]
        laurent = virtual[mqcd.sector_key(part["sector"])]
        for k, name in enumerate(mqcd.LAURENT):
            assert laurent[k] == pytest.approx(r * values[name], rel=1e-15)


@pytest.mark.loop_fast
def test_rule5_components_compare_each_component():
    """Rule 5's EW form: bare, R2, UV and their CT sum per sector against the
    provider's, relative to max(|a|, |r|, F_s)."""

    sector = mqcd.sector_key({"QED": 6})
    born = {mqcd.sector_key({"QED": 4}): Decimal(1)}
    laurents = {
        "bare": (Decimal("0.5"), Decimal("-0.25"), Decimal("0.125")),
        "r2": (Decimal("0.0625"), Decimal(0), Decimal(0)),
        "uv": (Decimal("0.03125"), Decimal("0.25"), Decimal(0)),
    }
    components = {name: {sector: value} for name, value in laurents.items()}
    dd = mqcd.Evaluated("double-double", born, {}, None, None, 0.0, components)
    reference = mqcd.Reference(
        "", ("madloop",), [], [], {}, None, [], [], [], 0.118,
        components=[{
            "bare": {mqcd.TOTAL: (0.5, -0.25, 0.125)},
            "r2": {mqcd.TOTAL: (0.0625, 0.0, 0.0)},
            "uv": {mqcd.TOTAL: (0.03125, 0.25, 0.0)},
            "ct": {mqcd.TOTAL: (0.09375, 0.25, 0.0)},
        }],
    )  # fmt: skip
    row = _row(corrections=["QED"])
    checks = mqcd.rule5_components(row, 0, dd, reference, {"QCD": 0.118, "QED": 0.0075})
    assert len(checks) == 12 and all(c.passed for c in checks)
    reference.components[0]["r2"][mqcd.TOTAL] = (0.0626, 0.0, 0.0)
    failed = [
        c
        for c in mqcd.rule5_components(row, 0, dd, reference, {"QED": 0.0075})
        if not c.passed
    ]
    assert {c.coefficient for c in failed} == {"r2 finite"}


@pytest.mark.loop_fast
def test_nightly_jobs_cover_the_tier_within_the_budget(manifest):
    """Every nightly id (an active gating row or variant with nightly points
    or a checked extra) is in exactly one file job, each job within its
    budget, each with its test file and no file without a job; the same for
    the restriction smokes."""

    ids = mqcd.nightly_ids(manifest)
    jobs = manifest["nightly_jobs"]["jobs"]
    listed = [name for job in jobs for name in job["rows"]]
    assert sorted(listed) == sorted(ids)
    budget = manifest["nightly_jobs"]["budget_seconds"]
    assert budget <= 1800 and all(job["seconds"] <= budget for job in jobs)
    files = {job["file"] for job in jobs} | {
        job["file"] for job in manifest["smokes"]["jobs"]
    }
    found = {
        str(path.relative_to(ROOT))
        for pattern in ("test_loop_v2_msm_nightly*.py", "test_loop_v2_msm_smokes*.py")
        for path in (ROOT / "tests" / "unit").glob(pattern)
    }
    assert found == files
    smokes = [n for job in manifest["smokes"]["jobs"] for n in job["smokes"]]
    assert sorted(smokes) == sorted(e["id"] for e in manifest["smokes"]["restrictions"])
