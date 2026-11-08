# SPDX-License-Identifier: 0BSD
"""The loop-induced validation campaign's run matrix, rules and data/lit.json
(m-lit PLAN §3.4 LIT-VAL; ``tools/loop_benchmark/li_manifest.py``). Fast: no
generation, no evaluation (synthetic records)."""

from __future__ import annotations

import json
import tomllib
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_benchmark import li_manifest as lm

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def entries():
    return lm.runs()


def test_every_campaign_run_is_one_entry_of_its_rows(entries):
    """Every reference run of a §4.1 row (lit-ref's manifest) is run once,
    with every row id, tier and check that lists it; the delegated rows (the
    TIR unit row, the Born-interfered backend row, the EFT and observable
    rows) are other packages'."""

    campaign = json.loads(lm.CAMPAIGN.read_text())
    by_fixture = {e["fixture"]: e for e in entries}
    assert len(by_fixture) == len(entries)
    listed = set()
    for row in campaign["rows"]:
        for run in row["runs"]:
            if row["row"] in lm.DELEGATED:
                continue
            entry = by_fixture[run["process"]]
            listed.add(run["process"])
            assert set(lm._ids(row["row"])) <= set(entry["ids"])
            assert row["tier"] in entry["tiers"]
            assert set(row["checks"]) <= set(entry["checks"])
            for provider in run.get("excluded", {}):
                assert provider not in entry["providers"]
    # This package's additional fixtures (the P15 point of LI-41) run too.
    extra = {path.stem for path in lm.EXTRA.glob("*.json")}
    assert "li_gg_hhgg_p15" in extra
    assert by_fixture["li_gg_hhgg_p15"]["ids"] == ["LI-41"]
    assert by_fixture["li_gg_hhgg_p15"]["key"] == "gg_hhgg"
    assert listed | extra == set(by_fixture)
    assert lm._ids("LI-W1/L1/P1/C1/S1") == ["LI-W1", "LI-L1", "LI-P1", "LI-C1", "LI-S1"]


def test_registry_keys_and_ap_runs(entries):
    """Every registry row with a process has its runs (by its ids and
    process); the complex-mass and zero runs are validation-only; one AP run
    per tier, the run of the tier with the fewest legs."""

    registry = lm.registry()
    keyed = lm.registry_runs(entries, registry)
    for row in registry["rows"]:
        if row.get("process"):
            assert keyed[row["key"]], row["key"]
        else:
            assert not keyed[row["key"]]
    assert set(keyed["gg_h"]) == {"li_gg_h", "li_gg_h_tb"}
    assert set(keyed["gg_aa"]) == {"li_gg_aa", "li_gg_aa_nf5"}
    by_fixture = {e["fixture"]: e for e in entries}
    assert by_fixture["li_gg_aa_cms"]["key"] is None
    assert by_fixture["li_gg_z"]["key"] is None
    tiers = {t for e in entries for t in e["tiers"]}
    for tier in tiers:
        (ap,) = [e for e in entries if tier in e["ap"]]
        assert ap["legs"] == min(e["legs"] for e in entries if tier in e["tiers"])


def test_cards_read_the_fixture(tmp_path):
    """The generation card states the fixture's model, correction, order
    limit, veto, the fixed parameters the model declares and the mass
    scheme; a UFO source gets its loading options."""

    def parsed(name, known=None):
        data = lm.fixture(name)
        fixed = set(data["scheme_data"]["fixed_parameters"])
        text = lm.card(data, tmp_path / "out", fixed if known is None else known, "P")
        return data, tomllib.loads(text)

    data, card = parsed("li_uux_ddxh")
    assert card["model"] == {"source": "loop_sm", "restriction": "no_b_mass"}
    assert card["process"]["max_coupling_orders"] == {"QED": 1}
    assert card["process"]["loop_coupling_orders"] == ["QCD"]
    assert (
        card["loop"]["preparation"]["fixed_parameters"]
        == data["scheme_data"]["fixed_parameters"]
    )
    _, card = parsed("li_gg_aa_nf5")
    assert card["generation"]["particle_veto"] == ["t"]
    _, card = parsed("li_gg_h_tb")
    assert card["model"]["restriction"] == "default"
    data, card = parsed("li_gg_zz_cms", known={"WT", "WW"})
    assert card["loop"]["model"]["scheme"] == {"mass_scheme": "complex-mass"}
    assert card["loop"]["preparation"]["compatibility"] == "gmu_particle_anti"
    assert card["loop"]["external_photons"] == "none"
    assert set(card["loop"]["preparation"]["fixed_parameters"]) == {"WT", "WW"}
    assert Path(card["model"]["source"]).name == "loop_qcd_qed_sm_Gmu_py3"
    _, card = parsed("li_h_aa")
    assert card["process"]["loop_coupling_orders"] == ["QED"]


@pytest.mark.parametrize("name", ["li_gg_hh", "li_gg_zzg", "li_h_aa"])
def test_the_identity_transform_is_exact(name):
    """The identity check's Lorentz transform keeps every fixture point an
    exact decimal point, on shell and momentum conserving, and moves it."""

    data = lm.fixture(name)
    initial = data["initial"]
    for point in data["points"]:
        moved = lm.lorentz(point)
        exact = [[Fraction(c) for c in p] for p in point]
        image = [[Fraction(c) for c in p] for p in moved]
        assert image != exact

        def square(p):
            return p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2

        for a, b in zip(exact, image, strict=True):
            assert square(a) == square(b)
        for k in range(4):
            assert sum(p[k] for p in image[:initial]) == sum(
                p[k] for p in image[initial:]
            )


def _point(value, error, *, status="success", finiteness=1e-15, pole=2e-15):
    return {
        "status": status,
        "detail": ""
        if status == "success"
        else "box [0, 1, 2, 3] refused at conditioning 1.6e-5",
        "reduction": None,
        "finiteness": finiteness if status == "success" else None,
        "values": None
        if status != "success"
        else [{"finite": repr(value), "single_pole": "1e-18", "double_pole": "0.0"}],
        "estimate": None
        if status != "success" or error is None
        else {
            "relative": error / value,
            "finite": error / value,
            "single_pole": pole,
            "double_pole": pole,
            "scale": pole,
            "absolute": [{"finite": error, "single_pole": 1e-15, "double_pole": 1e-15}],
            "checks": [],
        },
    }


def _record(name):
    """A synthetic record of fixture ``name`` (two of its points): TIRust
    serves point 0 and fails point 1 closed; double-double TIRust refused."""

    data = lm.fixture(name)
    v = [row["providers"]["madloop"]["value"] for row in data["rows"]]
    count = len(data["points"])
    near = [x * (1 + 1e-12) for x in v]
    n64 = {"points": [_point(x, 1e-12 * x) for x in near]}
    t64 = {
        "points": [_point(near[0] * (1 + 2e-13), None)]
        + [_point(0, None, status="failed")] * (count - 1)
    }
    t64c = {"points": [_point(near[0] * (1 + 2e-13), 1e-12 * v[0]), *t64["points"][1:]]}
    dd = {"points": [_point(x, 1e-30 * x, finiteness=1e-31, pole=1e-31) for x in v]}
    refused = {
        "refused": "ReductionUnavailable: TIRust does not serve the double-double carrier"
    }
    return {
        "fixture": name,
        "generation": {
            "probes": {
                "finiteness": {"passed": True, "worst": 0.1},
                "ward": {"legs": [1], "passed": True, "worst": 0.2},
                "precision": "double-double",
                "widths": "zero",
            }
        },
        "digits": None,
        "arms": {
            "ninjars/double/plain": n64,
            "ninjars/double/checked": n64,
            "tirust/double/plain": t64,
            "tirust/double/checked": t64c,
            "ninjars/double-double/checked": dd,
            "tirust/double-double/checked": refused,
            "identity/ninjars/double": n64,
            "identity/ninjars/double-double": dd,
        },
        "l3_probes": {
            "ninjars/double": [{"worst": 0.1, "passed": True}] * count,
            "ninjars/double-double": [{"worst": 0.05, "passed": True}] * count,
            "tirust/double": [{"worst": 0.2, "passed": True}]
            + [{"error": "FailedPoints: box [0, 1, 2, 3] refused"}] * (count - 1),
        },
        "complete": True,
    }


# The capability query of an extension with TIRust at every carrier (lit-tir-mp).
AFTER_TIR_MP = {
    "ninjars": ["arbitrary", "double", "double-double"],
    "tirust": ["arbitrary", "double", "double-double"],
}


def _after_tir_mp(record):
    """``record`` as taken by an extension serving TIRust at every carrier."""

    record["reduction_backends"] = AFTER_TIR_MP
    return record


def test_judge_records_partial_waits_and_passes(entries):
    """§4.0 on a synthetic record: L1 against the designated providers, L3
    finiteness against the pole estimate, L4 from the generation probes, L6
    through the one comparison on the points TIRust serves (a failed point
    makes it partial with TIRust's reason), the double-double arm waits on
    lit-tir-mp with TIRust's typed refusal."""

    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    judged = lm.judge(_record("li_gg_hh"), entry)
    assert judged["L1"]["status"] == "pass"
    assert set(judged["L1"]["max_relative"]) == set(entry["providers"])
    assert judged["L3"]["ninjars/double"]["status"] == "pass"
    assert judged["L3"]["tirust/double"]["status"] == "partial"
    assert judged["L3"]["tirust/double"]["served"] == 1
    # L4 without the fixture-point Ward probes: the generation-time ones
    # only (one point, the probed legs) are partial.
    assert judged["L4"]["status"] == "partial"
    assert judged["L4"]["generation"]["probes"]["ward"]["legs"] == [1]
    assert judged["L4"]["fixture_points"]["status"] == "not run"
    assert judged["L5"]["status"] == "pass"
    l6 = judged["L6"]
    assert l6["status"] == "partial" and l6["double"]["status"] == "partial"
    assert l6["waits_on"] == lm.TIR_MP
    assert l6["double"]["tirust_served"] == 1 and l6["double"]["of"] == 3
    assert l6["double"]["beyond_10E"] == 0 and l6["double"]["compared_entries"] == 3
    assert any("conditioning" in k for k in l6["double"]["tirust_failed_reasons"])
    assert l6["double-double"]["status"] == f"waits: {lm.TIR_MP}"
    assert "double-double" in l6["double-double"]["tirust"]
    # A probe residue beyond ten times its own estimate fails L3.
    record = _record("li_gg_hh")
    record["l3_probes"]["ninjars/double"] = [{"worst": 12.0, "passed": False}] * 3
    assert lm.judge(record, entry)["L3"]["ninjars/double"]["status"] == "fail"
    # A zero row's providers are recorded absolutely (L8 is its rule).
    zero = {e["fixture"]: e for e in entries}["li_gg_ha"]
    judged = lm.judge(_record("li_gg_ha"), zero)
    assert judged["L1"]["status"] == "recorded" and "L8" in judged


def test_arms_not_run_keep_their_reason(entries):
    """An arm recorded not run (``run --skip-arm``) keeps its reason: the AP
    arm makes L6 / L7 in AP 'not run', a skipped identity double-double arm
    leaves L5 on binary64 with the reason beside it, a skipped L3 probe arm
    makes L3 partial; ``LABEL=REASON`` arguments parse or are refused."""

    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    record = _after_tir_mp(_record("li_gg_hh"))
    for carrier in ("double", "double-double"):
        record["l3_probes"][f"tirust/{carrier}"] = [{"worst": 0.2, "passed": True}] * 3
    assert lm.judge(record, entry)["L3"]["status"] == "pass"
    reason = "the AP arm extrapolates to hours"
    record["arms"][f"ninjars/{lm.ARBITRARY}/plain"] = {"not_run": reason}
    record["arms"]["identity/ninjars/double-double"] = {"not_run": "slow"}
    record["l3_probes"]["ninjars/double-double"] = {"not_run": "slower"}
    judged = lm.judge(record, entry)
    assert judged["L7"] == {"status": "not run", "reason": reason}
    assert judged["L6"][lm.ARBITRARY]["status"] == "not run"
    assert judged["L5"]["status"] == "pass"
    assert judged["L5"]["double_double_not_run"] == "slow"
    assert judged["L5"]["double_double_max_relative"] is None
    assert judged["L3"]["ninjars/double-double"] == {
        "status": "not run",
        "reason": "slower",
    }
    assert judged["L3"]["status"] == "partial"
    assert lm.skip_arms(["l3/ninjars/double-double=slow", "a=b=c"]) == {
        "l3/ninjars/double-double": "slow",
        "a": "b=c",
    }
    with pytest.raises(ValueError, match="LABEL=REASON"):
        lm.skip_arms(["ninjars/arbitrary/plain"])
    # A run with plain double-double arms (``--high-plain``) is judged on them.
    plain = _record("li_gg_hh")
    for backend in lm.BACKENDS:
        arms = plain["arms"]
        arms[f"{backend}/double-double/plain"] = arms.pop(
            f"{backend}/double-double/checked"
        )
    judged = lm.judge(plain, entry)
    assert judged["E"]["status"] == "pass" and judged["E"]["ninjars"]["points"] == 3
    assert judged["L6"]["double-double"]["status"] == f"waits: {lm.TIR_MP}"


def test_lit_json_is_keyed_by_the_registry_and_read_by_the_report(entries, tmp_path):
    """data/lit.json: rows keyed by the registry keys in the LI_VALIDATION
    schema (status, references, points, note, source), the validation-only
    ids under ``ids``; the report builder's loop-induced section reads it."""

    import importlib.util

    records = {"li_gg_hh": _record("li_gg_hh")}
    out = lm.lit(records, entries, "test")
    row = out["rows"]["gg_hh"]
    assert row["status"] == "partial"
    assert set(row["references"]) == {"madloop", "openloops", "recola"}
    assert row["points"] == 3 and row["source"] == "test"
    assert row["note"].startswith(f"WAITS ON {lm.TIR_MP}") and row["waits_on"] == [
        lm.TIR_MP
    ]
    assert "TIRust fails 2 of 3 binary64 points closed" in row["note"]
    assert out["rows"]["gg_zz"]["status"] == "not run"
    assert "LI-24" in out["ids"] and "LI-A7" in out["ids"]
    # each run's checks once: an id's run a registry row carries names it
    assert out["ids"]["LI-24"]["runs"]["li_gg_hh"] == {
        "status": row["status"],
        "row": "gg_hh",
    }
    ward = out["ids"]["LI-W1"]["runs"]["li_gg_hh"]
    assert ward["row"] == "gg_hh" and "status" in ward
    path = (
        lm.WT / "docs" / "performance_reports" / "loop" / "scripts" / "build_summary.py"
    )
    spec = importlib.util.spec_from_file_location("build_summary_lit", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    files = {builder.LI_TIMING: None, builder.LI_VALIDATION: out}
    registry = {r["key"]: r for r in lm.registry()["rows"]}
    cells = builder._li_cells(registry["gg_hh"], False, files, {}, None, {}, "tirust")
    validation = cells["validation"]
    assert validation["status"] == "partial" and validation["points"] == 3
    assert next(r["key"] for r in validation["references"]) in row["references"]


def test_stress_fixtures_are_exact_and_current(entries):
    """PLAN §4.1's stress points and LI-Z1's exact colour zeros: the
    committed derived fixtures are what ``derived_fixtures`` builds from the
    specification (deterministic), every stress point exact (terminating
    decimals, on the mass shells, momentum conserving; the n-body kinds at
    their target sqrt(s) to ~1e-12, the planar kind's out-of-plane
    components scaled by the value), validation-only runs with the AP
    carrier; §4.1's kinds are all there (gg>VV pT on Z Z and W+ W-, the
    gg>hh threshold and forward, 2 m_t, gg>aa / ag forward, planar 2->3,
    several on-shell loop propagators under the complex-mass scheme)."""

    by_fixture = {e["fixture"]: e for e in entries}
    for data in lm.derived_fixtures():
        assert lm.fixture(data["name"]) == json.loads(json.dumps(data))
    kinds = set()
    for entry in lm.stress_entries():
        base = lm.fixture(entry["base"])
        stress = lm.fixture(lm.stress_name(entry))
        kinds.add((stress["madloop_process"].split("[")[0].strip(), entry["kind"]))
        assert stress["points"] == lm.stress_points(base, entry)
        assert stress["scheme_data"]["parameters"] == base["scheme_data"]["parameters"]
        initial = stress["initial"]
        shells = None
        for point in stress["points"]:
            p = [[Fraction(c) for c in m] for m in point]
            for k in range(4):
                assert sum(m[k] for m in p[:initial]) == sum(m[k] for m in p[initial:])
            squares = [m[0] ** 2 - m[1] ** 2 - m[2] ** 2 - m[3] ** 2 for m in p]
            assert squares[:initial] == [0] * initial
            assert shells is None or squares == shells
            shells = squares
        if "masses" in entry:
            assert shells[initial:] == lm._stress_masses(base, entry)
        if entry["kind"] == "planar":
            # each leg's out-of-plane over in-plane transverse component
            # scales with the value (5-point Gram ~ value^2)
            slopes = [
                [
                    Fraction(m[2]) / Fraction(m[1]) / Fraction(v)
                    for m in q[initial:]
                    if Fraction(m[1])
                ]
                for q, v in zip(stress["points"], entry["values"], strict=True)
            ]
            assert all(s == slopes[0] for s in slopes)
        e = by_fixture[stress["name"]]
        assert e["key"] is None and e["tiers"] == ["stress"]
        assert lm.ARBITRARY in stress["carriers"]
    assert {
        ("g g > z z", "pt"),
        ("g g > w+ w-", "pt"),
        ("g g > h h", "threshold"),
        ("g g > h h", "forward"),
        ("g g > a a", "forward"),
        ("g g > a g", "forward"),
        ("g g > h h", "top-threshold"),
        ("g g > h g g", "planar"),
        ("g g > z w+ w-", "top-threshold"),
    } <= kinds
    cms = lm.fixture("li_stress_gg_zwpwm_cms_top-threshold")
    assert cms["scheme_data"]["complex_mass_scheme"]
    assert cms["scheme_data"]["fixed_parameters"]["WW"] == "0"
    assert cms["scheme_data"]["fixed_parameters"]["WZ"] == "0"
    assert cms["normalization"]["denominator"] == 256
    # LI-Z1: validation-only exact colour zeros, never a tier's AP run.
    for name in ("li_h_ga", "li_z_ga"):
        e = by_fixture[name]
        assert e["ids"] == ["LI-Z1"] and e["exact_zero"] and not e["ap"]


def test_high_precision_backend_arm_is_ready_for_lit_tir_mp(entries):
    """Once TIRust serves double-double (lit-tir-mp), the same judge
    compares the arms through the one comparison within 1e-24 of the
    sector's largest coefficient; a deviation below ~1e-5 is recorded as
    lax, never a failure."""

    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    record = _record("li_gg_hh")
    dd = record["arms"]["ninjars/double-double/checked"]
    record["arms"]["tirust/double-double/checked"] = json.loads(json.dumps(dd))
    assert lm.judge(record, entry)["L6"]["double-double"]["status"] == "pass"
    moved = json.loads(json.dumps(dd))
    for point in moved["points"]:
        value = float(point["values"][0]["finite"])
        point["values"][0]["finite"] = repr(value * (1 + 1e-9))
    record["arms"]["tirust/double-double/checked"] = moved
    assert lm.judge(record, entry)["L6"]["double-double"]["status"] == "pass (lax)"


def test_a_failing_high_precision_arm_fails_the_row(entries):
    """L6's verdict is the worst of its carriers: a TIRust double-double /
    AP arm beyond ~1e-5 fails L6, the run and the row, though binary64
    passes; a silent backend switch fails L6 too."""

    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    record = _after_tir_mp(_record("li_gg_hh"))
    record["l3_probes"]["tirust/double"] = [{"worst": 0.2, "passed": True}] * 3
    record["arms"]["tirust/double/plain"] = json.loads(
        json.dumps(record["arms"]["ninjars/double/plain"])
    )
    record["arms"]["tirust/double/checked"] = json.loads(
        json.dumps(record["arms"]["ninjars/double/checked"])
    )
    moved = json.loads(json.dumps(record["arms"]["ninjars/double-double/checked"]))
    for point in moved["points"]:
        point["values"][0]["finite"] = repr(float(point["values"][0]["finite"]) * 1.001)
    record["arms"]["tirust/double-double/checked"] = moved
    record["arms"][f"tirust/{lm.ARBITRARY}/plain"] = moved
    record["arms"][f"ninjars/{lm.ARBITRARY}/plain"] = json.loads(
        json.dumps(record["arms"]["ninjars/double-double/checked"])
    )
    judged = lm.judge(record, entry)
    assert judged["L6"]["double"]["status"] == "pass"
    assert judged["L6"]["double-double"]["status"] == "fail"
    assert judged["L6"][lm.ARBITRARY]["status"] == "fail"
    assert judged["L6"]["status"] == "fail" and lm._run_status(judged) == "fail"
    assert (
        lm.lit({"li_gg_hh": record}, entries, "test")["rows"]["gg_hh"]["status"]
        == "fail"
    )
    # A served TIRust lane whose effective backend is NinjaRS: never compared.
    switched = _record("li_gg_hh")
    switched["arms"]["tirust/double/plain"]["points"][0]["reduction"] = [
        "tirust",
        "ninjars",
        "fallback",
    ]
    judged = lm.judge(switched, entry)
    assert judged["L6"]["status"] == "fail"
    assert judged["L6"]["double"]["effective_backend_switched"]


def test_tirust_arms_before_lit_tir_mp_wait_and_after_it_are_judged(
    entries, monkeypatch
):
    """Ruling (5): TIRust arms of an extension before lit-tir-mp (no TIRust
    at every carrier in its recorded capability query, or none recorded)
    wait on lit-tir-mp in L6 and L3 (a binary64 failure still fails); the
    same arms refreshed by an extension serving every carrier are judged,
    and a typed whole-arm refusal there is a coverage gap (partial), never
    a wait. A record refreshed with the staged extension (also one whose
    refresh was interrupted) is resumed, not run afresh."""

    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    stale = _record("li_gg_hh")
    for label in ("tirust/double/plain", "tirust/double/checked"):
        stale["arms"][label] = json.loads(
            json.dumps(stale["arms"]["ninjars/double/plain"])
        )
    stale["l3_probes"]["tirust/double"] = [{"worst": 0.2, "passed": True}] * 3
    judged = lm.judge(stale, entry)
    assert judged["L6"]["double"]["status"] == "pass"
    assert judged["L6"]["status"] == f"waits: {lm.TIR_MP}"
    assert judged["L6"]["waits_on"] == lm.TIR_MP
    assert judged["L3"]["tirust/double"]["status"] == f"waits: {lm.TIR_MP}"
    assert judged["L3"]["tirust/double-double"]["status"] == f"waits: {lm.TIR_MP}"
    assert judged["L3"]["waits_on"] == lm.TIR_MP
    row = lm.lit({"li_gg_hh": stale}, entries, "test")["rows"]["gg_hh"]
    assert row["status"] == "partial" and row["waits_on"] == [lm.TIR_MP]
    assert "L3" in row["note"] and "L6" in row["note"]
    # Refreshed by an extension with TIRust at every carrier: judged, and
    # the double-double refusal is a coverage gap with its typed reason.
    fresh = json.loads(json.dumps(stale))
    fresh["refreshed"] = [{"backend": "tirust", "reduction_backends": AFTER_TIR_MP}]
    fresh["l3_probes"]["tirust/double-double"] = [{"worst": 0.3, "passed": True}] * 3
    judged = lm.judge(fresh, entry)
    assert "waits_on" not in judged["L6"] and "waits_on" not in judged["L3"]
    assert judged["L6"]["double-double"]["status"] == "partial"
    assert "double-double" in judged["L6"]["double-double"]["tirust"]
    assert judged["L3"]["tirust/double"]["status"] == "pass"
    # A binary64 disagreement fails even before lit-tir-mp.
    wrong = json.loads(json.dumps(stale))
    for label in ("tirust/double/plain", "tirust/double/checked"):
        for point in wrong["arms"][label]["points"]:
            point["values"][0]["finite"] = repr(float(point["values"][0]["finite"]) * 1.01)
    assert lm.judge(wrong, entry)["L6"]["status"] == "fail"
    # Resumption: the staged extension's hash against the record's natives.
    monkeypatch.setattr(lm, "_native", lambda: {"extension_sha256": "new"})
    fresh["native"] = {"extension_sha256": "old"}
    fresh["refreshed"][0]["native"] = {"extension_sha256": "new"}
    assert lm._same_native(fresh) and lm._same_native(fresh, "tirust")
    assert not lm._same_native(fresh, "ninjars")
    assert not lm._same_native({"native": {"extension_sha256": "old"}})
    assert lm.tirust_carriers(fresh) == frozenset(AFTER_TIR_MP["tirust"])
    assert lm.tirust_carriers(stale) is None


def test_ward_probes_at_the_fixture_points_judge_l4(entries):
    """L4: the fixture-point Ward probes on every external massless
    gauge-boson leg, both backends in binary64 and double-double, with the
    generation-time probes: every leg within ten times its estimate passes,
    one leg beyond fails, a TIRust point failed closed is partial, an arm
    not run keeps its reason; a row without such a leg is recorded (no
    Ward probe reaches it), never passed."""

    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    record = _after_tir_mp(_record("li_gg_hh"))
    ok = [{"worst": {"1": 0.1, "2": 0.2}, "passed": True}] * 3
    record["ward_probes"] = {
        "legs": [1, 2],
        "zeroed_widths": [],
        "arms": {
            f"{b}/{c}": ok for b in lm.BACKENDS for c in ("double", "double-double")
        },
    }
    judged = lm.judge(record, entry)["L4"]
    assert judged["status"] == "pass" and judged["legs"] == [1, 2]
    assert judged["ninjars/double"]["worst_ratio"] == 0.2
    bad = json.loads(json.dumps(record))
    bad["ward_probes"]["arms"]["ninjars/double-double"] = [
        {"worst": {"1": 0.1, "2": 14.0}, "passed": False}
    ] * 3
    assert lm.judge(bad, entry)["L4"]["status"] == "fail"
    gap = json.loads(json.dumps(record))
    gap["ward_probes"]["arms"]["tirust/double"] = ok[:2] + [
        {"error": "FailedPoints: box refused"}
    ]
    gap["ward_probes"]["arms"]["ninjars/double-double"] = {"not_run": "slow"}
    judged = lm.judge(gap, entry)["L4"]
    assert judged["tirust/double"]["status"] == "partial"
    assert judged["ninjars/double-double"] == {"status": "not run", "reason": "slow"}
    assert judged["status"] == "partial"
    none = json.loads(json.dumps(record))
    none["ward_probes"] = {"legs": [], "zeroed_widths": [], "arms": {}}
    none["generation"]["probes"]["ward"] = {"legs": [], "passed": True, "worst": 0.0}
    judged = lm.judge(none, entry)["L4"]
    assert judged["status"] == "recorded"
    assert "no external massless gauge-boson leg" in judged["fixture_points"]["reason"]


def test_a_numerical_zero_compares_backends_absolutely(entries):
    """On a numerical-zero row (L8) the double-double / AP L6 comparison is
    absolute, within ten times the double-double estimates of both backends:
    two rounding-level values that differ relatively by O(1) agree, a value
    beyond the estimates fails."""

    entry = {e["fixture"]: e for e in entries}["li_gg_ha"]
    assert lm.ZERO_TIER in entry["tiers"]
    record = _after_tir_mp(_record("li_gg_ha"))
    count = len(record["arms"]["ninjars/double-double/checked"]["points"])
    noise = {"points": [_point(1e-60, 1e-28) for _ in range(count)]}
    other = {"points": [_point(3e-60, 1e-28) for _ in range(count)]}
    record["arms"]["ninjars/double-double/checked"] = noise
    record["arms"]["tirust/double-double/checked"] = other
    record["arms"][f"ninjars/{lm.ARBITRARY}/plain"] = noise
    record["arms"][f"tirust/{lm.ARBITRARY}/plain"] = other
    l6 = lm.judge(record, entry)["L6"]
    assert l6["double-double"]["status"] == "pass" and "rule" in l6["double-double"]
    assert l6[lm.ARBITRARY]["status"] == "pass"
    far = json.loads(json.dumps(record))
    far["arms"]["tirust/double-double/checked"] = {
        "points": [_point(1e-20, 1e-28) for _ in range(count)]
    }
    assert lm.judge(far, entry)["L6"]["double-double"]["status"] == "fail"


def test_an_interrupted_refresh_is_resumed(tmp_path, monkeypatch):
    """``_resumable``: a record unfinished in a refresh by the staged
    extension continues (its kept arms are never recomputed); a complete
    record refreshed by it is skipped on a second refresh."""

    monkeypatch.setattr(lm, "_native", lambda: {"extension_sha256": "new"})
    record = {
        "native": {"extension_sha256": "old"},
        "refreshed": [{"backend": "tirust", "native": {"extension_sha256": "new"}}],
        "arms": {"ninjars/double/plain": {"points": []}},
        "l3_probes": {},
        "ward_probes": {"legs": [], "arms": {}},
        "complete": False,
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    assert lm._resumable(path, "tirust", "x", tmp_path) == record
    assert lm._resumable(path, None, "x", tmp_path) == record
    record["complete"] = True
    path.write_text(json.dumps(record))
    assert lm._resumable(path, "tirust", "x", tmp_path)["complete"]
    # a record of another extension without a refresh: run afresh
    record["refreshed"] = []
    path.write_text(json.dumps({**record, "complete": False}))
    assert lm._resumable(path, "tirust", "x", tmp_path) is None


def test_exact_colour_zero_and_reference_findings(entries):
    """LI-Z1: an exact colour zero passes L8 only with the generation's
    colour record and every served value exactly 0 (Q13: the one exact-zero
    mechanism); its Ward probe is vacuous ('recorded'). A recorded HEFT
    finding relabels a failing L9 'reference defect' only where the limit
    is its expected value within ten times the extrapolation error."""

    entry = {e["fixture"]: e for e in entries}["li_h_ga"]
    record = _record("li_gg_hh")
    record["fixture"] = "li_h_ga"
    record["generation"]["probes"] = {"exact_colour_zero": True}
    for arm in record["arms"].values():
        for point in arm.get("points", []):
            if point["values"]:
                point["values"][0]["finite"] = "0.0"
    judged = lm.judge(record, entry)
    assert judged["L8"]["status"] == "pass" and judged["L8"]["exact_colour_zero_record"]
    assert judged["L4"]["status"] == "recorded"
    record["generation"]["probes"] = {}
    assert lm.judge(record, entry)["L8"]["status"] == "fail"
    finding = {"note": "the oracle is 9/8 too large", "expected_limit": "8/9"}
    explained = {
        "status": "fail",
        "points": [{"limit": 0.8889023, "extrapolation_error": 1.4e-4}],
    }
    lm._apply_finding(explained, finding)
    assert explained["status"] == "reference defect" and explained["finding_explains"]
    other = {"status": "fail", "points": [{"limit": 0.5, "extrapolation_error": 1e-6}]}
    lm._apply_finding(other, finding)
    assert other["status"] == "fail" and not other["finding_explains"]
    fixed = {"status": "pass", "points": [{"limit": 1.0, "extrapolation_error": 1e-8}]}
    lm._apply_finding(fixed, finding)
    assert fixed["status"] == "pass"


def test_madloop_stress_and_fig7_figures(entries):
    """The stress points' MadLoop QP rows are compared with the AP value
    (recorded, with MadLoop's own estimate); the Fig. 7 figures are the
    cumulative fractions of the true error and of the estimate."""

    record = _record("li_gg_hh")
    record["arms"][f"ninjars/{lm.ARBITRARY}/plain"] = record["arms"][
        "ninjars/double-double/checked"
    ]
    value = float(
        record["arms"]["ninjars/double-double/checked"]["points"][0]["values"][0][
            "finite"
        ]
    )
    madloop = {
        "rows": [{"value": value * (1 + 1e-12), "accuracy": 1e-11, "status": 217}] * 3,
        "seconds": 1.0,
    }
    judged = lm.judge_madloop(madloop, record)
    assert judged["status"] == "recorded" and judged["truth"].endswith("exact points")
    assert (
        judged["points"][0]["covered"]
        and judged["points"][0]["relative_to_truth"] < 1e-11
    )
    assert (
        lm.judge_madloop({"status": "not run", "reason": "x"}, record)["status"]
        == "not run"
    )
    result = {
        "own_scheme": False,
        "processes": {
            "li_gg_zz": {
                "points": 4,
                "selections": {
                    "all-minus": {
                        "errors": [1e-14, 1e-12, 1e-10, 1e-8],
                        "estimates": [1e-12, 1e-11, 1e-9, 1e-7],
                    }
                },
            }
        },
    }
    figures = lm.judge_fig7([result])
    selection = figures["runs"][0]["processes"]["li_gg_zz"]["selections"]["all-minus"]
    assert (
        selection["error_below"]["1e-09"] == 0.75
        and selection["estimate_below"]["1e-09"] == 0.5
    )
    assert selection["max_error_over_estimate"] == 0.1
    assert figures["runs"][0]["matched"] and "1604.01363" in figures["source"]


def _legs(*spec):
    return [{"flavour": f, "incoming": i, "fermion": fe} for f, i, fe in spec]


def test_crossing_maps_points_and_identity(entries):
    """C1 (the crossing of LI-26's runs): the maps are fixed by the crossed
    legs (identical-leg relabellings counted once), crossed points stay exact
    and momentum conserving with negative-energy crossed legs, and the
    identity holds with (-1)^(crossed fermions) and the fixtures' averages; a
    crossing whose crossed values vanish everywhere is partial with its
    reason, a mismatch beyond ten times the estimate is a miss. data/lit.json
    keys it 'crossing': LI-C1 stays the complex-mass-scheme row."""

    gg_hg = _legs(
        (21, True, False), (21, True, False), (25, False, False), (21, False, False)
    )
    ug_hu = _legs(
        (-2, True, True), (21, True, False), (25, False, False), (2, False, True)
    )
    uux_hg = _legs(
        (-2, True, True), (2, True, True), (25, False, False), (21, False, False)
    )
    assert len(lm.crossing_maps(gg_hg, gg_hg)) == 2
    assert lm.crossing_maps(ug_hu, ug_hu) == [] and lm.crossing_maps(gg_hg, ug_hu) == []
    (mapping,) = lm.crossing_maps(uux_hg, ug_hu)
    assert mapping == [[0, 1], [3, -1], [2, 1], [1, -1]]
    data = lm.fixture("li_ug_hu")
    for point in lm.crossed_points(data["points"], mapping):
        exact = [[Fraction(c) for c in p] for p in point]
        assert exact[1][0] < 0 and exact[3][0] < 0
        for k in range(4):
            assert exact[0][k] + exact[1][k] == exact[2][k] + exact[3][k]

    def point(value, error):
        if value:
            return _point(value, error * value)
        zero = _point(1.0, error)  # an exact zero with an absolute estimate
        zero["values"][0]["finite"] = "0.0"
        return zero

    def arms(values, scale=1.0):
        return {
            "double": {"points": [point(x * scale, 1e-12) for x in values]},
            "double-double": {"points": [point(x * scale, 1e-30) for x in values]},
            "tirust/double": {"points": [point(x * scale, 1e-12) for x in values]},
        }

    records = {n: _record(n) for n in ("li_gg_hg", "li_ug_hu")}
    for record in records.values():
        record["arms"]["tirust/double/checked"] = record["arms"][
            "ninjars/double/checked"
        ]
    near = [
        float(p["values"][0]["finite"])
        for p in records["li_gg_hg"]["arms"]["ninjars/double/checked"]["points"]
    ]
    exact = [
        float(p["values"][0]["finite"])
        for p in records["li_gg_hg"]["arms"]["ninjars/double-double/checked"]["points"]
    ]
    boson = lm.crossing_maps(gg_hg, gg_hg)[0]
    result = {
        "crossings": [
            {
                "target": "li_gg_hg",
                "source": "li_gg_hg",
                "map": boson,
                "crossed_fermions": 0,
                "points": data["points"],
                "arms": {
                    "double": arms(near)["double"],
                    "double-double": arms(exact)["double-double"],
                    "tirust/double": arms(near)["double"],
                },
            },
            {
                "target": "li_uux_hg",
                "source": "li_ug_hu",
                "map": mapping,
                "crossed_fermions": 1,
                "points": data["points"],
                "arms": arms([0.0] * 3),
            },
            {
                "target": "li_gg_hg",
                "source": "li_gg_hg",
                "map": lm.crossing_maps(gg_hg, gg_hg)[1],
                "crossed_fermions": 0,
                "points": data["points"],
                "arms": arms(near, 1.01),
            },
        ]
    }
    judged = lm.judge_crossing(result, records)
    statuses = [r["status"] for r in judged["runs"].values()]
    assert statuses == ["pass", "partial", "estimate miss"]
    first = next(iter(judged["runs"].values()))
    assert (
        first["double_double_max_relative"] == 0.0
        and first["tirust"]["status"] == "pass"
    )
    fermion = list(judged["runs"].values())[1]
    assert fermion["exactly_zero"] and fermion["finding"] == lm.CROSSING_FINDING
    assert judged["status"] == "estimate miss"
    cms = {e["fixture"]: e for e in entries}["li_gg_aa_cms"]
    assert "LI-C1" in cms["ids"]
    out = lm.lit(
        {**records, "li_gg_aa_cms": _record("li_gg_aa_cms")},
        entries,
        "test",
        crossing_result=result,
    )
    assert list(out["ids"]["LI-C1"]["runs"]) == ["li_gg_aa_cms"]
    assert out["ids"][lm.CROSSING_ID]["status"] == "estimate miss"


def test_lane_details_are_compacted_by_reason():
    """A failed point's detail lists every failed lane of every frame (tens
    of MB on a 2->3 row): it is recorded with each distinct reason (numbers
    masked) once, counted, never cut; '; ' inside parentheses and the items
    of a quoted list are respected; the arm's typed reasons count lanes."""

    lane = "family {f} lane LaneKey {{ family: {f}, order: 0, column: {c} }}: TIRust Unstable: conditioning {x}e-3 (refused by the gates; binary64 only)"
    detail = "; ".join(
        lane.format(f=f, c=c, x=f + 2) for f in range(2) for c in range(3)
    )
    compact = lm.compact_detail(detail)
    assert compact == "6 x " + lane.format(f=0, c=0, x=2)
    assert lm.compact_detail(compact) == compact
    listed = "ArithmeticError: failed: ['term scale 0: bad (a; b)', 'term scale 1: bad (a; b)']"
    assert lm.compact_detail(listed).count("term scale") == 2
    arm = {"points": [{"status": "failed", "detail": detail}] * 2}
    assert list(lm._reasons(arm).values()) == [12]


def _lanes(direct: dict, check: dict | None = None, virtual=(1.0, 0.5, 0.25)):
    """One point of lit-tir's ``lanes`` output (li_b1.py)."""

    return {
        "direct": direct,
        "check": direct if check is None else check,
        "virtual": {"o": list(virtual)},
    }


def test_born_lanes_compare_judge_and_split_numerical_zeros():
    """LI-B1's per-lane L6 (lit-tir's rule, DECISIONS 08:31): a lane within
    10 E passes; beyond it below ~1e-5 relative is 'pass (lax)'; a lane
    whose deviation sits within 10 x the floor of the point's lane scale is a
    numerical zero (its re-test reproduces its rounding: estimate 0), lax
    too and kept out of the worst relative deviation; >= 1e-5 relative on a
    lane that is not a zero, or a lane without an estimate, fails; a point a
    backend fails closed makes the row partial with its typed reasons."""

    margin, floor = 10.0, 4 * 2.0**-53
    big = [3.0, 1.0, 0.5]
    n = _lanes({"a": big, "z": [1e-18j, 1e-19j, 0j], "b": [1e-3, 1e-4, 0.0]})
    t = _lanes({"a": big, "z": [1.4e-17j, 1e-19j, 0j], "b": [1e-3, 1e-4, 0.0]})
    result = lm.compare_lanes({"ninjars": [n], "tirust": [t]}, margin, floor)
    assert result["lanes"] == 3 and result["beyond_10x"] == 1
    assert result["beyond_10x_zero"] == 1 and result["worst_beyond_10x"] == 0.0
    assert 0 < result["worst_zero_point_relative"] < margin * floor
    assert result["beyond"][0]["zero"] and result["beyond"][0]["lane"] == "z"
    assert lm.born_verdict(result) == "pass (lax)"
    # a real deviation >= LAX relative on a lane above the point's resolution
    off = _lanes({"a": big, "z": [1e-18j, 1e-19j, 0j], "b": [1.01e-3, 1e-4, 0.0]})
    result = lm.compare_lanes({"ninjars": [n], "tirust": [off]}, margin, floor)
    assert result["worst_beyond_10x"] >= lm.LAX and lm.born_verdict(result) == "fail"
    # below LAX relative, beyond 10 E: lax
    near = _lanes({"a": big, "z": [1e-18j, 1e-19j, 0j], "b": [1e-3 * (1 + 1e-9), 1e-4, 0.0]})
    result = lm.compare_lanes({"ninjars": [n], "tirust": [near]}, margin, floor)
    assert result["beyond_10x"] == 1 and not result["beyond_10x_zero"]
    assert 0 < result["worst_beyond_10x"] < lm.LAX
    assert lm.born_verdict(result) == "pass (lax)"
    # an unchecked lane fails; a failed point is partial with its reason
    blind = _lanes({"a": big}, {"a": None})
    result = lm.compare_lanes(
        {"ninjars": [_lanes({"a": big})], "tirust": [blind]}, margin, floor
    )
    assert result["no_estimate"] == 1 and lm.born_verdict(result) == "fail"
    closed = {"failed": ["family 3 lane 1: TIRust Unstable: ill-conditioned Gram 1e9"]}
    result = lm.compare_lanes(
        {"ninjars": [n, n], "tirust": [t, closed]}, margin, floor
    )
    assert result["failed_points"] == {"ninjars": 0, "tirust": 1}
    assert sum(result["failed"]["tirust"].values()) == 1
    assert lm.born_verdict(result) == "partial"
    result = lm.compare_lanes({"ninjars": [closed], "tirust": [closed]}, margin, floor)
    assert not result["lanes"] and lm.born_verdict(result) == "partial"


def test_forward_points_are_exact_terminating_decimals(monkeypatch):
    """LI-B1's theta rows: each 2 -> 2 row with massless beams and equal final
    masses gets exact points at pt = theta sqrt(s) / 2 (sqrt(s) a 2^x 5^y
    near the row's first point), every entry a terminating decimal, momentum
    conserved exactly and the final shells within 10^-DIGITS of the model's
    (derived, non-terminating) squared mass; other rows get a reason."""

    from decimal import Decimal
    from types import SimpleNamespace

    import pyamplicol.loops.v2.assemble as assemble

    mw = Fraction(80419002245, 10**9) + Fraction(1, 3 * 10**12)  # not decimal
    masses = [Fraction(0), Fraction(0), mw, mw]
    monkeypatch.setattr(assemble, "_leg_masses", lambda *_: tuple(m * m for m in masses))
    first = [["250", "0", "0", "250"], ["250", "0", "0", "-250"],
             ["250", "100", "0", "0"], ["250", "-100", "0", "0"]]
    run = SimpleNamespace(
        slice=None, parameters={}, reference=SimpleNamespace(points=[first], scales=["8315.0"])
    )
    values = ["1e-3", "1e-4"]
    points, scales, energy = lm.forward_points(run, values)
    assert scales == ["8315.0"] * 2 and 400 < energy < 625
    for theta, point in zip(values, points):
        exact = [[Fraction(Decimal(c)) for c in p] for p in point]
        for k in range(4):
            assert exact[0][k] + exact[1][k] == exact[2][k] + exact[3][k]
        for p, m in zip(exact, masses):
            shell = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            assert abs(shell - m * m) <= Fraction(1, 10**lm.DIGITS) * max(m * m, 1)
        assert abs(float(exact[2][1]) - float(theta) * energy / 2) <= 1e-12 * energy
    masses[3] = Fraction(91)
    assert lm.forward_points(run, values).startswith("no forward construction")


def test_run_skips_are_spec_data():
    """LI-44's measured skips live in the spec file (``skips``): the arms and
    their reasons, and ``high_plain``; a run without an entry skips nothing."""

    arms, plain = lm.run_skips("li_gg_zzzz")
    assert plain and "ward/ninjars/double-double" in arms
    assert all(isinstance(r, str) and r for r in arms.values())
    arms, plain = lm.run_skips("li_gg_aagg")
    assert not plain and "ninjars/arbitrary/plain" in arms
    assert lm.run_skips("li_gg_hh") == ({}, False)


def test_a_failing_lane_is_judged_against_the_double_double_truth():
    """Ruling (7): a binary64 lane beyond 10 E at >= 1e-5 relative is judged
    against the double-double lanes first: the backend off the truth by
    more than 10 x its own estimate has an estimate miss (the row's status,
    naming it); without a consistent truth the lane stays a failure."""

    margin, floor = 10.0, 4 * 2.0**-53
    truth = [3.0, 1.0, 0.5]
    off = _lanes({"a": [3.003, 1.0, 0.5]})  # 1e-3 off, its re-test blind
    found = {"ninjars": [off], "tirust": [_lanes({"a": truth})]}
    result = lm.compare_lanes(found, margin, floor)
    result["status"] = lm.born_verdict(result)
    assert result["status"] == "fail"
    agree = {b: [_lanes({"a": truth})] for b in lm.BACKENDS}
    lm.judge_truth(result, found, agree, [0], margin, floor)
    assert result["status"] == "estimate miss"
    assert result["estimate_miss"] == {"ninjars": 1, "tirust": 0}
    assert result["truth"][0]["ninjars"]["error"] > margin * floor
    disagree = {"ninjars": [_lanes({"a": truth})], "tirust": [_lanes({"a": [3.1, 1.0, 0.5]})]}
    result = lm.compare_lanes(found, margin, floor)
    result["status"] = lm.born_verdict(result)
    lm.judge_truth(result, found, disagree, [0], margin, floor)
    assert result["status"] == "fail" and "disagree" in result["truth"][0]["truth"]


def test_identity_l5_on_tirust_is_judged_and_folded(entries):
    """LI-L1 on TIRust (§4.1 backends N, T): its binary64 value at the
    transformed points within 10 E of its value at the fixture points;
    points it fails closed are partial with the typed reasons; an estimate
    miss and a missing record are told apart; TIRust's L5 folds into L5."""

    good = {"points": [_point(1.0, 1e-12), _point(2.0, 1e-12)]}
    moved = {"points": [_point(1.0 + 1e-12, None), _point(2.0, None)]}
    l5 = lm._identity_l5(good, moved, zero=False)
    assert l5["status"] == "pass" and l5["served"] == 2 and l5["failed_points"] == 0
    far = {"points": [_point(1.0 + 1e-9, None), _point(2.0, None)]}
    assert lm._identity_l5(good, far, zero=False)["status"] == "estimate miss"
    closed = {"points": [_point(1.0, 1e-12), _point(0, None, status="failed")]}
    l5 = lm._identity_l5(closed, moved, zero=False)
    assert l5["status"] == "partial" and l5["failed_points"] == 1 and l5["reasons"]
    assert lm._identity_l5(good, far, zero=True)["status"] == "recorded"
    entry = {e["fixture"]: e for e in entries}["li_gg_hh"]
    record = _after_tir_mp(_record("li_gg_hh"))
    assert lm.judge(record, entry)["L5"]["tirust"]["status"] == "not run"
    arm = record["arms"]["tirust/double/checked"]
    record["identity_tirust"] = {
        "fixture": arm,
        "transformed": arm,
        "bitwise_record": True,
        "native": {"build": "x"},
    }
    l5 = lm.judge(record, entry)["L5"]
    failed = sum(p["status"] != "success" for p in arm["points"])
    assert l5["tirust"]["bitwise_record"] and l5["tirust"]["failed_points"] == failed
    assert l5["status"] == lm._worst([l5["status"], l5["tirust"]["status"]])


def test_rambo_points_are_on_shell_and_conserve_momentum():
    """Fig. 7's massive RAMBO (the calibration tool's massless RAMBO,
    rescaled by a converged Newton): every final leg on its shell and
    momentum conserved to binary64 rounding."""

    import random

    rng = random.Random(7)
    for masses in ([91.1876] * 4, [0.0, 0.0, 125.0], [173.0, 173.0]):
        point = [[float(c) for c in v] for v in lm.rambo(masses, 1000.0, rng)]
        for v, m in zip(point[2:], masses, strict=True):
            shell = v[0] ** 2 - v[1] ** 2 - v[2] ** 2 - v[3] ** 2
            assert abs(shell - m * m) <= 1e-12 * 1000.0**2
        for k in range(4):
            total = sum(v[k] for v in point[:2]) - sum(v[k] for v in point[2:])
            assert abs(total) <= 1e-12 * 1000.0
