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
    assert card["loop"]["preparation"]["fixed_parameters"] == data["scheme_data"][
        "fixed_parameters"
    ]
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
            assert sum(p[k] for p in image[:initial]) == sum(p[k] for p in image[initial:])


def _point(value, error, *, status="success", finiteness=1e-15, pole=2e-15):
    return {
        "status": status,
        "detail": "" if status == "success" else "box [0, 1, 2, 3] refused at conditioning 1.6e-5",
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
    t64c = {"points": [_point(near[0] * (1 + 2e-13), 1e-12 * v[0])] + t64["points"][1:]}
    dd = {"points": [_point(x, 1e-30 * x, finiteness=1e-31, pole=1e-31) for x in v]}
    refused = {"refused": "ReductionUnavailable: TIRust does not serve the double-double carrier"}
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
    assert judged["L4"]["status"] == "pass"
    assert judged["L5"]["status"] == "pass"
    l6 = judged["L6"]
    assert l6["status"] == "partial"
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
    assert "TIRust serves 1/3" in row["note"]
    assert out["rows"]["gg_zz"]["status"] == "not run"
    assert "LI-24" in out["ids"] and "LI-A7" in out["ids"]
    path = lm.WT / "docs" / "performance_reports" / "loop" / "scripts" / "build_summary.py"
    spec = importlib.util.spec_from_file_location("build_summary_lit", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    files = {builder.LI_TIMING: None, builder.LI_VALIDATION: out}
    registry = {r["key"]: r for r in lm.registry()["rows"]}
    cells = builder._li_cells(registry["gg_hh"], False, files, {}, None, {}, "tirust")
    validation = cells["validation"]
    assert validation["status"] == "partial" and validation["points"] == 3
    assert [r["key"] for r in validation["references"]][0] in row["references"]
