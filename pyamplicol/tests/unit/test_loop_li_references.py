# SPDX-License-Identifier: 0BSD
"""The loop-induced reference fixtures and campaign manifest (M-LIT LIT-REF).

The fixtures ``tests/fixtures/loop/required/li/<process>.json`` are merged
provider references (``tools/loop_reference/li_references.py``) of the
``"loop_induced": true`` processes of ``tools/loop_reference/required_processes.json``;
``tools/loop_reference/li_campaign.json`` is the PLAN §4 matrix pointing at them.
These checks read committed data only (no provider runs).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tools" / "loop_reference"
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

import make_li_campaign  # noqa: E402
import required_spec  # noqa: E402

pytestmark = pytest.mark.loop_fast

L1 = 1e-5


def _li_processes():
    return [p["name"] for p in required_spec.load(required_spec.DEFAULT_SPEC)["processes"]
            if p.get("loop_induced")]


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", _li_processes())
def test_loop_induced_specification_and_fixture(name):
    spec = required_spec.load_process(required_spec.DEFAULT_SPEC, name)
    assert spec["born_orders"] is None and spec["born_orders_all"] == []
    assert spec["squared_orders"] == {k: 2 * v for k, v in spec["loop_orders"].items()}
    path = FIXTURES / f"{name}.json"
    assert path.exists(), f"no reference fixture for {name}"
    data = _fixture(name)
    assert data["pdgs"] == spec["pdgs"] and data["squared_orders"] == spec["squared_orders"]
    assert data["normalization"]["denominator"] == required_spec.normalization(spec)["denominator"]
    text = path.read_text()
    assert not re.search(r"/(shared|home|tmp)/", text), "absolute path left in the fixture"
    if spec["madloop"].get("unavailable"):
        # e.g. g g > a a a: MG5 removes Furry loops by rule (no diagram).
        assert "madloop" not in data["providers"] and data["unavailable"]["madloop"]
    elif {"export", "madloop"} & set(data["misses_45_min"]):
        assert "madloop" not in data["providers"]
    else:
        assert "madloop" in data["providers"]
        assert data["providers"]["madloop"]["settings"]["MadLoopParams"]["HelicityFilterLevel"] == "0"
        assert data["providers"]["madloop"]["settings"]["param_card"]
    if spec.get("zero"):
        assert data["zero_row"] and data["passed"] is None
        return
    _check_secondary_runs(data)
    compared = [p for p in data["providers"]]
    if len(compared) < 2:
        # A single reference only where the others missed the 45-minute rule
        # (recorded) or serve no such process (specification); it carries an
        # independent secondary run (another library, a rotated frame) at
        # every point.
        assert data["misses_45_min"] and data["passed"] is False, name
        assert data["secondary_runs"], f"{name}: a single reference without a secondary run"
        return
    assert data["passed"]
    assert data["max_provider_pair_relative_error"] <= L1
    for provider in ("recola", "openloops"):
        if spec.get(provider, {}).get("unavailable"):
            assert provider not in data["providers"]
        elif provider not in data["misses_45_min"]:
            assert provider in data["providers"], f"{name}: no {provider} reference"
    if spec.get("analytic"):
        # The oracle equals at least one provider far below L1 at every point
        # (under the complex-mass scheme MadLoop's a a > a a W loop is 5.5e-7
        # off Recola and the oracle, which agree to 1e-14).
        for row in data["rows"]:
            errors = row["provider_pair_relative_errors"]
            assert min(v for k, v in errors.items() if "analytic" in k) <= 1e-9


def _check_secondary_runs(data):
    """Secondary runs record their per-point deviation from the reference; a
    point where one exceeds L1 carries an ``l1_caution`` (and only such a
    point); production values taken from a timing record equal that record's
    values at the stream points that are the reference points."""
    for label, run in data["secondary_runs"].items():
        assert len(run["relative_deviation_from_reference"]) == len(data["rows"]), label
        source = (run.get("settings") or {}).get("timing_record")
        if source:
            record = json.loads((FIXTURES.parent.parent / source).read_text())
            stream = json.loads((ROOT / "tools" / "loop_benchmark" / "streams" / f"{data['name']}.json").read_text())
            rows = record["providers"][run["settings"]["record_provider"]]["rows"]
            for index, (point, number) in enumerate(zip(data["points"], run["settings"]["stream_points"])):
                assert stream["points"][number - 1] == point, (label, index)
                value = rows[number - 1]["finite"]
                assert run["relative_deviation_from_reference"][index] == pytest.approx(
                    abs(value - data["rows"][index]["value"]) / max(abs(value), abs(data["rows"][index]["value"])),
                    rel=1e-12), (label, index)
    if data.get("zero_row"):
        return
    for index, row in enumerate(data["rows"]):
        beyond = [k for k, v in row.get("secondary_relative_deviations", {}).items() if v > L1]
        assert bool(beyond) == ("l1_caution" in row), (data["name"], index + 1)


def _with_madloop_reference(name):
    # A reference without MadLoop records why (unavailable, or a 45-minute
    # miss): test_loop_induced_specification_and_fixture.
    path = FIXTURES / f"{name}.json"
    return path.exists() and "madloop" in json.loads(path.read_text())["providers"]


@pytest.mark.parametrize(
    "name",
    [n for n in _li_processes()
     if not required_spec.load_process(required_spec.DEFAULT_SPEC, n).get("zero") and _with_madloop_reference(n)])
def test_single_helicity_rows_sum_to_the_summed_value(name):
    # Q39: |A(h)|^2/avg (MadLoop's USERHEL value / HELAVGFACTOR) sums to the
    # helicity-summed value; a stale USERHEL would break the sum from point 2 on.
    data = _fixture(name)
    for row in data["rows"]:
        helicity = row.get("per_helicity")
        assert helicity, f"{name}: no single-helicity rows"
        total = sum(entry["value"] for entry in helicity)
        scale = max(abs(row["value"]), max(abs(entry["value"]) for entry in helicity))
        assert abs(total - row["value"]) <= 1e-9 * scale


def test_the_campaign_manifest_is_current_and_covers_every_reference():
    manifest = json.loads((REFERENCE / "li_campaign.json").read_text())
    assert manifest == json.loads(json.dumps(make_li_campaign.build()))
    referenced = {run["process"] for row in manifest["rows"] for run in row["runs"]}
    assert set(_li_processes()) <= referenced
    for row in manifest["rows"]:
        if row["tier"] in ("analytic", "2->2", "1->3", "zeros", "EW") or "gate" in row["tier"]:
            assert row["runs"] or row.get("owner") or row["row"] == "LI-Z1", row["row"]
        for run in row["runs"]:
            if run["status"] == "single provider":
                assert row["tier"] in ("2->4 optional", "2->5 stretch") and run["misses_45_min"], (row["row"], run)
                continue
            assert run["status"] in ("pass", "zero row"), (row["row"], run)
    # PLAN §4.1: the 2->4 optional rows are gated where run and matched, the
    # 2->5 stretch if a provider matches (both have provider timings).
    assert {r["row"] for r in manifest["rows"] if r["gated"]} >= {"LI-41", "LI-42", "LI-43", "LI-44", "LI-51"}
    # Every 2->3 row of PLAN §4.1 has its references (lit-ref-b).
    three = {run["process"] for row in manifest["rows"] if row["tier"] == "2->3 gate" for run in row["runs"]}
    assert {"li_emep_ggg", "li_gg_hhh", "li_gg_hgg", "li_gg_zzg", "li_gg_aag", "li_gg_wpwmg", "li_gg_zzz",
            "li_gg_hzg", "li_gg_agg", "li_gg_zgg", "li_uux_hgg", "li_uux_ddxh"} <= three


TIMED = sorted(p.name[: -len("-providers.json")] for p in (FIXTURES / "timing").glob("*-providers.json"))


def test_every_two_to_three_gate_row_has_provider_timings():
    manifest = json.loads((REFERENCE / "li_campaign.json").read_text())
    three = {run["process"] for row in manifest["rows"] if row["tier"] == "2->3 gate" for run in row["runs"]}
    assert three <= set(TIMED)


@pytest.mark.parametrize("name", TIMED)
def test_provider_timing_records(name):
    # The benchmark specification decides what is timed (Recola is timed on the
    # four-lepton rows although its width convention excludes its values).
    spec = required_spec.load_process(ROOT / "tools" / "loop_benchmark" / "processes.json", name)
    record = json.loads((FIXTURES / "timing" / f"{name}-providers.json").read_text())
    assert record["process"] == name and record["points"] == 16
    assert not re.search(r"/(shared|home|tmp)/", json.dumps(record)), "absolute path left in the record"
    for provider in ("madloop", "recola", "openloops"):
        entry = record["providers"][provider]
        if spec[provider].get("unavailable"):
            assert entry.get("unavailable"), (name, provider)
            continue
        assert "error" not in entry and entry["mean_seconds_per_point"] > 0
    madloop = record["providers"]["madloop"]
    assert madloop["madloop_params"]["MLReductionLib"] == "7|6|1"
    assert set(madloop["stages"]["stages_seconds_per_point"]) == {
        "loop numerator coefficients", "Born and counterterm HELAS", "reduction and masters",
        "colour flows and squaring"}
    opp = json.loads((FIXTURES / "timing" / f"{name}-providers-madloop-opp.json").read_text())
    assert opp["providers"]["madloop"]["madloop_params"]["MLReductionLib"] == "6|7|1"
    _check_provider_values(spec, record, opp)
    # The one-helicity column (MadEvent mode); one record per row, so the
    # summary table finds one production timing per provider.
    one = madloop["one_helicity"]
    if "miss" in one:  # g g > z z z z: 324 helicity calls per point
        assert "45-minute rule" in one["miss"]
        return
    assert one["madloop_params"]["MLReductionLib"] == "7|6|1"
    assert 0 < one["mean_seconds_per_call"] and one["calls"] == sum(one["kept_helicities_per_point"])
    assert all(0 < kept <= one["helicities"] for kept in one["kept_helicities_per_point"])


def _check_provider_values(spec, record, opp):
    """The providers' finite values in a timing record (MadLoop 6|7|1 from its
    own record included) agree within MadLoop's production stability
    threshold (the record's MLStabThres, lax), except a provider whose values
    the specification excludes (``values_excluded``, e.g. a width convention)
    and the stream points it excepts (``value_exceptions``, 1-based); the
    record carries those annotations, and every excepted point is a real
    outlier."""
    madloop = record["providers"]["madloop"]
    tolerance = float(madloop["madloop_params"]["MLStabThres"].lower().replace("d", "e"))
    values, exceptions = {}, {}
    for provider, entry in record["providers"].items():
        for field in ("values_excluded", "value_exceptions"):
            assert entry.get(field) == spec[provider].get(field), (record["process"], provider, field)
        if "rows" not in entry or entry.get("values_excluded"):
            continue
        values[provider] = [row["finite"] for row in entry["rows"]]
        exceptions[provider] = set(entry.get("value_exceptions", {}).get("points", ()))
        assert not exceptions[provider] or entry["value_exceptions"]["reason"]
    values["madloop 6|7|1"] = [row["finite"] for row in opp["providers"]["madloop"]["rows"]]
    exceptions["madloop 6|7|1"] = set()

    def deviation(a, b, index):
        x, y = values[a][index], values[b][index]
        return abs(x - y) / max(abs(x), abs(y))

    names = sorted(values)
    for index in range(record["points"]):
        kept = [n for n in names if index + 1 not in exceptions[n]]
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                assert deviation(a, b, index) <= tolerance, (record["process"], index + 1, a, b)
        for name in set(names) - set(kept):
            assert all(deviation(name, other, index) > tolerance for other in kept), (record["process"], index + 1)


def test_references_py_refuses_a_loop_induced_specification(tmp_path, monkeypatch):
    import provider_paths
    import references

    monkeypatch.delenv(provider_paths.ENVIRONMENT, raising=False)

    class Args:
        mg5 = None
        reference_root = None
        spec = required_spec.DEFAULT_SPEC
        process = "li_gg_h"
        provider = "recola"
        recola = tmp_path

    with pytest.raises(SystemExit, match="loop-induced"):
        references.command_run(Args())
