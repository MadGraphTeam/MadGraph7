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
    assert "madloop" in data["providers"]
    assert data["providers"]["madloop"]["settings"]["MadLoopParams"]["HelicityFilterLevel"] == "0"
    assert data["providers"]["madloop"]["settings"]["param_card"]
    if spec.get("zero"):
        assert data["zero_row"] and data["passed"] is None
        return
    compared = [p for p in data["providers"]]
    assert len(compared) >= 2 and data["passed"]
    assert data["max_provider_pair_relative_error"] <= L1
    for provider in ("recola", "openloops"):
        if spec.get(provider, {}).get("unavailable"):
            assert provider not in data["providers"]
        else:
            assert provider in data["providers"], f"{name}: no {provider} reference"
    if spec.get("analytic"):
        # The oracle equals at least one provider far below L1 at every point
        # (under the complex-mass scheme MadLoop's a a > a a W loop is 5.5e-7
        # off Recola and the oracle, which agree to 1e-14).
        for row in data["rows"]:
            errors = row["provider_pair_relative_errors"]
            assert min(v for k, v in errors.items() if "analytic" in k) <= 1e-9


@pytest.mark.parametrize(
    "name",
    [n for n in _li_processes()
     if not required_spec.load_process(required_spec.DEFAULT_SPEC, n).get("zero")])
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
        if row["tier"] in ("analytic", "2->2", "1->3", "zeros", "EW"):
            assert row["runs"] or row.get("owner") or row["row"] == "LI-Z1", row["row"]
        for run in row["runs"]:
            assert run["status"] in ("pass", "zero row"), (row["row"], run)


@pytest.mark.parametrize("name", ["li_gg_hh", "li_gg_hgg", "li_gg_zzg"])
def test_provider_timing_records_of_the_gated_and_two_to_two_rows(name):
    record = json.loads((FIXTURES / "timing" / f"{name}-providers.json").read_text())
    assert record["process"] == name
    for provider in ("madloop", "recola", "openloops"):
        entry = record["providers"][provider]
        assert "error" not in entry and entry["mean_seconds_per_point"] > 0
    madloop = record["providers"]["madloop"]
    assert madloop["madloop_params"]["MLReductionLib"] == "7|6|1"
    assert set(madloop["stages"]["stages_seconds_per_point"]) == {
        "loop numerator coefficients", "Born and counterterm HELAS", "reduction and masters",
        "colour flows and squaring"}
    opp = json.loads((FIXTURES / "timing" / f"{name}-providers-madloop-opp.json").read_text())
    assert opp["providers"]["madloop"]["madloop_params"]["MLReductionLib"] == "6|7|1"


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
