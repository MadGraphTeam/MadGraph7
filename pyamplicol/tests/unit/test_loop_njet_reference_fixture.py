# SPDX-License-Identifier: 0BSD
"""The NJet g g > g g g g g reference fixture and its pyAmpliCol comparison harness.

Pure JSON and exact arithmetic: no NJet build, no pyAmpliCol generation.
"""

from __future__ import annotations

import copy
import json
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_reference.exact_points import generate
from tools.loop_reference.njet import njet_fixture as nf

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "loop" / "required" / "gg_ggggg_njet.json"
PRIMARY = "njet_dd_oneloop_R1"

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_schema_points_and_universal_double_pole(fixture: dict) -> None:
    assert nf.validate_fixture(fixture)
    assert fixture["pdgs"] == [21] * 7
    assert fixture["born_orders"] == {"QCD": 5, "QED": 0}
    assert fixture["normalization_denominator"] == 256 * 120
    assert fixture["primary_provider"] == PRIMARY
    assert fixture["maximum_universal_double_pole_relative_error"] < 1e-12
    # Poles against the IR-pole (Catani) prediction from colour-correlated Borns.
    assert fixture["maximum_catani_single_pole_relative_error"] < 1e-8


def test_points_are_pyamplicols_and_read_like_the_required_fixtures(fixture: dict) -> None:
    # pyAmpliCol's campaign takes exact_points.generate([0]*7) in this order.
    first = generate([0] * 7, count=3)
    assert [[[Fraction(x) for x in p] for p in point] for point in fixture["points"][:3]] == [
        [list(p) for p in point] for point in first
    ]
    # The reading of tools/loop_reference/legacy_capture.control_points.
    data = json.loads(FIXTURE.read_text(), parse_int=str, parse_float=str)
    scales = [format(Decimal(mu) ** 2, "f") for mu in data["mu_r"]]
    assert scales[:3] == ["1000000"] * 3
    assert all(isinstance(x, str) for point in data["points"] for p in point for x in p)


def test_provenance_and_calibration(fixture: dict) -> None:
    njet = fixture["providers"][PRIMARY]
    assert njet["tool"] == "NJet 3.1.2"
    assert "-march=native" in njet["build_info"] and "--enable-5jet" in njet["build_info"]
    assert "--with-oneloop" in njet["build_info"]
    options = njet["options"]
    assert options["prec"] == "dd" and options["scaling"] == "1"
    assert options["scheme"] == "cdr" and options["renorm"] == "1" and options["nf"] == "5"
    assert options["rotation"]["name"] == "R1"
    assert fixture["evaluation_orientation"]["name"] == "R1"
    for row in fixture["rows"]:
        assert set(row["providers"]) >= {PRIMARY, "njet_double_R1", "njet_double_R2", "catani"}
    calibration = fixture["calibration"]
    # Conventions are validated against the provider references with top-free
    # OpenLoops (nf=5) and, through it, the gg_gggg fixture (nf=6).
    for label in ("gg_gg njet312 dd vs openloops nf5", "gg_gggg njet312 dd vs openloops nf5",
                  "gg_gggg openloops nf6 vs fixture recola"):
        deviations = calibration[label]["max_relative_deviation"]
        assert max(deviations.values()) < 1e-8, label


def _virtual_record(fixture: dict, *, top_free: bool, shift: float = 0.0) -> dict:
    return {
        "process": "g g > g g g g g",
        "top_quark_loops": not top_free,
        "points": [
            {
                "index": i,
                "born": float(row["born"]),
                "virtual": [float(row["finite"]) + shift, float(row["single_pole"]),
                            float(row["double_pole"])],
            }
            for i, row in enumerate(fixture["rows"][:3])
        ],
    }


def test_harness_virtual(fixture: dict) -> None:
    exact = nf.compare_pyamplicol(fixture, _virtual_record(fixture, top_free=True))
    for row in exact:
        assert max(row[k] for k in nf.COEFFICIENTS) < 1e-15
    # With loop_sm's top loops the finite difference is reported, never tested.
    shifted = nf.compare_pyamplicol(fixture, _virtual_record(fixture, top_free=False, shift=1e-9))
    for row in shifted:
        assert "finite" not in row
        assert row["finite_minus_reference"] == pytest.approx(1e-9, rel=1e-6)


def test_harness_bare_single_pole_shift(fixture: dict) -> None:
    # (n-2) beta_0(5) + (8/3) T_F = 5 * 23/3 + 4/3 = 119/3 at n = 7.
    units = 5 * nf.BETA0_LIGHT + Fraction(8, 3) * nf.T_F
    assert units == Fraction(119, 3)
    alphas = Fraction(fixture["parameters"]["aS"])
    record = {"process": "g g > g g g g g", "points": []}
    for i, row in enumerate(fixture["rows"][:3]):
        born = float(row["born"])
        shift = float(alphas) / (4 * math.pi) * float(units) * born
        record["points"].append({
            "index": i, "born": born,
            "bare": [0.0, float(row["single_pole"]) + shift, float(row["double_pole"])],
        })
    for row in nf.compare_pyamplicol(fixture, record):
        assert row["kind"] == "bare"
        assert row["single_pole"] < 1e-15 and row["double_pole"] < 1e-15
        assert "finite" not in row and "finite_minus_reference" not in row


def test_harness_refuses_other_points_or_processes(fixture: dict) -> None:
    record = _virtual_record(fixture, top_free=True)
    wrong = copy.deepcopy(record)
    wrong["points"][1]["born"] *= 1 + 1e-6
    with pytest.raises(ValueError, match="not the fixture's point"):
        nf.compare_pyamplicol(fixture, wrong)
    other = copy.deepcopy(record)
    other["process"] = "g g > g g g g"
    with pytest.raises(ValueError, match="is not"):
        nf.compare_pyamplicol(fixture, other)
