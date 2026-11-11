# SPDX-License-Identifier: 0BSD
"""The stability statistics of ``pyamplicol profile --stability``
(``loop_profile``) on synthetic digit lists, the flat points and the
rendered tables (no native runtime)."""

from __future__ import annotations

import io
import json
import math
from decimal import Decimal
from fractions import Fraction

import pytest

from pyamplicol.loop_profile import (
    BINARY64_DIGITS,
    PERCENTILES,
    BackendRuntime,
    BackendStability,
    LoopProfileReport,
    _backend_statistics,
    needed_points,
    percentile_table,
    relative_error,
    stable_digits,
)


def test_stable_digits_are_capped_to_binary64():
    assert stable_digits(0.0) == BINARY64_DIGITS
    assert stable_digits(2.0**-60) == BINARY64_DIGITS
    assert stable_digits(1e-7) == pytest.approx(7.0)
    assert stable_digits(1.0) == 0.0 and stable_digits(3.0) == 0.0
    assert stable_digits(math.inf) == 0.0 and stable_digits(math.nan) == 0.0
    assert BINARY64_DIGITS == pytest.approx(15.954589770191003)


def test_relative_error_against_a_decimal_reference():
    assert relative_error(1.0, Decimal("1.000001")) == pytest.approx(
        1e-6 / 1.000001, rel=1e-9
    )
    assert relative_error(0.0, Decimal(0)) == 0.0
    assert relative_error(1e-30, Decimal(0)) == math.inf
    # Formed above binary64: a one-ulp difference is resolved.
    assert relative_error(1.0, Decimal(1) + Decimal(2) ** -52) == pytest.approx(
        2.0**-52, rel=1e-9
    )


def test_percentile_classes_and_values_on_a_thousand_points():
    # Point k has true digits k / 100 (shuffled), estimate true + 1 except
    # the worst ten, whose estimate overstates by 3.
    n = 1000
    true = [k / 100 for k in range(n)]
    order = list(reversed(range(n)))  # worst last in the input
    true_in = [true[k] for k in order]
    est_in = [t + (3.0 if t < 0.1 else 1.0) for t in true_in]
    rows = percentile_table(true_in, est_in)
    assert [row.percentile for row in rows] == [0.001, 0.01, 0.1, 0.5]
    assert [row.rank for row in rows] == [1, 10, 100, 500]
    assert [row.count for row in rows] == [1, 9, 90, 400]
    # The value at percentile p is the true digits of the point of rank
    # floor(p N) of the sorted list (worst first).
    assert [row.digits for row in rows] == [0.0, 0.09, 0.99, 4.99]
    assert rows[0].deviation_min == rows[0].deviation_max == pytest.approx(3.0)
    assert rows[1].deviation_median == pytest.approx(3.0)
    assert rows[1].signed_median == pytest.approx(3.0)
    assert rows[2].deviation_min == pytest.approx(1.0)
    assert rows[2].deviation_max == pytest.approx(1.0)
    assert all(row.needed_points <= n for row in rows)


def test_small_samples_leave_classes_empty_and_name_the_needed_count():
    rows = percentile_table([5.0, 1.0, 3.0, 9.0], [5.0, 0.5, 4.0, 9.0])
    assert [row.rank for row in rows] == [0, 0, 0, 2]
    assert [row.count for row in rows] == [0, 0, 0, 2]
    assert [row.digits for row in rows] == [None, None, None, 3.0]
    assert rows[0].deviation_median is None and rows[0].signed_median is None
    # The median class: the two worst points (1.0 -> 0.5, 3.0 -> 4.0).
    assert rows[3].deviation_min == 0.5 and rows[3].deviation_max == 1.0
    assert rows[3].signed_median == pytest.approx(0.25)
    assert [row.needed_points for row in rows] == [1000, 100, 10, 2]
    assert needed_points(0) == 1000
    # One point: only the median rank floor(N/2) = 0 is unresolved as well.
    single = percentile_table([7.0], [8.0])
    assert all(row.count == 0 and row.digits is None for row in single)
    assert percentile_table([], []) == tuple(percentile_table([], [], PERCENTILES))
    with pytest.raises(ValueError, match="aligned"):
        percentile_table([1.0], [])


def test_needed_points_for_other_percentiles():
    # floor(N / 3) > floor(N / 4) first at N = 3.
    assert needed_points(1, (Fraction(1, 4), Fraction(1, 3))) == 3
    assert needed_points(1, (Fraction(1, 2), Fraction(3, 5))) == 5
    assert needed_points(0, (Fraction(1, 3),)) == 3


class _Laurent(tuple):
    @property
    def finite(self):
        return self[0]


class _Estimate:
    def __init__(self, relative, absolute):
        self.relative = relative
        self.finite = relative
        self.absolute = tuple(_Laurent((value, None, None)) for value in absolute)


def test_backend_statistics_from_values_estimates_and_references():
    # Two sectors per point; the reference in Decimal.
    plain = [((1.0, None, None), (1.0, None, None)), ((2.0, None, None),), None]
    reference = [
        ((Decimal("1.0000001"), None, None), (Decimal(1), None, None)),
        ((Decimal(2), None, None),),
        ((Decimal(1), None, None),),
    ]
    estimates = [_Estimate(1e-3, (1e-8, 1e-8)), _Estimate(1e-9, (2e-9,)), None]
    row = _backend_statistics("ninjars", plain, estimates, reference, 1e-5, 0.5)
    assert row.points == 2 and row.failed == 1
    assert row.true_digits[0] == pytest.approx(-math.log10(1e-7 / 2.0000001))
    assert row.true_digits[1] == BINARY64_DIGITS
    # Estimated: absolute finite errors summed over sectors over |total|.
    assert row.estimated_digits[0] == pytest.approx(-math.log10(2e-8 / 2.0))
    assert row.estimated_digits[1] == pytest.approx(9.0)
    # Point 0 flagged (relative 1e-3 > 1e-5) but accurate; nothing missed.
    assert (row.flagged, row.flagged_inaccurate, row.unflagged_inaccurate) == (
        1,
        0,
        0,
    )


def _report(stability_rows, runtime_rows=None):
    setup = {
        "points": 4,
        "seed": 1,
        "sqrt_s": 1000.0,
        "estimate": "the default rotation test",
        "target": 1e-5,
        "binary64_digits": BINARY64_DIGITS,
    }
    return LoopProfileReport(
        "gg_zgg",
        "g g > z g g",
        "loop-induced",
        "tirust",
        ("tirust", "ninjars"),
        None if runtime_rows is None else "default setup",
        runtime_rows,
        setup,
        stability_rows,
    )


def test_rendering_both_backends_a_refusal_and_json():
    from pyamplicol.cli.main import write_result

    digits = (1.0, 3.0, 5.0, 9.0)
    served = BackendStability(
        "tirust",
        points=4,
        true_digits=digits,
        estimated_digits=(2.0, 3.0, 4.0, 9.0),
        percentiles=percentile_table(digits, (2.0, 3.0, 4.0, 9.0)),
        median_true_digits=4.0,
        median_estimated_digits=3.5,
        flagged=1,
        flagged_inaccurate=1,
        unflagged_inaccurate=1,
        truly_inaccurate=2,
        seconds=1.5,
    )
    refused = BackendStability("ninjars", refusal="NinjaRS cannot serve rank 7")
    report = _report(
        (served, refused),
        (BackendRuntime("tirust", None, "refused: no TIRust in this build"),),
    )
    plain, coloured, machine = io.StringIO(), io.StringIO(), io.StringIO()
    write_result(report, format="human", stream=plain, color=False)
    write_result(report, format="human", stream=coloured, color=True)
    write_result(report, format="json", stream=machine, color=True)
    text = plain.getvalue()
    assert "\x1b[" not in text and "\x1b[" in coloured.getvalue()
    assert "TIRust true digits" in text and "NinjaRS cannot serve rank 7" in text
    assert "- (N >= 1000)" in text and "50 % (median)" in text
    assert "A populated 1 per mil class needs N >= 1000" in text
    assert "refused" in text and "no TIRust in this build" in text
    payload = json.loads(machine.getvalue())
    assert "\x1b[" not in machine.getvalue()
    assert payload["stability"][0]["percentiles"][3]["digits"] == 3.0
    assert payload["stability"][1]["refusal"] == "NinjaRS cannot serve rank 7"
    assert payload["runtime"][0]["refusal"].startswith("refused")


def test_flat_points_conserve_momentum_and_keep_the_masses(tmp_path):
    from pyamplicol.runtime.flat_points import (
        artifact_kinematics,
        external_masses,
        flat_points,
    )

    process = tmp_path / "processes" / "p"
    process.mkdir(parents=True)
    point = [
        {"momentum": ["500", "0", "0", "500"], "pdg": 2},
        {"momentum": ["500", "0", "0", "-500"], "pdg": -2},
        {
            "momentum": [
                "195.6991336459694",
                "-54.456572281957435",
                "38.451662872088697",
                "-62.650636608530753",
            ],
            "pdg": 6,
        },
        {
            "momentum": [
                "361.07092884968853",
                "-251.34762331697158",
                "171.4440017739098",
                "-88.738617123742969",
            ],
            "pdg": -6,
        },
        {
            "momentum": [
                "314.37354247181827",
                "192.74843324930768",
                "-230.27422748757118",
                "93.019061708983159",
            ],
            "pdg": 21,
        },
        {
            "momentum": [
                "128.85639503252366",
                "113.05576234962132",
                "20.378562841572641",
                "58.370192023290549",
            ],
            "pdg": 21,
        },
    ]
    (process / "validation-momenta.json").write_text(
        json.dumps({"available": True, "points": [point]})
    )
    kinematics = artifact_kinematics(tmp_path, "p")
    assert kinematics.incoming == 2 and kinematics.sqrt_s == pytest.approx(1000.0)
    masses = kinematics.masses
    assert masses[:2] == (0.0, 0.0) and masses[4:] == (0.0, 0.0)
    assert masses[2] == pytest.approx(173.0, rel=1e-9) == masses[3]
    points = flat_points(kinematics, 3, seed=7, sqrt_s=2000.0)
    assert flat_points(kinematics, 1, seed=8, sqrt_s=2000.0)[0] == points[1]
    for item in points:
        total = [sum(p[k] for p in item[2:]) for k in range(4)]
        assert total == pytest.approx([2000.0, 0.0, 0.0, 0.0], abs=1e-9)
        assert external_masses(item) == pytest.approx(masses, abs=1e-6)
    with pytest.raises(ValueError, match="must exceed"):
        flat_points(kinematics, 1, seed=1, sqrt_s=300.0)
