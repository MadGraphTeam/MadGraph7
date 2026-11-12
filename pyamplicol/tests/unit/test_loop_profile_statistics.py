# SPDX-License-Identifier: 0BSD
"""The stability statistics of ``pyamplicol profile --stability``
(``loop_profile``) on synthetic digit lists, the flat points and the
rendered tables (no native runtime)."""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from pyamplicol.loop_profile import (
    BINARY64_DIGITS,
    PERCENTILES,
    BackendRuntime,
    BackendStability,
    LoopProfileReport,
    StabilityPoint,
    _backend_statistics,
    failure_causes,
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
    assert abs(BINARY64_DIGITS - 15.954589770191003) < 1e-12


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
    # Two sectors per point; the reference in Decimal. Point 2 failed in
    # binary64 (no value, no estimate), point 3 has no reference.
    plain = [
        ((1.0, None, None), (1.0, None, None)),
        ((2.0, None, None),),
        None,
        ((3.0, None, None),),
    ]
    reference = [
        ((Decimal("1.0000001"), None, None), (Decimal(1), None, None)),
        ((Decimal(2), None, None),),
        None,
        None,
    ]
    estimates = [
        _Estimate(1e-3, (1e-8, 1e-8)),
        _Estimate(1e-9, (2e-9,)),
        None,
        _Estimate(1e-9, (3e-9,)),
    ]
    row = _backend_statistics(
        "ninjars",
        plain,
        estimates,
        reference,
        1e-5,
        0.5,
        reasons=[None, None, "unsupported: rank 7", None],
        reference_reasons=[None, None, "unsupported: rank 7", "failed: dd"],
    )
    # The binary64 failure is measured at 0 digits (whatever its reference);
    # the point without a reference is left out.
    assert (row.points, row.failed_binary64, row.failed_reference) == (3, 1, 1)
    assert row.true_digits[0] == pytest.approx(-math.log10(1e-7 / 2.0000001))
    assert row.true_digits[1] == BINARY64_DIGITS
    assert row.true_digits[2] == 0.0
    # Estimated: absolute finite errors summed over sectors over |total|.
    assert row.estimated_digits[0] == pytest.approx(-math.log10(2e-8 / 2.0))
    assert row.estimated_digits[1] == pytest.approx(9.0)
    assert row.estimated_digits[2] is None
    # Point 0 flagged (relative 1e-3 > 1e-5) but accurate; the failure is
    # flagged (detected) and truly inaccurate; nothing missed.
    assert (
        row.flagged,
        row.flagged_inaccurate,
        row.unflagged_inaccurate,
        row.truly_inaccurate,
    ) == (2, 1, 0, 1)
    assert [(p.index, p.status, p.reason) for p in row.per_point] == [
        (0, "measured", None),
        (1, "measured", None),
        (2, "binary64-failed", "unsupported: rank 7"),
        (3, "reference-failed", "failed: dd"),
    ]
    # The failure ranks at the worst end: the 50 % class (rank floor(3/2)
    # = 1) holds it alone, without an estimate.
    median = row.percentiles[3]
    assert (median.rank, median.digits, median.count, median.failed) == (
        1,
        0.0,
        1,
        1,
    )
    assert median.deviation_median is None and median.signed_median is None


def test_failed_points_rank_worst_and_classes_skip_their_estimates():
    true = [0.0, 0.0, 12.0, 13.0, 14.0, 15.0]
    estimated = [None, None, 10.0, 12.0, 11.0, 15.0]
    rows = percentile_table(true, estimated)
    # N = 6: only the 50 % class (ranks 1..3) is populated: both failures
    # and the 12-digit point, whose deviation is the class's only one.
    assert [row.rank for row in rows] == [0, 0, 0, 3]
    assert (rows[3].digits, rows[3].count, rows[3].failed) == (12.0, 3, 2)
    assert rows[3].deviation_min == rows[3].deviation_max == pytest.approx(2.0)
    assert rows[3].signed_median == pytest.approx(-2.0)


def test_failure_reason_reads_the_first_failed_channel():
    from pyamplicol.api.virtual_results import (
        VirtualFailureDiagnostic,
        VirtualLaneDiagnostic,
    )
    from pyamplicol.benchmarking import failure_reason

    lane = VirtualLaneDiagnostic(
        "unsupported",
        "double",
        53,
        False,
        "no TIRust reduction",
        failures=(VirtualFailureDiagnostic(0, None, None, 2, "reduction", "rank 7"),),
    )
    assert failure_reason(lane) == "unsupported: reduction: rank 7"
    # One message per failed lane, joined with "; " (also inside brackets and
    # quotes of a message): the first is kept, the others counted.
    joined = VirtualLaneDiagnostic(
        "failed",
        "double",
        53,
        False,
        'lane 1: Unstable("a; b") (c; d); lane 2: x; lane 3: y',
    )
    assert failure_reason(joined) == (
        'failed: lane 1: Unstable("a; b") (c; d) (+2 more)'
    )
    plain = VirtualLaneDiagnostic("failed", "double", 53, False, "nan value")
    assert failure_reason(plain) == "failed: nan value"
    assert failure_reason(None) == "failed"


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
        flagged=1,
        flagged_inaccurate=1,
        unflagged_inaccurate=1,
        truly_inaccurate=2,
        seconds=1.5,
        per_point=(
            StabilityPoint(0, "best-effort", 1.0, 2.0, "TIRust Unstable: x"),
            StabilityPoint(1, "measured", 3.0, 3.0),
            StabilityPoint(2, "measured", 5.0, 4.0),
            StabilityPoint(3, "measured", 9.0, 9.0),
        ),
        best_effort=1,
        best_effort_covered=0,
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
    assert "true digits" in text and "NinjaRS cannot serve rank 7" in text
    assert "- (N >= 1000)" in text and "50 %" in text
    # Tables only (C6): no legend; the descriptions come with
    # --fields-description-help.
    assert "Digits: -log10" not in text and "lanes:" not in text
    described = io.StringIO()
    write_result(report, format="human", stream=described, fields_help=True)
    flat = " ".join(described.getvalue().split())
    assert "Field descriptions" in flat and "arXiv:1103.0621" in flat
    assert "* marks the default reduction backend" in flat
    # Laptop terminal: no line wider than the report's width.
    assert max(len(line) for line in text.splitlines()) <= 100
    assert "refused" in text and "no TIRust in this build" in text
    # The best-effort points (m-lit ux-besteffort): counted apart and their
    # accuracy against the truth in their own table.
    assert "best-eff./none" in text and "unc. covers" in text
    payload = json.loads(machine.getvalue())
    assert "\x1b[" not in machine.getvalue()
    assert payload["stability"][0]["percentiles"][3]["digits"] == 3.0
    assert payload["stability"][1]["refusal"] == "NinjaRS cannot serve rank 7"
    assert payload["runtime"][0]["refusal"].startswith("refused")


def test_runtime_tables_compare_served_backends_only():
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.reporting import render_summary

    def profile(seconds, failed):
        return VirtualProfileResult(
            "gg_zgg",
            "g g > z g g",
            "loop-induced",
            4,
            2,
            0,
            seconds,
            seconds,
            seconds,
            {"trees": seconds / 2, "reduction": seconds / 2},
            {"lanes": {"tirust": 6, "ninjars": 2}},
            False,
            failed,
        )

    rows = (
        BackendRuntime("tirust", profile(1e-3, 0), ratio=1.0, requested="auto"),
        BackendRuntime("ninjars", profile(3e-3, 0), ratio=3.0),
    )
    text = render_summary(_report(None, rows), color=False)
    assert "fastest" in text and "x3.00" in text and "NinjaRS / TIRust" in text
    # One-line names, the default with an asterisk (C3); the lanes column
    # only because points were served by two backends; no failed column and
    # no notes when nothing failed.
    assert "TIRust *" in text and "(default)" not in text and "as auto" not in text
    assert "| lanes" in text and "NinjaRS 2" in text
    assert "| failed" not in text and "\n---\n" not in text and "lanes:" not in text
    assert max(len(line) for line in text.splitlines()) <= 100
    failed = (
        rows[0],
        BackendRuntime(
            "ninjars",
            replace(profile(3e-3, 8), failure_reasons=(("unsupported: rank 7", 8),)),
            ratio=None,
        ),
    )
    causes = failure_causes(
        [("ninjars", "runtime", "no value", "unsupported: rank 7")] * 8,
        {("ninjars", "runtime"): 8},
    )
    text = render_summary(
        replace(_report(None, failed), failure_causes=causes), color=False
    )
    assert "all failed" in text and "8 of 8" in text
    # The failure causes under the table (C10).
    lines = text.splitlines()
    assert "8 (100 %) NinjaRS | no value: unsupported: rank 7" in lines
    assert (
        lines[lines.index("8 (100 %) NinjaRS | no value: unsupported: rank 7") - 1]
        == "---"
    )
    assert "NinjaRS / TIRust" not in text


def test_failure_causes_group_by_masked_first_message():
    """ux-cli C10: a point's cause is its first refusal message without the
    status, the stage and the "(+N more ...)" counts; causes group with their
    numbers (and non-finite values) masked, most frequent first."""

    from pyamplicol.loop_profile import _strip_status

    reason = (
        "failed: v2-evaluation: family 386 lane: TIRust Unstable: Gram for "
        "[0, 1, 2, 3]: conditioning 1.3e-6 (rounding inf) (+1 more like this); "
        "family 7 lane: other (+2 more)"
    )
    first = _strip_status(reason, "failed")
    assert first == (
        "family 386 lane: TIRust Unstable: Gram for [0, 1, 2, 3]: "
        "conditioning 1.3e-6 (rounding inf)"
    )
    other = first.replace("386", "75").replace("1.3e-6", "4e-4").replace("inf", "2.1")
    causes = failure_causes(
        [("tirust", "stability", "best-effort", first)]
        + [("tirust", "stability", "best-effort", other)] * 2
        + [("tirust", "stability", "no reference", "gates only")]
        + [("tirust", "runtime", "no value", "x")],
        {("tirust", "stability"): 10, ("tirust", "runtime"): 4},
    )
    assert [(c.table, c.kind, c.count, c.total) for c in causes] == [
        ("runtime", "no value", 1, 4),
        ("stability", "best-effort", 3, 10),
        ("stability", "no reference", 1, 10),
    ]
    assert causes[1].reason == first


def test_stability_estimate_grows_while_a_stream_has_no_point(monkeypatch):
    """A backend and precision without a returned point costs at least the
    time its first point has been running (the tasks started when the first
    one returned, less its seconds), so the estimate rises while a slow
    stream's first point is pending; the first-call cost seen on the other
    streams is taken off, and a stream's later tasks give its warm cost."""

    import pyamplicol.loop_profile as lp

    clock = [100.0]
    monkeypatch.setattr(lp.time, "perf_counter", lambda: clock[0])
    tracker = lp._StabilityProgress(None, ("ninjars",), 11, 2)
    tracker.add("ninjars", "binary64", 1, 1.0)
    # Only the mean of the known costs: 1 s per point, 21 points left.
    assert tracker.eta() == pytest.approx((10 * 1.0 + 11 * 1.0) / 2)
    clock[0] = 160.0
    # The double-double point has run for 61 s (since 99 s) at least.
    assert tracker.eta() == pytest.approx((10 * 1.0 + 11 * 61.0) / 2)
    tracker.add("ninjars", "binary64", 5, 1.0)
    # binary64 warm: 0.2 s per point; its first point's 0.8 s extra is the
    # first-call cost, off the pending double-double point.
    assert tracker.eta() == pytest.approx((5 * 0.2 + 11 * (61.0 - 0.8)) / 2)
    tracker.add("ninjars", "double-double", 1, 70.0)
    assert tracker.eta() == pytest.approx((5 * 0.2 + 10 * 70.0) / 2)
