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


def _report(stability_rows, runtime_rows=None, truth=None):
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
        ground_truth=truth,
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


def test_stability_tasks_fill_whole_simd_chunks():
    """A stability task holds whole chunks of binary64 lanes (a padded lane
    costs as much as a filled one): about eight tasks per worker, at most
    one batch (its whole chunks)."""

    import pyamplicol.loop_profile as lp

    assert lp._chunk_size(40, 4, 128, 16) == 16
    assert lp._chunk_size(800, 4, 128, 16) == 32
    assert lp._chunk_size(160, 8, 128, 16) == 16
    assert lp._chunk_size(100_000, 8, 128, 16) == 128
    assert lp._chunk_size(100_000, 8, 100, 16) == 96
    # A batch smaller than a chunk is the task.
    assert lp._chunk_size(100, 1, 10, 16) == 10


def test_stability_estimate_grows_while_a_stream_has_no_point(monkeypatch):
    """The workers warm up before any timed task, so a returned task's
    seconds are its points' cost. A stream with nothing back costs at least
    the time its first task (started with the others) has been running per
    point of it, so the estimate rises while a slow stream is pending."""

    import pyamplicol.loop_profile as lp

    clock = [100.0]
    monkeypatch.setattr(lp.time, "perf_counter", lambda: clock[0])
    tracker = lp._StabilityProgress(None, ("ninjars",), 20, 2, 4)
    tracker.add("ninjars", "binary64", 4, 0.8)
    # 0.2 s per point; the double-double stream at least as costly.
    assert tracker.eta() == pytest.approx((16 * 0.2 + 20 * 0.2) / 2)
    # Its first task has been running since 99.2 s: 40.8 s for 4 points.
    clock[0] = 140.0
    assert tracker.eta() == pytest.approx((16 * 0.2 + 20 * 40.8 / 4) / 2)
    tracker.add("ninjars", "reference", 4, 280.0)
    assert tracker.eta() == pytest.approx((16 * 0.2 + 16 * 70.0) / 2)
    tracker.add("ninjars", "binary64", 8, 2.4)
    # 3.2 s over 12 points.
    assert tracker.eta() == pytest.approx((8 * 3.2 / 12 + 16 * 70.0) / 2)
    assert tracker.rate(("ninjars", "reference")) == pytest.approx(70.0)
    tracker.drop("ninjars")
    assert tracker.eta() == 0.0


def test_displayed_stability_estimate_is_smoothed(monkeypatch):
    """The displayed estimate follows the raw one in logarithm with the
    smoothing time constant: shown at once, a jump damped, a steady
    estimate kept (counting down)."""

    import math

    import pyamplicol.loop_profile as lp

    clock = [100.0]
    raw = [60.0]
    monkeypatch.setattr(lp.time, "perf_counter", lambda: clock[0])
    tracker = lp._StabilityProgress(None, ("ninjars",), 11, 1)
    monkeypatch.setattr(tracker, "eta", lambda: raw[0])
    assert tracker.shown_eta() == pytest.approx(60.0)
    # A tenfold jump one second later moves it by a small factor only.
    clock[0], raw[0] = 101.0, 590.0
    weight = 1.0 - math.exp(-1.0 / tracker.ETA_SMOOTHING_SECONDS)
    expected = math.exp((1 - weight) * math.log(59.0) + weight * math.log(590.0))
    assert tracker.shown_eta() == pytest.approx(expected)
    assert expected < 2 * 59.0
    # A steady estimate is kept; minutes later the new one has taken over.
    clock[0], raw[0] = 400.0, 300.0
    assert tracker.shown_eta() == pytest.approx(300.0, rel=1e-6)
    monkeypatch.setattr(tracker, "eta", lambda: None)
    assert tracker.shown_eta() is None


def test_runtime_and_stability_report_peak_ram_and_reference_timing():
    """The runtime table's ``peak RAM`` column (``<=`` for a lifetime bound,
    a muted ``-`` when not measured) and its ``(dd xN)`` time multiplier
    when the profile timed the reference; the stability table's ``time``:
    the binary64 total with the reference multiplier; no free lines under
    the tables; all in the JSON as fields. With an arbitrary-precision
    truth the labels say ``arb(D)`` (header, titles, cells)."""

    from pyamplicol.benchmarking import PrecisionTiming, VirtualProfileResult
    from pyamplicol.cli.main import write_result
    from pyamplicol.reporting import render_summary
    from pyamplicol.reporting.resources import PeakMemory

    base = VirtualProfileResult(
        "gg_zgg",
        "g g > z g g",
        "loop-induced",
        4,
        2,
        1,
        1e-3,
        9e-4,
        1.1e-3,
        {"trees": 5e-4, "reduction": 5e-4},
        {},
        False,
        0,
    )
    peak = PeakMemory(
        int(2.212 * 1024**3), 2 * 1024**3, 1024**3, "profiled region", "VmHWM"
    )
    bound = replace(peak, peak_rss_bytes=3 * 1024**3, scope="process lifetime")
    dd = PrecisionTiming("double-double", 4, 2, 1, 8, 2e-2, 1.9e-2, 2.1e-2, 0)
    rows = (
        BackendRuntime(
            "tirust", replace(base, peak_memory=peak, reference=dd), ratio=1.0
        ),
        BackendRuntime(
            "ninjars",
            replace(base, seconds_per_point=2e-3, peak_memory=bound),
            ratio=2.0,
        ),
    )
    timed = replace(
        BackendStability("tirust", points=4, true_digits=(9.0,) * 4, seconds=2.8),
        precision_timing={
            "binary64": {"points": 4, "seconds": 1.0, "seconds_per_point": 2e-3},
            "double-double": {"points": 4, "seconds": 1.8, "seconds_per_point": 4e-2},
        },
    )
    report = _report((timed,), rows)
    text = render_summary(report, color=False)
    lines = text.splitlines()
    runtime = next(line for line in lines if line.startswith("| TIRust"))
    assert "| 1 ms (dd x20) |" in runtime and "| 2.21 GiB |" in runtime, text
    slower = next(line for line in lines if line.startswith("| NinjaRS"))
    assert "| 2 ms " in slower and "dd x" not in slower
    assert "| <= 3 GiB |" in slower, text
    header = next(line for line in lines if "| time / point" in line)
    assert header.index("vs fastest") < header.index("peak RAM")
    assert any(line.endswith("| 1 s (dd x1.8) |") for line in lines), text
    assert not any("peak RAM:" in line or "time / point:" in line for line in lines)
    assert max(len(line) for line in lines) <= 100
    # Wide verdict cells: the multiplier goes under the time, within the
    # line width.
    wide = replace(
        timed,
        points=1000,
        flagged=120,
        flagged_inaccurate=100,
        unflagged_inaccurate=12,
        failed_binary64=2,
        precision_timing={
            "binary64": {"points": 1000, "seconds": 1.47},
            "double-double": {"points": 1000, "seconds": 26.3},
        },
    )
    wrapped = render_summary(_report((wide,), rows), color=False).splitlines()
    assert max(len(line) for line in wrapped) <= 100
    first = next(i for i, line in enumerate(wrapped) if "| 1.47 s" in line)
    assert wrapped[first + 1].rstrip(" |").endswith("(dd x18)")
    # Not measured: a muted dash.
    unmeasured = render_summary(
        _report(None, (BackendRuntime("tirust", base, ratio=1.0),)), color=False
    )
    assert next(
        line for line in unmeasured.splitlines() if line.startswith("| TIRust")
    ).endswith("| -        |")
    machine = io.StringIO()
    write_result(report, format="json", stream=machine, color=False)
    payload = json.loads(machine.getvalue())
    profile = payload["runtime"][0]["profile"]
    assert profile["peak_memory"]["increase_rss_bytes"] == 1024**3
    assert profile["reference"]["points"] == 8
    assert payload["runtime"][1]["profile"]["reference"] is None
    timing = payload["stability"][0]["precision_timing"]
    assert timing["double-double"]["points"] == 4
    assert timing["binary64"]["seconds"] == 1.0
    # Colour: the time bold (green: the fastest), the whole parenthesis
    # muted (bright black) after it, in both tables; without colour the text.
    import re

    from pyamplicol.reporting.loop_profile import _ANSI

    coloured = render_summary(report, color=True)
    assert _ANSI.sub("", coloured) == text
    assert "\x1b[90m(dd x20)\x1b[0m" in coloured
    assert "\x1b[90m(dd x1.8)\x1b[0m" in coloured
    assert re.search(r"\x1b\[1m\x1b\[32m1 ms\x1b\[0m \x1b\[90m\(dd x20\)", coloured)
    assert "\x1b[1m1 s\x1b[0m \x1b[90m(dd x1.8)" in coloured
    # An arbitrary-precision truth: its label everywhere, its timing key.
    truth = {"arithmetic": "arbitrary", "digits": 57, "label": "arb(57)"}
    arb = replace(dd, arithmetic="arbitrary", digits=57, seconds_per_point=0.21)
    arb_rows = (
        replace(rows[0], profile=replace(rows[0].profile, reference=arb)),
        rows[1],
    )
    arb_timed = replace(
        timed,
        precision_timing={
            "binary64": {"points": 4, "seconds": 1.0},
            "arbitrary": {"points": 4, "seconds": 42.0, "digits": 57},
        },
    )
    text = render_summary(_report((arb_timed,), arb_rows, truth), color=False)
    lines = text.splitlines()
    assert "| 1 ms (arb(57) x210) |" in next(
        line for line in lines if line.startswith("| TIRust")
    ), text
    assert any(
        "| ground truth  | arbitrary precision, 57 digits" in line for line in lines
    )
    assert any("Stable digits of binary64 vs arb(57)," in line for line in lines)
    # Past the line width on one line: the multiplier under the time.
    first = next(
        i for i, line in enumerate(lines) if line.endswith("| 1 s           |")
    )
    assert lines[first + 1].endswith("| (arb(57) x42) |"), text
    assert "dd x" not in text and max(len(line) for line in lines) <= 100
    # A wide runtime table (a lanes column, a long label): the multiplier
    # goes under the time, within the line width, no new line elsewhere.
    lanes = {"tirust": 3451, "ninjars": 5}
    wide_rows = (
        replace(
            arb_rows[0],
            profile=replace(
                arb_rows[0].profile,
                reference=replace(arb, seconds_per_point=1.808),
                reduction={"lanes": lanes},
            ),
        ),
        replace(
            arb_rows[1],
            profile=replace(arb_rows[1].profile, reduction={"lanes": {"ninjars": 9}}),
        ),
    )
    wide = render_summary(_report(None, wide_rows, truth), color=False).splitlines()
    assert max(len(line) for line in wide) <= 100, "\n".join(wide)
    first = next(i for i, line in enumerate(wide) if line.startswith("| TIRust"))
    assert wide[first].split("|")[2].strip() == "1 ms"
    assert wide[first + 1].split("|")[2].strip() == "(arb(57) x1808)"


def test_ground_truth_spellings():
    """``benchmark.ground_truth``: double-double (dd), arbitrary (arb) at the
    built-in rescue's digits, or arbitrary:DIGITS; anything else refused."""

    from pyamplicol.config import BenchmarkConfig
    from pyamplicol.config.errors import ConfigurationError
    from pyamplicol.loop_profile import GroundTruth, ground_truth
    from pyamplicol.loops.v2.stability import DEFAULT_RESCUE

    default = dict(DEFAULT_RESCUE)["arbitrary"]
    assert BenchmarkConfig().ground_truth == "double-double"
    for text, expected in (
        ("double-double", GroundTruth("double-double")),
        ("DD", GroundTruth("double-double")),
        ("arb", GroundTruth("arbitrary", default)),
        ("arbitrary", GroundTruth("arbitrary", default)),
        ("arb:60", GroundTruth("arbitrary", 60)),
        (" arbitrary:100 ", GroundTruth("arbitrary", 100)),
    ):
        canonical = BenchmarkConfig(ground_truth=text).ground_truth
        assert ground_truth(canonical) == expected == ground_truth(text)
    assert ground_truth("arb:60").label == "arb(60)"
    assert ground_truth("dd").label == "dd"
    assert ground_truth("arb").options == {
        "arithmetic": "arbitrary",
        "precision": default,
    }
    for bad in ("double", "arb:", "arb:0", "arb:x", "dd:30", "quad", 3):
        with pytest.raises(ConfigurationError):
            BenchmarkConfig(ground_truth=bad)


def test_relative_error_is_exact():
    """The true error against a reference of any precision is formed
    exactly: a 60-digit reference differing from the binary64 value far
    below a Decimal context's 28 digits is still resolved, non-finite
    values give nan, a zero reference 0 or inf."""

    from decimal import localcontext

    from pyamplicol.loop_profile import _exact_total, relative_error

    with localcontext() as context:
        context.prec = 100
        # binary64's 0.1 exactly, 1e-45 above it.
        reference = Decimal(0.1) + Decimal("1e-45")  # noqa: RUF032
    assert relative_error(0.1, reference) == pytest.approx(1e-44, rel=1e-6)
    assert relative_error(0.5, Fraction(1, 2)) == 0.0
    assert math.isnan(relative_error(math.nan, Decimal(1)))
    assert relative_error(1.0, Decimal(0)) == math.inf
    row = [[Decimal("1e30")], [Decimal("1")], [Decimal("-1e30")]]
    assert _exact_total(row) == 1


def test_peak_memory_measures_the_region():
    """``peak_memory`` over a block that maps and touches every page of 256
    MiB and holds it to the end: the peak covers the resident size the
    kernel counts with the block in (VmHWM is at least VmRSS in one read).
    The block is not released inside the region: the kernel records the
    high-water mark of released memory from its per-CPU batched counters,
    which can read short of the exact peak by up to a batch per CPU (the
    cause of the old 0.9-of-the-block assertion's failures on a busy host)."""

    import gc
    import mmap

    from pyamplicol.reporting import resources
    from pyamplicol.reporting.resources import peak_memory, peak_memory_text

    size = 256 * 1024**2
    page = mmap.PAGESIZE
    gc.disable()
    try:
        with peak_memory() as peak:
            block = mmap.mmap(-1, size)
            for offset in range(0, size, page):
                block[offset] = 1
            held = resources._proc_status_kib("VmRSS")
        block.close()
    finally:
        gc.enable()
    result = peak.result
    assert result is not None and result.peak_rss_bytes is not None
    assert result.peak_rss_bytes >= result.start_rss_bytes
    assert result.increase_rss_bytes == result.peak_rss_bytes - result.start_rss_bytes
    if held is not None and result.scope == "profiled region":
        assert result.peak_rss_bytes >= held
    text = peak_memory_text(result)
    assert text.endswith(("MiB", "GiB"))
    assert text.startswith("<= ") == (result.scope != "profiled region")
