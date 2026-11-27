# SPDX-License-Identifier: 0BSD
"""The progress ETA estimator on synthetic completion sequences."""

from __future__ import annotations

import pytest

from pyamplicol.reporting.eta import combined, count_estimate, remaining_estimate


def _marks(durations: list[float]) -> list[tuple[float, int]]:
    marks, now = [], 0.0
    for index, duration in enumerate(durations, start=1):
        now += duration
        marks.append((now, index))
    return marks


def _eta_after(durations: list[float], done: int, *, prior=None):
    marks = _marks(durations[:done])
    return count_estimate(
        total=len(durations),
        completed=done,
        started_at=0.0,
        marks=marks,
        now=marks[-1][0] if marks else 0.0,
        prior_seconds=prior,
    )


def _actual_left(durations: list[float], done: int) -> float:
    return sum(durations[done:])


def test_constant_rate_is_exact() -> None:
    durations = [2.0] * 20
    for done in (1, 5, 10, 19):
        estimate = _eta_after(durations, done)
        assert estimate is not None
        assert estimate.seconds == pytest.approx(_actual_left(durations, done))
    assert _eta_after(durations, 1).estimated
    assert not _eta_after(durations, 5).estimated


def test_slow_first_item_does_not_set_the_pace() -> None:
    durations = [30.0] + [2.0] * 29
    estimate = _eta_after(durations, 4)
    assert estimate is not None
    # A plain mean over the elapsed time would say 26 x 9 s = 234 s.
    assert estimate.seconds == pytest.approx(_actual_left(durations, 4), rel=0.05)


def test_load_drift_is_followed() -> None:
    durations = [1.0] * 20 + [2.0] * 20
    estimate = _eta_after(durations, 30)
    assert estimate is not None
    truth = _actual_left(durations, 30)
    overall_mean = 10 * (sum(durations[:30]) / 30)
    assert abs(estimate.seconds - truth) < abs(overall_mean - truth)
    assert estimate.seconds == pytest.approx(truth, rel=0.25)


def test_one_outlier_does_not_make_the_estimate_jump() -> None:
    durations = [2.0] * 10 + [20.0] + [2.0] * 9
    before = _eta_after(durations, 10)
    after = _eta_after(durations, 11)
    assert before is not None and after is not None
    # Remaining work shrinks by one item; the estimate may grow, not double.
    assert after.seconds < 2.0 * before.seconds


def test_countdown_between_updates_and_floor() -> None:
    marks = _marks([2.0] * 5)
    kwargs = dict(total=10, completed=5, started_at=0.0, marks=marks)
    at_mark = count_estimate(now=10.0, **kwargs)
    later = count_estimate(now=11.5, **kwargs)
    assert at_mark.seconds == pytest.approx(10.0)
    assert later.seconds == pytest.approx(8.5)
    # The item in progress takes long: the other four items still remain.
    stuck = count_estimate(now=20.0, **kwargs)
    assert stuck.seconds == pytest.approx(8.0)
    assert stuck.overdue == 0.0
    assert stuck.at_least and not later.at_least
    # Outlasting that time four times over: overdue, not a stale bound.
    outlasted = count_estimate(now=50.0, **kwargs)
    assert outlasted.seconds is None and outlasted.overdue > 0.0


def test_never_zero_while_work_remains() -> None:
    marks = _marks([2.0] * 9)
    for now in (18.0, 19.0, 19.9, 20.5, 25.0, 60.0):
        estimate = count_estimate(
            total=10, completed=9, started_at=0.0, marks=marks, now=now
        )
        assert estimate is not None
        # Some time left ("<1s" on the display), or overdue: never 0.
        assert estimate.seconds is None or estimate.seconds > 0.0
    late = count_estimate(total=10, completed=9, started_at=0.0, marks=marks, now=30.0)
    assert late.seconds is None
    assert late.overdue == pytest.approx(10.0)


def test_prior_before_any_item_and_its_blend() -> None:
    first = count_estimate(
        total=4, completed=0, started_at=0.0, marks=[], now=10.0, prior_seconds=100.0
    )
    assert first.estimated and first.seconds == pytest.approx(90.0)
    assert (
        count_estimate(total=4, completed=0, started_at=0.0, marks=[], now=1.0) is None
    )
    # One measured item of 10 s against a prior of 25 s per item.
    blended = count_estimate(
        total=4,
        completed=1,
        started_at=0.0,
        marks=[(10.0, 1)],
        now=10.0,
        prior_seconds=100.0,
    )
    assert 3 * 10.0 < blended.seconds < 3 * 25.0
    overdue = remaining_estimate(prior_seconds=5.0, started_at=0.0, now=8.0)
    assert overdue.seconds is None and overdue.overdue == pytest.approx(3.0)


def test_predictions_converge_on_a_noisy_sequence() -> None:
    import random

    rng = random.Random(7)
    durations = [8.0] + [max(rng.gauss(3.0, 0.8), 0.5) for _ in range(99)]
    total_time = sum(durations)
    for fraction, tolerance in ((0.1, 0.15), (0.25, 0.1), (0.5, 0.06), (0.75, 0.04)):
        done = int(fraction * len(durations))
        estimate = _eta_after(durations, done)
        predicted_end = sum(durations[:done]) + estimate.seconds
        assert predicted_end == pytest.approx(total_time, rel=tolerance)


def test_combined_total() -> None:
    active = count_estimate(
        total=10, completed=5, started_at=0.0, marks=_marks([2.0] * 5), now=10.0
    )
    total = combined(active, 30.0, later_known=True)
    assert total.seconds == pytest.approx(40.0) and not total.estimated
    bound = combined(active, 30.0, later_known=False)
    assert bound.estimated
    assert combined(None, 0.0, later_known=True) is None


def test_summary_tables_fit_the_terminal_width() -> None:
    import prettytable

    from pyamplicol.reporting.summary import _ANSI_STYLE, _FittedTables, table_width

    fitted = _FittedTables(prettytable, 60)
    table = fitted.PrettyTable(("field", "value"))
    table.add_row(("output", "/a/very/long/path/" * 6))
    table.add_row(("files", "\x1b[32m58 files; 32.56 MiB total; 1446 payloads\x1b[0m"))
    text = table.get_string()
    assert max(len(_ANSI_STYLE.sub("", line)) for line in text.splitlines()) <= 60
    assert "| field  |" in text  # the narrow column is not wrapped
    with table_width(None):
        pass


def test_log_heartbeat_breaks_every_silence(monkeypatch) -> None:
    import io
    import time as clock

    from pyamplicol.reporting import progress
    from pyamplicol.reporting.progress import ProgressEnd, ProgressStart

    monkeypatch.setattr(progress, "_LOG_CADENCE", ((float("inf"), 0.6),))
    stream = io.StringIO()
    sink = progress.progress_sink("log", stream=stream)
    sink.emit(ProgressStart("root", "Generating processes", total=1))
    sink.emit(ProgressStart("root:a", "A quick step", parent_task_id="root"))
    sink.emit(ProgressEnd("root:a"))
    sink.emit(ProgressStart("root:b", "A long step", parent_task_id="root"))
    clock.sleep(3.6)
    sink.emit(ProgressEnd("root:b", elapsed_seconds=3.6, details={"keep": True}))
    sink.emit(ProgressEnd("root"))
    progress.close_progress_sink(sink)
    lines = stream.getvalue().splitlines()
    assert not any("A quick step" in line for line in lines)
    assert lines[1] == "A long step"
    beats = [line for line in lines if line.startswith("A long step: ")]
    assert len(beats) >= 2 and all("elapsed" in line for line in beats)
    assert lines[-2] == "A long step done in 3s"


def test_fitted_tables_never_break_numbers_or_headers() -> None:
    import prettytable

    from pyamplicol.reporting.summary import _FittedTables

    fitted = _FittedTables(prettytable, 60)
    table = fitted.PrettyTable(("point", "virtual squared order", "finite [0]"))
    table.add_row(("0", "QCD^2, QED^4", "-0.00016690271388525922"))
    table.add_row(("1", "QED^6", "-1.079349250527398e-05"))
    text = table.get_string()
    # It cannot fit in 60 columns without breaking a number: natural width.
    assert "-0.00016690271388525922" in text
    assert "virtual squared order" in text


def test_a_long_description_gives_way_to_the_clocks() -> None:
    from pyamplicol.reporting.progress import _paint_fragments

    line = _paint_fragments(
        [
            ("Loading process output /a/very/long/path/to/an/artifact/OUT", "CYAN"),
            ("  elapsed 12s", None),
            ("  RSS 1.47/1.47G", None),
        ],
        width=60,
        enabled=False,
    )
    assert len(line) <= 60
    assert line.endswith("elapsed 12s  RSS 1.47/1.47G")
    assert line.startswith("Loading pro") and "..." in line


def test_a_phase_line_never_shows_eta_zero_while_work_remains() -> None:
    from pyamplicol.reporting.progress import _phase_line, _TaskState

    task = _TaskState(
        task_id="phase",
        description="Exporting",
        total=4,
        parent_task_id="root",
        unit="items",
        details={},
        started_at=0.0,
        updated_at=6.0,
        completed=3,
        marks=[[2.0, 1], [4.0, 2], [6.0, 3]],
    )
    for now in (6.0, 7.5, 8.5, 12.0, 60.0):
        line = _phase_line(task, now=now, active_children=0, width=200, color=False)
        assert "ETA 0s" not in line
        assert "ETA" in line or "overdue" in line
    assert "overdue" in _phase_line(
        task, now=60.0, active_children=0, width=200, color=False
    )
    task.completed = 4
    done = _phase_line(task, now=61.0, active_children=0, width=200, color=False)
    assert "ETA" not in done and "overdue" not in done


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-5", "1e308"])
def test_a_history_file_with_nonsense_seconds_gives_no_prior(tmp_path, value) -> None:
    import json

    from pyamplicol.generation.progress import GenerationHistory

    phases = ["a", "b"]
    steps = {name: "SECONDS" for name in phases}
    record = {"s": {"phases": phases, "steps": steps}}
    text = json.dumps(record).replace('"SECONDS"', value)
    (tmp_path / "generation-history.json").write_text(text, encoding="utf-8")

    history = GenerationHistory("s", root=tmp_path)

    later = history.later(set())
    assert later is None or (later[0] >= 0 and later[0] < float("inf"))
    for name in phases:
        prior = history.prior(name)
        assert prior is None or 0 <= prior < float("inf")
