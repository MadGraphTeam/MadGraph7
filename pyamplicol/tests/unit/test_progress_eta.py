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
    stuck = count_estimate(now=40.0, **kwargs)
    assert stuck.seconds == pytest.approx(8.0)
    assert stuck.overdue == 0.0


def test_never_zero_while_work_remains() -> None:
    marks = _marks([2.0] * 9)
    for now in (18.0, 19.0, 19.9, 20.5, 25.0, 60.0):
        estimate = count_estimate(
            total=10, completed=9, started_at=0.0, marks=marks, now=now
        )
        assert estimate is not None
        assert estimate.seconds is None or estimate.seconds >= 1.0
    late = count_estimate(total=10, completed=9, started_at=0.0, marks=marks, now=30.0)
    assert late.seconds is None
    assert late.overdue == pytest.approx(10.0)


def test_prior_before_any_item_and_its_blend() -> None:
    first = count_estimate(
        total=4, completed=0, started_at=0.0, marks=[], now=10.0, prior_seconds=100.0
    )
    assert first.estimated and first.seconds == pytest.approx(90.0)
    assert count_estimate(total=4, completed=0, started_at=0.0, marks=[], now=1.0) is None
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
