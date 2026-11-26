# SPDX-License-Identifier: 0BSD
"""The loop profile's time budgets (``profile_plan``), its scheduling loops
(``benchmarking._timed_calls``, ``_reference_calls``, the stability run's
admission gate) under a fake clock, the shared ETA estimator, and the live
progress of a slow reference pass (the user's "hang", 2026-11-16)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import pytest

from pyamplicol.config import BenchmarkConfig
from pyamplicol.profile_plan import (
    MEASURE_REPEAT_SECONDS,
    STABILITY_MINIMUM,
    TASK_SECONDS,
    WARMUP_CALL_SECONDS,
    Costs,
    affordable_chunk,
    measurement_affects_counts,
    minimum_counts,
    plan,
    reference_affects_counts,
    runtime_counts,
    stability_points,
    stability_wall,
    task_points,
    warmup_seconds,
)

LANES = 16
# The acceptance of a requested completion time: the achieved wall time
# within a factor 1.5 either way (the planner estimates from a measurement
# of first calls and priors, and re-plans as it goes; a factor of several
# would be a defect).
TOLERANCE = (0.5, 1.5)


def _costs(
    binary64: float, multiplier: float | None = 60.0, start: float = 2.0
) -> Costs:
    reference = None if multiplier is None else binary64 * multiplier
    return Costs(binary64, reference, 2.5 * binary64, reference, start)


def test_runtime_counts_are_whole_chunks_and_never_below_the_minimum():
    """A backend's runtime counts: whole SIMD chunks, at most the batch
    limit, one timed call at least, whatever the budget."""

    for binary64, budget in itertools.product(
        (1e-5, 1e-3, 0.5, 5.0), (0.0, 1.0, 10.0, 600.0)
    ):
        counts = runtime_counts(_costs(binary64), budget, lanes=LANES, batch_max=128)
        assert counts.batch % LANES == 0 and LANES <= counts.batch <= 128
        assert counts.minimum >= 1 and counts.warmup == 1
        assert counts.reference_budget >= _costs(binary64).reference
    cheap = runtime_counts(_costs(1e-5), 10.0, lanes=LANES, batch_max=128)
    dear = runtime_counts(_costs(0.5), 10.0, lanes=LANES, batch_max=128)
    assert cheap.batch == 128 and dear.batch == LANES


def test_a_reference_cost_that_cannot_change_the_counts_is_not_needed():
    """``reference_affects_counts``: under the default budget a backend whose
    share (its start-up excluded) affords no more than the binary64 warm-up
    gets the same counts whatever its reference costs (one chunk, the
    minimum calls, no timed budget), so its reference need not be measured;
    an estimated reference sets no budget, only the expected seconds."""

    from pyamplicol.profile_plan import DEFAULT_RUNTIME_SECONDS, REFERENCE_SHARE

    for binary64, start, backends in itertools.product(
        # 0.1 s at a 3 s start-up on 2 backends sits in the band where the
        # share (2 s) beats the warm-up (1.6 s) but its 3/4 do not (ptimereview2
        # m-a: the rule before b5c9f61da measured the reference there).
        (1e-5, 1e-3, 0.05, 0.1, 0.6, 200.0),
        (0.5, 3.0, 1200.0),
        (1, 2),
    ):
        share = max(DEFAULT_RUNTIME_SECONDS / backends - start, 0.0)
        needed = reference_affects_counts(
            binary64, start, backends=backends, lanes=LANES
        )
        left = (1.0 - REFERENCE_SHARE) * share
        assert needed == (left > warmup_seconds(LANES * binary64, binary64))
        if needed:
            # The exact condition: a free reference would leave a timed
            # budget that a dear one takes.
            cheap, dear = (
                runtime_counts(
                    _costs(binary64, multiplier, start),
                    share,
                    lanes=LANES,
                    batch_max=128,
                ).budget
                for multiplier in (0.0, 1e9)
            )
            assert cheap > 0.0 == dear
            continue
        counts = {
            (c.batch, c.warmup, c.minimum, c.budget)
            for multiplier in (0.0, 2.0, 60.0, 1e4)
            for c in [
                runtime_counts(
                    _costs(binary64, multiplier, start),
                    share,
                    lanes=LANES,
                    batch_max=128,
                )
            ]
        }
        assert counts == {(LANES, 1, 1, 0.0)}
    # The cheap processes measure it (their counts depend on it).
    assert reference_affects_counts(1e-3, 0.5, backends=2, lanes=LANES)
    # g g > g g g g measured 1.8 s for its first one-point call, 3.2 s start-up.
    assert not reference_affects_counts(1.8, 4.2, backends=2, lanes=LANES)
    estimated = replace(_costs(200.0, 30.0, 1200.0), reference_estimated=True)
    counts = runtime_counts(estimated, 0.0, lanes=LANES, batch_max=128)
    assert counts.reference_budget == REFERENCE_SHARE * 0.0
    assert counts.expected == pytest.approx(1200.0 + 200.0 + LANES * 200.0 + 2 * 6000.0)
    measured = runtime_counts(
        _costs(200.0, 30.0, 1200.0), 0.0, lanes=LANES, batch_max=128
    )
    assert measured.reference_budget == 6000.0
    assert (measured.batch, measured.budget, measured.expected) == (
        counts.batch,
        counts.budget,
        counts.expected,
    )


def test_a_start_up_beyond_the_share_leaves_the_minimum_counts():
    """``measurement_affects_counts``: when a backend's start-up takes its
    share of the default budget, the counts are the minimum whatever the
    costs (no measurement needed), and ``minimum_counts`` gives exactly
    them, its priors setting the expected seconds only."""

    from pyamplicol.profile_plan import DEFAULT_RUNTIME_SECONDS

    for backends in (1, 2):
        share = DEFAULT_RUNTIME_SECONDS / backends
        assert measurement_affects_counts(share - 0.5, backends=backends)
        assert not measurement_affects_counts(share, backends=backends)
        start = share + 1.0
        for binary64, multiplier in itertools.product(
            (1e-6, 1e-3, 0.5, 200.0), (None, 2.0, 60.0)
        ):
            counts = plan(
                {name: _costs(binary64, multiplier, start) for name in "ab"[:backends]},
                runtime=True,
                stability=False,
                workers=1,
                lanes=LANES,
                batch_max=128,
            ).runtime["a"]
            minimum = minimum_counts(
                start, lanes=LANES, reference=multiplier is not None
            )
            assert (
                (counts.batch, counts.warmup, counts.minimum, counts.budget)
                == (
                    minimum.batch,
                    minimum.warmup,
                    minimum.minimum,
                    minimum.budget,
                )
                == (LANES, 1, 1, 0.0)
            )
            assert minimum.reference_budget == 0.0 and minimum.expected == start
            with_priors = minimum_counts(
                start,
                lanes=LANES,
                reference=multiplier is not None,
                costs=_costs(binary64, multiplier, start),
            )
            assert with_priors.batch == LANES and with_priors.budget == 0.0
            assert with_priors.expected == pytest.approx(counts.expected)


def test_stability_points_are_chunks_and_at_least_the_minimum():
    costs = [_costs(1e-3), _costs(3e-3)]
    assert stability_points(costs, 0.0, 10, lanes=LANES) == STABILITY_MINIMUM
    many = stability_points(costs, 60.0, 10, lanes=LANES)
    assert many % LANES == 0 and many > LANES
    # More workers, more points in the same time.
    assert stability_points(costs, 60.0, 20, lanes=LANES) >= 2 * many - LANES


def test_the_plan_rules():
    """--completion-time overrules the counts given; without it the counts
    given are honoured, else the default budgets; a budget the minimums
    exceed is infeasible and the plan estimates the minimums' time."""

    costs = {"tirust": _costs(2e-3), "ninjars": _costs(6e-3)}
    common = dict(runtime=True, stability=True, workers=4, lanes=LANES, batch_max=128)
    budget = plan(costs, completion=600.0, **common, stability_points_given=1000)
    assert budget.rule == "completion time" and budget.feasible
    assert budget.stability_points != 1000
    assert TOLERANCE[0] * 600.0 <= budget.expected_seconds <= TOLERANCE[1] * 600.0
    given = plan(costs, **common, stability_points_given=1000)
    assert given.rule == "counts" and given.stability_points == 1000
    default = plan(costs, **common)
    assert (
        default.rule == "time budgets" and default.stability_points >= STABILITY_MINIMUM
    )
    # The user's process (seconds per point, NinjaRS dd x80): 10 s cannot
    # hold the minimums.
    dear = {"ninjars": _costs(0.45, 80.0, start=5.0), "tirust": _costs(0.2, 30.0, 5.0)}
    short = plan(dear, completion=10.0, **common)
    assert not short.feasible and short.minimum_seconds > 10.0
    assert short.stability_points == STABILITY_MINIMUM
    assert all(c.batch == LANES and c.minimum == 1 for c in short.runtime.values())


@dataclass
class _World:
    """The true costs of a fake process: seconds per point of binary64 and
    the reference, a stability point's in binary64 and the reference, a
    process start-up; a first call ``slow`` times dearer; costs scaled by
    ``drift`` after ``drift_at`` seconds of fake time (host load)."""

    binary64: float
    multiplier: float
    start: float
    slow: float = 3.0
    drift: float = 1.0
    drift_at: float = math.inf

    def factor(self, now: float) -> float:
        return self.drift if now >= self.drift_at else 1.0


def _simulate(
    world: _World,
    budget: float,
    workers: int,
    prior_error: float,
    given: dict | None = None,
) -> tuple[float, dict]:
    """The whole profile under a fake clock: measurement (one point, again
    when cheap, then a full chunk when that is cheap, in binary64 and the
    reference), each
    backend's runtime (``_timed_calls`` with the adaptive warm-up and
    ``_reference_calls`` on the counts re-planned from what is left) and
    the stability run (the real admission gate; tasks of ``task_points``
    per backend and precision, ``workers`` workers taking them in order),
    the plan from measured costs that are off by ``prior_error``; ``given``:
    the count and size settings the user gave with the completion time,
    through ``loop_profile.effective_benchmark`` and used as the profile
    uses them (a count that survived would be honoured, breaking the
    budget). The achieved seconds and the counts."""

    from dataclasses import replace

    from pyamplicol.benchmarking import _reference_calls, _timed_calls
    from pyamplicol.config.models import is_default
    from pyamplicol.loop_profile import effective_benchmark

    user = replace(BenchmarkConfig(), **(given or {}), completion_time=budget)
    bench_used, _ = effective_benchmark(user)
    batch_max = int(bench_used.batch_size)
    runtime_given = any(
        not is_default(getattr(bench_used, name))
        for name in ("target_runtime", "minimum_samples", "batch_size", "warmup_runs")
    )
    stability_given = (
        None
        if is_default(bench_used.stability_points)
        else int(bench_used.stability_points)
    )

    clock = [0.0]
    backends = ("tirust", "ninjars")
    scale = {"tirust": 1.0, "ninjars": 3.0}
    # Load and measurement.
    clock[0] += 2.0
    measured = {}
    for b in backends:
        cost = world.binary64 * scale[b]
        for per_point in (cost, cost * world.multiplier):
            clock[0] += per_point * world.slow
            if per_point * world.slow <= MEASURE_REPEAT_SECONDS:
                clock[0] += per_point
            if LANES * per_point <= WARMUP_CALL_SECONDS:
                clock[0] += LANES * per_point
        measured[b] = _costs(cost * prior_error, world.multiplier, world.start)
    planned = plan(
        measured,
        runtime=True,
        stability=True,
        workers=workers,
        lanes=LANES,
        batch_max=batch_max,
        completion=budget,
        elapsed=clock[0],
        stability_start=world.start,
    )
    counts_out: dict = {"stability": None, "runtime": {}}
    for position, b in enumerate(backends):
        cost = world.binary64 * scale[b]
        left = budget - clock[0] - planned.stability_seconds
        share = left / (len(backends) - position)
        counts = (
            runtime_counts(
                measured[b],
                max(share - world.start, 0.0),
                lanes=LANES,
                batch_max=batch_max,
            )
            if planned.feasible
            else planned.runtime[b]
        )
        clock[0] += world.start
        calls = [0]

        def call(k: int, counts=counts, cost=cost, calls=calls) -> float:
            # The warm-up: one point when a full call is dear.
            points = (
                1
                if k < counts.warmup and counts.batch * cost > WARMUP_CALL_SECONDS
                else counts.batch
            )
            wall = (
                points * cost * world.factor(clock[0]) * (world.slow if k == 0 else 1.0)
            )
            clock[0] += wall
            calls[0] += 1
            return wall

        bench = (
            bench_used
            if runtime_given
            else replace(
                bench_used,
                batch_size=counts.batch,
                warmup_runs=counts.warmup,
                minimum_samples=counts.minimum,
                target_runtime=max(counts.budget, 1e-9),
            )
        )
        walls = _timed_calls(bench, call)
        reference = cost * world.multiplier

        def ref_call(k: int, points: int, reference=reference) -> float:
            warm = k == 0
            wall = (
                points
                * reference
                * world.factor(clock[0])
                * (world.slow if warm else 1.0)
            )
            clock[0] += wall
            return wall

        refs, _ = _reference_calls(
            counts.reference_budget, True, LANES, ref_call, None, None
        )
        counts_out["runtime"][b] = (len(walls), counts.batch, len(refs))
        assert len(walls) >= 1 and len(refs) >= 1
    # The stability run: chunks admitted by the gate, run by the workers (a
    # count that survived: that many points, no deadline).
    deadline = math.inf if stability_given is not None else budget
    planned_points = stability_points(
        list(measured.values()),
        max(budget - clock[0] - world.start, 0.0),
        workers,
        lanes=LANES,
    )
    clock[0] += world.start
    free = [clock[0]] * workers
    points = 0
    streams = {
        (b, precision): factor * world.binary64 * scale[b]
        for b in backends
        for precision, factor in (("binary64", 2.5), ("reference", world.multiplier))
    }
    rates = {b: streams[(b, "binary64")] + streams[(b, "reference")] for b in backends}
    estimate = sum(rates.values()) * prior_error
    while True:
        now = min(free)
        queued = sum(max(f - now, 0.0) for f in free)
        if points:
            # As the tracker: the measured cost of the chunks back so far.
            estimate = sum(rates[b] * world.factor(now) for b in backends)
        if stability_given is not None:
            size = min(LANES, stability_given - points)
        elif points == 0:
            # As the run: the minimum first unless the plan affords several
            # chunks.
            size = min(planned_points, LANES) if planned_points >= 4 * LANES else 5
        else:
            size = affordable_chunk(
                now=now,
                deadline=deadline,
                queued_seconds=queued,
                point_seconds=estimate,
                workers=workers,
                points=points,
                lanes=LANES,
                stream_seconds=max(r * world.factor(now) for r in streams.values()),
            )
        if not size:
            break
        for rate in streams.values():
            piece = task_points(rate * prior_error if points == 0 else rate, size)
            for at in range(0, size, piece):
                k = free.index(min(free))
                free[k] += min(piece, size - at) * rate * world.factor(free[k])
        points += size
    clock[0] = max(free)
    counts_out["stability"] = points
    assert points >= STABILITY_MINIMUM
    return clock[0], counts_out


@pytest.mark.parametrize("prior_error", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "world",
    [
        _World(2e-4, 15.0, 1.5),
        _World(3e-3, 60.0, 2.0),
        _World(3e-2, 40.0, 3.0, drift=1.8, drift_at=60.0),
        _World(0.3, 80.0, 5.0, slow=1.5),
        # The user's u u~ > z g g g g: seconds per binary64 point, a
        # reference point of a minute on the dearer backend.
        _World(1.0, 20.0, 13.0, slow=1.2),
    ],
)
@pytest.mark.parametrize("workers", [1, 4, 8])
def test_the_completion_time_is_honoured(world, prior_error, workers):
    """Across costs from a cheap process to the user's 7-point one, priors
    off by a factor 2, a drift mid-run and 1 or 4 workers: when the minimums
    fit the budget the profile ends within the tolerance of it."""

    for budget in (120.0, 600.0, 1800.0):
        total, counts = _simulate(world, budget, workers, prior_error)
        minimum_plan = plan(
            {
                b: _costs(world.binary64 * s, world.multiplier, world.start)
                for b, s in (("tirust", 1.0), ("ninjars", 3.0))
            },
            runtime=True,
            stability=True,
            workers=workers,
            lanes=LANES,
            batch_max=128,
            completion=budget,
        )
        if minimum_plan.feasible and minimum_plan.minimum_seconds < 0.5 * budget:
            ratio = total / budget
            assert TOLERANCE[0] <= ratio <= TOLERANCE[1], (budget, total, counts)


def test_an_infeasible_budget_runs_the_minimums_and_estimates_them():
    """When the minimums exceed the budget the plan says so and its estimate
    of them is right within the tolerance."""

    world = _World(0.45, 80.0, 5.0, slow=1.2)
    budget = 60.0
    measured = {
        b: _costs(0.45 * s, 80.0, 5.0) for b, s in (("tirust", 1.0), ("ninjars", 3.0))
    }
    planned = plan(
        measured,
        runtime=True,
        stability=True,
        workers=4,
        lanes=LANES,
        batch_max=128,
        completion=budget,
        stability_start=5.0,
    )
    assert not planned.feasible
    total, counts = _simulate(world, budget, 4, 1.0)
    assert counts["stability"] == STABILITY_MINIMUM or counts["stability"] <= LANES
    assert all(calls >= 1 for calls, _, _ in counts["runtime"].values())
    ratio = total / max(planned.minimum_seconds, 1e-9)
    assert TOLERANCE[0] <= ratio <= TOLERANCE[1] * 1.5, (total, planned.minimum_seconds)


def test_the_profile_eta_on_synthetic_sequences(monkeypatch):
    """The profile's phase ETA (``recent_rate``, ``_BackendRun.remaining``): a
    prior until a call is back, then the median of the recent calls (a slow
    first call and one outlier do not move it, a drift is followed within a
    few calls), counted down while a call runs and never 0 while one
    remains; the whole command's ETA adds the later phases (a points-weighted
    mix: each backend's own calls at its own cost)."""

    import pyamplicol.loop_profile as lp
    from pyamplicol.benchmarking import recent_rate

    assert recent_rate([]) is None
    assert recent_rate([10.0, 1.0, 1.0, 1.0, 1.0, 1.0]) == 1.0
    assert recent_rate([1.0, 1.0, 1.0, 30.0, 1.0]) == 1.0
    assert recent_rate([1.0, 1.0, 3.0, 3.0, 3.0]) == 3.0
    clock = [0.0]
    monkeypatch.setattr(lp.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("pyamplicol._clock.command_start", lambda: None)
    monitor = lp._Monitor(None, "x", None)
    run = lp._BackendRun(monitor, "tirust", lp.GroundTruth("double-double"), 2.0)
    run.event(
        "binary64",
        {"done": 1, "total": 6, "eta_seconds": 50.0, "walls": [], "points": 16},
    )
    # A prior (the warm-up's 10 s per call): 5 calls left.
    assert run.remaining() == pytest.approx(50.0)
    clock[0] += 4.0
    assert run.remaining() == pytest.approx(46.0)
    walls = [1.0, 1.0, 1.0]
    run.event(
        "binary64",
        {"done": 4, "total": 6, "eta_seconds": 2.0, "walls": walls, "points": 16},
    )
    assert run.remaining() == pytest.approx(2.0)
    # A call overrunning its expected second: never 0, a tenth of a call.
    clock[0] += 5.0
    assert run.remaining() == pytest.approx(1.0 + 0.1)
    # The backend's planned later phases count until they start: in its
    # start-up, its binary64 and reference phases; in binary64, the
    # reference; then the phase alone. The phase line keeps its own.
    planned = lp._BackendRun(
        monitor,
        "ninjars",
        lp.GroundTruth("double-double"),
        3.0,
        12.0,
        16,
        {"binary64": 30.0, "reference": 70.0},
    )
    assert planned.remaining() == pytest.approx(3.0 + 30.0 + 70.0)
    assert planned._phase_remaining() == pytest.approx(3.0)
    planned.event(
        "binary64", {"done": 0, "total": 2, "warmup": 1, "walls": [], "points": 16}
    )
    assert planned.remaining() == pytest.approx(2 * 12.0 + 70.0)
    planned.event("binary64-result", {"profile": _Profile()})
    planned.event(
        "reference",
        {"done": 0, "total": 2, "eta_seconds": 60.0, "walls": [], "points": 1},
    )
    assert planned.remaining() == pytest.approx(60.0)


class _Profile:
    seconds_per_point = 0.6
    samples = 1
    batch = 16


class _Sink:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


def test_a_slow_reference_pass_reports_progress(monkeypatch):
    """The user's "hang": after the binary64 calls the reference pass runs
    for minutes. Driven through the real monitor with a fake clock, it is
    a phase of its own whose updates keep coming (every heartbeat), name the
    running call and its expected cost, and never show an ETA of 0 while a
    call remains; a call far past its expected cost says "no progress"."""

    import pyamplicol.loop_profile as lp
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.reporting.progress import ProgressEnd, ProgressStart, ProgressUpdate

    clock = [1000.0]
    monkeypatch.setattr(lp.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("pyamplicol._clock.command_start", lambda: None)
    sink = _Sink()
    monitor = lp._Monitor(sink, "Profile x", None)
    run = lp._BackendRun(monitor, "ninjars", lp.GroundTruth("double-double"), 3.0)
    run.event("loaded", {"seconds": 2.0})
    for k in (1, 2):
        clock[0] += 7.0
        run.event(
            "binary64",
            {
                "done": k,
                "total": 2,
                "eta_seconds": (2 - k) * 7.0,
                "walls": [7.0] if k == 2 else [],
                "points": 16,
                "seconds_per_point": 0.44 if k == 2 else None,
            },
        )
    profile = VirtualProfileResult(
        "p", "u u~ > z g g g g", "virtual", 16, 1, 1, 0.44, 0.44, 0.44, {}, {}, True, 0
    )
    run.event("binary64-result", {"profile": profile})
    clock[0] += 0.1
    run.event(
        "reference",
        {
            "done": 0,
            "total": 2,
            "eta_seconds": 2 * 16 * 30.0,
            "walls": [],
            "points": 16,
            "seconds_per_point": None,
            "estimated": True,
        },
    )
    start = len(sink.events)
    for _ in range(1600):  # 27 minutes of heartbeats inside one call
        clock[0] += 1.0
        run.tick()
    updates = [e for e in sink.events[start:] if isinstance(e, ProgressUpdate)]
    assert len(updates) == 1600
    reference = next(
        e
        for e in sink.events
        if isinstance(e, ProgressStart) and "reference" in e.description
    )
    assert all(e.task_id == reference.task_id for e in updates)
    for event in updates:
        assert event.completed < event.total
        assert (
            event.details["eta_seconds"] is not None
            and event.details["eta_seconds"] > 0.0
        )
        assert "running" in event.details["step"]
    # Past three times its expected cost (plus 30 s): "no progress".
    assert "of ~8m00s" in updates[0].details["step"]
    assert "stalled_seconds" not in updates[1400].details
    assert updates[-1].details.get("stalled_seconds", 0.0) > 0.0
    # The binary64 phase ended with its summary before the reference began.
    ended = [e for e in sink.events if isinstance(e, ProgressEnd)]
    assert any(e.message and "per point" in e.message for e in ended)
    # Ctrl-C now keeps the binary64 result, with a reference of no call.
    partial = run.partial()
    assert partial.seconds_per_point == 0.44 and partial.reference.samples == 0


def test_stability_tasks_and_warmups_follow_the_cost_of_a_point():
    """A stability task holds about TASK_SECONDS of work (an expensive
    point is a task of its own, a cheap chunk stays whole); the binary64
    warm-up is one point when a full call is dear; the wall of an expensive
    process's few points is that of its dearest point, not of a serial
    chunk."""

    assert task_points(1e-3, 16) == 16 and task_points(0.0, 16) == 16
    assert task_points(TASK_SECONDS / 4, 16) == 4
    assert task_points(35.0, 5) == 1 and task_points(35.0, 0) == 0
    assert warmup_seconds(1.0, 0.1) == 1.0
    assert warmup_seconds(48.0, 3.0) == 3.0
    dear = [Costs(0.6, 35.0, 1.5, 35.0, 4.0), Costs(3.0, 11.0, 7.5, 11.0, 4.0)]
    wall = stability_wall(dear, STABILITY_MINIMUM, 8)
    work = STABILITY_MINIMUM * sum(c.stability + c.stability_reference for c in dear)
    assert wall == pytest.approx(max(work / 8, 35.0))
    assert stability_wall(dear, STABILITY_MINIMUM, 1) == pytest.approx(work)


def test_the_users_process_fits_ten_minutes():
    """The user's u u~ > z g g g g on 8 cores (measured: NinjaRS 0.67 s and
    a 35 s dd point, TIRust 3.0 s and an 11 s dd point, 13 s per process
    start-up): --completion-time 10m is feasible, its minimums a few
    minutes (they ran in 10 minutes and were estimated at 29 before the
    reference was measured and the stability points spread over the
    workers)."""

    costs = {
        "ninjars": Costs(0.67, 35.0, 2.5 * 0.67, 35.0, 13.0),
        "tirust": Costs(3.0, 11.0, 7.5, 11.0, 13.0),
    }
    planned = plan(
        costs,
        runtime=True,
        stability=True,
        workers=8,
        lanes=LANES,
        batch_max=128,
        completion=600.0,
        elapsed=60.0,
        stability_start=23.0,
    )
    assert planned.feasible
    assert 150.0 <= planned.minimum_seconds <= 400.0, planned.minimum_seconds
    assert planned.stability_points is not None
    assert planned.stability_points >= STABILITY_MINIMUM


# Counts and sizes given with --completion-time: small and large, every one
# of them (they are overruled: the time is still honoured).
_GIVEN = [
    {
        "stability_points": 10,
        "batch_size": 16,
        "minimum_samples": 1,
        "target_runtime": 0.1,
        "warmup_runs": 0,
    },
    {
        "stability_points": 200_000,
        "batch_size": 4096,
        "minimum_samples": 50,
        "target_runtime": 900.0,
        "warmup_runs": 5,
    },
    {"stability_points": 10},
    {"batch_size": 32},
]


@pytest.mark.parametrize("given", _GIVEN)
@pytest.mark.parametrize(
    "world",
    [
        _World(2e-4, 15.0, 1.5),
        _World(3e-2, 40.0, 3.0, drift=1.8, drift_at=60.0),
        _World(1.0, 20.0, 13.0, slow=1.2),
    ],
)
@pytest.mark.parametrize("workers", [1, 8])
def test_the_completion_time_overrules_the_counts_given(world, given, workers):
    """--completion-time with conflicting counts and sizes (the verifier's
    D2: --stability-points 10 with 20 s ran 7.8 s): the counts are those of
    the time, the completion time honoured as without them."""

    for budget in (120.0, 600.0):
        total, counts = _simulate(world, budget, workers, 1.0, given)
        alone, _ = _simulate(world, budget, workers, 1.0)
        minimum_plan = plan(
            {
                b: _costs(world.binary64 * s, world.multiplier, world.start)
                for b, s in (("tirust", 1.0), ("ninjars", 3.0))
            },
            runtime=True,
            stability=True,
            workers=workers,
            lanes=LANES,
            batch_max=128,
            completion=budget,
        )
        assert total == pytest.approx(alone), (given, budget)
        if minimum_plan.feasible and minimum_plan.minimum_seconds < 0.5 * budget:
            assert TOLERANCE[0] <= total / budget <= TOLERANCE[1], (given, counts)
        assert all(batch <= 128 for _, batch, _ in counts["runtime"].values())


def test_effective_benchmark_resets_every_count_and_size():
    """Every count and size option (command line or card: the same fields)
    is named and reset under --completion-time, honoured without it."""

    from dataclasses import replace

    from pyamplicol.config.models import is_default
    from pyamplicol.loop_profile import _COUNT_OPTIONS, effective_benchmark

    given = _GIVEN[0]
    flags = {flag for flag, _ in _COUNT_OPTIONS}
    timed = replace(BenchmarkConfig(), **given, completion_time=20.0)
    used, overruled = effective_benchmark(timed)
    assert set(overruled) == flags and used.completion_time == 20.0
    assert all(is_default(getattr(used, name)) for _, name in _COUNT_OPTIONS)
    plain = replace(BenchmarkConfig(), **given)
    kept, named = effective_benchmark(plain)
    assert kept is plain and set(named) == flags


def test_a_budget_in_force_is_checked_whatever_else_is_given():
    """recheck2 N2: with a stability count given the runtime still runs on
    its default 10 s budget; on the user's process its minimums take minutes,
    so the plan names it unmet. A runtime count given makes --target-runtime
    (per backend) the budget in force; stability points given set none."""

    dear = {
        "ninjars": Costs(0.67, 35.0, 2.5 * 0.67, 35.0, 13.0),
        "tirust": Costs(3.0, 11.0, 7.5, 11.0, 13.0),
    }
    cheap = {name: _costs(2e-4) for name in ("ninjars", "tirust")}
    common = dict(runtime=True, stability=True, workers=8, lanes=LANES, batch_max=128)
    points = plan(dear, **common, stability_points_given=10)
    assert points.rule == "counts" and points.unmet == ("runtime",)
    assert not points.feasible and points.record()["unmet"] == ["runtime"]
    target = plan(dear, **common, runtime_counts_given=True, runtime_target=2.0)
    assert "runtime" in target.unmet
    roomy = plan(
        dear,
        **common,
        runtime_counts_given=True,
        runtime_target=600.0,
        stability_points_given=10,
    )
    assert roomy.feasible and roomy.unmet == ()
    assert plan(cheap, **common, stability_points_given=10).feasible
    assert plan(cheap, **common).feasible


class _StubResult:
    def __init__(self, size: int) -> None:
        self.valid = [True] * size
        self.diagnostics = [None] * size


class _StubBackend:
    def __init__(self, with_prepare: bool) -> None:
        self.prepared: list[dict] = []
        if with_prepare:
            self.prepare = lambda **kwargs: self.prepared.append(kwargs)

    def _reduction_summary(self):
        return {"requested": "ninjars", "auto_default": "ninjars"}


class _StubRuntime:
    def __init__(self, with_prepare: bool) -> None:
        self._backend = _StubBackend(with_prepare)
        self.calls: list[tuple[int, str | None]] = []

    def evaluate(self, batch, *, mu_r2=None, **options):
        self.calls.append((len(batch), options.get("arithmetic")))
        return _StubResult(len(batch))


@pytest.mark.parametrize(
    ("with_prepare", "entry", "prepared"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_the_warm_entry_replaces_a_dear_reference_warm_up(
    with_prepare, entry, prepared
):
    """The reference pass warms its carrier through the runtime's warm entry
    (``prepare``) instead of the untimed warm-up point when the caller says
    so (``reference["entry"]``, ``profile_plan.takes_entry``): the entry
    runs once in the timed calls' carrier, backend and unchecked mode, no
    untimed call is made and the first timed call is call 0; otherwise, or
    on a runtime without the entry, today's untimed point."""

    from pyamplicol.benchmarking import _PointBatches, _reference_timing

    runtime = _StubRuntime(with_prepare)
    points = [((1.0, 0.0, 0.0, 1.0),) * 4] * 4
    timing = _reference_timing(
        runtime,
        _PointBatches(1, points),
        BenchmarkConfig(batch_size=1, warmup_runs=1, minimum_samples=1),
        "ninjars",
        lambda batch: [1.0] * len(batch),
        {
            "arithmetic": "double-double",
            "reduction": "ninjars",
            "budget": 1e-9,
            "seconds_per_point": 10.0,
            "entry": entry,
        },
        binary64=1e-3,
    )
    calls = [c for c in runtime.calls if c[1] == "double-double"]
    if prepared:
        assert runtime._backend.prepared == [
            {"arithmetic": "double-double", "reduction": "ninjars", "unchecked": True}
        ]
        assert timing.warmup == 0 and timing.prepared_seconds is not None
        assert calls == [(1, "double-double")]
    else:
        assert runtime._backend.prepared == []
        assert timing.warmup == 1 and timing.prepared_seconds is None
        assert calls == [(1, "double-double"), (1, "double-double")]
    assert timing.samples == 1 and timing.points == 1


def test_the_scope_rules_at_their_thresholds():
    """``rule_b_drops`` / ``rule_a_drops`` on ``lower_bound``: exactly at the
    bound nothing is dropped, just above it is; rule A counts the reference
    only when asked to (invariant ii), and the stability part with workers."""

    from pyamplicol.profile_plan import (
        Known,
        lower_bound,
        rule_a_drops,
        rule_b_bound,
        rule_b_drops,
    )

    a = Known(point=0.8, chunk=0.3, reference=20.0, start=3.0)
    b = Known(point=1.0)
    with_reference = lower_bound(a, lanes=LANES, reference=True)
    assert with_reference == pytest.approx(16 * 0.3 + 0.3 + 20.0 + 0.3)
    alone = lower_bound(a, lanes=LANES, reference=False)
    known = {"ninjars": a, "tirust": b}
    both = with_reference + lower_bound(b, lanes=LANES, reference=True)
    # Rule B on the sum of every backend's lower bound (B-1: a cheap default
    # with a dear second backend's reference); one not measured counts 0.
    assert rule_b_drops(known, lambda n: both * (1 - 1e-9), lanes=LANES)
    assert not rule_b_drops(known, lambda n: both, lanes=LANES)
    assert not rule_b_drops(
        {"ninjars": a, "tirust": Known()}, lambda n: with_reference, lanes=LANES
    )
    assert rule_b_bound(known, "ninjars", lambda n: both, lanes=LANES) == (
        pytest.approx(with_reference)
    )

    def drops(bound, reference, workers=None):
        return rule_a_drops(
            known,
            "ninjars",
            lambda n: bound,
            lanes=LANES,
            reference=reference,
            stability_workers=workers,
        )

    assert not drops(alone, False) and drops(alone * (1 - 1e-9), False)
    # Not counted (rule B could drop it), a dear reference never drops a backend.
    assert not drops(with_reference / 2, False)
    assert drops(with_reference / 2, True)
    stability = lower_bound(a, lanes=LANES, reference=True, stability_workers=2)
    assert stability > with_reference
    assert not drops(stability, True, 2) and drops(stability * (1 - 1e-9), True, 2)
    # Without the runtime timing (--stability alone) its terms are not
    # counted; one worker leaves only its first reference point cold (B-2).
    alone_stability = lower_bound(
        a, lanes=LANES, reference=True, stability_workers=1, runtime=False
    )
    assert alone_stability == pytest.approx(5 * 0.3 + 20.0 + 4 * 0.3)
    assert lower_bound(
        a, lanes=LANES, reference=True, stability_workers=8, runtime=False
    ) == pytest.approx(20.0)
    # Through the warm entry (m-4): no cold point and no warm-up point, the
    # entries' measured seconds once they ran; the stability run's workers
    # still warm cold.
    entry = replace(a, warm_entry=True, reference_entry=True)
    assert lower_bound(entry, lanes=LANES, reference=True) == pytest.approx(
        16 * 0.3 + 0.3
    )
    ran = replace(entry, warm=0.12, reference_warm=0.25)
    assert lower_bound(ran, lanes=LANES, reference=True) == pytest.approx(
        16 * 0.3 + 0.12 + 0.25 + 0.3
    )
    assert lower_bound(
        ran, lanes=LANES, reference=True, stability_workers=1, runtime=False
    ) == pytest.approx(alone_stability)
    # One backend: rule A has nothing to drop; nothing known: nothing dropped.
    assert not rule_a_drops(
        {"ninjars": a}, "ninjars", lambda n: 0.0, lanes=LANES, reference=True
    )
    assert not rule_b_drops({"ninjars": Known()}, lambda n: 0.0, lanes=LANES)


def test_the_decider_counts_the_entry_not_the_cold_point():
    """``benchmarking._scope_drops`` (the decider's runtime timing, rule B
    open): a reference that warms through the entry counts the entry's
    measured seconds, none before it ran, never the parent's cold point;
    after an untimed point (no entry) that point counts (m-4)."""

    from types import SimpleNamespace

    from pyamplicol.benchmarking import _scope_drops

    profile = SimpleNamespace(seconds_per_point=0.3, prepared_seconds=0.12)
    call = 16 * 0.3 + 0.12 + 0.3  # the chunk call, its entry, a timed point

    def scope(entry):
        known = {"point": 0.8, "reference": 16.0, "start": 3.0}
        return {"bound": call + 1.0, "known": {**known, "reference_entry": entry}}

    # The cold point (16 s) would drop it; the entry's 0.25 s does not.
    assert _scope_drops(profile, scope(False), 16)
    assert not _scope_drops(profile, scope(True), 16)
    assert not _scope_drops(profile, scope(True), 16, entry=0.25)
    assert _scope_drops(profile, scope(True), 16, entry=1.5)
    # An untimed point ran (the entry did not): it counts.
    assert _scope_drops(profile, scope(True), 16, reference_point=16.0)


def _known_after(stage, true, known):
    """``known`` after one more measured number of a backend's profile
    (with its binary64 point: whether its streams warm through the entry;
    with its chunk: the binary64 entry's seconds where it ran)."""

    from pyamplicol.profile_plan import Known

    k = known or Known()
    numbers = {stage: true[stage]}
    if stage == "point":
        numbers.update(warm_entry=true["entry"], reference_entry=true["ref_entry"])
    if stage == "chunk":
        numbers["warm_entry"] = true["entry"]
        if true["entry"]:
            numbers["warm"] = true["warm"]
    return replace(k, **numbers)


def _reference_warms(known, true, *, scope):
    """The runtime timing's reference warming: the entry where its stream
    takes it (with rule B open, ``scope``, only with its cost known: else
    the untimed point measures it), else the untimed point (m-4). The
    updated ``known`` and whether the entry warmed it."""

    if true["ref_entry"] and (known.reference is not None or not scope):
        return replace(
            known, reference_entry=True, reference_warm=true["reference_warm"]
        ), True
    if known.reference is None:
        known = _known_after("reference", true, known)
    return replace(known, reference_entry=False), False


def _scope_path(path, mode, backends):
    """The profile's call points of the scope rules, on the numbers each
    path has measured at each point (``loop_profile``): the measurement
    (``measured``: the default's binary64 point, then its reference unless
    rule B has decided, then the other backends' points; its early stop),
    the decision before the runtime timing, the decider's worker after its
    binary64 call and after its untimed reference point, the default's row.
    The verdict: (reference timed, backends left out)."""

    from pyamplicol.profile_plan import Known, rule_a_drops, rule_b_drops

    names = list(backends)
    default = names[0]
    bound = mode["bound"]
    known = {name: Known() for name in names}
    # With rule B open no reference warms through the entry.
    backends = {
        name: {**costs, "ref_entry": costs["entry"] and not mode["rule_b"]}
        for name, costs in backends.items()
    }

    def b_drops():
        # Judged on the backends rule A has not left out.
        running = {n: k for n, k in known.items() if n not in omitted}
        return mode["rule_b"] and rule_b_drops(running, bound, lanes=LANES)

    def a_drops():
        return (
            mode["rules"]
            and not mode["asked"]
            and rule_a_drops(
                known,
                default,
                bound,
                lanes=LANES,
                reference=mode["a_reference"],
                stability_workers=mode["workers"],
                runtime=mode["runtime"],
            )
        )

    omitted = ()
    # Without the runtime timing there is no path without a measurement.
    if path == "measured" or not mode["runtime"]:
        for position, name in enumerate(names):
            known[name] = _known_after("point", backends[name], known[name])
            if not b_drops():
                known[name] = _known_after("reference", backends[name], known[name])
            if position == 0 and a_drops():
                omitted = tuple(names[1:])
                break
    if not omitted and a_drops():
        omitted = tuple(names[1:])
    timed = mode["truth"] and not b_drops()
    if not mode["runtime"]:
        return timed, omitted, known, {name: False for name in names}
    # The runtime timing: the default backend first.
    known[default] = _known_after("chunk", backends[default], known[default])
    # Whether each backend's reference warms through the entry on this path
    # (the default's decided below; the others' timings have no scope).
    through = {name: backends[name]["ref_entry"] for name in names}
    if timed and b_drops():
        timed = False
    elif timed:
        # Its warming (the entry, or its untimed point), rule B re-called
        # on it while open, then its timed point (all reported by the row).
        known[default], through[default] = _reference_warms(
            known[default], backends[default], scope=mode["rule_b"]
        )
        if mode["rule_b"] and b_drops():
            timed = False
        else:
            known[default] = _known_after(
                "reference_timed", backends[default], known[default]
            )
    if not omitted and a_drops():
        omitted = tuple(names[1:])
    return timed, omitted, known, through


def _true_minimum(costs, *, reference, workers, runtime=True, through=False):
    """A backend's real minimal profile (what the lower bounds bound): its
    binary64 warm-up its entry's seconds where its streams take the entry,
    else a point; its reference warmed by the entry (``through``) or by its
    cold point; a stability worker's first reference point cold, its others
    warm."""

    seconds = 0.0
    if runtime:
        warm = costs["warm"] if costs["entry"] else costs["point"]
        seconds += LANES * costs["chunk"] + warm
        if reference:
            untimed = costs["reference_warm"] if through else costs["reference"]
            seconds += untimed + costs["reference_timed"]
    if workers is not None:
        colds = min(workers, STABILITY_MINIMUM)
        stream = (
            colds * costs["reference"]
            + (STABILITY_MINIMUM - colds) * costs["reference_timed"]
        ) / STABILITY_MINIMUM
        seconds += stability_wall(
            [Costs(costs["point"], None, 2.5 * costs["point"], stream, 0)],
            STABILITY_MINIMUM,
            workers,
        )
    return seconds


_BINARY64 = {"cheap": (2e-3, 6e-4), "mid": (0.8, 0.3), "dear": (200.0, 90.0)}
# The reference's cold (first) and warm points: "cheap" scales with the
# binary64 point, "cold" is dear on its first point only.
_REFERENCE = {"cheap": None, "cold": (45.0, 6.0), "dear": (30.0, 25.0)}


def _modes():
    tolerance = 1.5
    yield (
        "default",
        {
            "bound": lambda n: tolerance * 10.0,
            "rule_b": True,
            "a_reference": False,
            "workers": None,
            "asked": False,
            "rules": True,
            "truth": True,
            "runtime": True,
        },
    )
    yield (
        "target 3",
        {
            "bound": lambda n: tolerance * 3.0 * n,
            "rule_b": True,
            "a_reference": False,
            "workers": None,
            "asked": False,
            "rules": True,
            "truth": True,
            "runtime": True,
        },
    )
    yield (
        "stability alone",
        {
            "bound": lambda n: tolerance * 60.0,
            "rule_b": False,
            "a_reference": True,
            "workers": 1,
            "asked": False,
            "rules": True,
            "truth": True,
            "runtime": False,
        },
    )
    yield (
        "runtime and stability",
        {
            "bound": lambda n: tolerance * 70.0,
            "rule_b": False,
            "a_reference": True,
            "workers": 2,
            "asked": False,
            "rules": True,
            "truth": True,
            "runtime": True,
        },
    )
    yield (
        "ground truth given",
        {
            "bound": lambda n: tolerance * 10.0,
            "rule_b": False,
            "a_reference": True,
            "workers": None,
            "asked": False,
            "rules": True,
            "truth": True,
            "runtime": True,
        },
    )
    yield (
        "all backends",
        {
            "bound": lambda n: tolerance * 10.0,
            "rule_b": True,
            "a_reference": False,
            "workers": None,
            "asked": True,
            "rules": True,
            "truth": True,
            "runtime": True,
        },
    )
    yield (
        "counts given",
        {
            "bound": lambda n: tolerance * 10.0,
            "rule_b": False,
            "a_reference": True,
            "workers": None,
            "asked": False,
            "rules": False,
            "truth": True,
            "runtime": True,
        },
    )


def _cases():
    """One backend, or two whose references may differ (the default's
    first): the same binary64 costs, a reference at least the binary64 point
    (a wider carrier costs at least the binary64 pass); without or with the
    warm entry on every stream (``entry``: a quarter of the binary64 point,
    an eighth of the cold reference point, m-4)."""

    pairs = [(r,) for r in _REFERENCE] + list(itertools.product(_REFERENCE, repeat=2))
    for mode_name, mode in _modes():
        for b64, refs, entry in itertools.product(_BINARY64, pairs, (False, True)):
            point, chunk = _BINARY64[b64]
            backends = {}
            for name, ref in zip(("ninjars", "tirust"), refs, strict=False):
                cold, warm = _REFERENCE[ref] or (10 * point, 8 * point)
                cold = max(cold, 10 * point)
                backends[name] = {
                    "point": point,
                    "chunk": chunk,
                    "reference": cold,
                    "reference_timed": max(warm, 8 * point),
                    "entry": entry,
                    "warm": point / 4,
                    "reference_warm": cold / 8,
                }
            yield (mode_name, b64, *refs, "entry" if entry else "point"), mode, backends


@pytest.mark.parametrize(
    ("case", "mode", "backends"),
    list(_cases()),
    ids=lambda value: "-".join(map(str, value)) if isinstance(value, tuple) else "",
)
def test_the_scope_verdicts_do_not_depend_on_the_path(case, mode, backends):
    """The verdicts on every path (invariant iii: the measured path, the
    path without a measurement phase; a previous profile's costs never
    decide, so with or without history is one of these), over cheap / mid /
    dear binary64, cheap / cold / dear references (the two backends' may
    differ), one or two backends and the budget modes: rule A the same on
    every path, rule B differing only by what the measured path measured;
    every drop on a real minimum past the budget (i); rule A never drops a
    backend for a reference rule B could drop (ii)."""

    from pyamplicol.profile_plan import Known, lower_bound

    measured, measured_omitted, _, measured_through = _scope_path(
        "measured", mode, backends
    )
    blind, blind_omitted, _, blind_through = _scope_path("blind", mode, backends)
    names = list(backends)
    default = backends[names[0]]
    bound = mode["bound"]
    assert measured_omitted == blind_omitted, case
    # Rule B: the measured path knows more (the other backends' binary64
    # points and references): it drops whenever the blind path does, and
    # may drop where the blind path cannot see it (ptimereview5 B-1). With
    # the warm entry the blind path's untimed cold point measures the
    # reference (its cost unknown) where the measured path warms it through
    # the entry: a drop on what that path ran (m-4).
    assert (
        blind
        or not measured
        or (measured_through[names[0]] and not blind_through[names[0]])
    ), case
    # Rule B is judged on the backends rule A keeps.
    running = [n for n in names if n not in measured_omitted]
    for timed, through in ((measured, measured_through), (blind, blind_through)):
        if mode["truth"] and not timed:
            # (i): the minimums of the running backends with their reference
            # as it warms on this path (the entry: never the cold point).
            assert sum(
                _true_minimum(
                    backends[n], reference=True, workers=None, through=through[n]
                )
                for n in running
            ) > bound(len(running)), case
    if mode["rule_b"] and measured:
        # B-1: what the measured path measured (every binary64 point and
        # reference) summed never exceeds the bound with the references kept.
        assert sum(
            lower_bound(
                Known(
                    point=costs["point"],
                    reference=costs["reference"],
                    warm_entry=costs["entry"],
                    reference_entry=costs["entry"] and not mode["rule_b"],
                ),
                lanes=LANES,
                reference=True,
            )
            for costs in (backends[n] for n in running)
        ) <= bound(len(running)), case
    if measured_omitted:
        assert _true_minimum(
            default,
            reference=mode["a_reference"],
            workers=mode["workers"],
            runtime=mode["runtime"],
            through=measured_through[names[0]],
        ) > bound(1), case
        # (ii): never for a reference rule B could drop.
        if mode["rule_b"]:
            assert _true_minimum(default, reference=False, workers=None) > bound(1)


def test_a_stream_takes_the_warm_entry_by_its_costs():
    """``takes_entry``: the entry when the point costs more than the entry's
    remembered cost, else (none remembered) more than the measured
    first-call excess, else (no cost known) the entry; and every plan counts
    the entry, not a point, for such a stream (ptimereview2 W-B1)."""

    from pyamplicol.profile_plan import (
        binary64_warmup,
        reference_warmup,
        runtime_minimum,
        takes_entry,
    )

    assert takes_entry(None) and takes_entry(1e-3)
    assert takes_entry(10.0, entry=5.0) and not takes_entry(5.0, entry=5.0)
    assert not takes_entry(0.7, excess=1.5) and takes_entry(20.0, excess=5.0)
    # The remembered entry wins over the excess.
    assert not takes_entry(3.0, entry=5.8, excess=1.0)
    point = _costs(200.0, 30.0, 10.0)
    warmed = replace(point, binary64_warm=0.5, reference_warm=70.0)
    call = LANES * 200.0
    assert binary64_warmup(point, call) == warmup_seconds(call, 200.0)
    assert binary64_warmup(warmed, call) == 0.5
    assert reference_warmup(point) == 6000.0 and reference_warmup(warmed) == 70.0
    assert runtime_minimum(warmed, lanes=LANES) == pytest.approx(
        10.0 + call + 0.5 + 70.0 + 6000.0
    )
    assert runtime_minimum(point, lanes=LANES) - runtime_minimum(
        warmed, lanes=LANES
    ) == pytest.approx(200.0 - 0.5 + 6000.0 - 70.0)
    counts = runtime_counts(warmed, 0.0, lanes=LANES, batch_max=128)
    assert counts.expected == pytest.approx(10.0 + 0.5 + call + 70.0 + 6000.0)


def test_the_lower_bound_counts_the_measured_precisions():
    """``lower_bound`` with --measured-precisions: binary64's calls only
    when listed (``binary64``), every other precision's stream at least two
    binary64 points in the runtime timing (its measuring call and one timed
    call) and one per stability point (``extra``); the scope rules pass
    them through, so a rule drops no earlier than the bound allows."""

    from pyamplicol.profile_plan import Known, lower_bound, rule_a_drops, rule_b_drops

    a = Known(point=0.8, chunk=0.3, start=3.0)
    alone = lower_bound(a, lanes=LANES, reference=False)
    assert alone == pytest.approx(16 * 0.3 + 0.3)
    two = lower_bound(a, lanes=LANES, reference=False, extra=2)
    assert two == pytest.approx(alone + 2 * 2 * 0.3)
    unlisted = lower_bound(a, lanes=LANES, reference=False, binary64=False, extra=1)
    assert unlisted == pytest.approx(2 * 0.3)
    stability = lower_bound(
        a, lanes=LANES, reference=False, stability_workers=1, runtime=False
    )
    more = lower_bound(
        a, lanes=LANES, reference=False, stability_workers=1, runtime=False, extra=1
    )
    assert more == pytest.approx(stability + STABILITY_MINIMUM * 0.3)
    known = {"ninjars": a, "tirust": Known(point=1.0)}
    bound = two * (1 - 1e-9)
    assert not rule_a_drops(
        known, "ninjars", lambda n: bound, lanes=LANES, reference=False
    )
    assert rule_a_drops(
        known, "ninjars", lambda n: bound, lanes=LANES, reference=False, extra=2
    )
    total = sum(
        lower_bound(k, lanes=LANES, reference=True, extra=1) for k in known.values()
    )
    assert rule_b_drops(known, lambda n: total * (1 - 1e-9), lanes=LANES, extra=1)
    assert not rule_b_drops(known, lambda n: total * (1 - 1e-9), lanes=LANES)
