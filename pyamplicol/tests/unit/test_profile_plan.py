# SPDX-License-Identifier: 0BSD
"""The loop profile's time budgets (``profile_plan``), its scheduling loops
(``benchmarking._timed_calls``, ``_reference_calls``, the stability run's
admission gate) under a fake clock, the shared ETA estimator, and the live
progress of a slow reference pass (the user's "hang", 2026-11-16)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import pytest

from pyamplicol.config import BenchmarkConfig
from pyamplicol.profile_plan import (
    MEASURE_REPEAT_SECONDS,
    STABILITY_MINIMUM,
    TASK_SECONDS,
    WARMUP_CALL_SECONDS,
    Costs,
    affordable_chunk,
    plan,
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
