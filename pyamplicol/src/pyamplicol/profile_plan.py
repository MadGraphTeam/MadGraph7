# SPDX-License-Identifier: 0BSD
"""Time budgets of ``pyamplicol profile`` of a loop artifact: from measured
costs and a budget to the counts of every phase (pure functions; the
scheduling loops in ``benchmarking`` and ``loop_profile`` re-plan from what
is left as they run).

Three rules set the counts, in this order (``Plan.rule``):

- ``--completion-time`` (``benchmark.completion_time``): the whole command
  aims at that wall time; it overrules every count. The budget left after
  the measurement phase goes first to the runtime timing (a stable median:
  ``STABLE_CALLS`` calls of up to ``batch_size`` points per backend, at most
  half of it when the stability run is asked for too), the rest to the
  stability run; the reference pass of each backend gets a quarter of its
  runtime share and its chunk shrinks to what that affords.
- counts given (the command line or a run card): honoured as given
  (``--target-runtime``, ``--minimum-samples``, ``--batch-size``,
  ``--warmup-runs`` for the runtime; ``--stability-points``).
- neither: the default budgets, ``DEFAULT_RUNTIME_SECONDS`` for the runtime
  timing of all backends and ``DEFAULT_STABILITY_SECONDS_PER_CORE`` of work
  per stability worker (about a minute of wall time whatever ``--n_core``;
  more cores, more points).

The budgets never undercut the minimums: one timed call of one filled SIMD
chunk per backend for the runtime, ``STABILITY_MINIMUM`` points per backend
for the stability run. When the minimums alone exceed a requested
completion time the plan says so (``Plan.feasible``) and estimates the time
they take instead."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# The user's minimums: one timed call (one filled SIMD chunk) per backend
# for the runtime, five points per backend for the stability run.
RUNTIME_MINIMUM_CALLS = 1
STABILITY_MINIMUM = 5
# The default budgets when neither --completion-time nor counts are given.
DEFAULT_RUNTIME_SECONDS = 10.0
DEFAULT_STABILITY_SECONDS_PER_CORE = 60.0
# The calls a stable median is planned on when the budget affords them.
STABLE_CALLS = 5
# The reference pass's share of a backend's runtime budget.
REFERENCE_SHARE = 0.25
# A stability point's cost relative to one optimistic binary64 pass before
# the stability run measured it: the checked call with the default
# rotation test evaluates the point twice, plus the checks' work.
STABILITY_PRIOR = 2.5


@dataclass(frozen=True, slots=True)
class Costs:
    """One backend's measured (or prior) costs: seconds per point of one
    optimistic binary64 pass (``binary64``), of the reference (``reference``,
    None without a ground truth), of a stability point in binary64 with its
    checks and in the reference (``stability``, ``stability_reference``),
    and the fixed seconds of starting a process that loads the artifact
    (``start``)."""

    binary64: float
    reference: float | None
    stability: float
    stability_reference: float | None
    start: float


@dataclass(frozen=True, slots=True)
class RuntimeCounts:
    """One backend's runtime timing: ``batch`` points per call (whole
    chunks), ``warmup`` untimed calls, at least ``minimum`` timed calls and
    more while they fit ``budget`` seconds; the reference pass's
    ``reference_budget`` seconds (its chunk shrinks to what that affords,
    at least one point); ``expected`` seconds of the whole backend
    (start-up included)."""

    batch: int
    warmup: int
    minimum: int
    budget: float
    reference_budget: float
    expected: float


@dataclass(frozen=True, slots=True)
class Plan:
    """The counts of a profile (module docstring): the ``rule`` that set
    them ("completion time", "counts", "time budgets"), per backend its
    ``RuntimeCounts`` (empty without the runtime timing), the stability
    run's ``stability_points`` (None: not asked for) and its expected wall
    seconds, the ``expected_seconds`` of everything planned, the requested
    ``budget_seconds`` (None without --completion-time), whether the budget
    can be met (``feasible``) and what the minimums alone take
    (``minimum_seconds``)."""

    rule: str
    runtime: Mapping[str, RuntimeCounts] = field(default_factory=dict)
    stability_points: int | None = None
    stability_seconds: float = 0.0
    expected_seconds: float = 0.0
    budget_seconds: float | None = None
    feasible: bool = True
    minimum_seconds: float = 0.0

    def record(self) -> dict[str, object]:
        """The JSON record of the plan."""

        return {
            "rule": self.rule,
            "runtime": {
                name: {
                    "batch": counts.batch,
                    "warmup": counts.warmup,
                    "minimum": counts.minimum,
                    "budget_seconds": counts.budget,
                    "reference_budget_seconds": counts.reference_budget,
                    "expected_seconds": counts.expected,
                }
                for name, counts in self.runtime.items()
            },
            "stability_points": self.stability_points,
            "stability_seconds": self.stability_seconds,
            "expected_seconds": self.expected_seconds,
            "budget_seconds": self.budget_seconds,
            "feasible": self.feasible,
            "minimum_seconds": self.minimum_seconds,
        }


def runtime_counts(
    costs: Costs, budget: float, *, lanes: int, batch_max: int
) -> RuntimeCounts:
    """A backend's runtime counts for ``budget`` seconds (its start-up
    excluded): the reference pass's share first (``REFERENCE_SHARE``, at
    least one point and its one-point warm-up), then the largest batch of
    whole chunks (at most ``batch_max``) whose ``STABLE_CALLS`` calls and one
    warm-up fit what is left, at least one chunk, timed while the calls fit
    the rest, at least ``RUNTIME_MINIMUM_CALLS``."""

    lanes = max(lanes, 1)
    chunks_max = max(batch_max // lanes, 1)
    reference = 0.0
    if costs.reference is not None:
        reference = max(REFERENCE_SHARE * budget, costs.reference)
    left = max(budget - reference - (costs.reference or 0.0), 0.0)
    per_chunk = lanes * costs.binary64
    chunks = int(left // ((STABLE_CALLS + 1) * per_chunk)) if per_chunk > 0 else chunks_max
    chunks = min(max(chunks, 1), chunks_max)
    batch = chunks * lanes
    call = batch * costs.binary64
    timed = max(left - call, RUNTIME_MINIMUM_CALLS * call)
    expected = costs.start + call + timed
    if costs.reference is not None:
        expected += costs.reference + reference
    return RuntimeCounts(batch, 1, RUNTIME_MINIMUM_CALLS, timed, reference, expected)


def runtime_minimum(costs: Costs, *, lanes: int) -> float:
    """The seconds of a backend's minimal runtime timing: its start-up, one
    warm-up and one timed call of one chunk, and the reference's one-point
    warm-up and one timed point."""

    seconds = costs.start + 2 * lanes * costs.binary64
    if costs.reference is not None:
        seconds += 2 * costs.reference
    return seconds


def stability_work(costs: Sequence[Costs]) -> float:
    """The worker seconds of one stability point on every backend: binary64
    with its checks and the reference."""

    return sum(c.stability + (c.stability_reference or 0.0) for c in costs)


def stability_wall(costs: Sequence[Costs], points: int, workers: int) -> float:
    """The wall seconds of ``points`` stability points on ``workers``
    workers: their work shared by the workers, no less than the dearest
    single task of ``points`` points (the tasks of one backend and
    precision of a chunk run on one worker)."""

    if not costs or points <= 0:
        return 0.0
    work = points * stability_work(costs)
    single = max(
        points * max(c.stability, c.stability_reference or 0.0) for c in costs
    )
    return max(work / max(workers, 1), min(single, work))


def stability_points(
    costs: Sequence[Costs], budget: float, workers: int, *, lanes: int
) -> int:
    """The stability points ``budget`` wall seconds afford on ``workers``
    workers, whole chunks of ``lanes`` from one chunk on, at least
    ``STABILITY_MINIMUM``."""

    work = stability_work(costs)
    if work <= 0.0:
        return STABILITY_MINIMUM
    points = int(budget * max(workers, 1) // work)
    if points >= lanes:
        points -= points % lanes
    return max(points, STABILITY_MINIMUM)


def plan(
    costs: Mapping[str, Costs],
    *,
    runtime: bool,
    stability: bool,
    workers: int,
    lanes: int,
    batch_max: int,
    completion: float | None = None,
    elapsed: float = 0.0,
    stability_start: float = 0.0,
    runtime_counts_given: bool = False,
    stability_points_given: int | None = None,
) -> Plan:
    """The counts of a profile (module docstring) from the backends'
    ``costs``: the phases asked for (``runtime``, ``stability``), the
    stability ``workers``, the SIMD ``lanes`` and the largest batch
    (``batch_max``); ``completion``: the requested wall seconds of the whole
    command, of which ``elapsed`` are spent; ``stability_start``: the wall
    seconds of starting the stability workers (loading, warm-up);
    ``runtime_counts_given`` / ``stability_points_given``: the counts given
    (honoured without ``completion``)."""

    backends = list(costs)
    values = [costs[name] for name in backends]
    minimum_runtime = (
        sum(runtime_minimum(c, lanes=lanes) for c in values) if runtime else 0.0
    )
    minimum_stability = (
        stability_start + stability_wall(values, STABILITY_MINIMUM, workers)
        if stability
        else 0.0
    )
    minimum = minimum_runtime + minimum_stability
    if completion is not None:
        left = max(completion - elapsed, 0.0)
        if left < minimum:
            counts = (
                {
                    name: runtime_counts(c, 0.0, lanes=lanes, batch_max=lanes)
                    for name, c in costs.items()
                }
                if runtime
                else {}
            )
            return Plan(
                "completion time",
                counts,
                STABILITY_MINIMUM if stability else None,
                minimum_stability,
                minimum,
                completion,
                False,
                minimum,
            )
        free = left - (stability_start if stability else 0.0)
        runtime_budget = 0.0
        counts = {}
        if runtime:
            stable = sum(
                c.start
                + (STABLE_CALLS + 1) * batch_max * c.binary64
                + (0.0 if c.reference is None else (lanes + 1) * c.reference)
                for c in values
            )
            runtime_budget = free if not stability else min(stable, free / 2.0)
            runtime_budget = max(runtime_budget, minimum_runtime)
            counts = _split_runtime(costs, runtime_budget, lanes, batch_max)
            runtime_budget = sum(c.expected for c in counts.values())
        points = None
        wall = 0.0
        if stability:
            budget = max(free - runtime_budget, 0.0)
            points = stability_points(values, budget, workers, lanes=lanes)
            wall = stability_start + stability_wall(values, points, workers)
        return Plan(
            "completion time",
            counts,
            points,
            wall,
            runtime_budget + wall,
            completion,
            True,
            minimum,
        )
    rule = (
        "counts"
        if (runtime and runtime_counts_given)
        or (stability and stability_points_given is not None)
        else "time budgets"
    )
    counts = {}
    runtime_seconds = 0.0
    if runtime and not runtime_counts_given:
        counts = _split_runtime(costs, DEFAULT_RUNTIME_SECONDS, lanes, batch_max)
        runtime_seconds = sum(c.expected for c in counts.values())
    points = None
    wall = 0.0
    if stability:
        points = (
            stability_points_given
            if stability_points_given is not None
            else stability_points(
                values,
                DEFAULT_STABILITY_SECONDS_PER_CORE,
                workers,
                lanes=lanes,
            )
        )
        wall = stability_start + stability_wall(values, points, workers)
    return Plan(rule, counts, points, wall, runtime_seconds + wall, None, True, minimum)


def _split_runtime(
    costs: Mapping[str, Costs], budget: float, lanes: int, batch_max: int
) -> dict[str, RuntimeCounts]:
    """The runtime budget shared equally by the backends, each its start-up
    excluded from its share."""

    share = budget / max(len(costs), 1)
    return {
        name: runtime_counts(
            c, max(share - c.start, 0.0), lanes=lanes, batch_max=batch_max
        )
        for name, c in costs.items()
    }


def admit_chunk(
    *,
    now: float,
    deadline: float | None,
    queued_seconds: float,
    chunk_seconds: float,
    workers: int,
    points: int,
    minimum: int = STABILITY_MINIMUM,
) -> bool:
    """Whether the stability run submits its next chunk: always until it
    holds ``minimum`` points, else while the work queued (``queued_seconds``
    of worker time, the running tasks' remainder included) and the chunk's
    (``chunk_seconds``) end, shared by the ``workers``, before ``deadline``
    (None: no deadline, every planned chunk)."""

    if points < minimum or deadline is None:
        return True
    finish = now + (queued_seconds + chunk_seconds) / max(workers, 1)
    return finish <= deadline + 0.5 * chunk_seconds / max(workers, 1)


def duration_text(seconds: float) -> str:
    """A duration the way the plan line shows it: "45s", "3m10s", "1h05m"."""

    seconds = max(seconds, 0.0)
    if seconds < 60.0:
        return f"{math.ceil(seconds)}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


__all__ = [
    "DEFAULT_RUNTIME_SECONDS",
    "DEFAULT_STABILITY_SECONDS_PER_CORE",
    "RUNTIME_MINIMUM_CALLS",
    "STABILITY_MINIMUM",
    "Costs",
    "Plan",
    "RuntimeCounts",
    "admit_chunk",
    "plan",
    "runtime_counts",
    "stability_points",
    "stability_wall",
]
