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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

# The user's minimums: one timed call (one filled SIMD chunk) per backend
# for the runtime, five points per backend for the stability run.
RUNTIME_MINIMUM_CALLS = 1
STABILITY_MINIMUM = 5
# The default budgets when neither --completion-time nor counts are given.
DEFAULT_RUNTIME_SECONDS = 10.0
DEFAULT_STABILITY_SECONDS_PER_CORE = 60.0
# A budget counts as met within this factor (the acceptance of
# --completion-time, and when the default budgets are said to be exceeded).
BUDGET_TOLERANCE = 1.5
# The serial seconds per record after the stability run: the statistics of
# one point of one backend, its record (--json, --samples-output, the write
# included) and one point's inputs (uux_ddx, 100 000 points on 2 backends:
# 12 s after the run, 120 us per point).
POST_SECONDS_PER_RECORD = 30e-6
RECORD_SECONDS_PER_RECORD = 20e-6
RECORD_SECONDS_PER_INPUT = 20e-6
# The share of the memory available when a stage starts its worker
# processes that their resident sets may take (``memory_workers``): the rest
# for the host's other processes and the per-worker estimate's error.
MEMORY_SHARE = 0.75
# The calls a stable median is planned on when the budget affords them.
STABLE_CALLS = 5
# With --runtime-threads N > 1: the share of a backend's binary64 budget for
# the one-thread timing of the same calls (the gain against one thread).
ONE_THREAD_SHARE = 0.25
# The reference pass's share of a backend's runtime budget.
REFERENCE_SHARE = 0.25
# A stability point's cost relative to one optimistic binary64 pass before
# the stability run measured it: the checked call with the default
# rotation test evaluates the point twice, plus the checks' work.
STABILITY_PRIOR = 2.5
# A stability task (one backend and precision) holds about this many
# seconds of work: a point that costs more is a task of its own, so that an
# expensive process's few points spread over the workers (a one-point call
# costs one to three times a full chunk's per-point price, a fixed cost of
# 0.03-0.3 s per call), a cheap one keeps whole SIMD chunks.
TASK_SECONDS = 4.0
# The binary64 warm-up is one point when a full call costs more than this
# (its first-call costs are then negligible), else one full call.
WARMUP_CALL_SECONDS = 2.0
# The measurement repeats its one-point call when it took at most this: a
# first call carries seconds of lazy set-up (2 s of NinjaRS's first 0.6 s
# point on u u~ > z g g g g), negligible against a dearer point.
MEASURE_REPEAT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class Costs:
    """One backend's measured (or prior) costs: seconds per point of one
    optimistic binary64 pass (``binary64``), of the reference (``reference``,
    None without a ground truth), of a stability point in binary64 with its
    checks and in the reference (``stability``, ``stability_reference``),
    and the fixed seconds of starting a process that loads the artifact
    (``start``). ``reference_estimated``: the reference cost is a prior, not
    measured (``reference_affects_counts`` False: the runtime timing's own
    warm-up measures it); it sets no budget, only the expected seconds."""

    binary64: float
    reference: float | None
    stability: float
    stability_reference: float | None
    start: float
    reference_estimated: bool = False
    # The seconds of the warm entry when a stream warms through it instead
    # of its untimed warm-up point (``takes_entry``): binary64's and the
    # reference's; None: the warm-up point (``binary64_warmup``,
    # ``reference_warmup``).
    binary64_warm: float | None = None
    reference_warm: float | None = None


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
    can be met (``feasible``: --completion-time, or the default budgets
    within ``BUDGET_TOLERANCE``) and the whole command's seconds with the
    minimums alone (``minimum_seconds``: the time spent so far included)."""

    rule: str
    runtime: Mapping[str, RuntimeCounts] = field(default_factory=dict)
    stability_points: int | None = None
    stability_seconds: float = 0.0  # its wall, the tail after it included
    expected_seconds: float = 0.0
    budget_seconds: float | None = None
    feasible: bool = True
    minimum_seconds: float = 0.0
    # The phases whose budget in force their minimums exceed ("runtime",
    # "stability"; ``feasible`` is False when there is one).
    unmet: tuple[str, ...] = ()

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
            "unmet": list(self.unmet),
        }


def warmup_seconds(call: float, per_point: float) -> float:
    """The binary64 warm-up's seconds: one full call of ``call`` seconds,
    one point (``per_point``) when the full call costs more than
    ``WARMUP_CALL_SECONDS``."""

    return per_point if call > WARMUP_CALL_SECONDS else call


def binary64_warmup(costs: Costs, call: float) -> float:
    """The seconds that warm a backend's binary64 stream before its timed
    calls of ``call`` seconds: its warm entry's (``Costs.binary64_warm``),
    else the untimed warm-up (``warmup_seconds``)."""

    if costs.binary64_warm is not None:
        return costs.binary64_warm
    return warmup_seconds(call, costs.binary64)


def reference_warmup(costs: Costs) -> float:
    """The seconds that warm a backend's reference stream: its warm entry's
    (``Costs.reference_warm``), else its untimed one-point call (the
    reference's cost per point; 0 without a reference)."""

    if costs.reference_warm is not None:
        return costs.reference_warm
    return costs.reference or 0.0


def takes_entry(
    point: float | None, *, entry: float | None = None, excess: float | None = None
) -> bool:
    """Whether a stream whose runtime offers the warm entry warms through it
    instead of its untimed warm-up ``point`` (its seconds; None: unknown):
    when the entry's remembered cost (``entry``, a previous profile's) is
    known, when the point costs more; else when the measured first-call
    excess of the stream (``excess``: its first one-point call over its
    second, what a first call builds lazily, about what the entry builds) is
    known, when the point costs more than it; else (no cost known: a dear
    point not measured twice, or nothing measured) the entry: it costs
    seconds, a dear point minutes."""

    if point is None:
        return True
    if entry is not None:
        return point > entry
    if excess is not None:
        return point > excess
    return True


def memory_workers(
    per_worker: float, available: float, *, share: float = MEMORY_SHARE
) -> int:
    """The worker processes that fit in ``share`` of the ``available``
    memory (bytes) at ``per_worker`` bytes each (a worker's resident size
    loaded and warm, as measured), at least one."""

    if per_worker <= 0.0:
        return 1 << 30
    return max(1, int(share * max(available, 0.0) // per_worker))


def task_points(point_seconds: float, points: int) -> int:
    """The points of one stability task (one backend and precision) out of
    a chunk of ``points``: about ``TASK_SECONDS`` of work at
    ``point_seconds`` each, at least one, at most the chunk (0 or unknown
    costs: the chunk)."""

    if points <= 0:
        return 0
    if point_seconds <= 0.0:
        return points
    return max(1, min(points, math.ceil(TASK_SECONDS / point_seconds)))


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
    measured = 0.0 if costs.reference_estimated else (costs.reference or 0.0)
    # The reference stream's warming (its untimed point, or its entry).
    warming = 0.0 if costs.reference_estimated else reference_warmup(costs)
    if costs.reference is not None:
        # An estimated cost sets no budget: the reference pass's own warm-up
        # fits its chunk to the share (at least one point).
        reference = max(REFERENCE_SHARE * budget, measured)
    left = max(budget - reference - warming, 0.0)
    per_chunk = lanes * costs.binary64
    chunks = (
        int(left // ((STABLE_CALLS + 1) * per_chunk)) if per_chunk > 0 else chunks_max
    )
    chunks = min(max(chunks, 1), chunks_max)
    batch = chunks * lanes
    call = batch * costs.binary64
    warm = binary64_warmup(costs, call)
    # The timed calls' budget: what is left (none: the minimum calls only,
    # however fast they turn out); the expected seconds at least the minimum.
    timed = max(left - warm, 0.0)
    expected = costs.start + warm + max(timed, RUNTIME_MINIMUM_CALLS * call)
    if costs.reference is not None:
        expected += reference_warmup(costs) + max(reference, costs.reference)
    return RuntimeCounts(batch, 1, RUNTIME_MINIMUM_CALLS, timed, reference, expected)


def measurement_affects_counts(start: float, *, backends: int) -> bool:
    """Whether any measured cost can change the runtime counts under the
    default budget (``DEFAULT_RUNTIME_SECONDS`` shared by ``backends``,
    each backend's ``start`` excluded, ``_split_runtime``): False when the
    start-up alone takes a backend's share, so that every backend gets the
    minimum counts (``minimum_counts``) whatever its costs. Then the
    measurement need not run: each runtime timing's warm-ups measure its
    costs."""

    return DEFAULT_RUNTIME_SECONDS / max(backends, 1) - start > 0.0


def minimum_counts(
    start: float, *, lanes: int, reference: bool, costs: Costs | None = None
) -> RuntimeCounts:
    """The runtime counts of a backend without a budget: one chunk of
    ``lanes``, one warm-up, ``RUNTIME_MINIMUM_CALLS`` timed calls, and with
    a ``reference`` one reference point after its warm-up (no budget);
    ``costs`` (priors) only set the expected seconds (without them: the
    ``start`` only)."""

    prior = costs or Costs(0.0, 0.0 if reference else None, 0.0, None, start)
    prior = replace(
        prior,
        start=start,
        reference=(prior.reference or 0.0) if reference else None,
        reference_estimated=True,
    )
    return runtime_counts(prior, 0.0, lanes=lanes, batch_max=lanes)


def reference_affects_counts(
    binary64: float,
    start: float,
    *,
    backends: int,
    lanes: int,
    warm: float | None = None,
) -> bool:
    """Whether a backend's reference cost can change its runtime counts
    under the default budget (``DEFAULT_RUNTIME_SECONDS`` shared by
    ``backends``, its ``start`` excluded, ``_split_runtime``) at ``binary64``
    seconds per point: False when what the share leaves after the
    reference's ``REFERENCE_SHARE`` (``runtime_counts`` takes it first, at
    least that much whatever the reference costs) affords no more than the
    binary64 warm-up (``warmup_seconds`` of one chunk of ``lanes``): then
    ``runtime_counts`` gives one chunk, one warm-up, the minimum timed calls
    and no timed budget left whatever the reference costs, and with a
    larger share a cheap reference would leave a timed budget a dear one
    would not (the exact condition). Then the measurement need not time the
    reference: the runtime timing's reference warm-up measures it.
    ``warm``: the binary64 stream's warming when it is not the warm-up
    (the warm entry's seconds, ``binary64_warmup``)."""

    budget = max(DEFAULT_RUNTIME_SECONDS / max(backends, 1) - start, 0.0)
    left = (1.0 - REFERENCE_SHARE) * budget
    if warm is None:
        warm = warmup_seconds(max(lanes, 1) * binary64, binary64)
    return left > warm


def scope_minimum(
    costs: Costs, *, lanes: int, reference: bool, start: bool = False
) -> float:
    """A backend's runtime minimum for the scope rules (``timing_scope``):
    its binary64 warm-up (``warmup_seconds``) and one timed call of one
    chunk of ``lanes``, with ``reference`` the reference's untimed and timed
    point, with ``start`` its start-up (``runtime_minimum`` with these
    parts)."""

    call = lanes * costs.binary64
    seconds = call + binary64_warmup(costs, call)
    if reference and costs.reference is not None:
        seconds += reference_warmup(costs) + costs.reference
    return seconds + (costs.start if start else 0.0)


def timing_scope(
    costs: Mapping[str, Costs],
    default: str | None,
    bound: Callable[[int], float],
    *,
    lanes: int,
    reference: bool,
    force_reference: bool,
    asked_backends: bool,
    runtime: bool = True,
    reference_rule: bool = True,
    start: bool = False,
    extra: Mapping[str, float] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """What a profile whose minimums exceed its budget runs (USER
    2026-11-19), from the backends' ``costs`` (those that can serve) and
    ``bound(n)``, the seconds the minimums of n backends may take before the
    budget counts as exceeded. First the reference of the runtime timing
    (rule B, ``reference_rule``: the runtime timing alone): none when the
    runtime minimums of every backend with their reference points
    (``scope_minimum``) exceed ``bound`` of them, unless ``force_reference``.
    Then the backends (rule A): the ``default`` alone when its minimum for
    everything asked (its runtime minimum when ``runtime``, with the
    reference as just decided, and ``extra[default]``: the stability run's
    minimum) exceeds ``bound(1)``, unless ``asked_backends``. The reference's
    decision (True: timed) and the backends left out."""

    names = list(costs)
    extra = extra or {}
    timed = reference
    if (
        reference_rule
        and runtime
        and timed
        and not force_reference
        and names
        and sum(
            scope_minimum(costs[b], lanes=lanes, reference=True, start=start)
            for b in names
        )
        > bound(len(names))
    ):
        timed = False
    omitted: tuple[str, ...] = ()
    if not asked_backends and default in costs and len(names) > 1:
        alone = extra.get(default, 0.0) + (
            scope_minimum(costs[default], lanes=lanes, reference=timed, start=start)
            if runtime
            else 0.0
        )
        if alone > bound(1):
            omitted = tuple(b for b in names if b != default)
    return timed, omitted


def runtime_minimum(costs: Costs, *, lanes: int) -> float:
    """The seconds of a backend's minimal runtime timing: its start-up, its
    binary64 warming (``binary64_warmup``: one warm-up, or the warm entry)
    and one timed call of one chunk, and the reference's warming
    (``reference_warmup``) and one timed point."""

    call = lanes * costs.binary64
    seconds = costs.start + call + binary64_warmup(costs, call)
    if costs.reference is not None:
        seconds += reference_warmup(costs) + costs.reference
    return seconds


def stability_work(costs: Sequence[Costs]) -> float:
    """The worker seconds of one stability point on every backend: binary64
    with its checks and the reference."""

    return sum(c.stability + (c.stability_reference or 0.0) for c in costs)


def stability_wall(
    costs: Sequence[Costs], points: int, workers: int, *, lanes: int = 16
) -> float:
    """The wall seconds of ``points`` stability points on ``workers``
    workers: their work shared by the workers, no less than the dearest
    single task (``task_points`` of a chunk of up to ``lanes`` points of
    one backend and precision run on one worker)."""

    if not costs or points <= 0:
        return 0.0
    work = points * stability_work(costs)
    chunk = min(points, lanes)
    single = max(
        task_points(seconds, chunk) * seconds
        for c in costs
        for seconds in (c.stability, c.stability_reference or 0.0)
    )
    return max(work / max(workers, 1), min(single, work))


def stability_points(
    costs: Sequence[Costs],
    budget: float,
    workers: int,
    *,
    lanes: int,
    tail: float = 0.0,
) -> int:
    """The stability points ``budget`` wall seconds afford on ``workers``
    workers (each point also ``tail`` serial seconds after the run: its
    statistics and records), whole chunks of ``lanes`` from one chunk on, at
    least ``STABILITY_MINIMUM``."""

    work = stability_work(costs)
    if work <= 0.0 and tail <= 0.0:
        return STABILITY_MINIMUM
    points = int(max(budget, 0.0) // (work / max(workers, 1) + tail))
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
    tail_seconds_per_point: float = 0.0,
    runtime_target: float = DEFAULT_RUNTIME_SECONDS,
) -> Plan:
    """The counts of a profile (module docstring) from the backends'
    ``costs``: the phases asked for (``runtime``, ``stability``), the
    stability ``workers``, the SIMD ``lanes`` and the largest batch
    (``batch_max``); ``completion``: the requested wall seconds of the whole
    command, of which ``elapsed`` are spent; ``stability_start``: the wall
    seconds of starting the stability workers (loading, warm-up);
    ``runtime_counts_given`` / ``stability_points_given``: the counts given
    (honoured without ``completion``); ``tail_seconds_per_point``: the
    serial seconds per stability point after the run (statistics, records),
    part of the completion time; ``runtime_target``: the runtime counts'
    time budget per backend when they are given (``--target-runtime``)."""

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
                elapsed + minimum,
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
            # The headline number first: a stable median (at least the
            # default runtime budget) before the stability run, at most half.
            stable = max(stable, DEFAULT_RUNTIME_SECONDS + sum(c.start for c in values))
            runtime_budget = free if not stability else min(stable, free / 2.0)
            runtime_budget = max(runtime_budget, minimum_runtime)
            counts = _split_runtime(costs, runtime_budget, lanes, batch_max)
            runtime_budget = sum(c.expected for c in counts.values())
        points = None
        wall = 0.0
        if stability:
            budget = max(free - runtime_budget, 0.0)
            points = stability_points(
                values, budget, workers, lanes=lanes, tail=tail_seconds_per_point
            )
            wall = (
                stability_start
                + stability_wall(values, points, workers)
                + points * tail_seconds_per_point
            )
        return Plan(
            "completion time",
            counts,
            points,
            wall,
            runtime_budget + wall,
            completion,
            True,
            elapsed + minimum,
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
    # A budget in force (the default ones, or --target-runtime with the
    # runtime counts given) cannot be met when its phase's minimums alone
    # (start-ups aside: outside the budgets by nature) exceed it by the
    # tolerance, whatever else was given. The stability points given set no
    # budget.
    unmet = []
    if runtime:
        budget = (
            DEFAULT_RUNTIME_SECONDS
            if not runtime_counts_given
            else runtime_target * len(values)
        )
        if minimum_runtime - sum(c.start for c in values) > BUDGET_TOLERANCE * budget:
            unmet.append("runtime")
    if (
        stability
        and stability_points_given is None
        and (
            minimum_stability - stability_start
            > BUDGET_TOLERANCE * DEFAULT_STABILITY_SECONDS_PER_CORE
        )
    ):
        unmet.append("stability")
    return Plan(
        rule,
        counts,
        points,
        wall,
        runtime_seconds + wall,
        None,
        not unmet,
        elapsed + minimum,
        tuple(unmet),
    )


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


def affordable_chunk(
    *,
    now: float,
    deadline: float,
    queued_seconds: float,
    point_seconds: float,
    workers: int,
    points: int,
    lanes: int,
    minimum: int = STABILITY_MINIMUM,
    stream_seconds: float = 0.0,
    tail_seconds: float = 0.0,
) -> int:
    """The points of the stability run's next chunk under a deadline: the
    minimum's remainder until it holds ``minimum`` points, then as many as
    fit, at most one whole SIMD chunk of ``lanes``: their work shared by the
    ``workers`` after the queued work (``point_seconds``: one point on every
    backend and precision), and their dearest single task (``stream_seconds``
    per point: one backend and precision, ``task_points`` of the chunk run
    by one worker) started when a worker frees up, both ending by the
    deadline less the serial ``tail_seconds`` per point after the run (its
    statistics and records, for the points so far and the chunk's); 0 when
    none fits."""

    if points < minimum:
        return minimum - points
    workers = max(workers, 1)
    deadline -= points * tail_seconds
    start = now + queued_seconds / workers
    room = (deadline - now) * workers - queued_seconds
    per_point = point_seconds + workers * tail_seconds
    fits = lanes if per_point <= 0.0 else int(room // per_point)
    if stream_seconds > 0.0:
        window = max(deadline - start, 0.0)
        if task_points(stream_seconds, lanes) * stream_seconds > window:
            fits = min(fits, int(window // stream_seconds))
    return max(min(fits, lanes), 0)


def duration_text(seconds: float) -> str:
    """A duration the way the plan line shows it: "45s", "3m10s", "1h05m"."""

    seconds = max(seconds, 0.0)
    if seconds < 60.0:
        return f"{math.ceil(seconds)}s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


__all__ = [
    "DEFAULT_RUNTIME_SECONDS",
    "DEFAULT_STABILITY_SECONDS_PER_CORE",
    "MEASURE_REPEAT_SECONDS",
    "MEMORY_SHARE",
    "ONE_THREAD_SHARE",
    "POST_SECONDS_PER_RECORD",
    "RECORD_SECONDS_PER_INPUT",
    "RECORD_SECONDS_PER_RECORD",
    "RUNTIME_MINIMUM_CALLS",
    "STABILITY_MINIMUM",
    "TASK_SECONDS",
    "WARMUP_CALL_SECONDS",
    "Costs",
    "Plan",
    "RuntimeCounts",
    "admit_chunk",
    "affordable_chunk",
    "binary64_warmup",
    "measurement_affects_counts",
    "memory_workers",
    "minimum_counts",
    "plan",
    "reference_affects_counts",
    "reference_warmup",
    "runtime_counts",
    "scope_minimum",
    "stability_points",
    "stability_wall",
    "takes_entry",
    "task_points",
    "timing_scope",
    "warmup_seconds",
]
