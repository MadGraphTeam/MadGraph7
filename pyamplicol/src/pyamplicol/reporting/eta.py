# SPDX-License-Identifier: 0BSD
"""Time-to-completion estimates for counted progress.

``count_estimate`` turns the completion marks of a counted task (when each
item finished) into the remaining time, for a live display that redraws
between updates:

* before any item is back, the task's prior (its expected duration, e.g.
  from the previous generation of the same process), marked as an estimate;
* then the measured time per item: the mean over the items since the first
  one finished (a slow first item, a warm-up, does not set the pace),
  averaged with the median over the most recent items (load drift, robust
  to one outlier), and
  blended with the prior's per-item time while few items are measured;
* between updates the estimate counts down from the last mark; when the
  item in progress takes longer than an item usually does, the estimate
  holds at the time the other remaining items need (it never reaches 0
  while work remains), and on the last item it turns into "overdue".
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

Mark = tuple[float, int]
"""A completion mark: (monotonic time, items completed at that time)."""

# The recent window: the last 30 % of the items, at least five.
_RECENT_FRACTION = 0.3
_RECENT_MINIMUM = 5
# Measured items needed before the measured pace outweighs a prior.
_PRIOR_WEIGHT = 2.0


@dataclass(frozen=True, slots=True)
class Estimate:
    """The remaining time of a task.

    ``seconds``: the best estimate of the time to completion (None when the
    expected end has passed and nothing else remains: see ``overdue``).
    ``estimated``: a prior rather than a measured pace ("~" on a display).
    ``overdue``: the seconds by which the expected end has passed (0 when
    it has not). ``at_least``: the item in progress takes longer than an
    item does, ``seconds`` is what the other remaining items need (">").
    """

    seconds: float | None
    estimated: bool
    overdue: float = 0.0
    at_least: bool = False


def _seconds_per_item(marks: Sequence[Mark], started_at: float) -> float | None:
    """The pace: the mean time per item since the first item finished (the
    whole run's pace without its warm-up) averaged with the median time per
    item of the recent items (load drift, robust to one outlier)."""

    if not marks:
        return None
    last_time, last_count = marks[-1]
    first_time, first_count = marks[0]
    if len(marks) >= 3 and last_count > first_count:
        overall = (last_time - first_time) / (last_count - first_count)
    else:
        overall = (last_time - started_at) / max(last_count, 1)
    steps = [
        (later[0] - earlier[0]) / (later[1] - earlier[1])
        for earlier, later in itertools.pairwise(marks)
        if later[1] > earlier[1]
    ]
    size = max(_RECENT_MINIMUM, int(len(steps) * _RECENT_FRACTION))
    window = sorted(steps[-size:])
    if len(window) >= _RECENT_MINIMUM:
        middle = len(window) // 2
        recent = (
            window[middle]
            if len(window) % 2
            else 0.5 * (window[middle - 1] + window[middle])
        )
    else:
        recent = overall
    return max(0.5 * (overall + recent), 0.0)


def count_estimate(
    *,
    total: int | None,
    completed: int,
    started_at: float,
    marks: Sequence[Mark],
    now: float,
    prior_seconds: float | None = None,
) -> Estimate | None:
    """The remaining time of a task of ``total`` items (None without a
    total, or when nothing is known yet and there is no prior)."""

    if total is None or total <= 0:
        return None
    remaining = total - min(completed, total)
    if remaining <= 0:
        return None
    if completed <= 0 or not marks:
        if prior_seconds is None:
            return None
        left = prior_seconds - max(now - started_at, 0.0)
        if left >= 1.0:
            return Estimate(left, estimated=True)
        return Estimate(None, estimated=True, overdue=max(-left, 0.0))
    per_item = _seconds_per_item(marks, started_at)
    assert per_item is not None
    measured = min(completed, total)
    if prior_seconds is not None and prior_seconds > 0.0:
        weight = measured / (measured + _PRIOR_WEIGHT)
        per_item = weight * per_item + (1.0 - weight) * prior_seconds / total
    estimated = measured < 2 and prior_seconds is None
    since = max(now - marks[-1][0], 0.0)
    counted_down = remaining * per_item - since
    floor = (remaining - 1) * per_item
    if counted_down >= max(floor, 1.0):
        return Estimate(counted_down, estimated=estimated)
    if floor >= 1.0:
        return Estimate(floor, estimated=estimated, at_least=True)
    # The last item, past the time an item takes.
    return Estimate(None, estimated=estimated, overdue=max(since - per_item, 0.0))


def remaining_estimate(
    *,
    prior_seconds: float | None,
    started_at: float,
    now: float,
) -> Estimate | None:
    """The remaining time of an uncounted task from its prior alone."""

    if prior_seconds is None:
        return None
    left = prior_seconds - max(now - started_at, 0.0)
    if left >= 1.0:
        return Estimate(left, estimated=True)
    return Estimate(None, estimated=True, overdue=max(-left, 0.0))


def combined(
    active: Estimate | None,
    later_seconds: float,
    *,
    later_known: bool,
) -> Estimate | None:
    """A whole command's estimate: the active phase's remaining time plus
    the priors of the phases still to come (``later_known`` False when a
    phase to come has none: the estimate is then a lower bound, shown
    as an estimate)."""

    if active is None and later_seconds <= 0.0:
        return None
    seconds = later_seconds + (active.seconds or 0.0 if active is not None else 0.0)
    if seconds < 1.0:
        return active
    return Estimate(
        seconds,
        estimated=(active is None or active.estimated) or not later_known,
    )


__all__ = ["Estimate", "Mark", "combined", "count_estimate", "remaining_estimate"]
