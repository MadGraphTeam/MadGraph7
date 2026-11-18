# SPDX-License-Identifier: 0BSD
"""The built-in stability rescue of a v2 virtual ``evaluate`` (user ruling
2026-10-31 Q9; M-FINAL MF-RESCUE).

A :class:`~pyamplicol.api.virtual_results.VirtualRescuePolicy` is an ordered
ladder of levels (carrier, stability checks, target), modelled on gammaloop's
``StabilitySettings``. ``rescue`` runs it on one batch through a level
evaluator, the runtime's own single-carrier evaluation of a subset of the
batch (``V2VirtualRuntime`` and ``V2ObservablesRuntime``):

1. level 0 evaluates the whole batch;
2. a point that failed (``FailedPoints`` reasons, failures of its rotated
   evaluations included) or whose estimate missed the level's target is
   evaluated again at the next level, from its original exact inputs (no
   value is promoted, MASTER_PLAN §2.18); an accepted point is never
   evaluated again; a point whose estimate is no smaller than its best
   estimate at fewer precision bits (same metric) is not escalated either
   (the stopping rule, ``_stalled``: more digits did not improve it, its
   error is not the carrier's rounding) and keeps that level's value with
   the target not met;
3. at the last level a point that evaluates keeps its value with the target
   marked as not met; a point that failed at every level is a failed lane,
   or, under ``on_failure="best-effort"``, a best-effort lane when the last
   level's unchecked reduction gives it a value (only the last level runs
   the best-effort evaluation, ``reduction.with_fallback``).

A level compares the public estimate's field named by its ``metric`` with
its target. An integrated zero (a Landau-Yang |A|^2 of a loop-induced
process, m-lit PLAN §2.a.8; numerical, never declared, RULED Q13) can never
meet a "relative" or "finite" target, its value being itself rounding: its
relative estimate grows with the precision (|A|^2 falls with the unit
roundoff squared, E with the unit roundoff), so the stopping rule keeps it
at the first level of higher precision whose estimate is no better, with
the target not met (g g > z on the loop-induced default stack: the
double-double level; before the rule it climbed to the last level). The
opt-in "scale" metric (``VirtualRescueLevel.metric``: E <= target x S,
the estimate against the pre-cancellation scale S of its floor,
``VirtualStabilityEstimate.scale``) accepts it at the first level, where
its E / S is the floor's FLOOR_UNITS x u (M-LIT LIT-STAB-N, ``tests/unit/
test_loop_v2_induced_stability.py``).

Every level evaluates the points it receives in ONE call: the native
executor fills its chunks (binary64 SIMD chunks of the narrowest width
holding the rest, the scalar lanes of the extended carriers in filled
chunks, ``rusticol-core loop_execution::v2::lane_chunks``), so no padded
lane is evaluated and one executor serves each carrier. Every point's
value is its own lane's: the same as in a call of its own. The
per-scale partition and flavour members stay inside the level evaluator.
The rescue is off unless a policy is passed (REQUIREMENTS_ADDENDUM C:
never on for timing); it is offered by the Python native ``evaluate``
only, in no SDK, the CLI or the configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

from pyamplicol.api.errors import CompatibilityError, EvaluationError
from pyamplicol.api.virtual_results import (
    VirtualEvaluation,
    VirtualLaneDiagnostic,
    VirtualRescueAttempt,
    VirtualRescueLevel,
    VirtualRescuePolicy,
    VirtualRescueRecord,
    splits_poles,
)
from pyamplicol.loops.v2.stability import relative_text

# One level's evaluation of the points at the given batch indices (in that
# order), failures returned as failed lanes (``on_failure="return"``).
LevelEvaluator = Callable[[tuple[int, ...], VirtualRescueLevel], VirtualEvaluation]

RESCUE_SCOPE = (
    "a VirtualRescuePolicy (stability=) applies to evaluate only; "
    "evaluate_resolved and evaluate_correlated(_many) take stability_checks "
    "and leave escalation to the caller"
)
RESCUE_OWNS_CARRIER = (
    "a VirtualRescuePolicy chooses each level's arithmetic, precision and "
    "stability checks; pass no arithmetic, precision or stability_checks "
    "with stability="
)


def loop_induced_stack() -> VirtualRescuePolicy:
    """The loop-induced default stack (m-lit F3 (b), USER 2026-11-08 ~15:40,
    DECISIONS 15:32): the default ladder's binary64 level on TIRust (at the
    default target, its acceptance calibrated from it), the same level on
    NinjaRS, then the default ladder's higher levels (``"auto"``)."""

    first, *rest = VirtualRescuePolicy().levels
    return VirtualRescuePolicy(
        (replace(first, reduction="tirust"), replace(first, reduction="ninjars"), *rest)
    )


def default_stack(
    policy: VirtualRescuePolicy | None,
    contribution: str,
    requested: str,
    fallback: str | None,
) -> VirtualRescuePolicy | None:
    """The stack a call runs: the loop-induced default stack
    (``loop_induced_stack``) for the default policy (``VirtualRescuePolicy()``)
    of a loop-induced call requesting ``"auto"`` whose ``"auto"`` takes
    TIRust in binary64 with a per-point NinjaRS fallback (``fallback``, the
    native policy's choice there: none on a TIR-only artifact or in a build
    without TIRust); any other policy as given. A call naming a backend runs
    the default ladder on it at every level (its levels are ``"auto"``)."""

    if (
        policy is None
        or policy != VirtualRescuePolicy()
        or contribution != "loop-induced"
        or requested != "auto"
        or fallback != "ninjars"
    ):
        return policy
    return loop_induced_stack()


def rescue_policy(
    stability: object,
    *,
    default: VirtualRescuePolicy | None = None,
    arithmetic: str | None = None,
    precision: int | None = None,
    default_precision: int | None = None,
    stability_checks: object = None,
) -> VirtualRescuePolicy | None:
    """The rescue policy of one call: ``stability`` when given (refused
    together with an explicit carrier or checks, which the levels own), else
    the load-time ``default`` unless the call names its own arithmetic, a
    precision other than ``default_precision`` or stability checks (such a
    call evaluates in that one carrier, without rescue)."""

    explicit = (
        arithmetic is not None
        or (precision is not None and precision != default_precision)
        or stability_checks is not None
    )
    if stability is None:
        return None if explicit else default
    if not isinstance(stability, VirtualRescuePolicy):
        raise TypeError("stability takes a VirtualRescuePolicy")
    if explicit:
        raise CompatibilityError(RESCUE_OWNS_CARRIER)
    return stability


@dataclass(frozen=True, slots=True)
class _Kept:
    """The evaluation (and its lane) that returns one point."""

    evaluation: VirtualEvaluation
    lane: int
    level: int
    met: bool
    # Kept below the last level by the stopping rule (``_stalled``): the
    # lower-precision attempt whose estimate it did not improve on.
    stalled: VirtualRescueAttempt | None = None


def rescue(
    policy: VirtualRescuePolicy,
    count: int,
    evaluate: LevelEvaluator,
    *,
    on_failure: str = "raise",
) -> VirtualEvaluation:
    """Run ``policy`` on a batch of ``count`` points through ``evaluate``
    (module docstring); one ``VirtualEvaluation`` in the batch order with
    ``rescue`` records. ``on_failure="raise"`` raises only for a point that
    failed at every level."""

    levels = policy.levels
    attempts: list[list[VirtualRescueAttempt]] = [[] for _ in range(count)]
    kept: list[_Kept | None] = [None] * count
    pending = tuple(range(count))
    template: VirtualEvaluation | None = None
    for k, level in enumerate(levels):
        if k and not pending:
            break
        last = k == len(levels) - 1
        batches = [pending]
        retry: list[int] = []
        for batch in batches:
            evaluation = evaluate(batch, level)
            if len(evaluation.diagnostics) != len(batch):
                raise EvaluationError("a rescue level returned another lane count")
            template = template or evaluation
            for lane, index in enumerate(batch):
                diagnostic = evaluation.diagnostics[lane]
                attempt = _attempt(evaluation, lane, level)
                attempts[index].append(attempt)
                met = (
                    diagnostic.valid
                    and attempt.estimate is not None
                    and attempt.estimate <= level.target
                )
                stalled = (
                    None
                    if met or last or not diagnostic.valid
                    else _stalled(attempts[index], levels)
                )
                if met or last or stalled is not None:
                    kept[index] = _Kept(evaluation, lane, k, met, stalled)
                else:
                    retry.append(index)
        pending = tuple(sorted(retry))
    if template is None:  # pragma: no cover - level 0 always runs
        raise EvaluationError("a rescue policy ran no level")
    return _merged(template, kept, attempts, policy, on_failure)


def _stalled(
    history: list[VirtualRescueAttempt], levels: tuple[VirtualRescueLevel, ...]
) -> VirtualRescueAttempt | None:
    """The stopping rule (USER 2026-11-18 ladder round): a point whose
    estimate at its latest level (the last of ``history``, one attempt per
    level from level 0) is no smaller than its best estimate at fewer
    precision bits under the same metric is not escalated: more digits did
    not improve it, so its error is not the carrier's rounding (an
    integrated zero, whose relative estimate grows as |A|^2 falls with the
    unit roundoff squared, e.g. g g > z: 4.7e15 in binary64, 8.4e33 in
    double-double, 3.0e99 at 57 digits). Returns that lower-precision
    attempt, or None (escalate). A failed attempt has no estimate and never
    stops a point; nor does a level of the same precision (the other
    backend)."""

    latest = history[-1]
    if latest.estimate is None:
        return None
    metric = levels[len(history) - 1].metric
    lower = [
        attempt
        for attempt, level in zip(history[:-1], levels, strict=False)
        if attempt.estimate is not None
        and attempt.precision_bits < latest.precision_bits
        and level.metric == metric
    ]
    if not lower:
        return None
    best = min(lower, key=lambda attempt: cast(float, attempt.estimate))
    return best if latest.estimate >= cast(float, best.estimate) else None


def _attempt(
    evaluation: VirtualEvaluation, lane: int, level: VirtualRescueLevel
) -> VirtualRescueAttempt:
    diagnostic = evaluation.diagnostics[lane]
    # The level's reduction backend as its evaluation recorded it
    # (``VirtualLaneDiagnostic.reduction``: requested, effective, reason).
    # A best-effort lane (the last level's, ``on_failure="best-effort"``)
    # is an attempt with a value: its estimate when the level checks.
    if not diagnostic.has_value:
        return VirtualRescueAttempt(
            diagnostic.arithmetic,
            diagnostic.precision_bits,
            None,
            diagnostic.detail or diagnostic.status,
            diagnostic.reduction,
            diagnostic.kinematics,
        )
    estimate = None if evaluation.stability is None else evaluation.stability[lane]
    value = None if estimate is None else float(getattr(estimate, level.metric))
    return VirtualRescueAttempt(
        diagnostic.arithmetic,
        diagnostic.precision_bits,
        value,
        reduction=diagnostic.reduction,
        kinematics=diagnostic.kinematics,
    )


def _detail(
    record: VirtualRescueRecord,
    metric: str,
    stalled: VirtualRescueAttempt | None = None,
) -> str:
    level = f"rescue level {record.level} ({record.arithmetic})"
    value = record.attempts[-1].estimate
    if value is None:
        return f"{level}: last level, no stability check; accuracy not estimated"
    if record.met:
        verdict = f"within the target {record.target:.1e}"
    elif stalled is not None:
        verdict = (
            f"target {record.target:.1e} not met; not escalated: no better than "
            f"{stalled.arithmetic}'s "
            f"{relative_text(cast(float, stalled.estimate))} at fewer bits"
        )
    else:
        verdict = f"target {record.target:.1e} not met at the final level"
    return f"{level}: estimated {metric} accuracy {relative_text(value)}, {verdict}"


def _merged(
    template: VirtualEvaluation,
    kept: list[_Kept | None],
    attempts: list[list[VirtualRescueAttempt]],
    policy: VirtualRescuePolicy,
    on_failure: str,
) -> VirtualEvaluation:
    """One evaluation of every point from the lanes that return them."""

    born: list[Any] = []
    values: list[Any] = []
    diagnostics: list[VirtualLaneDiagnostic] = []
    estimates: list[Any] = []
    parts: list[Any] = []
    records: list[VirtualRescueRecord] = []
    failed: dict[int, str] = {}
    for index, item in enumerate(kept):
        if item is None:  # pragma: no cover - the last level keeps every point
            raise EvaluationError(f"rescue kept no evaluation of point {index}")
        evaluation, lane = item.evaluation, item.lane
        level = policy.levels[item.level]
        original = evaluation.diagnostics[lane]
        record = VirtualRescueRecord(
            item.level,
            original.arithmetic,
            original.precision_bits,
            level.target,
            item.met,
            tuple(attempts[index]),
        )
        if original.status == "best-effort":
            # Every level refused the point; the last level's best-effort
            # value is returned, never as a success.
            diagnostic = replace(
                original,
                detail=f"{_detail(record, level.metric)}; {original.detail}",
                failures=tuple(
                    replace(failure, point_id=index) for failure in original.failures
                ),
            )
        elif original.valid:
            diagnostic = replace(
                original,
                status="success" if item.level == 0 else "rescued",
                accuracy_established=item.met,
                detail=_detail(record, level.metric, item.stalled),
            )
        else:
            reasons = "; ".join(
                f"level {k} ({attempt.arithmetic}): {attempt.failure}"
                for k, attempt in enumerate(record.attempts)
            )
            failed[index] = reasons
            diagnostic = replace(
                original,
                detail=f"failed at every rescue level: {reasons}",
                failures=tuple(
                    replace(failure, point_id=index) for failure in original.failures
                ),
            )
        born.append(evaluation.born_values[lane])
        values.append(evaluation.values[lane])
        diagnostics.append(diagnostic)
        estimates.append(
            None if evaluation.stability is None else evaluation.stability[lane]
        )
        parts.append(
            None if evaluation.pole_parts is None else evaluation.pole_parts[lane]
        )
        records.append(record)
    if failed and on_failure == "raise":
        index = min(failed)
        raise EvaluationError(
            f"v2 virtual evaluation failed at point {index} at every rescue "
            f"level: {failed[index]}",
            failures=diagnostics[index].failures,
        )
    checked = any(level.checks.enabled for level in policy.levels)
    return VirtualEvaluation(
        tuple(born),
        tuple(values),
        template.born_orders,
        template.sectors,
        tuple(diagnostics),
        template.r2,
        template.correlated,
        color_accuracy=template.color_accuracy,
        poles=template.poles,
        stability=tuple(estimates) if checked else None,
        pole_parts=tuple(parts) if splits_poles(template.poles) else None,
        rescue=tuple(records),
        # The kind of the levels' results ("virtual" or "loop-induced").
        contribution=template.contribution,
    )


__all__ = [
    "RESCUE_OWNS_CARRIER",
    "RESCUE_SCOPE",
    "LevelEvaluator",
    "rescue",
    "rescue_policy",
]
