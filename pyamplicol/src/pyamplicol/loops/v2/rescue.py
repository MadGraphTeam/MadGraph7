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
   evaluated again;
3. at the last level a point that evaluates keeps its value with the target
   marked as not met; a point that failed at every level is a failed lane.

A level compares the public estimate's field named by its ``metric`` with
its target. An integrated zero (a Landau-Yang |A|^2 of a loop-induced
process, m-lit PLAN §2.a.8; numerical, never declared, RULED Q13) can never
meet a "relative" or "finite" target, its value being itself rounding: it
climbs to the last level and is kept there with the target not met. The
opt-in "scale" metric (``VirtualRescueLevel.metric``: E <= target x S,
the estimate against the pre-cancellation scale S of its floor,
``VirtualStabilityEstimate.scale``) accepts it at the first level, where
its E / S is the floor's FLOOR_UNITS x u (M-LIT LIT-STAB-N, ``tests/unit/
test_loop_v2_induced_stability.py``).

Escalated levels evaluate one point per call: every lane of a non-binary64
carrier is a scalar loop that costs as much padded as filled, so a fixed
lane count of one gives each escalated carrier one native executor, which
compiles its generic-engine templates once (``plan_export.NativeExecutor.
lanes_for`` would otherwise build one executor per power-of-two subset
size). The per-scale partition and flavour members stay inside the level
evaluator. The rescue is off unless a policy is passed (REQUIREMENTS_ADDENDUM
C: never on for timing); it is offered by the Python native ``evaluate``
only, in no SDK, the CLI or the configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

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
        batches = [pending] if k == 0 else [(index,) for index in pending]
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
                if met or last:
                    kept[index] = _Kept(evaluation, lane, k, met)
                else:
                    retry.append(index)
        pending = tuple(sorted(retry))
    if template is None:  # pragma: no cover - level 0 always runs
        raise EvaluationError("a rescue policy ran no level")
    return _merged(template, kept, attempts, policy, on_failure)


def _attempt(
    evaluation: VirtualEvaluation, lane: int, level: VirtualRescueLevel
) -> VirtualRescueAttempt:
    diagnostic = evaluation.diagnostics[lane]
    # The level's reduction backend as its evaluation recorded it
    # (``VirtualLaneDiagnostic.reduction``: requested, effective, reason).
    if not diagnostic.valid:
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


def _detail(record: VirtualRescueRecord, metric: str) -> str:
    level = f"rescue level {record.level} ({record.arithmetic})"
    value = record.attempts[-1].estimate
    if value is None:
        return f"{level}: last level, no stability check; accuracy not estimated"
    verdict = (
        f"within the target {record.target:.1e}"
        if record.met
        else f"target {record.target:.1e} not met at the final level"
    )
    return f"{level}: estimated {metric} accuracy {value:.2e}, {verdict}"


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
        if original.valid:
            diagnostic = replace(
                original,
                status="success" if item.level == 0 else "rescued",
                accuracy_established=item.met,
                detail=_detail(record, level.metric),
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
