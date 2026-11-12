# SPDX-License-Identifier: 0BSD
"""``pyamplicol profile`` of a loop artifact: both reduction backends side by
side (``LoopProfileReport``).

RUNTIME (``--runtime``, the default): per backend the time per point and per
stage of ``benchmarking.profile_virtual`` in the default setup (binary64,
finite part, the checked reduction, no stability checks and no rescue), and
the ratio to the fastest backend.

STABILITY (``--stability``): per backend, over N flat (RAMBO) phase-space
points (``cli.flat_points``, ``evaluation.phase_space``'s seed and energy),
the accuracy of the pure binary64 result and of its own stability estimate,
measured as in sect. 3.4 of arXiv:1103.0621 (MadLoop) and the section
"Checks, stability, and recovery of numerically-unstable integral
reductions" of arXiv:1405.0301 (MadGraph5_aMC@NLO): the number of stable
digits of a value v against a reference r is -log10(|v - r| / |r|).

- TRUE digits: the binary64 value against a double-double re-run of the same
  point on the same backend (the reference only: no automatic rescue). A
  point the backend refused or failed in binary64 has no value: 0 true
  digits (it ranks at the worst end) and no estimate.
- ESTIMATED digits: the binary64 call's own stability estimate with the
  default rotation test (``VirtualStabilityChecks(1)``: one default
  rotation, what an evaluation's ``evaluation.stability_checks`` default
  gives; the rescue's own check set is stronger), its absolute finite-part
  errors summed over the sectors relative to the total.

Both are read on the finite part of the total (the sum over the
squared-order sectors; the poles are not computed) and capped to binary64's
range [0, -log10(2^-53)] (``stable_digits``). The MEASURED points are those
whose double-double reference exists and those that failed in binary64 (no
value, whatever the reference); a binary64 value without a reference leaves
the truth unknown and is counted apart. They are sorted by true digits from
worst to best; at each percentile p of ``PERCENTILES`` the true digits of
the point of rank floor(p N) (1-based, N measured points; 50 %: the lower
middle point) are reported, with the deviation |estimated - true| (min,
median, max) and the signed median (estimated - true: positive when the
estimate is optimistic) over the points with an estimate in the class of the
points of rank floor(p' N) < rank <= floor(p N), p' the previous percentile
(0 for the first). A class may be empty at small N (``needed_points``). The
default checks flag a point unstable when its estimate exceeds the default
target (the relative metric of the rescue, per sector) or when it failed in
binary64 (a failure is reported, so detected); a point is truly inaccurate
when its true relative error exceeds the same target or it has no binary64
value.

Every evaluation goes through the public ``Runtime.evaluate`` with explicit
``reduction=``; a backend that refuses the process (``ReductionUnavailable``)
is reported with its reason.
"""

from __future__ import annotations

import math
import os
import re
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.config import RunConfig

# The percentiles of the sorted true digits (worst first): 1 per mil, 1 %,
# 10 % and 50 % (the median).
PERCENTILES = (Fraction(1, 1000), Fraction(1, 100), Fraction(1, 10), Fraction(1, 2))
# binary64's range of stable digits: -log10 of its unit roundoff.
BINARY64_DIGITS = 53 * math.log10(2.0)


def stable_digits(relative_error: float) -> float:
    """-log10(relative error), capped to [0, ``BINARY64_DIGITS``]: an exact
    agreement (or one below binary64's roundoff) counts all of binary64's
    digits, an error of 100 % or more (or a non-finite one) none."""

    if math.isnan(relative_error) or relative_error >= 1.0:
        return 0.0
    if relative_error <= 2.0**-53:
        return BINARY64_DIGITS
    return min(max(-math.log10(relative_error), 0.0), BINARY64_DIGITS)


def relative_error(value: float, reference: Decimal | float) -> float:
    """|value - reference| / |reference|, formed in decimal arithmetic above
    the reference's precision; 0 for two exact zeros, inf for a nonzero value
    against a zero reference."""

    with localcontext() as context:
        context.prec = 60
        exact = Decimal(value)
        ref = Decimal(reference) if not isinstance(reference, Decimal) else reference
        if ref == 0:
            return 0.0 if exact == 0 else math.inf
        return float(abs(exact - ref) / abs(ref))


@dataclass(frozen=True, slots=True)
class StabilityPercentile:
    """One percentile row: ``rank`` = floor(p N) (0: below resolution, no
    point), the true digits at that rank, the class's point count (``failed``
    of them without a binary64 value) and the deviations |estimated - true|
    (min, median, max) and signed median (estimated - true) over its points
    with an estimate."""

    percentile: float
    rank: int
    digits: float | None
    count: int
    deviation_min: float | None
    deviation_median: float | None
    deviation_max: float | None
    signed_median: float | None
    needed_points: int
    failed: int = 0


def needed_points(index: int, percentiles: Sequence[Fraction] = PERCENTILES) -> int:
    """The smallest N whose class ``index`` holds a point: floor(p N) >
    floor(p' N) (found below ceil(1 / (p - p')), where the difference
    reaches one)."""

    p = percentiles[index]
    previous = percentiles[index - 1] if index else Fraction(0)
    bound = math.ceil(1 / (p - previous))
    for n in range(1, bound + 1):
        if math.floor(p * n) > math.floor(previous * n):
            return n
    return bound


def percentile_table(
    true_digits: Sequence[float],
    estimated_digits: Sequence[float | None],
    percentiles: Sequence[Fraction] = PERCENTILES,
) -> tuple[StabilityPercentile, ...]:
    """The percentile rows of points with ``true_digits`` and
    ``estimated_digits`` (aligned; None: a point without an estimate, a
    binary64 failure), sorted by true digits from worst to best (ties keep
    the point order)."""

    if len(true_digits) != len(estimated_digits):
        raise ValueError("true and estimated digits are aligned per point")
    n = len(true_digits)
    order = sorted(range(n), key=lambda k: true_digits[k])
    rows = []
    previous = 0
    for index, p in enumerate(percentiles):
        rank = math.floor(p * n)
        members = order[previous:rank]
        signed = [
            estimated - true_digits[k]
            for k in members
            if (estimated := estimated_digits[k]) is not None
        ]
        deviations = [abs(value) for value in signed]
        rows.append(
            StabilityPercentile(
                float(p),
                rank,
                true_digits[order[rank - 1]] if rank >= 1 else None,
                len(members),
                min(deviations) if deviations else None,
                statistics.median(deviations) if deviations else None,
                max(deviations) if deviations else None,
                statistics.median(signed) if signed else None,
                needed_points(index, percentiles),
                len(members) - len(signed),
            )
        )
        previous = max(previous, rank)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class BackendRuntime:
    """One backend's runtime profile, or its refusal (typed reason)."""

    backend: str
    profile: VirtualProfileResult | None
    refusal: str | None = None
    ratio: float | None = None
    # The reduction request of the timed calls: "auto" for the process
    # type's default backend (its per-point fallback included, the default
    # setup), the backend itself otherwise.
    requested: str | None = None


@dataclass(frozen=True, slots=True)
class StabilityPoint:
    """One flat point of a backend's stability run: its ``index`` among the
    flat points (point i from seed + i), ``status`` ("measured": accepted in
    binary64; "best-effort": refused by the backend's own checks, measured
    with its best-effort value, the unchecked reduction's, m-lit
    ux-besteffort; "binary64-failed": no binary64 value at all, 0 true digits
    and no estimate; "reference-failed": no double-double reference, not
    measured), the digits and the refusal or failure ``reason``;
    ``uncertainty_digits``: a best-effort point's stated uncertainty
    (``VirtualBestEffort.uncertainty``, m-lit ux-polish) in digits."""

    index: int
    status: str
    true_digits: float | None
    estimated_digits: float | None
    reason: str | None = None
    uncertainty_digits: float | None = None


@dataclass(frozen=True, slots=True)
class BackendStability:
    """One backend's stability statistics (``LoopProfileReport``), or its
    refusal. ``points``: the measured points (a reference exists; binary64
    failures included at 0 digits), ``failed_binary64`` of them failed in
    binary64; ``failed_reference``: points without a reference (left out);
    ``true_digits`` and ``estimated_digits`` per measured point (the flat
    points' order; None: no estimate), ``per_point`` every flat point;
    ``flagged``: measured points the default checks flag (estimate above
    ``target``, or a binary64 failure); ``flagged_inaccurate``: of those, the
    truly inaccurate ones; ``unflagged_inaccurate``: truly inaccurate points
    not flagged; ``truly_inaccurate``: all truly inaccurate ones.
    ``best_effort``: measured points the backend refused in binary64 and
    served with their best-effort value (counted flagged: never accepted),
    measured at their true digits; ``best_effort_covered``: of those with an
    uncertainty, the ones whose true relative error is within it."""

    backend: str
    refusal: str | None = None
    points: int = 0
    failed_binary64: int = 0
    failed_reference: int = 0
    true_digits: tuple[float, ...] = ()
    estimated_digits: tuple[float | None, ...] = ()
    percentiles: tuple[StabilityPercentile, ...] = ()
    flagged: int = 0
    flagged_inaccurate: int = 0
    unflagged_inaccurate: int = 0
    truly_inaccurate: int = 0
    seconds: float = 0.0
    per_point: tuple[StabilityPoint, ...] = ()
    best_effort: int = 0
    best_effort_covered: int = 0


@dataclass(frozen=True, slots=True)
class FailureCause:
    """One class of failed or refused points of a profile table (the
    distribution printed under the runtime and stability tables): the
    ``backend``, the ``table`` ("runtime" or "stability"), the ``kind``
    ("best-effort": refused by the backend's own checks and served with its
    best-effort value; "no value": no binary64 value at all; "no reference":
    no double-double reference, stability only), the ``count`` of points of
    that class among the backend's ``total`` points of the table, and the
    ``reason``: the first message of the class (reasons are grouped with
    their numbers masked, as ``plan_export.bounded_reasons`` groups them)."""

    backend: str
    table: str
    kind: str
    count: int
    total: int
    reason: str


def failure_causes(
    items: Sequence[tuple[str, str, str, str]], totals: Mapping[tuple[str, str], int]
) -> tuple[FailureCause, ...]:
    """The ``FailureCause`` classes of ``items`` = (backend, table, kind,
    reason) per failed point, grouped by backend, table, kind and the
    number-masked reason, most frequent first per table (ties: first seen);
    ``totals[(backend, table)]`` the backend's point count in that table."""

    from pyamplicol.loops.v2.plan_export import _NUMBER

    first: dict[tuple[str, str, str, str], str] = {}
    counts: dict[tuple[str, str, str, str], int] = {}
    for backend, table, kind, reason in items:
        masked = _NONFINITE.sub("#", _NUMBER.sub("#", reason))
        key = (backend, table, kind, masked)
        first.setdefault(key, reason)
        counts[key] = counts.get(key, 0) + 1
    order = {key: index for index, key in enumerate(first)}
    ranked = sorted(
        first, key=lambda key: (key[1] != "runtime", -counts[key], order[key])
    )
    return tuple(
        FailureCause(
            key[0], key[1], key[2], counts[key], totals[(key[0], key[1])], first[key]
        )
        for key in ranked
    )


# Non-finite numbers, masked with the finite ones.
_NONFINITE = re.compile(r"\b(?:[-+]?inf|nan)\b")
_MORE = re.compile(r"(?: \(\+\d+ more(?: like this)?\))+$")


def _strip_status(reason: str, status: str) -> str:
    """A point's failure cause: the first of its "; "-joined messages (one
    per refused lane or check, ``benchmarking._first_message``) without the
    status and the executor's stage name (the kind says it) and without the
    "(+N more ...)" counts, which differ from point to point."""

    from pyamplicol.benchmarking import _first_message

    for prefix in (f"{status}: v2-evaluation: ", f"{status}: "):
        if reason.startswith(prefix):
            reason = reason[len(prefix) :]
            break
    return _MORE.sub("", _MORE.sub("", _first_message(reason)))


@dataclass(frozen=True, slots=True)
class LoopProfileReport:
    """``pyamplicol profile`` of a loop artifact (module docstring):
    ``runtime`` and ``stability`` per backend (None when not requested),
    the backends in the order profiled (the process type's default first),
    and the setups measured."""

    process_id: str
    process_expression: str | None
    contribution: str
    default_backend: str | None
    backends: tuple[str, ...]
    runtime_setup: str | None
    runtime: tuple[BackendRuntime, ...] | None
    stability_setup: Mapping[str, object] | None
    stability: tuple[BackendStability, ...] | None
    # The failed and refused points of both tables per backend and reason
    # class (``FailureCause``; empty when nothing failed).
    failure_causes: tuple[FailureCause, ...] = ()


def _backends(requested: object, contribution: str) -> tuple[str, ...]:
    from pyamplicol.loops.v2.reduction import BACKENDS, DEFAULTS

    if requested is not None:
        return (str(requested),)
    default = DEFAULTS.get(contribution)
    return tuple(sorted(BACKENDS, key=lambda name: name != default))


def _validation_points(artifact: Path, process_id: str) -> list[Any]:
    import json

    record = json.loads(
        (artifact / "processes" / process_id / "validation-momenta.json").read_text()
    )
    return [[entry["momentum"] for entry in point] for point in record["points"]]


def _total_finite(row: Any) -> Any:
    total = row[0][0]
    for sector in row[1:]:
        total = total + sector[0]
    return total


def _estimated_relative(estimate: Any, total: float) -> float:
    """The estimate's relative error of the total's finite part: its absolute
    finite-part errors summed over the sectors over |total| (the per-
    coefficient relative estimate when no absolute one is recorded)."""

    if estimate.absolute is None:
        return float(estimate.finite)
    error = sum(float(sector.finite) for sector in estimate.absolute)
    if total == 0.0:
        return 0.0 if error == 0.0 else math.inf
    return error / abs(total)


def _served_points(profile: Any) -> int:
    """The timed points a profile's backend served (not failed)."""

    return profile.batch * profile.samples - profile.failed_points


def _runtime_setup(unchecked: bool, bench: Any) -> str:
    reduction = "unchecked reduction" if unchecked else "checked reduction"
    return (
        f"binary64, finite part, {reduction}; {bench.batch_size} points per "
        f"call, median of {max(bench.minimum_samples, 1)} calls after "
        f"{bench.warmup_runs} warm-up"
    )


def profile_loop(
    artifact: os.PathLike[str] | str,
    config: RunConfig,
    *,
    points: Sequence[Any] | None = None,
) -> LoopProfileReport:
    """Profile a loop artifact (module docstring): the runtime and/or the
    stability statistics of each reduction backend, as ``config.benchmark``
    selects (neither ``runtime`` nor ``stability``: the runtime only), on
    ``config.evaluation.reduction`` alone when it is set."""

    from pyamplicol.api import Runtime
    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.benchmarking import profile_virtual
    from pyamplicol.loops.v2.reduction import DEFAULTS

    bench = config.benchmark
    evaluation = config.evaluation
    path = Path(os.fspath(artifact)).expanduser().resolve(strict=False)
    options: dict[str, Any] = {}
    if evaluation.reduction_target is not None:
        options["reduction_target"] = evaluation.reduction_target
    if evaluation.reduction_unchecked:
        options["reduction_unchecked"] = True
    for name in ("kinematics_improvement", "kinematics_improvement_binary64"):
        if getattr(evaluation, name) is not None:
            options[name] = getattr(evaluation, name)
    runtime = Runtime.load(path, process=evaluation.process, **options)
    physics = runtime.physics
    contribution = runtime.contribution_kind
    backends = _backends(evaluation.reduction, contribution)
    want_stability = bench.stability
    want_runtime = bench.runtime or not want_stability

    runtime_rows = None
    if want_runtime:
        timed = list(points) if points else _validation_points(path, physics.process_id)
        rows = []
        default = DEFAULTS.get(contribution)
        for backend in backends:
            # The default backend in the default setup: "auto" (with its
            # per-point fallback), unless one backend was asked for.
            request = (
                "auto"
                if evaluation.reduction is None and backend == default
                else backend
            )
            try:
                # The default policy: a refused point is served with its
                # best-effort value, and timed as such (m-lit ux-besteffort).
                profile = profile_virtual(
                    runtime,
                    config,
                    points=timed,
                    reduction=request,
                    on_failure="best-effort",
                )
            except ReductionUnavailable as error:
                rows.append(
                    BackendRuntime(backend, None, str(error), requested=request)
                )
                continue
            rows.append(BackendRuntime(backend, profile, requested=request))
        # A backend that failed every timed point is not compared (its time
        # is that of failing, not of serving the process).
        served = [
            row.profile.seconds_per_point
            for row in rows
            if row.profile and _served_points(row.profile)
        ]
        fastest = min(served) if served else None
        runtime_rows = tuple(
            BackendRuntime(
                row.backend,
                row.profile,
                row.refusal,
                None
                if row.profile is None or not fastest or not _served_points(row.profile)
                else row.profile.seconds_per_point / fastest,
                row.requested,
            )
            for row in rows
        )

    stability_rows = None
    stability_setup = None
    if want_stability:
        stability_rows, stability_setup = _stability(
            runtime, path, physics, config, backends
        )
    causes: list[tuple[str, str, str, str]] = []
    totals: dict[tuple[str, str], int] = {}
    for row in runtime_rows or ():
        profile = row.profile
        if profile is None:
            continue
        totals[(row.backend, "runtime")] = profile.batch * profile.samples
        for kind, status, reasons in (
            ("best-effort", "best-effort", profile.best_effort_reasons),
            ("no value", "failed", profile.failure_reasons),
        ):
            for reason, count in reasons:
                text = _strip_status(reason, status)
                causes += [(row.backend, "runtime", kind, text)] * count
    for stability_row in stability_rows or ():
        totals[(stability_row.backend, "stability")] = len(stability_row.per_point)
        for point in stability_row.per_point:
            kind = _CAUSE_KINDS.get(point.status)
            if kind is not None:
                text = _strip_status(
                    point.reason or kind,
                    "best-effort" if kind == "best-effort" else "failed",
                )
                causes.append((stability_row.backend, "stability", kind, text))
    return LoopProfileReport(
        physics.process_id,
        physics.process,
        contribution,
        DEFAULTS.get(contribution),
        backends,
        _runtime_setup(bool(evaluation.reduction_unchecked), bench)
        if want_runtime
        else None,
        runtime_rows,
        stability_setup,
        stability_rows,
        failure_causes(causes, totals),
    )


# The stability point statuses of the failure distribution and their kinds.
_CAUSE_KINDS = {
    "best-effort": "best-effort",
    "binary64-failed": "no value",
    "reference-failed": "no reference",
}


def _stability(
    runtime: Any, artifact: Path, physics: Any, config: RunConfig, backends: Any
) -> tuple[tuple[BackendStability, ...], Mapping[str, object]]:
    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.cli.flat_points import draw_flat_points
    from pyamplicol.loops.v2.reduction import default_target

    bench = config.benchmark
    evaluation = config.evaluation
    space = evaluation.phase_space
    flat = draw_flat_points(
        artifact,
        process=physics.process_id,
        points=bench.stability_points,
        seed=space.seed,
        sqrt_s=space.sqrt_s,
    )
    points = flat.as_inputs(exact=False)
    configured = evaluation.stability_checks.to_checks()
    checks = configured if configured is not None else VirtualStabilityChecks(1)
    target = (
        evaluation.reduction_target
        if evaluation.reduction_target is not None
        else default_target()
    )
    scales = evaluation.mu_r2 if evaluation.mu_r2 is not None else flat.mu_r2()
    chunk = max(bench.batch_size, 1)
    rows = []
    for backend in backends:
        begin = time.perf_counter()
        try:
            plain: list[Any] = []
            estimates: list[Any] = []
            reasons: list[str | None] = []
            statuses: list[str] = []
            uncertainties: list[float | None] = []
            reference: list[Any] = []
            reference_reasons: list[str | None] = []
            for start in range(0, len(points), chunk):
                batch = points[start : start + chunk]
                mu = (
                    scales
                    if not isinstance(scales, list)
                    else scales[start : start + chunk]
                )
                common = {"mu_r2": mu, "poles": "none", "reduction": backend}
                # A point the backend refuses keeps its best-effort value
                # (the default policy, m-lit ux-besteffort), measured as such;
                # the double-double reference counts only accepted values.
                checked = runtime.evaluate(
                    batch,
                    stability_checks=checks,
                    on_failure="best-effort",
                    **common,
                )
                plain.extend(checked.values)
                estimates.extend(checked.stability)
                reasons.extend(_reasons(checked))
                statuses.extend(checked.statuses)
                uncertainties.extend(_uncertainties(checked))
                exact = runtime.evaluate(
                    batch, arithmetic="double-double", on_failure="return", **common
                )
                reference.extend(exact.values)
                reference_reasons.extend(_reasons(exact))
        except ReductionUnavailable as error:
            rows.append(BackendStability(backend, refusal=str(error)))
            continue
        rows.append(
            _backend_statistics(
                backend,
                plain,
                estimates,
                reference,
                target,
                time.perf_counter() - begin,
                reasons=reasons,
                reference_reasons=reference_reasons,
                statuses=statuses,
                uncertainties=uncertainties,
            )
        )
    setup = {
        "points": len(points),
        "seed": flat.seed,
        "sqrt_s": flat.sqrt_s,
        "generator": "flat (RAMBO), beams along +-z in the centre-of-mass frame",
        "measured": "finite part of the total (sum over the squared-order sectors)",
        "reference": "double-double re-run of the same point on the same backend",
        "estimate": (
            "the binary64 call's own estimate, "
            + (
                "evaluation.stability_checks"
                if configured is not None
                else "the default rotation test (one default rotation; "
                "the rescue's check set is not measured)"
            )
        ),
        "target": target,
        "binary64_digits": BINARY64_DIGITS,
        "percentiles": [float(p) for p in PERCENTILES],
        "mu_r2": evaluation.mu_r2 or "s of the initial state",
    }
    return tuple(rows), setup


def _uncertainties(result: Any) -> list[float | None]:
    """Per point the best-effort uncertainty (relative; None: no
    best-effort value, or none could be estimated)."""

    return [
        None if best is None else best.uncertainty
        for best in (getattr(lane, "best_effort", None) for lane in result.diagnostics)
    ]


def _reasons(result: Any) -> list[str | None]:
    from pyamplicol.benchmarking import failure_reason

    out: list[str | None] = []
    for valid, lane in zip(result.valid, result.diagnostics, strict=True):
        best = getattr(lane, "best_effort", None)
        out.append(
            None
            if valid
            else best.refusal
            if best is not None
            else failure_reason(lane)
        )
    return out


def _backend_statistics(
    backend: str,
    plain: Sequence[Any],
    estimates: Sequence[Any],
    reference: Sequence[Any],
    target: float,
    seconds: float,
    *,
    reasons: Sequence[str | None] | None = None,
    reference_reasons: Sequence[str | None] | None = None,
    statuses: Sequence[str] | None = None,
    uncertainties: Sequence[float | None] | None = None,
) -> BackendStability:
    true_digits: list[float] = []
    estimated_digits: list[float | None] = []
    per_point: list[StabilityPoint] = []
    flagged = flagged_inaccurate = unflagged_inaccurate = inaccurate = 0
    failed_binary64 = failed_reference = best_effort = covered = 0
    count = len(plain)
    statuses = statuses if statuses is not None else ["success"] * count
    reasons = reasons if reasons is not None else [None] * count
    reference_reasons = (
        reference_reasons if reference_reasons is not None else [None] * count
    )
    uncertainties = uncertainties if uncertainties is not None else [None] * count
    for index, (
        value,
        estimate,
        exact,
        reason,
        reference_reason,
        status,
        uncertainty,
    ) in enumerate(
        zip(
            plain,
            estimates,
            reference,
            reasons,
            reference_reasons,
            statuses,
            uncertainties,
            strict=True,
        )
    ):
        served = status == "best-effort"
        if value is None or (estimate is None and not served):
            # Refused or failed in binary64: no value (0 true digits, the
            # worst end, whatever the reference) and no estimate; the
            # failure is reported, so the default checks detect it.
            failed_binary64 += 1
            true_digits.append(0.0)
            estimated_digits.append(None)
            flagged += 1
            inaccurate += 1
            flagged_inaccurate += 1
            per_point.append(
                StabilityPoint(
                    index,
                    "binary64-failed",
                    0.0,
                    None,
                    reason or "no binary64 value or estimate",
                )
            )
            continue
        if exact is None:
            # No reference: the truth is unknown, the point is not measured.
            failed_reference += 1
            per_point.append(
                StabilityPoint(
                    index,
                    "reference-failed",
                    None,
                    None,
                    reference_reason or "no double-double value",
                )
            )
            continue
        total = float(_total_finite(value))
        true_relative = relative_error(total, _total_finite(exact))
        digits = stable_digits(true_relative)
        estimated = (
            None
            if estimate is None
            else stable_digits(_estimated_relative(estimate, total))
        )
        true_digits.append(digits)
        estimated_digits.append(estimated)
        per_point.append(
            StabilityPoint(
                index,
                "best-effort" if served else "measured",
                digits,
                estimated,
                reason if served else None,
                None
                if not served or uncertainty is None
                else stable_digits(uncertainty),
            )
        )
        if served:
            # Never accepted: flagged whatever its estimate says. Its stated
            # uncertainty (the call's and the recommended checks, m-lit
            # ux-polish) covers the truth or not.
            best_effort += 1
            if uncertainty is not None:
                covered += true_relative <= uncertainty
        flag = served or float(estimate.relative) > target
        wrong = true_relative > target
        flagged += flag
        inaccurate += wrong
        flagged_inaccurate += flag and wrong
        unflagged_inaccurate += wrong and not flag
    return BackendStability(
        backend,
        None,
        len(true_digits),
        failed_binary64,
        failed_reference,
        tuple(true_digits),
        tuple(estimated_digits),
        percentile_table(true_digits, estimated_digits),
        flagged,
        flagged_inaccurate,
        unflagged_inaccurate,
        inaccurate,
        seconds,
        tuple(per_point),
        best_effort,
        covered,
    )


__all__ = [
    "BINARY64_DIGITS",
    "PERCENTILES",
    "BackendRuntime",
    "BackendStability",
    "FailureCause",
    "LoopProfileReport",
    "StabilityPercentile",
    "StabilityPoint",
    "failure_causes",
    "needed_points",
    "percentile_table",
    "profile_loop",
    "relative_error",
    "stable_digits",
]
