# SPDX-License-Identifier: 0BSD
"""``pyamplicol profile`` of a loop artifact: both reduction backends side by
side (``LoopProfileReport``).

RUNTIME (``--runtime``, the default): per backend the time per point and per
stage of ``benchmarking.profile_virtual`` in the default setup (binary64,
finite part, the checked reduction, no stability checks and no rescue), and
the ratio to the fastest backend.

STABILITY (``--stability``): per backend, over N flat (RAMBO) phase-space
points (``runtime.flat_points``), the accuracy of the pure binary64 result
and of its own stability estimate, measured as in sect. 3.4 of
arXiv:1103.0621 (MadLoop) and the section "Checks, stability, and recovery
of numerically-unstable integral reductions" of arXiv:1405.0301
(MadGraph5_aMC@NLO): the number of stable digits of a value v against a
reference r is -log10(|v - r| / |r|).

- TRUE digits: the binary64 value against a double-double re-run of the same
  point on the same backend (the reference only: no automatic rescue).
- ESTIMATED digits: the binary64 call's own stability estimate with the
  default rotation test (``VirtualStabilityChecks(1)``: one default
  rotation; ``evaluation.stability_checks`` replaces it when set), its
  absolute finite-part errors summed over the sectors relative to the total.

Both are read on the finite part of the total (the sum over the squared-order
sectors; the poles are not computed) and capped to binary64's range
[0, -log10(2^-53)] (``stable_digits``). The points are sorted by true digits
from worst to best; at each percentile p of ``PERCENTILES`` the true digits of
the point of rank floor(p N) are reported, with the deviation |estimated -
true| (min, median, max) and the signed median (estimated - true: positive
when the estimate is optimistic) over the class of the points of rank
floor(p' N) < rank <= floor(p N), p' the previous percentile (0 for the
first). A class may be empty at small N (``needed_points``). The default
checks flag a point unstable when its estimate exceeds the default target
(the relative metric of the rescue, per sector); a point is truly inaccurate
when its true relative error exceeds the same target.

Every evaluation goes through the public ``Runtime.evaluate`` with explicit
``reduction=``; a backend that refuses the process (``ReductionUnavailable``)
is reported with its reason.
"""

from __future__ import annotations

import math
import os
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
    point), the true digits at that rank, the class's point count and its
    deviations |estimated - true| (min, median, max) and signed median
    (estimated - true)."""

    percentile: float
    rank: int
    digits: float | None
    count: int
    deviation_min: float | None
    deviation_median: float | None
    deviation_max: float | None
    signed_median: float | None
    needed_points: int


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
    estimated_digits: Sequence[float],
    percentiles: Sequence[Fraction] = PERCENTILES,
) -> tuple[StabilityPercentile, ...]:
    """The percentile rows of points with ``true_digits`` and
    ``estimated_digits`` (aligned), sorted by true digits from worst to best
    (ties keep the point order)."""

    if len(true_digits) != len(estimated_digits):
        raise ValueError("true and estimated digits are aligned per point")
    n = len(true_digits)
    order = sorted(range(n), key=lambda k: true_digits[k])
    rows = []
    previous = 0
    for index, p in enumerate(percentiles):
        rank = math.floor(p * n)
        members = order[previous:rank]
        signed = [estimated_digits[k] - true_digits[k] for k in members]
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
class BackendStability:
    """One backend's stability statistics (``LoopProfileReport``), or its
    refusal. ``true_digits`` and ``estimated_digits`` per evaluated point
    (the flat points' order, failed points left out: ``failed``);
    ``flagged``: points whose default checks exceed ``target``;
    ``flagged_inaccurate``: of those, the truly inaccurate ones;
    ``unflagged_inaccurate``: truly inaccurate points not flagged."""

    backend: str
    refusal: str | None = None
    points: int = 0
    failed: int = 0
    true_digits: tuple[float, ...] = ()
    estimated_digits: tuple[float, ...] = ()
    percentiles: tuple[StabilityPercentile, ...] = ()
    median_true_digits: float | None = None
    median_estimated_digits: float | None = None
    flagged: int = 0
    flagged_inaccurate: int = 0
    unflagged_inaccurate: int = 0
    truly_inaccurate: int = 0
    seconds: float = 0.0
    # Of ``failed``: the points without a binary64 value or estimate (the
    # backend refused or failed them; no rescue runs) and those whose
    # double-double reference failed.
    failed_binary64: int = 0
    failed_reference: int = 0


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


def _scales(points: Sequence[Any], initial: int, mu_r2: str | None) -> Any:
    from pyamplicol.benchmarking import _initial_invariant

    if mu_r2 is not None:
        return mu_r2
    return [_initial_invariant(point, initial) for point in points]


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
    reduction = (
        "unchecked reduction (timing mode, no accuracy certificate)"
        if unchecked
        else "checked reduction (the backends' own acceptance checks, the default)"
    )
    return (
        "default setup: binary64, finite part only (poles none), "
        f"{reduction}, no stability checks, no rescue; {bench.batch_size} points "
        f"per call, median of {max(bench.minimum_samples, 1)} calls after "
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
                profile = profile_virtual(
                    runtime, config, points=timed, reduction=request
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
    )


def _stability(
    runtime: Any, artifact: Path, physics: Any, config: RunConfig, backends: Any
) -> tuple[tuple[BackendStability, ...], Mapping[str, object]]:
    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.reduction import default_target
    from pyamplicol.runtime.flat_points import artifact_kinematics, flat_points

    bench = config.benchmark
    evaluation = config.evaluation
    kinematics = artifact_kinematics(artifact, physics.process_id)
    seed = (
        bench.stability_seed
        if bench.stability_seed is not None
        else config.generation.validation.seed
    )
    sqrt_s = kinematics.sqrt_s if bench.sqrt_s is None else bench.sqrt_s
    points = flat_points(kinematics, bench.stability_points, seed=seed, sqrt_s=sqrt_s)
    configured = evaluation.stability_checks.to_checks()
    checks = configured if configured is not None else VirtualStabilityChecks(1)
    target = (
        evaluation.reduction_target
        if evaluation.reduction_target is not None
        else default_target()
    )
    scales = _scales(points, kinematics.incoming, evaluation.mu_r2)
    chunk = max(bench.batch_size, 1)
    rows = []
    for backend in backends:
        begin = time.perf_counter()
        try:
            plain: list[Any] = []
            estimates: list[Any] = []
            reference: list[Any] = []
            for start in range(0, len(points), chunk):
                batch = points[start : start + chunk]
                mu = (
                    scales if isinstance(scales, str) else scales[start : start + chunk]
                )
                common = {
                    "mu_r2": mu,
                    "poles": "none",
                    "on_failure": "return",
                    "reduction": backend,
                }
                checked = runtime.evaluate(batch, stability_checks=checks, **common)
                plain.extend(checked.values)
                estimates.extend(checked.stability)
                exact = runtime.evaluate(batch, arithmetic="double-double", **common)
                reference.extend(exact.values)
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
            )
        )
    setup = {
        "points": len(points),
        "seed": seed,
        "sqrt_s": sqrt_s,
        "generator": "flat (RAMBO), beams along +-z in the centre-of-mass frame",
        "measured": "finite part of the total (sum over the squared-order sectors)",
        "reference": "double-double re-run of the same point on the same backend",
        "estimate": (
            "the binary64 call's own estimate, "
            + (
                "evaluation.stability_checks"
                if configured is not None
                else "the default rotation test (one default rotation)"
            )
        ),
        "target": target,
        "binary64_digits": BINARY64_DIGITS,
        "percentiles": [float(p) for p in PERCENTILES],
        "mu_r2": evaluation.mu_r2 or "s of the initial state",
    }
    return tuple(rows), setup


def _backend_statistics(
    backend: str,
    plain: Sequence[Any],
    estimates: Sequence[Any],
    reference: Sequence[Any],
    target: float,
    seconds: float,
) -> BackendStability:
    true_digits: list[float] = []
    estimated_digits: list[float] = []
    flagged = flagged_inaccurate = unflagged_inaccurate = inaccurate = 0
    failed = failed_binary64 = failed_reference = 0
    for value, estimate, exact in zip(plain, estimates, reference, strict=True):
        if value is None or estimate is None or exact is None:
            failed += 1
            if value is None or estimate is None:
                failed_binary64 += 1
            else:
                failed_reference += 1
            continue
        total = float(_total_finite(value))
        true_relative = relative_error(total, _total_finite(exact))
        true_digits.append(stable_digits(true_relative))
        estimated_digits.append(stable_digits(_estimated_relative(estimate, total)))
        flag = float(estimate.relative) > target
        wrong = true_relative > target
        flagged += flag
        inaccurate += wrong
        flagged_inaccurate += flag and wrong
        unflagged_inaccurate += wrong and not flag
    return BackendStability(
        backend,
        None,
        len(true_digits),
        failed,
        tuple(true_digits),
        tuple(estimated_digits),
        percentile_table(true_digits, estimated_digits),
        statistics.median(true_digits) if true_digits else None,
        statistics.median(estimated_digits) if estimated_digits else None,
        flagged,
        flagged_inaccurate,
        unflagged_inaccurate,
        inaccurate,
        seconds,
        failed_binary64,
        failed_reference,
    )


__all__ = [
    "BINARY64_DIGITS",
    "PERCENTILES",
    "BackendRuntime",
    "BackendStability",
    "LoopProfileReport",
    "StabilityPercentile",
    "needed_points",
    "percentile_table",
    "profile_loop",
    "relative_error",
    "stable_digits",
]
