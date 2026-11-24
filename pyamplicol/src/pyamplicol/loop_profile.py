# SPDX-License-Identifier: 0BSD
"""``pyamplicol profile`` of a loop artifact: both reduction backends side by
side (``LoopProfileReport``).

RUNTIME (``--runtime``, the default): per backend the time per point and per
stage of ``benchmarking.profile_virtual`` of one fully optimistic binary64
pass per point (finite part, the backend's unchecked reduction alone, no
stability checks and no rescue; ``benchmark.checked``: the checked mode), and
the ratio to the fastest backend. Every call times a new batch of distinct
flat points (``evaluation.phase_space``'s seed and energy, point i from seed
+ i), the same points per call on every backend; at least
``benchmark.minimum_samples`` calls, more until ``target_runtime`` seconds.
Each backend is profiled in its own spawned process that loads the artifact
(``_isolated_profile``) on one core (every thread pool at one thread,
``one_core``: the cost of one Monte Carlo worker), so that its peak RAM is
that of such a worker evaluating this matrix element on that backend
alone. After the binary64 calls the same process times the
reference precision (``benchmark.ground_truth``, ``GroundTruth``) as one
fully optimistic pass per point (``benchmarking._reference_timing``): the
multiplier of the time per point.

STABILITY (``--stability``): per backend, over N flat (RAMBO on diet) phase-space
points (``cli.flat_points``, ``evaluation.phase_space``'s seed and energy),
the accuracy of the pure binary64 result and of its own stability estimate,
measured as in sect. 3.4 of arXiv:1103.0621 (MadLoop) and the section
"Checks, stability, and recovery of numerically-unstable integral
reductions" of arXiv:1405.0301 (MadGraph5_aMC@NLO): the number of stable
digits of a value v against a reference r is -log10(|v - r| / |r|).

- TRUE digits: the binary64 value against a re-run of the same point on the
  same backend in the reference precision (``benchmark.ground_truth``:
  double-double by default, or arbitrary precision; the reference only, no
  automatic rescue; in arbitrary precision a reference is certified by
  another backend's agreeing one, else by its re-run with the recommended
  stability checks, and has none when that estimate exceeds the target). A
  point the backend refused or failed in binary64 has no value: 0 true
  digits (it ranks at the worst end) and no estimate.
- ESTIMATED digits: the binary64 call's own stability estimate with the
  DEFAULT checks (``DEFAULT_ESTIMATE``; ``evaluation.stability_checks``
  instead when set): one default rotation (``VirtualStabilityChecks(1)``)
  on a point NinjaRS serves, one default rotation and the rescaled
  reduction-direction check on a point TIRust serves
  (``reduction.default_checks``; per point under ``"auto"``), its absolute
  finite-part errors summed over the sectors relative to the total.

Both are read on the finite part of the total (the sum over the
squared-order sectors; the poles are not computed) and capped to binary64's
range [0, -log10(2^-53)] (``stable_digits``). The MEASURED points are those
whose reference exists and those that failed in binary64 (no value, whatever
the reference); a binary64 value without a reference leaves the truth
unknown and is counted apart. They are sorted by true digits from worst to
best; at each percentile p of ``PERCENTILES`` the true digits of the point
of rank floor(p N) (1-based, N measured points; 50 %: the lower middle
point; 100 %: the best point) are reported, with the deviation |estimated -
true| (min, median, max) and the signed median (estimated - true: positive
when the estimate is optimistic) over the points with an estimate in the
class of the points of rank floor(p' N) < rank <= floor(p N), p' the
previous percentile (0 for the first); the classes add up to N. The
--measured-precisions other than binary64 are read the same way at their own
precision (``MeasuredPrecision``). A class may be empty at small N
(``needed_points``). The default checks flag a point unstable when its
estimate exceeds the default target (the relative metric of the rescue, per
sector) or when it failed in binary64 (a failure is reported, so detected);
a point is truly inaccurate when its true relative error exceeds the same
target or it has no binary64 value.

Every evaluation goes through the public ``Runtime.evaluate`` with explicit
``reduction=``; a backend that refuses the process (``ReductionUnavailable``)
is reported with its reason.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.config import RunConfig
    from pyamplicol.generation.phase_space_cuts import PhaseSpaceCuts

# The stability points of a run whose plan has none (never used: a plan
# with the stability run asked for sets its points).
STABILITY_FALLBACK = 5
# The percentiles of the sorted true digits (worst first): 1 per mil, 1 %,
# 10 %, 50 % (the median) and 100 % (the best point); their classes add up
# to every measured point.
PERCENTILES = (
    Fraction(1, 1000),
    Fraction(1, 100),
    Fraction(1, 10),
    Fraction(1, 2),
    Fraction(1, 1),
)
# binary64's range of stable digits: -log10 of its unit roundoff.
BINARY64_DIGITS = 53 * math.log10(2.0)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The reference precision of a profile (``benchmark.ground_truth``):
    ``arithmetic`` "double-double" or "arbitrary" at ``digits`` decimal
    digits (None for double-double), or "none" (no reference: the
    estimates only). ``label`` names it in the tables: "dd",
    "arb(<digits>)" or "none"."""

    arithmetic: str
    digits: int | None = None

    @property
    def label(self) -> str:
        if self.arithmetic == "none":
            return "none"
        return "dd" if self.digits is None else f"arb({self.digits})"

    @property
    def name(self) -> str:
        """The name of the precision in titles and prose: "double-double",
        "arb(<digits>)" or "none"."""

        if self.arithmetic == "none":
            return "none"
        return "double-double" if self.digits is None else self.label

    @property
    def present(self) -> bool:
        """Whether there is a reference (not ``--ground-truth none``)."""

        return self.arithmetic != "none"

    @property
    def options(self) -> dict[str, Any]:
        """The ``Runtime.evaluate`` options of this precision."""

        if self.digits is None:
            return {"arithmetic": self.arithmetic}
        return {"arithmetic": self.arithmetic, "precision": self.digits}

    def record(self) -> dict[str, object]:
        """The JSON record of the precision: arithmetic, digits, label."""

        return {
            "arithmetic": self.arithmetic,
            "digits": self.digits,
            "label": self.label,
        }


def _truth_prose(truth: GroundTruth) -> str:
    """The reference precision in words: "double-double" or "arbitrary-
    precision (D digits)"."""

    if truth.arithmetic == "none":
        return "no"
    if truth.digits is None:
        return truth.arithmetic
    return f"arbitrary-precision ({truth.digits} digits)"


def ground_truth(text: str) -> GroundTruth:
    """The ``GroundTruth`` of a ``benchmark.ground_truth`` value (canonical
    after ``BenchmarkConfig``: "double-double", "arbitrary" or
    "arbitrary:DIGITS"); "arbitrary" without digits takes the default
    arbitrary-precision digits (500, the native policy's one definition,
    ``loops.v2.reduction.default_arbitrary_digits``)."""

    from pyamplicol.config.models import _ground_truth
    from pyamplicol.loops.v2.reduction import default_arbitrary_digits

    canonical = _ground_truth(text)
    if canonical in ("double-double", "none"):
        return GroundTruth(canonical)
    _, _, digits = canonical.partition(":")
    if digits:
        return GroundTruth("arbitrary", int(digits))
    return GroundTruth("arbitrary", default_arbitrary_digits())


# The stream key prefix of a measured precision other than binary64
# (``MeasuredPrecision.stream``): "measured:<label>".
_MEASURED = "measured:"


def precision_digits(arithmetic: str, digits: int | None = None) -> float:
    """The decimal digits a carrier resolves: binary64 -log10(2^-53)
    (``BINARY64_DIGITS``), double-double twice binary64's mantissa
    (-log10(2^-106)), arbitrary precision its ``digits``."""

    if arithmetic == "binary64":
        return BINARY64_DIGITS
    if arithmetic == "double-double":
        return 2 * BINARY64_DIGITS
    return float(digits or 0)


@dataclass(frozen=True, slots=True)
class MeasuredPrecision:
    """One entry of ``benchmark.measured_precisions``: the stability of its
    evaluations with the default checks (``arithmetic`` and ``digits`` as
    ``GroundTruth``'s, or "binary64"), against the ground truth below it,
    or its estimates only (``estimates_only``: the ground truth's own
    precision, or --ground-truth none). ``cap``: the digits a true or
    estimated reading can claim, the carrier's own range
    (``precision_digits``) and, against a truth, at most what the truth
    certifies (``truth_cap``: its range, and its cross-check's resolution
    ``_AGREEMENT``); ``capped_by_truth`` when that is below the carrier's
    range."""

    arithmetic: str
    digits: int | None
    estimates_only: bool
    cap: float
    capped_by_truth: bool = False

    @property
    def label(self) -> str:
        if self.arithmetic == "binary64":
            return "b64"
        return GroundTruth(self.arithmetic, self.digits).label

    @property
    def name(self) -> str:
        if self.arithmetic == "binary64":
            return "binary64"
        return GroundTruth(self.arithmetic, self.digits).name

    @property
    def stream(self) -> str:
        """The stability run's stream of this precision: "binary64" (today's
        binary64 stream), else ``_MEASURED`` + its label."""

        return "binary64" if self.arithmetic == "binary64" else _MEASURED + self.label

    @property
    def options(self) -> dict[str, Any]:
        """The ``Runtime.evaluate`` options of this precision."""

        if self.arithmetic == "binary64":
            return {}
        return GroundTruth(self.arithmetic, self.digits).options

    def record(self) -> dict[str, object]:
        return {
            "precision": self.name,
            "arithmetic": self.arithmetic,
            "digits": self.digits,
            "label": self.label,
            "estimates_only": self.estimates_only,
            "digits_cap": self.cap,
            "capped_by_truth": self.capped_by_truth,
        }


def truth_cap(truth: GroundTruth) -> float:
    """The digits a ground truth certifies: its carrier's range, and for an
    arbitrary-precision one at most its cross-check's resolution
    (``_AGREEMENT``: two backends agreeing to it certify the reference)."""

    digits = precision_digits(truth.arithmetic, truth.digits)
    if _reference_checks(truth) is not None:
        digits = min(digits, -math.log10(float(_AGREEMENT)))
    return digits


def measured_precisions(
    values: Sequence[str], truth: GroundTruth
) -> tuple[MeasuredPrecision, ...]:
    """The ``MeasuredPrecision`` of every ``benchmark.measured_precisions``
    entry (canonical, ``config.models._measured_precisions``), in order; a
    ``ConfigurationError`` naming the option and the ground truth for an
    entry above the ground truth (or the same one twice once "arbitrary"
    takes its default digits), before any work."""

    from pyamplicol.config.errors import ConfigurationError

    out: list[MeasuredPrecision] = []
    top = precision_digits(truth.arithmetic, truth.digits) if truth.present else None
    for value in values:
        if value == "binary64":
            arithmetic, digits = "binary64", None
        else:
            resolved = ground_truth(value)
            arithmetic, digits = resolved.arithmetic, resolved.digits
        own = precision_digits(arithmetic, digits)
        same = truth.present and (arithmetic, digits) == (
            truth.arithmetic,
            truth.digits,
        )
        if top is not None and not same and own >= top:
            raise ConfigurationError(
                f"--measured-precisions {value} is not below the --ground-truth "
                f"{truth.name} ({own:.2f} digits against its {top:.2f}): measure "
                "precisions below the ground truth, or the ground truth itself "
                "(its estimates only), or raise --ground-truth"
            )
        if any((m.arithmetic, m.digits) == (arithmetic, digits) for m in out):
            raise ConfigurationError(
                f"--measured-precisions lists {value} twice (with "
                f"--ground-truth {truth.name})"
            )
        estimates_only = not truth.present or same
        cap = own if estimates_only else min(own, truth_cap(truth))
        out.append(
            MeasuredPrecision(arithmetic, digits, estimates_only, cap, cap < own)
        )
    return tuple(out)


def stable_digits(relative_error: float, cap: float = BINARY64_DIGITS) -> float:
    """-log10(relative error), capped to [0, ``cap``] (default binary64's
    range, ``BINARY64_DIGITS``; a measured precision's own,
    ``MeasuredPrecision.cap``): an exact agreement (or one below 10^-cap)
    counts all of them, an error of 100 % or more (or a non-finite one)
    none."""

    if math.isnan(relative_error) or relative_error >= 1.0:
        return 0.0
    if relative_error <= 10.0**-cap:
        return cap
    return min(max(-math.log10(relative_error), 0.0), cap)


def relative_error(
    value: float | Fraction, reference: Decimal | Fraction | float
) -> float:
    """|value - reference| / |reference|, formed exactly (rational
    arithmetic: whatever the reference's precision, nothing is rounded
    before the final float); 0 for two exact zeros, inf for a nonzero value
    against a zero reference, nan for a non-finite value or reference."""

    if isinstance(reference, Decimal) and not reference.is_finite():
        return math.nan
    if not math.isfinite(value) or (
        isinstance(reference, float) and not math.isfinite(reference)
    ):
        return math.nan
    exact = Fraction(value)
    ref = Fraction(reference)
    if ref == 0:
        return 0.0 if exact == 0 else math.inf
    try:
        return float(abs(exact - ref) / abs(ref))
    except OverflowError:
        return math.inf


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
    # The reduction request of the timed calls: the backend itself (the
    # optimistic pass, the default), or with ``benchmark.checked`` "auto" for
    # the process type's default backend (its per-point fallback included).
    requested: str | None = None


@dataclass(frozen=True, slots=True)
class StabilityPoint:
    """One flat point of a backend's stability run: its ``index`` among the
    flat points (point i from seed + i), ``status`` ("measured": accepted in
    binary64; "best-effort": refused by the backend's own checks, measured
    with its best-effort value, the unchecked reduction's;
    "binary64-failed": no binary64 value at all, 0 true digits
    and no estimate; "reference-failed": no reference value, not
    measured), the digits and the refusal or failure ``reason``;
    ``uncertainty_digits``: a best-effort point's stated uncertainty
    (``VirtualBestEffort.uncertainty``) in digits."""

    index: int
    status: str
    true_digits: float | None
    estimated_digits: float | None
    reason: str | None = None
    uncertainty_digits: float | None = None
    # The point's lane rescue (``benchmarking.lane_rescue_reason``:
    # "reduction lanes rescued in <carrier>"), None without one.
    lane_rescue: str | None = None
    # The reduction's own rescue of the point (NinjaRS's cancellation
    # rescue in the checked mode, ``VirtualLaneRescue.local``): the lanes it
    # re-reduced in a higher precision, over the point's evaluations (the
    # reference and its rotated evaluations), and the highest such
    # precision ("double-double", "arbitrary"; None: no lane).
    reduction_rescue_lanes: int = 0
    reduction_rescue_carrier: str | None = None
    # Why lanes of the point were served without accuracy evidence
    # (``VirtualLaneDiagnostic.uncertified``, joined by "; "): its value is
    # returned, its accuracy is not established. None otherwise.
    uncertified: str | None = None
    # Why lanes of the point's REFERENCE were served without accuracy
    # evidence (the reference evaluation's ``VirtualLaneDiagnostic.
    # uncertified``, joined by "; "): the reference is not certified, the
    # true digits are measured against it all the same (flagged here). None
    # otherwise.
    reference_uncertified: str | None = None


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
    # Measured points served with a lane rescue (``StabilityPoint.
    # lane_rescue``).
    lane_rescued: int = 0
    # The reduction's own rescue (``StabilityPoint.reduction_rescue_lanes``):
    # the points with a lane it re-reduced, those lanes, and per precision
    # its lanes (``_with_reduction_rescues``).
    reduction_rescued_points: int = 0
    reduction_rescued_lanes: int = 0
    reduction_rescue_carriers: tuple[tuple[str, int], ...] = ()
    # The points served with lanes without accuracy evidence
    # (``StabilityPoint.uncertified``).
    uncertified_points: int = 0
    # The measured points whose reference was served with lanes without
    # accuracy evidence (``StabilityPoint.reference_uncertified``): measured
    # against a reference that is not certified.
    uncertified_reference_points: int = 0
    # The evaluations of the run per precision: "binary64" (the checked call
    # with its stability estimate) and the reference's arithmetic
    # ("double-double", or "arbitrary" with its "digits"), each
    # with its ``points``, ``seconds`` (the wall time of its batches summed
    # over the workers: its part of ``seconds``) and ``seconds_per_point``
    # (the wall time per point of its batches, the first-call cost of a
    # worker left out as the progress estimate does: ``_StabilityProgress``).
    precision_timing: Mapping[str, Mapping[str, float | int | None]] | None = None
    # Estimates only (``--ground-truth none``): no truth; ``percentiles``
    # rank the estimated digits (``true_digits`` empty), the truth-based
    # counts are 0 and not shown.
    estimate_only: bool = False
    # Every sample's complete record, columnar (``_sample_records``; the
    # JSON's per-sample dump, its fields documented there), or where they
    # were written (``--samples-output``); None: not asked for.
    samples: Mapping[str, object] | None = None

    def cli_json(self) -> dict[str, object]:
        """The JSON form (``cli.main``): the fields, ``per_point`` columnar
        (``PER_POINT_FIELDS``; one compact row per point)."""

        from dataclasses import fields as dataclass_fields

        from pyamplicol.reporting.raw_json import encode_rows

        out: dict[str, object] = {
            item.name: getattr(self, item.name) for item in dataclass_fields(self)
        }
        out["per_point"] = {
            "fields": list(PER_POINT_FIELDS),
            "rows": encode_rows(
                (
                    [
                        point.index,
                        point.status,
                        _rounded(point.true_digits, 3),
                        _rounded(point.estimated_digits, 3),
                        point.reason,
                        _rounded(point.uncertainty_digits, 3),
                        point.lane_rescue,
                        point.reduction_rescue_lanes,
                        point.reduction_rescue_carrier,
                        point.uncertified,
                        point.reference_uncertified,
                    ]
                    for point in self.per_point
                ),
                len(self.per_point),
            ),
        }
        return out


# The columns of a backend's ``per_point`` table in --json.
PER_POINT_FIELDS = (
    "index",
    "status",
    "true_digits",
    "estimated_digits",
    "reason",
    "uncertainty_digits",
    "lane_rescue",
    "reduction_rescue_lanes",
    "reduction_rescue_carrier",
    "uncertified",
    "reference_uncertified",
)
# The columns of a backend's per-sample records (``_sample_records``).
SAMPLE_FIELDS = (
    "index",
    "status",
    "value",
    "sectors",
    "estimate_relative",
    "estimate_finite_relative",
    "estimate_digits",
    "estimate_scale",
    "estimate_checks",
    "estimate_not_run",
    "flagged",
    "best_effort",
    "uncertainty",
    "reason",
    "reference_value",
    "reference_reason",
    "reference_certified",
    "true_relative_error",
    "true_digits",
    "binary64_task",
    "reference_task",
    "reference_checked_task",
    "lane_rescue",
    # The digits as measured, not capped to the precision's range (an
    # exact agreement: "inf").
    "estimate_digits_uncapped",
    "true_digits_uncapped",
    # The reduction's own rescue of the point (``StabilityPoint``).
    "reduction_rescue_lanes",
    "reduction_rescue_carrier",
    # Lanes served without accuracy evidence (``StabilityPoint``).
    "uncertified",
    # The reference's lanes served without accuracy evidence: its reasons
    # (``StabilityPoint.reference_uncertified``); such a reference has
    # ``reference_certified`` false.
    "reference_uncertified",
)
# The columns of the points' inputs (``stability_setup.samples``).
INPUT_FIELDS = ("index", "momenta", "mu_r2")


def _rounded(value: float | None, decimals: int) -> float | str | None:
    """A derived figure (digits) at ``decimals`` decimals; non-finite as
    text."""

    if value is None:
        return None
    if not math.isfinite(value):
        return str(value)
    return round(float(value), decimals)


def _significant(value: float | None, digits: int = 4) -> float | str | None:
    """A diagnostic figure (an estimated or true relative error, a scale) at
    ``digits`` significant digits; non-finite as text."""

    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return float(f"{value:.{digits}g}")


@dataclass(frozen=True, slots=True)
class FailureCause:
    """One class of failed or refused points of a profile table (the
    distribution printed under the runtime and stability tables): the
    ``backend``, the ``table`` ("runtime" or "stability"), the ``kind``
    ("best-effort": refused by the backend's own checks and served with its
    best-effort value; "no value": no binary64 value at all; "lane rescue":
    served, with reduction lanes refused in binary64 reduced again in a
    higher carrier (``VirtualLaneRescue``); "no reference":
    no reference value, stability only; "best-effort in <reference>", "no
    value in <reference>": the same for the runtime's reference timing,
    <reference> the ``GroundTruth.name``), the ``count`` of points of
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
    items: Sequence[tuple[str, str, str, str]],
    totals: Mapping[tuple[str, ...], int],
) -> tuple[FailureCause, ...]:
    """The ``FailureCause`` classes of ``items`` = (backend, table, kind,
    reason) per failed point, grouped by backend, table, kind and the
    number-masked reason, most frequent first per table (ties: first seen);
    ``totals[(backend, table)]`` the backend's point count in that table,
    ``totals[(backend, table, kind)]`` that of a kind counted over other
    points (the runtime's reference timing)."""

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
            key[0],
            key[1],
            key[2],
            counts[key],
            totals.get(key[:3], totals[(key[0], key[1])]),
            first[key],
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
    # Ctrl-C stopped the profile: the tables hold what was measured so far.
    interrupted: bool = False
    # The reference precision of both tables (``GroundTruth.record``).
    ground_truth: Mapping[str, object] | None = None
    # How the counts were set (``profile_plan.Plan.record``: the rule, the
    # counts, the expected and the requested seconds) and the whole
    # profile's wall seconds.
    plan: Mapping[str, object] | None = None
    seconds: float | None = None
    # Every other measured precision's runtime rows (``--measured-
    # precisions`` with --runtime): {"precision": record, "runtime": rows};
    # left out of the JSON when empty (``cli_json``).
    runtime_measured: tuple[Mapping[str, Any], ...] = ()

    def cli_json(self) -> dict[str, object]:
        """The JSON form (``cli.main``): every field, ``runtime_measured``
        only when a precision other than binary64 was timed."""

        from dataclasses import fields

        out = {item.name: getattr(self, item.name) for item in fields(self)}
        if not self.runtime_measured:
            out.pop("runtime_measured")
        return out

    def json_notice(self) -> str | None:
        """The line ``--json`` prints on stderr before a large dump (its
        per-sample records take seconds to write: never a silent tail), with
        its size estimated from the rows already encoded (the samples' and
        the inputs'), the per-point tables and digit lists at their row size
        (``json_size``); ``cli.main`` states the size written after it."""

        size, records = self.json_size()
        if size < 20_000_000:
            return None
        return (
            f"Writing the JSON result ({records} sample records, about "
            f"{size / 1e6:.0f} MB)"
        )

    def json_size(self) -> tuple[int, int]:
        """The estimated bytes of the --json document's per-sample parts and
        the number of sample records."""

        from pyamplicol.reporting.raw_json import RawJson

        def encoded(table: object) -> RawJson | None:
            if isinstance(table, Mapping):
                rows = table.get("rows")
                if isinstance(rows, RawJson):
                    return rows
            return None

        tables = [encoded(row.samples) for row in self.stability or ()]
        setup = self.stability_setup or {}
        inputs = encoded(setup.get("samples"))
        size = sum(len(t.text) for t in tables if t is not None)
        size += 0 if inputs is None else len(inputs.text)
        for row in self.stability or ():
            # One per-point row (~45 B) and two digit numbers (~20 B each).
            size += 45 * len(row.per_point)
            size += 20 * (len(row.true_digits) + len(row.estimated_digits))
        records = sum(t.rows for t in tables if t is not None)
        return size, records


def _backends(requested: object, contribution: str) -> tuple[str, ...]:
    from pyamplicol.loops.v2.reduction import BACKENDS, DEFAULTS

    if requested is not None:
        return (str(requested),)
    default = DEFAULTS.get(contribution)
    return tuple(sorted(BACKENDS, key=lambda name: name != default))


def _total_finite(row: Any) -> Any:
    total = row[0][0]
    for sector in row[1:]:
        total = total + sector[0]
    return total


def _exact_total(row: Any) -> Fraction:
    """The finite part of a reference value's total, its sectors summed
    exactly (a ``Decimal`` sum would round to its context's precision)."""

    return sum((Fraction(sector[0]) for sector in row), Fraction(0))


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


def _runtime_setup(
    checked: bool,
    seed: int | None,
    threads: Sequence[int] = (),
    rows: Sequence[Any] = (),
    truth: GroundTruth | None = None,
    lane_rescue: bool = True,
    cuts: PhaseSpaceCuts | None = None,
) -> str:
    """The runtime setup line; ``checked``: the checked mode
    (``benchmark.checked``), else the optimistic pass; ``lane_rescue``: the
    timed calls rescue refused reduction lanes (``evaluation.lane_rescue``,
    in both modes); ``seed``: of the flat
    points (None: the caller's points), drawn within ``cuts``;
    ``threads``: the native threads each backend's timed calls report (one:
    one core, ``one_core``; another count is named); ``rows``: the backends'
    runtimes (their calls, points per call, warm-ups and reference points)."""

    from pyamplicol.loops.v2.reduction import NAMES

    reduction = (
        "checked reduction, refused points served best-effort"
        if checked
        else "optimistic: unchecked reduction, no checks"
    )
    if lane_rescue:
        reduction += ", refused lanes rescued"

    points = (
        "points of --momenta"
        if seed is None
        else f"distinct flat points (seed {seed}"
        + ("" if cuts is None else "".join(f", {cut}" for cut in cuts.described()))
        + ")"
    )
    counts = sorted(set(threads) - {1})
    core = (
        "one core"
        if not counts
        else f"{'/'.join(map(str, sorted(set(threads))))} threads"
    )
    profiled = [row for row in rows if getattr(row, "profile", None) is not None]
    calls = ", ".join(
        f"{NAMES.get(row.backend, row.backend)} {row.profile.samples} x "
        f"{row.profile.batch}"
        for row in profiled
    )

    def warming(profile: Any) -> str:
        """How a stream was warmed: its untimed warm-up calls, the warm
        entry, or the entry and the warm-ups after it."""

        entry = getattr(profile, "prepared_seconds", None) is not None
        calls = f"{profile.warmup} warm-up"
        if not entry:
            return calls
        return "a warm entry" + (f" and {calls}" if profile.warmup else "")

    warmups = sorted({warming(row.profile) for row in profiled})
    text = (
        f"binary64, finite part, {reduction}, on {core}; {points}, a new batch "
        f"every call; the median of each backend's timed calls"
        + (f" ({calls} points)" if calls else "")
        + (f" after {' / '.join(warmups)}" if warmups else "")
    )
    references = [
        row.profile.reference.points
        for row in profiled
        if row.profile.reference is not None
    ]
    if truth is not None and truth.present and references:
        warmed = sorted(
            {
                warming(row.profile.reference)
                for row in profiled
                if row.profile.reference is not None
            }
        )
        text += (
            f"; the {truth.label} multiplier from "
            + " / ".join(map(str, references))
            + " reference points"
            + (f" after {' / '.join(warmed)}" if warmed != ["1 warm-up"] else "")
        )
    return text


# The count and size settings --completion-time overrules: (its option, the
# benchmark field it sets on the command line and in a run card).
_COUNT_OPTIONS = (
    ("--target-runtime", "target_runtime"),
    ("--minimum-samples", "minimum_samples"),
    ("--batch-size", "batch_size"),
    ("--warmup-runs", "warmup_runs"),
    ("--stability-points", "stability_points"),
)


def effective_benchmark(bench: Any) -> tuple[Any, list[str]]:
    """The benchmark settings a profile runs with, and the count and size
    options given (``_COUNT_OPTIONS``, command line or run card): with
    --completion-time every one of them back at its default (unmarked as
    given; the plan sets them), else ``bench`` as given."""

    from pyamplicol.config.models import BenchmarkConfig, is_default

    given = [
        flag for flag, name in _COUNT_OPTIONS if not is_default(getattr(bench, name))
    ]
    if bench.completion_time is None or not given:
        return bench, given
    defaults = BenchmarkConfig()
    reset = {name: getattr(defaults, name) for _, name in _COUNT_OPTIONS}
    return replace(bench, **reset), given


def profile_loop(
    artifact: os.PathLike[str] | str,
    config: RunConfig,
    *,
    points: Sequence[Any] | None = None,
    progress: Any = None,
) -> LoopProfileReport:
    """Profile a loop artifact (module docstring): the runtime and/or the
    stability statistics of each reduction backend, as ``config.benchmark``
    selects (neither ``runtime`` nor ``stability``: the runtime only), on
    ``config.evaluation.reduction`` alone when it is set. The runtime times
    ``points`` (default: ``benchmarking.profile_points``)."""

    from pyamplicol.api import Runtime
    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.config.models import is_default
    from pyamplicol.loops.v2.plan_export import DEFAULT_LANES
    from pyamplicol.loops.v2.reduction import DEFAULTS
    from pyamplicol.profile_plan import (
        BUDGET_TOLERANCE,
        DEFAULT_RUNTIME_SECONDS,
        DEFAULT_STABILITY_SECONDS_PER_CORE,
        MEASURE_REPEAT_SECONDS,
        ONE_THREAD_SHARE,
        POST_SECONDS_PER_RECORD,
        RECORD_SECONDS_PER_INPUT,
        RECORD_SECONDS_PER_RECORD,
        STABILITY_PRIOR,
        Costs,
        Known,
        Plan,
        binary64_warmup,
        measurement_affects_counts,
        minimum_counts,
        reference_affects_counts,
        reference_warmup,
        rule_a_drops,
        rule_b_bound,
        rule_b_drops,
        runtime_counts,
        takes_entry,
        warmup_seconds,
    )
    from pyamplicol.profile_plan import plan as plan_profile

    completion = config.benchmark.completion_time
    # --completion-time overrules every count and size (command line and run
    # card alike): the given ones are named once, then reset to their
    # defaults, so that nothing downstream sees them.
    bench, overruled = effective_benchmark(config.benchmark)
    if bench is not config.benchmark:
        config = replace(config, benchmark=bench)
    evaluation = config.evaluation
    path = Path(os.fspath(artifact)).expanduser().resolve(strict=False)
    monitor = _Monitor(progress, f"Profile {path.name}", completion)
    options: dict[str, Any] = {}
    if evaluation.reduction_target is not None:
        options["reduction_target"] = evaluation.reduction_target
    if evaluation.reduction_unchecked:
        options["reduction_unchecked"] = True
    if not evaluation.lane_rescue:
        # --no-lane-rescue: pure binary64 calls, refused points without a value.
        options["lane_rescue"] = False
    for name in ("kinematics_improvement", "kinematics_improvement_binary64"):
        if getattr(evaluation, name) is not None:
            options[name] = getattr(evaluation, name)
    truth = ground_truth(bench.ground_truth)
    # The measured precisions other than binary64 (``--measured-precisions``;
    # an entry above the ground truth refused here, before any work): the
    # stability plan counts their streams at their lower bounds
    # (``measured_prior``) until the run measures them.
    listed = measured_precisions(bench.measured_precisions, truth)
    extra_precisions = [m for m in listed if m.stream != "binary64"]
    binary64_listed = len(extra_precisions) < len(listed)
    # The ground truth listed: its own stream is its evaluation (one, not
    # two: the runtime's multiplier pass, the stability run's reference).
    truth_listed = next(
        (
            m
            for m in extra_precisions
            if truth.present
            and (m.arithmetic, m.digits) == (truth.arithmetic, truth.digits)
        ),
        None,
    )
    # binary64's runtime calls run when listed, or for the ground truth's
    # multiplier pass (in its worker) that another listed precision's table
    # shows when the truth is not listed itself.
    binary64_timed = binary64_listed or (
        truth.present and truth_listed is None and bool(extra_precisions)
    )

    def with_measured(cost: Any) -> Any:
        """``cost`` with the stability run's measured precisions: binary64's
        checked point only when listed, every other precision at its lower
        bound, the ground truth's own as its reference (counted once)."""

        if not extra_precisions:
            return cost
        evaluations = checked_evaluations(evaluation.stability_checks.to_checks())
        return replace(
            cost,
            stability=(cost.stability if binary64_listed else 0.0)
            + sum(
                measured_prior(cost, m, evaluations)
                for m in extra_precisions
                if m is not truth_listed
            )
            # A listed truth's stream is its reference (``stability_reference``,
            # one plain point): its checks' further evaluations on top.
            + (
                (evaluations - 1) * (cost.stability_reference or 0.0)
                if truth_listed is not None
                else 0.0
            ),
        )

    checked = bench.checked
    if checked and evaluation.reduction_unchecked:
        from pyamplicol.config.errors import ConfigurationError

        raise ConfigurationError(
            "benchmark.checked (--checked) times the checked reduction: drop "
            "evaluation.reduction_unchecked (--reduction-unchecked)"
        )
    # A previous profile of this artifact (``_ProfileHistory``): the whole
    # command's estimate from the first second, then the measurement's
    # priors; without one, a time budget stands for the estimate until the
    # first measured call.
    # Only when progress is shown (``--progress off`` passes a
    # NullProgressSink): as generate's history.
    from pyamplicol.reporting.progress import NullProgressSink

    history = (
        _ProfileHistory(path, evaluation.process, truth)
        if progress is not None and not isinstance(progress, NullProgressSink)
        else None
    )
    settings = "total:" + _settings_key(bench, config)
    previous_total = None if history is None else history.seconds(settings)
    provisional_end: list[float | None] = [None]
    if previous_total is not None:
        provisional_end[0] = monitor.begin + previous_total
    elif completion is not None:
        provisional_end[0] = monitor.begin + completion

    def provisional_remaining() -> tuple[float, bool] | None:
        end = provisional_end[0]
        if end is None:
            return None
        return max(end - time.perf_counter(), 1.0), True

    monitor.remaining = provisional_remaining
    monitor.tick()
    loading = monitor.phase("load", "Loading the artifact", None, "steps")
    loading.update(0, None, eta=None, step=f"loading {path.name}")
    load_begin = time.perf_counter()
    try:
        runtime = Runtime.load(path, process=evaluation.process, **options)
    except BaseException:
        loading.end("failed", success=False)
        monitor.close(False)
        raise
    load_seconds = time.perf_counter() - load_begin
    # This process's peak resident size once loaded (a worker's start).
    load_peak = _own_peak_bytes()
    loading.end(runtime.physics.process_id)
    physics = runtime.physics
    contribution = runtime.contribution_kind
    backends = _backends(evaluation.reduction, contribution)
    want_stability = bench.stability
    want_runtime = bench.runtime or not want_stability
    workers = max(1, min(bench.n_core, usable_cores()))
    # The runtime timing's native threads (--runtime-threads), at most the
    # usable cores (said once when reduced): the affinity mask and the CPU
    # quota, the runtime's own clamp (``effective_threads``, evaluation.
    # threads' rule); the stability workers keep one.
    threads = min(
        int(bench.runtime_threads),
        usable_cores(),
        _effective_threads(int(bench.runtime_threads)),
    )
    if threads < bench.runtime_threads:
        _notice(
            f"--runtime-threads {bench.runtime_threads} reduced to {threads}, "
            "the usable cores"
        )
        bench = replace(bench, runtime_threads=threads)
        config = replace(config, benchmark=bench)
    # The runtime counts given (honoured as given); a --target-runtime given
    # alone is each backend's timed binary64 budget, start-up and warm-up
    # outside it, the reference on top (origin/main's meaning), the counts
    # chosen from it (``profile_plan.target_counts``), not given.
    runtime_given = any(
        not is_default(getattr(bench, name))
        for name in ("minimum_samples", "batch_size", "warmup_runs")
    )
    runtime_share = (
        float(bench.target_runtime)
        if completion is None and not is_default(bench.target_runtime)
        else None
    )
    stability_given = (
        None if is_default(bench.stability_points) else int(bench.stability_points)
    )
    if completion is not None and overruled:
        _notice(
            "--completion-time "
            + _duration(completion)
            + " overrules "
            + ", ".join(overruled)
            + " (the counts are chosen from the time)"
        )
    interrupted = False
    batch_max = int(bench.batch_size)
    given_points = stability_given if completion is None else None
    # Every sample's record: for --json or --samples-output (else none: the
    # tables do not need them). The serial seconds per stability point after
    # the run (its statistics, its records): part of --completion-time.
    records = config.output.format == "json" or bench.samples_output is not None
    tail = len(backends) * POST_SECONDS_PER_RECORD + (
        len(backends) * RECORD_SECONDS_PER_RECORD + RECORD_SECONDS_PER_INPUT
        if records
        else 0.0
    )

    def planning(costs_map: Mapping[str, Any], elapsed: float) -> tuple[Any, float]:
        """The plan from ``costs_map`` with ``elapsed`` seconds spent, and a
        stability worker's start: the load and its warm-up (one point per
        stream; none for a stream dearer than MEASURE_REPEAT_SECONDS); the
        measured precisions' streams in the stability costs
        (``with_measured``)."""

        costs_map = {b: with_measured(c) for b, c in costs_map.items()}

        start = (
            load_seconds
            + 1.0
            + sum(
                seconds
                for c in costs_map.values()
                for seconds in (c.stability, c.stability_reference or 0.0)
                if seconds <= MEASURE_REPEAT_SECONDS
            )
        )
        return (
            plan_profile(
                costs_map,
                runtime=want_runtime,
                stability=want_stability,
                workers=workers,
                lanes=DEFAULT_LANES,
                batch_max=batch_max,
                completion=completion,
                elapsed=elapsed,
                stability_start=start,
                runtime_counts_given=runtime_given and completion is None,
                target=None if runtime_given else runtime_share,
                stability_points_given=given_points,
                tail_seconds_per_point=tail if completion is not None else 0.0,
                runtime_target=float(bench.target_runtime),
            ),
            start,
        )

    def future_of(planned: Any, costs_map: Mapping[str, Any]) -> dict[str, float]:
        """What is left of ``planned`` after each phase (the whole command's
        ETA)."""

        future = {
            f"runtime:{b}": counts.expected for b, counts in planned.runtime.items()
        }
        if want_runtime and not planned.runtime:
            # The counts given: their cost at the measured rates (at least the
            # warm-ups and the minimum calls, the budget, the reference's).
            for b, c in costs_map.items():
                calls = bench.warmup_runs + max(bench.minimum_samples, 1)
                seconds = c.start + max(
                    calls * bench.batch_size * c.binary64,
                    bench.warmup_runs * bench.batch_size * c.binary64
                    + float(bench.target_runtime),
                )
                if c.reference is not None:
                    seconds += c.reference + max(
                        float(bench.target_runtime), c.reference
                    )
                future[f"runtime:{b}"] = seconds
        if want_runtime and not binary64_timed:
            # Not timed: binary64's counts only start the others'.
            for b in list(future):
                if b.startswith("runtime:"):
                    future.pop(b)
        if want_runtime:
            # Every other measured precision's calls: its measuring point
            # and its budget (binary64's for that backend), at its lower
            # bound of a point (the ground truth's reference cost, else
            # binary64's) until it runs.
            for b, c in costs_map.items():
                counts = planned.runtime.get(b)
                budget = (
                    float(counts.budget)
                    if counts is not None
                    else float(bench.target_runtime)
                )
                for m in extra_precisions:
                    point = (
                        c.reference
                        if m is truth_listed and c.reference is not None
                        else c.binary64
                    )
                    future[f"runtime:{b}:{m.label}"] = (
                        c.start + point + max(budget, point)
                    )
        if want_stability:
            future["stability"] = planned.stability_seconds
        return future

    def provisional(costs_map: dict[str, Any], left: float) -> None:
        # The whole command from the costs known or expected so far.
        planned_now, _ = planning(costs_map, monitor.elapsed() + left)
        rest = sum(future_of(planned_now, costs_map).values())
        provisional_end[0] = time.perf_counter() + left + rest

    # An explicit --ground-truth keeps the runtime timing's reference pass.
    is_default_truth_forced = not is_default(bench.ground_truth)
    # The warm entry (``prepare`` of the runtime's backend, loadfast): the
    # streams that warm through it instead of their untimed warm-up point
    # (``profile_plan.takes_entry``), counted as such by every plan.
    has_entry = callable(
        getattr(getattr(runtime, "_backend", runtime), "prepare", None)
    )
    excess: dict[tuple[str, str], float] = {}

    def entry_cost(backend: str, precision: str) -> float | None:
        return None if history is None else history.cost(backend, f"{precision}-entry")

    def warms_through_entry(backend: str, precision: str, point: float | None) -> bool:
        """Whether this stream warms through the entry: the runtime offers
        it, the binary64 stream only in the optimistic pass, and
        ``takes_entry`` from the point's cost (None: unknown), the entry's
        remembered cost and the measured first-call excess. With rule B
        open no reference warms through it: its untimed point, which the
        measurement's cold point bounds, lets rule B drop the multiplier
        before it runs."""

        if not has_entry or (precision == "binary64" and checked):
            return False
        if precision == "reference" and rule_b:
            return False
        return takes_entry(
            point,
            entry=entry_cost(backend, precision),
            excess=excess.get((backend, precision)),
        )

    def entry_warming(backend: str, precision: str) -> float:
        """The entry's expected seconds: remembered, else the first-call
        excess it replaces, else none known (seconds, negligible against
        the dear point it replaces)."""

        known = entry_cost(backend, precision)
        if known is None:
            known = excess.get((backend, precision))
        return 0.0 if known is None else known

    def with_warming(backend: str, cost: Any) -> Any:
        """``cost`` with its streams' warming by the entry where they take
        it (``Costs.binary64_warm`` / ``reference_warm``)."""

        call = DEFAULT_LANES * cost.binary64
        point = warmup_seconds(call, cost.binary64)
        return replace(
            cost,
            binary64_warm=entry_warming(backend, "binary64")
            if warms_through_entry(backend, "binary64", point)
            else None,
            reference_warm=entry_warming(backend, "reference")
            if cost.reference is not None
            and warms_through_entry(backend, "reference", cost.reference)
            else None,
        )

    # The scope rules' inputs (below, after the measurement): counts given
    # (how many points or calls run) are honoured as given; a given
    # --target-runtime is the runtime timing's budget (counts given with it
    # win); --completion-time overrules every count.
    default = DEFAULTS.get(contribution)
    counts_given = completion is None and (
        any(
            not is_default(getattr(bench, name))
            for name in ("batch_size", "minimum_samples", "warmup_runs")
        )
        or (want_stability and stability_given is not None)
    )
    target_given = runtime_share is not None and not counts_given
    rules = not counts_given
    timing_alone = rules and want_runtime and not want_stability
    asked_backends = evaluation.reduction is not None or bench.all_backends is not None
    serving = [b for b in _serving(runtime, backends) if b in backends]
    early: list[str] = []
    # The backends rule A (or --no-all-backends) leaves out of the runtime
    # timing: set below, read by ``running``.
    omitted: list[str] = []
    # The scope rules (USER 2026-11-19; ``profile_plan.rule_b_drops`` and
    # ``rule_a_drops``), one decision for every path: each takes ``known``,
    # what this run has measured of every serving backend so far, and is
    # called again whenever a measured number arrives (the measurement's
    # steps, the decider worker after its binary64 call and after its
    # untimed reference point, every runtime row). Invariants: (i) nothing
    # is dropped but on a lower bound past the budget (``lower_bound``:
    # grows only, so a drop is final); (ii) rule A never counts a reference
    # rule B could drop; (iii) the verdict follows the measured numbers
    # alone, whatever the path (measured, without a measurement phase, with
    # or without history: a previous profile's costs set ETAs only); (iv)
    # the record and the counts row are re-planned from what ran.
    # A ground truth listed in --measured-precisions is a precision given:
    # rule B never leaves its timing out.
    rule_b = (
        timing_alone
        and truth.present
        and not is_default_truth_forced
        and truth_listed is None
    )
    a_reference = truth.present and not rule_b
    known: dict[str, Known] = {b: Known(start=load_seconds + 1.0) for b in serving}

    def budget_bound(count: int) -> float:
        """The seconds the minimums of ``count`` backends may take before
        the budget in force counts as exceeded, the counts row's rule
        (``profile_plan.plan``), start-ups aside: ``BUDGET_TOLERANCE`` x the
        budgets of the parts asked, the runtime timing's (the default for
        all backends, or a given --target-runtime per backend) and the
        stability run's (``DEFAULT_STABILITY_SECONDS_PER_CORE`` of wall);
        with --completion-time what is left of it, start-ups in."""

        if completion is not None:
            return completion - monitor.elapsed()
        runtime_budget = (
            (
                float(bench.target_runtime) * count
                if target_given
                else DEFAULT_RUNTIME_SECONDS
            )
            if want_runtime
            else 0.0
        )
        stability_budget = DEFAULT_STABILITY_SECONDS_PER_CORE if want_stability else 0.0
        return BUDGET_TOLERANCE * (runtime_budget + stability_budget)

    def running() -> dict[str, Known]:
        """``known`` of the backends that still run: rule A's left-out ones
        (``early``, ``omitted``) are not judged by rule B."""

        return {b: k for b, k in known.items() if b not in early and b not in omitted}

    # The measured precisions in the scope rules' lower bounds: binary64's
    # calls when listed, every other precision's stream but a listed ground
    # truth's (its cost is the reference's, counted as such).
    streams = {
        "binary64": binary64_listed,
        "extra": sum(1 for m in extra_precisions if m is not truth_listed),
    }

    def b_drops() -> bool:
        """Rule B on what is known now: every running serving backend's
        lower bound (one not measured yet: 0) summed."""

        return rule_b and rule_b_drops(
            running(),
            budget_bound,
            lanes=DEFAULT_LANES,
            start=completion is not None,
            **streams,
        )

    def a_drops() -> bool:
        """Rule A on what is known now."""

        return (
            rules
            and not asked_backends
            and rule_a_drops(
                known,
                default,
                budget_bound,
                lanes=DEFAULT_LANES,
                reference=a_reference,
                stability_workers=workers if want_stability else None,
                start=completion is not None,
                runtime=want_runtime,
                **streams,
            )
        )

    def learn(backend: str, **numbers: float | bool | None) -> None:
        """A measured number of ``backend`` arrives (``Known``'s fields)."""

        if backend in known:
            known[backend] = replace(known[backend], **numbers)

    def entries(
        backend: str, binary64: float, reference: float | None
    ) -> dict[str, bool]:
        """Whether ``backend``'s streams warm through the entry, as its
        runtime timing will decide (``warms_through_entry``): their untimed
        points then never run, the scope rules count the entry's measured
        seconds instead (m-4)."""

        return {
            "warm_entry": warms_through_entry(
                backend,
                "binary64",
                warmup_seconds(DEFAULT_LANES * binary64, binary64),
            ),
            "reference_entry": warms_through_entry(backend, "reference", reference),
        }

    def enough(found: Mapping[str, Any]) -> bool:
        """After each backend of the measurement (the default first): what
        it measured is known; rule A decided by it leaves the others out,
        unmeasured."""

        for backend, cost in found.items():
            learn(
                backend,
                point=cost.binary64,
                reference=None
                if cost.reference is None or cost.reference_estimated
                else cost.reference,
                **entries(backend, cost.binary64, cost.reference),
            )
        if set(found) == {default} and a_drops():
            early.extend(b for b in backends if b != default)
            return True
        return False

    # Under the default budget a start-up that takes a backend's share
    # leaves every backend the minimum counts whatever it costs: nothing is
    # measured here (no evaluation in this process); each runtime timing's
    # warm-ups measure its costs, the stability run's priors after it.
    blind = (
        want_runtime
        and completion is None
        and not runtime_given
        and not target_given
        and not measurement_affects_counts(
            load_seconds + 1.0, backends=len(_serving(runtime, backends))
        )
    )

    def reference_needed(backend: str, binary64: float, count: int) -> bool:
        """Whether the measurement times ``backend``'s reference: its cost
        can change the counts (rule 1; a given --target-runtime puts the
        reference on top: only the stability run's priors need it), and
        rule B, on what is known with this binary64 point, has not left it
        out (m-3: the backends measured before count)."""

        learn(backend, point=binary64, **entries(backend, binary64, None))
        if b_drops():
            return False
        if target_given:
            return want_stability
        return reference_affects_counts(
            binary64,
            load_seconds + 1.0,
            backends=count,
            lanes=DEFAULT_LANES,
            warm=with_warming(
                backend, Costs(binary64, None, 0.0, None, 0.0)
            ).binary64_warm,
        )

    try:
        costs = (
            {}
            if blind
            else _measure(
                runtime,
                path,
                physics,
                config,
                backends,
                truth,
                monitor,
                load_seconds,
                history=history,
                provisional=provisional,
                # Under the default budget a reference cost that cannot change
                # the runtime counts is not timed here: each backend's runtime
                # timing measures it with its own reference warm-up.
                reference_needed=(
                    reference_needed
                    if want_runtime and completion is None and not runtime_given
                    else None
                ),
                enough=enough,
                excess=excess,
            )
        )
    except KeyboardInterrupt:
        monitor.close(False)
        raise
    measured_at = monitor.elapsed()
    costs = {b: with_warming(b, c) for b, c in costs.items()}

    def cap_workers(profiles: Sequence[Any]) -> None:
        """The stability workers the memory holds, for the plan, its ETA and
        rule A's stability minimum (``_stability`` applies the same rule
        when it starts, ``_memory_record``); a --n_core given stays."""

        nonlocal workers
        from pyamplicol.profile_plan import memory_workers

        if not want_stability or not is_default(bench.n_core):
            return
        estimate = _worker_memory(
            load_peak, _own_peak_bytes() if costs and not blind else None, profiles
        )
        available = _available_memory()
        if estimate and available is not None:
            workers = min(workers, memory_workers(estimate, available))

    cap_workers(())
    priors: dict[str, Any] = {}
    # The minimum runtime timing, as the notices name it (the reference
    # point only when no scope rule can leave it out: given, or with
    # --stability, or counts given).
    minimum_calls = "one timed call of one chunk per backend" + (
        " and its reference point"
        if truth.present
        and (
            not is_default(bench.ground_truth)
            or want_stability
            or (runtime_given and completion is None)
        )
        else ", its reference point unless the minimums of every backend with "
        "theirs exceed the budget"
        if truth.present
        else ""
    )
    if blind:
        # The minimum counts; a previous profile's costs only for the ETA.
        priors = {
            backend: with_warming(
                backend,
                Costs(
                    binary64,
                    history.cost(backend, "reference") if truth.present else None,
                    0.0,
                    None,
                    load_seconds + 1.0,
                ),
            )
            for backend in backends
            if history is not None
            and (binary64 := history.cost(backend, "binary64")) is not None
        }
        planned = Plan(
            "time budgets",
            {
                backend: minimum_counts(
                    load_seconds + 1.0,
                    lanes=DEFAULT_LANES,
                    reference=truth.present,
                    costs=priors.get(backend),
                )
                for backend in backends
            },
        )
        stability_start = load_seconds + 1.0
        # Start-ups are outside the budgets: whether the minimums exceed it
        # is known once the runtime timing measured them (the counts row).
        share = DEFAULT_RUNTIME_SECONDS / max(len(_serving(runtime, backends)), 1)
        _notice(
            "starting a process that loads the artifact takes about "
            f"{_duration(load_seconds + 1.0)}, at least a backend's share "
            f"({share:g} s of the {DEFAULT_RUNTIME_SECONDS:g} s runtime "
            f"budget): running the minimums ({minimum_calls}), their costs "
            "measured as they run"
        )
    else:
        planned, stability_start = planning(costs, measured_at)
    # The scope rules before the runtime timing, on what the measurement
    # measured (nothing without a measurement phase); a stream's entry
    # cost is a previous profile's, not a bound: its row measures it (m-4).
    for backend, cost in costs.items():
        learn(backend, **entries(backend, cost.binary64, cost.reference))
    omitted = (
        [b for b in backends if b != default]
        if (bench.all_backends is False and default in backends) or early or a_drops()
        else []
    )
    timed_reference = truth.present and not b_drops()
    # The backend whose runtime timing decides rule B on its own measured
    # numbers when it is still open (all multipliers or none).
    reference_decider = next(
        (b for b in backends if b in serving and b not in omitted), None
    )

    def omit_notice(measured: bool = False) -> None:
        from pyamplicol.loops.v2.reduction import NAMES

        _notice(
            f"the default backend {NAMES.get(default, default)} alone exceeds "
            "the budget at its minimal counts"
            + (" (measured in its runtime timing)" if measured else "")
            + ": "
            + ", ".join(NAMES.get(b, b) for b in omitted)
            + " not profiled (--all-backends profiles every backend)"
        )

    if truth.present and not timed_reference:
        _notice(
            f"the runtime minimums of every backend with their {truth.name} "
            f"reference exceed the budget: no {truth.label} multiplier (an "
            "explicit --ground-truth measures it)"
        )
    if omitted and bench.all_backends is None:
        omit_notice()
    if costs and (omitted or not timed_reference):
        # The plan of what runs, no reference; the default backend alone
        # keeps the share it has beside the others (its row does not depend
        # on which other rows are shown).
        costs = {
            b: c
            if timed_reference
            else replace(
                c, reference=None, stability_reference=None, reference_estimated=False
            )
            for b, c in costs.items()
        }
        if not blind:
            # The shares of every serving backend: one left out unmeasured
            # (the measurement stopped) shares as the default does.
            shares = dict(costs)
            if default in costs:
                for backend in omitted:
                    if backend in serving:
                        shares.setdefault(backend, costs[default])
            planned, stability_start = planning(shares, measured_at)
            planned = replace(
                planned,
                runtime={b: c for b, c in planned.runtime.items() if b not in omitted},
            )
        minimum_calls = "one timed call of one chunk per backend" + (
            " and its reference point" if timed_reference else ""
        )
    if not planned.feasible:
        # One number, the header's too: the whole command with the minimums.
        if completion is not None:
            asked = f"--completion-time {_duration(completion)}"
        else:
            budgets = {
                "runtime": (
                    f"{_per_point_text(float(bench.target_runtime))} per backend "
                    "for the runtime timing (--target-runtime)"
                    if runtime_given or runtime_share is not None
                    else f"{DEFAULT_RUNTIME_SECONDS:g} s for the runtime timing"
                ),
                "stability": (
                    f"{_duration(DEFAULT_STABILITY_SECONDS_PER_CORE)} per "
                    "stability worker"
                ),
            }
            asked = (
                "the time budget"
                + ("s" if len(planned.unmet) > 1 else "")
                + " ("
                + ", ".join(budgets[part] for part in planned.unmet)
                + ")"
            )
        unmet_stability = completion is not None or "stability" in planned.unmet
        minimums = minimum_calls + (
            ", five stability points per backend"
            if want_stability and unmet_stability
            else ""
        )
        _notice(
            f"{asked} cannot be met on this process: the minimums ({minimums}) "
            f"take about {_duration(planned.minimum_seconds)} in all; running "
            "the minimums",
            warning=True,
        )
    future = future_of(planned, costs)
    for backend in omitted:
        future.pop(f"runtime:{backend}", None)
    if blind:
        # A backend without a previous profile's costs has no estimate yet.
        for backend in backends:
            if backend not in priors:
                future.pop(f"runtime:{backend}", None)
    if completion is None and (runtime_given or given_points is not None):
        # The counts given make the run as long as they are: said once, as
        # information (the user's own choice), when that is long.
        chosen = (
            sum(v for k, v in future.items() if k.startswith("runtime:"))
            if runtime_given
            else 0.0
        ) + (future.get("stability", 0.0) if given_points is not None else 0.0)
        defaults = (DEFAULT_RUNTIME_SECONDS if runtime_given else 0.0) + (
            stability_start + DEFAULT_STABILITY_SECONDS_PER_CORE
            if given_points is not None
            else 0.0
        )
        if chosen > BUDGET_TOLERANCE * defaults:
            _notice(
                "the counts given take about "
                f"{_duration(monitor.elapsed() + sum(future.values()))} in all "
                "(estimated)"
            )
    # The stability run evaluates in this process when it has one worker:
    # ``--n_core`` 1 (or one usable core), or, without a deadline (the points
    # given, no --completion-time), when its planned tasks are fewer than the
    # workers (``_stability`` lowers its count to them; a single stream of
    # up to a chunk of points is one task).
    in_process = want_stability and (
        workers == 1 or (stability_given is not None and completion is None)
    )
    if not in_process:
        # Nothing evaluates in this process any more (the runtime timings and
        # the stability workers load their own): its runtime is released and
        # the allocator's free memory given back while they run (8 gluons:
        # 23-38 GiB beside each worker's own).
        runtime = None
        _release_memory()
    current: list[Any] = [None]

    def remaining() -> tuple[float, bool] | None:
        now = current[0].remaining() if current[0] is not None else None
        rest = sum(future.values())
        if now is None and not rest:
            return None
        return (now or 0.0) + rest, True

    monitor.remaining = remaining

    runtime_rows = None
    # Every other measured precision's runtime rows (``--measured-precisions``).
    runtime_measured: list[dict[str, Any]] = []
    if want_runtime:
        # The caller's points, cycled; else the worker draws distinct flat
        # points as its calls need them (the same per call on every backend).
        timed = list(points) if points else None
        rows = []
        # Each backend's binary64 calls (its counts and budget): a measured
        # precision's calls start from them (``_measured_runtime``).
        benches: dict[str, Any] = {}
        default = DEFAULTS.get(contribution)
        for position, backend in enumerate(backends):
            if backend in omitted:
                continue
            # The optimistic pass: the backend alone. The checked mode: the
            # default backend as "auto" (with its per-point fallback), unless
            # one backend was asked for.
            request = (
                "auto"
                if checked and evaluation.reduction is None and backend == default
                else backend
            )
            backend_bench = bench
            reference_budget = float(bench.target_runtime)
            counts = planned.runtime.get(backend)
            if completion is not None and backend in costs and planned.feasible:
                # Re-planned from what is left: the backends to come share the
                # runtime's part of it (the measured precisions' calls after
                # them kept at their estimates).
                left = (
                    completion
                    - monitor.elapsed()
                    - future.get("stability", 0.0)
                    - sum(
                        seconds
                        for key, seconds in future.items()
                        if key.count(":") == 2
                    )
                )
                share = left / max(len(backends) - position, 1)
                if threads > 1:
                    # The one-thread timing's reload, warm-up and minimum
                    # call (at the measurement's one-thread cost).
                    share -= costs[backend].start + 2 * DEFAULT_LANES * (
                        costs[backend].binary64
                    )
                counts = runtime_counts(
                    costs[backend],
                    max(share - costs[backend].start, 0.0),
                    lanes=DEFAULT_LANES,
                    batch_max=batch_max,
                )
            if counts is not None:
                backend_bench = replace(
                    bench,
                    batch_size=counts.batch,
                    warmup_runs=counts.warmup,
                    minimum_samples=counts.minimum,
                    target_runtime=max(counts.budget, 1e-9),
                )
                reference_budget = counts.reference_budget
            one_thread = None
            if threads > 1:
                # A share of the binary64 budget for the one-thread timing of
                # the same calls (the gain), at least one call.
                one_thread = ONE_THREAD_SHARE * float(backend_bench.target_runtime)
                backend_bench = replace(
                    backend_bench,
                    target_runtime=max(
                        float(backend_bench.target_runtime) - one_thread, 1e-9
                    ),
                )
            benches[backend] = backend_bench
            future.pop(f"runtime:{backend}", None)
            if not binary64_timed:
                # binary64 not listed and not needed: its counts only (the
                # measured precisions start from them).
                continue
            # The planned seconds of this backend's phases after its
            # start-up: the root ETA counts them until they run.
            later: dict[str, float] = {}
            if backend in costs:
                cost = costs[backend]
                call = backend_bench.batch_size * cost.binary64
                warmups = int(backend_bench.warmup_runs)
                later["binary64"] = (
                    (binary64_warmup(cost, call) + (warmups - 1) * call)
                    if warmups
                    else 0.0
                ) + max(
                    float(backend_bench.target_runtime),
                    max(int(backend_bench.minimum_samples), 1) * call,
                )
                if timed_reference and cost.reference is not None:
                    later["reference"] = reference_warmup(cost) + max(
                        reference_budget, cost.reference
                    )
                if one_thread is not None:
                    # Its reload, one-point warm-up and calls (one-thread costs:
                    # the measurement's).
                    later["one-thread"] = (
                        cost.start
                        + warmup_seconds(call, cost.binary64)
                        + max(one_thread, call)
                    )
            run = _BackendRun(
                monitor,
                backend,
                truth,
                None if backend not in costs else costs[backend].start,
                (
                    costs[backend].binary64 * backend_bench.batch_size
                    if backend in costs
                    else None
                ),
                backend_bench.batch_size,
                later,
                threads,
            )
            current[0] = run
            reference = (
                None
                if not timed_reference or truth_listed is not None
                else {
                    **truth.options,
                    "reduction": backend,
                    "budget": max(reference_budget, 1e-9),
                    # Rule B still open: the decider's worker re-calls it on
                    # its own measured numbers (``benchmarking._scope_drops``)
                    # after its binary64 call and after its untimed point.
                    **(
                        {
                            "scope": {
                                # The bound less the others' known minimums.
                                "bound": rule_b_bound(
                                    running(),
                                    backend,
                                    budget_bound,
                                    lanes=DEFAULT_LANES,
                                    start=completion is not None,
                                    **streams,
                                ),
                                "known": {
                                    name: getattr(known[backend], name)
                                    for name in (
                                        "point",
                                        "reference",
                                        "start",
                                        "reference_entry",
                                    )
                                },
                            }
                        }
                        if rule_b and backend == reference_decider
                        else {}
                    ),
                    # Warmed through the entry (``takes_entry``; unknown
                    # cost: the entry).
                    "entry": warms_through_entry(
                        backend,
                        "reference",
                        costs[backend].reference
                        if backend in costs
                        else priors[backend].reference
                        if backend in priors
                        else None,
                    ),
                    **(
                        {"seconds_per_point": costs[backend].reference}
                        if backend in costs and costs[backend].reference is not None
                        else {}
                    ),
                }
            )
            try:
                result = _isolated_profile(
                    path,
                    evaluation.process,
                    options,
                    replace(config, benchmark=backend_bench),
                    timed,
                    request,
                    run,
                    reference=reference,
                    one_thread=one_thread,
                    # Whether the warm entry replaces the untimed binary64
                    # warm-up (``takes_entry`` from its cost: the
                    # measurement's, else a previous profile's, else
                    # unknown).
                    warm_entry=warms_through_entry(
                        backend,
                        "binary64",
                        None
                        if (prior := costs.get(backend) or priors.get(backend)) is None
                        else warmup_seconds(
                            backend_bench.batch_size * prior.binary64,
                            prior.binary64,
                        ),
                    ),
                )
            except ReductionUnavailable as error:
                run.abort()
                rows.append(
                    BackendRuntime(backend, None, str(error), requested=request)
                )
                continue
            except _Interrupted as stopped:
                # Ctrl-C after its binary64 calls: what is complete is shown.
                rows.append(BackendRuntime(backend, stopped.profile, requested=request))
                interrupted = True
                break
            except KeyboardInterrupt:
                # The backends timed so far are reported.
                run.abort()
                interrupted = True
                break
            finally:
                current[0] = None
            run.finish(result)
            rows.append(BackendRuntime(backend, result, requested=request))
            # What this row measured is known (``Known``).
            reference = result.reference
            # The streams' warming as it ran: the entry's seconds, or the
            # untimed reference point (m-4).
            warmed = (
                {"reference_entry": False}
                if result.reference_point_seconds is not None
                else {
                    "reference_entry": True,
                    "reference_warm": reference.prepared_seconds,
                }
                if reference is not None and reference.prepared_seconds is not None
                else {}
            )
            learn(
                backend,
                chunk=result.seconds_per_point if _served_points(result) else None,
                warm=result.prepared_seconds,
                warm_entry=result.prepared_seconds is not None,
                reference=result.reference_point_seconds
                if result.reference_point_seconds is not None
                else known.get(backend, Known()).reference,
                reference_timed=None
                if reference is None
                else reference.seconds_per_point,
                **warmed,
            )
            if backend == reference_decider and result.reference_skipped:
                from pyamplicol.loops.v2.reduction import NAMES

                # Rule B on this backend's measured numbers: no reference
                # for any backend (all or none).
                timed_reference = False
                _notice(
                    "the runtime minimums of every backend with their "
                    f"{truth.name} reference exceed the budget (measured in the "
                    f"runtime timing of {NAMES.get(backend, backend)}, "
                    + (
                        f"after one untimed {truth.label} point"
                        if result.reference_point_seconds is not None
                        else f"before its {truth.label} points"
                    )
                    + f"): no {truth.label} multiplier (an explicit "
                    "--ground-truth measures it)"
                )
            if (
                backend == default
                and not omitted
                and len(serving) > 1
                and _served_points(result)
                and a_drops()
            ):
                # Rule A on the default backend's measured row, before any
                # other backend starts (the ones after it read it here).
                omitted = [b for b in backends if b != default]
                omit_notice(measured=True)
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
        # Every other measured precision's runtime timing, after binary64's
        # (``--measured-precisions``): the same optimistic pass per backend
        # in its arithmetic, on binary64's counts and budget (its batch
        # fitted to its cost, ``benchmarking._fitted_calls``, unless a
        # --batch-size is given), the ground truth's multiplier against it
        # (none for the ground truth's own precision).
        if not interrupted:
            # The ground truth's own stream first: its timing is the truth's
            # (the multiplier of binary64's rows and of the others).
            truths = {
                row.backend: row.profile.reference
                for row in runtime_rows
                if row.profile is not None
            }
            timed_rows: dict[str, tuple[BackendRuntime, ...]] = {}

            def measured_bench(backend: str, bench_p: Any, label: str) -> Any:
                """A measured precision's calls of ``backend``: binary64's,
                with --completion-time what is left of it shared by the
                measured precisions' calls still to run (the stability
                run's planned seconds kept); its ETA entry leaves."""

                future.pop(f"runtime:{backend}:{label}", None)
                if completion is None:
                    return bench_p
                rest = [k for k in future if k.startswith("runtime:")]
                left = (
                    completion
                    - monitor.elapsed()
                    - future.get("stability", 0.0)
                    - sum(future[k] for k in rest)
                )
                start = costs[backend].start if backend in costs else 0.0
                return replace(bench_p, target_runtime=max(left - start, 1e-9))

            for measured in sorted(
                extra_precisions, key=lambda m: m is not truth_listed
            ):

                def timed_profile(
                    backend: str,
                    bench_p: Any,
                    precision: Mapping[str, Any],
                    measured: MeasuredPrecision = measured,
                ) -> Any:
                    label, name = measured.label, measured.name
                    return _isolated_profile(
                        path,
                        evaluation.process,
                        options,
                        replace(
                            config,
                            benchmark=measured_bench(backend, bench_p, label),
                        ),
                        timed,
                        backend,
                        _BackendRun(
                            monitor,
                            backend,
                            truth,
                            None if backend not in costs else costs[backend].start,
                            None,
                            bench_p.batch_size,
                            {},
                            threads,
                            precision=name,
                        ),
                        precision=precision,
                        warm_entry=warms_through_entry(backend, "binary64", None),
                    )

                measured_rows = _measured_runtime(
                    measured,
                    [b for b in backends if b not in omitted and b in benches],
                    benches,
                    truths,
                    truth,
                    timed_profile,
                    fit=is_default(bench.batch_size),
                )
                timed_rows[measured.stream] = measured_rows
                if measured is truth_listed:
                    truths = {
                        row.backend: _as_reference(row.profile, measured)
                        for row in measured_rows
                        if row.profile is not None
                    }
                    # binary64's multiplier from it (timed once).
                    runtime_rows = tuple(
                        row
                        if row.profile is None or row.backend not in truths
                        else replace(
                            row,
                            profile=replace(row.profile, reference=truths[row.backend]),
                        )
                        for row in runtime_rows
                    )
            # In the order given, binary64's rows in ``runtime`` (its entry
            # without rows); none with the default.
            if extra_precisions:
                runtime_measured.extend(
                    {
                        "precision": m.record(),
                        "runtime": None
                        if m.stream == "binary64"
                        else timed_rows[m.stream],
                    }
                    for m in listed
                    if m.stream == "binary64" or m.stream in timed_rows
                )
        # A reference cost the measurement left estimated: the runtime
        # timing's measured one from now on (the stability run's priors and
        # plan, the history).
        timed_references = {
            row.backend: row.profile.reference.seconds_per_point
            for row in rows
            if row.profile is not None
            and row.profile.reference is not None
            and row.backend in costs
            and costs[row.backend].reference_estimated
        }
        for backend, seconds in timed_references.items():
            costs[backend] = replace(
                costs[backend],
                reference=seconds,
                stability_reference=seconds,
                reference_estimated=False,
            )
        # The record, the counts row, the stability plan and the history
        # from what ran: every runtime row's measured costs (its chunk's
        # binary64 cost per point, its reference point, the entry where it
        # warmed), the backends left out or refused dropped; the counts
        # that ran stay the record's, their expected seconds re-planned.
        ran: dict[str, Any] = {}
        for row in rows:
            profile = row.profile
            if profile is None or not _served_points(profile):
                continue
            reference = profile.reference
            seconds = None if reference is None else reference.seconds_per_point
            ran[row.backend] = Costs(
                profile.seconds_per_point,
                seconds,
                STABILITY_PRIOR * profile.seconds_per_point,
                seconds,
                load_seconds + 1.0,
                binary64_warm=profile.prepared_seconds,
                reference_warm=None
                if reference is None
                else reference.prepared_seconds,
            )
        if ran:
            costs = ran
            cap_workers(
                [
                    row.profile
                    for row in rows
                    if row.profile is not None and row.backend not in omitted
                ]
            )
            replanned, stability_start = planning(costs, measured_at)
            planned = replace(
                replanned,
                runtime={
                    backend: (
                        replace(counts, expected=replanned.runtime[backend].expected)
                        if backend in replanned.runtime
                        else counts
                    )
                    for backend, counts in planned.runtime.items()
                    if backend not in omitted
                },
            )

    stability_rows = None
    stability_setup = None
    if want_stability and not interrupted:
        from pyamplicol.profile_plan import (
            DEFAULT_STABILITY_SECONDS_PER_CORE,
            stability_points,
        )

        future.pop("stability", None)
        # The stability run's points and deadline: what is left of
        # --completion-time, the points given, else a minute per worker.
        deadline = None
        count = planned.stability_points or STABILITY_FALLBACK
        if completion is not None:
            left = completion - monitor.elapsed()
            deadline = time.perf_counter() + max(left, 0.0)
            count = stability_points(
                [with_measured(c) for c in costs.values()],
                max(left - stability_start, 0.0),
                workers,
                lanes=DEFAULT_LANES,
                tail=tail,
            )
        elif stability_given is None:
            deadline = (
                time.perf_counter()
                + stability_start
                + DEFAULT_STABILITY_SECONDS_PER_CORE
            )
        stability_rows, stability_setup = _stability(
            runtime,
            path,
            physics,
            config,
            tuple(b for b in backends if b not in omitted),
            load_options=options,
            progress=progress,
            monitor=monitor,
            # A given count only without --completion-time (it overrules
            # them: ``effective_benchmark``).
            planned_points=count if given_points is None else given_points,
            deadline=deadline if given_points is None else None,
            costs=costs,
            current=current,
            worker_bytes=_worker_memory(
                load_peak,
                _own_peak_bytes() if costs and not blind else None,
                [
                    row.profile
                    for row in runtime_rows or ()
                    if row.profile is not None and row.backend not in omitted
                ],
            ),
            records=records,
            tail_seconds=tail if completion is not None else 0.0,
        )
        if records and bench.samples_output is not None:
            stability_rows, stability_setup = _write_samples(
                Path(bench.samples_output),
                stability_rows,
                stability_setup,
                monitor,
                physics.process_id,
            )
    causes, totals = _runtime_causes(runtime_rows or (), truth)
    more, more_totals = _stability_causes(stability_rows or ())
    causes += more
    totals.update(more_totals)
    # Every other measured precision's own causes, under its own tables.
    for entry in runtime_measured:
        if entry["runtime"] is not None:
            entry["failure_causes"] = failure_causes(
                *_runtime_causes(entry["runtime"], truth)
            )
    if stability_setup is not None and stability_setup.get("measured_precisions"):
        stability_setup = {
            **stability_setup,
            "measured_precisions": [
                entry
                if entry.get("stability") is None
                else {
                    **entry,
                    "failure_causes": failure_causes(
                        *_stability_causes(entry["stability"])
                    ),
                }
                for entry in stability_setup["measured_precisions"]
            ],
        }
    complete = not (
        interrupted
        or bool(stability_setup is not None and stability_setup.get("interrupted"))
    )
    if complete and history is not None:
        # The next profile's priors and whole estimate.
        history.store(
            backends,
            # The costs this run measured (without a measurement phase, or
            # with an estimated reference: the runtime timing's), the next
            # profile's priors.
            costs,
            {"load": load_seconds, settings: monitor.elapsed()},
        )
    monitor.close(
        not (
            interrupted
            or bool(stability_setup is not None and stability_setup.get("interrupted"))
        )
    )
    return LoopProfileReport(
        physics.process_id,
        physics.process,
        contribution,
        DEFAULTS.get(contribution),
        tuple(b for b in backends if b not in omitted),
        _runtime_setup(
            checked,
            None if points else evaluation.phase_space.seed,
            [
                row.profile.threads
                for row in runtime_rows or ()
                if row.profile is not None and row.profile.threads is not None
            ],
            runtime_rows or (),
            truth,
            evaluation.lane_rescue,
            evaluation.phase_space.cuts,
        )
        if want_runtime
        else None,
        runtime_rows,
        stability_setup,
        stability_rows,
        failure_causes(causes, totals),
        interrupted
        or bool(stability_setup is not None and stability_setup.get("interrupted")),
        truth.record(),
        {
            **planned.record(),
            "overruled": overruled if completion is not None else [],
            # Which counts were given (the rest from the time budgets).
            "given": {
                "runtime": bool(want_runtime and runtime_given and completion is None),
                "stability": bool(
                    want_stability
                    and stability_given is not None
                    and completion is None
                ),
            },
            # A --target-runtime given alone: the runtime budget per backend.
            "runtime_target": runtime_share if not runtime_given else None,
            # What the runtime timing left out on an expensive process: the
            # backends (rule A) and the reference multiplier (rule B).
            "omitted_backends": list(omitted),
            "omitted_reference": (
                truth.label
                if truth.present
                and want_runtime
                and (
                    not timed_reference
                    or any(
                        getattr(row.profile, "reference_skipped", False)
                        for row in runtime_rows or ()
                    )
                )
                else None
            ),
            "runtime_counts": {
                row.backend: {
                    "batch": row.profile.batch,
                    "calls": row.profile.samples,
                    "reference_points": (
                        None
                        if row.profile.reference is None
                        else row.profile.reference.points
                    ),
                }
                for row in runtime_rows or ()
                if row.profile is not None
            },
        },
        monitor.elapsed(),
        tuple(runtime_measured),
    )


class _Phase:
    """One phase of the live display (``_Monitor.phase``): a child task of
    the profile's root task with its counts, its ETA (counted down by the
    display between updates) and, when it ends, one scroll-back line with
    its summary."""

    def __init__(
        self,
        monitor: _Monitor,
        key: str,
        description: str,
        total: int | None,
        unit: str,
    ) -> None:
        from pyamplicol.reporting.progress import ProgressStart

        self.monitor = monitor
        self.task = f"loop-profile-{key}"
        self.begin = time.perf_counter()
        self.done = 0
        self.total = total
        self.ended = False
        if monitor.sink is not None:
            monitor.sink.emit(
                ProgressStart(
                    self.task,
                    description,
                    total=total,
                    parent_task_id=monitor.task,
                    unit=unit,
                    details={"eta_seconds": None, "readings": monitor.readings()},
                )
            )

    def update(
        self,
        done: int,
        total: int | None,
        *,
        eta: float | None,
        step: str,
        estimated: bool = False,
        heartbeat: bool = False,
        stalled: float | None = None,
        running: float | None = None,
    ) -> None:
        from pyamplicol.reporting.progress import ProgressUpdate

        self.done, self.total = done, total
        if self.monitor.sink is None or self.ended:
            return
        total = None if total is None else max(total, done)
        details: dict[str, Any] = {
            "step": step,
            "eta_seconds": None if eta is None else max(float(eta), 0.0),
            "eta_estimated": estimated,
            "readings": self.monitor.readings(),
            "heartbeat": heartbeat,
        }
        if stalled is not None:
            details["stalled_seconds"] = max(float(stalled), 0.0)
        if running is not None:
            details["running_seconds"] = max(float(running), 0.0)
        self.monitor.sink.emit(ProgressUpdate(self.task, done, total, details=details))

    def end(self, summary: str | None, *, success: bool = True) -> None:
        from pyamplicol.reporting.progress import ProgressEnd

        if self.ended:
            return
        self.ended = True
        if self.monitor.sink is not None:
            self.monitor.sink.emit(
                ProgressEnd(
                    self.task,
                    success=success,
                    message=summary,
                    elapsed_seconds=time.perf_counter() - self.begin,
                    details={"keep": True, "readings": self.monitor.readings()},
                )
            )


class _Monitor:
    """The live display of a loop profile (``--progress``): a root task with
    the whole command's elapsed time, ETA and budget (``remaining``: a
    callable of the seconds left, the plan's and the measured rates'
    estimate), its phases as child tasks (``_Phase``), and the readings every
    phase shows: per backend its current best estimate of the binary64 time
    per point with the running reference multiplier, or the stability run's
    points done and flagged (``set_reading``)."""

    def __init__(self, sink: Any, title: str, budget: float | None) -> None:
        from pyamplicol.reporting.progress import ProgressStart

        self.sink = sink
        self.task = "loop-profile"
        # The command's clock (``pyamplicol._clock``: the interpreter's start
        # and the imports count, the user waits for them too; an in-process
        # call from its entry), else this profile's start.
        from pyamplicol._clock import command_start

        started = command_start()
        self.begin = time.perf_counter() if started is None else started
        self.budget = budget
        self._readings: dict[str, tuple[str | None, str | None]] = {}
        self.remaining: Any = None
        self._count = 0
        if sink is not None:
            sink.emit(
                ProgressStart(
                    self.task,
                    title,
                    details={"budget_seconds": budget} if budget is not None else {},
                )
            )

    def elapsed(self) -> float:
        return time.perf_counter() - self.begin

    def set_reading(
        self, label: str, value: str | None, note: str | None = None
    ) -> None:
        self._readings[label] = (value, note)

    def clear_readings(self) -> None:
        self._readings.clear()

    def readings(self) -> str:
        from pyamplicol.reporting.progress import readings_detail

        return readings_detail(
            [(label, value, note) for label, (value, note) in self._readings.items()]
        )

    def phase(
        self, key: str, description: str, total: int | None = None, unit: str = "calls"
    ) -> _Phase:
        self._count += 1
        return _Phase(self, f"{self._count}-{key}", description, total, unit)

    def tick(self) -> None:
        """The root task's ETA and budget, from ``remaining``."""

        from pyamplicol.reporting.progress import ProgressUpdate

        if self.sink is None:
            return
        details: dict[str, Any] = {"heartbeat": True}
        remaining = self.remaining() if self.remaining is not None else None
        if remaining is not None:
            seconds, estimated = remaining
            details["eta_seconds"] = max(float(seconds), 0.0)
            details["eta_estimated"] = bool(estimated)
        if self.budget is not None:
            details["budget_seconds"] = self.budget - self.elapsed()
        self.sink.emit(ProgressUpdate(self.task, 0, None, details=details))

    def close(self, success: bool) -> None:
        from pyamplicol.reporting.progress import ProgressEnd

        if self.sink is not None:
            self.sink.emit(
                ProgressEnd(
                    self.task,
                    success=success,
                    message=None if success else "interrupted",
                    elapsed_seconds=self.elapsed(),
                )
            )


def _serving(runtime: Any, backends: Sequence[str]) -> list[str]:
    """The ``backends`` this loaded runtime can serve, read without an
    evaluation from its reduction record: those the artifact admits
    (``admitted``: a TIR-only artifact refuses NinjaRS) and the native build
    has (``native``); all of them when the record cannot be read. A backend
    refused only by its first evaluation (a TIRust plan refusal) still
    counts."""

    summary = getattr(getattr(runtime, "_backend", runtime), "_reduction_summary", None)
    try:
        record = summary() if callable(summary) else None
    except Exception:
        record = None
    if not isinstance(record, Mapping):
        return list(backends)
    admitted = record.get("admitted")
    native = record.get("native")
    return [
        backend
        for backend in backends
        if (admitted is None or backend in admitted)
        and (not isinstance(native, Mapping) or backend in native)
    ]


def _own_peak_bytes() -> float | None:
    """This process's peak resident set size (VmHWM; None where unknown)."""

    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return None


def _process_peak_bytes(pid: int | None) -> float | None:
    """Another process's peak resident set size (VmHWM; None unknown)."""

    if pid is None:
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return None


def _available_memory() -> float | None:
    """The memory available to new processes now (MemAvailable, bytes; None
    where unknown)."""

    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    return None


def _worker_memory(
    load_peak: float | None, measured_peak: float | None, profiles: Sequence[Any]
) -> float | None:
    """One stability worker's resident size, loaded and warm, as this run
    measured it: the parent's own peak when it loaded the artifact and
    measured every backend (``measured_peak``), and the load's peak plus
    each profiled backend's increase over it (its runtime worker's binary64
    and reference peaks, ``profiles``): the larger; None when nothing was
    measured."""

    candidates = [] if measured_peak is None else [measured_peak]
    peaks = []
    for profile in profiles:
        found = [
            float(memory.peak_rss_bytes)
            for memory in (
                profile.peak_memory,
                getattr(profile.reference, "peak_memory", None),
            )
            if memory is not None and memory.peak_rss_bytes is not None
        ]
        if found:
            peaks.append(max(found))
    if peaks:
        base = load_peak or 0.0
        candidates.append(base + sum(max(peak - base, 0.0) for peak in peaks))
    return max(candidates) if candidates else None


def _memory_record(
    workers: int, worker_bytes: float | None, bench: Any
) -> dict[str, Any] | None:
    """The stability workers' memory: one worker's measured bytes, the
    memory available now and how many workers fit in ``MEMORY_SHARE`` of it
    (``profile_plan.memory_workers``); None when either is unknown."""

    from pyamplicol.config.models import is_default
    from pyamplicol.profile_plan import MEMORY_SHARE, memory_workers

    available = _available_memory()
    if not worker_bytes or available is None:
        return None
    return {
        "per_worker_bytes": worker_bytes,
        "available_bytes": available,
        "share": MEMORY_SHARE,
        "fits": memory_workers(worker_bytes, available),
        "asked": workers,
        "n_core_given": not is_default(bench.n_core),
    }


def _gib(value: float) -> str:
    return f"{value / 2**30:.1f} GiB"


def _release_memory() -> None:
    """A full collection, then the C allocator's free memory returned to
    the system (glibc ``malloc_trim``; nothing where it is absent): freed
    native memory otherwise stays in this process's resident set."""

    gc.collect()
    with contextlib.suppress(Exception):
        import ctypes

        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim(0)


def _per_point_text(seconds: float | None) -> str | None:
    """A time per point for the readings ("2.31 ms"), None when unknown."""

    if seconds is None or not seconds > 0.0:
        return None
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def _multiplier_text(ratio: float) -> str:
    return f"x{ratio:.1f}" if round(ratio, 1) < 10.0 else f"x{ratio:.0f}"


# The worker events that carry calls of a phase (``_BackendRun.event``).
_CALL_PHASES = ("binary64", "reference", "one-thread")


class _BackendRun:
    """One backend's runtime timing seen from this process: its worker's
    events (``_runtime_worker``) turned into the live display's phases
    (start-up, binary64 calls, reference calls), its readings, and, between
    events, a heartbeat every second: the running call's time against its
    expected cost, "no progress" once it runs far past it. It keeps the
    binary64 result and the reference calls back so far, so that Ctrl-C
    reports them."""

    def __init__(
        self,
        monitor: _Monitor,
        backend: str,
        truth: GroundTruth,
        expected_start: float | None,
        expected_call: float | None = None,
        batch: int = 1,
        later: Mapping[str, float] | None = None,
        threads: int = 1,
        precision: str = "binary64",
    ) -> None:
        from pyamplicol.loops.v2.reduction import NAMES

        self.monitor = monitor
        # The precision of its timed calls (a measured precision's name).
        self.precision = precision
        # The native threads of the timed calls (``--runtime-threads``) and
        # the one-thread timing's time per point (None: not measured).
        self.threads = threads
        self.one_thread: float | None = None
        # The planned seconds of one binary64 call (the measured cost per
        # point times the batch): the first call's expected cost and ETA
        # until one is back.
        self.expected_call = expected_call
        # The planned seconds of the phases to come ("binary64",
        # "reference"): part of ``remaining`` until each one starts.
        self.later = dict(later or {})
        self.batch = max(int(batch), 1)
        self.name = NAMES.get(backend, backend)
        self.truth = truth
        self.profile: Any = None
        self.reference_walls: list[float] = []
        self.reference_points = 0
        self.binary64: float | None = None
        self.multiplier: float | None = None
        self.expected_start = expected_start
        self.kind = "start"
        self.phase: _Phase | None = monitor.phase(
            f"{backend}-start",
            f"{self.name}: start-up (process, artifact)",
            None,
            "steps",
        )
        self.last = time.perf_counter()
        self.state: dict[str, Any] = {"done": 0, "total": None, "eta": expected_start}
        self.load_seconds: float | None = None
        self.reading()

    def reading(self) -> None:
        note = None
        if self.truth.arithmetic != "none" and self.multiplier is not None:
            note = f"({self.truth.label} {_multiplier_text(self.multiplier)})"
        self.monitor.set_reading(self.name, _per_point_text(self.binary64), note)

    def remaining(self) -> float | None:
        """The seconds this backend has left: the current phase's (counted
        down since its last event, never below a fraction of a call while it
        runs) and the planned ones of the phases still to come."""

        order = ("start", "binary64", "between", "reference", "one-thread")
        position = order.index(self.kind) if self.kind in order else len(order)
        to_come = sum(
            seconds
            for kind, seconds in self.later.items()
            if order.index(kind) > position
        )
        now = self._phase_remaining()
        if now is None:
            return to_come or None
        return now + to_come

    def _phase_remaining(self) -> float | None:
        eta = self.state.get("eta")
        if eta is None:
            return None
        since = time.perf_counter() - self.last
        rate = self.state.get("rate")
        left_calls = max(int(self.state["total"] or 0) - int(self.state["done"]), 1)
        # The running call's expected cost: the recent calls', else the
        # phase's estimate shared by its calls; the calls after it, and at
        # least a tenth of a call while it runs (never 0).
        call = float(rate) if rate else float(eta) / left_calls
        if call <= 0.0:
            return max(float(eta) - since, 0.0)
        return (left_calls - 1) * call + max(call - since, 0.1 * call)

    def event(self, kind: str, payload: Mapping[str, Any]) -> None:
        now = time.perf_counter()
        if kind == "loaded":
            self.load_seconds = float(payload.get("seconds") or 0.0)
            return
        if kind in _CALL_PHASES and kind != self.kind:
            self._next(kind)
        if kind == "reference-result":
            # The reference pass is over; the one-thread timing follows.
            self.finish(payload["profile"])
            return
        if kind == "one-thread-result":
            single = payload["profile"]
            self.one_thread = single.seconds_per_point
            if self.phase is not None:
                gain = (
                    f": x{single.seconds_per_point / self.binary64:.2f} on "
                    f"{self.threads} threads"
                    if self.binary64
                    else ""
                )
                self.phase.end(
                    f"{_per_point_text(single.seconds_per_point)} per point on "
                    f"one thread{gain}"
                )
            self.phase = None
            return
        if kind == "binary64-result":
            self.profile = payload["profile"]
            self.binary64 = self.profile.seconds_per_point
            self.reading()
            if self.phase is not None:
                self.phase.end(
                    f"{_per_point_text(self.binary64)} per point, median of "
                    f"{self.profile.samples} calls of {self.profile.batch} points"
                )
            self.kind = "between"
            self.phase = None
            return
        if kind not in _CALL_PHASES:
            return
        walls = payload.get("walls") or []
        points = int(payload.get("points") or 1)
        done, total = int(payload["done"]), payload.get("total")
        rate = statistics.median(walls[-5:]) if walls else None
        if kind == "binary64" and payload.get("seconds_per_point") is not None:
            self.binary64 = float(payload["seconds_per_point"])
        if kind == "reference":
            self.reference_walls = list(walls)
            self.reference_points = points
            per = payload.get("seconds_per_point")
            if per is not None and self.binary64:
                self.multiplier = float(per) / self.binary64
        self.reading()
        eta = payload.get("eta_seconds")
        if eta is None and kind == "binary64" and self.expected_call:
            eta = max(int(total or 0) - done, 1) * self.expected_call
        self.state = {
            "done": done,
            "total": total,
            "eta": eta,
            "rate": rate,
            "warmup": int(payload.get("warmup") or 0)
            if kind in ("binary64", "one-thread")
            else 0,
        }
        self.last = now
        self._show(heartbeat=False)

    def _next(self, kind: str) -> None:
        if self.phase is not None:
            summary = None
            if self.kind == "start":
                summary = (
                    f"loaded in {_duration(self.load_seconds)}"
                    if self.load_seconds is not None
                    else None
                )
            self.phase.end(summary)
        label = {
            "binary64": f"{self.precision} calls",
            "one-thread": "one-thread calls (the gain)",
        }.get(kind, f"{self.truth.name} reference calls")
        self.kind = kind
        self.phase = self.monitor.phase(f"{self.name}-{kind}", f"{self.name}: {label}")
        self.state = {"done": 0, "total": None, "eta": None, "rate": None}
        self.last = time.perf_counter()

    def _show(self, *, heartbeat: bool) -> None:
        if self.phase is None:
            return
        since = time.perf_counter() - self.last
        done, total = self.state["done"], self.state["total"]
        rate = self.state.get("rate")
        eta = self._phase_remaining()
        if self.kind == "start":
            step = f"{self.name}: starting its process and loading the artifact"
            stalled = (
                since - 3.0 * self.expected_start
                if self.expected_start and since > 3.0 * self.expected_start + 30.0
                else None
            )
            self.phase.update(
                0,
                None,
                eta=eta,
                step=step,
                estimated=True,
                heartbeat=heartbeat,
                stalled=stalled,
                running=since,
            )
            return
        what = {"reference": "reference", "one-thread": "one thread"}.get(
            self.kind, "binary64"
        )
        shown = None if total is None else max(int(total), int(done) + 1)
        running = f"call {done + 1}" + (f"/{shown}" if shown else "")
        # The running call's expected cost: the recent calls', else the
        # phase's ETA shared by the calls left.
        expected = rate
        if not expected and self.state.get("eta") is not None:
            expected = float(self.state["eta"]) / max(int(total or 0) - int(done), 1)
        if int(done) < int(self.state.get("warmup") or 0):
            # The untimed warm-up: one point when a full call is dear
            # (``profile_plan.warmup_seconds``), else one full call.
            from pyamplicol.profile_plan import warmup_seconds

            running = "warm-up call"
            if self.expected_call:
                expected = warmup_seconds(
                    self.expected_call, self.expected_call / self.batch
                )
        step = f"{self.name} {what}: {running} running {_duration(since)}" + (
            f" of ~{_duration(float(expected))}" if expected else ""
        )
        stalled = None
        if expected and since > 3.0 * float(expected) + 30.0:
            stalled = since - float(expected)
        self.phase.update(
            int(done),
            None if total is None else max(int(total), int(done) + 1),
            eta=eta,
            step=step,
            estimated=rate is None,
            heartbeat=heartbeat,
            stalled=stalled,
            running=since,
        )

    def tick(self) -> None:
        self._show(heartbeat=True)

    def finish(self, profile: Any) -> None:
        if self.phase is not None:
            summary = None
            if self.kind == "reference" and profile.reference is not None:
                ratio = profile.reference.seconds_per_point / max(
                    profile.seconds_per_point, 1e-300
                )
                summary = (
                    f"{self.truth.label} {_multiplier_text(ratio)} from "
                    f"{profile.reference.points} points"
                )
            self.phase.end(summary)
            self.phase = None

    def partial(self) -> Any:
        """The binary64 profile with the reference calls back so far (none:
        a reference of no call, its multiplier shown muted), or None before
        the binary64 result."""

        from pyamplicol.benchmarking import PrecisionTiming

        if self.profile is None:
            return None
        if self.truth.arithmetic == "none":
            return self.profile
        walls = self.reference_walls
        size = max(self.reference_points, 1)
        timing = PrecisionTiming(
            self.truth.arithmetic,
            size,
            len(walls),
            1,
            size * len(walls),
            statistics.median(walls) / size if walls else 0.0,
            min(walls) / size if walls else 0.0,
            max(walls) / size if walls else 0.0,
            0,
            digits=self.truth.digits,
        )
        return replace(self.profile, reference=timing)

    def abort(self) -> None:
        if self.phase is not None:
            self.phase.end("interrupted", success=False)
            self.phase = None


def _duration(seconds: float | None) -> str:
    from pyamplicol.profile_plan import duration_text

    return "?" if seconds is None else duration_text(seconds)


def _notice(text: str, *, warning: bool = False) -> None:
    """One plain line for the user (the CLI's short notice; a warning in the
    warning colour)."""

    from pyamplicol.reporting.logging import get_logger, notice

    logger = get_logger("profile")
    if warning:
        logger.warning(text)
    else:
        notice(logger, "profile", text)


class _Interrupted(Exception):
    """Ctrl-C during a backend's runtime timing after its binary64 calls:
    ``profile`` holds them and the reference calls back so far."""

    def __init__(self, profile: Any) -> None:
        super().__init__("interrupted")
        self.profile = profile


def _measure(
    runtime: Any,
    path: Path,
    physics: Any,
    config: RunConfig,
    backends: Sequence[str],
    truth: GroundTruth,
    monitor: _Monitor,
    load_seconds: float,
    *,
    history: _ProfileHistory | None = None,
    provisional: Callable[[dict[str, Any], float], None] | None = None,
    reference_needed: Callable[[str, float, int], bool] | None = None,
    enough: Callable[[Mapping[str, Any]], bool] | None = None,
    excess: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """The measurement phase of a time-budgeted profile, on this process's
    loaded runtime, per backend: the binary64 cost per point of an
    optimistic call and the reference's (none without a ground truth; not
    timed when ``reference_needed(binary64 cost, backends measured)`` is
    False: then its prior, ``Costs.reference_estimated``, and the runtime
    timing's reference warm-up measures it), each
    from one point (twice when it costs at most ``MEASURE_REPEAT_SECONDS``:
    a first call's lazy set-up) and, when a full SIMD chunk costs at most
    ``WARMUP_CALL_SECONDS`` at that price, from one call of a full chunk
    (first calls: upper estimates); the stability run's binary64 cost from
    the prior ``STABILITY_PRIOR``; a process's start-up from this process's
    load. A backend that refuses the process is left out (its timing
    reports the refusal). Every call's expected cost (its ETA) and, after
    every step, ``provisional(costs, seconds)`` (every backend's costs so
    far, the steps left's expected seconds: the whole command's estimate
    long before the phase ends) come from what is known: a previous profile
    of the artifact (``history``), else the steps already measured (the
    other backend's binary64 cost; the reference's from the backend's
    multiplier prior, ``_MULTIPLIER_PRIOR``)."""

    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.benchmarking import REFERENCE_PRIOR, _initial_invariant
    from pyamplicol.cli.flat_points import draw_flat_points
    from pyamplicol.loops.v2.plan_export import DEFAULT_LANES
    from pyamplicol.loops.v2.reduction import NAMES
    from pyamplicol.profile_plan import (
        MEASURE_REPEAT_SECONDS,
        STABILITY_PRIOR,
        WARMUP_CALL_SECONDS,
        Costs,
    )

    space = config.evaluation.phase_space
    points = draw_flat_points(
        path,
        process=physics.process_id,
        points=DEFAULT_LANES,
        seed=space.seed,
        sqrt_s=space.sqrt_s,
        cuts=space.cuts,
    ).as_inputs(exact=False)
    initial = sum(1 for leg in physics.external_particles if leg.state == "incoming")
    scales = (
        config.evaluation.mu_r2
        if config.evaluation.mu_r2 is not None
        else [_initial_invariant(point, initial) for point in points]
    )
    precisions = ["binary64"] + ([] if truth.arithmetic == "none" else ["reference"])
    steps = len(backends) * len(precisions)
    phase = monitor.phase("measure", "Measuring the cost per point", steps)

    def cost(backend: str, precision: str, after: float = 0.0) -> float:
        options: dict[str, Any] = {
            "poles": "none",
            "on_failure": "return",
            "reduction": backend,
        }
        if precision == "binary64":
            options["reduction_unchecked"] = True
        else:
            options.update(truth.options)

        label = "binary64" if precision == "binary64" else truth.name
        name = NAMES.get(backend, backend)

        def seconds(size: int, expected: float | None) -> float:
            batch = points[:size]
            mu = scales[:size] if isinstance(scales, list) else scales
            begin = time.perf_counter()
            _while_ticking(
                lambda: runtime.evaluate(batch, mu_r2=mu, **options),
                lambda since: phase.update(
                    done,
                    steps,
                    eta=None
                    if expected is None
                    else max(expected - since, 0.1 * expected) + after,
                    step=f"{name}: {label} call of {size} point"
                    + ("s" if size > 1 else "")
                    + f" running {_duration(since)}"
                    + ("" if expected is None else f" of ~{_duration(expected)}"),
                    estimated=True,
                    heartbeat=True,
                    running=since,
                ),
                monitor.tick,
            )
            return (time.perf_counter() - begin) / size

        # One point; again when it was cheap (a first call carries seconds of
        # lazy set-up, negligible against a dearer point); one full chunk
        # when that costs at most WARMUP_CALL_SECONDS at the point's price.
        prior = prior_point(backend, precision)
        per_point = seconds(1, prior)
        if per_point <= MEASURE_REPEAT_SECONDS:
            first = per_point
            per_point = seconds(1, per_point)
            if excess is not None:
                # The first call's lazy set-up (``profile_plan.takes_entry``).
                excess[(backend, precision)] = max(first - per_point, 0.0)
        if per_point * len(points) <= WARMUP_CALL_SECONDS:
            per_point = seconds(len(points), per_point * len(points))
        return per_point

    # Every step measured so far, by (backend, precision).
    known: dict[tuple[str, str], float] = {}

    def prior_point(backend: str, precision: str) -> float | None:
        """The expected seconds per point of a step not measured yet."""

        if (backend, precision) in known:
            return known[(backend, precision)]
        if history is not None:
            remembered = history.cost(backend, precision)
            if remembered is not None:
                # The history seeds the priors until this run measures: then
                # it is rescaled by how far off it was on the steps measured
                # (a 100x-stale history showed ~3 h for a 5 min run).
                return remembered * history_ratio()
        if precision == "binary64":
            # The other carrier's binary64 point: the same matrix element.
            others = [v for (_, p), v in known.items() if p == "binary64"]
            return others[0] if others else None
        binary64 = prior_point(backend, "binary64")
        if binary64 is None:
            return None
        factor = (
            _MULTIPLIER_PRIOR.get(backend, REFERENCE_PRIOR["double-double"])
            if truth.arithmetic == "double-double"
            else REFERENCE_PRIOR.get(truth.arithmetic, REFERENCE_PRIOR["arbitrary"])
        )
        return binary64 * factor

    def history_ratio() -> float:
        """This run's measured costs over the history's, the median of the
        steps measured so far (1 before any)."""

        if history is None:
            return 1.0
        ratios = [
            value / remembered
            for (backend, precision), value in known.items()
            if (remembered := history.cost(backend, precision)) is not None
        ]
        return statistics.median(ratios) if ratios else 1.0

    def step_seconds(per_point: float | None) -> float:
        """A step's expected seconds at ``per_point``: its one-point calls
        and, when cheap, its full chunk (``cost``)."""

        if per_point is None:
            return 0.0
        total = per_point * (2.0 if per_point <= MEASURE_REPEAT_SECONDS else 1.0)
        if per_point * len(points) <= WARMUP_CALL_SECONDS:
            total += per_point * len(points)
        return total

    def report(step: int) -> None:
        """``provisional`` after ``step`` steps: the costs known or expected
        of every backend, the steps left's seconds."""

        if provisional is None:
            return
        expected: dict[str, Any] = {}
        for backend in backends:
            binary64 = prior_point(backend, "binary64")
            if binary64 is None:
                return
            reference = (
                prior_point(backend, "reference") if "reference" in precisions else None
            )
            expected[backend] = Costs(
                binary64,
                reference,
                STABILITY_PRIOR * binary64,
                reference,
                load_seconds + 1.0,
            )
        left = sum(
            step_seconds(prior_point(backend, precision))
            for position, (backend, precision) in enumerate(
                (b, p) for b in backends for p in precisions
            )
            if position >= step
        )
        provisional(expected, left)

    costs: dict[str, Any] = {}
    done = 0
    measured: dict[str, float] = {}
    # The backends whose reference cost is a prior (``reference_needed``).
    estimated: set[str] = set()
    report(0)
    for backend in backends:
        name = NAMES.get(backend, backend)
        measured.clear()
        try:
            for precision in precisions:
                if (
                    precision == "reference"
                    and reference_needed is not None
                    and not reference_needed(
                        backend, measured["binary64"], len(backends)
                    )
                ):
                    estimated.add(backend)
                    done += 1
                    report(done)
                    continue
                label = "binary64" if precision == "binary64" else truth.name
                phase.update(
                    done,
                    steps,
                    eta=None,
                    step=f"{name}: {label} cost per point",
                )
                # The steps after this one at their expected cost.
                order = [(b, q) for b in backends for q in precisions]
                after = sum(
                    step_seconds(prior_point(b, q))
                    for b, q in order[order.index((backend, precision)) + 1 :]
                )
                measured[precision] = cost(backend, precision, after)
                known[(backend, precision)] = measured[precision]
                done += 1
                report(done)
                ratio = (
                    measured["reference"] / measured["binary64"]
                    if "reference" in measured
                    else None
                )
                monitor.set_reading(
                    name,
                    _per_point_text(measured["binary64"]),
                    None
                    if ratio is None
                    else f"({truth.label} {_multiplier_text(ratio)})",
                )
        except ReductionUnavailable:
            done = (backends.index(backend) + 1) * len(precisions)
            continue
        binary64 = measured["binary64"]
        reference = measured.get("reference")
        if backend in estimated:
            reference = prior_point(backend, "reference")
        costs[backend] = Costs(
            binary64,
            reference,
            STABILITY_PRIOR * binary64,
            reference,
            load_seconds + 1.0,
            reference_estimated=backend in estimated,
        )
        if enough is not None and enough(costs):
            # The backends after this one are not profiled (``enough``).
            break
    for backend in [b for b in costs if b in estimated]:
        # A backend refused: the others' shares grew; a reference cost that
        # can now change the counts is measured after all.
        if reference_needed is None or reference_needed(
            backend, costs[backend].binary64, len(costs)
        ):
            reference = cost(backend, "reference")
            known[(backend, "reference")] = reference
            costs[backend] = replace(
                costs[backend],
                reference=reference,
                stability_reference=reference,
                reference_estimated=False,
            )
    phase.end(
        ", ".join(
            f"{NAMES.get(b, b)} {_per_point_text(c.binary64)}"
            + (
                ""
                if c.reference is None or c.reference_estimated
                else f" ({truth.label} {_multiplier_text(c.reference / c.binary64)})"
            )
            for b, c in costs.items()
        )
        + " per point"
    )
    monitor.clear_readings()
    return costs


def _settings_key(bench: Any, config: RunConfig) -> str:
    """The settings a whole profile's time depends on (the history's total
    of the previous profile with the same ones)."""

    import hashlib

    parts = (
        bench.completion_time,
        bench.runtime,
        bench.stability,
        *(getattr(bench, name) for _, name in _COUNT_OPTIONS),
        bench.n_core,
        bench.ground_truth,
        bench.checked,
        config.output.format,
        bench.samples_output is not None,
        config.evaluation.reduction,
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


class _ProfileHistory:
    """The costs and times of the previous profile of the same artifact
    (its ``artifact.json``), process, reference precision and host, kept in
    the pyAmpliCol cache with the generation history (``generation.progress.
    GenerationHistory``: bounded, atomic, a corrupt file ignored): the
    priors of the next profile's measurement and its whole estimate from the
    first second. Written only when the profile shows progress, as
    ``generate``'s."""

    def __init__(self, artifact: Path, process: str | None, truth: GroundTruth):
        import hashlib
        import platform

        from pyamplicol.generation.progress import GenerationHistory

        try:
            digest = hashlib.sha256(
                (artifact / "artifact.json").read_bytes()
            ).hexdigest()
        except OSError:
            digest = None
        signature = (
            None
            if digest is None
            else GenerationHistory.signature_of(
                "profile",
                digest,
                process,
                sorted(truth.record().items()),
                platform.machine(),
                os.cpu_count(),
            )
        )
        self._store = GenerationHistory(signature)

    @property
    def backends(self) -> list[str]:
        return list(self._store.order)

    def cost(self, backend: str, precision: str) -> float | None:
        """The previous profile's measured seconds per point."""

        # Stored in microseconds (the store keeps milliseconds' precision).
        value = self._store.prior(f"cost:{backend}:{precision}")
        return value * 1e-6 if value is not None and value > 0.0 else None

    def seconds(self, name: str) -> float | None:
        return self._store.prior(name)

    def store(
        self,
        backends: Sequence[str],
        costs: Mapping[str, Any],
        seconds: Mapping[str, float],
    ) -> None:
        for backend, cost in costs.items():
            self._store.observe(f"cost:{backend}:binary64", 1e6 * float(cost.binary64))
            if cost.reference is not None and not cost.reference_estimated:
                self._store.observe(
                    f"cost:{backend}:reference", 1e6 * float(cost.reference)
                )
            # The warm entry's seconds where it ran (``takes_entry``).
            for precision in ("binary64", "reference"):
                warm = getattr(cost, f"{precision}_warm", None)
                if warm:
                    self._store.observe(
                        f"cost:{backend}:{precision}-entry", 1e6 * float(warm)
                    )
        for name, value in seconds.items():
            self._store.observe(name, float(value))
        self._store.store(list(backends))


# The reference multiplier expected of each backend before it is measured
# (double-double over binary64, per point: TIRust x4-x16, NinjaRS x25-x100
# on the processes profiled so far).
_MULTIPLIER_PRIOR = {"tirust": 10.0, "ninjars": 50.0}


def _while_ticking(
    call: Callable[[], Any],
    show: Callable[[float], None],
    tick: Callable[[], None],
) -> Any:
    """``call()`` in a thread (the evaluation releases the GIL), and every
    second meanwhile ``show(seconds running)`` and ``tick()``: a long
    in-process call is never a silent phase. Its result, or its exception."""

    import threading

    outcome: list[Any] = []

    def run() -> None:
        try:
            outcome.append((True, call()))
        except BaseException as error:  # handed to the caller
            outcome.append((False, error))

    begin = time.perf_counter()
    worker = threading.Thread(target=run, name="profile-measure", daemon=True)
    worker.start()
    while True:
        worker.join(1.0)
        if not worker.is_alive():
            break
        show(time.perf_counter() - begin)
        tick()
    ok, value = outcome[0]
    if not ok:
        raise value
    return value


def _isolated_profile(
    artifact: Path,
    process: str | None,
    options: Mapping[str, Any],
    config: RunConfig,
    points: Sequence[Any] | None,
    request: str,
    run: _BackendRun | None,
    *,
    reference: Mapping[str, Any] | None = None,
    one_thread: float | None = None,
    warm_entry: bool = False,
    precision: Mapping[str, Any] | None = None,
) -> VirtualProfileResult:
    """``benchmarking.profile_virtual`` of one backend (``request``) on
    ``points`` (None: the artifact's distinct flat points) in a fresh
    spawned process that loads the artifact (with ``options``) and profiles
    that backend only, so that the profile's peak RAM is that of a Monte
    Carlo worker evaluating this matrix element on this backend: in a
    shared process the second backend's peak would hold the first one's
    prepared state and the memory the allocator kept (or, reusing it, would
    hide its own). The worker times on one core: every thread pool at one
    thread (``one_core``). ``reference``: the reference timing's options
    (``profile_virtual``), timed after the binary64 calls. ``run`` receives
    the worker's events and, every second without one, a heartbeat; the
    worker ignores Ctrl-C: this process stops it and raises ``_Interrupted``
    with the binary64 result and the reference calls back so far (when the
    binary64 calls are done), else KeyboardInterrupt. The worker never
    outlives the call."""

    import multiprocessing

    from pyamplicol.api.errors import EvaluationError

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    # The profile reads the evaluation and benchmark sections only (the
    # process section holds mapping proxies, which do not pickle).
    worker = context.Process(
        target=_runtime_worker,
        args=(
            sender,
            str(artifact),
            process,
            dict(options),
            config.evaluation,
            config.benchmark,
            None if points is None else list(points),
            request,
            None if reference is None else dict(reference),
            one_thread,
            warm_entry,
            None if precision is None else dict(precision),
        ),
        name="pyamplicol-loop-runtime-profile",
        daemon=True,
    )
    worker.start()
    sender.close()
    try:
        while True:
            if not receiver.poll(1.0):
                if run is not None:
                    run.tick()
                    run.monitor.tick()
                if not worker.is_alive() and not receiver.poll(0.1):
                    raise EvaluationError(
                        "the runtime profile's worker process ended without a "
                        f"result (exit code {worker.exitcode})"
                    )
                continue
            try:
                kind, *payload = receiver.recv()
            except EOFError:
                worker.join(timeout=10)
                raise EvaluationError(
                    "the runtime profile's worker process ended without a result "
                    f"(exit code {worker.exitcode})"
                ) from None
            if kind == "monitor":
                if run is not None:
                    run.event(payload[0], payload[1])
                    run.monitor.tick()
            elif kind == "result":
                return cast("VirtualProfileResult", payload[0])
            else:
                raise payload[0]
    except KeyboardInterrupt:
        partial = None if run is None else run.partial()
        if run is not None:
            run.abort()
        if partial is not None:
            raise _Interrupted(partial) from None
        raise
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=10)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)


def _runtime_worker(
    sender: Any,
    artifact: str,
    process: str | None,
    options: Mapping[str, Any],
    evaluation: Any,
    benchmark: Any,
    points: Sequence[Any] | None,
    request: str,
    reference: Mapping[str, Any] | None = None,
    one_thread: float | None = None,
    warm_entry: bool = False,
    precision: Mapping[str, Any] | None = None,
) -> None:
    """The worker process of ``_isolated_profile``: Ctrl-C left to the
    parent, every thread pool at ``benchmark.runtime_threads`` threads
    before anything native starts (``pin_threads``; the default one: the
    timing is that of one core), the artifact loaded,
    one backend profiled (the optimistic pass, or with ``benchmark.checked``
    the checked call, a refused point served with its best-effort value and
    timed as such, m-lit ux-besteffort) and its reference timing; its
    events (``profile_virtual``'s monitor, and "loaded"), then its result or
    error, sent back. ``one_thread`` (with more threads): the seconds of a
    one-thread timing of the same calls after them, the runtime reloaded
    with its pools at one thread (``benchmarking.one_thread_profile``; at
    least one timed call), its time per point in the result."""

    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threads = max(int(getattr(benchmark, "runtime_threads", 1) or 1), 1)
    pin_threads(threads)
    # This spawned process is the profile's own: its peak RAM is measured
    # from the reset high-water mark.
    from pyamplicol.reporting.resources import own_process_accounting

    own_process_accounting()
    try:
        from pyamplicol.api import Runtime
        from pyamplicol.benchmarking import profile_virtual
        from pyamplicol.config import Action, RunConfig

        config = RunConfig(
            action=Action.BENCHMARK, evaluation=evaluation, benchmark=benchmark
        )
        begin = time.perf_counter()
        # --runtime-threads is the runtime's thread bound (evaluation.threads'
        # mechanism): its load and every timed call on that many threads.
        runtime = Runtime.load(artifact, process=process, threads=threads, **options)
        sender.send(("monitor", "loaded", {"seconds": time.perf_counter() - begin}))
        profile = profile_virtual(
            runtime,
            config,
            points=points,
            artifact=artifact,
            reduction=request,
            # The checked mode: a refused point is served with its
            # best-effort value and timed as such (m-lit ux-besteffort).
            on_failure="best-effort" if benchmark.checked else "return",
            reference=reference,
            optimistic=not benchmark.checked,
            monitor=lambda kind, payload: sender.send(("monitor", kind, payload)),
            warm_entry=warm_entry,
            precision=precision,
        )
        if threads > 1 and one_thread is not None:
            # The gain: the same calls on one thread, after the N-thread
            # calls and their peak RAM (the N-thread runtime released first).
            import gc

            from pyamplicol.benchmarking import one_thread_profile

            sender.send(("monitor", "reference-result", {"profile": profile}))
            del runtime
            gc.collect()
            pin_threads(1)
            single = one_thread_profile(
                Runtime.load(artifact, process=process, threads=1, **options),
                config,
                budget=one_thread,
                points=points,
                artifact=artifact,
                reduction=request,
                on_failure="best-effort" if benchmark.checked else "return",
                optimistic=not benchmark.checked,
                monitor=lambda kind, payload: sender.send(("monitor", kind, payload)),
            )
            profile = replace(
                profile,
                one_thread_seconds_per_point=single.seconds_per_point,
                one_thread_calls=single.samples,
            )
        sender.send(("result", profile))
    except Exception as error:
        import pickle

        from pyamplicol.api.errors import EvaluationError

        try:
            # Sent as raised when it survives the trip, else by its text.
            pickle.loads(pickle.dumps(error))
        except Exception:
            error = EvaluationError(f"{type(error).__name__}: {error}")
        sender.send(("error", error))
    finally:
        sender.close()


# The stability point statuses of the failure distribution and their kinds.
_CAUSE_KINDS = {
    "best-effort": "best-effort",
    "binary64-failed": "no value",
    "reference-failed": "no reference",
}


@dataclass(frozen=True, slots=True)
class _EstimateSummary:
    """A binary64 call's stability estimate reduced to what the statistics
    and the per-point record read (``_backend_statistics``): its
    ``relative`` estimate (the flag), as ``finite`` the estimated relative
    error of the total's finite part (``_estimated_relative``; ``absolute``
    None so that it is read back unchanged), every check's own relative
    budget (``checks``), the checks that did not run (``not_run``) and the
    scale estimate (``scale``). Small and picklable: the worker processes
    return it."""

    relative: float
    finite: float
    absolute: None = None
    checks: tuple[tuple[str, float], ...] = ()
    not_run: tuple[tuple[str, str], ...] = ()
    scale: float | None = None


# The state of a stability worker (``_stability_worker``): the runtime, the
# check set, the reference and a point to warm up on; set once per process.
_WORKER: dict[str, Any] = {}

# The thread pools of a worker process, one thread each (``one_core``): a
# runtime worker times one core; N stability workers use about N cores (no
# oversubscription).
_WORKER_THREADS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _stability_worker(
    artifact: str,
    process: str | None,
    options: Mapping[str, Any],
    warm: tuple[Any, Any] | None,
    checks: Any,
    backends: Sequence[str] = (),
    truth: GroundTruth | None = None,
    cold: Sequence[tuple[str, str]] = (),
    defaulted: bool = False,
    measured: Mapping[str, Mapping[str, Any]] | None = None,
    streams: Sequence[str] | None = None,
) -> None:
    """Initialise a stability worker process: one thread per pool, Ctrl-C
    left to the parent (which collects what finished and stops the
    workers), the artifact loaded once with the reference precision
    ``truth`` (default double-double; "none": no reference), and warmed up
    on ``backends`` with ``warm`` (a point and its scale, ``_warm_up``; the
    ``cold`` streams not, nor a checked reference stream until its first
    task, ``_stability_task``)."""

    import signal

    one_core()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from pyamplicol.api import Runtime

    _WORKER.update(
        runtime=Runtime.load(artifact, process=process, threads=1, **options),
        warm=warm,
        checks=checks,
        defaulted=defaulted,
        truth=truth or GroundTruth("double-double"),
        cold=tuple(cold),
        measured=dict(measured or {}),
        streams=None if streams is None else tuple(streams),
    )
    _warm_up(backends, cold)


def one_core() -> None:
    """Size every thread pool this process starts from now on at one
    thread (``pin_threads``): its evaluations then use one core."""

    pin_threads(1)


def pin_threads(threads: int) -> None:
    """Size every thread pool this process starts from now on at
    ``threads`` threads (``_WORKER_THREADS``: rayon's, used by every native
    pool that sets no count, and the BLAS/OpenMP ones): set before anything
    native starts; a runtime loaded afterwards builds its executor's pool at
    that size. A pool already started keeps its threads."""

    for name in _WORKER_THREADS:
        os.environ[name] = str(int(threads))


def _precisions(truth: GroundTruth | None) -> tuple[str, ...]:
    """The evaluations of a stability run: binary64, and the reference
    unless ``--ground-truth none``."""

    if truth is not None and not truth.present:
        return ("binary64",)
    return _PRECISIONS


def _warm_up(backends: Sequence[str], cold: Sequence[tuple[str, str]] = ()) -> None:
    """Pay this process's first-call costs before any timed task, untimed:
    the stability task of one point on every backend and precision (the
    process's first evaluation, each backend's and carrier's first call and
    what it builds lazily; after native's fix one point per carrier builds
    the engine of every batch size), its result dropped. An evaluation does
    not depend on what ran before it (bit-identical values and estimates),
    so the statistics are unchanged; only the timings lose that cost. The
    ``cold`` streams (backend, precision) are left out: a point dearer than
    ``profile_plan.MEASURE_REPEAT_SECONDS`` makes its first-call cost
    negligible, and its warm-up would cost a whole point per worker. A
    backend's error is left to its timed tasks, which report it."""

    warm = _WORKER.get("warm")
    if not warm:
        return
    point, scale = warm
    for backend in backends:
        # The run's streams (``_stability``), else every one of the truth.
        streams = _WORKER.get("streams") or (
            *_precisions(_WORKER.get("truth")),
            *_WORKER.get("measured", {}),
        )
        for precision in streams:
            if (backend, precision) in cold:
                continue
            with contextlib.suppress(Exception):
                _stability_task(backend, precision, [0], [point], [scale])
    # The timed tasks start without the warm-up's garbage.
    gc.collect()


def _stability_task(
    backend: str,
    precision: str,
    indices: Sequence[int],
    batch: Sequence[Any],
    scales: Sequence[Any] | Any,
) -> tuple[str, str, list[tuple[Any, ...]], float]:
    """One batch of a stability run: the flat points ``indices`` (``batch``,
    at their ``scales``: one per point, or one for all) on ``backend`` in
    ``precision`` ("binary64": the checked call with its stability estimate,
    a refused point served with its best-effort value; "reference": the
    worker's ``GroundTruth``, accepted values only, the exact sum of their
    sectors). Per point a record (``_stability_records``), and the call's
    seconds, Python's cyclic garbage collector off during it (as in the
    runtime's timed calls, ``benchmarking._collector_off``)."""

    from pyamplicol.benchmarking import _collector_off

    if precision == _REFERENCE_CHECKED:
        _warm_checked(backend, batch, scales)
    with _collector_off():
        begin = time.perf_counter()
        records = _stability_records(backend, precision, indices, batch, scales)
        seconds = time.perf_counter() - begin
    return backend, precision, records, seconds


def _warm_checked(backend: str, batch: Sequence[Any], scales: Any) -> None:
    """Before this worker's first checked reference task of ``backend``,
    the checked call of its first point, untimed and dropped (``_warm_up``
    for the checked stream, which only some points need: its first-call
    costs, the checks' lazy set-up included, stay out of its timing); none
    when the backend's reference stream is cold (``_warm_up``)."""

    warmed = _WORKER.setdefault("warmed", set())
    if backend in warmed or (backend, "reference") in _WORKER.get("cold", ()):
        return
    warmed.add(backend)
    scale = scales[0] if isinstance(scales, list) else scales
    with contextlib.suppress(Exception):
        _stability_records(backend, _REFERENCE_CHECKED, [0], [batch[0]], [scale])
    gc.collect()


def _stability_records(
    backend: str,
    precision: str,
    indices: Sequence[int],
    batch: Sequence[Any],
    scales: Sequence[Any] | Any,
) -> list[tuple[Any, ...]]:
    """The per-point records of a stability task (``_stability_task``):
    binary64 (index, [[total]], estimate summary, reason, status,
    best-effort uncertainty, the sectors' finite parts); reference (index,
    [[exact total]], reason, its exact decimal text): one plain evaluation,
    or (``_REFERENCE_CHECKED``) the reference with its checks
    (``_reference_checks``), a point whose estimate exceeds binary64's unit
    roundoff (``_UNIT_ROUNDOFF``) then without a reference value, its reason
    saying so."""

    state = _WORKER
    common = {"mu_r2": scales, "poles": "none", "reduction": backend}
    # A measured precision other than binary64 (``MeasuredPrecision``): its
    # stream evaluates as binary64's, in its arithmetic, its total exact.
    measured = precision.startswith(_MEASURED)
    if precision == "binary64" or measured:
        from pyamplicol.loops.v2.reduction import default_checks

        # A point the backend refuses keeps its best-effort value (the
        # default policy, m-lit ux-besteffort), measured as such. Without
        # ``evaluation.stability_checks`` the checks are the DEFAULT set
        # (``reduction.default_checks``): one default rotation, and on a
        # point TIRust serves the rescaled reduction-direction check too.
        with default_checks() if state.get("defaulted") else contextlib.nullcontext():
            result = state["runtime"].evaluate(
                list(batch),
                stability_checks=state["checks"],
                on_failure="best-effort",
                **(state["measured"][precision] if measured else {}),
                **common,
            )
        records: list[tuple[Any, ...]] = []
        from pyamplicol.benchmarking import lane_rescue_reason, reduction_rescue

        for index, value, estimate, reason, status, uncertainty, lane in zip(
            indices,
            result.values,
            result.stability,
            _reasons(result),
            result.statuses,
            _uncertainties(result),
            result.diagnostics,
            strict=True,
        ):
            total = (
                None
                if value is None
                else _exact_total(value)
                if measured
                else _total_finite(value)
            )
            summary = (
                None
                if estimate is None or total is None
                else _EstimateSummary(
                    float(estimate.relative),
                    _estimated_relative(estimate, float(total)),
                    checks=tuple(estimate.checks),
                    not_run=tuple(getattr(estimate, "not_run", ()) or ()),
                    scale=None if estimate.scale is None else float(estimate.scale),
                )
            )
            records.append(
                (
                    index,
                    None if total is None else [[total]],
                    summary,
                    reason,
                    status,
                    uncertainty,
                    None
                    if value is None
                    else [
                        str(Decimal(sector[0])) if measured else float(sector[0])
                        for sector in value
                    ],
                    lane_rescue_reason(lane),
                    # A wider precision's total as its exact decimal text.
                    None if value is None or not measured else _exact_text(value),
                    # The reduction's own rescue: (lanes, precision) or None.
                    reduction_rescue(lane),
                    # Lanes served without accuracy evidence: the reasons.
                    "; ".join(getattr(lane, "uncertified", ()) or ()) or None,
                )
            )
    else:
        # The reference counts only accepted values, summed exactly; the
        # checked re-run of a reference the cross-check did not certify
        # (``_REFERENCE_CHECKED``) runs the reference's checks.
        truth = state.get("truth") or GroundTruth("double-double")
        checks = _reference_checks(truth) if precision == _REFERENCE_CHECKED else None
        result = state["runtime"].evaluate(
            list(batch),
            on_failure="return",
            **({} if checks is None else {"stability_checks": checks}),
            **truth.options,
            **common,
        )
        records = []
        for index, value, reason, estimate, lane in zip(
            indices,
            result.values,
            _reasons(result),
            result.stability if checks is not None else [None] * len(indices),
            result.diagnostics,
            strict=True,
        ):
            if (
                value is not None
                and estimate is not None
                and float(estimate.relative) > _UNIT_ROUNDOFF
            ):
                # A success its checks do not trust to resolve binary64's
                # digits is no reference: it takes the path of a failed
                # reference point (its estimate shown as relative_text).
                from pyamplicol.loops.v2.stability import relative_text

                value = None
                reason = (
                    "failed: stability checks: estimate "
                    f"{relative_text(float(estimate.relative), '.1e')} above "
                    f"binary64's unit roundoff {float(_UNIT_ROUNDOFF):.1e}"
                )
            exact = None if value is None else _exact_total(value)
            # The reference's lanes served without accuracy evidence: the
            # reference is then not certified (``harvest``).
            unsure = "; ".join(getattr(lane, "uncertified", ()) or ()) or None
            records.append(
                (
                    index,
                    None if exact is None else [[exact]],
                    reason,
                    None if value is None else _exact_text(value),
                    # A checked reference's own estimate (what it certifies:
                    # ``_point_caps``).
                    *(() if estimate is None else (float(estimate.relative),)),
                    *(
                        ()
                        if unsure is None or value is None
                        else (_ReferenceUncertified(unsure),)
                    ),
                )
            )
    return records


def _agreement_dispositions(
    totals: Mapping[str, Any], flagged: Container[str]
) -> tuple[dict[str, str], Fraction | None]:
    """How the backends' plain references of one point are certified
    (``decide``): per backend "none" (no value), "agreement" (it agrees with
    another backend's reference that is NOT flagged: a reference served
    with lanes without accuracy evidence is never certification evidence
    for anything) or "check" (its checked re-run decides); and the spread
    of the agreeing clean references (``agreed``: the agreement's
    resolution), None unless at least two references, none of them
    flagged, all agree."""

    valued = [b for b, total in totals.items() if total is not None]
    clean = [b for b in valued if b not in flagged]
    out: dict[str, str] = {}
    for backend, total in totals.items():
        if total is None:
            out[backend] = "none"
        elif any(
            _references_agree(total, totals[other])
            for other in clean
            if other != backend
        ):
            out[backend] = "agreement"
        else:
            out[backend] = "check"
    spread = None
    values = [totals[b] for b in valued]
    if (
        len(values) > 1
        and len(clean) == len(valued)
        and all(
            _references_agree(a, b)
            for k, a in enumerate(values)
            for b in values[k + 1 :]
        )
    ):
        spread = max(
            abs(a - b) / max(abs(a), abs(b)) if a or b else Fraction(0)
            for k, a in enumerate(values)
            for b in values[k + 1 :]
        )
    return out, spread


class _ReferenceUncertified(str):
    """The reasons of a reference point's lanes served without accuracy
    evidence, appended to its worker record (``_stability_records``) and
    taken off it on arrival (``harvest``)."""

    __slots__ = ()


# The task kind of a reference evaluated with its checks (``_stability_records``):
# the re-run of a reference the backends' cross-check did not certify, or
# the reference itself when only one backend is profiled.
_REFERENCE_CHECKED = "reference-checked"
# A reference exists to resolve the stable digits of a binary64 value, down to
# their cap ``BINARY64_DIGITS``: an error of binary64's unit roundoff u = 2^-53.
# A checked reference is kept only when its estimate is at most u (above it,
# it cannot tell a binary64 value right to its last digit from one a few
# units off), and two checked references that both pass but differ by more
# than u contradict their estimates: neither is kept.
_UNIT_ROUNDOFF = Fraction(1, 2**53)
# The agreement tolerance of the backends' cross-check (``_references_agree``):
# u squared, 2^-106. To leave the measure of an error of u exact to binary64's
# own resolution the reference must be right to u * u, the accuracy of the
# double-double ground truth (the default).
_AGREEMENT = _UNIT_ROUNDOFF**2


def _references_agree(
    first: Fraction, second: Fraction, tolerance: Fraction = _AGREEMENT
) -> bool:
    """Whether two backends' reference totals agree: their difference at
    most ``tolerance`` (default ``_AGREEMENT``) times the larger modulus (two
    exact zeros agree; a zero and a nonzero do not, nor do two values at
    rounding level of an integrated zero unless they agree to that
    tolerance, so such a point takes the checked re-run)."""

    return abs(first - second) <= tolerance * max(abs(first), abs(second))


def _reference_checks(truth: GroundTruth) -> Any:
    """The stability checks of a checked reference (``_REFERENCE_CHECKED``):
    the recommended ones (``loops.v2.stability.RECOMMENDED_CHECKS``) for an
    arbitrary-precision ground truth, whose carriers can return a wrong
    value with status success (NinjaRS at near-degenerate boxes): such a
    reference is cross-checked between the backends, and one the
    cross-check does not certify is evaluated again with these checks,
    dropped when its estimate exceeds binary64's unit roundoff; None for
    double-double (one plain evaluation, no cross-check)."""

    if truth.arithmetic != "arbitrary":
        return None
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    return VirtualStabilityChecks(
        RECOMMENDED_CHECKS.rotations,
        RECOMMENDED_CHECKS.reduction_direction,
        rescaled=RECOMMENDED_CHECKS.rescaled,
    )


def _exact_text(row: Any) -> str:
    """A reference value's total finite part as an exact decimal text: its
    sectors (decimal at the reference's precision) summed without
    rounding."""

    from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, Context

    exact = Context(prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN)
    total = Decimal(0)
    for sector in row:
        total = exact.add(total, Decimal(sector[0]))
    return str(total)


def _effective_threads(requested: int) -> int:
    """The thread count a runtime loaded with ``threads=requested`` uses
    (the native rule: clamped to the affinity mask and the CPU quota)."""

    from pyamplicol.loops.native import load_loop_native

    return int(load_loop_native().effective_threads(min(max(requested, 1), 2**64 - 1)))


def usable_cores() -> int:
    """The cores this process may run on (its CPU affinity where the
    platform reports one)."""

    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)


def _chunk_size(points: int, workers: int, batch_size: int, lanes: int = 1) -> int:
    """The points of one stability task: about eight tasks per worker and
    precision (progress and load balance), in whole SIMD chunks of
    ``lanes`` points (binary64 evaluates ``lanes`` points at a time, and a
    padded lane costs as much as a filled one: a task of fewer points
    would time empty lanes), at most one batch (whole chunks of a batch
    that holds one)."""

    lanes = max(lanes, 1)
    batch = max(batch_size, 1)
    if batch >= lanes:
        batch -= batch % lanes
    size = lanes * math.ceil(points / (8 * workers * lanes))
    return max(1, min(batch, size))


class _StabilityProgress:
    """The progress of a stability run (``profile --stability``): points done
    per backend and precision, and the estimated time to completion from the
    measured cost per point of each backend and precision (the workers
    warmed up first, ``_warm_up``: a task's seconds are its points' cost)."""

    def __init__(
        self,
        sink: Any,
        backends: Sequence[str],
        points: int,
        workers: int,
        first: int = 1,
        reference: str = "dd",
        *,
        monitor: _Monitor | None = None,
        precisions: Sequence[str] = ("binary64", "reference"),
        priors: Mapping[tuple[str, str], float] | None = None,
        deadline: float | None = None,
    ) -> None:
        from pyamplicol.reporting.progress import ProgressStart

        self.sink = sink
        self.monitor = monitor
        # The reference's label in the progress line (``GroundTruth.label``).
        self.reference = reference
        self.backends = tuple(backends)
        self.points = points
        self.workers = workers
        self.precisions = tuple(precisions)
        # The expected cost per point of each stream before any of its tasks
        # is back (the measurement phase's), and the run's deadline.
        self.priors = dict(priors or {})
        self.deadline = deadline
        # The serial seconds per point after the run (statistics, records).
        self.tail = 0.0
        self.flagged = dict.fromkeys(backends, 0)
        self.task = "loop-stability"
        self.done = {(b, p): 0 for b in backends for p in self.precisions}
        self.seconds = {key: 0.0 for key in self.done}
        # Per backend the checked reference re-runs queued or running
        # (``checking``) and the points and seconds of those back
        # (``checked``): their seconds count in the reference stream's time
        # (``seconds``), not in its plain rate (``rate``); ``stream_cost``
        # adds their expected share per point.
        self.checking = dict.fromkeys(backends, 0)
        self.checked = {b: [0, 0.0] for b in backends}
        # Per backend and precision the points and seconds of its returned
        # tasks (a refused backend's points count as done, not timed).
        self.timed = {key: [0, 0.0] for key in self.done}
        # The points of every stream's first task, started together.
        self.first = max(first, 1)
        self.begin = time.perf_counter()
        # When the first task came back, and its seconds: the workers were
        # computing since the difference.
        self.first_return: tuple[float, float] | None = None
        # The displayed estimate and when it was shown (shown_eta).
        self._shown: tuple[float, float] | None = None
        self.estimated = True
        if sink is not None:
            sink.emit(
                ProgressStart(
                    self.task,
                    f"Stability: {len(backends)} backend(s) on {workers} core(s)",
                    total=points * len(self.done),
                    parent_task_id=None if monitor is None else monitor.task,
                    unit="evaluations",
                    details={
                        "step": "starting the workers (load, one warm-up point each)",
                        "eta_seconds": None,
                    },
                )
            )

    def add(self, backend: str, precision: str, count: int, seconds: float) -> None:
        """A finished task of ``count`` points."""

        if self.first_return is None:
            self.first_return = (time.perf_counter(), seconds)
        key = (backend, precision)
        self.timed[key][0] += count
        self.timed[key][1] += seconds
        self.done[key] += count
        self.seconds[key] += seconds
        self.emit()

    def drop(self, backend: str) -> None:
        """A refused backend: its evaluations count as done."""

        for precision in self.precisions:
            self.done[(backend, precision)] = self.points
        self.checking[backend] = 0
        self.emit()

    def queue_checked(self, backend: str, count: int) -> None:
        """``count`` more references of ``backend`` to re-run checked."""

        self.checking[backend] += count

    def add_checked(self, backend: str, count: int, seconds: float) -> None:
        """A finished checked re-run of ``count`` references: its seconds
        are the reference stream's time, its points are not new points."""

        self.checking[backend] = max(self.checking[backend] - count, 0)
        self.checked[backend][0] += count
        self.checked[backend][1] += seconds
        key = (backend, "reference")
        if key in self.seconds:
            self.seconds[key] += seconds
        self.emit()

    def stream_cost(self, key: tuple[str, str]) -> float | None:
        """A stream's measured cost per point (None: nothing back): its
        rate, and for a reference stream the re-runs' seconds so far per
        point (their expected share of a future point)."""

        rate = self.rate(key)
        if rate is None or key[1] != "reference":
            return rate
        points = self.timed[key][0]
        return rate + self.checked[key[0]][1] / points

    def checked_rate(self, backend: str) -> float | None:
        """The measured cost of one checked reference re-run (None: none
        back yet)."""

        points, seconds = self.checked[backend]
        return seconds / points if points else None

    def set_points(self, points: int) -> None:
        """The run's planned points (the admission gate's, as it goes)."""

        self.points = max(points, max(self.done.values(), default=0))

    def flag(self, backend: str, count: int) -> None:
        """``count`` more points of ``backend`` the default checks flag
        (or served best-effort, or without a value)."""

        self.flagged[backend] = self.flagged.get(backend, 0) + count

    def remaining(self) -> float | None:
        """The run's remaining seconds as the display shows them, and the
        serial tail after it (``tail``: seconds per point submitted)."""

        eta = self.shown_eta()
        if eta is None:
            return None
        return eta + self.tail * max(self.points, 0)

    def rate(self, key: tuple[str, str]) -> float | None:
        """The measured cost per point of a stream (None: nothing back)."""

        points, seconds = self.timed[key]
        return seconds / points if points else None

    def eta(self) -> float | None:
        raw = self._eta()
        if self.deadline is None or raw is None:
            return raw
        # Chunks are admitted until the deadline: the run lasts about that
        # long, and no less than what is queued.
        return max(raw, self.deadline - time.perf_counter())

    def _eta(self) -> float | None:
        known = {
            key: rate
            for key in self.done
            if (rate := self.stream_cost(key)) is not None
        }
        self.estimated = len(known) < len(self.done)
        if not known:
            if not self.priors:
                return None
            # Nothing back yet: the measured priors.
            return sum(
                (self.points - done) * self.priors.get(key, 0.0)
                for key, done in self.done.items()
            ) / max(self.workers, 1)
        fallback = statistics.mean(known.values())
        if len(known) < len(self.done) and self.first_return is not None:
            # A stream with nothing back costs at least the running time of
            # its first task (started with the one that returned first) per
            # point of it.
            at, spent = self.first_return
            running = time.perf_counter() - (at - spent)
            fallback = max(fallback, running / self.first)
        remaining = 0.0
        for key, done in self.done.items():
            remaining += (self.points - done) * known.get(
                key, max(fallback, self.priors.get(key, 0.0))
            )
        for backend, count in self.checking.items():
            # The queued re-runs of points already done (the points' own
            # share in ``known`` covers future points only); one costs at
            # least a plain reference.
            if not count:
                continue
            rate = self.checked_rate(backend)
            if rate is None and (backend, "reference") in self.timed:
                rate = self.rate((backend, "reference"))
            rate = rate or fallback
            remaining += count * rate
        return remaining / max(self.workers, 1)

    #: The time constant (seconds) of the displayed estimate's smoothing: a
    #: display value, it never changes a result.
    ETA_SMOOTHING_SECONDS = 10.0

    def shown_eta(self) -> float | None:
        """The displayed estimate: :meth:`eta` smoothed over time, so the
        refinements of the first returns (a stream's cost bounded while
        nothing of it is back, then measured) do not make it jump. The
        remaining time is averaged in logarithm, with weight
        ``1 - exp(-dt / ETA_SMOOTHING_SECONDS)`` for an estimate ``dt``
        seconds after the previous one: a factor-ten jump moves it by about
        a factor 1.25 a second later and fully after half a minute."""

        raw = self.eta()
        now = time.perf_counter()
        previous = self._shown
        if raw is None or previous is None:
            self._shown = None if raw is None else (now, raw)
            return raw
        at, value = previous
        if raw <= 0.0:
            self._shown = (now, 0.0)
            return 0.0
        dt = max(now - at, 0.0)
        carried = max(value - dt, 1e-3)
        weight = 1.0 - math.exp(-dt / self.ETA_SMOOTHING_SECONDS)
        shown = math.exp((1.0 - weight) * math.log(carried) + weight * math.log(raw))
        self._shown = (now, shown)
        return shown

    def emit(self, *, heartbeat: bool = False) -> None:
        from pyamplicol.reporting.progress import ProgressUpdate

        if self.sink is None:
            return
        from pyamplicol.loops.v2.reduction import NAMES

        parts = []
        for backend in self.backends:
            name = NAMES.get(backend, backend)
            # Every stream's points (binary64 "b64", the reference by its
            # label, a measured precision by its own).
            done = [self.done[(backend, p)] for p in self.precisions]
            b64 = done[0] if done else 0
            part = name + "".join(
                f" {_stream_label(p, self.reference)} {count}/{self.points}"
                for p, count in zip(self.precisions, done, strict=True)
            )
            parts.append(part)
            if self.monitor is not None:
                self.monitor.set_reading(
                    name,
                    f"{b64}/{self.points}",
                    f"{self.flagged.get(backend, 0)} flagged",
                )
        elapsed = time.perf_counter() - self.begin
        done = sum(self.done.values())
        # The first stream's points (binary64's, when it runs).
        first = self.precisions[0] if self.precisions else "binary64"
        rate = sum(self.done.get((b, first), 0) for b in self.backends) / max(
            elapsed, 1e-9
        )
        eta = self.shown_eta()
        details: dict[str, Any] = {
            "step": " | ".join(parts) + (f"  {rate:.3g} pt/s" if done else ""),
            "eta_seconds": None if eta is None else round(max(eta, 0.0), 1),
            "eta_estimated": self.estimated,
            "heartbeat": heartbeat,
        }
        if self.monitor is not None:
            details["readings"] = self.monitor.readings()
        total = self.points * len(self.done)
        self.sink.emit(
            ProgressUpdate(
                self.task,
                min(done, total),
                max(total, done),
                details=details,
            )
        )

    def close(self, interrupted: bool) -> None:
        from pyamplicol.loops.v2.reduction import NAMES
        from pyamplicol.reporting.progress import ProgressEnd

        if self.sink is not None:
            first = self.precisions[0] if self.precisions else "binary64"
            points = min(
                (self.done.get((b, first), 0) for b in self.backends), default=0
            )
            summary = f"{points} points" + "".join(
                f", {NAMES.get(b, b)} {self.flagged.get(b, 0)} flagged"
                for b in self.backends
            )
            self.sink.emit(
                ProgressEnd(
                    self.task,
                    success=not interrupted,
                    message="interrupted" if interrupted else summary,
                    elapsed_seconds=time.perf_counter() - self.begin,
                    details={"keep": True},
                )
            )


# The two evaluations of a stability run: the binary64 call and the
# reference (the run's ``GroundTruth``).
_PRECISIONS = ("binary64", "reference")


# What the default estimate of a stability row is (the setup record, the
# field descriptions; ``reduction.default_checks``).
DEFAULT_ESTIMATE = (
    "the default checks: one default rotation on the points NinjaRS serves, "
    "one default rotation and the rescaled reduction-direction check on the "
    "points TIRust serves"
)


def _stream_label(stream: str, reference: str) -> str:
    """A stability stream's label in the progress line: "b64", the
    reference's (``GroundTruth.label``), a measured precision's own."""

    if stream == "binary64":
        return "b64"
    if stream == "reference":
        return reference
    return stream.removeprefix(_MEASURED)


def _stability(
    runtime: Any,
    artifact: Path,
    physics: Any,
    config: RunConfig,
    backends: Any,
    *,
    load_options: Mapping[str, Any] | None = None,
    progress: Any = None,
    monitor: _Monitor | None = None,
    planned_points: int | None = None,
    deadline: float | None = None,
    costs: Mapping[str, Any] | None = None,
    current: list[Any] | None = None,
    records: bool = True,
    tail_seconds: float = 0.0,
    worker_bytes: float | None = None,
) -> tuple[tuple[BackendStability, ...], Mapping[str, object]]:
    """The stability statistics of every backend (module docstring) over
    flat points, evaluated on ``benchmark.n_core`` worker processes (capped
    by the usable cores; 1: in this process, on ``runtime``): the
    ``planned_points`` (default ``benchmark.stability_points``) or, with a
    ``deadline`` (``time.perf_counter`` seconds), chunks of one SIMD chunk
    admitted while their work ends before it (``profile_plan.admit_chunk``,
    never fewer than ``STABILITY_MINIMUM`` points), from the ``costs``
    measured (the priors of the rates). Point i is drawn from seed + i and
    the statistics read the per-point records in point order, so the result
    does not depend on the worker count. ``--ground-truth none`` evaluates
    binary64 only; an arbitrary-precision reference is cross-checked
    between the backends and re-run checked where they do not agree (or
    only one has it): the re-runs go first, before the next chunk.
    Ctrl-C stops the run: the points evaluated so far
    (binary64 and reference both, or no binary64 value) make the
    statistics, ``setup["interrupted"]`` says so; a second Ctrl-C aborts.
    ``current[0]`` holds the tracker while it runs (the profile's ETA).
    ``records``: build every sample's record (``--json``, --samples-output;
    else None); ``tail_seconds``: the serial seconds per point after the run
    (statistics, records), left out of the deadline's time."""

    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.cli.flat_points import draw_flat_points
    from pyamplicol.config.models import DEFAULT_MEASURED_PRECISIONS
    from pyamplicol.loops.v2.plan_export import DEFAULT_LANES
    from pyamplicol.loops.v2.reduction import default_target
    from pyamplicol.profile_plan import (
        MEASURE_REPEAT_SECONDS,
        STABILITY_MINIMUM,
        affordable_chunk,
        task_points,
    )

    bench = config.benchmark
    evaluation = config.evaluation
    space = evaluation.phase_space
    configured = evaluation.stability_checks.to_checks()
    checks = configured if configured is not None else VirtualStabilityChecks(1)
    # The default set follows the serving backend (``reduction.default_checks``:
    # TIRust's points add the rescaled reduction-direction check).
    defaulted = configured is None
    target = (
        evaluation.reduction_target
        if evaluation.reduction_target is not None
        else default_target()
    )
    truth = ground_truth(bench.ground_truth)
    # The measured precisions (``--measured-precisions``): binary64's stream
    # always runs (its tables only when listed); each other one is a stream
    # of its own (``MeasuredPrecision.stream``).
    listed = measured_precisions(bench.measured_precisions, truth)
    extras = [m for m in listed if m.stream != "binary64"]
    # binary64 not listed: not evaluated (the first listed precision's
    # points are the evaluated ones).
    binary64_listed = len(extras) < len(listed)
    cross = _reference_checks(truth) is not None
    # The ground truth listed: its stream (the default checks) is also its
    # reference evaluation, accepted values only, as the plain one keeps
    # them (the same value bit for bit: verified on g g > g g g); the
    # certification of an arbitrary-precision truth as before. One backend
    # with a cross-checked truth keeps its checked references.
    unified = next(
        (
            m
            for m in extras
            if m.estimates_only and truth.present and not (cross and len(backends) == 1)
        ),
        None,
    )

    def stream_of(precision: str) -> str:
        # The reference evaluated as the ground truth's own stream.
        if precision == "reference" and unified is not None:
            return unified.stream
        return precision

    own_streams = tuple(
        p
        for p in _precisions(truth)
        if (p != "binary64" or binary64_listed)
        and (p != "reference" or unified is None)
    )
    precisions = own_streams + tuple(m.stream for m in extras)
    count = int(
        planned_points if planned_points is not None else bench.stability_points
    )
    count = max(count, 1)
    workers = max(1, min(bench.n_core, usable_cores()))
    # As many workers as the memory holds (``profile_plan.memory_workers``:
    # ``worker_bytes``, one worker's resident size as this run measured it,
    # against the memory available now); a --n_core given is honoured, and
    # warned about when it does not fit.
    memory = _memory_record(workers, worker_bytes, bench)
    if memory is not None and memory["fits"] < workers:
        from pyamplicol.config.models import is_default

        need = _gib(memory["per_worker_bytes"])
        held = _gib(memory["available_bytes"])
        if is_default(bench.n_core):
            _notice(
                f"the stability run starts {memory['fits']} "
                f"worker{'s' if memory['fits'] != 1 else ''} instead of "
                f"{workers}: each holds about {need} (measured), three quarters "
                f"of the {held} available hold {memory['fits']}"
            )
            workers = int(memory["fits"])
        else:
            _notice(
                f"--n_core {workers} given: its workers need about "
                f"{_gib(workers * memory['per_worker_bytes'])} ({need} each, "
                f"measured), more than three quarters of the {held} available",
                warning=True,
            )
    # A task of several points (a stream's point cheaper than a task's
    # work, ``task_points``) holds more than the one-point calls the
    # estimate measured: the first worker runs its first task alone and its
    # own peak sets the workers (``_run_pool``'s ``probe``); a task of one
    # point is that measurement's own evaluation, nothing to probe.
    probe = None
    if memory is not None and workers > 1:
        from pyamplicol.config.models import is_default
        from pyamplicol.profile_plan import memory_workers

        point_costs = [
            seconds
            for backend in backends
            if (cost := (costs or {}).get(backend)) is not None
            for seconds in (cost.stability, cost.stability_reference)
            if seconds
        ]
        chunked = not point_costs or any(
            task_points(seconds, DEFAULT_LANES) > 1 for seconds in point_costs
        )
        if chunked and is_default(bench.n_core):

            def probe(peak: float) -> int:
                available = _available_memory()
                fits = (
                    workers
                    if available is None
                    else min(workers, memory_workers(peak, available))
                )
                memory.update(
                    probed_bytes=peak, probed_workers=fits, available_bytes=available
                )
                if fits < workers:
                    _notice(
                        f"the stability run keeps {fits} of its {workers} workers: "
                        f"its first worker holds {_gib(peak)} after its first task, "
                        "three quarters of the "
                        f"{_gib(available or 0.0)} available hold {fits}"
                    )
                return fits

    # The flat points as the chunks need them (point i from seed + i), and
    # every point's inputs (the per-point record).
    inputs: dict[int, tuple[Any, Any]] = {}
    sqrt_s: list[float] = []
    cut_legs: list[tuple[int, ...]] = []

    def draw(start: int, size: int) -> tuple[list[Any], list[Any] | Any]:
        flat = draw_flat_points(
            artifact,
            process=physics.process_id,
            points=size,
            seed=space.seed + start,
            sqrt_s=space.sqrt_s,
            cuts=space.cuts,
        )
        if not sqrt_s:
            sqrt_s.append(flat.sqrt_s)
            cut_legs.append(flat.cut_legs)
        batch = list(flat.as_inputs(exact=False))
        scales = evaluation.mu_r2 if evaluation.mu_r2 is not None else flat.mu_r2()
        for k, point in enumerate(batch):
            inputs[start + k] = (
                point,
                scales if not isinstance(scales, list) else scales[k],
            )
        return batch, scales

    if deadline is None:
        chunk = _chunk_size(count, workers, bench.batch_size, DEFAULT_LANES)
        # One SIMD chunk of points first on every backend and precision,
        # started together (the time estimate, early), then tasks of
        # ``chunk`` points: every task but the last in whole chunks of
        # binary64 lanes.
        lead = min(chunk, DEFAULT_LANES, count)
        bounds = [(0, lead)] + [
            (start, min(start + chunk, count)) for start in range(lead, count, chunk)
        ]

        def prior(backend: str, precision: str) -> float:
            cost = (costs or {}).get(backend)
            if cost is None:
                return 0.0
            if precision == "binary64":
                return float(cost.stability)
            return float(cost.stability_reference or 0.0)

        planned_tasks = sum(
            math.ceil((end - start) / task_points(prior(b, p), end - start))
            for start, end in bounds
            for b in backends
            for p in precisions
        )
        workers = max(1, min(workers, planned_tasks))
    else:
        # A deadline: chunks of one SIMD chunk while they fit it; the first
        # holds the minimum, a whole chunk only when the plan affords several
        # (its costs are priors until the first chunk is back).
        lead = (
            min(count, DEFAULT_LANES)
            if count >= 4 * DEFAULT_LANES
            else STABILITY_MINIMUM
        )
        bounds = [(0, lead)]
    priors: dict[tuple[str, str], float] = {}
    for backend in backends:
        cost = (costs or {}).get(backend)
        if cost is not None:
            priors[(backend, "binary64")] = cost.stability
            if cost.stability_reference is not None:
                priors[(backend, "reference")] = cost.stability_reference
            for m in extras:
                priors[(backend, m.stream)] = measured_prior(
                    cost, m, checked_evaluations(configured)
                )
    # The streams whose point is dear enough that the workers skip their
    # warm-up (``_warm_up``).
    cold = tuple(
        key for key, seconds in priors.items() if seconds > MEASURE_REPEAT_SECONDS
    )
    tracker = _StabilityProgress(
        progress,
        backends,
        count,
        workers,
        lead,
        truth.label,
        monitor=monitor,
        precisions=precisions,
        priors=priors,
        deadline=deadline,
    )
    tracker.tail = tail_seconds
    if current is not None:
        current[0] = tracker
    b64: dict[str, dict[int, tuple[Any, ...]]] = {b: {} for b in backends}
    dd: dict[str, dict[int, tuple[Any, ...]]] = {b: {} for b in backends}
    seconds = dict.fromkeys(backends, 0.0)
    # Per backend, the task (its seconds and points) of every point and
    # precision (the per-sample record's timing).
    tasks_of: dict[str, dict[tuple[str, int], tuple[float, int]]] = {
        b: {} for b in backends
    }
    refused: dict[str, str] = {}
    submitted = [0]
    pending = list(bounds)
    # The backends' cross-check of an arbitrary-precision reference
    # (``_references_agree``): the plain references wait in ``plain`` until
    # every live backend's is back; one that agrees with another backend's
    # is certified ("agreement"), the others are queued for their checked
    # re-run (``checking``); once every re-run of the point is back
    # (``rerun``, ``rechecked``) each certifies ("checks") or drops its
    # reference, and two that pass but differ by more than binary64's unit
    # roundoff drop both (``settle``). With one live backend the reference
    # runs checked at once. Only certified references enter ``dd`` (an
    # interrupted run leaves the others out).
    plain: dict[str, dict[int, tuple[Any, ...]]] = {b: {} for b in backends}
    checking: list[tuple[str, int]] = []
    rerun: dict[int, set[str]] = {}
    rechecked: dict[int, dict[str, tuple[Any, ...]]] = {}
    # Per point certified by the cross-check, the agreement its references
    # reached (their largest relative difference): a measured precision
    # wider than the cross-check's tolerance is read to it (``_measured_rows``).
    agreed: dict[int, Fraction] = {}
    # Per backend and point certified by its checks (one backend, or the
    # checked re-run), that reference's own estimate: what it certifies.
    certified: dict[str, dict[int, float]] = {b: {} for b in backends}
    # Per backend the references served with lanes without accuracy evidence
    # (``_ReferenceUncertified``), by point: added to their records as the
    # sixth entry once every stream is back.
    reference_unsure: dict[str, dict[int, str]] = {b: {} for b in backends}
    # Per backend the reference tasks (plain or checked) submitted and not
    # harvested: a point may still need its checked re-run.
    flight = dict.fromkeys(backends, 0)
    # The records and seconds of every other measured precision's stream.
    ext: dict[str, dict[str, dict[int, tuple[Any, ...]]]] = {
        m.stream: {b: {} for b in backends} for m in extras
    }
    ext_seconds = {m.stream: dict.fromkeys(backends, 0.0) for m in extras}

    def live() -> list[str]:
        return [b for b in backends if b not in refused]

    def decide(index: int) -> None:
        """Certify or queue point ``index``'s references once every live
        backend's plain reference is back."""

        alive = live()
        if any(index not in plain[b] for b in alive):
            return
        found = {b: plain[b].pop(index) for b in alive}
        disposition, spread = _agreement_dispositions(
            {b: None if found[b][1] is None else found[b][1][0][0] for b in alive},
            {b for b in alive if index in reference_unsure[b]},
        )
        if spread is not None:
            agreed[index] = spread
        for backend in alive:
            record = found[backend]
            if disposition[backend] == "none":
                dd[backend][index] = (*record, None)
            elif disposition[backend] == "agreement":
                dd[backend][index] = (*record, "agreement")
            else:
                checking.append((backend, index))
                rerun.setdefault(index, set()).add(backend)
                tracker.queue_checked(backend, 1)

    def settle(index: int) -> None:
        """Certify or drop point ``index``'s re-run references once every
        live backend's re-run of it is back: each by its own checks, and
        none when two that pass differ by more than binary64's unit
        roundoff (their estimates are contradicted)."""

        alive = [b for b in sorted(rerun.get(index, ())) if b not in refused]
        found = rechecked.get(index, {})
        if any(b not in found for b in alive):
            return
        rerun.pop(index, None)
        rechecked.pop(index, None)
        values = [found[b][1][0][0] for b in alive if found[b][1] is not None]
        spread = max(
            (
                abs(a - b) / max(abs(a), abs(b))
                for k, a in enumerate(values)
                for b in values[k + 1 :]
                if not _references_agree(a, b, _UNIT_ROUNDOFF)
            ),
            default=None,
        )
        for backend in alive:
            record = found[backend]
            if record[1] is None:
                dd[backend][index] = (*record[:4], None)
            elif spread is not None:
                from pyamplicol.loops.v2.stability import relative_text

                dd[backend][index] = (
                    index,
                    None,
                    "failed: stability checks: the backends' checked references "
                    f"differ by {relative_text(float(spread), '.1e')}, above "
                    f"binary64's unit roundoff {float(_UNIT_ROUNDOFF):.1e}",
                    None,
                    None,
                )
            else:
                dd[backend][index] = (*record[:4], "checks")
                if len(record) > 4:
                    certified[backend][index] = record[4]

    def stream_rate(key: tuple[str, str]) -> float:
        rate = tracker.stream_cost(key)
        return rate if rate is not None else priors.get(key, 0.0)

    def waiting() -> bool:
        """Whether reference work is in flight (a checked re-run may follow)."""

        return any(flight[b] for b in live())

    def checked_tasks() -> list[tuple[Any, ...]]:
        """The queued checked re-runs, per backend in tasks of about
        TASK_SECONDS (``task_points`` at the measured cost of a checked
        reference, else of a plain one)."""

        tasks: list[tuple[Any, ...]] = []
        by_backend: dict[str, list[int]] = {}
        for backend, index in checking:
            if backend not in refused:
                by_backend.setdefault(backend, []).append(index)
        checking.clear()
        for backend, indices in by_backend.items():
            rate = tracker.checked_rate(backend)
            size = task_points(
                rate
                if rate is not None
                else stream_rate((backend, stream_of("reference"))),
                len(indices),
            )
            for at in range(0, len(indices), size):
                chosen = indices[at : at + size]
                flight[backend] += 1
                tasks.append(
                    (
                        backend,
                        _REFERENCE_CHECKED,
                        chosen,
                        [inputs[i][0] for i in chosen],
                        [inputs[i][1] for i in chosen],
                    )
                )
        return tasks

    def next_tasks() -> list[tuple[Any, ...]] | None:
        """The next tasks: the queued checked re-runs first, then the next
        chunk's (every backend and precision); an empty list while nothing
        can be admitted yet but reference work is in flight (it may queue
        re-runs); None when the planned points are submitted (or the
        deadline is near) and nothing more can follow."""

        if checking:
            return checked_tasks()
        if pending:
            start, end = pending.pop(0)
        elif deadline is not None:
            alive = live()
            unmeasured = any(
                tracker.rate((b, p)) is None for b in alive for p in precisions
            ) or any(
                tracker.checking[b] and tracker.checked_rate(b) is None for b in alive
            )
            if unmeasured and waiting():
                # No work beyond what is admitted on a stream whose cost is
                # not measured yet: ask again when a task is back.
                return []
            per_point = sum(stream_rate((b, p)) for b in backends for p in precisions)
            queued = sum(
                max(submitted[0] - tracker.done[(b, p)], 0) * stream_rate((b, p))
                for b in backends
                for p in precisions
            ) + sum(
                tracker.checking[b] * (tracker.checked_rate(b) or 0.0) for b in alive
            )
            size = affordable_chunk(
                now=time.perf_counter(),
                deadline=deadline,
                queued_seconds=queued,
                point_seconds=per_point,
                workers=workers,
                points=submitted[0],
                lanes=DEFAULT_LANES,
                stream_seconds=max(
                    (stream_rate((b, p)) for b in backends for p in precisions),
                    default=0.0,
                ),
                tail_seconds=tail_seconds,
            )
            if not size:
                return [] if waiting() else None
            start, end = submitted[0], submitted[0] + size
        else:
            return [] if waiting() else None
        batch, scales = draw(start, end - start)
        submitted[0] = max(submitted[0], end)
        tracker.set_points(max(submitted[0], count if deadline is None else 0))
        indices = list(range(start, end))
        tasks = []
        alone = len(live()) == 1
        for backend in backends:
            if backend in refused:
                continue
            for precision in precisions:
                # About TASK_SECONDS of work per task: an expensive point is
                # a task of its own, spread over the workers.
                size = task_points(stream_rate((backend, precision)), len(indices))
                kind = precision
                if cross and (
                    precision == "reference"
                    or (unified is not None and precision == unified.stream)
                ):
                    flight[backend] += math.ceil(len(indices) / size)
                    # One backend: no cross-check, every reference checked.
                    kind = _REFERENCE_CHECKED if alone else precision
                for at in range(0, len(indices), size):
                    tasks.append(
                        (
                            backend,
                            kind,
                            indices[at : at + size],
                            batch[at : at + size],
                            scales[at : at + size]
                            if isinstance(scales, list)
                            else scales,
                        )
                    )
        return tasks

    def harvest(backend: str, precision: str, records: Any, wall: float) -> None:
        if precision != "binary64" and precision not in ext:
            # A reference record's lanes served without accuracy evidence:
            # kept apart (``reference_unsure``), the record as before. A
            # checked re-run's record replaces the plain one's.
            stripped = []
            for record in records:
                if record and isinstance(record[-1], _ReferenceUncertified):
                    reference_unsure[backend][record[0]] = str(record[-1])
                    record = record[:-1]
                else:
                    reference_unsure[backend].pop(record[0], None)
                stripped.append(record)
            records = stripped
        if precision in ext:
            # A measured precision's stream: its own records and time.
            truth_stream = unified is not None and precision == unified.stream
            if truth_stream and cross:
                flight[backend] = max(flight[backend] - 1, 0)
            if backend in refused:
                return
            for record in records:
                ext[precision][backend][record[0]] = record
                tasks_of[backend][(precision, record[0])] = (wall, len(records))
            ext_seconds[precision][backend] += wall
            tracker.add(backend, precision, len(records), wall)
            if truth_stream:
                # Also the reference: its accepted values (a best-effort
                # or failed point has none, as the plain evaluation).
                references = [_reference_of(record) for record in records]
                for record in records:
                    unsure = record[10] if len(record) > 10 else None
                    if unsure and record[1] is not None:
                        reference_unsure[backend][record[0]] = unsure
                    else:
                        reference_unsure[backend].pop(record[0], None)
                for record in references:
                    tasks_of[backend][("reference", record[0])] = (
                        wall,
                        len(records),
                    )
                    if not cross:
                        dd[backend][record[0]] = (*record, None)
                    else:
                        plain[backend][record[0]] = record
                if cross:
                    for record in references:
                        decide(record[0])
            return
        if precision != "binary64" and cross:
            flight[backend] = max(flight[backend] - 1, 0)
        if backend in refused:
            return
        if (
            precision == _REFERENCE_CHECKED
            and records
            and (("reference", records[0][0]) in tasks_of[backend])
        ):
            # The checked re-run of references the cross-check did not
            # certify: its time is the reference stream's, its points are not.
            for record in records:
                rechecked.setdefault(record[0], {})[backend] = record
                tasks_of[backend][("checked", record[0])] = (wall, len(records))
            for record in records:
                settle(record[0])
            seconds[backend] += wall
            tracker.add_checked(backend, len(records), wall)
            return
        if precision == "binary64":
            for record in records:
                b64[backend][record[0]] = record
        elif not cross:
            for record in records:
                dd[backend][record[0]] = (*record, None)
        elif precision == _REFERENCE_CHECKED:
            # One live backend: the reference itself ran checked.
            for record in records:
                dd[backend][record[0]] = (
                    *record[:4],
                    None if record[1] is None else "checks",
                )
                if record[1] is not None and len(record) > 4:
                    certified[backend][record[0]] = record[4]
        else:
            for record in records:
                plain[backend][record[0]] = record
        stream = "binary64" if precision == "binary64" else "reference"
        for record in records:
            tasks_of[backend][(stream, record[0])] = (wall, len(records))
        if stream == "reference" and cross:
            for record in records:
                decide(record[0])
        seconds[backend] += wall
        if precision == "binary64":
            tracker.flag(
                backend,
                sum(
                    1
                    for record in records
                    if record[1] is None
                    or record[4] == "best-effort"
                    or (record[2] is not None and record[2].relative > target)
                ),
            )
        tracker.add(backend, stream, len(records), wall)

    def refuse(backend: str, error: Exception) -> None:
        if backend not in refused:
            refused[backend] = str(error)
            tracker.drop(backend)
            # The points waiting for its reference or its re-run: decided
            # without it.
            for index in sorted(set().union(*(plain[b] for b in live()))):
                decide(index)
            for index in sorted(rerun):
                settle(index)

    def tick() -> None:
        tracker.emit(heartbeat=True)
        if monitor is not None:
            monitor.tick()

    interrupted = False
    # Set by the pool at the first Ctrl-C: a second one aborts.
    first_interrupt: list[bool] = []
    first_point = draw(0, 1)
    warm = (first_point[0][0], inputs[0][1])
    try:
        if workers == 1:
            if runtime is None:
                # Released by the caller (``profile_loop``): loaded again.
                from pyamplicol.api import Runtime

                runtime = Runtime.load(
                    artifact, process=evaluation.process, **(load_options or {})
                )
            _WORKER.update(
                runtime=runtime,
                warm=warm,
                checks=checks,
                defaulted=defaulted,
                truth=truth,
                cold=cold,
                measured={m.stream: m.options for m in extras},
                streams=precisions,
            )
            try:
                _warm_up(backends, cold)
                while (tasks := next_tasks()) is not None:
                    for backend, precision, indices, batch, scales in tasks:
                        if backend in refused:
                            continue
                        try:
                            harvest(
                                *_stability_task(
                                    backend, precision, indices, batch, scales
                                )
                            )
                        except ReductionUnavailable as error:
                            refuse(backend, error)
                        tick()
            finally:
                _WORKER.clear()
        else:
            interrupted = _run_pool(
                workers,
                (
                    str(artifact),
                    physics.process_id,
                    dict(load_options or {}),
                    warm,
                    checks,
                    tuple(backends),
                    truth,
                    cold,
                    defaulted,
                    {m.stream: m.options for m in extras},
                    precisions,
                ),
                next_tasks,
                harvest,
                refuse,
                tick,
                first_interrupt,
                probe=probe,
            )
    except KeyboardInterrupt:
        interrupted = True
        if first_interrupt:
            # The second Ctrl-C: abort, no tables.
            raise
    finally:
        tracker.close(interrupted)
        if current is not None:
            current[0] = None
    count = submitted[0]
    rows = []
    complete: dict[str, int] = {}
    # The evaluated points of each backend: a binary64 record with its
    # reference, or one without a binary64 value.
    # The points of the first listed precision (binary64's, or the first
    # other one's when binary64 is not listed).
    base = b64 if binary64_listed else ext[extras[0].stream]
    evaluated = {
        backend: {
            index
            for index, record in base[backend].items()
            if not truth.present or index in dd[backend] or record[1] is None
        }
        for backend in backends
        if backend not in refused
    }
    # An interrupted run compares the backends like for like: the statistics
    # are taken over the points every backend evaluated (a complete run
    # evaluates them all).
    common = set.intersection(*evaluated.values()) if evaluated else set()
    # The references served with lanes without accuracy evidence: the reasons
    # as their record's sixth entry (``_sample_records``,
    # ``_with_reference_uncertified``).
    for backend, unsure in reference_unsure.items():
        for index, reasons in unsure.items():
            found = dd.get(backend, {}).get(index)
            if found is not None and found[1] is not None:
                head = (*found[:5], *(None,) * (5 - len(found[:5])))
                dd[backend][index] = (*head, reasons)
    # The statistics and the per-sample records take seconds on large runs
    # (a few per 10^4 points): a phase of their own, never a silent tail.
    from pyamplicol.loops.v2.reduction import NAMES

    collecting = (
        None
        if monitor is None
        else monitor.phase(
            "statistics",
            "Stability statistics and per-sample records",
            len(backends),
            unit="backends",
        )
    )
    for position, backend in enumerate(backends):
        name = NAMES.get(backend, backend)
        size = len(evaluated.get(backend, ()))

        def encoded(
            done: int, name: str = name, size: int = size, position: int = position
        ) -> None:
            if collecting is not None:
                collecting.update(
                    position,
                    len(backends),
                    eta=None,
                    step=f"{name}: records {done}/{size}",
                    heartbeat=True,
                )

        if collecting is not None:
            collecting.update(
                position,
                len(backends),
                eta=None,
                step=f"{name}: {size} points",
            )
        if backend in refused:
            rows.append(BackendStability(backend, refusal=refused[backend]))
            continue
        complete[backend] = len(evaluated[backend])
        if not binary64_listed:
            # Not listed: no binary64 rows (the other precisions' are).
            continue
        indices = sorted(common if interrupted else evaluated[backend])
        samples = (
            _sample_records(
                backend,
                indices,
                b64[backend],
                dd[backend],
                target,
                tasks_of[backend],
                truth,
                encoded,
            )
            if records
            else None
        )
        if not truth.present:
            rows.append(
                replace(
                    _estimate_statistics(
                        backend,
                        [b64[backend][i][1] for i in indices],
                        [b64[backend][i][2] for i in indices],
                        target,
                        seconds[backend],
                        reasons=[b64[backend][i][3] for i in indices],
                        statuses=[b64[backend][i][4] for i in indices],
                        indices=indices,
                        rescues=[_rescue_of(b64[backend][i]) for i in indices],
                    ),
                    samples=samples,
                )
            )
            rows[-1] = _with_reduction_rescues(rows[-1], b64[backend])
            continue
        rows.append(
            replace(
                _backend_statistics(
                    backend,
                    [b64[backend][i][1] for i in indices],
                    [b64[backend][i][2] for i in indices],
                    [dd[backend][i][1] if i in dd[backend] else None for i in indices],
                    target,
                    seconds[backend],
                    reasons=[b64[backend][i][3] for i in indices],
                    reference_reasons=[
                        dd[backend][i][2] if i in dd[backend] else None for i in indices
                    ],
                    statuses=[b64[backend][i][4] for i in indices],
                    uncertainties=[b64[backend][i][5] for i in indices],
                    indices=indices,
                    rescues=[_rescue_of(b64[backend][i]) for i in indices],
                ),
                samples=samples,
            )
        )
        rows[-1] = _with_reference_uncertified(
            _with_reduction_rescues(rows[-1], b64[backend]), dd[backend]
        )

    measured_rows = {
        m.stream: _measured_rows(
            m,
            backends,
            refused,
            {b: sorted(common if interrupted else evaluated[b]) for b in evaluated},
            ext[m.stream],
            dd,
            target,
            ext_seconds[m.stream],
            tracker,
            truth=truth,
            tasks=tasks_of,
            records_wanted=records,
            agreed=agreed,
            certified=certified,
        )
        for m in extras
    }
    from pyamplicol.reporting.raw_json import encode_rows

    sampled = sorted(set().union(*evaluated.values())) if evaluated else []
    if collecting is not None:
        collecting.end(f"{len(sampled)} points")
    rows = [
        row
        if row.refusal is not None
        else replace(
            row,
            precision_timing={
                ("binary64" if precision == "binary64" else truth.arithmetic): {
                    "points": tracker.done[(row.backend, stream_of(precision))],
                    "seconds": tracker.seconds[(row.backend, stream_of(precision))],
                    "seconds_per_point": tracker.stream_cost(
                        (row.backend, stream_of(precision))
                    ),
                    **(
                        {"digits": truth.digits}
                        if precision != "binary64" and truth.digits is not None
                        else {}
                    ),
                }
                for precision in _precisions(truth)
                if precision != "binary64" or binary64_listed
            },
        )
        for row in rows
    ]
    setup = {
        "points": count,
        "requested": count,
        "measured_points": (
            len(common) if interrupted else min(complete.values(), default=0)
        ),
        "evaluated_per_backend": complete,
        "interrupted": interrupted,
        "n_core": workers,
        # The workers' memory rule (``_memory_record``; None: not measured).
        "memory": None
        if memory is None
        else {**memory, "workers": int(memory.get("probed_workers", workers))},
        "seed": space.seed,
        "sqrt_s": sqrt_s[0] if sqrt_s else space.sqrt_s,
        # The cuts of the drawn points (None: no cut) and the legs (from 0, in
        # the process's order) they act on.
        "cuts": {
            **space.cuts.record(),
            "legs": list(cut_legs[0]) if cut_legs else [],
        },
        "generator": (
            "flat (RAMBO on diet), beams along +-z in the centre-of-mass frame"
        ),
        "measured": "finite part of the total (sum over the squared-order sectors)",
        "reference": (
            f"{_truth_prose(truth)} re-run of the same point on the same backend"
            + (
                ", certified by another backend's agreeing reference or by its "
                "re-run with the recommended stability checks"
                if cross
                else ""
            )
            if truth.present
            else "none (estimates only)"
        ),
        "ground_truth": truth.record(),
        "estimate": (
            "the binary64 call's own estimate with "
            + (
                "evaluation.stability_checks"
                if configured is not None
                else DEFAULT_ESTIMATE
            )
        ),
        "target": target,
        "binary64_digits": BINARY64_DIGITS,
        "percentiles": [float(p) for p in PERCENTILES],
        **(
            {}
            if tuple(bench.measured_precisions) == DEFAULT_MEASURED_PRECISIONS
            else {
                # Every measured precision in the order given, each with
                # its rows (binary64's are ``stability``).
                "measured_precisions": [
                    {
                        **m.record(),
                        "stability": None
                        if m.stream == "binary64"
                        else measured_rows[m.stream],
                    }
                    for m in measured_precisions(bench.measured_precisions, truth)
                ]
            }
        ),
        "mu_r2": evaluation.mu_r2 or "s of the initial state",
        "deadline": deadline is not None,
        # Every point's inputs, columnar (``INPUT_FIELDS``): point i is
        # drawn from seed + i (``generator``, ``sqrt_s``); its binary64
        # momenta (E, px, py, pz per leg, in the process's order) and mu_r2.
        "samples": (
            {
                "fields": list(INPUT_FIELDS),
                "rows": encode_rows(
                    (
                        [
                            index,
                            [[float(x) for x in leg] for leg in inputs[index][0]],
                            float(inputs[index][1]),
                        ]
                        for index in sampled
                        if index in inputs
                    ),
                    len(sampled),
                ),
            }
            if records
            else None
        ),
    }
    return tuple(rows), setup


def _reference_of(record: tuple[Any, ...]) -> tuple[Any, ...]:
    """A reference record (index, [[exact total]] or None, reason, its exact
    text) from the ground truth's own measured stream's record: its value
    only where it was accepted (a best-effort or failed point has none, as
    the plain reference evaluation keeps it)."""

    index, value, _, reason, status = record[:5]
    accepted = value is not None and status != "best-effort"
    text = record[8] if len(record) > 8 else None
    return (index, value if accepted else None, reason, text if accepted else None)


def measured_prior(
    cost: Any, precision: MeasuredPrecision, evaluations: int = 2
) -> float:
    """A lower bound of a measured precision's stability point before its
    stream is measured (``profile_plan.Costs``): the ground truth's own its
    checks' ``evaluations`` per point (``checked_evaluations``) of a plain
    reference point, one below it a checked binary64 point (a wider carrier
    costs at least the binary64 pass)."""

    if precision.estimates_only and cost.stability_reference is not None:
        return evaluations * float(cost.stability_reference)
    return float(cost.stability)


def checked_evaluations(checks: Any) -> int:
    """The evaluations per point of the stability checks ``checks``
    (``VirtualStabilityChecks``; None: the default set, one rotation): the
    point and each rotation (the other checks cost more, not counted: a
    lower bound)."""

    rotations = 1 if checks is None else int(getattr(checks, "rotations", 1) or 0)
    return 1 + rotations


def _measured_rows(
    precision: MeasuredPrecision,
    backends: Sequence[str],
    refused: Mapping[str, str],
    indices: Mapping[str, Sequence[int]],
    records: Mapping[str, Mapping[int, tuple[Any, ...]]],
    references: Mapping[str, Mapping[int, tuple[Any, ...]]],
    target: float,
    seconds: Mapping[str, float],
    tracker: Any,
    *,
    truth: GroundTruth,
    tasks: Mapping[str, Mapping[tuple[str, int], tuple[float, int]]],
    records_wanted: bool,
    agreed: Mapping[int, Fraction] | None = None,
    certified: Mapping[str, Mapping[int, float]] | None = None,
) -> tuple[BackendStability, ...]:
    """The stability rows of a measured precision other than binary64
    (``MeasuredPrecision``), per backend over the points binary64's rows
    read (``indices``) that its stream evaluated: against the ground truth
    (``references``), or its estimates only; digits capped at its ``cap``;
    its stream's timing as its ``precision_timing``."""

    rows: list[BackendStability] = []
    for backend in backends:
        if backend in refused:
            rows.append(BackendStability(backend, refusal=refused[backend]))
            continue
        own = records[backend]
        chosen = [i for i in indices.get(backend, ()) if i in own]
        caps = _point_caps(
            precision,
            truth,
            chosen,
            agreed or {},
            (certified or {}).get(backend, {}),
        )
        common: dict[str, Any] = {
            "reasons": [own[i][3] for i in chosen],
            "statuses": [own[i][4] for i in chosen],
            "indices": chosen,
            "rescues": [_rescue_of(own[i]) for i in chosen],
            "cap": [caps[i] for i in chosen]
            if not precision.estimates_only
            else precision.cap,
        }
        if precision.estimates_only:
            row = _estimate_statistics(
                backend,
                [own[i][1] for i in chosen],
                [own[i][2] for i in chosen],
                target,
                seconds[backend],
                **common,
            )
        else:
            known = references[backend]
            row = _backend_statistics(
                backend,
                [own[i][1] for i in chosen],
                [own[i][2] for i in chosen],
                [known[i][1] if i in known else None for i in chosen],
                target,
                seconds[backend],
                reference_reasons=[known[i][2] if i in known else None for i in chosen],
                uncertainties=[own[i][5] for i in chosen],
                **common,
            )
        key = (backend, precision.stream)
        samples = (
            _sample_records(
                backend,
                chosen,
                own,
                {} if precision.estimates_only else references[backend],
                target,
                tasks[backend],
                GroundTruth("none") if precision.estimates_only else truth,
                stream=precision.stream,
                cap=caps if not precision.estimates_only else precision.cap,
            )
            if records_wanted
            else None
        )
        rows.append(
            replace(
                _with_reduction_rescues(row, own)
                if precision.estimates_only
                else _with_reference_uncertified(
                    _with_reduction_rescues(row, own), references[backend]
                ),
                samples=samples,
                precision_timing={
                    precision.arithmetic: {
                        "points": tracker.done[key],
                        "seconds": tracker.seconds[key],
                        "seconds_per_point": tracker.stream_cost(key),
                        **(
                            {"digits": precision.digits}
                            if precision.digits is not None
                            else {}
                        ),
                    }
                },
            )
        )
    return tuple(rows)


def _point_caps(
    precision: MeasuredPrecision,
    truth: GroundTruth,
    indices: Sequence[int],
    agreed: Mapping[int, Fraction],
    certified: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """Each point's digits cap of a measured precision read against the
    truth: ``precision.cap``, and where that is what the truth certifies
    (``capped_by_truth``: wider than the cross-check's 2^-106) what this
    point's reference certifies, at most the precision's own range and the
    truth's: the agreement the backends' references reached on it, or (one
    backend, or the checked re-run) that reference's own checked estimate
    (``certified``), which may certify less than 2^-106. An arbitrary
    precision measured against a wider one is read to what its reference is
    known to."""

    caps = dict.fromkeys(indices, precision.cap)
    if not precision.capped_by_truth:
        return caps
    top = min(
        precision_digits(precision.arithmetic, precision.digits),
        precision_digits(truth.arithmetic, truth.digits),
    )
    checked = certified or {}
    for index in indices:
        if index in agreed:
            spread = float(agreed[index])
        elif index in checked:
            spread = checked[index]
        else:
            continue
        reached = top if spread <= 0.0 else -math.log10(spread)
        caps[index] = min(top, reached)
    return caps


def _runtime_causes(
    rows: Sequence[BackendRuntime], truth: GroundTruth
) -> tuple[list[tuple[str, str, str, str]], dict[tuple[str, ...], int]]:
    """The failed points of runtime ``rows`` (``failure_causes``' items and
    totals): best-effort, no value and lane rescue of the timed calls, and
    the points without a value of their reference timing."""

    causes: list[tuple[str, str, str, str]] = []
    totals: dict[tuple[str, ...], int] = {}
    for row in rows:
        profile = row.profile
        if profile is None:
            continue
        totals[(row.backend, "runtime")] = profile.batch * profile.samples
        for kind, status, reasons in (
            ("best-effort", "best-effort", profile.best_effort_reasons),
            ("no value", "failed", profile.failure_reasons),
            ("lane rescue", "success", profile.lane_rescue_reasons),
        ):
            for reason, count in reasons:
                text = _strip_status(reason, status)
                causes += [(row.backend, "runtime", kind, text)] * count
        # The reference timing's points without a value (among its own
        # timed points).
        timing = profile.reference
        if timing is None:
            continue
        for reason, count in timing.failure_reasons:
            status = "best-effort" if reason.startswith("best-effort") else "failed"
            outcome = "best-effort" if status == "best-effort" else "no value"
            totals[(row.backend, "runtime", f"{outcome} in {truth.name}")] = (
                timing.points
            )
            text = _strip_status(reason, status)
            causes += [
                (row.backend, "runtime", f"{outcome} in {truth.name}", text)
            ] * count
    return causes, totals


def _stability_causes(
    rows: Sequence[BackendStability],
) -> tuple[list[tuple[str, str, str, str]], dict[tuple[str, ...], int]]:
    """The failed, best-effort and rescued points of stability ``rows``
    (``failure_causes``' items and totals)."""

    causes: list[tuple[str, str, str, str]] = []
    totals: dict[tuple[str, ...], int] = {}
    for stability_row in rows:
        totals[(stability_row.backend, "stability")] = len(stability_row.per_point)
        for point in stability_row.per_point:
            if point.lane_rescue is not None:
                causes.append(
                    (
                        stability_row.backend,
                        "stability",
                        "lane rescue",
                        point.lane_rescue,
                    )
                )
            if point.reduction_rescue_lanes:
                causes.append(
                    (
                        stability_row.backend,
                        "stability",
                        "reduction rescue",
                        "lanes re-reduced in "
                        f"{point.reduction_rescue_carrier} within the reduction",
                    )
                )
            kind = _CAUSE_KINDS.get(point.status)
            if kind is not None:
                text = _strip_status(
                    point.reason or kind,
                    "best-effort" if kind == "best-effort" else "failed",
                )
                causes.append((stability_row.backend, "stability", kind, text))
    return causes, totals


def _as_reference(profile: Any, precision: MeasuredPrecision) -> Any:
    """A measured precision's runtime profile as a reference timing
    (``benchmarking.PrecisionTiming``): the ground truth's own stream
    timed once, the multiplier of the other precisions' rows."""

    from pyamplicol.benchmarking import PrecisionTiming

    return PrecisionTiming(
        precision.arithmetic,
        profile.batch,
        profile.samples,
        profile.warmup,
        profile.batch * profile.samples,
        profile.seconds_per_point,
        profile.low,
        profile.high,
        profile.failed_points,
        tuple(profile.failure_reasons),
        peak_memory=profile.peak_memory,
        digits=precision.digits,
        prepared_seconds=profile.prepared_seconds,
    )


def _measured_runtime(
    precision: MeasuredPrecision,
    backends: Sequence[str],
    benches: Mapping[str, Any],
    references: Mapping[str, Any],
    truth: GroundTruth,
    profile: Callable[[str, Any, Mapping[str, Any]], Any],
    *,
    fit: bool,
) -> tuple[BackendRuntime, ...]:
    """A measured precision's runtime rows (``profile_loop``): per backend
    ``profile(backend, bench, precision options)`` on that backend's
    binary64 bench (``benches``); the ground truth's timing of its binary64
    row (``references``) attached as the row's reference (its multiplier
    against this precision), none for the ground truth's own precision; a
    refusal a row of its own; the ratio to the fastest."""

    from pyamplicol.api.errors import ReductionUnavailable

    own = truth.present and (precision.arithmetic, precision.digits) == (
        truth.arithmetic,
        truth.digits,
    )
    rows: list[BackendRuntime] = []
    for backend in backends:
        try:
            result = profile(
                backend, benches[backend], {**precision.options, "fit": fit}
            )
        except ReductionUnavailable as error:
            rows.append(BackendRuntime(backend, None, str(error), requested=backend))
            continue
        rows.append(
            BackendRuntime(
                backend,
                replace(result, reference=None if own else references.get(backend)),
                requested=backend,
            )
        )
    served = [
        row.profile.seconds_per_point
        for row in rows
        if row.profile and _served_points(row.profile)
    ]
    fastest = min(served) if served else None
    return tuple(
        replace(
            row,
            ratio=None
            if row.profile is None or not fastest or not _served_points(row.profile)
            else row.profile.seconds_per_point / fastest,
        )
        for row in rows
    )


def _write_samples(
    path: Path,
    rows: tuple[BackendStability, ...],
    setup: Mapping[str, object],
    monitor: _Monitor,
    process: str,
) -> tuple[tuple[BackendStability, ...], Mapping[str, object]]:
    """``--samples-output``: every sample's record to ``path`` (one JSON
    document: ``inputs`` as the setup's ``samples``, ``backends`` each
    backend's as its ``samples``), as its own phase; the report then names
    the file where the records were."""

    from pyamplicol.reporting.raw_json import RawJson, splice

    phase = monitor.phase("samples", f"Writing the per-sample records to {path}")
    fragments: list[RawJson] = []

    def held(value: object) -> object:
        # The pre-encoded rows as placeholders (``raw_json.splice``).
        from pyamplicol.reporting.raw_json import placeholder

        if isinstance(value, RawJson):
            fragments.append(value)
            return placeholder(len(fragments) - 1)
        if isinstance(value, Mapping):
            return {str(key): held(entry) for key, entry in value.items()}
        return value

    document = {
        "schema": "pyamplicol.profile-samples/1",
        "process": process,
        "seed": setup.get("seed"),
        "sqrt_s": setup.get("sqrt_s"),
        "cuts": setup.get("cuts"),
        "inputs": held(setup.get("samples")),
        "backends": {
            row.backend: held(row.samples) for row in rows if row.samples is not None
        },
    }
    entries = cast("list[dict[str, Any]]", setup.get("measured_precisions") or [])
    if entries:
        # Every other measured precision's records, per backend.
        document["measured_precisions"] = {
            entry["label"]: {
                row.backend: held(row.samples)
                for row in entry["stability"]
                if row.samples is not None
            }
            for entry in entries
            if entry.get("stability") is not None
        }
    text = splice(json.dumps(document, indent=1, allow_nan=False), fragments)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    points = int(cast(int, setup.get("measured_points") or 0))
    phase.end(f"{points} points, {len(text) / 1e6:.1f} MB")
    where = {"output": str(path), "points": points}

    def named(rows: Sequence[BackendStability]) -> tuple[BackendStability, ...]:
        return tuple(
            row if row.samples is None else replace(row, samples=where) for row in rows
        )

    return (
        named(rows),
        {
            **setup,
            "samples": where,
            **(
                {
                    "measured_precisions": [
                        entry
                        if entry.get("stability") is None
                        else {**entry, "stability": named(entry["stability"])}
                        for entry in entries
                    ]
                }
                if entries
                else {}
            ),
        },
    )


def _run_pool(
    workers: int,
    initargs: tuple[Any, ...],
    tasks: Any,
    harvest: Any,
    refuse: Any,
    tick: Any = None,
    first_interrupt: list[bool] | None = None,
    probe: Callable[[float], int] | None = None,
) -> bool:
    """Run the stability tasks on ``workers`` spawned processes
    (``_stability_worker``), handing every finished task to ``harvest``;
    True when Ctrl-C stopped the run. ``tasks()`` gives the next chunk's
    tasks (None: no more; empty: none until a task is back), pulled while
    fewer than twice ``workers`` tasks
    are queued, so every worker stays busy and a deadline is checked chunk
    by chunk; ``tick()`` about every second. On Ctrl-C (or any error) nothing more
    is submitted, the finished tasks are collected and every worker is
    stopped and joined: no process outlives the call. The first Ctrl-C is
    recorded in ``first_interrupt``; a second one, while the run winds
    down, kills the workers and propagates."""

    import concurrent.futures as futures
    import multiprocessing

    from pyamplicol.api.errors import ReductionUnavailable

    context = multiprocessing.get_context("spawn")
    executor = futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_stability_worker,
        initargs=initargs,
    )
    pending: dict[Any, tuple[str, str]] = {}
    interrupted = False
    source = [tasks]
    last_tick = time.perf_counter()
    # ``probe``: the first worker runs its first task alone (the pool spawns
    # a process per submit while none is idle), then ``probe(its peak
    # resident size)`` sets how many may run (``_stability``'s memory rule);
    # fewer than ``workers``: at most that many tasks are queued, so no
    # further process is spawned.
    probing = [probe is not None and workers > 1]
    limit = [workers]

    def queued() -> int:
        if probing[0]:
            return 1
        return 2 * workers if limit[0] >= workers else limit[0]

    def measure() -> None:
        probing[0] = False
        peaks = [
            peak
            for process in list(getattr(executor, "_processes", {}).values())
            if (peak := _process_peak_bytes(process.pid)) is not None
        ]
        if peaks and probe is not None:
            limit[0] = max(1, min(workers, probe(max(peaks))))

    # Tasks of a chunk not submitted yet (while probing: all but one).
    held: list[Any] = []

    def submit(task: Any) -> None:
        backend, precision, indices, batch, scales = task
        future = executor.submit(
            _stability_task, backend, precision, indices, batch, scales
        )
        pending[future] = (backend, precision)

    def refill() -> None:
        while (held or source) and len(pending) < queued():
            if held:
                submit(held.pop(0))
                continue
            chunk = source[0]()
            if chunk is None:
                source.clear()
                return
            if not chunk:
                # Nothing to submit until a task is back.
                return
            # One task at a time into the pool: no process is spawned for
            # tasks that ``queued`` does not admit yet (the probe's first
            # worker runs alone).
            held.extend(chunk)

    try:
        refill()
        while pending or held:
            if not pending:
                refill()
                continue
            done, _ = futures.wait(
                tuple(pending), timeout=0.5, return_when=futures.FIRST_COMPLETED
            )
            if tick is not None and time.perf_counter() - last_tick >= 1.0:
                # The display moves while nothing returns.
                last_tick = time.perf_counter()
                tick()
            for future in done:
                backend, _precision = pending.pop(future)
                try:
                    harvest(*future.result())
                except ReductionUnavailable as error:
                    refuse(backend, error)
            if probing[0] and done:
                measure()
            refill()
    except KeyboardInterrupt:
        interrupted = True
        if first_interrupt is not None:
            first_interrupt.append(True)
        for future in list(pending):
            if future.done() and not future.cancelled():
                backend, _precision = pending.pop(future)
                try:
                    harvest(*future.result())
                except ReductionUnavailable as error:
                    refuse(backend, error)
                except Exception:
                    # A failed task of an interrupted run: its points stay
                    # out of the partial statistics.
                    pass
    finally:
        _stop_pool(executor)
    return interrupted


def _stop_pool(executor: Any) -> None:
    """Stop a process pool at once: cancel what has not started, terminate
    the workers still computing, and join them."""

    processes = list(getattr(executor, "_processes", {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    except KeyboardInterrupt:
        # A Ctrl-C while stopping: kill at once, still no orphan.
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join(timeout=5)
        raise


def _reduction_rescue_of(record: Sequence[Any]) -> tuple[int, str] | None:
    """A binary64 stability record's reduction rescue (its tenth entry:
    ``(lanes, precision)``; absent in a shorter record)."""

    return record[9] if len(record) > 9 else None


def _with_reduction_rescues(
    row: BackendStability, records: Mapping[int, Sequence[Any]]
) -> BackendStability:
    """``row`` with the reduction's own rescue of each of its points (its
    record's, ``_reduction_rescue_of``) and their totals."""

    per_point: list[StabilityPoint] = []
    points = lanes = uncertain = 0
    carriers: dict[str, int] = {}
    for point in row.per_point:
        record = records.get(point.index)
        local = None if record is None else _reduction_rescue_of(record)
        reasons = None if record is None or len(record) <= 10 else record[10]
        if reasons:
            uncertain += 1
            point = replace(point, uncertified=reasons)
        if local is None:
            per_point.append(point)
            continue
        count, carrier = local
        points += 1
        lanes += count
        carriers[carrier] = carriers.get(carrier, 0) + count
        per_point.append(
            replace(
                point,
                reduction_rescue_lanes=count,
                reduction_rescue_carrier=carrier,
            )
        )
    return replace(
        row,
        per_point=tuple(per_point),
        reduction_rescued_points=points,
        reduction_rescued_lanes=lanes,
        reduction_rescue_carriers=tuple(sorted(carriers.items())),
        uncertified_points=uncertain,
    )


def _with_reference_uncertified(
    row: BackendStability, references: Mapping[int, Sequence[Any]]
) -> BackendStability:
    """``row`` with each measured point's reference served with lanes without
    accuracy evidence (its reference record's sixth entry: the reasons)
    marked (``StabilityPoint.reference_uncertified``) and counted
    (``BackendStability.uncertified_reference_points``); every other number
    as it was."""

    per_point: list[StabilityPoint] = []
    count = 0
    for point in row.per_point:
        record = references.get(point.index)
        reasons = None if record is None or len(record) <= 5 else record[5]
        if reasons and point.status != "reference-failed":
            count += 1
            point = replace(point, reference_uncertified=reasons)
        per_point.append(point)
    if not count:
        return row
    return replace(row, per_point=tuple(per_point), uncertified_reference_points=count)


def _rescue_of(record: Sequence[Any]) -> str | None:
    """A binary64 stability record's lane rescue (its eighth entry, absent
    in a shorter record)."""

    return record[7] if len(record) > 7 else None


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


def _estimate_statistics(
    backend: str,
    plain: Sequence[Any],
    estimates: Sequence[Any],
    target: float,
    seconds: float,
    *,
    reasons: Sequence[str | None],
    statuses: Sequence[str],
    indices: Sequence[int],
    rescues: Sequence[str | None] | None = None,
    cap: float = BINARY64_DIGITS,
) -> BackendStability:
    """The statistics of a run without a reference (``--ground-truth
    none``): the estimated digits of every point ranked at the percentiles
    (a point without a binary64 value or estimate at 0 digits, the worst
    end), the points the default checks flag (estimate above ``target``, a
    best-effort value, or no value) and the best-effort points; nothing
    truth-based."""

    digits: list[float] = []
    per_point: list[StabilityPoint] = []
    flagged = failed = best_effort = rescued = 0
    rescues = rescues if rescues is not None else [None] * len(plain)
    for index, value, estimate, reason, status, rescue in zip(
        indices, plain, estimates, reasons, statuses, rescues, strict=True
    ):
        served = status == "best-effort"
        if value is None or (estimate is None and not served):
            failed += 1
            flagged += 1
            digits.append(0.0)
            per_point.append(
                StabilityPoint(
                    index, "binary64-failed", None, None, reason or "no binary64 value"
                )
            )
            continue
        total = float(_total_finite(value))
        estimated = (
            0.0
            if estimate is None
            else stable_digits(_estimated_relative(estimate, total), cap)
        )
        digits.append(estimated)
        best_effort += served
        flagged += served or (
            estimate is not None and float(estimate.relative) > target
        )
        rescued += rescue is not None
        per_point.append(
            StabilityPoint(
                index,
                "best-effort" if served else "measured",
                None,
                estimated,
                reason if served else None,
                lane_rescue=rescue,
            )
        )
    return BackendStability(
        backend,
        None,
        len(digits),
        failed,
        0,
        (),
        tuple(digits),
        percentile_table(digits, [None] * len(digits)),
        flagged,
        0,
        0,
        0,
        seconds,
        tuple(per_point),
        best_effort,
        0,
        rescued,
        estimate_only=True,
    )


def _finite_or_text(value: float | None) -> float | str | None:
    """A float for the JSON: itself when finite, else "inf" / "nan"."""

    if value is None or math.isfinite(value):
        return value
    return str(value)


def _sample_records(
    backend: str,
    indices: Sequence[int],
    b64: Mapping[int, tuple[Any, ...]],
    reference: Mapping[int, tuple[Any, ...]],
    target: float,
    tasks: Mapping[tuple[str, int], tuple[float, int]],
    truth: GroundTruth,
    progress: Callable[[int], None] | None = None,
    *,
    stream: str = "binary64",
    cap: float | Mapping[int, float] = BINARY64_DIGITS,
) -> dict[str, object]:
    """Every sample's complete record of one backend, columnar (the JSON
    dump of ``--stability --json``; its point's inputs are in the setup's
    ``samples``): ``fields`` (``SAMPLE_FIELDS``), ``tasks`` ([seconds,
    points] of each task that evaluated points: no per-sample timing exists
    inside a batched call) and ``rows``, one per point: ``index``;
    ``status`` (the binary64 call's: success, best-effort, failed ...);
    ``value`` (the binary64 total finite part) and ``sectors`` (each
    squared-order sector's); the estimate (null: none): ``relative`` (the
    flag's metric), ``finite_relative`` (the total's estimated relative
    error), ``digits``, ``scale``, ``checks`` (each check's own relative
    budget by name: rotation-1, reduction-rescaled, ...) and ``not_run``;
    ``flagged`` (the default checks flag it: estimate above the target, a
    best-effort value or none); ``best_effort`` and its ``uncertainty``
    (relative); ``reason`` (a refusal or failure); the reference's ``value``
    (the exact decimal text of its total finite part at its precision; null
    when it failed or without a ground truth), ``reason`` and ``certified``
    (an arbitrary-precision reference's certification: "agreement" with
    another backend's, "checks" from its checked re-run; False for a
    reference served with lanes without accuracy evidence, whose reasons are
    the row's last field ``reference_uncertified``; null otherwise);
    ``true_relative_error`` and ``true_digits`` (null without both values);
    ``binary64_task``, ``reference_task`` (the reference's first
    evaluation: plain, or checked with one backend) and
    ``reference_checked_task`` (its checked re-run; null without one),
    indices into ``tasks``; ``lane_rescue`` (the point's lane rescue,
    ``benchmarking.lane_rescue_reason``; null without one);
    ``estimate_digits_uncapped`` and ``true_digits_uncapped`` (the digits
    before the cap ``cap`` of the tables). Values, sectors and the
    reference are exact (binary64 as JSON numbers, shortest round trip; a
    wider measured precision's ``stream``, ``MeasuredPrecision``, as the
    exact decimal text of its carrier's value, its ``binary64_task`` that
    stream's task; ``truth`` absent: no reference fields); the relative
    errors and scales carry 4 significant digits, the digits 3 decimals.
    ``progress(rows)`` as the rows are encoded."""

    from pyamplicol.reporting.raw_json import encode_rows

    task_ids: dict[tuple[float, int], int] = {}

    def task(key: tuple[str, int]) -> int | None:
        found = tasks.get(key)
        if found is None:
            return None
        return task_ids.setdefault(found, len(task_ids))

    def rows() -> Any:
        for index in indices:
            record = b64.get(index)
            if record is None:
                continue
            point_cap = cap if isinstance(cap, (int, float)) else cap[index]
            padded: tuple[Any, ...] = (*record, *(None,) * (11 - len(record)))
            _, value, summary, reason, status, uncertainty, sectors, rescue = padded[:8]
            text = padded[8]
            local = padded[9]
            uncertified = padded[10]
            exact_total = None if value is None else _total_finite(value)
            total = None if exact_total is None else float(exact_total)
            estimate: list[object] = [None] * 6
            uncapped_estimate = uncapped_true = None
            if summary is not None and total is not None:
                finite = _estimated_relative(summary, total)
                uncapped_estimate = _rounded(_uncapped_digits(finite), 3)
                estimate = [
                    _significant(summary.relative),
                    _significant(finite),
                    _rounded(stable_digits(finite, point_cap), 3),
                    _significant(summary.scale),
                    {name: _significant(budget) for name, budget in summary.checks},
                    dict(summary.not_run),
                ]
            served = status == "best-effort"
            flagged = (
                value is None
                or served
                or summary is None
                or float(summary.relative) > target
            )
            reference_value = reference_reason = certified = error = digits = None
            unsure = None
            if truth.present:
                ref = reference.get(index)
                if ref is not None:
                    reference_value = None if len(ref) < 4 else ref[3]
                    reference_reason = ref[2]
                    certified = None if len(ref) < 5 else ref[4]
                    unsure = None if len(ref) < 6 else ref[5]
                    if unsure:
                        # Lanes of the reference served without accuracy
                        # evidence: not certified, whatever its estimate or
                        # agreement says.
                        certified = False
                    if ref[1] is not None and exact_total is not None:
                        exact = relative_error(exact_total, _total_finite(ref[1]))
                        error = _significant(exact)
                        digits = _rounded(stable_digits(exact, point_cap), 3)
                        uncapped_true = _rounded(_uncapped_digits(exact), 3)
            yield [
                index,
                status,
                text if text is not None else _finite_or_text(total),
                None
                if sectors is None
                else [
                    x if isinstance(x, str) else _finite_or_text(float(x))
                    for x in sectors
                ],
                *estimate,
                bool(flagged),
                served,
                _significant(uncertainty),
                reason,
                reference_value,
                reference_reason,
                certified,
                error,
                digits,
                task((stream, index)),
                task(("reference", index)),
                task(("checked", index)),
                rescue,
                uncapped_estimate,
                uncapped_true,
                0 if local is None else local[0],
                None if local is None else local[1],
                uncertified,
                unsure,
            ]

    encoded = encode_rows(rows(), len(indices), progress)
    return {
        "fields": list(SAMPLE_FIELDS),
        "tasks": [[round(seconds, 6), points] for seconds, points in task_ids],
        "rows": encoded,
    }


def _uncapped_digits(relative: float) -> float:
    """-log10(relative error) as measured, not capped (``stable_digits``
    caps): inf for an exact agreement, nan for a non-finite error."""

    if math.isnan(relative):
        return math.nan
    if relative <= 0.0:
        return math.inf
    return -math.log10(relative)


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
    indices: Sequence[int] | None = None,
    rescues: Sequence[str | None] | None = None,
    cap: float | Sequence[float] = BINARY64_DIGITS,
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
    rescues = rescues if rescues is not None else [None] * count
    rescued = 0
    for position, (
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
        # The flat point's index (``indices``: the evaluated points of an
        # interrupted run).
        index = position if indices is None else indices[position]
        point_cap = cap if isinstance(cap, (int, float)) else cap[position]
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
                    reference_reason or "no reference value",
                )
            )
            continue
        # A binary64 total is a float, a wider precision's exact (its
        # relative error formed exactly either way).
        exact_total = _total_finite(value)
        total = float(exact_total)
        true_relative = relative_error(exact_total, _total_finite(exact))
        digits = stable_digits(true_relative, point_cap)
        estimated = (
            None
            if estimate is None
            else stable_digits(_estimated_relative(estimate, total), point_cap)
        )
        true_digits.append(digits)
        estimated_digits.append(estimated)
        rescue = rescues[position]
        rescued += rescue is not None
        per_point.append(
            StabilityPoint(
                index,
                "best-effort" if served else "measured",
                digits,
                estimated,
                reason if served else None,
                None
                if not served or uncertainty is None
                else stable_digits(uncertainty, point_cap),
                rescue,
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
        rescued,
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
