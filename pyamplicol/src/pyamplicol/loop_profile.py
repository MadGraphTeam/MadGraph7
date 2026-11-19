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
  automatic rescue). A
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
whose reference exists and those that failed in binary64 (no
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

import contextlib
import gc
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pyamplicol.benchmarking import VirtualProfileResult
    from pyamplicol.config import RunConfig

# The stability points of a run whose plan has none (never used: a plan
# with the stability run asked for sets its points).
STABILITY_FALLBACK = 5
# The percentiles of the sorted true digits (worst first): 1 per mil, 1 %,
# 10 % and 50 % (the median).
PERCENTILES = (Fraction(1, 1000), Fraction(1, 100), Fraction(1, 10), Fraction(1, 2))
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
    "arbitrary:DIGITS"); "arbitrary" without digits takes the built-in
    rescue's arbitrary-precision level's digits (``loops.v2.stability.
    DEFAULT_RESCUE``, the project's truth carrier)."""

    from pyamplicol.config.models import _ground_truth
    from pyamplicol.loops.v2.stability import DEFAULT_RESCUE

    canonical = _ground_truth(text)
    if canonical in ("double-double", "none"):
        return GroundTruth(canonical)
    _, _, digits = canonical.partition(":")
    if digits:
        return GroundTruth("arbitrary", int(digits))
    default = next(d for a, d in DEFAULT_RESCUE if a == "arbitrary" and d is not None)
    return GroundTruth("arbitrary", default)


def stable_digits(relative_error: float) -> float:
    """-log10(relative error), capped to [0, ``BINARY64_DIGITS``]: an exact
    agreement (or one below binary64's roundoff) counts all of binary64's
    digits, an error of 100 % or more (or a non-finite one) none."""

    if math.isnan(relative_error) or relative_error >= 1.0:
        return 0.0
    if relative_error <= 2.0**-53:
        return BINARY64_DIGITS
    return min(max(-math.log10(relative_error), 0.0), BINARY64_DIGITS)


def relative_error(value: float, reference: Decimal | Fraction | float) -> float:
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
    with its best-effort value, the unchecked reduction's, m-lit
    ux-besteffort; "binary64-failed": no binary64 value at all, 0 true digits
    and no estimate; "reference-failed": no reference value, not
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
    "true_relative_error",
    "true_digits",
    "binary64_task",
    "reference_task",
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
    best-effort value; "no value": no binary64 value at all; "no reference":
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
) -> str:
    """The runtime setup line; ``checked``: the checked mode
    (``benchmark.checked``), else the optimistic pass; ``seed``: of the flat
    points (None: the caller's points); ``threads``: the native threads each
    backend's timed calls report (one: one core, ``one_core``; another count
    is named); ``rows``: the backends' runtimes (their calls, points per
    call, warm-ups and reference points)."""

    from pyamplicol.loops.v2.reduction import NAMES

    reduction = (
        "checked reduction, refused points served best-effort"
        if checked
        else "optimistic: unchecked reduction, no checks"
    )
    points = (
        "points of --momenta" if seed is None else f"distinct flat points (seed {seed})"
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
    warmups = sorted({row.profile.warmup for row in profiled})
    text = (
        f"binary64, finite part, {reduction}, on {core}; {points}, a new batch "
        f"every call; the median of each backend's timed calls"
        + (f" ({calls} points)" if calls else "")
        + (f" after {'/'.join(map(str, warmups))} warm-up" if warmups else "")
    )
    references = [
        row.profile.reference.points
        for row in profiled
        if row.profile.reference is not None
    ]
    if truth is not None and truth.present and references:
        text += (
            f"; the {truth.label} multiplier from "
            + " / ".join(map(str, references))
            + " reference points"
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
        runtime_counts,
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
    for name in ("kinematics_improvement", "kinematics_improvement_binary64"):
        if getattr(evaluation, name) is not None:
            options[name] = getattr(evaluation, name)
    truth = ground_truth(bench.ground_truth)
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
    loading.end(runtime.physics.process_id)
    physics = runtime.physics
    contribution = runtime.contribution_kind
    backends = _backends(evaluation.reduction, contribution)
    want_stability = bench.stability
    want_runtime = bench.runtime or not want_stability
    workers = max(1, min(bench.n_core, usable_cores()))
    # The runtime timing's native threads (--runtime-threads), at most the
    # usable cores (said once when reduced); the stability workers keep one.
    threads = min(int(bench.runtime_threads), usable_cores())
    if threads < bench.runtime_threads:
        _notice(
            f"--runtime-threads {bench.runtime_threads} reduced to {threads}, "
            "the usable cores"
        )
        bench = replace(bench, runtime_threads=threads)
        config = replace(config, benchmark=bench)
    runtime_given = any(
        not is_default(getattr(bench, name))
        for name in ("target_runtime", "minimum_samples", "batch_size", "warmup_runs")
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
        stream; none for a stream dearer than MEASURE_REPEAT_SECONDS)."""

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
        if want_stability:
            future["stability"] = planned.stability_seconds
        return future

    def provisional(costs_map: dict[str, Any], left: float) -> None:
        # The whole command from the costs known or expected so far.
        planned_now, _ = planning(costs_map, monitor.elapsed() + left)
        rest = sum(future_of(planned_now, costs_map).values())
        provisional_end[0] = time.perf_counter() + left + rest

    try:
        costs = _measure(
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
        )
    except KeyboardInterrupt:
        monitor.close(False)
        raise
    planned, stability_start = planning(costs, monitor.elapsed())
    if not planned.feasible:
        # One number, the header's too: the whole command with the minimums.
        if completion is not None:
            asked = f"--completion-time {_duration(completion)}"
        else:
            budgets = {
                "runtime": (
                    f"{_duration(float(bench.target_runtime))} per backend for the "
                    "runtime timing (--target-runtime)"
                    if runtime_given
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
        _notice(
            f"{asked} cannot be met on this process: the minimums (one timed "
            "call of one chunk per backend and its reference point"
            + (
                ", five stability points per backend"
                if want_stability and unmet_stability
                else ""
            )
            + f") take about {_duration(planned.minimum_seconds)} in all; "
            "running the minimums",
            warning=True,
        )
    future = future_of(planned, costs)
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
    current: list[Any] = [None]

    def remaining() -> tuple[float, bool] | None:
        now = current[0].remaining() if current[0] is not None else None
        rest = sum(future.values())
        if now is None and not rest:
            return None
        return (now or 0.0) + rest, True

    monitor.remaining = remaining

    runtime_rows = None
    if want_runtime:
        # The caller's points, cycled; else the worker draws distinct flat
        # points as its calls need them (the same per call on every backend).
        timed = list(points) if points else None
        rows = []
        default = DEFAULTS.get(contribution)
        for position, backend in enumerate(backends):
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
                # runtime's part of it.
                left = completion - monitor.elapsed() - future.get("stability", 0.0)
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
            future.pop(f"runtime:{backend}", None)
            # The planned seconds of this backend's phases after its
            # start-up: the root ETA counts them until they run.
            later: dict[str, float] = {}
            if backend in costs:
                cost = costs[backend]
                call = backend_bench.batch_size * cost.binary64
                warmups = int(backend_bench.warmup_runs)
                later["binary64"] = (
                    (warmup_seconds(call, cost.binary64) + (warmups - 1) * call)
                    if warmups
                    else 0.0
                ) + max(
                    float(backend_bench.target_runtime),
                    max(int(backend_bench.minimum_samples), 1) * call,
                )
                if truth.present and cost.reference is not None:
                    later["reference"] = cost.reference + max(
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
                if truth.arithmetic == "none"
                else {
                    **truth.options,
                    "reduction": backend,
                    "budget": max(reference_budget, 1e-9),
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
                list(costs.values()),
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
            backends,
            load_options=options,
            progress=progress,
            monitor=monitor,
            # A given count only without --completion-time (it overrules
            # them: ``effective_benchmark``).
            planned_points=count if given_points is None else given_points,
            deadline=deadline if given_points is None else None,
            costs=costs,
            current=current,
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
    causes: list[tuple[str, str, str, str]] = []
    totals: dict[tuple[str, ...], int] = {}
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
    complete = not (
        interrupted
        or bool(stability_setup is not None and stability_setup.get("interrupted"))
    )
    if complete and history is not None:
        # The next profile's priors and whole estimate.
        history.store(
            backends,
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
        backends,
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
    ) -> None:
        from pyamplicol.loops.v2.reduction import NAMES

        self.monitor = monitor
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
            "binary64": "binary64 calls",
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
) -> dict[str, Any]:
    """The measurement phase of a time-budgeted profile, on this process's
    loaded runtime, per backend: the binary64 cost per point of an
    optimistic call and the reference's (none without a ground truth), each
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
            per_point = seconds(1, per_point)
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
    report(0)
    for backend in backends:
        name = NAMES.get(backend, backend)
        measured.clear()
        try:
            for precision in precisions:
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
        costs[backend] = Costs(
            binary64,
            reference,
            STABILITY_PRIOR * binary64,
            reference,
            load_seconds + 1.0,
        )
    phase.end(
        ", ".join(
            f"{NAMES.get(b, b)} {_per_point_text(c.binary64)}"
            + (
                ""
                if c.reference is None
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
            if cost.reference is not None:
                self._store.observe(
                    f"cost:{backend}:reference", 1e6 * float(cost.reference)
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
        runtime = Runtime.load(artifact, process=process, **options)
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
                Runtime.load(artifact, process=process, **options),
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
) -> None:
    """Initialise a stability worker process: one thread per pool, Ctrl-C
    left to the parent (which collects what finished and stops the
    workers), the artifact loaded once with the reference precision
    ``truth`` (default double-double; "none": no reference), and warmed up
    on ``backends`` with ``warm`` (a point and its scale, ``_warm_up``;
    the ``cold`` streams not)."""

    import signal

    one_core()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from pyamplicol.api import Runtime

    _WORKER.update(
        runtime=Runtime.load(artifact, process=process, **options),
        warm=warm,
        checks=checks,
        truth=truth or GroundTruth("double-double"),
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
        for precision in _precisions(_WORKER.get("truth")):
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

    with _collector_off():
        begin = time.perf_counter()
        records = _stability_records(backend, precision, indices, batch, scales)
        seconds = time.perf_counter() - begin
    return backend, precision, records, seconds


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
    [[exact total]], reason, its exact decimal text)."""

    state = _WORKER
    common = {"mu_r2": scales, "poles": "none", "reduction": backend}
    if precision == "binary64":
        # A point the backend refuses keeps its best-effort value (the
        # default policy, m-lit ux-besteffort), measured as such.
        result = state["runtime"].evaluate(
            list(batch),
            stability_checks=state["checks"],
            on_failure="best-effort",
            **common,
        )
        records: list[tuple[Any, ...]] = []
        for index, value, estimate, reason, status, uncertainty in zip(
            indices,
            result.values,
            result.stability,
            _reasons(result),
            result.statuses,
            _uncertainties(result),
            strict=True,
        ):
            total = None if value is None else _total_finite(value)
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
                    None if value is None else [float(sector[0]) for sector in value],
                )
            )
    else:
        # The reference counts only accepted values, summed exactly.
        truth = state.get("truth") or GroundTruth("double-double")
        result = state["runtime"].evaluate(
            list(batch), on_failure="return", **truth.options, **common
        )
        records = []
        for index, value, reason in zip(
            indices, result.values, _reasons(result), strict=True
        ):
            exact = None if value is None else _exact_total(value)
            records.append(
                (
                    index,
                    None if exact is None else [[exact]],
                    reason,
                    None if value is None else _exact_text(value),
                )
            )
    return records


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
        self.emit()

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
        known = {key: rate for key in self.done if (rate := self.rate(key)) is not None}
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
            b64 = self.done[(backend, "binary64")]
            part = f"{name} b64 {b64}/{self.points}"
            if "reference" in self.precisions:
                reference = self.done[(backend, "reference")]
                part += f" {self.reference} {reference}/{self.points}"
            parts.append(part)
            if self.monitor is not None:
                self.monitor.set_reading(
                    name,
                    f"{b64}/{self.points}",
                    f"{self.flagged.get(backend, 0)} flagged",
                )
        elapsed = time.perf_counter() - self.begin
        done = sum(self.done.values())
        rate = sum(self.done[(b, "binary64")] for b in self.backends) / max(
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
            points = min((self.done[(b, "binary64")] for b in self.backends), default=0)
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
    binary64 only. Ctrl-C stops the run: the points evaluated so far
    (binary64 and reference both, or no binary64 value) make the
    statistics, ``setup["interrupted"]`` says so; a second Ctrl-C aborts.
    ``current[0]`` holds the tracker while it runs (the profile's ETA).
    ``records``: build every sample's record (``--json``, --samples-output;
    else None); ``tail_seconds``: the serial seconds per point after the run
    (statistics, records), left out of the deadline's time."""

    from pyamplicol.api.errors import ReductionUnavailable
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.cli.flat_points import draw_flat_points
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
    target = (
        evaluation.reduction_target
        if evaluation.reduction_target is not None
        else default_target()
    )
    truth = ground_truth(bench.ground_truth)
    precisions = _precisions(truth)
    count = int(
        planned_points if planned_points is not None else bench.stability_points
    )
    count = max(count, 1)
    workers = max(1, min(bench.n_core, usable_cores()))
    # The flat points as the chunks need them (point i from seed + i), and
    # every point's inputs (the per-point record).
    inputs: dict[int, tuple[Any, Any]] = {}
    sqrt_s: list[float] = []

    def draw(start: int, size: int) -> tuple[list[Any], list[Any] | Any]:
        flat = draw_flat_points(
            artifact,
            process=physics.process_id,
            points=size,
            seed=space.seed + start,
            sqrt_s=space.sqrt_s,
        )
        if not sqrt_s:
            sqrt_s.append(flat.sqrt_s)
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

    def stream_rate(key: tuple[str, str]) -> float:
        rate = tracker.rate(key)
        return rate if rate is not None else priors.get(key, 0.0)

    def next_tasks() -> list[tuple[Any, ...]] | None:
        """The next chunk's tasks (every backend and precision), or None
        when the planned points are submitted or the deadline is near."""

        if pending:
            start, end = pending.pop(0)
        elif deadline is not None:
            per_point = sum(stream_rate((b, p)) for b in backends for p in precisions)
            queued = sum(
                max(submitted[0] - tracker.done[(b, p)], 0) * stream_rate((b, p))
                for b in backends
                for p in precisions
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
                return None
            start, end = submitted[0], submitted[0] + size
        else:
            return None
        batch, scales = draw(start, end - start)
        submitted[0] = max(submitted[0], end)
        tracker.set_points(max(submitted[0], count if deadline is None else 0))
        indices = list(range(start, end))
        tasks = []
        for backend in backends:
            if backend in refused:
                continue
            for precision in precisions:
                # About TASK_SECONDS of work per task: an expensive point is
                # a task of its own, spread over the workers.
                size = task_points(stream_rate((backend, precision)), len(indices))
                for at in range(0, len(indices), size):
                    tasks.append(
                        (
                            backend,
                            precision,
                            indices[at : at + size],
                            batch[at : at + size],
                            scales[at : at + size]
                            if isinstance(scales, list)
                            else scales,
                        )
                    )
        return tasks

    def harvest(backend: str, precision: str, records: Any, wall: float) -> None:
        if backend in refused:
            return
        store = b64 if precision == "binary64" else dd
        for record in records:
            store[backend][record[0]] = record
        for record in records:
            tasks_of[backend][(precision, record[0])] = (wall, len(records))
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
        tracker.add(backend, precision, len(records), wall)

    def refuse(backend: str, error: Exception) -> None:
        if backend not in refused:
            refused[backend] = str(error)
            tracker.drop(backend)

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
            _WORKER.update(runtime=runtime, warm=warm, checks=checks, truth=truth)
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
                ),
                next_tasks,
                harvest,
                refuse,
                tick,
                first_interrupt,
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
    evaluated = {
        backend: {
            index
            for index, record in b64[backend].items()
            if not truth.present or index in dd[backend] or record[1] is None
        }
        for backend in backends
        if backend not in refused
    }
    # An interrupted run compares the backends like for like: the statistics
    # are taken over the points every backend evaluated (a complete run
    # evaluates them all).
    common = set.intersection(*evaluated.values()) if evaluated else set()
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
                    ),
                    samples=samples,
                )
            )
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
                ),
                samples=samples,
            )
        )
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
                    "points": tracker.done[(row.backend, precision)],
                    "seconds": tracker.seconds[(row.backend, precision)],
                    "seconds_per_point": tracker.rate((row.backend, precision)),
                    **(
                        {"digits": truth.digits}
                        if precision != "binary64" and truth.digits is not None
                        else {}
                    ),
                }
                for precision in precisions
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
        "seed": space.seed,
        "sqrt_s": sqrt_s[0] if sqrt_s else space.sqrt_s,
        "generator": (
            "flat (RAMBO on diet), beams along +-z in the centre-of-mass frame"
        ),
        "measured": "finite part of the total (sum over the squared-order sectors)",
        "reference": (
            f"{_truth_prose(truth)} re-run of the same point on the same backend"
            if truth.present
            else "none (estimates only)"
        ),
        "ground_truth": truth.record(),
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
        "inputs": held(setup.get("samples")),
        "backends": {
            row.backend: held(row.samples) for row in rows if row.samples is not None
        },
    }
    text = splice(json.dumps(document, indent=1, allow_nan=False), fragments)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    points = int(cast(int, setup.get("measured_points") or 0))
    phase.end(f"{points} points, {len(text) / 1e6:.1f} MB")
    where = {"output": str(path), "points": points}
    return (
        tuple(
            row if row.samples is None else replace(row, samples=where) for row in rows
        ),
        {**setup, "samples": where},
    )


def _run_pool(
    workers: int,
    initargs: tuple[Any, ...],
    tasks: Any,
    harvest: Any,
    refuse: Any,
    tick: Any = None,
    first_interrupt: list[bool] | None = None,
) -> bool:
    """Run the stability tasks on ``workers`` spawned processes
    (``_stability_worker``), handing every finished task to ``harvest``;
    True when Ctrl-C stopped the run. ``tasks()`` gives the next chunk's
    tasks (None: no more), pulled while fewer than twice ``workers`` tasks
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

    def refill() -> None:
        while source and len(pending) < 2 * workers:
            chunk = source[0]()
            if chunk is None:
                source.clear()
                return
            for backend, precision, indices, batch, scales in chunk:
                future = executor.submit(
                    _stability_task, backend, precision, indices, batch, scales
                )
                pending[future] = (backend, precision)

    try:
        refill()
        while pending:
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
) -> BackendStability:
    """The statistics of a run without a reference (``--ground-truth
    none``): the estimated digits of every point ranked at the percentiles
    (a point without a binary64 value or estimate at 0 digits, the worst
    end), the points the default checks flag (estimate above ``target``, a
    best-effort value, or no value) and the best-effort points; nothing
    truth-based."""

    digits: list[float] = []
    per_point: list[StabilityPoint] = []
    flagged = failed = best_effort = 0
    for index, value, estimate, reason, status in zip(
        indices, plain, estimates, reasons, statuses, strict=True
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
            else stable_digits(_estimated_relative(estimate, total))
        )
        digits.append(estimated)
        best_effort += served
        flagged += served or (
            estimate is not None and float(estimate.relative) > target
        )
        per_point.append(
            StabilityPoint(
                index,
                "best-effort" if served else "measured",
                None,
                estimated,
                reason if served else None,
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
    when it failed or without a ground truth) and ``reason``;
    ``true_relative_error`` and ``true_digits`` (null without both values);
    ``binary64_task`` and ``reference_task`` (indices into ``tasks``).
    Values, sectors and the reference are exact; the relative errors and
    scales carry 4 significant digits, the digits 3 decimals.
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
            padded: tuple[Any, ...] = (*record, *(None,) * (7 - len(record)))
            _, value, summary, reason, status, uncertainty, sectors = padded[:7]
            total = None if value is None else float(_total_finite(value))
            estimate: list[object] = [None] * 6
            if summary is not None and total is not None:
                finite = _estimated_relative(summary, total)
                estimate = [
                    _significant(summary.relative),
                    _significant(finite),
                    _rounded(stable_digits(finite), 3),
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
            reference_value = reference_reason = error = digits = None
            if truth.present:
                ref = reference.get(index)
                if ref is not None:
                    reference_value = None if len(ref) < 4 else ref[3]
                    reference_reason = ref[2]
                    if ref[1] is not None and total is not None:
                        exact = relative_error(total, _total_finite(ref[1]))
                        error = _significant(exact)
                        digits = _rounded(stable_digits(exact), 3)
            yield [
                index,
                status,
                _finite_or_text(total),
                None
                if sectors is None
                else [_finite_or_text(float(x)) for x in sectors],
                *estimate,
                bool(flagged),
                served,
                _significant(uncertainty),
                reason,
                reference_value,
                reference_reason,
                error,
                digits,
                task(("binary64", index)),
                task(("reference", index)),
            ]

    encoded = encode_rows(rows(), len(indices), progress)
    return {
        "fields": list(SAMPLE_FIELDS),
        "tasks": [[round(seconds, 6), points] for seconds, points in task_ids],
        "rows": encoded,
    }


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
