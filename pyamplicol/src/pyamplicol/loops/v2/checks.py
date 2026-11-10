# SPDX-License-Identifier: 0BSD
"""On-demand physics checks of one-loop requests (``pyamplicol check``,
m-lit lit-examples, USER 2026-11-09 ~08:30: "a pyAmpliCol subcommand
`check` (which does ward and pole checks (catani for trees, and zero poles
for loop-induced))").

A check runs on an in-memory slice of the request (nothing is written): the
generation-time probe slice (``induced.probe_request_of``,
``ward.ward_slice``), i.e. the request on the model's full tree kernels with
every width zeroed outside the complex-mass scheme, at seeded phase-space
points of the generation's validation convention lifted to exact momentum
conservation (``induced.probe_point``; every carrier evaluates the same
exact point), with mu_R^2 the squared invariant mass of the initial state
(``induced._point_scale``). Every check reduces on ``induced.
PROBE_REDUCTION`` (NinjaRS) at every carrier. The checks and their
criteria are the existing ones; nothing here is a new threshold:

* **Poles, Born-interfered** (``born_interfered_rows``): per point, squared-order
  sector and pole coefficient the renormalised virtual (poles "all") against
  Catani's prediction of the requested corrections (QCD and QED summed per
  sector, formed natively, ``NativeExecutor(catani=True)``): in binary64 with
  ``stability.RECOMMENDED_CHECKS`` it passes iff |pole - Catani| <=
  ``ESTIMATES`` E + ``CATANI_FLOOR`` |Catani| (``catani_comparison``); in
  double-double (no checks) iff |pole - Catani| <= ``CATANI_DD_FLOOR``
  |Catani| (``catani_dd_comparison``). A sector the operators do not predict
  is listed with its reason (``poles.UNPREDICTED_REASONS``), never as a
  pass; a sector they predict without a pole row at a point is "not run"
  (an applicable check that did not run). A binary64 point whose relative
  estimate exceeds the default target accuracy (``stability.RESCUE_TARGET``,
  the built-in rescue's level criterion) is out of the binary64 scope. When
  binary64 evaluated a point and the double-double run is refused, the
  undecided pole and Ward rows keep binary64's verdict (never a pass) and
  say that double-double was refused.
* **Ward, Born-interfered**: ``ward.ward_check`` on every external gluon and
  photon leg (``ward.gauge_legs``) and both factors (``ward.WARD_FACTORS``),
  its tree-level Born part included, with ``ward.ward_row``'s criteria.
* **Loop-induced**: the generation-time probes' per-point pieces
  (``induced.point_probes``): the finiteness probe (the amplitude's poles
  against the row term floor S_k(h), the pre-cancellation scale) and the
  amplitude-level Ward probe of one leg of every gluon and photon species,
  each within ``induced.ESTIMATE_FACTOR`` times its own estimate.
* **Goldstone** (``induced.goldstone_point``): k.A(V) = eta M_V A(G) where
  the probe serves the slice (``induced.goldstone_applicability``:
  't Hooft-Feynman gauge, an external massive vector with a Goldstone
  partner, a loop-induced amplitude); otherwise a "not applicable" row
  with the reason.

Carriers (``induced.probe_carriers``): binary64 first and double-double
only for a row whose binary64 run fails, is refused or is out of its
criterion's scope; an explicit carrier alone otherwise. The deciding carrier
and the fallback's reason are recorded per row.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# The pole criterion against Catani's operator (moved here from
# tools/loop_benchmark/stored_checks.py, which imports it): each deviation
# within ESTIMATES x E plus CATANI_FLOOR of the prediction (binary64 with
# the recommended checks), or within CATANI_DD_FLOOR of the prediction in
# double-double (precision is lax).
ESTIMATES = 10
CATANI_FLOOR = 1e-12
CATANI_DD_FLOOR = 1e-12

# Row verdicts: only PASS is a pass; NOT_APPLICABLE, NO_PREDICTION and
# NOT_RUN carry their reason and never count as one.
PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "not applicable"
NO_PREDICTION = "no prediction"
NOT_RUN = "not run"
VERDICTS = (PASS, FAIL, NOT_APPLICABLE, NO_PREDICTION, NOT_RUN)

# Process statuses and the exit code of a report (``CheckReport.exit_code``).
PASSED = "passed"
FAILED = "failed"
NOTHING_CHECKED = "nothing checked"
EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_NOTHING_CHECKED = 3

POLE_NAMES = {1: "1/eps", 2: "1/eps^2"}


def catani_comparison(
    values: Sequence[float],
    poles: Sequence[Any],
    absolute: Sequence[float] | None = None,
) -> tuple[list[float], list[float], list[float], bool]:
    """The binary64 pole criterion of one sector: ``values`` the real Laurent
    coefficients (finite, 1/eps, 1/eps^2), ``poles`` Catani's (1/eps,
    1/eps^2) prediction (``PointResult.ir_poles[sector]``), ``absolute`` the
    estimate E per coefficient (None: no estimate). Returns the prediction,
    the deviations |pole - Catani|, their bounds ESTIMATES E + CATANI_FLOOR
    |Catani| and whether both are within them."""

    catani = [complex(poles[0]).real, complex(poles[1]).real]
    deviations: list[float] = []
    bounds: list[float] = []
    ok = True
    for k in (1, 2):
        deviation = abs(values[k] - catani[k - 1])
        bound = CATANI_FLOOR * abs(catani[k - 1])
        if absolute is not None:
            bound += ESTIMATES * absolute[k]
        deviations.append(deviation)
        bounds.append(bound)
        ok = ok and deviation <= bound
    return catani, deviations, bounds, ok


def catani_dd_comparison(
    values: Sequence[float], poles: Sequence[Any]
) -> tuple[list[float], list[float], bool]:
    """The double-double pole criterion of one sector: the relative
    deviations |pole - Catani| / |Catani| of the 1/eps and 1/eps^2
    coefficients and whether both are within ``CATANI_DD_FLOOR``."""

    catani = [complex(poles[0]).real, complex(poles[1]).real]
    errors = [
        abs(values[k] - catani[k - 1]) / max(abs(catani[k - 1]), 1e-300) for k in (1, 2)
    ]
    return catani, errors, all(e <= CATANI_DD_FLOOR for e in errors)


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class CheckRow:
    """One judged check: ``check`` ("poles", "ward" or "goldstone"),
    ``subject`` (a sector and pole coefficient, a leg and factor, or the
    finiteness of every loop order and row), the point index, the carrier
    that decided, the verdict (``VERDICTS``), ``ratio`` the measured value
    over its allowed bound (<= 1 passes: the deviation over ESTIMATES E plus
    the floor, a probe's residue over ten times its estimate),
    ``deviation`` and ``estimate`` the deviation and E over the row's scale
    (where formed): a binary64 pole row's pre-cancellation scale S, a
    double-double pole row's |Catani| (its criterion is relative; no E), a
    Born-interfered Ward row's physical scale (``ward.ward_scales``), a
    probe's row term floor S_k(h); ``fallback`` why the first carrier did
    not decide, ``reason`` why a row is not a verdict."""

    check: str
    subject: str
    point: int | None
    carrier: str | None
    verdict: str
    ratio: float | None = None
    deviation: float | None = None
    estimate: float | None = None
    fallback: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ProcessCheck:
    """The checks of one process: its rows, the slice they ran on and the
    status (``PASSED``: every row with a verdict passed and one did;
    ``FAILED``: a row failed; ``NOTHING_CHECKED``: no row passed or failed,
    or a check that applies did not run)."""

    process: str
    expression: str
    contribution: str
    widths: str
    zeroed_widths: tuple[str, ...]
    reduction: str
    rows: tuple[CheckRow, ...]
    seconds: float
    status: str = ""

    def __post_init__(self) -> None:
        verdicts = {row.verdict for row in self.rows}
        if FAIL in verdicts:
            status = FAILED
        elif PASS in verdicts and NOT_RUN not in verdicts:
            status = PASSED
        else:
            status = NOTHING_CHECKED
        object.__setattr__(self, "status", status)


@dataclass(frozen=True)
class CheckReport:
    """``pyamplicol check``'s result: what was checked (``source``: the run
    card or artifact whose request was regenerated in memory), the point
    count, seed and carriers, every process's checks, and the exit code
    (``EXIT_PASSED`` iff every process passed, ``EXIT_FAILED`` on any
    failure, else ``EXIT_NOTHING_CHECKED``)."""

    source: str
    source_kind: str
    points: int
    seed: int
    precision: str
    carriers: tuple[str, ...]
    processes: tuple[ProcessCheck, ...]
    regeneration: str
    seconds: float
    exit_code: int = EXIT_PASSED

    def __post_init__(self) -> None:
        statuses = {process.status for process in self.processes}
        if FAILED in statuses:
            code = EXIT_FAILED
        elif statuses == {PASSED}:
            code = EXIT_PASSED
        else:
            code = EXIT_NOTHING_CHECKED
        object.__setattr__(self, "exit_code", code)

    @property
    def passed(self) -> bool:
        return self.exit_code == EXIT_PASSED


# ------------------------------------------------------------- the checks


def _sector_name(sector: Any) -> str:
    return " ".join(f"{name}^{power}" for name, power in sector)


def _short(value: float | None) -> float | None:
    """``induced._short`` with an infinite ratio kept a float."""

    from .induced import _short as short

    return short(value, infinite=float("inf"))


def _carrier_name(carrier: Any) -> str:
    return str(carrier.arithmetic)


def _evaluate(executor: Any, point: Any, scale: Any, carrier: Any, checks: Any) -> Any:
    """One point's ``PointResult`` (poles "all"); a refused point raises
    ``plan_export.FailedPoints``."""

    (result,) = executor.evaluate([point], [scale], precision=carrier, stability=checks)
    return result


def _pole_rows_binary64(result: Any, index: int, target: float) -> dict[Any, CheckRow]:
    """The binary64 pole rows of one evaluated point, keyed by (sector, k).
    A row without its criterion (no estimate E) or out of the binary64 scope
    (relative estimate above ``target``) is "not run" with the reason: the
    next carrier decides it, and with binary64 alone it is no pass."""

    estimate = result.stability
    rows: dict[Any, CheckRow] = {}
    reason = None
    if estimate is None:
        reason = "no criterion: the binary64 evaluation carries no estimate E"
    elif float(estimate.relative) > target:
        reason = (
            f"out of the binary64 scope: relative estimate "
            f"{float(estimate.relative):.3g} above the default target "
            f"accuracy {target:g}"
        )
    for sector, laurent in result.components["virtual"].items():
        values = [complex(x).real for x in laurent.coefficients]
        poles = (result.ir_poles or {}).get(sector)
        if poles is None or len(values) != 3:
            continue
        absolute = scale = None
        if estimate is not None:
            absolute = [abs(float(x)) for x in estimate.absolute[sector].coefficients]
            scale = [abs(float(x)) for x in estimate.scale[sector].coefficients]
        _catani, deviations, bounds, _ok = catani_comparison(values, poles, absolute)
        for k in (1, 2):
            deviation, bound = deviations[k - 1], bounds[k - 1]
            ratio = (
                deviation / bound if bound else (0.0 if not deviation else float("inf"))
            )
            verdict = NOT_RUN if reason else PASS if deviation <= bound else FAIL
            over = scale[k] if scale is not None else 0.0
            rows[(sector, k)] = CheckRow(
                "poles",
                f"{_sector_name(sector)} {POLE_NAMES[k]}",
                index,
                "double",
                verdict,
                _short(ratio),
                _short(deviation / over) if over else None,
                _short(absolute[k] / over) if over and absolute is not None else None,
                reason=reason,
            )
    return rows


def _pole_rows_dd(result: Any, index: int) -> dict[Any, CheckRow]:
    """The double-double pole rows of one evaluated point (no checks: the
    relative deviation from Catani's double-double prediction)."""

    rows: dict[Any, CheckRow] = {}
    for sector, laurent in result.components["virtual"].items():
        values = [complex(x).real for x in laurent.coefficients]
        poles = (result.ir_poles or {}).get(sector)
        if poles is None or len(values) != 3:
            continue
        _catani, errors, _ok = catani_dd_comparison(values, poles)
        for k in (1, 2):
            rows[(sector, k)] = CheckRow(
                "poles",
                f"{_sector_name(sector)} {POLE_NAMES[k]}",
                index,
                "double-double",
                PASS if errors[k - 1] <= CATANI_DD_FLOOR else FAIL,
                _short(errors[k - 1] / CATANI_DD_FLOOR),
                _short(errors[k - 1]),
                None,
            )
    return rows


def _unpredicted_rows(
    slice_: Any,
    sectors: Sequence[Any],
    formed: bool,
    index: int,
    judged: Sequence[Any] = (),
) -> list[CheckRow]:
    """A row per sector without a pole row, with its reason, never omitted:
    "no prediction" for the sectors the operators do not predict
    (``poles.UNPREDICTED_REASONS``) or where no prediction could be formed,
    "not run" for a sector they predict that got no pole row at this point
    (``judged``): an applicable check that did not run (exit code 3)."""

    unpredicted = slice_.unpredicted_sectors()
    rows = []
    for sector in sectors:
        verdict = NO_PREDICTION
        if sector in unpredicted:
            reason = (
                f"{unpredicted[sector]}: the requested corrections' operators do "
                "not predict this sector's poles (poles.sector_predictions)"
            )
        elif not formed:
            reason = (
                "no prediction formed: Catani's native operator needs a "
                "full-colour request with two coloured legs (dipoles.correlator)"
            )
        elif sector not in judged:
            verdict = NOT_RUN
            reason = (
                "an applicable check did not run: the operators predict this "
                "sector's poles, but no prediction or pole coefficients were "
                "formed for it at this point"
            )
        else:
            continue
        rows.append(
            CheckRow("poles", _sector_name(sector), index, None, verdict, reason=reason)
        )
    return rows


def ward_applicability(target: Any) -> str | None:
    """None where the Ward rows of a probe slice are checked, else why they
    are not applicable (USER 2026-11-10 ~00:17: with EW corrections and an
    on-shell zero-width external top "it may break some EW gauge relations
    ... ignore ward check there"). One rule on the request's own data,
    decided before any Ward evaluation: its corrections include one other
    than the model's strong correction (``ew_policy.correction_orders``,
    identified by the gluon's vertices, never by an order's name) AND an
    external leg is a massive quark with zero width at the check's
    parameters: a fermion (``ew_policy.FERMION_SPIN``) of colour triplet or
    anti-triplet (|colour| = 3; an antiquark's record carries -3) whose mass
    parameter is not zero and whose width parameter is (``ew_policy.
    is_zero`` on the probe request's fixed parameters: outside the
    complex-mass scheme the probe zeroes every width, ``ew_policy.
    probe_widths``). Every external leg is on shell at a check point
    (``induced.probe_point``). Ward rows are not split by squared-order
    sector, so a pure-QCD-loop sector of such a request is not checked
    either."""

    from .ew_policy import FERMION_SPIN, correction_orders, is_zero, request_corrections

    compiled, request = target.compiled, target.request
    fixed = request.fixed_parameters
    strong = correction_orders(compiled, fixed).strong
    others = sorted(set(request_corrections(compiled, request)) - {strong})
    if not others:
        return None
    particles = {int(record.pdg_code): record for record in compiled.ir.particles}
    parameters = {record.name: record for record in compiled.ir.parameters}
    quarks = [
        int(leg.label)
        for leg in sorted(target.blocks.process.legs, key=lambda x: x.label)
        if (record := particles.get(int(leg.pdg))) is not None
        and int(record.spin) == FERMION_SPIN
        and abs(int(record.color)) == 3
        and not is_zero(str(record.mass), parameters, fixed)
        and is_zero(str(record.width), parameters, fixed)
    ]
    if not quarks:
        return None
    return (
        f"EW corrections ({', '.join(others)}) with an on-shell zero-width "
        f"external massive quark (leg {', '.join(map(str, quarks))}): the "
        "gluon / photon Ward identity of the asymmetric check is not "
        "guaranteed there (USER 2026-11-10); not evaluated, every "
        "squared-order sector alike"
    )


def _not_applicable_ward(legs: Sequence[int], reason: str) -> CheckRow:
    """The one Ward row of a process whose Ward rows are not applicable
    (``ward_applicability``), naming the legs not checked."""

    return CheckRow(
        "ward",
        "external gluons and photons",
        None,
        None,
        NOT_APPLICABLE,
        reason=f"legs {', '.join(map(str, legs))} not checked: {reason}",
    )


def born_interfered_rows(
    target: Any,
    points: Sequence[tuple[Any, Any]],
    carriers: Sequence[Any],
) -> list[CheckRow]:
    """The pole and Ward rows of a Born-interfered probe slice (module
    docstring) at the lifted ``points`` ((point, mu_R^2) pairs)."""

    from . import induced, stability, ward
    from .plan_export import FailedPoints, NativeExecutor

    executor = NativeExecutor(target, inputs="native", forward=False, catani=True)
    target_accuracy = float(stability.RESCUE_TARGET)
    legs = ward.gauge_legs(target)
    excluded = ward_applicability(target) if legs else None
    rows: list[CheckRow] = []
    for index, (point, scale) in enumerate(points):
        # Poles against Catani's operator.
        decided: dict[Any, CheckRow] = {}
        sectors: list[Any] = []
        formed = evaluated = False
        refusal: str | None = None
        for carrier in carriers:
            pending = [key for key, row in decided.items() if row.verdict != PASS]
            if evaluated and not pending:
                break
            double = _carrier_name(carrier) == "double"
            try:
                result = _evaluate(
                    executor,
                    point,
                    scale,
                    carrier,
                    stability.RECOMMENDED_CHECKS if double else None,
                )
            except FailedPoints as error:
                refusal = f"{_carrier_name(carrier)} refused: {error}"
                if evaluated:
                    # The earlier carrier's rows stay (never a pass) and say
                    # that this carrier was refused.
                    for key in pending:
                        decided[key] = _refused(decided[key], refusal)
                continue
            sectors = list(result.components["virtual"])
            formed = result.ir_poles is not None
            found = (
                _pole_rows_binary64(result, index, target_accuracy)
                if double
                else _pole_rows_dd(result, index)
            )
            if not evaluated:
                decided = {
                    key: _with(row, fallback=refusal) if refusal else row
                    for key, row in found.items()
                }
            else:
                for key in pending:
                    if key in found:
                        earlier = decided[key]
                        decided[key] = _with(
                            found[key],
                            fallback=earlier.reason
                            or f"{earlier.carrier} ratio {earlier.ratio}",
                        )
            evaluated = True
            refusal = None
        if not evaluated:
            rows.append(
                CheckRow("poles", "every sector", index, None, FAIL, reason=refusal)
            )
        rows.extend(decided.values())
        rows.extend(
            _unpredicted_rows(
                target, sectors, formed, index, {key[0] for key in decided}
            )
        )
        # Ward identities on every gluon and photon leg.
        if legs and excluded is None:
            rows.extend(_born_ward_rows(executor, point, scale, legs, carriers, index))
    if excluded is not None:
        rows.append(_not_applicable_ward(legs, excluded))
    if not legs:
        rows.append(
            CheckRow(
                "ward",
                "external gluons and photons",
                None,
                None,
                NOT_APPLICABLE,
                reason="the process has no external gluon or photon",
            )
        )
    reason = induced.goldstone_applicability(target) or (
        "the Goldstone-equivalence probe is amplitude-level (loop-induced "
        "requests); a Born-interfered virtual has none"
    )
    rows.append(
        CheckRow(
            "goldstone",
            "external massive vectors",
            None,
            None,
            NOT_APPLICABLE,
            reason=reason,
        )
    )
    return rows


def _with(row: CheckRow, **changes: Any) -> CheckRow:
    from dataclasses import replace

    return replace(row, **changes)


def _refused(row: CheckRow, refusal: str) -> CheckRow:
    """``row`` (an earlier carrier's undecided or failed row) with a later
    carrier's refusal added to its reason; the verdict is kept."""

    reason = refusal if row.reason is None else f"{row.reason}; {refusal}"
    return _with(row, reason=reason)


def _ward_row(row: Mapping[str, Any], carrier: str, index: int) -> CheckRow:
    """A ``ward.ward_row`` record as a check row: the ratio is the largest of
    the criterion's parts over their bounds (<= 1 passes)."""

    from . import ward

    parts = [row["born"] / ward.BORN_BOUNDS[carrier]]
    virtual = max(row["parts"]["virtual"])
    if carrier in ward.VIRTUAL_BOUNDS:
        parts.append(virtual / ward.VIRTUAL_BOUNDS[carrier])
    if row.get("estimate_over") is not None:
        parts.append(max(row["estimate_over"]) / ward.ESTIMATE_FACTOR)
    if row.get("o1_over") is not None:
        parts.append(max(row["o1_over"]))
    reason = None
    if row.get("in_scope") is False:
        reason = (
            f"out of the binary64 scope: E above {ward.BINARY64_SCOPE:g} of the "
            "physical scale"
        )
    elif row["passed"] is None:
        reason = "no criterion: the binary64 evaluation carries no estimate"
    factor = "c = 1" if tuple(row["factor"]) == (1, 0) else "c = i"
    if row["passed"] is True:
        verdict = PASS
    elif not row["born_passed"] or reason is None:
        verdict = FAIL
    else:
        # The virtual part without its binary64 criterion (no estimate, or
        # out of scope) and the tree-level Born part passed: the next
        # carrier decides; with binary64 alone the row is no pass.
        verdict = NOT_RUN
    return CheckRow(
        "ward",
        f"leg {row['leg']}, {factor}",
        index,
        carrier,
        verdict,
        _short(max(parts)),
        _short(virtual),
        _short(max(row["estimate"])) if row.get("estimate") else None,
        reason=reason if row["passed"] is not True else None,
    )


def _born_ward_rows(
    executor: Any,
    point: Any,
    scale: Any,
    legs: Sequence[int],
    carriers: Sequence[Any],
    index: int,
) -> list[CheckRow]:
    from . import ward
    from .plan_export import FailedPoints

    decided: dict[Any, CheckRow] = {}
    fallback: str | None = None
    for carrier in carriers:
        name = _carrier_name(carrier)
        pending = [
            key
            for key in ((leg, factor) for leg in legs for factor in ward.WARD_FACTORS)
            if key not in decided or decided[key].verdict != PASS
        ]
        if not pending:
            break
        found: dict[Any, CheckRow] = {}
        # One batch for every pending row when they are every leg's factors
        # (the physical evaluation shared), else one per leg.
        if len(pending) == len(legs) * len(ward.WARD_FACTORS):
            batches = [(list(legs), list(ward.WARD_FACTORS))]
        else:
            batches = [
                ([leg], [f for other, f in pending if other == leg])
                for leg in sorted({leg for leg, _ in pending})
            ]
        try:
            for batch_legs, factors in batches:
                check = ward.ward_check(
                    executor, point, scale, name, legs=batch_legs, factors=factors
                )
                for row in check["rows"]:
                    found[(row["leg"], tuple(row["factor"]))] = _ward_row(
                        row, name, index
                    )
        except FailedPoints as error:
            fallback = f"{name} refused: {error}"
            for key in pending:
                earlier = decided.get(key)
                decided[key] = (
                    CheckRow(
                        "ward",
                        f"leg {key[0]}",
                        index,
                        name,
                        FAIL,
                        reason=fallback,
                    )
                    if earlier is None
                    else _refused(earlier, fallback)
                )
            continue
        for key in pending:
            row = found[key]
            earlier = decided.get(key)
            if earlier is not None:
                row = _with(
                    row,
                    fallback=earlier.reason
                    or f"{earlier.carrier} ratio {earlier.ratio}",
                )
            decided[key] = row
    return list(decided.values())


def induced_rows(
    target: Any,
    compiled: Any,
    request: Any,
    native: Any,
    points: Sequence[tuple[Any, Any]],
    carriers: Sequence[Any],
    *,
    memo: Any = None,
) -> list[CheckRow]:
    """The finiteness, Ward and Goldstone rows of a loop-induced probe slice
    (``induced.point_probes``, ``induced.goldstone_point``) at the lifted
    ``points``; ``memo`` shares the Goldstone sides' model preparation."""

    from . import induced
    from .plan_export import NativeExecutor

    if target.amplitude.colour_zero:
        return [
            CheckRow(
                check,
                "every loop order",
                None,
                None,
                NOT_APPLICABLE,
                reason="an exact colour zero: nothing to probe",
            )
            for check in ("poles", "ward", "goldstone")
        ]
    legs = induced.probe_legs(target)
    excluded = ward_applicability(target) if legs else None
    probed = legs if excluded is None else []
    executor = NativeExecutor(
        target,
        forward=False,
        inputs="native",
        lanes=induced.probe_lanes(1 + len(probed)),
    )
    applicability = induced.goldstone_applicability(target)
    sides: Mapping[int, Any] = {}
    skipped: Mapping[str, str] = {}
    if applicability is None:
        sides, skipped, _seconds = induced.goldstone_sides(
            target, compiled, request, native, memo=memo
        )
    executors: dict[Any, Any] = {}
    rows: list[CheckRow] = []
    for index, (point, scale) in enumerate(points):
        try:
            floors, _record = induced.probe_floors(executor, point, scale)
        except induced.ProbePointRefused as error:
            rows.append(
                CheckRow(
                    "poles",
                    "finiteness",
                    index,
                    None,
                    FAIL,
                    reason=f"the point's physical floors were refused: {error}",
                )
            )
            continue
        entries, decided = induced.point_probes(
            executor, point, scale, probed, carriers, floors
        )
        for entry, result in zip(entries, decided, strict=True):
            rows.append(_probe_row(entry, result, index))
        if sides:
            results, gentries, _phases, _times = induced.goldstone_point(
                target, sides, point, scale, carriers, floors, executors
            )
            for entry, (_leg, result) in zip(gentries, results, strict=True):
                rows.append(_probe_row(entry, result, index))
    if not legs:
        rows.append(
            CheckRow(
                "ward",
                "external gluons and photons",
                None,
                None,
                NOT_APPLICABLE,
                reason="the process has no external gluon or photon",
            )
        )
    if excluded is not None:
        rows.append(_not_applicable_ward(legs, excluded))
    if applicability is not None:
        rows.append(
            CheckRow(
                "goldstone",
                "external massive vectors",
                None,
                None,
                NOT_APPLICABLE,
                reason=applicability,
            )
        )
    for leg, text in skipped.items():
        rows.append(
            CheckRow("goldstone", f"leg {leg}", None, None, NOT_RUN, reason=text)
        )
    return rows


def _probe_row(entry: Mapping[str, Any], result: Any, index: int) -> CheckRow:
    """A generation-time probe's decided result as a check row."""

    from .induced import ESTIMATE_FACTOR

    probe = entry["probe"]
    check = {"finiteness": "poles", "ward": "ward", "goldstone": "goldstone"}[probe]
    subject = (
        "finiteness (every loop order and row)"
        if probe == "finiteness"
        else f"leg {entry['leg']}"
    )
    fallback = entry.get("fallback")
    if isinstance(fallback, list):
        fallback = fallback[-1]
    fallback_text = None
    if fallback is not None:
        fallback_text = (
            f"{fallback['carrier']} refused: {fallback['detail']}"
            if fallback["reason"] == "refused"
            else f"{fallback['carrier']} ratio {fallback['worst']} times its estimate"
        )
    if isinstance(result, str):
        return CheckRow(
            check,
            subject,
            index,
            entry["carrier"],
            FAIL,
            fallback=fallback_text,
            reason=result,
        )
    deviation = estimate = None
    relative = getattr(result, "relative", None) or {}
    if relative and result.ratios:
        key = max(result.ratios, key=lambda k: result.ratios[k])
        if key in relative:
            deviation, estimate = relative[key]
    return CheckRow(
        check,
        subject,
        index,
        entry["carrier"],
        PASS if result.passed else FAIL,
        _short(result.worst / ESTIMATE_FACTOR),
        _short(deviation),
        _short(estimate),
        fallback=fallback_text,
    )


def check_request(
    process: str,
    expression: str,
    compiled: Any,
    request: Any,
    native: Any,
    momenta: Sequence[Sequence[Sequence[float]]],
    carriers: Sequence[Any],
    *,
    memo: Any = None,
) -> ProcessCheck:
    """The checks of one generation request (``SliceRequest``) at the
    binary64 phase-space points ``momenta`` (module docstring): its probe
    slice is generated in memory, the points lifted once at the digits of
    the last carrier. ``memo`` (the check's ``preparation_cache.
    PreparationMemo``) shares the model preparation of the probe and
    Goldstone slices, as at generation."""

    from .induced import (
        PROBE_REDUCTION,
        _complex_mass,
        _point_scale,
        probe_point,
        probe_request_of,
    )
    from .reduction import running
    from .ward import ward_slice

    start = time.perf_counter()
    with running(PROBE_REDUCTION):
        probe, zeroed = probe_request_of(compiled, request)
        target = ward_slice(compiled, probe, native, memo=memo)
        digits = 2 * max(carrier.working_digits for carrier in carriers)
        initial = sum(1 for leg in target.blocks.process.legs if leg.is_initial)
        points = []
        for momentum in momenta:
            point, _distance = probe_point(target, momentum, digits)
            points.append((point, _point_scale(point, initial)))
        if request.contribution == "loop-induced":
            rows = induced_rows(
                target, compiled, probe, native, points, carriers, memo=memo
            )
        else:
            rows = born_interfered_rows(target, points, carriers)
    return ProcessCheck(
        process,
        expression,
        request.contribution,
        "complex-mass" if not zeroed and _complex_mass(target) else "zero",
        tuple(zeroed),
        PROBE_REDUCTION,
        tuple(rows),
        round(time.perf_counter() - start, 3),
    )


__all__ = [
    "CATANI_DD_FLOOR",
    "CATANI_FLOOR",
    "ESTIMATES",
    "EXIT_FAILED",
    "EXIT_NOTHING_CHECKED",
    "EXIT_PASSED",
    "FAIL",
    "FAILED",
    "NOTHING_CHECKED",
    "NOT_APPLICABLE",
    "NOT_RUN",
    "NO_PREDICTION",
    "PASS",
    "PASSED",
    "VERDICTS",
    "CheckReport",
    "CheckRow",
    "ProcessCheck",
    "born_interfered_rows",
    "catani_comparison",
    "catani_dd_comparison",
    "check_request",
    "induced_rows",
]
