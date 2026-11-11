# SPDX-License-Identifier: 0BSD
"""Human tables of ``pyamplicol profile`` of a loop artifact
(``loop_profile.LoopProfileReport``): the runtime and the stability
statistics of the reduction backends side by side."""

from __future__ import annotations

import math
from typing import Any

from . import palette


def _name(backend: str) -> str:
    from pyamplicol.loops.v2.reduction import NAMES

    return NAMES.get(backend, backend)


def _backend_label(backend: str, default: str | None, color: bool) -> str:
    text = palette.paint(
        _name(backend), palette.BACKEND.get(backend), color=color, bold=True
    )
    return text + (" (default)" if backend == default else "")


# The widest line of the human report: the terminal's width, at most this
# many columns (a readable line on a laptop terminal; a display constant).
_LINE_WIDTH = 100


def _width() -> int:
    import shutil

    return max(
        min(shutil.get_terminal_size((_LINE_WIDTH, 24)).columns, _LINE_WIDTH), 60
    )


def _notes(notes: list[str]) -> str:
    """The footnotes wrapped to the terminal width (``_width``)."""

    import textwrap

    width = _width()
    return "\n".join(
        textwrap.fill(note, width=width, initial_indent="  ", subsequent_indent="    ")
        for note in notes
    )


def _header(report: Any, prettytable: Any, color: bool) -> str:
    setup = (
        (("runtime setup", report.runtime_setup),)
        if report.runtime_setup is not None
        else ()
    )
    table = palette.table(
        prettytable, ("field", "value"), "pyAmpliCol loop profile", color=color
    )
    table.max_width = _width() - 24
    table.add_rows(
        (
            (
                "process",
                f"{report.process_id} ({report.process_expression})",
            ),
            ("contribution", report.contribution),
            *setup,
            (
                "backends",
                ", ".join(
                    _backend_label(name, report.default_backend, color)
                    for name in report.backends
                )
                + (
                    f"; default for {report.contribution}: "
                    f"{_name(report.default_backend)}"
                    if report.default_backend
                    else ""
                ),
            ),
        )
    )
    return str(table.get_string())


def _served(row: Any) -> bool:
    """A profiled backend that served at least one timed point."""

    profile = row.profile
    return (
        profile is not None and profile.failed_points < profile.batch * profile.samples
    )


# The width of a stage bar (a display constant).
_BAR = 8
# The characters of a failure reason shown under the runtime table (the full
# reasons are in the JSON; a display constant).
_REASON = 160


def _seconds_text(seconds: float) -> str:
    """A duration with three significant digits."""

    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if abs(seconds) >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def _short(reason: str) -> str:
    text = " ".join(reason.split())
    return text if len(text) <= _REASON else text[: _REASON - 3] + "..."


def _runtime_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    from .virtual import _STAGE_COLOURS

    rows = report.runtime
    compared = [row for row in rows if _served(row) and row.ratio is not None]
    fastest = min((row.profile.seconds_per_point for row in compared), default=None)
    summary = palette.table(
        prettytable,
        (
            "backend",
            "time / point",
            "min - max call",
            "vs fastest",
            "lanes",
            "failed",
        ),
        "Runtime per reduction backend (the setup above)",
        color=color,
    )
    summary.max_width = 26
    reasons: list[str] = []
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        if row.requested == "auto":
            label += "\n" + palette.paint("as auto", palette.MUTED, color=color)
        if row.profile is None:
            summary.add_row(
                (
                    label,
                    palette.paint("refused", palette.BAD, color=color, bold=True),
                    "",
                    "",
                    "",
                    "",
                )
            )
            reasons.append(f"{_name(row.backend)} refused: {row.refusal}")
            continue
        profile = row.profile
        timed = profile.batch * profile.samples
        if profile.failed_points:
            shown = profile.failure_reasons[:2]
            more = len(profile.failure_reasons) - len(shown)
            reasons.append(
                f"{_name(row.backend)} failed {profile.failed_points} of {timed} "
                "timed points"
                + (
                    f" ({len(profile.failure_reasons)} distinct reasons)"
                    if more
                    else ""
                )
                + ": "
                + "; ".join(f"{count}x {_short(reason)}" for reason, count in shown)
                + (f"; {more} more in --json" if more else "")
            )
        if not _served(row):
            # Every timed point failed: the time is that of failing.
            summary.add_row(
                (
                    label,
                    palette.paint(
                        _seconds_text(profile.seconds_per_point),
                        palette.MUTED,
                        color=color,
                    ),
                    "",
                    palette.paint("all failed", palette.BAD, color=color, bold=True),
                    "-",
                    palette.paint(
                        f"{profile.failed_points} of {timed}", palette.BAD, color=color
                    ),
                )
            )
            continue
        best = profile.seconds_per_point == fastest
        if len(compared) < 2:
            ratio = palette.paint(
                "only one served" if len(rows) > 1 else "-", palette.MUTED, color=color
            )
        elif best:
            ratio = palette.paint("fastest", palette.GOOD, color=color, bold=True)
        else:
            ratio = palette.paint(f"x{row.ratio:.2f}", palette.RECOVERED, color=color)
        lanes = profile.reduction.get("lanes") or {}
        lane_text = "\n".join(
            palette.paint(
                f"{_name(name)} {count}",
                palette.BACKEND.get(name) if name == row.backend else palette.RECOVERED,
                color=color,
            )
            + ("" if name == row.backend else " (fallback)")
            for name, count in sorted(lanes.items(), key=lambda i: i[0] != row.backend)
        )
        summary.add_row(
            (
                label,
                palette.paint(
                    _seconds_text(profile.seconds_per_point),
                    palette.GOOD if best and len(compared) > 1 else None,
                    color=color,
                    bold=True,
                ),
                f"{_seconds_text(profile.low)} - {_seconds_text(profile.high)}",
                ratio,
                lane_text,
                palette.paint(
                    f"{profile.failed_points} of {timed}",
                    palette.GOOD if profile.failed_points == 0 else palette.BAD,
                    color=color,
                ),
            )
        )
    blocks = [str(summary.get_string())]
    notes = [
        "lanes: the timed points per effective backend (as auto, the default "
        "backend's per-point fallback serves the points it refuses). A backend "
        "that failed every timed point shows the time of failing (grey) and is "
        "not compared."
    ]
    notes += reasons
    blocks.append(_notes(notes))
    served = [row for row in rows if _served(row)]
    if not served:
        return blocks
    groups = list(dict.fromkeys(g for row in served for g in row.profile.stages))
    fields = ["stage"]
    for row in served:
        name = _name(row.backend)
        fields += [name, f"{name} share"]
    pair = len(served) == 2
    if pair:
        fields.append(f"{_name(served[1].backend)} / {_name(served[0].backend)}")
    stages = palette.table(
        prettytable,
        tuple(fields),
        "Time per point and stage (native executor stage reports of the timed calls)",
        color=color,
    )
    totals = [sum(row.profile.stages.values()) or 1.0 for row in served]
    for group in [*groups, "total"]:
        colour = _STAGE_COLOURS.get(group)
        cells = [palette.paint(group, colour, color=color, bold=True)]
        seconds = []
        for row, total in zip(served, totals, strict=True):
            value = (
                row.profile.seconds_per_point
                if group == "total"
                else row.profile.stages.get(group, 0.0)
            )
            seconds.append(value)
            share = 1.0 if group == "total" else value / total
            bar = "#" * max(round(share * _BAR), 1 if value else 0)
            cells += [
                _seconds_text(value),
                f"{100 * share:5.1f} % "
                + palette.paint(bar.ljust(_BAR), colour, color=color),
            ]
        if pair:
            first, second = seconds
            if first > 0.0 and second > 0.0:
                ratio = second / first
                cells.append(
                    palette.paint(
                        f"{ratio:.2f}",
                        palette.GOOD if ratio < 1.0 else palette.RECOVERED,
                        color=color,
                    )
                )
            else:
                cells.append("-")
        stages.add_row(tuple(cells))
    blocks.append(str(stages.get_string()))
    return blocks


def _digits_colour(digits: float, target_digits: float) -> str:
    """Red below the target's digits, yellow below the target's digits plus
    half of binary64's remaining ones (a display convention: less than half
    of the margin above the target is left), green above."""

    from pyamplicol.loop_profile import BINARY64_DIGITS

    if digits < target_digits:
        return palette.BAD
    if digits < (target_digits + BINARY64_DIGITS) / 2:
        return palette.RECOVERED
    return palette.GOOD


def _number(value: float | None, color: bool, colour: str | None = None) -> str:
    if value is None:
        return palette.paint("-", palette.MUTED, color=color)
    return palette.paint(f"{value:.2f}", colour, color=color)


def _signed(value: float | None, color: bool) -> str:
    if value is None:
        return palette.paint("-", palette.MUTED, color=color)
    # Positive: the estimate claims more digits than the truth (optimistic).
    colour = palette.BAD if value > 0 else palette.DOUBLE_POLE if value < 0 else None
    return palette.paint(f"{value:+.2f}", colour, color=color)


def _percent(part: int, whole: int) -> str:
    return f"{part} ({100 * part / whole:.1f} %)" if whole else str(part)


_LABELS = {0.001: "1 per mil", 0.01: "1 %", 0.1: "10 %", 0.5: "50 %"}


def _stability_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    setup = report.stability_setup
    target = float(setup["target"])
    target_digits = -math.log10(target)
    rows = report.stability
    served = [row for row in rows if row.refusal is None and row.points]
    title = (
        f"Stable digits of pure binary64 vs double-double, {setup['points']} flat "
        f"points (sqrt(s) = {setup['sqrt_s']:g}, seed {setup['seed']})"
    )
    table = palette.table(
        prettytable,
        (
            "percentile",
            "backend",
            "true digits",
            "class",
            "|est-true| min/med/max",
            "est-true med",
        ),
        title,
        color=color,
    )
    table.max_width = 30
    count = len(served[0].percentiles) if served else 0
    for index in range(count):
        for position, row in enumerate(served):
            item = row.percentiles[index]
            digits = (
                _number(item.digits, color, _digits_colour(item.digits, target_digits))
                if item.digits is not None
                else palette.paint(
                    f"- (N >= {item.needed_points})", palette.MUTED, color=color
                )
            )
            spread = (
                palette.paint("-", palette.MUTED, color=color)
                if item.deviation_min is None
                else " / ".join(
                    _number(value, color, palette.ESTIMATE)
                    for value in (
                        item.deviation_min,
                        item.deviation_median,
                        item.deviation_max,
                    )
                )
            )
            points = str(item.count) + (
                palette.paint(f" ({item.failed} failed)", palette.BAD, color=color)
                if item.failed
                else ""
            )
            table.add_row(
                (
                    _LABELS.get(item.percentile, f"{100 * item.percentile:g} %")
                    if position == 0
                    else "",
                    palette.paint(
                        _name(row.backend),
                        palette.BACKEND.get(row.backend),
                        color=color,
                    ),
                    digits,
                    points,
                    spread,
                    _signed(item.signed_median, color),
                ),
                divider=position == len(served) - 1 and index < count - 1,
            )
    blocks = [str(table.get_string())]
    verdicts = palette.table(
        prettytable,
        (
            "backend",
            "measured",
            "best-effort",
            "no value",
            "flagged",
            "flagged, inaccurate",
            "missed",
            "time",
        ),
        f"Default checks against the truth (target {target:g} = "
        f"{target_digits:.2f} digits)",
        color=color,
    )
    verdicts.max_width = 14
    refusals = []
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        if row.refusal is not None:
            verdicts.add_row(
                (
                    label,
                    palette.paint("refused", palette.BAD, color=color, bold=True),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                )
            )
            refusals.append(f"{_name(row.backend)} refused: {row.refusal}")
            continue
        measured = str(row.points) + (
            palette.paint(
                f" (+{row.failed_reference} no ref.)", palette.MUTED, color=color
            )
            if row.failed_reference
            else ""
        )
        verdicts.add_row(
            (
                label,
                measured,
                palette.paint(
                    _percent(row.best_effort, row.points),
                    palette.GOOD if row.best_effort == 0 else palette.RECOVERED,
                    color=color,
                ),
                palette.paint(
                    _percent(row.failed_binary64, row.points),
                    palette.GOOD if row.failed_binary64 == 0 else palette.BAD,
                    color=color,
                    bold=row.failed_binary64 > 0,
                ),
                _percent(row.flagged, row.points),
                f"{row.flagged_inaccurate} of {row.flagged}",
                palette.paint(
                    _percent(row.unflagged_inaccurate, row.points),
                    palette.GOOD if row.unflagged_inaccurate == 0 else palette.BAD,
                    color=color,
                    bold=row.unflagged_inaccurate > 0,
                ),
                f"{row.seconds:.1f} s",
            )
        )
    blocks.append(str(verdicts.get_string()))
    best = _best_effort_table(served, prettytable, color)
    if best is not None:
        blocks.append(best)
    first = served[0].percentiles if served else ()
    notes = [
        "Digits: -log10(|ME_b64 - ME_dd| / |ME_dd|) on the finite part of the "
        "total (summed over the squared-order sectors; poles not computed), "
        f"capped to [0, {setup['binary64_digits']:.2f}] (binary64). Estimated: "
        f"{setup['estimate']}; its absolute finite-part errors summed over the "
        "sectors over |total|, capped the same way. Measure of MadLoop, "
        "arXiv:1103.0621 sect. 3.4, and arXiv:1405.0301.",
        "Measured points: those with a double-double reference, plus the points "
        "without any binary64 value (no value), which count 0 true digits "
        "(worst end) and have no estimate. A point the backend's own checks "
        "refuse is measured with its best-effort value (the unchecked "
        "reduction's, returned by default with status best-effort; never "
        "accepted, so always flagged) at its true digits. Points with a "
        "binary64 value but no reference are left out (no ref.).",
        "Sorted by true digits, worst first; N = measured points. The row of "
        "percentile p shows the true digits of the point of rank floor(p N) "
        "(50 %: the median, the lower middle point for even N). Its class "
        "holds the points of rank floor(p' N) < rank <= floor(p N), p' the "
        "previous percentile (0 for the first). |est - true| (min / med / max) "
        "and the median of est - true are over the class's points with an "
        "estimate: positive (red) = the estimate is optimistic, negative "
        "(blue) = pessimistic.",
    ]
    if first:
        notes.append(
            "Class points ("
            + _name(served[0].backend)
            + "): "
            + ", ".join(
                f"{_LABELS.get(item.percentile, item.percentile)} {item.count}"
                for item in first
            )
            + f" of N = {served[0].points}. A populated 1 per mil class needs "
            f"N >= {first[0].needed_points} measured points."
        )
    notes.append(
        "Best-effort: points the backend refused in binary64, served with their "
        "best-effort value; no value: points without any binary64 value (no "
        "rescue ran). Flagged, inaccurate: of the flagged points, the truly "
        "inaccurate ones; missed: truly inaccurate points the checks did "
        "not flag; time: the backend's binary64 and double-double calls. "
        "Percentages of the measured points."
    )
    notes.append(
        "Flagged: the binary64 call's estimate (relative, per sector) above the "
        "target with the check set named above (one default rotation unless "
        "evaluation.stability_checks is set; the default rescue runs a "
        "stronger set), or a binary64 failure (reported, so detected). Truly "
        "inaccurate: the true relative error above the target, or no binary64 "
        "value. No rescue ran: pure binary64 statistics; the double-double "
        "re-run is the reference only. Digits: red below the target's digits, "
        "yellow in the lower half of the margin above it, green above."
    )
    notes += refusals
    blocks.append(_notes(notes))
    return blocks


def _best_effort_table(served: Any, prettytable: Any, color: bool) -> str | None:
    """How accurate the best-effort values and their uncertainties are
    (m-lit ux-besteffort): per backend with best-effort points, their true
    digits (min / median / max), the median digits of their uncertainty,
    the median |uncertainty - true| digits and how many uncertainties cover
    the true error."""

    rows = [row for row in served if row.best_effort]
    if not rows:
        return None
    table = palette.table(
        prettytable,
        (
            "backend",
            "best-effort",
            "true digits min/med/max",
            "unc. digits med",
            "|unc-true| med",
            "unc. covers truth",
        ),
        "Best-effort values (refused in binary64) against the truth",
        color=color,
    )
    table.max_width = 24
    for row in rows:
        points = [item for item in row.per_point if item.status == "best-effort"]
        true = sorted(float(item.true_digits) for item in points)
        estimated = [item.estimated_digits for item in points]
        known = sorted(float(e) for e in estimated if e is not None)
        gaps = sorted(
            abs(float(item.estimated_digits) - float(item.true_digits))
            for item in points
            if item.estimated_digits is not None
        )

        def med(values: list[float]) -> float | None:
            return values[(len(values) - 1) // 2] if values else None

        def cell(value: float | None) -> str:
            if value is None:
                return palette.paint("-", palette.MUTED, color=color)
            return _number(value, color, palette.ESTIMATE)

        table.add_row(
            (
                palette.paint(
                    _name(row.backend), palette.BACKEND.get(row.backend), color=color
                ),
                str(len(points)),
                " / ".join(cell(v) for v in (true[0], med(true), true[-1])),
                cell(med(known)),
                cell(med(gaps)),
                f"{row.best_effort_covered} of {len(known)}"
                + (
                    palette.paint(
                        f" ({len(points) - len(known)} without)",
                        palette.MUTED,
                        color=color,
                    )
                    if len(known) < len(points)
                    else ""
                ),
            )
        )
    return str(table.get_string())


def loop_profile_summary(report: Any, *, prettytable: Any, color: bool) -> str:
    """The header, the runtime tables and the stability tables (whichever
    the report holds)."""

    blocks = [_header(report, prettytable, color)]
    if report.runtime is not None:
        blocks += _runtime_tables(report, prettytable, color)
    if report.stability is not None:
        blocks += _stability_tables(report, prettytable, color)
    return "\n\n".join(blocks)


__all__ = ["loop_profile_summary"]
