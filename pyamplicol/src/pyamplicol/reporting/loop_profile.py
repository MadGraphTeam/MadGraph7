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
    """The backend's name on one line, the process type's default backend
    marked with an asterisk."""

    text = palette.paint(
        _name(backend), palette.BACKEND.get(backend), color=color, bold=True
    )
    return text + (" *" if backend == default else "")


# The widest line of the human report: the terminal's width, at most this
# many columns (a readable line on a laptop terminal; a display constant).
_LINE_WIDTH = 100


def _width() -> int:
    import shutil

    return max(
        min(shutil.get_terminal_size((_LINE_WIDTH, 24)).columns, _LINE_WIDTH), 60
    )


def _fraction(count: int, total: int) -> str:
    """``count / total`` in percent, three significant digits."""

    if not total:
        return "-"
    value = 100.0 * count / total
    return f"{value:.3g} %"


def failure_lines(causes: Any, table: str, color: bool) -> str | None:
    """The distribution of the failed and refused points of one profile
    table (``loop_profile.FailureCause``): after a ``---`` line, one line
    per backend and reason class, most frequent first, ``<count>
    (<fraction> %) <backend> | <kind>: <reason>``, the reason wrapped under
    itself within the line width; None when nothing failed."""

    import textwrap

    selected = [cause for cause in causes if cause.table == table]
    if not selected:
        return None
    width = _width()
    lines = [palette.paint("---", palette.MUTED, color=color)]
    for cause in selected:
        label = (
            f"{cause.count} ({_fraction(cause.count, cause.total)}) "
            f"{_name(cause.backend)} | "
        )
        kind = cause.kind
        body = textwrap.wrap(
            f"{kind}: {' '.join(cause.reason.split())}",
            width=max(width - len(label), 30),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [kind]
        colour = palette.RECOVERED if kind == "best-effort" else palette.BAD
        lines.append(
            palette.paint(label, colour, color=color)
            + body[0]
            + "".join("\n" + " " * len(label) + part for part in body[1:])
        )
    return "\n".join(lines)


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
                ),
            ),
        )
    )
    return str(table.get_string())


def _served(row: Any) -> bool:
    """A profiled backend that served at least one timed point (accepted or
    with its best-effort value)."""

    profile = row.profile
    return (
        profile is not None and profile.failed_points < profile.batch * profile.samples
    )


# The width of a stage bar (a display constant).
_BAR = 8


def _seconds_text(seconds: float) -> str:
    """A duration with three significant digits."""

    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if abs(seconds) >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def _lanes(row: Any) -> dict[str, int]:
    profile = row.profile
    return dict(profile.reduction.get("lanes") or {}) if profile is not None else {}


def _runtime_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    from .virtual import _STAGE_COLOURS

    rows = report.runtime
    compared = [row for row in rows if _served(row) and row.ratio is not None]
    fastest = min((row.profile.seconds_per_point for row in compared), default=None)
    # Optional columns: only when they say something (C3: time both
    # backends, no flourish).
    show_lanes = any(len(_lanes(row)) > 1 for row in rows)
    show_best = any(row.profile and row.profile.best_effort_points for row in rows)
    show_failed = any(row.profile and row.profile.failed_points for row in rows)
    fields = ["backend", "time / point", "min - max call", "vs fastest"]
    fields += ["best-effort"] if show_best else []
    fields += ["lanes"] if show_lanes else []
    fields += ["failed"] if show_failed else []
    summary = palette.table(
        prettytable, tuple(fields), "Runtime per reduction backend", color=color
    )
    summary.max_width = 26
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        cells: list[str]
        if row.profile is None:
            cells = [
                label,
                palette.paint("refused", palette.BAD, color=color, bold=True),
                "",
                "",
            ]
            cells += [""] * (len(fields) - len(cells))
            summary.add_row(tuple(cells))
            continue
        profile = row.profile
        timed = profile.batch * profile.samples
        best = profile.seconds_per_point == fastest
        if not _served(row):
            # Every timed point failed without a value: the time of failing.
            time_cell = palette.paint(
                _seconds_text(profile.seconds_per_point), palette.MUTED, color=color
            )
            ratio = palette.paint("all failed", palette.BAD, color=color, bold=True)
            spread = ""
        else:
            time_cell = palette.paint(
                _seconds_text(profile.seconds_per_point),
                palette.GOOD if best and len(compared) > 1 else None,
                color=color,
                bold=True,
            )
            spread = f"{_seconds_text(profile.low)} - {_seconds_text(profile.high)}"
            if len(compared) < 2:
                ratio = palette.paint("-", palette.MUTED, color=color)
            elif best:
                ratio = palette.paint("fastest", palette.GOOD, color=color, bold=True)
            else:
                ratio = palette.paint(
                    f"x{row.ratio:.2f}", palette.RECOVERED, color=color
                )
        cells = [label, time_cell, spread, ratio]
        if show_best:
            cells.append(
                palette.paint(
                    f"{profile.best_effort_points} of {timed}",
                    palette.RECOVERED if profile.best_effort_points else None,
                    color=color,
                )
                if profile.best_effort_points
                else ""
            )
        if show_lanes:
            lanes = _lanes(row)
            cells.append(
                ", ".join(
                    palette.paint(
                        f"{_name(name)} {count}",
                        palette.BACKEND.get(name)
                        if name == row.backend
                        else palette.RECOVERED,
                        color=color,
                    )
                    for name, count in sorted(
                        lanes.items(), key=lambda i: i[0] != row.backend
                    )
                )
            )
        if show_failed:
            cells.append(
                palette.paint(
                    f"{profile.failed_points} of {timed}",
                    palette.BAD if profile.failed_points else None,
                    color=color,
                )
                if profile.failed_points
                else ""
            )
        summary.add_row(tuple(cells))
    blocks = [str(summary.get_string())]
    causes = failure_lines(report.failure_causes, "runtime", color)
    refusals = [
        f"{_backend_label(row.backend, report.default_backend, color)} refused: "
        f"{row.refusal}"
        for row in rows
        if row.profile is None
    ]
    if causes is not None or refusals:
        blocks.append(
            "\n".join(
                ([causes] if causes is not None else [])
                + (
                    [palette.paint("---", palette.MUTED, color=color)]
                    if causes is None
                    else []
                )
                + refusals
            )
        )
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
        "Time per point and stage",
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


def interrupted_line(setup: Any, color: bool) -> str | None:
    """The one-line notice of an interrupted stability run (Ctrl-C): how
    many of the requested points were measured; None otherwise."""

    if not setup or not setup.get("interrupted"):
        return None
    return palette.paint(
        f"interrupted: {setup.get('measured', 0)} of {setup.get('requested', 0)} "
        "requested points evaluated; the tables hold those points",
        palette.RECOVERED,
        color=color,
        bold=True,
    )


def _stability_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    setup = report.stability_setup
    target = float(setup["target"])
    target_digits = -math.log10(target)
    rows = report.stability
    served = [row for row in rows if row.refusal is None and row.points]
    title = (
        f"Stable digits of binary64 vs double-double, {setup['points']} flat "
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
                    _backend_label(row.backend, report.default_backend, color),
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
            "best-eff./none",
            "flagged",
            "flagged inacc.",
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
                )
            )
            refusals.append(f"{label} refused: {row.refusal}")
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
                    str(row.best_effort),
                    palette.GOOD if row.best_effort == 0 else palette.RECOVERED,
                    color=color,
                )
                + " / "
                + palette.paint(
                    str(row.failed_binary64),
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
    best = _best_effort_table(report, served, prettytable, color)
    if best is not None:
        blocks.append(best)
    causes = failure_lines(report.failure_causes, "stability", color)
    if causes is not None or refusals:
        blocks.append(
            "\n".join(
                ([causes] if causes is not None else [])
                + (
                    [palette.paint("---", palette.MUTED, color=color)]
                    if causes is None
                    else []
                )
                + refusals
            )
        )
    notice = interrupted_line(setup, color)
    if notice is not None:
        blocks.append(notice)
    return blocks


def _best_effort_table(
    report: Any, served: Any, prettytable: Any, color: bool
) -> str | None:
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
            "points",
            "true min/med/max",
            "unc. median",
            "|unc-true| med",
            "unc. covers",
        ),
        "Best-effort values (refused in binary64) against the truth",
        color=color,
    )
    table.max_width = 20
    for row in rows:
        points = [item for item in row.per_point if item.status == "best-effort"]
        true = sorted(float(item.true_digits) for item in points)
        stated = [item.uncertainty_digits for item in points]
        known = sorted(float(e) for e in stated if e is not None)
        gaps = sorted(
            abs(float(item.uncertainty_digits) - float(item.true_digits))
            for item in points
            if item.uncertainty_digits is not None
        )

        def med(values: list[float]) -> float | None:
            return values[(len(values) - 1) // 2] if values else None

        def cell(value: float | None) -> str:
            if value is None:
                return palette.paint("-", palette.MUTED, color=color)
            return _number(value, color, palette.ESTIMATE)

        table.add_row(
            (
                _backend_label(row.backend, report.default_backend, color),
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
