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


def _header(report: Any, prettytable: Any, color: bool) -> str:
    setup = (
        (("runtime setup", report.runtime_setup),)
        if report.runtime_setup is not None
        else ()
    )
    table = palette.table(
        prettytable, ("field", "value"), "pyAmpliCol loop profile", color=color
    )
    table.max_width = 100
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


def _runtime_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    from .summary import _seconds_text
    from .virtual import _STAGE_COLOURS

    rows = report.runtime
    served = [row for row in rows if row.profile is not None]
    fastest = min((row.profile.seconds_per_point for row in served), default=None)
    summary = palette.table(
        prettytable,
        (
            "backend",
            "time per point",
            "fastest - slowest call",
            "vs fastest",
            "effective (reason)",
            "failed points",
        ),
        "Runtime per reduction backend (the setup above)",
        color=color,
    )
    summary.max_width = 60
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        if row.requested == "auto":
            label += "\n" + palette.paint(
                "as auto: per-point fallback", palette.MUTED, color=color
            )
        if row.profile is None:
            summary.add_row(
                (
                    label,
                    palette.paint("refused", palette.BAD, color=color, bold=True),
                    "",
                    "",
                    palette.paint(row.refusal or "", palette.BAD, color=color),
                    "",
                )
            )
            continue
        profile = row.profile
        best = profile.seconds_per_point == fastest
        time_text = _seconds_text(profile.seconds_per_point)
        ratio = (
            palette.paint("fastest", palette.GOOD, color=color, bold=True)
            if best and len(served) > 1
            else ""
            if len(served) < 2
            else palette.paint(f"x{row.ratio:.2f}", palette.RECOVERED, color=color)
        )
        reduction = profile.reduction
        effective = str(reduction.get("effective", "N/A"))
        reason = str(reduction.get("reason", ""))
        lanes = reduction.get("lanes") or {}
        if len(lanes) > 1:
            reason += "; lanes: " + ", ".join(
                f"{name} {count}" for name, count in sorted(lanes.items())
            )
        summary.add_row(
            (
                label,
                palette.paint(
                    time_text,
                    palette.GOOD if best and len(served) > 1 else None,
                    color=color,
                    bold=True,
                ),
                f"{_seconds_text(profile.low)} - {_seconds_text(profile.high)}",
                ratio,
                f"{_name(effective)} ({reason})" if reason else _name(effective),
                palette.paint(
                    str(profile.failed_points),
                    palette.GOOD if profile.failed_points == 0 else palette.BAD,
                    color=color,
                ),
            )
        )
    blocks = [str(summary.get_string())]
    if not served:
        return blocks
    groups = list(dict.fromkeys(g for row in served for g in row.profile.stages))
    fields = ["stage"]
    for row in served:
        fields += [f"{_name(row.backend)} time", f"{_name(row.backend)} share"]
    if len(served) == 2:
        fields.append(f"{_name(served[1].backend)} / {_name(served[0].backend)}")
    stages = palette.table(
        prettytable,
        tuple(fields),
        "Time per point and stage (native executor stage reports of the timed calls)",
        color=color,
    )
    totals = [sum(row.profile.stages.values()) or 1.0 for row in served]
    for group in [*groups, "total"]:
        cells = [
            palette.paint(group, _STAGE_COLOURS.get(group), color=color, bold=True)
        ]
        seconds = []
        for row, total in zip(served, totals, strict=True):
            value = (
                row.profile.seconds_per_point
                if group == "total"
                else row.profile.stages.get(group, 0.0)
            )
            seconds.append(value)
            share = 1.0 if group == "total" else value / total
            cells += [_seconds_text(value), f"{100 * share:5.1f} %"]
        if len(served) == 2:
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
    """Red below the target's digits, yellow below half of binary64's digits
    (half the precision lost), green above."""

    from pyamplicol.loop_profile import BINARY64_DIGITS

    if digits < target_digits:
        return palette.BAD
    if digits < BINARY64_DIGITS / 2:
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


def _stability_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    setup = report.stability_setup
    target = float(setup["target"])
    target_digits = -math.log10(target)
    rows = report.stability
    served = [row for row in rows if row.refusal is None and row.points]
    fields = ["percentile (worst first)", "class points"]
    for row in served:
        name = _name(row.backend)
        fields += [
            f"{name} true digits",
            f"{name} |est - true| min / median / max",
            f"{name} signed median",
        ]
    title = (
        f"Stability over {setup['points']} flat points (sqrt(s) = "
        f"{setup['sqrt_s']:g}, seed {setup['seed']}): stable digits of the "
        "pure binary64 finite part of the total vs a double-double re-run"
    )
    table = palette.table(prettytable, tuple(fields), title, color=color)
    table.max_width = 40
    labels = {0.001: "1 per mil", 0.01: "1 %", 0.1: "10 %", 0.5: "50 % (median)"}
    if served:
        for index, first in enumerate(served[0].percentiles):
            counts = {row.percentiles[index].count for row in served}
            cells = [
                labels.get(first.percentile, f"{100 * first.percentile:g} %"),
                " / ".join(str(count) for count in sorted(counts)),
            ]
            for row in served:
                item = row.percentiles[index]
                digits = (
                    _number(
                        item.digits,
                        color,
                        _digits_colour(item.digits, target_digits),
                    )
                    if item.digits is not None
                    else palette.paint(
                        f"- (N >= {item.needed_points})", palette.MUTED, color=color
                    )
                )
                spread = (
                    palette.paint("-", palette.MUTED, color=color)
                    if item.count == 0
                    else " / ".join(
                        _number(value, color, palette.ESTIMATE)
                        for value in (
                            item.deviation_min,
                            item.deviation_median,
                            item.deviation_max,
                        )
                    )
                )
                cells += [digits, spread, _signed(item.signed_median, color)]
            table.add_row(tuple(cells))
    blocks = [str(table.get_string())]
    verdicts = palette.table(
        prettytable,
        (
            "backend",
            "points (failed)",
            "median digits true / estimated",
            f"flagged unstable (estimate > {target:g})",
            "of which truly inaccurate",
            "truly inaccurate, not flagged",
            "seconds",
        ),
        "Default checks against the truth (target "
        f"{target:g} = {target_digits:.2f} digits)",
        color=color,
    )
    verdicts.max_width = 32
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        if row.refusal is not None:
            verdicts.add_row(
                (
                    label,
                    palette.paint("refused", palette.BAD, color=color, bold=True),
                    palette.paint(row.refusal, palette.BAD, color=color),
                    "",
                    "",
                    "",
                    "",
                )
            )
            continue
        verdicts.add_row(
            (
                label,
                f"{row.points} ("
                + palette.paint(
                    str(row.failed),
                    palette.GOOD if row.failed == 0 else palette.BAD,
                    color=color,
                )
                + ")",
                _number(
                    row.median_true_digits,
                    color,
                    None
                    if row.median_true_digits is None
                    else _digits_colour(row.median_true_digits, target_digits),
                )
                + " / "
                + _number(row.median_estimated_digits, color, palette.ESTIMATE),
                _percent(row.flagged, row.points),
                palette.paint(
                    f"{row.flagged_inaccurate} of {row.flagged}",
                    None,
                    color=color,
                ),
                palette.paint(
                    _percent(row.unflagged_inaccurate, row.points),
                    palette.GOOD if row.unflagged_inaccurate == 0 else palette.BAD,
                    color=color,
                    bold=row.unflagged_inaccurate > 0,
                ),
                f"{row.seconds:.1f}",
            )
        )
    blocks.append(str(verdicts.get_string()))
    first = served[0].percentiles if served else ()
    notes = [
        "Digits: -log10(|ME_b64 - ME_dd| / |ME_dd|) of the finite part of the "
        "total (summed over the squared-order sectors; poles not computed), "
        f"capped to [0, {setup['binary64_digits']:.2f}] (binary64); estimated: "
        f"{setup['estimate']}, its absolute finite-part errors summed over the "
        "sectors over |total|, capped the same way (MadLoop, arXiv:1103.0621 "
        "sect. 3.4; arXiv:1405.0301).",
        "Points sorted by true digits, worst first. The row of percentile p "
        "shows the true digits of the point of rank floor(p N); its class holds "
        "the points of rank floor(p' N) < rank <= floor(p N) (p' the previous "
        "percentile, 0 for the first), over which |est - true| (min / median / "
        "max) and the signed median of est - true are taken: positive (red) = "
        "the estimate is optimistic, negative (blue) = pessimistic.",
    ]
    if first:
        notes.append(
            "Class points: "
            + ", ".join(
                f"{labels.get(item.percentile, item.percentile)} {item.count}"
                for item in first
            )
            + f". A populated 1 per mil class needs N >= {first[0].needed_points}."
        )
    notes.append(
        "Flagged: the binary64 call's estimate (relative, per sector) above the "
        "target, as the default rescue reads it; truly inaccurate: the true "
        "relative error above the target. No rescue ran: the statistics are "
        "those of pure binary64; the double-double re-run is the reference only."
    )
    blocks.append("\n".join(f"  {note}" for note in notes))
    return blocks


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
