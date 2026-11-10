# SPDX-License-Identifier: 0BSD
"""Human tables of ``pyamplicol check`` (``loops.v2.checks.CheckReport``)."""

from __future__ import annotations

from typing import Any

_VERDICT_COLOURS = {
    "pass": "GREEN",
    "fail": "RED",
    "undecided": "RED",
    "not applicable": "YELLOW",
    "no prediction": "YELLOW",
    "not run": "YELLOW",
}
_STATUS_COLOURS = {"passed": "GREEN", "failed": "RED", "nothing checked": "YELLOW"}


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.3g}"


def check_report_summary(report: Any, *, prettytable: Any, color: bool) -> str:
    """The header (what was checked and how), one table per process (every
    row: check, subject, point, deciding carrier, ratio to its bound,
    deviation and E over the row's scale (``checks.CheckRow``), verdict,
    fallback or reason) and the overall verdict with the exit code."""

    from pyamplicol.loops.v2 import checks
    from pyamplicol.loops.v2.induced import ESTIMATE_FACTOR

    from .summary import _paint

    header = prettytable.PrettyTable(("field", "value"))
    header.title = _paint("pyAmpliCol check", "CYAN", enabled=color, bold=True)
    header.align = "l"
    header.max_width = 96
    header.add_rows(
        (
            ("source", f"{report.source} ({report.source_kind})"),
            ("regeneration", report.regeneration),
            (
                "points",
                f"{report.points} per process (seeds {report.seed} to "
                f"{report.seed + report.points - 1}), momenta lifted to exact "
                "conservation, mu_R^2 = s of the initial state",
            ),
            (
                "carriers",
                f"{report.precision}: " + " then ".join(report.carriers)
                if len(report.carriers) > 1
                else f"{report.precision}: {report.carriers[0]}",
            ),
            (
                "criteria",
                f"poles: |pole - Catani| <= {checks.ESTIMATES} E + "
                f"{checks.CATANI_FLOOR:g} |Catani| (binary64, recommended "
                f"checks; double-double {checks.CATANI_DD_FLOOR:g} relative); "
                "loop-induced finiteness, Ward and Goldstone probes within "
                f"{ESTIMATE_FACTOR}x their own estimate; Born-interfered Ward: "
                "ward.ward_row; arbitrary precision (the rung, digits derived "
                "from the double-double shortfall) by the double-double "
                "criterion. ratio <= 1 passes",
            ),
            ("seconds", f"{report.seconds:.3g}"),
        )
    )
    blocks = [header.get_string()]
    for process in report.processes:
        table = prettytable.PrettyTable(
            (
                "check",
                "subject",
                "point",
                "carrier",
                "ratio",
                "dev / scale",
                "E / scale",
                "verdict",
                "fallback / reason",
            )
        )
        status = _paint(
            process.status,
            _STATUS_COLOURS.get(process.status),
            enabled=color,
            bold=True,
        )
        zeroed = (
            f", zeroed {', '.join(process.zeroed_widths)}"
            if process.zeroed_widths
            else ""
        )
        table.title = (
            f"{process.process}: {process.expression} ({process.contribution}; "
            f"widths {process.widths}{zeroed}; reduction {process.reduction}; "
            f"{process.seconds:.3g} s): {status}"
        )
        table.align = "l"
        table.max_width["fallback / reason"] = 56
        table.max_width["subject"] = 40
        for row in process.rows:
            verdict = _paint(
                row.verdict, _VERDICT_COLOURS.get(row.verdict), enabled=color
            )
            table.add_row(
                (
                    row.check,
                    row.subject,
                    "" if row.point is None else row.point,
                    f"{row.carrier} ({row.digits} digits)"
                    if row.digits
                    else row.carrier or "",
                    _number(row.ratio),
                    _number(row.deviation),
                    _number(row.estimate),
                    verdict,
                    row.fallback or row.reason or "",
                )
            )
        blocks.append(table.get_string())
    verdict = (
        "PASSED"
        if report.exit_code == 0
        else ("FAILED" if report.exit_code == 1 else "NOTHING CHECKED")
    )
    blocks.append(
        _paint(
            f"check {verdict} (exit code {report.exit_code})",
            "GREEN" if report.exit_code == 0 else "RED",
            enabled=color,
            bold=True,
        )
    )
    return "\n\n".join(blocks)


__all__ = ["check_report_summary"]
