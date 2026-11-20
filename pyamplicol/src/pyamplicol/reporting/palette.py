# SPDX-License-Identifier: 0BSD
"""The colour roles of the human loop tables (evaluate and profile).

One place names what each kind of cell means, so the evaluate and the profile
tables read alike: the Laurent coefficients apart (finite part, single and
double pole), the axes (squared order, helicity, colour component), the
carrier and the reduction backend, and the verdicts (green: within the
target or a direct success; yellow: recovered, by a higher-precision rescue
or another backend; red: unstable, failed or refused). Colour is applied by
``summary._paint`` only when the output's colour setting allows it
(``output.color``: auto, always or never); JSON output is never coloured.
"""

from __future__ import annotations

from typing import Any

TITLE = "CYAN"
ORDER = "MAGENTA"
BORN = "CYAN"
FINITE = "GREEN"
SINGLE_POLE = "YELLOW"
DOUBLE_POLE = "LIGHTBLUE_EX"
HELICITY = "LIGHTCYAN_EX"
COMPONENT = "LIGHTMAGENTA_EX"
ESTIMATE = "LIGHTYELLOW_EX"
GOOD = "GREEN"
RECOVERED = "YELLOW"
BAD = "RED"
MUTED = "LIGHTBLACK_EX"

ARITHMETIC = {
    "double": "LIGHTBLUE_EX",
    "double-double": "MAGENTA",
    "arbitrary": "LIGHTMAGENTA_EX",
}
BACKEND = {"tirust": "LIGHTCYAN_EX", "ninjars": "LIGHTYELLOW_EX"}
STATUS = {
    "success": GOOD,
    "rescued": RECOVERED,
    # A value without an accepted accuracy (m-lit ux-besteffort).
    "best-effort": RECOVERED,
    "accuracy-not-established": BAD,
    "unsupported": BAD,
    "failed": BAD,
}


def paint(text: object, colour: str | None, *, color: bool, bold: bool = False) -> str:
    from .summary import _paint

    return _paint(str(text), colour, enabled=color, bold=bold)


def table(prettytable: Any, fields: tuple[str, ...], title: str, *, color: bool) -> Any:
    """A left-aligned framed table with a bold coloured title and bold
    column headers (the plain texts without colour)."""

    out = prettytable.PrettyTable(
        tuple(paint(name, None, color=color, bold=True) for name in fields)
    )
    out.title = paint(title, TITLE, color=color, bold=True)
    out.align = "l"
    out.hrules = prettytable.HRuleStyle.FRAME
    return out


def within(value: float | None, target: float, *, color: bool, text: str) -> str:
    """``text`` green when ``value`` is within ``target``, red otherwise
    (also for an infinite estimate), uncoloured when there is no value."""

    if value is None:
        return text
    return paint(text, GOOD if value <= target else BAD, color=color)
