# SPDX-License-Identifier: 0BSD
"""Human tables of ``pyamplicol profile`` of a loop artifact
(``loop_profile.LoopProfileReport``): the runtime and the stability
statistics of the reduction backends side by side."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
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
        colour = palette.RECOVERED if kind.startswith("best-effort") else palette.BAD
        lines.append(
            palette.paint(label, colour, color=color)
            + body[0]
            + "".join("\n" + " " * len(label) + part for part in body[1:])
        )
    return "\n".join(lines)


def _truth(report: Any) -> tuple[str, str, str]:
    """The reference precision of a report (``LoopProfileReport.
    ground_truth``; double-double when it has none): its label ("dd",
    "arb(D)"), its name in titles ("double-double", "arb(D)") and its
    ``precision_timing`` key (the arithmetic)."""

    record = getattr(report, "ground_truth", None) or {}
    digits = record.get("digits")
    if record.get("arithmetic") == "none":
        return "none", "none", "none"
    if digits is None:
        return "dd", "double-double", "double-double"
    label = str(record.get("label") or f"arb({digits})")
    return label, label, str(record.get("arithmetic") or "arbitrary")


def _truth_cell(report: Any) -> str | None:
    """The header's ``ground truth`` value: the precision in words and its
    label in the tables."""

    record = getattr(report, "ground_truth", None)
    if not record:
        return None
    if record.get("arithmetic") == "none":
        return "none (estimates only)"
    digits = record.get("digits")
    if digits is None:
        return "double-double (dd)"
    return f"arbitrary precision, {digits} digits ({_truth(report)[0]})"


def _counts_cell(report: Any, color: bool) -> str | None:
    """The header's ``counts`` value: which rule set the counts (``Plan.
    rule``: --completion-time, the counts given, the time budgets), what it
    expected and what the profile took; a requested completion time that
    cannot be met in the warning colour, with the minimums' estimate."""

    from pyamplicol.profile_plan import (
        DEFAULT_RUNTIME_SECONDS,
        DEFAULT_STABILITY_SECONDS_PER_CORE,
        duration_text,
    )

    plan = getattr(report, "plan", None)
    if not plan:
        return None
    took = getattr(report, "seconds", None)
    took_text = "" if took is None else f", took {duration_text(float(took))}"
    rule = plan.get("rule")
    if rule == "completion time":
        budget = duration_text(float(plan.get("budget_seconds") or 0.0))
        text = f"from --completion-time {budget}{took_text}"
        overruled = plan.get("overruled") or []
        if overruled:
            text += f" (overrules {', '.join(map(str, overruled))})"
        if not plan.get("feasible", True):
            text += palette.paint(
                "; cannot be met: the minimums take about "
                + duration_text(float(plan.get("minimum_seconds") or 0.0)),
                palette.RECOVERED,
                color=color,
            )
        return text
    given = plan.get("given") or {}
    if rule == "counts" and not all(
        given.get(part, True) for part in ("runtime", "stability")
    ):
        # Some counts given, the rest from their budget.
        parts = []
        if plan.get("runtime") or plan.get("runtime_counts"):
            parts.append(
                "runtime as given"
                if given.get("runtime")
                else f"runtime from its {DEFAULT_RUNTIME_SECONDS:g} s budget"
            )
        if plan.get("stability_points") is not None:
            parts.append(
                f"{plan['stability_points']} stability points as given"
                if given.get("stability")
                else "stability points from "
                f"{duration_text(DEFAULT_STABILITY_SECONDS_PER_CORE)} per worker"
            )
        text = "; ".join(parts) + took_text
    elif rule == "counts":
        text = f"as given{took_text}"
    else:
        text = (
            f"from the time budgets: {DEFAULT_RUNTIME_SECONDS:g} s for the runtime "
            f"timing, {duration_text(DEFAULT_STABILITY_SECONDS_PER_CORE)} per "
            f"stability worker{took_text}"
        )
    if not plan.get("feasible", True):
        # A budget in force the minimums exceed (``Plan.unmet``).
        text += palette.paint(
            "; cannot be met: the minimums take about "
            + duration_text(float(plan.get("minimum_seconds") or 0.0)),
            palette.RECOVERED,
            color=color,
        )
    return text


def _header(report: Any, prettytable: Any, color: bool) -> str:
    setup = (
        (("runtime setup", report.runtime_setup),)
        if report.runtime_setup is not None
        else ()
    )
    truth = _truth_cell(report)
    reference = (("ground truth", truth),) if truth is not None else ()
    counts_text = _counts_cell(report, color)
    counts = (("counts", counts_text),) if counts_text is not None else ()
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
            *counts,
            *reference,
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


def _reference_ratio(ratio: float, label: str, color: bool) -> str:
    """``(<label> xN)`` after a binary64 time: the reference precision
    (``label``: "dd", "arb(D)") takes N times as long (one decimal below 10,
    none from 10 on), the whole parenthesis muted so that the time it
    follows (bold) dominates."""

    text = f"x{ratio:.1f}" if round(ratio, 1) < 10.0 else f"x{ratio:.0f}"
    return palette.paint(f"({label} {text})", palette.MUTED, color=color)


def _peak_cell(profile: Any, color: bool) -> str:
    """The ``peak RAM`` cell of a profiled backend (``resources.
    peak_memory_text``): a muted ``-`` when it is not measurable."""

    from .resources import peak_memory_text

    text = peak_memory_text(getattr(profile, "peak_memory", None))
    return palette.paint("-", palette.MUTED, color=color) if text is None else text


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
    fields = ["backend", "time / point", "min - max call", "vs fastest", "peak RAM"]
    fields += ["best-effort"] if show_best else []
    fields += ["lanes"] if show_lanes else []
    fields += ["failed"] if show_failed else []
    summary = palette.table(
        prettytable, tuple(fields), "Runtime per reduction backend", color=color
    )
    entries: list[list[Cell]] = []
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        cells: list[Cell]
        if row.profile is None:
            cells = [
                label,
                palette.paint("refused", palette.BAD, color=color, bold=True),
            ]
            entries.append([*cells, *[""] * (len(fields) - len(cells))])
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
            spread: Cell = ""
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
        reference = getattr(profile, "reference", None)
        time_parts: tuple[str, ...] = (time_cell,)
        if reference is not None and reference.samples == 0:
            # Ctrl-C before a reference call was back: no multiplier.
            time_parts += (
                palette.paint(f"({_truth(report)[0]} -)", palette.MUTED, color=color),
            )
        elif (
            reference is not None
            and profile.seconds_per_point > 0.0
            and reference.points > reference.failed_points
        ):
            time_parts += (
                _reference_ratio(
                    reference.seconds_per_point / profile.seconds_per_point,
                    _truth(report)[0],
                    color,
                ),
            )
        single = getattr(profile, "one_thread_seconds_per_point", None)
        if single and profile.seconds_per_point > 0.0:
            # --runtime-threads N: the same calls' time on one thread over
            # this one (the gain of the N threads), muted as the multiplier.
            gain = single / profile.seconds_per_point
            time_parts += (
                palette.paint(f"(x{gain:.1f} vs 1 thread)", palette.MUTED, color=color),
            )
        cells = [
            label,
            time_parts,
            spread,
            ratio,
            _peak_cell(profile, color),
        ]
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
            ordered = sorted(_lanes(row).items(), key=lambda i: i[0] != row.backend)
            cells.append(
                ", ".join(
                    palette.paint(
                        f"{_name(name)} {count}",
                        palette.BACKEND.get(name)
                        if name == row.backend
                        else palette.RECOVERED,
                        color=color,
                    )
                    for name, count in ordered
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
        entries.append(cells)
    blocks = [_fit(summary, [(cells, False) for cells in entries])]
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
    stage_rows: list[tuple[Sequence[Cell], bool]] = []
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
        stage_rows.append((cells, False))
    blocks.append(_fit(stages, stage_rows))
    return blocks


# The terminal colour codes of a painted text (not part of its width).
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# A table cell: one text, or its parts (a value, then what annotates it),
# on one line or one part per line (``_fit``).
Cell = str | tuple[str, ...]


def _visible(text: str) -> int:
    """The widest line of ``text`` without its colour codes."""

    return max((len(line) for line in _ANSI.sub("", text).splitlines()), default=0)


def _wrap_painted(text: str, width: int) -> str:
    """``text`` (one colour run around it, or none) wrapped at its spaces
    within ``width`` columns, each line painted as the whole was."""

    import textwrap

    found = re.fullmatch(r"((?:\x1b\[[0-9;]*m)*)(.*?)((?:\x1b\[[0-9;]*m)*)", text, re.S)
    if found is None:
        return text
    head, body, tail = found.groups()
    lines = textwrap.wrap(body, max(width, 1), break_on_hyphens=False) or [body]
    return "\n".join(head + line + tail for line in lines)


def _fit(table: Any, rows: Sequence[tuple[Sequence[Cell], bool]]) -> str:
    """``table`` with ``rows`` (its cells, and whether a divider follows the
    row) within the line width (``_width``): every cell's parts on one line;
    else, column by column (the one that saves the most width first), one
    part per line (a value over its annotation, as the time over its
    multiplier); else, as a last resort, the widest column narrowed
    step by step until the table fits, its texts wrapped at their spaces
    (uncoloured: the table's own wrapping would count the colour codes),
    never below its header or longest word unless nothing else is left. A
    column is as wide as its widest text or header: nothing that fits is
    wrapped."""

    import textwrap

    width = _width()
    names = list(table.field_names)
    floor = dict.fromkeys(names, 1)

    def joined(wrapped: set[int]) -> list[tuple[list[str], bool]]:
        return [
            (
                [
                    cell
                    if isinstance(cell, str)
                    else ("\n" if k in wrapped else " ").join(cell)
                    for k, cell in enumerate(cells)
                ],
                divider,
            )
            for cells, divider in rows
        ]

    def render(texts: list[tuple[list[str], bool]], limits: dict[str, int]) -> str:
        table.clear_rows()
        raw = dict.fromkeys(names, 1)
        body = []
        for cells, divider in texts:
            narrowed = []
            for name, text in zip(names, cells, strict=True):
                if _visible(text) > limits[name]:
                    text = "\n".join(
                        "\n".join(
                            textwrap.wrap(
                                line,
                                limits[name],
                                break_on_hyphens=False,
                                break_long_words=limits[name] < floor[name],
                            )
                            or [""]
                        )
                        for line in _ANSI.sub("", text).splitlines()
                    )
                raw[name] = max(raw[name], _visible(text))
                narrowed.append(text)
            body.append((narrowed, divider))
        # A header wider than its column goes on several lines (at its
        # spaces, painted as it is): the headers are then the first row,
        # a rule under them (a table header holds one line).
        widths = {name: max(limits[name], raw[name]) for name in names}
        table.header = all(_visible(name) <= widths[name] for name in names)
        if not table.header:
            shown = [
                _wrap_painted(name, widths[name])
                if _visible(name) > widths[name]
                else name
                for name in names
            ]
            for label, name in zip(shown, names, strict=True):
                raw[name] = max(raw[name], _visible(label))
            body.insert(0, (shown, True))
        for narrowed, divider in body:
            table.add_row(narrowed, divider=divider)
        table.max_width = raw
        return str(table.get_string())

    def natural(texts: list[tuple[list[str], bool]]) -> dict[str, int]:
        return {
            name: max([_visible(name)] + [_visible(t[k]) for t, _ in texts])
            for k, name in enumerate(names)
        }

    # A title wider than the line goes on several lines, each painted as
    # the title is.
    title = table.title or ""
    if _visible(title) > width - 4:
        found = re.fullmatch(
            r"((?:\x1b\[[0-9;]*m)*)(.*?)((?:\x1b\[[0-9;]*m)*)", title, re.S
        )
        if found is not None:
            head, body, tail = found.groups()
            table.title = "\n".join(
                head + line + tail
                for line in textwrap.wrap(body, width - 4, break_on_hyphens=False)
            )
    # One line per cell; else the annotated columns wrapped one at a time,
    # the one that saves the most width first, until the table fits.
    wrapped: set[int] = set()
    texts = joined(wrapped)
    text = render(texts, natural(texts))
    candidates = {
        k
        for cells, _ in rows
        for k, cell in enumerate(cells)
        if not isinstance(cell, str) and len(cell) > 1
    }
    while _visible(text) > width and candidates - wrapped:
        now = natural(texts)
        savings = {
            k: now[names[k]] - natural(joined(wrapped | {k}))[names[k]]
            for k in sorted(candidates - wrapped)
        }
        wrapped.add(max(savings, key=lambda k: savings[k]))
        texts = joined(wrapped)
        text = render(texts, natural(texts))
    if _visible(text) <= width:
        return text
    limits = natural(texts)
    # A column's floor: the longest word of its header and of its cells.
    floor.update(
        {
            name: max(
                [len(word) for word in _ANSI.sub("", name).split()]
                + [
                    len(word)
                    for cells, _ in texts
                    for word in _ANSI.sub("", cells[k]).split()
                ]
            )
            for k, name in enumerate(names)
        }
    )
    text = render(texts, limits)
    while _visible(text) > width:
        roomy = [name for name in names if limits[name] > floor[name]]
        if not roomy and width < _LINE_WIDTH:
            # A terminal narrower than the report's width: words and numbers
            # stay whole (the terminal wraps the rest).
            break
        if not roomy:
            # Every column at its floor: break the words of a column wider
            # than its header (a header is never narrowed), down to the
            # narrowest: the report's own width holds in every case.
            roomy = [
                name
                for name in names
                if limits[name]
                > max(
                    [_NARROWEST] + [len(word) for word in _ANSI.sub("", name).split()]
                )
            ]
            if not roomy:
                break
        widest = max(roomy, key=lambda name: limits[name] - floor[name])
        limits[widest] -= 1
        text = render(texts, limits)
    return text


# The narrowest a column is made to fit the line (a display constant).
_NARROWEST = 6


def _stability_time(row: Any, report: Any, color: bool) -> tuple[str, ...]:
    """The parts of the ``time`` cell of a backend's stability run: its
    binary64 calls' total wall time (with their checks, summed over the
    workers) and how many times longer its reference calls took in total
    (``_reference_ratio``); the sum of both (``row.seconds``) when the run
    has no split per precision."""

    label, _, key = _truth(report)
    timing = getattr(row, "precision_timing", None) or {}
    binary64 = timing.get("binary64", {}).get("seconds")
    reference = timing.get(key, {}).get("seconds")
    if binary64 is None:
        return (
            palette.paint(_seconds_text(row.seconds), None, color=color, bold=True),
        )
    text = palette.paint(_seconds_text(float(binary64)), None, color=color, bold=True)
    if reference is None or not binary64 > 0.0:
        return (text,)
    return text, _reference_ratio(float(reference) / float(binary64), label, color)


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


def _percent_parts(part: int, whole: int) -> tuple[str, ...]:
    """``part`` and its share of ``whole`` (the parts of a ``Cell``)."""

    return (str(part), f"({100 * part / whole:.1f} %)") if whole else (str(part),)


_LABELS = {0.001: "1 per mil", 0.01: "1 %", 0.1: "10 %", 0.5: "50 %"}


def interrupted_line(setup: Any, color: bool) -> str | None:
    """The notice of an interrupted stability run (Ctrl-C): how many of the
    requested points were measured, wrapped to the line width; None
    otherwise."""

    import textwrap

    if not setup or not setup.get("interrupted"):
        return None
    per = setup.get("evaluated_per_backend") or {}
    detail = ", ".join(f"{_name(name)} {count}" for name, count in per.items())
    text = (
        f"interrupted: {setup.get('measured_points', 0)} of "
        f"{setup.get('requested', 0)} "
        "requested points evaluated"
        + (
            f" by every backend ({detail}); the tables compare those points"
            if len(set(per.values())) > 1
            else "; the tables hold those points"
        )
    )
    return "\n".join(
        palette.paint(line, palette.RECOVERED, color=color, bold=True)
        for line in textwrap.wrap(text, _width(), break_on_hyphens=False)
    )


def _estimate_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    """The stability tables of a run without a reference (``--ground-truth
    none``): the estimated digits at the percentiles (no truth to compare
    them with) and the default checks' counts and the binary64 time."""

    setup = report.stability_setup
    target = float(setup["target"])
    target_digits = -math.log10(target)
    rows = report.stability
    served = [row for row in rows if row.refusal is None and row.points]
    points = (
        f"{setup.get('measured_points', 0)} of {setup['points']}"
        if setup.get("interrupted")
        else str(setup["points"])
    )
    table = palette.table(
        prettytable,
        ("percentile", "backend", "est. digits", "class"),
        f"Estimated digits of binary64 (no ground truth), {points} flat points "
        f"(sqrt(s) = {setup['sqrt_s']:g}, seed {setup['seed']})",
        color=color,
    )
    count = len(served[0].percentiles) if served else 0
    percentile_rows: list[tuple[Sequence[Cell], bool]] = []
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
            percentile_rows.append(
                (
                    (
                        _LABELS.get(item.percentile, f"{100 * item.percentile:g} %")
                        if position == 0
                        else "",
                        _backend_label(row.backend, report.default_backend, color),
                        digits,
                        str(item.count),
                    ),
                    position == len(served) - 1 and index < count - 1,
                )
            )
    blocks = [_fit(table, percentile_rows)]
    verdicts = palette.table(
        prettytable,
        ("backend", "evaluated", "best-eff./none", "flagged", "time"),
        f"Default checks, estimates only (target {target:g} = "
        f"{target_digits:.2f} digits)",
        color=color,
    )
    entries: list[tuple[Sequence[Cell], bool]] = []
    refusals = []
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        if row.refusal is not None:
            entries.append(
                (
                    (
                        label,
                        palette.paint("refused", palette.BAD, color=color, bold=True),
                        "",
                        "",
                        "",
                    ),
                    False,
                )
            )
            refusals.append(f"{label} refused: {row.refusal}")
            continue
        entries.append(
            (
                (
                    label,
                    str(row.points),
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
                    _percent_parts(row.flagged, row.points),
                    _stability_time(row, report, color),
                ),
                False,
            )
        )
    blocks.append(_fit(verdicts, entries))
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


def _stability_tables(report: Any, prettytable: Any, color: bool) -> list[str]:
    if any(getattr(row, "estimate_only", False) for row in report.stability or ()):
        return _estimate_tables(report, prettytable, color)
    setup = report.stability_setup
    target = float(setup["target"])
    target_digits = -math.log10(target)
    rows = report.stability
    served = [row for row in rows if row.refusal is None and row.points]
    points = (
        f"{setup.get('measured_points', 0)} of {setup['points']}"
        if setup.get("interrupted")
        else str(setup["points"])
    )
    title = (
        f"Stable digits of binary64 vs {_truth(report)[1]}, {points} flat "
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
    count = len(served[0].percentiles) if served else 0
    percentile_rows: list[tuple[Sequence[Cell], bool]] = []
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
            counted: Cell = (str(item.count),) + (
                (palette.paint(f"({item.failed} failed)", palette.BAD, color=color),)
                if item.failed
                else ()
            )
            percentile_rows.append(
                (
                    (
                        _LABELS.get(item.percentile, f"{100 * item.percentile:g} %")
                        if position == 0
                        else "",
                        _backend_label(row.backend, report.default_backend, color),
                        digits,
                        counted,
                        spread,
                        _signed(item.signed_median, color),
                    ),
                    position == len(served) - 1 and index < count - 1,
                )
            )
    blocks = [_fit(table, percentile_rows)]
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
    entries: list[Sequence[Cell]] = []
    refusals = []
    for row in rows:
        label = _backend_label(row.backend, report.default_backend, color)
        if row.refusal is not None:
            entries.append(
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
        measured: Cell = (str(row.points),) + (
            (
                palette.paint(
                    f"(+{row.failed_reference} no ref.)", palette.MUTED, color=color
                ),
            )
            if row.failed_reference
            else ()
        )
        entries.append(
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
                _percent_parts(row.flagged, row.points),
                f"{row.flagged_inaccurate} of {row.flagged}",
                tuple(
                    palette.paint(
                        part,
                        palette.GOOD if row.unflagged_inaccurate == 0 else palette.BAD,
                        color=color,
                        bold=row.unflagged_inaccurate > 0,
                    )
                    for part in _percent_parts(row.unflagged_inaccurate, row.points)
                ),
                _stability_time(row, report, color),
            )
        )
    blocks.append(_fit(verdicts, [(cells, False) for cells in entries]))
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
    best_rows: list[tuple[Sequence[Cell], bool]] = []
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

        best_rows.append(
            (
                (
                    _backend_label(row.backend, report.default_backend, color),
                    str(len(points)),
                    " / ".join(cell(v) for v in (true[0], med(true), true[-1])),
                    cell(med(known)),
                    cell(med(gaps)),
                    (f"{row.best_effort_covered} of {len(known)}",)
                    + (
                        (
                            palette.paint(
                                f"({len(points) - len(known)} without)",
                                palette.MUTED,
                                color=color,
                            ),
                        )
                        if len(known) < len(points)
                        else ()
                    ),
                ),
                False,
            )
        )
    return _fit(table, best_rows)


def loop_profile_summary(report: Any, *, prettytable: Any, color: bool) -> str:
    """The header, the runtime tables and the stability tables (whichever
    the report holds)."""

    blocks = [_header(report, prettytable, color)]
    if report.runtime is not None:
        blocks += _runtime_tables(report, prettytable, color)
    if report.interrupted and report.stability is None:
        blocks.append(
            palette.paint(
                "interrupted: the runtime table holds the backends timed so far",
                palette.RECOVERED,
                color=color,
                bold=True,
            )
        )
    if report.stability is not None:
        blocks += _stability_tables(report, prettytable, color)
    return "\n\n".join(blocks)


__all__ = ["loop_profile_summary"]
