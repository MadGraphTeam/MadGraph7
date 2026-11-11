# SPDX-License-Identifier: 0BSD
"""Complete Laurent/order tables without narrowing high-precision values."""

from __future__ import annotations

from typing import Any

from pyamplicol.api.virtual_results import (
    ResolvedVirtualEvaluation,
    VirtualEvaluation,
    VirtualScalar,
)
from pyamplicol.config import PoleSelection
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.models.base import CouplingOrders

from . import palette


def _order(order: CouplingOrders) -> str:
    return ", ".join(f"{name}^{power}" for name, power in order) or "1"


def _scalar(value: VirtualScalar | None) -> str:
    # Decimal.__str__ preserves its stored digits without using the active context.
    if value is None:
        return "absent"  # a finite-only evaluation computed no poles
    if isinstance(value, CorrelatedValue):
        return f"({value.real}) + ({value.imag})i"
    return str(value)


def _target(result: Any) -> float:
    """The accuracy target the call's estimates are read against: the
    reduction's recorded target, else the default target (1e-5)."""

    for lane in result.diagnostics:
        reduction = getattr(lane, "reduction", None)
        if reduction is not None and reduction.target is not None:
            return float(reduction.target)
    from pyamplicol.loops.v2.reduction import default_target

    return float(default_target())


def _estimate_row(estimate: Any, target: float, color: bool) -> tuple[str, ...]:
    def text(value: float | None) -> str:
        if value is None:
            return "absent"
        return palette.within(value, target, color=color, text=f"{value:.3e}")

    return (
        text(estimate.relative),
        text(estimate.finite),
        text(estimate.single_pole),
        text(estimate.double_pole),
        ", ".join(
            f"{palette.paint(name, palette.ESTIMATE, color=color)} {value:.2e}"
            for name, value in estimate.checks
        ),
    )


def _axes(resolved: bool, color: bool, helicity: Any, component: Any) -> tuple:
    if not resolved:
        return ()
    return (
        palette.paint(helicity, palette.HELICITY, color=color),
        palette.paint(component, palette.COMPONENT, color=color),
    )


def _stability_table(result: Any, prettytable: Any, color: bool = False) -> Any:
    """Per-point (resolved: per-entry) stability estimates, or None."""

    if result.stability is None:
        return None
    resolved = isinstance(result, ResolvedVirtualEvaluation)
    axes = ("helicity", "colour component") if resolved else ()
    target = _target(result)
    table = palette.table(
        prettytable,
        ("point", *axes, "relative", "finite", "single pole", "double pole", "checks"),
        "Stability estimates (relative; requested checks only; green within "
        f"the target {target:g}, red above)",
        color=color,
    )
    for point, estimate in enumerate(result.stability):
        if estimate is None:
            table.add_row(
                (
                    point,
                    *("",) * len(axes),
                    palette.paint("unavailable", palette.BAD, color=color),
                    "",
                    "",
                    "",
                    "",
                )
            )
        elif resolved:
            for h, helicity in enumerate(result.helicity_ids):
                for c, component in enumerate(result.virtual_color_ids):
                    table.add_row(
                        (
                            point,
                            *_axes(True, color, helicity, component),
                            *_estimate_row(estimate[h][c], target, color),
                        )
                    )
        else:
            table.add_row((point, *_estimate_row(estimate, target, color)))
    return table


def _pole_parts_table(result: Any, prettytable: Any, color: bool = False) -> Any:
    """Both origins' pole parts per component (``poles`` "uv" or "ir"), or
    None; resolved results per helicity and colour component."""

    parts = getattr(result, "pole_parts", None)
    if parts is None:
        return None
    resolved = isinstance(result, ResolvedVirtualEvaluation)
    axes = ("helicity", "colour component") if resolved else ()
    table = palette.table(
        prettytable,
        (
            "point",
            "virtual squared order",
            *axes,
            "origin",
            "component",
            "single pole [-1]",
            "double pole [-2]",
        ),
        "Pole parts by origin (P/eps = P_UV/eps_UV + P_IR/eps_IR)",
        color=color,
    )

    def entries(value: Any) -> Any:
        if not resolved:
            yield (), value
            return
        for h, helicity in enumerate(result.helicity_ids):
            for c, component in enumerate(result.virtual_color_ids):
                yield _axes(True, color, helicity, component), value[h][c]

    for point, lane in enumerate(parts):
        if lane is None:
            table.add_row(
                (
                    point,
                    "",
                    *("",) * len(axes),
                    "",
                    "",
                    palette.paint("unavailable", palette.BAD, color=color),
                    "",
                )
            )
            continue
        for index, sector in enumerate(result.sectors):
            for origin in ("uv", "ir"):
                for component, values in lane.origin(origin).items():
                    for labels, laurent in entries(values[index]):
                        table.add_row(
                            (
                                point,
                                palette.paint(
                                    _order(sector.order), palette.ORDER, color=color
                                ),
                                *labels,
                                origin,
                                component,
                                palette.paint(
                                    _scalar(laurent.single_pole),
                                    palette.SINGLE_POLE,
                                    color=color,
                                ),
                                palette.paint(
                                    _scalar(laurent.double_pole),
                                    palette.DOUBLE_POLE,
                                    color=color,
                                ),
                            )
                        )
    return table


def _laurent_cells(values: Any, color: bool) -> tuple[str, ...]:
    """A Laurent triple's cells: the finite part, the single and the double
    pole in their own colours (``unavailable`` in red for a failed lane)."""

    if values is None:
        return (palette.paint("unavailable", palette.BAD, color=color),) * 3
    finite, single, double = map(_scalar, values)
    return (
        palette.paint(finite, palette.FINITE, color=color, bold=True),
        palette.paint(single, palette.SINGLE_POLE, color=color),
        palette.paint(double, palette.DOUBLE_POLE, color=color),
    )


def _reduction_text(reduction: Any, fallbacks: set[str], color: bool) -> str:
    """The lane's effective reduction backend (coloured): "auto" calls name
    their per-point fallback; a lane the fallback served (its backend is
    another lane's fallback) is marked in yellow, as recovered."""

    if reduction is None:
        return ""
    effective = str(reduction.effective)
    requested = str(reduction.requested)
    served_by_fallback = (
        requested == "auto" and reduction.fallback is None and effective in fallbacks
    )
    text = palette.paint(
        effective,
        palette.RECOVERED if served_by_fallback else palette.BACKEND.get(effective),
        color=color,
        bold=True,
    )
    if served_by_fallback:
        text += " " + palette.paint(
            "(per-point fallback of auto)", palette.RECOVERED, color=color
        )
    elif requested == "auto":
        text += " (auto" + (
            f"; per-point fallback {reduction.fallback})"
            if reduction.fallback is not None
            else ")"
        )
    if reduction.unchecked:
        text += " " + palette.paint("unchecked", palette.RECOVERED, color=color)
    return text


def virtual_evaluation_summary(
    result: VirtualEvaluation | ResolvedVirtualEvaluation,
    *,
    prettytable: Any,
    color: bool = False,
) -> str:
    """Render all sectors, poles, pairing metadata and per-lane diagnostics
    (coloured when ``color``: ``reporting.palette``)."""
    resolved = isinstance(result, ResolvedVirtualEvaluation)
    axes = (
        (
            "helicity",
            "Born anchor" if result.color_accuracy == "lc" else "colour component",
        )
        if resolved
        else ()
    )
    loop_induced = getattr(result, "contribution", "virtual") == "loop-induced"
    born = palette.table(
        prettytable,
        ("point", "Born squared order", *axes, "Born"),
        "Born contributions",
        color=color,
    )
    virtual = palette.table(
        prettytable,
        (
            "point",
            "virtual squared order",
            *axes,
            "finite [0]",
            "single pole [-1]",
            "double pole [-2]",
        ),
        "|A_loop|^2 (loop-induced)" if loop_induced else "Born-loop interference",
        color=color,
    )
    pairs = palette.table(
        prettytable,
        ("virtual squared order", "Born x loop orders"),
        "Contributing order pairs",
        color=color,
    )
    status = palette.table(
        prettytable,
        (
            "point",
            "status",
            "arithmetic",
            "bits",
            "reduction",
            "accuracy established",
            "detail",
        ),
        "Evaluation diagnostics",
        color=color,
    )

    def order_cell(order: CouplingOrders) -> str:
        return palette.paint(_order(order), palette.ORDER, color=color)

    def born_cell(value: Any) -> str:
        if value is None:
            return palette.paint("unavailable", palette.BAD, color=color)
        return palette.paint(_scalar(value), palette.BORN, color=color)

    if isinstance(result, ResolvedVirtualEvaluation):
        for point, row in enumerate(result.born_values):
            for order_index, order in enumerate(result.born_orders):
                for h, helicity in enumerate(result.helicity_ids):
                    for c, component in enumerate(result.born_color_ids):
                        born.add_row(
                            (
                                point,
                                order_cell(order),
                                *_axes(True, color, helicity, component),
                                born_cell(
                                    None if row is None else row[order_index][h][c]
                                ),
                            )
                        )
        for point, virtual_row in enumerate(result.values):
            for order_index, sector in enumerate(result.sectors):
                for h, helicity in enumerate(result.helicity_ids):
                    for c, component in enumerate(result.virtual_color_ids):
                        virtual.add_row(
                            (
                                point,
                                order_cell(sector.order),
                                *_axes(True, color, helicity, component),
                                *_laurent_cells(
                                    None
                                    if virtual_row is None
                                    else virtual_row[order_index][h][c],
                                    color,
                                ),
                            )
                        )
    else:
        for point, inclusive_row in enumerate(result.born_values):
            for index, order in enumerate(result.born_orders):
                born.add_row(
                    (
                        point,
                        order_cell(order),
                        born_cell(
                            None if inclusive_row is None else inclusive_row[index]
                        ),
                    )
                )
        for point, inclusive_virtual_row in enumerate(result.values):
            for index, sector in enumerate(result.sectors):
                virtual.add_row(
                    (
                        point,
                        order_cell(sector.order),
                        *_laurent_cells(
                            None
                            if inclusive_virtual_row is None
                            else inclusive_virtual_row[index],
                            color,
                        ),
                    )
                )
    for sector in result.sectors:
        pairs.add_row(
            (
                order_cell(sector.order),
                "; ".join(
                    f"({_order(pair.born)}) x ({_order(pair.loop)})"
                    for pair in sector.pairs
                ),
            )
        )
    fallbacks = {
        str(lane.reduction.fallback)
        for lane in result.diagnostics
        if getattr(lane, "reduction", None) is not None
        and lane.reduction.fallback is not None
    }
    for point, item in enumerate(result.diagnostics):
        status.add_row(
            (
                point,
                palette.paint(
                    item.status,
                    palette.STATUS.get(item.status),
                    color=color,
                    bold=True,
                ),
                palette.paint(
                    item.arithmetic,
                    palette.ARITHMETIC.get(item.arithmetic),
                    color=color,
                ),
                item.precision_bits,
                _reduction_text(getattr(item, "reduction", None), fallbacks, color),
                palette.paint(
                    "yes" if item.accuracy_established else "no",
                    palette.GOOD if item.accuracy_established else palette.MUTED,
                    color=color,
                ),
                item.detail,
            )
        )
    header = (
        palette.paint("Virtual coefficients", palette.TITLE, color=color, bold=True)
        + ": Cepsilon=(4*pi)^epsilon/Gamma(1-epsilon); "
        "epsilon powers ("
        + palette.paint("0", palette.FINITE, color=color)
        + ", "
        + palette.paint("-1", palette.SINGLE_POLE, color=color)
        + ", "
        + palette.paint("-2", palette.DOUBLE_POLE, color=color)
        + "); "
        f"colour accuracy={result.color_accuracy.upper()}.\n"
        f"R2 requested={result.r2.requested.value}; "
        f"effective={result.r2.effective.value}; "
        f"{result.r2.reason}"
    )
    if result.correlated:
        header += (
            "\nDirected colour correlations retain complex Born "
            "and Laurent coefficients."
        )
    if result.color_accuracy == "lc" and resolved:
        header += (
            "\nBorn anchors are signed, potentially overcomplete components; "
            "they are not sampling probabilities."
        )
    if result.poles is PoleSelection.NONE:
        header += "\nFinite-only evaluation: the poles were not computed."
    elif result.poles is PoleSelection.UV:
        header += (
            "\nPoles: the ultraviolet poles of the bare loop amplitude; both "
            "origins' parts per component, counterterms included, follow."
        )
    elif result.poles is PoleSelection.IR:
        header += (
            "\nPoles: the infrared poles of the renormalised virtual (the "
            "double pole is infrared); both origins' parts per component follow."
        )
    # A loop-induced result has no Born: its Born table would stay empty.
    tables = [virtual, pairs, status]
    if not loop_induced:
        tables.insert(0, born)
    parts = _pole_parts_table(result, prettytable, color)
    if parts is not None:
        tables.append(parts)
    stability = _stability_table(result, prettytable, color)
    if stability is not None:
        tables.append(stability)
    out = "\n\n".join((header, *(str(table.get_string()) for table in tables)))
    return out + warnings_line(result, color)


def warnings_line(result: Any, color: bool) -> str:
    """The one dim line under a result's tables that names its captured
    warnings (``result.warnings``, m-lit ux-besteffort), or nothing."""

    from pyamplicol.api.result_warnings import summary_line

    line = summary_line(tuple(getattr(result, "warnings", ())))
    return (
        "" if line is None else "\n" + palette.paint(line, palette.MUTED, color=color)
    )


def virtual_order_summary(result: Any, *, prettytable: Any, color: bool = False) -> str:
    """A ``VirtualOrderEvaluation``: the evaluation's tables (every sector
    and Born order apart) followed by the explicit sums over the squared
    orders, labelled as such."""

    table = palette.table(
        prettytable,
        (
            "point",
            "Born summed over orders",
            "finite [0]",
            "single pole [-1]",
            "double pole [-2]",
        ),
        "Explicit sums over the squared coupling orders (sum_orders)",
        color=color,
    )
    born = result.sum_born_orders()
    for point, laurent in enumerate(result.sum_orders()):
        table.add_row(
            (
                point,
                palette.paint("unavailable", palette.BAD, color=color)
                if born[point] is None
                else palette.paint(_scalar(born[point]), palette.BORN, color=color),
                *_laurent_cells(laurent, color),
            )
        )
    return (
        virtual_evaluation_summary(
            result.evaluation, prettytable=prettytable, color=color
        )
        + "\n\n"
        + str(table.get_string())
    )


# Bar colours of the profile's stage groups (``runtime.virtual.PROFILE_GROUPS``).
_STAGE_COLOURS = {
    "trees": "CYAN",
    "dressing": "MAGENTA",
    "scatter": "YELLOW",
    "reduction": "GREEN",
    "rest": "WHITE",
}
_BAR_WIDTH = 40


def virtual_profile_summary(result: Any, *, prettytable: Any, color: bool) -> str:
    """A ``benchmarking.VirtualProfileResult``: the run (process, backend,
    setting, batch, samples, per-point time) and the coloured per-stage table
    (seconds per point, share and a bar per stage group)."""

    from .summary import _paint, _seconds_text

    reduction = result.reduction
    backend = reduction.get("effective", "N/A")
    if reduction.get("requested") and reduction.get("requested") != backend:
        backend = f"{backend} (requested {reduction['requested']})"
    lanes = reduction.get("lanes") or {}
    if len(lanes) > 1:
        backend += "; lanes per effective backend: " + ", ".join(
            f"{name} {count}" for name, count in sorted(lanes.items())
        )
    header = prettytable.PrettyTable(("field", "value"))
    header.title = _paint("pyAmpliCol loop profile", "CYAN", enabled=color, bold=True)
    header.align = "l"
    header.max_width = 96
    header.add_rows(
        (
            ("process", f"{result.process_id} ({result.process_expression})"),
            ("contribution", result.contribution),
            ("reduction", backend),
            (
                "setting",
                "unchecked (timing mode, no accuracy certificate)"
                if result.unchecked
                else "checked (the backends' own acceptance checks)",
            ),
            (
                "batch",
                f"{result.batch} points per call, finite part only; "
                f"{result.warmup} warm-up + {result.samples} timed calls",
            ),
            (
                "time per point",
                f"{_seconds_text(result.seconds_per_point)} (median call; "
                f"{_seconds_text(result.low)} to {_seconds_text(result.high)})",
            ),
            ("failed points", str(result.failed_points)),
        )
    )
    stages = prettytable.PrettyTable(("stage", "time per point", "share", "bar"))
    stages.title = (
        "Time per stage (native executor stage report of the timed calls)"
        if result.label is None
        else f"Time per stage ({result.label})"
    )
    stages.align = "l"
    total = sum(result.stages.values()) or 1.0
    for group, seconds in result.stages.items():
        share = seconds / total
        bar = "#" * max(round(share * _BAR_WIDTH), 1 if seconds else 0)
        stages.add_row(
            (
                _paint(group, _STAGE_COLOURS.get(group), enabled=color, bold=True),
                _seconds_text(seconds),
                f"{100 * share:5.1f} %",
                _paint(bar, _STAGE_COLOURS.get(group), enabled=color),
            )
        )
    return str(header.get_string()) + "\n\n" + str(stages.get_string())
