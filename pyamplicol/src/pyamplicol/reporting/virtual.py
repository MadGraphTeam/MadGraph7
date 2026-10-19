# SPDX-License-Identifier: 0BSD
"""Complete Laurent/order tables without narrowing high-precision values."""

from __future__ import annotations

from typing import Any

from pyamplicol.api.virtual_results import (
    ResolvedVirtualEvaluation,
    VirtualEvaluation,
    VirtualScalar,
)
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.models.base import CouplingOrders


def _order(order: CouplingOrders) -> str:
    return ", ".join(f"{name}^{power}" for name, power in order) or "1"


def _scalar(value: VirtualScalar) -> str:
    # Decimal.__str__ preserves its stored digits without using the active context.
    if isinstance(value, CorrelatedValue):
        return f"({value.real}) + ({value.imag})i"
    return str(value)


def virtual_evaluation_summary(
    result: VirtualEvaluation | ResolvedVirtualEvaluation, *, prettytable: Any
) -> str:
    """Render all sectors, poles, pairing metadata and per-lane diagnostics."""
    resolved = isinstance(result, ResolvedVirtualEvaluation)
    axes = (
        (
            "helicity",
            "Born anchor" if result.color_accuracy == "lc" else "colour component",
        )
        if resolved
        else ()
    )
    born = prettytable.PrettyTable(("point", "Born squared order", *axes, "Born"))
    virtual = prettytable.PrettyTable(
        (
            "point",
            "virtual squared order",
            *axes,
            "finite [0]",
            "single pole [-1]",
            "double pole [-2]",
        )
    )
    pairs = prettytable.PrettyTable(("virtual squared order", "Born x loop orders"))
    status = prettytable.PrettyTable(
        ("point", "status", "arithmetic", "bits", "accuracy established", "detail")
    )
    for table, title in (
        (born, "Born contributions"),
        (virtual, "Born-loop interference"),
        (pairs, "Contributing order pairs"),
        (status, "Evaluation diagnostics"),
    ):
        table.title = title
        table.align = "l"
        table.hrules = prettytable.HRuleStyle.FRAME
    if isinstance(result, ResolvedVirtualEvaluation):
        for point, row in enumerate(result.born_values):
            for order_index, order in enumerate(result.born_orders):
                for h, helicity in enumerate(result.helicity_ids):
                    for c, component in enumerate(result.born_color_ids):
                        born.add_row(
                            (
                                point,
                                _order(order),
                                helicity,
                                component,
                                "unavailable"
                                if row is None
                                else _scalar(row[order_index][h][c]),
                            )
                        )
        for point, virtual_row in enumerate(result.values):
            for order_index, sector in enumerate(result.sectors):
                for h, helicity in enumerate(result.helicity_ids):
                    for c, component in enumerate(result.virtual_color_ids):
                        coefficients = (
                            ("unavailable",) * 3
                            if virtual_row is None
                            else tuple(map(_scalar, virtual_row[order_index][h][c]))
                        )
                        virtual.add_row(
                            (
                                point,
                                _order(sector.order),
                                helicity,
                                component,
                                *coefficients,
                            )
                        )
    else:
        for point, inclusive_row in enumerate(result.born_values):
            for index, order in enumerate(result.born_orders):
                born.add_row(
                    (
                        point,
                        _order(order),
                        "unavailable"
                        if inclusive_row is None
                        else _scalar(inclusive_row[index]),
                    )
                )
        for point, inclusive_virtual_row in enumerate(result.values):
            for index, sector in enumerate(result.sectors):
                coefficients = (
                    ("unavailable",) * 3
                    if inclusive_virtual_row is None
                    else tuple(map(_scalar, inclusive_virtual_row[index]))
                )
                virtual.add_row((point, _order(sector.order), *coefficients))
    for sector in result.sectors:
        pairs.add_row(
            (
                _order(sector.order),
                "; ".join(
                    f"({_order(pair.born)}) x ({_order(pair.loop)})"
                    for pair in sector.pairs
                ),
            )
        )
    for point, item in enumerate(result.diagnostics):
        status.add_row(
            (
                point,
                item.status,
                item.arithmetic,
                item.precision_bits,
                "yes" if item.accuracy_established else "no",
                item.detail,
            )
        )
    header = (
        "Virtual coefficients: Cepsilon=(4*pi)^epsilon/Gamma(1-epsilon); "
        "epsilon powers (0, -1, -2); "
        f"colour accuracy={result.color_accuracy.upper()}.\n"
        f"R2 requested={result.r2.requested.value}; "
        f"effective={result.r2.effective.value}; "
        f"{result.r2.reason}"
    )
    if result.color_accuracy == "lc" and resolved:
        header += (
            "\nBorn anchors are signed, potentially overcomplete components; "
            "they are not sampling probabilities."
        )
    return "\n\n".join(
        (header, *(str(table.get_string()) for table in (born, virtual, pairs, status)))
    )
