# SPDX-License-Identifier: 0BSD
"""Complete Laurent/order tables without narrowing high-precision values."""

from __future__ import annotations

from typing import Any

from pyamplicol.api.virtual_results import VirtualEvaluation, VirtualScalar
from pyamplicol.correlators import CorrelatedValue
from pyamplicol.models.base import CouplingOrders


def _order(order: CouplingOrders) -> str:
    return ", ".join(f"{name}^{power}" for name, power in order) or "1"


def _scalar(value: VirtualScalar) -> str:
    # Decimal.__str__ preserves its stored digits without using the active context.
    if isinstance(value, CorrelatedValue):
        return f"({value.real}) + ({value.imag})i"
    return str(value)


def virtual_evaluation_summary(result: VirtualEvaluation, *, prettytable: Any) -> str:
    """Render all sectors, poles, pairing metadata and per-lane diagnostics."""
    born = prettytable.PrettyTable(("point", "Born squared order", "Born"))
    virtual = prettytable.PrettyTable(
        (
            "point",
            "virtual squared order",
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
    for point, row in enumerate(result.born_values):
        for index, order in enumerate(result.born_orders):
            born.add_row(
                (
                    point,
                    _order(order),
                    "unavailable" if row is None else _scalar(row[index]),
                )
            )
    for point, virtual_row in enumerate(result.values):
        for index, sector in enumerate(result.sectors):
            values = (
                ("unavailable",) * 3
                if virtual_row is None
                else tuple(map(_scalar, virtual_row[index]))
            )
            virtual.add_row((point, _order(sector.order), *values))
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
        "epsilon powers (0, -1, -2).\n"
        f"R2 requested={result.r2.requested.value}; "
        f"effective={result.r2.effective.value}; "
        f"{result.r2.reason}"
    )
    return "\n\n".join(
        (header, *(str(table.get_string()) for table in (born, virtual, pairs, status)))
    )
