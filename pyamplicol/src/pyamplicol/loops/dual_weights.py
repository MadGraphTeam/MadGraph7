# SPDX-License-Identifier: 0BSD
"""Exact dual-Born weights shared independently of denominator families.

Only Symbolica evaluator instructions and current-column identities are kept.
The map runs once per native point/source row; no event values are cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .physical import _saved

if TYPE_CHECKING:
    from .color_components import BornColorComponents
    from .color_orbits import RectangularColorCertificate
    from .physical_color import FormalColorMetrics


@dataclass(frozen=True, slots=True)
class PreparedDualBornWeights:
    state: bytes
    # Each output retains direction, squared-order sector and physical loop term.
    columns: tuple[tuple[int, int, int], ...]
    directions: int
    color_certificate: DualBornColorCertificate | None = None

    def bind(self, bare):
        if self.color_certificate is not None:
            self.color_certificate.require_owner(bare)
        return bare.backend.CoefficientEvaluator(self.state, **bare.options)

    def family_columns(self, bare, indices, sectors):
        terms = {
            bare.loop_term_offsets[index] + local
            for index in indices
            for local in range(len(bare.loop_reduction_powers[index]))
        }
        return tuple(
            column
            for column, (_, sector, term) in enumerate(self.columns)
            if sector in sectors and term in terms
        )


@dataclass(frozen=True, slots=True)
class DualBornColorCertificate:
    """Cold exact operator banks linked to their common-degree metric owner.

    These are ordinary/resolved Born weights, not directed inserted operators.
    No Fourier execution is selected by this record. The saved exact evaluator
    remains the numerical schedule until a native transform backend is admitted.
    """

    formal_metrics: FormalColorMetrics
    rectangle: RectangularColorCertificate
    # Explicit direction/component and squared-order sector -> rectangle bank.
    banks: tuple[tuple[int, int, str], ...]
    resolved: bool
    component_owner: BornColorComponents | None = None

    def require_owner(self, bare):
        self.formal_metrics.require_owner(bare)
        if self.resolved and self.component_owner is not bare.born_color_components:
            raise ValueError(
                "colour certificate has a different resolved-component owner"
            )


def prepare_certified_dual_born_weights(bare, labels, *, resolved=False):
    """Certify actual formal rectangles before external averaging/substitution.

    Action labels are supplied explicitly and proved using dependency tensors.
    Denominator families never enter this domain. Missing formal data cannot be
    reconstructed from bound execution weights; callers retain the direct plan.
    """
    from symbolica import E

    from .color_orbits import prepare_rectangular_color_certificate
    from .physical_color import FormalColorMetrics

    formal = getattr(bare, "formal_color_metrics", None)
    if not isinstance(formal, FormalColorMetrics):
        raise ValueError(
            "colour certification requires retained formal projection data"
        )
    formal.require_owner(bare)
    components = bare.born_color_components if resolved else None
    metrics = components.mask_bra(formal.loop) if resolved else (formal.loop,)
    banks, bank_keys = {}, []
    for direction, metric in enumerate(metrics):
        for sector, selected in enumerate(formal.virtual_pairs):
            pairs = frozenset(selected)
            name = f"direction{direction}:sector{sector}"
            bank_keys.append((direction, sector, name))
            banks[name] = tuple(
                tuple(
                    value
                    if (formal.born_orders[i], formal.loop_orders[j]) in pairs
                    else E("0")
                    for j, value in enumerate(row)
                )
                for i, row in enumerate(metric)
            )
    rectangle = prepare_rectangular_color_certificate(
        formal.born_tensors,
        formal.loop_tensors,
        born_orders=formal.born_orders,
        loop_orders=formal.loop_orders,
        labels=labels,
        metrics=banks,
        nc=formal.nc,
    )
    bound = tuple(
        tuple(
            tuple(value.replace(formal.nc, 3) / formal.initial_average for value in row)
            for row in metric
        )
        for metric in metrics
    )
    weights = prepare_dual_born_weights(bare, bound)
    return PreparedDualBornWeights(
        weights.state,
        weights.columns,
        weights.directions,
        DualBornColorCertificate(
            formal, rectangle, tuple(bank_keys), resolved, components
        ),
    )


def prepare_dual_born_weights(bare, metrics=None):
    """Compile each exact order/metric bank before any family routing scatter."""
    from symbolica import E, S

    metrics = (bare.loop_metric,) if metrics is None else tuple(metrics)
    born = tuple(S(f"loop_dual_weights::born{i}") for i in range(len(bare.born_colors)))
    loop_orders = tuple(
        graph.order
        for graph, terms in zip(
            bare.loop_graphs, bare.loop_reduction_powers, strict=True
        )
        for _ in terms
    )
    values, columns = [], []
    for direction, metric in enumerate(metrics):
        for sector, selection in enumerate(bare.order_layout.virtual):
            pairs = {(pair.born, pair.loop) for pair in selection.pairs}
            for term, order in enumerate(loop_orders):
                admitted = tuple(
                    index
                    for index, born_order in enumerate(bare.born_term_orders)
                    if (born_order, order) in pairs
                )
                if not admitted:
                    continue
                # Keep admitted columns even when this operator gives zero.
                columns.append((direction, sector, term))
                values.append(
                    sum(
                        (
                            born[index].conj() * metric[index][term]
                            for index in admitted
                        ),
                        E("0"),
                    )
                )
    return PreparedDualBornWeights(_saved(values, born), tuple(columns), len(metrics))


def family_weight_projection_state(
    bare, weights, indices, powers, sectors, *, helicities=None
):
    """Assemble routed coefficients using explicit selections of shared weights."""
    from symbolica import E, S

    helicities = bare.helicities if helicities is None else helicities
    selected = weights.family_columns(bare, indices, sectors)
    projected = {
        (direction, sector, power): E("0")
        for direction in range(weights.directions)
        for sector in sectors
        for power in powers
    }
    parameters = []
    for helicity in range(len(helicities)):
        symbols = {
            weights.columns[column]: S(f"loop_dual_weights::weight{helicity}_{column}")
            for column in selected
        }
        parameters.extend(symbols.values())
        for index in indices:
            for local, labels in enumerate(bare.loop_reduction_powers[index]):
                term = bare.loop_term_offsets[index] + local
                for column, power in enumerate(labels):
                    coefficient = S(
                        f"loop_dual_weights::coefficient{helicity}_{index}_{local}_{column}"
                    )
                    parameters.append(coefficient)
                    if power is None:
                        continue
                    for direction in range(weights.directions):
                        for sector in sectors:
                            weight = symbols.get((direction, sector, term))
                            if weight is not None:
                                projected[direction, sector, power] += (
                                    weight * coefficient
                                )
    outputs = list(projected.values())
    return _saved(outputs, parameters), len(parameters), len(outputs)
