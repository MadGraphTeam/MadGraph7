# SPDX-License-Identifier: 0BSD
"""Exact dual-Born weights shared independently of denominator families.

Only Symbolica evaluator instructions and current-column identities are kept.
The map runs once per native point/source row; no event values are cached.
"""

from __future__ import annotations

from dataclasses import dataclass

from .physical import _saved


@dataclass(frozen=True, slots=True)
class PreparedDualBornWeights:
    state: bytes
    # Each output retains direction, squared-order sector and physical loop term.
    columns: tuple[tuple[int, int, int], ...]
    directions: int

    def bind(self, bare):
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
