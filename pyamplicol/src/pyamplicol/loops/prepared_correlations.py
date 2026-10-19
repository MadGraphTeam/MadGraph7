# SPDX-License-Identifier: 0BSD
"""Immutable directed-observable instructions over retained physical currents.

Only dependency evaluator bytes and existing owner/layout references are kept.
The portable writer can reference these same bytes. These records introduce no
current engine or public correlated runtime. Reduction precedes conjugation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .family_projectors import bind_family_projector
from .physical import _saved


@dataclass(frozen=True, slots=True)
class PreparedDirectedFamily:
    state: bytes
    inputs: int
    outputs: int

    def bind(self, bare):
        return bind_family_projector(
            self.state,
            self.inputs,
            self.outputs,
            backend=bare.backend,
            options=bare.options,
            engine=bare.projector_engine,
        )


@dataclass(frozen=True, slots=True)
class PreparedDirectedBare:
    plan: Any
    histories: tuple
    spin_layout: Any
    helicities: tuple
    color_legs: tuple
    born_state: bytes
    families: tuple[PreparedDirectedFamily, ...]
    postprocess_state: bytes
    channels: tuple

    def require_owner(self, bare):
        if bare.prepared is not self.plan:
            raise ValueError(
                "directed preparation belongs to a different retained bare plan"
            )

    def bind(self, bare):
        from .physical_correlations import NativeDirectedBareProgram

        return NativeDirectedBareProgram(
            bare, self.histories, spin_layout=self.spin_layout, _prepared=self
        )


@dataclass(frozen=True, slots=True)
class PreparedDirectedModel:
    plan: Any
    bare: PreparedDirectedBare
    counterterm_state: bytes
    assembly_state: bytes
    sum_order_state: bytes

    def require_owner(self, model):
        if model.prepared is not self.plan:
            raise ValueError(
                "directed preparation belongs to a different retained model plan"
            )
        self.bare.require_owner(model.bare)

    def bind(self, model):
        from .physical_model_correlations import NativeDirectedModelProgram

        return NativeDirectedModelProgram(
            model,
            self.bare.histories,
            spin_layout=self.bare.spin_layout,
            _prepared=self,
        )


def prepare_directed_bare(bare, histories, *, spin_layout=None):
    from symbolica import E, S

    from pyamplicol.color.connections import ColorConnection

    from .color_bilinears import SUNColorSpace
    from .color_connections import directed_color_bilinears
    from .physical_correlations import _physical_color_ports

    helicities = bare.helicities if spin_layout is None else spin_layout.helicities
    helicity_options = {} if spin_layout is None else {"helicities": helicities}
    histories = tuple(tuple(pair) for pair in histories)
    if not histories or any(
        len(pair) != 2 or any(not isinstance(x, ColorConnection) for x in pair)
        for pair in histories
    ):
        raise ValueError("expected nonempty ordered colour-history pairs")
    color_legs, born, loop = _physical_color_ports(bare)
    if any(
        connection.initial_legs != color_legs
        for pair in histories
        for connection in pair
    ):
        raise ValueError("history legs differ from the native physical colour ports")
    nc = S("loop_directed_physical::Nc", is_real=True)
    space = SUNColorSpace(nc)

    def fixed(matrix):
        return tuple(
            tuple(
                (value.replace(nc, 3) / bare.initial_average).together()
                for value in row
            )
            for row in matrix
        )

    born_metrics, loop_metrics = [], []
    for bra, ket in histories:
        kernels = directed_color_bilinears(space, born, loop, bra, ket)
        born_metrics.append(
            fixed(directed_color_bilinears(space, born, born, bra, ket).born_loop)
        )
        loop_metrics.extend(
            (
                fixed(kernels.born_loop),
                fixed(
                    tuple(
                        tuple(kernels.loop_born[j][i].conj() for j in range(len(loop)))
                        for i in range(len(born))
                    )
                ),
            )
        )
    born_state = bare._born_projection_state(born_metrics, **helicity_options)
    families = tuple(
        PreparedDirectedFamily(
            *bare._family_projection_state(
                family.graph_indices,
                family.powers,
                family.sector_indices,
                metrics=loop_metrics,
                **helicity_options,
            )
        )
        for family in bare.families
    )
    directions = 2 * len(histories)
    channels = tuple(
        (direction, sector, family.index)
        for family in bare.families
        for direction in range(directions)
        for sector in family.sector_indices
    )
    poles = {
        channel: tuple(
            S(f"loop_directed_physical::pole_{index}_{pole}") for pole in range(3)
        )
        for index, channel in enumerate(channels)
    }
    first, second = [], []
    for history in range(len(histories)):
        for sector in range(len(bare.order_layout.virtual)):
            for pole in range(3):

                def amplitude(direction, sector=sector, pole=pole):
                    return (
                        E("1i")
                        * sum(
                            (
                                values[pole]
                                for (d, s, _), values in poles.items()
                                if d == direction and s == sector
                            ),
                            E("0"),
                        )
                        / (16 * E("pi") ** 2)
                    )

                first.append(amplitude(2 * history))
                second.append(amplitude(2 * history + 1).conj())
    postprocess_state = _saved(
        [a + b for a, b in zip(first, second, strict=True)] + first + second,
        [value for values in poles.values() for value in values],
    )

    return PreparedDirectedBare(
        bare.prepared,
        histories,
        spin_layout,
        tuple(helicities),
        color_legs,
        born_state,
        families,
        postprocess_state,
        channels,
    )


def prepare_directed_model(model, prepared):
    from symbolica import E, S

    from .color_bilinears import SUNColorSpace
    from .color_connections import directed_color_bilinears
    from .physical_correlations import (
        _physical_color_ports,
        _rename_physical_color_tensors,
    )

    prepared.require_owner(model.bare)
    bare = model.bare
    _, born_colors, _ = _physical_color_ports(bare)
    nc = S("loop_directed_model::Nc", is_real=True)
    space = SUNColorSpace(nc)
    color_cache = {}

    def matrices(colors, key):
        if key not in color_cache:
            values = []
            for bra, ket in prepared.histories:
                kernels = directed_color_bilinears(space, born_colors, colors, bra, ket)
                forward = kernels.born_loop
                swapped = tuple(
                    tuple(kernels.loop_born[j][i].conj() for j in range(len(colors)))
                    for i in range(len(born_colors))
                )
                values.extend(
                    tuple(
                        tuple(
                            (entry.replace(nc, 3) / bare.initial_average).together()
                            for entry in row
                        )
                        for row in matrix
                    )
                    for matrix in (forward, swapped)
                )
            color_cache[key] = tuple(values)
        return color_cache[key]

    current_weights = tuple(
        matrices(
            _rename_physical_color_tensors(current.color_tensors),
            ("ct", id(current.color_tensors)),
        )
        for current in model.currents
    )
    field_weights = tuple(
        matrices(
            tuple(born_colors[index] for index in current.born_indices),
            ("field", current.born_indices),
        )
        for current in model.field_currents
    )
    sectors = len(bare.order_layout.virtual)
    width = len(prepared.histories) * sectors * 3
    # Components are local UV, R2, external fields; each retains two terms.
    cross_terms = [[E("0")] * width for _ in range(6)]
    parameters = []
    for helicity in range(len(prepared.helicities)):
        born = tuple(
            S(f"loop_directed_model::born_{helicity}_{index}")
            for index in range(len(born_colors))
        )
        parameters.extend(born)
        for index, (current, weights) in enumerate(
            zip(model.currents, current_weights, strict=True)
        ):
            amplitudes = tuple(
                S(f"loop_directed_model::ct_{helicity}_{index}_{term}")
                for term in range(current.term_count)
            )
            parameters.extend(amplitudes)
            component = 1 if current.insertion.rule.kind == "R2" else 0
            _accumulate_counterterm(
                model,
                prepared.histories,
                cross_terms,
                component,
                born,
                amplitudes,
                weights,
                current.insertion.graph.order,
                current.epsilon_power,
                sectors,
            )
        for index, (current, weights) in enumerate(
            zip(model.field_currents, field_weights, strict=True)
        ):
            factor = S(f"loop_directed_model::field_{helicity}_{index}")
            parameters.append(factor)
            amplitudes = tuple(factor * born[right] for right in current.born_indices)
            _accumulate_counterterm(
                model,
                prepared.histories,
                cross_terms,
                2,
                born,
                amplitudes,
                weights,
                current.insertion.order,
                current.epsilon_power,
                sectors,
            )
    counterterm_state = _saved(
        [value for row in cross_terms for value in row], parameters
    )
    # Input rows are bare first/second then the six effective-rule blocks.
    inputs = tuple(S(f"loop_directed_model::sum_{i}") for i in range(8 * width))
    first = [
        sum((inputs[k * width + i] for k in (0, 2, 4, 6)), E("0")) for i in range(width)
    ]
    second = [
        sum((inputs[k * width + i] for k in (1, 3, 5, 7)), E("0")) for i in range(width)
    ]
    local, rational, field = (
        [inputs[k * width + i] + inputs[(k + 1) * width + i] for i in range(width)]
        for k in (2, 4, 6)
    )
    assembly_state = _saved(
        [
            *[a + b for a, b in zip(first, second, strict=True)],
            *[a + b for a, b in zip(local, field, strict=True)],
            *rational,
            *local,
            *field,
            *first,
            *second,
        ],
        inputs,
    )

    return PreparedDirectedModel(
        model.prepared,
        prepared,
        counterterm_state,
        assembly_state,
        _directed_order_sum_state(len(prepared.histories), bare.order_layout),
    )


def _directed_order_sum_state(histories, order_layout):
    """Sum orders within each history, retaining all complex component values.

    Inputs concatenate Born then full/bare/UV/R2/local/field/first/second blocks;
    Each block packs history/order/(pole). Outputs pack history/(Born + eight
    Laurent triples).
    """
    from symbolica import E, S

    born, virtual = len(order_layout.born), len(order_layout.virtual)
    width = histories * virtual * 3
    inputs = tuple(
        S(f"loop_directed_model::order_sum_{index}")
        for index in range(histories * born + 8 * width)
    )
    outputs = []
    for history in range(histories):
        outputs.append(sum(inputs[history * born : (history + 1) * born], E("0")))
        outputs.extend(
            sum(
                (
                    inputs[
                        histories * born
                        + component * width
                        + (history * virtual + sector) * 3
                        + pole
                    ]
                    for sector in range(virtual)
                ),
                E("0"),
            )
            for component in range(8)
            for pole in range(3)
        )
    return _saved(outputs, inputs)


def _accumulate_counterterm(
    model,
    histories,
    outputs,
    component,
    born,
    amplitudes,
    weights,
    order,
    power,
    sectors,
):
    from symbolica import E

    for history in range(len(histories)):
        for sector, pairs in enumerate(model.sector_pairs):
            for direction in (0, 1):
                value = sum(
                    (
                        left.conj() * weight * amplitude
                        for left, left_order, row in zip(
                            born,
                            model.bare.born_term_orders,
                            weights[2 * history + direction],
                            strict=True,
                        )
                        if (left_order, order) in pairs
                        for weight, amplitude in zip(row, amplitudes, strict=True)
                    ),
                    E("0"),
                )
                index = 3 * (history * sectors + sector) - power
                outputs[2 * component + direction][index] += (
                    value if direction == 0 else value.conj()
                )
