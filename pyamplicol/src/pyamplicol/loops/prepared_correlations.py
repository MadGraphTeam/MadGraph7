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
from .symbolic_sums import symbolic_sum


@dataclass(frozen=True, slots=True)
class PreparedDirectedFamily:
    state: bytes
    inputs: int
    outputs: int
    weight_columns: tuple[int, ...]

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
    weight_plan: Any
    families: tuple[PreparedDirectedFamily, ...]
    postprocess_state: bytes
    channels: tuple
    color_bank: Any = None

    def require_owner(self, bare):
        if bare.prepared is not self.plan:
            raise ValueError(
                "directed preparation belongs to a different retained bare plan"
            )
        if self.color_bank is not None:
            self.color_bank.require_bare(bare)
            if self.color_bank.histories != self.histories:
                raise ValueError("directed colour bank has different history requests")

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
        if self.bare.color_bank is not None:
            self.bare.color_bank.require_owner(model)

    def bind(self, model):
        from .physical_model_correlations import NativeDirectedModelProgram

        return NativeDirectedModelProgram(
            model,
            self.bare.histories,
            spin_layout=self.bare.spin_layout,
            _prepared=self,
        )


def prepare_directed_bare(bare, histories, *, spin_layout=None, color_bank=None):
    from symbolica import E, S

    from pyamplicol.color.connections import ColorConnection

    from .color_bilinears import SUNColorSpace
    from .color_connections import directed_color_bilinears
    from .physical_correlations import (
        PROJECTED_IDENTITY_CONVENTION,
        _physical_color_ports,
        directed_color_convention,
    )

    if spin_layout is not None:
        spin_layout.require_owner(bare)
    helicities = bare.helicities if spin_layout is None else spin_layout.helicities
    helicity_options = {} if spin_layout is None else {"helicities": helicities}
    histories = tuple(tuple(pair) for pair in histories)
    if not histories or any(
        len(pair) != 2 or any(not isinstance(x, ColorConnection) for x in pair)
        for pair in histories
    ):
        raise ValueError("expected nonempty ordered colour-history pairs")
    if color_bank is not None:
        color_bank.require_bare(bare)
        if color_bank.histories != histories:
            raise ValueError("directed colour bank has different history requests")
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
    convention = directed_color_convention(bare, histories, color_bank=color_bank)
    if color_bank is not None:
        born_metrics, loop_metrics, _, _ = color_bank.bound()
    elif convention == PROJECTED_IDENTITY_CONVENTION:
        # The model has already projected bare+UV+R2+fields with one structural
        # degree per sector. Source replacement cannot choose a new degree.
        # For identity history, the adjoint swapped metric equals the forward one.
        born_metrics = [bare.born_metric] * len(histories)
        loop_metrics = [bare.loop_metric] * (2 * len(histories))
    else:
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
                            tuple(
                                kernels.loop_born[j][i].conj() for j in range(len(loop))
                            )
                            for i in range(len(born))
                        )
                    ),
                )
            )
    born_state = bare._born_projection_state(born_metrics, **helicity_options)
    from .dual_weights import family_weight_projection_state, prepare_dual_born_weights

    weight_plan = prepare_dual_born_weights(bare, loop_metrics)
    families = tuple(
        PreparedDirectedFamily(
            *family_weight_projection_state(
                bare,
                weight_plan,
                family.graph_indices,
                family.powers,
                family.sector_indices,
                **helicity_options,
            ),
            weight_plan.family_columns(
                bare, family.graph_indices, family.sector_indices
            ),
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
                        * symbolic_sum(
                            values[pole]
                            for (d, s, _), values in poles.items()
                            if d == direction and s == sector
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
        weight_plan,
        families,
        postprocess_state,
        channels,
        color_bank,
    )


def prepare_directed_model(model, prepared):
    from .physical_model_correlations import require_directed_model_scope

    require_directed_model_scope(
        model,
        () if prepared is None else prepared.histories,
        color_bank=None if prepared is None else prepared.color_bank,
    )
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

    if prepared.color_bank is not None:
        prepared.color_bank.require_owner(model)
        _, _, current_weights, field_weights = prepared.color_bank.bound()
    elif model.color_projection is not None:
        # These weights already include each current's scalar colour grade,
        # active tensor mask, the common sector degree and initial averaging.
        current_weights = tuple(
            (current.color_weights,) * (2 * len(prepared.histories))
            for current in model.currents
        )
        field_weights = tuple(
            (current.color_weights,) * (2 * len(prepared.histories))
            for current in model.field_currents
        )
    else:
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
    # Terms are collected per output and added once (``+=`` is quadratic).
    cross_terms = [[[] for _ in range(width)] for _ in range(6)]
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
        [symbolic_sum(terms) for row in cross_terms for terms in row], parameters
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


def prepare_directed_complete(model, histories, *, spin_layout=None, color_bank=None):
    """Choose complete family degrees before compiling bare or effective terms.

    The returned bank may be passed to further source preparations for this same
    model and history catalogue. Source values and spectator helicities cannot
    select a different leading degree.
    """
    from .physical_model_correlations import require_directed_model_components

    require_directed_model_components(model)
    histories = tuple(tuple(pair) for pair in histories)
    if (
        color_bank is None
        and model.color_projection is not None
        and any(item.emissions for pair in histories for item in pair)
    ):
        from .directed_color_projection import prepare_directed_color_bank

        color_bank = prepare_directed_color_bank(model, histories)
    if color_bank is not None:
        color_bank.require_owner(model)
    prepared = prepare_directed_bare(
        model.bare, histories, spin_layout=spin_layout, color_bank=color_bank
    )
    return prepare_directed_model(model, prepared)


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
    for history in range(len(histories)):
        for sector, pairs in enumerate(model.sector_pairs):
            for direction in (0, 1):
                value = symbolic_sum(
                    left.conj() * weight * amplitude
                    for left, left_order, row in zip(
                        born,
                        model.bare.born_term_orders,
                        weights[2 * history + direction],
                        strict=True,
                    )
                    if (left_order, order) in pairs
                    for weight, amplitude in zip(row, amplitudes, strict=True)
                )
                index = 3 * (history * sectors + sector) - power
                outputs[2 * component + direction][index].append(
                    value if direction == 0 else value.conj()
                )
