# SPDX-License-Identifier: 0BSD
"""Exact complete-family colour projection over existing dependency tensors.

The ordinary metric cannot certify a directed operator's support. These cold
banks include every compatible ordered history, both directions and all model
components before selecting one structural degree per squared-order sector.
Tensor algebra and Laurent decomposition remain dependency operations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from pyamplicol.color.connection_catalogue import (
    build_color_correlator_catalogue,
    canonical_color_connections,
)
from pyamplicol.color.connections import ColorConnection, ColorLeg
from pyamplicol.color.correlator_matrices import ColorCorrelator
from pyamplicol.config.models import ColorAccuracy

from .color_bilinears import SUNColorSpace
from .color_connections import apply_native_color_connection
from .color_projection import prepare_color_projection
from .correlation_labels import native_color_history, public_color_history


@dataclass(frozen=True, slots=True)
class DirectedColorColumns:
    """One coherent right-hand column block, before any numerical substitution."""

    tensors: tuple
    orders: tuple
    factors: tuple

    def __post_init__(self):
        if len(self.tensors) != len(self.orders) or len(self.orders) != len(
            self.factors
        ):
            raise ValueError("directed tensor columns, orders and scalar grades differ")


@dataclass(frozen=True, slots=True)
class DirectedColorInputs:
    """Exact arguments of one actual colour assembly, before projection.

    Domains distinguish bare, UV, R2 and field columns. They do not describe
    numerical caches or identify an owner; the model factory owns that check.
    """

    born_tensors: tuple
    born_orders: tuple
    born_pairs: tuple
    virtual_pairs: tuple
    columns: tuple[DirectedColorColumns, ...]
    domains: tuple[str, ...]

    def __post_init__(self):
        if len(self.born_tensors) != len(self.born_orders):
            raise ValueError("Born tensor columns and orders differ")
        if len(self.columns) != len(self.domains) or any(
            not isinstance(domain, str) or not domain for domain in self.domains
        ):
            raise ValueError("directed column blocks and actual domains differ")


@dataclass(frozen=True, slots=True)
class DirectedColorFamily:
    histories: tuple[ColorConnection, ...]
    born_degrees: tuple[int | None, ...]
    virtual_degrees: tuple[int | None, ...]

    @property
    def order(self):
        return self.histories[0].order

    @property
    def output_legs(self):
        return self.histories[0].output_legs


@dataclass(frozen=True, slots=True)
class DirectedColorProjection:
    nc: Any
    accuracy: str
    families: tuple[DirectedColorFamily, ...]
    history_families: tuple[int, ...]
    born: tuple
    columns: tuple

    def bound(self, initial_average):
        def matrix(values):
            return tuple(
                tuple(
                    (value.replace(self.nc, 3) / initial_average).together()
                    for value in row
                )
                for row in values
            )

        return (
            tuple(map(matrix, self.born)),
            tuple(tuple(map(matrix, group)) for group in self.columns),
        )


def _complete_families(histories):
    """Resolve native requests through the existing complete public catalogue."""
    histories = tuple(tuple(pair) for pair in histories)
    if not histories or any(
        len(pair) != 2 or any(not isinstance(x, ColorConnection) for x in pair)
        for pair in histories
    ):
        raise ValueError("expected nonempty ordered colour-history pairs")
    initial = histories[0][0].initial_legs
    if any(item.initial_legs != initial for pair in histories for item in pair):
        raise ValueError("directed families have different initial typed legs")
    public_legs = tuple(ColorLeg(leg.label + 1, leg.representation) for leg in initial)
    catalogues, families, selections = {}, {}, []
    for alpha, beta in histories:
        request = public_color_history(
            ColorCorrelator("family", alpha.emissions, beta.emissions)
        )
        left, right = canonical_color_connections(
            (
                ColorConnection(public_legs, request.bra),
                ColorConnection(public_legs, request.ket),
            )
        )
        key = (left.order, left.output_legs)
        if key not in families:
            if left.order == 0:
                connected = (ColorConnection(initial),)
            else:
                if left.order not in catalogues:
                    by_output = defaultdict(list)
                    for item in build_color_correlator_catalogue(
                        public_legs,
                        through_order=left.order,
                        include_physical_splittings=True,
                    ):
                        if item.order == left.order and item.bra == item.ket:
                            value = ColorConnection(public_legs, item.bra)
                            by_output[value.output_legs].append(item)
                    catalogues[left.order] = by_output
                connected = tuple(
                    ColorConnection(initial, native_color_history(item).bra)
                    for item in catalogues[left.order][left.output_legs]
                )
            families[key] = connected
        native = native_color_history(
            ColorCorrelator("family", left.emissions, right.emissions)
        )
        requested = (
            ColorConnection(initial, native.bra),
            ColorConnection(initial, native.ket),
        )
        positions = {item: i for i, item in enumerate(families[key])}
        if any(item not in positions for item in requested):
            raise ValueError(
                "requested history is absent from its complete colour family"
            )
        selections.append((key, positions[requested[0]], positions[requested[1]]))
    return tuple(families.values()), tuple(
        (tuple(families).index(key), left, right) for key, left, right in selections
    )


def project_directed_color_columns(
    nc,
    accuracy,
    histories,
    born_tensors,
    born_orders,
    born_pairs,
    virtual_pairs,
    columns,
):
    """Project complete operator families on exact supplied tensor/order columns.

    This mathematical helper does not certify a model continuation. The model
    factory below supplies its actual retained ordinary/CT/field owners.
    ``columns`` contains bare loop, then graded CT and field blocks. Each scalar
    factor already includes its exact active-term mask. Both integrated terms
    use this same family projection, with the second selecting swapped histories.
    """
    from symbolica import E

    accuracy = ColorAccuracy(accuracy).value
    born_tensors, born_orders, columns = (
        tuple(born_tensors),
        tuple(born_orders),
        tuple(columns),
    )
    if len(born_tensors) != len(born_orders):
        raise ValueError("Born tensor columns and orders differ")
    family_histories, selections = _complete_families(histories)
    space = SUNColorSpace(nc)
    images, overlaps = {}, {}

    def image(value, history):
        key = (id(value), history)
        if key not in images:
            images[key] = apply_native_color_connection(value, history)
        return images[key]

    def matrix(alpha, beta, group):
        result = []
        for left in born_tensors:
            row = []
            for right, factor in zip(group.tensors, group.factors, strict=True):
                if factor == 0:
                    row.append(E("0"))
                    continue
                a, b = image(left, alpha), image(right, beta)
                key = (id(a), id(b))
                if key not in overlaps:
                    overlaps[key] = space.overlap(a, b)
                row.append((factor * overlaps[key]).together())
            result.append(tuple(row))
        return tuple(result)

    def project(blocks, orders, pairs):
        owners = {pair: index for index, sector in enumerate(pairs) for pair in sector}
        rows, row_owners = [], []
        for values, right_orders in zip(blocks, orders, strict=True):
            for left_order, values_row in zip(born_orders, values, strict=True):
                for value, right_order in zip(values_row, right_orders, strict=True):
                    rows.append([value])
                    row_owners.append(owners.get((left_order, right_order)))
        output = [E("0")] * len(rows)
        degrees = []
        for sector in range(len(pairs)):
            indices = [i for i, owner in enumerate(row_owners) if owner == sector]
            plan = prepare_color_projection(nc, [rows[i] for i in indices])
            degrees.append(plan.leading_degree)
            for i, value in zip(indices, plan.project(accuracy), strict=True):
                output[i] = value
        values = iter(output)
        projected = tuple(
            tuple(tuple(next(values) for _ in row) for row in block) for block in blocks
        )
        return projected, tuple(degrees)

    born_group = DirectedColorColumns(
        born_tensors, born_orders, (E("1"),) * len(born_tensors)
    )
    family_born, family_virtual, families = [], [], []
    for connected in family_histories:
        pairs = tuple((alpha, beta) for alpha in connected for beta in connected)
        born_blocks, born_degrees = project(
            tuple(matrix(a, b, born_group) for a, b in pairs),
            (born_orders,) * len(pairs),
            born_pairs,
        )
        virtual_blocks, virtual_degrees = project(
            tuple(matrix(a, b, group) for a, b in pairs for group in columns),
            tuple(group.orders for _ in pairs for group in columns),
            virtual_pairs,
        )
        family_born.append(born_blocks)
        family_virtual.append(virtual_blocks)
        families.append(DirectedColorFamily(connected, born_degrees, virtual_degrees))
    selected_born, selected_columns = [], [[] for _ in columns]
    for family, alpha, beta in selections:
        count = len(families[family].histories)
        selected_born.append(family_born[family][alpha * count + beta])
        # K_LB(alpha,beta)^dagger = K_BL(beta,alpha). This selects a separate
        # exact matrix. Numerical conjugation still occurs only after reduction.
        for direction in (alpha * count + beta, beta * count + alpha):
            for group, selected in enumerate(selected_columns):
                selected.append(
                    family_virtual[family][direction * len(columns) + group]
                )
    return DirectedColorProjection(
        nc,
        accuracy,
        tuple(families),
        tuple(item[0] for item in selections),
        tuple(selected_born),
        tuple(tuple(group) for group in selected_columns),
    )


def reproject_directed_color_columns(
    reference: DirectedColorProjection,
    accuracy,
    reference_inputs: DirectedColorInputs,
    target_inputs: DirectedColorInputs,
):
    """Reproject exact full rows using their complete-family degree authority.

    This mathematical operation makes no physical owner claim. Exact support
    must agree in each typed tensor/order/grade/domain class. Repeated target
    columns keep explicit output positions but do not alter the structural
    maximum. The reference must retain full expressions, never bound Nc=3 rows.
    """
    from symbolica import E

    from .color_orbits import _canonical_with_ports
    from .exact import require_exact_numbers

    accuracy = ColorAccuracy(accuracy).value
    if reference.accuracy != "full":
        raise ValueError("directed reprojection requires a full colour reference")
    if (
        reference_inputs.born_pairs != target_inputs.born_pairs
        or reference_inputs.virtual_pairs != target_inputs.virtual_pairs
    ):
        raise ValueError("directed reprojection changes squared-order sectors")
    histories = len(reference.history_families)
    if len(reference.born) != histories or len(reference.columns) != len(
        reference_inputs.columns
    ):
        raise ValueError("full directed rectangles differ from their column inputs")
    for family in reference.families:
        if len(family.born_degrees) != len(reference_inputs.born_pairs) or len(
            family.virtual_degrees
        ) != len(reference_inputs.virtual_pairs):
            raise ValueError("directed family degrees differ from order sectors")
    if any(
        type(index) is not int or not 0 <= index < len(reference.families)
        for index in reference.history_families
    ):
        raise ValueError("directed history has no complete family degree owner")

    def require_matrix(matrix, width):
        if len(matrix) != len(reference_inputs.born_tensors) or any(
            len(row) != width for row in matrix
        ):
            raise ValueError("full directed rectangle has an invalid column shape")

    for matrix in reference.born:
        require_matrix(matrix, len(reference_inputs.born_tensors))
    for group, columns in zip(reference.columns, reference_inputs.columns, strict=True):
        if len(group) != 2 * histories:
            raise ValueError("full directed rectangle has an invalid direction axis")
        for matrix in group:
            require_matrix(matrix, len(columns.tensors))

    canonical = {}

    def tensor_key(tensor):
        key = id(tensor)
        if key not in canonical:
            expression, reason = _canonical_with_ports(tensor)
            if reason is not None:
                raise NotImplementedError(
                    f"directed colour reuse has an unproved tensor: {reason}"
                )
            canonical[key] = (expression, frozenset(tensor.axes))
        return canonical[key]

    def born_keys(inputs):
        return tuple(
            (tensor_key(tensor), order)
            for tensor, order in zip(
                inputs.born_tensors, inputs.born_orders, strict=True
            )
        )

    def column_keys(inputs):
        groups = []
        for domain, group in zip(inputs.domains, inputs.columns, strict=True):
            keys = []
            for tensor, order, factor in zip(
                group.tensors, group.orders, group.factors, strict=True
            ):
                require_exact_numbers(factor)
                # Keep tensor and grade/mask separate, even for an inactive
                # column. No compensating colour-factor rescaling is inferred.
                keys.append((domain, tensor_key(tensor), order, factor.together()))
            groups.append(tuple(keys))
        return tuple(groups)

    left_born, right_born = born_keys(reference_inputs), born_keys(target_inputs)
    left_columns, right_columns = (
        column_keys(reference_inputs),
        column_keys(target_inputs),
    )
    if set(left_born) != set(right_born) or {
        key for group in left_columns for key in group
    } != {key for group in right_columns for key in group}:
        raise ValueError("directed reuse changes complete tensor/order/grade support")
    born_indices = {}
    for index, key in enumerate(left_born):
        born_indices.setdefault(key, index)
    born_map = tuple(born_indices[key] for key in right_born)
    column_indices = {}
    for group, keys in enumerate(left_columns):
        for index, key in enumerate(keys):
            column_indices.setdefault(key, (group, index))
    column_maps = tuple(
        tuple(column_indices[key] for key in keys) for keys in right_columns
    )

    def project(matrix, right_orders, pairs, degrees):
        owners = {}
        for sector, entries in enumerate(pairs):
            for pair in entries:
                if pair in owners:
                    raise ValueError("directed order pair belongs to multiple sectors")
                owners[pair] = sector
        rows, row_owners = [], []
        for order, values in zip(target_inputs.born_orders, matrix, strict=True):
            for value, right_order in zip(values, right_orders, strict=True):
                rows.append([value])
                row_owners.append(owners.get((order, right_order)))
        output = [E("0")] * len(rows)
        for sector, degree in enumerate(degrees):
            indices = [i for i, owner in enumerate(row_owners) if owner == sector]
            plan = prepare_color_projection(reference.nc, [rows[i] for i in indices])
            if plan.leading_degree is not None and (
                degree is None or plan.leading_degree > degree
            ):
                raise ValueError("selected row exceeds its complete-family degree")
            for index, value in zip(
                indices,
                replace(plan, leading_degree=degree).project(accuracy),
                strict=True,
            ):
                output[index] = value
        if any(rows[i][0] != 0 for i, owner in enumerate(row_owners) if owner is None):
            raise ValueError("full directed row is outside its retained order sectors")
        values = iter(output)
        return tuple(tuple(next(values) for _ in row) for row in matrix)

    born = []
    columns = [[] for _ in target_inputs.columns]
    for history, family_index in enumerate(reference.history_families):
        family = reference.families[family_index]
        born.append(
            project(
                tuple(
                    tuple(reference.born[history][i][j] for j in born_map)
                    for i in born_map
                ),
                target_inputs.born_orders,
                target_inputs.born_pairs,
                family.born_degrees,
            )
        )
        for direction in (2 * history, 2 * history + 1):
            for group, mapping in enumerate(column_maps):
                columns[group].append(
                    project(
                        tuple(
                            tuple(
                                reference.columns[source_group][direction][i][j]
                                for source_group, j in mapping
                            )
                            for i in born_map
                        ),
                        target_inputs.columns[group].orders,
                        target_inputs.virtual_pairs,
                        family.virtual_degrees,
                    )
                )
    return DirectedColorProjection(
        reference.nc,
        accuracy,
        reference.families,
        reference.history_families,
        tuple(born),
        tuple(tuple(group) for group in columns),
    )


@dataclass(frozen=True, slots=True)
class PreparedDirectedColorBank:
    plan: Any
    formal: Any
    histories: tuple
    projection: DirectedColorProjection
    current_count: int

    def require_bare(self, bare):
        if (
            bare.prepared is not self.plan.bare
            or self.projection.accuracy != self.formal.accuracy
        ):
            raise ValueError("directed colour bank has a different retained bare owner")
        # The factory proves the exact tensor/metric binding once. Rebound
        # arithmetic views retain this immutable preparation, so numerical calls
        # must not repeat dependency algebra or rebuild the family catalogue.
        self.formal.require_owner_layout(bare)

    def require_owner(self, model):
        if model.prepared is not self.plan:
            raise ValueError(
                "directed colour bank has a different complete model owner"
            )
        self.require_bare(model.bare)

    def bound(self):
        born, columns = self.projection.bound(self.formal.initial_average)
        return (
            born,
            columns[0],
            columns[1 : 1 + self.current_count],
            columns[1 + self.current_count :],
        )


def _model_color_inputs(model):
    """Extract the existing actual assembly once for preparation and reuse."""
    from symbolica import E

    from .physical_correlations import (
        _physical_color_ports,
        _rename_physical_color_tensors,
    )
    from .scalar_incidence import scalar_grade_term_mask

    formal = getattr(model.bare, "formal_color_metrics", None)
    if (
        formal is None
        or model.color_projection is None
        or model.color_projection.formal_metrics is not formal
    ):
        raise ValueError(
            "directed formal colour requires the actual complete projection owner"
        )
    formal.require_owner(model.bare)
    if model.r2_strategy != "model":
        raise NotImplementedError("directed formal completion requires model R2")
    legs, born, loop = _physical_color_ports(model.bare)
    columns = [DirectedColorColumns(loop, formal.loop_orders, (E("1"),) * len(loop))]
    domains = ["bare"]
    for current in model.currents:
        if type(current.color_grade) is not int:
            raise ValueError("formal counterterm has no retained scalar colour grade")
        active = scalar_grade_term_mask(
            current.dag,
            current.executor.prepared.output_nodes,
            current.insertion.rule.active_coupling_symbols,
            current.coefficient_expressions,
        )
        colors = _rename_physical_color_tensors(current.color_tensors)
        columns.append(
            DirectedColorColumns(
                colors,
                (current.insertion.graph.order,) * len(colors),
                tuple(
                    formal.nc**current.color_grade if selected else E("0")
                    for selected in active
                ),
            )
        )
        domains.append(current.insertion.rule.kind)
    for current in model.field_currents:
        if type(current.color_grade) is not int:
            raise ValueError(
                "formal external field has no retained scalar colour grade"
            )
        colors = tuple(born[i] for i in current.born_indices)
        factor = (
            formal.nc**current.color_grade
            if current.coefficient_expression != 0
            else E("0")
        )
        columns.append(
            DirectedColorColumns(
                colors,
                (current.insertion.order,) * len(colors),
                (factor,) * len(colors),
            )
        )
        domains.append("field")
    return (
        formal,
        legs,
        DirectedColorInputs(
            born,
            formal.born_orders,
            formal.born_pairs,
            formal.virtual_pairs,
            tuple(columns),
            tuple(domains),
        ),
    )


def prepare_directed_color_bank(model, histories):
    """Build one source-independent bank from the actual complete formal owner."""
    formal, legs, inputs = _model_color_inputs(model)
    histories = tuple(tuple(pair) for pair in histories)
    if any(item.initial_legs != legs for pair in histories for item in pair):
        raise ValueError(
            "directed history legs differ from actual physical colour ports"
        )
    projection = project_directed_color_columns(
        formal.nc,
        formal.accuracy,
        histories,
        inputs.born_tensors,
        inputs.born_orders,
        inputs.born_pairs,
        inputs.virtual_pairs,
        inputs.columns,
    )
    _require_identity_consistency(model, projection, histories)
    return PreparedDirectedColorBank(
        model.prepared, formal, histories, projection, len(model.currents)
    )


def reproject_directed_color_bank(reference_model, reference_bank, target_model):
    """Make a new LC/NLC/full bank for the same actual process and source.

    The target keeps its normally prepared currents, formal metrics and source
    states. Cross-species owner substitution is not part of this factory.
    """
    reference_bank.require_owner(reference_model)
    source = reference_model.source_color
    if (
        source is None
        or target_model.source_color is not source
        or any(
            dict(model.prepared.metadata).get("source_color") is not source
            for model in (reference_model, target_model)
        )
        or reference_model.bare.process is not target_model.bare.process
        or reference_model.bare.model_state is None
        or target_model.bare.model_state is None
        or reference_model.bare.model_state.plan
        is not target_model.bare.model_state.plan
    ):
        raise ValueError(
            "directed reprojection requires the same actual process/source"
        )
    left, left_legs, reference_inputs = _model_color_inputs(reference_model)
    right, right_legs, target_inputs = _model_color_inputs(target_model)
    if (
        reference_bank.formal is not left
        or reference_bank.projection.nc != left.nc
        or left.nc != right.nc
        or left.initial_average != right.initial_average
        or left.born_degrees != right.born_degrees
        or left.virtual_degrees != right.virtual_degrees
        or left_legs != right_legs
    ):
        raise ValueError(
            "directed reprojection changes formal normalization or sectors"
        )
    projection = reproject_directed_color_columns(
        reference_bank.projection, right.accuracy, reference_inputs, target_inputs
    )
    _require_identity_consistency(target_model, projection, reference_bank.histories)
    return PreparedDirectedColorBank(
        target_model.prepared,
        right,
        reference_bank.histories,
        projection,
        len(target_model.currents),
    )


def _require_identity_consistency(model, projection, histories):
    """Order zero must retain the already certified ordinary metric authority."""
    formal = model.bare.formal_color_metrics

    def same(left, right):
        return len(left) == len(right) and all(
            len(a) == len(b)
            and all((x - y).together() == 0 for x, y in zip(a, b, strict=True))
            for a, b in zip(left, right, strict=True)
        )

    for index, pair in enumerate(histories):
        if any(item.emissions for item in pair):
            continue
        family = projection.families[projection.history_families[index]]
        if (
            family.born_degrees != formal.born_degrees
            or family.virtual_degrees != formal.virtual_degrees
            or not same(projection.born[index], formal.born)
            or any(
                not same(projection.columns[0][direction], formal.loop)
                for direction in (2 * index, 2 * index + 1)
            )
        ):
            raise ValueError(
                "identity history differs from ordinary formal colour owner"
            )
        # The old record retains exact pre-Nc bare metrics but only bound CT
        # weights. New CT rows come from actual scalar grades/masks/tensors;
        # equality at Nc=3 is a consistency check, not their continuation proof.
        for current, group in zip(
            (*model.currents, *model.field_currents),
            projection.columns[1:],
            strict=True,
        ):
            for direction in (2 * index, 2 * index + 1):
                bound = tuple(
                    tuple(
                        value.replace(formal.nc, 3) / formal.initial_average
                        for value in row
                    )
                    for row in group[direction]
                )
                if not same(bound, current.color_weights):
                    raise ValueError(
                        "identity history differs from ordinary effective "
                        "colour weights"
                    )
