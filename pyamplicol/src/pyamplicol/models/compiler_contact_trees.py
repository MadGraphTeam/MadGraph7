# SPDX-License-Identifier: 0BSD
"""Color-singlet four-point contact-tree compilation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_contacts import _execute_dense_tensor
from .compiler_kernels import (
    _as_expression,
    _canonicalize_oriented_kernel_component,
    _remap_kernel_symbols,
    _replace_expression_symbols,
    _spin_dimension,
    _spin_representations,
    _spin_slots,
)
from .compiler_records import _replace_evaluator_constants
from .contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
    _ContactTreeNode,
)


def _compile_color_singlet_contact_trees(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Lower arbitrary color-singlet contacts to balanced trivalent trees."""

    particle_by_name = {particle.name: particle for particle in particles}
    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_000_000, max(used_pdgs, default=0) + 1)
    auxiliary_particles: list[CompiledParticleRecord] = []
    kernels: list[CompiledOrientedKernel] = []
    final_template_cache: dict[
        tuple[object, ...],
        tuple[int, tuple[_sym.Expression, ...]],
    ] = {}

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    for term in terms:
        if term.valence < 4 or not _contact_term_is_color_singlet(term):
            continue
        if term.valence == 4 and "ufo_momentum_" not in term.lorentz_expression:
            continue
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        scalar_product_tree = all(particle.spin == 1 for particle in source_particles)
        oriented_result_particles: set[str] = set()
        for result_leg in range(term.valence):
            source_result = source_particles[result_leg]
            if source_result.name in oriented_result_particles:
                continue
            oriented_result_particles.add(source_result.name)
            input_legs = tuple(leg for leg in range(term.valence) if leg != result_leg)

            def build_node(
                legs: tuple[int, ...],
                *,
                source_particles: tuple[CompiledParticleRecord, ...] = source_particles,
                scalar_product_tree: bool = scalar_product_tree,
                term: CompiledVertexTerm = term,
                result_leg: int = result_leg,
            ) -> _ContactTreeNode:
                if len(legs) == 1:
                    leg = legs[0]
                    return _ContactTreeNode(
                        legs=legs,
                        particle=source_particles[leg],
                        physical_leg=leg,
                    )
                split = len(legs) // 2
                left_node = build_node(legs[:split])
                right_node = build_node(legs[split:])
                auxiliary_name = (
                    f"__pyamplicol_contact_tree_{term.id}_r{result_leg}_"
                    + "_".join(str(leg) for leg in legs)
                )
                auxiliary_dimension = (
                    1
                    if scalar_product_tree
                    else sum(
                        _spin_dimension(source_particles[leg].spin) + 4 for leg in legs
                    )
                )
                auxiliary = CompiledParticleRecord(
                    name=auxiliary_name,
                    antiname=auxiliary_name,
                    pdg_code=allocate_pdg(),
                    spin=-1,
                    color=1,
                    mass="ZERO",
                    width="ZERO",
                    charge=0.0,
                    ghost_number=0,
                    propagating=False,
                    goldstoneboson=False,
                    propagator=None,
                    component_dimension=auxiliary_dimension,
                    auxiliary_kind=(
                        f"ufo-contact-tree:{term.id}:result-{result_leg}:"
                        + ",".join(str(leg) for leg in legs)
                    ),
                )
                auxiliary_particles.append(auxiliary)
                node = _ContactTreeNode(
                    legs=legs,
                    particle=auxiliary,
                    left=left_node,
                    right=right_node,
                )
                _append_contact_tree_partial_kernels(
                    kernels,
                    term=term,
                    node=node,
                    source_particles=source_particles,
                    scalar_product_tree=scalar_product_tree,
                    start_kind=start_kind,
                )
                return node

            split = len(input_legs) // 2
            left_root = build_node(input_legs[:split])
            right_root = build_node(input_legs[split:])
            assignment_multiplicity = _contact_tree_assignment_multiplicity(
                input_legs,
                source_particles,
                left_root,
                right_root,
            )
            _append_contact_tree_final_kernels(
                kernels,
                term=term,
                source_particles=source_particles,
                left_node=left_root,
                right_node=right_root,
                result_leg=result_leg,
                result_name=particle_by_name[source_result.antiname].name,
                scalar_product_tree=scalar_product_tree,
                assignment_multiplicity=assignment_multiplicity,
                start_kind=start_kind,
                template_cache=final_template_cache,
                model_symbols=model_symbols,
            )
    return tuple(auxiliary_particles), tuple(kernels)


def _append_contact_tree_partial_kernels(
    kernels: list[CompiledOrientedKernel],
    *,
    term: CompiledVertexTerm,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    scalar_product_tree: bool,
    start_kind: int,
) -> None:
    if node.left is None or node.right is None:
        raise ValueError("contact tree partial node is missing a child")
    orientations = (
        ((node.left, node.right),)
        if node.left.particle.name == node.right.particle.name
        else ((node.left, node.right), (node.right, node.left))
    )
    for actual_left, actual_right in orientations:
        kind = start_kind + len(kernels)
        canonical_left_side = "left" if actual_left is node.left else "right"
        canonical_right_side = "right" if actual_right is node.right else "left"
        left_payload = _contact_tree_node_payload(
            kind,
            canonical_left_side,
            node.left,
            source_particles,
            scalar_product_tree=scalar_product_tree,
        )
        right_payload = _contact_tree_node_payload(
            kind,
            canonical_right_side,
            node.right,
            source_particles,
            scalar_product_tree=scalar_product_tree,
        )
        components = (
            (left_payload[0] * right_payload[0],)
            if scalar_product_tree
            else (*left_payload, *right_payload)
        )
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=term.id,
                vertex=f"{term.vertex}::contact-tree-partial",
                particles=(
                    actual_left.particle.name,
                    actual_right.particle.name,
                    node.particle.name,
                ),
                source_particle_legs=(
                    _contact_tree_source_leg(actual_left),
                    _contact_tree_source_leg(actual_right),
                    -1,
                ),
                component_expressions=tuple(
                    component.to_canonical_string() for component in components
                ),
                coupling_expression="1",
                coupling_orders=(),
                runtime_parameters=(),
                color_source="1",
                color_expression="1",
                lc_color_normalization_power=0,
                term_ids=(),
            )
        )


def _append_contact_tree_final_kernels(
    kernels: list[CompiledOrientedKernel],
    *,
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    left_node: _ContactTreeNode,
    right_node: _ContactTreeNode,
    result_leg: int,
    result_name: str,
    scalar_product_tree: bool,
    assignment_multiplicity: int,
    start_kind: int,
    template_cache: dict[
        tuple[object, ...],
        tuple[int, tuple[_sym.Expression, ...]],
    ],
    model_symbols: ModelSymbolRegistry,
) -> None:
    orientations = (
        ((left_node, right_node),)
        if left_node.particle.name == right_node.particle.name
        else ((left_node, right_node), (right_node, left_node))
    )
    canonical_kind = start_kind + len(kernels)
    if scalar_product_tree:
        left_payload = _contact_tree_node_payload(
            canonical_kind,
            "left",
            left_node,
            source_particles,
            scalar_product_tree=True,
        )
        right_payload = _contact_tree_node_payload(
            canonical_kind,
            "right",
            right_node,
            source_particles,
            scalar_product_tree=True,
        )
        canonical_components = (
            left_payload[0] * right_payload[0] * _sym.E(term.lorentz_expression),
        )
    else:
        template_key = (
            term.lorentz_expression,
            tuple(particle.spin for particle in source_particles),
            result_leg,
            left_node.legs,
            right_node.legs,
        )
        cached = template_cache.get(template_key)
        if cached is None:
            canonical_components = _contact_tree_final_component_expressions(
                term,
                source_particles,
                left_node=left_node,
                right_node=right_node,
                result_leg=result_leg,
                kind=canonical_kind,
                canonical_left_side="left",
                canonical_right_side="right",
            )
            template_cache[template_key] = canonical_kind, canonical_components
        else:
            template_kind, template_components = cached
            canonical_components = tuple(
                _remap_kernel_symbols(
                    component,
                    old_kind=template_kind,
                    new_kind=canonical_kind,
                )
                for component in template_components
            )
    prefactor = (
        symbols.derived_coupling(model_symbols.model_name, term.id)
        / assignment_multiplicity
    )
    canonical_weighted_components = tuple(
        _canonicalize_oriented_kernel_component(component * prefactor)
        for component in canonical_components
    )
    for actual_left, actual_right in orientations:
        kind = start_kind + len(kernels)
        components = tuple(
            _remap_kernel_symbols(
                component,
                old_kind=canonical_kind,
                new_kind=kind,
                swap_sides=actual_left is right_node,
            )
            for component in canonical_weighted_components
        )
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=term.id,
                vertex=f"{term.vertex}::contact-tree-final",
                particles=(
                    actual_left.particle.name,
                    actual_right.particle.name,
                    result_name,
                ),
                source_particle_legs=(
                    _contact_tree_source_leg(actual_left),
                    _contact_tree_source_leg(actual_right),
                    result_leg,
                ),
                component_expressions=tuple(
                    component.to_canonical_string() for component in components
                ),
                coupling_expression="1",
                coupling_orders=term.coupling_orders,
                runtime_parameters=(f"derived_coupling_{term.id}",),
                color_source=term.color_source,
                color_expression=term.color_expression,
                lc_color_normalization_power=term.lc_color_normalization_power,
                term_ids=(term.id,),
            )
        )


def _contact_tree_node_payload(
    kind: int,
    side: str,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    scalar_product_tree: bool,
) -> tuple[_sym.Expression, ...]:
    if node.is_leaf:
        if node.physical_leg is None:
            raise ValueError("contact tree leaf has no source leg")
        dimension = _spin_dimension(source_particles[node.physical_leg].spin)
        components = tuple(
            symbols.kernel_component(kind, side, index) for index in range(dimension)
        )
        if scalar_product_tree:
            return components
        momenta = tuple(
            symbols.kernel_momentum(kind, side, index) for index in range(4)
        )
        return (*components, *momenta)
    dimension = node.particle.component_dimension
    if dimension is None:
        raise ValueError("contact tree auxiliary has no component dimension")
    return tuple(
        symbols.kernel_component(kind, side, index) for index in range(dimension)
    )


def _contact_tree_final_component_expressions(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    left_node: _ContactTreeNode,
    right_node: _ContactTreeNode,
    result_leg: int,
    kind: int,
    canonical_left_side: str,
    canonical_right_side: str,
) -> tuple[_sym.Expression, ...]:
    payload_by_leg = {
        **_contact_tree_payload_by_leg(
            kind,
            canonical_left_side,
            left_node,
            source_particles,
        ),
        **_contact_tree_payload_by_leg(
            kind,
            canonical_right_side,
            right_node,
            source_particles,
        ),
    }
    momentum_by_leg = {
        leg: momentum for leg, (_components, momentum) in payload_by_leg.items()
    }
    momentum_by_leg[result_leg] = tuple(
        -sum(
            (momentum[component] for momentum in momentum_by_leg.values()),
            _sym.E("0"),
        )
        for component in range(4)
    )
    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    for leg, (components, _momentum) in sorted(payload_by_leg.items()):
        expression *= _contact_tree_physical_tensor_expression(
            library,
            kind=kind,
            leg=leg,
            spin=source_particles[leg].spin,
            components=components,
        )
    minkowski = _sym.Representation.mink(4)
    for leg, momentum in momentum_by_leg.items():
        library.register(
            _sym.LibraryTensor.dense(
                _sym.TensorName(symbols.ufo_momentum_tensor_name(leg + 1))(minkowski),
                momentum,
            )
        )
    result = _execute_dense_tensor(expression, library)
    expected = _spin_dimension(source_particles[result_leg].spin)
    if len(result) != expected:
        raise ValueError(
            f"contact tree final {term.vertex}/{term.id} produced {len(result)} "
            f"components, expected {expected}"
        )
    return tuple(
        _replace_evaluator_constants(_as_expression(result[index]))
        for index in range(len(result))
    )


def eager_color_singlet_vertex_term_components(
    term: CompiledVertexTerm,
    particles: Sequence[CompiledParticleRecord],
    *,
    result_leg: int,
    input_components: Mapping[int, Sequence[_sym.Expression]],
    input_momenta: Mapping[int, Sequence[_sym.Expression]],
    coupling: _sym.Expression | None = None,
) -> tuple[_sym.Expression, ...]:
    """Contract one original color-singlet n-ary term without its lowered tree."""

    _sym._ensure_symbolica()

    if not _contact_term_is_color_singlet(term):
        raise ValueError("the eager n-ary oracle currently requires a color singlet")
    if not 0 <= result_leg < term.valence:
        raise ValueError(f"result leg {result_leg} is outside valence {term.valence}")
    particle_by_name = {particle.name: particle for particle in particles}
    source_particles = tuple(particle_by_name[name] for name in term.particles)
    input_legs = set(range(term.valence)) - {result_leg}
    if set(input_components) != input_legs or set(input_momenta) != input_legs:
        raise ValueError(
            "eager n-ary inputs must provide every non-result leg exactly once"
        )

    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    momentum_by_leg: dict[int, tuple[_sym.Expression, ...]] = {}
    for leg in sorted(input_legs):
        components = tuple(input_components[leg])
        expected_dimension = _spin_dimension(source_particles[leg].spin)
        if len(components) != expected_dimension:
            raise ValueError(
                f"input leg {leg} has {len(components)} components, "
                f"expected {expected_dimension}"
            )
        momentum = tuple(input_momenta[leg])
        if len(momentum) != 4:
            raise ValueError(f"input leg {leg} momentum must have four components")
        momentum_by_leg[leg] = momentum
        expression *= _contact_tree_physical_tensor_expression(
            library,
            kind=term.id,
            leg=leg,
            spin=source_particles[leg].spin,
            components=components,
        )

    momentum_by_leg[result_leg] = tuple(
        -sum(
            (momentum[component] for momentum in momentum_by_leg.values()),
            _sym.E("0"),
        )
        for component in range(4)
    )
    minkowski = _sym.Representation.mink(4)
    for leg, momentum in momentum_by_leg.items():
        library.register(
            _sym.LibraryTensor.dense(
                _sym.TensorName(symbols.ufo_momentum_tensor_name(leg + 1))(minkowski),
                momentum,
            )
        )
    if coupling is not None:
        expression *= coupling
    result = _execute_dense_tensor(expression, library)
    expected_dimension = _spin_dimension(source_particles[result_leg].spin)
    if len(result) != expected_dimension:
        raise ValueError(
            f"eager n-ary term {term.vertex}/{term.id} produced {len(result)} "
            f"components, expected {expected_dimension}"
        )
    return tuple(
        _replace_evaluator_constants(_as_expression(result[index]))
        for index in range(len(result))
    )


def _contact_tree_payload_by_leg(
    kind: int,
    side: str,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
) -> dict[int, tuple[tuple[_sym.Expression, ...], tuple[_sym.Expression, ...]]]:
    payload = _contact_tree_node_payload(
        kind,
        side,
        node,
        source_particles,
        scalar_product_tree=False,
    )
    result: dict[
        int, tuple[tuple[_sym.Expression, ...], tuple[_sym.Expression, ...]]
    ] = {}
    cursor = 0
    for leg in node.legs:
        dimension = _spin_dimension(source_particles[leg].spin)
        components = tuple(payload[cursor : cursor + dimension])
        cursor += dimension
        momentum = tuple(payload[cursor : cursor + 4])
        cursor += 4
        result[leg] = components, momentum
    if cursor != len(payload):
        raise ValueError("contact tree payload layout mismatch")
    return result


def _contact_tree_physical_tensor_expression(
    library: _sym.TensorLibrary,
    *,
    kind: int,
    leg: int,
    spin: int,
    components: Sequence[_sym.Expression],
) -> _sym.Expression:
    representations = _spin_representations(spin)
    if not representations:
        if len(components) != 1:
            raise ValueError("scalar contact input must have one component")
        return components[0]
    name = _sym.TensorName(symbols.contact_leg_tensor_name(kind, leg))
    library.register(_sym.LibraryTensor.dense(name(*representations), components))
    return name(*_spin_slots(spin, leg + 1)).to_expression()


def _contact_tree_assignment_multiplicity(
    input_legs: Sequence[int],
    source_particles: Sequence[CompiledParticleRecord],
    left_root: _ContactTreeNode,
    right_root: _ContactTreeNode,
) -> int:
    species_counts: dict[str, int] = {}
    for leg in input_legs:
        name = source_particles[leg].name
        species_counts[name] = species_counts.get(name, 0) + 1
    permutations = math.prod(math.factorial(count) for count in species_counts.values())
    symmetry_nodes = (
        _contact_tree_same_input_node_count(left_root)
        + _contact_tree_same_input_node_count(right_root)
        + int(left_root.particle.name == right_root.particle.name)
    )
    symmetry_divisor = 2**symmetry_nodes
    if permutations % symmetry_divisor:
        raise ValueError("contact tree assignment symmetry is not integral")
    return permutations // symmetry_divisor


def _contact_tree_same_input_node_count(node: _ContactTreeNode) -> int:
    if node.is_leaf:
        return 0
    if node.left is None or node.right is None:
        raise ValueError("contact tree internal node is missing a child")
    return (
        int(node.left.particle.name == node.right.particle.name)
        + _contact_tree_same_input_node_count(node.left)
        + _contact_tree_same_input_node_count(node.right)
    )


def _contact_tree_source_leg(node: _ContactTreeNode) -> int:
    return -1 if node.physical_leg is None else node.physical_leg


def _contact_term_is_color_singlet(term: CompiledVertexTerm) -> bool:
    return term.color_source in {"1", "UFO::{}::1"} or term.color_expression == "1"


def _deduplicate_contact_partials(
    auxiliary_particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    terms: Sequence[CompiledVertexTerm],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    term_by_id = {term.id: term for term in terms}
    auxiliary_by_name = {particle.name: particle for particle in auxiliary_particles}
    representative_by_signature: dict[tuple[object, ...], str] = {}
    replacement: dict[str, str] = {}
    retained: list[CompiledOrientedKernel] = []

    for kernel in kernels:
        if not kernel.vertex.endswith("::contact-partial"):
            retained.append(kernel)
            continue
        auxiliary = auxiliary_by_name[kernel.particles[2]]
        term = term_by_id[kernel.term_id]
        substitutions = {
            symbols.derived_coupling(
                model_symbols.model_name,
                int(name.rsplit("_", 1)[1]),
            ): _sym.E(term.coupling_expression)
            for name in kernel.runtime_parameters
            if name.startswith("derived_coupling_")
        }
        normalized_components = tuple(
            _replace_expression_symbols(
                _remap_kernel_symbols(
                    _sym.E(component),
                    old_kind=kernel.kind,
                    new_kind=0,
                ),
                substitutions,
            ).to_canonical_string()
            for component in kernel.component_expressions
        )
        signature = (
            kernel.particles[:2],
            auxiliary.component_dimension,
            auxiliary.color,
            kernel.coupling_orders,
            kernel.color_source,
            kernel.lc_color_normalization_power,
            normalized_components,
        )
        representative = representative_by_signature.get(signature)
        if representative is None:
            representative_by_signature[signature] = auxiliary.name
            retained.append(kernel)
        else:
            replacement[auxiliary.name] = representative

    if not replacement:
        return tuple(auxiliary_particles), tuple(kernels)

    rewritten = tuple(
        replace(
            kernel,
            particles=tuple(replacement.get(name, name) for name in kernel.particles),
        )
        for kernel in retained
    )
    particles = tuple(
        particle for particle in auxiliary_particles if particle.name not in replacement
    )
    return particles, rewritten
