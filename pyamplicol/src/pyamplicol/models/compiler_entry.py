# SPDX-License-Identifier: 0BSD
"""Entry point for compiling external UFO model IR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_contact_trees import (
    _compile_color_singlet_contact_trees,
    _deduplicate_contact_partials,
)
from .compiler_contacts import (
    _compress_contact_components,
    _contact_auxiliary_color,
    _contact_final_component_expressions,
    _contact_partial_component_expressions,
    _four_point_contact_color_split,
    _fuse_contact_finals,
)
from .compiler_kernels import (
    _canonicalize_oriented_kernel_component,
    _fuse_oriented_kernels,
    _lc_color_normalization_power,
    _oriented_component_expressions,
    _permutation_sign,
    _remap_kernel_symbols,
    _replace_expression_symbols,
    _spin_dimension,
)
from .compiler_records import (
    _coupling,
    _mappings,
    _order,
    _parameter,
    _particle,
    _propagator,
    _resolve_coupling_records,
    _resolve_parameter_records,
    _sequence,
)
from .contracts import (
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from .tensors import (
    classify_trilinear_color_expression,
    normalize_color_expression,
    normalize_lorentz_expression,
)


def compile_ufo_model_ir(model: Mapping[str, object]) -> CompiledModelIR:
    _sym._ensure_symbolica()
    model_name = str(model.get("name", "unnamed-model"))
    model_symbols = symbols.model(model_name)
    particles = tuple(_particle(item) for item in _mappings(model.get("particles")))
    particle_by_name = {particle.name: particle for particle in particles}
    lorentz_by_name = {
        str(item["name"]): item for item in _mappings(model.get("lorentz_structures"))
    }
    parameter_records = tuple(
        _parameter(item, model_symbols=model_symbols)
        for item in _mappings(model.get("parameters"))
    )
    parameter_records = _resolve_parameter_records(parameter_records, model_symbols)
    coupling_records = tuple(
        _coupling(item, model_symbols=model_symbols)
        for item in _mappings(model.get("couplings"))
    )
    coupling_records = _resolve_coupling_records(
        coupling_records,
        parameter_records,
        model_symbols,
    )
    coupling_by_name = {coupling.name: coupling for coupling in coupling_records}
    terms: list[CompiledVertexTerm] = []
    for vertex in _mappings(model.get("vertex_rules")):
        particle_names = tuple(str(value) for value in _sequence(vertex["particles"]))
        try:
            vertex_particles = tuple(particle_by_name[name] for name in particle_names)
        except KeyError as exc:
            raise ValueError(
                f"vertex {vertex.get('name')} refers to unknown particle {exc.args[0]}"
            ) from exc
        colors = tuple(particle.color for particle in vertex_particles)
        color_sources = tuple(
            str(value) for value in _sequence(vertex["color_structures"])
        )
        lorentz_names = tuple(
            str(value) for value in _sequence(vertex["lorentz_structures"])
        )
        coupling_matrix = _sequence(vertex["couplings"])
        if len(coupling_matrix) != len(color_sources):
            raise ValueError(
                f"vertex {vertex.get('name')} coupling rows do not match "
                "color structures"
            )
        normalized_colors = tuple(
            normalize_color_expression(source, colors) for source in color_sources
        )
        normalized_lorentz = []
        for name in lorentz_names:
            try:
                lorentz = lorentz_by_name[name]
            except KeyError as exc:
                raise ValueError(
                    f"vertex {vertex.get('name')} refers to unknown Lorentz "
                    f"structure {name}"
                ) from exc
            normalized_lorentz.append(
                normalize_lorentz_expression(
                    str(lorentz["structure"]),
                    tuple(int(value) for value in _sequence(lorentz["spins"])),
                    model_symbols=model_symbols,
                )
            )
        for color_index, row_value in enumerate(coupling_matrix):
            row = _sequence(row_value)
            if len(row) != len(lorentz_names):
                raise ValueError(
                    f"vertex {vertex.get('name')} coupling columns do not match "
                    "Lorentz structures"
                )
            for lorentz_index, coupling_value in enumerate(row):
                if coupling_value is None:
                    continue
                coupling_name = str(coupling_value)
                try:
                    coupling = coupling_by_name[coupling_name]
                except KeyError as exc:
                    raise ValueError(
                        f"vertex {vertex.get('name')} refers to unknown coupling "
                        f"{coupling_name}"
                    ) from exc
                source_lorentz = lorentz_by_name[lorentz_names[lorentz_index]]
                terms.append(
                    CompiledVertexTerm(
                        id=len(terms),
                        vertex=str(vertex["name"]),
                        particles=particle_names,
                        color_index=color_index,
                        lorentz_index=lorentz_index,
                        color_source=color_sources[color_index],
                        color_expression=normalized_colors[color_index].expression,
                        lorentz_name=lorentz_names[lorentz_index],
                        lorentz_source=str(source_lorentz["structure"]),
                        lorentz_expression=normalized_lorentz[lorentz_index].expression,
                        coupling=coupling.name,
                        coupling_expression=coupling.resolved_expression,
                        coupling_orders=coupling.orders,
                        lc_color_normalization_power=(
                            _lc_color_normalization_power(color_sources[color_index])
                        ),
                    )
                )
    propagators = tuple(
        _propagator(item, particles) for item in _mappings(model.get("propagators"))
    )
    oriented_kernels = _compile_oriented_kernels(
        terms,
        particles,
        parameter_records,
        model_symbols,
    )
    contact_particles, contact_kernels = _compile_four_point_contact_kernels(
        terms,
        particles,
        start_kind=len(oriented_kernels),
        model_symbols=model_symbols,
    )
    contact_particles, contact_kernels = _deduplicate_contact_partials(
        contact_particles,
        contact_kernels,
        terms,
        model_symbols=model_symbols,
    )
    contact_kernels = _fuse_contact_finals(
        contact_kernels,
        terms,
        model_symbols=model_symbols,
    )
    particles = (*particles, *contact_particles)
    oriented_kernels = (*oriented_kernels, *contact_kernels)
    tree_start_kind = max((kernel.kind for kernel in oriented_kernels), default=-1) + 1
    tree_particles, tree_kernels = _compile_color_singlet_contact_trees(
        terms,
        particles,
        start_kind=tree_start_kind,
        model_symbols=model_symbols,
    )
    particles = (*particles, *tree_particles)
    oriented_kernels = (*oriented_kernels, *tree_kernels)
    oriented_kernels = _annotate_oriented_kernel_color_projections(
        oriented_kernels,
        particles,
    )
    oriented_kernels = _annotate_oriented_kernel_evaluation_equivalence(
        oriented_kernels,
        particles,
        terms,
        model_symbols,
    )
    return CompiledModelIR(
        name=model_name,
        orders=tuple(_order(item) for item in _mappings(model.get("orders"))),
        parameters=parameter_records,
        particles=particles,
        couplings=coupling_records,
        propagators=propagators,
        vertex_terms=tuple(terms),
        oriented_kernels=oriented_kernels,
    )


def _annotate_oriented_kernel_color_projections(
    kernels: Sequence[CompiledOrientedKernel],
    particles: Sequence[CompiledParticleRecord],
) -> tuple[CompiledOrientedKernel, ...]:
    particle_by_name = {particle.name: particle for particle in particles}
    annotated: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        representations = tuple(
            particle_by_name[name].color for name in kernel.particles
        )
        structure, coefficient = classify_trilinear_color_expression(
            kernel.color_expression,
            kernel.color_source,
            representations,
        )
        annotated.append(
            replace(
                kernel,
                color_projection_structure=structure,
                color_projection_coefficient=(
                    float(coefficient.real),
                    float(coefficient.imag),
                ),
            )
        )
    return tuple(annotated)


def _annotate_oriented_kernel_evaluation_equivalence(
    kernels: Sequence[CompiledOrientedKernel],
    particles: Sequence[CompiledParticleRecord],
    terms: Sequence[CompiledVertexTerm],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    """Prove exact signed/permuted kernel relations from lowered expressions.

    The comparison uses the concrete component expressions after resolving
    generated coupling aliases.  It is therefore independent of how a UFO
    author spelled or ordered the source Lorentz/color structures.  Only exact
    Symbolica canonical equality up to an overall sign and input exchange is
    recorded; kernels that do not pass that test remain in distinct classes.
    """

    particle_by_name = {particle.name: particle for particle in particles}
    term_by_id = {term.id: term for term in terms}
    annotated: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        derived_couplings = {
            symbols.derived_coupling(model_symbols.model_name, term_id): _sym.E(
                term_by_id[term_id].coupling_expression
            )
            for name in kernel.runtime_parameters
            if name.startswith("derived_coupling_")
            for term_id in (int(name.rsplit("_", 1)[1]),)
            if term_id in term_by_id
        }
        coupling = _replace_expression_symbols(
            _sym.E(kernel.coupling_expression),
            derived_couplings,
        )
        dimensions = tuple(
            (
                particle.component_dimension
                if particle.component_dimension is not None
                else _spin_dimension(particle.spin)
            )
            for particle in (
                particle_by_name[kernel.particles[0]],
                particle_by_name[kernel.particles[1]],
                particle_by_name[kernel.particles[2]],
            )
        )
        candidates: list[tuple[str, tuple[int, int], tuple[float, float], str]] = []
        for input_order, swap_sides in (((0, 1), False), ((1, 0), True)):
            oriented_components = tuple(
                _canonicalize_oriented_kernel_component(
                    _replace_expression_symbols(
                        _remap_kernel_symbols(
                            _sym.E(component),
                            old_kind=kernel.kind,
                            new_kind=0,
                            model_symbols=model_symbols,
                            swap_sides=swap_sides,
                        ),
                        derived_couplings,
                    )
                    * coupling
                )
                for component in kernel.component_expressions
            )
            oriented_dimensions = (
                dimensions[input_order[0]],
                dimensions[input_order[1]],
                dimensions[2],
            )
            for sign in (1.0, -1.0):
                component_strings = tuple(
                    _canonicalize_oriented_kernel_component(
                        sign * component
                    ).to_canonical_string()
                    for component in oriented_components
                )
                signature = json.dumps(
                    {
                        "input_dimensions": list(oriented_dimensions[:2]),
                        "output_dimension": oriented_dimensions[2],
                        "components": list(component_strings),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                candidates.append(
                    (
                        signature,
                        input_order,
                        (sign, 0.0),
                        hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                    )
                )
        _signature, input_order, factor, class_digest = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
        annotated.append(
            replace(
                kernel,
                evaluation_class=f"symbolica-sha256:{class_digest}",
                evaluation_factor=factor,
                evaluation_input_order=input_order,
                evaluation_equivalence_verified=True,
            )
        )
    reference_factor_by_class: dict[str, complex] = {}
    normalized: list[CompiledOrientedKernel] = []
    for kernel in annotated:
        class_factor_complex = complex(*kernel.evaluation_factor)
        reference_factor = reference_factor_by_class.setdefault(
            kernel.evaluation_class,
            class_factor_complex,
        )
        relative_factor = class_factor_complex / reference_factor
        normalized.append(
            replace(
                kernel,
                evaluation_factor=(
                    float(relative_factor.real),
                    float(relative_factor.imag),
                ),
            )
        )
    return tuple(normalized)


def _compile_oriented_kernels(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    parameters: Sequence[CompiledParameterRecord],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    particle_by_name = {particle.name: particle for particle in particles}
    external_parameters = {
        parameter.name for parameter in parameters if parameter.nature == "external"
    }
    kernels: list[CompiledOrientedKernel] = []
    for term in terms:
        if term.valence != 3:
            continue
        oriented_result_particles: set[str] = set()
        for result_leg in range(3):
            result_source_name = term.particles[result_leg]
            if result_source_name in oriented_result_particles:
                continue
            oriented_result_particles.add(result_source_name)
            input_legs = tuple(leg for leg in range(3) if leg != result_leg)
            input_orders = (
                (input_legs,)
                if term.particles[input_legs[0]] == term.particles[input_legs[1]]
                else (input_legs, tuple(reversed(input_legs)))
            )
            for left_leg, right_leg in input_orders:
                result_source = particle_by_name[term.particles[result_leg]]
                try:
                    result_name = particle_by_name[result_source.antiname].name
                except KeyError as exc:
                    raise ValueError(
                        f"vertex {term.vertex} particle {result_source.name} refers to "
                        f"absent antiparticle {result_source.antiname}"
                    ) from exc
                components = _oriented_component_expressions(
                    term,
                    particle_by_name,
                    left_leg=left_leg,
                    right_leg=right_leg,
                    result_leg=result_leg,
                    kind=len(kernels),
                    model_symbols=model_symbols,
                )
                coupling_symbols = set(
                    _sym.E(term.coupling_expression).get_all_symbols(False)
                )
                runtime_parameters = tuple(
                    sorted(
                        name
                        for name in external_parameters
                        if model_symbols.symbol(name) in coupling_symbols
                    )
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=len(kernels),
                        term_id=term.id,
                        vertex=term.vertex,
                        particles=(
                            term.particles[left_leg],
                            term.particles[right_leg],
                            result_name,
                        ),
                        source_particle_legs=(left_leg, right_leg, result_leg),
                        component_expressions=components,
                        coupling_expression=term.coupling_expression,
                        coupling_orders=term.coupling_orders,
                        runtime_parameters=runtime_parameters,
                        color_source=term.color_source,
                        color_expression=term.color_expression,
                        lc_color_normalization_power=(
                            term.lc_color_normalization_power
                        ),
                        term_ids=(term.id,),
                    )
                )
    return _fuse_oriented_kernels(
        kernels,
        model_name=model_symbols.model_name,
    )


def _compile_four_point_contact_kernels(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Lower momentum-independent four-point tensors through dense auxiliaries."""

    particle_by_name = {particle.name: particle for particle in particles}
    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_000_000, max(used_pdgs, default=0) + 1)
    auxiliary_particles: list[CompiledParticleRecord] = []
    kernels: list[CompiledOrientedKernel] = []

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    for term in terms:
        if term.valence != 4:
            continue
        if "ufo_momentum_" in term.lorentz_expression:
            continue
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        oriented_result_particles: set[str] = set()
        for result_leg in range(4):
            source_result = source_particles[result_leg]
            if source_result.name in oriented_result_particles:
                continue
            oriented_result_particles.add(source_result.name)
            contact_split = _four_point_contact_color_split(term, result_leg)
            (
                pair_legs,
                remaining_leg,
                _outer_color_source,
                _final_color_source,
                outer_color_power,
                final_color_power,
                outer_color_factor,
                final_color_factor,
                color_dummy,
            ) = contact_split
            assignment_multiplicity = sum(
                source_particles[leg].name == source_particles[remaining_leg].name
                for leg in range(4)
                if leg != result_leg
            )
            open_legs = tuple(sorted((remaining_leg, result_leg)))
            auxiliary_name = f"__pyamplicol_contact_{term.id}_r{result_leg}"
            canonical_kind = start_kind + len(kernels)
            canonical_components = _contact_partial_component_expressions(
                term,
                particle_by_name,
                left_leg=pair_legs[0],
                right_leg=pair_legs[1],
                open_legs=open_legs,
                kind=canonical_kind,
                model_symbols=model_symbols,
            )
            representative_indices, component_expansion = _compress_contact_components(
                canonical_components
            )
            auxiliary_dimension = len(representative_indices)
            auxiliary = CompiledParticleRecord(
                name=auxiliary_name,
                antiname=auxiliary_name,
                pdg_code=allocate_pdg(),
                spin=-1,
                color=_contact_auxiliary_color(
                    term,
                    source_particles,
                    remaining_leg=remaining_leg,
                    result_leg=result_leg,
                ),
                mass="ZERO",
                width="ZERO",
                charge=0.0,
                quantum_numbers=(("electric_charge", "0"),),
                ghost_number=0,
                propagating=False,
                goldstoneboson=False,
                propagator=None,
                component_dimension=auxiliary_dimension,
                auxiliary_kind=(
                    f"ufo-contact:{term.id}:result-{result_leg}:"
                    + ",".join(str(leg) for leg in open_legs)
                ),
            )
            auxiliary_particles.append(auxiliary)

            pair_orders = (
                (pair_legs,)
                if source_particles[pair_legs[0]].name
                == source_particles[pair_legs[1]].name
                else (pair_legs, tuple(reversed(pair_legs)))
            )
            canonical_outer_sign = (
                _permutation_sign(
                    outer_color_factor,
                    (color_dummy, pair_legs[0] + 1, pair_legs[1] + 1),
                )
                if outer_color_factor
                else 1
            )
            for left_leg, right_leg in pair_orders:
                kind = start_kind + len(kernels)
                components = _contact_partial_component_expressions(
                    term,
                    particle_by_name,
                    left_leg=left_leg,
                    right_leg=right_leg,
                    open_legs=open_legs,
                    kind=kind,
                    model_symbols=model_symbols,
                )
                outer_sign = (
                    _permutation_sign(
                        outer_color_factor,
                        (color_dummy, left_leg + 1, right_leg + 1),
                    )
                    if outer_color_factor
                    else 1
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=kind,
                        term_id=term.id,
                        vertex=f"{term.vertex}::contact-partial",
                        particles=(
                            source_particles[left_leg].name,
                            source_particles[right_leg].name,
                            auxiliary.name,
                        ),
                        source_particle_legs=(left_leg, right_leg, -1),
                        component_expressions=tuple(
                            _canonicalize_oriented_kernel_component(
                                _sym.E(components[index])
                                * outer_sign
                                / canonical_outer_sign
                            ).to_canonical_string()
                            for index in representative_indices
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

            result_name = particle_by_name[source_result.antiname].name
            final_orders = (
                ((True, auxiliary.name, source_particles[remaining_leg].name),)
                if auxiliary.name == source_particles[remaining_leg].name
                else (
                    (True, auxiliary.name, source_particles[remaining_leg].name),
                    (False, source_particles[remaining_leg].name, auxiliary.name),
                )
            )
            for auxiliary_on_left, left_name, right_name in final_orders:
                kind = start_kind + len(kernels)
                final_input_tokens = (
                    (color_dummy, remaining_leg + 1)
                    if auxiliary_on_left
                    else (remaining_leg + 1, color_dummy)
                )
                final_sign = (
                    _permutation_sign(
                        final_color_factor,
                        (result_leg + 1, *final_input_tokens),
                    )
                    if final_color_factor
                    else 1
                )
                derived_coupling = symbols.derived_coupling(
                    model_symbols.model_name,
                    term.id,
                )
                final_prefactor = (
                    canonical_outer_sign
                    * final_sign
                    * derived_coupling
                    / assignment_multiplicity
                )
                combined_color_source = (
                    "UFO::{}::f(1,2,3)*UFO::{}::f(1,2,3)"
                    if outer_color_factor
                    else term.color_source
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=kind,
                        term_id=term.id,
                        vertex=f"{term.vertex}::contact-final",
                        particles=(left_name, right_name, result_name),
                        source_particle_legs=(
                            -1 if auxiliary_on_left else remaining_leg,
                            remaining_leg if auxiliary_on_left else -1,
                            result_leg,
                        ),
                        component_expressions=tuple(
                            (_sym.E(component) * final_prefactor).to_canonical_string()
                            for component in _contact_final_component_expressions(
                                source_particles,
                                auxiliary,
                                open_legs=open_legs,
                                remaining_leg=remaining_leg,
                                result_leg=result_leg,
                                kind=kind,
                                auxiliary_on_left=auxiliary_on_left,
                                component_expansion=component_expansion,
                                model_symbols=model_symbols,
                            )
                        ),
                        coupling_expression="1",
                        coupling_orders=term.coupling_orders,
                        runtime_parameters=(f"derived_coupling_{term.id}",),
                        color_source=combined_color_source,
                        color_expression=combined_color_source,
                        lc_color_normalization_power=(
                            outer_color_power + final_color_power
                        ),
                        term_ids=(term.id,),
                    )
                )
    return tuple(auxiliary_particles), tuple(kernels)
