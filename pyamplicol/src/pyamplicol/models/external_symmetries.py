# SPDX-License-Identifier: 0BSD
"""Algebraic symmetry certificates for compiled external models.

The certificates in this module are deliberately derived from normalized
Lorentz/color tensors and exact coupling relations. UFO particle names,
vertex names, PDG assignments, and coupling-order labels are not proof input.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .contracts import (
    CompiledCouplingRecord,
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from .tensors import (
    _constant_expression_ratio,
    classify_trilinear_color_expression,
    normalize_color_expression,
    normalize_lorentz_expression,
)

_YM_THREE_VECTOR = (
    "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
    "*UFO::P(UFO::idx(1,3),UFO::idx(1,2))"
    "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,2),UFO::idx(1,1))"
    "-UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,1),UFO::idx(1,3))"
    "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
    "*UFO::P(UFO::idx(1,3),UFO::idx(1,1))"
    "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,2),UFO::idx(1,3))"
    "+UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,1),UFO::idx(1,2))"
)

_YM_FOUR_VECTOR_BASIS = (
    (
        "UFO::f(-1,1,2)*UFO::f(3,4,-1)",
        "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,4))"
        "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,4))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))",
    ),
    (
        "UFO::f(-1,1,3)*UFO::f(2,4,-1)",
        "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
        "*UFO::Metric(UFO::idx(1,3),UFO::idx(1,4))"
        "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,4))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))",
    ),
    (
        "UFO::f(-1,1,4)*UFO::f(2,3,-1)",
        "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
        "*UFO::Metric(UFO::idx(1,3),UFO::idx(1,4))"
        "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,4))",
    ),
)

_CONTACT_ORIGIN = re.compile(r"^ufo-contact:(?P<term>[0-9]+):")


@dataclass(frozen=True, slots=True)
class ExternalSymmetryCertificates:
    """Fail-closed model-level certificates consumed by DAG generation."""

    parity_kernel_kinds: frozenset[int]
    yang_mills_kernel_kinds: frozenset[int]
    yang_mills_adjoint_names: frozenset[str]
    adjoint_current_reflection_phases: tuple[
        tuple[int, tuple[float, float]], ...
    ]


def derive_external_symmetry_certificates(
    ir: CompiledModelIR,
) -> ExternalSymmetryCertificates:
    """Recognize exact vectorlike and Yang--Mills sectors in compiled IR."""

    _sym._ensure_symbolica()
    model_symbols = symbols.model(ir.name)
    particles = {particle.name: particle for particle in ir.particles}
    parameters = {parameter.name: parameter for parameter in ir.parameters}
    couplings = {coupling.name: coupling for coupling in ir.couplings}

    ym_three_reference = normalize_lorentz_expression(
        _YM_THREE_VECTOR,
        (3, 3, 3),
        model_symbols=model_symbols,
    ).expression
    ym_cubic_terms: dict[int, str] = {}
    ym_coupling_by_adjoint: dict[str, CompiledCouplingRecord] = {}
    vectorlike_terms: set[int] = set()

    for term in ir.vertex_terms:
        records = tuple(particles[name] for name in term.particles)
        if _is_yang_mills_cubic_term(
            term,
            records,
            ym_three_reference=ym_three_reference,
            parameters=parameters,
        ):
            adjoint_name = records[0].name
            ym_cubic_terms[term.id] = adjoint_name
            ym_coupling_by_adjoint.setdefault(adjoint_name, couplings[term.coupling])
            continue
        if _is_vectorlike_gauge_term(
            term,
            records,
            model_symbols=model_symbols,
            parameters=parameters,
        ):
            vectorlike_terms.add(term.id)

    quartic_groups = _certified_yang_mills_quartic_groups(
        ir,
        particles,
        couplings,
        ym_coupling_by_adjoint,
        model_symbols=model_symbols,
        parameters=parameters,
    )
    ym_quartic_terms = {
        term_id
        for term_ids in quartic_groups.values()
        for term_id in term_ids
    }
    complete_adjoint_names = frozenset(
        name
        for name in ym_coupling_by_adjoint
        if name in quartic_groups
    )

    parity_kinds: set[int] = set()
    yang_mills_kinds: set[int] = set()
    for kernel in ir.oriented_kernels:
        term_ids = frozenset(kernel.term_ids or (kernel.term_id,))
        if term_ids and term_ids <= vectorlike_terms:
            parity_kinds.add(kernel.kind)
        if term_ids and term_ids <= set(ym_cubic_terms):
            names = {ym_cubic_terms[term_id] for term_id in term_ids}
            if names <= complete_adjoint_names:
                parity_kinds.add(kernel.kind)
                yang_mills_kinds.add(kernel.kind)
                continue
        if term_ids and term_ids <= ym_quartic_terms:
            group_names = {
                name
                for name, group_term_ids in quartic_groups.items()
                if term_ids <= group_term_ids
            }
            if group_names:
                parity_kinds.add(kernel.kind)
                yang_mills_kinds.add(kernel.kind)
                continue

        origin_term_id = _contact_origin_term_id(kernel.particles, particles)
        if origin_term_id is not None and origin_term_id in ym_quartic_terms:
            parity_kinds.add(kernel.kind)
            yang_mills_kinds.add(kernel.kind)

    reflection_phases = _certified_two_source_adjoint_current_reflections(
        ir,
        particles,
        parameters,
        model_symbols=model_symbols,
    )

    return ExternalSymmetryCertificates(
        parity_kernel_kinds=frozenset(parity_kinds),
        yang_mills_kernel_kinds=frozenset(yang_mills_kinds),
        yang_mills_adjoint_names=complete_adjoint_names,
        adjoint_current_reflection_phases=tuple(sorted(reflection_phases.items())),
    )


def _is_yang_mills_cubic_term(
    term: CompiledVertexTerm,
    particles: tuple[CompiledParticleRecord, ...],
    *,
    ym_three_reference: str,
    parameters: dict[str, CompiledParameterRecord],
) -> bool:
    if len(particles) != 3 or len({particle.name for particle in particles}) != 1:
        return False
    particle = particles[0]
    if not _is_compile_time_massless_adjoint_vector(particle, parameters):
        return False
    structure, _coefficient = classify_trilinear_color_expression(
        term.color_expression,
        term.color_source,
        tuple(item.color for item in particles),
    )
    return (
        structure == "adjoint-structure-constant"
        and _expressions_are_constant_multiples(
            term.lorentz_expression,
            ym_three_reference,
        )
    )


def _is_vectorlike_gauge_term(
    term: CompiledVertexTerm,
    particles: tuple[CompiledParticleRecord, ...],
    *,
    model_symbols: ModelSymbolRegistry,
    parameters: dict[str, CompiledParameterRecord],
) -> bool:
    if len(particles) != 3:
        return False
    vector_legs = tuple(
        index for index, particle in enumerate(particles) if particle.spin == 3
    )
    fermion_legs = tuple(
        index for index, particle in enumerate(particles) if particle.spin == 2
    )
    if len(vector_legs) != 1 or len(fermion_legs) != 2:
        return False
    vector = particles[vector_legs[0]]
    left_fermion, right_fermion = (particles[index] for index in fermion_legs)
    if not _is_compile_time_massless_adjoint_vector(vector, parameters):
        return False
    if sorted((left_fermion.color, right_fermion.color)) != [-3, 3]:
        return False
    if (
        left_fermion.antiname != right_fermion.name
        or right_fermion.antiname != left_fermion.name
    ):
        return False
    structure, _coefficient = classify_trilinear_color_expression(
        term.color_expression,
        term.color_source,
        tuple(item.color for item in particles),
    )
    if structure != "fundamental-generator":
        return False

    vector_leg = vector_legs[0] + 1
    references = tuple(
        normalize_lorentz_expression(
            "UFO::Gamma("
            f"UFO::idx(1,{vector_leg}),"
            f"UFO::idx(1,{output_leg + 1}),"
            f"UFO::idx(1,{input_leg + 1}))",
            tuple(particle.spin for particle in particles),
            model_symbols=model_symbols,
        ).expression
        for output_leg, input_leg in (
            fermion_legs,
            tuple(reversed(fermion_legs)),
        )
    )
    return any(
        _expressions_are_constant_multiples(term.lorentz_expression, reference)
        for reference in references
    )


def _certified_yang_mills_quartic_groups(
    ir: CompiledModelIR,
    particles: dict[str, CompiledParticleRecord],
    couplings: dict[str, CompiledCouplingRecord],
    cubic_couplings: dict[str, CompiledCouplingRecord],
    *,
    model_symbols: ModelSymbolRegistry,
    parameters: dict[str, CompiledParameterRecord],
) -> dict[str, frozenset[int]]:
    reference_signatures = Counter(
        _tensor_product_signature(
            normalize_color_expression(color, (8, 8, 8, 8)).expression,
            normalize_lorentz_expression(
                lorentz,
                (3, 3, 3, 3),
                model_symbols=model_symbols,
            ).expression,
        )
        for color, lorentz in _YM_FOUR_VECTOR_BASIS
    )
    terms_by_particles: dict[
        tuple[str, ...], list[CompiledVertexTerm]
    ] = defaultdict(list)
    for term in ir.vertex_terms:
        if len(term.particles) == 4:
            terms_by_particles[term.particles].append(term)

    certified: dict[str, frozenset[int]] = {}
    for particle_names, terms in terms_by_particles.items():
        if len(set(particle_names)) != 1:
            continue
        name = particle_names[0]
        particle = particles[name]
        cubic = cubic_couplings.get(name)
        if cubic is None or not _is_compile_time_massless_adjoint_vector(
            particle,
            parameters,
        ):
            continue
        actual_signatures = Counter(
            _tensor_product_signature(term.color_expression, term.lorentz_expression)
            for term in terms
        )
        if actual_signatures != reference_signatures:
            continue
        quartic_couplings = {term.coupling for term in terms}
        if len(quartic_couplings) != 1:
            continue
        quartic = couplings[next(iter(quartic_couplings))]
        if not _has_yang_mills_coupling_relation(cubic, quartic):
            continue
        certified[name] = frozenset(term.id for term in terms)
    return certified


def _has_yang_mills_coupling_relation(
    cubic: CompiledCouplingRecord,
    quartic: CompiledCouplingRecord,
) -> bool:
    cubic_expression = _sym.E(cubic.expression)
    if cubic_expression == _sym.E("0"):
        return False
    ratio = (_sym.E(quartic.expression) / cubic_expression**2).cancel()
    if ratio.get_all_symbols(False):
        return False
    try:
        return complex(ratio) == 1j
    except (RuntimeError, TypeError, ValueError):
        return False


def _tensor_product_signature(color: str, lorentz: str) -> str:
    return str((_sym.E(color) * _sym.E(lorentz)).expand().to_canonical_string())


def _expressions_are_constant_multiples(target: str, reference: str) -> bool:
    target_expression = _sym.E(target)
    reference_expression = _sym.E(reference)
    if target_expression == reference_expression:
        return True
    return (
        _constant_expression_ratio(target_expression, reference_expression) is not None
    )


def _is_compile_time_massless_adjoint_vector(
    particle: CompiledParticleRecord,
    parameters: dict[str, CompiledParameterRecord],
) -> bool:
    return (
        particle.spin == 3
        and particle.color == 8
        and particle.self_conjugate is True
        and _parameter_is_compile_time_zero(particle.mass, parameters)
        and particle.propagating
        and particle.ghost_number == 0
    )


def _parameter_is_compile_time_zero(
    name: str,
    parameters: dict[str, CompiledParameterRecord],
) -> bool:
    parameter = parameters.get(name)
    if parameter is None or parameter.nature != "internal":
        return False
    return (
        _sym.E(parameter.resolved_expression).expand().to_canonical_string()
        == _sym.E("0").to_canonical_string()
    )


def _certified_two_source_adjoint_current_reflections(
    ir: CompiledModelIR,
    particles: dict[str, CompiledParticleRecord],
    parameters: dict[str, CompiledParameterRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> dict[int, tuple[float, float]]:
    """Prove local two-input current reflection from lowered expressions.

    The proof is intentionally narrower than a Yang--Mills model
    classification.  Every cubic kernel of a candidate gauge species must be
    exactly antisymmetric under exchanging its formal input currents and
    momenta, must carry an exact structure-constant color tensor, and must use
    the compiler's standard linear propagator.  One failed kernel disables the
    optimization for the complete species.
    """

    propagators = {propagator.name: propagator for propagator in ir.propagators}
    kernels_by_name: dict[str, list[CompiledOrientedKernel]] = defaultdict(list)
    for kernel in ir.oriented_kernels:
        if len(set(kernel.particles)) == 1:
            kernels_by_name[kernel.particles[0]].append(kernel)

    certified: dict[int, tuple[float, float]] = {}
    for name, kernels in kernels_by_name.items():
        particle = particles.get(name)
        if particle is None or not _is_compile_time_massless_adjoint_vector(
            particle,
            parameters,
        ):
            continue
        propagator = (
            None
            if particle.propagator is None
            else propagators.get(particle.propagator)
        )
        if propagator is None or propagator.custom:
            continue
        if not kernels or any(
            not _kernel_has_exact_adjoint_reflection(
                kernel,
                model_symbols=model_symbols,
            )
            for kernel in kernels
        ):
            continue
        certified.update({kernel.kind: (-1.0, 0.0) for kernel in kernels})
    return certified


def _kernel_has_exact_adjoint_reflection(
    kernel: CompiledOrientedKernel,
    *,
    model_symbols: ModelSymbolRegistry,
) -> bool:
    from .compiler_kernels import (
        _remap_kernel_symbols,
    )

    color_reference = normalize_color_expression(
        "UFO::f(1,2,3)",
        (8, 8, 8),
    ).expression
    color_ratio = _constant_expression_ratio(
        _sym.E(kernel.color_expression),
        _sym.E(color_reference),
    )
    if color_ratio is None or color_ratio == 0j:
        return False

    coupling = _sym.E(kernel.coupling_expression)
    zero = _sym.E("0").to_canonical_string()
    for component_source in kernel.component_expressions:
        component = _sym.E(component_source) * coupling
        direct = _remap_kernel_symbols(
            component,
            old_kind=kernel.kind,
            new_kind=0,
            model_symbols=model_symbols,
        )
        swapped = _remap_kernel_symbols(
            component,
            old_kind=kernel.kind,
            new_kind=0,
            model_symbols=model_symbols,
            swap_sides=True,
        )
        if (swapped + direct).expand().to_canonical_string() != zero:
            return False
    return True


def _contact_origin_term_id(
    particle_names: tuple[str, str, str],
    particles: dict[str, CompiledParticleRecord],
) -> int | None:
    for name in particle_names:
        auxiliary_kind = particles[name].auxiliary_kind
        if auxiliary_kind is None:
            continue
        match = _CONTACT_ORIGIN.match(auxiliary_kind)
        if match is not None:
            return int(match.group("term"))
    return None


__all__ = [
    "ExternalSymmetryCertificates",
    "derive_external_symmetry_certificates",
]
