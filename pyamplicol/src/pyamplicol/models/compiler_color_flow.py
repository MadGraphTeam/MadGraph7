# SPDX-License-Identifier: 0BSD
"""Compiler-owned color-flow auxiliaries derived from proven model tensors."""

from __future__ import annotations

from collections.abc import Sequence

from .._internal.physics.symbols import ModelSymbolRegistry
from . import compiler_symbolica as _sym
from .compiler_kernels import (
    _canonicalize_oriented_kernel_component,
    _remap_kernel_symbols,
)
from .contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    compiled_particle_component_dimension,
)
from .tensors import normalize_color_expression

_U1_SUBTRACTION_AUXILIARY = "u1-subtraction-color-flow-vector"


def synthesize_fundamental_fierz_auxiliaries(
    particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    propagators: Sequence[CompiledPropagatorRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Materialize the singlet term of a proven fundamental Fierz identity.

    A certified ``T^a_ij`` kernel implies the SU(3) color-flow decomposition
    into an ordinary line connection and a ``1 / N_c`` singlet subtraction.
    The synthetic current reuses the exact compiled Lorentz/coupling kernel;
    no particle name or PDG label participates in the proof.

    This milestone deliberately targets the supported QCD-like contract:
    massless self-conjugate adjoint vectors with a default propagator and
    fundamental fermion lines. Models outside that contract keep the existing
    exact sector-partition fallback instead of acquiring an unproven auxiliary.
    """

    particle_by_name = {particle.name: particle for particle in particles}
    propagator_by_particle = {
        propagator.particle: propagator for propagator in propagators
    }
    eligible_sources: set[str] = set()
    for kernel in kernels:
        source_name = _eligible_adjoint_source(
            kernel,
            particle_by_name,
            propagator_by_particle,
        )
        if source_name is not None:
            eligible_sources.add(source_name)
    if not eligible_sources:
        return tuple(particles), tuple(kernels)

    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_100_000, max(used_pdgs, default=0) + 1)

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    quantum_number_names = sorted(
        {
            name
            for particle in particles
            for name, _expression in particle.quantum_numbers
        }
        | {"electric_charge"}
    )
    auxiliary_by_source: dict[str, CompiledParticleRecord] = {}
    for source_name in sorted(eligible_sources):
        source = particle_by_name[source_name]
        auxiliary_name = f"__pyamplicol_u1_subtraction_{source_name}"
        auxiliary_by_source[source_name] = CompiledParticleRecord(
            name=auxiliary_name,
            antiname=auxiliary_name,
            pdg_code=allocate_pdg(),
            spin=source.spin,
            color=1,
            mass=source.mass,
            width=source.width,
            charge=0.0,
            quantum_numbers=tuple((name, "0") for name in quantum_number_names),
            ghost_number=0,
            propagating=False,
            goldstoneboson=False,
            propagator=None,
            component_dimension=compiled_particle_component_dimension(source),
            auxiliary_kind=_U1_SUBTRACTION_AUXILIARY,
        )

    synthetic: list[CompiledOrientedKernel] = []
    next_kind = max((kernel.kind for kernel in kernels), default=-1) + 1
    for kernel in kernels:
        source_name = _kernel_adjoint_source(
            kernel,
            particle_by_name,
            eligible_sources,
        )
        if source_name is None:
            continue
        source_slot = kernel.particles.index(source_name)
        representations = tuple(
            particle_by_name[name].color for name in kernel.particles
        )
        if source_slot == 2 and representations[:2] != (3, -3):
            # Match the canonical oriented closure used by the LC recursion.
            # The reverse input ordering is an equivalent physical kernel, but
            # materializing both would create duplicate subtraction currents.
            continue

        auxiliary = auxiliary_by_source[source_name]
        synthetic_particles = list(kernel.particles)
        synthetic_particles[source_slot] = auxiliary.name
        synthetic_representations = tuple(
            auxiliary.color if index == source_slot else representations[index]
            for index in range(3)
        )
        colored_legs = tuple(
            index + 1
            for index, representation in enumerate(synthetic_representations)
            if abs(representation) != 1
        )
        if len(colored_legs) != 2:
            raise ValueError(
                f"Fierz auxiliary for kernel {kernel.kind} did not leave two "
                "fundamental color legs"
            )
        color_source = f"UFO::Identity({colored_legs[0]},{colored_legs[1]})"
        color_expression = normalize_color_expression(
            color_source,
            synthetic_representations,
        ).expression

        kind = next_kind + len(synthetic)
        coefficient = complex(*(kernel.color_projection_coefficient or (1.0, 0.0)))
        if source_slot == 2:
            fundamental_dimension = abs(representations[0])
            coefficient /= fundamental_dimension
        source_legs = list(kernel.source_particle_legs)
        source_legs[source_slot] = -1
        synthetic.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=kernel.term_id,
                vertex=f"{kernel.vertex}::u1-subtraction",
                particles=tuple(synthetic_particles),
                source_particle_legs=tuple(source_legs),
                component_expressions=tuple(
                    _canonicalize_oriented_kernel_component(
                        _remap_kernel_symbols(
                            _sym.E(component),
                            old_kind=kernel.kind,
                            new_kind=kind,
                            model_symbols=model_symbols,
                        )
                    ).to_canonical_string()
                    for component in kernel.component_expressions
                ),
                coupling_expression=kernel.coupling_expression,
                coupling_orders=kernel.coupling_orders,
                runtime_parameters=kernel.runtime_parameters,
                color_source=color_source,
                color_expression=color_expression,
                color_projection_structure="color-identity",
                color_projection_coefficient=(
                    float(coefficient.real),
                    float(coefficient.imag),
                ),
                lc_color_normalization_power=kernel.lc_color_normalization_power,
                term_ids=kernel.term_ids,
            )
        )

    return (
        (*particles, *auxiliary_by_source.values()),
        (*kernels, *synthetic),
    )


def _kernel_adjoint_source(
    kernel: CompiledOrientedKernel,
    particles: dict[str, CompiledParticleRecord],
    eligible_sources: set[str],
) -> str | None:
    if kernel.color_projection_structure != "fundamental-generator":
        return None
    matches = tuple(name for name in kernel.particles if name in eligible_sources)
    if len(matches) != 1:
        return None
    source_name = matches[0]
    representations = tuple(particles[name].color for name in kernel.particles)
    if representations.count(8) != 1:
        return None
    if sorted(abs(value) for value in representations if value != 8) != [3, 3]:
        return None
    return source_name


def _eligible_adjoint_source(
    kernel: CompiledOrientedKernel,
    particles: dict[str, CompiledParticleRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> str | None:
    if kernel.color_projection_structure != "fundamental-generator":
        return None
    representations = tuple(particles[name].color for name in kernel.particles)
    adjoint_slots = tuple(
        index
        for index, representation in enumerate(representations)
        if representation == 8
    )
    if len(adjoint_slots) != 1:
        return None
    if sorted(abs(value) for value in representations if value != 8) != [3, 3]:
        return None
    if not all(
        particles[name].statistics == "fermion"
        for index, name in enumerate(kernel.particles)
        if index != adjoint_slots[0]
    ):
        return None
    source = particles[kernel.particles[adjoint_slots[0]]]
    propagator = propagators.get(source.name)
    if (
        source.spin != 3
        or source.mass.upper() != "ZERO"
        or source.width.upper() != "ZERO"
        or not source.self_conjugate
        or (propagator is not None and propagator.custom)
    ):
        return None
    return source.name


__all__ = [
    "synthesize_fundamental_fierz_auxiliaries",
]
