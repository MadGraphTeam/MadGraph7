# SPDX-License-Identifier: 0BSD
"""Compiler-owned color-flow auxiliaries derived from proven model tensors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry
from . import compiler_symbolica as _sym
from .compiler_kernels import (
    _canonicalize_oriented_kernel_component,
    _remap_kernel_symbols,
)
from .contracts import (
    U1_SUBTRACTION_AUXILIARY_KIND,
    CompiledLCColorTransitionTerm,
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    compiled_particle_component_dimension,
)
from .tensors import normalize_color_expression

U1_SUBTRACTION_PREFIX = "__pyamplicol_u1_subtraction_"
U1_IDENTITY_PREFIX = "__pyamplicol_u1_identity_"
# Auxiliary kind suffix of the identity contact lines (their contact line's
# kind, so that contact bookkeeping keeps reading their origin).
U1_IDENTITY_SUFFIX = ":u1-identity"
U1_PROJECTION_SUFFIX = "::u1-projection"
# Colour rules of oriented stages that couple to the U(1) part of an adjoint
# leg: the d-symbol and identities between adjoint legs. Structure constants
# and fundamental generators do not (or have the Fierz auxiliaries).
_U1_COUPLING_RULES = frozenset({"adjoint-symmetric-invariant", "color-identity"})
_U1_FREE_RULES = frozenset({"adjoint-structure-constant", "fundamental-generator"})


class _CompiledLCColorTransitionCompilation(tuple):
    """Transition tuple carrying compiler-owned closure companions.

    The oriented-kernel compiler call site predates recurrence closure
    contracts and assigns only the transition tuple. ``CompiledOrientedKernel``
    consumes this private companion immediately and persists the two catalogs
    as separate fields.
    """

    compiler_closure_terms: tuple[CompiledLCColorTransitionTerm, ...]

    def __new__(
        cls,
        transition_terms: tuple[CompiledLCColorTransitionTerm, ...],
        closure_terms: tuple[CompiledLCColorTransitionTerm, ...],
    ) -> _CompiledLCColorTransitionCompilation:
        instance = super().__new__(cls, transition_terms)
        instance.compiler_closure_terms = closure_terms
        return instance


def compile_lc_color_transition_terms(
    kernel: CompiledOrientedKernel,
    oriented_representations: tuple[int, int, int],
    *,
    proof_source: str,
    provenance: tuple[tuple[str, str], ...] = (),
    tensor_role_representations: tuple[int, int, int] | None = None,
) -> tuple[CompiledLCColorTransitionTerm, ...]:
    """Compile exact ordered-word operations from a certified local tensor.

    This function is part of model compilation. Runtime and recurrence-builder
    code consume its closed records and never repeat tensor-family inference.
    """

    structure = kernel.color_projection_structure
    if structure is None:
        raise ValueError(f"kernel {kernel.kind} has no certified color projection")
    tensor_representations = (
        oriented_representations
        if tensor_role_representations is None
        else tuple(int(value) for value in tensor_role_representations)
    )
    if tuple(abs(value) for value in tensor_representations) != tuple(
        abs(value) for value in oriented_representations
    ):
        raise ValueError(
            f"kernel {kernel.kind} tensor-role representations "
            f"{tensor_representations!r} do not match its oriented current shapes "
            f"{oriented_representations!r}"
        )
    shapes = tuple(_lc_color_shape(value) for value in oriented_representations)
    result_shape = shapes[2]

    def term(
        permutation: tuple[int, int],
        operation: str,
        result_component_kind: str | None,
        exact_factor_expression: str = "1",
        *,
        closure: bool = False,
    ) -> CompiledLCColorTransitionTerm:
        result_component_role = (
            "passive"
            if operation == "concatenate-join" and abs(output_representation) == 1
            else "active"
            if operation in {"concatenate-join", "inherit-left", "inherit-right"}
            else "none"
        )
        payload = {
            "color_expression": kernel.color_expression,
            "color_source": kernel.color_source,
            "exact_factor_expression": exact_factor_expression,
            "input_permutation": list(permutation),
            "input_shape_kinds": list(shapes[:2]),
            "kernel_kind": kernel.kind,
            "operation": operation,
            "oriented_representations": list(oriented_representations),
            "tensor_role_representations": list(tensor_representations),
            "proof_source": proof_source,
            "result_component_kind": result_component_kind,
            "result_component_role": result_component_role,
            "result_shape_kind": None if closure else result_shape,
            "source_particle_legs": list(kernel.source_particle_legs),
            "structure": structure,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        term_provenance = tuple(
            sorted(
                (
                    ("compiler-proof-source", proof_source),
                    (
                        "contract-kind",
                        "closure" if closure else "transition",
                    ),
                    ("kernel-kind", str(kernel.kind)),
                    (
                        "source-particle-legs",
                        ",".join(str(value) for value in kernel.source_particle_legs),
                    ),
                    *provenance,
                )
            )
        )
        return CompiledLCColorTransitionTerm(
            input_permutation=permutation,
            reverse_parent_mask=0,
            component_operation=operation,
            result_component_kind=result_component_kind,
            result_component_role=result_component_role,
            input_shape_kinds=(shapes[0], shapes[1]),
            result_shape_kind=None if closure else result_shape,
            exact_factor_expression=exact_factor_expression,
            proof_digest=digest,
            provenance=term_provenance,
        )

    def compiled(
        *transition_terms: CompiledLCColorTransitionTerm,
    ) -> tuple[CompiledLCColorTransitionTerm, ...]:
        left_representation, right_representation = oriented_representations[:2]
        if abs(left_representation) == abs(right_representation) == 1:
            closure_kind = None
        elif {left_representation, right_representation} == {3, -3}:
            closure_kind = "open-string"
        elif left_representation == right_representation == 8:
            closure_kind = "trace"
        else:
            closure_terms: tuple[CompiledLCColorTransitionTerm, ...] = ()
            return _CompiledLCColorTransitionCompilation(
                tuple(transition_terms),
                closure_terms,
            )
        closure_terms = (
            term(
                (0, 1),
                "close",
                closure_kind,
                closure=True,
            ),
        )
        return _CompiledLCColorTransitionCompilation(
            tuple(transition_terms),
            closure_terms,
        )

    left_representation, right_representation, output_representation = (
        oriented_representations
    )
    if structure == "singlet":
        if oriented_representations != (1, 1, 1):
            raise ValueError(
                "a colored literal-singlet transition requires explicit contact "
                f"provenance; kernel {kernel.kind} has {oriented_representations!r}"
            )
        return compiled(term((0, 1), "empty", None))
    if structure == "color-identity":
        colored_inputs = tuple(
            index
            for index, representation in enumerate(oriented_representations[:2])
            if abs(representation) != 1
        )
        if len(colored_inputs) == 0:
            return compiled(term((0, 1), "empty", None))
        if len(colored_inputs) == 1:
            operation = "inherit-left" if colored_inputs[0] == 0 else "inherit-right"
            return compiled(term((0, 1), operation, None))
        if len(colored_inputs) == 2 and abs(output_representation) == 1:
            if {left_representation, right_representation} == {3, -3}:
                permutation = (0, 1) if left_representation == 3 else (1, 0)
                result_component = "open-string"
            elif left_representation == right_representation == 8:
                permutation = (0, 1)
                result_component = "trace"
            else:
                raise ValueError(
                    f"kernel {kernel.kind} has unsupported identity contraction "
                    f"{oriented_representations!r}"
                )
            return compiled(
                term(
                    permutation,
                    "concatenate-join",
                    result_component,
                )
            )
        raise ValueError(
            f"kernel {kernel.kind} has unsupported color-identity orientation "
            f"{oriented_representations!r}"
        )
    if structure == "fundamental-generator":
        try:
            fundamental = tensor_representations.index(3)
            antifundamental = tensor_representations.index(-3)
            adjoint = tensor_representations.index(8)
        except ValueError as exc:
            raise ValueError(
                f"kernel {kernel.kind} fundamental generator has invalid oriented "
                f"tensor roles {tensor_representations!r} for current shapes "
                f"{oriented_representations!r}"
            ) from exc
        tensor_output_representation = tensor_representations[2]
        if tensor_output_representation == -3:
            permutation = (fundamental, adjoint)
            result_component = "open-string"
        elif tensor_output_representation == 3:
            permutation = (adjoint, antifundamental)
            result_component = "open-string"
        elif tensor_output_representation == 8:
            permutation = (fundamental, antifundamental)
            result_component = "adjoint-segment"
        else:
            raise ValueError(
                f"kernel {kernel.kind} fundamental generator has unsupported output "
                f"tensor representation {tensor_output_representation}"
            )
        if set(permutation) != {0, 1}:
            raise ValueError(
                f"kernel {kernel.kind} fundamental generator output is not slot 2"
            )
        return compiled(
            term(
                permutation,
                "concatenate-join",
                result_component,
            )
        )
    if structure == "adjoint-structure-constant":
        if oriented_representations != (8, 8, 8):
            raise ValueError(
                f"kernel {kernel.kind} structure constant has representations "
                f"{oriented_representations!r}"
            )
        return compiled(
            term((0, 1), "concatenate-join", "adjoint-segment"),
            term(
                (1, 0),
                "concatenate-join",
                "adjoint-segment",
                "-1",
            ),
        )
    if structure == "adjoint-symmetric-invariant":
        # d^abc = (tr(abc) + tr(acb)) / sqrt(2) with tau = sqrt(2) t: both
        # orders with the same sign (the structure constant's differ).
        if oriented_representations != (8, 8, 8):
            raise ValueError(
                f"kernel {kernel.kind} d-symbol has representations "
                f"{oriented_representations!r}"
            )
        return compiled(
            term((0, 1), "concatenate-join", "adjoint-segment"),
            term((1, 0), "concatenate-join", "adjoint-segment"),
        )
    raise ValueError(
        f"kernel {kernel.kind} has unsupported recurrence color projection "
        f"{structure!r}"
    )


def _lc_color_shape(representation: int) -> str:
    return {
        1: "singlet-forest",
        3: "fundamental-open-string",
        -3: "antifundamental-open-string",
        8: "adjoint-segment",
    }.get(representation) or _unsupported_lc_color_shape(representation)


def _unsupported_lc_color_shape(representation: int) -> str:
    raise ValueError(
        f"recurrence LC color witnesses do not support representation {representation}"
    )


def synthesize_fundamental_fierz_auxiliaries(
    particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    propagators: Sequence[CompiledPropagatorRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[
    tuple[CompiledParticleRecord, ...],
    tuple[CompiledOrientedKernel, ...],
    tuple[CompiledPropagatorRecord, ...],
]:
    """Materialize the singlet term of a proven fundamental Fierz identity.

    A certified ``T^a_ij`` kernel implies the SU(3) color-flow decomposition
    into an ordinary line connection and a ``1 / N_c`` singlet subtraction.
    The synthetic current reuses the exact compiled Lorentz/coupling kernel;
    no particle name or PDG label participates in the proof.

    This milestone deliberately targets the supported QCD-like contract:
    massless self-conjugate adjoint vectors and fundamental fermion lines.
    Models outside that contract keep the existing exact sector-partition
    fallback instead of acquiring an unproven auxiliary. The auxiliary
    propagates like its source: a source with a model-supplied propagator
    (e.g. a counterterm clone line, ``models.ct_line_insertions``) gives its
    auxiliary a copy of that propagator, returned with the propagators.
    """

    particle_by_name = {particle.name: particle for particle in particles}
    propagator_by_particle = {
        propagator.particle: propagator for propagator in propagators
    }
    eligible_sources: set[str] = set()
    for kernel in kernels:
        source_name = _eligible_adjoint_source(kernel, particle_by_name)
        if source_name is not None:
            eligible_sources.add(source_name)
    if not eligible_sources:
        return tuple(particles), tuple(kernels), tuple(propagators)

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
    auxiliary_propagators: list[CompiledPropagatorRecord] = []
    for source_name in sorted(eligible_sources):
        source = particle_by_name[source_name]
        auxiliary_name = f"{U1_SUBTRACTION_PREFIX}{source_name}"
        custom = propagator_by_particle.get(source_name)
        if custom is not None and not custom.custom:
            custom = None
        if custom is not None:
            auxiliary_propagators.append(
                replace(custom, name=f"{auxiliary_name}_prop", particle=auxiliary_name)
            )
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
            propagator=None if custom is None else f"{auxiliary_name}_prop",
            component_dimension=compiled_particle_component_dimension(source),
            auxiliary_kind=U1_SUBTRACTION_AUXILIARY_KIND,
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
        synthetic_kernel = CompiledOrientedKernel(
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
        synthetic.append(
            replace(
                synthetic_kernel,
                lc_color_transition_terms=compile_lc_color_transition_terms(
                    synthetic_kernel,
                    synthetic_representations,  # type: ignore[arg-type]
                    proof_source="fundamental-fierz-identity-v1",
                    provenance=(("source-kernel-kind", str(kernel.kind)),),
                ),
            )
        )

    return (
        (*particles, *auxiliary_by_source.values()),
        (*kernels, *synthetic),
        (*propagators, *auxiliary_propagators),
    )


def synthesize_u1_projections(
    particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    *,
    projected_vertices: frozenset[str],
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Project the adjoint legs of colour-flow stages onto SU(N) (design §6.1).

    The tree recursion carries a gluon ``G`` between fermion lines as its
    U(N) colour-flow current plus the U(1)-subtraction auxiliary ``aux_G`` of
    ``synthesize_fundamental_fierz_auxiliaries``: the SU(N) current is
    ``G_U + alpha 1`` with ``alpha = -tr(G_U) / N`` the auxiliary. Only
    fermion vertices produce and absorb ``aux_G``, which is exact as long as
    every other vertex of ``G`` is blind to the U(1) part (structure
    constants). A stage with a d-symbol or an identity between adjoint legs
    is not, so for the stages of ``projected_vertices`` (base names of
    vertex rules, e.g. the counterterm vertices of ``models.loading``) this
    adds the exact projection of each ``G`` leg:

    * **absorption**: the stage with an input ``G`` replaced by ``aux_G``
      (``alpha`` times the stage with the identity for that leg; colour
      weight ``-c``, the sign of ``alpha``);
    * **production**: a ``G`` output replaced by ``aux_G`` (minus the trace
      of the output over ``N``: colour weight ``-c`` and the exact ``1/3``
      in the coupling, the numeric ``1/N_c`` of the Fierz auxiliaries,
      graded like theirs);
    * **identity dummies**: a d-symbol partial stage of a contact with both
      inputs replaced gives the identity on its adjoint contact line; it is
      carried by a colour-singlet clone of that line
      (``__pyamplicol_u1_identity_*``), read by copies of the d-symbol final
      stages.

    Terms that vanish are not generated: an output that is the identity
    (projected out together with its production) and closings of the
    identity alone. A closing whose left input is replaced sums to zero over
    the patterns of a right input that is a gluon (``tr(G_U) + N alpha =
    0``), so closings substitute their right input only; the colour engine
    keeps the pattern whose replaced input has the larger first label.

    Returns the inputs unchanged when a stage outside ``projected_vertices``
    couples to the U(1) part of a ``G`` leg: the invariant
    ``tr(G_U) = -N alpha`` the closings rely on then fails, and callers keep
    their exact fallback (``loops.v2.ct_closure``).
    """

    particle_by_name = {particle.name: particle for particle in particles}
    auxiliary_of = {
        particle.name.removeprefix(U1_SUBTRACTION_PREFIX): particle
        for particle in particles
        if particle.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        and particle.name.startswith(U1_SUBTRACTION_PREFIX)
    }
    if not auxiliary_of or not projected_vertices:
        return tuple(particles), tuple(kernels)

    def base(kernel: CompiledOrientedKernel) -> str:
        return kernel.vertex.split("::", 1)[0]

    def role(name: str) -> str:
        particle = particle_by_name[name]
        if name in auxiliary_of:
            return "gluon"
        if particle.auxiliary_kind is not None and particle.auxiliary_kind.startswith(
            "ufo-contact"
        ):
            return "dummy-adjoint" if particle.color == 8 else "dummy-singlet"
        return "adjoint" if particle.color == 8 else "other"

    for kernel in kernels:
        if base(kernel) in projected_vertices:
            continue
        if any(name in auxiliary_of for name in kernel.particles) and (
            kernel.lc_recurrence_color_rule_kind not in _U1_FREE_RULES
        ):
            return tuple(particles), tuple(kernels)

    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_200_000, max(used_pdgs, default=0) + 1)
    next_kind = max((kernel.kind for kernel in kernels), default=-1) + 1
    identity_of: dict[str, CompiledParticleRecord] = {}
    synthetic: list[CompiledOrientedKernel] = []

    def identity_dummy(name: str) -> CompiledParticleRecord:
        nonlocal next_pdg
        found = identity_of.get(name)
        if found is None:
            while next_pdg in used_pdgs:
                next_pdg += 1
            used_pdgs.add(next_pdg)
            source = particle_by_name[name]
            found = identity_of[name] = replace(
                source,
                name=f"{U1_IDENTITY_PREFIX}{name}",
                antiname=f"{U1_IDENTITY_PREFIX}{name}",
                pdg_code=next_pdg,
                color=1,
                color_role="",
                auxiliary_kind=f"{source.auxiliary_kind}{U1_IDENTITY_SUFFIX}",
            )
        return found

    def copy(
        kernel: CompiledOrientedKernel,
        new_particles: tuple[str, str, str],
        weight: complex,
        *,
        produce: bool = False,
    ) -> None:
        kind = next_kind + len(synthetic)
        representations = tuple(
            1
            if name.startswith(U1_IDENTITY_PREFIX)
            else particle_by_name[name].color
            for name in new_particles
        )
        coloured = tuple(
            index + 1
            for index, representation in enumerate(representations)
            if abs(representation) != 1
        )
        color_source = (
            f"UFO::Identity({coloured[0]},{coloured[1]})" if len(coloured) == 2 else "1"
        )
        coupling = _sym.E(kernel.coupling_expression)
        if produce:
            coupling = coupling * _sym.E("1/3")
        source_legs = tuple(
            -1 if new != old else leg
            for new, old, leg in zip(
                new_particles, kernel.particles, kernel.source_particle_legs, strict=True
            )
        )
        synthetic_kernel = CompiledOrientedKernel(
            kind=kind,
            term_id=kernel.term_id,
            vertex=f"{kernel.vertex}{U1_PROJECTION_SUFFIX}",
            particles=new_particles,
            source_particle_legs=source_legs,  # type: ignore[arg-type]
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
            coupling_expression=coupling.to_canonical_string(),
            coupling_orders=kernel.coupling_orders,
            runtime_parameters=kernel.runtime_parameters,
            color_source=color_source,
            color_expression=normalize_color_expression(
                color_source, representations
            ).expression,
            color_projection_structure="color-identity",
            color_projection_coefficient=(float(weight.real), float(weight.imag)),
            lc_color_normalization_power=kernel.lc_color_normalization_power,
            term_ids=kernel.term_ids,
            lc_recurrence_color_rule_kind="color-identity",
        )
        synthetic.append(
            replace(
                synthetic_kernel,
                lc_color_transition_terms=compile_lc_color_transition_terms(
                    synthetic_kernel,
                    representations,  # type: ignore[arg-type]
                    proof_source="u1-projection-v1",
                    provenance=(("source-kernel-kind", str(kernel.kind)),),
                ),
            )
        )

    d_finals: dict[str, list[CompiledOrientedKernel]] = {}
    for kernel in kernels:
        if (
            base(kernel) in projected_vertices
            and kernel.lc_recurrence_color_rule_kind == "adjoint-symmetric-invariant"
        ):
            for name in kernel.particles[:2]:
                if role(name) == "dummy-adjoint":
                    d_finals.setdefault(name, []).append(kernel)

    for kernel in kernels:
        rule = kernel.lc_recurrence_color_rule_kind
        if base(kernel) not in projected_vertices or rule not in _U1_COUPLING_RULES:
            continue
        roles = tuple(role(name) for name in kernel.particles)
        if not any(value == "gluon" for value in roles):
            continue
        weight = _vertex_weight(kernel)
        left, right, output = kernel.particles
        coloured_inputs = [
            index
            for index in (0, 1)
            if roles[index] in {"gluon", "dummy-adjoint", "adjoint"}
        ]
        closing = roles[2] in {"other", "dummy-singlet"} and len(coloured_inputs) == 2
        substitutable = [index for index in (0, 1) if roles[index] == "gluon"]
        patterns: list[tuple[int, ...]] = [()]
        if closing:
            # Closings replace their right input only (docstring).
            patterns += [(1,)] if 1 in substitutable else []
        else:
            patterns += [(index,) for index in substitutable]
            if len(substitutable) == 2:
                patterns.append((0, 1))
        for pattern in patterns:
            names = list(kernel.particles)
            for index in pattern:
                names[index] = auxiliary_of[kernel.particles[index]].name
            remaining = [index for index in coloured_inputs if index not in pattern]
            sign = (-1) ** len(pattern)
            if pattern:
                if remaining:
                    copy(kernel, tuple(names), sign * weight)  # type: ignore[arg-type]
                elif roles[2] == "dummy-adjoint" and output in d_finals:
                    identity = identity_dummy(output)
                    names[2] = identity.name
                    copy(kernel, tuple(names), sign * weight)  # type: ignore[arg-type]
            if roles[2] == "gluon" and remaining:
                names[2] = auxiliary_of[output].name
                copy(kernel, tuple(names), -sign * weight, produce=True)  # type: ignore[arg-type]
    # The identity on a d-symbol contact line: the final stage reads it as
    # the identity matrix, d(1, C) = 2 C from its two orders.
    for name, finals in d_finals.items():
        if name not in identity_of:
            continue
        identity = identity_of[name].name
        for kernel in finals:
            names = [identity if value == name else value for value in kernel.particles]
            other = next(
                index for index in (0, 1) if kernel.particles[index] != name
            )
            weight = _vertex_weight(kernel)
            if role(kernel.particles[other]) == "gluon" or (
                role(kernel.particles[other]) == "adjoint"
            ):
                copy(kernel, tuple(names), weight)  # type: ignore[arg-type]
                if role(kernel.particles[2]) == "gluon":
                    produced = list(names)
                    produced[2] = auxiliary_of[kernel.particles[2]].name
                    copy(kernel, tuple(produced), -weight, produce=True)  # type: ignore[arg-type]
    if not synthetic:
        return tuple(particles), tuple(kernels)
    return (*particles, *identity_of.values()), (*kernels, *synthetic)


def _vertex_weight(kernel: CompiledOrientedKernel) -> complex:
    """The colour weight of an oriented stage (``vertex_color_weight``)."""

    coefficient = complex(*(kernel.color_projection_coefficient or (1.0, 0.0)))
    structure = kernel.color_projection_structure
    if structure in {"adjoint-structure-constant", "adjoint-structure-constant-product"}:
        return coefficient * (-1j) ** kernel.lc_color_normalization_power
    if structure == "adjoint-trilinear-product":
        return coefficient * (-1j) ** kernel.color_source.count("f(")
    return coefficient


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
    if (
        source.spin != 3
        or source.mass.upper() != "ZERO"
        or source.width.upper() != "ZERO"
        or not source.self_conjugate
    ):
        return None
    return source.name


__all__ = [
    "U1_IDENTITY_PREFIX",
    "U1_IDENTITY_SUFFIX",
    "U1_PROJECTION_SUFFIX",
    "U1_SUBTRACTION_PREFIX",
    "synthesize_fundamental_fierz_auxiliaries",
    "synthesize_u1_projections",
]
