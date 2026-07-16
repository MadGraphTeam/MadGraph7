# SPDX-License-Identifier: 0BSD
"""Four-point contact component compression and fusion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_kernels import (
    _as_expression,
    _canonicalize_oriented_kernel_component,
    _component_symbols,
    _function_arguments,
    _input_tensor_expression,
    _ordered_dense_tensor_components,
    _remap_kernel_symbols,
    _replace_expression_symbols,
    _spin_axis_labels,
    _spin_dimension,
    _spin_representations,
    _spin_slots,
)
from .compiler_records import _replace_evaluator_constants
from .contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
)


def _fuse_contact_finals(
    kernels: Sequence[CompiledOrientedKernel],
    terms: Sequence[CompiledVertexTerm],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    coupling_by_term = {term.id: term.coupling_expression for term in terms}
    groups: dict[tuple[object, ...], list[CompiledOrientedKernel]] = {}
    passthrough: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        if not kernel.vertex.endswith("::contact-final"):
            passthrough.append(kernel)
            continue
        key = (
            kernel.particles,
            kernel.coupling_orders,
            kernel.color_source,
            kernel.lc_color_normalization_power,
        )
        groups.setdefault(key, []).append(kernel)

    fused: list[CompiledOrientedKernel] = []
    for members in groups.values():
        first = members[0]
        runtime_aliases: dict[_sym.Expression, _sym.Expression] = {}
        canonical_runtime_name: str | None = None
        if all(
            coupling_by_term.get(member.term_id) == coupling_by_term.get(first.term_id)
            for member in members
        ):
            canonical_runtime_name = f"derived_coupling_{first.term_id}"
            canonical_runtime = symbols.derived_coupling(
                model_symbols.model_name,
                first.term_id,
            )
            runtime_aliases = {
                symbols.derived_coupling(
                    model_symbols.model_name,
                    member.term_id,
                ): canonical_runtime
                for member in members
            }
        components = tuple(
            _canonicalize_oriented_kernel_component(
                sum(
                    (
                        _replace_expression_symbols(
                            _remap_kernel_symbols(
                                _sym.E(member.component_expressions[index]),
                                old_kind=member.kind,
                                new_kind=first.kind,
                                model_symbols=model_symbols,
                            ),
                            runtime_aliases,
                        )
                        for member in members
                    ),
                    _sym.E("0"),
                )
            ).to_canonical_string()
            for index in range(len(first.component_expressions))
        )
        fused.append(
            replace(
                first,
                vertex=first.vertex.replace(
                    "::contact-final",
                    "::contact-final-fused",
                ),
                component_expressions=components,
                runtime_parameters=(
                    (canonical_runtime_name,)
                    if canonical_runtime_name is not None
                    else tuple(
                        sorted(
                            {
                                name
                                for member in members
                                for name in member.runtime_parameters
                            }
                        )
                    )
                ),
                term_ids=tuple(
                    term_id
                    for member in members
                    for term_id in (member.term_ids or (member.term_id,))
                ),
            )
        )
    return tuple(sorted((*passthrough, *fused), key=lambda kernel: kernel.kind))


def _contact_partial_component_expressions(
    term: CompiledVertexTerm,
    particle_by_name: Mapping[str, CompiledParticleRecord],
    *,
    left_leg: int,
    right_leg: int,
    open_legs: tuple[int, ...],
    kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[str, ...]:
    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    particles = tuple(particle_by_name[name] for name in term.particles)
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="left",
        spin=particles[left_leg].spin,
        leg=left_leg + 1,
        components=_component_symbols(
            kind,
            "left",
            particles[left_leg].spin,
            model_symbols=model_symbols,
        ),
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="right",
        spin=particles[right_leg].spin,
        leg=right_leg + 1,
        components=_component_symbols(
            kind,
            "right",
            particles[right_leg].spin,
            model_symbols=model_symbols,
        ),
        model_symbols=model_symbols,
    )
    result = _execute_dense_tensor(
        expression,
        library,
        axis_labels=tuple(
            label
            for leg in open_legs
            for label in _spin_axis_labels(particles[leg].spin, leg + 1)
        ),
    )
    expected = math.prod(_spin_dimension(particles[leg].spin) for leg in open_legs)
    if len(result) != expected:
        raise ValueError(
            f"contact partial {term.vertex}/{term.id} produced {len(result)} "
            f"components, expected {expected}"
        )
    return tuple(
        _replace_evaluator_constants(
            _as_expression(result[index])
        ).to_canonical_string()
        for index in range(len(result))
    )


def _contact_final_component_expressions(
    particles: Sequence[CompiledParticleRecord],
    auxiliary: CompiledParticleRecord,
    *,
    open_legs: tuple[int, ...],
    remaining_leg: int,
    result_leg: int,
    kind: int,
    auxiliary_on_left: bool,
    component_expansion: tuple[tuple[int, int] | None, ...],
    model_symbols: ModelSymbolRegistry,
) -> tuple[str, ...]:
    library = _sym.TensorLibrary.hep_lib_atom()
    auxiliary_side = "left" if auxiliary_on_left else "right"
    physical_side = "right" if auxiliary_on_left else "left"
    auxiliary_symbols = tuple(
        model_symbols.kernel_component(kind, auxiliary_side, component)
        for component in range(auxiliary.component_dimension or 0)
    )
    expanded_auxiliary = tuple(
        _sym.E("0") if entry is None else entry[1] * auxiliary_symbols[entry[0]]
        for entry in component_expansion
    )
    expression = _contact_auxiliary_tensor_expression(
        library,
        kind=kind,
        side=auxiliary_side,
        particles=particles,
        open_legs=open_legs,
        components=expanded_auxiliary,
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side=physical_side,
        spin=particles[remaining_leg].spin,
        leg=remaining_leg + 1,
        components=_component_symbols(
            kind,
            physical_side,
            particles[remaining_leg].spin,
            model_symbols=model_symbols,
        ),
        model_symbols=model_symbols,
    )
    result = _execute_dense_tensor(
        expression,
        library,
        axis_labels=_spin_axis_labels(particles[result_leg].spin, result_leg + 1),
    )
    expected = _spin_dimension(particles[result_leg].spin)
    if len(result) != expected:
        raise ValueError(
            f"contact final for source leg {result_leg} produced {len(result)} "
            f"components, expected {expected}"
        )
    return tuple(
        _replace_evaluator_constants(
            _as_expression(result[index])
        ).to_canonical_string()
        for index in range(len(result))
    )


def _contact_auxiliary_tensor_expression(
    library: _sym.TensorLibrary,
    *,
    kind: int,
    side: str,
    particles: Sequence[CompiledParticleRecord],
    open_legs: tuple[int, ...],
    components: Sequence[_sym.Expression],
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    representations = tuple(
        representation
        for leg in open_legs
        for representation in _spin_representations(particles[leg].spin)
    )
    if not representations:
        if len(components) != 1:
            raise ValueError("scalar contact auxiliary must have one component")
        return components[0]
    slots = tuple(
        slot for leg in open_legs for slot in _spin_slots(particles[leg].spin, leg + 1)
    )
    name = _sym.TensorName(model_symbols.kernel_tensor_name(kind, side))
    library.register(_sym.LibraryTensor.dense(name(*representations), components))
    return name(*slots).to_expression()


def _execute_dense_tensor(
    expression: _sym.Expression,
    library: _sym.TensorLibrary,
    *,
    axis_labels: Sequence[str],
) -> tuple[object, ...]:
    if not axis_labels and set(expression.get_all_symbols()) == set(
        expression.get_all_symbols(include_function_symbols=False)
    ):
        # A rank-zero expression with no function indeterminates contains no
        # tensor heads to contract. Spenso need not construct a tensor network.
        return (expression,)
    network = _sym.TensorNetwork(expression, library)
    network.execute(library=library)
    result = network.result_tensor(library)
    return _ordered_dense_tensor_components(result, axis_labels)


def _contact_auxiliary_color(
    term: CompiledVertexTerm,
    particles: Sequence[CompiledParticleRecord],
    *,
    remaining_leg: int,
    result_leg: int,
) -> int:
    colors = tuple(particle.color for particle in particles)
    if all(color == 1 for color in colors):
        return 1
    if "f(" in term.color_source or "::f(" in term.color_source:
        return 8
    remaining = abs(colors[remaining_leg])
    result = abs(colors[result_leg])
    if remaining == 1:
        return colors[result_leg]
    if result == 1:
        return colors[remaining_leg]
    if remaining == result == 8:
        return 1
    return 1


def _compress_contact_components(
    components: Sequence[str],
) -> tuple[tuple[int, ...], tuple[tuple[int, int] | None, ...]]:
    representatives: list[_sym.Expression] = []
    representative_indices: list[int] = []
    expansion: list[tuple[int, int] | None] = []
    zero = _sym.E("0")
    for index, source in enumerate(components):
        expression = _sym.E(source)
        if str(expression) == "0":
            expansion.append(None)
            continue
        match: tuple[int, int] | None = None
        for basis_index, representative in enumerate(representatives):
            if str((expression - representative).expand()) == "0":
                match = (basis_index, 1)
                break
            if str((expression + representative).expand()) == "0":
                match = (basis_index, -1)
                break
        if match is None:
            match = (len(representatives), 1)
            representatives.append(expression)
            representative_indices.append(index)
        expansion.append(match)
    if not representatives:
        representatives.append(zero)
        representative_indices.append(0)
    return tuple(representative_indices), tuple(expansion)


def _four_point_contact_color_split(
    term: CompiledVertexTerm,
    result_leg: int,
) -> tuple[
    tuple[int, int],
    int,
    str,
    str,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    int,
] | None:
    factors = _normalized_structure_constant_factors(term.color_expression)
    if len(factors) == 2:
        shared_dummies = set(value for value in factors[0] if value < 0) & set(
            value for value in factors[1] if value < 0
        )
        if len(shared_dummies) == 1:
            dummy = next(iter(shared_dummies))
            result_index = result_leg + 1
            final_factor = next(
                (factor for factor in factors if result_index in factor),
                None,
            )
            if final_factor is not None:
                outer_factor = factors[1] if final_factor is factors[0] else factors[0]
                pair = tuple(value - 1 for value in outer_factor if value > 0)
                remaining = tuple(
                    value - 1
                    for value in final_factor
                    if value > 0 and value != result_index
                )
                if len(pair) == 2 and len(remaining) == 1:
                    canonical_f = "UFO::{}::f(1,2,3)"
                    return (
                        (pair[0], pair[1]),
                        remaining[0],
                        canonical_f,
                        canonical_f,
                        1,
                        1,
                        outer_factor,
                        final_factor,
                        dummy,
                    )

    if term.color_source in {"1", "UFO::{}::1"} or term.color_expression == "1":
        input_legs = tuple(leg for leg in range(4) if leg != result_leg)
        return (
            (input_legs[0], input_legs[1]),
            input_legs[2],
            term.color_source,
            "1",
            term.lc_color_normalization_power,
            0,
            (),
            (),
            -1,
        )
    return None


def _normalized_structure_constant_factors(
    expression: str,
) -> tuple[tuple[int, ...], ...]:
    """Return typed f-tensor index words from a normalized color monomial."""

    if expression.count("::f(") != 2:
        return ()
    result: list[tuple[int, ...]] = []
    for arguments in _function_arguments(expression, "::f"):
        indices: list[int] = []
        for argument in arguments:
            dummy = re.search(r"ufo_c_dummy_([0-9]+)_adjoint", argument)
            if dummy is not None:
                indices.append(-int(dummy.group(1)))
                continue
            external = re.search(r"ufo_c_([0-9]+)", argument)
            if external is None:
                return ()
            indices.append(int(external.group(1)))
        if len(indices) != 3:
            return ()
        result.append(tuple(indices))
    return tuple(result)
