# SPDX-License-Identifier: 0BSD
"""Oriented-kernel fusion and component expression lowering."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from itertools import product

from .._internal.physics.canonical_order import CanonicalRenaming, canonical_apply
from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_records import _replace_evaluator_constants, _sequence
from .compiler_tensor_ordering import (
    OrderedComponents,
    identity_ordering_for_materialized_axes,
)
from .contracts import (
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from .tensors import LEVI_CIVITA, levi_civita_components


def _function_arguments(source: str, head_suffix: str) -> tuple[tuple[str, ...], ...]:
    """Extract balanced canonical arguments for one fully qualified function."""

    marker = head_suffix + "("
    result: list[tuple[str, ...]] = []
    cursor = 0
    while True:
        marker_start = source.find(marker, cursor)
        if marker_start < 0:
            break
        open_index = marker_start + len(head_suffix)
        depth = 0
        brace_depth = 0
        argument_start = open_index + 1
        arguments: list[str] = []
        index = open_index
        while index < len(source):
            character = source[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    arguments.append(source[argument_start:index])
                    result.append(tuple(arguments))
                    cursor = index + 1
                    break
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth -= 1
            elif character == "," and depth == 1 and brace_depth == 0:
                arguments.append(source[argument_start:index])
                argument_start = index + 1
            index += 1
        else:
            raise ValueError(f"unbalanced canonical function {head_suffix}")
    return tuple(result)


def _permutation_sign(
    actual: tuple[int, ...],
    canonical: tuple[int, ...],
) -> int:
    if sorted(actual) != sorted(canonical):
        raise ValueError(
            f"color-factor indices {actual} do not match local orientation {canonical}"
        )
    positions = {value: index for index, value in enumerate(canonical)}
    permutation = tuple(positions[value] for value in actual)
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _fuse_oriented_kernels(
    kernels: Sequence[CompiledOrientedKernel],
    *,
    model_name: str,
) -> tuple[CompiledOrientedKernel, ...]:
    model_symbols = symbols.model(model_name)
    groups: dict[tuple[object, ...], list[CompiledOrientedKernel]] = {}
    for kernel in kernels:
        key = (
            kernel.vertex,
            kernel.particles,
            kernel.source_particle_legs,
            kernel.coupling_orders,
            kernel.color_source,
            kernel.color_expression,
            kernel.lc_color_normalization_power,
            kernel.input_ordering_ids,
            kernel.output_ordering_id,
        )
        groups.setdefault(key, []).append(kernel)

    fused: list[CompiledOrientedKernel] = []
    for members in groups.values():
        kind = len(fused)
        first = members[0]
        components = _fused_components(members, kind, model_symbols)
        coupling_expression = "1"
        fused.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=first.term_id,
                vertex=first.vertex,
                particles=first.particles,
                source_particle_legs=first.source_particle_legs,
                component_expressions=components,
                coupling_expression=coupling_expression,
                coupling_orders=first.coupling_orders,
                runtime_parameters=tuple(
                    sorted({f"derived_coupling_{member.term_id}" for member in members})
                ),
                color_source=first.color_source,
                color_expression=first.color_expression,
                lc_color_normalization_power=first.lc_color_normalization_power,
                term_ids=tuple(
                    term_id
                    for member in members
                    for term_id in (member.term_ids or (member.term_id,))
                ),
                input_ordering_ids=first.input_ordering_ids,
                output_ordering_id=first.output_ordering_id,
            )
        )
    return tuple(fused)


def _fused_components(
    members: Sequence[CompiledOrientedKernel],
    kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[str, ...]:
    """The canonical strings of the fused kernel ``kind``'s components:
    ``sum_m component_m * derived_coupling(term_m)`` over the ``members``,
    their kernel symbols renamed to ``kind``, canonicalised
    (``_canonicalize_oriented_kernel_component``).

    Fusions equal up to the members' kinds and term ids (the vertex copies of
    counterterm clone lines, repeated structures) are formed once
    (``_FUSED``) and renamed on the text (``_renamed_fused_texts``): when
    each member's components hold only its own kind's kernel symbols
    (``_single_kind``) and the term ids are in the same numeric order, both
    sums are equal on the canonical symbols of ``canonical_apply``, so the
    canonicalisation is the same up to the renaming."""

    model_name = model_symbols.model_name
    term_ids = tuple(member.term_id for member in members)
    key = None
    if len(set(term_ids)) == len(term_ids) and all(
        _single_kind(component, member.kind, model_symbols)
        for member in members
        for component in member.component_expressions
    ):
        key = (
            model_symbols.namespace,
            tuple(
                tuple(
                    _renamed_kernel_text(
                        component,
                        old_kind=member.kind,
                        new_kind=0,
                        model_symbols=model_symbols,
                    )
                    for component in member.component_expressions
                )
                for member in members
            ),
            tuple(sorted(range(len(term_ids)), key=term_ids.__getitem__)),
        )
        found = _FUSED.get(key)
        if found is not None:
            formed_kind, formed_ids, texts = found
            renamed = _renamed_fused_texts(
                texts,
                old_kind=formed_kind,
                new_kind=kind,
                old_ids=formed_ids,
                new_ids=term_ids,
                model_symbols=model_symbols,
            )
            if renamed is not None:
                return renamed
    remapped_components = [
        tuple(
            _sym.E(
                _renamed_kernel_text(
                    component,
                    old_kind=member.kind,
                    new_kind=kind,
                    model_symbols=model_symbols,
                )
            )
            for component in member.component_expressions
        )
        for member in members
    ]
    texts = tuple(
        _canonicalize_oriented_kernel_component(
            sum(
                (
                    remapped[index]
                    * symbols.derived_coupling(model_name, member.term_id)
                    for member, remapped in zip(
                        members, remapped_components, strict=True
                    )
                ),
                _sym.E("0"),
            )
        ).to_canonical_string()
        for index in range(len(remapped_components[0]))
    )
    if key is not None:
        if len(_FUSED) >= _FUSED_BOUND:
            _FUSED.clear()
        _FUSED[key] = (kind, term_ids, texts)
    return texts


# ``_fused_components``' results by (model namespace, the members' components
# on kind 0, the numeric order of their term ids), with the kind and term ids
# they were formed for. Cleared beyond the bound.
_FUSED: dict[tuple[object, ...], tuple[int, tuple[int, ...], tuple[str, ...]]] = {}
_FUSED_BOUND = 1 << 15

# A derived coupling symbol in a canonical string, with its term id.
_DERIVED_TOKEN = re.compile(r"::derived_coupling_([0-9]+)(?![0-9A-Za-z_])")


def _renamed_fused_texts(
    texts: Sequence[str],
    *,
    old_kind: int,
    new_kind: int,
    old_ids: Sequence[int],
    new_ids: Sequence[int],
    model_symbols: ModelSymbolRegistry,
) -> tuple[str, ...] | None:
    """``texts`` (canonical strings of a fusion formed for kernel
    ``old_kind`` and the term ids ``old_ids``) with the kind and the term ids
    renamed to ``new_kind`` and ``new_ids``, or None when the renaming might
    change an order of the canonical strings.

    A derived coupling's id is renamed in place when the texts hold no other
    derived coupling than the model's of ``old_ids``, no id is a prefix of
    another (old or new), and the ids are in the same lexicographic order:
    two canonical strings then first differ either outside the ids or
    inside two different ids at a digit that compares as the ids do. The
    kind is renamed by ``_renamed_kernel_text``."""

    old_names = [str(term_id) for term_id in old_ids]
    new_names = [str(term_id) for term_id in new_ids]
    for names in (old_names, new_names):
        if any(a != b and b.startswith(a) for a in names for b in names):
            return None
    if any(
        (old_names[i] < old_names[j]) != (new_names[i] < new_names[j])
        for i in range(len(old_names))
        for j in range(len(old_names))
    ):
        return None
    prefix = _derived_token_prefix(model_symbols)
    mapping = dict(zip(old_names, new_names, strict=True))
    renamed: list[str] = []
    for text in texts:
        ids = _DERIVED_TOKEN.findall(text)
        occurrences = text.count("derived_coupling_")
        if (
            len(ids) != occurrences
            or text.count(prefix) != occurrences
            or any(term_id not in mapping for term_id in ids)
        ):
            return None
        text = _DERIVED_TOKEN.sub(
            lambda match: f"::derived_coupling_{mapping[match.group(1)]}", text
        )
        renamed.append(
            _renamed_kernel_text(
                text,
                old_kind=old_kind,
                new_kind=new_kind,
                model_symbols=model_symbols,
            )
        )
    return tuple(renamed)


def _derived_token_prefix(model_symbols: ModelSymbolRegistry) -> str:
    """The canonical-string text of the model's derived couplings up to their
    term id."""

    found = _DERIVED_TOKEN_PREFIXES.get(model_symbols.namespace)
    if found is None:
        text = symbols.derived_coupling(
            model_symbols.model_name, 0
        ).to_canonical_string()
        if not text.endswith("::derived_coupling_0"):
            raise ValueError(f"unexpected canonical derived coupling {text}")
        found = _DERIVED_TOKEN_PREFIXES[model_symbols.namespace] = text[:-1]
    return found


_DERIVED_TOKEN_PREFIXES: dict[str, str] = {}


def _canonicalize_oriented_kernel_component(
    expression: _sym.Expression,
) -> _sym.Expression:
    """Cancel numeric sums, factor couplings, and group primitive inputs.

    The factorisation and the Horner scheme depend on Symbolica's symbol
    order, so they run on canonical symbols (``canonical_order``): the same
    kernel in every process history."""

    if len(_CANONICAL_COMPONENTS) >= _CANONICAL_COMPONENTS_BOUND:
        _CANONICAL_COMPONENTS.clear()
    return canonical_apply(
        _canonical_component,
        expression,
        memo=_CANONICAL_COMPONENTS,
    )


def _canonical_component(canonical: _sym.Expression) -> _sym.Expression:
    return canonical.expand_num().collect_factors().collect_horner()


# ``_canonicalize_oriented_kernel_component``'s results on canonical symbols
# by the canonical text of their argument (``canonical_apply``): the vertex
# copies of counterterm clone lines, and every kernel of a repeated
# structure, share one Horner scheme. Cleared beyond the bound.
_CANONICAL_COMPONENTS: dict[str, _sym.Expression] = {}
_CANONICAL_COMPONENTS_BOUND = 1 << 16


# A kernel's input component and momentum symbols, by name
# (``ModelSymbolRegistry.kernel_component``, ``kernel_momentum``).
_KERNEL_SYMBOL = re.compile(r"kernel_([0-9]+)_(left|right)_(momentum_)?([0-9]+)")


def _remap_kernel_symbols(
    expression: _sym.Expression,
    *,
    old_kind: int,
    new_kind: int,
    model_symbols: ModelSymbolRegistry,
    swap_sides: bool = False,
) -> _sym.Expression:
    """``expression`` with the input symbols of kernel ``old_kind`` renamed
    to those of ``new_kind`` (the sides exchanged with ``swap_sides``). The
    symbols are found by name among the expression's variables, so neither a
    canonical string nor a pattern per kind is formed."""

    kernel_symbols = set()
    for symbol in expression.get_all_symbols(False):
        found = _KERNEL_SYMBOL.fullmatch(symbol.get_name().rsplit("::", 1)[-1])
        if found is not None and int(found.group(1)) == old_kind:
            kernel_symbols.add(
                (found.group(2), found.group(3) is not None, int(found.group(4)))
            )
    replacements: list[_sym.Replacement] = []
    for side, is_momentum, index in sorted(kernel_symbols):
        target_side = ("right" if side == "left" else "left") if swap_sides else side
        if is_momentum:
            replacements.append(
                _sym.Replacement(
                    model_symbols.kernel_momentum(old_kind, side, index),
                    model_symbols.kernel_momentum(new_kind, target_side, index),
                )
            )
        else:
            replacements.append(
                _sym.Replacement(
                    model_symbols.kernel_component(old_kind, side, index),
                    model_symbols.kernel_component(new_kind, target_side, index),
                )
            )
    return expression.replace_multiple(replacements) if replacements else expression


# A kernel's input component or momentum symbol in a canonical string, with
# its kind (``ModelSymbolRegistry.kernel_component``, ``kernel_momentum``).
_KERNEL_TOKEN = re.compile(
    r"::kernel_([0-9]+)_(?:left|right)_(?:momentum_)?[0-9]+(?![0-9A-Za-z_])"
)


def _kernel_token_prefix(model_symbols: ModelSymbolRegistry) -> str:
    """The canonical-string text of the model's kernel symbols up to their
    kind (``<namespace>::compiler::{}::kernel_``)."""

    found = _KERNEL_TOKEN_PREFIXES.get(model_symbols.namespace)
    if found is None:
        text = model_symbols.kernel_component(0, "left", 0).to_canonical_string()
        tail = "0_left_0"
        if not text.endswith(f"::kernel_{tail}"):
            raise ValueError(f"unexpected canonical kernel symbol {text}")
        found = _KERNEL_TOKEN_PREFIXES[model_symbols.namespace] = text[: -len(tail)]
    return found


_KERNEL_TOKEN_PREFIXES: dict[str, str] = {}


def _single_kind(text: str, kind: int, model_symbols: ModelSymbolRegistry) -> bool:
    """Whether every kernel symbol of the canonical string ``text`` is an
    input component or momentum symbol of kernel ``kind`` in the model's
    namespace (``_renamed_kernel_text``'s condition)."""

    prefix = _kernel_token_prefix(model_symbols)
    kinds = _KERNEL_TOKEN.findall(text)
    occurrences = text.count("::kernel_")
    return (
        len(kinds) == occurrences
        and text.count(f"{prefix}{kind}_") == occurrences
        and all(found == str(kind) for found in kinds)
    )


def _canonical_product_text(
    text: str,
    factor: str,
    *,
    kind: int,
    model_symbols: ModelSymbolRegistry,
) -> str:
    """``_canonicalize_oriented_kernel_component(E(text) * E(factor))
    .to_canonical_string()`` for a canonical string ``text`` of kernel
    ``kind`` and a ``factor`` without kernel symbols.

    Formed once per (``text`` on kind 0, ``factor``) when every kernel symbol
    of ``text`` has the one kind (``_single_kind``): the natural order of the
    canonical names then does not depend on the kind, so the product is the
    same on the canonical symbols of ``canonical_apply`` and its result is
    the formed one with the kind renamed (``_renamed_kernel_text``)."""

    key = None
    if "::kernel_" not in factor and _single_kind(text, kind, model_symbols):
        key = (
            model_symbols.namespace,
            _renamed_kernel_text(
                text, old_kind=kind, new_kind=0, model_symbols=model_symbols
            ),
            factor,
        )
        found = _CANONICAL_PRODUCTS.get(key)
        if found is not None:
            formed, result = found
            return _renamed_kernel_text(
                result, old_kind=formed, new_kind=kind, model_symbols=model_symbols
            )
    result = _canonicalize_oriented_kernel_component(
        _sym.E(text) * _sym.E(factor)
    ).to_canonical_string()
    if key is not None:
        if len(_CANONICAL_PRODUCTS) >= _CANONICAL_PRODUCTS_BOUND:
            _CANONICAL_PRODUCTS.clear()
        _CANONICAL_PRODUCTS[key] = (kind, result)
    return result


# ``_canonical_product_text``'s results by (model namespace, text on kind 0,
# factor), with the kind they were formed for. Cleared beyond the bound.
_CANONICAL_PRODUCTS: dict[tuple[str, str, str], tuple[int, str]] = {}
_CANONICAL_PRODUCTS_BOUND = 1 << 16


def _renamed_kernel_text(
    text: str,
    *,
    old_kind: int,
    new_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> str:
    """``_remap_kernel_symbols(E(text), old_kind=..., new_kind=...)
    .to_canonical_string()`` of a canonical string ``text``.

    Formed on the text when every kernel symbol it holds is an input
    component or momentum symbol of kernel ``old_kind`` in the model's
    namespace. Symbolica's canonical string sorts the arguments, factors and
    terms of every node by their own canonical strings; two such strings
    first differ outside the kind of a kernel symbol (all of them have the
    same kind, so a difference inside one would be one between two kinds),
    so writing ``new_kind`` for ``old_kind`` in place keeps every order and
    gives the renamed expression's canonical string. Any other text goes
    through Symbolica."""

    if old_kind == new_kind:
        return text
    if _single_kind(text, old_kind, model_symbols):
        prefix = _kernel_token_prefix(model_symbols)
        return text.replace(f"{prefix}{old_kind}_", f"{prefix}{new_kind}_")
    return _remap_kernel_symbols(
        _sym.E(text),
        old_kind=old_kind,
        new_kind=new_kind,
        model_symbols=model_symbols,
    ).to_canonical_string()


def _replace_expression_symbols(
    expression: _sym.Expression,
    substitutions: Mapping[_sym.Expression, _sym.Expression],
) -> _sym.Expression:
    result = expression
    for source, target in substitutions.items():
        result = result.replace(source, target)
    return result


def _oriented_component_expressions(
    term: CompiledVertexTerm,
    particle_by_name: Mapping[str, CompiledParticleRecord],
    *,
    left_leg: int,
    right_leg: int,
    result_leg: int,
    kind: int,
    model_symbols: ModelSymbolRegistry,
    use_transverse_massless_yang_mills: bool = False,
) -> OrderedComponents:
    library = DenseTensors()
    expression = _sym.E(term.lorentz_expression)
    particles = tuple(particle_by_name[name] for name in term.particles)
    left_symbols = _component_symbols(
        kind,
        "left",
        particles[left_leg].spin,
        model_symbols=model_symbols,
    )
    right_symbols = _component_symbols(
        kind,
        "right",
        particles[right_leg].spin,
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="left",
        spin=particles[left_leg].spin,
        leg=left_leg + 1,
        components=left_symbols,
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="right",
        spin=particles[right_leg].spin,
        leg=right_leg + 1,
        components=right_symbols,
        model_symbols=model_symbols,
    )
    left_momentum = tuple(
        model_symbols.kernel_momentum(kind, "left", component) for component in range(4)
    )
    right_momentum = tuple(
        model_symbols.kernel_momentum(kind, "right", component)
        for component in range(4)
    )
    result_momentum = tuple(
        -(left_momentum[component] + right_momentum[component])
        for component in range(4)
    )
    momentum_by_leg = {
        left_leg: left_momentum,
        right_leg: right_momentum,
        result_leg: result_momentum,
    }
    minkowski = _sym.Representation.mink(4)
    for leg, momentum in momentum_by_leg.items():
        library.register_dense(
            model_symbols.ufo_momentum_tensor_name(leg + 1), (minkowski,), momentum
        )
    ordered_result = _execute_dense_network(
        expression,
        library,
        _spin_axis_labels(particles[result_leg].spin, result_leg + 1),
    )
    result_components = ordered_result.values
    expected_dimension = _spin_dimension(particles[result_leg].spin)
    if len(result_components) != expected_dimension:
        raise ValueError(
            f"oriented kernel {term.vertex}/{term.id} produced "
            f"{len(result_components)} "
            f"components for spin {particles[result_leg].spin}, expected "
            f"{expected_dimension}"
        )
    components = tuple(
        _replace_evaluator_constants(_as_expression(component))
        for component in result_components
    )
    if all(particle.spin == 3 for particle in particles):
        compact = _compact_yang_mills_three_vector_components(
            left_leg=left_leg,
            right_leg=right_leg,
            result_leg=result_leg,
            left_components=left_symbols,
            right_components=right_symbols,
            momentum_by_leg=momentum_by_leg,
        )
        scale = _equivalent_component_scale(components, compact)
        if scale is not None:
            if use_transverse_massless_yang_mills:
                compact = _transverse_yang_mills_three_vector_components(
                    left_components=left_symbols,
                    right_components=right_symbols,
                    left_momentum=left_momentum,
                    right_momentum=right_momentum,
                )
            components = tuple(scale * component for component in compact)
    return OrderedComponents(
        ordering=ordered_result.ordering,
        values=tuple(component.to_canonical_string() for component in components),
    )


def _compact_yang_mills_three_vector_components(
    *,
    left_leg: int,
    right_leg: int,
    result_leg: int,
    left_components: Sequence[_sym.Expression],
    right_components: Sequence[_sym.Expression],
    momentum_by_leg: Mapping[int, Sequence[_sym.Expression]],
) -> tuple[_sym.Expression, ...]:
    """Return the compact canonical Yang-Mills three-vector contraction.

    The caller retains this form only after proving algebraic equivalence to the
    fully materialized spenso tensor. This makes the optimization independent
    of UFO Lorentz names and source-expression layout.
    """

    components_by_leg = {
        left_leg: tuple(left_components),
        right_leg: tuple(right_components),
    }
    terms = (
        ((0, 1), 0, 1, 2),
        ((0, 2), 2, 0, 1),
        ((1, 2), 1, 2, 0),
    )
    outputs: list[_sym.Expression] = []
    for component in range(4):
        output = _sym.E("0")
        for (
            metric_legs,
            positive_momentum_leg,
            negative_momentum_leg,
            vector_leg,
        ) in terms:
            momentum = tuple(
                momentum_by_leg[positive_momentum_leg][index]
                - momentum_by_leg[negative_momentum_leg][index]
                for index in range(4)
            )
            if result_leg == vector_leg:
                output += momentum[component] * _minkowski_dot(
                    components_by_leg[metric_legs[0]],
                    components_by_leg[metric_legs[1]],
                )
                continue
            other_metric_leg = (
                metric_legs[1] if metric_legs[0] == result_leg else metric_legs[0]
            )
            output += components_by_leg[other_metric_leg][component] * _minkowski_dot(
                momentum,
                components_by_leg[vector_leg],
            )
        outputs.append(output)
    return tuple(outputs)


def _transverse_yang_mills_three_vector_components(
    *,
    left_components: Sequence[_sym.Expression],
    right_components: Sequence[_sym.Expression],
    left_momentum: Sequence[_sym.Expression],
    right_momentum: Sequence[_sym.Expression],
) -> tuple[_sym.Expression, ...]:
    """Return the transverse Berends-Giele massless gauge current.

    The full Yang-Mills contraction differs only by terms proportional to each
    parent current's self-momentum contraction. AmpliCol removes those terms
    for massless adjoint gauge currents. Applying the same tensor-derived
    reduction preserves its fixed-width recursion convention without relying
    on model-specific particle or Lorentz names.
    """

    left = tuple(left_components)
    right = tuple(right_components)
    left_p = tuple(left_momentum)
    right_p = tuple(right_momentum)
    dot = _minkowski_dot(left, right)
    left_dot_right_p = _minkowski_dot(left, right_p)
    right_dot_left_p = _minkowski_dot(right, left_p)
    return tuple(
        dot * (left_p[index] - right_p[index])
        + 2 * (left_dot_right_p * right[index] - right_dot_left_p * left[index])
        for index in range(4)
    )


def _equivalent_component_scale(
    materialized: Sequence[_sym.Expression],
    compact: Sequence[_sym.Expression],
) -> _sym.Expression | None:
    if len(materialized) != len(compact) or not materialized:
        return None
    if any(
        frozenset(
            symbol.to_canonical_string() for symbol in dense.get_all_symbols(False)
        )
        != frozenset(
            symbol.to_canonical_string() for symbol in candidate.get_all_symbols(False)
        )
        for dense, candidate in zip(materialized, compact, strict=True)
    ):
        return None
    for scale in (_sym.E("1"), _sym.E("-1")):
        if all(
            (dense - scale * candidate).expand().to_canonical_string() == "0"
            for dense, candidate in zip(materialized, compact, strict=True)
        ):
            return scale
    # The compact Yang-Mills convention is only certified for an exact sign.
    # Retaining the fully materialized tensor is always correct for every other
    # normalization.  In particular, do not derive a general ratio here:
    # Symbolica's polynomial cancellation aborts the process for float atoms.
    return None


def _is_compile_time_zero_parameter(
    name: str,
    parameters: Mapping[str, CompiledParameterRecord],
) -> bool:
    if name.upper() == "ZERO":
        return True
    parameter = parameters.get(name)
    if parameter is None or parameter.nature != "internal":
        return False
    expression = _sym.E(parameter.resolved_expression)
    if expression.get_all_symbols(False):
        return False
    try:
        return complex(expression.evaluate({})) == 0.0
    except (TypeError, ValueError):
        return False


def _is_single_structure_constant(expression: str) -> bool:
    return (
        expression.count("::f(") == 1
        and len(_function_arguments(expression, "::f")) == 1
        and "::t(" not in expression.lower()
        and "::d(" not in expression.lower()
    )


def _minkowski_dot(
    left: Sequence[_sym.Expression],
    right: Sequence[_sym.Expression],
) -> _sym.Expression:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Minkowski dot products require four components")
    return left[0] * right[0] - sum(
        (left[index] * right[index] for index in range(1, 4)),
        _sym.E("0"),
    )


def _input_tensor_expression(
    library: DenseTensors | _sym.TensorLibrary,
    *,
    kind: int,
    side: str,
    spin: int,
    leg: int,
    components: Sequence[_sym.Expression],
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    representations = _spin_representations(spin)
    if not representations:
        if len(components) != 1:
            raise ValueError("scalar current must have exactly one component")
        return components[0]
    name = model_symbols.kernel_tensor_name(kind, side)
    register_dense(library, name, representations, components)
    slots = _spin_slots(spin, leg)
    return _sym.TensorName(name)(*slots).to_expression()


class DenseTensors:
    """The dense tensors of one tensor network, by qualified head name.

    ``_execute_dense_network`` registers them in a Spenso library on
    canonical symbols."""

    def __init__(self) -> None:
        self.tensors: list[tuple[str, tuple[object, ...], tuple[object, ...]]] = []

    def register_dense(
        self,
        name: str,
        representations: Sequence[object],
        components: Sequence[object],
    ) -> None:
        # The head is a Spenso tensor from here on, as on a library.
        _sym.TensorName(name)
        self.tensors.append((name, tuple(representations), tuple(components)))


def register_dense(
    library: DenseTensors | _sym.TensorLibrary,
    name: str,
    representations: Sequence[object],
    components: Sequence[object],
) -> None:
    """Register a dense tensor in ``DenseTensors`` or directly in a Spenso
    library (callers that execute their own network)."""

    if isinstance(library, DenseTensors):
        library.register_dense(name, representations, components)
    else:
        library.register(
            _sym.Tensor.dense(_sym.TensorName(name)(*representations), components)
        )


# Symbolica's and Spenso's own symbols: registered when they are imported,
# with the same ids in every process.
_BUILTIN_NAMESPACES = frozenset({"spenso", "symbolica"})


def _execute_dense_network(
    expression: _sym.Expression,
    tensors: DenseTensors,
    axis_labels: Sequence[str],
) -> OrderedComponents:
    """Execute the Spenso network of ``expression`` with ``tensors`` on
    canonical symbols and return its components in ``axis_labels`` order on
    the original symbols.

    Spenso contracts in an order that follows Symbolica's normal form, i.e.
    the symbol ids (``_internal.physics.canonical_order``), and the sums it
    accumulates normalise into differently nested (equal) expressions: the
    same scalar_gravity three-graviton vertex gave other component
    structures, and after the Horner scheme other kernels, when the index
    symbols had been created in another order. Every symbol of the network
    (index symbols, tensor heads, components, parameters) but Symbolica's and
    Spenso's own is renamed to canonical symbols first; a symbol with other
    attributes keeps its name (none in current models)."""

    _sym._ensure_symbolica()
    every = {
        symbol.to_canonical_string(): symbol
        for symbol in expression.get_all_symbols(True)
    }
    dense = list(tensors.tensors)
    if any(symbol.get_name() == LEVI_CIVITA for symbol in every.values()) and all(
        name != LEVI_CIVITA for name, _r, _c in dense
    ):
        # UFO ``Epsilon`` (``tensors.LEVI_CIVITA``): a dense tensor like the
        # inputs, so the network contracts it on canonical symbols too.
        minkowski = _sym.Representation.mink(4)
        dense.append((LEVI_CIVITA, (minkowski,) * 4, levi_civita_components()))
    heads = []
    for name, _representations, components in dense:
        head = symbols.qualified(name)
        heads.append(head)
        every[head.to_canonical_string()] = head
        for component in components:
            for symbol in _network_component(component).get_all_symbols(True):
                every[symbol.to_canonical_string()] = symbol
    renamed, tensor_heads = [], []
    for token, symbol in every.items():
        if token.split("::", 1)[0] in _BUILTIN_NAMESPACES:
            continue
        tags, attributes = symbol.get_tags(), symbol.get_attributes()
        if not tags and not attributes:
            renamed.append(symbol)
        elif tags == ["spenso::tensor"] and not attributes:
            renamed.append(symbol)
            tensor_heads.append(symbol)
    renaming = CanonicalRenaming(renamed, tensor_heads=tensor_heads)
    library = _sym.TensorLibrary.hep_lib_atom()
    for head, (_name, representations, components) in zip(heads, dense, strict=True):
        library.register(
            _sym.Tensor.dense(
                _sym.TensorName(renaming.name(head))(*representations),
                [
                    renaming.canonical(_network_component(component))
                    for component in components
                ],
            )
        )
    canonical = renaming.canonical(expression)
    # Plain expressions already carry the correct empty scalar interface.
    network = _sym.TensorNetwork(
        _sym.as_tensor(canonical) if axis_labels else canonical, library
    )
    network.execute(library=library)
    ordered = _ordered_dense_tensor_components(
        network.result_tensor(library),
        axis_labels,
        restore=renaming.from_canonical,
    )
    return OrderedComponents(
        ordering=ordered.ordering,
        values=tuple(
            renaming.restore(_as_expression(value)) for value in ordered.values
        ),
    )


def _spin_representations(spin: int) -> tuple[_sym.Representation, ...]:
    minkowski = _sym.Representation.mink(4)
    if spin in {-1, 1}:
        return ()
    if spin == 2:
        return (_sym.Representation.bis(4),)
    if spin == 3:
        return (minkowski,)
    if spin == 5:
        return (minkowski, minkowski)
    raise ValueError(f"unsupported UFO spin code {spin}")


def _spin_slots(spin: int, leg: int):
    representations = _spin_representations(spin)
    if spin == 2:
        return (representations[0](f"ufo_s_1_{leg}"),)
    if spin == 3:
        return (representations[0](f"ufo_l_1_{leg}"),)
    if spin == 5:
        return (
            representations[0](f"ufo_l_1_{leg}"),
            representations[1](f"ufo_l_2_{leg}"),
        )
    return ()


def _spin_axis_labels(spin: int, leg: int) -> tuple[str, ...]:
    if spin == 2:
        return (f"ufo_s_1_{leg}",)
    if spin == 3:
        return (f"ufo_l_1_{leg}",)
    if spin == 5:
        return (f"ufo_l_1_{leg}", f"ufo_l_2_{leg}")
    return ()


def _network_component(value: object) -> _sym.Expression:
    """A dense tensor component as an expression (an integer as a number)."""

    if isinstance(value, int) and not isinstance(value, bool):
        return _sym.Expression.num(value)
    return _as_expression(value)


def _ordered_dense_tensor_components(
    tensor: object,
    expected_axis_labels: Sequence[str],
    *,
    restore: Callable[[str], str] | None = None,
) -> OrderedComponents:
    """Flatten a dense tensor in explicit physical-index order (``restore``:
    the original names of a canonical network's index symbols)."""

    tensor = tensor.to_dense()
    expected = tuple(str(label) for label in expected_axis_labels)
    if not expected:
        if len(tensor) != 1:
            raise ValueError("scalar tensor result unexpectedly has open indices")
        return OrderedComponents(
            ordering=identity_ordering_for_materialized_axes((), ()),
            values=tuple(tensor[index] for index in range(len(tensor))),
        )

    # Component coordinates follow the tensor's logical axes, not its canonical
    # structure signature, which can put the same axes in a different order.
    arguments = tuple(slot.to_expression() for slot in tensor.axes)
    actual: list[str] = []
    for argument in arguments:
        source = argument.to_canonical_string()
        if restore is not None:
            source = restore(source)
        labels = re.findall(r"ufo_[sl]_[0-9]+_[0-9]+", source)
        if len(labels) != 1:
            raise ValueError(
                "tensor result has an unrecognized physical index: " + source
            )
        actual.append(labels[0])
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError(
            "tensor result index labels do not match the requested physical order: "
            f"actual={actual}, expected={list(expected)}"
        )

    positions = tuple(actual.index(label) for label in expected)
    shape = tuple(int(dimension) for dimension in tensor.shape)
    if len(shape) != len(actual) or math.prod(shape) != len(tensor):
        raise ValueError(
            "tensor result does not cover the complete Cartesian component grid"
        )
    by_coordinates: dict[tuple[int, ...], object] = {}
    for flat_index, coordinates in enumerate(product(*(range(size) for size in shape))):
        key = tuple(int(coordinates[position]) for position in positions)
        by_coordinates[key] = tensor[flat_index]
    ordered_shape = tuple(shape[position] for position in positions)
    canonical_coordinates = product(*(range(size) for size in ordered_shape))
    ordering = identity_ordering_for_materialized_axes(
        expected,
        ordered_shape,
    )
    return OrderedComponents(
        ordering=ordering,
        values=tuple(by_coordinates[key] for key in canonical_coordinates),
    )


def _spin_dimension(spin: int) -> int:
    return {-1: 1, 1: 1, 2: 4, 3: 4, 5: 16}[spin]


def _component_symbols(
    kind: int,
    side: str,
    spin: int,
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[_sym.Expression, ...]:
    return tuple(
        model_symbols.kernel_component(kind, side, component)
        for component in range(_spin_dimension(spin))
    )


def _lc_color_normalization_power(source: str) -> int:
    """Count normalized non-Abelian tensors in one UFO color monomial."""

    expression = _sym.E(source)
    return sum(
        len(list(expression.match(_sym.E(pattern))))
        for pattern in (
            "UFO::T(a_,b_,c_)",
            "UFO::f(a_,b_,c_)",
            "UFO::d(a_,b_,c_)",
        )
    )


def _as_expression(value: object) -> _sym.Expression:
    _sym._ensure_symbolica()
    if isinstance(value, _sym.Expression):
        return value
    raise TypeError(
        f"spenso kernel component is not an Expression: {type(value).__name__}"
    )


def cast_tuple3(value: object) -> tuple[str, str, str]:
    values = tuple(str(item) for item in _sequence(value))
    if len(values) != 3:
        raise ValueError("oriented kernel particles must have length three")
    return values[0], values[1], values[2]
