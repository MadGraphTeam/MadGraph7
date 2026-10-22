# SPDX-License-Identifier: 0BSD
"""Ordered physical charges on dependency tensors, for loop colour bilinears.

The existing ColorConnection defines typed legs and ordered operations. This
adapter inserts native generator/structure-constant tensors; all contractions,
adjoints, dummy scoping and invariant reduction remain dependency operations.
It is a generation helper, not a correlated numerical loop evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyamplicol.color.connections import ColorConnection, EmitGluon

from .._internal.physics.symbols import symbols
from .color_bilinears import SUNColorSpace
from .exact import require_exact_numbers


def _ports(legs):
    from symbolica.community import tensor as t

    representations = {
        3: t.Representation.cof(3),
        -3: t.Representation.cof(3).dual(),
        8: t.Representation.coad(8),
    }
    return tuple(
        representations[leg.representation](leg.label)
        for leg in legs
        if leg.representation != 1
    )


def apply_native_color_connection(expression: Any, connection: ColorConnection):
    """Apply all-outgoing physical charges with Tr(Ta Tb) = delta(ab)/2.

    Free indices must be the integer leg labels with native SU(3) representations.
    Formal continuation is performed afterward by SUNColorSpace. Singlets have
    no colour index. Emitted legs may be used by later operations, including a
    gluon splitting into a quark pair. No kinematic splitting factor is included.
    """
    from symbolica import E
    from symbolica.community import tensor as t

    if not isinstance(connection, ColorConnection):
        raise TypeError("expected an ordered ColorConnection")
    if not isinstance(expression, t.TensorExpression):
        raise TypeError("expected a dependency TensorExpression")
    require_exact_numbers(expression.to_expression())
    initial = _ports(connection.initial_legs)
    output = _ports(connection.output_legs)
    if expression.to_expression() == 0 and not expression.axes:
        return t.TensorExpression(E("0"), structure=t.TensorStructure(output))
    if set(expression.axes) != set(initial):
        raise ValueError("tensor ports differ from the typed initial colour legs")
    # Integer external labels cannot collide with the inserted symbolic dummies;
    # scope existing dummies first so even user-chosen internal names are safe.
    value = expression.wrap_indices(
        symbols.qualified("loop_color_connection::input"), dummies_only=True
    )
    active = {leg.label: leg.representation for leg in connection.initial_legs}
    generator = t.TensorExpression.color_t(8, 3)
    structure = t.TensorExpression.color_f(8)
    for index, operation in enumerate(connection.emissions):
        dummy = symbols.qualified(f"loop_color_connection::contracted_{index}")
        if isinstance(operation, EmitGluon):
            parent, emitted = operation.emitter_label, operation.emitted_label
            representation = active[parent]
            value = value.rename_indices({parent: dummy})
            if representation == 3:
                charge = generator(emitted, parent, dummy)
            elif representation == -3:
                charge = -generator(emitted, dummy, parent)
            else:
                charge = E("1i") * structure(parent, emitted, dummy)
            value *= charge
            active[emitted] = 8
        else:
            # ColorConnection already validates the remaining SplitGluon variant.
            parent = operation.parent_label
            value = value.rename_indices({parent: dummy}) * generator(
                dummy, operation.quark_label, operation.antiquark_label
            )
            del active[parent]
            active[operation.quark_label] = 3
            active[operation.antiquark_label] = -3
    if set(value.axes) != set(output):
        raise ValueError("connected tensor ports differ from the output colour legs")
    return value


@dataclass(frozen=True, slots=True)
class DirectedColorBilinears:
    """Two rectangular kernels for one ordered pair of connection histories.

    Their numerical virtual observable is B† K_BL L + L† K_LB B. The second
    matrix is the adjoint of K_BL for swapped histories, not in general the
    adjoint of this record's first matrix. Reduction must precede conjugation
    when the numerical executor later assembles the second cross term.
    """

    born_loop: tuple[tuple[Any, ...], ...]
    loop_born: tuple[tuple[Any, ...], ...]


def directed_color_bilinears(
    space: SUNColorSpace,
    born: tuple[Any, ...],
    loop: tuple[Any, ...],
    bra: ColorConnection,
    ket: ColorConnection,
) -> DirectedColorBilinears:
    """Construct formal rectangular weights for both directed cross terms."""
    if bra.initial_legs != ket.initial_legs:
        raise ValueError("connection histories have different initial colour legs")
    if bra.output_legs != ket.output_legs:
        raise ValueError("connection histories have different output colour legs")
    born_bra = tuple(apply_native_color_connection(x, bra) for x in born)
    born_ket = tuple(apply_native_color_connection(x, ket) for x in born)
    loop_bra = tuple(apply_native_color_connection(x, bra) for x in loop)
    loop_ket = tuple(apply_native_color_connection(x, ket) for x in loop)
    return DirectedColorBilinears(
        space.rectangular(born_bra, loop_ket),
        space.rectangular(loop_bra, born_ket),
    )
