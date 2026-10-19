# SPDX-License-Identifier: 0BSD
"""Explicit semantic colour views of paired, exact source scalar bindings.

This preparation boundary does not certify an SU(Nc) model continuation. In
particular, fixed-SU3 effective tensors need their own model-supplied extension.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .physical_color import split_color_grades
from .scalar_bindings import ScalarBindingPlan, prepare_scalar_bindings
from .scalar_frontend import ScalarExpression


@dataclass(frozen=True, slots=True)
class SourceColorScalarPlan:
    """Nc-independent native outputs, indexed by grade and original output.

    Each grade carries the same original active diagnostics. Exact zero
    coefficients remain zero and must not determine a physical leading degree.
    A wholly zero value catalogue retains grade zero for diagnostic evaluation.
    """

    nc: Any
    formal: ScalarBindingPlan
    coefficients: ScalarBindingPlan
    grades: tuple[int, ...]
    columns: tuple[tuple[int, ...], ...]


def prepare_source_color_scalars(
    raw,
    outputs,
    *,
    nc,
    parameter_identities,
    program=None,
    fixed_parameters=None,
    anonymous_definitions=None,
):
    """Retain explicit named invariants before recursive scalar substitution.

    At Nc=3 each declared value must reproduce the original source parameter
    identically, including any ordinary external inputs. Mutable external
    invariants are rejected by scalar preparation unless explicitly fixed.
    Diagnostics and helper ownership are preserved, never algebraically
    interpreted as colour powers or dropped when a value coefficient vanishes.
    """
    from symbolica import AtomType, E, Expression

    if not isinstance(nc, Expression) or nc.get_type() != AtomType.Var:
        raise TypeError("Nc must be an exact symbolic variable")
    if not isinstance(parameter_identities, Mapping):
        raise TypeError("semantic colour identities require a mapping")
    identities = dict(parameter_identities)
    formal = prepare_scalar_bindings(
        raw,
        outputs,
        semantic_parameters=identities,
        anonymous_definitions=anonymous_definitions,
        program=program,
        fixed_parameters=fixed_parameters,
    )
    if identities:
        original = prepare_scalar_bindings(
            raw,
            tuple((name, 0) for name in identities),
            program=formal.program,
            fixed_parameters=fixed_parameters,
        )
        # This establishes compatibility at the physical colour value, not a
        # proof of the supplied continuation away from that value.
        for name, expression in zip(identities, original.expressions, strict=True):
            available = {
                nc,
                *formal.input_symbols,
                *expression.value.get_all_symbols(False),
            }
            unknown = set(identities[name].get_all_symbols(False)) - available
            if unknown:
                raise ValueError(
                    f"semantic colour identity for {name!r} has unbound symbols: "
                    + ", ".join(map(str, sorted(unknown, key=str)))
                )
            if (
                expression.value - identities[name].replace(nc, 3)
            ).together().expand() != 0:
                raise ValueError(
                    f"semantic colour identity for {name!r} differs from source at Nc=3"
                )
    for output, expression in zip(formal.outputs, formal.expressions, strict=True):
        if nc in expression.diagnostic.get_all_symbols():
            raise NotImplementedError(
                f"scalar {output!r} has an Nc-dependent active diagnostic"
            )
    try:
        graded = split_color_grades((part.value for part in formal.expressions), nc)
    except (ValueError, NotImplementedError) as error:
        raise NotImplementedError(
            f"source scalar outputs {formal.outputs!r} cannot be graded: {error}"
        ) from error
    if not graded:
        graded = ((0, (E("0"),) * len(formal.outputs)),)
    coefficients, names, columns = [], [], []
    for degree, values in graded:
        columns.append(tuple(range(len(coefficients), len(coefficients) + len(values))))
        for (name, power), original, value in zip(
            formal.outputs, formal.expressions, values, strict=True
        ):
            coefficients.append(ScalarExpression(value, original.diagnostic))
            names.append((f"{name}[Nc^{degree}]", power))
    return SourceColorScalarPlan(
        nc,
        formal,
        replace(formal, outputs=tuple(names), expressions=tuple(coefficients)),
        tuple(degree for degree, _ in graded),
        tuple(columns),
    )
