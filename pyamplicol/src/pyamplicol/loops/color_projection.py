# SPDX-License-Identifier: 0BSD
"""Structural fixed-Nf colour projection for loop bilinears.

Symbolica owns exact combination and Laurent decomposition. This cold-path
projection does not provide or certify a model's formal-Nc continuation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Any

from pyamplicol.config.models import ColorAccuracy

from .exact import require_exact_numbers


@dataclass(frozen=True, slots=True)
class StructuralColorProjection:
    """One squared-order sector and compatible observable/source family.

    The rows cover every helicity, source, numerator and epsilon component
    sharing this structural leading degree. Build once before numerical input
    substitution. A vanishing row never chooses its own lower leading degree.
    External spin/colour averages must be applied after projection at Nc=3.
    """

    nc: Any
    leading_degree: int | None
    rows: tuple[tuple[tuple[int, Any], ...], ...]

    def project(self, accuracy: ColorAccuracy | str) -> tuple[Any, ...]:
        """Return exact expressions; NLC includes odd suppression powers."""
        from symbolica import E

        accuracy = ColorAccuracy(accuracy)
        depth = {ColorAccuracy.LC: 0, ColorAccuracy.NLC: 2}.get(accuracy)
        return tuple(
            sum(
                (
                    coefficient * self.nc**degree
                    for degree, coefficient in row
                    if depth is None
                    or self.leading_degree - depth <= degree <= self.leading_degree
                ),
                E("0"),
            )
            for row in self.rows
        )


def prepare_color_projection(
    nc: Any,
    bilinears: Sequence[Iterable[Any]],
    *,
    substitutions: Sequence[tuple[Any, Any]] = (),
) -> StructuralColorProjection:
    """Combine contributions to each bilinear before choosing a common degree.

    Each row contains contributions to one symbolic bilinear coefficient,
    including any physical, counterterm and rational pieces that contribute to
    it. Born and loop bases may be rectangular; their contraction belongs to the
    caller. ``substitutions`` are explicit semantic colour identities supplied
    by a certified model, never guesses inferred from numeric constants. Other
    symbols, including Nf, have degree zero. Functions or non-Laurent rational
    dependence on Nc fail instead of being assigned an incorrect degree.

    This is a loop-only definition: existing tree colour approximations are
    unchanged. Generic full-colour models need not use this formal projector.
    """
    from symbolica import AtomType, E, Expression

    if not isinstance(nc, Expression) or nc.get_type() != AtomType.Var:
        raise TypeError("Nc must be an exact symbolic variable")
    identities = {}
    for name, replacement in substitutions:
        if not isinstance(name, Expression) or name.get_type() != AtomType.Var:
            raise TypeError("colour identities must replace named semantic invariants")
        if not isinstance(replacement, Expression):
            raise TypeError("colour identities require exact symbolic expressions")
        require_exact_numbers(replacement)
        if name == nc:
            raise ValueError("Nc cannot be numerically substituted before projection")
        if name in identities and identities[name] != replacement:
            raise ValueError("conflicting definitions of a colour invariant")
        identities[name] = replacement
    dependencies = {
        name: set(replacement.get_all_symbols()) & identities.keys()
        for name, replacement in identities.items()
    }
    resolved = {}
    for name in TopologicalSorter(dependencies).static_order():
        replacement = identities[name]
        for dependency in dependencies[name]:
            replacement = replacement.replace(dependency, resolved[dependency])
        resolved[name] = replacement

    rows = []
    leading_degree = None
    for contributions in bilinears:
        expression = E("0")
        for contribution in contributions:
            if not isinstance(contribution, Expression):
                raise TypeError(
                    "bilinear contributions require exact symbolic expressions"
                )
            require_exact_numbers(contribution)
            expression += contribution
        for name, replacement in resolved.items():
            expression = expression.replace(name, replacement)
        # together() removes exact rational cancellations before degree selection.
        expression = expression.together().expand()
        row = []
        for monomial, coefficient in expression.coefficient_list(nc):
            if coefficient == 0:
                continue
            if nc in coefficient.get_all_symbols():
                raise ValueError(
                    "colour bilinear is not a finite Laurent polynomial in Nc"
                )
            power = monomial.to_rational_polynomial(vars=[nc])
            degree = power.numerator().degree(nc) - power.denominator().degree(nc)
            if monomial != nc**degree:
                raise ValueError("colour bilinear has an unsupported Nc exponent")
            row.append((degree, coefficient))
            leading_degree = (
                degree if leading_degree is None else max(leading_degree, degree)
            )
        rows.append(tuple(sorted(row)))
    return StructuralColorProjection(nc, leading_degree, tuple(rows))
