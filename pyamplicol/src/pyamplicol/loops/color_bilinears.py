# SPDX-License-Identifier: 0BSD
"""Dependency-backed rectangular color overlaps with formal SU(Nc) dimensions.

These cold-path helpers preserve native tensors before component evaluation.
They provide neither a model's dimensional-continuation certificate nor a
projection policy. Counterterm scalar invariants must remain formal separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .exact import require_exact_numbers


def external_color_tensor(graph: Any, expression: Any) -> Any:
    """Label free native color ports by physical external leg, without evaluation."""
    from symbolica.community import hepkit as h

    indices = {axis.index for axis in expression.axes}
    labels = {}
    for edge in graph.native.external_edges:
        half = next(x for x in graph.native.half_edges if x.edge.id == edge.id)
        index = h.Symbols.half_edge()(half.id, 1)
        if index in indices:
            labels[index] = f"physical_color_leg_{edge.external_index}"
    if set(labels) != indices:
        raise ValueError("color tensor has free ports not assigned to external legs")
    return expression.rename_indices(labels)


@dataclass(frozen=True, slots=True)
class SUNColorSpace:
    """Explicit fundamental/adjoint SU(Nc) continuation of native SU(3) tensors.

    Only representation dimensions are lifted. Numeric scalar coefficients are
    untouched: the caller must supply a valid continuation of the model's color
    tensors and any scalar CA/CF/TR dependence. This class never infers an
    invariant from a numerical constant such as 3 or 4/3.
    """

    nc: Any
    _adjoint_dimension: Any = field(init=False, repr=False)

    def __post_init__(self):
        from symbolica import AtomType, Expression, S

        if not isinstance(self.nc, Expression) or self.nc.get_type() != AtomType.Var:
            raise TypeError("formal Nc must be a symbolic variable")
        object.__setattr__(
            self,
            "_adjoint_dimension",
            S(f"{self.nc.get_name()}__adjoint_dimension", is_real=True),
        )

    def lift(self, expression: Any) -> Any:
        """Use dependency port patterns for physical representation dimensions."""
        from symbolica import S
        from symbolica.community import tensor as t

        value = expression.to_expression()
        require_exact_numbers(value)
        index = S("loop_color_bilinear::index_")
        for old, new in (
            (t.Representation.cof(3), t.Representation.cof(self.nc)),
            (t.Representation.cof(3).dual(), t.Representation.cof(self.nc).dual()),
            (t.Representation.coad(8), t.Representation.coad(self._adjoint_dimension)),
        ):
            value = value.replace(
                t.PortPattern.exact(old, index), t.PortPattern.exact(new, index)
            )
            # Compact traces/chains and representation invariants carry an
            # unindexed channel too. Lifting ports alone leaves those at SU(3).
            value = value.replace(old.to_expression(), new.to_expression())
        return t.TensorExpression(value)

    def overlap(self, left: Any, right: Any) -> Any:
        """Return left† right with no external averaging or Nc projection.

        Each expression's dummy indices are scoped independently by the tensor
        dependency. Adjoint construction preserves physical external labels.
        Complex rectangular weights are retained exactly.
        """
        from symbolica import E, S
        from symbolica.community import tensor as t

        if left.to_expression() == 0 or right.to_expression() == 0:
            return E("0")
        if set(left.axes) != set(right.axes):
            raise ValueError("Born and loop color ports differ")
        bra = (
            self.lift(left)
            .wrap_indices(S("loop_color_bilinear::bra"), dummies_only=True)
            .dirac_adjoint(preserve_indices=True)
        )
        ket = self.lift(right).wrap_indices(
            S("loop_color_bilinear::ket"), dummies_only=True
        )
        overlap = (bra * ket).simplify_algebra(
            gamma=False,
            color=True,
            color_substitute_cof_dimension_invariants=True,
        )
        if overlap.axes:
            raise NotImplementedError(
                "formal color overlap retained uncontracted ports"
            )

        # Named representation invariants implement the declared SU(N) family.
        # All trace/Fierz/index operations above belong to the dependency.
        def bind_invariants(value):
            return (
                value.to_expression()
                .replace(t.Representation.cof(self.nc).dynkin_index(), E("1/2"))
                .replace(
                    t.Representation.coad(self._adjoint_dimension).casimir(), self.nc
                )
                .replace(self._adjoint_dimension, self.nc**2 - 1)
                .together()
            )

        result = bind_invariants(overlap)
        if set(result.get_all_symbols()) - {self.nc}:
            # Some products of symmetric colour traces stay compact after the
            # first pass. The dependency's normalized-projector expansion makes
            # their generator permutations available to its Fierz reduction.
            # Expand only on this unresolved path; ordinary overlaps stay small.
            expanded = overlap.expand_projectors()
            if expanded.to_expression() != overlap.to_expression():
                overlap = expanded.simplify_algebra(
                    gamma=False,
                    color=True,
                    color_substitute_cof_dimension_invariants=True,
                )
                if overlap.axes:
                    raise NotImplementedError(
                        "formal color overlap retained uncontracted ports"
                    )
                result = bind_invariants(overlap)
        if set(result.get_all_symbols()) - {self.nc}:
            # Quartic Gram invariants can require the explicit relation
            # dA = Nc**2 - 1. Only after ordinary tensor reduction/expansion,
            # let the dependency substitute its recognized SU(N) invariants.
            # A remaining indexed tensor may not admit a composite dimension;
            # it remains unsupported rather than being coerced to a scalar.
            try:
                bound = t.TensorExpression(result)
            except ValueError:
                pass
            else:
                result = bind_invariants(
                    bound.simplify_algebra(
                        gamma=False,
                        color=True,
                        color_substitute_cof_dimension_invariants=True,
                    )
                )
        if set(result.get_all_symbols()) - {self.nc}:
            raise NotImplementedError(
                "formal color overlap retained unsupported tensors or scalar symbols"
            )
        require_exact_numbers(result)
        return result

    def rectangular(
        self, born: Sequence[Any], loop: Sequence[Any]
    ) -> tuple[tuple[Any, ...], ...]:
        """Exact, possibly complex Born-by-loop overlap matrix."""
        return tuple(
            tuple(self.overlap(left, right) for right in loop) for left in born
        )
