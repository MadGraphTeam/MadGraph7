# SPDX-License-Identifier: 0BSD
"""Dependency-backed rectangular physical and formal SU(Nc) color overlaps.

These cold-path helpers preserve native tensors before component evaluation.
They provide neither a model's dimensional-continuation certificate nor a
projection policy. Counterterm scalar invariants must remain formal separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .._internal.physics.symbols import symbols
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
class PhysicalColorSpace:
    """Exact overlaps at the tensors' declared physical color dimensions.

    No dimension is lifted and no scalar coefficient is reinterpreted as a
    Casimir. The dependency supplies trace, projector, Fierz and SU(N) invariant
    reduction. A cold process plan owns this cache; numerical model states and
    event values never enter its keys or values.
    """

    _overlaps: dict[tuple[Any, Any], Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _bras: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)
    _kets: dict[Any, Any] = field(default_factory=dict, init=False, repr=False)

    def overlap(self, left: Any, right: Any) -> Any:
        """Return exact left† right, without averaging or formal continuation."""
        from symbolica import E
        from symbolica.community import tensor as t

        if left.to_expression() == 0 or right.to_expression() == 0:
            return E("0")
        if set(left.axes) != set(right.axes):
            raise ValueError("Born and loop color ports differ")
        key = (left.to_expression(), right.to_expression())
        if key not in self._overlaps:
            require_exact_numbers(key[0])
            require_exact_numbers(key[1])
            # Rectangular matrices reuse each exact left and right tensor many
            # times. Keep their dependency-prepared, independently scoped forms;
            # no canonicalization or numerical equality enters the cache key.
            if key[0] not in self._bras:
                self._bras[key[0]] = left.wrap_indices(
                    symbols.qualified("physical_color_bilinear::bra"), dummies_only=True
                ).dirac_adjoint(preserve_indices=True)
            if key[1] not in self._kets:
                self._kets[key[1]] = right.wrap_indices(
                    symbols.qualified("physical_color_bilinear::ket"), dummies_only=True
                )
            bra, ket = self._bras[key[0]], self._kets[key[1]]
            settings = dict(
                gamma=False,
                color=True,
                color_substitute_cof_dimension_invariants=True,
            )
            original = bra * ket
            overlap = original.simplify_algebra(**settings)
            result = overlap.to_expression().together()
            if result.get_all_symbols():
                expanded = overlap.expand_projectors()
                if expanded.to_expression() != overlap.to_expression():
                    overlap = expanded.simplify_algebra(**settings)
                    result = overlap.to_expression().together()
            if overlap.axes or result.get_all_symbols():
                # Algebraic saturation need not reduce every closed invariant
                # network to a scalar. The dependency's physical representation
                # library supplies exact components for its supported spaces.
                # Evaluate the original independently scoped network: opening a
                # residual compact trace is neither necessary nor assumed safe.
                try:
                    components = original.to_tensor(t.TensorLibrary.hep_lib_atom())
                except (ValueError, TypeError, RuntimeError) as error:
                    raise NotImplementedError(
                        "physical color overlap retained unsupported tensors; "
                        "the native component library cannot evaluate this network"
                    ) from error
                if components.axes:
                    raise NotImplementedError(
                        "physical color overlap retained unsupported open ports"
                    )
                result = components[0].together()
                if result.get_all_symbols():
                    raise NotImplementedError(
                        "physical color overlap retained unsupported scalar symbols"
                    )
            require_exact_numbers(result)
            self._overlaps[key] = result
        return self._overlaps[key]

    def rectangular(
        self, born: Sequence[Any], loop: Sequence[Any]
    ) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            tuple(self.overlap(left, right) for right in loop) for left in born
        )


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
        from symbolica import AtomType, Expression

        if not isinstance(self.nc, Expression) or self.nc.get_type() != AtomType.Var:
            raise TypeError("formal Nc must be a symbolic variable")
        object.__setattr__(
            self,
            "_adjoint_dimension",
            symbols.qualified(f"{self.nc.get_name()}__adjoint_dimension", is_real=True),
        )

    def lift(self, expression: Any) -> Any:
        """Use dependency port patterns for physical representation dimensions."""
        from symbolica.community import tensor as t

        value = expression.to_expression()
        require_exact_numbers(value)
        index = symbols.qualified("loop_color_bilinear::index_")
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
        from symbolica import E
        from symbolica.community import tensor as t

        if left.to_expression() == 0 or right.to_expression() == 0:
            return E("0")
        if set(left.axes) != set(right.axes):
            raise ValueError("Born and loop color ports differ")
        bra = (
            self.lift(left)
            .wrap_indices(
                symbols.qualified("loop_color_bilinear::bra"), dummies_only=True
            )
            .dirac_adjoint(preserve_indices=True)
        )
        ket = self.lift(right).wrap_indices(
            symbols.qualified("loop_color_bilinear::ket"), dummies_only=True
        )
        original = bra * ket
        overlap = original.simplify_algebra(
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
            # Closed higher-rank invariant networks can contain signed-zero
            # terms hidden inside compact traces and products of sums. Keep
            # their formal invariants until the dependency has exposed and
            # canonized those terms. The ordinary fast paths above are unchanged.
            canonical = (
                original.simplify_algebra(
                    gamma=False,
                    color=True,
                    color_substitute_cof_dimension_invariants=False,
                )
                .contract()
                .expand_projectors()
                .undo_trace()
                .undo_chain()
                .expand()
                .canonize()
                .simplify_algebra(
                    gamma=False,
                    color=True,
                    color_substitute_cof_dimension_invariants=True,
                )
            )
            if canonical.axes:
                raise NotImplementedError(
                    "formal color overlap retained uncontracted ports"
                )
            result = bind_invariants(canonical)
            # Apply the dimension relation only to closed scalar invariants;
            # indexed representations require their original symbolic labels.
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
