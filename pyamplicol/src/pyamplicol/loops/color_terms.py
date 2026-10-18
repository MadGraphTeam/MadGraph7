# SPDX-License-Identifier: 0BSD
"""Local colour alternatives over the native coefficient-current DAG.

Only a single vertex/propagator is symbolically expanded. Lorentz currents stay
as coefficient-node references. Native tensor operations own colour products,
index contractions and sums; no colour identities or SU(3) reconstruction are
implemented here. This is a preparation primitive, not a process evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from .kernels import CoefficientDAG
from .recurrence import separate_local_color


@dataclass(frozen=True, slots=True)
class LocalColorTerm:
    lorentz: Any = field(repr=False)
    color: Any = field(repr=False)
    source: Any = None


def split_local_color_terms(expression: Any) -> tuple[LocalColorTerm, ...]:
    """Split one local native expression, retaining independent colour terms.

    The existing one-colour proof classifies each expanded monomial. Equal
    native colour factors share a summed local Lorentz factor; no nonlocal
    current expressions are accepted or substituted by this operation.
    """
    from symbolica import AtomType
    from symbolica.community import tensor as t

    expanded = expression.expand()
    terms = list(expanded) if expanded.get_type() == AtomType.Add else [expanded]
    groups = {}
    for term in terms:
        lorentz, color = separate_local_color(t.TensorExpression(term))
        if lorentz.to_expression() == 0:
            continue
        key = color.to_expression()
        if key in groups:
            groups[key] = groups[key] + lorentz
        else:
            groups[key] = lorentz
    result = tuple(
        LocalColorTerm(lorentz, t.TensorExpression(color))
        for color, lorentz in groups.items()
        if lorentz.to_expression() != 0
    )
    reconstructed = sum((term.lorentz * term.color for term in result), 0)
    if (reconstructed - expression).expand().to_expression() != 0:
        raise ValueError("native local colour-term reconstruction failed")
    return result


def model_basis_terms(
    catalogue: Any, entries: Any, *, epsilon_power: int = 0
) -> tuple[LocalColorTerm, ...]:
    """Keep selected UFO basis entries separate and lower their exact coefficient.

    All entries must belong to one local vertex. The caller must select eligible
    loop groups first; this function does not infer process/UV/R2 eligibility.
    The selected Laurent power stays explicit. Native local tensor axes and
    all-incoming momenta are the catalogue's interface, without graph rebinding.
    """
    entries = tuple(entries)
    if len({(entry.collection, entry.vertex_id) for entry in entries}) > 1:
        raise ValueError("local basis entries must belong to one vertex")
    result = []
    for entry in entries:
        scalar = entry.coupling.value.coefficient(epsilon_power)
        if scalar == 0:
            continue
        native = catalogue.lower(entry)
        result.append(LocalColorTerm(scalar * native.lorentz, native.color, entry.key))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ColorCoefficientTerm:
    node: int
    color: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class ColorCurrent:
    dag: CoefficientDAG = field(repr=False, compare=False)
    terms: tuple[ColorCoefficientTerm, ...]

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def closed(self) -> bool:
        """Whether all alternatives have closed their Lorentz/spin indices."""
        return all(not self.dag.nodes[term.node].axes for term in self.terms)


class ColorTermDAG:
    """Branch at local basis choices while retaining reusable coefficient nodes.

    At each named site, equal local tensors with the same dependency nodes share
    one coefficient instruction. If alternatives have that same instruction,
    their colour tensors are added natively. Distinct coefficient instructions
    remain explicit alternatives; no global Lorentz expansion or automatic
    large-Nc truncation occurs. Branch counts remain visible in ColorCurrent.
    """

    def __init__(self, dag: CoefficientDAG):
        self.dag = dag
        self._instructions: dict[Any, int] = {}

    def add(
        self,
        name: str,
        local_terms: Any,
        dependencies: tuple[ColorCurrent, ...] = (),
        *,
        library: Any = None,
        materialize: Any = None,
    ) -> ColorCurrent:
        if any(current.dag is not self.dag for current in dependencies):
            raise ValueError("colour currents must belong to this coefficient DAG")
        if any(current.is_zero for current in dependencies):
            return ColorCurrent(self.dag, ())
        groups = {}
        for local in local_terms:
            if local.lorentz.to_expression() == 0 or local.color.to_expression() == 0:
                continue
            factor = (
                local.lorentz.to_tensor(library)
                if materialize is None
                else materialize(local)
            ).to_dense()
            factor_key = (tuple(factor.axes), tuple(factor.shape), tuple(factor))
            for inputs in product(*(current.terms for current in dependencies)):
                nodes = tuple(term.node for term in inputs)
                key = name, factor_key, nodes
                if key not in self._instructions:
                    self._instructions[key] = self.dag.add(name, factor, nodes)
                node = self._instructions[key]
                color = local.color
                for term in inputs:
                    color = color * term.color
                groups[node] = groups[node] + color if node in groups else color
        return ColorCurrent(
            self.dag,
            tuple(
                ColorCoefficientTerm(node, color)
                for node, color in groups.items()
                if color.to_expression() != 0
            ),
        )
