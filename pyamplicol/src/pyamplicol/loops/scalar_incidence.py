# SPDX-License-Identifier: 0BSD
"""Constructive scalar-alias incidence through existing coefficient instructions.

Symbolica collects each local kernel. Only dependency support and homogeneous
alias degree propagate between kernels; no whole-current expression is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Support:
    constant: bool = False
    linear: frozenset[int] = frozenset()
    nonlinear: bool = False

    @property
    def nonzero(self):
        return self.constant or bool(self.linear) or self.nonlinear

    def union(self, other):
        return _Support(
            self.constant or other.constant,
            self.linear | other.linear,
            self.nonlinear or other.nonlinear,
        )

    def product(self, other):
        return _Support(
            self.constant and other.constant,
            (self.linear if other.constant else frozenset())
            | (other.linear if self.constant else frozenset()),
            (self.nonlinear and other.nonzero)
            or (other.nonlinear and self.nonzero)
            or (bool(self.linear) and bool(other.linear)),
        )

    def power(self, exponent):
        if exponent == 0:
            return _Support(constant=True)
        if exponent == 1:
            return self
        return _Support(
            self.constant,
            self.linear if self.constant else frozenset(),
            self.nonlinear or bool(self.linear),
        )


def scalar_alias_incidence(dag, output_nodes, aliases):
    """Prove scalar CT outputs homogeneous of degree one in selected aliases.

    Returned sets index ``aliases`` in the requested output-node order. Local
    polynomial cancellation is exact. Distinct input currents remain independent
    for this constructive proof; identities between whole currents are neither
    guessed nor expanded. Their supports may therefore conservatively include
    an alias that cancels only after substituting whole currents. This cannot
    remove a valid colour contribution. Zero outputs have empty support.
    """
    from symbolica import AtomType, E, S

    aliases, output_nodes = tuple(aliases), tuple(output_nodes)
    if len(set(aliases)) != len(aliases) or any(
        alias.get_type() != AtomType.Var for alias in aliases
    ):
        raise ValueError("scalar aliases must be distinct symbolic variables")
    required = set(output_nodes)
    if any(
        type(node) is not int or not 0 <= node < len(dag.nodes) for node in required
    ):
        raise ValueError("scalar alias outputs must name coefficient nodes")
    for index in range(len(dag.nodes) - 1, -1, -1):
        if index in required:
            required.update(dag.nodes[index].dependencies)
    supports = {}
    alias_support = tuple(_Support(linear=frozenset((i,))) for i in range(len(aliases)))
    for index, node in enumerate(dag.nodes):
        if index not in required:
            continue
        inputs, labels, input_support = [], [], []
        for position, dependency in enumerate(node.dependencies):
            for column, label in enumerate(dag.nodes[dependency].output_labels):
                inputs.append(S(f"loop_coefficient::input_{position}_{column}"))
                labels.append((dependency, label))
                input_support.append(supports[dependency][column])
        if tuple(labels) != node.input_labels or set(inputs) & set(aliases):
            raise ValueError(
                f"scalar alias kernel {node.name!r} has inconsistent inputs"
            )
        variables = (*aliases, *inputs)
        variable_support = (*alias_support, *input_support)
        outputs = []
        for expression in node.expressions:
            support = _Support()
            collected = expression.expand().coefficient_list(*variables)
            for monomial, coefficient in collected:
                if coefficient == 0:
                    continue
                try:
                    if set(coefficient.get_all_symbols(False)) & set(variables):
                        raise ValueError("functional dependence on an alias or input")
                    terms = monomial.to_polynomial(vars=variables).coefficient_list(
                        variables
                    )
                    if len(terms) != 1:
                        raise ValueError("non-monomial alias or input powers")
                    powers, residual = terms[0]
                    if residual.to_expression() != E("1"):
                        raise ValueError("unsupported alias or input monomial")
                except (ValueError, TypeError) as error:
                    raise NotImplementedError(
                        f"scalar alias kernel {node.name!r} is not polynomial "
                        "in its aliases and input currents"
                    ) from error
                term = _Support(constant=True)
                for value, power in zip(variable_support, powers, strict=True):
                    term = term.product(value.power(power))
                support = support.union(term)
            outputs.append(support)
        supports[index] = tuple(outputs)
    incidence = []
    for index in output_nodes:
        node = dag.nodes[index]
        if (
            node.axes
            or len(node.output_labels) != 1
            or any(node.output_labels[0].powers)
        ):
            raise ValueError(
                "scalar alias incidence requires a closed scalar CT output"
            )
        support = supports[index][0]
        if support.constant or support.nonlinear:
            raise NotImplementedError(
                f"counterterm output {node.name!r} lacks a constructive degree-one "
                "alias proof (alias-independent or nonlinear contribution)"
            )
        incidence.append(support.linear)
    return tuple(incidence)


def scalar_grade_term_mask(incidence, coefficients):
    """Keep tensor terms reached by a nonzero alias in this scalar grade."""
    coefficients = tuple(coefficients)
    active = {index for index, value in enumerate(coefficients) if value != 0}
    if any(
        type(index) is not int or not 0 <= index < len(coefficients)
        for support in incidence
        for index in support
    ):
        raise ValueError("scalar incidence and coefficient layouts differ")
    return tuple(bool(support & active) for support in incidence)
