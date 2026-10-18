# SPDX-License-Identifier: 0BSD
"""Exact local tensor-coefficient kernels for the open-loop prototype.

Symbolica owns tensor contractions, polynomial algebra and evaluator storage.
Each instruction sees placeholders for its input currents, never their expanded
expressions. The saved evaluator is the shared-core boundary for native bulk
binary64, DoubleFloat and arbitrary-precision execution. Only the binary64 bulk
oracle is executed here; Python's precise evaluator API is single-point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from .exact import require_exact_numbers


@dataclass(frozen=True, slots=True)
class CoefficientLabel:
    component: int
    powers: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CoefficientKernel:
    id: int
    name: str
    dependencies: tuple[int, ...]
    parameters: tuple[Any, ...]
    input_labels: tuple[tuple[int, CoefficientLabel], ...]
    output_labels: tuple[CoefficientLabel, ...]
    axes: tuple[Any, ...]
    shape: tuple[int, ...]
    expressions: tuple[Any, ...] = field(repr=False, compare=False)
    evaluator: Any = field(repr=False, compare=False)

    @property
    def rank(self) -> int:
        """Structural polynomial rank; coefficient relations may lower it."""
        return max((sum(label.powers) for label in self.output_labels), default=0)

    def evaluator_bytes(self) -> bytes:
        """Dependency-owned exact evaluator storage, with ordered labels above."""
        return self.evaluator.save()


class CoefficientDAG:
    """Local contraction DAG; a Python node loop dispatches whole native batches."""

    domain = "four-dimensional-model-rules"
    epsilon_power = 0
    mu2_power = 0
    direct_r2_certified = False

    def __init__(self, variables: tuple[Any, ...]):
        if len(variables) != 4 or len(set(variables)) != 4:
            raise ValueError("four distinct loop-momentum component variables required")
        self.variables = variables
        self.nodes: list[CoefficientKernel] = []

    def add(self, name: str, factor: Any, dependencies: tuple[int, ...] = ()) -> int:
        """Contract a local exact tensor with already-computed open currents."""
        from symbolica import E, Expression, S
        from symbolica.community import tensor as t

        if len(set(dependencies)) != len(dependencies):
            raise ValueError(
                "an input current cannot be consumed twice in one contraction"
            )
        if any(i < 0 or i >= len(self.nodes) for i in dependencies):
            raise ValueError("dependencies must refer to preceding coefficient kernels")
        factor = factor.to_dense()
        for component in range(len(factor)):
            require_exact_numbers(factor[component])
        network = factor
        input_symbols = []
        input_labels = []
        for position, dependency in enumerate(dependencies):
            node = self.nodes[dependency]
            components = [E("0")] * prod(node.shape)
            for column, label in enumerate(node.output_labels):
                symbol = S(f"loop_coefficient::input_{position}_{column}")
                input_symbols.append(symbol)
                input_labels.append((dependency, label))
                monomial = symbol
                for variable, power in zip(self.variables, label.powers, strict=True):
                    monomial *= variable**power
                components[label.component] += monomial
            current = t.Tensor.dense(
                t.TensorName("loop_coefficient::current")(*node.axes), components
            )
            network = network * current
        if dependencies:
            network.execute()
            output = network.result_tensor().to_dense()
        else:
            output = factor.to_dense()
        labels = []
        expressions = []
        for component in range(len(output)):
            value = output[component]
            if not isinstance(value, Expression):
                raise TypeError(
                    "local coefficient kernels require exact expression tensors"
                )
            for monomial, expression in value.expand().coefficient_list(
                *self.variables
            ):
                terms = monomial.to_polynomial(vars=self.variables).coefficient_list(
                    self.variables
                )
                if len(terms) != 1:
                    raise ValueError(
                        "native coefficient extraction returned a non-monomial"
                    )
                powers, residual = terms[0]
                if residual.to_expression() != E("1") or any(
                    expression.contains(q) for q in self.variables
                ):
                    raise ValueError("local factor is not polynomial in loop momentum")
                labels.append(CoefficientLabel(component, tuple(powers)))
                expressions.append(expression)
        if not expressions:
            labels.append(CoefficientLabel(0, (0, 0, 0, 0)))
            expressions.append(E("0"))
        inputs = set(input_symbols)
        parameters = tuple(
            sorted(
                {
                    s
                    for expression in expressions
                    for s in expression.get_all_symbols(False)
                }
                - inputs,
                key=lambda s: s.to_canonical_string(),
            )
        )
        evaluator = Expression.evaluator_multiple(
            expressions, [*parameters, *input_symbols], n_cores=1, jit_compile=False
        )
        index = len(self.nodes)
        self.nodes.append(
            CoefficientKernel(
                index,
                name,
                dependencies,
                parameters,
                tuple(input_labels),
                tuple(labels),
                tuple(output.axes),
                tuple(output.shape),
                tuple(expressions),
                evaluator,
            )
        )
        return index

    @property
    def parameters(self) -> tuple[Any, ...]:
        return tuple(
            sorted(
                {p for node in self.nodes for p in node.parameters},
                key=lambda p: p.to_canonical_string(),
            )
        )

    def evaluate_batch(self, values: Any) -> tuple[Any, ...]:
        """Binary64 oracle: batch axis first, parameters in ``parameters`` order.

        This intentionally does not coerce precise inputs or simulate a precise
        batch by calling a single-point evaluator in a Python loop.
        """
        import numpy as np

        values = np.asarray(values)
        if values.dtype not in (np.dtype("float64"), np.dtype("complex128")):
            raise TypeError("binary64 oracle requires float64 or complex128 input")
        if values.ndim != 2 or values.shape[1] != len(self.parameters):
            raise ValueError("expected (batch, parameter) input")
        positions = {parameter: i for i, parameter in enumerate(self.parameters)}
        outputs = []
        for node in self.nodes:
            columns = [values[:, [positions[p]]] for p in node.parameters]
            columns.extend(outputs[dependency] for dependency in node.dependencies)
            inputs = (
                np.concatenate(columns, axis=1)
                if columns
                else np.empty((len(values), 0), dtype=complex)
            )
            outputs.append(node.evaluator.evaluate_complex(inputs))
        return tuple(outputs)


class NativeCoefficientDAG:
    """Execute saved local instructions in native batches at one fixed precision.

    Decimal complex pairs are transport only. Every arithmetic operation is in
    the shared core's binary64, DoubleFloat or arbitrary-precision carrier.
    """

    def __init__(
        self,
        dag: CoefficientDAG,
        *,
        arithmetic: str = "double",
        precision: int | None = None,
        backend: Any = None,
    ):
        if backend is None:
            import _loop_reduction as backend
        self.dag = dag
        self.precision = precision
        self.arithmetic = arithmetic
        self.evaluators = tuple(
            backend.CoefficientEvaluator(
                node.evaluator_bytes(), arithmetic=arithmetic, precision=precision
            )
            for node in dag.nodes
        )

    def evaluate_decimal(self, values: Any) -> tuple[Any, ...]:
        import numpy as np

        values = np.asarray(values, dtype=object)
        if values.ndim != 3 or values.shape[1:] != (len(self.dag.parameters), 2):
            raise ValueError(
                "expected (batch, parameter, real/imaginary) decimal input"
            )
        if any(not isinstance(value, str) for value in values.flat):
            raise TypeError("native precise input components must be decimal strings")
        rows = values.shape[0]
        positions = {parameter: i for i, parameter in enumerate(self.dag.parameters)}
        outputs = []
        for node, evaluator in zip(self.dag.nodes, self.evaluators, strict=True):
            columns = [values[:, [positions[p]], :] for p in node.parameters]
            columns.extend(outputs[dependency] for dependency in node.dependencies)
            inputs = (
                np.concatenate(columns, axis=1)
                if columns
                else np.empty((rows, 0, 2), dtype=object)
            )
            result = evaluator.evaluate_decimal(inputs.reshape(-1, 2).tolist(), rows)
            outputs.append(
                np.asarray(result, dtype=object).reshape(
                    rows, len(node.output_labels), 2
                )
            )
        return tuple(outputs)
