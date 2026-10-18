# SPDX-License-Identifier: 0BSD
"""Exact local tensor-coefficient kernels for the open-loop prototype.

Symbolica owns tensor contractions, polynomial algebra and evaluator storage.
Each instruction sees placeholders for its input currents, never their expanded
expressions. The saved evaluator is the shared-core boundary for native bulk
binary64, DoubleFloat and arbitrary-precision execution. The native scheduler
executes all nodes without serializing intermediate values; the binary64
dependency evaluator remains an independent oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from .exact import require_exact_numbers


@dataclass(frozen=True, slots=True)
class CoefficientLabel:
    component: int
    powers: tuple[int, ...]


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
    variable_weights: tuple[int, ...] = (1, 1, 1, 1)

    @property
    def rank(self) -> int:
        """Structural weighted rank; coefficient relations may lower it."""
        return max(
            (
                sum(
                    p * w
                    for p, w in zip(label.powers, self.variable_weights, strict=True)
                )
                for label in self.output_labels
            ),
            default=0,
        )

    def evaluator_bytes(self) -> bytes:
        """Dependency-owned exact evaluator storage, with ordered labels above."""
        return self.evaluator.save()


class CoefficientDAG:
    """Local contraction DAG; a Python node loop dispatches whole native batches."""

    domain = "four-dimensional-model-rules"
    epsilon_power = 0
    mu2_power = 0
    direct_r2_certified = False

    def __init__(
        self,
        variables: tuple[Any, ...],
        *,
        extra_variables: tuple[Any, ...] = (),
        extra_weights: tuple[int, ...] = (),
    ):
        """Keep auxiliary polynomial powers explicit without interpreting them.

        The four momentum components carry weight one. Callers specify weights
        for auxiliary variables, e.g. one for an evanescent slash parameter and
        zero for epsilon. This bookkeeping does not certify any dimensional
        algebra or permit truncating intermediate auxiliary powers.
        """
        if len(variables) != 4 or len(set(variables)) != 4:
            raise ValueError("four distinct loop-momentum component variables required")
        if len(extra_variables) != len(extra_weights) or any(
            type(weight) is not int or weight < 0 for weight in extra_weights
        ):
            raise ValueError(
                "each auxiliary variable requires a nonnegative integer weight"
            )
        self.variables = variables + extra_variables
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("all polynomial variables must be distinct")
        self.variable_weights = (1, 1, 1, 1, *extra_weights)
        if extra_variables:
            self.domain = "uninterpreted-auxiliary-polynomial-kernels"
            self.epsilon_power = self.mu2_power = None
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
            labels.append(CoefficientLabel(0, (0,) * len(self.variables)))
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
                self.variable_weights,
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


def require_finite_batch(values: Any, stage: str, *, rows_per_point: int = 1) -> None:
    """Fail this diagnostic batch before exposing nonfinite native results.

    Only failing row identities cross Python. This is not a rescue policy.
    Helicity-packed callers declare their grouping to retain original point IDs.
    """
    failures = values.nonfinite_rows()
    if failures:
        locations = [(row // rows_per_point, row) for row in failures]
        raise ArithmeticError(f"nonfinite {stage} (point, lane): {locations}")


class NativeCoefficientDAG:
    """One native schedule over saved local instructions at a fixed precision.

    The arena is global inputs followed by each node's ordered output slots.
    No current is expanded symbolically and no numerical intermediate crosses
    Python. ``output_nodes`` optionally restricts the returned packed columns.
    """

    def __init__(
        self,
        dag: CoefficientDAG,
        *,
        arithmetic: str = "double",
        precision: int | None = None,
        backend: Any = None,
        output_nodes: tuple[int, ...] | None = None,
        coefficient_engine: str = "interpreted",
    ):
        if backend is None:
            import _loop_reduction as backend
        self.dag, self.backend = dag, backend
        self.native_options = {"arithmetic": arithmetic, "precision": precision}
        self.output_nodes = (
            tuple(range(len(dag.nodes))) if output_nodes is None else output_nodes
        )
        if any(i < 0 or i >= len(dag.nodes) for i in self.output_nodes):
            raise ValueError("output nodes must refer to existing coefficient kernels")
        positions = {parameter: i for i, parameter in enumerate(dag.parameters)}
        width = len(positions)
        node_slots, input_slots = [], []
        for node in dag.nodes:
            slots = [positions[p] for p in node.parameters]
            slots.extend(
                slot
                for dependency in node.dependencies
                for slot in node_slots[dependency]
            )
            input_slots.append(slots)
            node_slots.append(list(range(width, width + len(node.output_labels))))
            width += len(node.output_labels)
        self.plan = backend.CoefficientPlan(
            [node.evaluator_bytes() for node in dag.nodes],
            input_slots,
            len(positions),
            [slot for index in self.output_nodes for slot in node_slots[index]],
            engine=coefficient_engine,
            **self.native_options,
        )
        self.coefficient_engine = self.plan.engine

    def evaluate_batch(self, values: Any) -> Any:
        """Return one typed native buffer; rows and their identities are retained."""
        return self.plan.evaluate_batch(values)

    def evaluate_decimal(self, values: Any) -> tuple[Any, ...]:
        """Decimal boundary convenience; only inputs and selected outputs convert."""
        import numpy as np

        values = np.asarray(values, dtype=object)
        if values.ndim != 3 or values.shape[1:] != (len(self.dag.parameters), 2):
            raise ValueError(
                "expected (batch, parameter, real/imaginary) decimal input"
            )
        if any(not isinstance(value, str) for value in values.flat):
            raise TypeError("native precise input components must be decimal strings")
        rows = len(values)
        native = self.backend.NativeBatch.from_decimal(
            values.reshape(-1, 2).tolist(),
            rows,
            len(self.dag.parameters),
            **self.native_options,
        )
        evaluated = self.evaluate_batch(native)
        packed = np.asarray(evaluated.to_decimal(), dtype=object).reshape(
            rows, evaluated.columns, 2
        )
        widths = [
            len(self.dag.nodes[index].output_labels) for index in self.output_nodes
        ]
        return tuple(np.split(packed, np.cumsum(widths)[:-1], axis=1))
