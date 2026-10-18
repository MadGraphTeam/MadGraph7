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

from collections.abc import Mapping, Sequence
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
        self.momentum_variables = variables
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


@dataclass(frozen=True, slots=True)
class CoefficientInstructions:
    """Dependency bytecode and integer arena layout, independent of arithmetic."""

    states: tuple[bytes, ...]
    input_slots: tuple[tuple[int, ...], ...]
    parameter_count: int
    output_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedCoefficientDAG:
    """Retained instructions plus exact producer-side binding/domain metadata.

    The instruction payload contains no graph or evaluator objects. Parameter
    symbols and polynomial labels preserve the existing exact sharing proof;
    they are separate from the native arena descriptor and numerical workspace.
    This internal record introduces no artifact persistence format.
    """

    instructions: CoefficientInstructions
    parameters: tuple[Any, ...]
    output_nodes: tuple[int, ...]
    output_widths: tuple[int, ...]
    output_metadata: tuple[Any, ...]
    variables: tuple[Any, ...]
    variable_weights: tuple[int, ...]
    domain: tuple[Any, ...]

    @classmethod
    def from_dag(cls, dag: CoefficientDAG, *, output_nodes=None):
        parameters = dag.parameters
        output_nodes = (
            tuple(range(len(dag.nodes)))
            if output_nodes is None
            else tuple(output_nodes)
        )
        if any(i < 0 or i >= len(dag.nodes) for i in output_nodes):
            raise ValueError("output nodes must refer to existing coefficient kernels")
        output_widths = tuple(
            len(dag.nodes[index].output_labels) for index in output_nodes
        )
        positions = {parameter: i for i, parameter in enumerate(parameters)}
        width = len(positions)
        node_slots, input_slots = [], []
        for node in dag.nodes:
            slots = [positions[p] for p in node.parameters]
            slots.extend(
                slot
                for dependency in node.dependencies
                for slot in node_slots[dependency]
            )
            input_slots.append(tuple(slots))
            node_slots.append(tuple(range(width, width + len(node.output_labels))))
            width += len(node.output_labels)
        return cls(
            CoefficientInstructions(
                tuple(node.evaluator_bytes() for node in dag.nodes),
                tuple(input_slots),
                len(positions),
                tuple(slot for index in output_nodes for slot in node_slots[index]),
            ),
            parameters,
            output_nodes,
            output_widths,
            tuple(
                (dag.nodes[index].shape, dag.nodes[index].output_labels)
                for index in output_nodes
            ),
            dag.variables,
            dag.variable_weights,
            (dag.domain, dag.epsilon_power, dag.mu2_power, dag.direct_r2_certified),
        )

    @property
    def identity(self):
        instructions = self.instructions
        return (
            instructions.states,
            instructions.input_slots,
            instructions.parameter_count,
            instructions.output_slots,
            self.output_metadata,
            self.variables,
            self.variable_weights,
            self.domain,
        )


class NativeCoefficientDAG:
    """One native schedule over saved local instructions at a fixed precision.

    The arena is global inputs followed by each node's ordered output slots.
    No current is expanded symbolically and no numerical intermediate crosses
    Python. ``output_nodes`` optionally restricts the returned packed columns.
    """

    def __init__(
        self,
        dag: CoefficientDAG | PreparedCoefficientDAG,
        *,
        arithmetic: str = "double",
        precision: int | None = None,
        backend: Any = None,
        output_nodes: tuple[int, ...] | None = None,
        coefficient_engine: str = "interpreted",
        kernel_cache: Any = None,
    ):
        if backend is None:
            import _loop_reduction as backend
        self.backend = backend
        if isinstance(dag, PreparedCoefficientDAG):
            if output_nodes is not None:
                raise ValueError("prepared coefficient outputs are already selected")
            self.dag, self.prepared = None, dag
        else:
            self.dag = dag  # Producer metadata only; evaluation never reads it.
            self.prepared = PreparedCoefficientDAG.from_dag(
                dag, output_nodes=output_nodes
            )
        self.parameters = self.prepared.parameters
        self.native_options = {"arithmetic": arithmetic, "precision": precision}
        self.output_nodes = self.prepared.output_nodes
        self.output_widths = self.prepared.output_widths
        instructions = self.prepared.instructions
        cache_options = {} if kernel_cache is None else {"kernel_cache": kernel_cache}
        self.plan = backend.CoefficientPlan(
            instructions.states,
            instructions.input_slots,
            instructions.parameter_count,
            instructions.output_slots,
            engine=coefficient_engine,
            **self.native_options,
            **cache_options,
        )
        self.coefficient_engine = self.plan.engine
        # Positional dependency bytecode includes exact constants and ordered
        # results. Freeze the packed layout and domain, keeping graph metadata
        # separate. Later mutations of a source DAG cannot alter this key.
        self.schedule_identity = (
            *self.prepared.identity,
            arithmetic,
            self.plan.precision_bits,
            self.coefficient_engine,
        )

    def rebind(
        self,
        *,
        arithmetic: str,
        precision: int | None = None,
        coefficient_engine: str = "interpreted",
        kernel_cache: Any = None,
    ) -> NativeCoefficientDAG:
        """Create a fresh native workspace from retained exact instructions."""
        return NativeCoefficientDAG(
            self.prepared,
            backend=self.backend,
            arithmetic=arithmetic,
            precision=precision,
            coefficient_engine=coefficient_engine,
            kernel_cache=kernel_cache,
        )

    def sharing_key(self, model_bindings: Mapping[str, Any]) -> tuple[Any, ...]:
        """Prove equal inputs from fixed exact symbolic model definitions.

        Bindings are the native-name expressions used by the model evaluator.
        Unbound momentum/state parameters retain their complete identities.
        No sampled value, flavour multiplicity or graph equivalence is inferred.
        """
        bindings = tuple(
            model_bindings.get(parameter.get_name(), parameter)
            for parameter in self.parameters
        )
        for binding in bindings:
            require_exact_numbers(binding)
        return self.schedule_identity, bindings

    def evaluate_batch(self, values: Any) -> Any:
        """Return one typed native buffer; rows and their identities are retained."""
        return self.plan.evaluate_batch(values)

    def evaluate_decimal(self, values: Any) -> tuple[Any, ...]:
        """Decimal boundary convenience; only inputs and selected outputs convert."""
        import numpy as np

        values = np.asarray(values, dtype=object)
        if values.ndim != 3 or values.shape[1:] != (len(self.parameters), 2):
            raise ValueError(
                "expected (batch, parameter, real/imaginary) decimal input"
            )
        if any(not isinstance(value, str) for value in values.flat):
            raise TypeError("native precise input components must be decimal strings")
        rows = len(values)
        native = self.backend.NativeBatch.from_decimal(
            values.reshape(-1, 2).tolist(),
            rows,
            len(self.parameters),
            **self.native_options,
        )
        evaluated = self.evaluate_batch(native)
        packed = np.asarray(evaluated.to_decimal(), dtype=object).reshape(
            rows, evaluated.columns, 2
        )
        return tuple(np.split(packed, np.cumsum(self.output_widths)[:-1], axis=1))


@dataclass(frozen=True, slots=True)
class CoefficientSharingGroups:
    """Execution representatives with every original graph position retained."""

    representatives: tuple[int, ...]
    representative_for: tuple[int, ...]


def coefficient_sharing_groups(
    executors: Sequence[NativeCoefficientDAG], *, model_bindings: Mapping[str, Any]
) -> CoefficientSharingGroups:
    """Group exact schedules for output reuse within one evaluation call.

    Every graph retains its colour, multiplicity, polynomial-label and reduction
    metadata. This helper never reads or retains numerical event values.
    """
    representatives, representative_for, seen = [], [], {}
    for index, executor in enumerate(executors):
        key = executor.sharing_key(model_bindings)
        representative = seen.get(key)
        if representative is None:
            representative = seen[key] = index
            representatives.append(index)
        representative_for.append(representative)
    return CoefficientSharingGroups(tuple(representatives), tuple(representative_for))
