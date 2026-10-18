# SPDX-License-Identifier: 0BSD
"""Internal native bare-process assembly over verified graph families.

This physical control supports on-shell Dirac and massless vector states.
Local currents, explicit perturbative-order pairs, exact SU(3)
bilinears and native family maps determine every numerator. UV/R2 completion,
formal-Nc projection and public process dispatch remain separate boundaries.
Supplied momenta must satisfy the model's external masses. This internal control
preserves them; a general tolerance and mass-consistency gate is not yet provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np

from .color_bilinears import external_color_tensor
from .exact import require_exact_numbers
from .graph_basis import (
    build_color_loop_recurrence,
    build_color_tree_recurrence,
    routed_edge_denominator,
)
from .graphs import build_loop_process
from .kernels import (
    CoefficientSharingGroups,
    NativeCoefficientDAG,
    coefficient_sharing_groups,
    require_finite_batch,
)
from .kinematics import build_loop_kinematics
from .physical import _saved
from .physical_orders import (
    OrderedBareProcessResult,
    PhysicalOrderLayout,
    SummedBareValues,
)
from .recurrence import momentum_component_library


@dataclass(frozen=True, slots=True)
class BareProcessResult:
    """Batch-first decimal pairs; poles ordered finite, single, double."""

    born: Any
    virtual: Any
    statuses: tuple[tuple[tuple[int, str], ...], ...]
    family_indices: tuple[int, ...]
    arithmetic: str
    precision: int | None
    reduction_engine: str
    coefficient_engine: str
    color_accuracy: str = "full"
    born_color_degree: int | None = None
    virtual_color_degree: int | None = None
    rational: Any = None
    convention: str = "MadLoop-Cepsilon-bare-four-dimensional-no-R2-no-UV"


@dataclass(slots=True)
class NativePhysicalBatch:
    """Shared physical states and global model inputs for bare/CT assembly."""

    size: int
    rows: int
    point_ids: list[int]
    lane_ids: list[int]
    bound: Any
    positions: dict[str, int]
    model_arguments: Any
    scales_squared: Any
    helicity_count: int
    model_snapshot: Any = None
    spin_legs: tuple[int, ...] = ()


@dataclass(slots=True)
class NativeBareEvaluation:
    batch: NativePhysicalBatch
    born_amplitudes: Any
    born: Any
    virtual: Any
    statuses: tuple[tuple[tuple[int, str], ...], ...]
    rational: Any = None
    requested_point_ids: tuple[int, ...] = ()
    failures: tuple[Any, ...] = ()


@dataclass(slots=True)
class NativeBareCurrents:
    """One native physical batch shared by every bare contraction direction."""

    batch: NativePhysicalBatch
    born: Any
    loops: tuple[Any, ...]
    model: Any
    momenta_pairs: Any


def _physical_point_ids(momenta, scales, legs, point_ids):
    """Validate the original batch before any optional compact gather."""
    if momenta.ndim != 3 or momenta.shape[1:] != (legs, 4):
        raise ValueError("expected (batch, leg, four components) momenta")
    size = len(momenta)
    if scales.shape != (size,):
        raise ValueError("physical scale shape differs")
    point_ids = list(range(size)) if point_ids is None else list(point_ids)
    if (
        len(point_ids) != size
        or any(type(point) is not int or point < 0 for point in point_ids)
        or len(set(point_ids)) != size
    ):
        raise ValueError("physical point IDs must be distinct nonnegative integers")
    return point_ids


def _evaluate_coefficient_groups(bound, executors, columns, groups):
    """Execute exact representatives and retain every original graph position."""
    outputs = {
        index: executors[index].evaluate_batch(bound.take_columns(columns[index]))
        for index in groups.representatives
    }
    return [outputs[index] for index in groups.representative_for]


def _external_color(graph, expression):
    """Identify color ports by physical leg before comparing distinct graphs."""
    from symbolica.community import hepkit as h

    replacements = {}
    indices = {axis.index for axis in expression.axes}
    for edge in graph.native.external_edges:
        half = next(half for half in graph.native.half_edges if half.edge.id == edge.id)
        label = h.Symbols.half_edge()(half.id, 1)
        if label in indices:
            replacements[label] = f"physical_color_leg_{edge.external_index}"
    return expression.rename_indices(replacements).to_tensor().to_dense()


def _color_bilinear(left, right, average):
    from symbolica import E

    if set(left.axes) != set(right.axes):
        raise ValueError("physical Born/loop color ports disagree")
    right = right.permute_axes([right.axes.index(axis) for axis in left.axes])
    return sum((left[i].conj() * right[i] for i in range(len(left))), E("0")) / average


@dataclass(slots=True)
class _FamilyProgram:
    index: int
    graph_indices: tuple[int, ...]
    sector_indices: tuple[int, ...]
    powers: tuple[tuple[int, ...], ...]
    kinematics: Any
    exact_evaluator: Any
    masses: Any
    mass_parameters: tuple[Any, ...]
    mass_columns: tuple[int, ...]
    projector: Any
    reducer: Any
    rational_reducer: Any = None


def _reduction_numerators(powers, names, *, rational=False):
    """Translate closed (q[4], mu2, epsilon) labels to reducer sectors.

    Epsilon does not contribute to tensor rank. The caller has already applied
    its provider's closed-trace parity projection; open currents never enter
    this boundary. N4 and its dimensional remainder are disjoint selectors.
    """
    sectors = [[], [], []]
    for name, power in zip(names, powers, strict=True):
        if len(power) != 6 or any(type(p) is not int or p < 0 for p in power):
            raise ValueError("expected nonnegative q[4], mu2, epsilon powers")
        *momentum, mu2, epsilon = power
        if epsilon > 2:
            raise NotImplementedError("numerator epsilon power exceeds two")
        if bool(mu2 or epsilon) != rational:
            continue
        factors = [name]
        factors.extend(
            f"{variable}^{exponent}"
            for variable, exponent in zip(
                ("q0", "q1", "q2", "q3", "mu2"), (*momentum, mu2), strict=True
            )
            if exponent
        )
        sectors[epsilon].append("*".join(factors))
    return ["+".join(sector) or "0" for sector in sectors]


class NativeBareProcessProgram:
    """Reusable graph/family assembly, with global exact model input formulas.

    ``model_expressions`` maps native model symbol names to exact expressions
    in ``model_inputs``. Alternatively ``model_state`` supplies checked native
    source outputs from one immutable snapshot. A source scale view evaluates
    those outputs at each point's mu_r2; other model inputs remain global.
    The caller must supply on-shell momenta with the declared model masses;
    native wavefunctions use those four-vectors without repairing their shell.
    ``dimensional_numerators`` requires a provider for every selected loop. Its
    closed DAG supplies both N4 and the auxiliary remainder, which are reduced
    independently for diagnostics; no integrated-amplitude subtraction occurs.
    """

    def __init__(
        self,
        process,
        *,
        state_kinds: tuple[str, ...],
        initial_average: Any,
        model_expressions: dict[str, Any] | None = None,
        model_inputs: tuple[Any, ...] = (),
        model_state: Any = None,
        model_output_indices: dict[str, int] | None = None,
        model_scale_view: Any = None,
        vertex_spinor_frame: str | None = None,
        arithmetic: str = "double",
        precision: int | None = None,
        reduction_engine: str = "generic",
        coefficient_engine: str = "interpreted",
        dimensional_numerators: bool = False,
        cache_coefficient_kernels: bool = True,
        share_coefficient_schedules: bool = True,
        backend: Any = None,
    ):
        if backend is None:
            import _loop_reduction as backend
        from symbolica import E, S
        from symbolica.community import tensor as t

        self.order_layout = PhysicalOrderLayout.from_plan(process.orders)
        if dimensional_numerators and not self.order_layout.is_single:
            raise NotImplementedError(
                "multi-order dimensional assembly is not certified"
            )
        if any(
            kind not in {"u", "v", "u_bar", "v_bar", "epsilon", "epsilon_bar"}
            for kind in state_kinds
        ):
            raise NotImplementedError(
                "physical control requires Dirac or vector states"
            )
        require_exact_numbers(initial_average)
        self.model_state = model_state
        self.vertex_spinor_frame = vertex_spinor_frame
        self.model_scale_view = model_scale_view
        if model_scale_view is not None and (
            model_state is None
            or model_scale_view.global_state is not model_state
            or model_scale_view.plan is not model_state.plan
        ):
            raise ValueError(
                "ordinary scale view must share the source model state plan"
            )
        if model_state is not None:
            if model_expressions is not None or model_inputs:
                raise ValueError("source model state replaces legacy model formulas")
            if model_state.backend is not backend or model_state.options != {
                "arithmetic": arithmetic,
                "precision": precision,
            }:
                raise ValueError("source model state backend or arithmetic differs")
            if not model_output_indices:
                raise ValueError("source model outputs require native symbol bindings")
            if any(
                not isinstance(name, str)
                or type(index) is not int
                or not 0 <= index < len(model_state.plan.outputs)
                for name, index in model_output_indices.items()
            ):
                raise ValueError("invalid source model output binding")
            model_inputs = model_state.plan.input_symbols
            model_expressions = {
                name: model_state.plan.expressions[index].value
                for name, index in model_output_indices.items()
            }
        elif model_output_indices is not None:
            raise ValueError("source model output bindings require their owner state")
        elif model_expressions is None:
            raise ValueError(
                "exact model formulas or a source model state are required"
            )
        self.model_expressions = MappingProxyType(dict(model_expressions))
        for expression in self.model_expressions.values():
            require_exact_numbers(expression)
        selected_loop_orders = {
            pair.loop for sector in process.orders.sectors for pair in sector.pairs
        }
        selected_loop_ids = {
            contribution.id
            for contribution in process.orders.contributions
            if contribution.kind == "loop"
        }
        self.loop_graphs = tuple(
            graph
            for graph in process.loops
            if graph.id in selected_loop_ids and graph.order in selected_loop_orders
        )
        self.process, self.backend = process, backend
        self.external_masses = tuple(
            edge.particle.mass
            for edge in sorted(
                process.born[0].native.external_edges,
                key=lambda edge: edge.external_index,
            )
        )
        if any(
            kind in {"epsilon", "epsilon_bar"} and mass != 0
            for kind, mass in zip(state_kinds, self.external_masses, strict=True)
        ):
            raise NotImplementedError(
                "physical vector control currently requires massless states"
            )
        self.initial_average = initial_average
        self.state_kinds = state_kinds
        self.helicities = tuple(product((-1, 1), repeat=len(state_kinds)))
        self.arithmetic, self.precision = arithmetic, precision
        self.reduction_engine = reduction_engine
        self.dimensional_numerators = dimensional_numerators
        self.options = {"arithmetic": arithmetic, "precision": precision}
        self.cache_coefficient_kernels = cache_coefficient_kernels
        self.share_coefficient_schedules = share_coefficient_schedules
        self.coefficient_kernel_cache = (
            backend.CoefficientKernelCache(engine=coefficient_engine, **self.options)
            if cache_coefficient_kernels
            else None
        )
        self.model_inputs = model_inputs
        self.scale_squared_symbol = S("loop_physical_process::mu_r2")
        self.model_names = tuple(sorted(self.model_expressions))
        self.model_positions = {name: i for i, name in enumerate(self.model_names)}
        self.model_output_indices = (
            tuple(model_output_indices[name] for name in self.model_names)
            if model_state is not None
            else ()
        )
        self.model_evaluator = (
            backend.CoefficientEvaluator(
                _saved(
                    [self.model_expressions[name] for name in self.model_names],
                    model_inputs,
                ),
                **self.options,
            )
            if model_state is None
            else None
        )
        if dimensional_numerators:
            from .direct_kernels import dimensional_kernels

        self.recurrences = tuple(
            build_color_loop_recurrence(
                graph,
                routing="family",
                kernel_provider=dimensional_kernels(graph)
                if dimensional_numerators
                else None,
                vertex_spinor_frame=vertex_spinor_frame,
            )
            for graph in self.loop_graphs
        )
        if not self.recurrences:
            raise ValueError("physical process has no selected loop graphs")
        representative = self.recurrences[0]
        self.momenta = representative.momentum_parameters
        self.momentum_parameters = tuple(x for vector in self.momenta for x in vector)
        if len(self.momenta) != len(state_kinds):
            raise ValueError("physical state count differs from native process")
        self.external_states = representative.external_states
        legs = len(state_kinds)
        self.bound_positions = {
            p.get_name(): i for i, p in enumerate(self.momentum_parameters)
        }
        self.bound_positions.update(
            {
                symbol.get_name(): 4 * legs + 4 * leg + component
                for leg, symbols in self.external_states
                for component, symbol in enumerate(symbols)
            }
        )
        self.bound_positions.update(
            {name: 8 * legs + i for i, name in enumerate(self.model_names)}
        )
        self.bound_columns = 8 * legs + len(self.model_names)
        variables = representative.dag.momentum_variables
        self.born_recurrences = tuple(
            build_color_tree_recurrence(
                graph,
                variables=variables,
                momenta=self.momenta,
                vertex_spinor_frame=vertex_spinor_frame,
            )
            for graph in process.born
        )
        self.born_dags = tuple(item.dag for item in self.born_recurrences)
        self.born_term_graphs = tuple(
            graph_index
            for graph_index, item in enumerate(self.born_recurrences)
            for _ in item.terms
        )
        for item in self.born_recurrences:
            if any(
                len(item.dag.nodes[term.node].output_labels) != 1
                or any(item.dag.nodes[term.node].output_labels[0].powers)
                for term in item.terms
            ):
                raise ValueError("Born color term must have one scalar coefficient")
        self.born_color_tensors = tuple(
            external_color_tensor(item.graph, term.color)
            for item in self.born_recurrences
            for term in item.terms
        )
        born_colors = tuple(
            _external_color(item.graph, term.color)
            for item in self.born_recurrences
            for term in item.terms
        )
        self.born_colors = born_colors
        self.loop_finals, loop_colors, self.loop_term_offsets = [], [], []
        self.loop_color_tensors = []
        for recurrence in self.recurrences:
            if recurrence.momentum_parameters != self.momenta:
                raise ValueError("native graph external momentum frames differ")
            library = momentum_component_library(
                recurrence.graph, (recurrence.loop_components,), self.momenta
            )
            bridge = E("1")
            for edge in recurrence.graph.native.internal_edges:
                if edge.id in recurrence.graph.loop_edges:
                    continue
                denominator = routed_edge_denominator(edge, library)
                if any(denominator.contains(q) for q in variables):
                    raise ValueError("tree bridge denominator depends on loop momentum")
                bridge /= denominator
            self.loop_term_offsets.append(len(loop_colors))
            self.loop_finals.append(
                tuple(
                    recurrence.dag.add(
                        "physical-tree-denominators",
                        t.TensorExpression(bridge).to_tensor(library),
                        (term.node,),
                    )
                    for term in recurrence.terms
                )
            )
            for term in recurrence.terms:
                loop_colors.append(_external_color(recurrence.graph, term.color))
                self.loop_color_tensors.append(
                    external_color_tensor(recurrence.graph, term.color)
                )
        self.born_executors = tuple(
            NativeCoefficientDAG(
                item.dag,
                output_nodes=item.final_nodes,
                backend=backend,
                coefficient_engine=coefficient_engine,
                kernel_cache=self.coefficient_kernel_cache,
                **self.options,
            )
            for item in self.born_recurrences
        )
        self.born_term_orders = tuple(
            item.graph.order for item in self.born_recurrences for _ in item.terms
        )
        self.loop_executors = tuple(
            NativeCoefficientDAG(
                recurrence.dag,
                output_nodes=finals,
                backend=backend,
                coefficient_engine=coefficient_engine,
                kernel_cache=self.coefficient_kernel_cache,
                **self.options,
            )
            for recurrence, finals in zip(
                self.recurrences, self.loop_finals, strict=True
            )
        )
        groups = tuple(
            coefficient_sharing_groups(executors, model_bindings=self.model_expressions)
            if share_coefficient_schedules
            else CoefficientSharingGroups(
                tuple(range(len(executors))), tuple(range(len(executors)))
            )
            for executors in (self.born_executors, self.loop_executors)
        )
        self.born_sharing, self.loop_sharing = groups
        self.born_input_columns = tuple(
            tuple(self.bound_positions[p.get_name()] for p in executor.parameters)
            for executor in self.born_executors
        )
        self.loop_input_columns = tuple(
            tuple(self.bound_positions[p.get_name()] for p in executor.parameters)
            for executor in self.loop_executors
        )
        if dimensional_numerators:
            from .direct_kernels import closed_reduction_powers

        self.loop_reduction_powers = tuple(
            tuple(
                closed_reduction_powers(recurrence, node=node)
                if dimensional_numerators
                else tuple(
                    (*label.powers, 0, 0)
                    for label in recurrence.dag.nodes[node].output_labels
                )
                for node in finals
            )
            for recurrence, finals in zip(
                self.recurrences, self.loop_finals, strict=True
            )
        )
        self.coefficient_engine = self.born_executors[0].coefficient_engine
        self.born_metric = tuple(
            tuple(
                _color_bilinear(left, right, initial_average) for right in born_colors
            )
            for left in born_colors
        )
        self.loop_metric = tuple(
            tuple(
                _color_bilinear(left, right, initial_average) for right in loop_colors
            )
            for left in born_colors
        )
        self._compile_born_projection()
        self.families = []
        for family in process.families:
            indices = tuple(
                i
                for i, graph in enumerate(self.loop_graphs)
                if graph.family.family_index == family.index
            )
            if not indices:
                continue
            sector_indices = tuple(
                sector_index
                for sector_index, sector in enumerate(self.order_layout.virtual)
                if any(
                    self.loop_graphs[index].order == pair.loop
                    for index in indices
                    for pair in sector.pairs
                )
            )
            plans = [
                build_loop_kinematics(self.recurrences[i], family_order=True)
                for i in indices
            ]
            plan = plans[0]
            if any(
                (p.shifts, p.masses_squared, p.s_matrix)
                != (plan.shifts, plan.masses_squared, plan.s_matrix)
                for p in plans[1:]
            ):
                raise ValueError(
                    "native family graphs have different physical denominator bindings"
                )
            powers = tuple(
                sorted(
                    {
                        power
                        for i in indices
                        for term in self.loop_reduction_powers[i]
                        for power in term
                        if power is not None
                    }
                )
            )
            if max(sum(p[:4]) + 2 * p[4] for p in powers) > plan.propagators + 1:
                raise NotImplementedError("numerator rank exceeds the reducer domain")
            names = [f"c{i}" for i in range(len(powers))]
            rational_numerators = _reduction_numerators(powers, names, rational=True)
            mass_parameters = tuple(
                sorted(
                    {
                        symbol
                        for mass in plan.masses_squared
                        for symbol in mass.get_all_symbols(False)
                    },
                    key=lambda s: s.to_canonical_string(),
                )
            )
            self.families.append(
                _FamilyProgram(
                    family.index,
                    indices,
                    sector_indices,
                    powers,
                    plan,
                    backend.ExactPolynomialEvaluator(plan.evaluator_bytes()),
                    backend.CoefficientEvaluator(
                        _saved(plan.masses_squared, mass_parameters), **self.options
                    ),
                    mass_parameters,
                    tuple(self.model_positions[p.get_name()] for p in mass_parameters),
                    self._compile_family_projection(indices, powers, sector_indices),
                    backend.PolynomialPlan(
                        plan.propagators,
                        _reduction_numerators(powers, names),
                        names,
                        engine=reduction_engine,
                        **self.options,
                    ),
                    backend.PolynomialPlan(
                        plan.propagators,
                        rational_numerators,
                        names,
                        engine=reduction_engine,
                        **self.options,
                    )
                    if rational_numerators != ["0", "0", "0"]
                    else None,
                )
            )
        self.order_channels = tuple(
            (sector, family.index)
            for family in self.families
            for sector in family.sector_indices
        )
        poles = {
            (sector, family): [
                S(f"loop_physical_process::family{family}_order{sector}_pole{p}")
                for p in range(3)
            ]
            for sector, family in self.order_channels
        }
        values = [
            E("1i")
            * sum(
                (row[p] for (order, _), row in poles.items() if order == sector),
                E("0"),
            )
            / (16 * E("pi") ** 2)
            for sector in range(len(self.order_layout.virtual))
            for p in range(3)
        ]
        self.postprocess = backend.CoefficientEvaluator(
            _saved(
                [value + value.conj() for value in values],
                [p for row in poles.values() for p in row],
            ),
            **self.options,
        )
        self.zero_poles = backend.CoefficientEvaluator(
            _saved([E("0")] * 3, []), **self.options
        )
        born_orders = [
            S(f"loop_physical_process::born_order{i}")
            for i in range(len(self.order_layout.born))
        ]
        virtual_orders = [
            [S(f"loop_physical_process::virtual_order{i}_pole{p}") for p in range(3)]
            for i in range(len(self.order_layout.virtual))
        ]
        self.sum_order_values = backend.CoefficientEvaluator(
            _saved(
                [sum(born_orders, E("0"))]
                + [sum((row[p] for row in virtual_orders), E("0")) for p in range(3)],
                [*born_orders, *(p for row in virtual_orders for p in row)],
            ),
            **self.options,
        )

    def _born_symbols(self, *, helicities=None):
        from symbolica import S

        helicities = self.helicities if helicities is None else helicities
        return [
            [
                S(f"loop_physical_process::born{helicity}_{i}")
                for i in range(len(self.born_colors))
            ]
            for helicity in range(len(helicities))
        ]

    def _compile_born_projection(self):
        self.born_projector = self._make_born_projector((self.born_metric,))

    def _make_born_projector(self, metrics, *, helicities=None):
        from symbolica import E

        symbols = self._born_symbols(helicities=helicities)
        values = [
            sum(
                (
                    left.conj() * right * metric[i][j]
                    for row in symbols
                    for i, left in enumerate(row)
                    for j, right in enumerate(row)
                    if (self.born_term_orders[i], self.born_term_orders[j])
                    in sector.pairs
                ),
                E("0"),
            )
            for metric in metrics
            for sector in self.order_layout.born
        ]
        return self.backend.CoefficientEvaluator(
            _saved(values, [value for row in symbols for value in row]), **self.options
        )

    def _compile_family_projection(
        self, indices, powers, sector_indices, *, metrics=None, helicities=None
    ):
        from symbolica import E, S

        metrics = (self.loop_metric,) if metrics is None else metrics
        projected = {
            (direction, sector, power): E("0")
            for direction in range(len(metrics))
            for sector in sector_indices
            for power in powers
        }
        pairs = {
            sector: {
                (pair.born, pair.loop)
                for pair in self.order_layout.virtual[sector].pairs
            }
            for sector in sector_indices
        }
        parameters = []
        for helicity, born in enumerate(self._born_symbols(helicities=helicities)):
            parameters.extend(born)
            for index in indices:
                for local_term, labels in enumerate(self.loop_reduction_powers[index]):
                    term_index = self.loop_term_offsets[index] + local_term
                    for column, power in enumerate(labels):
                        coefficient = S(
                            f"loop_physical_process::coefficient{helicity}_{index}_{local_term}_{column}"
                        )
                        parameters.append(coefficient)
                        if power is None:
                            continue
                        for direction, metric in enumerate(metrics):
                            for sector in sector_indices:
                                projected[direction, sector, power] += sum(
                                    (
                                        value.conj()
                                        * metric[b][term_index]
                                        * coefficient
                                        for b, value in enumerate(born)
                                        if (
                                            self.born_term_orders[b],
                                            self.loop_graphs[index].order,
                                        )
                                        in pairs[sector]
                                    ),
                                    E("0"),
                                )
        return self.backend.CoefficientEvaluator(
            _saved(
                [
                    projected[direction, sector, power]
                    for direction in range(len(metrics))
                    for sector in sector_indices
                    for power in powers
                ],
                parameters,
            ),
            **self.options,
        )

    def set_color_metrics(self, born_metric, loop_metric):
        """Recompile only physical weight evaluators after a shared projection.

        The supplied matrices include averaging and are ready for arithmetic;
        graph currents, model evaluators, kinematics and reducers are retained.
        """
        if getattr(self, "_model_color_metrics_bound", False):
            raise ValueError(
                "bare color metrics are already bound to a model program; "
                "construct a fresh bare program before changing color projection"
            )
        if getattr(self, "_directed_color_metrics_bound", False):
            raise ValueError(
                "bare color metrics are already bound to a directed program"
            )
        if (
            len(born_metric) != len(self.born_colors)
            or any(len(row) != len(self.born_colors) for row in born_metric)
            or len(loop_metric) != len(self.born_colors)
            or any(len(row) != len(self.loop_color_tensors) for row in loop_metric)
        ):
            raise ValueError("physical color metric dimensions differ")
        for row in (*born_metric, *loop_metric):
            for value in row:
                require_exact_numbers(value)
        self.born_metric, self.loop_metric = (
            tuple(map(tuple, born_metric)),
            tuple(map(tuple, loop_metric)),
        )
        self._compile_born_projection()
        for family in self.families:
            family.projector = self._compile_family_projection(
                family.graph_indices, family.powers, family.sector_indices
            )
        self._color_metrics_revision = getattr(self, "_color_metrics_revision", 0) + 1

    def evaluate_decimal(self, momenta, scales_squared, *, model_values=None):
        result = self.evaluate_native(
            momenta, scales_squared, model_values=model_values
        )
        return BareProcessResult(
            np.asarray(result.born.to_decimal(), dtype=object).reshape(
                result.batch.size, 2
            ),
            np.asarray(result.virtual.to_decimal(), dtype=object).reshape(
                result.batch.size, 3, 2
            ),
            result.statuses,
            tuple(f.index for f in self.families),
            self.arithmetic,
            self.precision,
            self.reduction_engine,
            self.coefficient_engine,
            getattr(self, "color_accuracy", "full"),
            getattr(self, "born_color_degree", None),
            getattr(self, "virtual_color_degree", None),
            None
            if result.rational is None
            else np.asarray(result.rational.to_decimal(), dtype=object).reshape(
                result.batch.size, 3, 2
            ),
        )

    def require_single_order(self, context):
        if not self.order_layout.is_single:
            raise NotImplementedError(
                f"{context} requires one Born squared order and one virtual squared "
                "order; use ordered bare evaluation for multiple sectors"
            )

    def evaluate_native(self, momenta, scales_squared, *, model_values=None):
        """Single-sector adapter with an explicit check before removing order axes."""
        self.require_single_order("single-sector physical adapter")
        return self.evaluate_ordered_native(
            momenta, scales_squared, model_values=model_values
        )

    def evaluate_ordered_decimal(self, momenta, scales_squared, *, model_values=None):
        result = self.evaluate_ordered_native(
            momenta, scales_squared, model_values=model_values
        )
        size = result.batch.size

        def values(batch, *shape):
            return np.asarray(batch.to_decimal(), dtype=object).reshape(size, *shape, 2)

        totals = self.sum_order_values.evaluate_batch(
            self.backend.NativeBatch.concat_columns([result.born, result.virtual])
        )
        require_finite_batch(totals, "sum over perturbative orders")
        summed = values(totals, 4)
        rational = None
        summed_rational = None
        if result.rational is not None:
            rational = values(result.rational, len(self.order_layout.virtual), 3)
            # Dimensional assembly currently has one explicit sector.
            summed_rational = rational[:, 0]
        return OrderedBareProcessResult(
            values(result.born, len(self.order_layout.born)),
            values(result.virtual, len(self.order_layout.virtual), 3),
            self.order_layout,
            self.order_channels,
            result.statuses,
            self.arithmetic,
            self.precision,
            self.reduction_engine,
            self.coefficient_engine,
            SummedBareValues(summed[:, 0], summed[:, 1:], summed_rational),
            rational,
            convention="MadLoop-Cepsilon-separated-N4-and-direct-R2-no-UV"
            if self.dimensional_numerators
            else "MadLoop-Cepsilon-bare-four-dimensional-no-R2-no-UV",
        )

    def prepare_currents_native(
        self,
        momenta,
        scales_squared,
        *,
        model_values=None,
        helicities=None,
        spin_sources=None,
        point_ids=None,
        _failures=None,
    ):
        """Bind one physical batch and evaluate its Born and loop currents once."""
        momenta, scales = (
            np.asarray(momenta, dtype=object),
            np.asarray(scales_squared, dtype=object),
        )
        helicity_states = self.helicities if helicities is None else helicities
        legs, helicities = len(self.state_kinds), len(helicity_states)
        point_ids = _physical_point_ids(momenta, scales, legs, point_ids)
        size, rows = len(momenta), len(momenta) * helicities
        snapshot = None
        if self.model_state is None:
            if model_values is None or len(model_values) != len(self.model_inputs):
                raise ValueError("global model input shape differs")
            decimal_model_values = model_values
        else:
            if model_values is not None:
                raise ValueError(
                    "update the source model state through its named inputs"
                )
            decimal_model_values = ()
        if any(
            not isinstance(value, str)
            for value in [*momenta.flat, *scales.flat, *decimal_model_values]
        ):
            raise TypeError("physical precise inputs must be decimal strings")
        batch_type = self.backend.NativeBatch

        def pairs(values):
            values = np.asarray(values, dtype=object).reshape(-1)
            return np.stack(
                [values, np.full(len(values), "0", dtype=object)], axis=1
            ).tolist()

        def native(values, nrows, columns, row_ids=None):
            constructor = (
                batch_type.from_decimal
                if _failures is None
                else batch_type.from_exact_decimal
            )
            return constructor(
                pairs(values), nrows, columns, row_ids=row_ids, **self.options
            )

        lane_ids = [
            point * helicities + h for point in point_ids for h in range(helicities)
        ]
        momenta_pairs = pairs(momenta)
        physical = native(momenta, size, 4 * legs, row_ids=point_ids)
        lanes = physical.take_rows(
            np.repeat(np.arange(size), helicities).tolist(), row_ids=lane_ids
        )
        state_ids = [lane * legs + leg for lane in lane_ids for leg in range(legs)]
        state_inputs = lanes.reshape_rows(rows * legs, 4, state_ids)
        state_helicities = (
            np.tile(np.asarray(helicity_states), (size, 1)).reshape(-1).tolist()
        )
        state_kinds = list(self.state_kinds) * rows
        native_scales = None
        if _failures is None:
            states = self.backend.wavefunctions_batch(
                state_inputs, state_helicities, state_kinds
            )
        else:
            # Validate all original scale literals before dropping any point.
            native_scales = native(scales, size, 1, row_ids=point_ids)
            _failures.require_finite(native_scales, "renormalization scale ingress")
            states, state_statuses = self.backend.wavefunctions_partial_batch(
                state_inputs, state_helicities, state_kinds
            )
            for state_id, status in state_statuses:
                if status != "success":
                    _failures.record(
                        state_id // (helicities * legs),
                        "external state preparation",
                        status,
                    )
            indices = _failures.successful_indices(point_ids)
            if len(indices) != size:
                lane_rows = [
                    point * helicities + h
                    for point in indices
                    for h in range(helicities)
                ]
                point_ids = [point_ids[point] for point in indices]
                lane_ids = [lane_ids[lane] for lane in lane_rows]
                size, rows = len(indices), len(lane_rows)
                momenta_pairs = pairs(momenta[indices])
                lanes = lanes.take_rows(lane_rows)
                native_scales = native_scales.take_rows(indices)
                positions = {
                    state_id: row for row, state_id in enumerate(states.row_ids)
                }
                states = states.take_rows(
                    [
                        positions[lane * legs + leg]
                        for lane in lane_ids
                        for leg in range(legs)
                    ]
                )
        states = states.reshape_rows(rows, 4 * legs, lane_ids)
        if spin_sources:
            from .spin_sources import replace_vector_sources

            states = replace_vector_sources(
                states,
                spin_sources,
                point_rows=np.repeat(np.arange(size), helicities).tolist(),
                lane_ids=lane_ids,
                state_kinds=self.state_kinds,
                backend=self.backend,
            )
        scales = (
            native(scales, size, 1, row_ids=point_ids)
            if native_scales is None
            else native_scales
        )
        if self.model_state is None:
            model_arguments = native(model_values, 1, len(model_values))
            model = self.model_evaluator.evaluate_batch(model_arguments)
            require_finite_batch(model, "global model formulas")
        else:
            snapshot = self.model_state.snapshot()
            model_arguments = snapshot.arguments
            selected = (
                self.model_scale_view.evaluate_native(scales, snapshot=snapshot)
                if self.model_scale_view is not None
                else snapshot.values
            )
            model = selected.take_columns(self.model_output_indices)
        model_rows = (
            np.repeat(np.arange(size), helicities).tolist()
            if self.model_scale_view is not None
            else [0] * rows
        )
        bound = batch_type.concat_columns(
            [lanes, states, model.take_rows(model_rows, row_ids=lane_ids)]
        )
        born = batch_type.concat_columns(
            _evaluate_coefficient_groups(
                bound,
                self.born_executors,
                self.born_input_columns,
                self.born_sharing,
            )
        )
        currents = _evaluate_coefficient_groups(
            bound, self.loop_executors, self.loop_input_columns, self.loop_sharing
        )
        return NativeBareCurrents(
            NativePhysicalBatch(
                size,
                rows,
                point_ids,
                lane_ids,
                bound,
                dict(self.bound_positions),
                model_arguments,
                scales,
                helicities,
                snapshot,
                tuple(sorted(spin_sources or {})),
            ),
            born,
            tuple(currents),
            model,
            momenta_pairs,
        )

    def family_kinematics_native(self, currents, family):
        """Exact ingress invariants and global or source scale-dependent masses."""
        bindings = family.exact_evaluator.evaluate_decimal(
            currents.momenta_pairs,
            currents.batch.size,
            row_ids=currents.batch.point_ids,
            **self.options,
        )
        n = family.kinematics.propagators
        return (
            bindings.take_columns(list(range(4 * n))),
            family.masses.evaluate_batch(
                currents.model.take_columns(family.mass_columns)
            ),
            bindings.take_columns(list(range(4 * n, 4 * n + n * n))),
        )

    def family_projection_arguments(self, currents, family):
        inputs = self.backend.NativeBatch.concat_columns(
            [currents.born, *(currents.loops[i] for i in family.graph_indices)]
        )
        return inputs.reshape_rows(
            currents.batch.size,
            currents.batch.helicity_count * inputs.columns,
            currents.batch.point_ids,
        )

    def evaluate_ordered_partial_native(
        self, momenta, scales_squared, *, model_values=None, point_ids=None
    ):
        """Return only successful native points, with explicit failure metadata.

        This is a numerical-failure boundary, not an accuracy or rescue policy.
        Original IDs may be supplied for a compacted retry batch.
        """
        from .numerical_lanes import LaneFailures, require_partial_domain

        require_partial_domain(self)
        requested_ids = (
            tuple(range(len(momenta))) if point_ids is None else tuple(point_ids)
        )
        failures = LaneFailures()
        currents = self.prepare_currents_native(
            momenta,
            scales_squared,
            model_values=model_values,
            point_ids=requested_ids,
            _failures=failures,
        )
        result = self._evaluate_currents_native(currents, failures)
        result.requested_point_ids = requested_ids
        return result

    def evaluate_ordered_native(self, momenta, scales_squared, *, model_values=None):
        """Original decimal input enters exact kinematics and native arithmetic."""
        currents = self.prepare_currents_native(
            momenta, scales_squared, model_values=model_values
        )
        return self._evaluate_currents_native(currents)

    def _evaluate_currents_native(self, currents, failures=None):
        from .numerical_lanes import select_bare_currents

        batch_type = self.backend.NativeBatch
        requested_ids = tuple(currents.batch.point_ids)
        size, point_ids = currents.batch.size, currents.batch.point_ids
        helicities = currents.batch.helicity_count
        born_squared = self.born_projector.evaluate_batch(
            currents.born.reshape_rows(
                size, helicities * currents.born.columns, point_ids
            )
        )
        integrated, rational_integrated, status_maps = [], [], []
        guard = require_finite_batch if failures is None else failures.require_finite
        guard(born_squared, "Born projection")

        def compact(*point_buffers):
            nonlocal currents, born_squared, size, point_ids
            if failures is None:
                return point_buffers
            indices = failures.successful_indices(point_ids)
            if len(indices) == size:
                return point_buffers
            currents = select_bare_currents(currents, indices)
            born_squared = born_squared.take_rows(indices)
            integrated[:] = [value.take_rows(indices) for value in integrated]
            rational_integrated[:] = [
                value.take_rows(indices) for value in rational_integrated
            ]
            size, point_ids = currents.batch.size, currents.batch.point_ids
            return tuple(value.take_rows(indices) for value in point_buffers)

        if failures is not None:
            guard(currents.born, "Born current", rows_per_point=helicities)
            for index, current in enumerate(currents.loops):
                guard(current, f"loop current {index}", rows_per_point=helicities)
            compact()
        for family in self.families:
            coefficients = family.projector.evaluate_batch(
                self.family_projection_arguments(currents, family)
            )
            if failures is not None:
                guard(coefficients, f"family {family.index} numerator projection")
                (coefficients,) = compact(coefficients)
            shifts, masses, matrix = self.family_kinematics_native(currents, family)
            if failures is not None:
                guard(shifts, f"family {family.index} shifts")
                guard(matrix, f"family {family.index} invariant matrix")
                # Built-in masses are global. Invalid global model formulas were
                # already rejected at ingress and cannot be rescued pointwise.
                require_finite_batch(masses, f"family {family.index} global masses")
                coefficients, shifts, matrix = compact(coefficients, shifts, matrix)
            channel_count = len(family.sector_indices)
            row_ids = [
                point * len(self.order_layout.virtual) + sector
                for point in point_ids
                for sector in family.sector_indices
            ]
            point_rows = np.repeat(np.arange(size), channel_count).tolist()
            coefficients = coefficients.reshape_rows(
                size * channel_count, len(family.powers), row_ids
            )
            shifts = shifts.take_rows(point_rows, row_ids=row_ids)
            matrix = matrix.take_rows(point_rows, row_ids=row_ids)
            masses = masses.take_rows(
                point_rows if masses.rows == size else [0] * (size * channel_count),
                row_ids=row_ids,
            )
            channel_scales = currents.batch.scales_squared.take_rows(
                point_rows, row_ids=row_ids
            )
            reduced = family.reducer.evaluate_batch(
                shifts, masses, channel_scales, coefficients, s_matrix=matrix
            )
            bad = [
                (*divmod(row, len(self.order_layout.virtual)), family.index, status)
                for row, status in reduced.statuses
                if status != "success"
            ]
            if bad and failures is None:
                raise ArithmeticError(
                    "native process reduction failed (point, order, family, status): "
                    f"{bad}"
                )
            if failures is not None:
                for point, order, index, status in bad:
                    failures.record(
                        point,
                        "loop reduction",
                        status,
                        order_index=order,
                        family_index=index,
                    )
                values = reduced.successful_values()
                for row in values.nonfinite_rows():
                    point, order = divmod(row, len(self.order_layout.virtual))
                    failures.record(
                        point,
                        "loop reduction",
                        "nonfinite successful reduction",
                        order_index=order,
                        family_index=family.index,
                    )
                compact()
                positions = {row: i for i, row in enumerate(values.row_ids)}
                values = values.take_rows(
                    [
                        positions[point * len(self.order_layout.virtual) + sector]
                        for point in point_ids
                        for sector in family.sector_indices
                    ]
                )
            else:
                values = reduced.values
            integrated.append(values.reshape_rows(size, channel_count * 3, point_ids))
            status_by_row = dict(reduced.statuses)
            status_maps.extend(
                {
                    point: (
                        point,
                        status_by_row[point * len(self.order_layout.virtual) + sector],
                    )
                    for point in point_ids
                }
                for sector in family.sector_indices
            )
            if self.dimensional_numerators:
                if family.rational_reducer is None:
                    rational_integrated.append(
                        self.zero_poles.evaluate_batch(
                            currents.batch.scales_squared.take_columns([])
                        )
                    )
                else:
                    rational = family.rational_reducer.evaluate_batch(
                        shifts, masses, channel_scales, coefficients, s_matrix=matrix
                    )
                    bad = [
                        (point, family.index, status)
                        for point, status in rational.statuses
                        if status != "success"
                    ]
                    if bad:
                        raise ArithmeticError(
                            "native dimensional remainder reduction failed "
                            f"(point, family, status): {bad}"
                        )
                    rational_integrated.append(
                        rational.values.reshape_rows(size, channel_count * 3, point_ids)
                    )
        virtual = self.postprocess.evaluate_batch(batch_type.concat_columns(integrated))
        guard(virtual, "bare virtual assembly")
        rational = None
        if self.dimensional_numerators:
            rational = self.postprocess.evaluate_batch(
                batch_type.concat_columns(rational_integrated)
            )
            guard(rational, "direct rational assembly")
        (virtual,) = compact(virtual)
        return NativeBareEvaluation(
            currents.batch,
            currents.born,
            born_squared,
            virtual,
            tuple(
                tuple(family[point] for family in status_maps) for point in point_ids
            ),
            rational,
            requested_ids,
            () if failures is None else tuple(failures.entries),
        )


def distinct_flavour_qcd_program(**options):
    """Native six-flavour u ubar -> d dbar control; model values are alpha_s, mt."""
    return _qcd_fermion_program(("d", "d~"), **options)


def top_pair_qcd_program(**options):
    """Native u ubar -> t tbar bare control; inputs are alpha_s and top mass.

    All supplied four-vectors must already be on shell for that mass. Controlled
    decimal fixtures prove this exactly; no general input tolerance is implied.
    """
    return _qcd_fermion_program(("t", "t~"), **options)


def _qcd_fermion_program(outgoing, **options):
    return _qcd_process_program(
        ("u", "u~"), outgoing, ("u", "v_bar", "u_bar", "v"), 36, **options
    )


def gluon_scattering_qcd_program(**options):
    """Native gg -> gg control, including the identical-final-state factor 1/2."""
    return _qcd_process_program(
        ("g", "g"),
        ("g", "g"),
        ("epsilon", "epsilon", "epsilon_bar", "epsilon_bar"),
        512,
        **options,
    )


def quark_gluon_qcd_program(**options):
    """Native u ubar -> gg control, with the final identical-state factor 1/2."""
    return _qcd_process_program(
        ("u", "u~"),
        ("g", "g"),
        ("u", "v_bar", "epsilon_bar", "epsilon_bar"),
        72,
        **options,
    )


def _qcd_process_program(incoming, outgoing, state_kinds, average, **options):
    from symbolica import E, S
    from symbolica.community import hepkit as h

    model = h.Model.qcd_qed()
    process = build_loop_process(
        model.process(incoming, outgoing),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    alpha_s, top_mass = S("loop_model_input::alpha_s", "loop_model_input::top_mass")
    strong = (4 * E("pi") * alpha_s) ** E("1/2")
    expressions = {"UFO::MB": E("0"), "UFO::MC": E("0"), "UFO::MT": top_mass}
    for name in ("GC_10", "GC_11", "GC_12"):
        coupling = model.coupling(name)
        expressions[coupling.symbol.get_name()] = coupling.expression.replace(
            S("UFO::G"), strong
        )
    return NativeBareProcessProgram(
        process,
        state_kinds=state_kinds,
        initial_average=E(str(average)),
        model_expressions=expressions,
        model_inputs=(alpha_s, top_mass),
        **options,
    )
