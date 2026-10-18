# SPDX-License-Identifier: 0BSD
"""Internal native bare-process assembly over verified graph families.

This physical control supports on-shell Dirac and massless vector states,
one Born order and one virtual squared-order sector. Local currents, exact SU(3)
bilinears and native family maps determine every numerator. UV/R2 completion,
formal-Nc projection and public process dispatch remain separate boundaries.
Supplied momenta must satisfy the model's external masses. This internal control
preserves them; a general tolerance and mass-consistency gate is not yet provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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
from .kernels import NativeCoefficientDAG, require_finite_batch
from .kinematics import build_loop_kinematics
from .physical import _saved
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


@dataclass(slots=True)
class NativeBareEvaluation:
    batch: NativePhysicalBatch
    born_amplitudes: Any
    born: Any
    virtual: Any
    statuses: tuple[tuple[tuple[int, str], ...], ...]
    rational: Any = None


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
    in ``model_inputs``. Numerical values bind once per call, never per event.
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
        model_expressions: dict[str, Any],
        model_inputs: tuple[Any, ...],
        arithmetic: str = "double",
        precision: int | None = None,
        reduction_engine: str = "generic",
        coefficient_engine: str = "interpreted",
        dimensional_numerators: bool = False,
        backend: Any = None,
    ):
        if backend is None:
            import _loop_reduction as backend
        from symbolica import E, S
        from symbolica.community import tensor as t

        if len(process.orders.born_orders) != 1 or len(process.orders.sectors) != 1:
            raise NotImplementedError("physical control requires one Born/order sector")
        if any(
            kind not in {"u", "v", "u_bar", "v_bar", "epsilon", "epsilon_bar"}
            for kind in state_kinds
        ):
            raise NotImplementedError(
                "physical control requires Dirac or vector states"
            )
        require_exact_numbers(initial_average)
        for expression in model_expressions.values():
            require_exact_numbers(expression)
        selected_loop_orders = {pair.loop for pair in process.orders.sectors[0].pairs}
        self.loop_graphs = tuple(
            graph for graph in process.loops if graph.order in selected_loop_orders
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
        self.model_inputs = model_inputs
        self.scale_squared_symbol = S("loop_physical_process::mu_r2")
        self.model_names = tuple(sorted(model_expressions))
        self.model_positions = {name: i for i, name in enumerate(self.model_names)}
        self.model_evaluator = backend.CoefficientEvaluator(
            _saved(
                [model_expressions[name] for name in self.model_names], model_inputs
            ),
            **self.options,
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
                graph, variables=variables, momenta=self.momenta
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
                **self.options,
            )
            for item in self.born_recurrences
        )
        self.loop_executors = tuple(
            NativeCoefficientDAG(
                recurrence.dag,
                output_nodes=finals,
                backend=backend,
                coefficient_engine=coefficient_engine,
                **self.options,
            )
            for recurrence, finals in zip(
                self.recurrences, self.loop_finals, strict=True
            )
        )
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
                    powers,
                    plan,
                    backend.ExactPolynomialEvaluator(plan.evaluator_bytes()),
                    backend.CoefficientEvaluator(
                        _saved(plan.masses_squared, mass_parameters), **self.options
                    ),
                    mass_parameters,
                    tuple(self.model_positions[p.get_name()] for p in mass_parameters),
                    self._compile_family_projection(indices, powers),
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
        poles = [
            [S(f"loop_physical_process::family{i}_pole{p}") for p in range(3)]
            for i in range(len(self.families))
        ]
        values = [
            E("1i") * sum((row[p] for row in poles), E("0")) / (16 * E("pi") ** 2)
            for p in range(3)
        ]
        self.postprocess = backend.CoefficientEvaluator(
            _saved(
                [value + value.conj() for value in values],
                [p for row in poles for p in row],
            ),
            **self.options,
        )
        self.zero_poles = backend.CoefficientEvaluator(
            _saved([E("0")] * 3, []), **self.options
        )

    def _born_symbols(self):
        from symbolica import S

        return [
            [
                S(f"loop_physical_process::born{helicity}_{i}")
                for i in range(len(self.born_colors))
            ]
            for helicity in range(len(self.helicities))
        ]

    def _compile_born_projection(self):
        from symbolica import E

        symbols = self._born_symbols()
        value = sum(
            (
                left.conj() * right * self.born_metric[i][j]
                for row in symbols
                for i, left in enumerate(row)
                for j, right in enumerate(row)
            ),
            E("0"),
        )
        self.born_projector = self.backend.CoefficientEvaluator(
            _saved([value], [value for row in symbols for value in row]), **self.options
        )

    def _compile_family_projection(self, indices, powers):
        from symbolica import E, S

        projected, parameters = {power: E("0") for power in powers}, []
        for helicity, born in enumerate(self._born_symbols()):
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
                        projected[power] += sum(
                            (
                                value.conj()
                                * self.loop_metric[b][term_index]
                                * coefficient
                                for b, value in enumerate(born)
                            ),
                            E("0"),
                        )
        return self.backend.CoefficientEvaluator(
            _saved([projected[p] for p in powers], parameters), **self.options
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
                family.graph_indices, family.powers
            )

    def evaluate_decimal(self, momenta, scales_squared, *, model_values):
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

    def evaluate_native(self, momenta, scales_squared, *, model_values):
        """Original decimal input enters exact kinematics and native arithmetic."""
        momenta, scales = (
            np.asarray(momenta, dtype=object),
            np.asarray(scales_squared, dtype=object),
        )
        legs, helicities = len(self.state_kinds), len(self.helicities)
        if momenta.ndim != 3 or momenta.shape[1:] != (legs, 4):
            raise ValueError("expected (batch, leg, four components) momenta")
        size, rows = len(momenta), len(momenta) * helicities
        if scales.shape != (size,) or len(model_values) != len(self.model_inputs):
            raise ValueError("physical scale or global model input shape differs")
        if any(
            not isinstance(value, str)
            for value in [*momenta.flat, *scales.flat, *model_values]
        ):
            raise TypeError("physical precise inputs must be decimal strings")
        batch_type = self.backend.NativeBatch

        def pairs(values):
            values = np.asarray(values, dtype=object).reshape(-1)
            return np.stack(
                [values, np.full(len(values), "0", dtype=object)], axis=1
            ).tolist()

        def native(values, nrows, columns):
            return batch_type.from_decimal(
                pairs(values), nrows, columns, **self.options
            )

        point_ids, lane_ids = list(range(size)), list(range(rows))
        momenta_pairs = pairs(momenta)
        physical = native(momenta, size, 4 * legs)
        lanes = physical.take_rows(
            np.repeat(np.arange(size), helicities).tolist(), row_ids=lane_ids
        )
        states = self.backend.wavefunctions_batch(
            lanes.reshape_rows(rows * legs, 4, list(range(rows * legs))),
            np.tile(np.asarray(self.helicities), (size, 1)).reshape(-1).tolist(),
            list(self.state_kinds) * rows,
        ).reshape_rows(rows, 4 * legs, lane_ids)
        model_arguments = native(model_values, 1, len(model_values))
        model = self.model_evaluator.evaluate_batch(model_arguments)
        bound = batch_type.concat_columns(
            [lanes, states, model.take_rows([0] * rows, row_ids=lane_ids)]
        )
        born = batch_type.concat_columns(
            [
                executor.evaluate_batch(bound.take_columns(columns))
                for executor, columns in zip(
                    self.born_executors, self.born_input_columns, strict=True
                )
            ]
        )
        currents = [
            executor.evaluate_batch(bound.take_columns(columns))
            for executor, columns in zip(
                self.loop_executors, self.loop_input_columns, strict=True
            )
        ]
        born_squared = self.born_projector.evaluate_batch(
            born.reshape_rows(size, helicities * born.columns, point_ids)
        )
        require_finite_batch(born_squared, "Born projection")
        scales = native(scales, size, 1)
        integrated, rational_integrated, statuses = [], [], []
        for family in self.families:
            inputs = batch_type.concat_columns(
                [born, *(currents[i] for i in family.graph_indices)]
            )
            coefficients = family.projector.evaluate_batch(
                inputs.reshape_rows(size, helicities * inputs.columns, point_ids)
            )
            bindings = family.exact_evaluator.evaluate_decimal(
                momenta_pairs, size, **self.options
            )
            n = family.kinematics.propagators
            shifts, matrix = (
                bindings.take_columns(list(range(4 * n))),
                bindings.take_columns(list(range(4 * n, 4 * n + n * n))),
            )
            masses = family.masses.evaluate_batch(
                model.take_columns(family.mass_columns)
            ).take_rows([0] * size, row_ids=point_ids)
            reduced = family.reducer.evaluate_batch(
                shifts, masses, scales, coefficients, s_matrix=matrix
            )
            failures = [
                (point, family.index, status)
                for point, status in reduced.statuses
                if status != "success"
            ]
            if failures:
                raise ArithmeticError(
                    "native process reduction failed (point, family, status): "
                    f"{failures}"
                )
            integrated.append(reduced.values)
            statuses.append(tuple(reduced.statuses))
            if self.dimensional_numerators:
                if family.rational_reducer is None:
                    rational_integrated.append(
                        self.zero_poles.evaluate_batch(scales.take_columns([]))
                    )
                else:
                    rational = family.rational_reducer.evaluate_batch(
                        shifts, masses, scales, coefficients, s_matrix=matrix
                    )
                    failures = [
                        (point, family.index, status)
                        for point, status in rational.statuses
                        if status != "success"
                    ]
                    if failures:
                        raise ArithmeticError(
                            "native dimensional remainder reduction failed "
                            f"(point, family, status): {failures}"
                        )
                    rational_integrated.append(rational.values)
        virtual = self.postprocess.evaluate_batch(batch_type.concat_columns(integrated))
        require_finite_batch(virtual, "bare virtual assembly")
        rational = None
        if self.dimensional_numerators:
            rational = self.postprocess.evaluate_batch(
                batch_type.concat_columns(rational_integrated)
            )
            require_finite_batch(rational, "direct rational assembly")
        return NativeBareEvaluation(
            NativePhysicalBatch(
                size,
                rows,
                point_ids,
                lane_ids,
                bound,
                dict(self.bound_positions),
                model_arguments,
                scales,
            ),
            born,
            born_squared,
            virtual,
            tuple(tuple(family[point] for family in statuses) for point in range(size)),
            rational,
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
