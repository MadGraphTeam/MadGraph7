# SPDX-License-Identifier: 0BSD
"""Internal native bare-process assembly over verified graph families.

This first physical process boundary supports on-shell external Dirac states,
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
from .graphs import build_loop_process
from .kernels import NativeCoefficientDAG, require_finite_batch
from .kinematics import build_loop_kinematics
from .physical import _born_dag, _denominator, _saved
from .recurrence import build_open_loop_recurrence, momentum_component_library


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
    projector: Any
    reducer: Any


class NativeBareProcessProgram:
    """Reusable graph/family assembly, with global exact model input formulas.

    ``model_expressions`` maps native model symbol names to exact expressions
    in ``model_inputs``. Numerical values bind once per call, never per event.
    The caller must supply on-shell momenta with the declared model masses;
    native wavefunctions use those four-vectors without repairing their shell.
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
        backend: Any = None,
    ):
        if backend is None:
            import _loop_reduction as backend
        from symbolica import E, S
        from symbolica.community import tensor as t

        if len(process.orders.born_orders) != 1 or len(process.orders.sectors) != 1:
            raise NotImplementedError("physical control requires one Born/order sector")
        if any(kind not in {"u", "v", "u_bar", "v_bar"} for kind in state_kinds):
            raise NotImplementedError("physical control requires external Dirac states")
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
        self.initial_average = initial_average
        self.state_kinds = state_kinds
        self.helicities = tuple(product((-1, 1), repeat=len(state_kinds)))
        self.arithmetic, self.precision = arithmetic, precision
        self.reduction_engine = reduction_engine
        self.options = {"arithmetic": arithmetic, "precision": precision}
        self.model_inputs = model_inputs
        self.scale_squared_symbol = S("loop_physical_process::mu_r2")
        self.model_names = tuple(sorted(model_expressions))
        self.model_evaluator = backend.CoefficientEvaluator(
            _saved(
                [model_expressions[name] for name in self.model_names], model_inputs
            ),
            **self.options,
        )
        self.recurrences = tuple(
            build_open_loop_recurrence(graph, routing="family")
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
        variables = representative.dag.variables
        self.born_dags, born_colors = [], []
        self.born_color_tensors = []
        for graph in process.born:
            dag, color = _born_dag(graph, variables, self.momenta)
            self.born_dags.append(dag)
            self.born_color_tensors.append(external_color_tensor(graph, color))
            born_colors.append(_external_color(graph, color))
        self.loop_finals, loop_colors = [], []
        self.loop_color_tensors = []
        self.born_colors = tuple(born_colors)
        for recurrence in self.recurrences:
            if recurrence.momentum_parameters != self.momenta:
                raise ValueError("native graph external momentum frames differ")
            library = momentum_component_library(
                recurrence.graph, (recurrence.loop_components,), self.momenta
            )
            bridge = E("1")
            for edge in recurrence.graph.native.internal_edges:
                if edge.id not in recurrence.graph.loop_edges:
                    denominator = _denominator(edge, library)
                    if any(denominator.contains(q) for q in variables):
                        raise ValueError(
                            "tree bridge denominator depends on loop momentum"
                        )
                    bridge /= denominator
            self.loop_finals.append(
                recurrence.dag.add(
                    "physical-tree-denominators",
                    t.TensorExpression(bridge).to_tensor(library),
                    (recurrence.final_node,),
                )
            )
            loop_colors.append(
                _external_color(recurrence.graph, recurrence.color_expression)
            )
            self.loop_color_tensors.append(
                external_color_tensor(recurrence.graph, recurrence.color_expression)
            )
        self.born_executors = tuple(
            NativeCoefficientDAG(
                dag,
                output_nodes=(len(dag.nodes) - 1,),
                backend=backend,
                coefficient_engine=coefficient_engine,
                **self.options,
            )
            for dag in self.born_dags
        )
        self.loop_executors = tuple(
            NativeCoefficientDAG(
                r.dag,
                output_nodes=(final,),
                backend=backend,
                coefficient_engine=coefficient_engine,
                **self.options,
            )
            for r, final in zip(self.recurrences, self.loop_finals, strict=True)
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
            labels = {
                i: self.recurrences[i].dag.nodes[self.loop_finals[i]].output_labels
                for i in indices
            }
            powers = tuple(
                sorted({label.powers for group in labels.values() for label in group})
            )
            if max(map(sum, powers)) > plan.propagators + 1:
                raise NotImplementedError("numerator rank exceeds the reducer domain")
            names = [f"c{i}" for i in range(len(powers))]
            numerator = "+".join(
                "*".join(
                    [
                        name,
                        *(
                            f"q{axis}^{power}"
                            for axis, power in enumerate(exponents)
                            if power
                        ),
                    ]
                )
                for name, exponents in zip(names, powers, strict=True)
            )
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
                    self._compile_family_projection(indices, powers),
                    backend.PolynomialPlan(
                        plan.propagators,
                        [numerator, "0", "0"],
                        names,
                        engine=reduction_engine,
                        **self.options,
                    ),
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

    def _born_symbols(self):
        from symbolica import S

        return [
            [
                S(f"loop_physical_process::born{helicity}_{i}")
                for i in range(len(self.born_dags))
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
                labels = (
                    self.recurrences[index]
                    .dag.nodes[self.loop_finals[index]]
                    .output_labels
                )
                for column, label in enumerate(labels):
                    coefficient = S(
                        f"loop_physical_process::coefficient{helicity}_{index}_{column}"
                    )
                    parameters.append(coefficient)
                    projected[label.powers] += sum(
                        (
                            value.conj() * self.loop_metric[b][index] * coefficient
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
            len(born_metric) != len(self.born_dags)
            or any(len(row) != len(self.born_dags) for row in born_metric)
            or len(loop_metric) != len(self.born_dags)
            or any(len(row) != len(self.recurrences) for row in loop_metric)
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
        positions = {p.get_name(): i for i, p in enumerate(self.momentum_parameters)}
        positions.update(
            {
                symbol.get_name(): 4 * legs + 4 * leg + component
                for leg, symbols in self.external_states
                for component, symbol in enumerate(symbols)
            }
        )
        positions.update(
            {name: 8 * legs + i for i, name in enumerate(self.model_names)}
        )

        def execute(executor, dag):
            return executor.evaluate_batch(
                bound.take_columns([positions[p.get_name()] for p in dag.parameters])
            )

        born = batch_type.concat_columns(
            [
                execute(executor, dag)
                for executor, dag in zip(
                    self.born_executors, self.born_dags, strict=True
                )
            ]
        )
        currents = [
            execute(executor, recurrence.dag)
            for executor, recurrence in zip(
                self.loop_executors, self.recurrences, strict=True
            )
        ]
        born_squared = self.born_projector.evaluate_batch(
            born.reshape_rows(size, helicities * born.columns, point_ids)
        )
        require_finite_batch(born_squared, "Born projection")
        scales = native(scales, size, 1)
        model_positions = {name: i for i, name in enumerate(self.model_names)}
        integrated, statuses = [], []
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
                model.take_columns(
                    [model_positions[p.get_name()] for p in family.mass_parameters]
                )
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
        virtual = self.postprocess.evaluate_batch(batch_type.concat_columns(integrated))
        require_finite_batch(virtual, "bare virtual assembly")
        return NativeBareEvaluation(
            NativePhysicalBatch(
                size,
                rows,
                point_ids,
                lane_ids,
                bound,
                positions,
                model_arguments,
                scales,
            ),
            born,
            born_squared,
            virtual,
            tuple(tuple(family[point] for family in statuses) for point in range(size)),
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
    from symbolica import E, S
    from symbolica.community import hepkit as h

    model = h.Model.qcd_qed()
    process = build_loop_process(
        model.process(["u", "u~"], outgoing),
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
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("36"),
        model_expressions=expressions,
        model_inputs=(alpha_s, top_mass),
        **options,
    )
