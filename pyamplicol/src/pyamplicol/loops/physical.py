# SPDX-License-Identifier: 0BSD
"""Internal photon-only massless Drell--Yan bare virtual control.

This connects native physical spinors, local coefficient recurrences and Ninja
reduction. It is not the general loop API and contains no analytic loop form
factor. The returned virtual term excludes model R2 and UV insertions. SU(3)
color tensors are explicitly contracted for this control, not used to infer a
formal-Nc continuation. Model couplings are global; momenta/scales are batched.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .graphs import build_loop_process
from .kernels import CoefficientDAG, NativeCoefficientDAG
from .recurrence import (
    attach_external_spin_components,
    build_open_loop_recurrence,
    momentum_component_library,
    separate_local_color,
)
from .symbolic_sums import symbolic_sum


@dataclass(frozen=True, slots=True)
class DrellYanBareResult:
    """Decimal complex pairs, batch first; virtual order finite, single, double."""

    born: Any
    virtual: Any
    statuses: tuple[tuple[str, ...], ...]
    arithmetic: str
    precision: int | None
    reduction_schedule: str
    reduction_engine: str
    coefficient_engine: str
    convention: str = "MadLoop-Cepsilon-bare-four-dimensional-no-R2-no-UV"


def _denominator(edge, library):
    from symbolica import S
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    edge_id, momentum, mass, value = S(
        "loop_denominator::e_",
        "loop_denominator::p_",
        "loop_denominator::m_",
        "loop_denominator::v_",
    )
    expression = (
        edge.denominator_expression(dimension=4)
        .to_expression()
        .replace(h.Symbols.denominator()(edge_id, momentum, mass, value), value)
    )
    return t.TensorExpression(expression).to_tensor(library)[0]


def _born_dag(graph, variables, momenta):
    from symbolica import E
    from symbolica.community import tensor as t

    library = momentum_component_library(graph, (), momenta)
    local, color = {}, E("1")
    for kind, items in (
        ("vertex", graph.native.vertices),
        ("edge", graph.native.internal_edges),
    ):
        for item in items:
            lorentz, color_term = separate_local_color(item.numerator_expression())
            local[kind, item.id] = lorentz
            color *= color_term
    attach_external_spin_components(graph, local, library)
    tree = graph.native.depth_first_traverse(graph.vertices[0].id)
    dag, currents = CoefficientDAG(variables), {}
    for vertex in reversed(tree.nodes):
        children = tuple(currents[child.id] for child in tree.children(vertex.id))
        node = dag.add(
            f"born-vertex:{vertex.id}",
            local["vertex", vertex.id].to_tensor(library),
            children,
        )
        parent = tree.parent(vertex.id)
        if parent is not None:
            edge = next(
                edge
                for edge in tree.subgraph.internal_edges
                if {edge.source, edge.target} == {parent.id, vertex.id}
            )
            denominator = _denominator(edge, library)
            node = dag.add(
                f"born-propagator:{edge.id}",
                (local["edge", edge.id] / denominator).to_tensor(library),
                (node,),
            )
        currents[vertex.id] = node
    final = dag.add(
        "born-global-factor",
        t.TensorExpression(
            graph.native.overall_factor_expression(evaluate=True)
        ).to_tensor(library),
        (node,),
    )
    if dag.nodes[final].axes or len(dag.nodes[final].output_labels) != 1:
        raise ValueError("Born tree did not close to a scalar")
    return dag, t.TensorExpression(color)


def _saved(expressions, parameters):
    from symbolica import Expression

    # For large projector linear forms the build time is dominated by the
    # unbounded common-pair elimination (cpe_iterations=None): 524,288 Born
    # terms take 280 s, the six-leg CT projector 89 s. cpe_iterations=0 builds
    # them in about 1 s with slightly fewer instructions, but it reassociates
    # the arithmetic (results change at the last-bit level), so it is not
    # enabled here. n_cores > 1 gives no speedup at iterations=1.
    return Expression.evaluator_multiple(
        expressions, parameters, n_cores=1, jit_compile=False
    ).save()


class DrellYanBareProgram:
    """Reusable internal physical control; every numerical batch stays native."""

    helicities = tuple(product((-1, 1), repeat=4))
    state_kinds = ("u", "v_bar", "u_bar", "v")

    def __init__(
        self,
        *,
        arithmetic: str = "double",
        precision: int | None = None,
        backend: Any = None,
        reduction_schedule: str = "born-weighted",
        reduction_engine: str = "generic",
        coefficient_engine: str = "interpreted",
    ):
        if backend is None:
            from .native import load_loop_native

            backend = load_loop_native()
        from symbolica import E, S
        from symbolica.community import tensor as t

        from .control_models import qcd_qed_control_model

        if reduction_schedule not in {"born-weighted", "per-helicity-oracle"}:
            raise ValueError("unknown Drell--Yan reduction schedule")
        self.reduction_schedule = reduction_schedule
        self.reduction_engine = reduction_engine
        self.backend, self.arithmetic, self.precision = backend, arithmetic, precision
        self.native_options = {"arithmetic": arithmetic, "precision": precision}
        self.process = build_loop_process(
            qcd_qed_control_model().process(["u", "u~"], ["e-", "e+"]),
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
        )
        if len(self.process.born) != 1 or len(self.process.loops) != 1:
            raise ValueError("Drell--Yan control requires one Born and one loop graph")
        self.loop = build_open_loop_recurrence(self.process.loops[0])
        variables, momenta = self.loop.dag.variables, self.loop.momentum_parameters
        self.born_dag, born_color = _born_dag(self.process.born[0], variables, momenta)
        library = momentum_component_library(self.loop.graph, (variables,), momenta)
        bridge_factor = E("1")
        for edge in self.loop.graph.native.internal_edges:
            if edge.id not in self.loop.graph.loop_edges:
                bridge_factor /= _denominator(edge, library)
        self.loop_final = self.loop.dag.add(
            "physical-tree-denominators",
            t.TensorExpression(bridge_factor).to_tensor(library),
            (self.loop.final_node,),
        )
        self.born_executor = NativeCoefficientDAG(
            self.born_dag,
            backend=backend,
            output_nodes=(len(self.born_dag.nodes) - 1,),
            coefficient_engine=coefficient_engine,
            **self.native_options,
        )
        self.loop_executor = NativeCoefficientDAG(
            self.loop.dag,
            backend=backend,
            output_nodes=(self.loop_final,),
            coefficient_engine=coefficient_engine,
            **self.native_options,
        )
        self.coefficient_engine = self.born_executor.coefficient_engine
        # Exact component contractions establish this explicit-Nc=3 control.
        born_color = born_color.to_tensor().to_dense()
        loop_color = self.loop.color_expression.to_tensor().to_dense()
        loop_color = loop_color.permute_axes(
            [loop_color.axes.index(axis) for axis in born_color.axes]
        )
        if born_color.axes != loop_color.axes:
            raise ValueError("Born/loop external color ports disagree")
        self.born_weight = (
            sum(
                (born_color[i].conj() * born_color[i] for i in range(len(born_color))),
                E("0"),
            )
            / 36
        )
        self.interference_weight = (
            sum(
                (born_color[i].conj() * loop_color[i] for i in range(len(born_color))),
                E("0"),
            )
            / 36
        )
        alpha_s, inverse_alpha = S(
            "loop_physical::alpha_s", "loop_physical::inverse_alpha"
        )
        electromagnetic = (4 * E("pi") / inverse_alpha) ** E("1/2")
        strong = (4 * E("pi") * alpha_s) ** E("1/2")
        self.couplings = backend.CoefficientEvaluator(
            _saved(
                [
                    E("2/3") * E("1i") * electromagnetic,
                    -E("1i") * electromagnetic,
                    E("1i") * strong,
                ],
                [alpha_s, inverse_alpha],
            ),
            **self.native_options,
        )
        self.coupling_names = ("UFO::GC_2", "UFO::GC_3", "UFO::GC_11")
        self.momentum_parameters = tuple(
            component for vector in momenta for component in vector
        )
        shifts = []
        for edge in self.loop.graph.edges:
            if edge.id in self.loop.graph.loop_edges:
                coefficient = edge.loop_momentum[0]
                if coefficient not in (-1, 1) or edge.mass != "0":
                    raise ValueError(
                        "Drell--Yan control requires standard massless cycle lines"
                    )
                shifts.extend(
                    coefficient
                    * sum(
                        (
                            weight * vector[axis]
                            for weight, vector in zip(
                                edge.external_momentum, momenta, strict=True
                            )
                        ),
                        E("0"),
                    )
                    for axis in range(4)
                )
        self.shifts = backend.CoefficientEvaluator(
            _saved(shifts, self.momentum_parameters), **self.native_options
        )
        labels = self.loop.dag.nodes[self.loop_final].output_labels
        coefficient_names = [f"c{i}" for i in range(len(labels))]
        numerator = "+".join(
            "*".join(
                [
                    name,
                    *(
                        f"q{axis}^{power}"
                        for axis, power in enumerate(label.powers)
                        if power
                    ),
                ]
            )
            for name, label in zip(coefficient_names, labels, strict=True)
        )
        self.reducer = backend.PolynomialPlan(
            3,
            [numerator, "0", "0"],
            coefficient_names,
            engine=reduction_engine,
            **self.native_options,
        )
        # This projection is complex-linear in the loop coefficients. Taking a
        # real part here would discard absorptive contributions after integration.
        projection_parameters, weighted = [], [[] for _ in labels]
        post_parameters, born_sum, virtual = [], [], [[], [], []]
        for helicity in range(len(self.helicities)):
            born = S(f"loop_physical::born{helicity}")
            coefficients = [
                S(f"loop_physical::coefficient{helicity}_{i}")
                for i in range(len(labels))
            ]
            projection_parameters.extend([born, *coefficients])
            for i, coefficient in enumerate(coefficients):
                weighted[i].append(self.interference_weight * born.conj() * coefficient)
            poles = [S(f"loop_physical::loop{helicity}_{pole}") for pole in range(3)]
            post_parameters.extend([born, *poles])
            born_sum.append(self.born_weight * born.conj() * born)
            for pole in range(3):
                value = (
                    self.interference_weight
                    * born.conj()
                    * E("1i")
                    * poles[pole]
                    / (16 * E("pi") ** 2)
                )
                virtual[pole].extend((value, value.conj()))
        born_sum = symbolic_sum(born_sum)
        weighted = [symbolic_sum(terms) for terms in weighted]
        virtual = [symbolic_sum(terms) for terms in virtual]
        self.project_born = backend.CoefficientEvaluator(
            _saved([born_sum, *weighted], projection_parameters), **self.native_options
        )
        self.oracle_postprocess = backend.CoefficientEvaluator(
            _saved([born_sum, *virtual], post_parameters), **self.native_options
        )
        projected_born, *projected_poles = S(
            "loop_physical::born_sum",
            "loop_physical::finite",
            "loop_physical::single",
            "loop_physical::double",
        )
        integrated = [E("1i") * pole / (16 * E("pi") ** 2) for pole in projected_poles]
        self.weighted_postprocess = backend.CoefficientEvaluator(
            _saved(
                [projected_born, *(value + value.conj() for value in integrated)],
                [projected_born, *projected_poles],
            ),
            **self.native_options,
        )
        # A future CT tree supplies a separate native complex amplitude buffer;
        # it must use the Born/color bilinear but not this loop-measure factor.

    def evaluate_decimal(
        self,
        momenta: Any,
        scales_squared: Any,
        *,
        alpha_s: str = "0.118",
        inverse_alpha: str = "132.507",
    ) -> DrellYanBareResult:
        """Positive physical momenta, exact decimal strings; global model inputs."""
        momenta = np.asarray(momenta, dtype=object)
        scales = np.asarray(scales_squared, dtype=object)
        if momenta.ndim != 3 or momenta.shape[1:] != (4, 4):
            raise ValueError("expected (batch, four legs, four components) momenta")
        size = len(momenta)
        if scales.shape != (size,):
            raise ValueError("one scale squared per phase-space point is required")
        if any(
            not isinstance(value, str)
            for value in [*momenta.flat, *scales.flat, alpha_s, inverse_alpha]
        ):
            raise TypeError("physical precise inputs must be decimal strings")
        helicities = len(self.helicities)
        rows = size * helicities
        point_ids, lane_ids = list(range(size)), list(range(rows))
        batch_type = self.backend.NativeBatch

        def from_real(values, batch_rows, columns):
            values = np.asarray(values, dtype=object).reshape(-1)
            pairs = np.stack([values, np.full(len(values), "0", dtype=object)], axis=1)
            return batch_type.from_decimal(
                pairs.tolist(), batch_rows, columns, **self.native_options
            )

        point_momenta = from_real(momenta, size, 16)
        lane_momenta = point_momenta.take_rows(
            np.repeat(np.arange(size), helicities).tolist(), row_ids=lane_ids
        )
        states = self.backend.massless_spinors_batch(
            lane_momenta.reshape_rows(rows * 4, 4, list(range(rows * 4))),
            np.tile(np.asarray(self.helicities), (size, 1)).reshape(-1).tolist(),
            list(self.state_kinds) * rows,
        ).reshape_rows(rows, 16, lane_ids)
        couplings = self.couplings.evaluate_batch(
            from_real([alpha_s, inverse_alpha], 1, 2)
        ).take_rows([0] * rows, row_ids=lane_ids)
        bound = batch_type.concat_columns([lane_momenta, states, couplings])
        positions = {
            parameter.get_name(): i
            for i, parameter in enumerate(self.momentum_parameters)
        }
        positions.update(
            {
                symbol.get_name(): 16 + 4 * leg + component
                for leg, symbols in self.loop.external_states
                for component, symbol in enumerate(symbols)
            }
        )
        positions.update({name: 32 + i for i, name in enumerate(self.coupling_names)})

        def inputs(dag):
            return bound.take_columns([positions[p.get_name()] for p in dag.parameters])

        born = self.born_executor.evaluate_batch(inputs(self.born_dag))
        coefficients = self.loop_executor.evaluate_batch(inputs(self.loop.dag))
        scales = from_real(scales, size, 1)
        if self.reduction_schedule == "born-weighted":
            projected = self.project_born.evaluate_batch(
                batch_type.concat_columns([born, coefficients]).reshape_rows(
                    size, helicities * (1 + coefficients.columns), point_ids
                )
            )
            born = projected.take_columns([0])
            coefficients = projected.take_columns(list(range(1, projected.columns)))
            reduction_momenta, reduction_rows = point_momenta, size
            statuses_per_point = 1
        else:
            reduction_momenta, reduction_rows = lane_momenta, rows
            scales = scales.take_rows(
                np.repeat(np.arange(size), helicities).tolist(), row_ids=lane_ids
            )
            statuses_per_point = helicities
        shifts = self.shifts.evaluate_batch(reduction_momenta)
        masses = from_real(["0"] * (reduction_rows * 3), reduction_rows, 3)
        reduction = self.reducer.evaluate_batch(shifts, masses, scales, coefficients)
        failures = [
            (lane // statuses_per_point, lane, status)
            for lane, status in reduction.statuses
            if status != "success"
        ]
        if failures:
            raise ArithmeticError(
                "native Drell--Yan reduction failed (point, lane, status): "
                f"{failures}; no physical result returned"
            )
        poles = reduction.values
        combined = batch_type.concat_columns([born, poles])
        if self.reduction_schedule == "born-weighted":
            result = self.weighted_postprocess.evaluate_batch(combined)
        else:
            result = self.oracle_postprocess.evaluate_batch(
                combined.reshape_rows(size, helicities * 4, point_ids)
            )
        result = np.asarray(result.to_decimal(), dtype=object).reshape(size, 4, 2)
        statuses = tuple(
            tuple(
                status
                for _, status in reduction.statuses[start : start + statuses_per_point]
            )
            for start in range(0, reduction_rows, statuses_per_point)
        )
        return DrellYanBareResult(
            result[:, 0],
            result[:, 1:],
            statuses,
            self.arithmetic,
            self.precision,
            self.reduction_schedule,
            self.reduction_engine,
            self.coefficient_engine,
        )
