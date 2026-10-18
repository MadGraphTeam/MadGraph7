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


@dataclass(frozen=True, slots=True)
class DrellYanBareResult:
    """Decimal complex pairs, batch first; virtual order finite, single, double."""

    born: Any
    virtual: Any
    statuses: tuple[tuple[str, ...], ...]
    arithmetic: str
    precision: int | None
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

    return Expression.evaluator_multiple(
        expressions, parameters, n_cores=1, jit_compile=False
    ).save()


class DrellYanBareProgram:
    """Reusable internal physical control; every numerical batch stays native."""

    reduction_schedule = "per-helicity-oracle"
    helicities = tuple(product((-1, 1), repeat=4))
    state_kinds = ("u", "v_bar", "u_bar", "v")

    def __init__(
        self,
        *,
        arithmetic: str = "double",
        precision: int | None = None,
        backend: Any = None,
    ):
        if backend is None:
            import _loop_reduction as backend
        from symbolica import E, S
        from symbolica.community import hepkit as h
        from symbolica.community import tensor as t

        self.backend, self.arithmetic, self.precision = backend, arithmetic, precision
        self.native_options = {"arithmetic": arithmetic, "precision": precision}
        self.process = build_loop_process(
            h.Model.qcd_qed().process(["u", "u~"], ["e-", "e+"]),
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
            self.born_dag, backend=backend, **self.native_options
        )
        self.loop_executor = NativeCoefficientDAG(
            self.loop.dag, backend=backend, **self.native_options
        )
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
            3, [numerator, "0", "0"], coefficient_names, **self.native_options
        )
        post_parameters, born_sum, virtual = [], E("0"), [E("0")] * 3
        for helicity in range(len(self.helicities)):
            born = S(f"loop_physical::born{helicity}")
            poles = [S(f"loop_physical::loop{helicity}_{pole}") for pole in range(3)]
            post_parameters.extend([born, *poles])
            born_sum += self.born_weight * born.conj() * born
            for pole in range(3):
                value = (
                    self.interference_weight
                    * born.conj()
                    * E("1i")
                    * poles[pole]
                    / (16 * E("pi") ** 2)
                )
                virtual[pole] += value + value.conj()
        self.postprocess = backend.CoefficientEvaluator(
            _saved([born_sum, *virtual], post_parameters), **self.native_options
        )

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
        lane_momenta = np.repeat(momenta, helicities, axis=0)
        spinors = self.backend.massless_spinors_decimal(
            lane_momenta.reshape(-1, 4).tolist(),
            np.tile(np.asarray(self.helicities), (size, 1)).reshape(-1).tolist(),
            list(self.state_kinds) * rows,
            **self.native_options,
        )
        states = np.asarray(spinors, dtype=object).reshape(rows, 4, 4, 2)
        couplings = np.asarray(
            self.couplings.evaluate_decimal([[alpha_s, "0"], [inverse_alpha, "0"]], 1),
            dtype=object,
        )
        bindings = {
            name: np.broadcast_to(value, (rows, 2))
            for name, value in zip(self.coupling_names, couplings, strict=True)
        }
        momentum_values = np.stack(
            [lane_momenta.reshape(rows, 16), np.full((rows, 16), "0", dtype=object)],
            axis=2,
        )
        bindings.update(
            {
                parameter.get_name(): momentum_values[:, i, :]
                for i, parameter in enumerate(self.momentum_parameters)
            }
        )
        bindings.update(
            {
                symbol.get_name(): states[:, leg, component, :]
                for leg, symbols in self.loop.external_states
                for component, symbol in enumerate(symbols)
            }
        )

        def inputs(dag):
            return np.stack([bindings[p.get_name()] for p in dag.parameters], axis=1)

        born = self.born_executor.evaluate_decimal(inputs(self.born_dag))[-1][:, 0, :]
        coefficients = self.loop_executor.evaluate_decimal(inputs(self.loop.dag))[
            self.loop_final
        ]
        shifts = self.shifts.evaluate_decimal(
            momentum_values.reshape(-1, 2).tolist(), rows
        )
        scales = np.repeat(scales, helicities)
        reduction = self.reducer.evaluate_decimal(
            list(range(rows)),
            np.asarray(shifts, dtype=object).reshape(rows * 3, 4, 2).tolist(),
            [["0", "0"]] * (rows * 3),
            np.stack([scales, np.full(rows, "0", dtype=object)], axis=1).tolist(),
            coefficients.reshape(-1, 2).tolist(),
        )
        if any(
            status != "success" or values is None for _, status, values in reduction
        ):
            raise ArithmeticError(
                "native Drell--Yan reduction failed; no physical result returned"
            )
        poles = np.asarray(
            [values for _, _, values in reduction], dtype=object
        ).reshape(rows, 3, 2)
        combined = np.concatenate([born[:, None, :], poles], axis=1).reshape(
            size, helicities * 4, 2
        )
        result = np.asarray(
            self.postprocess.evaluate_decimal(combined.reshape(-1, 2).tolist(), size),
            dtype=object,
        ).reshape(size, 4, 2)
        statuses = tuple(
            tuple(status for _, status, _ in reduction[start : start + helicities])
            for start in range(0, rows, helicities)
        )
        return DrellYanBareResult(
            result[:, 0], result[:, 1:], statuses, self.arithmetic, self.precision
        )
