# SPDX-License-Identifier: 0BSD
"""Coupling orders of the Born amplitudes behind one normalization payload.

A model whose stage evaluators omit the coupling constants (the built-in SM)
restores them with a global factor whose powers are the amplitude coupling
orders of the Born. ``RuntimeNormalizationRequest`` hands a model's
``runtime_normalization_payload`` the process and, computed only when the
model asks, those orders: from the amplitude roots of a compiled DAG, or,
before any DAG exists, from the closures the generation's coupling-order
limits admit.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..color.plan import GenericColorPlan, build_color_plan
from ..models.base import CouplingOrders, Model
from ..processes.ir import CanonicalProcessIR
from .dag_algorithms import (
    _closure_totals_from_color_plan,
    _infer_minimal_coupling_order_limits_from_color_plan,
)
from .dag_reachability import _combine_coupling_order_tuples
from .dag_types import GenericDAG


class RuntimeNormalizationRequest:
    """The process of one normalization payload and its Born coupling orders."""

    __slots__ = ("_orders", "_provider", "process")

    def __init__(
        self,
        process: CanonicalProcessIR,
        born_amplitude_orders: Callable[[], Iterable[CouplingOrders]] | None = None,
    ) -> None:
        self.process = process
        self._provider = born_amplitude_orders
        self._orders: frozenset[CouplingOrders] | None = None

    @classmethod
    def of_dag(cls, dag: GenericDAG, model: Model) -> RuntimeNormalizationRequest:
        return cls(dag.process, lambda: dag_born_amplitude_orders(dag, model))

    @property
    def born_amplitude_orders(self) -> frozenset[CouplingOrders]:
        """Distinct total coupling orders of the Born amplitudes."""

        if self._orders is None:
            if self._provider is None:
                raise TypeError(
                    f"normalization of {self.process.process!r} was requested "
                    "without the coupling orders of its Born amplitudes"
                )
            self._orders = frozenset(
                _combine_coupling_order_tuples(tuple(orders))
                for orders in self._provider()
            )
        return self._orders


def dag_born_amplitude_orders(
    dag: GenericDAG,
    model: Model,
) -> frozenset[CouplingOrders]:
    """Total coupling orders of the amplitude roots of a compiled DAG.

    A vertex closure adds its vertex's orders to those of its two currents;
    every other root contracts its currents directly.
    """

    vertices: dict[tuple[int, tuple[int, ...]], Any] = {}
    for vertex in model.vertices:
        vertices.setdefault((int(vertex.kind), tuple(vertex.particles)), vertex)
    orders: set[CouplingOrders] = set()
    for root in dag.amplitude_roots:
        left = dag.currents[root.left_id].index
        right = dag.currents[root.right_id].index
        if root.kind != "vertex-closure":
            orders.add(
                _combine_coupling_order_tuples(
                    tuple(left.coupling_orders), tuple(right.coupling_orders)
                )
            )
            continue
        vertex = (
            None
            if root.vertex_kind is None or root.vertex_particles is None
            else vertices.get((int(root.vertex_kind), tuple(root.vertex_particles)))
        )
        if vertex is None:
            raise ValueError(
                f"amplitude root {root.id} of {dag.process.process!r} names no "
                f"model vertex {root.vertex_kind} {root.vertex_particles}"
            )
        orders.add(
            _combine_coupling_order_tuples(
                tuple(model.combine_coupling_orders(left, right, vertex))
            )
        )
    return frozenset(orders)


def process_born_amplitude_orders(
    process: CanonicalProcessIR,
    *,
    model: Model,
    max_coupling_orders: Mapping[str, int] | None,
    color_plan: GenericColorPlan | None = None,
    minimal: bool = False,
) -> frozenset[CouplingOrders]:
    """Total coupling orders of the closures a generation admits, pre-DAG.

    ``max_coupling_orders`` are the generation's limits; ``minimal`` first
    narrows them to the minimal-order envelope, as generation does.
    """

    if color_plan is None:
        color_plan = build_color_plan(
            process,
            color_accuracy=process.color_accuracy,
            fold_trace_reflections=(
                model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
    limits = dict(max_coupling_orders or {})
    if minimal:
        limits = (
            _infer_minimal_coupling_order_limits_from_color_plan(
                process,
                model=model,
                color_plan=color_plan,
                max_coupling_orders=limits or None,
            )
            or limits
        )
    return frozenset(
        _closure_totals_from_color_plan(
            process,
            model=model,
            color_plan=color_plan,
            max_coupling_orders=limits or None,
        )
    )
