# SPDX-License-Identifier: 0BSD
"""Coupling orders of the Born amplitudes behind one normalization payload.

A model whose stage evaluators omit the coupling constants (the built-in SM)
restores them with a global factor whose powers are the amplitude coupling
orders of the Born. ``RuntimeNormalizationRequest`` hands a model's
``runtime_normalization_payload`` the process and, computed only when the
model asks, those orders: from the amplitude roots of a compiled DAG, or,
before any DAG exists, from the closures the generation's coupling-order
limits admit (over the generation's colour plan, or colour-blind where the
generation builds none).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from ..color.plan import GenericColorPlan
from ..models.base import CouplingOrders, Model, Vertex
from ..processes.ir import CanonicalProcessIR
from .dag_algorithms import (
    _closure_totals_from_color_plan,
    _minimal_coupling_order_limits_from_totals,
)
from .dag_reachability import (
    _combine_coupling_order_tuples,
    _coupling_orders_within_limits,
    _normalize_coupling_order_limits,
)
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

    vertices: dict[tuple[int, tuple[int, ...]], Vertex] = {}
    for model_vertex in model.vertices:
        vertices.setdefault(
            (int(model_vertex.kind), tuple(model_vertex.particles)), model_vertex
        )
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
        vertex: Vertex | None = (
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
    closure_totals: Iterable[CouplingOrders] | None = None,
    minimal: bool = False,
) -> frozenset[CouplingOrders]:
    """Total coupling orders of the closures a generation admits, pre-DAG.

    ``max_coupling_orders`` are the generation's limits; ``minimal`` narrows
    them to the minimal-order envelope, as generation does. Coupling orders
    only grow along a recursion, so the closures within the envelope are the
    envelope's subset of those within the limits: one sweep serves both, and
    ``closure_totals`` hands in one the generation already ran within the
    limits (``_closure_totals_from_color_plan``).

    The sweep runs over ``color_plan`` when the generation has one; without
    it, it is colour-blind (``_colour_blind_closure_totals``) and costs no
    colour ordering.
    """

    limits = dict(max_coupling_orders or {}) or None
    if closure_totals is not None:
        normalized = _normalize_coupling_order_limits(limits)
        totals = tuple(
            total
            for total in closure_totals
            if _coupling_orders_within_limits(total, normalized)
        )
    elif color_plan is None:
        totals = _colour_blind_closure_totals(
            process, model=model, max_coupling_orders=limits
        )
    else:
        totals = _closure_totals_from_color_plan(
            process,
            model=model,
            color_plan=color_plan,
            max_coupling_orders=limits,
        )
    if minimal:
        envelope = _minimal_coupling_order_limits_from_totals(totals, model=model)
        if envelope:
            totals = tuple(
                total
                for total in totals
                if _coupling_orders_within_limits(total, envelope)
            )
    return frozenset(totals)


def _colour_blind_closure_totals(
    process: CanonicalProcessIR,
    *,
    model: Model,
    max_coupling_orders: Mapping[str, int] | None,
) -> tuple[CouplingOrders, ...]:
    """The closure sweep of ``_closure_totals_from_color_plan`` without colour.

    An empty colour plan keeps the sweep's species, coupling-order and
    process-level vertex filters (``ColorEngine.vertex_allowed`` reads only the
    process and the model) and closes on the canonical sink leg. Every tree
    reaches every external leg, so one sink sees all of their totals, over
    O(3^n) leg-set splits instead of the (n-1)! orderings of an LC colour plan.
    What it drops, the pruning of current leg sets to segments of LC colour
    words, could only remove the totals of trees no colour word keeps planar;
    over the built-in-SM sweep processes (LC and full colour, minimal and
    explicit orders) both sweeps give the same totals.
    """

    return _closure_totals_from_color_plan(
        process,
        model=model,
        color_plan=GenericColorPlan(
            process=process, color_accuracy=process.color_accuracy, sectors=()
        ),
        max_coupling_orders=max_coupling_orders,
        color_order_mask_pruning=False,
    )
