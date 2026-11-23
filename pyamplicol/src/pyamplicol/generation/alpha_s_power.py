# SPDX-License-Identifier: 0BSD
"""The proven power of alpha_s in the squared matrix element of a process.

With it, ``ME(alpha_s) = (alpha_s / alpha_s_0)^k ME(alpha_s_0)`` exactly, so that a
per-event alpha_s needs no new evaluation. It holds if every amplitude of the
process is homogeneous of one degree ``k / 2`` in alpha_s. The proof uses the
model expressions, not coupling-order names:

* every vertex term coupling is homogeneous in alpha_s, by Euler's identity
  ``a dC/da = d C`` (alpha_s enters a kernel only through its term couplings);
* no other expression of the model (masses, widths, propagators, Lorentz and
  colour structures) depends on alpha_s;
* the coupling-order reachability sweep of the DAG compiler, without its
  pruning to minimal orders, finds one total degree for all amplitude closures.
  It may overestimate the closures, which can only make the proof fail.
"""

from __future__ import annotations

import json
import re
import weakref
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

from ..color.plan import GenericColorPlan
from ..models.base import CouplingOrders, Model
from ..processes.ir import CanonicalProcessIR
from .dag_color import ColorEngine
from .dag_ordering import (
    _closure_candidate_splits,
    _closure_side_reachable_masks,
    _labels_mask,
    _lc_color_order_reachable_masks,
)
from .dag_reachability import (
    _closure_right_particles_by_left,
    _combine_coupling_order_tuples,
    _coupling_orders_within_limits,
    _mask_allowed_by_reachability,
    _masks_by_size,
    _normalize_coupling_order_limits,
    _ordered_splits,
    _right_particles_by_left,
)

# a namespaced symbol of an expression, e.g. model_sm::{}::aS
_SYMBOL = re.compile(r"::(?:\{\}::)?([A-Za-z_][A-Za-z0-9_]*)")
_DERIVED_COUPLING = re.compile(r"derived_coupling_([0-9]+)")
# twice the alpha_s degree of every oriented kernel kind, per model and parameter
_KERNEL_DEGREES: weakref.WeakKeyDictionary[Any, dict[str, dict[int, int] | None]] = (
    weakref.WeakKeyDictionary()
)


def alpha_s_parameter(
    parameters: Iterable[Mapping[str, Any]], explicit: str | None = None
) -> str | None:
    """The external alpha_s parameter: the explicit one, or the unique SMINPUTS 3."""
    if explicit:
        return str(explicit)
    candidates = [
        str(p["name"])
        for p in parameters
        if str(p.get("lhablock") or "").upper() == "SMINPUTS"
        and list(p.get("lhacode") or ()) == [3]
        and p.get("nature") == "external"
    ]
    return candidates[0] if len(candidates) == 1 else None


def _twice_degree(expression: str, symbol: Any) -> int | None:
    """``2 d`` for an expression homogeneous of degree ``d`` in the symbol."""
    from symbolica import E

    # An expression Symbolica cannot handle is not proven homogeneous.
    try:
        value = E(expression)
        derivative = value.derivative(symbol)
        if derivative.expand().to_canonical_string() == "0":
            return 0
        ratio = (symbol * derivative / value).together()
        if not ratio.is_constant():
            return None
        degree = 2 * Fraction(ratio.to_canonical_string())
    except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
        return None
    return int(degree) if degree.denominator == 1 else None


def _drop_copies(value: Any, key: str, copy: object) -> None:
    """Removes the entries ``key: copy`` at any depth, e.g. certificates of a term."""
    if isinstance(value, dict):
        if value.get(key) == copy:
            del value[key]
        for item in value.values():
            _drop_copies(item, key, copy)
    elif isinstance(value, list):
        for item in value:
            _drop_copies(item, key, copy)


def _kernel_degrees(ir: Any, name: str) -> dict[int, int] | None:
    cached = _KERNEL_DEGREES.setdefault(ir, {})
    if name not in cached:
        cached[name] = _compute_kernel_degrees(ir, name)
    return cached[name]


def _compute_kernel_degrees(ir: Any, name: str) -> dict[int, int] | None:
    from symbolica import E

    records = {record.name: record for record in ir.parameters}
    external = records.get(name)
    if external is None or external.nature != "external":
        return None
    symbol = E(external.resolved_expression)
    # parameters and couplings that depend on alpha_s
    dependent = {name}
    for record in (*ir.parameters, *ir.couplings):
        if (
            record.name != name
            and _twice_degree(record.resolved_expression, symbol) != 0
        ):
            dependent.add(record.name)
    term_degrees: dict[int, int | None] = {}
    for term in ir.vertex_terms:
        term_degrees[term.id] = _twice_degree(term.coupling_expression, symbol)
    # Outside of parameters, couplings and term couplings, no expression may name
    # a dependent parameter, nor may a particle mass or width. False matches of
    # symbol names only make the proof fail.
    payload = ir.to_dict()
    for section in ("parameters", "couplings"):
        payload.pop(section, None)
    for term in payload.get("vertex_terms", ()):
        term.pop("coupling", None)
        _drop_copies(term, "coupling_expression", term.pop("coupling_expression", None))
    named = set(_SYMBOL.findall(json.dumps(payload, ensure_ascii=False)))
    named.update(name for p in ir.particles for name in (p.mass, p.width))
    if dependent & named:
        return None
    degrees: dict[int, int] = {}
    for kernel in ir.oriented_kernels:
        kernel_degrees = set()
        for parameter in kernel.runtime_parameters:
            match = _DERIVED_COUPLING.fullmatch(parameter)
            if match:
                kernel_degrees.add(term_degrees.get(int(match.group(1))))
            elif parameter in dependent:
                return None
        # several term couplings of one kernel are summed
        if None in kernel_degrees or len(kernel_degrees) > 1:
            return None
        degrees[int(kernel.kind)] = next(iter(kernel_degrees), 0)
    return degrees


def alpha_s_power(
    process: CanonicalProcessIR,
    model: Model,
    color_plan: GenericColorPlan,
    max_coupling_orders: Mapping[str, int] | None = None,
) -> dict[str, object] | None:
    """``{"parameter", "power"}`` if every amplitude closure of the process, with the
    colour plan and coupling-order limits of its generation, has one degree in the
    model's alpha_s parameter; otherwise None."""
    ir = getattr(getattr(model, "compiled", None), "ir", None)
    if ir is None:
        return None
    name = alpha_s_parameter(record.to_dict() for record in ir.parameters)
    degrees = None if name is None else _kernel_degrees(ir, name)
    if degrees is None:
        return None
    totals = _closure_degrees(process, model, color_plan, degrees, max_coupling_orders)
    if totals is None or len(totals) != 1:
        return None
    return {"parameter": name, "power": next(iter(totals))}


def _closure_degrees(process, model, color_plan, degrees, max_coupling_orders):
    """Twice the alpha_s degrees of all amplitude closures, like
    ``_closure_total_coupling_orders`` but without its Pareto pruning."""
    limits = _normalize_coupling_order_limits(max_coupling_orders)
    accuracy = process.color_accuracy
    color_engine = ColorEngine(color_plan, model)
    full_mask = _labels_mask(leg.label for leg in process.legs)
    splits = _closure_candidate_splits(process, model, color_engine)
    closure_masks = _closure_side_reachable_masks(full_mask, splits)
    color_masks = _lc_color_order_reachable_masks(process, color_plan, model)

    def allowed(mask: int) -> bool:
        return _mask_allowed_by_reachability(mask, closure_masks, color_masks)

    def vertices(left: int, right: int):
        for vertex in model.vertices_accepting(left, right, color_accuracy=accuracy):
            if color_engine.vertex_allowed(
                vertex
            ) and not model.skip_duplicate_vertex_orientation(vertex):
                yield vertex

    def combine(left_states, right_states, orders, degree):
        for a, da in left_states:
            for b, db in right_states:
                combined = _combine_coupling_order_tuples(a, b, orders)
                if _coupling_orders_within_limits(combined, limits):
                    yield combined, da + db + degree

    # (coupling orders, twice the alpha_s degree) of the currents of every mask
    states: dict[int, dict[int, set[tuple[CouplingOrders, int]]]] = {}
    for leg in process.legs:
        if leg.outgoing_pdg is not None:
            states.setdefault(1 << (leg.label - 1), {}).setdefault(
                int(leg.outgoing_pdg), set()
            ).add(((), 0))
    rights = _right_particles_by_left(model, color_accuracy=accuracy)
    for mask in _masks_by_size(full_mask):
        if mask & (mask - 1) == 0 or mask == full_mask or not allowed(mask):
            continue
        for left_mask, right_mask in _ordered_splits(mask):
            if not (allowed(left_mask) and allowed(right_mask)):
                continue
            left, right = states.get(left_mask), states.get(right_mask)
            if not left or not right:
                continue
            for left_pdg, left_states in tuple(left.items()):
                for right_pdg in rights.get(left_pdg, ()):
                    right_states = right.get(right_pdg)
                    if not right_states:
                        continue
                    for vertex in vertices(left_pdg, right_pdg):
                        degree = degrees.get(int(vertex.kind))
                        if degree is None:
                            return None
                        states.setdefault(mask, {}).setdefault(
                            vertex.particles[2], set()
                        ).update(
                            combine(
                                left_states,
                                right_states,
                                model.vertex_coupling_orders(vertex),
                                degree,
                            )
                        )
    closure_rights = _closure_right_particles_by_left(model, color_accuracy=accuracy)
    totals: set[int] = set()
    for left_mask, right_mask in splits:
        left, right = states.get(left_mask), states.get(right_mask)
        if not left or not right:
            continue
        for left_pdg, left_states in left.items():
            for right_pdg in closure_rights.get(left_pdg, ()):
                right_states = right.get(right_pdg)
                if not right_states:
                    continue
                if model.direct_contraction_possible(left_pdg, right_pdg):
                    totals.update(
                        d for _, d in combine(left_states, right_states, (), 0)
                    )
                for vertex in vertices(left_pdg, right_pdg):
                    if model.closure_contraction_ir(vertex.particles[2]) is None:
                        continue
                    degree = degrees.get(int(vertex.kind))
                    if degree is None:
                        return None
                    totals.update(
                        d
                        for _, d in combine(
                            left_states,
                            right_states,
                            model.vertex_coupling_orders(vertex),
                            degree,
                        )
                    )
    return totals


__all__ = ["alpha_s_parameter", "alpha_s_power"]
