# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import CouplingOrders, Model, Vertex


def _flat_index(indices: tuple[int, ...], dims: tuple[int, ...]) -> int:
    index = 0
    for value, dim in zip(indices, dims, strict=True):
        index = index * dim + value
    return index


def _number(value: complex | float) -> Any:
    from symbolica import Expression

    return Expression.num(value)


def _as_expression(value: Any) -> Any:
    if isinstance(value, int | float | complex):
        return _number(value)
    return value


def _minkowski_square_expression(momentum: Sequence[Any]) -> Any:
    if len(momentum) != 4:
        raise ValueError("Minkowski momentum needs four components")
    if all(isinstance(value, int | float | complex) for value in momentum):
        p0, p1, p2, p3 = momentum
        return p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3
    p0, p1, p2, p3 = (_as_expression(value) for value in momentum)
    return p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3


def _model_vertex_result_chiralities(
    model: Model,
    vertex: Vertex,
    left_index: Any,
    right_index: Any,
) -> tuple[int, ...]:
    resolver = getattr(model, "_vertex_result_chiralities", None)
    if resolver is None:
        return (0,)
    return tuple(resolver(vertex, left_index, right_index))


def _index_particle_id(index: Any) -> int:
    if hasattr(index, "particle_id"):
        return int(index.particle_id)
    return int(index.pdg)


def _index_chirality(index: Any) -> int:
    return int(getattr(index, "chirality", 0))


def _index_flavour_flow(index: Any) -> tuple[int, ...]:
    flow = getattr(index, "flavour_flow", None)
    if flow is None:
        return (_index_particle_id(index),)
    return tuple(int(value) for value in flow)


def _index_coupling_orders(index: Any) -> CouplingOrders:
    orders = getattr(index, "coupling_orders", None)
    if orders is None:
        return ()
    return tuple(
        sorted(
            (str(name).upper(), int(value)) for name, value in orders if int(value) != 0
        )
    )


def _append_flavour_transition(
    flow: tuple[int, ...],
    result_particle: int,
) -> tuple[int, ...]:
    if flow and flow[-1] == result_particle:
        return flow
    return (*flow, result_particle)


def _expr_vector_slash_terms(vector: tuple[Any, ...]) -> tuple[Any, Any, Any, Any]:
    v0, v1, v2, v3 = vector
    return v0 + v3, v0 - v3, v1 + 1j * v2, v1 - 1j * v2


def _expr_fermion_propagator_weyl(
    fermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    chirality: int,
) -> tuple[Any, ...]:
    energy, px, py, pz = momentum
    denominator = _minkowski_square_expression(momentum)
    prefactor = 1j / denominator
    tmp1 = energy + pz
    tmp2 = energy - pz
    tmp3 = px + 1j * py
    tmp4 = px - 1j * py
    f1, f2 = fermion
    if chirality == 1:
        return (
            (tmp1 * f1 + tmp3 * f2) * prefactor,
            (tmp2 * f2 + tmp4 * f1) * prefactor,
        )
    if chirality == -1:
        return (
            (tmp2 * f1 - tmp3 * f2) * prefactor,
            (tmp1 * f2 - tmp4 * f1) * prefactor,
        )
    raise ValueError("Weyl fermion propagator expression needs nonzero chirality")


def _expr_antifermion_propagator_weyl(
    antifermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    chirality: int,
) -> tuple[Any, ...]:
    energy, px, py, pz = momentum
    denominator = _minkowski_square_expression(momentum)
    prefactor = 1j / denominator
    tmp1 = -(energy + pz)
    tmp2 = -(energy - pz)
    tmp3 = -(px + 1j * py)
    tmp4 = -(px - 1j * py)
    a1, a2 = antifermion
    if chirality == 1:
        return (
            (tmp2 * a1 - tmp4 * a2) * prefactor,
            (tmp1 * a2 - tmp3 * a1) * prefactor,
        )
    if chirality == -1:
        return (
            (tmp1 * a1 + tmp4 * a2) * prefactor,
            (tmp2 * a2 + tmp3 * a1) * prefactor,
        )
    raise ValueError("Weyl antifermion propagator expression needs nonzero chirality")


def _expr_fermion_propagator_dirac(
    fermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    mass: float,
    width: float,
) -> tuple[Any, ...]:
    if len(fermion) != 4 or len(momentum) != 4:
        raise ValueError("Dirac fermion propagator expects four components")
    energy, px, py, pz = momentum
    denominator = (
        _minkowski_square_expression(momentum) - mass * mass + 1j * mass * width
    )
    prefactor = 1j / denominator
    tmp1 = energy + pz
    tmp2 = energy - pz
    tmp3 = px + 1j * py
    tmp4 = px - 1j * py
    f1, f2, f3, f4 = fermion
    return (
        (tmp1 * f3 + tmp3 * f4 + mass * f1) * prefactor,
        (tmp2 * f4 + tmp4 * f3 + mass * f2) * prefactor,
        (tmp2 * f1 - tmp3 * f2 + mass * f3) * prefactor,
        (tmp1 * f2 - tmp4 * f1 + mass * f4) * prefactor,
    )


def _expr_antifermion_propagator_dirac(
    antifermion: tuple[Any, ...],
    momentum: tuple[Any, ...],
    mass: float,
    width: float,
) -> tuple[Any, ...]:
    if len(antifermion) != 4 or len(momentum) != 4:
        raise ValueError("Dirac antifermion propagator expects four components")
    energy, px, py, pz = momentum
    denominator = (
        _minkowski_square_expression(momentum) - mass * mass + 1j * mass * width
    )
    prefactor = 1j / denominator
    tmp1 = -(energy + pz)
    tmp2 = -(energy - pz)
    tmp3 = -(px + 1j * py)
    tmp4 = -(px - 1j * py)
    a1, a2, a3, a4 = antifermion
    return (
        (tmp2 * a3 - tmp4 * a4 + mass * a1) * prefactor,
        (tmp1 * a4 - tmp3 * a3 + mass * a2) * prefactor,
        (tmp1 * a1 + tmp4 * a2 + mass * a3) * prefactor,
        (tmp2 * a2 + tmp3 * a1 + mass * a4) * prefactor,
    )


def _expr_minkowski_dot(
    left: tuple[Any, ...],
    right: tuple[Any, ...],
) -> Any:
    return (
        left[0] * right[0]
        - left[1] * right[1]
        - left[2] * right[2]
        - left[3] * right[3]
    )
