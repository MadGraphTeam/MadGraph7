# SPDX-License-Identifier: 0BSD
"""Canonical IR compilation for the hand-written built-in SM."""

from __future__ import annotations

from collections.abc import Mapping

from .. import compiler_symbolica as _sym
from ..compiler_records import _mappings, _order, _parameter, _particle, _sequence
from ..contracts import CompiledModelIR, CompiledVertexTerm


def compile_model_ir(model: Mapping[str, object]) -> CompiledModelIR:
    _sym._ensure_symbolica()
    particles = tuple(_particle(item) for item in _mappings(model.get("particles")))
    terms = tuple(
        CompiledVertexTerm(
            id=index,
            vertex=str(vertex["name"]),
            particles=tuple(str(value) for value in _sequence(vertex["particles"])),
            color_index=0,
            lorentz_index=0,
            color_source="built-in",
            color_expression="built-in",
            lorentz_name=str(vertex["builtin_kind"]),
            lorentz_source="built-in",
            lorentz_expression="built-in",
            coupling=f"builtin_coupling_{index}",
            coupling_expression=str(vertex.get("builtin_coupling", [1.0, 0.0])),
            coupling_orders=(),
            backend="built-in",
        )
        for index, vertex in enumerate(_mappings(model.get("vertex_rules")))
    )
    return CompiledModelIR(
        name=str(model.get("name", "built-in-sm")),
        orders=tuple(_order(item) for item in _mappings(model.get("orders"))),
        parameters=tuple(
            _parameter(item) for item in _mappings(model.get("parameters"))
        ),
        particles=particles,
        couplings=(),
        propagators=(),
        vertex_terms=terms,
        oriented_kernels=(),
    )


__all__ = ["compile_model_ir"]
