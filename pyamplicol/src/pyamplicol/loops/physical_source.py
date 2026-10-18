# SPDX-License-Identifier: 0BSD
"""Internal source-owned scalar binding for native bare process controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graphs import build_loop_process
from .physical_process import NativeBareProcessProgram
from .scalar_bindings import NativeScalarBindings, NativeScaleBindings
from .vertex_frames import certified_ffv_source


@dataclass(frozen=True, slots=True)
class NativeSourceBareProgram:
    """One source structure, one exact global state, and one native bare program.

    Parameters remain global. Updating the state recomputes selected values and
    active diagnostics atomically. Numerical event evaluation then consumes its
    immutable native snapshot directly. The local Dirac source frame is mapped
    to the certified physical FFV frame. This internal interface does not
    certify UV/R2 completion, a general model or a complex-mass scheme.
    """

    structure: Any
    state: NativeScalarBindings
    bare: NativeBareProcessProgram

    def update_parameters(self, changes):
        return self.state.update(changes)

    def evaluate_decimal(self, momenta, scales_squared):
        return self.bare.evaluate_decimal(momenta, scales_squared)

    def evaluate_ordered_decimal(self, momenta, scales_squared):
        return self.bare.evaluate_ordered_decimal(momenta, scales_squared)


def source_bare_program(
    structure,
    incoming,
    outgoing,
    *,
    selected_corrections,
    state_kinds,
    initial_average,
    born_order_bounds=None,
    loop_order_bounds=None,
    allowed_pairs=None,
    parameters=None,
    arithmetic="double",
    precision=None,
    backend=None,
    **options,
):
    """Prepare source graph currents without lowering scalars through floats."""
    if backend is None:
        import _loop_reduction as backend
    from symbolica import S

    if structure.scalars.library.raw is not structure.raw:
        raise ValueError("source structure and scalar owner differ")
    if options.get("dimensional_numerators", False):
        raise NotImplementedError("dimensional source vertex frames are not certified")
    if "vertex_spinor_frame" in options:
        raise ValueError(
            "physical source programs require their certified vertex frame"
        )
    perturbative = {
        order["name"]: order["fields"].get("perturbative_expansion", 0)
        for order in structure.raw.collections["coupling_orders"]
    }
    process = build_loop_process(
        structure.native.process(incoming, outgoing),
        selected_corrections=selected_corrections,
        perturbative_orders=perturbative,
        born_order_bounds=born_order_bounds,
        loop_order_bounds=loop_order_bounds,
        allowed_pairs=allowed_pairs,
    )
    graphs = (*process.born, *process.loops)
    rules = {vertex.interaction for graph in graphs for vertex in graph.vertices}
    coupling_names = {
        name
        for rule in structure.native.vertex_rules
        if rule.name in rules
        for row in rule.couplings
        for name in row
        if name is not None
    }
    for rule in structure.native.vertex_rules:
        if rule.name not in rules:
            continue
        for column, name in enumerate(rule.lorentz_structures):
            if not any(row[column] is not None for row in rule.couplings):
                continue
            lorentz = structure.native.lorentz_structure(name)
            expression = lorentz.structure
            if 2 in lorentz.spins and not certified_ffv_source(lorentz):
                raise NotImplementedError(
                    f"source fermion tensor {name} lacks physical frame conformance"
                )
            if S("UFO::Epsilon") in expression.get_all_symbols():
                raise NotImplementedError(
                    "source epsilon tensors require the certified native phase adapter"
                )
    symbols = {
        structure.native.coupling(name).symbol.get_name(): name
        for name in coupling_names
    }
    source_parameters = {
        S(f"UFO::{record['name']}"): record["name"]
        for record in structure.raw.collections["parameters"]
    }
    for graph in graphs:
        for edge in graph.native.edges:
            for symbol in edge.particle.mass.get_all_symbols():
                if symbol not in source_parameters:
                    raise NotImplementedError(
                        "physical source masses require declared parameter symbols"
                    )
                symbols[symbol.get_name()] = source_parameters[symbol]
    outputs = tuple((name, 0) for name in sorted(set(symbols.values())))
    plan = structure.scalar_bindings(outputs)
    positions = {name: index for index, (name, _power) in enumerate(outputs)}
    state = NativeScalarBindings(
        plan,
        arithmetic=arithmetic,
        precision=precision,
        parameters=parameters,
        backend=backend,
    )
    scale_view = NativeScaleBindings(state) if "MU_R" in plan.external_names else None
    bare = NativeBareProcessProgram(
        process,
        state_kinds=tuple(state_kinds),
        initial_average=initial_average,
        model_state=state,
        model_output_indices={
            symbol: positions[name] for symbol, name in symbols.items()
        },
        model_scale_view=scale_view,
        vertex_spinor_frame="native-physical",
        arithmetic=arithmetic,
        precision=precision,
        backend=backend,
        **options,
    )
    return NativeSourceBareProgram(structure, state, bare)
