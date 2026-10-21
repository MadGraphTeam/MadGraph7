# SPDX-License-Identifier: 0BSD
"""Internal source-owned scalar binding for native bare process controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .._internal.physics.symbols import symbols as symbol_registry
from .graphs import build_loop_process
from .physical_process import NativeBareProcessProgram
from .request import LoopExternalRequest, prepare_loop_external_request
from .scalar_bindings import NativeScalarBindings, NativeScaleBindings
from .vertex_frames import certified_source_dirac_frame


@dataclass(frozen=True, slots=True)
class NativeSourceBareProgram:
    """One source structure, one exact global state, and one native bare program.

    Parameters remain global. Updating the state recomputes selected values and
    active diagnostics atomically. Numerical event evaluation then consumes its
    immutable native snapshot directly. The local Dirac source frame is mapped
    to the certified physical FFV/FFS frame. This internal interface does not
    certify UV/R2 completion, a general model or a complex-mass scheme.
    """

    structure: Any
    state: NativeScalarBindings
    bare: NativeBareProcessProgram
    external_request: LoopExternalRequest

    @property
    def prepared(self):
        return self.bare.prepared

    def rebind(self, **options):
        """Re-evaluate one captured exact global state in a fresh carrier."""
        bare = self.bare.rebind(**options)
        return replace(self, state=bare.model_state, bare=bare)

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
    state_kinds=None,
    initial_average=None,
    born_order_bounds=None,
    loop_order_bounds=None,
    allowed_pairs=None,
    parameters=None,
    stability=None,
    arithmetic="double",
    precision=None,
    backend=None,
    prepared_counterterms=None,
    **options,
):
    """Prepare source graph currents without lowering scalars through floats."""
    if backend is None:
        from .native import load_loop_native

        backend = load_loop_native()
    from symbolica import E

    if structure.scalars.library.raw is not structure.raw:
        raise ValueError("source structure and scalar owner differ")
    if structure.request_scope is not None:
        structure.request_scope.require_bounds(born_order_bounds, loop_order_bounds)
    if options.get("dimensional_numerators", False):
        raise NotImplementedError("dimensional source vertex frames are not certified")
    if "vertex_spinor_frame" in options:
        raise ValueError(
            "physical source programs require their certified vertex frame"
        )
    request = prepare_loop_external_request(structure.native, incoming, outgoing)
    incoming, outgoing = request.incoming, request.outgoing
    if state_kinds is None:
        state_kinds = request.state_kinds
    if initial_average is None:
        initial_average = E(str(request.initial_average))
    model = structure.native
    if prepared_counterterms is not None:
        if (
            prepared_counterterms.ordinary_model is not structure.native
            or prepared_counterterms.source.raw is not structure.raw
            or prepared_counterterms.source.scalar_program is not structure.scalars
            or prepared_counterterms.vertex_spinor_frame != "ufo"
        ):
            raise ValueError(
                "source counterterms must share the structural model and frame"
            )
        model = prepared_counterterms.native
    perturbative = {
        order["name"]: order["fields"].get("perturbative_expansion", 0)
        for order in structure.raw.collections["coupling_orders"]
    }
    process = build_loop_process(
        structure.native.process(incoming, outgoing)
        if prepared_counterterms is None
        else prepared_counterterms.process(incoming, outgoing),
        ordinary_model=structure.native,
        perturbative_species_orders=structure.perturbative_species_orders,
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
        for rule in model.vertex_rules
        if rule.name in rules
        for row in rule.couplings
        for name in row
        if name is not None
    }
    for rule in model.vertex_rules:
        if rule.name not in rules:
            continue
        for column, name in enumerate(rule.lorentz_structures):
            if not any(row[column] is not None for row in rule.couplings):
                continue
            lorentz = model.lorentz_structure(name)
            expression = lorentz.structure
            if 2 in lorentz.spins and not certified_source_dirac_frame(lorentz):
                raise NotImplementedError(
                    f"source fermion tensor {name} lacks physical frame conformance"
                )
            if (
                symbol_registry.qualified("UFO::Epsilon")
                in expression.get_all_symbols()
            ):
                raise NotImplementedError(
                    "source epsilon tensors require the certified native phase adapter"
                )
    symbols = {model.coupling(name).symbol.get_name(): name for name in coupling_names}
    source_parameters = {
        symbol_registry.qualified(f"UFO::{record['name']}"): record["name"]
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
        stability=stability,
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
    return NativeSourceBareProgram(structure, state, bare, request)


@dataclass(frozen=True, slots=True)
class NativeSourceModelProgram:
    """Source-owned model-rule assembly with one immutable snapshot per call."""

    source_bare: NativeSourceBareProgram
    virtual: Any
    matched: Any

    @property
    def prepared(self):
        return self.virtual.prepared

    def rebind(self, **options):
        """Reuse complete instructions and one captured parameter revision."""
        virtual = self.virtual.rebind(**options)
        source_bare = replace(
            self.source_bare, state=virtual.bare.model_state, bare=virtual.bare
        )
        return replace(self, source_bare=source_bare, virtual=virtual)

    def update_parameters(self, changes):
        return self.source_bare.update_parameters(changes)

    def evaluate_decimal(self, momenta, scales_squared):
        return self.virtual.evaluate_decimal(momenta, scales_squared)

    def evaluate_ordered_decimal(self, momenta, scales_squared):
        return self.virtual.evaluate_ordered_decimal(momenta, scales_squared)


def source_model_program(
    structure,
    prepared,
    incoming,
    outgoing,
    *,
    source_color=None,
    color_accuracy="full",
    **options,
):
    """Assemble admitted source CT rules with the same physical vertex frame.

    The supplied prepared model must retain every relevant source rule and
    certify every matched tensor basis. This internal interface does not infer
    arbitrary model completeness or a renormalization scheme from the rule list.
    """
    from .counterterms import match_counterterms
    from .physical_counterterms import NativeModelVirtualProgram
    from .wavefunction import match_external_field_counterterms

    scope = getattr(structure, "request_scope", None)
    if scope is not None:
        scope.require_local_counterterms(prepared)
    elif not prepared.source_entries_complete or prepared.unavailable_entries:
        raise ValueError(
            "source model assembly requires the complete source CT collection "
            "and all its species; prepare counterterms without an entry subset "
            "against the full structural model"
        )
    if source_color is not None and source_color.model is not prepared.source:
        raise ValueError(
            "source colour continuation belongs to a different prepared model"
        )
    source_bare = source_bare_program(
        structure, incoming, outgoing, prepared_counterterms=prepared, **options
    )
    process = source_bare.bare.process
    corrections = process.orders.correction_names
    matched = match_counterterms(
        prepared,
        prepared.process(
            source_bare.external_request.incoming,
            source_bare.external_request.outgoing,
            counterterms=True,
        ),
        process,
        selected_corrections=corrections,
    )
    fields = match_external_field_counterterms(
        prepared.source,
        process,
        selected_corrections=corrections,
        allowed_amplitude_orders={
            pair.loop for sector in process.orders.sectors for pair in sector.pairs
        },
    )
    virtual = NativeModelVirtualProgram(
        source_bare.bare,
        matched.insertions,
        source_bindings=(),
        external_fields=fields.insertions,
        source_color=source_color,
        color_accuracy=color_accuracy,
    )
    return NativeSourceModelProgram(source_bare, virtual, matched)
