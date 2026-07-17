# SPDX-License-Identifier: 0BSD
"""Neutral schema-v3 runtime layout assembly for generated DAGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models.base import Model
from .contracts import RuntimeExpressionSchema, runtime_coupling_parameter_names
from .dag_types import CurrentNode, GenericDAG, InteractionNode
from .physics_metadata import build_resolved_physics_payload
from .runtime_amplitudes import build_runtime_amplitude_stage

RUNTIME_PHYSICS_SCHEMA_VERSION = 1


def build_runtime_expression_schema(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
) -> RuntimeExpressionSchema:
    return RuntimeExpressionSchema.from_mapping(
        build_runtime_schema(dag, model, process_id=process_id)
    )


def build_runtime_schema(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
) -> dict[str, object]:
    """Build the stage compiler and runtime metadata for one concrete process."""

    public_process_id = process_id or dag.process.key

    current_slots = _current_slots(dag)
    interaction_inputs = {
        current_id
        for interaction in dag.interactions
        for current_id in (interaction.left_id, interaction.right_id)
    }
    amplitude_inputs = {
        current_id
        for root in dag.amplitude_roots
        for current_id in (root.left_id, root.right_id)
    }
    value_slot_records = _value_slots(
        dag,
        model,
        current_slots=current_slots,
        interaction_inputs=interaction_inputs,
        amplitude_inputs=amplitude_inputs,
    )
    value_slots = {
        (int(slot["current_id"]), str(slot["variant"])): slot
        for slot in value_slot_records
    }
    momentum_slots = _momentum_slots(dag)
    momentum_slot_by_mask = {
        int(slot["momentum_mask"]): int(slot["momentum_slot_id"])
        for slot in momentum_slots
    }
    amplitude_stage = build_runtime_amplitude_stage(
        dag,
        model,
        current_slots=current_slots,
        value_slots=value_slots,
    )
    model_parameters = _model_parameter_records(
        dag,
        model,
        amplitude_stage=amplitude_stage,
    )
    stages = _stage_records(
        dag,
        model,
        current_slots=current_slots,
        value_slots=value_slots,
        momentum_slot_by_mask=momentum_slot_by_mask,
    )
    external_particles = _external_particles(dag)
    normalization = _normalization(dag, model)
    physics = build_resolved_physics_payload(
        dag,
        model,
        process_id=public_process_id,
        amplitude_stage=amplitude_stage,
        model_parameters=model_parameters,
        normalization=normalization,
    )
    value_count = (
        int(value_slot_records[-1]["component_stop"]) if value_slot_records else 0
    )
    momentum_count = 4 * len(momentum_slots)

    return {
        "schema_version": 1,
        "kind": "pyamplicol-runtime-expression-schema",
        "process_key": public_process_id,
        "process": dag.process.process,
        "color_accuracy": dag.process.color_accuracy,
        "external_particles": external_particles,
        "momentum_conventions": _momentum_conventions(dag),
        "model": _model_payload(dag, model),
        "normalization": normalization,
        "parameter_layout": {
            "value_component_count": value_count,
            "momentum_parameter_count": momentum_count,
            "model_parameter_count": len(model_parameters),
            "parameter_count_if_flattened": (
                value_count + momentum_count + len(model_parameters)
            ),
            "momentum_components_real": True,
            "model_parameters_real": True,
        },
        "model_parameters": model_parameters,
        "physics": physics,
        "current_storage": {
            "component_count": (
                int(current_slots[-1]["component_stop"]) if current_slots else 0
            ),
            "number_type": "complex",
            "current_slots": current_slots,
        },
        "value_storage": {
            "component_count": value_count,
            "number_type": "complex",
            "value_slots": value_slot_records,
        },
        "source_fill": {
            "source_count": len(dag.sources),
            "sources": [
                _source_record(
                    dag,
                    model,
                    current_id=current_id,
                    source_index=source_index,
                    current_slot=current_slots[current_id],
                    value_slot=value_slots[(current_id, "source")],
                )
                for source_index, current_id in enumerate(dag.sources)
            ],
        },
        "momentum_slots": momentum_slots,
        "stages": stages,
        "amplitude_stage": amplitude_stage,
    }


def _current_slots(dag: GenericDAG) -> list[dict[str, object]]:
    offset = 0
    slots: list[dict[str, object]] = []
    for current in dag.currents:
        start = offset
        offset += current.dimension
        slots.append(
            {
                "current_id": current.id,
                "component_start": start,
                "component_stop": offset,
                "dimension": current.dimension,
                "is_source": current.is_source,
                "particle_id": current.index.particle_id,
                "external_mask": current.index.external_mask,
                "external_labels": list(current.index.external_labels),
                "momentum_mask": current.index.momentum_mask,
                "helicity_ancestry": str(current.index.helicity_ancestry),
                "chirality": current.index.chirality,
                "spin_state": _spin_state(current.index.spin_state),
                "flavour_flow": list(current.index.flavour_flow),
                "quantum_number_flow": [
                    [name, expression]
                    for name, expression in current.index.quantum_number_flow
                ],
                "color_state": current.index.color_state.to_json_dict(),
                "auxiliary_kind": current.index.auxiliary_kind,
            }
        )
    return slots


def _value_slots(
    dag: GenericDAG,
    model: Model,
    *,
    current_slots: Sequence[Mapping[str, object]],
    interaction_inputs: set[int],
    amplitude_inputs: set[int],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    offset = 0

    def add(current: CurrentNode, variant: str, applies_propagator: bool) -> None:
        nonlocal offset
        current_slot = current_slots[current.id]
        propagator = model._propagator_ir(
            current.index.particle_id,
            current.index.chirality,
        )
        start = offset
        offset += current.dimension
        records.append(
            {
                "value_slot_id": len(records),
                "current_id": current.id,
                "variant": variant,
                "component_start": start,
                "component_stop": offset,
                "dimension": current.dimension,
                "current_component_start": current_slot["component_start"],
                "current_component_stop": current_slot["component_stop"],
                "is_source": current.is_source,
                "applies_propagator": applies_propagator,
                "particle_id": current.index.particle_id,
                "external_mask": current.index.external_mask,
                "external_labels": list(current.index.external_labels),
                "momentum_mask": current.index.momentum_mask,
                "chirality": current.index.chirality,
                "propagator": propagator.to_json_dict(),
                "used_as_interaction_input": current.id in interaction_inputs,
                "used_as_amplitude_input": current.id in amplitude_inputs,
            }
        )

    for current in dag.currents:
        if current.is_source:
            add(current, "source", False)
            continue
        propagator = model._propagator_ir(
            current.index.particle_id,
            current.index.chirality,
        )
        needs_propagated = (
            current.id in interaction_inputs and propagator.applies_propagator
        )
        needs_unpropagated = current.id in amplitude_inputs
        if (
            needs_unpropagated
            or not needs_propagated
            or (current.id in interaction_inputs and not propagator.applies_propagator)
        ):
            add(current, "unpropagated", False)
        if needs_propagated:
            add(current, "propagated", True)
    return records


def _momentum_slots(dag: GenericDAG) -> list[dict[str, object]]:
    masks = sorted(
        {current.index.momentum_mask for current in dag.currents},
        key=lambda mask: (mask.bit_count(), mask),
    )
    incoming = {leg.label for leg in dag.process.initial_legs}
    result: list[dict[str, object]] = []
    for slot_id, mask in enumerate(masks):
        labels = _mask_labels(mask)
        start = 4 * slot_id
        result.append(
            {
                "momentum_slot_id": slot_id,
                "momentum_mask": mask,
                "external_labels": list(labels),
                "component_start": start,
                "component_stop": start + 4,
                "component_order": ["E", "px", "py", "pz"],
                "real_valued": True,
                "crossed_incoming_labels": [
                    label for label in labels if label in incoming
                ],
                "construction": (
                    "sum all-outgoing momenta for external labels; negate "
                    "physical incoming momenta first"
                ),
            }
        )
    return result


def _stage_records(
    dag: GenericDAG,
    model: Model,
    *,
    current_slots: Sequence[Mapping[str, object]],
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
    momentum_slot_by_mask: Mapping[int, int],
) -> list[dict[str, object]]:
    by_size: dict[int, list[InteractionNode]] = {}
    for interaction in dag.interactions:
        size = len(dag.currents[interaction.result_id].index.external_labels)
        by_size.setdefault(size, []).append(interaction)

    stages: list[dict[str, object]] = []
    for stage_index, size in enumerate(sorted(by_size), start=1):
        interactions = by_size[size]
        records = [
            _interaction_record(
                dag,
                model,
                interaction,
                current_slots=current_slots,
                value_slots=value_slots,
                momentum_slot_by_mask=momentum_slot_by_mask,
            )
            for interaction in interactions
        ]
        input_current_ids = {
            current_id
            for interaction in interactions
            for current_id in (interaction.left_id, interaction.right_id)
        }
        output_current_ids = {interaction.result_id for interaction in interactions}
        input_value_slot_ids = {
            int(record[side]["value_slot_id"])
            for record in records
            for side in ("left_value_slot", "right_value_slot")
        }
        output_value_slot_ids = {
            int(slot["value_slot_id"])
            for record in records
            for slot in _mapping_sequence(record["result_value_slots"])
        }
        momentum_ids = {
            int(slot_id)
            for record in records
            for slot_id in _mapping(record["momentum_slots"]).values()
        }
        stages.append(
            {
                "stage_index": stage_index,
                "stage_kind": "current-combine",
                "subset_size": size,
                "input_current_ids": sorted(input_current_ids),
                "output_current_ids": sorted(output_current_ids),
                "input_value_slot_ids": sorted(input_value_slot_ids),
                "output_value_slot_ids": sorted(output_value_slot_ids),
                "input_momentum_slot_ids": sorted(momentum_ids),
                "interaction_count": len(records),
                "interaction_evaluation_count": len(
                    {
                        (
                            "group",
                            interaction.evaluation_group_id,
                        )
                        if interaction.evaluation_group_id is not None
                        else ("interaction", interaction.id)
                        for interaction in interactions
                    }
                ),
                "interaction_ids": [],
                "interactions_compacted": False,
                "interactions": records,
            }
        )
    return stages


def _interaction_record(
    dag: GenericDAG,
    model: Model,
    interaction: InteractionNode,
    *,
    current_slots: Sequence[Mapping[str, object]],
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
    momentum_slot_by_mask: Mapping[int, int],
) -> dict[str, object]:
    left = dag.currents[interaction.left_id]
    right = dag.currents[interaction.right_id]
    result = dag.currents[interaction.result_id]
    left_value = _input_value_slot(left, model, value_slots)
    right_value = _input_value_slot(right, model, value_slots)
    result_values = _result_value_slots(result, value_slots)
    rule = model.vertex_lowering_rule(interaction.vertex_kind)
    return {
        "interaction_id": interaction.id,
        "vertex_kind": interaction.vertex_kind,
        "vertex_particles": list(interaction.vertex_particles),
        "left_current_id": interaction.left_id,
        "right_current_id": interaction.right_id,
        "result_current_id": interaction.result_id,
        "left_slot": _current_slot_ref(current_slots[interaction.left_id]),
        "right_slot": _current_slot_ref(current_slots[interaction.right_id]),
        "result_slot": _current_slot_ref(current_slots[interaction.result_id]),
        "left_value_slot": _value_slot_ref(left_value),
        "right_value_slot": _value_slot_ref(right_value),
        "result_value_slots": [_value_slot_ref(slot) for slot in result_values],
        "result_requires_propagated_value": any(
            slot["variant"] == "propagated" for slot in result_values
        ),
        "result_requires_unpropagated_value": any(
            slot["variant"] == "unpropagated" for slot in result_values
        ),
        "momentum_slots": {
            "left": momentum_slot_by_mask[left.index.momentum_mask],
            "right": momentum_slot_by_mask[right.index.momentum_mask],
            "result": momentum_slot_by_mask[result.index.momentum_mask],
        },
        "coupling": list(interaction.coupling),
        "coupling_parameter_names": runtime_coupling_parameter_names(
            interaction.vertex_kind,
            interaction.vertex_particles,
            interaction.coupling,
            model=model,
        ),
        "color_weight": list(interaction.color_weight),
        "evaluation_group_id": interaction.evaluation_group_id,
        "evaluation_factor": list(interaction.evaluation_factor),
        "accumulation": "sum-into-result-current",
        "lowering": rule.to_json_dict(),
        "full_tensor_network_ready": interaction.full_tensor_network_ready,
    }


def _model_parameter_records(
    dag: GenericDAG,
    model: Model,
    *,
    amplitude_stage: Mapping[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(name: str, kind: str, default: float, **metadata: object) -> None:
        if name in seen:
            return
        seen.add(name)
        records.append(
            {
                "name": name,
                "kind": kind,
                "parameter_index": len(records),
                "default": float(default),
                **metadata,
            }
        )

    def add_complex(
        name: str,
        value: object,
        *,
        kind: str,
        **metadata: object,
    ) -> None:
        real, imaginary = _complex_pair(value, name)
        for component, default in (("real", real), ("imag", imaginary)):
            add(
                f"{name}.{component}",
                kind,
                default,
                runtime_name=name,
                complex_component=component,
                **metadata,
            )

    for raw_name, value in sorted(
        model.runtime_normalization_parameter_defaults().items()
    ):
        add(str(raw_name), "normalization", float(value))
    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    if callable(defaults_provider):
        type_provider = getattr(model, "runtime_parameter_type", None)
        for raw_name, value in sorted(defaults_provider().items()):
            name = str(raw_name)
            declared = (
                str(type_provider(name)).lower()
                if callable(type_provider)
                else "complex"
            )
            if declared == "complex":
                add_complex(name, value, kind="external_parameter_component")
            else:
                real, imaginary = _complex_pair(value, name)
                if imaginary != 0.0:
                    raise ValueError(
                        "real runtime model parameter "
                        f"{name!r} has an imaginary default"
                    )
                add(name, "external_parameter", real, parameter_type=declared)
        _add_derived_parameter_records(
            dag,
            model,
            amplitude_stage=amplitude_stage,
            add_complex=add_complex,
        )
        return records

    for particle in sorted(model.particles.values(), key=lambda item: item.pdg):
        if float(particle.mass) != 0.0:
            add(
                f"particle.{particle.pdg}.mass",
                "particle_mass",
                float(particle.mass),
                pdg=particle.pdg,
            )
        if float(particle.width) != 0.0:
            add(
                f"particle.{particle.pdg}.width",
                "particle_width",
                float(particle.width),
                pdg=particle.pdg,
            )
    for kind, particles, coupling in _coupling_signatures(dag, amplitude_stage):
        names = runtime_coupling_parameter_names(
            kind,
            particles,
            coupling,
            model=model,
        )
        for component, name in enumerate(names):
            if name is None:
                continue
            add(
                name,
                "coupling_component",
                float(coupling[component]),
                vertex_kind=kind,
                vertex_particles=list(particles),
                component=component,
            )
    return records


def _add_derived_parameter_records(
    dag: GenericDAG,
    model: Model,
    *,
    amplitude_stage: Mapping[str, object],
    add_complex: Any,
) -> None:
    defaults_provider = getattr(model, "runtime_derived_parameter_defaults_for", None)
    if not callable(defaults_provider):
        return
    used = _used_coupling_parameter_names(dag, model, amplitude_stage)
    values = defaults_provider(tuple(sorted(used)))
    domains_provider = getattr(model, "runtime_derived_parameter_domains_for", None)
    domains = (
        domains_provider(tuple(sorted(used))) if callable(domains_provider) else {}
    )
    for raw_name, value in sorted(values.items()):
        name = str(raw_name)
        domain = str(domains.get(raw_name, domains.get(name, "complex")))
        if domain not in {"real", "imaginary", "complex"}:
            raise ValueError(f"unsupported runtime parameter domain {domain!r}")
        add_complex(
            name,
            value,
            kind="derived_parameter_component",
            derived=True,
            complex_domain=domain,
        )


def _used_coupling_parameter_names(
    dag: GenericDAG,
    model: Model,
    amplitude_stage: Mapping[str, object],
) -> set[str]:
    names = {
        name
        for interaction in dag.interactions
        for name in runtime_coupling_parameter_names(
            interaction.vertex_kind,
            interaction.vertex_particles,
            interaction.coupling,
            model=model,
        )
        if name is not None
    }
    for root in _mapping_sequence(amplitude_stage["roots"]):
        raw_names = root.get("coupling_parameter_names")
        if isinstance(raw_names, list):
            names.update(str(name) for name in raw_names if isinstance(name, str))
    runtime_particle_names = getattr(
        model,
        "runtime_parameter_names_for_particle",
        None,
    )
    if callable(runtime_particle_names):
        for particle_id in {current.index.particle_id for current in dag.currents}:
            names.update(str(name) for name in runtime_particle_names(int(particle_id)))
    return names


def _coupling_signatures(
    dag: GenericDAG,
    amplitude_stage: Mapping[str, object],
) -> tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...]:
    signatures = {
        (
            interaction.vertex_kind,
            tuple(interaction.vertex_particles),
            tuple(interaction.coupling),
        )
        for interaction in dag.interactions
    }
    for root in _mapping_sequence(amplitude_stage["roots"]):
        kind = root.get("vertex_kind")
        particles = root.get("vertex_particles")
        coupling = root.get("coupling")
        if (
            isinstance(kind, int)
            and isinstance(particles, list)
            and isinstance(coupling, list)
        ):
            signatures.add(
                (
                    kind,
                    tuple(int(pdg) for pdg in particles),
                    tuple(float(value) for value in coupling),
                )
            )
    return tuple(sorted(signatures))


def _source_record(
    dag: GenericDAG,
    model: Model,
    *,
    current_id: int,
    source_index: int,
    current_slot: Mapping[str, object],
    value_slot: Mapping[str, object],
) -> dict[str, object]:
    current = dag.currents[current_id]
    leg = next(
        (
            candidate
            for candidate in dag.process.legs
            if candidate.label == current.source_leg_label
        ),
        None,
    )
    source_start = sum(
        dag.currents[source_id].dimension for source_id in dag.sources[:source_index]
    )
    source_ir = model._source_ir(current.index.particle_id)
    applied_crossing = (
        source_ir.crossing
        if leg is not None and leg.is_initial
        else type(source_ir.crossing).identity()
    )
    expected_states = {
        (
            state.helicity,
            state.chirality,
            state.spin_state,
        )
        for state in (
            applied_crossing.apply(declared_state)
            for declared_state in source_ir.states
        )
    }
    current_state = (
        int(current.source_helicity),
        int(current.index.chirality),
        current.index.spin_state,
    )
    if current_state not in expected_states:
        raise ValueError(
            f"source current {current_id} state {current_state!r} is not declared "
            f"by model source metadata"
        )
    crossing = (
        "negate-incoming-momentum"
        if applied_crossing.momentum_transform == "negate-four-momentum"
        else "identity"
    )
    return {
        "source_id": source_index,
        "current_id": current_id,
        "current_component_start": current_slot["component_start"],
        "current_component_stop": current_slot["component_stop"],
        "value_slot": _value_slot_ref(value_slot),
        "source_parameter_start": source_start,
        "source_parameter_stop": source_start + current.dimension,
        "leg_label": current.source_leg_label,
        "input_momentum_slot": None if leg is None else leg.label - 1,
        "side": None if leg is None else leg.side,
        "crossing": crossing,
        "physical_pdg": None if leg is None else leg.pdg,
        "outgoing_pdg": current.index.particle_id,
        "particle_id": current.index.particle_id,
        "anti_particle_id": source_ir.identity.anti_pdg_label,
        "source_kind": "external-wavefunction",
        "wavefunction_kind": source_ir.wavefunction_family,
        "source_orientation": source_ir.identity.orientation,
        "source_basis": source_ir.basis,
        "source_ir": source_ir.to_json_dict(),
        "applied_crossing": applied_crossing.to_json_dict(),
        "source_helicity": current.source_helicity,
        "chirality": current.index.chirality,
        "spin_state": _spin_state(current.index.spin_state),
        "dimension": current.dimension,
        "helicity_ancestry": str(current.index.helicity_ancestry),
        "color_state": current.index.color_state.to_json_dict(),
    }


def _normalization(dag: GenericDAG, model: Model) -> dict[str, object]:
    return dict(model.runtime_normalization_payload(dag))


def _model_payload(dag: GenericDAG, model: Model) -> dict[str, object]:
    particle_ids = {current.index.particle_id for current in dag.currents}
    vertex_kinds = {interaction.vertex_kind for interaction in dag.interactions}
    vertex_kinds.update(
        root.vertex_kind for root in dag.amplitude_roots if root.vertex_kind is not None
    )
    return {
        "name": model.name,
        "particles": [
            {
                "pdg": particle.pdg,
                "anti_pdg": particle.anti_pdg,
                "spin": particle.spin,
                "dimension": particle.dimension,
                "color_rep": particle.color_rep,
                "mass": particle.mass,
                "width": particle.width,
                "mass_parameter": _runtime_particle_parameter_name(
                    model,
                    particle.pdg,
                    kind="mass",
                ),
                "width_parameter": _runtime_particle_parameter_name(
                    model,
                    particle.pdg,
                    kind="width",
                ),
                "charge": particle.charge,
            }
            for particle in sorted(model.particles.values(), key=lambda item: item.pdg)
            if particle.pdg in particle_ids or particle.anti_pdg in particle_ids
        ],
        "vertices": [
            {
                "kind": vertex.kind,
                "particles": list(vertex.particles),
                "coupling": list(vertex.coupling),
                "lowering": model.vertex_lowering_rule(vertex.kind).to_json_dict(),
            }
            for vertex in model.vertices
            if vertex.kind in vertex_kinds
        ],
    }


def _runtime_particle_parameter_name(
    model: Model,
    pdg: int,
    *,
    kind: str,
) -> str | None:
    provider = getattr(model, f"runtime_{kind}_parameter_name", None)
    if not callable(provider):
        return None
    name = provider(int(pdg))
    return None if name is None else str(name)


def _external_particles(dag: GenericDAG) -> list[dict[str, object]]:
    return [
        {
            "label": leg.label,
            "index": leg.label - 1,
            "side": leg.side,
            "role": "initial" if leg.is_initial else "final",
            "particle": leg.particle,
            "outgoing_particle": leg.outgoing_particle,
            "pdg": leg.pdg,
            "outgoing_pdg": leg.outgoing_pdg,
            "statistics": leg.statistics,
            "wavefunction_family": leg.wavefunction_family,
            "color_role": leg.color_role,
            "source_orientation": leg.source_orientation,
            "momentum_slot": leg.label - 1,
            "momentum_components": ["E", "px", "py", "pz"],
        }
        for leg in dag.process.legs
    ]


def _momentum_conventions(dag: GenericDAG) -> dict[str, object]:
    incoming = [leg.label for leg in dag.process.initial_legs]
    return {
        "input_shape": ["batch", len(dag.process.legs), 4],
        "component_order": ["E", "px", "py", "pz"],
        "input_momenta": "physical external four-momenta in process order",
        "incoming_labels": incoming,
        "final_state_labels": [leg.label for leg in dag.process.final_legs],
        "all_outgoing_convention": {
            "crossed_incoming_labels": incoming,
            "operation": "negate incoming four-vectors before current/source use",
        },
        "metric": "mostly-minus",
    }


def _input_value_slot(
    current: CurrentNode,
    model: Model,
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> Mapping[str, object]:
    if current.is_source:
        variant = "source"
    else:
        propagator = model._propagator_ir(
            current.index.particle_id,
            current.index.chirality,
        )
        variant = "propagated" if propagator.applies_propagator else "unpropagated"
    try:
        return value_slots[(current.id, variant)]
    except KeyError as exc:
        raise ValueError(
            f"interaction current {current.id} has no {variant} runtime value slot"
        ) from exc


def _result_value_slots(
    current: CurrentNode,
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    result = tuple(
        value_slots[(current.id, variant)]
        for variant in ("unpropagated", "propagated")
        if (current.id, variant) in value_slots
    )
    if result:
        return result
    if current.is_source:
        return (value_slots[(current.id, "source")],)
    raise ValueError(f"result current {current.id} has no runtime value slots")


def _current_slot_ref(slot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: slot[key]
        for key in (
            "current_id",
            "component_start",
            "component_stop",
            "dimension",
        )
    }


def _value_slot_ref(slot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: slot[key]
        for key in (
            "value_slot_id",
            "current_id",
            "variant",
            "component_start",
            "component_stop",
            "dimension",
        )
    }


def _complex_pair(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, complex):
        return float(value.real), float(value.imag)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != 2:
            raise ValueError(
                f"runtime model parameter {name!r} must have two components"
            )
        return float(value[0]), float(value[1])
    if not isinstance(value, str | int | float):
        raise ValueError(f"runtime model parameter {name!r} is not numeric")
    return float(value), 0.0


def _spin_state(value: object) -> object:
    return list(value) if isinstance(value, tuple) else value


def _mask_labels(mask: int) -> tuple[int, ...]:
    return tuple(index + 1 for index in range(mask.bit_length()) if mask & (1 << index))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime schema value must be an object")
    return value


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("runtime schema value must be an array")
    return value


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _sequence(value))


__all__ = [
    "RUNTIME_PHYSICS_SCHEMA_VERSION",
    "build_runtime_expression_schema",
    "build_runtime_schema",
]
