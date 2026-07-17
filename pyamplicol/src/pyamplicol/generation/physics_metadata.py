# SPDX-License-Identifier: 0BSD
"""Strict runtime-physics-v1 metadata derived from one compiled DAG."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product

from ..models.base import Model
from .dag_types import GenericDAG

_HELICITY_WEIGHT_TOLERANCE = 1.0e-12
_NORMALIZATION_EXTENSION_KEYS = (
    "color_accuracy",
    "color_factor",
    "average_factor",
    "identical_factor",
    "global_coupling_factor",
    "qcd_coupling_power",
    "electroweak_coupling_power",
    "couplings_in_stage_evaluators",
    "coupling_policy",
)


def build_resolved_physics_payload(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str,
    amplitude_stage: Mapping[str, object],
    model_parameters: Sequence[Mapping[str, object]],
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Build strict public axes and exact representative-reuse metadata."""

    coherent_groups = _mapping_sequence(amplitude_stage.get("coherent_groups", ()))
    helicities, group_helicities, structural_zero_count = _helicity_metadata(
        dag,
        model,
        coherent_groups,
    )
    color_components, group_colors = _color_metadata(dag, coherent_groups)
    helicity_ids = {
        tuple(_integer(value) for value in _sequence(record["values"])): str(
            record["id"]
        )
        for record in helicities
    }
    reduction_groups = _reduction_groups(
        coherent_groups,
        group_helicities=group_helicities,
        group_colors=group_colors,
        helicity_ids=helicity_ids,
    )
    lc_color_complete = (
        dag.color_coverage == "complete"
        and not dag.color_plan.truncated
        and len(color_components) == _expected_lc_physical_color_count(dag)
    )
    is_lc = dag.process.color_accuracy == "lc"
    return {
        "schema_version": 1,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": dag.process.process,
        "color_accuracy": dag.process.color_accuracy,
        "coverage": {
            "helicities": dag.helicity_coverage,
            "color": ("complete" if is_lc and lc_color_complete else "selected")
            if is_lc
            else "contracted",
            "color_kind": "physical-lc-flows" if is_lc else "contracted-color",
            "structural_zero_helicity_count": structural_zero_count,
        },
        "external_particles": _external_particles(dag),
        "helicities": helicities,
        "color_components": color_components,
        "reduction": {
            "kind": "lc-diagonal" if is_lc else "contracted-color",
            "groups": reduction_groups,
        },
        "model_parameters": _public_model_parameters(model_parameters),
        "selectors": {
            "helicity": True,
            "color_flow": is_lc,
            "contracted_color": False,
        },
        "extensions": {
            "process_key": dag.process.key,
            "coherent_groups": [dict(group) for group in coherent_groups],
            "color_contraction": amplitude_stage.get("color_contraction"),
            "normalization": {
                key: normalization[key]
                for key in _NORMALIZATION_EXTENSION_KEYS
                if key in normalization
            },
            "selected_source_helicities": {
                str(label): helicity
                for label, helicity in dag.selected_source_helicities
            },
        },
    }


def _helicity_metadata(
    dag: GenericDAG,
    model: Model,
    groups: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[int, tuple[tuple[int, ...], ...]], int]:
    represented: dict[tuple[int, ...], tuple[tuple[int, ...], bool]] = {}
    group_members: dict[int, tuple[tuple[int, ...], ...]] = {}
    for group in groups:
        group_id = _integer(group["group_id"])
        vector = tuple(_integer(value) for value in _sequence(group["helicities"]))
        weight = _number(group.get("helicity_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"invalid helicity weight {weight!r} in group {group_id}")
        members = [vector]
        if weight > 1.0 + _HELICITY_WEIGHT_TOLERANCE:
            if not math.isclose(
                weight,
                2.0,
                rel_tol=_HELICITY_WEIGHT_TOLERANCE,
                abs_tol=_HELICITY_WEIGHT_TOLERANCE,
            ):
                raise ValueError(
                    f"unsupported helicity reuse weight {weight!r} in group {group_id}"
                )
            flipped = tuple(-value for value in vector)
            if flipped != vector:
                members.append(flipped)
        group_members[group_id] = tuple(members)
        for index, member in enumerate(members):
            candidate = (vector, index == 0)
            previous = represented.setdefault(member, candidate)
            if previous != candidate:
                raise ValueError(
                    f"inconsistent helicity representative for physical state {member}"
                )

    possible = _possible_helicity_vectors(dag, model)
    structural_zeros = sorted(set(possible) - represented.keys())
    for vector in structural_zeros:
        represented[vector] = (vector, False)

    identifiers = {vector: _helicity_id(vector) for vector in represented}
    records: list[dict[str, object]] = []
    for index, vector in enumerate(sorted(represented)):
        representative, computed = represented[vector]
        structural_zero = vector in structural_zeros
        records.append(
            {
                "id": identifiers[vector],
                "index": index,
                "values": list(vector),
                "computed": computed and not structural_zero,
                "structural_zero": structural_zero,
                "representative_id": identifiers[representative],
                "coefficient": 0.0 if structural_zero else 1.0,
            }
        )
    return records, group_members, len(structural_zeros)


def _possible_helicity_vectors(
    dag: GenericDAG,
    model: Model,
) -> tuple[tuple[int, ...], ...]:
    selected = dict(dag.selected_source_helicities)
    per_leg: list[tuple[int, ...]] = []
    for leg in dag.process.legs:
        if leg.outgoing_pdg is None:
            per_leg.append((0,))
            continue
        particle_id = int(leg.outgoing_pdg)
        source_ir = model._source_ir(particle_id)
        values: set[int] = set()
        for declared_state in source_ir.states:
            state = (
                source_ir.crossing.apply(declared_state)
                if leg.is_initial
                else declared_state
            )
            values.add(int(state.helicity))
        if leg.label in selected:
            requested = selected[leg.label]
            values.intersection_update((requested,))
            if not values:
                raise ValueError(
                    f"selected helicity {requested} is unavailable for external "
                    f"leg {leg.label}"
                )
        per_leg.append(tuple(sorted(values)))
    return tuple(tuple(values) for values in product(*per_leg))


def _color_metadata(
    dag: GenericDAG,
    groups: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[int, tuple[str, ...]]]:
    if dag.process.color_accuracy != "lc":
        identifier = "color:contracted"
        return [
            {
                "kind": "contracted-color",
                "id": identifier,
                "index": 0,
                "description": "coherent sparse contraction of the full color basis",
            }
        ], {_integer(group["group_id"]): (identifier,) for group in groups}

    records: dict[str, dict[str, object]] = {}
    members_by_group: dict[int, tuple[str, ...]] = {}
    for group in groups:
        group_id = _integer(group["group_id"])
        word = tuple(_integer(value) for value in _sequence(group["color_word"]))
        identifier = _color_id(word)
        members = [identifier]
        records.setdefault(
            identifier,
            {
                "kind": "lc-flow",
                "id": identifier,
                "word": list(word),
                "computed": True,
                "representative_id": identifier,
                "coefficient": 1.0,
            },
        )
        helicity_weight = _number(group.get("helicity_weight", 1.0))
        all_sector_weight = _number(group.get("all_sector_weight", helicity_weight))
        if word and all_sector_weight > helicity_weight + _HELICITY_WEIGHT_TOLERANCE:
            reflected = (word[0], *reversed(word[1:]))
            reflected_id = _color_id(reflected)
            if reflected_id != identifier:
                members.append(reflected_id)
                records.setdefault(
                    reflected_id,
                    {
                        "kind": "lc-flow",
                        "id": reflected_id,
                        "word": list(reflected),
                        "computed": False,
                        "representative_id": identifier,
                        "coefficient": 1.0,
                    },
                )
        members_by_group[group_id] = tuple(members)
    components = [records[key] for key in sorted(records)]
    for index, component in enumerate(components):
        component["index"] = index
    return components, members_by_group


def _reduction_groups(
    groups: Sequence[Mapping[str, object]],
    *,
    group_helicities: Mapping[int, tuple[tuple[int, ...], ...]],
    group_colors: Mapping[int, tuple[str, ...]],
    helicity_ids: Mapping[tuple[int, ...], str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group in groups:
        group_id = _integer(group["group_id"])
        helicities = group_helicities[group_id]
        colors = group_colors[group_id]
        records.append(
            {
                "id": f"reduction:{group_id}",
                "representative_helicity_id": helicity_ids[helicities[0]],
                "representative_color_id": colors[0],
                "physical_helicity_ids": [
                    helicity_ids[helicity] for helicity in helicities
                ],
                "physical_color_ids": list(colors),
            }
        )
    return records


def _public_model_parameters(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    parameters: dict[str, dict[str, object]] = {}
    for record in records:
        raw_kind = str(record.get("kind", ""))
        if raw_kind == "runtime_control":
            continue
        name = str(record.get("runtime_name", record.get("name", "")))
        if not name:
            raise ValueError("runtime model parameter has no name")
        kind = _public_parameter_kind(raw_kind)
        parameter = parameters.setdefault(
            name,
            {
                "name": name,
                "kind": kind,
                "default_real": 0.0,
                "default_imaginary": 0.0,
                "mutable": kind != "derived",
            },
        )
        if parameter["kind"] != kind:
            raise ValueError(f"runtime parameter {name!r} has inconsistent kinds")
        component = str(record.get("complex_component", "real"))
        field = "default_imaginary" if component == "imag" else "default_real"
        parameter[field] = _number(record.get("default", 0.0))
    return list(parameters.values())


def _public_parameter_kind(kind: str) -> str:
    if kind == "normalization":
        return "normalization"
    if kind == "particle_mass":
        return "mass"
    if kind == "particle_width":
        return "width"
    if kind == "coupling_component":
        return "coupling"
    if kind in {"external_parameter", "external_parameter_component"}:
        return "external"
    if kind == "derived_parameter_component":
        return "derived"
    raise ValueError(f"unsupported public model-parameter kind {kind!r}")


def _external_particles(dag: GenericDAG) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "label": leg.label,
            "particle": leg.particle,
            "pdg": leg.pdg,
            "role": "initial" if leg.is_initial else "final",
            "momentum_slot": index,
            "momentum_components": ["E", "px", "py", "pz"],
        }
        for index, leg in enumerate(dag.process.legs)
    ]


def _expected_lc_physical_color_count(dag: GenericDAG) -> int:
    if dag.process.color_accuracy != "lc":
        return 0
    count = 0
    for sector in dag.color_plan.sectors:
        count += 1
        if (
            dag.color_plan.trace_reflections_folded
            and sector.kind == "single-trace"
            and len(sector.trace_labels) > 2
        ):
            reflected = (sector.trace_labels[0], *reversed(sector.trace_labels[1:]))
            if reflected != sector.trace_labels:
                count += 1
    return count


def _helicity_id(values: Sequence[int]) -> str:
    return "h:" + ",".join(f"{int(value):+d}" for value in values)


def _color_id(word: Sequence[int]) -> str:
    if word:
        return "flow:" + ",".join(str(int(label)) for label in word)
    return "flow:singlet"


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime physics record must be a mapping")
    return {str(key): item for key, item in value.items()}


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _sequence(value))


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("runtime physics field must be a sequence")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("runtime physics integer field is invalid")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("runtime physics numeric field is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("runtime physics numeric field must be finite")
    return result


__all__ = ["build_resolved_physics_payload"]
