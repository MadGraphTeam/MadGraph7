# SPDX-License-Identifier: 0BSD
"""Public physics and parameter metadata for compact recurrence artifacts.

The recurrence builder forks before :class:`GenericDAG`, so its public axes
must be derived from the authenticated process projection rather than from a
materialized DAG.  This module keeps that derivation small and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import product
from types import SimpleNamespace

from .._internal.versions import (
    RECURRENCE_BUILDER_INPUT_ABI,
    RECURRENCE_DIRECT_TEMPLATE_ABI,
    RECURRENCE_PLAN_ABI,
    RECURRENCE_RUNTIME_LAYOUT_ABI,
)
from ..models.base import Model
from ..models.recurrence_template import RecurrenceTemplateCatalog
from ..processes.ir import CanonicalProcessIR
from .recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceNormalizationV1,
    RecurrencePublicLCFlowV1,
)

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


def build_recurrence_normalization(
    process: CanonicalProcessIR,
    model: Model,
) -> tuple[RecurrenceNormalizationV1, dict[str, object]]:
    """Bind the existing process normalization without constructing a DAG."""

    # Existing model implementations inspect only ``dag.process``.  Keep the
    # compatibility adapter local while preserving the established payload.
    payload = dict(
        model.runtime_normalization_payload(SimpleNamespace(process=process))
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return (
        RecurrenceNormalizationV1(
            factor=ExactComplexRationalV1(1),
            convention="runtime-normalization-extension-v1",
            semantic_digest=hashlib.sha256(encoded).hexdigest(),
        ),
        payload,
    )


def build_recurrence_physics(
    process: CanonicalProcessIR,
    logical: RecurrenceBuilderLogicalInputV1,
    catalog: RecurrenceTemplateCatalog,
    *,
    process_id: str,
    resolved_helicities: Sequence[Sequence[int]],
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Build strict ``pyamplicol-resolved-physics`` metadata pre-DAG."""

    if logical.process_id != process.key:
        raise ValueError("recurrence projection does not belong to the process IR")
    if logical.layout == "topology-replay" and not resolved_helicities:
        raise ValueError("topology-replay recurrence has no resolved helicities")

    possible_helicities = _possible_helicities(logical)
    resolved = {tuple(int(value) for value in row) for row in resolved_helicities}
    unknown = resolved - set(possible_helicities)
    if unknown:
        raise ValueError(
            "recurrence lowering returned helicities outside generated coverage: "
            f"{sorted(unknown)!r}"
        )
    # All-flow-union selects source states at runtime and deliberately has no
    # helicity-expanded destinations.  Every retained assignment is therefore
    # executable even though some may evaluate to zero dynamically.
    computed_helicities = (
        set(possible_helicities) if logical.layout == "all-flow-union" else resolved
    )
    helicities = []
    structural_zero_count = 0
    for index, values in enumerate(possible_helicities):
        identifier = _helicity_id(values)
        structural_zero = values not in computed_helicities
        structural_zero_count += int(structural_zero)
        helicities.append(
            {
                "id": identifier,
                "index": index,
                "values": list(values),
                "computed": not structural_zero,
                "structural_zero": structural_zero,
                "representative_id": identifier,
                "coefficient": 0.0 if structural_zero else 1.0,
            }
        )

    selected_flow_ids = (
        None
        if logical.selected_public_flow_ids is None
        else set(logical.selected_public_flow_ids)
    )
    retained_flows = _retained_public_flows(logical)
    if not retained_flows:
        raise ValueError("recurrence physics has no retained public LC flow")
    labels_by_slot = {
        leg.source_slot: leg.public_label for leg in logical.external_legs
    }
    color_components = []
    for index, flow in enumerate(retained_flows):
        weight = _complex_factor(flow.reduction_weight)
        color_components.append(
            {
                "kind": "lc-flow",
                "id": flow.public_id,
                "index": index,
                "word": [labels_by_slot[slot] for slot in flow.word_source_slots],
                "computed": True,
                "representative_id": flow.public_id,
                "coefficient": float(
                    weight.real * weight.real + weight.imag * weight.imag
                ),
            }
        )

    selected_sources = {
        row.source_slot: tuple(row.source_state_indices)
        for row in logical.selected_source_coverage or ()
    }
    selected_source_helicities = {}
    for leg in logical.external_legs:
        retained = selected_sources.get(leg.source_slot)
        if retained is None:
            continue
        values = {leg.source_states[index].public_helicity for index in retained}
        if len(values) == 1:
            selected_source_helicities[str(leg.public_label)] = values.pop()

    color_coverage = "complete" if selected_flow_ids is None else "selected"
    helicity_coverage = (
        "complete" if logical.selected_source_coverage is None else "selected"
    )
    public_parameters = _public_model_parameters(catalog)
    return {
        "schema_version": 1,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": process.process,
        "color_accuracy": "lc",
        "coverage": {
            "helicities": helicity_coverage,
            "color": color_coverage,
            "color_kind": "physical-lc-flows",
            "structural_zero_helicity_count": structural_zero_count,
        },
        "external_particles": [
            {
                "index": index,
                "label": int(leg.label),
                "particle": str(leg.particle),
                "pdg": int(leg.pdg),
                "role": "initial" if leg.is_initial else "final",
                "momentum_slot": index,
                "momentum_components": ["E", "px", "py", "pz"],
            }
            for index, leg in enumerate(process.legs)
        ],
        "helicities": helicities,
        "color_components": color_components,
        # The compact recurrence plan owns the high-cardinality destination
        # expansion.  Rusticol hydrates these groups while loading the plan.
        "reduction": {"kind": "lc-diagonal", "groups": []},
        "model_parameters": public_parameters,
        "selectors": {
            "helicity": True,
            "color_flow": True,
            "contracted_color": False,
        },
        "extensions": {
            "process_key": process.key,
            "normalization": {
                key: normalization[key]
                for key in _NORMALIZATION_EXTENSION_KEYS
                if key in normalization
            },
            "selected_source_helicities": selected_source_helicities,
            "runtime_selectors": {
                "kind": "pyamplicol-runtime-selectors",
                "contract_version": 1,
                "provenance": RECURRENCE_PLAN_ABI,
                "axes": {
                    "helicity": {
                        "generation_coverage": helicity_coverage,
                        "generation_selection": selected_source_helicities,
                        "runtime_contract": (
                            "complete-reusable"
                            if helicity_coverage == "complete"
                            else "generation-specialized"
                        ),
                    },
                    "color_flow": {
                        "generation_coverage": color_coverage,
                        "generation_selection": sorted(selected_flow_ids or ()),
                        "runtime_contract": (
                            "complete-reusable"
                            if color_coverage == "complete"
                            else "generation-specialized"
                        ),
                    },
                },
                "generation_specialized_axes": [
                    *([] if helicity_coverage == "complete" else ["helicity"]),
                    *([] if color_coverage == "complete" else ["color_flow"]),
                ],
            },
            "recurrence_runtime_reduction": {
                "kind": "pyamplicol-recurrence-native-reduction-v2",
                "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
                "container_path": "recurrence-runtime.pacbin",
                "plan_member_path": "plan/recurrence-direct-plan-v2.bin",
            },
            "recurrence": {
                "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
                "plan_abi": RECURRENCE_PLAN_ABI,
                "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
                "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
                "lc_flow_layout": logical.layout,
            },
        },
    }


def build_recurrence_runtime_metadata(
    logical: RecurrenceBuilderLogicalInputV1,
    catalog: RecurrenceTemplateCatalog,
    model: Model,
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Return bounded source, parameter, and normalization runtime metadata."""

    values_by_name: dict[str, complex] = {}
    for provider_name in (
        "runtime_parameter_defaults",
        "runtime_derived_parameter_defaults",
    ):
        provider = getattr(model, provider_name, None)
        if not callable(provider):
            continue
        for name, raw_value in provider().items():
            values_by_name[str(name)] = _complex_value(raw_value)

    prepared_defaults = [0j] * len(catalog.parameters)
    for parameter in catalog.parameters:
        prepared_id = parameter.prepared_parameter_id
        if prepared_id is None:
            continue
        if prepared_id >= len(prepared_defaults):
            raise ValueError(
                f"prepared parameter ID {prepared_id} exceeds catalog size"
            )
        value = values_by_name.get(parameter.name)
        if value is None and parameter.default_value is not None:
            value = _complex_factor(parameter.default_value)
        if value is None:
            raise ValueError(
                f"recurrence prepared parameter {parameter.name!r} has no default"
            )
        prepared_defaults[prepared_id] = value

    runtime_parameters = []
    for projection in sorted(
        logical.parameter_projection, key=lambda item: item.runtime_slot
    ):
        parameter = catalog.parameters[projection.parameter_template_id]
        if parameter.name != projection.runtime_name:
            raise ValueError(
                "recurrence parameter projection name disagrees with its template"
            )
        value = values_by_name.get(parameter.name)
        if value is None and parameter.default_value is not None:
            value = _complex_factor(parameter.default_value)
        if value is None:
            raise ValueError(
                f"recurrence runtime parameter {parameter.name!r} has no default"
            )
        is_complex = parameter.value_type == "complex"
        component_name = "real" if projection.component == 0 else "imag"
        runtime_parameters.append(
            {
                "name": (
                    f"{parameter.name}.{component_name}"
                    if is_complex
                    else parameter.name
                ),
                "kind": parameter.parameter_kind,
                "parameter_index": projection.runtime_slot,
                "default": float(
                    value.real if projection.component == 0 else value.imag
                ),
                "runtime_name": parameter.name if is_complex else None,
                "complex_component": component_name if is_complex else None,
            }
        )

    state_index_by_id = {
        state.template_id: index for index, state in enumerate(catalog.current_states)
    }
    referenced_source_ids = sorted(
        {
            state.source_template_id
            for leg in logical.external_legs
            for state in leg.source_states
        }
    )
    source_templates = []
    particle_masses: dict[int, float] = {}
    parameter_names = {parameter.name for parameter in catalog.parameters}
    for source_template_id in referenced_source_ids:
        source = catalog.sources[source_template_id]
        state = next(
            item
            for item in catalog.current_states
            if item.template_id == source.state_template_id
        )
        source_ir = model._source_ir(state.particle_id)
        particle = model.particle(state.particle_id)
        mass_parameter = _runtime_particle_parameter_name(
            source_ir.mass_parameter,
            particle_pdg=int(particle.pdg),
            kind="mass",
            available_names=parameter_names,
        )
        width_parameter = _runtime_particle_parameter_name(
            source_ir.width_parameter,
            particle_pdg=int(particle.pdg),
            kind="width",
            available_names=parameter_names,
        )
        try:
            crossing = _runtime_crossing(json.loads(source.crossing))
        except json.JSONDecodeError as exc:  # pragma: no cover - catalog validated
            raise ValueError(
                f"recurrence source {source.template_id!r} has malformed crossing JSON"
            ) from exc
        source_ir_payload = source_ir.to_json_dict()
        # Built-in models expose particle masses directly rather than through
        # named SourceIR parameters.  Prepared kernels already use the generic
        # ``particle.<pdg>.<kind>`` fallback; retain the same name here so
        # source filling and runtime parameter updates share one contract.
        source_ir_payload["mass_parameter"] = mass_parameter
        source_ir_payload["width_parameter"] = width_parameter
        if crossing != source_ir_payload["crossing"]:
            raise ValueError(
                f"recurrence source {source.template_id!r} crossing disagrees "
                "with its typed SourceIR"
            )
        source_templates.append(
            {
                "source_template_id": source_template_id,
                "current_state_template_id": state_index_by_id[state.template_id],
                "dimension": state.dimension,
                "helicity": source.helicity,
                "chirality": state.chirality,
                "spin_state": source.spin_state,
                "source_ir": source_ir_payload,
                "crossing": crossing,
            }
        )
        mass = float(model.mass(state.particle_id))
        if mass_parameter is not None:
            mass_value = values_by_name.get(mass_parameter)
            if mass_value is not None:
                if mass_value.imag != 0.0:
                    raise ValueError(
                        "recurrence source mass parameter must be real: "
                        f"{mass_parameter!r}"
                    )
                mass = float(mass_value.real)
        previous = particle_masses.setdefault(state.particle_id, mass)
        if previous != mass:
            raise ValueError(
                "recurrence particle "
                f"{state.particle_id} has inconsistent source masses"
            )

    return {
        "public_color_flows": [
            {
                "public_id": flow.public_id,
                "construction_sector_id": flow.construction_sector_id,
                "target_sector_id": target_sector_id,
            }
            for target_sector_id, flow in enumerate(_retained_public_flows(logical))
        ],
        "runtime_parameters": runtime_parameters,
        "prepared_parameter_defaults": [
            [float(value.real), float(value.imag)] for value in prepared_defaults
        ],
        "parameter_projection": [
            {
                "runtime_slot": row.runtime_slot,
                "runtime_name": row.runtime_name,
                "parameter_template_id": row.parameter_template_id,
                "prepared_parameter_id": row.prepared_parameter_id,
                "component": row.component,
            }
            for row in logical.parameter_projection
        ],
        "source_templates": source_templates,
        "external_legs": [
            {
                "source_slot": leg.source_slot,
                "public_label": leg.public_label,
                "physical_pdg": leg.physical_pdg,
                "outgoing_pdg": leg.outgoing_pdg,
                "is_initial": leg.is_initial,
            }
            for leg in logical.external_legs
        ],
        "particle_masses": [
            {"outgoing_pdg": pdg, "mass": mass}
            for pdg, mass in sorted(particle_masses.items())
        ],
        "normalization": {
            key: normalization[key]
            for key in _NORMALIZATION_EXTENSION_KEYS
            if key in normalization
        },
    }


def _runtime_particle_parameter_name(
    declared: str | None,
    *,
    particle_pdg: int,
    kind: str,
    available_names: set[str],
) -> str | None:
    """Resolve the model-owned name or its prepared-catalog fallback."""

    if declared is not None:
        return declared
    fallback = f"particle.{particle_pdg}.{kind}"
    return fallback if fallback in available_names else None


def _retained_public_flows(
    logical: RecurrenceBuilderLogicalInputV1,
) -> tuple[RecurrencePublicLCFlowV1, ...]:
    selected = (
        None
        if logical.selected_public_flow_ids is None
        else set(logical.selected_public_flow_ids)
    )
    retained = tuple(
        flow
        for flow in logical.public_flows
        if selected is None or flow.flow_id in selected
    )
    if not retained:
        raise ValueError("recurrence metadata has no retained public LC flow")
    return retained


def recurrence_referenced_kernel_ids(
    logical: RecurrenceBuilderLogicalInputV1,
) -> frozenset[int]:
    """Return exactly the prepared kernels named by process template references."""

    return frozenset(
        reference.prepared_kernel_id
        for reference in logical.semantic_template_references
        if reference.prepared_kernel_id is not None
    )


def _possible_helicities(
    logical: RecurrenceBuilderLogicalInputV1,
) -> tuple[tuple[int, ...], ...]:
    selected = {
        row.source_slot: set(row.source_state_indices)
        for row in logical.selected_source_coverage or ()
    }
    choices = []
    for leg in logical.external_legs:
        retained = selected.get(leg.source_slot)
        choices.append(
            tuple(
                state.public_helicity
                for state in leg.source_states
                if retained is None or state.state_index in retained
            )
        )
    return tuple(sorted(product(*choices)))


def _public_model_parameters(
    catalog: RecurrenceTemplateCatalog,
) -> list[dict[str, object]]:
    result = []
    for parameter in sorted(catalog.parameters, key=lambda item: item.name):
        if parameter.parameter_kind == "constant":
            continue
        default = (
            0j
            if parameter.default_value is None
            else _complex_factor(parameter.default_value)
        )
        if parameter.name.startswith("normalization."):
            kind = "normalization"
        elif parameter.parameter_kind == "derived":
            kind = "derived"
        else:
            kind = "external"
        result.append(
            {
                "name": parameter.name,
                "kind": kind,
                "default_real": float(default.real),
                "default_imaginary": float(default.imag),
                "mutable": bool(parameter.mutable),
            }
        )
    return result


def _complex_factor(value: object) -> complex:
    return complex(
        int(value.real_numerator) / int(value.real_denominator),
        int(value.imag_numerator) / int(value.imag_denominator),
    )


def _runtime_crossing(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("recurrence source crossing must be an object")
    result = dict(value)
    phase = result.get("phase")
    if not isinstance(phase, Mapping):
        raise ValueError("recurrence source crossing phase must be exact")
    result["phase"] = [
        int(phase["real_numerator"]) / int(phase["real_denominator"]),
        int(phase["imag_numerator"]) / int(phase["imag_denominator"]),
    ]
    return result


def _complex_value(value: object) -> complex:
    if isinstance(value, tuple):
        return complex(*value)
    return complex(value)  # type: ignore[arg-type]


def _helicity_id(values: Sequence[int]) -> str:
    return "h:" + ",".join(f"{int(value):+d}" for value in values)


__all__ = [
    "build_recurrence_normalization",
    "build_recurrence_physics",
    "build_recurrence_runtime_metadata",
    "recurrence_referenced_kernel_ids",
]
