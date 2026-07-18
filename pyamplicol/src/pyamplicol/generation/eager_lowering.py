# SPDX-License-Identifier: 0BSD
"""Lower proven process DAGs into backend-independent eager execution tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..models.base import Model
from .contracts import runtime_coupling_parameter_names
from .dag_types import CurrentNode, GenericDAG, InteractionNode
from .eager_tables import (
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
    pack_rows,
)

EAGER_RUNTIME_KIND = "pyamplicol-runtime-eager-execution"


class EagerKernelResolver(Protocol):
    """Resolve model-local prepared kernels without compiling them."""

    def vertex_kernel_id(self, interaction: InteractionNode) -> int: ...

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None: ...

    def closure_kernel_id(self, root: Mapping[str, object]) -> int | None: ...


@dataclass(frozen=True, slots=True)
class MappingEagerKernelResolver:
    """Small immutable resolver used by prepared kernel-pack manifests."""

    vertex_kernels: Mapping[int, int]
    propagator_kernels: Mapping[tuple[int, int], int]
    closure_kernels: Mapping[tuple[str, int | None], int]

    def vertex_kernel_id(self, interaction: InteractionNode) -> int:
        try:
            return int(self.vertex_kernels[interaction.vertex_kind])
        except KeyError as error:
            raise ValueError(
                f"prepared model has no vertex kernel for kind "
                f"{interaction.vertex_kind}"
            ) from error

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None:
        if not bool(propagator.get("applies_propagator", False)):
            return None
        key = (int(current.index.particle_id), int(current.index.chirality))
        try:
            return int(self.propagator_kernels[key])
        except KeyError as error:
            raise ValueError(
                f"prepared model has no propagator kernel for particle/chirality {key}"
            ) from error

    def closure_kernel_id(self, root: Mapping[str, object]) -> int | None:
        kind = str(root.get("kind"))
        vertex_kind = root.get("vertex_kind")
        key = (kind, None if vertex_kind is None else int(vertex_kind))
        if kind == "direct-contraction":
            return None
        try:
            return int(self.closure_kernels[key])
        except KeyError as error:
            raise ValueError(
                f"prepared model has no closure kernel for {key}"
            ) from error


@dataclass(frozen=True, slots=True)
class EagerStageTables:
    stage_index: int
    subset_size: int
    invocations: tuple[EagerInvocationRow, ...]
    attachments: tuple[EagerAttachmentRow, ...]
    finalizations: tuple[EagerFinalizationRow, ...]

    def __post_init__(self) -> None:
        cursor = 0
        for invocation in self.invocations:
            if invocation.attachment_start != cursor:
                raise ValueError("eager attachment ranges must be contiguous")
            cursor += invocation.attachment_count
        if cursor != len(self.attachments):
            raise ValueError("eager attachment ranges must cover the attachment table")


@dataclass(frozen=True, slots=True)
class EagerExecutionTables:
    process_key: str
    couplings: tuple[EagerCouplingRow, ...]
    stages: tuple[EagerStageTables, ...]
    closures: tuple[EagerClosureRow, ...]

    @property
    def invocation_count(self) -> int:
        return sum(len(stage.invocations) for stage in self.stages)

    @property
    def attachment_count(self) -> int:
        return sum(len(stage.attachments) for stage in self.stages)

    def binary_payloads(self, *, prefix: str = "eager") -> dict[str, bytes]:
        payloads = {f"{prefix}/couplings.bin": pack_rows(self.couplings)}
        for stage in self.stages:
            base = f"{prefix}/stage-{stage.stage_index}"
            payloads[f"{base}-invocations.bin"] = pack_rows(stage.invocations)
            payloads[f"{base}-attachments.bin"] = pack_rows(stage.attachments)
            payloads[f"{base}-finalizations.bin"] = pack_rows(stage.finalizations)
        payloads[f"{prefix}/closures.bin"] = pack_rows(self.closures)
        return payloads

    def to_metadata(self, *, prefix: str = "eager") -> dict[str, object]:
        return {
            "kind": EAGER_RUNTIME_KIND,
            "eager_plan_abi": EAGER_PLAN_ABI,
            "required_runtime_capabilities": [EAGER_RUNTIME_CAPABILITY],
            "process_key": self.process_key,
            "couplings": {
                "path": f"{prefix}/couplings.bin",
                "count": len(self.couplings),
                "row_size": EagerCouplingRow._STRUCT.size,
            },
            "stages": [
                {
                    "stage_index": stage.stage_index,
                    "subset_size": stage.subset_size,
                    "invocations": {
                        "path": (f"{prefix}/stage-{stage.stage_index}-invocations.bin"),
                        "count": len(stage.invocations),
                        "row_size": EagerInvocationRow._STRUCT.size,
                    },
                    "attachments": {
                        "path": (f"{prefix}/stage-{stage.stage_index}-attachments.bin"),
                        "count": len(stage.attachments),
                        "row_size": EagerAttachmentRow._STRUCT.size,
                    },
                    "finalizations": {
                        "path": (
                            f"{prefix}/stage-{stage.stage_index}-finalizations.bin"
                        ),
                        "count": len(stage.finalizations),
                        "row_size": EagerFinalizationRow._STRUCT.size,
                    },
                }
                for stage in self.stages
            ],
            "closures": {
                "path": f"{prefix}/closures.bin",
                "count": len(self.closures),
                "row_size": EagerClosureRow._STRUCT.size,
            },
        }


class _CouplingCatalog:
    def __init__(self, model_parameters: Sequence[Mapping[str, object]]) -> None:
        self.rows: list[EagerCouplingRow] = []
        self._row_ids: dict[EagerCouplingRow, int] = {}
        self._direct = {
            str(record["name"]): int(record["parameter_index"])
            for record in model_parameters
        }
        logical: dict[str, dict[str, int]] = {}
        for record in model_parameters:
            runtime_name = record.get("runtime_name")
            component = record.get("complex_component")
            if isinstance(runtime_name, str) and component in {"real", "imag"}:
                logical.setdefault(runtime_name, {})[str(component)] = int(
                    record["parameter_index"]
                )
        self._logical = logical

    def add(
        self,
        coupling: Sequence[object],
        parameter_names: Sequence[object] | None,
    ) -> int:
        if len(coupling) != 2:
            raise ValueError("eager coupling metadata must contain two components")
        constants = (float(coupling[0]), float(coupling[1]))
        names = tuple(parameter_names or ())
        real_parameter = MISSING_U32
        imag_parameter = MISSING_U32

        first_name = names[0] if names else None
        if isinstance(first_name, str) and first_name in self._logical:
            components = self._logical[first_name]
            if set(components) != {"real", "imag"}:
                raise ValueError(
                    f"logical coupling {first_name!r} lacks real/imaginary slots"
                )
            real_parameter = components["real"]
            imag_parameter = components["imag"]
        else:
            for component, name in enumerate(names[:2]):
                if not isinstance(name, str):
                    continue
                parameter_id = self._direct.get(name)
                if parameter_id is None:
                    continue
                if component == 0:
                    real_parameter = parameter_id
                else:
                    imag_parameter = parameter_id

        row = EagerCouplingRow(
            real_parameter,
            imag_parameter,
            constants[0],
            constants[1],
        )
        existing = self._row_ids.get(row)
        if existing is not None:
            return existing
        row_id = len(self.rows)
        self.rows.append(row)
        self._row_ids[row] = row_id
        return row_id


def lower_eager_execution_tables(
    dag: GenericDAG,
    model: Model,
    runtime_schema: Mapping[str, object],
    resolver: EagerKernelResolver,
) -> EagerExecutionTables:
    """Lower a proven DAG without constructing any backend evaluator."""

    value_slots = {
        int(slot["value_slot_id"]): slot
        for slot in _mapping_sequence(
            _mapping(runtime_schema.get("value_storage"), "value_storage").get(
                "value_slots"
            ),
            "value_storage.value_slots",
        )
    }
    value_slots_by_current_variant = {
        (int(slot["current_id"]), str(slot["variant"])): int(slot_id)
        for slot_id, slot in value_slots.items()
    }
    momentum_slot_by_mask = {
        int(slot["momentum_mask"]): int(slot["momentum_slot_id"])
        for slot in _mapping_sequence(
            runtime_schema.get("momentum_slots"),
            "momentum_slots",
        )
    }
    model_parameters = _mapping_sequence(
        runtime_schema.get("model_parameters"),
        "model_parameters",
    )
    coupling_catalog = _CouplingCatalog(model_parameters)
    stages: list[EagerStageTables] = []

    for stage_record in _mapping_sequence(runtime_schema.get("stages"), "stages"):
        stage_index = int(stage_record["stage_index"])
        subset_size = int(stage_record["subset_size"])
        interaction_ids = tuple(
            int(value)
            for value in _sequence(
                stage_record.get("interaction_ids"), "interaction_ids"
            )
        )
        input_slot_by_current = {
            int(value_slots[slot_id]["current_id"]): slot_id
            for slot_id in (
                int(value)
                for value in _sequence(
                    stage_record.get("input_value_slot_ids"),
                    "input_value_slot_ids",
                )
            )
        }
        grouped: dict[tuple[str, int], list[InteractionNode]] = {}
        for interaction_id in interaction_ids:
            interaction = dag.interactions[interaction_id]
            group = (
                ("group", int(interaction.evaluation_group_id))
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            grouped.setdefault(group, []).append(interaction)

        invocations: list[EagerInvocationRow] = []
        attachments: list[EagerAttachmentRow] = []
        for interactions in grouped.values():
            representative = interactions[0]
            representative_factor = complex(*representative.evaluation_factor)
            if representative_factor == 0j:
                raise ValueError(
                    "eager evaluation representative factor must be nonzero"
                )
            parameter_names = runtime_coupling_parameter_names(
                representative.vertex_kind,
                representative.vertex_particles,
                representative.coupling,
                model=model,
            )
            coupling_slot_id = coupling_catalog.add(
                representative.coupling,
                parameter_names,
            )
            attachment_start = len(attachments)
            for interaction in interactions:
                if interaction.coupling != representative.coupling:
                    raise ValueError(
                        "one eager evaluation group contains different couplings"
                    )
                factor = (
                    complex(*interaction.color_weight)
                    * complex(*interaction.evaluation_factor)
                    / representative_factor
                )
                attachments.append(
                    EagerAttachmentRow(
                        interaction.result_id,
                        factor.real,
                        factor.imag,
                    )
                )
            left = dag.currents[representative.left_id]
            right = dag.currents[representative.right_id]
            invocations.append(
                EagerInvocationRow(
                    resolver.vertex_kernel_id(representative),
                    input_slot_by_current[left.id],
                    input_slot_by_current[right.id],
                    momentum_slot_by_mask[left.index.momentum_mask],
                    momentum_slot_by_mask[right.index.momentum_mask],
                    coupling_slot_id,
                    attachment_start,
                    len(interactions),
                )
            )

        finalizations = tuple(
            _finalization_row(
                dag.currents[int(current_id)],
                value_slots,
                value_slots_by_current_variant,
                momentum_slot_by_mask,
                resolver,
            )
            for current_id in _sequence(
                stage_record.get("output_current_ids"),
                "output_current_ids",
            )
        )
        stages.append(
            EagerStageTables(
                stage_index=stage_index,
                subset_size=subset_size,
                invocations=tuple(invocations),
                attachments=tuple(attachments),
                finalizations=finalizations,
            )
        )

    amplitude_stage = _mapping(
        runtime_schema.get("amplitude_stage"),
        "amplitude_stage",
    )
    closures: list[EagerClosureRow] = []
    for root in _mapping_sequence(
        amplitude_stage.get("roots"), "amplitude_stage.roots"
    ):
        kernel_id = resolver.closure_kernel_id(root)
        coupling_slot_id = MISSING_U32
        if kernel_id is not None:
            coupling_slot_id = coupling_catalog.add(
                _sequence(root.get("coupling"), "root.coupling"),
                _optional_sequence(root.get("coupling_parameter_names")),
            )
        color_weight = _complex_pair(root.get("color_weight"), "root.color_weight")
        closures.append(
            EagerClosureRow(
                MISSING_U32 if kernel_id is None else kernel_id,
                int(
                    _mapping(root.get("left_value_slot"), "left_value_slot")[
                        "value_slot_id"
                    ]
                ),
                int(
                    _mapping(root.get("right_value_slot"), "right_value_slot")[
                        "value_slot_id"
                    ]
                ),
                int(root["output_index"]),
                coupling_slot_id,
                color_weight.real,
                color_weight.imag,
            )
        )

    return EagerExecutionTables(
        process_key=str(runtime_schema.get("process_key", dag.process.key)),
        couplings=tuple(coupling_catalog.rows),
        stages=tuple(stages),
        closures=tuple(closures),
    )


def _finalization_row(
    current: CurrentNode,
    value_slots: Mapping[int, Mapping[str, object]],
    value_slots_by_current_variant: Mapping[tuple[int, str], int],
    momentum_slot_by_mask: Mapping[int, int],
    resolver: EagerKernelResolver,
) -> EagerFinalizationRow:
    unpropagated = value_slots_by_current_variant.get(
        (current.id, "unpropagated"),
        MISSING_U32,
    )
    propagated = value_slots_by_current_variant.get(
        (current.id, "propagated"),
        MISSING_U32,
    )
    if unpropagated == MISSING_U32 and propagated == MISSING_U32:
        raise ValueError(f"eager current {current.id} has no output value slot")
    propagator_slot_id = propagated if propagated != MISSING_U32 else unpropagated
    propagator = _mapping(
        value_slots[propagator_slot_id].get("propagator"),
        f"current {current.id} propagator",
    )
    kernel_id = resolver.propagator_kernel_id(current, propagator)
    if propagated != MISSING_U32 and kernel_id is None:
        raise ValueError(
            f"eager propagated current {current.id} has no prepared propagator kernel"
        )
    return EagerFinalizationRow(
        MISSING_U32 if kernel_id is None else kernel_id,
        current.id,
        unpropagated,
        propagated,
        momentum_slot_by_mask[current.index.momentum_mask],
    )


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a sequence")
    return value


def _optional_sequence(value: object) -> Sequence[object] | None:
    if value is None:
        return None
    return _sequence(value, "optional sequence")


def _mapping_sequence(value: object, context: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, f"{context}[{index}]")
        for index, item in enumerate(_sequence(value, context))
    )


def _complex_pair(value: object, context: str) -> complex:
    components = _sequence(value, context)
    if len(components) != 2:
        raise ValueError(f"{context} must contain two components")
    return complex(float(components[0]), float(components[1]))


__all__ = [
    "EAGER_RUNTIME_KIND",
    "EagerExecutionTables",
    "EagerKernelResolver",
    "EagerStageTables",
    "MappingEagerKernelResolver",
    "lower_eager_execution_tables",
]
