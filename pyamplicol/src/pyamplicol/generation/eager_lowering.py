# SPDX-License-Identifier: 0BSD
"""Lower proven process DAGs into backend-independent eager execution tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..models.base import Model
from ..models.prepared_catalog import (
    ClosureKernelKey,
    PropagatorKernelKey,
    VertexKernelKey,
)
from .contracts import runtime_coupling_parameter_names
from .dag_types import CurrentNode, GenericDAG, InteractionNode
from .eager_tables import (
    EAGER_PLAN_ABI,
    EAGER_RUNTIME_CAPABILITY,
    EAGER_SELECTOR_DOMAINS_ABI,
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
    EagerSelectorDomainIdRow,
    EagerSelectorDomainRow,
    EagerSelectorGroupRow,
    pack_rows,
)

EAGER_RUNTIME_KIND = "pyamplicol-runtime-eager-execution"


class EagerKernelResolver(Protocol):
    """Resolve model-local prepared kernels without compiling them."""

    def vertex_kernel(self, interaction: InteractionNode) -> EagerResolvedKernel: ...

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None: ...

    def closure_kernel(
        self, root: Mapping[str, object]
    ) -> EagerResolvedKernel | None: ...


@dataclass(frozen=True, slots=True)
class EagerResolvedKernel:
    """Prepared callable plus the exact transformation into its canonical ABI."""

    kernel_id: int
    canonical_input_order: tuple[int, int] = (0, 1)
    normalization_factor: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if self.kernel_id < 0:
            raise ValueError("prepared kernel ID must be nonnegative")
        if self.canonical_input_order not in ((0, 1), (1, 0)):
            raise ValueError("prepared binary input order must be a permutation")
        if complex(*self.normalization_factor) == 0j:
            raise ValueError("prepared kernel normalization factor must be nonzero")


@dataclass(frozen=True, slots=True)
class MappingEagerKernelResolver:
    """Small immutable resolver used by prepared kernel-pack manifests."""

    vertex_kernels: Mapping[int, int]
    propagator_kernels: Mapping[tuple[int, int], int]
    closure_kernels: Mapping[tuple[str, int | None], int]

    def vertex_kernel(self, interaction: InteractionNode) -> EagerResolvedKernel:
        try:
            return EagerResolvedKernel(
                int(self.vertex_kernels[interaction.vertex_kind])
            )
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

    def closure_kernel(
        self, root: Mapping[str, object]
    ) -> EagerResolvedKernel | None:
        kind = str(root.get("kind"))
        vertex_kind = root.get("vertex_kind")
        key = (kind, None if vertex_kind is None else int(vertex_kind))
        if kind == "direct-contraction":
            return None
        try:
            return EagerResolvedKernel(int(self.closure_kernels[key]))
        except KeyError as error:
            raise ValueError(
                f"prepared model has no closure kernel for {key}"
            ) from error


class PreparedCatalogEagerKernelResolver:
    """Resolve typed model bindings retained in a prepared-model bundle."""

    def __init__(
        self,
        dag: GenericDAG,
        manifest: Mapping[str, object],
    ) -> None:
        if manifest.get("abi") != "pyamplicol-prepared-kernel-catalog-v1":
            raise ValueError("prepared model has an incompatible resolver ABI")
        self._dag = dag
        self._vertices: dict[VertexKernelKey, EagerResolvedKernel] = {}
        self._propagators: dict[PropagatorKernelKey, int | None] = {}
        self._closures: dict[ClosureKernelKey, EagerResolvedKernel] = {}
        for raw in _mapping_sequence(
            manifest.get("vertex_bindings"), "resolver.vertex_bindings"
        ):
            key = _vertex_kernel_key(
                _mapping(raw.get("key"), "resolver.vertex.key")
            )
            self._vertices[key] = _resolved_kernel(raw, "resolver.vertex")
        for raw in _mapping_sequence(
            manifest.get("propagator_bindings"), "resolver.propagator_bindings"
        ):
            key_data = _mapping(raw.get("key"), "resolver.propagator.key")
            key = PropagatorKernelKey(
                int(key_data["particle_id"]), int(key_data["chirality"])
            )
            kernel_id = raw.get("kernel_id")
            self._propagators[key] = None if kernel_id is None else int(kernel_id)
        for raw in _mapping_sequence(
            manifest.get("closure_bindings"), "resolver.closure_bindings"
        ):
            key = _closure_kernel_key(
                _mapping(raw.get("key"), "resolver.closure.key")
            )
            self._closures[key] = _resolved_kernel(raw, "resolver.closure")

    def vertex_kernel(self, interaction: InteractionNode) -> EagerResolvedKernel:
        left = self._dag.currents[interaction.left_id]
        right = self._dag.currents[interaction.right_id]
        result = self._dag.currents[interaction.result_id]
        key = VertexKernelKey(
            kind=int(interaction.vertex_kind),
            particles=tuple(int(value) for value in interaction.vertex_particles),
            left_chirality=int(left.index.chirality),
            right_chirality=int(right.index.chirality),
            result_chirality=int(result.index.chirality),
            coupling=tuple(float(value) for value in interaction.coupling),
        )
        try:
            return self._vertices[key]
        except KeyError as error:
            raise ValueError(
                f"prepared model has no vertex binding for {key}"
            ) from error

    def propagator_kernel_id(
        self,
        current: CurrentNode,
        propagator: Mapping[str, object],
    ) -> int | None:
        if not bool(propagator.get("applies_propagator", False)):
            return None
        key = PropagatorKernelKey(
            int(current.index.particle_id), int(current.index.chirality)
        )
        try:
            kernel_id = self._propagators[key]
        except KeyError as error:
            raise ValueError(
                f"prepared model has no propagator binding for {key}"
            ) from error
        if kernel_id is None:
            raise ValueError(
                f"prepared propagator binding {key} does not provide a kernel"
            )
        return kernel_id

    def closure_kernel(
        self, root: Mapping[str, object]
    ) -> EagerResolvedKernel | None:
        if str(root.get("kind")) == "direct-contraction":
            return None
        kind = root.get("vertex_kind")
        particles = root.get("vertex_particles")
        if kind is None or particles is None:
            raise ValueError("prepared vertex closure lacks vertex identity")
        left = self._dag.currents[int(root["left_current_id"])]
        right = self._dag.currents[int(root["right_current_id"])]
        key = ClosureKernelKey(
            kind=int(kind),
            particles=tuple(
                int(value)
                for value in _sequence(particles, "closure.vertex_particles")
            ),
            left_chirality=int(left.index.chirality),
            right_chirality=int(right.index.chirality),
            coupling=tuple(
                float(value)
                for value in _sequence(root.get("coupling"), "closure.coupling")
            ),
        )
        try:
            return self._closures[key]
        except KeyError as error:
            raise ValueError(
                f"prepared model has no closure binding for {key}"
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
class EagerStageSelectorDomains:
    """Domain references aligned one-for-one with one eager execution stage."""

    stage_index: int
    invocation_domains: tuple[EagerSelectorDomainIdRow, ...]
    attachment_domains: tuple[EagerSelectorDomainIdRow, ...]
    unpropagated_finalization_domains: tuple[EagerSelectorDomainIdRow, ...]
    propagated_finalization_domains: tuple[EagerSelectorDomainIdRow, ...]


@dataclass(frozen=True, slots=True)
class EagerSelectorClosureTables:
    """Interned coherent-group dependency domains for selector pruning."""

    domains: tuple[EagerSelectorDomainRow, ...]
    domain_group_ids: tuple[EagerSelectorGroupRow, ...]
    stages: tuple[EagerStageSelectorDomains, ...]
    closure_domains: tuple[EagerSelectorDomainIdRow, ...]

    def __post_init__(self) -> None:
        cursor = 0
        for domain_id, domain in enumerate(self.domains):
            if domain.member_start != cursor:
                raise ValueError(
                    f"eager selector domain {domain_id} does not start at {cursor}"
                )
            stop = cursor + domain.member_count
            if stop > len(self.domain_group_ids):
                raise ValueError(
                    f"eager selector domain {domain_id} exceeds its membership table"
                )
            members = tuple(
                row.coherent_group_id for row in self.domain_group_ids[cursor:stop]
            )
            if members != tuple(sorted(set(members))):
                raise ValueError(
                    f"eager selector domain {domain_id} members must be unique "
                    "and sorted"
                )
            cursor = stop
        if cursor != len(self.domain_group_ids):
            raise ValueError(
                "eager selector domains do not cover their membership table"
            )

        domain_count = len(self.domains)
        for reference in self._domain_references():
            if reference.domain_id >= domain_count:
                raise ValueError(
                    f"eager selector domain ID {reference.domain_id} is out of range"
                )

    def _domain_references(self) -> tuple[EagerSelectorDomainIdRow, ...]:
        return (
            *self.closure_domains,
            *(
                reference
                for stage in self.stages
                for references in (
                    stage.invocation_domains,
                    stage.attachment_domains,
                    stage.unpropagated_finalization_domains,
                    stage.propagated_finalization_domains,
                )
                for reference in references
            ),
        )

    def binary_payloads(self, *, prefix: str) -> dict[str, bytes]:
        payloads = {
            f"{prefix}/selector-domains.bin": pack_rows(self.domains),
            f"{prefix}/selector-domain-group-ids.bin": pack_rows(
                self.domain_group_ids
            ),
            f"{prefix}/closure-domains.bin": pack_rows(self.closure_domains),
        }
        for stage in self.stages:
            base = f"{prefix}/stage-{stage.stage_index}"
            payloads[f"{base}-invocation-domains.bin"] = pack_rows(
                stage.invocation_domains
            )
            payloads[f"{base}-attachment-domains.bin"] = pack_rows(
                stage.attachment_domains
            )
            payloads[f"{base}-unpropagated-finalization-domains.bin"] = pack_rows(
                stage.unpropagated_finalization_domains
            )
            payloads[f"{base}-propagated-finalization-domains.bin"] = pack_rows(
                stage.propagated_finalization_domains
            )
        return payloads

    def to_metadata(self, *, prefix: str) -> dict[str, object]:
        def table(path: str, count: int, row_size: int) -> dict[str, object]:
            return {"path": path, "count": count, "row_size": row_size}

        return {
            "abi": EAGER_SELECTOR_DOMAINS_ABI,
            "domains": table(
                f"{prefix}/selector-domains.bin",
                len(self.domains),
                EagerSelectorDomainRow._STRUCT.size,
            ),
            "domain_group_ids": table(
                f"{prefix}/selector-domain-group-ids.bin",
                len(self.domain_group_ids),
                EagerSelectorGroupRow._STRUCT.size,
            ),
            "stages": [
                {
                    "stage_index": stage.stage_index,
                    "invocation_domains": table(
                        f"{prefix}/stage-{stage.stage_index}-invocation-domains.bin",
                        len(stage.invocation_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                    "attachment_domains": table(
                        f"{prefix}/stage-{stage.stage_index}-attachment-domains.bin",
                        len(stage.attachment_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                    "unpropagated_finalization_domains": table(
                        f"{prefix}/stage-{stage.stage_index}"
                        "-unpropagated-finalization-domains.bin",
                        len(stage.unpropagated_finalization_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                    "propagated_finalization_domains": table(
                        f"{prefix}/stage-{stage.stage_index}"
                        "-propagated-finalization-domains.bin",
                        len(stage.propagated_finalization_domains),
                        EagerSelectorDomainIdRow._STRUCT.size,
                    ),
                }
                for stage in self.stages
            ],
            "closure_domains": table(
                f"{prefix}/closure-domains.bin",
                len(self.closure_domains),
                EagerSelectorDomainIdRow._STRUCT.size,
            ),
        }


@dataclass(frozen=True, slots=True)
class EagerExecutionTables:
    process_key: str
    couplings: tuple[EagerCouplingRow, ...]
    stages: tuple[EagerStageTables, ...]
    closures: tuple[EagerClosureRow, ...]
    selector_closures: EagerSelectorClosureTables | None = None

    def __post_init__(self) -> None:
        selector_closures = self.selector_closures
        if selector_closures is None:
            return
        if len(selector_closures.stages) != len(self.stages):
            raise ValueError("eager selector domains do not cover every stage")
        for stage, domains in zip(
            self.stages,
            selector_closures.stages,
            strict=True,
        ):
            if domains.stage_index != stage.stage_index:
                raise ValueError("eager selector-domain stage index mismatch")
            expected_counts = (
                len(stage.invocations),
                len(stage.attachments),
                len(stage.finalizations),
                len(stage.finalizations),
            )
            actual_counts = (
                len(domains.invocation_domains),
                len(domains.attachment_domains),
                len(domains.unpropagated_finalization_domains),
                len(domains.propagated_finalization_domains),
            )
            if actual_counts != expected_counts:
                raise ValueError(
                    f"eager selector-domain row counts do not match stage "
                    f"{stage.stage_index}"
                )
        if len(selector_closures.closure_domains) != len(self.closures):
            raise ValueError("eager selector domains do not cover every closure")

    @property
    def invocation_count(self) -> int:
        return sum(len(stage.invocations) for stage in self.stages)

    @property
    def attachment_count(self) -> int:
        return sum(len(stage.attachments) for stage in self.stages)

    @property
    def referenced_kernel_ids(self) -> frozenset[int]:
        return frozenset(
            {
                *(
                    row.kernel_id
                    for stage in self.stages
                    for row in stage.invocations
                ),
                *(
                    row.kernel_id
                    for stage in self.stages
                    for row in stage.finalizations
                    if row.kernel_id != MISSING_U32
                ),
                *(
                    row.kernel_id
                    for row in self.closures
                    if row.kernel_id != MISSING_U32
                ),
            }
        )

    def binary_payloads(self, *, prefix: str = "eager") -> dict[str, bytes]:
        payloads = {f"{prefix}/couplings.bin": pack_rows(self.couplings)}
        for stage in self.stages:
            base = f"{prefix}/stage-{stage.stage_index}"
            payloads[f"{base}-invocations.bin"] = pack_rows(stage.invocations)
            payloads[f"{base}-attachments.bin"] = pack_rows(stage.attachments)
            payloads[f"{base}-finalizations.bin"] = pack_rows(stage.finalizations)
        payloads[f"{prefix}/closures.bin"] = pack_rows(self.closures)
        if self.selector_closures is not None:
            payloads.update(self.selector_closures.binary_payloads(prefix=prefix))
        return payloads

    def to_metadata(self, *, prefix: str = "eager") -> dict[str, object]:
        metadata: dict[str, object] = {
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
        if self.selector_closures is not None:
            metadata["selector_closures"] = self.selector_closures.to_metadata(
                prefix=prefix
            )
        return metadata


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
            resolved_kernel = resolver.vertex_kernel(representative)
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
                    * complex(*resolved_kernel.normalization_factor)
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
            input_currents = (left, right)
            ordered_currents = tuple(
                input_currents[index]
                for index in resolved_kernel.canonical_input_order
            )
            invocations.append(
                EagerInvocationRow(
                    resolved_kernel.kernel_id,
                    input_slot_by_current[ordered_currents[0].id],
                    input_slot_by_current[ordered_currents[1].id],
                    momentum_slot_by_mask[ordered_currents[0].index.momentum_mask],
                    momentum_slot_by_mask[ordered_currents[1].index.momentum_mask],
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
        resolved_kernel = resolver.closure_kernel(root)
        coupling_slot_id = MISSING_U32
        if resolved_kernel is not None:
            coupling_slot_id = coupling_catalog.add(
                _sequence(root.get("coupling"), "root.coupling"),
                _optional_sequence(root.get("coupling_parameter_names")),
            )
        color_weight = _complex_pair(root.get("color_weight"), "root.color_weight")
        left_slot_id = int(
            _mapping(root.get("left_value_slot"), "left_value_slot")[
                "value_slot_id"
            ]
        )
        right_slot_id = int(
            _mapping(root.get("right_value_slot"), "right_value_slot")[
                "value_slot_id"
            ]
        )
        if resolved_kernel is not None:
            slots = (left_slot_id, right_slot_id)
            left_slot_id, right_slot_id = (
                slots[index] for index in resolved_kernel.canonical_input_order
            )
            color_weight *= complex(*resolved_kernel.normalization_factor)
        closures.append(
            EagerClosureRow(
                MISSING_U32 if resolved_kernel is None else resolved_kernel.kernel_id,
                left_slot_id,
                right_slot_id,
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
        selector_closures=_build_selector_closure_tables(
            tuple(stages),
            tuple(closures),
            amplitude_stage,
        ),
    )


def _build_selector_closure_tables(
    stages: tuple[EagerStageTables, ...],
    closures: tuple[EagerClosureRow, ...],
    amplitude_stage: Mapping[str, object],
) -> EagerSelectorClosureTables:
    """Propagate coherent-group dependencies backwards through eager rows."""

    roots = _mapping_sequence(amplitude_stage.get("roots"), "amplitude_stage.roots")
    if len(roots) != len(closures):
        raise ValueError("eager closures do not match runtime amplitude roots")

    declared_groups = tuple(
        int(group["group_id"])
        for group in _mapping_sequence(
            amplitude_stage.get("coherent_groups"),
            "amplitude_stage.coherent_groups",
        )
    )
    if len(declared_groups) != len(set(declared_groups)) or any(
        group_id < 0 or group_id > MISSING_U32 for group_id in declared_groups
    ):
        raise ValueError("runtime coherent-group IDs must be unique unsigned integers")
    declared_group_set = set(declared_groups)

    value_domains: dict[int, set[int]] = {}
    closure_domains: list[frozenset[int]] = []
    for closure, root in zip(closures, roots, strict=True):
        if closure.amplitude_index != int(root["output_index"]):
            raise ValueError(
                "eager closure order does not match runtime amplitude roots"
            )
        group_id = int(root["coherent_group_id"])
        if group_id not in declared_group_set:
            raise ValueError(
                f"amplitude root references undeclared coherent group {group_id}"
            )
        domain = frozenset((group_id,))
        closure_domains.append(domain)
        value_domains.setdefault(closure.left_value_slot_id, set()).add(group_id)
        value_domains.setdefault(closure.right_value_slot_id, set()).add(group_id)

    stage_domain_sets: dict[
        int,
        tuple[
            tuple[frozenset[int], ...],
            tuple[frozenset[int], ...],
            tuple[frozenset[int], ...],
            tuple[frozenset[int], ...],
        ],
    ] = {}
    empty_domain: frozenset[int] = frozenset()

    for stage in reversed(stages):
        unpropagated_domains: list[frozenset[int]] = []
        propagated_domains: list[frozenset[int]] = []
        current_domains: dict[int, frozenset[int]] = {}
        for finalization in stage.finalizations:
            if finalization.current_id in current_domains:
                raise ValueError(
                    f"eager stage {stage.stage_index} finalizes current "
                    f"{finalization.current_id} more than once"
                )
            unpropagated = (
                frozenset(
                    value_domains.get(finalization.unpropagated_value_slot_id, ())
                )
                if finalization.stores_unpropagated
                else empty_domain
            )
            propagated = (
                frozenset(
                    value_domains.get(finalization.propagated_value_slot_id, ())
                )
                if finalization.stores_propagated
                else empty_domain
            )
            unpropagated_domains.append(unpropagated)
            propagated_domains.append(propagated)
            current_domains[finalization.current_id] = unpropagated | propagated

        attachment_domains: list[frozenset[int]] = []
        for attachment in stage.attachments:
            try:
                attachment_domains.append(current_domains[attachment.result_current_id])
            except KeyError as error:
                raise ValueError(
                    f"eager attachment targets current {attachment.result_current_id} "
                    f"without a stage-{stage.stage_index} finalization"
                ) from error

        invocation_domains: list[frozenset[int]] = []
        for invocation in stage.invocations:
            start = invocation.attachment_start
            stop = start + invocation.attachment_count
            domain = frozenset(
                group_id
                for attachment_domain in attachment_domains[start:stop]
                for group_id in attachment_domain
            )
            invocation_domains.append(domain)
            value_domains.setdefault(invocation.left_value_slot_id, set()).update(
                domain
            )
            value_domains.setdefault(invocation.right_value_slot_id, set()).update(
                domain
            )

        stage_domain_sets[stage.stage_index] = (
            tuple(invocation_domains),
            tuple(attachment_domains),
            tuple(unpropagated_domains),
            tuple(propagated_domains),
        )

    unique_domains = {empty_domain, *closure_domains}
    for domain_sets in stage_domain_sets.values():
        for domains in domain_sets:
            unique_domains.update(domains)
    ordered_domains = tuple(
        sorted(unique_domains, key=lambda domain: (len(domain), tuple(sorted(domain))))
    )
    domain_ids = {domain: domain_id for domain_id, domain in enumerate(ordered_domains)}

    domain_rows: list[EagerSelectorDomainRow] = []
    group_rows: list[EagerSelectorGroupRow] = []
    for domain in ordered_domains:
        members = tuple(sorted(domain))
        domain_rows.append(EagerSelectorDomainRow(len(group_rows), len(members)))
        group_rows.extend(EagerSelectorGroupRow(group_id) for group_id in members)

    def references(
        domains: Sequence[frozenset[int]],
    ) -> tuple[EagerSelectorDomainIdRow, ...]:
        return tuple(EagerSelectorDomainIdRow(domain_ids[domain]) for domain in domains)

    selector_stages = []
    for stage in stages:
        (
            invocation_domains,
            attachment_domains,
            unpropagated_domains,
            propagated_domains,
        ) = stage_domain_sets[stage.stage_index]
        selector_stages.append(
            EagerStageSelectorDomains(
                stage_index=stage.stage_index,
                invocation_domains=references(invocation_domains),
                attachment_domains=references(attachment_domains),
                unpropagated_finalization_domains=references(
                    unpropagated_domains
                ),
                propagated_finalization_domains=references(propagated_domains),
            )
        )

    return EagerSelectorClosureTables(
        domains=tuple(domain_rows),
        domain_group_ids=tuple(group_rows),
        stages=tuple(selector_stages),
        closure_domains=references(closure_domains),
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
    output_slot = value_slots[propagator_slot_id]
    propagator = _mapping(
        output_slot.get("propagator"),
        f"current {current.id} propagator",
    )
    kernel_id = (
        resolver.propagator_kernel_id(current, propagator)
        if bool(output_slot.get("applies_propagator", False))
        else None
    )
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


def _resolved_kernel(
    record: Mapping[str, object],
    context: str,
) -> EagerResolvedKernel:
    order = tuple(
        int(value)
        for value in _sequence(
            record.get("canonical_input_order"),
            f"{context}.canonical_input_order",
        )
    )
    factor = tuple(
        float(value)
        for value in _sequence(
            record.get("equivalence_factor"),
            f"{context}.equivalence_factor",
        )
    )
    if len(order) != 2 or len(factor) != 2:
        raise ValueError(f"{context} has malformed input transformation metadata")
    return EagerResolvedKernel(
        kernel_id=int(record["kernel_id"]),
        canonical_input_order=(order[0], order[1]),
        normalization_factor=(factor[0], factor[1]),
    )


def _vertex_kernel_key(record: Mapping[str, object]) -> VertexKernelKey:
    particles = tuple(
        int(value)
        for value in _sequence(record.get("particles"), "vertex.particles")
    )
    coupling = tuple(
        float(value)
        for value in _sequence(record.get("coupling"), "vertex.coupling")
    )
    if len(particles) != 3 or len(coupling) != 2:
        raise ValueError("prepared vertex key has malformed particles or coupling")
    return VertexKernelKey(
        kind=int(record["kind"]),
        particles=(particles[0], particles[1], particles[2]),
        left_chirality=int(record["left_chirality"]),
        right_chirality=int(record["right_chirality"]),
        result_chirality=int(record["result_chirality"]),
        coupling=(coupling[0], coupling[1]),
    )


def _closure_kernel_key(record: Mapping[str, object]) -> ClosureKernelKey:
    particles = tuple(
        int(value)
        for value in _sequence(record.get("particles"), "closure.particles")
    )
    coupling = tuple(
        float(value)
        for value in _sequence(record.get("coupling"), "closure.coupling")
    )
    if len(particles) != 3 or len(coupling) != 2:
        raise ValueError("prepared closure key has malformed particles or coupling")
    return ClosureKernelKey(
        kind=int(record["kind"]),
        particles=(particles[0], particles[1], particles[2]),
        left_chirality=int(record["left_chirality"]),
        right_chirality=int(record["right_chirality"]),
        coupling=(coupling[0], coupling[1]),
    )


__all__ = [
    "EAGER_RUNTIME_KIND",
    "EagerExecutionTables",
    "EagerKernelResolver",
    "EagerResolvedKernel",
    "EagerSelectorClosureTables",
    "EagerStageSelectorDomains",
    "EagerStageTables",
    "MappingEagerKernelResolver",
    "PreparedCatalogEagerKernelResolver",
    "lower_eager_execution_tables",
]
