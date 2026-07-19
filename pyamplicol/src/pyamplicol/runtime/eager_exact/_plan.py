# SPDX-License-Identifier: 0BSD
"""Validated in-memory representation of an eager exact-execution plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.artifacts.manifest import ArtifactManifest
from pyamplicol.artifacts.security import confined_path, normalize_relative_path
from pyamplicol.generation.eager_tables import (
    EAGER_KERNEL_ABI,
    MISSING_U32,
    EagerAttachmentRow,
    EagerClosureRow,
    EagerCouplingRow,
    EagerFinalizationRow,
    EagerInvocationRow,
)
from pyamplicol.models.prepared import (
    PreparedKernelPack,
    PreparedKernelRecord,
    PreparedModelBundleError,
)
from pyamplicol.runtime.eager_exact._contracts import (
    _PREPARED_CATALOG_ABI,
    _SUPPORTED_PREPARED_BACKENDS,
    _component_slots,
    _ComponentSlot,
    _direct_coefficients,
    _integer,
    _joined_payload_path,
    _KernelLoader,
    _LazyExactKernel,
    _load_table,
    _mapping,
    _PayloadIndex,
    _read_json,
    _sequence,
    _validate_execution_header,
)


@dataclass(frozen=True, slots=True)
class _ExactStage:
    stage_index: int
    invocations: tuple[EagerInvocationRow, ...]
    attachments: tuple[EagerAttachmentRow, ...]
    finalizations: tuple[EagerFinalizationRow, ...]


@dataclass(slots=True)
class _EagerExactPlan:
    runtime_schema: Mapping[str, object]
    kernels: Mapping[int, _LazyExactKernel]
    value_slots: tuple[_ComponentSlot, ...]
    momentum_slots: tuple[_ComponentSlot, ...]
    current_slots: tuple[_ComponentSlot, ...]
    value_component_count: int
    momentum_component_count: int
    current_component_count: int
    parameter_count: int
    amplitude_count: int
    couplings: tuple[EagerCouplingRow, ...]
    stages: tuple[_ExactStage, ...]
    closures: tuple[EagerClosureRow, ...]

    @classmethod
    def load(
        cls,
        *,
        artifact_root: Path,
        process_root: Path,
        process_id: str,
        execution: Mapping[str, object],
        manifest: ArtifactManifest,
        kernel_loader: _KernelLoader,
    ) -> _EagerExactPlan:
        _validate_execution_header(execution)
        payloads = _PayloadIndex.from_manifest(manifest)
        kernel_reference = _mapping(execution.get("kernel_pack"), "kernel_pack")
        pack_path = cast(str, kernel_reference.get("manifest_path"))
        if not isinstance(pack_path, str):
            raise ArtifactError("kernel_pack.manifest_path must be a string")
        payload_root_name = kernel_reference.get("payload_root")
        if not isinstance(payload_root_name, str):
            raise ArtifactError("kernel_pack.payload_root must be a string")
        pack_path = normalize_relative_path(pack_path)
        payload_root_name = normalize_relative_path(payload_root_name)
        payloads.require(pack_path, role="evaluator-manifest", process_id=None)
        pack_payload = _read_json(
            confined_path(artifact_root, pack_path), "eager kernel pack"
        )
        raw_backend = pack_payload.get("backend")
        if raw_backend not in _SUPPORTED_PREPARED_BACKENDS:
            raise CompatibilityError(
                f"unsupported prepared eager backend {raw_backend!r}"
            )
        mutable_pack = dict(pack_payload)
        kernel_abi = mutable_pack.pop("eager_kernel_abi", None)
        if kernel_abi != EAGER_KERNEL_ABI:
            raise CompatibilityError(f"unsupported eager kernel ABI {kernel_abi!r}")
        try:
            pack = PreparedKernelPack.from_dict(mutable_pack)
        except PreparedModelBundleError as exc:
            raise ArtifactError(f"eager kernel pack is malformed: {exc}") from exc
        if pack.resolver_manifest.get("abi") != _PREPARED_CATALOG_ABI:
            raise CompatibilityError("unsupported prepared eager kernel catalog ABI")
        payload_root = artifact_root / payload_root_name
        if not payload_root.is_dir() or payload_root.is_symlink():
            raise ArtifactError("eager kernel payload root is missing or invalid")
        for kernel in pack.kernels:
            exact_path = _joined_payload_path(
                payload_root_name, kernel.exact_evaluator_state_path
            )
            payloads.require(exact_path, role="evaluator-state", process_id=None)

        runtime_schema = _mapping(execution.get("runtime_schema"), "runtime_schema")
        layout = _mapping(runtime_schema.get("parameter_layout"), "parameter_layout")
        value_count = _integer(
            layout.get("value_component_count"), "value_component_count"
        )
        momentum_count = _integer(
            layout.get("momentum_parameter_count"), "momentum_parameter_count"
        )
        parameter_count = _integer(
            layout.get("model_parameter_count"), "model_parameter_count"
        )
        current_storage = _mapping(
            runtime_schema.get("current_storage"), "current_storage"
        )
        value_storage = _mapping(runtime_schema.get("value_storage"), "value_storage")
        current_count = _integer(
            current_storage.get("component_count"), "current component_count"
        )
        values = _component_slots(
            value_storage.get("value_slots"),
            id_field="value_slot_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field="dimension",
            component_count=value_count,
            context="value slots",
        )
        momenta = _component_slots(
            runtime_schema.get("momentum_slots"),
            id_field="momentum_slot_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field=None,
            component_count=momentum_count,
            context="momentum slots",
        )
        currents = _component_slots(
            current_storage.get("current_slots"),
            id_field="current_id",
            start_field="component_start",
            stop_field="component_stop",
            dimension_field="dimension",
            component_count=current_count,
            context="current slots",
        )
        kernel_map = {
            kernel.kernel_id: _LazyExactKernel(kernel, payload_root, kernel_loader)
            for kernel in pack.kernels
            if kernel.contract_kind != "model-parameter"
        }
        plan_record = _mapping(execution.get("plan"), "plan")
        process_prefix = f"processes/{process_id}"
        couplings = _load_table(
            process_root,
            process_prefix,
            _mapping(plan_record.get("couplings"), "plan.couplings"),
            EagerCouplingRow,
            payloads,
            process_id,
            "couplings",
        )
        raw_runtime_stages = _sequence(runtime_schema.get("stages"), "runtime stages")
        raw_stages = _sequence(plan_record.get("stages"), "plan.stages")
        if len(raw_stages) != len(raw_runtime_stages):
            raise ArtifactError("eager plan stage count does not match runtime schema")
        stages: list[_ExactStage] = []
        previous_stage = -1
        for index, (raw_stage, raw_runtime_stage) in enumerate(
            zip(raw_stages, raw_runtime_stages, strict=True)
        ):
            stage = _mapping(raw_stage, f"plan.stages[{index}]")
            runtime_stage = _mapping(
                raw_runtime_stage, f"runtime_schema.stages[{index}]"
            )
            stage_index = _integer(stage.get("stage_index"), "stage_index")
            if stage_index <= previous_stage:
                raise ArtifactError("eager stage indices must be strictly increasing")
            previous_stage = stage_index
            if stage_index != _integer(runtime_stage.get("stage_index"), "stage_index"):
                raise ArtifactError("eager plan stage does not match runtime schema")
            stages.append(
                _ExactStage(
                    stage_index,
                    _load_table(
                        process_root,
                        process_prefix,
                        _mapping(stage.get("invocations"), "stage.invocations"),
                        EagerInvocationRow,
                        payloads,
                        process_id,
                        f"stage {stage_index} invocations",
                    ),
                    _load_table(
                        process_root,
                        process_prefix,
                        _mapping(stage.get("attachments"), "stage.attachments"),
                        EagerAttachmentRow,
                        payloads,
                        process_id,
                        f"stage {stage_index} attachments",
                    ),
                    _load_table(
                        process_root,
                        process_prefix,
                        _mapping(stage.get("finalizations"), "stage.finalizations"),
                        EagerFinalizationRow,
                        payloads,
                        process_id,
                        f"stage {stage_index} finalizations",
                    ),
                )
            )
        closures = _load_table(
            process_root,
            process_prefix,
            _mapping(plan_record.get("closures"), "plan.closures"),
            EagerClosureRow,
            payloads,
            process_id,
            "closures",
        )
        amplitude_stage = _mapping(
            runtime_schema.get("amplitude_stage"), "amplitude_stage"
        )
        amplitude_count = _integer(
            amplitude_stage.get("output_count"), "amplitude output count", minimum=1
        )
        result = cls(
            runtime_schema=runtime_schema,
            kernels=kernel_map,
            value_slots=values,
            momentum_slots=momenta,
            current_slots=currents,
            value_component_count=value_count,
            momentum_component_count=momentum_count,
            current_component_count=current_count,
            parameter_count=parameter_count,
            amplitude_count=amplitude_count,
            couplings=couplings,
            stages=tuple(stages),
            closures=closures,
        )
        result._validate()
        return result

    def _validate(self) -> None:
        for index, coupling in enumerate(self.couplings):
            for component, parameter_id in (
                ("real", coupling.real_parameter_id),
                ("imaginary", coupling.imag_parameter_id),
            ):
                if parameter_id != MISSING_U32 and parameter_id >= self.parameter_count:
                    raise ArtifactError(
                        f"eager coupling {index} {component} parameter is out of range"
                    )
        finalized: set[int] = set()
        stored_values: set[int] = set()
        for stage in self.stages:
            cursor = 0
            attached: set[int] = set()
            for index, invocation in enumerate(stage.invocations):
                kernel = self._require_kernel(
                    invocation.kernel_id, "vertex", f"invocation {index}"
                )
                left = self._slot(
                    self.value_slots,
                    invocation.left_value_slot_id,
                    "invocation left value",
                )
                right = self._slot(
                    self.value_slots,
                    invocation.right_value_slot_id,
                    "invocation right value",
                )
                left_momentum = self._slot(
                    self.momentum_slots,
                    invocation.left_momentum_slot_id,
                    "invocation left momentum",
                )
                right_momentum = self._slot(
                    self.momentum_slots,
                    invocation.right_momentum_slot_id,
                    "invocation right momentum",
                )
                if invocation.coupling_slot_id >= len(self.couplings):
                    raise ArtifactError("eager invocation coupling is out of range")
                if (
                    invocation.attachment_count == 0
                    or invocation.attachment_start != cursor
                ):
                    raise ArtifactError(
                        f"eager stage {stage.stage_index} has invalid attachment ranges"
                    )
                stop = cursor + invocation.attachment_count
                if stop > len(stage.attachments):
                    raise ArtifactError(
                        "eager invocation attachment range is out of bounds"
                    )
                self._validate_kernel_inputs(
                    kernel.record,
                    first_current=left.width,
                    second_current=right.width,
                    first_momentum=left_momentum.width,
                    second_momentum=right_momentum.width,
                    has_coupling=True,
                    context=f"invocation {index}",
                )
                for attachment in stage.attachments[cursor:stop]:
                    current = self._slot(
                        self.current_slots,
                        attachment.result_current_id,
                        "attachment result current",
                    )
                    if current.width != kernel.record.output_arity:
                        raise ArtifactError(
                            "eager invocation output width does not match "
                            "attached current"
                        )
                    attached.add(attachment.result_current_id)
                cursor = stop
            if cursor != len(stage.attachments):
                raise ArtifactError("eager attachment table is not fully referenced")
            stage_finalized: set[int] = set()
            for index, finalization in enumerate(stage.finalizations):
                if (
                    finalization.current_id in stage_finalized
                    or finalization.current_id in finalized
                ):
                    raise ArtifactError(
                        f"eager current {finalization.current_id} is finalized "
                        "more than once"
                    )
                stage_finalized.add(finalization.current_id)
                finalized.add(finalization.current_id)
                current = self._slot(
                    self.current_slots,
                    finalization.current_id,
                    "finalization current",
                )
                outputs = []
                for name, slot_id in (
                    ("unpropagated", finalization.unpropagated_value_slot_id),
                    ("propagated", finalization.propagated_value_slot_id),
                ):
                    if slot_id == MISSING_U32:
                        continue
                    if slot_id in stored_values:
                        raise ArtifactError(
                            f"eager value slot {slot_id} is finalized more than once"
                        )
                    stored_values.add(slot_id)
                    output = self._slot(self.value_slots, slot_id, f"{name} value")
                    if output.width != current.width:
                        raise ArtifactError("finalization output/current widths differ")
                    outputs.append((name, output))
                if not outputs:
                    raise ArtifactError("eager finalization stores no value")
                momentum = self._slot(
                    self.momentum_slots,
                    finalization.momentum_slot_id,
                    "finalization momentum",
                )
                if finalization.kernel_id == MISSING_U32:
                    if finalization.propagated_value_slot_id != MISSING_U32:
                        raise ArtifactError(
                            "eager propagated current has no finalization kernel"
                        )
                else:
                    kernel = self._require_kernel(
                        finalization.kernel_id,
                        "propagator",
                        f"finalization {index}",
                    )
                    if finalization.propagated_value_slot_id == MISSING_U32:
                        raise ArtifactError(
                            "eager finalization kernel has no propagated output"
                        )
                    if kernel.record.output_arity != current.width:
                        raise ArtifactError(
                            "eager finalization kernel/current widths differ"
                        )
                    self._validate_kernel_inputs(
                        kernel.record,
                        first_current=current.width,
                        second_current=0,
                        first_momentum=momentum.width,
                        second_momentum=0,
                        has_coupling=False,
                        context=f"finalization {index}",
                    )
            if not attached.issubset(stage_finalized):
                missing = min(attached - stage_finalized)
                raise ArtifactError(
                    f"eager stage {stage.stage_index} does not finalize "
                    f"current {missing}"
                )

        amplitude_stage = _mapping(
            self.runtime_schema.get("amplitude_stage"), "amplitude_stage"
        )
        roots = _sequence(amplitude_stage.get("roots"), "amplitude roots")
        if len(roots) != len(self.closures):
            raise ArtifactError("eager closure rows do not match amplitude roots")
        for index, (closure, raw_root) in enumerate(
            zip(self.closures, roots, strict=True)
        ):
            root = _mapping(raw_root, f"amplitude root {index}")
            left = self._slot(
                self.value_slots, closure.left_value_slot_id, "closure left value"
            )
            right = self._slot(
                self.value_slots, closure.right_value_slot_id, "closure right value"
            )
            if closure.amplitude_index >= self.amplitude_count:
                raise ArtifactError("eager closure amplitude index is out of range")
            if closure.kernel_id == MISSING_U32:
                if closure.coupling_slot_id != MISSING_U32:
                    raise ArtifactError("direct eager closure references a coupling")
                if root.get("kind") != "direct-contraction":
                    raise ArtifactError(
                        "direct eager closure lacks contraction metadata"
                    )
                coefficients = _direct_coefficients(root, index)
                if left.width != right.width or left.width != len(coefficients):
                    raise ArtifactError("direct eager closure component widths differ")
            else:
                if root.get("kind") == "direct-contraction":
                    raise ArtifactError("kernel eager closure has direct metadata")
                if closure.coupling_slot_id >= len(self.couplings):
                    raise ArtifactError("eager closure coupling is out of range")
                kernel = self._require_kernel(
                    closure.kernel_id, "closure", f"closure {index}"
                )
                if kernel.record.output_arity != 1:
                    raise ArtifactError(
                        "eager closure kernels must return one component"
                    )
                self._validate_kernel_inputs(
                    kernel.record,
                    first_current=left.width,
                    second_current=right.width,
                    first_momentum=0,
                    second_momentum=0,
                    has_coupling=True,
                    context=f"closure {index}",
                )

    def _require_kernel(
        self, kernel_id: int, kind: str, context: str
    ) -> _LazyExactKernel:
        kernel = self.kernels.get(kernel_id)
        if kernel is None:
            raise ArtifactError(
                f"{context} references missing eager kernel {kernel_id}"
            )
        if kernel.record.contract_kind != kind:
            raise ArtifactError(
                f"{context} requires a {kind} kernel, but {kernel_id} is "
                f"{kernel.record.contract_kind}"
            )
        return kernel

    @staticmethod
    def _slot(
        slots: Sequence[_ComponentSlot],
        slot_id: int,
        context: str,
    ) -> _ComponentSlot:
        if slot_id < 0 or slot_id >= len(slots):
            raise ArtifactError(f"{context} references unknown slot {slot_id}")
        return slots[slot_id]

    def _validate_kernel_inputs(
        self,
        record: PreparedKernelRecord,
        *,
        first_current: int,
        second_current: int,
        first_momentum: int,
        second_momentum: int,
        has_coupling: bool,
        context: str,
    ) -> None:
        seen: set[tuple[str, int]] = set()
        for input_index, contract in enumerate(record.input_contracts):
            role = str(contract["role"])
            component = _integer(
                contract["component"],
                f"kernel {record.kernel_id} input component",
            )
            if role == "left-current":
                allowed, bound, descriptor = True, first_current, (role, component)
            elif role == "right-current":
                allowed, bound, descriptor = (
                    record.contract_kind != "propagator",
                    second_current,
                    (role, component),
                )
            elif role == "current":
                allowed, bound, descriptor = (
                    record.contract_kind == "propagator",
                    first_current,
                    ("left-current", component),
                )
            elif role == "left-momentum":
                allowed, bound, descriptor = (
                    record.contract_kind != "closure",
                    first_momentum,
                    (role, component),
                )
            elif role == "right-momentum":
                allowed, bound, descriptor = (
                    record.contract_kind == "vertex",
                    second_momentum,
                    (role, component),
                )
            elif role == "momentum":
                allowed, bound, descriptor = (
                    record.contract_kind == "propagator",
                    first_momentum,
                    ("left-momentum", component),
                )
            elif role in {"coupling-real", "coupling-imag"}:
                allowed, bound, descriptor = has_coupling, 1, (role, 0)
            elif role == "model-parameter":
                parameter_id = contract.get("model_parameter_index")
                if isinstance(parameter_id, bool) or not isinstance(parameter_id, int):
                    raise ArtifactError(
                        f"kernel {record.kernel_id} model-parameter input lacks "
                        "an index"
                    )
                allowed, bound, component, descriptor = (
                    parameter_id < self.parameter_count,
                    self.parameter_count,
                    parameter_id,
                    (role, parameter_id),
                )
            else:
                raise ArtifactError(
                    f"kernel {record.kernel_id} has unsupported input role {role!r}"
                )
            if not allowed or component >= bound:
                raise ArtifactError(
                    f"{context} kernel {record.kernel_id} input {input_index} "
                    "is out of range"
                )
            if descriptor in seen:
                raise ArtifactError(
                    f"kernel {record.kernel_id} repeats eager input {descriptor!r}"
                )
            seen.add(descriptor)
