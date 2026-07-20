# SPDX-License-Identifier: 0BSD
"""Validated in-memory representation of an eager exact-execution plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.artifacts.manifest import ArtifactManifest
from pyamplicol.artifacts.security import confined_path, normalize_relative_path
from pyamplicol.generation.eager_tables import (
    EAGER_KERNEL_ABI,
    EAGER_OUTPUT_FACTOR_NONE,
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


@dataclass(frozen=True, slots=True)
class _RuntimeParameterSlots:
    real: int
    imaginary: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class _PreparedParameterProjectionEntry:
    name: str
    prepared_index: int
    runtime_real_index: int
    runtime_imaginary_index: int | None


@dataclass(frozen=True, slots=True)
class _PreparedParameterProjection:
    parameter_count: int
    runtime_parameter_count: int
    entries: tuple[_PreparedParameterProjectionEntry, ...]

    def project(
        self, runtime_parameters: Sequence[Decimal]
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        if len(runtime_parameters) != self.runtime_parameter_count:
            raise ArtifactError(
                f"eager runtime has {len(runtime_parameters)} model parameters, "
                f"expected {self.runtime_parameter_count}"
            )
        zero = Decimal(0)
        result = [(zero, zero) for _ in range(self.parameter_count)]
        for entry in self.entries:
            imaginary = (
                runtime_parameters[entry.runtime_imaginary_index]
                if entry.runtime_imaginary_index is not None
                else zero
            )
            result[entry.prepared_index] = (
                runtime_parameters[entry.runtime_real_index],
                imaginary,
            )
        return tuple(result)

    def entry(self, name: str) -> _PreparedParameterProjectionEntry | None:
        return next((entry for entry in self.entries if entry.name == name), None)


@dataclass(frozen=True, slots=True)
class _DerivedParameterTarget:
    name: str
    prepared_index: int | None
    runtime_real_index: int
    runtime_imaginary_index: int | None


@dataclass(frozen=True, slots=True)
class _ExactModelParameterState:
    runtime: tuple[Decimal, ...]
    prepared: tuple[tuple[Decimal, Decimal], ...]


@dataclass(frozen=True, slots=True)
class _ExactParameterDerivation:
    kernel: _LazyExactKernel
    input_parameter_indices: tuple[int, ...]
    targets: tuple[_DerivedParameterTarget | None, ...]

    def evaluate(
        self,
        runtime_parameters: Sequence[Decimal],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        precision: int,
    ) -> _ExactModelParameterState:
        inputs = tuple(
            prepared_parameters[index] for index in self.input_parameter_indices
        )
        outputs = self.kernel.evaluate(inputs, precision)
        if len(outputs) != len(self.targets):
            raise ArtifactError(
                "exact model-parameter output count does not match its layout"
            )
        runtime = list(runtime_parameters)
        prepared = list(prepared_parameters)
        zero = Decimal(0)
        for output, target in zip(outputs, self.targets, strict=True):
            if target is None:
                continue
            if target.prepared_index is not None:
                prepared[target.prepared_index] = output
            runtime[target.runtime_real_index] = output[0]
            if target.runtime_imaginary_index is not None:
                runtime[target.runtime_imaginary_index] = output[1]
            elif output[1] != zero:
                raise EvaluationError(
                    f"exact derived scalar parameter {target.name!r} has a "
                    "nonzero imaginary component"
                )
        return _ExactModelParameterState(tuple(runtime), tuple(prepared))


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
    parameter_projection: _PreparedParameterProjection
    parameter_derivation: _ExactParameterDerivation | None
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
        parameter_projection = _prepared_parameter_projection(
            pack.kernels,
            runtime_schema,
            parameter_count,
        )
        parameter_derivation = _exact_parameter_derivation(
            pack.kernels,
            runtime_schema,
            parameter_projection,
            payload_root,
            kernel_loader,
        )
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
            parameter_projection=parameter_projection,
            parameter_derivation=parameter_derivation,
            amplitude_count=amplitude_count,
            couplings=couplings,
            stages=tuple(stages),
            closures=closures,
        )
        result._validate()
        return result

    def project_model_parameters(
        self, runtime_parameters: Sequence[Decimal]
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        return self.parameter_projection.project(runtime_parameters)

    def resolve_model_parameters(
        self,
        runtime_parameters: Sequence[Decimal],
        precision: int,
    ) -> _ExactModelParameterState:
        runtime = tuple(runtime_parameters)
        prepared = self.parameter_projection.project(runtime)
        if self.parameter_derivation is None:
            return _ExactModelParameterState(runtime, prepared)
        return self.parameter_derivation.evaluate(runtime, prepared, precision)

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
                if closure.output_factor_source != EAGER_OUTPUT_FACTOR_NONE:
                    raise ArtifactError(
                        "direct eager closure has a dynamic output factor"
                    )
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
                    parameter_id < self.parameter_projection.parameter_count,
                    self.parameter_projection.parameter_count,
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


def _prepared_parameter_projection(
    kernels: Sequence[PreparedKernelRecord],
    runtime_schema: Mapping[str, object],
    runtime_parameter_count: int,
) -> _PreparedParameterProjection:
    runtime_slots = _runtime_parameter_slots(
        runtime_schema,
        runtime_parameter_count,
    )
    by_name: dict[str, int] = {}
    by_index: dict[int, str] = {}
    for kernel in kernels:
        for contract in kernel.input_contracts:
            if contract.get("role") != "model-parameter":
                continue
            name = contract.get("model_parameter_name")
            if not isinstance(name, str) or not name:
                raise ArtifactError(
                    "prepared model-parameter input lacks its logical name"
                )
            index = contract.get("model_parameter_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ArtifactError(
                    "prepared model-parameter input lacks its stable index"
                )
            previous_index = by_name.setdefault(name, index)
            if previous_index != index:
                raise ArtifactError(
                    f"prepared parameter {name!r} has conflicting stable indices"
                )
            previous_name = by_index.setdefault(index, name)
            if previous_name != name:
                raise ArtifactError(
                    f"prepared parameter index {index} names multiple parameters"
                )

    entries = []
    for name, prepared_index in sorted(by_name.items(), key=lambda item: item[1]):
        slots = runtime_slots.get(name)
        if slots is None:
            raise ArtifactError(
                f"prepared parameter {name!r} is absent from the process runtime schema"
            )
        entries.append(
            _PreparedParameterProjectionEntry(
                name=name,
                prepared_index=prepared_index,
                runtime_real_index=slots.real,
                runtime_imaginary_index=slots.imaginary,
            )
        )
    parameter_count = max(by_index, default=-1) + 1
    return _PreparedParameterProjection(
        parameter_count=parameter_count,
        runtime_parameter_count=runtime_parameter_count,
        entries=tuple(entries),
    )


def _runtime_parameter_slots(
    runtime_schema: Mapping[str, object],
    parameter_count: int,
) -> Mapping[str, _RuntimeParameterSlots]:
    direct: dict[str, _RuntimeParameterSlots] = {}
    complex_components: dict[str, list[int | None]] = {}
    complex_kinds: dict[str, str] = {}
    seen_indices: set[int] = set()
    records = _sequence(runtime_schema.get("model_parameters"), "model parameters")
    for record_index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"model parameters[{record_index}]")
        parameter_index = _integer(
            record.get("parameter_index"),
            f"model parameters[{record_index}].parameter_index",
        )
        if parameter_index >= parameter_count or parameter_index in seen_indices:
            raise ArtifactError("runtime model-parameter indices are invalid")
        seen_indices.add(parameter_index)
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise ArtifactError("runtime model parameter has no name")
        runtime_name = record.get("runtime_name")
        kind = record.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ArtifactError("runtime model parameter has no kind")
        if runtime_name is None:
            if name in direct or name in complex_components:
                raise ArtifactError(f"runtime model parameter {name!r} is duplicated")
            direct[name] = _RuntimeParameterSlots(parameter_index, None, kind)
            continue
        if not isinstance(runtime_name, str) or not runtime_name:
            raise ArtifactError("complex runtime model parameter has no logical name")
        component = record.get("complex_component")
        if component not in {"real", "imag"}:
            raise ArtifactError(
                f"runtime model parameter {runtime_name!r} has invalid component "
                f"{component!r}"
            )
        slots = complex_components.setdefault(runtime_name, [None, None])
        previous_kind = complex_kinds.setdefault(runtime_name, kind)
        if previous_kind != kind:
            raise ArtifactError(
                f"runtime model parameter {runtime_name!r} has mixed component kinds"
            )
        component_index = 0 if component == "real" else 1
        if slots[component_index] is not None:
            raise ArtifactError(
                f"runtime model parameter {runtime_name!r} repeats a complex component"
            )
        slots[component_index] = parameter_index

    if seen_indices != set(range(parameter_count)):
        raise ArtifactError(
            "runtime model-parameter indices must be contiguous from zero"
        )
    for name, (real, imaginary) in complex_components.items():
        if real is None:
            raise ArtifactError(
                f"complex runtime model parameter {name!r} lacks a real component"
            )
        if name in direct:
            raise ArtifactError(
                f"runtime model parameter {name!r} has scalar and complex records"
            )
        if imaginary is None:
            raise ArtifactError(
                f"complex runtime model parameter {name!r} lacks an imaginary component"
            )
        direct[name] = _RuntimeParameterSlots(real, imaginary, complex_kinds[name])
    return direct


def _exact_parameter_derivation(
    kernels: Sequence[PreparedKernelRecord],
    runtime_schema: Mapping[str, object],
    projection: _PreparedParameterProjection,
    payload_root: Path,
    kernel_loader: _KernelLoader,
) -> _ExactParameterDerivation | None:
    records = tuple(
        kernel for kernel in kernels if kernel.contract_kind == "model-parameter"
    )
    if not records:
        return None
    if len(records) != 1:
        raise ArtifactError(
            "eager kernel pack declares multiple model-parameter kernels"
        )
    record = records[0]
    runtime_slots = _runtime_parameter_slots(
        runtime_schema,
        projection.runtime_parameter_count,
    )
    output_names: set[str] = set()
    parsed_outputs: list[str] = []
    prefix = "model-parameter:"
    if not record.output_layout:
        raise ArtifactError("model-parameter kernel declares no outputs")
    for index, layout in enumerate(record.output_layout):
        if not layout.startswith(prefix) or len(layout) == len(prefix):
            raise ArtifactError(
                f"model-parameter kernel output {index} has invalid layout {layout!r}"
            )
        output_name = layout[len(prefix) :]
        if output_name != output_name.strip():
            raise ArtifactError(
                f"model-parameter kernel output {index} has invalid layout {layout!r}"
            )
        if output_name in output_names:
            raise ArtifactError(
                f"model-parameter kernel repeats output parameter {output_name!r}"
            )
        output_names.add(output_name)
        parsed_outputs.append(output_name)

    input_names: set[str] = set()
    input_parameter_indices: list[int] = []
    for index, contract in enumerate(record.input_contracts):
        if contract.get("role") != "model-parameter":
            raise ArtifactError(
                f"model-parameter kernel input {index} is not a model parameter"
            )
        input_name = contract.get("model_parameter_name")
        stable_index = contract.get("model_parameter_index")
        component = contract.get("component")
        if not isinstance(input_name, str) or not input_name:
            raise ArtifactError(
                f"model-parameter kernel input {index} lacks a logical name"
            )
        if (
            isinstance(stable_index, bool)
            or not isinstance(stable_index, int)
            or stable_index < 0
        ):
            raise ArtifactError(
                f"model-parameter kernel input {index} lacks a stable index"
            )
        if component != 0:
            raise ArtifactError(
                f"model-parameter kernel input {index} has invalid component "
                f"{component!r}"
            )
        entry = projection.entry(input_name)
        if entry is None or entry.prepared_index != stable_index:
            raise ArtifactError(
                f"model-parameter kernel input {input_name!r} has no matching "
                "projection"
            )
        input_slots = runtime_slots[input_name]
        if input_slots.kind not in {
            "external_parameter",
            "external_parameter_component",
        }:
            raise ArtifactError(
                f"model-parameter kernel input {input_name!r} is not an "
                "external/base "
                "parameter"
            )
        if input_name in input_names:
            raise ArtifactError(
                f"model-parameter kernel repeats input parameter {input_name!r}"
            )
        input_names.add(input_name)
        input_parameter_indices.append(stable_index)
    overlap = input_names & output_names
    if overlap:
        raise ArtifactError(
            "model-parameter kernel inputs overlap its derived outputs: "
            + ", ".join(sorted(overlap))
        )

    derived_runtime_names = {
        runtime_name
        for runtime_name, runtime_slots_record in runtime_slots.items()
        if runtime_slots_record.kind == "derived_parameter_component"
    }
    missing_outputs = derived_runtime_names - output_names
    if missing_outputs:
        raise ArtifactError(
            "model-parameter kernel does not output runtime derived parameters: "
            + ", ".join(repr(name) for name in sorted(missing_outputs))
        )

    targets: list[_DerivedParameterTarget | None] = []
    for output_name in parsed_outputs:
        output_slots = runtime_slots.get(output_name)
        entry = projection.entry(output_name)
        if output_slots is None:
            if entry is not None:
                raise ArtifactError(
                    f"derived prepared parameter {output_name!r} has no runtime slots"
                )
            targets.append(None)
            continue
        if output_slots.kind != "derived_parameter_component":
            raise ArtifactError(
                f"model-parameter kernel output {output_name!r} does not target "
                "derived runtime slots"
            )
        if output_slots.imaginary is None:
            raise ArtifactError(
                f"derived runtime parameter {output_name!r} lacks an imaginary "
                "component"
            )
        targets.append(
            _DerivedParameterTarget(
                name=output_name,
                prepared_index=(None if entry is None else entry.prepared_index),
                runtime_real_index=output_slots.real,
                runtime_imaginary_index=output_slots.imaginary,
            )
        )
    return _ExactParameterDerivation(
        _LazyExactKernel(record, payload_root, kernel_loader),
        tuple(input_parameter_indices),
        tuple(targets),
    )
