# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import gc
import hashlib
import importlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

from pyamplicol.api.errors import GenerationError, ModelError, PyAmpliColError
from pyamplicol.api.models import CompiledModel, _compiled_model_payload
from pyamplicol.api.requests import (
    ModelSource,
    ProcessAlias,
    ProcessRequest,
    ProcessSet,
)
from pyamplicol.api.results import GenerationPlan, GenerationResult
from pyamplicol.config import (
    ClampRequest,
    ConfigResolution,
    GenerationConfig,
    RunConfig,
    config_to_dict,
    resolve_config,
)
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
)

from ..color.plan import build_color_plan, build_lc_topology_replay_plan
from ..models.base import Model
from ..models.loading import CompiledModel as _CompiledModelPayload
from ..models.prepared import PreparedKernelPack
from ..models.prepared_target import PreparedTargetError, validate_prepared_target
from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from .artifact_writer import (
    EAGER_PLAN_V3_ABI,
    EAGER_PLAN_V3_RUNTIME_CAPABILITY,
    EAGER_RUNTIME_CONTAINER_KIND,
    EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION,
    EAGER_RUNTIME_LAYOUT_ABI,
    EAGER_RUNTIME_STORAGE_ABI,
    CompiledColorSelectorExecutionArtifact,
    CompiledExecutionArtifact,
    CompiledHelicitySelectorExecutionArtifact,
    CompiledProcessArtifact,
    EagerPlanV3ProcessArtifact,
    EagerProcessArtifact,
    _GenerationConfigProvenance,
    write_schema_v3_artifact,
)
from .contracts import RuntimeExpressionSchema, StageCompilationInput
from .dag_algorithms import (
    contributing_color_sector_ids,
    filter_dag_to_color_sectors,
    infer_minimal_coupling_order_limits,
    prune_dag_to_amplitude_roots,
    prune_global_helicity_flip_equivalent_roots,
)
from .dag_compiler import _restrict_color_plan, compile_generic_dag
from .dag_types import GenericDAG
from .eager_columnar import (
    EAGER_LOWERING_INPUT_ABI,
    EagerLoweringInputV1,
    build_eager_lowering_input_v1,
)
from .eager_lowering import (
    PreparedCatalogEagerKernelIndex,
    PreparedCatalogEagerKernelResolver,
    lower_fused_eager_execution,
)
from .eager_tables import MISSING_U32
from .helicity_materialization import materialize_helicity_recurrence
from .helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
    build_helicity_recurrence_plan,
)
from .progress import GenerationPhaseReporter, PhaseHandle
from .runtime_schema import build_runtime_expression_schema
from .stage_compiler import (
    build_and_write_generic_stage_evaluator_artifacts,
    write_model_parameter_evaluator_artifact,
)
from .validation import ValidationPointRecord, build_validation_point

if TYPE_CHECKING:
    from ..licensing import SymbolicaLicenseState
    from .artifact_writer import ApiBundleHook


_ProcessInput = TypeVar("_ProcessInput")
_ProcessOutput = TypeVar("_ProcessOutput")
_MISSING_PROCESS_RESULT = object()
_LOGGER = logging.getLogger("pyamplicol.generation")
_MAX_FUSED_LC_HELICITY_SELECTOR_SECTORS = 128
_EAGER_PLAN_VERSION_ENV = "PYAMPLICOL_EAGER_PLAN_VERSION"
_EAGER_PLAN_V2 = "v2"
_EAGER_PLAN_V3 = "v3"
_EAGER_LOWERING_RESULT_KIND = "pyamplicol-eager-runtime-lowering-result"
_EAGER_LOWERING_RESULT_SCHEMA_VERSION = 1
# Symbolica releases the GIL while retaining process-wide mutable symbol state.
# Keep each complete lowering/compilation transaction atomic across generators.
_SYMBOLICA_MATERIALIZATION_LOCK = Lock()


class _RustEagerLoweringBinding(Protocol):
    def __call__(
        self,
        lowering_input: EagerLoweringInputV1,
        destination: str,
        /,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _RustEagerLoweringOutput:
    physics: Mapping[str, object]
    inspection_summary: Mapping[str, object]
    payload_path: Path
    payload_size_bytes: int
    payload_sha256: str
    member_count: int
    unpacked_size_bytes: int
    index_sha256: str


def _selected_eager_plan_version() -> Literal["v2", "v3"]:
    value = os.environ.get(_EAGER_PLAN_VERSION_ENV, _EAGER_PLAN_V2).strip().lower()
    if value not in {_EAGER_PLAN_V2, _EAGER_PLAN_V3}:
        raise GenerationError(
            f"{_EAGER_PLAN_VERSION_ENV} must be {_EAGER_PLAN_V2!r} or "
            f"{_EAGER_PLAN_V3!r}, got {value!r}"
        )
    return cast(Literal["v2", "v3"], value)


def _invoke_rust_eager_lowering_v1(
    lowering_input: EagerLoweringInputV1,
    destination: Path,
) -> _RustEagerLoweringOutput:
    try:
        module = importlib.import_module("pyamplicol._rusticol")
    except ImportError as exc:
        raise GenerationError(
            "eager plan-v3 was requested, but pyamplicol._rusticol is unavailable; "
            f"set {_EAGER_PLAN_VERSION_ENV}={_EAGER_PLAN_V2} to use plan-v2"
        ) from exc
    candidate = getattr(module, "_lower_eager_runtime_v1", None)
    if not callable(candidate):
        raise GenerationError(
            "eager plan-v3 was requested, but pyamplicol._rusticol does not provide "
            "the private _lower_eager_runtime_v1 binding; "
            f"set {_EAGER_PLAN_VERSION_ENV}={_EAGER_PLAN_V2} to use plan-v2"
        )
    binding = cast(_RustEagerLoweringBinding, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise GenerationError(
            f"Rust eager lowering destination already exists: {destination}"
        )
    try:
        raw_result = binding(lowering_input, os.fspath(destination))
        result = _validate_rust_eager_lowering_result(raw_result, lowering_input)
        if destination.is_symlink() or not destination.is_file():
            raise GenerationError(
                "Rust eager lowering did not produce a regular eager-runtime.pacbin"
            )
        payload_size = destination.stat().st_size
        if payload_size <= 0:
            raise GenerationError(
                "Rust eager lowering produced an empty runtime payload"
            )
        payload_sha256 = _file_sha256(destination)
    except Exception as exc:
        _discard_rust_eager_output(destination)
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError(f"Rust eager plan-v3 lowering failed: {exc}") from exc
    return _RustEagerLoweringOutput(
        physics=result["physics"],
        inspection_summary=result["inspection_summary"],
        payload_path=destination,
        payload_size_bytes=payload_size,
        payload_sha256=payload_sha256,
        member_count=cast(int, result["member_count"]),
        unpacked_size_bytes=cast(int, result["unpacked_size_bytes"]),
        index_sha256=cast(str, result["index_sha256"]),
    )


def _validate_rust_eager_lowering_result(
    value: object,
    lowering_input: EagerLoweringInputV1,
) -> dict[str, object]:
    result = _strict_mapping(
        value,
        "Rust eager lowering result",
        {
            "kind",
            "schema_version",
            "lowering_input_abi",
            "lowering_input_sha256",
            "eager_plan_abi",
            "runtime_layout_abi",
            "required_runtime_capabilities",
            "runtime_container",
            "physics",
            "inspection_summary",
        },
    )
    expected = {
        "kind": _EAGER_LOWERING_RESULT_KIND,
        "schema_version": _EAGER_LOWERING_RESULT_SCHEMA_VERSION,
        "lowering_input_abi": EAGER_LOWERING_INPUT_ABI,
        "lowering_input_sha256": lowering_input.digest,
        "eager_plan_abi": EAGER_PLAN_V3_ABI,
        "runtime_layout_abi": EAGER_RUNTIME_LAYOUT_ABI,
    }
    mismatched = [
        name
        for name, expected_value in expected.items()
        if result[name] != expected_value
    ]
    if mismatched:
        raise GenerationError(
            "Rust eager lowering result has incompatible contract fields: "
            + ", ".join(mismatched)
        )
    capabilities = result["required_runtime_capabilities"]
    if (
        isinstance(capabilities, str | bytes)
        or not isinstance(capabilities, Sequence)
        or tuple(capabilities) != (EAGER_PLAN_V3_RUNTIME_CAPABILITY,)
    ):
        raise GenerationError(
            "Rust eager lowering result has an incompatible runtime capability"
        )
    container = _strict_mapping(
        result["runtime_container"],
        "Rust eager runtime container metadata",
        {
            "kind",
            "schema_version",
            "storage_abi",
            "member_count",
            "unpacked_size_bytes",
            "index_sha256",
        },
    )
    if container["kind"] != EAGER_RUNTIME_CONTAINER_KIND:
        raise GenerationError("Rust eager runtime container kind is incompatible")
    if container["schema_version"] != EAGER_RUNTIME_CONTAINER_SCHEMA_VERSION:
        raise GenerationError("Rust eager runtime container schema is incompatible")
    if container["storage_abi"] != EAGER_RUNTIME_STORAGE_ABI:
        raise GenerationError(
            "Rust eager runtime container storage ABI is incompatible"
        )
    member_count = _result_integer(
        container["member_count"],
        "Rust eager runtime member count",
        minimum=1,
    )
    unpacked_size = _result_integer(
        container["unpacked_size_bytes"],
        "Rust eager runtime unpacked size",
        minimum=0,
    )
    index_sha256 = _result_sha256(
        container["index_sha256"],
        "Rust eager runtime index SHA-256",
    )
    physics = _canonical_json_mapping(result["physics"], "Rust eager physics metadata")
    if (
        physics.get("schema_version") != 1
        or physics.get("kind") != "pyamplicol-resolved-physics"
        or physics.get("process_id") != lowering_input.process_key
    ):
        raise GenerationError(
            "Rust eager lowering returned incompatible physics metadata"
        )
    inspection = _canonical_json_mapping(
        result["inspection_summary"],
        "Rust eager inspection summary",
    )
    return {
        "physics": physics,
        "inspection_summary": inspection,
        "member_count": member_count,
        "unpacked_size_bytes": unpacked_size,
        "index_sha256": index_sha256,
    }


def _strict_mapping(
    value: object,
    context: str,
    fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationError(f"{context} must be an object")
    result = {str(key): item for key, item in value.items()}
    if set(result) != fields:
        raise GenerationError(f"{context} fields do not match the v1 binding protocol")
    return result


def _canonical_json_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GenerationError(f"{context} must be an object")
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"{context} is not canonical JSON: {exc}") from exc
    if not isinstance(result, dict):  # pragma: no cover - Mapping encoded above
        raise GenerationError(f"{context} must encode as an object")
    return cast(dict[str, object], result)


def _result_integer(value: object, context: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GenerationError(f"{context} must be at least {minimum}")
    return value


def _result_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GenerationError(f"{context} must be a lowercase hexadecimal digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _discard_rust_eager_output(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except OSError:
        pass


def _eager_lowering_kernel_ids(
    lowering_input: EagerLoweringInputV1,
) -> frozenset[int]:
    references: set[int] = set()
    for table_name, column_name in (
        ("currents", "propagator_kernel_id"),
        ("interactions", "kernel_id"),
        ("roots", "kernel_id"),
    ):
        for raw_value in lowering_input.table(table_name).column(column_name):
            value = int(raw_value)
            if value != MISSING_U32:
                references.add(value)
    return frozenset(references)


def _builtin_sm_model() -> Model:
    from ..models import BuiltinSMModel

    return BuiltinSMModel()


def _nested_config_value(values: Mapping[str, object], path: str) -> object:
    current: object = values
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise GenerationError(
                f"prepared model references unknown configuration path {path!r}"
            )
        current = current[component]
    return current


def _prepared_pack_effective_values(
    pack: PreparedKernelPack,
) -> dict[str, object]:
    settings = pack.optimization_settings

    def required(name: str) -> object:
        if name not in settings:
            raise GenerationError(
                f"prepared kernel pack omits optimization setting {name!r}"
            )
        return settings[name]

    values: dict[str, object] = {
        "evaluator.backend": pack.backend,
        "evaluator.optimization.horner_iterations": required("iterations"),
        "evaluator.optimization.cpe_iterations": required("cpe_iterations"),
        "evaluator.optimization.max_horner_variables": required(
            "max_horner_scheme_variables"
        ),
        "evaluator.optimization.max_common_pair_cache_entries": required(
            "max_common_pair_cache_entries"
        ),
        "evaluator.optimization.max_common_pair_distance": required(
            "max_common_pair_distance"
        ),
        "evaluator.optimization.collect_factors": required("collect_factors"),
    }
    if pack.backend == "jit":
        values["evaluator.jit.optimization_level"] = required("jit_optimization_level")
        return values
    optimization_level = required("compiled_optimization_level")
    if isinstance(optimization_level, bool) or not isinstance(optimization_level, int):
        raise GenerationError(
            "prepared native kernel pack has an invalid optimization level"
        )
    values.update(
        {
            "evaluator.cpp.optimization": f"O{optimization_level}",
            "evaluator.cpp.native_arch": required("compiled_native"),
            "evaluator.cpp.compiler": required("compiler_path"),
            "evaluator.cpp.extra_flags": required("compiler_flags"),
        }
    )
    return values


@dataclass(frozen=True, slots=True)
class _ResolvedModel:
    source: ModelSource
    model: Model | None
    compiled: _CompiledModelPayload | None = None
    use_compiled_process_catalog: bool = True
    eager_kernel_index: PreparedCatalogEagerKernelIndex | None = None


@dataclass(frozen=True, slots=True)
class _ProcessSelection:
    max_color_sectors: int | None = None
    reference_color_order: tuple[int, ...] | None = None
    selected_color_sector_ids: frozenset[int] | None = None
    selected_source_helicities: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        if self.reference_color_order is not None:
            object.__setattr__(
                self,
                "reference_color_order",
                tuple(int(label) for label in self.reference_color_order),
            )
        if self.selected_color_sector_ids is not None:
            object.__setattr__(
                self,
                "selected_color_sector_ids",
                frozenset(int(value) for value in self.selected_color_sector_ids),
            )
        if self.selected_source_helicities is not None:
            object.__setattr__(
                self,
                "selected_source_helicities",
                MappingProxyType(
                    {
                        int(label): int(helicity)
                        for label, helicity in self.selected_source_helicities.items()
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class _ExpandedProcess:
    request: ProcessRequest
    process_ir: CanonicalProcessIR
    aliases: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class _DagProcess:
    expanded: _ExpandedProcess
    dag: GenericDAG
    coverage: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _CompiledProcess:
    expanded: _ExpandedProcess
    dag: GenericDAG
    helicity_sum_dag: GenericDAG | None
    helicity_selector_union_dag: GenericDAG | None
    coverage: Mapping[str, object]
    filters: Mapping[str, object]
    validation_points: tuple[ValidationPointRecord, ...]


@dataclass(frozen=True, slots=True)
class _EvaluatorProcess:
    compiled: _CompiledProcess
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput
    helicity_sum_runtime_schema: RuntimeExpressionSchema | None = None
    helicity_sum_stage_input: StageCompilationInput | None = None
    helicity_selector_lanes: tuple[_HelicitySelectorLane, ...] = ()
    color_selector_lanes: tuple[_ColorSelectorLane, ...] = ()


@dataclass(frozen=True, slots=True)
class _ColorSelectorLane:
    materialized_sector_id: int
    dag: GenericDAG
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput


@dataclass(frozen=True, slots=True)
class _HelicitySelectorLane:
    selector_domain_ids: tuple[int, ...]
    active_current_ids: tuple[int, ...]
    active_root_ids: tuple[int, ...]
    dag: GenericDAG
    runtime_schema: RuntimeExpressionSchema
    stage_input: StageCompilationInput
    schedule_mode: Literal["parent-closure", "nested-runtime"] = "parent-closure"
    child_lanes: tuple[_HelicitySelectorLane, ...] = ()


def _compiled_lc_color_selector_lane_dags(
    dag: GenericDAG,
) -> tuple[tuple[int, GenericDAG], ...]:
    if dag.color_plan.color_accuracy != "lc" or dag.color_coverage != "complete":
        return ()
    lanes: list[tuple[int, GenericDAG]] = []
    for sector_id in contributing_color_sector_ids(dag):
        lane_dag = filter_dag_to_color_sectors(dag, (sector_id,))
        if not lane_dag.amplitude_roots:
            raise GenerationError(
                "compiled LC colour-selector lane has no amplitude roots for "
                f"materialized sector {sector_id}"
            )
        lanes.append((sector_id, lane_dag))
    return tuple(lanes)


def _compiled_helicity_selector_closure_lanes(
    dag: GenericDAG,
    schema: RuntimeExpressionSchema,
    model: Model,
) -> tuple[_HelicitySelectorLane, ...]:
    """Compile one reusable evaluator lane per exact helicity closure class."""

    materialization = dag.helicity_materialization
    if materialization is None:
        return ()
    grouped_domains: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[int],
    ] = {}
    for schedule in materialization.selector_schedules:
        if schedule.structural_zero:
            continue
        closure = (schedule.active_current_ids, schedule.active_root_ids)
        grouped_domains.setdefault(closure, []).append(schedule.selector_domain_id)

    stripped_dag = replace(
        dag,
        helicity_recurrence=None,
        helicity_materialization=None,
    )
    compile_dag = replace(stripped_dag, lc_topology_replay=None)
    lanes: list[_HelicitySelectorLane] = []
    for (active_current_ids, active_root_ids), domain_ids in sorted(
        grouped_domains.items(),
        key=lambda item: tuple(item[1]),
    ):
        lane_schema = _helicity_closure_runtime_schema(
            schema,
            dag,
            active_current_ids=active_current_ids,
        )
        if lane_schema is None:
            # Keep the exact partitioned primary path for an unusual closure
            # that does not traverse every compiled recursion stage.
            continue
        lanes.append(
            _HelicitySelectorLane(
                selector_domain_ids=tuple(sorted(domain_ids)),
                active_current_ids=active_current_ids,
                active_root_ids=active_root_ids,
                dag=stripped_dag,
                runtime_schema=lane_schema,
                stage_input=StageCompilationInput(
                    compile_dag,
                    model,
                    lane_schema,
                ),
            )
        )
    return tuple(lanes)


def _helicity_closure_runtime_schema(
    schema: RuntimeExpressionSchema,
    dag: GenericDAG,
    *,
    active_current_ids: Sequence[int],
) -> RuntimeExpressionSchema | None:
    """Filter compiled stages while retaining the parent's stable storage ABI."""

    payload = schema.to_mapping()
    if payload.pop("helicity_recurrence", None) is None:
        raise GenerationError(
            "compiled helicity closure lane has no recurrence metadata to strip"
        )
    active = {int(current_id) for current_id in active_current_ids}
    value_slots = {
        int(record["value_slot_id"]): record
        for record in cast(
            "Sequence[Mapping[str, object]]",
            cast("Mapping[str, object]", payload["value_storage"])["value_slots"],
        )
    }
    momentum_slot_by_mask = {
        int(record["momentum_mask"]): int(record["momentum_slot_id"])
        for record in cast(
            "Sequence[Mapping[str, object]]",
            payload["momentum_slots"],
        )
    }
    filtered_stages: list[dict[str, object]] = []
    for raw_stage in cast("Sequence[Mapping[str, object]]", payload["stages"]):
        stage = dict(raw_stage)
        interaction_ids = tuple(
            int(interaction_id)
            for interaction_id in cast(
                "Sequence[object]",
                stage.get("interaction_ids", ()),
            )
            if dag.interactions[int(interaction_id)].result_id in active
        )
        if not interaction_ids:
            return None
        interactions = tuple(dag.interactions[item] for item in interaction_ids)
        input_current_ids = {
            current_id
            for interaction in interactions
            for current_id in (interaction.left_id, interaction.right_id)
        }
        output_current_ids = {interaction.result_id for interaction in interactions}
        input_value_slot_ids = tuple(
            sorted(
                int(slot_id)
                for slot_id in cast(
                    "Sequence[object]",
                    stage["input_value_slot_ids"],
                )
                if int(value_slots[int(slot_id)]["current_id"]) in input_current_ids
            )
        )
        output_value_slot_ids = tuple(
            sorted(
                int(slot_id)
                for slot_id in cast(
                    "Sequence[object]",
                    stage["output_value_slot_ids"],
                )
                if int(value_slots[int(slot_id)]["current_id"]) in output_current_ids
            )
        )
        momentum_slot_ids = {
            momentum_slot_by_mask[dag.currents[current_id].index.momentum_mask]
            for interaction in interactions
            for current_id in (
                interaction.left_id,
                interaction.right_id,
                interaction.result_id,
            )
        }
        evaluation_groups = {
            (
                "group",
                int(interaction.evaluation_group_id),
            )
            if interaction.evaluation_group_id is not None
            else ("interaction", interaction.id)
            for interaction in interactions
        }
        stage.update(
            {
                "input_current_ids": sorted(input_current_ids),
                "output_current_ids": sorted(output_current_ids),
                "input_value_slot_ids": list(input_value_slot_ids),
                "output_value_slot_ids": list(output_value_slot_ids),
                "input_momentum_slot_ids": sorted(momentum_slot_ids),
                "interaction_count": len(interaction_ids),
                "interaction_evaluation_count": len(evaluation_groups),
                "interaction_ids": list(interaction_ids),
                "interactions_compacted": True,
                "interactions": [],
            }
        )
        filtered_stages.append(stage)
    payload["stages"] = filtered_stages
    return RuntimeExpressionSchema.from_mapping(payload)


def _complete_helicity_domain_contract(
    dag: GenericDAG,
) -> tuple[
    frozenset[tuple[tuple[int, int], ...]],
    frozenset[tuple[tuple[int, int], ...]],
]:
    plan = dag.helicity_recurrence
    if plan is None:
        raise GenerationError("helicity selector lane has no recurrence proof")
    complete_by_id = {
        domain.id: domain.source_states
        for domain in plan.selector_domains
        if domain.complete
    }
    complete = frozenset(complete_by_id.values())
    structural_zero = frozenset(
        complete_by_id[domain_id]
        for domain_id in plan.structural_zero_selector_domain_ids
        if domain_id in complete_by_id
    )
    return complete, structural_zero


def _validate_matching_helicity_selector_domains(
    parent: GenericDAG,
    nested: GenericDAG,
) -> None:
    parent_complete, parent_zero = _complete_helicity_domain_contract(parent)
    nested_complete, nested_zero = _complete_helicity_domain_contract(nested)
    if parent_complete != nested_complete:
        raise GenerationError(
            "complete-color helicity selector lane disagrees with the parent "
            "source-state domain"
        )
    if parent_zero != nested_zero:
        raise GenerationError(
            "complete-color helicity selector lane disagrees with the parent "
            "structural-zero domain"
        )


def _map_process_phase(
    items: Sequence[_ProcessInput],
    operation: Callable[[_ProcessInput], _ProcessOutput],
    *,
    executor: ThreadPoolExecutor | None,
    max_in_flight: int,
    phase_name: str,
    item_name: Callable[[_ProcessInput], str],
) -> tuple[_ProcessOutput, ...]:
    """Apply one process phase with bounded submissions and stable output order."""

    if max_in_flight < 1:
        raise ValueError("process phase max_in_flight must be positive")
    if executor is None:
        results: list[_ProcessOutput] = []
        for item in items:
            try:
                results.append(operation(item))
            except Exception as exc:
                raise GenerationError(
                    f"{phase_name} failed for process {item_name(item)!r}: {exc}"
                ) from exc
        return tuple(results)

    ordered: list[_ProcessOutput | object] = [_MISSING_PROCESS_RESULT for _ in items]
    active: dict[Future[_ProcessOutput], int] = {}
    next_index = 0

    def submit_one(index: int) -> None:
        item = items[index]
        try:
            future = executor.submit(operation, item)
        except Exception as exc:
            for pending in active:
                pending.cancel()
            if active:
                wait(tuple(active))
            raise GenerationError(
                f"{phase_name} failed to schedule process {item_name(item)!r}: {exc}"
            ) from exc
        active[future] = index

    while next_index < len(items) and len(active) < max_in_flight:
        submit_one(next_index)
        next_index += 1

    while active:
        done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
        failures: list[tuple[int, Exception]] = []
        for future in done:
            index = active.pop(future)
            try:
                ordered[index] = future.result()
            except Exception as exc:
                failures.append((index, exc))
        if failures:
            for pending in active:
                pending.cancel()
            if active:
                wait(tuple(active))
            index, exc = min(failures, key=lambda failure: failure[0])
            item = items[index]
            raise GenerationError(
                f"{phase_name} failed for process {item_name(item)!r}: {exc}"
            ) from exc
        while next_index < len(items) and len(active) < max_in_flight:
            submit_one(next_index)
            next_index += 1

    if any(result is _MISSING_PROCESS_RESULT for result in ordered):
        raise GenerationError(f"{phase_name} did not produce every process result")
    return tuple(cast(_ProcessOutput, result) for result in ordered)


class GenerationBackend:
    def __init__(
        self,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
        progress: ProgressSink | None,
        *,
        api_bundle_hook: ApiBundleHook | None = None,
        process_selection: _ProcessSelection | None = None,
    ) -> None:
        self._resource_config = config
        self._configuration = _GenerationConfigProvenance.from_config(config)
        self._config = self._configuration.effective
        self._progress = progress
        self._api_bundle_hook = api_bundle_hook
        self._process_selection_override = process_selection
        self._prepared_pack_warning_emitted = False

    def plan(
        self,
        processes: ProcessSet,
        *,
        model: ModelSource | CompiledModel | None = None,
    ) -> GenerationPlan:
        source = model or self._configured_model_source()
        task_id = "generation-plan"
        if self._progress is not None:
            self._progress.emit(
                ProgressStart(
                    task_id,
                    "Planning processes",
                    total=len(processes.requests),
                )
            )
        try:
            license_state = self._detect_symbolica_license()
            self._apply_symbolica_resource_policy(license_state)
            resolved_model = self._resolve_model_for_plan(source)
            self._require_eager_kernel_pack(resolved_model)
            self._apply_prepared_kernel_pack_policy(resolved_model)
            expanded = self._expand_process_set(
                processes,
                resolved_model,
                PhaseHandle(task_id, self._progress, len(processes.requests)),
            )
            self._apply_symbolica_resource_policy(
                license_state,
                process_count=len(expanded),
            )
            coverage: list[dict[str, object]] = []
            for entry in expanded:
                coverage.append(
                    self._plan_concrete_process(
                        entry.process_ir,
                        model=resolved_model.model,
                    )
                )
            result = GenerationPlan(
                concrete_processes=tuple(entry.request for entry in expanded),
                estimated_coverage={
                    "model_kind": resolved_model.source.kind,
                    "color_accuracy": self._color_accuracy,
                    "process_count": len(expanded),
                    "alias_count": sum(len(entry.aliases) for entry in expanded),
                    "processes": tuple(coverage),
                },
                requested_settings=self._configuration.requested,
                effective_settings=self._config,
                adjustments=self._configuration.adjustments,
                unsupported_features=(),
            )
        except Exception as exc:
            if self._progress is not None:
                self._progress.emit(
                    ProgressEnd(task_id, success=False, message=str(exc))
                )
            if isinstance(exc, (GenerationError, ModelError)):
                raise
            raise GenerationError(str(exc)) from exc
        if self._progress is not None:
            self._progress.emit(ProgressEnd(task_id))
        return result

    def generate(
        self,
        processes: ProcessSet,
        output: os.PathLike[str] | str,
        *,
        model: ModelSource | CompiledModel | None = None,
        mode: str = "error",
    ) -> GenerationResult:
        if mode not in ("error", "append", "replace"):
            raise ValueError("generation mode must be 'error', 'append', or 'replace'")
        write_mode = cast(Literal["error", "append", "replace"], mode)
        reporter = GenerationPhaseReporter(self._progress)
        run_config = self._run_config
        execution_mode = (
            "compiled"
            if run_config is None
            else str(run_config.evaluator.execution_mode)
        )
        backend = "jit" if run_config is None else str(run_config.evaluator.backend)
        if run_config is None:
            optimization = "O3"
        elif backend == "jit":
            optimization = f"O{run_config.evaluator.jit.optimization_level}"
        else:
            optimization = str(run_config.evaluator.cpp.optimization)
        reporter.start_generation(
            total=7 if execution_mode == "eager" else 8,
            details={
                "execution_mode": execution_mode,
                "backend": backend.upper(),
                "optimization": optimization.upper(),
            },
        )
        generation_started = time.perf_counter()
        try:
            license_state = self._detect_symbolica_license()
            self._apply_symbolica_resource_policy(license_state)
            source = model or self._configured_model_source()
            with reporter.phase(
                "model-loading",
                "Loading and compiling model",
                total=1,
            ) as phase:
                resolved_model = self._resolve_model(source)
                artifact_model = self._artifact_model(resolved_model)
                self._require_eager_kernel_pack(resolved_model)
                resolved_model = self._index_eager_kernel_pack(resolved_model)
                self._apply_prepared_kernel_pack_policy(resolved_model)
                phase.update(1, message=artifact_model.name)
            generation_model = resolved_model.model
            if generation_model is None:
                raise GenerationError("generation model resolution produced no model")

            with reporter.phase(
                "process-expansion",
                "Expanding concrete processes",
                total=len(processes.requests),
            ) as phase:
                expanded = self._expand_process_set(processes, resolved_model, phase)
            self._apply_symbolica_resource_policy(
                license_state,
                process_count=len(expanded),
            )

            with TemporaryDirectory(prefix="pyamplicol-generation-") as temporary:
                temporary_root = Path(temporary)
                worker_count = self._process_worker_count(len(expanded))
                executor = (
                    ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="pyamplicol-generation",
                    )
                    if worker_count > 1
                    else None
                )
                try:
                    with reporter.phase(
                        "dag",
                        "Compiling process DAGs",
                        total=len(expanded),
                    ) as phase:
                        compiled = _map_process_phase(
                            expanded,
                            lambda entry: self._compile_for_generation(
                                entry,
                                generation_model,
                                phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="DAG compilation",
                            item_name=lambda entry: entry.request.name,
                        )

                    indexed_compiled = tuple(enumerate(compiled))
                    with reporter.phase(
                        "warmup-filter",
                        (
                            "Applying structural reductions and preparing "
                            "validation points"
                        ),
                        total=len(compiled),
                    ) as phase:
                        prepared = _map_process_phase(
                            indexed_compiled,
                            lambda indexed: self._prepare_warmup_process(
                                indexed[1],
                                generation_model,
                                index=indexed[0],
                                phase=phase,
                            ),
                            executor=executor,
                            max_in_flight=worker_count,
                            phase_name="warmup and validation-point preparation",
                            item_name=lambda indexed: indexed[1].expanded.request.name,
                        )

                    if self._eager_execution_enabled:
                        with reporter.phase(
                            "eager-lowering",
                            "Lowering prepared-kernel DAG invocation tables",
                            total=len(prepared),
                        ) as phase:
                            artifact_processes = _map_process_phase(
                                prepared,
                                lambda entry: self._construct_eager_artifact(
                                    entry,
                                    generation_model,
                                    resolved_model,
                                    temporary_root,
                                    phase,
                                ),
                                executor=executor,
                                max_in_flight=worker_count,
                                phase_name="eager DAG-table lowering",
                                item_name=lambda entry: entry.expanded.request.name,
                            )
                    else:
                        with reporter.phase(
                            "evaluator-construction",
                            "Constructing runtime evaluator schemas",
                            total=len(prepared),
                        ) as phase:
                            evaluators = _map_process_phase(
                                prepared,
                                lambda entry: self._construct_evaluator(
                                    entry,
                                    generation_model,
                                    phase,
                                ),
                                executor=executor,
                                max_in_flight=worker_count,
                                phase_name="runtime evaluator-schema construction",
                                item_name=lambda entry: entry.expanded.request.name,
                            )

                        jit_total = sum(
                            _runtime_stage_count(evaluator.runtime_schema)
                            + 1
                            + (
                                0
                                if evaluator.helicity_sum_runtime_schema is None
                                else _runtime_stage_count(
                                    evaluator.helicity_sum_runtime_schema
                                )
                                + 1
                            )
                            + sum(
                                _helicity_selector_lane_compile_task_count(lane)
                                for lane in evaluator.helicity_selector_lanes
                            )
                            + sum(
                                _runtime_stage_count(lane.runtime_schema) + 1
                                for lane in evaluator.color_selector_lanes
                            )
                            for evaluator in evaluators
                        )
                        with reporter.phase(
                            "jit",
                            "Compiling and materializing stage evaluators",
                            total=jit_total,
                        ) as phase:
                            # Symbolica expressions are created while resolving the
                            # model on this caller thread. Keep materialization on that
                            # same thread. The process-wide lock already made this phase
                            # serial; moving between worker threads can violate backend
                            # thread affinity.
                            artifact_processes = _map_process_phase(
                                evaluators,
                                lambda evaluator: self._materialize_evaluator(
                                    evaluator,
                                    generation_model,
                                    temporary_root,
                                    phase,
                                ),
                                executor=None,
                                max_in_flight=1,
                                phase_name="evaluator materialization",
                                item_name=lambda evaluator: (
                                    evaluator.compiled.expanded.request.name
                                ),
                            )
                finally:
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)

                with reporter.phase(
                    "artifact-writing",
                    "Writing schema-v3 artifact",
                    total=1,
                ) as phase:

                    def artifact_progress(event: dict[str, object]) -> None:
                        details = {
                            str(key): value
                            for key, value in event.items()
                            if isinstance(value, (str, int, float, bool, type(None)))
                        }
                        phase.update(
                            int(event.get("completed", phase.completed)),
                            total=(
                                None
                                if event.get("total") is None
                                else int(event["total"])
                            ),
                            message=str(event.get("step", "writing artifact")),
                            details=details,
                        )

                    write_result = write_schema_v3_artifact(
                        Path(output),
                        mode=write_mode,
                        source=resolved_model.source,
                        compiled_model=artifact_model,
                        configuration=self._configuration,
                        processes=artifact_processes,
                        timings=reporter.timings,
                        api_bundle_hook=self._api_bundle_hook,
                        progress_callback=(
                            artifact_progress if phase.sink is not None else None
                        ),
                    )
                    phase.update(
                        phase.total or phase.completed,
                        message=str(write_result.output),
                        details={
                            "step": "artifact ready",
                            "file_count": len(write_result.files),
                        },
                    )

                concrete_requests = tuple(entry.expanded.request for entry in prepared)
                validation_points_by_process = {
                    process.expanded.request.name: process.validation_points
                    for process in prepared
                }
                expected_process_ids = tuple(
                    process.process_id for process in artifact_processes
                )
                del compiled
                del indexed_compiled
                del prepared
                del artifact_processes
                gc.collect()

            with reporter.phase(
                "validation",
                "Validating generated artifact",
                total=1,
            ) as phase:
                validation = self._generation_config.validation
                if validation.post_build_validation:
                    self._validate_generated_artifact(
                        write_result.output,
                        expected_process_ids,
                        validation_points=validation_points_by_process,
                        expected_api_bundle_path=write_result.api_bundle_path,
                        progress=phase,
                    )
                    message = (
                        f"schema and {validation.samples} numerical samples"
                        if validation.enabled
                        else "schema, references, hashes, and target"
                    )
                else:
                    message = "post-build validation disabled"
                phase.update(1, message=message)

            reporter.timings["total"] = time.perf_counter() - generation_started
        except KeyboardInterrupt:
            reporter.finish_generation(success=False, message="interrupted")
            raise
        except Exception as exc:
            reporter.finish_generation(success=False, message=str(exc))
            if isinstance(exc, PyAmpliColError):
                raise
            raise GenerationError(str(exc)) from exc

        reporter.finish_generation()
        return GenerationResult(
            output=write_result.output,
            processes=ProcessSet(
                requests=concrete_requests,
                aliases=processes.aliases,
            ),
            mode=write_mode,
            files=write_result.files,
        )

    def _artifact_model(self, resolved: _ResolvedModel) -> _CompiledModelPayload:
        if resolved.compiled is not None:
            return resolved.compiled
        from ..models.loading import compile_model_source

        return compile_model_source("BUILTIN_SM", use_cache=False)

    def _expand_process_set(
        self,
        processes: ProcessSet,
        resolved: _ResolvedModel,
        phase: PhaseHandle,
    ) -> tuple[_ExpandedProcess, ...]:
        aliases_by_target: dict[str, list[ProcessAlias]] = {}
        for alias in processes.aliases:
            aliases_by_target.setdefault(alias.process_name, []).append(alias)
        result: list[_ExpandedProcess] = []
        names: set[str] = set()
        for request_index, request in enumerate(processes.requests, start=1):
            expanded = self._expand_request(request, resolved)
            request_aliases = aliases_by_target.get(request.name, [])
            if request_aliases and len(expanded) != 1:
                raise GenerationError(
                    f"process aliases for {request.name!r} are ambiguous after "
                    f"expansion into {len(expanded)} concrete processes"
                )
            for concrete_index, process_ir in enumerate(expanded, start=1):
                name = _expanded_name(
                    request.name,
                    concrete_index,
                    len(expanded),
                )
                if name in names:
                    raise GenerationError(f"expanded process ID is not unique: {name}")
                names.add(name)
                concrete = ProcessRequest.parse(process_ir.process, name=name)
                alias_records: list[Mapping[str, object]] = []
                for alias in request_aliases:
                    permutation = tuple(alias.particle_permutation) or tuple(
                        range(len(process_ir.legs))
                    )
                    if len(permutation) != len(process_ir.legs):
                        raise GenerationError(
                            f"alias {alias.name!r} permutation has length "
                            f"{len(permutation)}, expected {len(process_ir.legs)}"
                        )
                    if permutation[:2] != (0, 1):
                        raise GenerationError(
                            f"alias {alias.name!r} may only permute final-state "
                            "particles; genuine crossing reuse is not enabled"
                        )
                    alias_expression, alias_pdgs = _permuted_process_identity(
                        process_ir,
                        permutation,
                    )
                    alias_records.append(
                        {
                            "id": alias.name,
                            "expression": alias_expression,
                            "external_pdgs": list(alias_pdgs),
                            "external_permutation": list(permutation),
                        }
                    )
                result.append(
                    _ExpandedProcess(
                        request=concrete,
                        process_ir=process_ir,
                        aliases=tuple(alias_records),
                    )
                )
            phase.update(request_index, message=request.name)
        return tuple(result)

    def _compile_for_generation(
        self,
        expanded: _ExpandedProcess,
        model: Model,
        phase: PhaseHandle,
    ) -> _DagProcess:
        process_name = expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: DAG construction",
            details={
                "process": process_name,
                "step": "colour-plan",
            },
        ) as task:

            def report(details: Mapping[str, str | int]) -> None:
                payload = {"process": process_name, **details}
                completed = task.completed
                total: int | None = None
                if "mask_index" in details and "mask_total" in details:
                    completed = int(details["mask_index"])
                    total = int(details["mask_total"])
                task.update(
                    completed,
                    total=total,
                    message=str(details.get("step", "DAG construction")),
                    details=payload,
                )

            dag, coverage = self._compile_concrete_process(
                expanded.process_ir,
                model,
                progress_callback=report if task.sink is not None else None,
            )
            task.update(
                task.completed,
                message="DAG complete",
                details={
                    "process": process_name,
                    "step": "DAG complete",
                    "current_count": len(dag.currents),
                    "interaction_count": len(dag.interactions),
                    "amplitude_count": len(dag.amplitude_roots),
                },
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "DAG complete"},
        )
        return _DagProcess(expanded=expanded, dag=dag, coverage=coverage)

    def _prepare_warmup_process(
        self,
        process: _DagProcess,
        model: Model,
        *,
        index: int,
        phase: PhaseHandle,
    ) -> _CompiledProcess:
        process_name = process.expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: structural reduction",
            total=2,
            details={"process": process_name, "step": "structural reduction"},
        ) as task:
            result = self._prepare_warmup_process_inner(
                process,
                model,
                index=index,
                progress=task,
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "validation points ready"},
        )
        return result

    def _prepare_warmup_process_inner(
        self,
        process: _DagProcess,
        model: Model,
        *,
        index: int,
        progress: PhaseHandle,
    ) -> _CompiledProcess:
        process_name = process.expanded.request.name
        progress.update(
            0,
            message="structural reduction",
            details={
                "process": process_name,
                "step": "structural reduction",
                "current_count": len(process.dag.currents),
                "interaction_count": len(process.dag.interactions),
                "amplitude_count": len(process.dag.amplitude_roots),
            },
        )
        parity_preweighted = any(
            float(root.helicity_weight) > 1.0 for root in process.dag.amplitude_roots
        )
        reduced = (
            prune_dag_to_amplitude_roots(process.dag)
            if parity_preweighted
            else prune_global_helicity_flip_equivalent_roots(process.dag, model)
        )
        helicity_sum_dag: GenericDAG | None = None
        helicity_selector_union_dag: GenericDAG | None = None
        all_flow_union = self._all_flow_union_enabled
        helicity_recurrence = build_helicity_recurrence_plan(reduced, model)
        if helicity_recurrence is not None:
            if self._eager_execution_enabled:
                # Eager invocation tables already carry selector domains. Keep
                # the exact recurrence proof, but do not serialize the
                # compiled lane's evaluator-chunk schedules into an eager
                # artifact.
                reduced = replace(
                    reduced,
                    helicity_recurrence=helicity_recurrence,
                    helicity_materialization=None,
                )
            else:
                # Topology replay keeps its existing fused helicity-sum lane.
                # The all-flow union is itself the primary reusable lane, so
                # compiling that second payload would duplicate the expensive
                # cross-flow recurrence without helping its target workload.
                if not all_flow_union:
                    helicity_sum_dag = replace(
                        reduced,
                        helicity_recurrence=None,
                        helicity_materialization=None,
                    )
                materialization = materialize_helicity_recurrence(
                    reduced,
                    helicity_recurrence,
                )
                reduced = replace(
                    materialization.dag,
                    helicity_recurrence=helicity_recurrence,
                    helicity_materialization=materialization,
                )
                if not all_flow_union:
                    helicity_selector_union_dag = (
                        self._compile_complete_lc_helicity_selector_union(
                            process,
                            model,
                        )
                    )
                if helicity_selector_union_dag is not None:
                    _validate_matching_helicity_selector_domains(
                        reduced,
                        helicity_selector_union_dag,
                    )
        before_amplitude_roots = (
            round(
                sum(float(root.helicity_weight) for root in process.dag.amplitude_roots)
            )
            if parity_preweighted
            else len(process.dag.amplitude_roots)
        )
        validation = self._generation_config.validation
        filters: dict[str, object] = {
            **(
                {}
                if self._color_accuracy != "lc"
                else {
                    "lc_flow_layout": (
                        "all-flow-union" if all_flow_union else "topology-replay"
                    )
                }
            ),
            "structural_helicity_reduction": {
                "applied": parity_preweighted or reduced is not process.dag,
                "before_amplitude_roots": before_amplitude_roots,
                "after_amplitude_roots": len(reduced.amplitude_roots),
                "mode": "proven global-helicity-flip equivalence",
            },
            **(
                {}
                if reduced.helicity_recurrence is None
                else {
                    "helicity_recurrence": {
                        "contract_version": (HELICITY_RECURRENCE_CONTRACT_VERSION),
                        **reduced.helicity_recurrence.proof_counts(),
                        **(
                            {}
                            if reduced.helicity_materialization is None
                            else {
                                "materialized_current_count": (
                                    reduced.helicity_materialization.materialized_current_count
                                ),
                                "materialized_amplitude_count": (
                                    reduced.helicity_materialization.materialized_root_count
                                ),
                                "materialization_strategy": (
                                    reduced.helicity_materialization.strategy
                                ),
                            }
                        ),
                    }
                }
            ),
        }
        progress.update(
            1,
            message="validation-point preparation",
            details={
                "process": process_name,
                "step": "validation-point preparation",
                "current_count": len(reduced.currents),
                "interaction_count": len(reduced.interactions),
                "amplitude_count": len(reduced.amplitude_roots),
            },
        )
        sample_count = (
            validation.samples
            if validation.enabled and validation.post_build_validation
            else 1
        )
        points = tuple(
            build_validation_point(
                reduced,
                model,
                process_id=process.expanded.request.name,
                seed=validation.seed + index * sample_count + sample_index,
            )
            for sample_index in range(sample_count)
        )
        point = points[0]
        progress.update(
            2,
            message=(
                process_name
                if point.available
                else f"{process_name}: metadata-only validation"
            ),
            details={
                "process": process_name,
                "step": "validation points ready",
                "samples": len(points),
            },
        )
        coverage = {
            **dict(process.coverage),
            "current_count": len(reduced.currents),
            "interaction_count": len(reduced.interactions),
            "interaction_evaluation_count": reduced.interaction_evaluation_count,
            "amplitude_root_count": len(reduced.amplitude_roots),
            **(
                {}
                if reduced.helicity_recurrence is None
                else {
                    "helicity_recurrence_contract_version": (
                        HELICITY_RECURRENCE_CONTRACT_VERSION
                    ),
                    **reduced.helicity_recurrence.proof_counts(),
                    **(
                        {}
                        if reduced.helicity_materialization is None
                        else {
                            "proof_current_count": (
                                reduced.helicity_materialization.proof_current_count
                            ),
                            "proof_amplitude_count": (
                                reduced.helicity_materialization.proof_root_count
                            ),
                            "helicity_materialization_strategy": (
                                reduced.helicity_materialization.strategy
                            ),
                        }
                    ),
                }
            ),
        }
        return _CompiledProcess(
            expanded=process.expanded,
            dag=reduced,
            helicity_sum_dag=helicity_sum_dag,
            helicity_selector_union_dag=helicity_selector_union_dag,
            coverage=coverage,
            filters=filters,
            validation_points=points,
        )

    def _construct_evaluator(
        self,
        process: _CompiledProcess,
        model: Model,
        phase: PhaseHandle,
    ) -> _EvaluatorProcess:
        process_name = process.expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: runtime schema",
            details={"process": process_name, "step": "runtime layout"},
        ) as task:
            schema = build_runtime_expression_schema(
                process.dag,
                model,
                process_id=process_name,
            )
            helicity_sum_schema = (
                None
                if process.helicity_sum_dag is None
                else build_runtime_expression_schema(
                    process.helicity_sum_dag,
                    model,
                    process_id=process_name,
                )
            )
            helicity_selector_lanes: tuple[_HelicitySelectorLane, ...] = ()
            if (
                process.dag.helicity_coverage == "complete"
                and not process.dag.selected_source_helicities
                and process.dag.helicity_recurrence is not None
                and process.dag.helicity_materialization is not None
            ):
                union_dag = process.helicity_selector_union_dag
                if union_dag is not None:
                    union_schema = build_runtime_expression_schema(
                        union_dag,
                        model,
                        process_id=process_name,
                    )
                    materialization = process.dag.helicity_materialization
                    assert materialization is not None
                    helicity_selector_lanes = (
                        _HelicitySelectorLane(
                            selector_domain_ids=tuple(
                                sorted(
                                    schedule.selector_domain_id
                                    for schedule in materialization.selector_schedules
                                    if not schedule.structural_zero
                                )
                            ),
                            active_current_ids=(),
                            active_root_ids=(),
                            dag=union_dag,
                            runtime_schema=union_schema,
                            stage_input=StageCompilationInput(
                                union_dag,
                                model,
                                union_schema,
                            ),
                            schedule_mode="nested-runtime",
                            child_lanes=(
                                _compiled_helicity_selector_closure_lanes(
                                    union_dag,
                                    union_schema,
                                    model,
                                )
                            ),
                        ),
                    )
                else:
                    helicity_selector_lanes = _compiled_helicity_selector_closure_lanes(
                        process.dag,
                        schema,
                        model,
                    )
            color_selector_dag = (
                process.dag
                if process.helicity_sum_dag is None
                else process.helicity_sum_dag
            )
            color_selector_lanes: list[_ColorSelectorLane] = []
            sector_lanes = (
                ()
                if self._all_flow_union_enabled
                else _compiled_lc_color_selector_lane_dags(color_selector_dag)
            )
            for sector_id, lane_dag in sector_lanes:
                lane_schema = build_runtime_expression_schema(
                    lane_dag,
                    model,
                    process_id=process_name,
                )
                color_selector_lanes.append(
                    _ColorSelectorLane(
                        materialized_sector_id=sector_id,
                        dag=lane_dag,
                        runtime_schema=lane_schema,
                        stage_input=StageCompilationInput(
                            lane_dag,
                            model,
                            lane_schema,
                        ),
                    )
                )
            stage_count = _runtime_stage_count(schema) + (
                0
                if helicity_sum_schema is None
                else _runtime_stage_count(helicity_sum_schema)
            )
            stage_count += sum(
                _helicity_selector_lane_stage_count(lane)
                for lane in helicity_selector_lanes
            )
            stage_count += sum(
                _runtime_stage_count(lane.runtime_schema)
                for lane in color_selector_lanes
            )
            task.update(
                stage_count,
                total=stage_count,
                message="runtime schema ready",
                details={
                    "process": process_name,
                    "step": "runtime schema ready",
                    "stage_index": stage_count,
                    "stage_total": stage_count,
                    "current_count": len(process.dag.currents),
                    "interaction_count": len(process.dag.interactions),
                    "output_count": len(process.dag.amplitude_roots),
                    "helicity_sum_lane": helicity_sum_schema is not None,
                    "helicity_selector_closure_lane_count": len(
                        helicity_selector_lanes
                    ),
                    "color_selector_lane_count": len(color_selector_lanes),
                },
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "runtime schema ready"},
        )
        return _EvaluatorProcess(
            compiled=process,
            runtime_schema=schema,
            stage_input=StageCompilationInput(process.dag, model, schema),
            helicity_sum_runtime_schema=helicity_sum_schema,
            helicity_sum_stage_input=(
                None
                if helicity_sum_schema is None or process.helicity_sum_dag is None
                else StageCompilationInput(
                    process.helicity_sum_dag,
                    model,
                    helicity_sum_schema,
                )
            ),
            helicity_selector_lanes=helicity_selector_lanes,
            color_selector_lanes=tuple(color_selector_lanes),
        )

    def _construct_eager_artifact(
        self,
        process: _CompiledProcess,
        model: Model,
        resolved_model: _ResolvedModel,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> EagerProcessArtifact | EagerPlanV3ProcessArtifact:
        process_name = process.expanded.request.name
        eager_kernel_index = resolved_model.eager_kernel_index
        if eager_kernel_index is None:
            raise GenerationError(
                "eager DAG lowering requires an indexed prepared model bundle"
            )
        resolver = PreparedCatalogEagerKernelResolver(
            process.dag,
            eager_kernel_index,
        )
        if _selected_eager_plan_version() == _EAGER_PLAN_V3:
            return self._construct_eager_plan_v3_artifact(
                process,
                model,
                resolver,
                temporary_root,
                phase,
            )
        with phase.child(
            process_name,
            f"{process_name}: eager DAG lowering",
            details={"process": process_name, "step": "runtime layout"},
        ) as task:

            def report(details: Mapping[str, str | int]) -> None:
                completed = int(details.get("stage_index", task.completed))
                total_value = details.get("stage_total")
                total = None if total_value is None else int(total_value)
                task.update(
                    completed,
                    total=total,
                    message=str(details.get("step", "eager lowering")),
                    details={"process": process_name, **details},
                )

            schema_mapping, eager_tables = lower_fused_eager_execution(
                dag=process.dag,
                model=model,
                resolver=resolver,
                process_id=process_name,
                progress_callback=report if task.sink is not None else None,
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "eager plan ready"},
        )
        run = self._run_config
        if run is None:  # pragma: no cover - eager mode requires RunConfig
            raise GenerationError("eager generation has no evaluator configuration")
        ir = process.expanded.process_ir
        dag = process.dag
        return EagerProcessArtifact(
            process_id=process_name,
            expression=ir.process,
            color_accuracy=ir.color_accuracy,
            external_pdgs=(*ir.initial_pdgs, *ir.final_pdgs),
            aliases=process.expanded.aliases,
            runtime_schema=schema_mapping,
            eager_tables=eager_tables,
            point_tile_size=run.evaluator.eager.point_tile_size,
            workspace_mib=run.evaluator.eager.workspace_mib,
            dag_summary={
                "current_count": len(dag.currents),
                "source_count": len(dag.sources),
                "interaction_count": len(dag.interactions),
                "amplitude_root_count": len(dag.amplitude_roots),
                "truncated": False,
            },
            validation_point=process.validation_points[0],
            generation_filters=process.filters,
        )

    def _construct_eager_plan_v3_artifact(
        self,
        process: _CompiledProcess,
        model: Model,
        resolver: PreparedCatalogEagerKernelResolver,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> EagerPlanV3ProcessArtifact:
        process_name = process.expanded.request.name
        with phase.child(
            process_name,
            f"{process_name}: Rust eager runtime lowering",
            total=2,
            details={"process": process_name, "step": "columnar lowering input"},
        ) as task:
            lowering_input = build_eager_lowering_input_v1(
                dag=process.dag,
                model=model,
                resolver=resolver,
                process_id=process_name,
            )
            task.update(
                1,
                message="columnar lowering input ready",
                details={
                    "process": process_name,
                    "step": "columnar lowering input ready",
                },
            )
            output = _invoke_rust_eager_lowering_v1(
                lowering_input,
                temporary_root
                / "eager-plan-v3"
                / process_name
                / "eager-runtime.pacbin",
            )
            task.update(
                2,
                message="Rust eager runtime ready",
                details={"process": process_name, "step": "eager runtime ready"},
            )
        phase.advance(
            message=process_name,
            details={"process": process_name, "step": "eager runtime ready"},
        )
        run = self._run_config
        if run is None:  # pragma: no cover - eager mode requires RunConfig
            raise GenerationError("eager generation has no evaluator configuration")
        ir = process.expanded.process_ir
        dag = process.dag
        return EagerPlanV3ProcessArtifact(
            process_id=process_name,
            expression=ir.process,
            color_accuracy=ir.color_accuracy,
            external_pdgs=(*ir.initial_pdgs, *ir.final_pdgs),
            aliases=process.expanded.aliases,
            physics=output.physics,
            eager_runtime_path=output.payload_path,
            eager_runtime_size_bytes=output.payload_size_bytes,
            eager_runtime_sha256=output.payload_sha256,
            eager_runtime_member_count=output.member_count,
            eager_runtime_unpacked_size_bytes=output.unpacked_size_bytes,
            eager_runtime_index_sha256=output.index_sha256,
            lowering_input_sha256=lowering_input.digest,
            referenced_kernel_ids=_eager_lowering_kernel_ids(lowering_input),
            inspection_summary=output.inspection_summary,
            point_tile_size=run.evaluator.eager.point_tile_size,
            workspace_mib=run.evaluator.eager.workspace_mib,
            dag_summary={
                "current_count": len(dag.currents),
                "source_count": len(dag.sources),
                "interaction_count": len(dag.interactions),
                "amplitude_root_count": len(dag.amplitude_roots),
                "truncated": False,
            },
            validation_point=process.validation_points[0],
            generation_filters=process.filters,
        )

    def _materialize_evaluator(
        self,
        process: _EvaluatorProcess,
        model: Model,
        temporary_root: Path,
        phase: PhaseHandle,
    ) -> CompiledProcessArtifact:
        process_id = process.compiled.expanded.request.name
        run = self._run_config
        backend = "JIT" if run is None else str(run.evaluator.backend).upper()
        with phase.child(
            process_id,
            f"{process_id}: evaluator compilation",
            details={
                "process": process_id,
                "step": "waiting for compiler",
                "backend": backend,
            },
        ) as task:
            task.update(
                0,
                message="waiting for compiler",
                details={
                    "process": process_id,
                    "step": "waiting for compiler",
                    "backend": backend,
                },
            )
            with _SYMBOLICA_MATERIALIZATION_LOCK:
                return self._materialize_evaluator_unlocked(
                    process,
                    model,
                    temporary_root,
                    phase,
                    task,
                    backend=backend,
                )

    def _materialize_evaluator_unlocked(
        self,
        process: _EvaluatorProcess,
        model: Model,
        temporary_root: Path,
        phase: PhaseHandle,
        progress: PhaseHandle,
        *,
        backend: str,
    ) -> CompiledProcessArtifact:
        process_id = process.compiled.expanded.request.name
        evaluator_root = temporary_root / process_id
        total_stage_count = _runtime_stage_count(process.runtime_schema) + 1
        if process.helicity_sum_runtime_schema is not None:
            total_stage_count += (
                _runtime_stage_count(process.helicity_sum_runtime_schema) + 1
            )
        total_stage_count += sum(
            _helicity_selector_lane_compile_task_count(lane)
            for lane in process.helicity_selector_lanes
        )
        total_stage_count += sum(
            _runtime_stage_count(lane.runtime_schema) + 1
            for lane in process.color_selector_lanes
        )

        def evaluator_progress(
            event: dict[str, object],
            *,
            execution_lane: str,
        ) -> None:
            details = {
                str(key): value
                for key, value in event.items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
            details.update(
                {
                    "process": process_id,
                    "backend": backend,
                    "execution_lane": execution_lane,
                    "lane_stage_total": event.get("total"),
                }
            )
            if event.get("stage") == "stage complete":
                completed = progress.completed + int(event.get("increment", 1))
                progress.update(
                    completed,
                    total=total_stage_count,
                    message=str(event.get("item", "stage complete")),
                    details=details,
                )
                phase.advance(
                    message=str(event.get("item", process_id)),
                    details=details,
                )
                return
            progress.update(
                progress.completed,
                total=total_stage_count,
                message=str(event.get("item", event.get("stage", "compiling"))),
                details=details,
            )

        def compile_lane(
            stage_input: StageCompilationInput,
            runtime_schema: RuntimeExpressionSchema,
            output_root: Path,
            *,
            execution_lane: str,
        ) -> tuple[Mapping[str, object], Mapping[str, object] | None]:
            callback = (
                None
                if progress.sink is None
                else lambda event: evaluator_progress(
                    event,
                    execution_lane=execution_lane,
                )
            )
            _blueprint, stage_manifest = (
                build_and_write_generic_stage_evaluator_artifacts(
                    stage_input,
                    runtime_schema.to_mapping(),
                    output_root,
                    model=model,
                    stage_local_parameter_layout=True,
                    symbolica_settings=self._symbolica_settings(),
                    jit_compile=True,
                    evaluator_progress_callback=callback,
                )
            )
            if not bool(stage_manifest.get("runtime_available")):
                raise GenerationError(
                    f"{execution_lane} stage evaluator for {process_id!r} "
                    "is not runtime-available"
                )
            model_parameter_evaluator = write_model_parameter_evaluator_artifact(
                model,
                runtime_schema.to_mapping(),
                output_root,
                symbolica_settings=self._symbolica_settings(),
                jit_compile=True,
                progress_callback=callback,
            )
            return stage_manifest, model_parameter_evaluator

        stage_manifest, model_parameter_evaluator = compile_lane(
            process.stage_input,
            process.runtime_schema,
            evaluator_root,
            execution_lane="selected-helicity",
        )
        helicity_sum_execution: CompiledExecutionArtifact | None = None
        if (
            process.helicity_sum_runtime_schema is not None
            or process.helicity_sum_stage_input is not None
        ):
            if (
                process.helicity_sum_runtime_schema is None
                or process.helicity_sum_stage_input is None
                or process.compiled.helicity_sum_dag is None
            ):
                raise GenerationError(
                    f"helicity-sum execution inputs for {process_id!r} are inconsistent"
                )
            helicity_sum_root = temporary_root / ".helicity-sum" / process_id
            (
                helicity_sum_stage_manifest,
                helicity_sum_model_parameter_evaluator,
            ) = compile_lane(
                process.helicity_sum_stage_input,
                process.helicity_sum_runtime_schema,
                helicity_sum_root,
                execution_lane="helicity-sum",
            )
            helicity_sum_dag = process.compiled.helicity_sum_dag
            helicity_sum_execution = CompiledExecutionArtifact(
                runtime_schema=process.helicity_sum_runtime_schema,
                stage_manifest=helicity_sum_stage_manifest,
                model_parameter_evaluator=(helicity_sum_model_parameter_evaluator),
                dag_summary={
                    "current_count": len(helicity_sum_dag.currents),
                    "source_count": len(helicity_sum_dag.sources),
                    "interaction_count": len(helicity_sum_dag.interactions),
                    "amplitude_root_count": len(helicity_sum_dag.amplitude_roots),
                    "truncated": False,
                },
                evaluator_root=helicity_sum_root,
            )
        helicity_selector_executions: list[
            CompiledHelicitySelectorExecutionArtifact
        ] = []

        def compile_helicity_selector_lane(
            lane: _HelicitySelectorLane,
            lane_root: Path,
            *,
            execution_lane: str,
        ) -> CompiledHelicitySelectorExecutionArtifact:
            lane_stage_manifest, lane_model_parameter_evaluator = compile_lane(
                lane.stage_input,
                lane.runtime_schema,
                lane_root,
                execution_lane=execution_lane,
            )
            child_executions = tuple(
                compile_helicity_selector_lane(
                    child,
                    lane_root.parent / f"{lane_root.name}-closure-{child_index}",
                    execution_lane=(f"{execution_lane}-closure-{child_index}"),
                )
                for child_index, child in enumerate(lane.child_lanes)
            )
            return CompiledHelicitySelectorExecutionArtifact(
                selector_domain_ids=lane.selector_domain_ids,
                schedule_mode=lane.schedule_mode,
                execution=CompiledExecutionArtifact(
                    runtime_schema=lane.runtime_schema,
                    stage_manifest=lane_stage_manifest,
                    model_parameter_evaluator=(lane_model_parameter_evaluator),
                    # Closure lanes retain their parent's stable current and
                    # amplitude storage ABI while filtering stage interactions.
                    dag_summary=_runtime_schema_dag_summary(lane.runtime_schema),
                    evaluator_root=lane_root,
                    helicity_selector_executions=child_executions,
                ),
            )

        for lane_index, lane in enumerate(process.helicity_selector_lanes):
            helicity_selector_root = (
                temporary_root
                / ".helicity-selector-union"
                / process_id
                / f"class-{lane_index}"
            )
            helicity_selector_executions.append(
                compile_helicity_selector_lane(
                    lane,
                    helicity_selector_root,
                    execution_lane=f"helicity-selector-class-{lane_index}",
                )
            )
        color_selector_executions: list[CompiledColorSelectorExecutionArtifact] = []
        selector_root_name = (
            ".helicity-sum-color-selector"
            if helicity_sum_execution is not None
            else ".color-selector"
        )
        for lane in process.color_selector_lanes:
            lane_root = (
                temporary_root
                / selector_root_name
                / process_id
                / f"sector-{lane.materialized_sector_id}"
            )
            lane_stage_manifest, lane_model_parameter_evaluator = compile_lane(
                lane.stage_input,
                lane.runtime_schema,
                lane_root,
                execution_lane=(f"color-selector sector-{lane.materialized_sector_id}"),
            )
            color_selector_executions.append(
                CompiledColorSelectorExecutionArtifact(
                    materialized_sector_id=lane.materialized_sector_id,
                    execution=CompiledExecutionArtifact(
                        runtime_schema=lane.runtime_schema,
                        stage_manifest=lane_stage_manifest,
                        model_parameter_evaluator=(lane_model_parameter_evaluator),
                        dag_summary={
                            "current_count": len(lane.dag.currents),
                            "source_count": len(lane.dag.sources),
                            "interaction_count": len(lane.dag.interactions),
                            "amplitude_root_count": len(lane.dag.amplitude_roots),
                            "truncated": False,
                        },
                        evaluator_root=lane_root,
                    ),
                )
            )
        primary_color_selector_executions: tuple[
            CompiledColorSelectorExecutionArtifact, ...
        ] = tuple(color_selector_executions)
        if helicity_sum_execution is not None:
            helicity_sum_execution = replace(
                helicity_sum_execution,
                color_selector_executions=tuple(color_selector_executions),
            )
            primary_color_selector_executions = ()
        progress.update(
            progress.completed,
            message="model parameters ready",
            details={
                "process": process_id,
                "step": "model parameters ready",
                "backend": backend,
            },
        )
        phase.advance(
            message=f"{process_id}: model parameters",
            details={
                "process": process_id,
                "step": "model parameters ready",
                "backend": backend,
            },
        )
        ir = process.compiled.expanded.process_ir
        dag = process.compiled.dag
        return CompiledProcessArtifact(
            process_id=process_id,
            expression=ir.process,
            color_accuracy=ir.color_accuracy,
            external_pdgs=(*ir.initial_pdgs, *ir.final_pdgs),
            aliases=process.compiled.expanded.aliases,
            runtime_schema=process.runtime_schema,
            stage_manifest=stage_manifest,
            model_parameter_evaluator=model_parameter_evaluator,
            dag_summary={
                "current_count": len(dag.currents),
                "source_count": len(dag.sources),
                "interaction_count": len(dag.interactions),
                "amplitude_root_count": len(dag.amplitude_roots),
                "truncated": False,
            },
            evaluator_root=evaluator_root,
            validation_point=process.compiled.validation_points[0],
            generation_filters=process.compiled.filters,
            helicity_sum_execution=helicity_sum_execution,
            helicity_selector_executions=tuple(helicity_selector_executions),
            color_selector_executions=primary_color_selector_executions,
        )

    def _process_worker_count(self, process_count: int) -> int:
        if process_count < 1:
            return 1
        configured = self._generation_config.workers
        requested = (
            max(1, os.cpu_count() or 1)
            if configured == "auto"
            else max(1, int(configured))
        )
        return min(process_count, requested)

    def _detect_symbolica_license(self) -> SymbolicaLicenseState:
        from ..licensing import detect_symbolica_license

        run = self._run_config
        return detect_symbolica_license(
            suggest=True if run is None else run.symbolica.suggest_license,
            json_mode=False if run is None else str(run.output.format) == "json",
        )

    def _apply_symbolica_resource_policy(
        self,
        state: SymbolicaLicenseState,
        *,
        process_count: int | None = None,
    ) -> None:
        from ..licensing import resolve_symbolica_resource_config

        resolution = resolve_symbolica_resource_config(
            self._resource_config,
            state,
            process_count=process_count,
        )
        self._resource_config = resolution
        self._configuration = _GenerationConfigProvenance.from_config(resolution)
        self._config = resolution.effective

    def _symbolica_settings(self) -> object:
        from ..evaluators.symbolica import SymbolicaEvaluatorSettings

        run = self._run_config
        if run is None:
            return SymbolicaEvaluatorSettings()
        optimization = run.evaluator.optimization
        if optimization.horner_iterations < 1:
            raise GenerationError(
                "evaluator.optimization.horner_iterations must be positive "
                "for generation"
            )
        backend = str(run.evaluator.backend)
        evaluator_backend = "jit" if backend == "jit" else "compiled-complex"
        cores = (
            max(1, os.cpu_count() or 1)
            if optimization.cores == "auto"
            else int(optimization.cores)
        )
        collect_factors = (
            False
            if optimization.collect_factors == "auto"
            else bool(optimization.collect_factors)
        )
        cpp_level = _cpp_optimization_level(run.evaluator.cpp.optimization)
        return SymbolicaEvaluatorSettings(
            backend=evaluator_backend,
            iterations=optimization.horner_iterations,
            cpe_iterations=optimization.cpe_iterations,
            n_cores=cores,
            jit_direct_translation=False,
            jit_optimization_level=run.evaluator.jit.optimization_level,
            max_horner_scheme_variables=optimization.max_horner_variables,
            max_common_pair_cache_entries=(optimization.max_common_pair_cache_entries),
            max_common_pair_distance=optimization.max_common_pair_distance,
            collect_factors=collect_factors,
            compiled_inline_asm="default" if backend == "asm" else "none",
            compiled_optimization_level=cpp_level,
            compiled_native=run.evaluator.cpp.native_arch,
            compiler_path=run.evaluator.cpp.compiler,
            compiler_flags=run.evaluator.cpp.extra_flags,
            compiled_output_chunk_size=run.evaluator.output_chunk_size,
            output_chunk_strategy="uniform",
            output_chunk_autotune_batch_size=run.evaluator.batch_size,
            compiled_chunk_compile_workers=1,
        )

    def _validate_generated_artifact(
        self,
        output: Path,
        process_ids: Sequence[str],
        *,
        validation_points: Mapping[str, Sequence[ValidationPointRecord]],
        expected_api_bundle_path: str | None,
        progress: PhaseHandle | None = None,
    ) -> None:
        import cmath

        from pyamplicol.api import Runtime
        from pyamplicol.artifacts import load_manifest

        manifest = load_manifest(output)
        expected_ids = set(process_ids)
        actual_ids = {str(process["id"]) for process in manifest.processes}
        if not expected_ids.issubset(actual_ids):
            raise GenerationError("generated artifact omitted concrete processes")
        by_path = {record.path: record for record in manifest.payloads}
        process_total = len(process_ids)
        for process_index, process_id in enumerate(process_ids, start=1):
            if progress is not None:
                progress.update(
                    process_index - 1,
                    total=process_total,
                    message=f"loading {process_id}",
                    details={
                        "process": process_id,
                        "step": "runtime validation",
                        "sample_index": 0,
                    },
                )
            prefix = f"processes/{process_id}"
            required = {
                f"{prefix}/physics.json",
                f"{prefix}/execution.json",
                f"{prefix}/validation-momenta.json",
            }
            if not required.issubset(by_path):
                raise GenerationError(
                    f"artifact payload set is incomplete for {process_id!r}"
                )
            runtime = Runtime.load(output, process=process_id)
            if runtime.physics.process_id != process_id:
                raise GenerationError(
                    f"Rusticol selected the wrong process for {process_id!r}"
                )
            validation = self._generation_config.validation
            samples = tuple(
                point.four_vectors
                for point in validation_points.get(process_id, ())
                if point.available
            )
            if validation.enabled and samples:
                total = runtime.evaluate(samples)
                resolved_total = runtime.evaluate_resolved(samples).total()
                if len(total) != len(samples) or len(resolved_total) != len(samples):
                    raise GenerationError(
                        f"Rusticol returned an invalid validation shape for "
                        f"{process_id!r}"
                    )
                for sample_index, (summed, resolved) in enumerate(
                    zip(total, resolved_total, strict=True),
                    start=1,
                ):
                    if progress is not None:
                        progress.update(
                            process_index - 1,
                            total=process_total,
                            message=f"{process_id}: sample {sample_index}",
                            details={
                                "process": process_id,
                                "step": "numerical validation",
                                "sample_index": sample_index,
                                "sample_total": len(samples),
                            },
                        )
                    difference = abs(complex(summed) - complex(resolved))
                    if not cmath.isclose(
                        complex(summed),
                        complex(resolved),
                        rel_tol=validation.relative_tolerance,
                        abs_tol=validation.absolute_tolerance,
                    ):
                        raise GenerationError(
                            "resolved Rusticol validation does not reduce to the "
                            f"total for {process_id!r} sample {sample_index} "
                            f"(absolute difference {difference:.3e})"
                        )
            if progress is not None:
                progress.update(
                    process_index,
                    total=process_total,
                    message=process_id,
                    details={
                        "process": process_id,
                        "step": "validated",
                        "samples": len(samples),
                    },
                )
        actual_api_bundle_path = manifest.runtime.get("api_bundle_path")
        if actual_api_bundle_path != expected_api_bundle_path:
            raise GenerationError("artifact API bundle outcome is inconsistent")

    @property
    def _run_config(self) -> RunConfig | None:
        return self._config if isinstance(self._config, RunConfig) else None

    @property
    def _eager_execution_enabled(self) -> bool:
        run = self._run_config
        return run is not None and str(run.evaluator.execution_mode) == "eager"

    @property
    def _all_flow_union_enabled(self) -> bool:
        run = self._run_config
        if run is None:
            return False
        return str(getattr(run.color, "lc_flow_layout", "topology-replay")) == (
            "all-flow-union"
        )

    @property
    def _generation_config(self) -> GenerationConfig:
        config = self._config
        return config.generation if isinstance(config, RunConfig) else config

    @property
    def _builtin_process_options(self):
        from ..models.builtin.process_types import BuiltinProcessOptions

        run = self._run_config
        if run is None:
            return BuiltinProcessOptions()
        max_lines = run.process.max_quark_lines
        return BuiltinProcessOptions(
            flavour_scheme=run.process.flavor_scheme,
            include_3qqbar=max_lines is None or max_lines >= 3,
        )

    @property
    def _max_quark_pairs(self) -> int | None:
        run = self._run_config
        return None if run is None else run.process.max_quark_lines

    @property
    def _color_accuracy(self) -> str:
        run = self._run_config
        return "lc" if run is None else run.color.accuracy

    @property
    def _coupling_order_limits(self) -> dict[str, int]:
        run = self._run_config
        if run is None:
            return {}
        return {
            str(name).upper(): int(value)
            for name, value in run.process.max_coupling_orders.items()
        }

    @property
    def _process_selection(self) -> _ProcessSelection:
        if self._process_selection_override is not None:
            return self._process_selection_override
        run = self._run_config
        if run is None:
            return _ProcessSelection()
        process = run.process
        return _ProcessSelection(
            max_color_sectors=process.max_color_sectors,
            reference_color_order=(
                tuple(int(label) for label in process.reference_color_order)
                if process.reference_color_order
                else None
            ),
            selected_color_sector_ids=(
                frozenset(
                    int(sector_id) for sector_id in process.selected_color_sector_ids
                )
                if process.selected_color_sector_ids
                else None
            ),
            selected_source_helicities=(
                {
                    int(label): int(helicity)
                    for label, helicity in process.selected_source_helicities.items()
                }
                if process.selected_source_helicities
                else None
            ),
        )

    def _configured_model_source(self) -> ModelSource:
        run = self._run_config
        if run is None:
            return ModelSource.built_in_sm()
        return ModelSource.from_config(run.model)

    def _resolve_model_for_plan(
        self,
        source: ModelSource | CompiledModel,
    ) -> _ResolvedModel:
        if isinstance(source, CompiledModel):
            compiled = self._private_compiled_model(source)
            is_builtin = self._is_builtin_compiled_model(compiled)
            return _ResolvedModel(
                self._source_for_compiled_model(compiled),
                _builtin_sm_model() if is_builtin else None,
                compiled,
                use_compiled_process_catalog=not is_builtin,
            )
        if source.kind == "built-in-sm":
            if not self._eager_execution_enabled:
                return _ResolvedModel(source, _builtin_sm_model())
            source = self._packaged_builtin_eager_source()
        if source.path is None:
            raise ModelError("external model source has no path")

        from ..models.loading import load_cached_model_source, load_compiled_model

        try:
            if source.kind in {"compiled", "prepared"}:
                compiled = load_compiled_model(source.path)
            else:
                run = self._run_config
                use_cache = True if run is None else run.model.cache
                if not use_cache:
                    raise ModelError(
                        "dry-run does not compile external model sources and model "
                        "caching is disabled; compile the ModelSource explicitly and "
                        "pass its CompiledModel"
                    )
                cache_dir = None if run is None else run.model.cache_dir
                compiled = load_cached_model_source(
                    source.path,
                    restriction=(
                        "default"
                        if source.restriction is None
                        else str(source.restriction)
                    ),
                    simplify=source.simplify,
                    cache_dir=cache_dir,
                )
                if compiled is None:
                    raise ModelError(
                        "dry-run does not compile external model sources; call "
                        "ModelSource.compile() first and pass the returned "
                        "CompiledModel, or populate the configured model cache"
                    )
        except ModelError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc
        is_builtin = self._is_builtin_compiled_model(compiled)
        return _ResolvedModel(
            source,
            _builtin_sm_model() if is_builtin else None,
            compiled,
            use_compiled_process_catalog=not is_builtin,
        )

    @staticmethod
    def _private_compiled_model(compiled: CompiledModel) -> _CompiledModelPayload:
        payload = _compiled_model_payload(compiled)
        if not isinstance(payload, _CompiledModelPayload):
            raise ModelError("compiled model handle has an invalid private payload")
        return payload

    @staticmethod
    def _source_for_compiled_model(compiled: _CompiledModelPayload) -> ModelSource:
        if compiled.prepared_bundle is not None:
            return ModelSource(kind="prepared", path=compiled.prepared_bundle.path)
        return ModelSource(kind="compiled", path=compiled._serialized_path)

    @staticmethod
    def _is_builtin_compiled_model(compiled: _CompiledModelPayload) -> bool:
        return compiled.source.get("kind") == "built-in-sm"

    def _resolve_model(
        self,
        source: ModelSource | CompiledModel,
    ) -> _ResolvedModel:
        if isinstance(source, CompiledModel):
            compiled = self._private_compiled_model(source)
            if self._is_builtin_compiled_model(compiled):
                return _ResolvedModel(
                    self._source_for_compiled_model(compiled),
                    _builtin_sm_model(),
                    compiled,
                    use_compiled_process_catalog=False,
                )

            from ..models.external import CompiledUFOModel

            return _ResolvedModel(
                self._source_for_compiled_model(compiled),
                CompiledUFOModel(compiled),
                compiled,
            )
        if source.kind == "built-in-sm":
            if not self._eager_execution_enabled:
                return _ResolvedModel(source, _builtin_sm_model())
            source = self._packaged_builtin_eager_source()
        if source.path is None:
            raise ModelError("external model source has no path")
        from ..models.external import CompiledUFOModel
        from ..models.loading import compile_model_source, load_compiled_model

        try:
            if source.kind in {"compiled", "prepared"}:
                compiled = load_compiled_model(source.path)
            else:
                run = self._run_config
                use_cache = True if run is None else run.model.cache
                cache_dir = None if run is None else run.model.cache_dir
                compiled = compile_model_source(
                    source.path,
                    restriction=(
                        "default"
                        if source.restriction is None
                        else str(source.restriction)
                    ),
                    simplify=source.simplify,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ModelError(str(exc)) from exc
        if self._is_builtin_compiled_model(compiled):
            return _ResolvedModel(
                source,
                _builtin_sm_model(),
                compiled,
                use_compiled_process_catalog=False,
            )
        return _ResolvedModel(source, CompiledUFOModel(compiled), compiled)

    @staticmethod
    def _packaged_builtin_eager_source() -> ModelSource:
        from ..assets.prepared_models import (
            PackagedPreparedModelError,
            materialize_packaged_prepared_model,
        )

        try:
            path = materialize_packaged_prepared_model()
        except (OSError, PackagedPreparedModelError, RuntimeError, ValueError) as exc:
            raise ModelError(
                "the wheel-owned built-in-SM eager model is unavailable or stale: "
                f"{exc}"
            ) from exc
        return ModelSource.from_path(path)

    def _require_eager_kernel_pack(self, resolved: _ResolvedModel) -> None:
        run = self._run_config
        if run is None or str(run.evaluator.execution_mode) != "eager":
            return
        compiled = resolved.compiled
        if compiled is None or compiled.prepared_bundle is None:
            source = resolved.source.path or resolved.source.kind
            backend = str(run.evaluator.backend)
            raise GenerationError(
                "eager generation requires a prepared model kernel pack; run: "
                f"pyamplicol model compile {source} MODEL.pyamplicol-model "
                f"--backend {backend}"
            )
        pack = compiled.prepared_bundle.kernel_pack
        try:
            validate_prepared_target(
                pack.target,
                backend=pack.backend,
                symjit_application_abi=cast(
                    str | None,
                    pack.dependency_abis.get("symjit_application"),
                ),
            )
        except PreparedTargetError as exc:
            source = resolved.source.path or resolved.source.kind
            raise GenerationError(
                f"eager prepared model is incompatible with this host: {exc}; "
                "prepare a matching bundle with: "
                f"pyamplicol model compile {source} MODEL.pyamplicol-model "
                f"--backend {pack.backend}"
            ) from exc

    def _index_eager_kernel_pack(self, resolved: _ResolvedModel) -> _ResolvedModel:
        if not self._eager_execution_enabled:
            return resolved
        compiled = resolved.compiled
        bundle = None if compiled is None else compiled.prepared_bundle
        if bundle is None:  # pragma: no cover - guarded by _require_eager_kernel_pack
            raise GenerationError("eager generation has no prepared kernel pack")
        try:
            index = PreparedCatalogEagerKernelIndex.from_manifest(
                bundle.kernel_pack.resolver_manifest
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GenerationError(
                f"prepared model has an invalid eager kernel resolver: {exc}"
            ) from exc
        return replace(resolved, eager_kernel_index=index)

    def _apply_prepared_kernel_pack_policy(self, resolved: _ResolvedModel) -> None:
        if not self._eager_execution_enabled:
            return
        compiled = resolved.compiled
        bundle = None if compiled is None else compiled.prepared_bundle
        if bundle is None:  # pragma: no cover - guarded by _require_eager_kernel_pack
            raise GenerationError("eager generation has no prepared kernel pack")
        resolution = self._resource_config
        if not isinstance(resolution, ConfigResolution):
            resolution = resolve_config(config_to_dict(self._config))
        effective_values = config_to_dict(resolution.effective)
        pack_values = _prepared_pack_effective_values(bundle.kernel_pack)
        replacement_paths = frozenset(pack_values)
        inherited = [
            ClampRequest(item.path, item.effective, item.reason)
            for item in resolution.clamps
            if item.path not in replacement_paths
        ]
        prepared_clamps: list[ClampRequest] = []
        for path, value in pack_values.items():
            if _nested_config_value(effective_values, path) == value:
                continue
            prepared_clamps.append(
                ClampRequest(
                    path,
                    value,
                    "prepared eager kernel pack is authoritative for backend "
                    "and code-shaping optimization settings",
                )
            )
        if not prepared_clamps:
            return
        updated = resolve_config(
            config_to_dict(resolution.requested),
            clamps=(*inherited, *prepared_clamps),
        )
        self._resource_config = updated
        self._configuration = _GenerationConfigProvenance.from_config(updated)
        self._config = updated.effective
        if not self._prepared_pack_warning_emitted:
            rendered = ", ".join(clamp.path for clamp in prepared_clamps)
            _LOGGER.warning(
                "eager generation uses prepared model settings for: %s",
                rendered,
            )
            self._prepared_pack_warning_emitted = True

    def _expand_request(
        self,
        request: ProcessRequest,
        resolved_model: _ResolvedModel,
    ) -> tuple[CanonicalProcessIR, ...]:
        if (
            resolved_model.compiled is None
            or not resolved_model.use_compiled_process_catalog
        ):
            from ..models.builtin.process_ir import build_process_ir
            from ..models.builtin.process_selection import (
                enumerate_generic_process_set,
            )

            enumeration = enumerate_generic_process_set(
                request.expression,
                self._builtin_process_options,
                max_quark_pairs=self._max_quark_pairs,
            )
            return tuple(
                build_process_ir(
                    entry.process,
                    color_accuracy=self._color_accuracy,
                    options=self._builtin_process_options,
                )
                for entry in enumeration.entries
            )

        from ..processes.model import (
            ModelParticleCatalog,
            build_model_process_ir,
            expand_model_processes,
        )

        catalog = ModelParticleCatalog(
            resolved_model.compiled.ir.name,
            resolved_model.compiled.ir.particles,
            resolved_model.compiled.ir.parameters,
        )
        run = self._run_config
        multiparticles = None if run is None else run.process.multiparticles
        candidates = tuple(
            build_model_process_ir(
                process,
                resolved_model.compiled.ir,
                color_accuracy=self._color_accuracy,
            )
            for process in expand_model_processes(
                request.expression,
                catalog,
                multiparticles=multiparticles,
            )
        )
        selected, rejected = _select_color_ready_processes(
            candidates,
            color_accuracy=self._color_accuracy,
        )
        if not selected:
            detail = "; ".join(rejected) or "no concrete processes"
            raise GenerationError(
                f"process request {request.expression!r} has no usable color plan: "
                f"{detail}"
            )
        return selected

    def _plan_concrete_process(
        self,
        process: CanonicalProcessIR,
        *,
        model: Model | None,
    ) -> dict[str, object]:
        selection = self._process_selection
        self._validate_lc_flow_layout(selection)
        if (
            selection.selected_color_sector_ids is not None
            and self._color_accuracy != "lc"
        ):
            raise GenerationError(
                "process.selected_color_sector_ids is available only for LC generation"
            )
        color_plan = build_color_plan(
            process,
            color_accuracy=self._color_accuracy,
            max_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            fold_trace_reflections=(
                model is not None
                and model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
        color_plan, missing_sector_ids = _restrict_color_plan(
            color_plan,
            selection.selected_color_sector_ids,
        )
        if missing_sector_ids:
            raise GenerationError(
                f"process {process.process!r} did not materialize requested LC "
                "colour sector ids: "
                + ", ".join(str(sector_id) for sector_id in missing_sector_ids)
            )
        if not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise GenerationError(
                f"process {process.process!r} has no usable color plan: {detail}"
            )
        return {
            "key": process.key,
            "process": process.process,
            "external_particle_count": len(process.legs),
            "color_sector_count": color_plan.sector_count,
            "color_coverage": (
                "selected"
                if color_plan.truncated
                or selection.selected_color_sector_ids is not None
                else "complete"
            ),
            "helicity_coverage": (
                "selected"
                if selection.selected_source_helicities is not None
                else "complete"
            ),
            "color_diagnostics": tuple(color_plan.diagnostics),
            "coupling_order_limits": self._coupling_order_limits,
            "dag_compilation_deferred": True,
        }

    def _compile_concrete_process(
        self,
        process: CanonicalProcessIR,
        model: Model,
        *,
        progress_callback: Callable[[Mapping[str, str | int]], None] | None = None,
    ) -> tuple[GenericDAG, dict[str, object]]:
        selection = self._process_selection
        self._validate_lc_flow_layout(selection)
        if (
            selection.selected_color_sector_ids is not None
            and self._color_accuracy != "lc"
        ):
            raise GenerationError(
                "process.selected_color_sector_ids is available only for LC generation"
            )
        complete_color_plan = build_color_plan(
            process,
            color_accuracy=self._color_accuracy,
            max_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            fold_trace_reflections=(
                model.lc_trace_reflection_equivalence_is_proven(process)
            ),
        )
        color_plan, missing_sector_ids = _restrict_color_plan(
            complete_color_plan,
            selection.selected_color_sector_ids,
        )
        if missing_sector_ids:
            raise GenerationError(
                f"process {process.process!r} did not materialize requested LC "
                "colour sector ids: "
                + ", ".join(str(sector_id) for sector_id in missing_sector_ids)
            )
        if not color_plan.sectors:
            detail = "; ".join(color_plan.diagnostics) or "no color sectors"
            raise GenerationError(
                f"process {process.process!r} has no usable color plan: {detail}"
            )
        run = self._run_config
        limits = self._coupling_order_limits
        if run is not None and run.process.coupling_order_policy == "minimal":
            inferred = infer_minimal_coupling_order_limits(
                process,
                model=model,
                max_coupling_orders=limits or None,
            )
            limits = inferred or limits
        replay_plan = None
        materialized_sector_ids = selection.selected_color_sector_ids
        if (
            self._color_accuracy == "lc"
            and not self._all_flow_union_enabled
            and selection.selected_color_sector_ids is None
            and selection.selected_source_helicities is None
        ):
            replay_plan = build_lc_topology_replay_plan(
                complete_color_plan,
                model,
            )
            if replay_plan is not None and replay_plan.optimized:
                materialized_sector_ids = frozenset(replay_plan.materialized_sector_ids)
        dag = compile_generic_dag(
            process,
            model=model,
            color_plan=complete_color_plan,
            max_color_sectors=selection.max_color_sectors,
            reference_color_order=selection.reference_color_order,
            selected_color_sector_ids=materialized_sector_ids,
            max_coupling_orders=limits or None,
            max_quark_pairs=self._max_quark_pairs,
            selected_source_helicities=selection.selected_source_helicities,
            online_evaluation_reuse=(
                self._eager_execution_enabled or self._all_flow_union_enabled
            ),
            backward_live_planning=self._eager_execution_enabled,
            progress_callback=progress_callback,
        )
        if replay_plan is not None and replay_plan.optimized:
            root_sector_ids = {
                int(root.color_sector_id)
                if root.color_sector_id is not None
                else int(dag.currents[root.left_id].index.color_state.sector_id)
                for root in dag.amplitude_roots
            }
            missing_representatives = {
                int(partition.materialized_sector_id)
                for partition in replay_plan.partitions
                if int(partition.materialized_sector_id) not in root_sector_ids
            }
            if missing_representatives:
                replay_plan = None
                dag = compile_generic_dag(
                    process,
                    model=model,
                    color_plan=complete_color_plan,
                    max_color_sectors=selection.max_color_sectors,
                    reference_color_order=selection.reference_color_order,
                    selected_color_sector_ids=None,
                    max_coupling_orders=limits or None,
                    max_quark_pairs=self._max_quark_pairs,
                    selected_source_helicities=selection.selected_source_helicities,
                    online_evaluation_reuse=self._eager_execution_enabled,
                    backward_live_planning=self._eager_execution_enabled,
                    progress_callback=progress_callback,
                )
        if replay_plan is not None and replay_plan.optimized:
            dag = replace(
                dag,
                color_plan=complete_color_plan,
                color_coverage="complete",
                selected_color_sector_ids=(),
                lc_topology_replay=replay_plan,
            )
        if dag.truncated:
            raise GenerationError(
                f"process {process.process!r} DAG was unexpectedly truncated"
            )
        if not dag.has_amplitudes:
            raise GenerationError(
                f"process {process.process!r} has no model-supported amplitudes"
            )
        return (
            dag,
            {
                "key": process.key,
                "process": process.process,
                "color_sector_count": dag.color_plan.sector_count,
                **(
                    {}
                    if self._color_accuracy != "lc"
                    else {
                        "lc_flow_layout": (
                            "all-flow-union"
                            if self._all_flow_union_enabled
                            else "topology-replay"
                        )
                    }
                ),
                **(
                    {}
                    if dag.lc_topology_replay is None
                    else {
                        "materialized_color_sector_count": len(
                            dag.lc_topology_replay.materialized_sector_ids
                        ),
                        "replayed_color_sector_count": (
                            dag.lc_topology_replay.replayed_sector_count
                        ),
                        "residual_color_sector_count": len(
                            dag.lc_topology_replay.residual_sector_ids
                        ),
                    }
                ),
                "color_coverage": dag.color_coverage,
                "helicity_coverage": dag.helicity_coverage,
                "source_count": len(dag.sources),
                "current_count": len(dag.currents),
                "interaction_count": len(dag.interactions),
                "interaction_evaluation_count": dag.interaction_evaluation_count,
                "amplitude_root_count": len(dag.amplitude_roots),
                "coupling_order_limits": limits,
            },
        )

    def _validate_lc_flow_layout(self, selection: _ProcessSelection) -> None:
        if not self._all_flow_union_enabled:
            return
        if self._color_accuracy != "lc":
            raise GenerationError(
                "color.lc_flow_layout='all-flow-union' is available only for "
                "LC generation"
            )
        incompatible: list[str] = []
        if selection.max_color_sectors is not None:
            incompatible.append("process.max_color_sectors")
        if selection.selected_color_sector_ids is not None:
            incompatible.append("process.selected_color_sector_ids")
        if selection.selected_source_helicities is not None:
            incompatible.append("process.selected_source_helicities")
        if incompatible:
            raise GenerationError(
                "color.lc_flow_layout='all-flow-union' requires complete runtime "
                "flow and helicity coverage; remove " + ", ".join(incompatible)
            )

    def _compile_complete_lc_helicity_selector_union(
        self,
        process: _DagProcess,
        model: Model,
    ) -> GenericDAG | None:
        """Build a bounded all-colour lane for runtime helicity selection.

        The ordinary complete LC artifact keeps topology replay for fast
        runtime flow selection and helicity sums.  Replaying that compact DAG
        once per physical flow is inefficient for the complementary
        all-flows/selected-helicity workload, so bounded color spaces receive
        an auxiliary complete-color recurrence quotient.
        """

        parent = process.dag
        if (
            self._eager_execution_enabled
            or parent.process.color_accuracy != "lc"
            or parent.color_coverage != "complete"
            or parent.selected_color_sector_ids
            or parent.selected_source_helicities
            or parent.helicity_coverage != "complete"
            or parent.lc_topology_replay is None
            or len(parent.color_plan.sectors) > _MAX_FUSED_LC_HELICITY_SELECTOR_SECTORS
        ):
            return None

        limits_payload = process.coverage.get("coupling_order_limits", {})
        if not isinstance(limits_payload, Mapping):
            raise GenerationError("process coupling-order coverage is not a mapping")
        limits = {str(name): int(value) for name, value in limits_payload.items()}
        selection = self._process_selection
        union = compile_generic_dag(
            process.expanded.process_ir,
            model=model,
            color_plan=parent.color_plan,
            max_color_sectors=None,
            reference_color_order=selection.reference_color_order,
            selected_color_sector_ids=None,
            max_coupling_orders=limits or None,
            max_quark_pairs=self._max_quark_pairs,
            selected_source_helicities=None,
            online_evaluation_reuse=False,
            backward_live_planning=False,
            progress_callback=None,
        )
        if union.truncated or not union.has_amplitudes:
            raise GenerationError(
                "complete-color helicity selector lane did not produce a "
                "complete amplitude DAG"
            )
        parity_preweighted = any(
            float(root.helicity_weight) > 1.0 for root in union.amplitude_roots
        )
        reduced = (
            prune_dag_to_amplitude_roots(union)
            if parity_preweighted
            else prune_global_helicity_flip_equivalent_roots(union, model)
        )
        recurrence = build_helicity_recurrence_plan(reduced, model)
        if recurrence is None:
            return None
        materialization = materialize_helicity_recurrence(
            reduced,
            recurrence,
        )
        return replace(
            materialization.dag,
            lc_topology_replay=None,
            helicity_recurrence=recurrence,
            helicity_materialization=materialization,
        )


def _runtime_stage_count(schema: RuntimeExpressionSchema) -> int:
    stages = schema.to_mapping().get("stages")
    if isinstance(stages, str | bytes) or not isinstance(stages, Sequence):
        raise GenerationError("runtime expression schema stages must be a sequence")
    return len(stages)


def _helicity_selector_lane_stage_count(
    lane: _HelicitySelectorLane,
) -> int:
    return _runtime_stage_count(lane.runtime_schema) + sum(
        _helicity_selector_lane_stage_count(child) for child in lane.child_lanes
    )


def _helicity_selector_lane_compile_task_count(
    lane: _HelicitySelectorLane,
) -> int:
    return (
        _runtime_stage_count(lane.runtime_schema)
        + 1
        + sum(
            _helicity_selector_lane_compile_task_count(child)
            for child in lane.child_lanes
        )
    )


def _runtime_schema_dag_summary(
    schema: RuntimeExpressionSchema,
) -> dict[str, object]:
    payload = schema.to_mapping()
    current_storage = cast("Mapping[str, object]", payload["current_storage"])
    source_fill = cast("Mapping[str, object]", payload["source_fill"])
    amplitude_stage = cast("Mapping[str, object]", payload["amplitude_stage"])
    stages = cast("Sequence[Mapping[str, object]]", payload["stages"])
    current_slots = cast("Sequence[object]", current_storage["current_slots"])
    return {
        "current_count": len(current_slots),
        "source_count": int(source_fill["source_count"]),
        "interaction_count": sum(int(stage["interaction_count"]) for stage in stages),
        "amplitude_root_count": int(amplitude_stage["output_count"]),
        "truncated": False,
    }


def _select_color_ready_processes(
    processes: Sequence[CanonicalProcessIR],
    *,
    color_accuracy: str,
) -> tuple[tuple[CanonicalProcessIR, ...], tuple[str, ...]]:
    """Drop structurally impossible children of an external multiparticle request."""

    selected: list[CanonicalProcessIR] = []
    rejected: list[str] = []
    for process in processes:
        color_plan = build_color_plan(
            process,
            color_accuracy=color_accuracy,
        )
        if color_plan.ready_for_requested_colour:
            selected.append(process)
            continue
        detail = "; ".join(color_plan.diagnostics) or "no color sectors"
        rejected.append(f"{process.process}: {detail}")
    return tuple(selected), tuple(rejected)


def _expanded_name(base: str, index: int, count: int) -> str:
    return base if count == 1 else f"{base}_{index}"


def _permuted_process_identity(
    process: CanonicalProcessIR,
    representative_to_alias: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    """Return public process metadata in a final-state alias's external order."""

    legs = process.legs
    if len(representative_to_alias) != len(legs) or sorted(
        representative_to_alias
    ) != list(range(len(legs))):
        raise GenerationError("alias particle permutation is not complete")
    alias_legs: list[ProcessLegIR | None] = [None] * len(legs)
    for representative_index, alias_index in enumerate(representative_to_alias):
        alias_legs[alias_index] = legs[representative_index]
    if any(leg is None for leg in alias_legs):
        raise GenerationError("alias particle permutation is incomplete")
    ordered_legs = tuple(cast(ProcessLegIR, leg) for leg in alias_legs)
    if any(leg.pdg is None for leg in ordered_legs):
        raise GenerationError("process alias has an unresolved external particle")
    initial = " ".join(leg.particle for leg in ordered_legs if leg.is_initial)
    final = " ".join(leg.particle for leg in ordered_legs if leg.is_final)
    if not initial or not final:
        raise GenerationError("process alias must retain initial and final states")
    return (
        f"{initial} > {final}",
        tuple(int(leg.pdg) for leg in ordered_legs if leg.pdg is not None),
    )


def _cpp_optimization_level(value: str) -> int:
    normalized = value.strip().upper()
    if normalized.startswith("-"):
        normalized = normalized[1:]
    if normalized not in {"O0", "O1", "O2", "O3"}:
        raise GenerationError("evaluator.cpp.optimization must be O0, O1, O2, or O3")
    return int(normalized[1])


def create_generator_backend(
    config: GenerationConfig | RunConfig | ConfigResolution | None,
    progress: ProgressSink | None,
    *,
    api_bundle_hook: ApiBundleHook | None = None,
) -> GenerationBackend:
    return GenerationBackend(config, progress, api_bundle_hook=api_bundle_hook)


__all__ = ["GenerationBackend", "create_generator_backend"]
