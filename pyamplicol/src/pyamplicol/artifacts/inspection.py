# SPDX-License-Identifier: 0BSD
"""Read-only summaries of generated process artifacts."""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pyamplicol.api.errors import ArtifactError

from .manifest import ArtifactManifest, load_manifest
from .security import confined_path, normalize_relative_path

_EAGER_RUNTIME_KIND = "pyamplicol-runtime-eager-execution"
_MISSING_U32 = (1 << 32) - 1
_EAGER_INVOCATION = struct.Struct("<IIIIIIQQ")
_EAGER_FINALIZATION = struct.Struct("<IIIII")
_EAGER_CLOSURE = struct.Struct("<IIIIIdd")
_EAGER_SELECTOR_DOMAINS_ABI = "pyamplicol-eager-selector-domains-v1"
_EAGER_SELECTOR_DOMAIN = struct.Struct("<QQ")
_EAGER_SELECTOR_GROUP = struct.Struct("<I")
_EAGER_SELECTOR_DOMAIN_ID = struct.Struct("<I")
_COMPILED_PROFILE_PHASES = (
    "source-fill",
    "momentum-setup",
    "stage-input-pack",
    "stage-evaluator-call",
    "output-assign",
    "amplitude-input-pack",
    "amplitude-evaluator-call",
    "reduction",
)
_EAGER_PROFILE_PHASES = (
    "source-fill",
    "momentum-setup",
    "eager-execution-aggregate",
)


@dataclass(frozen=True, slots=True)
class ArtifactAliasInspection:
    id: str
    expression: str
    representative_id: str
    external_pdgs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArtifactProcessInspection:
    id: str
    expression: str
    color_accuracy: str
    external_pdgs: tuple[int, ...]
    default: bool
    physical_helicities: int
    computed_helicities: int
    physical_color_components: int
    computed_color_components: int
    helicity_coverage: str
    color_coverage: str
    aliases: tuple[ArtifactAliasInspection, ...]
    execution_mode: str = "compiled"
    prepared_backend: str | None = None
    prepared_kernel_count: int | None = None
    referenced_kernel_count: int | None = None
    invocation_count: int | None = None
    attachment_count: int | None = None
    evaluation_alias_count: int | None = None
    maximum_fanout: int | None = None
    finalization_count: int | None = None
    closure_count: int | None = None
    selector_closure_available: bool = False
    selector_domain_count: int | None = None
    selector_domain_membership_count: int | None = None
    requested_point_tile_size: int | None = None
    effective_point_tile_size: int | None = None
    workspace_limit_bytes: int | None = None
    workspace_bytes: int | None = None
    native_profile_phases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArtifactDependencyInspection:
    name: str
    version: str
    license: str
    source: str


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    kind: str
    path: Path
    artifact_kind: str
    artifact_id: str
    created_utc: str
    producer_version: str
    target: str
    cpu_features: tuple[str, ...]
    model_name: str
    model_source: str
    model_restriction: str | None
    default_process_id: str | None
    runtime_engine: str
    runtime_version: str
    runtime_capabilities: tuple[str, ...]
    payload_count: int
    payload_size_bytes: int
    integrity: str
    processes: tuple[ArtifactProcessInspection, ...]
    dependencies: tuple[ArtifactDependencyInspection, ...]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} must be an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} must be an array")
    return value


def _computed(value: object, context: str) -> bool:
    record = _mapping(value, context)
    if record.get("kind") == "contracted-color":
        return True
    computed = record.get("computed")
    if not isinstance(computed, bool):
        raise ArtifactError(f"{context}.computed must be a boolean")
    return computed


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactError(f"{context} must be an integer >= {minimum}")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{context} must be a non-empty string")
    return value


def _json_mapping(path: Path, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(f"cannot read {context}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid {context}: {exc}") from exc
    return _mapping(value, context)


def _artifact_path(
    manifest: ArtifactManifest,
    relative: str | Path,
    context: str,
) -> Path:
    try:
        normalized = normalize_relative_path(Path(relative).as_posix())
    except ArtifactError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{context} is not a valid artifact path") from exc
    return confined_path(manifest.root, normalized)


def _execution_paths(manifest: ArtifactManifest) -> Mapping[str, Path]:
    evaluator_relative = _string(
        manifest.runtime.get("evaluator_manifest_path"),
        "runtime.evaluator_manifest_path",
    )
    evaluator_path = _artifact_path(
        manifest,
        evaluator_relative,
        "runtime.evaluator_manifest_path",
    )
    evaluator_set = _json_mapping(evaluator_path, "runtime evaluator manifest")
    records = _sequence(
        evaluator_set.get("processes"),
        "runtime evaluator manifest.processes",
    )
    base = Path(evaluator_relative).parent
    result: dict[str, Path] = {}
    for index, raw in enumerate(records):
        record = _mapping(raw, f"runtime evaluator manifest.processes[{index}]")
        process_id = _string(
            record.get("process_id"),
            f"runtime evaluator manifest.processes[{index}].process_id",
        )
        manifest_path = _string(
            record.get("manifest_path"),
            f"runtime evaluator manifest.processes[{index}].manifest_path",
        )
        if process_id in result:
            raise ArtifactError(
                f"runtime evaluator manifest repeats process {process_id!r}"
            )
        result[process_id] = _artifact_path(
            manifest,
            base / manifest_path,
            f"runtime evaluator manifest for {process_id!r}",
        )
    expected = {str(process["id"]) for process in manifest.processes}
    if set(result) != expected:
        raise ArtifactError(
            "runtime evaluator manifest process IDs do not match artifact processes"
        )
    return result


@dataclass(frozen=True, slots=True)
class _ExecutionInspection:
    execution_mode: str
    prepared_backend: str | None = None
    prepared_kernel_count: int | None = None
    referenced_kernel_count: int | None = None
    invocation_count: int | None = None
    attachment_count: int | None = None
    evaluation_alias_count: int | None = None
    maximum_fanout: int | None = None
    finalization_count: int | None = None
    closure_count: int | None = None
    selector_closure_available: bool = False
    selector_domain_count: int | None = None
    selector_domain_membership_count: int | None = None
    requested_point_tile_size: int | None = None
    effective_point_tile_size: int | None = None
    workspace_limit_bytes: int | None = None
    workspace_bytes: int | None = None
    native_profile_phases: tuple[str, ...] = ()


def _table_record(
    value: object,
    context: str,
) -> tuple[str, int, int]:
    record = _mapping(value, context)
    return (
        _string(record.get("path"), f"{context}.path"),
        _integer(record.get("count"), f"{context}.count"),
        _integer(record.get("row_size"), f"{context}.row_size", minimum=1),
    )


def _table_rows(
    manifest: ArtifactManifest,
    execution_root: Path,
    table: object,
    layout: struct.Struct,
    context: str,
) -> tuple[tuple[int | float, ...], ...]:
    relative, count, row_size = _table_record(table, context)
    if row_size != layout.size:
        raise ArtifactError(
            f"{context}.row_size is {row_size}, expected {layout.size}"
        )
    path = _artifact_path(
        manifest,
        execution_root.relative_to(manifest.root) / relative,
        f"{context}.path",
    )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"cannot read {context} payload: {exc}") from exc
    expected_size = count * row_size
    if len(payload) != expected_size:
        raise ArtifactError(
            f"{context} declares {count} rows but has {len(payload)} bytes"
        )
    return cast(
        tuple[tuple[int | float, ...], ...],
        tuple(tuple(row) for row in layout.iter_unpack(payload)),
    )


def _optional_positive_integer(
    value: object,
    context: str,
) -> int | None:
    if value is None:
        return None
    return _integer(value, context, minimum=1)


def _profile_phases(execution: Mapping[str, object]) -> tuple[str, ...]:
    raw = execution.get("native_profile_phases")
    if raw is None:
        return _EAGER_PROFILE_PHASES
    phases = tuple(
        _string(value, f"execution.native_profile_phases[{index}]")
        for index, value in enumerate(
            _sequence(raw, "execution.native_profile_phases")
        )
    )
    if len(phases) != len(set(phases)):
        raise ArtifactError("execution.native_profile_phases contains duplicates")
    return phases


def _selector_domain_ids(
    manifest: ArtifactManifest,
    execution_root: Path,
    table: object,
    *,
    expected_count: int,
    domain_count: int,
    context: str,
) -> tuple[int, ...]:
    rows = _table_rows(
        manifest,
        execution_root,
        table,
        _EAGER_SELECTOR_DOMAIN_ID,
        context,
    )
    if len(rows) != expected_count:
        raise ArtifactError(
            f"{context} has {len(rows)} rows, expected {expected_count}"
        )
    result = tuple(int(row[0]) for row in rows)
    invalid = next((value for value in result if value >= domain_count), None)
    if invalid is not None:
        raise ArtifactError(f"{context} references unknown domain {invalid}")
    return result


def _selector_domain_inspection(
    manifest: ArtifactManifest,
    execution: Mapping[str, object],
    execution_root: Path,
    selector_closures: object,
    *,
    stages: Sequence[
        tuple[
            Mapping[str, object],
            tuple[tuple[int | float, ...], ...],
            int,
            tuple[tuple[int | float, ...], ...],
        ]
    ],
    closures: tuple[tuple[int | float, ...], ...],
) -> tuple[int, int]:
    selector = _mapping(
        selector_closures,
        "eager execution.plan.selector_closures",
    )
    abi = _string(
        selector.get("abi"),
        "eager execution.plan.selector_closures.abi",
    )
    if abi != _EAGER_SELECTOR_DOMAINS_ABI:
        raise ArtifactError(f"unsupported eager selector-domain ABI {abi!r}")

    domain_rows = _table_rows(
        manifest,
        execution_root,
        selector.get("domains"),
        _EAGER_SELECTOR_DOMAIN,
        "eager execution.plan.selector_closures.domains",
    )
    group_rows = _table_rows(
        manifest,
        execution_root,
        selector.get("domain_group_ids"),
        _EAGER_SELECTOR_GROUP,
        "eager execution.plan.selector_closures.domain_group_ids",
    )
    memberships: list[frozenset[int]] = []
    cursor = 0
    for domain_id, row in enumerate(domain_rows):
        start, count = int(row[0]), int(row[1])
        if start != cursor:
            raise ArtifactError(
                f"eager selector domain {domain_id} starts at {start}, "
                f"expected {cursor}"
            )
        stop = start + count
        if stop > len(group_rows):
            raise ArtifactError(
                f"eager selector domain {domain_id} exceeds its membership table"
            )
        members = tuple(int(member[0]) for member in group_rows[start:stop])
        if members != tuple(sorted(set(members))):
            raise ArtifactError(
                f"eager selector domain {domain_id} members are not unique and sorted"
            )
        memberships.append(frozenset(members))
        cursor = stop
    if cursor != len(group_rows):
        raise ArtifactError(
            "eager selector domains do not cover their membership table"
        )
    if len(memberships) != len(set(memberships)):
        raise ArtifactError("eager selector domains contain duplicate memberships")

    runtime_schema = _mapping(
        execution.get("runtime_schema"), "eager execution.runtime_schema"
    )
    amplitude_stage = _mapping(
        runtime_schema.get("amplitude_stage"),
        "eager execution.runtime_schema.amplitude_stage",
    )
    roots = _sequence(
        amplitude_stage.get("roots"),
        "eager execution.runtime_schema.amplitude_stage.roots",
    )
    groups_by_output: dict[int, int] = {}
    for index, raw_root in enumerate(roots):
        root = _mapping(
            raw_root,
            f"eager execution.runtime_schema.amplitude_stage.roots[{index}]",
        )
        output_index = _integer(
            root.get("output_index"),
            f"eager execution.runtime_schema.amplitude_stage.roots[{index}]"
            ".output_index",
        )
        group_id = _integer(
            root.get("coherent_group_id"),
            f"eager execution.runtime_schema.amplitude_stage.roots[{index}]"
            ".coherent_group_id",
        )
        if output_index in groups_by_output:
            raise ArtifactError(
                f"eager amplitude roots repeat output index {output_index}"
            )
        groups_by_output[output_index] = group_id

    declared_groups = set(groups_by_output.values())
    unknown_groups = (
        set().union(*memberships) - declared_groups if memberships else set()
    )
    if unknown_groups:
        rendered = ", ".join(str(value) for value in sorted(unknown_groups))
        raise ArtifactError(
            f"eager selector domains reference undeclared coherent groups: {rendered}"
        )

    selector_stages = _sequence(
        selector.get("stages"),
        "eager execution.plan.selector_closures.stages",
    )
    if len(selector_stages) != len(stages):
        raise ArtifactError("eager selector domains do not cover every stage")
    domain_count = len(memberships)
    for index, (raw_selector_stage, execution_stage) in enumerate(
        zip(selector_stages, stages, strict=True)
    ):
        selector_stage = _mapping(
            raw_selector_stage,
            f"eager execution.plan.selector_closures.stages[{index}]",
        )
        stage, invocations, attachment_count, finalizations = execution_stage
        stage_index = _integer(
            stage.get("stage_index"),
            f"eager execution.plan.stages[{index}].stage_index",
        )
        selector_stage_index = _integer(
            selector_stage.get("stage_index"),
            f"eager execution.plan.selector_closures.stages[{index}].stage_index",
        )
        if selector_stage_index != stage_index:
            raise ArtifactError("eager selector-domain stage index mismatch")
        invocation_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("invocation_domains"),
            expected_count=len(invocations),
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".invocation_domains"
            ),
        )
        attachment_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("attachment_domains"),
            expected_count=attachment_count,
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".attachment_domains"
            ),
        )
        unpropagated_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("unpropagated_finalization_domains"),
            expected_count=len(finalizations),
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".unpropagated_finalization_domains"
            ),
        )
        propagated_domains = _selector_domain_ids(
            manifest,
            execution_root,
            selector_stage.get("propagated_finalization_domains"),
            expected_count=len(finalizations),
            domain_count=domain_count,
            context=(
                f"eager execution.plan.selector_closures.stages[{index}]"
                ".propagated_finalization_domains"
            ),
        )

        for invocation, domain_id in zip(
            invocations, invocation_domains, strict=True
        ):
            start, count = int(invocation[6]), int(invocation[7])
            stop = start + count
            if stop > len(attachment_domains):
                raise ArtifactError(
                    "eager invocation selector range exceeds attachment domains"
                )
            expected_domain = frozenset(
                group_id
                for attachment_domain_id in attachment_domains[start:stop]
                for group_id in memberships[attachment_domain_id]
            )
            if memberships[domain_id] != expected_domain:
                raise ArtifactError(
                    "eager invocation selector domain does not equal its "
                    "attachment-domain union"
                )
        for finalization, unpropagated, propagated in zip(
            finalizations,
            unpropagated_domains,
            propagated_domains,
            strict=True,
        ):
            if int(finalization[2]) == _MISSING_U32 and memberships[unpropagated]:
                raise ArtifactError(
                    "missing eager unpropagated output has a nonempty selector domain"
                )
            if int(finalization[3]) == _MISSING_U32 and memberships[propagated]:
                raise ArtifactError(
                    "missing eager propagated output has a nonempty selector domain"
                )

    closure_domains = _selector_domain_ids(
        manifest,
        execution_root,
        selector.get("closure_domains"),
        expected_count=len(closures),
        domain_count=domain_count,
        context="eager execution.plan.selector_closures.closure_domains",
    )
    for closure, domain_id in zip(closures, closure_domains, strict=True):
        output_index = int(closure[3])
        try:
            expected = frozenset((groups_by_output[output_index],))
        except KeyError as exc:
            raise ArtifactError(
                f"eager closure references unknown amplitude output {output_index}"
            ) from exc
        if memberships[domain_id] != expected:
            raise ArtifactError(
                "eager closure selector domain does not match its coherent group"
            )

    return domain_count, len(group_rows)


def _eager_execution_inspection(
    manifest: ArtifactManifest,
    execution: Mapping[str, object],
    execution_path: Path,
) -> _ExecutionInspection:
    runtime_options = _mapping(
        execution.get("runtime_options"), "eager execution.runtime_options"
    )
    requested_tile = _integer(
        runtime_options.get("point_tile_size"),
        "eager execution.runtime_options.point_tile_size",
        minimum=1,
    )
    workspace_mib = _integer(
        runtime_options.get("workspace_mib"),
        "eager execution.runtime_options.workspace_mib",
        minimum=1,
    )
    effective_tile = _optional_positive_integer(
        runtime_options.get("effective_point_tile_size"),
        "eager execution.runtime_options.effective_point_tile_size",
    )
    workspace_bytes = _optional_positive_integer(
        runtime_options.get("workspace_bytes"),
        "eager execution.runtime_options.workspace_bytes",
    )
    plan = _mapping(execution.get("plan"), "eager execution.plan")
    stages = _sequence(plan.get("stages"), "eager execution.plan.stages")
    execution_root = execution_path.parent

    invocation_count = 0
    attachment_count = 0
    finalization_count = 0
    maximum_fanout = 0
    referenced_kernel_ids: set[int] = set()
    inspected_stages: list[
        tuple[
            Mapping[str, object],
            tuple[tuple[int | float, ...], ...],
            int,
            tuple[tuple[int | float, ...], ...],
        ]
    ] = []
    for index, raw_stage in enumerate(stages):
        stage = _mapping(raw_stage, f"eager execution.plan.stages[{index}]")
        invocations = _table_rows(
            manifest,
            execution_root,
            stage.get("invocations"),
            _EAGER_INVOCATION,
            f"eager execution.plan.stages[{index}].invocations",
        )
        _attachment_path, stage_attachments, _attachment_size = _table_record(
            stage.get("attachments"),
            f"eager execution.plan.stages[{index}].attachments",
        )
        finalizations = _table_rows(
            manifest,
            execution_root,
            stage.get("finalizations"),
            _EAGER_FINALIZATION,
            f"eager execution.plan.stages[{index}].finalizations",
        )
        inspected_stages.append(
            (stage, invocations, stage_attachments, finalizations)
        )
        invocation_count += len(invocations)
        attachment_count += stage_attachments
        finalization_count += len(finalizations)
        for row in invocations:
            referenced_kernel_ids.add(int(row[0]))
            maximum_fanout = max(maximum_fanout, int(row[-1]))
        referenced_kernel_ids.update(
            int(row[0]) for row in finalizations if int(row[0]) != _MISSING_U32
        )

    closures = _table_rows(
        manifest,
        execution_root,
        plan.get("closures"),
        _EAGER_CLOSURE,
        "eager execution.plan.closures",
    )
    referenced_kernel_ids.update(
        int(row[0]) for row in closures if int(row[0]) != _MISSING_U32
    )
    if attachment_count < invocation_count:
        raise ArtifactError(
            "eager execution has fewer attachments than canonical invocations"
        )

    kernel_pack = _mapping(
        execution.get("kernel_pack"), "eager execution.kernel_pack"
    )
    pack_path = _artifact_path(
        manifest,
        _string(
            kernel_pack.get("manifest_path"),
            "eager execution.kernel_pack.manifest_path",
        ),
        "eager execution.kernel_pack.manifest_path",
    )
    pack = _json_mapping(pack_path, "prepared eager kernel pack")
    kernels = _sequence(pack.get("kernels"), "prepared eager kernel pack.kernels")
    kernel_ids = {
        _integer(
            _mapping(raw, f"prepared eager kernel pack.kernels[{index}]").get(
                "kernel_id"
            ),
            f"prepared eager kernel pack.kernels[{index}].kernel_id",
        )
        for index, raw in enumerate(kernels)
    }
    if not referenced_kernel_ids <= kernel_ids:
        missing = ", ".join(
            str(value) for value in sorted(referenced_kernel_ids - kernel_ids)
        )
        raise ArtifactError(
            "eager execution references kernels absent from its prepared pack: "
            f"{missing}"
        )

    selector_closures = plan.get("selector_closures")
    selector_closure_available = False
    selector_domain_count = None
    selector_domain_membership_count = None
    if selector_closures is not None:
        (
            selector_domain_count,
            selector_domain_membership_count,
        ) = _selector_domain_inspection(
            manifest,
            execution,
            execution_root,
            selector_closures,
            stages=tuple(inspected_stages),
            closures=closures,
        )
        selector_closure_available = True

    return _ExecutionInspection(
        execution_mode="eager",
        prepared_backend=_string(
            pack.get("backend"), "prepared eager kernel pack.backend"
        ),
        prepared_kernel_count=len(kernels),
        referenced_kernel_count=len(referenced_kernel_ids),
        invocation_count=invocation_count,
        attachment_count=attachment_count,
        evaluation_alias_count=attachment_count - invocation_count,
        maximum_fanout=maximum_fanout,
        finalization_count=finalization_count,
        closure_count=len(closures),
        selector_closure_available=selector_closure_available,
        selector_domain_count=selector_domain_count,
        selector_domain_membership_count=selector_domain_membership_count,
        requested_point_tile_size=requested_tile,
        effective_point_tile_size=effective_tile,
        workspace_limit_bytes=workspace_mib * 1024 * 1024,
        workspace_bytes=workspace_bytes,
        native_profile_phases=_profile_phases(execution),
    )


def _execution_inspection(
    manifest: ArtifactManifest,
    execution_path: Path,
) -> _ExecutionInspection:
    execution = _json_mapping(execution_path, "process execution manifest")
    kind = _string(execution.get("kind"), "process execution manifest.kind")
    if kind == _EAGER_RUNTIME_KIND:
        return _eager_execution_inspection(manifest, execution, execution_path)
    if kind == "pyamplicol-runtime-execution":
        return _ExecutionInspection(
            execution_mode="compiled",
            native_profile_phases=_COMPILED_PROFILE_PHASES,
        )
    raise ArtifactError(f"unsupported process execution kind {kind!r}")


def _physics_counts(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
) -> tuple[int, int, int, int, str, str]:
    relative = str(process["physics_path"])
    path = manifest.root / relative
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactError(
            f"cannot read runtime physics metadata {relative}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"invalid runtime physics metadata {relative}: {exc}"
        ) from exc

    physics = _mapping(payload, f"runtime physics metadata {relative}")
    helicities = _sequence(physics.get("helicities"), f"{relative}.helicities")
    colors = _sequence(physics.get("color_components"), f"{relative}.color_components")
    coverage = _mapping(physics.get("coverage"), f"{relative}.coverage")

    computed_helicities = sum(
        _computed(item, f"{relative}.helicities[{index}]")
        for index, item in enumerate(helicities)
    )
    computed_colors = sum(
        _computed(item, f"{relative}.color_components[{index}]")
        for index, item in enumerate(colors)
    )
    return (
        len(helicities),
        computed_helicities,
        len(colors),
        computed_colors,
        str(coverage.get("helicities", "unknown")),
        str(coverage.get("color", "unknown")),
    )


def _process_inspection(
    manifest: ArtifactManifest,
    process: Mapping[str, object],
    execution: _ExecutionInspection,
) -> ArtifactProcessInspection:
    process_id = str(process["id"])
    aliases: list[ArtifactAliasInspection] = []
    for index, raw_alias in enumerate(
        _sequence(process.get("aliases"), f"process {process_id}.aliases")
    ):
        alias = _mapping(raw_alias, f"process {process_id}.aliases[{index}]")
        aliases.append(
            ArtifactAliasInspection(
                id=str(alias["id"]),
                expression=str(alias["expression"]),
                representative_id=process_id,
                external_pdgs=tuple(
                    cast(int, value)
                    for value in _sequence(
                        alias.get("external_pdgs"),
                        f"process {process_id}.aliases[{index}].external_pdgs",
                    )
                ),
            )
        )
    (
        physical_helicities,
        computed_helicities,
        physical_colors,
        computed_colors,
        helicity_coverage,
        color_coverage,
    ) = _physics_counts(manifest, process)
    return ArtifactProcessInspection(
        id=process_id,
        expression=str(process["expression"]),
        color_accuracy=str(process["color_accuracy"]),
        external_pdgs=tuple(
            cast(int, value)
            for value in _sequence(
                process.get("external_pdgs"), f"process {process_id}.external_pdgs"
            )
        ),
        default=process_id == manifest.default_process_id,
        physical_helicities=physical_helicities,
        computed_helicities=computed_helicities,
        physical_color_components=physical_colors,
        computed_color_components=computed_colors,
        helicity_coverage=helicity_coverage,
        color_coverage=color_coverage,
        aliases=tuple(aliases),
        execution_mode=execution.execution_mode,
        prepared_backend=execution.prepared_backend,
        prepared_kernel_count=execution.prepared_kernel_count,
        referenced_kernel_count=execution.referenced_kernel_count,
        invocation_count=execution.invocation_count,
        attachment_count=execution.attachment_count,
        evaluation_alias_count=execution.evaluation_alias_count,
        maximum_fanout=execution.maximum_fanout,
        finalization_count=execution.finalization_count,
        closure_count=execution.closure_count,
        selector_closure_available=execution.selector_closure_available,
        selector_domain_count=execution.selector_domain_count,
        selector_domain_membership_count=execution.selector_domain_membership_count,
        requested_point_tile_size=execution.requested_point_tile_size,
        effective_point_tile_size=execution.effective_point_tile_size,
        workspace_limit_bytes=execution.workspace_limit_bytes,
        workspace_bytes=execution.workspace_bytes,
        native_profile_phases=execution.native_profile_phases,
    )


def inspect_artifact(artifact: str | Path) -> ArtifactInspection:
    """Validate and summarize one generated artifact without loading evaluators."""

    manifest = load_manifest(artifact)
    target = _mapping(manifest.producer["target"], "producer.target")
    model = manifest.model
    runtime = manifest.runtime
    execution_paths = _execution_paths(manifest)
    processes = tuple(
        _process_inspection(
            manifest,
            process,
            _execution_inspection(manifest, execution_paths[str(process["id"])]),
        )
        for process in manifest.processes
    )
    dependencies = tuple(
        ArtifactDependencyInspection(
            name=str(dependency["name"]),
            version=str(dependency["version"]),
            license=str(dependency["license"]),
            source=str(dependency["source"]),
        )
        for dependency in manifest.dependencies
    )
    restriction = model.get("restriction")
    return ArtifactInspection(
        kind="pyamplicol-artifact-inspection",
        path=manifest.root,
        artifact_kind=manifest.kind,
        artifact_id=manifest.artifact_id,
        created_utc=manifest.created_utc,
        producer_version=str(manifest.producer["version"]),
        target=str(target["triple"]),
        cpu_features=tuple(
            str(value)
            for value in _sequence(
                target.get("cpu_features"), "producer.target.cpu_features"
            )
        ),
        model_name=str(model["name"]),
        model_source=str(model["source_kind"]),
        model_restriction=None if restriction is None else str(restriction),
        default_process_id=manifest.default_process_id,
        runtime_engine=str(runtime["engine"]),
        runtime_version=str(runtime["engine_version"]),
        runtime_capabilities=tuple(
            str(value)
            for value in _sequence(
                runtime.get("required_runtime_capabilities"),
                "runtime.required_runtime_capabilities",
            )
        ),
        payload_count=len(manifest.payloads),
        payload_size_bytes=sum(payload.size_bytes for payload in manifest.payloads),
        integrity="verified",
        processes=processes,
        dependencies=dependencies,
    )


__all__ = [
    "ArtifactAliasInspection",
    "ArtifactDependencyInspection",
    "ArtifactInspection",
    "ArtifactProcessInspection",
    "inspect_artifact",
]
