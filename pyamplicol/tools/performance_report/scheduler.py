# SPDX-License-Identifier: 0BSD
"""Dependency-aware, resource-supervised performance-report campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .agreements import incoming_agreement_edges
from .artifacts import ArtifactAction, ArtifactStore, CurrentRecord
from .cache import validate_measurement
from .campaign_policy import (
    MACBOOK_M3_Z_TABLE_F_POLICY,
    STRICT_POLICY,
    X86_EPYC_NATIVE_COMPILER_SLOTS,
    X86_EPYC_PROFILE,
    CampaignPolicy,
    CampaignPolicyError,
    PolicyCensorKind,
    PolicyMeasurementState,
    dependency_reference,
    generation_limit_for_cell,
    policy_censor_measurement,
    resource_frontier_reference,
    resource_lane_identity,
    validate_campaign_settings,
    validate_policy_measurement,
    validate_policy_profile,
)
from .campaign_reset import OriginalAmplicolSeed, load_seed_if_present
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement import failure_measurement
from .measurement_lineage import MeasurementLineage
from .models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .phase_state import WorkerPhaseChannel
from .resources import (
    GenerationPhaseEvidence,
    ResourceUsage,
    supervise_worker,
)
from .runner import DEFAULT_TARGET_RUNTIME_SECONDS
from .service import CANONICAL_REPORT_ENTRYPOINT, ReportService
from .source_identity import require_eligible_report_source
from .study_contract import (
    StudyContractError,
    authenticate_z_table_f_study_contract,
    bind_z_table_f_attempt,
    require_z_table_f_attempt_binding,
    z_table_f_attempt_binding,
    z_table_f_cell_ids,
    z_table_f_worker_harness_identity,
)
from .worker_harness import (
    POLICY_ENTRYPOINT,
    WorkerHarnessError,
    attach_worker_harness_identity,
    require_worker_harness_identity,
    validate_worker_harness_identity,
)

_NATIVE_COMPILER_GATE_DIR_ENV = "PYAMPLICOL_NATIVE_COMPILER_GATE_DIR"
_NATIVE_COMPILER_GATE_SLOT_COUNT_ENV = "PYAMPLICOL_NATIVE_COMPILER_SLOT_COUNT"


@dataclass(frozen=True, slots=True)
class CellSelection:
    datasets: frozenset[str] = frozenset()
    modes: frozenset[ExecutionMode] = frozenset()
    models: frozenset[ModelKey] = frozenset()
    accuracies: frozenset[Accuracy] = frozenset()
    process_keys: frozenset[str] = frozenset()
    processes: frozenset[str] = frozenset()
    multiplicities: frozenset[int] = frozenset()
    variants: frozenset[str] = frozenset()
    workloads: frozenset[Workload] = frozenset()
    cell_ids: frozenset[str] = frozenset()

    def matches(self, cell: CellSpec) -> bool:
        model = cell.measurement.model
        checks = (
            not self.datasets or cell.dataset_id in self.datasets,
            not self.modes or cell.measurement.execution_mode in self.modes,
            not self.models or model in self.models,
            not self.accuracies or cell.measurement.accuracy in self.accuracies,
            not self.process_keys or cell.process_key in self.process_keys,
            not self.processes or cell.process in self.processes,
            not self.multiplicities or cell.n_final in self.multiplicities,
            not self.variants or cell.variant in self.variants,
            not self.workloads or cell.workload in self.workloads,
            not self.cell_ids or cell.cell_id in self.cell_ids,
        )
        return all(checks)


@dataclass(frozen=True, slots=True)
class CampaignSettings:
    workers: int = 1
    cell_cores: int = 1
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS
    batch_size: int = 128
    timeout_seconds: float | None = None
    generation_time_limit_seconds: float | None = None
    max_rss_bytes: int | None = None
    campaign_max_rss_bytes: int | None = None
    artifact_policy: ArtifactPolicy = ArtifactPolicy.REGENERATE
    missing_only: bool = False
    rerun: bool = False
    allow_symbolica_parallel: bool = False
    campaign_policy: CampaignPolicy = STRICT_POLICY
    report_profile: str | None = None
    study_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.workers < 1 or self.cell_cores < 1:
            raise ValueError("workers and cell_cores must be positive")
        if self.target_runtime_seconds <= 0.0 or self.batch_size < 1:
            raise ValueError("target runtime and batch size must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.generation_time_limit_seconds is not None and (
            self.generation_time_limit_seconds <= 0.0
            or not math.isfinite(self.generation_time_limit_seconds)
        ):
            raise ValueError("generation_time_limit_seconds must be positive")
        if self.max_rss_bytes is not None and self.max_rss_bytes <= 0:
            raise ValueError("max_rss_bytes must be positive")
        if self.campaign_max_rss_bytes is not None and self.campaign_max_rss_bytes <= 0:
            raise ValueError("campaign_max_rss_bytes must be positive")
        if self.missing_only and self.rerun:
            raise ValueError("--missing-only and --rerun are mutually exclusive")
        profile = self.report_profile or ""
        if self.campaign_policy is not STRICT_POLICY or self.report_profile is not None:
            validate_policy_profile(self.campaign_policy, profile)
        if self.campaign_policy is MACBOOK_M3_Z_TABLE_F_POLICY:
            if self.study_contract_sha256 is None:
                raise ValueError(
                    "the Z-table F policy requires a study contract SHA-256"
                )
            try:
                z_table_f_attempt_binding(self.study_contract_sha256)
            except StudyContractError as error:
                raise ValueError(str(error)) from error
        elif self.study_contract_sha256 is not None:
            raise ValueError(
                "a study contract SHA-256 requires the Z-table F policy"
            )
        validate_campaign_settings(self.campaign_policy, self)


@dataclass(frozen=True, slots=True)
class PlannedCell:
    cell: CellSpec
    dependency: bool
    baseline_cell_id: str | None
    rank: int
    comparison_peer_ids: tuple[str, ...] = ()
    force_recompare: bool = False


@dataclass(frozen=True, slots=True)
class CellOutcome:
    cell_id: str
    status: str
    detail: str


def _worker_environment_overrides(
    settings: CampaignSettings,
    coordination_root: Path,
) -> dict[str, str]:
    """Return identity-neutral cross-controller worker controls."""

    if settings.report_profile != X86_EPYC_PROFILE:
        return {}
    return {
        _NATIVE_COMPILER_GATE_DIR_ENV: os.fspath(
            coordination_root.resolve() / "native-compiler-slots"
        ),
        _NATIVE_COMPILER_GATE_SLOT_COUNT_ENV: str(X86_EPYC_NATIVE_COMPILER_SLOTS),
    }


@dataclass(frozen=True, slots=True)
class CampaignResult:
    planned: tuple[PlannedCell, ...]
    outcomes: tuple[CellOutcome, ...]

    @property
    def failed(self) -> tuple[CellOutcome, ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.status
            not in {
                "ok",
                "reused",
                "skipped-current",
                PolicyMeasurementState.GENERATION_LIMIT.value,
                PolicyMeasurementState.MEMORY_LIMIT.value,
                PolicyMeasurementState.DEPENDENCY.value,
                PolicyMeasurementState.RESOURCE_FRONTIER.value,
            }
        )


def select_cells(
    selection: CellSelection,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    return tuple(
        sorted(
            (cell for cell in catalog.measurement_cells() if selection.matches(cell)),
            key=lambda cell: cell.cell_id,
        )
    )


def _rank(cell: CellSpec) -> int:
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        return 0
    if cell.measurement.execution_mode is ExecutionMode.RECURRENCE:
        return 1
    if cell.dataset_id in {"scalar_contact", "scalar_gravity"}:
        return 0
    return 2


def _successful_current(
    store: ArtifactStore,
    cell_id: str,
    *,
    expected_revision: str | None = None,
    expected_tree: str | None = None,
    expected_cell: CellSpec | None = None,
    measurement_lineage: MeasurementLineage | None = None,
    expected_study_contract_sha256: str | None = None,
    expected_worker_harness: Mapping[str, object] | None = None,
) -> CurrentRecord | None:
    current = store.load_current(cell_id, missing_ok=True)
    if current is None or current.result.get("status") != ResultStatus.OK.value:
        return None
    try:
        validate_measurement(current.result, expected_cell=expected_cell)
        if expected_study_contract_sha256 is not None:
            require_z_table_f_attempt_binding(
                current.result,
                expected_study_contract_sha256,
            )
            if expected_worker_harness is None:
                raise WorkerHarnessError(
                    "the expected worker harness is unavailable"
                )
            require_worker_harness_identity(
                current.result,
                expected=expected_worker_harness,
            )
    except (StudyContractError, ValueError):
        return None
    if expected_revision is not None:
        provenance = current.result.get("provenance")
        if measurement_lineage is not None and expected_tree is not None:
            source = measurement_lineage.source_for_current(
                current,
                active_revision=expected_revision,
                active_tree=expected_tree,
            )
        else:
            source = (
                (expected_revision, expected_tree or "")
                if isinstance(provenance, Mapping)
                and provenance.get("report_source_revision") == expected_revision
                and (
                    expected_tree is None
                    or provenance.get("report_source_tree") == expected_tree
                )
                else None
            )
        if source is None:
            return None
    return current


def _policy_current(
    store: ArtifactStore,
    cell: CellSpec,
    *,
    settings: CampaignSettings,
    expected_revision: str | None,
    expected_tree: str | None = None,
    measurement_lineage: MeasurementLineage | None = None,
    original_amplicol_seed: OriginalAmplicolSeed | None = None,
    expected_worker_harness: Mapping[str, object] | None = None,
) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
    if settings.campaign_policy is STRICT_POLICY:
        current = _successful_current(
            store,
            cell.cell_id,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            expected_cell=cell,
            measurement_lineage=measurement_lineage,
            expected_study_contract_sha256=(
                settings.study_contract_sha256
            ),
            expected_worker_harness=expected_worker_harness,
        )
        return None if current is None else (current, PolicyMeasurementState.SUCCESS)
    current = store.load_current(cell.cell_id, missing_ok=True)
    if current is None or expected_revision is None:
        return None
    provenance = current.result.get("provenance")
    expected_source = (
        measurement_lineage.source_for_current(
            current,
            active_revision=expected_revision,
            active_tree=expected_tree or "",
        )
        if measurement_lineage is not None and expected_tree is not None
        else (
            (expected_revision, expected_tree)
            if isinstance(provenance, Mapping)
            and provenance.get("report_source_revision") == expected_revision
            and (
                expected_tree is None
                or provenance.get("report_source_tree") == expected_tree
            )
            else None
        )
    )
    if expected_source is None and original_amplicol_seed is not None:
        expected_source = original_amplicol_seed.source_for_current(
            current,
            active_revision=expected_revision,
            active_tree=expected_tree or "",
        )
    if expected_source is None:
        return None
    try:
        validate_measurement(current.result, expected_cell=cell)
        if settings.study_contract_sha256 is not None:
            require_z_table_f_attempt_binding(
                current.result,
                settings.study_contract_sha256,
            )
            if expected_worker_harness is None:
                raise WorkerHarnessError(
                    "the expected worker harness is unavailable"
                )
            require_worker_harness_identity(
                current.result,
                expected=expected_worker_harness,
            )
        state = validate_policy_measurement(
            settings.campaign_policy,
            settings.report_profile or "",
            cell,
            current.result,
            expected_source_revision=expected_source[0],
            expected_source_tree=expected_source[1],
        )
    except (CampaignPolicyError, StudyContractError, ValueError):
        return None
    return current, state


def _resource_lane_key(cell: CellSpec) -> tuple[object, ...]:
    """Identify one monotone multiplicity lane across the report catalog."""

    return tuple(resource_lane_identity(cell).items())


def _resource_lane_lock_name(cell: CellSpec) -> str:
    encoded = json.dumps(
        resource_lane_identity(cell),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return f"campaign-resource-lane-{hashlib.sha256(encoded).hexdigest()}"


def _resource_frontier_source(
    store: ArtifactStore,
    cell: CellSpec,
    *,
    settings: CampaignSettings,
    catalog: ReportCatalog,
    expected_revision: str | None,
    expected_tree: str | None,
    measurement_lineage: MeasurementLineage | None = None,
    original_amplicol_seed: OriginalAmplicolSeed | None = None,
    expected_worker_harness: Mapping[str, object] | None = None,
    lane_cells: Sequence[CellSpec] | None = None,
) -> tuple[CellSpec, CurrentRecord, PolicyMeasurementState] | None:
    """Return the first authenticated lower-multiplicity hard resource censor."""

    if not settings.campaign_policy.allow_terminal_censors:
        return None
    lane = _resource_lane_key(cell)
    sources: list[tuple[CellSpec, CurrentRecord, PolicyMeasurementState]] = []
    candidates = catalog.measurement_cells() if lane_cells is None else lane_cells
    for candidate in candidates:
        if candidate.n_final >= cell.n_final or _resource_lane_key(candidate) != lane:
            continue
        current = _policy_current(
            store,
            candidate,
            settings=settings,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            measurement_lineage=measurement_lineage,
            original_amplicol_seed=original_amplicol_seed,
            expected_worker_harness=expected_worker_harness,
        )
        if current is None:
            continue
        record, state = current
        if state in {
            PolicyMeasurementState.GENERATION_LIMIT,
            PolicyMeasurementState.MEMORY_LIMIT,
        }:
            sources.append((candidate, record, state))
    return min(sources, key=lambda item: item[0].n_final) if sources else None


def _frontier_reference(
    source: tuple[CellSpec, CurrentRecord, PolicyMeasurementState],
    cell: CellSpec,
) -> dict[str, object]:
    source_cell, record, _state = source
    return resource_frontier_reference(cell, source_cell, record.result)


def _measurement_frontier(
    measurement: Mapping[str, object],
) -> object:
    provenance = measurement.get("provenance")
    censor = (
        provenance.get("policy_censor") if isinstance(provenance, Mapping) else None
    )
    return censor.get("frontier") if isinstance(censor, Mapping) else None


def _fresh_equivalent_current(
    store: ArtifactStore,
    cell: CellSpec,
    *,
    catalog: ReportCatalog,
    expected_revision: str,
    expected_tree: str | None = None,
    measurement_lineage: MeasurementLineage | None = None,
    expected_study_contract_sha256: str | None = None,
    expected_worker_harness: Mapping[str, object] | None = None,
) -> CurrentRecord | None:
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        return None
    for equivalent in catalog.equivalent_cells(cell):
        current = _successful_current(
            store,
            equivalent.cell_id,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            expected_cell=equivalent,
            measurement_lineage=measurement_lineage,
            expected_study_contract_sha256=(
                expected_study_contract_sha256
            ),
            expected_worker_harness=expected_worker_harness,
        )
        if current is not None:
            return current
    return None


def plan_campaign(
    requested: Sequence[CellSpec],
    *,
    store: ArtifactStore,
    settings: CampaignSettings,
    catalog: ReportCatalog = REPORT_CATALOG,
    expected_revision: str | None = None,
    expected_tree: str | None = None,
    measurement_lineage: MeasurementLineage | None = None,
    original_amplicol_seed: OriginalAmplicolSeed | None = None,
    excluded_cell_ids: frozenset[str] = frozenset(),
    expected_worker_harness: Mapping[str, object] | None = None,
) -> tuple[PlannedCell, ...]:
    requested_ids = {cell.cell_id for cell in requested}
    needed: dict[str, CellSpec] = {}
    force_recompare_ids: set[str] = set()
    visiting: set[str] = set()
    resource_lanes: dict[tuple[object, ...], list[CellSpec]] = {}
    if original_amplicol_seed is None and settings.report_profile is not None:
        original_amplicol_seed = load_seed_if_present(
            profile=settings.report_profile,
            store=store,
        )
    if settings.campaign_policy.allow_terminal_censors:
        for catalog_cell in catalog.measurement_cells():
            if catalog.static_na_reason(catalog_cell) is not None:
                continue
            resource_lanes.setdefault(
                _resource_lane_key(catalog_cell),
                [],
            ).append(catalog_cell)

    def dependencies(cell: CellSpec) -> tuple[CellSpec, ...]:
        baseline = catalog.validation_baseline_cell(cell)
        peers = tuple(
            edge.baseline for edge in incoming_agreement_edges(cell, catalog=catalog)
        )
        return tuple(
            {
                dependency.cell_id: dependency
                for dependency in (
                    *((baseline,) if baseline is not None else ()),
                    *peers,
                )
            }.values()
        )

    exclusion_memo: dict[str, bool] = {}
    exclusion_visiting: set[str] = set()

    def blocked_by_exclusion(cell: CellSpec) -> bool:
        cached = exclusion_memo.get(cell.cell_id)
        if cached is not None:
            return cached
        if cell.cell_id in excluded_cell_ids:
            exclusion_memo[cell.cell_id] = True
            return True
        if cell.cell_id in exclusion_visiting:
            raise ValueError(
                f"report comparison dependency cycle reaches {cell.cell_id!r}"
            )
        exclusion_visiting.add(cell.cell_id)
        try:
            blocked = any(
                blocked_by_exclusion(dependency)
                for dependency in dependencies(cell)
                if _policy_current(
                    store,
                    dependency,
                    settings=settings,
                    expected_revision=expected_revision,
                    expected_tree=expected_tree,
                    measurement_lineage=measurement_lineage,
                    original_amplicol_seed=original_amplicol_seed,
                    expected_worker_harness=expected_worker_harness,
                )
                is None
            )
        finally:
            exclusion_visiting.remove(cell.cell_id)
        exclusion_memo[cell.cell_id] = blocked
        return blocked

    def include(cell: CellSpec, *, explicitly_requested: bool) -> None:
        if catalog.static_na_reason(cell) is not None:
            return
        if blocked_by_exclusion(cell):
            return
        cell_dependencies = dependencies(cell)
        dependency_currents = {
            dependency.cell_id: _policy_current(
                store,
                dependency,
                settings=settings,
                expected_revision=expected_revision,
                expected_tree=expected_tree,
                measurement_lineage=measurement_lineage,
                original_amplicol_seed=original_amplicol_seed,
                expected_worker_harness=expected_worker_harness,
            )
            for dependency in cell_dependencies
        }
        missing_dependencies = tuple(
            dependency
            for dependency in cell_dependencies
            if dependency_currents[dependency.cell_id] is None
        )
        terminal_dependencies = tuple(
            dependency_reference(
                dependency.cell_id,
                dependency_currents[dependency.cell_id][0].result,  # type: ignore[index]
            )
            for dependency in cell_dependencies
            if (
                dependency_currents[dependency.cell_id] is not None
                and dependency_currents[dependency.cell_id][1]  # type: ignore[index]
                is not PolicyMeasurementState.SUCCESS
            )
        )
        current = _policy_current(
            store,
            cell,
            settings=settings,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            measurement_lineage=measurement_lineage,
            original_amplicol_seed=original_amplicol_seed,
            expected_worker_harness=expected_worker_harness,
        )
        frontier_source = _resource_frontier_source(
            store,
            cell,
            settings=settings,
            catalog=catalog,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            measurement_lineage=measurement_lineage,
            original_amplicol_seed=original_amplicol_seed,
            expected_worker_harness=expected_worker_harness,
            lane_cells=resource_lanes.get(_resource_lane_key(cell), ()),
        )
        expected_frontier = (
            None
            if frontier_source is None
            else _frontier_reference(frontier_source, cell)
        )
        fresh_requested = False
        if settings.missing_only and explicitly_requested and current is not None:
            _record, state = current
            if frontier_source is not None:
                fresh_requested = (
                    state is PolicyMeasurementState.RESOURCE_FRONTIER
                    and _measurement_frontier(_record.result) == expected_frontier
                )
            elif state is PolicyMeasurementState.SUCCESS:
                fresh_requested = not missing_dependencies and not terminal_dependencies
            elif state is PolicyMeasurementState.DEPENDENCY:
                provenance = _record.result.get("provenance")
                censor = (
                    provenance.get("policy_censor")
                    if isinstance(provenance, Mapping)
                    else None
                )
                observed = (
                    censor.get("dependencies") if isinstance(censor, Mapping) else None
                )
                fresh_requested = (
                    not missing_dependencies
                    and bool(terminal_dependencies)
                    and observed
                    == list(
                        sorted(
                            terminal_dependencies,
                            key=lambda item: str(item["cell_id"]),
                        )
                    )
                )
            elif state is PolicyMeasurementState.RESOURCE_FRONTIER:
                fresh_requested = False
            else:
                fresh_requested = True
        if fresh_requested:
            return
        if settings.missing_only and explicitly_requested and current is not None:
            force_recompare_ids.add(cell.cell_id)
        if cell.cell_id in needed:
            return
        if cell.cell_id in visiting:
            raise ValueError(
                f"report comparison dependency cycle reaches {cell.cell_id!r}"
            )
        visiting.add(cell.cell_id)
        if frontier_source is None:
            for dependency in missing_dependencies:
                include(dependency, explicitly_requested=False)
        visiting.remove(cell.cell_id)
        needed[cell.cell_id] = cell

    for cell in requested:
        include(cell, explicitly_requested=True)

    if settings.campaign_policy.allow_terminal_censors:
        scheduled = tuple(needed.values())
        for cell in requested:
            if cell.cell_id in needed:
                continue
            if catalog.static_na_reason(cell) is not None:
                continue
            if blocked_by_exclusion(cell):
                continue
            if any(
                predecessor.n_final < cell.n_final
                and _resource_lane_key(predecessor) == _resource_lane_key(cell)
                for predecessor in scheduled
            ):
                # The lower cell will run in this campaign. Retain the higher
                # fresh current for a post-wave frontier rescan; _run_cell
                # returns without launching it when the lower cell succeeds.
                needed[cell.cell_id] = cell

    ranks: dict[str, int] = {}

    def planned_rank(cell: CellSpec) -> int:
        if cell.cell_id in ranks:
            return ranks[cell.cell_id]
        rank = _rank(cell)
        for dependency in dependencies(cell):
            scheduled = needed.get(dependency.cell_id)
            if scheduled is not None:
                rank = max(rank, planned_rank(scheduled) + 1)
        if settings.campaign_policy.allow_terminal_censors:
            predecessors = tuple(
                candidate
                for candidate in needed.values()
                if candidate.n_final < cell.n_final
                and _resource_lane_key(candidate) == _resource_lane_key(cell)
            )
            if predecessors:
                predecessor = max(
                    predecessors,
                    key=lambda item: item.n_final,
                )
                rank = max(rank, planned_rank(predecessor) + 1)
        ranks[cell.cell_id] = rank
        return rank

    def validation_baseline_id(cell: CellSpec) -> str | None:
        baseline = catalog.validation_baseline_cell(cell)
        return None if baseline is None else baseline.cell_id

    return tuple(
        PlannedCell(
            cell=cell,
            dependency=cell.cell_id not in requested_ids,
            baseline_cell_id=validation_baseline_id(cell),
            rank=planned_rank(cell),
            comparison_peer_ids=tuple(
                edge.baseline.cell_id
                for edge in incoming_agreement_edges(cell, catalog=catalog)
            ),
            force_recompare=cell.cell_id in force_recompare_ids,
        )
        for cell in sorted(
            needed.values(),
            key=lambda item: (planned_rank(item), item.cell_id),
        )
    )


def validate_campaign_plan(
    planned: Sequence[PlannedCell],
    settings: CampaignSettings,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> None:
    """Enforce study-wide plan constraints before any attempt is created."""

    if settings.campaign_policy is not MACBOOK_M3_Z_TABLE_F_POLICY:
        return
    if len(planned) != 1 or planned[0].dependency:
        raise StudyContractError(
            "the Z-table F policy requires exactly one runnable requested "
            "cell; all comparison dependencies must already have "
            "authenticated currents"
        )
    cell_id = planned[0].cell.cell_id
    if cell_id not in frozenset(z_table_f_cell_ids()):
        raise StudyContractError(
            f"cell {cell_id!r} is outside the contracted 28-cell Z-table F scope"
        )
    baseline = catalog.validation_baseline_cell(planned[0].cell)
    expected_baseline_id = None if baseline is None else baseline.cell_id
    expected_peer_ids = tuple(
        edge.baseline.cell_id
        for edge in incoming_agreement_edges(planned[0].cell, catalog=catalog)
    )
    if (
        planned[0].baseline_cell_id != expected_baseline_id
        or planned[0].comparison_peer_ids != expected_peer_ids
    ):
        raise StudyContractError(
            "the Z-table F runnable cell does not carry its canonical "
            "comparison dependency plan"
        )


def _resource_payload(
    usage: ResourceUsage,
    generation_phase: GenerationPhaseEvidence | None = None,
    *,
    memory_limit_bytes: int | None = None,
    memory_limit_reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "available": usage.available,
        "current_rss_bytes": usage.current_rss_bytes,
        "peak_rss_bytes": usage.peak_rss_bytes,
        "child_count": usage.child_count,
        "cpu_seconds": usage.cpu_seconds,
        "wall_seconds": usage.wall_seconds,
        "probe_error": usage.error,
    }
    if usage.memory_metric_abi is not None:
        payload.update(
            {
                "memory_metric_abi": usage.memory_metric_abi,
                "current_physical_footprint_bytes": (
                    usage.current_physical_footprint_bytes
                ),
                "peak_physical_footprint_bytes": (usage.peak_physical_footprint_bytes),
                "current_guard_bytes": usage.current_guard_bytes,
                "peak_guard_bytes": usage.peak_guard_bytes,
                "memory_limit_bytes": memory_limit_bytes,
                "memory_limit_reason": memory_limit_reason,
                "memory_probe_reason": usage.memory_probe_reason,
            }
        )
    if generation_phase is not None:
        payload["generation_phase"] = generation_phase.as_dict()
    return payload


def _attempt_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"manifest.json", "result.json"}
            and not path.is_symlink()
        )
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )


class CampaignScheduler:
    def __init__(
        self,
        service: ReportService,
        *,
        settings: CampaignSettings,
        catalog: ReportCatalog = REPORT_CATALOG,
        measurement_lineage: MeasurementLineage | None = None,
        measurement_lineage_authenticated: bool = False,
        study_contract: Mapping[str, object] | None = None,
        study_contract_wrapper_root: Path | None = None,
    ) -> None:
        self.service = service
        self.settings = settings
        self.catalog = catalog
        self.source_identity = require_eligible_report_source(service.paths.repo_root)
        self.source_revision = self.source_identity.revision
        self.source_tree = self.source_identity.tree
        if settings.campaign_policy is MACBOOK_M3_Z_TABLE_F_POLICY:
            if study_contract is None or study_contract_wrapper_root is None:
                raise StudyContractError(
                    "the Z-table F scheduler requires the authenticated "
                    "study contract, not only a SHA-256"
                )
            authenticated_contract = authenticate_z_table_f_study_contract(
                study_contract,
                service.paths.repo_root,
                study_contract_wrapper_root,
                prior_store=service.store,
            )
            if (
                authenticated_contract.get("sha256")
                != settings.study_contract_sha256
            ):
                raise StudyContractError(
                    "the Z-table F scheduler contract SHA-256 differs from "
                    "campaign settings"
                )
            self.study_contract = authenticated_contract
            wrapper_record = authenticated_contract.get("policy_wrapper")
            source_record = authenticated_contract.get("measured_source")
            if not isinstance(wrapper_record, Mapping) or not isinstance(
                source_record,
                Mapping,
            ):
                raise StudyContractError(
                    "the Z-table F worker identities are missing"
                )
            self.worker_wrapper_root = (
                study_contract_wrapper_root.expanduser().resolve(strict=True)
            )
            self.worker_entrypoint = (
                self.worker_wrapper_root / POLICY_ENTRYPOINT
            )
            if (
                self.worker_entrypoint.is_symlink()
                or not self.worker_entrypoint.is_file()
            ):
                raise StudyContractError(
                    "the authenticated policy-wrapper entrypoint is unavailable"
                )
            self.worker_harness_identity = (
                z_table_f_worker_harness_identity(authenticated_contract)
            )
        else:
            if study_contract is not None or study_contract_wrapper_root is not None:
                raise StudyContractError(
                    "a study contract authorization requires the Z-table F policy"
                )
            self.study_contract = None
            self.worker_wrapper_root = service.paths.repo_root
            self.worker_entrypoint = (
                service.paths.repo_root / CANONICAL_REPORT_ENTRYPOINT
            )
            self.worker_harness_identity = None
        self.measurement_lineage = (
            measurement_lineage
            if measurement_lineage_authenticated
            else (
                None
                if settings.report_profile is None
                else service._measurement_lineage()
            )
        )
        self.measurement_source_tree = (
            self.source_tree if settings.report_profile is not None else None
        )
        self.service.bind_measurement_lineage(self.measurement_lineage)
        self.original_amplicol_seed = (
            None
            if settings.report_profile is None
            else self.service._original_amplicol_seed()
        )
        self.service.bind_original_amplicol_seed(self.original_amplicol_seed)
        self._prepared_model_paths: dict[ModelKey, Path] = {}
        self._resource_lanes: dict[tuple[object, ...], tuple[CellSpec, ...]] = {}
        if settings.campaign_policy.allow_terminal_censors:
            grouped: dict[tuple[object, ...], list[CellSpec]] = {}
            for cell in catalog.measurement_cells():
                if catalog.static_na_reason(cell) is not None:
                    continue
                grouped.setdefault(_resource_lane_key(cell), []).append(cell)
            self._resource_lanes = {
                lane: tuple(sorted(cells, key=lambda item: item.n_final))
                for lane, cells in grouped.items()
            }

    def _current(
        self,
        cell: CellSpec,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        return _policy_current(
            self.service.store,
            cell,
            settings=self.settings,
            expected_revision=self.source_revision,
            expected_tree=self.measurement_source_tree,
            measurement_lineage=self.measurement_lineage,
            original_amplicol_seed=self.original_amplicol_seed,
            expected_worker_harness=self.worker_harness_identity,
        )

    def _validate_z_table_f_plan(
        self,
        planned: Sequence[PlannedCell],
    ) -> None:
        validate_campaign_plan(
            planned,
            self.settings,
            catalog=self.catalog,
        )
        if self.settings.campaign_policy is not MACBOOK_M3_Z_TABLE_F_POLICY:
            return
        item = planned[0]
        baseline = self.catalog.validation_baseline_cell(item.cell)
        dependencies = (
            *((baseline,) if baseline is not None else ()),
            *(
                edge.baseline
                for edge in incoming_agreement_edges(
                    item.cell,
                    catalog=self.catalog,
                )
            ),
        )
        missing = tuple(
            dependency.cell_id
            for dependency in dependencies
            if self._current(dependency) is None
        )
        if missing:
            raise StudyContractError(
                "the Z-table F runnable cell has unauthenticated comparison "
                f"dependencies: {', '.join(sorted(missing))}"
            )

    def _prepare_legacy_workspace(self, attempt_id: str) -> Path:
        """Create one pinned writable legacy checkout for one worker only."""

        from .legacy import MaintainedLegacyApi

        api = MaintainedLegacyApi()
        source = api.default_repository.expanduser().resolve(strict=True)
        api.validate_checkout(source)
        destination = (
            self.service.paths.artifact_root / "legacy-workspaces" / attempt_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"legacy worker workspace already exists: {destination}")
        commands = (
            (
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--",
                os.fspath(source),
                os.fspath(destination),
            ),
            (
                "git",
                "-C",
                os.fspath(destination),
                "checkout",
                "--detach",
                api.expected_revision(),
            ),
        )
        for command in commands:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    "cannot prepare isolated legacy worker checkout: "
                    f"{detail or f'exit {completed.returncode}'}"
                )
        api.validate_checkout(destination)
        return destination

    def _service_path_arguments(self) -> tuple[str, ...]:
        paths = self.service.paths
        return (
            "--docs-dir",
            os.fspath(paths.docs_dir),
            "--artifact-root",
            os.fspath(paths.artifact_root),
            "--coordination-root",
            os.fspath(paths.coordination_root),
        )

    def _worker_harness_arguments(self) -> tuple[str, ...]:
        identity = self.worker_harness_identity
        if identity is None:
            return ()
        return (
            "--measurement-source-root",
            os.fspath(self.service.paths.repo_root),
            "--expected-measurement-source-revision",
            str(identity["measured_source_revision"]),
            "--expected-measurement-source-tree",
            str(identity["measured_source_tree"]),
            "--expected-policy-wrapper-revision",
            str(identity["policy_wrapper_revision"]),
            "--expected-policy-wrapper-tree",
            str(identity["policy_wrapper_tree"]),
            "--expected-policy-entrypoint-sha256",
            str(identity["policy_entrypoint_sha256"]),
            "--expected-legacy-adapter-sha256",
            str(identity["legacy_adapter_sha256"]),
            "--study-contract-sha256",
            str(identity["study_contract_sha256"]),
        )

    def _bind_study_result(
        self,
        result: dict[str, object],
        *,
        require_existing_harness: bool = False,
    ) -> None:
        if self.settings.study_contract_sha256 is None:
            return
        identity = self.worker_harness_identity
        if identity is None:
            raise StudyContractError(
                "the Z-table F worker harness identity is unavailable"
            )
        if require_existing_harness:
            require_worker_harness_identity(result, expected=identity)
        else:
            attach_worker_harness_identity(result, identity)
        bind_z_table_f_attempt(
            result,
            self.settings.study_contract_sha256,
        )

    def _ensure_prepared_model(self, planned: Sequence[PlannedCell]) -> None:
        models = {
            item.cell.measurement.model
            for item in planned
            if item.cell.measurement.execution_mode
            in {ExecutionMode.EAGER, ExecutionMode.RECURRENCE}
            and item.cell.measurement.model in {ModelKey.BUILTIN_SM, ModelKey.UFO_SM}
        }
        for model in sorted(models, key=lambda item: item.value):  # type: ignore[union-attr]
            assert model is not None
            result_path = (
                self.service.paths.artifact_root
                / "prepared-models"
                / f".preflight-{model.value}-{uuid.uuid4().hex}.json"
            )
            command = (
                sys.executable,
                "-I",
                "-S",
                "-B",
                os.fspath(self.worker_entrypoint),
                "--repo-root",
                os.fspath(self.service.paths.repo_root),
                *self._service_path_arguments(),
                *self._worker_harness_arguments(),
                "_prepare",
                "--model",
                model.value,
                "--result-json",
                os.fspath(result_path),
                "--cell-cores",
                str(self.settings.cell_cores),
            )
            supervised = supervise_worker(
                command,
                timeout_seconds=self.settings.timeout_seconds,
                max_rss_bytes=self._effective_cell_rss_limit(),
                environment_overrides=_worker_environment_overrides(
                    self.settings,
                    self.service.paths.coordination_root,
                ),
                scrub_import_environment=(
                    self.worker_harness_identity is not None
                ),
                working_directory=(
                    self.service.paths.repo_root
                    if self.worker_harness_identity is not None
                    else None
                ),
            )
            try:
                if (
                    supervised.reason != "completed"
                    or supervised.returncode != 0
                    or not result_path.is_file()
                ):
                    detail = None
                    if result_path.is_file():
                        try:
                            detail = json.loads(
                                result_path.read_text(encoding="ascii")
                            ).get("error")
                        except (AttributeError, json.JSONDecodeError, OSError):
                            detail = None
                    raise RuntimeError(
                        f"{model.value} prepared-model preflight failed: "
                        f"reason={supervised.reason}, "
                        f"exit={supervised.returncode}, detail={detail}"
                    )
                payload = json.loads(result_path.read_text(encoding="ascii"))
                if not isinstance(payload, Mapping):
                    raise RuntimeError(
                        f"{model.value} prepared-model preflight returned "
                        "a non-object payload"
                    )
                if self.worker_harness_identity is not None:
                    validate_worker_harness_identity(
                        payload.get("worker_harness"),
                        expected=self.worker_harness_identity,
                    )
                path = Path(str(payload["path"])).resolve(strict=True)
                self._prepared_model_paths[model] = path
            finally:
                result_path.unlink(missing_ok=True)

    def run(self, planned: Sequence[PlannedCell]) -> CampaignResult:
        ordered = tuple(planned)
        self._validate_z_table_f_plan(ordered)
        static_na = tuple(
            (
                item.cell.cell_id,
                self.catalog.static_na_reason(item.cell),
            )
            for item in ordered
            if self.catalog.static_na_reason(item.cell) is not None
        )
        if static_na:
            cell_id, reason = static_na[0]
            raise ValueError(
                f"campaign plan contains catalog static N/A cell {cell_id!r}: {reason}"
            )
        self._ensure_prepared_model(ordered)
        outcomes: list[CellOutcome] = []
        effective_workers = self.settings.workers
        if not self.settings.allow_symbolica_parallel and any(
            item.cell.measurement.model is ModelKey.UFO_SM
            and item.cell.measurement.execution_mode is ExecutionMode.COMPILED
            for item in ordered
        ):
            effective_workers = 1
        for rank in sorted({item.rank for item in ordered}):
            wave = tuple(item for item in ordered if item.rank == rank)
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {executor.submit(self._run_cell, item): item for item in wave}
                for future in as_completed(futures):
                    outcomes.append(future.result())
        return CampaignResult(
            planned=ordered,
            outcomes=tuple(sorted(outcomes, key=lambda item: item.cell_id)),
        )

    def _run_cell(self, planned: PlannedCell) -> CellOutcome:
        if not self.settings.campaign_policy.allow_terminal_censors:
            return self._run_cell_in_lane(planned)
        with self.service.store.named_lock(_resource_lane_lock_name(planned.cell)):
            return self._run_cell_in_lane(planned)

    def _run_cell_in_lane(self, planned: PlannedCell) -> CellOutcome:
        if self.settings.campaign_policy is MACBOOK_M3_Z_TABLE_F_POLICY:
            self._validate_z_table_f_plan((planned,))
        cell = planned.cell
        with self.service.store.named_lock(f"campaign-cell-{cell.cell_id}"):
            fresh = self._current(cell)
            frontier_source = _resource_frontier_source(
                self.service.store,
                cell,
                settings=self.settings,
                catalog=self.catalog,
                expected_revision=self.source_revision,
                expected_tree=self.measurement_source_tree,
                measurement_lineage=self.measurement_lineage,
                original_amplicol_seed=self.original_amplicol_seed,
                expected_worker_harness=self.worker_harness_identity,
                lane_cells=self._resource_lanes.get(
                    _resource_lane_key(cell),
                    (),
                ),
            )
            if (
                self.settings.missing_only
                and not planned.force_recompare
                and fresh is not None
                and frontier_source is None
            ):
                return CellOutcome(cell.cell_id, "skipped-current", "already complete")
            if frontier_source is not None:
                expected_frontier = _frontier_reference(frontier_source, cell)
                if (
                    fresh is not None
                    and fresh[1] is PolicyMeasurementState.RESOURCE_FRONTIER
                    and _measurement_frontier(fresh[0].result) == expected_frontier
                ):
                    return CellOutcome(
                        cell.cell_id,
                        "skipped-current",
                        "resource frontier already authenticated",
                    )
            decision = self.service.store.decide(
                cell.cell_id,
                self.settings.artifact_policy,
            )
            if frontier_source is not None:
                return self._publish_resource_frontier(
                    cell,
                    _frontier_reference(frontier_source, cell),
                    current=decision.current,
                )
            current_is_fresh = fresh is not None
            if (
                decision.action is ArtifactAction.REUSE_CURRENT
                and decision.current is not None
                and current_is_fresh
                and not self.settings.rerun
                and not planned.force_recompare
            ):
                return CellOutcome(cell.cell_id, "reused", decision.current.attempt_id)

            equivalent_record = (
                _fresh_equivalent_current(
                    self.service.store,
                    cell,
                    catalog=self.catalog,
                    expected_revision=self.source_revision,
                    expected_tree=self.measurement_source_tree,
                    measurement_lineage=self.measurement_lineage,
                    expected_study_contract_sha256=(
                        self.settings.study_contract_sha256
                    ),
                    expected_worker_harness=self.worker_harness_identity,
                )
                if (
                    not self.settings.rerun
                    and self.settings.artifact_policy
                    in {ArtifactPolicy.REUSE, ArtifactPolicy.RETIME}
                )
                else None
            )
            baseline = (
                None
                if planned.baseline_cell_id is None
                else self.catalog.cell(planned.baseline_cell_id)
            )
            dependency_cells = {
                dependency.cell_id: dependency
                for dependency in (
                    *((baseline,) if baseline is not None else ()),
                    *(
                        self.catalog.cell(peer_cell_id)
                        for peer_cell_id in planned.comparison_peer_ids
                    ),
                )
            }
            dependency_records: dict[str, CurrentRecord] = {}
            terminal_dependencies: list[dict[str, object]] = []
            for dependency in dependency_cells.values():
                current_dependency = self._current(dependency)
                if current_dependency is None:
                    return self._publish_skip(
                        cell,
                        f"required dependency {dependency.cell_id!r} is unavailable",
                        current=decision.current,
                    )
                dependency_record, dependency_state = current_dependency
                if dependency_state is PolicyMeasurementState.SUCCESS:
                    dependency_records[dependency.cell_id] = dependency_record
                else:
                    terminal_dependencies.append(
                        dependency_reference(
                            dependency.cell_id,
                            dependency_record.result,
                        )
                    )
            if terminal_dependencies:
                return self._publish_dependency_censor(
                    cell,
                    terminal_dependencies,
                    current=decision.current,
                )
            baseline_record = (
                None if baseline is None else dependency_records[baseline.cell_id]
            )
            peer_records: dict[str, CurrentRecord] = {}
            for peer_cell_id in planned.comparison_peer_ids:
                peer_records[peer_cell_id] = dependency_records[peer_cell_id]

            with self.service.store.new_attempt(
                cell.cell_id,
                self.settings.artifact_policy,
                based_on=decision.current,
            ) as attempt:
                worker_result = attempt.path("worker-result.json")
                worker_log = attempt.path("worker.log")
                generation_timeout = (
                    self.settings.generation_time_limit_seconds
                    if self.settings.campaign_policy is STRICT_POLICY
                    else generation_limit_for_cell(
                        self.settings.campaign_policy,
                        cell,
                    )
                )
                phase_channel = (
                    None
                    if generation_timeout is None
                    else WorkerPhaseChannel.create(
                        attempt.path("worker-phase-state.json")
                    )
                )
                command = [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    os.fspath(self.worker_entrypoint),
                    "--repo-root",
                    os.fspath(self.service.paths.repo_root),
                    *self._service_path_arguments(),
                    *self._worker_harness_arguments(),
                    "_worker",
                    "--cell-id",
                    cell.cell_id,
                    "--attempt-root",
                    os.fspath(attempt.root),
                    "--result-json",
                    os.fspath(worker_result),
                    "--log-path",
                    os.fspath(worker_log),
                    "--target-runtime",
                    str(self.settings.target_runtime_seconds),
                    "--batch-size",
                    str(self.settings.batch_size),
                    "--cell-cores",
                    str(self.settings.cell_cores),
                ]
                if phase_channel is not None:
                    command.extend(
                        (
                            "--phase-state-path",
                            os.fspath(phase_channel.path),
                            "--phase-state-run-id",
                            phase_channel.run_id,
                            "--phase-state-authentication-key",
                            phase_channel.authentication_key,
                        )
                    )
                if baseline_record is not None:
                    command.extend(
                        ("--baseline-json", os.fspath(baseline_record.result_path))
                    )
                for peer_cell_id, peer_record in sorted(peer_records.items()):
                    command.extend(
                        (
                            "--peer-json",
                            peer_cell_id,
                            os.fspath(peer_record.result_path),
                        )
                    )
                prepared_model = self._prepared_model_paths.get(
                    cell.measurement.model  # type: ignore[arg-type]
                )
                if prepared_model is not None:
                    command.extend(("--prepared-model", os.fspath(prepared_model)))
                if (
                    self.settings.campaign_policy.allow_terminal_censors
                    and cell.measurement.execution_mode is ExecutionMode.AMPLICOL
                ):
                    legacy_repository = self._prepare_legacy_workspace(
                        attempt.attempt_id
                    )
                    command.extend(
                        ("--legacy-repository", os.fspath(legacy_repository))
                    )
                reusable_record = (
                    decision.current
                    if (
                        decision.action is ArtifactAction.RETIME_CURRENT
                        and decision.current is not None
                        and current_is_fresh
                        and not self.settings.rerun
                    )
                    else equivalent_record
                )
                if reusable_record is not None:
                    command.extend(
                        (
                            "--reused-measurement-json",
                            os.fspath(reusable_record.result_path),
                        )
                    )
                supervised = supervise_worker(
                    command,
                    timeout_seconds=self.settings.timeout_seconds,
                    generation_timeout_seconds=generation_timeout,
                    phase_channel=phase_channel,
                    max_rss_bytes=self._effective_cell_rss_limit(),
                    environment_overrides=_worker_environment_overrides(
                        self.settings,
                        self.service.paths.coordination_root,
                    ),
                    scrub_import_environment=(
                        self.worker_harness_identity is not None
                    ),
                    working_directory=(
                        self.service.paths.repo_root
                        if self.worker_harness_identity is not None
                        else None
                    ),
                )
                generation_phase = supervised.generation_phase
                resources = _resource_payload(
                    supervised.usage,
                    generation_phase,
                    memory_limit_bytes=supervised.memory_limit_bytes,
                    memory_limit_reason=supervised.memory_limit_reason,
                )
                policy_state: PolicyMeasurementState | None = None
                if (
                    supervised.reason == "generation_timeout"
                    and generation_timeout is not None
                    and self.settings.campaign_policy.allow_terminal_censors
                ):
                    phase_evidence = (
                        None if generation_phase is None else generation_phase.as_dict()
                    )
                    observed_generation = (
                        phase_evidence.get("generation_elapsed_seconds")
                        if isinstance(phase_evidence, Mapping)
                        else None
                    )
                    if (
                        isinstance(phase_evidence, Mapping)
                        and isinstance(observed_generation, (int, float))
                        and not isinstance(observed_generation, bool)
                    ):
                        result = policy_censor_measurement(
                            self.settings.campaign_policy,
                            self.settings.report_profile or "",
                            cell,
                            kind=PolicyCensorKind.GENERATION_LIMIT,
                            source_identity=self.source_identity,
                            resources=resources,
                            observed_generation_seconds=float(observed_generation),
                            phase_evidence=phase_evidence,
                        )
                        policy_state = PolicyMeasurementState.GENERATION_LIMIT
                    else:
                        result = failure_measurement(
                            ResultStatus.ERROR,
                            "generation timeout lacked authenticated phase evidence",
                            resources=resources,
                        )
                elif (
                    supervised.reason == "memory_limit"
                    and self.settings.campaign_policy.allow_terminal_censors
                ):
                    peak_rss = resources.get("peak_rss_bytes")
                    peak_guard = resources.get("peak_guard_bytes")
                    metric_abi = resources.get("memory_metric_abi")
                    limit_reason = resources.get("memory_limit_reason")
                    if (
                        isinstance(peak_rss, int)
                        and not isinstance(peak_rss, bool)
                        and isinstance(peak_guard, int)
                        and not isinstance(peak_guard, bool)
                        and isinstance(metric_abi, str)
                        and isinstance(limit_reason, str)
                    ):
                        result = policy_censor_measurement(
                            self.settings.campaign_policy,
                            self.settings.report_profile or "",
                            cell,
                            kind=PolicyCensorKind.MEMORY_LIMIT,
                            source_identity=self.source_identity,
                            resources=resources,
                            observed_rss_bytes=peak_rss,
                            observed_guard_bytes=peak_guard,
                            memory_metric_abi=metric_abi,
                            memory_limit_reason=limit_reason,
                        )
                        policy_state = PolicyMeasurementState.MEMORY_LIMIT
                    elif (
                        isinstance(peak_rss, int)
                        and not isinstance(peak_rss, bool)
                        and metric_abi is None
                        and self.settings.campaign_policy
                        is not MACBOOK_M3_Z_TABLE_F_POLICY
                    ):
                        result = policy_censor_measurement(
                            self.settings.campaign_policy,
                            self.settings.report_profile or "",
                            cell,
                            kind=PolicyCensorKind.MEMORY_LIMIT,
                            source_identity=self.source_identity,
                            resources=resources,
                            observed_rss_bytes=peak_rss,
                        )
                        policy_state = PolicyMeasurementState.MEMORY_LIMIT
                    else:
                        result = failure_measurement(
                            ResultStatus.ERROR,
                            "memory limit lacked authenticated RSS evidence",
                            resources=resources,
                        )
                elif supervised.reason != "completed":
                    status = {
                        "timeout": ResultStatus.TIMEOUT,
                        "generation_timeout": ResultStatus.TIMEOUT,
                        "memory_limit": ResultStatus.MEMORY_LIMIT,
                        "memory_probe_error": ResultStatus.ERROR,
                        "phase_state_error": ResultStatus.ERROR,
                    }[supervised.reason]
                    result = failure_measurement(
                        status,
                        f"worker terminated by {supervised.reason}",
                        resources=resources,
                    )
                elif supervised.returncode != 0 or not worker_result.is_file():
                    result = failure_measurement(
                        ResultStatus.ERROR,
                        f"worker exited with code {supervised.returncode}",
                        resources=resources,
                    )
                else:
                    raw = json.loads(worker_result.read_text(encoding="ascii"))
                    if not isinstance(raw, Mapping):
                        raise TypeError("worker result must be a JSON object")
                    result = dict(raw)
                    result["resources"] = resources
                    self._bind_study_result(
                        result,
                        require_existing_harness=True,
                    )
                if supervised.reason != "completed" or (
                    supervised.returncode != 0 or not worker_result.is_file()
                ):
                    self._bind_study_result(result)
                validate_measurement(result, expected_cell=cell)
                if result["status"] == ResultStatus.OK.value:
                    policy_state = (
                        PolicyMeasurementState.SUCCESS
                        if self.settings.campaign_policy is STRICT_POLICY
                        else validate_policy_measurement(
                            self.settings.campaign_policy,
                            self.settings.report_profile or "",
                            cell,
                            result,
                            expected_source_revision=self.source_revision,
                            expected_source_tree=self.source_tree,
                        )
                    )
                elif policy_state is not None:
                    validated_state = validate_policy_measurement(
                        self.settings.campaign_policy,
                        self.settings.report_profile or "",
                        cell,
                        result,
                        expected_source_revision=self.source_revision,
                        expected_source_tree=self.source_tree,
                    )
                    if validated_state is not policy_state:
                        raise RuntimeError(
                            "policy censor state changed during validation"
                        )
                _write_json(worker_result, result)
                paths = _attempt_files(attempt.root)
                if policy_state is not None:
                    record = attempt.publish(result, artifact_paths=paths)
                    return CellOutcome(
                        cell.cell_id,
                        (
                            "ok"
                            if policy_state is PolicyMeasurementState.SUCCESS
                            else policy_state.value
                        ),
                        record.attempt_id,
                    )
                attempt.mark_failed(
                    str(result.get("failure")),
                    artifact_paths=paths,
                )
                return CellOutcome(
                    cell.cell_id,
                    str(result["status"]),
                    (
                        attempt.attempt_id
                        if decision.current is None
                        else "previous valid current preserved"
                    ),
                )

    def _publish_dependency_censor(
        self,
        cell: CellSpec,
        dependencies: Sequence[Mapping[str, object]],
        *,
        current: CurrentRecord | None,
    ) -> CellOutcome:
        with self.service.store.new_attempt(
            cell.cell_id,
            self.settings.artifact_policy,
            based_on=current,
        ) as attempt:
            result = policy_censor_measurement(
                self.settings.campaign_policy,
                self.settings.report_profile or "",
                cell,
                kind=PolicyCensorKind.DEPENDENCY,
                source_identity=self.source_identity,
                resources=None,
                dependencies=tuple(
                    sorted(
                        (dict(item) for item in dependencies),
                        key=lambda item: str(item["cell_id"]),
                    )
                ),
            )
            self._bind_study_result(result)
            validate_measurement(result, expected_cell=cell)
            record = attempt.publish(result)
            return CellOutcome(
                cell.cell_id,
                PolicyMeasurementState.DEPENDENCY.value,
                record.attempt_id,
            )

    def _publish_resource_frontier(
        self,
        cell: CellSpec,
        frontier: Mapping[str, object],
        *,
        current: CurrentRecord | None,
    ) -> CellOutcome:
        with self.service.store.new_attempt(
            cell.cell_id,
            self.settings.artifact_policy,
            based_on=current,
        ) as attempt:
            result = policy_censor_measurement(
                self.settings.campaign_policy,
                self.settings.report_profile or "",
                cell,
                kind=PolicyCensorKind.RESOURCE_FRONTIER,
                source_identity=self.source_identity,
                resources=None,
                frontier=frontier,
            )
            self._bind_study_result(result)
            validate_measurement(result, expected_cell=cell)
            record = attempt.publish(result)
            return CellOutcome(
                cell.cell_id,
                PolicyMeasurementState.RESOURCE_FRONTIER.value,
                record.attempt_id,
            )

    def _effective_cell_rss_limit(self) -> int | None:
        limits = [
            limit
            for limit in (
                self.settings.max_rss_bytes,
                (
                    None
                    if self.settings.campaign_max_rss_bytes is None
                    else self.settings.campaign_max_rss_bytes // self.settings.workers
                ),
            )
            if limit is not None
        ]
        return min(limits) if limits else None

    def _publish_skip(
        self,
        cell: CellSpec,
        message: str,
        *,
        current: CurrentRecord | None,
    ) -> CellOutcome:
        with self.service.store.new_attempt(
            cell.cell_id,
            self.settings.artifact_policy,
            based_on=current,
        ) as attempt:
            result = failure_measurement(ResultStatus.SKIP, message)
            self._bind_study_result(result)
            attempt.write_json("worker-result.json", result)
            attempt.mark_failed(
                message,
                artifact_paths=("worker-result.json",),
            )
            return CellOutcome(
                cell.cell_id,
                "skip",
                (
                    attempt.attempt_id
                    if current is None
                    else "previous valid current preserved"
                ),
            )


__all__ = [
    "CampaignResult",
    "CampaignScheduler",
    "CampaignSettings",
    "CellOutcome",
    "CellSelection",
    "PlannedCell",
    "plan_campaign",
    "select_cells",
    "validate_campaign_plan",
]
