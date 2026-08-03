# SPDX-License-Identifier: 0BSD
"""Dependency-aware, resource-supervised performance-report campaigns."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .agreements import (
    incoming_agreement_edges,
    validation_baseline_fallback_peers,
    validation_baseline_is_required,
)
from .artifacts import (
    ArtifactAction,
    ArtifactStore,
    CurrentRecord,
    LockCancelledError,
    LockTimeoutError,
)
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
    policy_measurement_state_hint,
    policy_status_label,
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
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_TERMINATION_GRACE_SECONDS,
    GenerationPhaseEvidence,
    ResourceUsage,
    SupervisedResult,
    WorkerObservation,
    supervise_worker,
)
from .runner import DEFAULT_TARGET_RUNTIME_SECONDS
from .service import CANONICAL_REPORT_ENTRYPOINT, ReportService
from .source_identity import ReportSourceIdentity, require_eligible_report_source
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
    warmup_runs: int = 2
    minimum_samples: int = 5
    timeout_seconds: float | None = None
    generation_time_limit_seconds: float | None = None
    profiling_time_limit_seconds: float | None = None
    validation_time_limit_seconds: float | None = None
    max_rss_bytes: int | None = None
    campaign_max_rss_bytes: int | None = None
    artifact_policy: ArtifactPolicy = ArtifactPolicy.REGENERATE
    missing_only: bool = False
    rerun: bool = False
    allow_symbolica_parallel: bool = False
    resource_sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS
    source_identity_override: ReportSourceIdentity | None = None
    campaign_invocation_id: str | None = None
    progress_observer: Callable[[Mapping[str, object]], None] | None = None
    cancellation_requested: Callable[[], bool] | None = None
    manual_terminal_censors: bool = False
    discard_cancelled_attempts: bool = False
    remove_heavy_attempt_artifacts: bool = False
    campaign_policy: CampaignPolicy = STRICT_POLICY
    report_profile: str | None = None
    study_contract_sha256: str | None = None
    reuse_cross_source_comparison_dependencies: bool = False
    original_amplicol_repository: Path | None = None
    original_amplicol_revision: str | None = None

    def __post_init__(self) -> None:
        if self.workers < 1 or self.cell_cores < 1:
            raise ValueError("workers and cell_cores must be positive")
        if (
            not math.isfinite(self.target_runtime_seconds)
            or self.target_runtime_seconds <= 0.0
            or self.batch_size < 1
        ):
            raise ValueError("target runtime and batch size must be positive")
        if self.warmup_runs < 0 or self.minimum_samples < 5:
            raise ValueError("warmups must be non-negative and samples at least five")
        if (
            not math.isfinite(self.resource_sample_interval_seconds)
            or self.resource_sample_interval_seconds <= 0.0
        ):
            raise ValueError("resource sample interval must be positive")
        if (
            not math.isfinite(self.termination_grace_seconds)
            or self.termination_grace_seconds < 0.0
        ):
            raise ValueError("termination grace must be non-negative")
        if self.manual_terminal_censors and self.campaign_policy is not STRICT_POLICY:
            raise ValueError("manual terminal censors require the strict base policy")
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0
        ):
            raise ValueError("timeout_seconds must be positive")
        if self.generation_time_limit_seconds is not None and (
            self.generation_time_limit_seconds <= 0.0
            or not math.isfinite(self.generation_time_limit_seconds)
        ):
            raise ValueError("generation_time_limit_seconds must be positive")
        for field_name, value in (
            ("profiling_time_limit_seconds", self.profiling_time_limit_seconds),
            ("validation_time_limit_seconds", self.validation_time_limit_seconds),
        ):
            if value is not None and (value <= 0.0 or not math.isfinite(value)):
                raise ValueError(f"{field_name} must be positive")
        if self.max_rss_bytes is not None and self.max_rss_bytes <= 0:
            raise ValueError("max_rss_bytes must be positive")
        if self.campaign_max_rss_bytes is not None and self.campaign_max_rss_bytes <= 0:
            raise ValueError("campaign_max_rss_bytes must be positive")
        if self.campaign_invocation_id is not None and (
            not self.campaign_invocation_id
            or len(self.campaign_invocation_id) > 128
            or not self.campaign_invocation_id.isprintable()
        ):
            raise ValueError(
                "campaign_invocation_id must be 1..128 printable characters"
            )
        if self.missing_only and self.rerun:
            raise ValueError("--missing-only and --rerun are mutually exclusive")
        if (self.original_amplicol_repository is None) != (
            self.original_amplicol_revision is None
        ):
            raise ValueError(
                "original AmpliCol repository and revision must be specified together"
            )
        if (
            self.original_amplicol_revision is not None
            and re.fullmatch(r"[0-9a-f]{40}", self.original_amplicol_revision) is None
        ):
            raise ValueError("original AmpliCol revision must be lowercase 40-hex")
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
            raise ValueError("a study contract SHA-256 requires the Z-table F policy")
        validate_campaign_settings(self.campaign_policy, self)


@dataclass(frozen=True, slots=True)
class PlannedCell:
    cell: CellSpec
    dependency: bool
    baseline_cell_id: str | None
    rank: int
    comparison_peer_ids: tuple[str, ...] = ()
    optional_baseline_cell_id: str | None = None
    optional_comparison_peer_ids: tuple[str, ...] = ()
    force_recompare: bool = False
    prerequisite_cell_ids: tuple[str, ...] = ()
    resource_predecessor_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CellOutcome:
    cell_id: str
    status: str
    detail: str
    prerequisite_cell_ids: tuple[str, ...] = ()


class _CoordinationDeferred(RuntimeError):
    """Signal that a ready cell lost a non-blocking coordination-lock race."""

    def __init__(self, lock_name: str) -> None:
        super().__init__(f"coordination lock is busy: {lock_name}")
        self.lock_name = lock_name


class _PreparationFailed(RuntimeError):
    """Memoized failure of one shared prepared-model prerequisite."""

    def __init__(
        self,
        model: ModelKey,
        reason: str,
        detail: str,
        supervised: SupervisedResult | None = None,
    ) -> None:
        super().__init__(detail)
        self.model = model
        self.reason = reason
        self.detail = detail
        self.supervised = supervised


def _artifact_consumer_cell_ids(
    cell: CellSpec,
    *,
    catalog: ReportCatalog,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                cell.cell_id,
                *(candidate.cell_id for candidate in catalog.equivalent_cells(cell)),
            }
        )
    )


def archive_cell_attempt_history(
    service: ReportService,
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[str, ...]:
    """Archive one cell's obsolete attempts while preserving live owners."""

    return service.store.archive_obsolete_attempts(
        cell.cell_id,
        consumer_cell_ids=_artifact_consumer_cell_ids(cell, catalog=catalog),
    )


def reconcile_attempt_history(
    service: ReportService,
    cells: Sequence[CellSpec],
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[tuple[str, str], ...]:
    """Best-effort bounded startup cleanup for an explicit campaign closure."""

    warnings: list[tuple[str, str]] = []
    for cell in sorted(cells, key=lambda candidate: candidate.cell_id):
        try:
            with service.store.named_lock(
                f"campaign-cell-{cell.cell_id}", timeout=0.0
            ):
                archive_cell_attempt_history(service, cell, catalog=catalog)
        except LockTimeoutError:
            continue
        except Exception as error:
            warnings.append((cell.cell_id, f"{type(error).__name__}: {error}"))
    return tuple(warnings)


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


def _symbolica_generation_lock_path(
    settings: CampaignSettings,
    coordination_root: Path,
    cell: CellSpec,
) -> Path | None:
    if (
        settings.allow_symbolica_parallel
        or cell.measurement.model is not ModelKey.UFO_SM
        or cell.measurement.execution_mode is not ExecutionMode.COMPILED
    ):
        return None
    return coordination_root.resolve() / "symbolica-ufo-compiled-generation.lock"


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
                PolicyMeasurementState.WORKER_TIMEOUT.value,
                PolicyMeasurementState.PROFILING_TIMEOUT.value,
                PolicyMeasurementState.VALIDATION_TIMEOUT.value,
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
                raise WorkerHarnessError("the expected worker harness is unavailable")
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
    comparison_dependency: bool = False,
) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
    if settings.campaign_policy is STRICT_POLICY:
        current = _successful_current(
            store,
            cell.cell_id,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            expected_cell=cell,
            measurement_lineage=measurement_lineage,
            expected_study_contract_sha256=(settings.study_contract_sha256),
            expected_worker_harness=expected_worker_harness,
        )
        if current is not None:
            return current, PolicyMeasurementState.SUCCESS
        if not settings.manual_terminal_censors:
            return None
        terminal = store.load_current(cell.cell_id, missing_ok=True)
        if terminal is None:
            return None
        state = policy_measurement_state_hint(terminal.result)
        if state is None or policy_status_label(terminal.result) is None:
            return None
        provenance = terminal.result.get("provenance")
        if expected_revision is not None and (
            not isinstance(provenance, Mapping)
            or provenance.get("report_source_revision") != expected_revision
        ):
            return None
        try:
            validate_measurement(terminal.result, expected_cell=cell)
        except ValueError:
            return None
        return terminal, state
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
    cross_source = (
        expected_source is None
        and comparison_dependency
        and settings.reuse_cross_source_comparison_dependencies
        and isinstance(provenance, Mapping)
        and isinstance(provenance.get("report_source_revision"), str)
        and isinstance(provenance.get("report_source_tree"), str)
    )
    if cross_source:
        expected_source = (
            str(provenance["report_source_revision"]),
            str(provenance["report_source_tree"]),
        )
    if expected_source is None:
        return None
    try:
        validate_measurement(current.result, expected_cell=cell)
        if settings.study_contract_sha256 is not None and not cross_source:
            require_z_table_f_attempt_binding(
                current.result,
                settings.study_contract_sha256,
            )
            if expected_worker_harness is None:
                raise WorkerHarnessError("the expected worker harness is unavailable")
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
    current_resolver: (
        Callable[
            [CellSpec],
            tuple[CurrentRecord, PolicyMeasurementState] | None,
        ]
        | None
    ) = None,
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
        current = (
            current_resolver(candidate)
            if current_resolver is not None
            else _policy_current(
                store,
                candidate,
                settings=settings,
                expected_revision=expected_revision,
                expected_tree=expected_tree,
                measurement_lineage=measurement_lineage,
                original_amplicol_seed=original_amplicol_seed,
                expected_worker_harness=expected_worker_harness,
            )
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


def _partition_dependency_records(
    *,
    baseline_cell_id: str | None,
    comparison_peer_ids: Sequence[str],
    optional_baseline_cell_id: str | None = None,
    optional_comparison_peer_ids: Sequence[str] = (),
    baseline_fallback_peer_ids: Sequence[str] = (),
    currents: Mapping[
        str,
        tuple[CurrentRecord, PolicyMeasurementState],
    ],
) -> tuple[
    CurrentRecord | None,
    dict[str, CurrentRecord],
    tuple[dict[str, object], ...],
]:
    """Partition successful dependencies and retain every terminal blocker."""

    baseline_record: CurrentRecord | None = None
    terminal_required: list[dict[str, object]] = []
    terminal_ids: set[str] = set()
    if baseline_cell_id is not None:
        record, state = currents[baseline_cell_id]
        if state is PolicyMeasurementState.SUCCESS:
            baseline_record = record
        else:
            terminal_required.append(
                dependency_reference(baseline_cell_id, record.result)
            )
            terminal_ids.add(baseline_cell_id)
    elif optional_baseline_cell_id is not None:
        optional = currents.get(optional_baseline_cell_id)
        if optional is not None and optional[1] is PolicyMeasurementState.SUCCESS:
            baseline_record = optional[0]
    peer_records: dict[str, CurrentRecord] = {}
    for peer_cell_id in comparison_peer_ids:
        record, state = currents[peer_cell_id]
        if state is PolicyMeasurementState.SUCCESS:
            peer_records[peer_cell_id] = record
        elif peer_cell_id not in terminal_ids:
            terminal_required.append(
                dependency_reference(peer_cell_id, record.result)
            )
            terminal_ids.add(peer_cell_id)
    for peer_cell_id in optional_comparison_peer_ids:
        optional = currents.get(peer_cell_id)
        if optional is not None and optional[1] is PolicyMeasurementState.SUCCESS:
            peer_records[peer_cell_id] = optional[0]
    if baseline_record is None:
        for peer_cell_id in baseline_fallback_peer_ids:
            fallback = currents.get(peer_cell_id)
            if fallback is not None and fallback[1] is PolicyMeasurementState.SUCCESS:
                baseline_record = fallback[0]
                break
    return baseline_record, peer_records, tuple(terminal_required)


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
            expected_study_contract_sha256=(expected_study_contract_sha256),
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
    current_resolver: (
        Callable[
            [CellSpec],
            tuple[CurrentRecord, PolicyMeasurementState] | None,
        ]
        | None
    ) = None,
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

    def resolve_current(
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if current_resolver is not None:
            return current_resolver(cell)
        return _policy_current(
            store,
            cell,
            settings=settings,
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            measurement_lineage=measurement_lineage,
            original_amplicol_seed=original_amplicol_seed,
            expected_worker_harness=expected_worker_harness,
            comparison_dependency=comparison_dependency,
        )

    def dependency_roles(
        cell: CellSpec,
    ) -> tuple[
        CellSpec | None,
        CellSpec | None,
        tuple[CellSpec, ...],
        tuple[CellSpec, ...],
    ]:
        candidate_baseline = catalog.validation_baseline_cell(cell)
        baseline_required = validation_baseline_is_required(cell, candidate_baseline)
        edges = incoming_agreement_edges(cell, catalog=catalog)
        return (
            candidate_baseline if baseline_required else None,
            candidate_baseline if not baseline_required else None,
            tuple(edge.baseline for edge in edges if edge.required),
            tuple(edge.baseline for edge in edges if not edge.required),
        )

    def dependencies(cell: CellSpec) -> tuple[CellSpec, ...]:
        baseline, _optional_baseline, peers, _optional_peers = dependency_roles(cell)
        return tuple(
            {
                dependency.cell_id: dependency
                for dependency in (
                    *((baseline,) if baseline is not None else ()),
                    *peers,
                )
            }.values()
        )

    def optional_dependencies(cell: CellSpec) -> tuple[CellSpec, ...]:
        _baseline, optional_baseline, _peers, optional_peers = dependency_roles(cell)
        return tuple(
            {
                dependency.cell_id: dependency
                for dependency in (
                    *((optional_baseline,) if optional_baseline is not None else ()),
                    *optional_peers,
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
                if resolve_current(
                    dependency,
                    comparison_dependency=True,
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
            dependency.cell_id: resolve_current(
                dependency,
                comparison_dependency=True,
            )
            for dependency in cell_dependencies
        }
        missing_dependencies = tuple(
            dependency
            for dependency in cell_dependencies
            if dependency_currents[dependency.cell_id] is None
        )
        terminal_dependency_cells = tuple(
            dependency
            for dependency in cell_dependencies
            if (
                dependency_currents[dependency.cell_id] is not None
                and dependency_currents[dependency.cell_id][1]  # type: ignore[index]
                is not PolicyMeasurementState.SUCCESS
            )
        )
        terminal_dependencies = tuple(
            dependency_reference(
                dependency.cell_id,
                dependency_currents[dependency.cell_id][0].result,  # type: ignore[index]
            )
            for dependency in terminal_dependency_cells
            if not settings.manual_terminal_censors
        )
        current = resolve_current(cell)
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
            current_resolver=resolve_current,
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
                fresh_requested = (
                    not missing_dependencies and not terminal_dependency_cells
                )
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
        for dependency in (*dependencies(cell), *optional_dependencies(cell)):
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
        baseline, _optional, _peers, _optional_peers = dependency_roles(cell)
        return None if baseline is None else baseline.cell_id

    def optional_baseline_id(cell: CellSpec) -> str | None:
        _baseline, optional, _peers, _optional_peers = dependency_roles(cell)
        return None if optional is None else optional.cell_id

    def comparison_peer_ids(cell: CellSpec) -> tuple[str, ...]:
        _baseline, _optional, peers, _optional_peers = dependency_roles(cell)
        return tuple(peer.cell_id for peer in peers)

    def optional_comparison_peer_ids(cell: CellSpec) -> tuple[str, ...]:
        _baseline, _optional, _peers, optional_peers = dependency_roles(cell)
        return tuple(peer.cell_id for peer in optional_peers)

    def scheduled_prerequisite_ids(cell: CellSpec) -> tuple[str, ...]:
        return tuple(
            sorted(
                dependency.cell_id
                for dependency in (*dependencies(cell), *optional_dependencies(cell))
                if dependency.cell_id in needed
            )
        )

    def resource_predecessor_ids(cell: CellSpec) -> tuple[str, ...]:
        if not settings.campaign_policy.allow_terminal_censors:
            return ()
        predecessors = tuple(
            candidate
            for candidate in needed.values()
            if candidate.n_final < cell.n_final
            and _resource_lane_key(candidate) == _resource_lane_key(cell)
        )
        if not predecessors:
            return ()
        predecessor = max(
            predecessors,
            key=lambda item: (item.n_final, item.cell_id),
        )
        return (predecessor.cell_id,)

    return tuple(
        PlannedCell(
            cell=cell,
            dependency=cell.cell_id not in requested_ids,
            baseline_cell_id=validation_baseline_id(cell),
            rank=planned_rank(cell),
            comparison_peer_ids=comparison_peer_ids(cell),
            optional_baseline_cell_id=optional_baseline_id(cell),
            optional_comparison_peer_ids=optional_comparison_peer_ids(cell),
            force_recompare=cell.cell_id in force_recompare_ids,
            prerequisite_cell_ids=scheduled_prerequisite_ids(cell),
            resource_predecessor_ids=resource_predecessor_ids(cell),
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
    baseline_required = validation_baseline_is_required(planned[0].cell, baseline)
    expected_baseline_id = (
        baseline.cell_id if baseline is not None and baseline_required else None
    )
    expected_optional_baseline_id = (
        baseline.cell_id if baseline is not None and not baseline_required else None
    )
    edges = incoming_agreement_edges(planned[0].cell, catalog=catalog)
    expected_peer_ids = tuple(
        edge.baseline.cell_id for edge in edges if edge.required
    )
    expected_optional_peer_ids = tuple(
        edge.baseline.cell_id for edge in edges if not edge.required
    )
    if (
        planned[0].baseline_cell_id != expected_baseline_id
        or planned[0].comparison_peer_ids != expected_peer_ids
        or planned[0].optional_baseline_cell_id
        != expected_optional_baseline_id
        or planned[0].optional_comparison_peer_ids
        != expected_optional_peer_ids
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
    supervised: SupervisedResult | None = None,
    source_revision: str | None = None,
    campaign_invocation_id: str | None = None,
    supervisor_phase: str | None = None,
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
    if supervised is not None:
        payload["supervisor"] = {
            "abi": "pyamplicol-report-worker-supervisor-v1",
            "reason": supervised.reason,
            "returncode": supervised.returncode,
            "signal_number": supervised.signal_number,
            "signal_name": supervised.signal_name,
            "pid": supervised.pid,
            "member_pids": list(supervised.member_pids),
            "phase": (
                supervisor_phase
                if supervisor_phase is not None
                else (
                    None
                    if generation_phase is None
                    else generation_phase.final_phase
                )
            ),
            "phase_state_error": supervised.phase_state_error,
            "stderr": supervised.supervisor_stderr,
            "stderr_truncated": supervised.supervisor_stderr_truncated,
            "stderr_limit_bytes": supervised.supervisor_stderr_limit_bytes,
            "source_revision": source_revision,
            "campaign_invocation_id": campaign_invocation_id,
            "started_at_utc": supervised.started_at_utc,
            "finished_at_utc": supervised.finished_at_utc,
        }
    return payload


class WorkerProcessExitError(RuntimeError):
    """A worker process exited independently of a supervisor resource cap."""


def _worker_process_exit_error(supervised: SupervisedResult) -> WorkerProcessExitError:
    if supervised.signal_name is not None and supervised.signal_number is not None:
        detail = (
            f"worker terminated by {supervised.signal_name} "
            f"(signal {supervised.signal_number}, return code {supervised.returncode})"
        )
    else:
        detail = f"worker exited with code {supervised.returncode}"
    return WorkerProcessExitError(detail)


def _attempt_files(
    root: Path,
    *,
    include_heavy_artifact: bool = True,
) -> tuple[str, ...]:
    members: list[str] = []
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.name in {"manifest.json", "result.json"}
            or path.is_symlink()
        ):
            continue
        relative = path.relative_to(root)
        if not include_heavy_artifact and relative.parts[0] == "artifact":
            continue
        members.append(relative.as_posix())
    return tuple(sorted(members))


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
        self.source_identity = (
            settings.source_identity_override
            if settings.source_identity_override is not None
            else require_eligible_report_source(service.paths.repo_root)
        )
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
            if authenticated_contract.get("sha256") != settings.study_contract_sha256:
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
                raise StudyContractError("the Z-table F worker identities are missing")
            self.worker_wrapper_root = study_contract_wrapper_root.expanduser().resolve(
                strict=True
            )
            self.worker_entrypoint = self.worker_wrapper_root / POLICY_ENTRYPOINT
            self.worker_module: str | None = None
            if (
                self.worker_entrypoint.is_symlink()
                or not self.worker_entrypoint.is_file()
            ):
                raise StudyContractError(
                    "the authenticated policy-wrapper entrypoint is unavailable"
                )
            self.worker_harness_identity = z_table_f_worker_harness_identity(
                authenticated_contract
            )
        else:
            if study_contract is not None or study_contract_wrapper_root is not None:
                raise StudyContractError(
                    "a study contract authorization requires the Z-table F policy"
                )
            self.study_contract = None
            self.worker_wrapper_root = service.paths.repo_root
            source_entrypoint = service.paths.repo_root / CANONICAL_REPORT_ENTRYPOINT
            if __package__.startswith("pyamplicol."):
                self.worker_entrypoint: Path | None = None
                self.worker_module = "pyamplicol._performance_report"
            else:
                self.worker_entrypoint = source_entrypoint
                self.worker_module = None
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
        self._prepared_model_failures: dict[ModelKey, _PreparationFailed] = {}
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

    def _observe(self, event: str, cell: CellSpec, **values: object) -> None:
        callback = self.settings.progress_observer
        if callback is None:
            return
        # Dashboard/lease updates are informational and must not change a
        # campaign result.
        with suppress(Exception):
            callback({"event": event, "cell_id": cell.cell_id, **values})

    def _observe_schedule(
        self,
        *,
        ready: int,
        waiting_dependency: int,
        waiting_coordination_lock: int,
    ) -> None:
        callback = self.settings.progress_observer
        if callback is None:
            return
        with suppress(Exception):
            callback(
                {
                    "event": "scheduler-state",
                    "ready": ready,
                    "waiting_dependency": waiting_dependency,
                    "waiting_coordination_lock": waiting_coordination_lock,
                }
            )

    def _cancelled(self) -> bool:
        callback = self.settings.cancellation_requested
        return callback is not None and callback()

    def _attach_manual_provenance(
        self,
        result: dict[str, object],
        *,
        cell: CellSpec,
        censor: Mapping[str, object] | None = None,
        progress_path: Path | None = None,
    ) -> None:
        if self.settings.source_identity_override is None:
            return
        # Imported lazily to keep the ordinary report scheduler independent of
        # the optional manual UI during module initialization.
        from .manual_campaign import reproduction_recipe

        recipe = reproduction_recipe(
            cell,
            repo_root=self.service.paths.repo_root,
            artifact_root=self.service.paths.artifact_root,
            cores=self.settings.cell_cores,
            target_runtime=self.settings.target_runtime_seconds,
            batch_size=self.settings.batch_size,
            warmups=self.settings.warmup_runs,
            minimum_samples=self.settings.minimum_samples,
            measurement=result,
        )
        existing = result.get("provenance")
        result["provenance"] = {
            **({} if not isinstance(existing, Mapping) else dict(existing)),
            **self.source_identity.provenance(),
            "manual_campaign": {
                "cell_identity": {
                    "cell_id": cell.cell_id,
                    "dataset_id": cell.dataset_id,
                    "process_key": cell.process_key,
                    "process": cell.process,
                    "n_final": cell.n_final,
                    "workload": cell.workload.value,
                    "execution_mode": cell.measurement.execution_mode.value,
                    "model": (
                        None
                        if cell.measurement.model is None
                        else cell.measurement.model.value
                    ),
                    "accuracy": cell.measurement.accuracy.value,
                    "backend": cell.measurement.backend,
                    "variant": cell.variant,
                },
                "generation_limit_seconds": (
                    self.settings.generation_time_limit_seconds
                ),
                "worker_wall_limit_seconds": self.settings.timeout_seconds,
                "profiling_time_limit_seconds": (
                    self.settings.profiling_time_limit_seconds
                ),
                "validation_time_limit_seconds": (
                    self.settings.validation_time_limit_seconds
                ),
                "memory_limit_bytes": self._effective_cell_rss_limit(),
                "workers": self.settings.workers,
                "cores_per_worker": self.settings.cell_cores,
                "target_runtime_seconds": self.settings.target_runtime_seconds,
                "warmup_runs": self.settings.warmup_runs,
                "minimum_samples": self.settings.minimum_samples,
                "batch_size": self.settings.batch_size,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "public_cli_reproduction": recipe.as_dict(),
            },
            **({} if censor is None else {"policy_censor": dict(censor)}),
        }
        from .phase_timeline import build_phase_timeline

        persisted_provenance = result["provenance"]
        if not isinstance(persisted_provenance, dict):
            raise TypeError("manual result provenance must be mutable")
        manual = persisted_provenance["manual_campaign"]
        if not isinstance(manual, dict):
            raise TypeError("manual campaign provenance must be mutable")
        manual["phase_timeline"] = build_phase_timeline(
            result,
            progress_path=progress_path,
        )

    def _manual_terminal_result(
        self,
        *,
        cell: CellSpec,
        reason: str,
        resources: Mapping[str, object],
        progress_path: Path | None = None,
    ) -> tuple[dict[str, object], PolicyMeasurementState]:
        state, censor = self._manual_terminal_contract(reason)
        status = (
            ResultStatus.MEMORY_LIMIT
            if reason == "memory_limit"
            else ResultStatus.TIMEOUT
        )
        result = failure_measurement(
            status,
            f"worker terminated by {reason}",
            resources=resources,
        )
        self._attach_manual_provenance(
            result,
            cell=cell,
            censor=censor,
            progress_path=progress_path,
        )
        return result, state

    def _manual_terminal_contract(
        self,
        reason: str,
    ) -> tuple[PolicyMeasurementState, dict[str, object]]:
        if reason == "generation_timeout":
            state = PolicyMeasurementState.GENERATION_LIMIT
            censor = {
                "kind": PolicyCensorKind.GENERATION_LIMIT.value,
                "terminal_reason": reason,
                "generation_limit_seconds": (
                    self.settings.generation_time_limit_seconds
                ),
            }
        elif reason == "memory_limit":
            state = PolicyMeasurementState.MEMORY_LIMIT
            censor = {
                "kind": PolicyCensorKind.MEMORY_LIMIT.value,
                "terminal_reason": reason,
                "memory_limit_bytes": self._effective_cell_rss_limit(),
            }
        elif reason == "worker_timeout":
            state = PolicyMeasurementState.WORKER_TIMEOUT
            censor = {
                "kind": PolicyCensorKind.WORKER_TIMEOUT.value,
                "terminal_reason": reason,
                "worker_wall_limit_seconds": self.settings.timeout_seconds,
            }
        elif reason == "profiling_timeout":
            state = PolicyMeasurementState.PROFILING_TIMEOUT
            censor = {
                "kind": PolicyCensorKind.PROFILING_TIMEOUT.value,
                "terminal_reason": reason,
                "profiling_time_limit_seconds": (
                    self.settings.profiling_time_limit_seconds
                ),
            }
        elif reason == "validation_timeout":
            state = PolicyMeasurementState.VALIDATION_TIMEOUT
            censor = {
                "kind": PolicyCensorKind.VALIDATION_TIMEOUT.value,
                "terminal_reason": reason,
                "validation_time_limit_seconds": (
                    self.settings.validation_time_limit_seconds
                ),
            }
        else:
            raise ValueError(f"unsupported manual terminal reason {reason!r}")
        return state, censor

    def _current(
        self,
        cell: CellSpec,
        *,
        comparison_dependency: bool = False,
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
            comparison_dependency=comparison_dependency,
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
        edges = incoming_agreement_edges(
            item.cell,
            catalog=self.catalog,
        )
        dependencies = (
            *(
                (baseline,)
                if validation_baseline_is_required(item.cell, baseline)
                else ()
            ),
            *(edge.baseline for edge in edges if edge.required),
        )
        missing = tuple(
            dependency.cell_id
            for dependency in dependencies
            if self._current(dependency, comparison_dependency=True) is None
        )
        if missing:
            raise StudyContractError(
                "the Z-table F runnable cell has unauthenticated comparison "
                f"dependencies: {', '.join(sorted(missing))}"
            )

    def _legacy_workspace_paths(self, attempt_id: str) -> tuple[Path, Path]:
        """Resolve legacy copy paths without doing controller-side work."""

        from .legacy import MaintainedLegacyApi

        api = MaintainedLegacyApi()
        source = (
            (
                api.default_repository
                if self.settings.original_amplicol_repository is None
                else self.settings.original_amplicol_repository
            )
            .expanduser()
            .resolve(strict=True)
        )
        destination = (
            self.service.paths.artifact_root / "legacy-workspaces" / attempt_id
        )
        return source, destination

    @staticmethod
    def _remove_legacy_workspace(destination: Path) -> None:
        if not destination.exists():
            return
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError(
                "legacy worker workspace is not its canonical regular directory: "
                f"{destination}"
            )
        shutil.rmtree(destination)

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

    def _worker_invocation(self) -> tuple[str, ...]:
        if self.worker_module is not None:
            return (
                sys.executable,
                "-I",
                "-B",
                "-m",
                self.worker_module,
            )
        if self.worker_entrypoint is None:
            raise RuntimeError("performance-report worker entrypoint is unavailable")
        return (
            sys.executable,
            "-I",
            "-S",
            "-B",
            os.fspath(self.worker_entrypoint),
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
            preflight_attempt_id = f"prepared-model-{model.value}"
            representative = next(
                item
                for item in planned
                if item.cell.measurement.model is model
                and item.cell.measurement.execution_mode
                in {ExecutionMode.EAGER, ExecutionMode.RECURRENCE}
            )
            result_path = (
                self.service.paths.artifact_root
                / "prepared-models"
                / f".preflight-{model.value}-{uuid.uuid4().hex}.json"
            )
            progress_path = result_path.with_suffix(".progress.jsonl")
            command = (
                *self._worker_invocation(),
                "--repo-root",
                os.fspath(self.service.paths.repo_root),
                *self._service_path_arguments(),
                *self._worker_harness_arguments(),
                "_prepare",
                "--model",
                model.value,
                "--result-json",
                os.fspath(result_path),
                "--progress-jsonl",
                os.fspath(progress_path),
                "--cell-cores",
                str(self.settings.cell_cores),
                *(
                    (
                        "--producer-revision",
                        self.source_revision,
                    )
                    if self.settings.source_identity_override is not None
                    else ()
                ),
            )
            preflight_timeout = self.settings.generation_time_limit_seconds
            if self.settings.timeout_seconds is not None:
                preflight_timeout = (
                    self.settings.timeout_seconds
                    if preflight_timeout is None
                    else min(preflight_timeout, self.settings.timeout_seconds)
                )

            self._observe(
                "started",
                representative.cell,
                dependency=representative.dependency,
            )
            self._observe(
                "worker",
                representative.cell,
                attempt_id=preflight_attempt_id,
                progress_path=os.fspath(progress_path),
            )

            def observe_preflight_resources(
                observation: WorkerObservation,
                *,
                cell: CellSpec = representative.cell,
                attempt_id: str = preflight_attempt_id,
            ) -> None:
                self._observe(
                    "resource",
                    cell,
                    attempt_id=attempt_id,
                    pid=observation.pid,
                    member_pids=list(observation.member_pids),
                    phase="preparation",
                    phase_sequence=None,
                    wall_seconds=observation.usage.wall_seconds,
                    cpu_seconds=observation.usage.cpu_seconds,
                    current_rss_bytes=observation.usage.current_rss_bytes,
                    peak_rss_bytes=observation.usage.peak_rss_bytes,
                    current_physical_footprint_bytes=(
                        observation.usage.current_physical_footprint_bytes
                    ),
                    peak_physical_footprint_bytes=(
                        observation.usage.peak_physical_footprint_bytes
                    ),
                    current_guard_bytes=observation.usage.current_guard_bytes,
                    peak_guard_bytes=observation.usage.peak_guard_bytes,
                    child_count=observation.usage.child_count,
                )

            try:
                supervised = supervise_worker(
                    command,
                    timeout_seconds=preflight_timeout,
                    max_rss_bytes=self._effective_cell_rss_limit(),
                    interval_seconds=(self.settings.resource_sample_interval_seconds),
                    termination_grace_seconds=self.settings.termination_grace_seconds,
                    cancellation_requested=self.settings.cancellation_requested,
                    observation_callback=observe_preflight_resources,
                    environment_overrides=_worker_environment_overrides(
                        self.settings,
                        self.service.paths.coordination_root,
                    ),
                    scrub_import_environment=(self.worker_harness_identity is not None),
                    working_directory=(
                        self.service.paths.repo_root
                        if self.worker_harness_identity is not None
                        else None
                    ),
                    capture_stderr=True,
                )
                if (
                    supervised.reason != "completed"
                    or supervised.returncode != 0
                    or not result_path.is_file()
                ):
                    effective_reason = supervised.reason
                    if supervised.reason == "worker_timeout":
                        generation_limit = self.settings.generation_time_limit_seconds
                        worker_limit = self.settings.timeout_seconds
                        effective_reason = (
                            "generation_timeout"
                            if generation_limit is not None
                            and (
                                worker_limit is None or generation_limit <= worker_limit
                            )
                            else "worker_timeout"
                        )
                    detail = None
                    if result_path.is_file():
                        try:
                            detail = json.loads(
                                result_path.read_text(encoding="ascii")
                            ).get("error")
                        except (AttributeError, json.JSONDecodeError, OSError):
                            detail = None
                    process_detail = (
                        str(_worker_process_exit_error(supervised))
                        if supervised.reason == "worker_exit"
                        else (
                            f"reason={supervised.reason}, "
                            f"exit={supervised.returncode}"
                        )
                    )
                    message = (
                        f"{model.value} prepared-model preflight failed: "
                        f"{process_detail}, detail={detail}"
                    )
                    raise _PreparationFailed(
                        model,
                        effective_reason,
                        message,
                        supervised,
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
                self._observe(
                    "preflight-finished",
                    representative.cell,
                    attempt_id=preflight_attempt_id,
                )
                result_path.unlink(missing_ok=True)
                progress_path.unlink(missing_ok=True)

    def _prepare_model_for(self, planned: PlannedCell) -> None:
        cell = planned.cell
        model = cell.measurement.model
        if cell.measurement.execution_mode not in {
            ExecutionMode.EAGER,
            ExecutionMode.RECURRENCE,
        } or model not in {ModelKey.BUILTIN_SM, ModelKey.UFO_SM}:
            return
        assert model is not None
        if model in self._prepared_model_paths:
            return
        failure = self._prepared_model_failures.get(model)
        if failure is not None:
            raise failure
        lock_name = f"campaign-prepared-model-{model.value}"
        try:
            with self.service.store.named_lock(
                lock_name,
                timeout=0.0,
                cancellation_requested=self.settings.cancellation_requested,
            ):
                if model in self._prepared_model_paths:
                    return
                failure = self._prepared_model_failures.get(model)
                if failure is not None:
                    raise failure
                try:
                    self._ensure_prepared_model((planned,))
                except _PreparationFailed as error:
                    self._prepared_model_failures[model] = error
                    raise
                except Exception as error:
                    failure = _PreparationFailed(
                        model,
                        "error",
                        f"{type(error).__name__}: {error}",
                    )
                    self._prepared_model_failures[model] = failure
                    raise failure from error
        except LockTimeoutError as error:
            raise _CoordinationDeferred(lock_name) from error

    def _preparation_failure_outcome(
        self,
        cell: CellSpec,
        failure: _PreparationFailed,
    ) -> CellOutcome:
        if failure.reason == "cancelled":
            return CellOutcome(cell.cell_id, "cancelled", failure.detail)
        supervised = failure.supervised
        manual_reason = {
            "generation_timeout": "generation_timeout",
            "worker_timeout": "worker_timeout",
            "memory_limit": "memory_limit",
        }.get(failure.reason)
        if (
            self.settings.manual_terminal_censors
            and supervised is not None
            and manual_reason is not None
        ):
            resources = _resource_payload(
                supervised.usage,
                None,
                memory_limit_bytes=supervised.memory_limit_bytes,
                memory_limit_reason=supervised.memory_limit_reason,
                supervised=supervised,
                source_revision=self.source_revision,
                campaign_invocation_id=self.settings.campaign_invocation_id,
            )
            resources["terminal_reason"] = manual_reason
            with self.service.store.new_attempt(
                cell.cell_id,
                self.settings.artifact_policy,
                based_on=self.service.store.load_current(
                    cell.cell_id,
                    missing_ok=True,
                ),
            ) as attempt:
                result, state = self._manual_terminal_result(
                    cell=cell,
                    reason=manual_reason,
                    resources=resources,
                )
                validate_measurement(result, expected_cell=cell)
                record = attempt.publish(result)
                return CellOutcome(cell.cell_id, state.value, record.attempt_id)
        if supervised is not None and failure.reason == "worker_exit":
            resources = _resource_payload(
                supervised.usage,
                supervised.generation_phase,
                memory_limit_bytes=supervised.memory_limit_bytes,
                memory_limit_reason=supervised.memory_limit_reason,
                supervised=supervised,
                source_revision=self.source_revision,
                campaign_invocation_id=self.settings.campaign_invocation_id,
                supervisor_phase="preparation",
            )
            result = failure_measurement(
                ResultStatus.ERROR,
                _worker_process_exit_error(supervised),
                resources=resources,
            )
            self._attach_manual_provenance(result, cell=cell)
            self._bind_study_result(result)
            validate_measurement(result, expected_cell=cell)
            with self.service.store.new_attempt(
                cell.cell_id,
                self.settings.artifact_policy,
                based_on=self.service.store.load_current(
                    cell.cell_id,
                    missing_ok=True,
                ),
            ) as attempt:
                attempt.write_json("worker-result.json", result)
                attempt.mark_failed(
                    str(result.get("failure")),
                    artifact_paths=("worker-result.json",),
                )
                return CellOutcome(
                    cell.cell_id,
                    ResultStatus.ERROR.value,
                    attempt.attempt_id,
                )
        return CellOutcome(
            cell.cell_id,
            "preparation_error",
            failure.detail,
        )

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
        if self._cancelled():
            return CampaignResult(planned=ordered, outcomes=())
        if not ordered:
            return CampaignResult(planned=ordered, outcomes=())

        by_id = {item.cell.cell_id: item for item in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("campaign plan contains duplicate cell IDs")
        planned_ids = frozenset(by_id)
        prerequisites: dict[str, set[str]] = {}
        dependents: dict[str, set[str]] = {cell_id: set() for cell_id in planned_ids}
        for item in ordered:
            declared = set(item.prerequisite_cell_ids) | set(
                item.resource_predecessor_ids
            )
            # Hand-written/older plans did not carry explicit DAG edges. Preserve
            # their dependency semantics while plan_campaign now records them.
            declared.update(
                cell_id
                for cell_id in (
                    item.baseline_cell_id,
                    *item.comparison_peer_ids,
                )
                if cell_id is not None and cell_id in planned_ids
            )
            missing = declared - planned_ids
            if missing:
                raise ValueError(
                    f"campaign plan for {item.cell.cell_id!r} references "
                    f"unscheduled prerequisites: {', '.join(sorted(missing))}"
                )
            if item.cell.cell_id in declared:
                raise ValueError(
                    "campaign plan contains a self dependency for "
                    f"{item.cell.cell_id!r}"
                )
            prerequisites[item.cell.cell_id] = declared
            for predecessor_id in declared:
                dependents[predecessor_id].add(item.cell.cell_id)

        ready: list[tuple[int, str]] = [
            (item.rank, item.cell.cell_id)
            for item in ordered
            if not prerequisites[item.cell.cell_id]
        ]
        heapq.heapify(ready)
        queued_ids = {cell_id for _rank_value, cell_id in ready}
        if not ready and ordered:
            raise ValueError("campaign plan contains a dependency cycle")

        outcomes: list[CellOutcome] = []
        futures: dict[Future[CellOutcome], PlannedCell] = {}
        waiting_locks: dict[str, float] = {}
        retry_delay_seconds = min(
            0.05,
            self.settings.resource_sample_interval_seconds,
        )

        def observe_schedule() -> None:
            self._observe_schedule(
                ready=len(queued_ids - waiting_locks.keys()),
                waiting_dependency=sum(bool(value) for value in prerequisites.values()),
                waiting_coordination_lock=len(waiting_locks),
            )

        def pop_lockable_candidate(now: float) -> PlannedCell | None:
            deferred: list[tuple[int, str]] = []
            selected: PlannedCell | None = None
            while ready:
                key = heapq.heappop(ready)
                retry_at = waiting_locks.get(key[1])
                if retry_at is not None and retry_at > now:
                    deferred.append(key)
                    continue
                selected = by_id[key[1]]
                queued_ids.remove(key[1])
                waiting_locks.pop(key[1], None)
                break
            for key in deferred:
                heapq.heappush(ready, key)
            return selected

        observe_schedule()
        with ThreadPoolExecutor(max_workers=self.settings.workers) as executor:
            while futures or queued_ids:
                cancelled = self._cancelled()
                now = time.monotonic()
                while not cancelled and len(futures) < self.settings.workers:
                    item = pop_lockable_candidate(now)
                    if item is None:
                        break
                    futures[executor.submit(self._run_cell, item)] = item
                observe_schedule()

                if not futures:
                    if cancelled:
                        break
                    if not queued_ids:
                        break
                    next_retry = min(waiting_locks.values(), default=now)
                    time.sleep(max(0.001, min(retry_delay_seconds, next_retry - now)))
                    continue

                completed, _pending = wait(
                    tuple(futures),
                    timeout=retry_delay_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    completed,
                    key=lambda candidate: futures[candidate].cell.cell_id,
                ):
                    item = futures.pop(future)
                    cell_id = item.cell.cell_id
                    try:
                        outcome = future.result()
                    except _CoordinationDeferred:
                        waiting_locks[cell_id] = time.monotonic() + retry_delay_seconds
                        queued_ids.add(cell_id)
                        heapq.heappush(ready, (item.rank, cell_id))
                        continue
                    outcomes.append(outcome)
                    prerequisites.pop(cell_id, None)
                    waiting_locks.pop(cell_id, None)
                    for dependent_id in sorted(dependents[cell_id]):
                        remaining = prerequisites[dependent_id]
                        remaining.discard(cell_id)
                        if not remaining and dependent_id not in queued_ids:
                            dependent = by_id[dependent_id]
                            queued_ids.add(dependent_id)
                            heapq.heappush(ready, (dependent.rank, dependent_id))
                if not futures and not queued_ids and prerequisites:
                    blocked = ", ".join(sorted(prerequisites))
                    raise ValueError(
                        "campaign plan contains an unresolved dependency cycle: "
                        f"{blocked}"
                    )
            observe_schedule()
        return CampaignResult(
            planned=ordered,
            outcomes=tuple(sorted(outcomes, key=lambda item: item.cell_id)),
        )

    def _coordination_lock_names(self, cell: CellSpec) -> tuple[str, ...]:
        names = {f"campaign-cell-{cell.cell_id}"}
        if self.settings.campaign_policy.allow_terminal_censors:
            names.add(_resource_lane_lock_name(cell))
        return tuple(sorted(names))

    def _run_cell(self, planned: PlannedCell) -> CellOutcome:
        cell = planned.cell
        if self._cancelled():
            outcome = CellOutcome(cell.cell_id, "cancelled", "not started")
            self._observe(
                "finished",
                cell,
                status=outcome.status,
                detail=outcome.detail,
            )
            return outcome
        completed_at_ns: int | None = None
        completion_observed = False
        try:
            with ExitStack() as locks:
                for lock_name in self._coordination_lock_names(cell):
                    try:
                        locks.enter_context(
                            self.service.store.named_lock(
                                lock_name,
                                timeout=0.0,
                                cancellation_requested=(
                                    self.settings.cancellation_requested
                                ),
                            )
                        )
                    except LockTimeoutError as error:
                        raise _CoordinationDeferred(lock_name) from error
                try:
                    try:
                        outcome = self._run_cell_in_lane(planned)
                    except _PreparationFailed as error:
                        # Publishing the shared-preparation terminal must remain
                        # inside the same cell/lane exclusion as ordinary results.
                        outcome = self._preparation_failure_outcome(cell, error)
                    completed_at_ns = time.time_ns()
                    completion_observed = True
                    self._observe(
                        "finished",
                        cell,
                        status=outcome.status,
                        detail=outcome.detail,
                        prerequisite_cell_ids=outcome.prerequisite_cell_ids,
                        completed_at_ns=completed_at_ns,
                    )
                except (_CoordinationDeferred, LockCancelledError):
                    raise
                except BaseException as error:
                    if not completion_observed:
                        completed_at_ns = time.time_ns()
                        completion_observed = True
                        self._observe(
                            "finished",
                            cell,
                            status="error",
                            detail=f"{type(error).__name__}: {error}",
                            completed_at_ns=completed_at_ns,
                        )
                    raise
                finally:
                    if self.settings.remove_heavy_attempt_artifacts:
                        try:
                            archive_cell_attempt_history(
                                self.service,
                                cell,
                                catalog=self.catalog,
                            )
                        except Exception as error:
                            self._observe(
                                "cleanup-warning",
                                cell,
                                detail=f"{type(error).__name__}: {error}",
                            )
        except _CoordinationDeferred:
            # A lock race is scheduling state, not a worker failure. The ready
            # queue will defer and retry this cell without emitting completion.
            raise
        except LockCancelledError:
            outcome = CellOutcome(cell.cell_id, "cancelled", "lock wait cancelled")
            completed_at_ns = time.time_ns()
        except BaseException as error:
            if not completion_observed:
                self._observe(
                    "finished",
                    cell,
                    status="error",
                    detail=f"{type(error).__name__}: {error}",
                    completed_at_ns=(completed_at_ns or time.time_ns()),
                )
            raise
        assert completed_at_ns is not None
        if not completion_observed:
            self._observe(
                "finished",
                cell,
                status=outcome.status,
                detail=outcome.detail,
                prerequisite_cell_ids=outcome.prerequisite_cell_ids,
                completed_at_ns=completed_at_ns,
            )
        return outcome

    def _run_cell_in_lane(self, planned: PlannedCell) -> CellOutcome:
        if self.settings.campaign_policy is MACBOOK_M3_Z_TABLE_F_POLICY:
            self._validate_z_table_f_plan((planned,))
        cell = planned.cell
        # Coordination locks are acquired non-blockingly by _run_cell before a
        # worker slot performs any artifact or dependency work.
        with ExitStack() as lane_locks:
            fresh = self._current(cell)
            frontier_source = (
                _resource_frontier_source(
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
                if planned.dependency
                else None
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
            optional_baseline = (
                None
                if planned.optional_baseline_cell_id is None
                else self.catalog.cell(planned.optional_baseline_cell_id)
            )
            required_dependency_ids = {
                dependency_id
                for dependency_id in (
                    planned.baseline_cell_id,
                    *planned.comparison_peer_ids,
                )
                if dependency_id is not None
            }
            dependency_cells = {
                dependency.cell_id: dependency
                for dependency in (
                    *((baseline,) if baseline is not None else ()),
                    *((optional_baseline,) if optional_baseline is not None else ()),
                    *(
                        self.catalog.cell(peer_cell_id)
                        for peer_cell_id in (
                            *planned.comparison_peer_ids,
                            *planned.optional_comparison_peer_ids,
                        )
                    ),
                )
            }
            dependency_currents: dict[
                str,
                tuple[CurrentRecord, PolicyMeasurementState],
            ] = {}
            missing_dependency_ids: list[str] = []
            for dependency in dependency_cells.values():
                current_dependency = self._current(
                    dependency,
                    comparison_dependency=True,
                )
                if (
                    current_dependency is None
                    and dependency.cell_id in required_dependency_ids
                ):
                    missing_dependency_ids.append(dependency.cell_id)
                elif current_dependency is not None:
                    dependency_currents[dependency.cell_id] = current_dependency
            if missing_dependency_ids:
                return self._publish_blocked_dependency(
                    cell,
                    missing_dependency_ids,
                    current=decision.current,
                )
            if self.settings.manual_terminal_censors:
                terminal_dependency_ids = tuple(
                    dependency_id
                    for dependency_id, (_record, state) in dependency_currents.items()
                    if dependency_id in required_dependency_ids
                    if state is not PolicyMeasurementState.SUCCESS
                )
                if terminal_dependency_ids:
                    return self._publish_blocked_dependency(
                        cell,
                        terminal_dependency_ids,
                        current=decision.current,
                    )
            baseline_record, peer_records, terminal_dependencies = (
                _partition_dependency_records(
                    baseline_cell_id=(None if baseline is None else baseline.cell_id),
                    comparison_peer_ids=planned.comparison_peer_ids,
                    optional_baseline_cell_id=(
                        None
                        if optional_baseline is None
                        else optional_baseline.cell_id
                    ),
                    optional_comparison_peer_ids=(
                        planned.optional_comparison_peer_ids
                    ),
                    baseline_fallback_peer_ids=tuple(
                        peer.cell_id
                        for peer in validation_baseline_fallback_peers(
                            cell,
                            catalog=self.catalog,
                        )
                    ),
                    currents=dependency_currents,
                )
            )
            if terminal_dependencies:
                return self._publish_dependency_censor(
                    cell,
                    terminal_dependencies,
                    current=decision.current,
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
                artifact_use_lock = (
                    "campaign-artifact-use-" f"{reusable_record.attempt_id}"
                )
                try:
                    lane_locks.enter_context(
                        self.service.store.named_lock(
                            artifact_use_lock,
                            timeout=0.0,
                            cancellation_requested=(
                                self.settings.cancellation_requested
                            ),
                        )
                    )
                except LockTimeoutError as error:
                    raise _CoordinationDeferred(artifact_use_lock) from error
                artifact = reusable_record.result.get("artifact")
                artifact_path = (
                    artifact.get("path")
                    if isinstance(artifact, Mapping)
                    else None
                )
                if isinstance(artifact_path, str) and not Path(artifact_path).is_dir():
                    reusable_record = None
            if reusable_record is None:
                self._prepare_model_for(planned)
            self._observe("started", cell, dependency=planned.dependency)

            with self.service.store.new_attempt(
                cell.cell_id,
                self.settings.artifact_policy,
                based_on=decision.current,
            ) as attempt:
                worker_result = attempt.path("worker-result.json")
                worker_log = attempt.path("worker.log")
                worker_progress = attempt.path("worker-progress.jsonl")
                generation_timeout = (
                    self.settings.generation_time_limit_seconds
                    if self.settings.campaign_policy is STRICT_POLICY
                    else generation_limit_for_cell(
                        self.settings.campaign_policy,
                        cell,
                    )
                )
                generation_lock_path = (
                    None
                    if reusable_record is not None
                    else _symbolica_generation_lock_path(
                        self.settings,
                        self.service.paths.coordination_root,
                        cell,
                    )
                )
                phase_channel = (
                    None
                    if generation_timeout is None
                    and self.settings.profiling_time_limit_seconds is None
                    and self.settings.validation_time_limit_seconds is None
                    and generation_lock_path is None
                    else WorkerPhaseChannel.create(
                        attempt.path("worker-phase-state.json")
                    )
                )
                command = [
                    *self._worker_invocation(),
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
                    "--warmup-runs",
                    str(self.settings.warmup_runs),
                    "--minimum-samples",
                    str(self.settings.minimum_samples),
                    "--progress-jsonl",
                    os.fspath(worker_progress),
                ]
                for option, limit in (
                    ("--worker-wall-limit", self.settings.timeout_seconds),
                    (
                        "--profiling-time-limit",
                        self.settings.profiling_time_limit_seconds,
                    ),
                    (
                        "--validation-time-limit",
                        self.settings.validation_time_limit_seconds,
                    ),
                ):
                    if limit is not None:
                        command.extend((option, str(limit)))
                if self.settings.source_identity_override is not None:
                    command.extend(
                        (
                            "--manual-source-revision",
                            self.source_revision,
                            "--manual-source-tree",
                            self.source_tree,
                        )
                    )
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
                if generation_lock_path is not None:
                    command.extend(
                        ("--generation-lock-path", os.fspath(generation_lock_path))
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
                legacy_workspace: Path | None = None
                if cell.measurement.execution_mode is ExecutionMode.AMPLICOL and (
                    self.settings.campaign_policy.allow_terminal_censors
                    or self.settings.source_identity_override is not None
                ):
                    legacy_source, legacy_workspace = self._legacy_workspace_paths(
                        attempt.attempt_id
                    )
                    command.extend(
                        (
                            "--legacy-source-repository",
                            os.fspath(legacy_source),
                            "--legacy-workspace",
                            os.fspath(legacy_workspace),
                        )
                    )
                    if self.settings.source_identity_override is not None:
                        command.append("--legacy-copy-source")
                    if self.settings.original_amplicol_revision is not None:
                        command.extend(
                            (
                                "--legacy-source-revision",
                                self.settings.original_amplicol_revision,
                            )
                        )
                if reusable_record is not None:
                    command.extend(
                        (
                            "--reused-measurement-json",
                            os.fspath(reusable_record.result_path),
                        )
                    )
                self._observe(
                    "worker",
                    cell,
                    attempt_id=attempt.attempt_id,
                    log_path=os.fspath(worker_log),
                    progress_path=os.fspath(worker_progress),
                    phase_path=(
                        None if phase_channel is None else os.fspath(phase_channel.path)
                    ),
                )

                def observe_resources(observation: WorkerObservation) -> None:
                    self._observe(
                        "resource",
                        cell,
                        attempt_id=attempt.attempt_id,
                        pid=observation.pid,
                        member_pids=list(observation.member_pids),
                        phase=observation.phase,
                        phase_sequence=observation.phase_sequence,
                        wall_seconds=observation.usage.wall_seconds,
                        cpu_seconds=observation.usage.cpu_seconds,
                        current_rss_bytes=observation.usage.current_rss_bytes,
                        peak_rss_bytes=observation.usage.peak_rss_bytes,
                        current_physical_footprint_bytes=(
                            observation.usage.current_physical_footprint_bytes
                        ),
                        peak_physical_footprint_bytes=(
                            observation.usage.peak_physical_footprint_bytes
                        ),
                        current_guard_bytes=observation.usage.current_guard_bytes,
                        peak_guard_bytes=observation.usage.peak_guard_bytes,
                        child_count=observation.usage.child_count,
                    )

                supervised = supervise_worker(
                    command,
                    timeout_seconds=self.settings.timeout_seconds,
                    generation_timeout_seconds=generation_timeout,
                    profiling_timeout_seconds=(
                        self.settings.profiling_time_limit_seconds
                    ),
                    validation_timeout_seconds=(
                        self.settings.validation_time_limit_seconds
                    ),
                    generation_guard_includes_preparation=True,
                    phase_channel=phase_channel,
                    max_rss_bytes=self._effective_cell_rss_limit(),
                    environment_overrides=_worker_environment_overrides(
                        self.settings,
                        self.service.paths.coordination_root,
                    ),
                    scrub_import_environment=(self.worker_harness_identity is not None),
                    working_directory=(
                        self.service.paths.repo_root
                        if self.worker_harness_identity is not None
                        else None
                    ),
                    interval_seconds=self.settings.resource_sample_interval_seconds,
                    termination_grace_seconds=self.settings.termination_grace_seconds,
                    observation_callback=observe_resources,
                    cancellation_requested=self.settings.cancellation_requested,
                    capture_stderr=True,
                )
                if legacy_workspace is not None:
                    self._remove_legacy_workspace(legacy_workspace)
                generation_phase = supervised.generation_phase
                resources = _resource_payload(
                    supervised.usage,
                    generation_phase,
                    memory_limit_bytes=supervised.memory_limit_bytes,
                    memory_limit_reason=supervised.memory_limit_reason,
                    supervised=supervised,
                    source_revision=self.source_revision,
                    campaign_invocation_id=self.settings.campaign_invocation_id,
                )
                if supervised.reason == "cancelled":
                    if self.settings.discard_cancelled_attempts:
                        attempt.discard()
                        detail = (
                            "incomplete attempt discarded"
                            if decision.current is None
                            else (
                                "previous valid current preserved; incomplete "
                                "attempt discarded"
                            )
                        )
                    else:
                        attempt.mark_interrupted(
                            "worker terminated by cancellation",
                            artifact_paths=_attempt_files(
                                attempt.root,
                                include_heavy_artifact=(
                                    not self.settings.remove_heavy_attempt_artifacts
                                ),
                            ),
                        )
                        detail = (
                            attempt.attempt_id
                            if decision.current is None
                            else "previous valid current preserved"
                        )
                    return CellOutcome(
                        cell.cell_id,
                        "cancelled",
                        detail,
                    )
                policy_state: PolicyMeasurementState | None = None
                if self.settings.manual_terminal_censors and supervised.reason in {
                    "generation_timeout",
                    "memory_limit",
                    "worker_timeout",
                    "profiling_timeout",
                    "validation_timeout",
                }:
                    resources["terminal_reason"] = supervised.reason
                    result, policy_state = self._manual_terminal_result(
                        cell=cell,
                        reason=supervised.reason,
                        resources=resources,
                        progress_path=worker_progress,
                    )
                elif (
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
                        "worker_timeout": ResultStatus.TIMEOUT,
                        "generation_timeout": ResultStatus.TIMEOUT,
                        "profiling_timeout": ResultStatus.TIMEOUT,
                        "validation_timeout": ResultStatus.TIMEOUT,
                        "memory_limit": ResultStatus.MEMORY_LIMIT,
                        "memory_probe_error": ResultStatus.ERROR,
                        "phase_state_error": ResultStatus.ERROR,
                        "worker_exit": ResultStatus.ERROR,
                    }[supervised.reason]
                    error: BaseException | str = (
                        _worker_process_exit_error(supervised)
                        if supervised.reason == "worker_exit"
                        else f"worker terminated by {supervised.reason}"
                    )
                    result = failure_measurement(
                        status,
                        error,
                        resources=resources,
                    )
                elif supervised.returncode != 0 or not worker_result.is_file():
                    result = failure_measurement(
                        ResultStatus.ERROR,
                        _worker_process_exit_error(supervised),
                        resources=resources,
                    )
                else:
                    raw = json.loads(worker_result.read_text(encoding="ascii"))
                    if not isinstance(raw, Mapping):
                        raise TypeError("worker result must be a JSON object")
                    result = dict(raw)
                    child_resources = result.get("resources")
                    child_failure = result.get("failure")
                    child_terminal_reason = (
                        child_resources.get("terminal_reason")
                        if isinstance(child_resources, Mapping)
                        else None
                    )
                    authenticated_child_profile_timeout = (
                        result.get("status") == ResultStatus.TIMEOUT.value
                        and child_terminal_reason == "profiling_timeout"
                        and isinstance(child_failure, Mapping)
                        and child_failure.get("kind") == "ProfilingTimeLimitError"
                    )
                    if authenticated_child_profile_timeout:
                        resources["terminal_reason"] = "profiling_timeout"
                    result["resources"] = resources
                    if (
                        self.settings.manual_terminal_censors
                        and authenticated_child_profile_timeout
                    ):
                        policy_state, censor = self._manual_terminal_contract(
                            "profiling_timeout"
                        )
                        self._attach_manual_provenance(
                            result,
                            cell=cell,
                            censor=censor,
                            progress_path=worker_progress,
                        )
                    else:
                        self._attach_manual_provenance(
                            result,
                            cell=cell,
                            progress_path=worker_progress,
                        )
                    self._bind_study_result(
                        result,
                        require_existing_harness=True,
                    )
                if supervised.reason != "completed" or (
                    supervised.returncode != 0 or not worker_result.is_file()
                ):
                    provenance = result.get("provenance")
                    manual_provenance = (
                        provenance.get("manual_campaign")
                        if isinstance(provenance, Mapping)
                        else None
                    )
                    if not isinstance(manual_provenance, Mapping):
                        self._attach_manual_provenance(
                            result,
                            cell=cell,
                            progress_path=worker_progress,
                        )
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
                elif (
                    policy_state is not None
                    and not self.settings.manual_terminal_censors
                ):
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
                attempt.write_json("worker-result.json", result)
                paths = _attempt_files(
                    attempt.root,
                    include_heavy_artifact=(
                        policy_state is not None
                        or not self.settings.remove_heavy_attempt_artifacts
                    ),
                )
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

    def _publish_blocked_dependency(
        self,
        cell: CellSpec,
        prerequisite_cell_ids: str | Sequence[str],
        *,
        current: CurrentRecord | None,
    ) -> CellOutcome:
        normalized = tuple(
            sorted(
                {
                    value
                    for value in (
                        (prerequisite_cell_ids,)
                        if isinstance(prerequisite_cell_ids, str)
                        else prerequisite_cell_ids
                    )
                    if isinstance(value, str) and value
                }
            )
        )
        if not normalized:
            raise ValueError("blocked dependency requires a prerequisite cell ID")
        if len(normalized) == 1:
            message = (
                "blocked by dependency: required prerequisite "
                f"{normalized[0]!r} is unavailable"
            )
        else:
            rendered = ", ".join(repr(value) for value in normalized)
            message = (
                "blocked by dependency: required prerequisites "
                f"{rendered} are unavailable"
            )
        with self.service.store.new_attempt(
            cell.cell_id,
            self.settings.artifact_policy,
            based_on=current,
        ) as attempt:
            result = failure_measurement(ResultStatus.SKIP, message)
            result["blocked_dependency"] = {
                "prerequisite_cell_ids": list(normalized),
            }
            self._bind_study_result(result)
            attempt.write_json("worker-result.json", result)
            attempt.mark_failed(
                message,
                artifact_paths=("worker-result.json",),
            )
            return CellOutcome(
                cell.cell_id,
                "blocked_dependency",
                message,
                normalized,
            )

    def _publish_skip(
        self,
        cell: CellSpec,
        message: str,
        *,
        current: CurrentRecord | None,
    ) -> CellOutcome:
        """Retain the generic failed-attempt helper for non-dependency callers."""

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
                ResultStatus.SKIP.value,
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
    "archive_cell_attempt_history",
    "plan_campaign",
    "reconcile_attempt_history",
    "select_cells",
    "validate_campaign_plan",
]
