# SPDX-License-Identifier: 0BSD
"""Dependency-aware, resource-supervised performance-report campaigns."""

from __future__ import annotations

import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactAction, ArtifactStore, CurrentRecord
from .cache import validate_measurement
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement import failure_measurement, source_revision
from .models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .resources import ResourceUsage, supervise_worker
from .runner import DEFAULT_TARGET_RUNTIME_SECONDS
from .service import ReportService


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
    max_rss_bytes: int | None = None
    campaign_max_rss_bytes: int | None = None
    artifact_policy: ArtifactPolicy = ArtifactPolicy.REGENERATE
    missing_only: bool = False
    rerun: bool = False
    allow_symbolica_parallel: bool = False

    def __post_init__(self) -> None:
        if self.workers < 1 or self.cell_cores < 1:
            raise ValueError("workers and cell_cores must be positive")
        if self.target_runtime_seconds <= 0.0 or self.batch_size < 1:
            raise ValueError("target runtime and batch size must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_rss_bytes is not None and self.max_rss_bytes <= 0:
            raise ValueError("max_rss_bytes must be positive")
        if (
            self.campaign_max_rss_bytes is not None
            and self.campaign_max_rss_bytes <= 0
        ):
            raise ValueError("campaign_max_rss_bytes must be positive")
        if self.missing_only and self.rerun:
            raise ValueError("--missing-only and --rerun are mutually exclusive")


@dataclass(frozen=True, slots=True)
class PlannedCell:
    cell: CellSpec
    dependency: bool
    baseline_cell_id: str | None
    rank: int


@dataclass(frozen=True, slots=True)
class CellOutcome:
    cell_id: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class CampaignResult:
    planned: tuple[PlannedCell, ...]
    outcomes: tuple[CellOutcome, ...]

    @property
    def failed(self) -> tuple[CellOutcome, ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.status not in {"ok", "reused", "skipped-current"}
        )


def select_cells(
    selection: CellSelection,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSpec, ...]:
    return tuple(
        sorted(
            (
                cell
                for cell in catalog.measurement_cells()
                if selection.matches(cell)
            ),
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
) -> CurrentRecord | None:
    current = store.load_current(cell_id, missing_ok=True)
    if current is None or current.result.get("status") != ResultStatus.OK.value:
        return None
    if expected_revision is not None:
        provenance = current.result.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("report_source_revision") != expected_revision
        ):
            return None
    return current


def _fresh_equivalent_current(
    store: ArtifactStore,
    cell: CellSpec,
    *,
    catalog: ReportCatalog,
    expected_revision: str,
) -> CurrentRecord | None:
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        return None
    for equivalent in catalog.equivalent_cells(cell):
        current = _successful_current(
            store,
            equivalent.cell_id,
            expected_revision=expected_revision,
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
) -> tuple[PlannedCell, ...]:
    requested_ids = {cell.cell_id for cell in requested}
    needed: dict[str, CellSpec] = {}

    def include(cell: CellSpec, *, explicitly_requested: bool) -> None:
        if (
            settings.missing_only
            and explicitly_requested
            and _successful_current(
                store,
                cell.cell_id,
                expected_revision=expected_revision,
            )
            is not None
        ):
            return
        if cell.cell_id in needed:
            return
        baseline = catalog.baseline_cell(cell)
        if baseline is not None and _successful_current(
            store,
            baseline.cell_id,
            expected_revision=expected_revision,
        ) is None:
            include(baseline, explicitly_requested=False)
        needed[cell.cell_id] = cell

    for cell in requested:
        include(cell, explicitly_requested=True)
    return tuple(
        PlannedCell(
            cell=cell,
            dependency=cell.cell_id not in requested_ids,
            baseline_cell_id=(
                None
                if catalog.baseline_cell(cell) is None
                else catalog.baseline_cell(cell).cell_id  # type: ignore[union-attr]
            ),
            rank=_rank(cell),
        )
        for cell in sorted(
            needed.values(),
            key=lambda item: (_rank(item), item.cell_id),
        )
    )


def _resource_payload(usage: ResourceUsage) -> dict[str, object]:
    return {
        "available": usage.available,
        "current_rss_bytes": usage.current_rss_bytes,
        "peak_rss_bytes": usage.peak_rss_bytes,
        "child_count": usage.child_count,
        "cpu_seconds": usage.cpu_seconds,
        "wall_seconds": usage.wall_seconds,
        "probe_error": usage.error,
    }


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
    ) -> None:
        self.service = service
        self.settings = settings
        self.catalog = catalog
        self.source_revision = source_revision(service.paths.repo_root)
        self._prepared_model_paths: dict[ModelKey, Path] = {}

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

    def _ensure_prepared_model(self, planned: Sequence[PlannedCell]) -> None:
        models = {
            item.cell.measurement.model
            for item in planned
            if item.cell.measurement.execution_mode
            in {ExecutionMode.EAGER, ExecutionMode.RECURRENCE}
            and item.cell.measurement.model
            in {ModelKey.BUILTIN_SM, ModelKey.UFO_SM}
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
                os.fspath(self.service.paths.repo_root / "docs/result_tables.py"),
                "--repo-root",
                os.fspath(self.service.paths.repo_root),
                *self._service_path_arguments(),
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
                path = Path(str(payload["path"])).resolve(strict=True)
                self._prepared_model_paths[model] = path
            finally:
                result_path.unlink(missing_ok=True)

    def run(self, planned: Sequence[PlannedCell]) -> CampaignResult:
        ordered = tuple(planned)
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
                futures = {
                    executor.submit(self._run_cell, item): item for item in wave
                }
                for future in as_completed(futures):
                    outcomes.append(future.result())
            self.service.publish(reset=False, merge_artifacts=True)
        return CampaignResult(
            planned=ordered,
            outcomes=tuple(sorted(outcomes, key=lambda item: item.cell_id)),
        )

    def _run_cell(self, planned: PlannedCell) -> CellOutcome:
        cell = planned.cell
        with self.service.store.named_lock(f"campaign-cell-{cell.cell_id}"):
            if (
                self.settings.missing_only
                and _successful_current(
                    self.service.store,
                    cell.cell_id,
                    expected_revision=self.source_revision,
                )
                is not None
            ):
                return CellOutcome(cell.cell_id, "skipped-current", "already complete")
            decision = self.service.store.decide(
                cell.cell_id,
                self.settings.artifact_policy,
            )
            current_is_fresh = (
                _successful_current(
                    self.service.store,
                    cell.cell_id,
                    expected_revision=self.source_revision,
                )
                is not None
            )
            if (
                decision.action is ArtifactAction.REUSE_CURRENT
                and decision.current is not None
                and current_is_fresh
                and not self.settings.rerun
                and decision.current.result.get("status")
                == ResultStatus.OK.value
            ):
                return CellOutcome(cell.cell_id, "reused", decision.current.attempt_id)

            equivalent_record = (
                _fresh_equivalent_current(
                    self.service.store,
                    cell,
                    catalog=self.catalog,
                    expected_revision=self.source_revision,
                )
                if (
                    not self.settings.rerun
                    and self.settings.artifact_policy
                    in {ArtifactPolicy.REUSE, ArtifactPolicy.RETIME}
                )
                else None
            )
            baseline = self.catalog.baseline_cell(cell)
            baseline_record = (
                None
                if baseline is None
                else _successful_current(
                    self.service.store,
                    baseline.cell_id,
                    expected_revision=self.source_revision,
                )
            )
            if baseline is not None and baseline_record is None:
                return self._publish_skip(
                    cell,
                    f"required baseline {baseline.cell_id!r} is unavailable",
                    current=decision.current,
                )

            with self.service.store.new_attempt(
                cell.cell_id,
                self.settings.artifact_policy,
                based_on=decision.current,
            ) as attempt:
                worker_result = attempt.path("worker-result.json")
                worker_log = attempt.path("worker.log")
                command = [
                    sys.executable,
                    os.fspath(
                        self.service.paths.repo_root / "docs/result_tables.py"
                    ),
                    "--repo-root",
                    os.fspath(self.service.paths.repo_root),
                    *self._service_path_arguments(),
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
                if baseline_record is not None:
                    command.extend(
                        ("--baseline-json", os.fspath(baseline_record.result_path))
                    )
                prepared_model = self._prepared_model_paths.get(
                    cell.measurement.model  # type: ignore[arg-type]
                )
                if prepared_model is not None:
                    command.extend(
                        ("--prepared-model", os.fspath(prepared_model))
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
                    max_rss_bytes=self._effective_cell_rss_limit(),
                )
                resources = _resource_payload(supervised.usage)
                if supervised.reason != "completed":
                    status = (
                        ResultStatus.TIMEOUT
                        if supervised.reason == "timeout"
                        else ResultStatus.MEMORY_LIMIT
                    )
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
                validate_measurement(result)
                _write_json(worker_result, result)
                paths = _attempt_files(attempt.root)
                if result["status"] == ResultStatus.OK.value:
                    record = attempt.publish(result, artifact_paths=paths)
                    return CellOutcome(cell.cell_id, "ok", record.attempt_id)
                if decision.current is None:
                    record = attempt.publish(result, artifact_paths=paths)
                    return CellOutcome(
                        cell.cell_id,
                        str(result["status"]),
                        record.attempt_id,
                    )
                attempt.mark_failed(
                    str(result.get("failure")),
                    artifact_paths=paths,
                )
                return CellOutcome(
                    cell.cell_id,
                    str(result["status"]),
                    "previous valid current preserved",
                )

    def _effective_cell_rss_limit(self) -> int | None:
        limits = [
            limit
            for limit in (
                self.settings.max_rss_bytes,
                (
                    None
                    if self.settings.campaign_max_rss_bytes is None
                    else self.settings.campaign_max_rss_bytes
                    // self.settings.workers
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
            if current is None:
                record = attempt.publish(result)
                return CellOutcome(cell.cell_id, "skip", record.attempt_id)
            attempt.mark_failed(message)
            return CellOutcome(
                cell.cell_id,
                "skip",
                "previous valid current preserved",
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
]
