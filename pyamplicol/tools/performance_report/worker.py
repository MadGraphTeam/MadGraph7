# SPDX-License-Identifier: 0BSD
"""Isolated one-cell worker entry point used by the campaign scheduler."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyamplicol.reporting import ProgressEvent, ProgressSink

from .agreements import (
    LC_CROSS_LAYOUT_COMPONENT,
    attach_direct_agreements,
    incoming_agreement_edges,
    independent_numerical_authorities,
)
from .artifacts import _filesystem_lock, _raise_disk_full
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement import (
    attach_validation_failure_precision_diagnostic,
    failure_measurement,
    generated_artifact_from_measurement,
    load_measurement,
    measure_pyamplicol_cell,
)
from .models import ExecutionMode, ResultStatus
from .phase_state import WorkerPhaseChannel, WorkerPhaseReporter
from .runner import ProfilingTimeLimitError, RunnerSettings
from .service import ReportPaths
from .source_identity import ReportSourceIdentity, require_eligible_report_source
from .worker_harness import attach_worker_harness_identity

_GIT_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{40}$")


class _JsonlProgressSink:
    """Thread-safe, append-only typed progress capture for dashboard readers."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        from pyamplicol.reporting import ProgressEnd, ProgressStart, ProgressUpdate

        payload: dict[str, object] = {
            "timestamp_unix": time.time(),
            "pid": os.getpid(),
            "task_id": event.task_id,
        }
        if isinstance(event, ProgressStart):
            payload.update(
                {
                    "event": "start",
                    "description": event.description,
                    "total": event.total,
                    "parent_task_id": event.parent_task_id,
                    "unit": event.unit,
                    "details": dict(event.details),
                }
            )
        elif isinstance(event, ProgressUpdate):
            payload.update(
                {
                    "event": "update",
                    "completed": event.completed,
                    "total": event.total,
                    "message": event.message,
                    "details": dict(event.details),
                }
            )
        elif isinstance(event, ProgressEnd):
            payload.update(
                {
                    "event": "end",
                    "success": event.success,
                    "message": event.message,
                    "elapsed_seconds": event.elapsed_seconds,
                    "details": dict(event.details),
                }
            )
        else:  # pragma: no cover - ProgressEvent is a closed typed union.
            raise TypeError(f"unsupported progress event: {type(event).__name__}")
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        with self._lock:
            descriptor = os.open(self.path, flags, 0o600)
            try:
                os.write(descriptor, encoded)
            finally:
                os.close(descriptor)


def _source_identity(
    repo_root: Path,
    revision: str | None,
    tree: str | None,
) -> ReportSourceIdentity:
    if (revision is None) != (tree is None):
        raise ValueError("manual source revision and tree must be specified together")
    if revision is None:
        return require_eligible_report_source(repo_root)
    assert tree is not None
    if _GIT_OBJECT_ID.fullmatch(revision) is None:
        raise ValueError("manual source revision must be a 40-digit hexadecimal ID")
    if _GIT_OBJECT_ID.fullmatch(tree) is None:
        raise ValueError("manual source tree must be a 40-digit hexadecimal ID")
    return ReportSourceIdentity(revision.lower(), tree.lower(), ())


def _portable_current_paths(
    *,
    repo_root: Path,
    attempt_root: Path,
) -> ReportPaths | None:
    """Recognize the canonical copied-campaign attempt shape.

    Ordinary developer/report stores retain their historical raw-current ABI.
    A copied manual campaign has one literal visible state root, so its worker
    can materialize authenticated peer/current files without another CLI flag.
    """

    lexical_attempt = Path(os.path.abspath(attempt_root.expanduser()))
    if len(lexical_attempt.parents) < 4:
        return None
    artifact_root = lexical_attempt.parents[3]
    if artifact_root.name != "campaign_artifacts":
        return None
    relative = lexical_attempt.relative_to(artifact_root)
    if (
        len(relative.parts) != 4
        or relative.parts[0] != "cells"
        or not relative.parts[1]
        or relative.parts[2] != "attempts"
    ):
        raise ValueError("manual campaign worker attempt path is not canonical")
    try:
        attempt_id = str(uuid.UUID(relative.parts[3]))
    except ValueError as error:
        raise ValueError("manual campaign worker attempt ID is invalid") from error
    if attempt_id != relative.parts[3]:
        raise ValueError("manual campaign worker attempt ID is not canonical")
    try:
        resolved_attempt = attempt_root.expanduser().resolve(strict=True)
        resolved_root = artifact_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("manual campaign worker state root is unavailable") from error
    if (
        resolved_attempt != lexical_attempt
        or resolved_root != artifact_root
        or artifact_root.is_symlink()
        or not artifact_root.is_dir()
    ):
        raise ValueError("manual campaign worker state root is not canonical")
    coordination_root = artifact_root / "coordination"
    if (
        coordination_root.is_symlink()
        or not coordination_root.is_dir()
        or coordination_root.resolve(strict=True) != coordination_root
    ):
        raise ValueError("manual campaign coordination root is not canonical")
    docs_dir = artifact_root.parent
    return ReportPaths(
        repo_root=repo_root.expanduser().resolve(strict=False),
        docs_dir=docs_dir,
        results_dir=docs_dir / "results",
        artifact_root=artifact_root,
        coordination_root=coordination_root,
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        stream = os.fdopen(descriptor, "w", encoding="ascii")
        descriptor = None
        with stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        _raise_disk_full(error, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _selector_provider_measurement(
    cell_id: str,
    peers: Mapping[str, Mapping[str, object]],
    *,
    catalog: ReportCatalog,
) -> Mapping[str, object] | None:
    """Return the scheduled selected-flow peer used to seed LC all-flow."""

    cell = catalog.cell(cell_id)
    providers = tuple(
        edge.baseline.cell_id
        for edge in incoming_agreement_edges(cell, catalog=catalog)
        if edge.kind == LC_CROSS_LAYOUT_COMPONENT
    )
    if not providers:
        return None
    if len(providers) != 1:
        raise ValueError(f"{cell_id}: selector provider is not unique")
    provider_id = providers[0]
    provider = peers.get(provider_id)
    if provider is None:
        raise ValueError(
            f"{cell_id}: required selector provider {provider_id!r} is unavailable"
        )
    return provider


@contextmanager
def _worker_legacy_workspace(
    *,
    repository: Path | None,
    source_repository: Path | None,
    workspace: Path | None,
    copy_source: bool,
) -> Iterator[Path | None]:
    """Prepare an isolated legacy checkout inside the supervised worker."""

    if source_repository is None:
        if workspace is not None or copy_source:
            raise ValueError(
                "legacy workspace/copy options require a source repository"
            )
        yield repository
        return
    if repository is not None:
        raise ValueError(
            "legacy repository and legacy source repository are mutually exclusive"
        )
    if workspace is None:
        raise ValueError("legacy source repository requires a workspace destination")
    source = source_repository.expanduser().resolve(strict=True)
    destination = workspace.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"legacy worker workspace already exists: {destination}")

    from .legacy import MaintainedLegacyApi
    from .legacy_structure import legacy_structural_probe_lock

    api = MaintainedLegacyApi()
    print(
        f"Preparing isolated original-AmpliCol workspace {destination}",
        flush=True,
    )
    if copy_source:
        with legacy_structural_probe_lock(source):
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(".git", ".DS_Store"),
            )
    else:
        api.validate_checkout(source)
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
            print(f"Legacy workspace command: {' '.join(command)}", flush=True)
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
    try:
        yield destination
    finally:
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise RuntimeError(
                    "legacy worker workspace is not its canonical regular "
                    f"directory: {destination}"
                )
            shutil.rmtree(destination)


def measure_cell(
    cell_id: str,
    *,
    repo_root: Path,
    attempt_root: Path,
    target_runtime_seconds: float,
    batch_size: int,
    worker_cores: int,
    warmup_runs: int = 2,
    minimum_samples: int = 5,
    progress_jsonl: Path | None = None,
    worker_wall_limit_seconds: float | None = None,
    profiling_time_limit_seconds: float | None = None,
    validation_time_limit_seconds: float | None = None,
    manual_source_revision: str | None = None,
    manual_source_tree: str | None = None,
    baseline_json: Path | None = None,
    expected_authority_cell_ids: Sequence[str] = (),
    selected_authority_cell_id: str | None = None,
    peer_json: Sequence[tuple[str, Path]] = (),
    prepared_model_path: Path | None = None,
    reused_measurement_json: Path | None = None,
    phase_reporter: WorkerPhaseReporter | None = None,
    legacy_repository: Path | None = None,
    legacy_source_repository: Path | None = None,
    legacy_workspace: Path | None = None,
    legacy_copy_source: bool = False,
    legacy_source_revision: str | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, object]:
    worker_started_monotonic = time.monotonic()
    for name, value in (
        ("worker_wall_limit_seconds", worker_wall_limit_seconds),
        ("profiling_time_limit_seconds", profiling_time_limit_seconds),
        ("validation_time_limit_seconds", validation_time_limit_seconds),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    cell = catalog.cell(cell_id)
    static_na_reason = catalog.static_na_reason(cell)
    if static_na_reason is not None:
        description = catalog.static_na_description(cell)
        raise ValueError(
            f"{cell_id}: catalog static N/A cell cannot be measured "
            f"({static_na_reason}: {description})"
        )
    authority_ids = tuple(expected_authority_cell_ids)
    canonical_authority_ids = tuple(
        authority.cell_id
        for authority in independent_numerical_authorities(cell, catalog=catalog)
    )
    if authority_ids != canonical_authority_ids:
        raise ValueError(
            "expected numerical authority cell IDs differ from the canonical "
            f"catalog chain: expected={canonical_authority_ids!r}, "
            f"observed={authority_ids!r}"
        )
    if selected_authority_cell_id is not None and (
        selected_authority_cell_id not in authority_ids
        or baseline_json is None
    ):
        raise ValueError("selected numerical authority is not in the expected chain")
    if authority_ids and (baseline_json is not None) != (
        selected_authority_cell_id is not None
    ):
        raise ValueError(
            "compiled/eager baseline and selected numerical authority must be "
            "specified together"
        )
    source_identity = _source_identity(
        repo_root,
        manual_source_revision,
        manual_source_tree,
    )
    progress: ProgressSink | None = (
        None if progress_jsonl is None else _JsonlProgressSink(progress_jsonl)
    )
    publication_paths = _portable_current_paths(
        repo_root=repo_root,
        attempt_root=attempt_root,
    )
    baseline = (
        None
        if baseline_json is None
        else load_measurement(
            baseline_json,
            publication_paths=publication_paths,
        )
    )
    peers = {
        peer_cell_id: load_measurement(
            path,
            publication_paths=publication_paths,
        )
        for peer_cell_id, path in peer_json
    }
    if len(peers) != len(peer_json):
        raise ValueError("direct-agreement peer cell IDs must be unique")
    reused_artifact = (
        None
        if reused_measurement_json is None
        else generated_artifact_from_measurement(
            load_measurement(
                reused_measurement_json,
                publication_paths=publication_paths,
            )
        )
    )
    selector_provider = _selector_provider_measurement(
        cell.cell_id,
        peers,
        catalog=catalog,
    )
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        from .legacy import LegacyMeasurementAdapter, LegacySettings

        with _worker_legacy_workspace(
            repository=legacy_repository,
            source_repository=legacy_source_repository,
            workspace=legacy_workspace,
            copy_source=legacy_copy_source,
        ) as prepared_legacy_repository:
            result = LegacyMeasurementAdapter().measure(
                cell,
                artifact_path=attempt_root / "artifact",
                settings=LegacySettings(
                    target_runtime_seconds=target_runtime_seconds,
                    jobs=worker_cores,
                    repository=prepared_legacy_repository,
                    validate_checkout=manual_source_revision is None,
                    source_revision=legacy_source_revision,
                    profiling_time_limit_seconds=profiling_time_limit_seconds,
                    worker_deadline_monotonic=(
                        None
                        if worker_wall_limit_seconds is None
                        else worker_started_monotonic + worker_wall_limit_seconds
                    ),
                ),
                phase_reporter=phase_reporter,
                selector_provider=selector_provider,
            )
    else:
        result = measure_pyamplicol_cell(
            cell,
            artifact_path=attempt_root / "artifact",
            settings=RunnerSettings(
                target_runtime_seconds=target_runtime_seconds,
                batch_size=batch_size,
                worker_cores=worker_cores,
                warmup_runs=warmup_runs,
                minimum_samples=minimum_samples,
                model_cache_dir=attempt_root.parent.parent.parent / "model-cache",
                progress=progress,
                source_revision_override=(
                    None if manual_source_revision is None else source_identity.revision
                ),
                profiling_time_limit_seconds=profiling_time_limit_seconds,
                worker_deadline_monotonic=(
                    None
                    if worker_wall_limit_seconds is None
                    else worker_started_monotonic + worker_wall_limit_seconds
                ),
            ),
            repo_root=repo_root,
            baseline=baseline,
            expected_authority_cell_ids=authority_ids,
            selected_authority_cell_id=selected_authority_cell_id,
            validation_peers=peers,
            selector_provider=selector_provider,
            prepared_model_path=prepared_model_path,
            reused_artifact=reused_artifact,
            phase_reporter=phase_reporter,
            catalog=catalog,
        )
    attach_direct_agreements(
        cell,
        result,
        peers,
        catalog=catalog,
    )
    attach_validation_failure_precision_diagnostic(
        cell,
        result,
        baseline=baseline,
        peers=peers,
    )
    source_identity_postflight = _source_identity(
        repo_root,
        manual_source_revision,
        manual_source_tree,
    )
    if source_identity_postflight != source_identity:
        raise RuntimeError("report source identity changed during cell measurement")
    provenance = result.get("provenance")
    result["provenance"] = {
        **({} if not isinstance(provenance, Mapping) else dict(provenance)),
        **source_identity.provenance(),
    }
    if phase_reporter is not None:
        phase_reporter.complete()
    return result


def write_cell_result(
    cell_id: str,
    result_path: Path,
    *,
    log_path: Path | None = None,
    phase_state_path: Path | None = None,
    phase_state_run_id: str | None = None,
    phase_state_authentication_key: str | None = None,
    worker_harness: Mapping[str, object] | None = None,
    worker_wall_limit_seconds: float | None = None,
    profiling_time_limit_seconds: float | None = None,
    validation_time_limit_seconds: float | None = None,
    generation_lock_path: Path | None = None,
    **kwargs: object,
) -> dict[str, object]:
    try:
        phase_arguments = (
            phase_state_path,
            phase_state_run_id,
            phase_state_authentication_key,
        )
        if any(argument is not None for argument in phase_arguments):
            if not all(argument is not None for argument in phase_arguments):
                raise ValueError(
                    "worker phase-state arguments must be specified together"
                )
            assert phase_state_path is not None
            assert phase_state_run_id is not None
            assert phase_state_authentication_key is not None
            phase_reporter = WorkerPhaseReporter(
                WorkerPhaseChannel(
                    path=phase_state_path.expanduser().resolve(strict=False),
                    run_id=phase_state_run_id,
                    authentication_key=phase_state_authentication_key,
                ),
                track_post_generation_stages=(
                    profiling_time_limit_seconds is not None
                    or validation_time_limit_seconds is not None
                ),
                generation_gate=(
                    None
                    if generation_lock_path is None
                    else lambda: _filesystem_lock(
                        generation_lock_path,
                        timeout=None,
                        poll_interval=0.05,
                    )
                ),
            )
        else:
            phase_reporter = None
        if log_path is None:
            result = measure_cell(
                cell_id,
                phase_reporter=phase_reporter,
                worker_wall_limit_seconds=worker_wall_limit_seconds,
                profiling_time_limit_seconds=profiling_time_limit_seconds,
                validation_time_limit_seconds=validation_time_limit_seconds,
                **kwargs,  # type: ignore[arg-type]
            )
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                log_path.open("a", encoding="utf-8") as stream,
                redirect_stdout(stream),
                redirect_stderr(stream),
            ):
                result = measure_cell(
                    cell_id,
                    phase_reporter=phase_reporter,
                    worker_wall_limit_seconds=worker_wall_limit_seconds,
                    profiling_time_limit_seconds=profiling_time_limit_seconds,
                    validation_time_limit_seconds=validation_time_limit_seconds,
                    **kwargs,  # type: ignore[arg-type]
                )
    except ProfilingTimeLimitError as error:
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
        result = failure_measurement(
            ResultStatus.TIMEOUT,
            error,
            resources={"terminal_reason": "profiling_timeout"},
        )
    except Exception as error:
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
        result = failure_measurement(ResultStatus.ERROR, error)
    if worker_harness is not None:
        attach_worker_harness_identity(result, worker_harness)
    provenance = result.get("provenance")
    if isinstance(provenance, Mapping):
        result["provenance"] = {
            **provenance,
            "worker_environment": {
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
                "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH"),
            },
            "worker_log": None if log_path is None else os.fspath(log_path),
        }
    _atomic_json(result_path, result)
    return result


__all__ = ["measure_cell", "write_cell_result"]
