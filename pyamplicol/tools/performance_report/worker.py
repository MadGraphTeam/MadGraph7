# SPDX-License-Identifier: 0BSD
"""Isolated one-cell worker entry point used by the campaign scheduler."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from .agreements import attach_direct_agreements
from .catalog import REPORT_CATALOG, ReportCatalog
from .measurement import (
    failure_measurement,
    generated_artifact_from_measurement,
    load_measurement,
    measure_pyamplicol_cell,
)
from .models import ExecutionMode, ResultStatus
from .runner import RunnerSettings
from .source_identity import require_eligible_report_source


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
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
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def measure_cell(
    cell_id: str,
    *,
    repo_root: Path,
    attempt_root: Path,
    target_runtime_seconds: float,
    batch_size: int,
    worker_cores: int,
    baseline_json: Path | None = None,
    peer_json: Sequence[tuple[str, Path]] = (),
    prepared_model_path: Path | None = None,
    reused_measurement_json: Path | None = None,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> dict[str, object]:
    source_identity = require_eligible_report_source(repo_root)
    cell = catalog.cell(cell_id)
    baseline = None if baseline_json is None else load_measurement(baseline_json)
    peers = {
        peer_cell_id: load_measurement(path)
        for peer_cell_id, path in peer_json
    }
    if len(peers) != len(peer_json):
        raise ValueError("direct-agreement peer cell IDs must be unique")
    reused_artifact = (
        None
        if reused_measurement_json is None
        else generated_artifact_from_measurement(
            load_measurement(reused_measurement_json)
        )
    )
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        from .legacy import LegacyMeasurementAdapter, LegacySettings

        result = LegacyMeasurementAdapter().measure(
            cell,
            artifact_path=attempt_root / "artifact",
            settings=LegacySettings(
                target_runtime_seconds=target_runtime_seconds,
                jobs=worker_cores,
            ),
        )
    else:
        result = measure_pyamplicol_cell(
            cell,
            artifact_path=attempt_root / "artifact",
            settings=RunnerSettings(
                target_runtime_seconds=target_runtime_seconds,
                batch_size=batch_size,
                worker_cores=worker_cores,
                model_cache_dir=attempt_root.parent.parent.parent / "model-cache",
            ),
            repo_root=repo_root,
            baseline=baseline,
            prepared_model_path=prepared_model_path,
            reused_artifact=reused_artifact,
        )
    attach_direct_agreements(
        cell,
        result,
        peers,
        catalog=catalog,
    )
    source_identity_postflight = require_eligible_report_source(repo_root)
    if source_identity_postflight != source_identity:
        raise RuntimeError(
            "report source identity changed during cell measurement"
        )
    provenance = result.get("provenance")
    result["provenance"] = {
        **({} if not isinstance(provenance, Mapping) else dict(provenance)),
        **source_identity.provenance(),
    }
    return result


def write_cell_result(
    cell_id: str,
    result_path: Path,
    *,
    log_path: Path | None = None,
    **kwargs: object,
) -> dict[str, object]:
    try:
        if log_path is None:
            result = measure_cell(cell_id, **kwargs)  # type: ignore[arg-type]
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                log_path.open("a", encoding="utf-8") as stream,
                redirect_stdout(stream),
                redirect_stderr(stream),
            ):
                result = measure_cell(cell_id, **kwargs)  # type: ignore[arg-type]
    except Exception as error:
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
        result = failure_measurement(ResultStatus.ERROR, error)
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
