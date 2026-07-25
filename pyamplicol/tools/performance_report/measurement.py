"""One-cell measurement orchestration over the public pyAmpliCol Python API."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

from .cache import empty_measurement
from .models import Accuracy, CellSpec, ResultStatus
from .runner import (
    GeneratedArtifact,
    RunnerError,
    RunnerSettings,
    SelectorContract,
    generate_artifact,
    pointwise_validation,
    profile_runtime,
    provenance_payload,
    validate_artifact_contract,
)


def shared_validation_points(process: str) -> object:
    from pyamplicol.models.builtin.validation import generic_validation_point

    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(process)
        ),
    )


def source_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_selector_contract(
    baseline: Mapping[str, object] | None,
) -> SelectorContract | None:
    if baseline is None:
        return None
    raw = baseline.get("selector_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RunnerError("baseline selector_contract must be an object")
    return SelectorContract.from_mapping(raw)


def _baseline_matrix_element(
    baseline: Mapping[str, object] | None,
) -> float | None:
    if baseline is None:
        return None
    if baseline.get("status") != ResultStatus.OK.value:
        raise RunnerError("candidate baseline is not a valid completed measurement")
    value = baseline.get("matrix_element")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError("candidate baseline has no matrix element")
    return float(value)


def _load_runtime(artifact_path: Path, process_id: str) -> object:
    from pyamplicol.api import Runtime

    return Runtime.load(artifact_path, process=process_id)


def _resolution_benchmark_config(effective_config: Mapping[str, object]) -> object:
    from pyamplicol.config import Action
    from pyamplicol.config.resolver import resolve_config

    return resolve_config(
        effective_config,
        action=Action.GENERATE,
        base_dir=Path.cwd(),
    ).effective.benchmark


def measure_pyamplicol_cell(
    cell: CellSpec,
    *,
    artifact_path: Path,
    settings: RunnerSettings,
    repo_root: Path,
    baseline: Mapping[str, object] | None,
    prepared_model_path: Path | None = None,
    reused_artifact: GeneratedArtifact | None = None,
) -> dict[str, object]:
    """Generate or retime one complete-coverage pyAmpliCol artifact."""

    generated = reused_artifact or generate_artifact(
        cell,
        artifact_path,
        settings=settings,
        repo_root=repo_root,
        prepared_model_path=prepared_model_path,
    )
    validate_artifact_contract(cell, generated.path)
    runtime = _load_runtime(generated.path, generated.process_id)
    points = shared_validation_points(cell.process)
    contract = (
        _baseline_selector_contract(baseline)
        if cell.measurement.accuracy is Accuracy.LC
        else None
    )
    if cell.measurement.accuracy is Accuracy.LC and contract is None:
        from .runner import derive_selector_contract

        contract = derive_selector_contract(runtime, points)
    profile = profile_runtime(
        runtime,
        points,
        cell=cell,
        benchmark_config=_resolution_benchmark_config(generated.effective_config),
        selector_contract=contract,
    )
    validation: dict[str, object] = {
        "resolved_sum": profile.pop("resolved_sum_validation"),
    }
    baseline_value = _baseline_matrix_element(baseline)
    if baseline_value is not None:
        validation["pointwise"] = pointwise_validation(
            float(profile["matrix_element"]),
            baseline_value,
        )
    statuses = {
        str(record.get("status"))
        for record in validation.values()
        if isinstance(record, Mapping)
    }
    validation["status"] = (
        ResultStatus.VALIDATION_FAILED.value
        if ResultStatus.VALIDATION_FAILED.value in statuses
        else ResultStatus.OK.value
    )
    status = str(profile["status"])
    if ResultStatus.VALIDATION_FAILED.value in statuses:
        status = ResultStatus.VALIDATION_FAILED.value

    measurement = empty_measurement()
    measurement.update(profile)
    measurement.update(
        {
            "status": status,
            "generation_seconds": generated.generation_seconds,
            "artifact": {
                "path": str(generated.path),
                "process_id": generated.process_id,
                "policy": "reused" if reused_artifact is not None else "generated",
            },
            "selector_contract": None if contract is None else contract.as_dict(),
            "validation": validation,
            "resources": None,
            "provenance": {
                **provenance_payload(),
                "source_revision": source_revision(repo_root),
                "requested_config": dict(generated.requested_config),
                "effective_config": dict(generated.effective_config),
                "model_preparation_seconds": generated.model_preparation_seconds,
                "model_preparation_reused": generated.model_preparation_reused,
                "generation_timer_excludes_model_preparation": True,
            },
            "failure": None,
        }
    )
    return measurement


def failure_measurement(
    status: ResultStatus,
    error: BaseException | str,
    *,
    resources: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if status in {ResultStatus.NOT_AVAILABLE, ResultStatus.OK}:
        raise ValueError("failure measurement requires a terminal failure status")
    message = str(error)
    failure = empty_measurement()
    failure.update(
        {
            "status": status.value,
            "resources": None if resources is None else dict(resources),
            "failure": {
                "kind": type(error).__name__ if isinstance(error, BaseException) else None,
                "message": message,
            },
        }
    )
    return failure


def load_measurement(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("measurement file must contain an object")
    return dict(payload)


__all__ = [
    "failure_measurement",
    "file_digest",
    "load_measurement",
    "measure_pyamplicol_cell",
    "shared_validation_points",
    "source_revision",
]
