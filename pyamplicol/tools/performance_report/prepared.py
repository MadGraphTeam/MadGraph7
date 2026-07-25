# SPDX-License-Identifier: 0BSD
"""Deterministic, lock-protected prepared-model assets for report workers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .artifacts import ArtifactStore
from .models import ModelKey

PREPARED_RECORD_SCHEMA = "pyamplicol-report-prepared-model-v1"


class PreparedModelError(RuntimeError):
    """Raised when a report prepared-model bundle is absent or invalid."""


class CompilableModelSource(Protocol):
    def compile(
        self,
        *,
        cache_dir: os.PathLike[str] | str | None = None,
        use_cache: bool = True,
        require_supported: bool = True,
        prepared_output: os.PathLike[str] | str | None = None,
        evaluator: object | None = None,
    ) -> object: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepared_identity(
    *,
    model: ModelKey,
    backend: str,
    jit_optimization_level: int,
    source_digest: str,
    producer_revision: str,
    target_platform: str | None = None,
) -> dict[str, object]:
    if len(source_digest) != 64:
        raise ValueError("source_digest must be SHA-256")
    if not producer_revision:
        raise ValueError("producer_revision must not be empty")
    return {
        "model": model.value,
        "backend": backend,
        "jit_optimization_level": jit_optimization_level,
        "source_digest": source_digest,
        "producer_revision": producer_revision,
        "target_platform": (
            target_platform
            if target_platform is not None
            else f"{platform.system().lower()}-{platform.machine().lower()}"
        ),
    }


def _record_path(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(bundle_path.suffix + ".report.json")


def validate_prepared_record(
    bundle_path: Path,
    *,
    expected_identity: Mapping[str, object],
) -> dict[str, object]:
    record_path = _record_path(bundle_path)
    try:
        record = json.loads(record_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparedModelError(
            f"cannot read prepared-model record: {error}"
        ) from error
    if not isinstance(record, Mapping):
        raise PreparedModelError("prepared-model record must be an object")
    if record.get("schema") != PREPARED_RECORD_SCHEMA:
        raise PreparedModelError("prepared-model record schema is unsupported")
    if record.get("identity") != dict(expected_identity):
        raise PreparedModelError("prepared-model identity does not match")
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise PreparedModelError("prepared-model bundle is not a regular file")
    if record.get("bundle_size") != bundle_path.stat().st_size:
        raise PreparedModelError("prepared-model bundle size does not match")
    if record.get("bundle_sha256") != _sha256(bundle_path):
        raise PreparedModelError("prepared-model bundle digest does not match")
    return dict(record)


@contextmanager
def ensure_prepared_model(
    *,
    store: ArtifactStore,
    bundle_path: Path,
    source: CompilableModelSource,
    evaluator: object,
    identity: Mapping[str, object],
    model_cache_dir: Path,
) -> Iterator[tuple[Path, bool]]:
    """Create one prepared bundle atomically or reuse its validated record."""

    lock_name = "prepared-" + hashlib.sha256(
        _canonical_bytes(dict(identity))
    ).hexdigest()
    with store.named_lock(lock_name):
        try:
            validate_prepared_record(bundle_path, expected_identity=identity)
        except PreparedModelError:
            reused = False
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            staging = bundle_path.with_name(
                f".{bundle_path.stem}.{uuid.uuid4().hex}.staging"
                ".pyamplicol-model"
            )
            try:
                source.compile(
                    cache_dir=model_cache_dir,
                    use_cache=True,
                    require_supported=True,
                    prepared_output=staging,
                    evaluator=evaluator,
                )
                if not staging.is_file() or staging.is_symlink():
                    raise PreparedModelError(
                        "prepared-model compiler did not publish a regular bundle"
                    )
                digest = _sha256(staging)
                size = staging.stat().st_size
                staging.replace(bundle_path)
                _atomic_write(
                    _record_path(bundle_path),
                    {
                        "schema": PREPARED_RECORD_SCHEMA,
                        "identity": dict(identity),
                        "bundle_sha256": digest,
                        "bundle_size": size,
                    },
                )
                validate_prepared_record(
                    bundle_path,
                    expected_identity=identity,
                )
            finally:
                staging.unlink(missing_ok=True)
        else:
            reused = True
        yield bundle_path, reused


def _git_revision(repo_root: Path) -> str:
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


def ensure_report_prepared_model(
    *,
    store: ArtifactStore,
    repo_root: Path,
    worker_cores: int,
    model: ModelKey,
) -> tuple[Path, bool]:
    """Create or reuse one portable report JIT O2 prepared bundle."""

    from pyamplicol.api import ModelSource
    from pyamplicol.config import (
        EvaluatorBackend,
        EvaluatorConfig,
        EvaluatorExecutionMode,
        EvaluatorOptimizationConfig,
        JITConfig,
    )

    revision = _git_revision(repo_root)
    if model is ModelKey.BUILTIN_SM:
        source_path = None
        source = ModelSource.built_in_sm()
        digest = hashlib.sha256(
            f"built-in-sm:{revision}".encode("ascii")
        ).hexdigest()
        stem = "built-in-sm"
    elif model is ModelKey.UFO_SM:
        source_path = (
            repo_root / "src/pyamplicol/assets/models/json/sm/sm.json"
        ).resolve()
        source = ModelSource.from_path(source_path)
        digest = _sha256(source_path)
        stem = "ufo-sm"
    else:
        raise ValueError("report prepared packs support built-in SM and UFO-SM")
    identity = prepared_identity(
        model=model,
        backend="jit",
        jit_optimization_level=2,
        source_digest=digest,
        producer_revision=revision,
    )
    identity_digest = hashlib.sha256(
        _canonical_bytes(identity)
    ).hexdigest()[:20]
    bundle_path = (
        store.artifact_root
        / "prepared-models"
        / f"{stem}-jit-o2-{identity_digest}.pyamplicol-model"
    )
    evaluator = EvaluatorConfig(
        backend=EvaluatorBackend.JIT,
        execution_mode=EvaluatorExecutionMode.RECURRENCE,
        optimization=EvaluatorOptimizationConfig(cores=worker_cores),
        jit=JITConfig(optimization_level=2),
    )
    with ensure_prepared_model(
        store=store,
        bundle_path=bundle_path,
        source=source,
        evaluator=evaluator,
        identity=identity,
        model_cache_dir=store.artifact_root / "model-cache",
    ) as result:
        return result


def ensure_report_ufo_sm_prepared_model(
    *,
    store: ArtifactStore,
    repo_root: Path,
    worker_cores: int,
) -> tuple[Path, bool]:
    return ensure_report_prepared_model(
        store=store,
        repo_root=repo_root,
        worker_cores=worker_cores,
        model=ModelKey.UFO_SM,
    )


__all__ = [
    "PREPARED_RECORD_SCHEMA",
    "PreparedModelError",
    "ensure_prepared_model",
    "ensure_report_prepared_model",
    "ensure_report_ufo_sm_prepared_model",
    "prepared_identity",
    "validate_prepared_record",
]
