"""Deterministic, lock-protected prepared-model assets for report workers."""

from __future__ import annotations

import hashlib
import json
import os
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
                f".{bundle_path.name}.{uuid.uuid4().hex}.staging"
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


__all__ = [
    "PREPARED_RECORD_SCHEMA",
    "PreparedModelError",
    "ensure_prepared_model",
    "prepared_identity",
    "validate_prepared_record",
]
