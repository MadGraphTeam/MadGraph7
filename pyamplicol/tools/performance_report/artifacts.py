# SPDX-License-Identifier: 0BSD
"""Immutable report artifacts, validated current pointers, and filesystem locks."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO

from .models import ArtifactPolicy

if TYPE_CHECKING:
    from .service import ReportPaths

ATTEMPT_SCHEMA = "pyamplicol-performance-attempt-v1"
CURRENT_SCHEMA = "pyamplicol-performance-current-v1"
PORTABLE_CURRENT_SCHEMA = "pyamplicol-performance-portable-current-v1"

_ATTEMPT_KEYS = {
    "schema",
    "cell_id",
    "attempt_id",
    "status",
    "artifact_policy",
    "based_on",
    "result_path",
    "artifacts",
    "error",
}
_CURRENT_KEYS = {
    "schema",
    "cell_id",
    "attempt_id",
    "manifest_path",
    "manifest_sha256",
}
_FILE_KEYS = {"path", "size", "sha256"}
_BASED_ON_KEYS = {"attempt_id", "manifest_sha256"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LOCK_STATE = threading.local()


class ArtifactStoreError(RuntimeError):
    """Base error for the immutable report artifact store."""


class ManifestValidationError(ArtifactStoreError):
    """An artifact manifest or current pointer failed validation."""


class LockTimeoutError(ArtifactStoreError):
    """A filesystem lock could not be acquired before its deadline."""


class LockCancelledError(ArtifactStoreError):
    """A caller cancelled while waiting for a filesystem lock."""


class DiskFullError(ArtifactStoreError):
    """A report write failed because its filesystem has no free space."""


class ArtifactAction(StrEnum):
    """Concrete action selected from an artifact policy and current state."""

    REUSE_CURRENT = "reuse-current"
    RETIME_CURRENT = "retime-current"
    GENERATE = "generate"


def _thread_lock_depths() -> dict[Path, int]:
    process_id = os.getpid()
    if getattr(_LOCK_STATE, "process_id", None) != process_id:
        _LOCK_STATE.process_id = process_id
        _LOCK_STATE.depths = {}
    depths = getattr(_LOCK_STATE, "depths", None)
    if depths is None:
        depths = {}
        _LOCK_STATE.depths = depths
    return depths


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    relative_path: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CurrentRecord:
    cell_id: str
    attempt_id: str
    manifest_path: Path
    manifest_sha256: str
    result_path: Path
    result: Mapping[str, Any]
    artifacts: tuple[ArtifactFile, ...]
    artifact_policy: ArtifactPolicy


@dataclass(frozen=True, slots=True)
class ArtifactDecision:
    requested_policy: ArtifactPolicy
    action: ArtifactAction
    current: CurrentRecord | None

    @property
    def requires_generation(self) -> bool:
        return self.action is ArtifactAction.GENERATE

    @property
    def requires_timing(self) -> bool:
        return self.action is not ArtifactAction.REUSE_CURRENT


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactStoreError(f"payload is not canonical JSON: {error}") from error
    return f"{encoded}\n".encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _raise_disk_full(error: OSError, path: Path) -> None:
    if error.errno != errno.ENOSPC:
        raise error
    probe = path if path.is_dir() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
        detail = f"; {free} bytes available"
    except OSError:
        detail = ""
    raise DiskFullError(f"disk full while writing {path}{detail}") from error


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4()}")
        data = _canonical_json_bytes(payload)
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        _raise_disk_full(error, path)
    finally:
        if "temporary" in locals():
            temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"could not read {description} {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ManifestValidationError(f"{description} must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ManifestValidationError(f"{field} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ManifestValidationError(f"{field} must be a UUID string") from error
    if str(parsed) != value:
        raise ManifestValidationError(f"{field} must use canonical UUID spelling")
    return value


def _validate_cell_id(cell_id: object) -> str:
    if not isinstance(cell_id, str) or not cell_id or "\x00" in cell_id:
        raise ManifestValidationError("cell_id must be a non-empty string")
    if len(cell_id.encode()) > 4096:
        raise ManifestValidationError("cell_id is unreasonably long")
    return cell_id


def _safe_component(value: str, *, prefix_length: int = 48) -> str:
    digest = hashlib.sha256(value.encode()).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not prefix:
        prefix = "item"
    return f"{prefix[:prefix_length]}-{digest}"


def _strict_relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestValidationError(f"{field} contains an unsafe path: {value!r}")
    return path


def _resolve_member(
    root: Path,
    relative: object,
    *,
    field: str,
    require_file: bool = True,
) -> tuple[str, Path]:
    relative_path = _strict_relative_path(relative, field=field)
    candidate = root.joinpath(relative_path)
    if candidate.is_symlink():
        raise ManifestValidationError(
            f"{field} is not a regular attempt file: {relative_path}"
        )
    try:
        resolved = candidate.resolve(strict=require_file)
    except OSError as error:
        raise ManifestValidationError(
            f"{field} does not resolve to a readable file: {relative_path}"
        ) from error
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ManifestValidationError(
            f"{field} escapes its attempt directory: {relative_path}"
        ) from error
    if require_file and (not resolved.is_file() or candidate.is_symlink()):
        raise ManifestValidationError(
            f"{field} is not a regular attempt file: {relative_path}"
        )
    return relative_path.as_posix(), resolved


@contextmanager
def _filesystem_lock(
    path: Path,
    *,
    timeout: float | None,
    poll_interval: float,
    cancellation_requested: Callable[[], bool] | None = None,
) -> Iterator[None]:
    if timeout is not None and timeout < 0:
        raise ValueError("lock timeout must be non-negative or None")
    if poll_interval <= 0:
        raise ValueError("lock poll interval must be positive")
    lock_path = path.expanduser().resolve(strict=False)
    depths = _thread_lock_depths()
    depth = depths.get(lock_path, 0)
    if depth:
        depths[lock_path] = depth + 1
        try:
            yield
        finally:
            remaining = depths[lock_path] - 1
            if remaining:
                depths[lock_path] = remaining
            else:
                del depths[lock_path]
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream: BinaryIO = lock_path.open("a+b")
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while True:
            if cancellation_requested is not None and cancellation_requested():
                raise LockCancelledError(
                    f"cancelled while waiting for filesystem lock {lock_path}"
                )
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockTimeoutError(
                        f"timed out acquiring filesystem lock {lock_path}"
                    ) from error
                remaining = (
                    poll_interval
                    if deadline is None
                    else min(poll_interval, max(0.0, deadline - time.monotonic()))
                )
                time.sleep(remaining)
        depths[lock_path] = 1
        yield
    finally:
        depths.pop(lock_path, None)
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


class ArtifactStore:
    """Store immutable attempts and atomically select one validated current result."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        lock_root: Path,
        current_publication_paths: ReportPaths | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).expanduser().resolve(strict=False)
        self.lock_root = Path(lock_root).expanduser().resolve(strict=False)
        self.current_publication_paths = current_publication_paths
        if current_publication_paths is not None and (
            current_publication_paths.artifact_root != self.artifact_root
            or current_publication_paths.coordination_root != self.lock_root
        ):
            raise ValueError(
                "portable current roots must match the artifact store roots"
            )
        self.cells_root = self.artifact_root / "cells"
        self.attempt_history_root = self.artifact_root / "attempt-history"
        self.cells_root.mkdir(parents=True, exist_ok=True)
        self.lock_root.mkdir(parents=True, exist_ok=True)

    @property
    def current_schema(self) -> str:
        return (
            CURRENT_SCHEMA
            if self.current_publication_paths is None
            else PORTABLE_CURRENT_SCHEMA
        )

    def _stored_current_result(
        self,
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.current_publication_paths is None:
            return result
        from .publication import portable_current_value

        portable = portable_current_value(result, self.current_publication_paths)
        if not isinstance(portable, Mapping):  # pragma: no cover - input is a mapping.
            raise ArtifactStoreError("portable current result must remain an object")
        return portable

    def materialize_current_result(
        self,
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Resolve an authenticated stored current against this store's roots."""

        if self.current_publication_paths is None:
            return result
        from .publication import PublicationPortabilityError, materialize_current_value

        try:
            materialized = materialize_current_value(
                result,
                self.current_publication_paths,
            )
        except PublicationPortabilityError as error:
            raise ManifestValidationError(
                f"portable current result contains an unsafe locator: {error}"
            ) from error
        if not isinstance(materialized, Mapping):  # pragma: no cover - input is object.
            raise ManifestValidationError(
                "portable current result did not materialize as an object"
            )
        return materialized

    def _cell_root(self, cell_id: str) -> Path:
        validated = _validate_cell_id(cell_id)
        return self.cells_root / _safe_component(validated)

    def _validate_cells_root(self) -> None:
        if self.cells_root.is_symlink() or not self.cells_root.is_dir():
            raise ManifestValidationError(
                f"artifact cells root is not a regular directory: {self.cells_root}"
            )
        try:
            resolved = self.cells_root.resolve(strict=True)
        except OSError as error:
            raise ManifestValidationError(
                f"artifact cells root is unavailable: {self.cells_root}"
            ) from error
        if resolved != self.cells_root:
            raise ManifestValidationError(
                f"artifact cells root escapes its canonical path: {self.cells_root}"
            )

    def _validate_cell_root_path(
        self,
        cell_root: Path,
        *,
        missing_ok: bool,
    ) -> Path | None:
        self._validate_cells_root()
        if cell_root.parent != self.cells_root:
            raise ManifestValidationError(
                f"cell root is outside the artifact inventory: {cell_root}"
            )
        if cell_root.is_symlink():
            raise ManifestValidationError(
                f"cell root is a symbolic link: {cell_root}"
            )
        if not cell_root.exists():
            if missing_ok:
                return None
            raise ManifestValidationError(
                f"artifact cell root is absent: {cell_root}"
            )
        if not cell_root.is_dir():
            raise ManifestValidationError(
                f"cell root is not a regular directory: {cell_root}"
            )
        try:
            resolved = cell_root.resolve(strict=True)
        except OSError as error:
            raise ManifestValidationError(
                f"cell root is unavailable: {cell_root}"
            ) from error
        if resolved != cell_root:
            raise ManifestValidationError(
                f"cell root escapes its canonical path: {cell_root}"
            )
        return cell_root

    def _existing_cell_root(
        self,
        cell_id: str,
        *,
        missing_ok: bool,
    ) -> Path | None:
        return self._validate_cell_root_path(
            self._cell_root(cell_id),
            missing_ok=missing_ok,
        )

    def _cell_lock_path(self, cell_id: str) -> Path:
        validated = _validate_cell_id(cell_id)
        return self.lock_root / "cells" / f"{_safe_component(validated)}.lock"

    def _attempt_root(self, cell_id: str, attempt_id: str) -> Path:
        validated_cell = _validate_cell_id(cell_id)
        validated_attempt = _validate_uuid(attempt_id, field="attempt_id")
        return self._cell_root(validated_cell) / "attempts" / validated_attempt

    def _attempt_history_cell_root(self, cell_id: str) -> Path:
        validated = _validate_cell_id(cell_id)
        return self.attempt_history_root / _safe_component(validated)

    @staticmethod
    def _validate_direct_inventory_directory(
        path: Path,
        *,
        parent: Path,
        description: str,
        missing_ok: bool,
    ) -> Path | None:
        if path.parent != parent:
            raise ManifestValidationError(
                f"{description} is outside its canonical inventory: {path}"
            )
        if path.is_symlink():
            raise ManifestValidationError(f"{description} is a symbolic link: {path}")
        if not path.exists():
            if missing_ok:
                return None
            raise ManifestValidationError(f"{description} is absent: {path}")
        if not path.is_dir():
            raise ManifestValidationError(
                f"{description} is not a regular directory: {path}"
            )
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ManifestValidationError(
                f"{description} is unavailable: {path}"
            ) from error
        if resolved != path:
            raise ManifestValidationError(
                f"{description} escapes its canonical path: {path}"
            )
        return path

    def _existing_live_attempt_root(
        self,
        cell_id: str,
        attempt_id: str,
    ) -> Path | None:
        cell_root = self._existing_cell_root(cell_id, missing_ok=True)
        if cell_root is None:
            return None
        attempts_root = self._validate_direct_inventory_directory(
            cell_root / "attempts",
            parent=cell_root,
            description="attempts root",
            missing_ok=True,
        )
        if attempts_root is None:
            return None
        return self._validate_direct_inventory_directory(
            attempts_root / _validate_uuid(attempt_id, field="attempt_id"),
            parent=attempts_root,
            description="attempt root",
            missing_ok=True,
        )

    def _existing_archived_attempt_root(
        self,
        cell_id: str,
        attempt_id: str,
    ) -> Path | None:
        history_root = self._validate_direct_inventory_directory(
            self.attempt_history_root,
            parent=self.artifact_root,
            description="attempt history root",
            missing_ok=True,
        )
        if history_root is None:
            return None
        history_cell = self._validate_direct_inventory_directory(
            self._attempt_history_cell_root(cell_id),
            parent=history_root,
            description="attempt history cell root",
            missing_ok=True,
        )
        if history_cell is None:
            return None
        return self._validate_direct_inventory_directory(
            history_cell / _validate_uuid(attempt_id, field="attempt_id"),
            parent=history_cell,
            description="archived attempt root",
            missing_ok=True,
        )

    @contextmanager
    def cell_lock(
        self,
        cell_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        with _filesystem_lock(
            self._cell_lock_path(cell_id),
            timeout=timeout,
            poll_interval=poll_interval,
            cancellation_requested=cancellation_requested,
        ):
            yield

    @contextmanager
    def named_lock(
        self,
        name: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        validated = _validate_cell_id(name)
        path = self.lock_root / "named" / f"{_safe_component(validated)}.lock"
        with _filesystem_lock(
            path,
            timeout=timeout,
            poll_interval=poll_interval,
            cancellation_requested=cancellation_requested,
        ):
            yield

    def decide(
        self,
        cell_id: str,
        policy: ArtifactPolicy,
    ) -> ArtifactDecision:
        current = self.load_current(cell_id, missing_ok=True)
        if policy is ArtifactPolicy.REUSE and current is not None:
            action = ArtifactAction.REUSE_CURRENT
        elif policy is ArtifactPolicy.RETIME and current is not None:
            action = ArtifactAction.RETIME_CURRENT
        else:
            action = ArtifactAction.GENERATE
        return ArtifactDecision(policy, action, current)

    def new_attempt(
        self,
        cell_id: str,
        policy: ArtifactPolicy,
        *,
        based_on: CurrentRecord | None = None,
    ) -> ArtifactAttempt:
        validated = _validate_cell_id(cell_id)
        if based_on is not None and based_on.cell_id != validated:
            raise ArtifactStoreError("based-on record belongs to a different cell")
        attempt_id = str(uuid.uuid4())
        self._validate_cells_root()
        cell_root = self._cell_root(validated)
        if cell_root.exists() or cell_root.is_symlink():
            self._validate_cell_root_path(cell_root, missing_ok=False)
        root = cell_root / "attempts" / attempt_id
        try:
            root.mkdir(parents=True, exist_ok=False)
            _fsync_directory(root.parent)
        except OSError as error:
            _raise_disk_full(error, root)
        return ArtifactAttempt(
            store=self,
            cell_id=validated,
            attempt_id=attempt_id,
            root=root,
            artifact_policy=policy,
            based_on=based_on,
        )

    def seal_existing_worker_result(
        self,
        cell_id: str,
        attempt_id: str,
        *,
        worker_result_sha256: str,
        artifact_policy: ArtifactPolicy,
        validate_result: Callable[[Mapping[str, Any], Path], None],
    ) -> CurrentRecord:
        """Seal one controller-orphaned worker result without rerunning it."""

        validated_cell = _validate_cell_id(cell_id)
        validated_attempt = _validate_uuid(attempt_id, field="attempt_id")
        expected_worker_digest = _validate_sha256(
            worker_result_sha256,
            field="worker_result_sha256",
        )
        cell_root = self._existing_cell_root(validated_cell, missing_ok=False)
        assert cell_root is not None
        attempt_root = cell_root / "attempts" / validated_attempt
        with self.cell_lock(validated_cell):
            pointer_path = cell_root / "current.json"
            if pointer_path.exists() or pointer_path.is_symlink():
                raise ArtifactStoreError(
                    "orphan sealing cannot replace an existing current"
                )
            if (
                attempt_root.is_symlink()
                or not attempt_root.is_dir()
                or attempt_root.parent.is_symlink()
            ):
                raise ArtifactStoreError(
                    "orphan attempt is not its canonical regular directory"
                )
            for reserved in ("result.json", "manifest.json"):
                path = attempt_root / reserved
                if path.exists() or path.is_symlink():
                    raise ArtifactStoreError(
                        f"orphan attempt already contains {reserved}"
                    )
            _, worker_result = _resolve_member(
                attempt_root,
                "worker-result.json",
                field="worker-result.json",
            )
            if _sha256(worker_result) != expected_worker_digest:
                raise ArtifactStoreError("worker-result.json digest mismatch")
            result = _read_json_object(
                worker_result,
                description="orphan worker result",
            )
            validate_result(result, attempt_root)
            artifact_paths: list[str] = []
            for path in sorted(attempt_root.rglob("*")):
                if path.is_symlink():
                    raise ArtifactStoreError(
                        f"orphan attempt contains a symbolic link: {path}"
                    )
                if path.is_file():
                    artifact_paths.append(path.relative_to(attempt_root).as_posix())
                elif not path.is_dir():
                    raise ArtifactStoreError(
                        f"orphan attempt contains a special file: {path}"
                    )
            if "worker-result.json" not in artifact_paths:
                raise ArtifactStoreError("orphan attempt lacks worker-result.json")
            attempt = ArtifactAttempt(
                store=self,
                cell_id=validated_cell,
                attempt_id=validated_attempt,
                root=attempt_root,
                artifact_policy=artifact_policy,
                based_on=None,
            )
            record = attempt.publish(
                result,
                artifact_paths=artifact_paths,
            )
            if _sha256(worker_result) != expected_worker_digest:
                raise ArtifactStoreError(
                    "worker-result.json changed during orphan sealing"
                )
            return record

    def load_current(
        self,
        cell_id: str,
        *,
        missing_ok: bool = False,
    ) -> CurrentRecord | None:
        validated = _validate_cell_id(cell_id)
        cell_root = self._existing_cell_root(validated, missing_ok=True)
        if cell_root is None:
            if missing_ok:
                return None
            raise FileNotFoundError(f"no artifact cell for {validated!r}")
        pointer_path = cell_root / "current.json"
        if pointer_path.is_symlink():
            raise ManifestValidationError(
                f"current pointer is a symbolic link: {pointer_path}"
            )
        if not pointer_path.exists():
            if missing_ok:
                return None
            raise FileNotFoundError(f"no current artifact for cell {validated!r}")
        if not pointer_path.is_file():
            raise ManifestValidationError(
                f"current pointer is not a regular file: {pointer_path}"
            )
        return self._validate_current_pointer(
            pointer_path,
            expected_cell_id=validated,
            expected_cell_root=cell_root,
        )

    def recover_current_records(self) -> tuple[CurrentRecord, ...]:
        records: list[CurrentRecord] = []
        self._validate_cells_root()
        for pointer_path in sorted(self.cells_root.glob("*/current.json")):
            self._validate_cell_root_path(pointer_path.parent, missing_ok=False)
            records.append(
                self._validate_current_pointer(
                    pointer_path,
                    expected_cell_id=None,
                    expected_cell_root=pointer_path.parent,
                )
            )
        return tuple(records)

    def cell_attempt_ids(self, cell_id: str) -> tuple[str, ...]:
        """Return the canonical immutable-attempt inventory for one cell."""

        cell_root = self._existing_cell_root(cell_id, missing_ok=True)
        if cell_root is None:
            return ()
        attempts_root = cell_root / "attempts"
        if not attempts_root.exists():
            return ()
        if attempts_root.is_symlink() or not attempts_root.is_dir():
            raise ManifestValidationError(
                f"attempt inventory is not a regular directory: {attempts_root}"
            )
        attempt_ids: list[str] = []
        for path in sorted(attempts_root.iterdir()):
            if path.is_symlink() or not path.is_dir():
                raise ManifestValidationError(
                    f"attempt inventory contains a non-directory: {path}"
                )
            attempt_ids.append(
                _validate_uuid(path.name, field="attempt inventory member")
            )
        return tuple(attempt_ids)

    def load_sealed_unsuccessful_worker_result(
        self,
        cell_id: str,
        attempt_id: str,
    ) -> Mapping[str, Any]:
        """Load one explicitly named failed result through its sealed manifest.

        This is deliberately not a history search.  Presentation metadata
        names the exact attempt UUID, and this method authenticates only that
        attempt's compact ``worker-result.json``.  It supports the canonical
        live and archived locations used by bounded artifact cleanup.
        """

        validated_cell = _validate_cell_id(cell_id)
        validated_attempt = _validate_uuid(attempt_id, field="attempt_id")
        live_root = self._existing_live_attempt_root(
            validated_cell,
            validated_attempt,
        )
        archived_root = self._existing_archived_attempt_root(
            validated_cell,
            validated_attempt,
        )
        roots = tuple(
            root
            for root in (live_root, archived_root)
            if root is not None
        )
        if len(roots) != 1:
            raise ManifestValidationError(
                "named unsuccessful attempt is absent or has colliding locations"
            )
        root = roots[0]
        archived = root == archived_root
        if root.is_symlink() or not root.is_dir():
            raise ManifestValidationError(
                f"named unsuccessful attempt is not a regular directory: {root}"
            )
        manifest_path = root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ManifestValidationError(
                f"named unsuccessful attempt has no regular manifest: {root}"
            )
        manifest = _read_json_object(
            manifest_path,
            description="unsuccessful attempt manifest",
        )
        if (
            set(manifest) != _ATTEMPT_KEYS
            or manifest.get("schema") != ATTEMPT_SCHEMA
            or manifest.get("cell_id") != validated_cell
            or manifest.get("attempt_id") != validated_attempt
            or manifest.get("status") not in {"failed", "interrupted"}
            or manifest.get("result_path") is not None
            or not isinstance(manifest.get("error"), str)
            or not manifest.get("error")
        ):
            raise ManifestValidationError(
                "named unsuccessful attempt manifest is invalid"
            )
        try:
            ArtifactPolicy(manifest.get("artifact_policy"))
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                "unsuccessful attempt artifact policy is invalid"
            ) from error
        self._validate_based_on(manifest.get("based_on"))
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ManifestValidationError(
                "unsuccessful attempt artifacts must be an array"
            )
        artifacts: list[ArtifactFile] = []
        for index, raw in enumerate(raw_artifacts):
            if not isinstance(raw, dict) or set(raw) != _FILE_KEYS:
                raise ManifestValidationError(
                    f"artifact record {index} has an unsupported schema shape"
                )
            relative = _strict_relative_path(
                raw.get("path"),
                field=f"artifacts[{index}].path",
            )
            self._validate_unsuccessful_artifact_ancestors(root, relative)
            candidate = root.joinpath(relative)
            if archived and not candidate.exists() and not candidate.is_symlink():
                if not relative.parts or relative.parts[0] != "artifact":
                    raise ManifestValidationError(
                        "archived compact diagnostic artifact is missing: "
                        f"{relative.as_posix()}"
                    )
                self._validate_missing_archived_artifact_record(
                    raw,
                    index=index,
                )
                continue
            artifacts.append(self._validate_artifact_file(root, raw, index=index))
        paths = [
            _strict_relative_path(raw.get("path"), field="artifact path").as_posix()
            for raw in raw_artifacts
            if isinstance(raw, Mapping)
        ]
        if len(paths) != len(set(paths)):
            raise ManifestValidationError(
                "unsuccessful attempt manifest has duplicate artifacts"
            )
        matches = tuple(
            artifact
            for artifact in artifacts
            if artifact.relative_path == "worker-result.json"
        )
        if len(matches) != 1:
            raise ManifestValidationError(
                "unsuccessful attempt does not authenticate worker-result.json"
            )
        return self.materialize_current_result(
            _read_json_object(
                matches[0].path,
                description="unsuccessful worker result",
            )
        )

    @staticmethod
    def _validate_unsuccessful_artifact_ancestors(
        attempt_root: Path,
        relative: Path,
    ) -> None:
        cursor = attempt_root
        for offset, part in enumerate(relative.parts):
            cursor = cursor / part
            if cursor.is_symlink():
                raise ManifestValidationError(
                    "unsuccessful artifact traverses a symbolic link: "
                    f"{relative.as_posix()}"
                )
            if (
                offset < len(relative.parts) - 1
                and cursor.exists()
                and not cursor.is_dir()
            ):
                raise ManifestValidationError(
                    "unsuccessful artifact traverses a non-directory: "
                    f"{relative.as_posix()}"
                )

    @staticmethod
    def _validate_missing_archived_artifact_record(
        raw_record: Mapping[str, object],
        *,
        index: int,
    ) -> None:
        size = raw_record.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ManifestValidationError(
                f"artifacts[{index}].size must be a non-negative integer"
            )
        _validate_sha256(
            raw_record.get("sha256"),
            field=f"artifacts[{index}].sha256",
        )

    def lightweight_current_payload(
        self,
        cell_id: str,
        *,
        missing_ok: bool = False,
    ) -> tuple[str, Mapping[str, Any]] | None:
        """Read authenticated pointer/manifest/result metadata without big files.

        This deliberately authenticates the small manifest and result records
        only.  It is used solely to protect live artifact owners during history
        pruning; ordinary current loading retains full artifact validation.
        """

        validated = _validate_cell_id(cell_id)
        cell_root = self._existing_cell_root(validated, missing_ok=True)
        if cell_root is None:
            if missing_ok:
                return None
            raise FileNotFoundError(f"no artifact cell for {validated!r}")
        pointer_path = cell_root / "current.json"
        if pointer_path.is_symlink():
            raise ManifestValidationError(
                f"current pointer is a symbolic link: {pointer_path}"
            )
        if not pointer_path.exists():
            if missing_ok:
                return None
            raise FileNotFoundError(f"no current artifact for cell {validated!r}")
        if not pointer_path.is_file():
            raise ManifestValidationError(
                f"current pointer is not a regular file: {pointer_path}"
            )
        pointer = _read_json_object(pointer_path, description="current pointer")
        if (
            set(pointer) != _CURRENT_KEYS
            or pointer.get("schema") != self.current_schema
        ):
            raise ManifestValidationError("current pointer has an unsupported shape")
        if pointer.get("cell_id") != validated:
            raise ManifestValidationError("current pointer cell_id does not match")
        attempt_id = _validate_uuid(pointer.get("attempt_id"), field="attempt_id")
        expected_manifest = f"attempts/{attempt_id}/manifest.json"
        if pointer.get("manifest_path") != expected_manifest:
            raise ManifestValidationError(
                "current pointer manifest_path is not its canonical attempt manifest"
            )
        _, manifest_path = _resolve_member(
            cell_root,
            expected_manifest,
            field="manifest_path",
        )
        expected_digest = _validate_sha256(
            pointer.get("manifest_sha256"), field="manifest_sha256"
        )
        if _sha256(manifest_path) != expected_digest:
            raise ManifestValidationError("current manifest digest mismatch")
        manifest = _read_json_object(manifest_path, description="attempt manifest")
        if (
            set(manifest) != _ATTEMPT_KEYS
            or manifest.get("schema") != ATTEMPT_SCHEMA
            or manifest.get("cell_id") != validated
            or manifest.get("attempt_id") != attempt_id
            or manifest.get("status") != "ok"
            or manifest.get("error") is not None
        ):
            raise ManifestValidationError("current attempt manifest is invalid")
        try:
            ArtifactPolicy(manifest.get("artifact_policy"))
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                "current manifest artifact policy is invalid"
            ) from error
        self._validate_based_on(manifest.get("based_on"))
        result_relative = _strict_relative_path(
            manifest.get("result_path"), field="result_path"
        ).as_posix()
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise ManifestValidationError("current manifest artifacts are invalid")
        artifact_paths: list[str] = []
        for index, record in enumerate(raw_artifacts):
            if not isinstance(record, dict) or set(record) != _FILE_KEYS:
                raise ManifestValidationError(
                    f"current artifact record {index} is invalid"
                )
            relative = _strict_relative_path(
                record.get("path"), field=f"artifacts[{index}].path"
            ).as_posix()
            _validate_sha256(record.get("sha256"), field=f"artifacts[{index}].sha256")
            size = record.get("size")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ManifestValidationError(
                    f"artifacts[{index}].size must be a non-negative integer"
                )
            artifact_paths.append(relative)
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ManifestValidationError("current manifest has duplicate artifacts")
        result_record = next(
            (
                record
                for record in raw_artifacts
                if isinstance(record, dict) and record.get("path") == result_relative
            ),
            None,
        )
        if result_record is None:
            raise ManifestValidationError("current result is absent from manifest")
        result_file = self._validate_artifact_file(
            manifest_path.parent,
            result_record,
            index=raw_artifacts.index(result_record),
        )
        result = self.materialize_current_result(
            _read_json_object(result_file.path, description="attempt result")
        )
        return attempt_id, result

    def protected_attempt_ids(
        self,
        cell_id: str,
        *,
        consumer_cell_ids: Iterable[str],
    ) -> frozenset[str]:
        """Return target attempts retained by live current/equivalent consumers."""

        validated = _validate_cell_id(cell_id)
        attempt_ids = self.cell_attempt_ids(validated)
        candidate_artifacts = {
            self._attempt_root(validated, attempt_id)
            .joinpath("artifact")
            .resolve(strict=False): attempt_id
            for attempt_id in attempt_ids
        }
        protected: set[str] = set()
        own = self.lightweight_current_payload(validated, missing_ok=True)
        if own is not None:
            protected.add(own[0])
        consumers = {_validate_cell_id(value) for value in consumer_cell_ids}
        for consumer_id in sorted(consumers):
            current = self.lightweight_current_payload(consumer_id, missing_ok=True)
            if current is None:
                continue
            artifact = current[1].get("artifact")
            raw_path = artifact.get("path") if isinstance(artifact, Mapping) else None
            if not isinstance(raw_path, str) or not raw_path:
                continue
            if not Path(raw_path).is_absolute():
                raise ManifestValidationError(
                    f"current artifact path is not absolute: {raw_path!r}"
                )
            referenced_artifact = Path(raw_path)
            if referenced_artifact.is_symlink() or not referenced_artifact.is_dir():
                raise ManifestValidationError(
                    f"current artifact path is unavailable or unsafe: {raw_path!r}"
                )
            owner = candidate_artifacts.get(
                referenced_artifact.expanduser().resolve(strict=False)
            )
            if owner is not None:
                protected.add(owner)
        return frozenset(protected)

    def archive_obsolete_attempts(
        self,
        cell_id: str,
        *,
        consumer_cell_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """Archive sealed noncurrent attempts and remove only heavy payloads.

        The caller must already hold the campaign-cell lock.  Per-attempt use
        locks are acquired non-blockingly so a concurrent equivalent reuse is
        never disturbed.
        """

        validated = _validate_cell_id(cell_id)
        if self.attempt_history_root.exists() and (
            self.attempt_history_root.is_symlink()
            or not self.attempt_history_root.is_dir()
        ):
            raise ArtifactStoreError(
                "attempt history root is not a regular directory: "
                f"{self.attempt_history_root}"
            )
        history_cell = self._attempt_history_cell_root(validated)
        try:
            self._prune_archived_payloads(history_cell)
        except OSError as error:
            _raise_disk_full(error, history_cell)
        archived: list[str] = []
        for attempt_id in self.cell_attempt_ids(validated):
            try:
                with self.named_lock(
                    f"campaign-artifact-use-{attempt_id}", timeout=0.0
                ):
                    protected = self.protected_attempt_ids(
                        validated,
                        consumer_cell_ids=consumer_cell_ids,
                    )
                    if attempt_id in protected:
                        continue
                    source = self._attempt_root(validated, attempt_id)
                    manifest = source / "manifest.json"
                    if (
                        source.is_symlink()
                        or not source.is_dir()
                        or manifest.is_symlink()
                        or not manifest.is_file()
                    ):
                        continue
                    raw = _read_json_object(manifest, description="attempt manifest")
                    if (
                        set(raw) != _ATTEMPT_KEYS
                        or raw.get("schema") != ATTEMPT_SCHEMA
                        or raw.get("cell_id") != validated
                        or raw.get("attempt_id") != attempt_id
                        or raw.get("status") not in {"ok", "failed", "interrupted"}
                    ):
                        raise ManifestValidationError(
                            f"sealed attempt metadata is invalid: {source}"
                        )
                    if history_cell.exists() and (
                        history_cell.is_symlink() or not history_cell.is_dir()
                    ):
                        raise ArtifactStoreError(
                            "attempt history is not a regular directory: "
                            f"{history_cell}"
                        )
                    history_cell.mkdir(parents=True, exist_ok=True)
                    destination = history_cell / attempt_id
                    if destination.exists() or destination.is_symlink():
                        raise ArtifactStoreError(
                            f"attempt history collision: {destination}"
                        )
                    self._validate_attempt_tree(source)
                    os.replace(source, destination)
                    _fsync_directory(source.parent)
                    _fsync_directory(history_cell)
                    self._remove_archived_artifact(destination)
                    archived.append(attempt_id)
            except LockTimeoutError:
                continue
            except OSError as error:
                _raise_disk_full(error, history_cell)
        return tuple(archived)

    def _prune_archived_payloads(self, history_cell: Path) -> None:
        if not history_cell.exists():
            return
        if history_cell.is_symlink() or not history_cell.is_dir():
            raise ArtifactStoreError(
                f"attempt history is not a regular directory: {history_cell}"
            )
        for archived in sorted(history_cell.iterdir()):
            if archived.is_symlink() or not archived.is_dir():
                raise ArtifactStoreError(
                    f"attempt history contains a non-directory: {archived}"
                )
            _validate_uuid(archived.name, field="attempt history member")
            self._remove_archived_artifact(archived)

    @staticmethod
    def _validate_attempt_tree(attempt: Path) -> None:
        for root, directories, files in os.walk(attempt, topdown=True):
            root_path = Path(root)
            for name in (*directories, *files):
                member = root_path / name
                if member.is_symlink() or (
                    not member.is_dir() and not member.is_file()
                ):
                    raise ArtifactStoreError(
                        f"attempt contains an unsafe member: {member}"
                    )

    @staticmethod
    def _validate_archived_artifact(archived_attempt: Path) -> None:
        artifact = archived_attempt / "artifact"
        if not artifact.exists() and not artifact.is_symlink():
            return
        if artifact.is_symlink() or not artifact.is_dir():
            raise ArtifactStoreError(
                f"archived artifact payload is not a regular directory: {artifact}"
            )
        for root, directories, files in os.walk(artifact, topdown=True):
            root_path = Path(root)
            for name in (*directories, *files):
                member = root_path / name
                if member.is_symlink() or (
                    not member.is_dir() and not member.is_file()
                ):
                    raise ArtifactStoreError(
                        f"archived artifact payload contains an unsafe member: {member}"
                    )

    @classmethod
    def _remove_archived_artifact(cls, archived_attempt: Path) -> None:
        artifact = archived_attempt / "artifact"
        cls._validate_archived_artifact(archived_attempt)
        if not artifact.exists():
            return
        shutil.rmtree(artifact)
        _fsync_directory(archived_attempt)

    def _validate_current_pointer(
        self,
        pointer_path: Path,
        *,
        expected_cell_id: str | None,
        expected_cell_root: Path,
    ) -> CurrentRecord:
        self._validate_cell_root_path(expected_cell_root, missing_ok=False)
        if (
            pointer_path.parent != expected_cell_root
            or pointer_path.name != "current.json"
            or pointer_path.is_symlink()
            or not pointer_path.is_file()
        ):
            raise ManifestValidationError(
                f"current pointer is not its canonical regular file: {pointer_path}"
            )
        pointer = _read_json_object(pointer_path, description="current pointer")
        if set(pointer) != _CURRENT_KEYS:
            raise ManifestValidationError(
                f"current pointer has an unsupported schema shape: {pointer_path}"
            )
        if pointer["schema"] != self.current_schema:
            raise ManifestValidationError(
                f"unsupported current pointer schema: {pointer['schema']!r}"
            )
        cell_id = _validate_cell_id(pointer["cell_id"])
        if expected_cell_id is not None and cell_id != expected_cell_id:
            raise ManifestValidationError("current pointer cell_id does not match")
        if expected_cell_root != self._cell_root(cell_id):
            raise ManifestValidationError(
                "current pointer is stored under the wrong cell directory"
            )
        attempt_id = _validate_uuid(pointer["attempt_id"], field="attempt_id")
        expected_manifest = f"attempts/{attempt_id}/manifest.json"
        if pointer["manifest_path"] != expected_manifest:
            raise ManifestValidationError(
                "current pointer manifest_path is not its canonical attempt manifest"
            )
        _, manifest_path = _resolve_member(
            expected_cell_root,
            pointer["manifest_path"],
            field="manifest_path",
        )
        expected_digest = _validate_sha256(
            pointer["manifest_sha256"],
            field="manifest_sha256",
        )
        actual_digest = _sha256(manifest_path)
        if actual_digest != expected_digest:
            raise ManifestValidationError("current manifest digest mismatch")
        return self._validate_attempt_manifest(
            manifest_path,
            expected_cell_id=cell_id,
            expected_attempt_id=attempt_id,
            expected_digest=expected_digest,
        )

    def _validate_attempt_manifest(
        self,
        manifest_path: Path,
        *,
        expected_cell_id: str,
        expected_attempt_id: str,
        expected_digest: str,
    ) -> CurrentRecord:
        manifest = _read_json_object(manifest_path, description="attempt manifest")
        if set(manifest) != _ATTEMPT_KEYS:
            raise ManifestValidationError(
                f"attempt manifest has an unsupported schema shape: {manifest_path}"
            )
        if manifest["schema"] != ATTEMPT_SCHEMA:
            raise ManifestValidationError(
                f"unsupported attempt manifest schema: {manifest['schema']!r}"
            )
        if _validate_cell_id(manifest["cell_id"]) != expected_cell_id:
            raise ManifestValidationError("attempt manifest cell_id does not match")
        if (
            _validate_uuid(manifest["attempt_id"], field="attempt_id")
            != expected_attempt_id
        ):
            raise ManifestValidationError("attempt manifest attempt_id does not match")
        if manifest["status"] != "ok":
            raise ManifestValidationError(
                "current pointer targets a non-success attempt"
            )
        if manifest["error"] is not None:
            raise ManifestValidationError("successful attempt contains an error")
        try:
            policy = ArtifactPolicy(manifest["artifact_policy"])
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                "attempt manifest has an unsupported artifact policy"
            ) from error
        self._validate_based_on(manifest["based_on"])
        result_relative = _strict_relative_path(
            manifest["result_path"],
            field="result_path",
        ).as_posix()
        raw_artifacts = manifest["artifacts"]
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ManifestValidationError(
                "successful attempt must contain artifact records"
            )
        attempt_root = manifest_path.parent
        artifacts: list[ArtifactFile] = []
        seen: set[str] = set()
        for index, raw_record in enumerate(raw_artifacts):
            artifacts.append(
                self._validate_artifact_file(
                    attempt_root,
                    raw_record,
                    index=index,
                )
            )
            relative = artifacts[-1].relative_path
            if relative in seen:
                raise ManifestValidationError(
                    f"duplicate artifact path in attempt manifest: {relative}"
                )
            seen.add(relative)
        if result_relative not in seen:
            raise ManifestValidationError(
                "result_path is not authenticated as an artifact"
            )
        result_file = next(
            artifact
            for artifact in artifacts
            if artifact.relative_path == result_relative
        )
        result = self.materialize_current_result(
            _read_json_object(result_file.path, description="attempt result")
        )
        return CurrentRecord(
            cell_id=expected_cell_id,
            attempt_id=expected_attempt_id,
            manifest_path=manifest_path,
            manifest_sha256=expected_digest,
            result_path=result_file.path,
            result=result,
            artifacts=tuple(artifacts),
            artifact_policy=policy,
        )

    @staticmethod
    def _validate_based_on(raw: object) -> None:
        if raw is None:
            return
        if not isinstance(raw, dict) or set(raw) != _BASED_ON_KEYS:
            raise ManifestValidationError("based_on has an unsupported schema shape")
        _validate_uuid(raw["attempt_id"], field="based_on.attempt_id")
        _validate_sha256(
            raw["manifest_sha256"],
            field="based_on.manifest_sha256",
        )

    @staticmethod
    def _validate_artifact_file(
        attempt_root: Path,
        raw_record: object,
        *,
        index: int,
    ) -> ArtifactFile:
        if not isinstance(raw_record, dict) or set(raw_record) != _FILE_KEYS:
            raise ManifestValidationError(
                f"artifact record {index} has an unsupported schema shape"
            )
        relative, path = _resolve_member(
            attempt_root,
            raw_record["path"],
            field=f"artifacts[{index}].path",
        )
        size = raw_record["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ManifestValidationError(
                f"artifacts[{index}].size must be a non-negative integer"
            )
        if path.stat().st_size != size:
            raise ManifestValidationError(f"artifact size mismatch: {relative}")
        expected_digest = _validate_sha256(
            raw_record["sha256"],
            field=f"artifacts[{index}].sha256",
        )
        if _sha256(path) != expected_digest:
            raise ManifestValidationError(f"artifact digest mismatch: {relative}")
        return ArtifactFile(relative, path, size, expected_digest)

    def _publish(self, attempt: ArtifactAttempt) -> CurrentRecord:
        manifest_path = attempt.root / "manifest.json"
        digest = _sha256(manifest_path)
        pointer = {
            "schema": self.current_schema,
            "cell_id": attempt.cell_id,
            "attempt_id": attempt.attempt_id,
            "manifest_path": f"attempts/{attempt.attempt_id}/manifest.json",
            "manifest_sha256": digest,
        }
        with self.cell_lock(attempt.cell_id):
            self._validate_attempt_manifest(
                manifest_path,
                expected_cell_id=attempt.cell_id,
                expected_attempt_id=attempt.attempt_id,
                expected_digest=digest,
            )
            pointer_path = self._cell_root(attempt.cell_id) / "current.json"
            _atomic_write_json(pointer_path, pointer)
            return self._validate_current_pointer(
                pointer_path,
                expected_cell_id=attempt.cell_id,
                expected_cell_root=pointer_path.parent,
            )


class ArtifactAttempt:
    """One immutable UUID-qualified attempt for a report cell."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        cell_id: str,
        attempt_id: str,
        root: Path,
        artifact_policy: ArtifactPolicy,
        based_on: CurrentRecord | None,
    ) -> None:
        self.store = store
        self.cell_id = cell_id
        self.attempt_id = attempt_id
        self.root = root
        self.artifact_policy = artifact_policy
        self.based_on = based_on
        self._sealed = False

    def __enter__(self) -> ArtifactAttempt:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        if not self._sealed:
            if exception_type is None:
                self.mark_interrupted("attempt exited without publication")
            else:
                self.mark_interrupted(f"{exception_type.__name__}: {exception}")
        return False

    def path(self, relative_path: str) -> Path:
        if self._sealed:
            raise ArtifactStoreError("attempt is already sealed")
        _, path = _resolve_member(
            self.root,
            relative_path,
            field="attempt output path",
            require_file=False,
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            _raise_disk_full(error, path)
        return path

    def write_json(
        self,
        relative_path: str,
        payload: Mapping[str, Any],
    ) -> Path:
        path = self.path(relative_path)
        _atomic_write_json(path, payload)
        return path

    def publish(
        self,
        result: Mapping[str, Any],
        *,
        artifact_paths: Iterable[str] = (),
    ) -> CurrentRecord:
        self._require_open()
        self.write_json("result.json", self.store._stored_current_result(result))
        relative_paths = {"result.json", *artifact_paths}
        records = [self._file_record(relative) for relative in sorted(relative_paths)]
        manifest = self._manifest(
            status="ok",
            result_path="result.json",
            artifacts=records,
            error=None,
        )
        _atomic_write_json(self.root / "manifest.json", manifest)
        self._sealed = True
        return self.store._publish(self)

    def mark_failed(
        self,
        error: str,
        *,
        artifact_paths: Iterable[str] = (),
    ) -> None:
        self._seal_unsuccessful(
            "failed",
            error,
            artifact_paths=artifact_paths,
        )

    def mark_interrupted(
        self,
        error: str,
        *,
        artifact_paths: Iterable[str] = (),
    ) -> None:
        self._seal_unsuccessful(
            "interrupted",
            error,
            artifact_paths=artifact_paths,
        )

    def discard(self) -> None:
        """Remove an unpublished attempt instead of retaining partial history."""

        self._require_open()
        if self.root.is_symlink() or not self.root.is_dir():
            raise ArtifactStoreError(
                f"attempt is not its canonical regular directory: {self.root}"
            )
        shutil.rmtree(self.root)
        self._sealed = True

    def _seal_unsuccessful(
        self,
        status: str,
        error: str,
        *,
        artifact_paths: Iterable[str] = (),
    ) -> None:
        self._require_open()
        if not isinstance(error, str) or not error:
            raise ArtifactStoreError("failed attempt error must be a non-empty string")
        manifest = self._manifest(
            status=status,
            result_path=None,
            artifacts=[
                self._file_record(relative) for relative in sorted(set(artifact_paths))
            ],
            error=error,
        )
        _atomic_write_json(self.root / "manifest.json", manifest)
        self._sealed = True

    def _file_record(self, relative_path: str) -> dict[str, Any]:
        relative, path = _resolve_member(
            self.root,
            relative_path,
            field="artifact path",
        )
        if path.name == "manifest.json":
            raise ArtifactStoreError("manifest.json cannot authenticate itself")
        return {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

    def _manifest(
        self,
        *,
        status: str,
        result_path: str | None,
        artifacts: list[dict[str, Any]],
        error: str | None,
    ) -> dict[str, Any]:
        based_on = None
        if self.based_on is not None:
            based_on = {
                "attempt_id": self.based_on.attempt_id,
                "manifest_sha256": self.based_on.manifest_sha256,
            }
        return {
            "schema": ATTEMPT_SCHEMA,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "status": status,
            "artifact_policy": self.artifact_policy.value,
            "based_on": based_on,
            "result_path": result_path,
            "artifacts": artifacts,
            "error": error,
        }

    def _require_open(self) -> None:
        if self._sealed:
            raise ArtifactStoreError("attempt is already sealed")


__all__ = [
    "ATTEMPT_SCHEMA",
    "CURRENT_SCHEMA",
    "PORTABLE_CURRENT_SCHEMA",
    "ArtifactAction",
    "ArtifactAttempt",
    "ArtifactDecision",
    "ArtifactFile",
    "ArtifactStore",
    "ArtifactStoreError",
    "CurrentRecord",
    "DiskFullError",
    "LockCancelledError",
    "LockTimeoutError",
    "ManifestValidationError",
]
