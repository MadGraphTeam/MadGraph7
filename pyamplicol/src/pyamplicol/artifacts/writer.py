# SPDX-License-Identifier: 0BSD
"""High-level schema-v3 artifact builder."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .manifest import (
    MANIFEST_NAME,
    PayloadRecord,
    canonical_manifest_bytes,
    compute_artifact_id,
    load_manifest,
)
from .security import (
    confined_path,
    executable_bit,
    fsync_file,
    normalize_relative_path,
    sha256_file,
)
from .transaction import ArtifactTransaction, ArtifactWriteMode


class ArtifactBuilder:
    """Build one artifact in a private staging directory."""

    def __init__(
        self,
        destination: str | Path,
        *,
        mode: ArtifactWriteMode = "error",
    ) -> None:
        self._transaction = ArtifactTransaction(destination, mode=mode)
        self.root: Path | None = None
        self._payloads: dict[str, PayloadRecord] = {}

    def __enter__(self) -> ArtifactBuilder:
        self.root = self._transaction.__enter__()
        if self._transaction.mode == "append":
            existing = self.root / MANIFEST_NAME
            if existing.is_file():
                manifest = load_manifest(self.root)
                self._payloads.update(
                    (record.path, record) for record in manifest.payloads
                )
                existing.unlink()
        return self

    def _root(self) -> Path:
        if self.root is None:
            raise RuntimeError("artifact builder is not active")
        return self.root

    def add_bytes(
        self,
        relative: str,
        content: bytes,
        *,
        role: str,
        media_type: str,
        executable: bool = False,
        target: Mapping[str, object] | None = None,
        process_id: str | None = None,
    ) -> PayloadRecord:
        path_value = normalize_relative_path(relative)
        if path_value == MANIFEST_NAME:
            raise ValueError(f"{MANIFEST_NAME} is reserved for the artifact manifest")
        path = confined_path(self._root(), path_value, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o755 if executable else 0o644)
        os.replace(temporary, path)
        record = PayloadRecord(
            path=path_value,
            role=role,
            media_type=media_type,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            executable=executable_bit(path),
            target=dict(target) if target is not None else None,
            process_id=process_id,
        )
        self._payloads[path_value] = record
        return record

    def add_json(
        self,
        relative: str,
        value: object,
        *,
        role: str,
        process_id: str | None = None,
        compact: bool = False,
    ) -> PayloadRecord:
        content = (
            json.dumps(
                value,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        return self.add_bytes(
            relative,
            content,
            role=role,
            media_type="application/json",
            process_id=process_id,
        )

    def add_file(
        self,
        relative: str,
        source: str | Path,
        *,
        role: str,
        media_type: str,
        executable: bool | None = None,
        target: Mapping[str, object] | None = None,
        process_id: str | None = None,
    ) -> PayloadRecord:
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f"artifact source must be a regular file: {source_path}")
        return self.add_bytes(
            relative,
            source_path.read_bytes(),
            role=role,
            media_type=media_type,
            executable=(
                executable_bit(source_path) if executable is None else executable
            ),
            target=target,
            process_id=process_id,
        )

    def finalize(
        self,
        *,
        kind: Literal["pyamplicol-process", "pyamplicol-process-set"],
        producer: Mapping[str, object],
        model: Mapping[str, object],
        configuration: Mapping[str, object],
        processes: Sequence[Mapping[str, object]],
        runtime: Mapping[str, object],
        dependencies: Sequence[Mapping[str, object]] = (),
        default_process_id: str | None = None,
        extensions: Mapping[str, object] | None = None,
    ) -> Path:
        if not self._payloads:
            raise ValueError("an artifact must contain at least one payload")
        manifest: dict[str, object] = {
            "schema_version": 3,
            "kind": kind,
            "artifact_id": "0" * 64,
            "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "producer": dict(producer),
            "model": dict(model),
            "configuration": dict(configuration),
            "processes": [dict(process) for process in processes],
            "runtime": dict(runtime),
            "payloads": [
                record.as_dict()
                for record in sorted(
                    self._payloads.values(),
                    key=lambda item: item.path,
                )
            ],
            "dependencies": [dict(dependency) for dependency in dependencies],
            "extensions": dict(extensions or {}),
        }
        if default_process_id is not None:
            manifest["default_process_id"] = default_process_id
        manifest["artifact_id"] = compute_artifact_id(manifest)
        path = self._root() / MANIFEST_NAME
        path.write_bytes(canonical_manifest_bytes(manifest))
        path.chmod(0o644)
        fsync_file(path)
        return path

    def __exit__(self, *exception: object) -> bool:
        if exception[0] is None and not (self._root() / MANIFEST_NAME).is_file():
            raise RuntimeError("artifact builder exited without finalize()")
        return self._transaction.__exit__(*exception)  # type: ignore[arg-type]


__all__ = ["ArtifactBuilder"]
