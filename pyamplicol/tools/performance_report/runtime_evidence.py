# SPDX-License-Identifier: 0BSD
"""Checked identities for the Python bytes eligible during final measurements."""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

_HASH_CHUNK_BYTES = 1024 * 1024
_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_PREIMPORT_RUNTIME_IDENTITY: dict[str, object] | None = None


class RuntimeEvidenceError(RuntimeError):
    """A runtime tree cannot be authenticated without a namespace race."""


def _isolated_startup_flags() -> tuple[bool, bool, bool]:
    return (
        bool(sys.flags.isolated),
        bool(sys.flags.no_site),
        bool(sys.flags.ignore_environment),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def source_only_bytecode_policy() -> dict[str, object]:
    """Prove that this process cannot read or write package-local bytecode.

    ``-B`` prevents writes, while an absolute, absent ``PYTHONPYCACHEPREFIX``
    makes the interpreter's cache lookup namespace empty.  The exact random
    prefix is intentionally omitted so identities remain comparable across
    isolated workers.
    """

    prefix_value = sys.pycache_prefix
    isolated, no_site, ignore_environment = _isolated_startup_flags()
    if not (isolated and no_site and ignore_environment):
        raise RuntimeEvidenceError(
            "exact runtime evidence requires isolated Python startup (-I -S)"
        )
    if not sys.dont_write_bytecode:
        raise RuntimeEvidenceError(
            "exact runtime evidence requires Python -B "
            "(sys.dont_write_bytecode must be true)"
        )
    if not isinstance(prefix_value, str) or not prefix_value:
        raise RuntimeEvidenceError(
            "exact runtime evidence requires an absent external "
            "PYTHONPYCACHEPREFIX"
        )
    prefix = Path(prefix_value).expanduser()
    if not prefix.is_absolute():
        raise RuntimeEvidenceError(
            "exact runtime evidence requires an absolute PYTHONPYCACHEPREFIX"
        )
    if prefix.exists():
        raise RuntimeEvidenceError(
            "exact runtime evidence requires PYTHONPYCACHEPREFIX to remain absent"
        )
    return {
        "kind": "pyamplicol-source-only-bytecode-policy-v1",
        "dont_write_bytecode": True,
        "external_pycache_prefix": True,
        "external_pycache_prefix_absent": True,
        "package_local_bytecode_eligible": False,
        "isolated_startup": True,
        "site_initialization": False,
        "python_environment_ignored_at_startup": True,
    }


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in _STABLE_STAT_FIELDS
    )


def _relative_directory(path: str) -> PurePosixPath:
    if path in {"", "."}:
        return PurePosixPath()
    normalized = path[2:] if path.startswith("./") else path
    logical = PurePosixPath(normalized)
    if (
        logical.is_absolute()
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != normalized
    ):
        raise RuntimeEvidenceError(
            f"package traversal produced a noncanonical directory: {path!r}"
        )
    return logical


def _regular_file_record(
    directory_fd: int,
    name: str,
    logical_path: str,
) -> dict[str, object]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor: int | None = None
    try:
        namespace_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(namespace_before.st_mode):
            raise RuntimeEvidenceError(
                f"package member is not a regular file: {logical_path}"
            )
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        opened_before = os.fstat(descriptor)
        if not _same_stat(namespace_before, opened_before):
            raise RuntimeEvidenceError(
                f"package member changed while it was opened: {logical_path}"
            )
        digest = hashlib.sha256()
        byte_count = 0
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            for block in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(block)
                byte_count += len(block)
            opened_after = os.fstat(stream.fileno())
        namespace_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise RuntimeEvidenceError(
            f"cannot authenticate installed package member: {logical_path}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        byte_count != opened_before.st_size
        or not _same_stat(opened_before, opened_after)
        or not _same_stat(opened_after, namespace_after)
    ):
        raise RuntimeEvidenceError(
            f"package member changed while it was hashed: {logical_path}"
        )
    return {
        "path": logical_path,
        "size": byte_count,
        "sha256": digest.hexdigest(),
    }


def _path_file_identity(path: Path) -> dict[str, object]:
    try:
        parent = path.parent.resolve(strict=True)
        name = path.name
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(parent, flags)
        try:
            parent_before = os.fstat(directory_fd)
            record = _regular_file_record(directory_fd, name, name)
            parent_after = os.fstat(directory_fd)
            parent_namespace = os.stat(parent, follow_symlinks=False)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise RuntimeEvidenceError(
            f"cannot authenticate runtime file: {path}"
        ) from error
    if (
        not _same_stat(parent_before, parent_after)
        or not _same_stat(parent_after, parent_namespace)
    ):
        raise RuntimeEvidenceError(
            f"runtime file parent changed while it was authenticated: {path}"
        )
    return {
        "path": str(parent / name),
        "size": record["size"],
        "sha256": record["sha256"],
    }


def _scan_package_tree(
    root_fd: int,
) -> tuple[list[dict[str, object]], tuple[object, ...]]:
    records: list[dict[str, object]] = []
    namespace: list[object] = []
    try:
        walker = os.fwalk(
            ".",
            topdown=True,
            follow_symlinks=False,
            dir_fd=root_fd,
        )
        for directory, directory_names, file_names, directory_fd in walker:
            logical_directory = _relative_directory(directory)
            directory_names.sort()
            file_names.sort()
            if "__pycache__" in directory_names:
                directory_names.remove("__pycache__")
            directory_metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_metadata.st_mode):
                raise RuntimeEvidenceError(
                    f"package member is not a directory: {logical_directory}"
                )
            namespace.append(
                (
                    "directory",
                    logical_directory.as_posix(),
                    *(
                        getattr(directory_metadata, field)
                        for field in _STABLE_STAT_FIELDS
                    ),
                )
            )
            for name in directory_names:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                logical = (logical_directory / name).as_posix()
                if not stat.S_ISDIR(metadata.st_mode):
                    raise RuntimeEvidenceError(
                        "installed pyamplicol package tree contains an "
                        f"unsupported directory entry: {logical}"
                    )
                namespace.append(
                    (
                        "directory-entry",
                        logical,
                        *(getattr(metadata, field) for field in _STABLE_STAT_FIELDS),
                    )
                )
            for name in file_names:
                logical = (logical_directory / name).as_posix()
                if Path(name).suffix in {".pyc", ".pyo"}:
                    raise RuntimeEvidenceError(
                        "installed pyamplicol package tree contains executable "
                        f"sourceless bytecode: {logical}"
                    )
                record = _regular_file_record(directory_fd, name, logical)
                records.append(record)
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                namespace.append(
                    (
                        "file",
                        logical,
                        *(getattr(metadata, field) for field in _STABLE_STAT_FIELDS),
                    )
                )
    except OSError as error:
        raise RuntimeEvidenceError(
            "cannot traverse installed pyamplicol package through checked descriptors"
        ) from error
    records.sort(key=lambda record: str(record["path"]))
    namespace.sort(key=repr)
    return records, tuple(namespace)


def _resolved_package_roots(package_roots: Path | Sequence[Path]) -> tuple[Path, ...]:
    raw_roots = (
        (package_roots,)
        if isinstance(package_roots, Path)
        else tuple(package_roots)
    )
    if not raw_roots:
        raise RuntimeEvidenceError("installed pyamplicol package has no search roots")
    roots: list[Path] = []
    for raw_root in raw_roots:
        try:
            root = raw_root.expanduser().resolve(strict=True)
        except OSError as error:
            raise RuntimeEvidenceError(
                f"cannot resolve installed pyamplicol package tree: {raw_root}"
            ) from error
        if root not in roots:
            roots.append(root)
    return tuple(roots)


def _package_tree_records(root: Path) -> list[dict[str, object]]:
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        root_fd = os.open(root, flags)
        root_before = os.fstat(root_fd)
        if not stat.S_ISDIR(root_before.st_mode):
            raise RuntimeEvidenceError(
                f"installed pyamplicol package tree is not a directory: {root}"
            )
        first_records, first_namespace = _scan_package_tree(root_fd)
        second_records, second_namespace = _scan_package_tree(root_fd)
        root_after = os.fstat(root_fd)
        namespace_root = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise RuntimeEvidenceError(
            f"cannot authenticate installed pyamplicol package tree: {root}"
        ) from error
    finally:
        if "root_fd" in locals():
            os.close(root_fd)
    if (
        not _same_stat(root_before, root_after)
        or not _same_stat(root_after, namespace_root)
        or first_records != second_records
        or first_namespace != second_namespace
    ):
        raise RuntimeEvidenceError(
            "installed pyamplicol package tree changed while it was authenticated"
        )
    if not first_records:
        raise RuntimeEvidenceError(
            "installed pyamplicol package tree contains no source/runtime files"
        )
    return first_records


def python_package_tree_identity(
    package_roots: Path | Sequence[Path],
) -> dict[str, object]:
    """Hash every ordered package search root through anchored directory fds."""

    bytecode_policy = source_only_bytecode_policy()
    roots = _resolved_package_roots(package_roots)
    records: list[dict[str, object]] = []
    for root_index, root in enumerate(roots):
        for record in _package_tree_records(root):
            records.append({"root_index": root_index, **record})
    total_bytes = sum(int(record["size"]) for record in records)
    return {
        "kind": "pyamplicol-python-package-tree-v2",
        "root": str(roots[0]),
        "roots": [str(root) for root in roots],
        "file_count": len(records),
        "total_bytes": total_bytes,
        "sha256": hashlib.sha256(_canonical_json(records)).hexdigest(),
        "member_set_stable": True,
        "namespace_bound_to_root_fd": True,
        "bytecode_policy": bytecode_policy,
    }


def native_extension_in_package(package_root: Path) -> Path:
    """Return the one importable native extension in an ordered package root."""

    try:
        root = package_root.expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeEvidenceError(
            f"cannot resolve native package root: {package_root}"
        ) from error
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    matches = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and path.name.startswith("_rusticol")
            and path.name.endswith(suffixes)
        ),
        key=lambda path: path.name,
    )
    if len(matches) != 1:
        raise RuntimeEvidenceError(
            "exact runtime evidence requires one native _rusticol extension in "
            f"{root}; found {len(matches)}"
        )
    return matches[0]


def preimport_python_runtime_identity(
    package_roots: Path | Sequence[Path],
    *,
    native_extension: Path,
) -> dict[str, object]:
    """Authenticate candidate Python/native files before importing pyamplicol."""

    global _PREIMPORT_RUNTIME_IDENTITY
    if any(
        name == "pyamplicol" or name.startswith("pyamplicol.")
        for name in sys.modules
    ):
        raise RuntimeEvidenceError(
            "pyamplicol was imported before exact runtime preauthentication"
        )
    identity = {
        "kind": "pyamplicol-preimport-runtime-identity-v1",
        "python_package_tree": python_package_tree_identity(package_roots),
        "native_extension": _path_file_identity(native_extension),
    }
    _PREIMPORT_RUNTIME_IDENTITY = identity
    return identity


def established_preimport_runtime_identity() -> dict[str, object]:
    """Return the identity established before the first pyamplicol import."""

    if _PREIMPORT_RUNTIME_IDENTITY is None:
        raise RuntimeEvidenceError(
            "exact runtime evidence has no preimport Python/native identity"
        )
    return _PREIMPORT_RUNTIME_IDENTITY


def loaded_pyamplicol_origin_policy(
    package_roots: Path | Sequence[Path],
    *,
    native_extension: Path,
    expected_package_identity: dict[str, object] | None = None,
    expected_native_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Require every loaded pyamplicol module to originate in authenticated bytes."""

    source_only_bytecode_policy()
    try:
        roots = _resolved_package_roots(package_roots)
        native = native_extension.expanduser().resolve(strict=True)
    except OSError as error:
        raise RuntimeEvidenceError(
            "cannot resolve loaded pyamplicol module origins"
        ) from error
    current_package_identity = python_package_tree_identity(roots)
    if (
        expected_package_identity is not None
        and current_package_identity != expected_package_identity
    ):
        raise RuntimeEvidenceError(
            "pyamplicol package roots differ from their preimport identity"
        )
    current_native_identity = _path_file_identity(native)
    if (
        expected_native_identity is not None
        and current_native_identity != expected_native_identity
    ):
        raise RuntimeEvidenceError(
            "native extension differs from its preimport identity"
        )
    member_records: list[dict[str, object]] = []
    for root_index, root in enumerate(roots):
        member_records.extend(
            {"root_index": root_index, **record}
            for record in _package_tree_records(root)
        )
    members = {
        (int(record["root_index"]), str(record["path"])): record
        for record in member_records
    }
    prefix = Path(str(sys.pycache_prefix))
    observed = 0
    observations: list[dict[str, object]] = []
    for name, module in tuple(sys.modules.items()):
        if name != "pyamplicol" and not name.startswith("pyamplicol."):
            continue
        specification = getattr(module, "__spec__", None)
        origin_value = getattr(specification, "origin", None)
        if not isinstance(origin_value, str) or origin_value in {
            "built-in",
            "frozen",
        }:
            origin_value = getattr(module, "__file__", None)
        if not isinstance(origin_value, str) or not origin_value:
            locations = getattr(specification, "submodule_search_locations", None)
            if locations is None:
                raise RuntimeEvidenceError(
                    f"loaded pyamplicol module has no filesystem origin: {name}"
                )
            for raw_location in locations:
                try:
                    location = Path(str(raw_location)).expanduser().resolve(strict=True)
                except OSError as error:
                    raise RuntimeEvidenceError(
                        "loaded pyamplicol namespace is outside the authenticated "
                        f"package tree: {name}"
                    ) from error
                if not any(
                    location == root or location.is_relative_to(root)
                    for root in roots
                ):
                    raise RuntimeEvidenceError(
                        "loaded pyamplicol namespace is outside the authenticated "
                        f"package tree: {name}"
                    )
                root_index = next(
                    index
                    for index, root in enumerate(roots)
                    if location == root or location.is_relative_to(root)
                )
                relative = location.relative_to(roots[root_index]).as_posix()
                if "__pycache__" in PurePosixPath(relative).parts:
                    raise RuntimeEvidenceError(
                        f"loaded pyamplicol namespace uses excluded cache bytes: {name}"
                    )
                observations.append(
                    {
                        "module": name,
                        "kind": "namespace",
                        "root_index": root_index,
                        "path": relative,
                    }
                )
            observed += 1
            continue
        try:
            origin = Path(origin_value).expanduser().resolve(strict=True)
        except OSError as error:
            raise RuntimeEvidenceError(
                f"loaded pyamplicol module origin is unavailable: {name}"
            ) from error
        in_package = any(
            origin == root or origin.is_relative_to(root) for root in roots
        )
        if not in_package and origin != native:
            raise RuntimeEvidenceError(
                "loaded pyamplicol module is outside the authenticated package "
                f"tree/native image: {name} -> {origin}"
            )
        if origin.suffix in {".pyc", ".pyo"}:
            raise RuntimeEvidenceError(
                f"loaded pyamplicol module used sourceless bytecode: {name}"
            )
        if origin == native:
            observations.append(
                {
                    "module": name,
                    "kind": "native-extension",
                    "path": current_native_identity["path"],
                    "size": current_native_identity["size"],
                    "sha256": current_native_identity["sha256"],
                }
            )
        else:
            root_index = next(
                index
                for index, root in enumerate(roots)
                if origin == root or origin.is_relative_to(root)
            )
            relative = origin.relative_to(roots[root_index]).as_posix()
            if "__pycache__" in PurePosixPath(relative).parts:
                raise RuntimeEvidenceError(
                    f"loaded pyamplicol module uses excluded cache bytes: {name}"
                )
            member = members.get((root_index, relative))
            if member is None:
                raise RuntimeEvidenceError(
                    "loaded pyamplicol module is absent from the authenticated "
                    f"member set: {name}"
                )
            actual = _path_file_identity(origin)
            if (
                actual["size"] != member["size"]
                or actual["sha256"] != member["sha256"]
            ):
                raise RuntimeEvidenceError(
                    f"loaded pyamplicol module bytes changed: {name}"
                )
            observations.append(
                {
                    "module": name,
                    "kind": "package-member",
                    "root_index": root_index,
                    "path": relative,
                    "size": member["size"],
                    "sha256": member["sha256"],
                }
            )
        cached_value = getattr(module, "__cached__", None)
        if isinstance(cached_value, str) and cached_value:
            cached = Path(cached_value).expanduser()
            try:
                cached.relative_to(prefix)
                under_prefix = True
            except ValueError:
                under_prefix = False
            if not cached.is_absolute() or not under_prefix or cached.exists():
                raise RuntimeEvidenceError(
                    "loaded pyamplicol module exposes an eligible bytecode cache: "
                    f"{name}"
                )
        observed += 1
    if observed == 0:
        raise RuntimeEvidenceError("no loaded pyamplicol modules were observed")
    observations.sort(
        key=lambda observation: (
            str(observation["module"]),
            str(observation["kind"]),
            str(observation["path"]),
        )
    )
    return {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": observed,
        "observations": observations,
        "observations_sha256": hashlib.sha256(
            _canonical_json(observations)
        ).hexdigest(),
    }


__all__ = [
    "RuntimeEvidenceError",
    "established_preimport_runtime_identity",
    "loaded_pyamplicol_origin_policy",
    "native_extension_in_package",
    "preimport_python_runtime_identity",
    "python_package_tree_identity",
    "source_only_bytecode_policy",
]
