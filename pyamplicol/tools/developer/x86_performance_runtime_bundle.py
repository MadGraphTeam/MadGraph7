#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build-time authentication helpers for the x86 performance runtime bundle.

The frozen pre-Arena candidate predates reproducible dependency-state
timestamps.  ``freeze-baseline`` restores the already accepted generated state
only after every path-independent dependency input matches its pinned digest.
The other commands content-bind or verify the two installed candidates, their
shared external dependency distributions, wheel bytes, and the UFO prepared
model used by the recurrence comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import venv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import compiled_mode_matrix as matrix  # noqa: E402
from tools.developer import compiled_mode_regression as regression  # noqa: E402

BUNDLE_KIND = "pyamplicol-x86-performance-runtime-bundle"
BASELINE_ATTESTATION_KIND = "pyamplicol-frozen-baseline-source-attestation"
SCHEMA_VERSION = 1
CONTENT_IDENTITY_ALGORITHM = "sha256-canonical-json-body-v1"
FROZEN_SOURCE_ROOT = Path("/tmp/pyamplicol-eager-compiled-arena-base-src")
FROZEN_CANDIDATE_LOCK_SHA256 = (
    "3ec1fd00ed1369adcb736a6816495bb21bcc11a377d0922a59cc0b845a0df3ef"
)
FROZEN_CARGO_CONFIG_SHA256 = (
    "f14f5d6595a26680e636c5f25f2ea9d88772a6e44b6777c363cb8b810e2ca661"
)
FROZEN_INSTALL_STATE_SHA256 = (
    "d4835339f2e35ac5e3ba4ec10941f6cffa4465fec505e3560e25e7eba534bb09"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_JSON_LIMIT_BYTES = 64 * 1024 * 1024
_BASELINE_GENERATED_FILES = (
    "dependencies/candidate-Cargo.lock",
    "dependencies/candidate-cargo-config.toml",
    "dependencies/install-state.json",
)
_BASELINE_BOOTSTRAP_ROOTS = (
    ".venv/",
    "dependencies/checkouts/",
    "dependencies/wheelhouse/",
)

FROZEN_INSTALL_STATE: dict[str, object] = {
    "candidate_lock_sha256": FROZEN_CANDIDATE_LOCK_SHA256,
    "cargo_config_sha256": FROZEN_CARGO_CONFIG_SHA256,
    "contributor_lock_sha256": (
        "91a4cd4d03bc3d35b7e2794f04bed4580428f5d17ea2f846fa530c5df8197cb5"
    ),
    "created_utc": "2026-07-24T16:52:37.275744+00:00",
    "patches": [
        {
            "applies_to_revision": "48197f32536c894b51ef25b2cf05ddd05c22675f",
            "name": "symjit-aarch64-compression-and-direct-arena",
            "path": "patches/symjit/0001-aarch64-compression-and-direct-arena.patch",
            "sha256": (
                "6d456e69fc160ec5361188f60f994d10fb2dd3360eb47a91c4979a1bde69626e"
            ),
            "target": "symjit",
        }
    ],
    "publishable": False,
    "python_runtime_lock_sha256": (
        "f8c929cfe925630a96e1a9d80bb8d6106964ace84a4efc4df171784a4f8f522b"
    ),
    "release_lock_sha256": (
        "3302cacd840eb9f14e9e00e4c4c712d1a4735a45c1702224b84e4f34385f50ee"
    ),
    "schema_version": 1,
    "sources": {
        "gammaloop": {
            "revision": "fff1d66ee7ca039b9e165fe8d29da91f4c27113c",
            "url": "https://github.com/alphal00p/gammaloop.git",
            "worktree_sha256": (
                "2576a8cd3bd5cd58c649af7c6b480d7a8c3ed86afd0c7ac8b20be605d7e9f005"
            ),
        },
        "legacy-amplicol": {
            "branch": "amplicol_with_patches",
            "revision": "79c96cecf2a722e50c3d2030b6894d755f96518a",
            "url": "git@github.com:rikkert-frederix/AmpliCol.git",
            "worktree_sha256": (
                "2b6678c6fe477fcc0f99b5ed636e34b6c783ba778c328268e57d34de44bd7d65"
            ),
        },
        "symbolica": {
            "revision": "77c137481904b8a5531ede86e3ef36b82beed7fd",
            "url": "https://github.com/symbolica-dev/symbolica.git",
            "worktree_sha256": (
                "3a52b012e62783988a9567260d022b049d9d07183a8604e7cbfc99218344cd59"
            ),
        },
        "symbolica-community": {
            "revision": "0deaa0f1484ec5c24f5f109176fa66f0d30134ca",
            "url": "https://github.com/symbolica-dev/symbolica-community.git",
            "worktree_sha256": (
                "bbe726e158192c2bac478cf482f68576d8bd49d06f28282e0e4cc5fa43c1c479"
            ),
        },
        "symjit": {
            "archive_sha256": (
                "876930348cc06761ca780570fb282d009f143f4a469e321e3b5039c5ee217424"
            ),
            "patch_sha256": (
                "6d456e69fc160ec5361188f60f994d10fb2dd3360eb47a91c4979a1bde69626e"
            ),
            "revision": "48197f32536c894b51ef25b2cf05ddd05c22675f",
            "url": (
                "https://github.com/siravan/symjit/archive/"
                "48197f32536c894b51ef25b2cf05ddd05c22675f.tar.gz"
            ),
            "version": "2.21.1",
            "worktree_sha256": (
                "932bb24df2633cc8bbf9c743a80282662d11e70b692885de5ff7a3ed20b3df31"
            ),
        },
    },
}


class BundleError(RuntimeError):
    """Raised when the runtime bundle cannot be authenticated."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _attach_content_identity(body: Mapping[str, object]) -> dict[str, object]:
    result = dict(body)
    result["content_identity"] = {
        "algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sha256": _canonical_sha256(body),
    }
    return result


def _require_content_identity(payload: Mapping[str, object]) -> None:
    identity = payload.get("content_identity")
    body = dict(payload)
    body.pop("content_identity", None)
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"algorithm", "sha256"}
        or identity.get("algorithm") != CONTENT_IDENTITY_ALGORITHM
        or identity.get("sha256") != _canonical_sha256(body)
    ):
        raise BundleError("runtime bundle content identity is invalid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    expanded = path.expanduser().absolute()
    try:
        before = expanded.lstat()
    except OSError as error:
        raise BundleError(f"cannot inspect bundle input: {path}") from error
    if not stat.S_ISREG(before.st_mode) or expanded.is_symlink():
        raise BundleError(f"bundle input is not a regular non-symlink file: {path}")
    resolved = expanded.resolve(strict=True)
    digest = _sha256_file(resolved)
    try:
        after = expanded.lstat()
    except OSError as error:
        raise BundleError(f"cannot recheck bundle input: {path}") from error
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise BundleError(f"bundle input changed while it was hashed: {path}")
    return {
        "relative_path": None,
        "size_bytes": before.st_size,
        "sha256": digest,
    }


def _reject_constant(value: str) -> None:
    raise BundleError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _checked_json(path: Path, *, label: str) -> dict[str, Any]:
    expanded = path.expanduser().absolute()
    try:
        before = expanded.lstat()
    except OSError as error:
        raise BundleError(f"cannot inspect {label}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or expanded.is_symlink()
        or before.st_size <= 0
        or before.st_size > _JSON_LIMIT_BYTES
    ):
        raise BundleError(f"{label} is not a bounded regular non-symlink file")
    try:
        encoded = expanded.read_bytes()
        after = expanded.lstat()
    except OSError as error:
        raise BundleError(f"cannot read {label}") from error
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise BundleError(f"{label} changed while it was read")
    try:
        payload = json.loads(
            encoded,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"{label} is not strict JSON") from error
    if not isinstance(payload, dict):
        raise BundleError(f"{label} is not a JSON object")
    return payload


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BundleError(f"git inspection failed: {completed.stderr.strip()}")
    return completed.stdout


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frozen_install_state_bytes() -> bytes:
    return (
        json.dumps(
            FROZEN_INSTALL_STATE,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _baseline_ignored_inventory(root: Path) -> dict[str, object]:
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise BundleError("frozen baseline has tracked or unexpected source changes")
    raw_ignored = _git_output(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    ignored = sorted(path for path in raw_ignored.split("\0") if path)
    unexpected = [
        path
        for path in ignored
        if path not in _BASELINE_GENERATED_FILES
        and not any(path.startswith(prefix) for prefix in _BASELINE_BOOTSTRAP_ROOTS)
    ]
    if unexpected:
        raise BundleError(
            "frozen baseline has unexpected ignored source/bootstrap files: "
            + ", ".join(unexpected[:20])
        )
    missing_generated = [
        relative
        for relative in _BASELINE_GENERATED_FILES
        if not (root / relative).is_file()
    ]
    missing_roots = [
        relative
        for relative in _BASELINE_BOOTSTRAP_ROOTS
        if not (root / relative).is_dir()
    ]
    if missing_generated or missing_roots:
        raise BundleError(
            "frozen baseline bootstrap is incomplete: "
            f"files={missing_generated}, roots={missing_roots}"
        )
    return {
        "tracked_and_untracked_status_clean": True,
        "ignored_file_count": len(ignored),
        "allowed_generated_files": list(_BASELINE_GENERATED_FILES),
        "allowed_bootstrap_roots": list(_BASELINE_BOOTSTRAP_ROOTS),
        "unexpected_ignored_files": [],
    }


def _baseline_generated_identities(root: Path) -> dict[str, object]:
    generated_files: dict[str, object] = {}
    for relative in _BASELINE_GENERATED_FILES:
        identity = _file_identity(root / relative)
        identity["relative_path"] = relative
        generated_files[relative] = identity
    bootstrap_trees: dict[str, object] = {}
    for relative in _BASELINE_BOOTSTRAP_ROOTS:
        logical_path = relative.rstrip("/")
        identity = regression._tree_identity(root / relative)
        identity["relative_path"] = logical_path
        bootstrap_trees[logical_path] = identity
    return {
        "generated_files": generated_files,
        "bootstrap_trees": bootstrap_trees,
    }


def freeze_baseline(source_root: Path) -> dict[str, object]:
    root = source_root.expanduser().resolve(strict=True)
    if root != FROZEN_SOURCE_ROOT:
        raise BundleError(
            f"frozen baseline must use the path {FROZEN_SOURCE_ROOT}, got {root}"
        )
    if _git_output(root, "rev-parse", "--verify", "HEAD").strip() != (
        matrix.FROZEN_BASELINE_SOURCE_REVISION
    ):
        raise BundleError("baseline checkout is not the frozen source revision")
    ignored_inventory = _baseline_ignored_inventory(root)
    dependencies = root / "dependencies"
    lock = dependencies / "candidate-Cargo.lock"
    config = dependencies / "candidate-cargo-config.toml"
    state = dependencies / "install-state.json"
    identities = {
        "candidate_lock": _sha256_file(lock),
        "cargo_config": _sha256_file(config),
    }
    if identities != {
        "candidate_lock": FROZEN_CANDIDATE_LOCK_SHA256,
        "cargo_config": FROZEN_CARGO_CONFIG_SHA256,
    }:
        raise BundleError(f"generated baseline dependency inputs differ: {identities}")
    if state.is_file():
        try:
            generated = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BundleError("generated baseline install state is invalid") from error
        if not isinstance(generated, dict):
            raise BundleError("generated baseline install state is not an object")
        for key in (
            "candidate_lock_sha256",
            "cargo_config_sha256",
            "contributor_lock_sha256",
            "patches",
            "publishable",
            "python_runtime_lock_sha256",
            "release_lock_sha256",
            "schema_version",
        ):
            if generated.get(key) != FROZEN_INSTALL_STATE[key]:
                raise BundleError(
                    f"generated baseline state disagrees with frozen field {key}"
                )
        generated_sources = generated.get("sources")
        frozen_sources = FROZEN_INSTALL_STATE["sources"]
        if not isinstance(generated_sources, Mapping) or not isinstance(
            frozen_sources, Mapping
        ):
            raise BundleError("generated baseline source inventory is invalid")
        for name, frozen in frozen_sources.items():
            if name == "legacy-amplicol" and name not in generated_sources:
                continue
            if generated_sources.get(name) != frozen:
                raise BundleError(
                    f"generated baseline source {name} differs from the freeze"
                )
    encoded = frozen_install_state_bytes()
    if hashlib.sha256(encoded).hexdigest() != FROZEN_INSTALL_STATE_SHA256:
        raise AssertionError("embedded frozen install state bytes changed")
    temporary = state.with_name(f".{state.name}.tmp-{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, state)
    result = _attach_content_identity(
        {
            "kind": BASELINE_ATTESTATION_KIND,
            "schema_version": SCHEMA_VERSION,
            "source_root": str(root),
            "source_revision": matrix.FROZEN_BASELINE_SOURCE_REVISION,
            "native_build_inputs_sha256": (
                matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
            ),
            "candidate_lock_sha256": identities["candidate_lock"],
            "cargo_config_sha256": identities["cargo_config"],
            "install_state_sha256": _sha256_file(state),
            "source_inventory": ignored_inventory,
            "generated_identity": _baseline_generated_identities(root),
            "passes": _sha256_file(state) == FROZEN_INSTALL_STATE_SHA256,
        }
    )
    if (
        result["passes"] is not True
        or not _valid_baseline_attestation_payload(result)
    ):
        raise BundleError("frozen install state could not be materialized exactly")
    return result


def _stable_installation(identity: Mapping[str, Any]) -> dict[str, object]:
    build_files = identity.get("build_info_files")
    native_modules = identity.get("native_modules")
    distribution = identity.get("distribution_content")
    if (
        not isinstance(build_files, list)
        or len(build_files) != 1
        or not isinstance(build_files[0], Mapping)
        or not isinstance(build_files[0].get("payload"), Mapping)
        or not isinstance(native_modules, list)
        or len(native_modules) != 1
        or not isinstance(native_modules[0], Mapping)
        or not isinstance(distribution, Mapping)
    ):
        raise BundleError("installed pyamplicol identity is incomplete")
    payload = dict(build_files[0]["payload"])
    return {
        "package_version": identity.get("package_version"),
        "build_info": payload,
        "build_info_sha256": build_files[0].get("sha256"),
        "distribution_content": {
            key: distribution.get(key)
            for key in ("algorithm", "sha256", "file_count", "size_bytes")
        },
        "native_module": {
            key: native_modules[0].get(key)
            for key in ("relative_path", "sha256", "size_bytes")
        },
    }


def _installation(python: Path) -> dict[str, object]:
    identity = regression._installed_pyamplicol_identity(
        python.expanduser().resolve(strict=True),
        environment=regression._environment(),
    )
    return _stable_installation(identity)


def _stable_dependency_site(path: Path) -> dict[str, object]:
    identity = regression._dependency_site_identity(path)
    distributions = identity["distributions"]
    return {
        "algorithm": identity["algorithm"],
        "sha256": identity["sha256"],
        "distributions": {
            name: {
                key: record[key]
                for key in (
                    "name",
                    "version",
                    "algorithm",
                    "sha256",
                    "file_count",
                    "size_bytes",
                )
            }
            for name, record in sorted(distributions.items())
        },
    }


def _wheel_inventory(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    wheels_root = root / "wheels"
    if (
        not wheels_root.is_dir()
        or wheels_root.is_symlink()
        or {path.name for path in wheels_root.iterdir()} != {"baseline", "current"}
    ):
        raise BundleError("runtime bundle wheel lane inventory is not exact")
    for lane in ("baseline", "current"):
        lane_root = wheels_root / lane
        entries = sorted(lane_root.iterdir(), key=lambda path: path.name)
        candidates = [
            path
            for path in entries
            if path.suffix == ".whl" and path.is_file() and not path.is_symlink()
        ]
        if len(candidates) != 1 or entries != candidates:
            raise BundleError(
                f"runtime bundle requires exactly one file, the {lane} wheel"
            )
        identity = _file_identity(candidates[0])
        identity["relative_path"] = candidates[0].relative_to(root).as_posix()
        result[lane] = identity
    return result


def _prepared_inventory(root: Path) -> dict[str, object]:
    path = root / "prepared-models" / "ufo-sm-jit-o2.pyamplicol-model"
    identity = _file_identity(path)
    identity["relative_path"] = path.relative_to(root).as_posix()
    return identity


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_integer(value: object) -> bool:
    return _nonnegative_integer(value) and value > 0


def _valid_generated_file_identity(
    value: object,
    *,
    relative_path: str,
    expected_sha256: str,
) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"relative_path", "size_bytes", "sha256"}
        and value.get("relative_path") == relative_path
        and _positive_integer(value.get("size_bytes"))
        and value.get("sha256") == expected_sha256
    )


def _valid_bootstrap_tree_identity(
    value: object,
    *,
    relative_path: str,
) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            "algorithm",
            "relative_path",
            "sha256",
            "file_count",
            "size_bytes",
        }
        and value.get("algorithm") == "sha256-relative-path-size-content-v1"
        and value.get("relative_path") == relative_path
        and isinstance(value.get("sha256"), str)
        and _SHA256.fullmatch(str(value.get("sha256"))) is not None
        and _positive_integer(value.get("file_count"))
        and _positive_integer(value.get("size_bytes"))
    )


def _valid_baseline_attestation_payload(payload: Mapping[str, object]) -> bool:
    source_inventory = payload.get("source_inventory")
    generated_identity = payload.get("generated_identity")
    if (
        not isinstance(source_inventory, Mapping)
        or set(source_inventory)
        != {
            "tracked_and_untracked_status_clean",
            "ignored_file_count",
            "allowed_generated_files",
            "allowed_bootstrap_roots",
            "unexpected_ignored_files",
        }
        or source_inventory.get("tracked_and_untracked_status_clean") is not True
        or not _positive_integer(source_inventory.get("ignored_file_count"))
        or source_inventory.get("allowed_generated_files")
        != list(_BASELINE_GENERATED_FILES)
        or source_inventory.get("allowed_bootstrap_roots")
        != list(_BASELINE_BOOTSTRAP_ROOTS)
        or source_inventory.get("unexpected_ignored_files") != []
        or not isinstance(generated_identity, Mapping)
        or set(generated_identity) != {"generated_files", "bootstrap_trees"}
    ):
        return False
    generated_files = generated_identity.get("generated_files")
    bootstrap_trees = generated_identity.get("bootstrap_trees")
    expected_file_sha256 = {
        "dependencies/candidate-Cargo.lock": FROZEN_CANDIDATE_LOCK_SHA256,
        "dependencies/candidate-cargo-config.toml": FROZEN_CARGO_CONFIG_SHA256,
        "dependencies/install-state.json": FROZEN_INSTALL_STATE_SHA256,
    }
    expected_bootstrap_paths = {
        relative.rstrip("/") for relative in _BASELINE_BOOTSTRAP_ROOTS
    }
    if (
        not isinstance(generated_files, Mapping)
        or set(generated_files) != set(_BASELINE_GENERATED_FILES)
        or not all(
            _valid_generated_file_identity(
                generated_files.get(relative),
                relative_path=relative,
                expected_sha256=expected_file_sha256[relative],
            )
            for relative in _BASELINE_GENERATED_FILES
        )
        or not isinstance(bootstrap_trees, Mapping)
        or set(bootstrap_trees) != expected_bootstrap_paths
        or not all(
            _valid_bootstrap_tree_identity(
                bootstrap_trees.get(relative),
                relative_path=relative,
            )
            for relative in expected_bootstrap_paths
        )
    ):
        return False
    ignored_file_count = len(_BASELINE_GENERATED_FILES) + sum(
        int(bootstrap_trees[relative]["file_count"])
        for relative in expected_bootstrap_paths
    )
    return source_inventory.get("ignored_file_count") == ignored_file_count


def _baseline_attestation_inventory(root: Path) -> dict[str, object]:
    path = root / "frozen-baseline-attestation.json"
    payload = _checked_json(path, label="frozen baseline source attestation")
    _require_content_identity(payload)
    if (
        set(payload)
        != {
            "kind",
            "schema_version",
            "source_root",
            "source_revision",
            "native_build_inputs_sha256",
            "candidate_lock_sha256",
            "cargo_config_sha256",
            "install_state_sha256",
            "source_inventory",
            "generated_identity",
            "passes",
            "content_identity",
        }
        or payload.get("kind") != BASELINE_ATTESTATION_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("source_root") != str(FROZEN_SOURCE_ROOT)
        or payload.get("source_revision") != matrix.FROZEN_BASELINE_SOURCE_REVISION
        or payload.get("native_build_inputs_sha256")
        != matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
        or payload.get("candidate_lock_sha256")
        != FROZEN_CANDIDATE_LOCK_SHA256
        or payload.get("cargo_config_sha256") != FROZEN_CARGO_CONFIG_SHA256
        or payload.get("install_state_sha256") != FROZEN_INSTALL_STATE_SHA256
        or not _valid_baseline_attestation_payload(payload)
        or payload.get("passes") is not True
    ):
        raise BundleError("frozen baseline source attestation is invalid")
    identity = _file_identity(path)
    identity["relative_path"] = path.relative_to(root).as_posix()
    return {
        "file": identity,
        "content_sha256": payload["content_identity"]["sha256"],
    }


def create_manifest(arguments: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise BundleError("x86 runtime bundle must be produced on Linux x86-64")
    bundle_root = arguments.bundle_root.expanduser().resolve(strict=True)
    baseline = _installation(arguments.baseline_python)
    current = _installation(arguments.current_python)
    for lane, identity, expected_source, expected_inputs in (
        (
            "baseline",
            baseline,
            matrix.FROZEN_BASELINE_SOURCE_REVISION,
            matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256,
        ),
        (
            "current",
            current,
            arguments.expected_current_revision,
            arguments.expected_current_native_inputs_sha256,
        ),
    ):
        build = identity["build_info"]
        if (
            not isinstance(build, Mapping)
            or build.get("source_revision") != expected_source
            or build.get("native_build_inputs_sha256") != expected_inputs
            or build.get("publishable") is not False
        ):
            raise BundleError(f"{lane} installed build identity is not exact")
    payload = _attach_content_identity(
        {
            "kind": BUNDLE_KIND,
            "schema_version": SCHEMA_VERSION,
            "target": "x86_64-unknown-linux-gnu",
            "workflow_run_id": arguments.workflow_run_id,
            "expected_current_revision": arguments.expected_current_revision,
            "installations": {"baseline": baseline, "current": current},
            "dependency_site": _stable_dependency_site(arguments.dependency_site),
            "wheels": _wheel_inventory(bundle_root),
            "prepared_models": {"ufo-sm": _prepared_inventory(bundle_root)},
            "frozen_baseline_source_attestation": (
                _baseline_attestation_inventory(bundle_root)
            ),
            "frozen_baseline": {
                "source_root": str(FROZEN_SOURCE_ROOT),
                "source_revision": matrix.FROZEN_BASELINE_SOURCE_REVISION,
                "native_build_inputs_sha256": (
                    matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
                ),
                "candidate_lock_sha256": FROZEN_CANDIDATE_LOCK_SHA256,
                "cargo_config_sha256": FROZEN_CARGO_CONFIG_SHA256,
                "install_state_sha256": FROZEN_INSTALL_STATE_SHA256,
            },
            "passes": True,
        }
    )
    _write_json_atomic(arguments.output, payload)
    return payload


def _checked_manifest(path: Path) -> dict[str, Any]:
    payload = _checked_json(path, label="runtime bundle manifest")
    _require_content_identity(payload)
    if (
        payload.get("kind") != BUNDLE_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("target") != "x86_64-unknown-linux-gnu"
        or payload.get("passes") is not True
    ):
        raise BundleError("runtime bundle manifest contract is invalid")
    return payload


def verify_manifest(arguments: argparse.Namespace) -> dict[str, object]:
    payload = _checked_manifest(arguments.manifest)
    if (
        payload.get("workflow_run_id") != arguments.workflow_run_id
        or payload.get("expected_current_revision")
        != arguments.expected_current_revision
    ):
        raise BundleError("runtime bundle workflow/source binding changed")
    observed = {
        "installations": {
            "baseline": _installation(arguments.baseline_python),
            "current": _installation(arguments.current_python),
        },
        "dependency_site": _stable_dependency_site(arguments.dependency_site),
    }
    expected = {
        "installations": payload.get("installations"),
        "dependency_site": payload.get("dependency_site"),
    }
    if observed != expected:
        raise BundleError("installed runtime/dependency identity differs from bundle")
    bundle_root = arguments.bundle_root.expanduser().resolve(strict=True)
    if _wheel_inventory(bundle_root) != payload.get("wheels") or {
        "ufo-sm": _prepared_inventory(bundle_root)
    } != payload.get("prepared_models"):
        raise BundleError("runtime bundle files differ from their manifest")
    if _baseline_attestation_inventory(bundle_root) != payload.get(
        "frozen_baseline_source_attestation"
    ):
        raise BundleError("frozen baseline attestation differs from its manifest")
    return {
        "kind": "pyamplicol-x86-performance-runtime-bundle-verification",
        "schema_version": 1,
        "manifest_content_sha256": payload["content_identity"]["sha256"],
        "passes": True,
    }


def bundle_dependencies(source_site: Path, destination: Path) -> dict[str, object]:
    source = source_site.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise BundleError("dependency bundle destination is unsafe")
        if any(destination.iterdir()):
            raise BundleError("dependency bundle destination must be empty")
    else:
        destination.mkdir(parents=True)
    available = {
        re.sub(r"[-_.]+", "-", str(dist.metadata.get("Name", ""))).casefold(): dist
        for dist in importlib.metadata.distributions(path=[str(source)])
    }
    for distribution_name, _import_name in regression.DEPENDENCY_DISTRIBUTIONS:
        key = re.sub(r"[-_.]+", "-", distribution_name).casefold()
        distribution = available.get(key)
        if distribution is None:
            raise BundleError(f"dependency site lacks {distribution_name}")
        for entry in distribution.files or ():
            relative = Path(str(entry))
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "__pycache__" in relative.parts
                or relative.suffix == ".pyc"
            ):
                continue
            source_file = (source / relative).resolve(strict=True)
            source_file.relative_to(source)
            if not source_file.is_file():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
    source_identity = _stable_dependency_site(source)
    destination_identity = _stable_dependency_site(destination)
    if source_identity != destination_identity:
        raise BundleError("copied dependency bundle identity changed")
    return destination_identity


def _empty_external_directory(path: Path, *, label: str) -> Path:
    destination = path.expanduser().absolute()
    checkout = ROOT.resolve(strict=True)
    if destination == Path(destination.anchor):
        raise BundleError(f"{label} cannot be a filesystem root")
    try:
        destination.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise BundleError(f"{label} must be outside the source checkout")
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise BundleError(f"{label} is unsafe")
        if any(destination.iterdir()):
            raise BundleError(f"{label} must be empty")
    else:
        destination.mkdir(parents=True)
    return destination


def _venv_python(root: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return (root / relative).resolve(strict=True)


def _venv_purelib(python: Path) -> Path:
    completed = subprocess.run(
        (
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BundleError("cannot resolve materialized runtime site-packages")
    try:
        result = Path(completed.stdout.strip()).resolve(strict=True)
    except OSError as error:
        raise BundleError(
            "materialized runtime site-packages is unavailable"
        ) from error
    if not result.is_dir():
        raise BundleError("materialized runtime site-packages is not a directory")
    return result


def materialize_runtime(
    bundle_root: Path,
    runtime_root: Path,
) -> dict[str, object]:
    """Rebuild the two fixed-path venvs from portable bundle members."""

    bundle = bundle_root.expanduser().resolve(strict=True)
    destination = _empty_external_directory(
        runtime_root,
        label="materialized runtime root",
    )
    dependencies = (bundle / "dependency-site").resolve(strict=True)
    if not dependencies.is_dir():
        raise BundleError("runtime bundle has no dependency site")
    dependency_identity = _stable_dependency_site(dependencies)
    wheels = _wheel_inventory(bundle)
    lanes: dict[str, dict[str, object]] = {}
    for lane in ("baseline", "current"):
        lane_root = destination / lane
        # A copied interpreter keeps ``resolve(strict=True)`` inside the lane;
        # symlinked venv interpreters would collapse to the setup-python host
        # executable and lose the selected site-packages identity.
        venv.EnvBuilder(with_pip=True, symlinks=False).create(lane_root)
        python = _venv_python(lane_root)
        purelib = _venv_purelib(python)
        for entry in sorted(dependencies.iterdir(), key=lambda path: path.name):
            target = purelib / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target)
            elif entry.is_file():
                shutil.copy2(entry, target)
            else:
                raise BundleError(f"dependency bundle member is unsafe: {entry}")
        if _stable_dependency_site(purelib) != dependency_identity:
            raise BundleError(f"{lane} materialized dependency identity changed")
        wheel = bundle / str(wheels[lane]["relative_path"])
        environment = os.environ.copy()
        environment.update(
            {
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        completed = subprocess.run(
            (
                str(python),
                "-m",
                "pip",
                "install",
                "--no-compile",
                "--no-deps",
                str(wheel),
            ),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise BundleError(
                f"cannot install {lane} runtime wheel: {completed.stderr.strip()}"
            )
        lanes[lane] = {
            "python": str(python),
            "site_packages": str(purelib),
            "installation": _installation(python),
        }
    return {
        "runtime_root": str(destination),
        "dependency_site": str(dependencies),
        "dependency_site_identity": dependency_identity,
        "lanes": lanes,
        "passes": True,
    }


def prepare_ufo(output: Path, artifact_root: Path) -> dict[str, object]:
    from tools.performance_report.artifacts import ArtifactStore
    from tools.performance_report.prepared import ensure_report_ufo_sm_prepared_model

    store = ArtifactStore(
        artifact_root=artifact_root.expanduser().absolute(),
        lock_root=artifact_root.expanduser().absolute() / "locks",
    )
    prepared, _reused = ensure_report_ufo_sm_prepared_model(
        store=store,
        repo_root=ROOT,
        worker_cores=1,
    )
    destination = output.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prepared, destination)
    return _file_identity(destination)


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256")
    return value


def _git_sha_argument(value: str) -> str:
    if _GIT_SHA.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase Git SHA")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze-baseline")
    frozen.add_argument("--source-root", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)

    dependencies = commands.add_parser("bundle-dependencies")
    dependencies.add_argument("--source-site", type=Path, required=True)
    dependencies.add_argument("--destination", type=Path, required=True)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--bundle-root", type=Path, required=True)
    materialize.add_argument("--runtime-root", type=Path, required=True)

    prepared = commands.add_parser("prepare-ufo")
    prepared.add_argument("--output", type=Path, required=True)
    prepared.add_argument("--artifact-root", type=Path, required=True)

    for name in ("create-manifest", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--bundle-root", type=Path, required=True)
        command.add_argument("--baseline-python", type=Path, required=True)
        command.add_argument("--current-python", type=Path, required=True)
        command.add_argument("--dependency-site", type=Path, required=True)
        command.add_argument("--workflow-run-id", required=True)
        command.add_argument(
            "--expected-current-revision",
            type=_git_sha_argument,
            required=True,
        )
        if name == "create-manifest":
            command.add_argument(
                "--expected-current-native-inputs-sha256",
                type=_sha256_argument,
                required=True,
            )
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--manifest", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "freeze-baseline":
            payload = freeze_baseline(arguments.source_root)
            _write_json_atomic(arguments.output, payload)
        elif arguments.command == "bundle-dependencies":
            payload = bundle_dependencies(
                arguments.source_site,
                arguments.destination,
            )
        elif arguments.command == "materialize":
            payload = materialize_runtime(
                arguments.bundle_root,
                arguments.runtime_root,
            )
        elif arguments.command == "prepare-ufo":
            payload = prepare_ufo(arguments.output, arguments.artifact_root)
        elif arguments.command == "create-manifest":
            payload = create_manifest(arguments)
        else:
            payload = verify_manifest(arguments)
    except (BundleError, OSError, regression.RegressionError) as error:
        print(f"x86-performance-runtime-bundle: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passes", True) is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
