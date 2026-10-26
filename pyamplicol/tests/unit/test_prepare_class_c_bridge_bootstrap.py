# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from tools.developer import prepare_class_c_bridge as bootstrap


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_bind_ancestor_runtime_reconstructs_only_staged_runtime(
    tmp_path: Path,
) -> None:
    live = tmp_path / "live-descendant"
    ancestor = tmp_path / "ancestor"
    for root in (live, ancestor):
        _write(root / "pyproject.toml", b"[project]\nname='pyamplicol'\n")
        _write(root / "Cargo.toml", b"[workspace]\n")
        _write(root / "dependencies/contributor-lock.toml", b"[dependency]\n")
        _write(root / "rust/lib.rs", b"pub fn tracked() {}\n")
        _write(root / "src/pyamplicol/__init__.py", b"# tracked package\n")

    live_package = live / "src/pyamplicol"
    _write(live_package / "descendant_only.py", b"raise RuntimeError\n")
    extension = live_package / "_rusticol.abi3.so"
    _write(extension, b"retained native extension")
    for relative in (
        "_sdk/fortran/rusticol.f90",
        "_sdk/include/rusticol.h",
        "_sdk/lib/librusticol_capi.a",
        "_sdk/link.json",
    ):
        _write(live_package / relative, relative.encode("ascii"))
    _write(
        live_package / "_sdk/metadata.json",
        json.dumps({"target": "test-target"}).encode("ascii"),
    )
    _write(
        live_package / "assets/selftest/test-target/expected.json",
        b'{"ok":true}\n',
    )
    (live / "dependencies/checkouts").mkdir(parents=True)
    for relative in bootstrap._IGNORED_NATIVE_INPUTS:
        _write(live / relative, f"{relative}\n".encode("ascii"))

    shadow = tmp_path / "shadow"
    shutil.copytree(ancestor, shadow)
    for relative in bootstrap._IGNORED_NATIVE_INPUTS:
        _write(shadow / relative, (live / relative).read_bytes())
    native_digest = bootstrap._native_build_inputs_digest(shadow)
    extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    _write(
        live / ".artifacts/source-runtime/_build_info.json",
        (
            json.dumps(
                {
                    "candidate_fingerprint": "candidate-test",
                    "source_runtime": {
                        "extension_name": extension.name,
                        "extension_sha256": extension_sha256,
                        "native_build_inputs_sha256": native_digest,
                    },
                    "version": "0.1.0.dev0+candidate.test",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )

    identity = bootstrap._bind_ancestor_runtime(live, ancestor)

    ancestor_extension = ancestor / "src/pyamplicol" / extension.name
    assert not ancestor_extension.is_symlink()
    assert os.stat(ancestor_extension).st_ino == os.stat(extension).st_ino
    linked_header = ancestor / "src/pyamplicol/_sdk/include/rusticol.h"
    assert os.stat(linked_header).st_ino == os.stat(
        live_package / "_sdk/include/rusticol.h"
    ).st_ino
    assert (ancestor / ".artifacts/source-runtime").is_symlink()
    assert (ancestor / "dependencies/checkouts").is_symlink()
    assert all(
        (ancestor / path).is_symlink()
        for path in bootstrap._IGNORED_NATIVE_INPUTS
    )
    assert not (ancestor / "src/pyamplicol/descendant_only.py").exists()
    assert bootstrap._native_build_inputs_digest(ancestor) == native_digest
    assert identity == {
        "candidate_fingerprint": "candidate-test",
        "extension_name": extension.name,
        "extension_sha256": extension_sha256,
        "native_build_inputs_sha256": native_digest,
        "package_version": "0.1.0.dev0+candidate.test",
        "target": "test-target",
    }


def test_bind_release_runtime_uses_release_projection_without_candidate_inputs(
    tmp_path: Path,
) -> None:
    live = tmp_path / "live-descendant"
    ancestor = tmp_path / "ancestor"
    for root in (live, ancestor):
        _write(root / "pyproject.toml", b"[project]\nname='pyamplicol'\n")
        _write(root / "Cargo.toml", b"[workspace]\n")
        _write(root / "src/pyamplicol/__init__.py", b"# tracked package\n")
    _write(
        ancestor / "build_backend/native_build_identity.py",
        (
            b"import hashlib\n"
            b"def native_build_inputs_digest("
            b"root, *, normalize_release_cargo_lock=False):\n"
            b"    if not normalize_release_cargo_lock:\n"
            b"        raise RuntimeError('release projection required')\n"
            b"    return hashlib.sha256((root / 'Cargo.toml').read_bytes()).hexdigest()\n"
        ),
    )

    live_package = live / "src/pyamplicol"
    extension = live_package / "_rusticol.abi3.so"
    _write(extension, b"release native extension")
    for relative in (
        "_sdk/fortran/rusticol.f90",
        "_sdk/include/rusticol.h",
        "_sdk/lib/librusticol_capi.a",
        "_sdk/link.json",
    ):
        _write(live_package / relative, relative.encode("ascii"))
    _write(
        live_package / "_sdk/metadata.json",
        json.dumps({"target": "release-target"}).encode("ascii"),
    )
    _write(
        live_package / "assets/selftest/release-target/expected.json",
        b'{"ok":true}\n',
    )

    native_digest = bootstrap._native_build_inputs_digest(
        ancestor,
        normalize_release_cargo_lock=True,
    )
    extension_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    _write(
        live / ".artifacts/source-runtime/_build_info.json",
        (
            json.dumps(
                {
                    "native_build_inputs_sha256": native_digest,
                    "publishable": True,
                    "source_runtime": {
                        "extension_name": extension.name,
                        "extension_sha256": extension_sha256,
                        "mode": "release",
                        "native_build_inputs_sha256": native_digest,
                    },
                    "version": "0.1.0",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )

    identity = bootstrap._bind_ancestor_runtime(live, ancestor)

    assert not (ancestor / "dependencies/checkouts").exists()
    assert all(not (ancestor / path).exists() for path in bootstrap._IGNORED_NATIVE_INPUTS)
    assert identity == {
        "candidate_fingerprint": None,
        "extension_name": extension.name,
        "extension_sha256": extension_sha256,
        "native_build_inputs_sha256": native_digest,
        "package_version": "0.1.0",
        "target": "release-target",
    }
