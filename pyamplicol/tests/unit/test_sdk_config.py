# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pyamplicol._sdk import config


def _sdk(tmp_path: Path) -> Path:
    root = tmp_path / "installed path with spaces" / "_sdk"
    (root / "include").mkdir(parents=True)
    (root / "fortran").mkdir()
    (root / "lib").mkdir()
    (root / "sboms").mkdir()
    (root / "include" / "rusticol.h").write_text("", encoding="utf-8")
    (root / "include" / "rusticol.hpp").write_text("", encoding="utf-8")
    (root / "fortran" / "rusticol.f90").write_text("", encoding="utf-8")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"archive")
    (root / "sboms" / "rusticol-capi.cyclonedx.json").write_bytes(b"{}\n")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi_version": 1,
                "version": "0.1.0",
                "target": "aarch64-apple-darwin",
                "archive": "lib/librusticol_capi.a",
                "archive_sha256": hashlib.sha256(b"archive").hexdigest(),
                "sbom": "sboms/rusticol-capi.cyclonedx.json",
                "sbom_sha256": hashlib.sha256(b"{}\n").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (root / "link.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "aarch64-apple-darwin",
                "system_libraries": ["System", "m"],
                "frameworks": ["Security"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_sdk_info_returns_a_complete_native_link_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config.importlib.metadata, "version", lambda _name: "0.1.0")

    info = config.load_sdk_info()
    assert info.library == root / "lib" / "librusticol_capi.a"
    assert info.sbom == root / "sboms" / "rusticol-capi.cyclonedx.json"
    assert info.link_flags == (
        str(root / "lib" / "librusticol_capi.a"),
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    )


def test_sdk_metadata_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata["archive"] = "../../outside.a"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config.importlib.metadata, "version", lambda _name: "0.1.0")

    with pytest.raises(config.SdkUnavailableError, match="escapes"):
        config.load_sdk_info()


def test_sdk_metadata_requires_exact_version_and_archive_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(
        config.importlib.metadata,
        "version",
        lambda _name: "0.1.0.dev0+candidate.deadbeef",
    )
    with pytest.raises(config.SdkUnavailableError, match="version"):
        config.load_sdk_info()

    monkeypatch.setattr(config.importlib.metadata, "version", lambda _name: "0.1.0")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"tampered")
    with pytest.raises(config.SdkUnavailableError, match="digest"):
        config.load_sdk_info()

    (root / "lib" / "librusticol_capi.a").write_bytes(b"archive")
    (root / "sboms" / "rusticol-capi.cyclonedx.json").write_bytes(b"tampered")
    with pytest.raises(config.SdkUnavailableError, match="SBOM digest"):
        config.load_sdk_info()
