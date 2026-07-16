# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from pyamplicol._sdk import config


def _sdk(tmp_path: Path) -> Path:
    root = tmp_path / "installed path with spaces" / "_sdk"
    (root / "include").mkdir(parents=True)
    (root / "fortran").mkdir()
    (root / "lib").mkdir()
    (root / "rust").mkdir()
    (root / "include" / "rusticol.h").write_text("", encoding="utf-8")
    (root / "include" / "rusticol.hpp").write_text("", encoding="utf-8")
    (root / "fortran" / "rusticol.f90").write_text("", encoding="utf-8")
    (root / "rust" / "rusticol.rs").write_text("", encoding="utf-8")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"archive")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi_version": 1,
                "version": "0.1.0",
                "target": "aarch64-apple-darwin",
                "archive": "lib/librusticol_capi.a",
                "rust_source": "rust/rusticol.rs",
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
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    info = config.load_sdk_info()
    assert info.library == root / "lib" / "librusticol_capi.a"
    assert info.rust_source == root / "rust" / "rusticol.rs"
    assert info.link_flags == (
        str(root / "lib" / "librusticol_capi.a"),
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    )
    assert info.rust_flags == tuple(
        token
        for link_flag in info.link_flags
        for token in ("-C", f"link-arg={link_flag}")
    )
    assert info.to_json()["rust_flags"] == list(info.rust_flags)
    assert info.to_json()["rust_source"] == str(info.rust_source)
    assert info.cargo_encoded_rust_flags.split("\x1f") == list(info.rust_flags)


def test_rustflags_cli_emits_shell_safe_rustc_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    assert config.main(["--rustflags"]) == 0
    output = capsys.readouterr().out.strip()
    assert shlex.split(output) == list(config.load_sdk_info().rust_flags)
    assert str(root / "lib" / "librusticol_capi.a") in output

    assert config.main(["--rust-source"]) == 0
    assert capsys.readouterr().out.strip() == str(root / "rust" / "rusticol.rs")

    assert config.main(["--cargo-rustflags"]) == 0
    encoded = capsys.readouterr().out.rstrip("\n")
    assert encoded.split("\x1f") == list(config.load_sdk_info().rust_flags)


@pytest.mark.parametrize("field", ("archive", "rust_source"))
def test_sdk_metadata_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root = _sdk(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata[field] = "../../outside"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    with pytest.raises(config.SdkUnavailableError, match="escapes"):
        config.load_sdk_info()


def test_sdk_metadata_requires_exact_version_and_uses_wheel_record_for_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(
        config,
        "_package_version",
        lambda: "0.1.0.dev0+candidate.deadbeef",
    )
    with pytest.raises(config.SdkUnavailableError, match="version"):
        config.load_sdk_info()

    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"tampered")
    info = config.load_sdk_info()
    assert info.library.read_bytes() == b"tampered"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert "archive_sha256" not in metadata
