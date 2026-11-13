# SPDX-License-Identifier: 0BSD
"""Shared GMP/MPFR/MPC of distribution wheels (build_backend/arithmetic.py)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import arithmetic  # noqa: E402


def test_the_locked_gmp_mpfr_sys_release_is_the_source() -> None:
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    locked = [
        package["version"]
        for package in lock["package"]
        if package["name"] == arithmetic.SOURCE_CRATE
    ]
    assert [arithmetic.locked_source_version(ROOT)] == locked


def test_the_cargo_feature_selects_system_libraries_in_every_artifact_crate() -> None:
    for crate in ("rusticol-python", "rusticol-capi"):
        manifest = tomllib.loads(
            (ROOT / "rust" / "crates" / crate / "Cargo.toml").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["features"][arithmetic.CARGO_FEATURE] == [
            f"dep:{arithmetic.SOURCE_CRATE}",
            f"{arithmetic.SOURCE_CRATE}/use-system-libs",
        ]
        assert manifest["dependencies"][arithmetic.SOURCE_CRATE] == {
            "workspace": True,
            "optional": True,
        }
        assert arithmetic.CARGO_FEATURE not in manifest["features"].get("default", [])


def test_the_sdk_profile_drops_lto_bitcode_from_the_archive() -> None:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    profile = pyproject["tool"]["pyamplicol"]["native-build"]["sdk"]["profile"]
    assert cargo["profile"][profile] == {"inherits": "release", "lto": "off"}


@pytest.mark.parametrize(
    ("soname", "private"),
    [
        ("libgmp.so.10", "libgmp-pyamplicol.so.10"),
        ("libmpfr.so.6", "libmpfr-pyamplicol.so.6"),
        ("libmpc.so.3", "libmpc-pyamplicol.so.3"),
    ],
)
def test_linux_libraries_get_private_sonames(soname: str, private: str) -> None:
    assert arithmetic._private_soname(soname) == private


def test_bundled_source_trees_are_found_once_each(tmp_path: Path) -> None:
    for name in ("gmp-6.3.0-c", "mpfr-4.2.2-c", "mpc-1.4.1-c"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "configure").write_text("#!/bin/sh\n", encoding="utf-8")
    sources = arithmetic._library_sources(tmp_path)
    assert {name: path.name for name, path in sources.items()} == {
        "gmp": "gmp-6.3.0-c",
        "mpfr": "mpfr-4.2.2-c",
        "mpc": "mpc-1.4.1-c",
    }
    (tmp_path / "gmp-6.2.1-c").mkdir()
    (tmp_path / "gmp-6.2.1-c" / "configure").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="one gmp source tree"):
        arithmetic._library_sources(tmp_path)


@pytest.mark.parametrize("platform", ("linux", "darwin"))
def test_cargo_environment_points_the_system_library_mode_at_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
) -> None:
    monkeypatch.setattr(arithmetic.sys, "platform", platform)
    runtime = arithmetic.ArithmeticRuntime(tmp_path, {})
    environment = runtime.cargo_environment("--remap-path-prefix=/a=/b")
    library = str(tmp_path / "lib")
    assert environment["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") == [
        "--remap-path-prefix=/a=/b",
        "-L",
        f"native={library}",
    ]
    assert environment["C_INCLUDE_PATH"] == str(tmp_path / "include")
    assert environment["LIBRARY_PATH"] == library
    assert ("LD_LIBRARY_PATH" in environment) == (platform == "linux")


def test_real_file_follows_relative_links_without_canonicalising(
    tmp_path: Path,
) -> None:
    real = tmp_path / "libgmp.10.dylib"
    real.write_bytes(b"x")
    (tmp_path / "libgmp.dylib").symlink_to(real.name)
    assert arithmetic._real_file(tmp_path / "libgmp.dylib") == real
    (tmp_path / "loop-a").symlink_to("loop-b")
    (tmp_path / "loop-b").symlink_to("loop-a")
    with pytest.raises(RuntimeError, match="symlink loop"):
        arithmetic._real_file(tmp_path / "loop-a")
