# SPDX-License-Identifier: 0BSD
"""The release tools' hermetic runtime PATH (``tools/release/_common.
runtime_environment``; m-lit lit-sdk-b item [10], lit-rel-xz's NixOS minor):
it finds the base utilities the UMAMI provider Makefiles call (``mkdir``,
``uname``) wherever the caller's PATH has them, never a Rust toolchain, and
is unchanged where /usr/bin or /bin hold them (the release runners)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import _common  # noqa: E402

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX PATH layout")


def _tool(directory: Path, name: str) -> None:
    path = directory / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


def test_base_utilities_come_from_the_callers_path(tmp_path, monkeypatch):
    utilities, toolchain = tmp_path / "coreutils", tmp_path / "toolchain"
    utilities.mkdir()
    toolchain.mkdir()
    for name in _common.BASE_UTILITIES:
        _tool(utilities, name)
    _tool(toolchain, "cargo")
    _tool(toolchain, "mkdir")
    empty = tmp_path / "empty-system"
    empty.mkdir()
    monkeypatch.setenv("PATH", os.pathsep.join([str(toolchain), str(utilities)]))
    # A host whose system directories lack the utilities (NixOS): the
    # directory holding a Rust toolchain is skipped for one without.
    found = _common._base_utility_paths([empty])
    assert found == [utilities]
    monkeypatch.setenv("PATH", str(utilities))
    assert _common._base_utility_paths([empty]) == [utilities]
    # A system directory holding them (a release runner): nothing is added.
    assert _common._base_utility_paths([utilities]) == []


def test_the_runtime_path_holds_no_rust_toolchain(tmp_path):
    environment = _common.runtime_environment(tmp_path / "venv")
    for directory in environment["PATH"].split(os.pathsep):
        for tool in _common.RUST_TOOLCHAIN:
            assert not (Path(directory) / tool).exists(), (directory, tool)
