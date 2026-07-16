# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_maturin_recursively_includes_every_sdist_source_tree() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = {entry["path"] for entry in pyproject["tool"]["maturin"]["include"]}

    assert {
        "build_backend/**/*",
        "dependencies/patches/**/*",
        "docs/**/*",
        "examples/**/*",
        "rust/**/*",
        "schemas/**/*",
        "tests/**/*",
        "tools/developer/**/*",
        "tools/release/**/*",
        "tools/typing/**/*",
    } <= includes
    assert {
        "justfile",
        "dependencies/install_dependencies.py",
        "rust-toolchain.toml",
    } <= includes
    assert not {path for path in includes if path.endswith("/**")}

    excludes = set(pyproject["tool"]["maturin"]["exclude"])
    assert {
        "docs/*.aux",
        "docs/*.fdb_latexmk",
        "docs/*.fls",
        "docs/*.log",
        "docs/*.toc",
    } <= excludes


def test_root_resources_are_sdist_only_and_wheel_resources_are_namespaced() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    includes = {
        entry["path"]: set(entry["format"])
        for entry in pyproject["tool"]["maturin"]["include"]
        if "format" in entry
    }

    for path in (
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "licenses/**/*",
        "dependencies/release-lock.toml",
        "schemas/**/*",
    ):
        assert includes[path] == {"sdist"}
