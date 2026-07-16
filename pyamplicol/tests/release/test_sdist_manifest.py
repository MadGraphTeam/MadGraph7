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
        "build_backend/python_lock.py",
        "dependencies/contributor-lock.toml",
        "dependencies/install_dependencies.py",
        "dependencies/patches/**",
        "dependencies/python-runtime-lock.toml",
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
        "THIRD_PARTY_NOTICES.md",
        "licenses/**/*",
        "dependencies/release-lock.toml",
        "schemas/**/*",
    ):
        assert includes[path] == {"sdist"}
    assert includes["README.md"] == {"sdist"}


def test_maturin_generated_sboms_are_disabled() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["maturin"]["sbom"] == {
        "rust": False,
        "auditwheel": False,
    }
