# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

from audit_sdist import (  # noqa: E402
    PREPARED_MODEL_SDIST_MEMBERS,
    REQUIRED_SDIST_MEMBERS,
)


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
        "flake.nix",
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


def test_sdist_inventory_requires_both_architecture_prepared_packs() -> None:
    root = "src/pyamplicol/assets/prepared_models"
    expected = {
        f"{root}/__init__.py",
        f"{root}/built-in-sm-jit-o3-aarch64.metadata.json",
        f"{root}/built-in-sm-jit-o3-aarch64.pyamplicol-model",
        f"{root}/built-in-sm-jit-o3-x86_64.metadata.json",
        f"{root}/built-in-sm-jit-o3-x86_64.pyamplicol-model",
    }
    assert expected == PREPARED_MODEL_SDIST_MEMBERS
    assert PREPARED_MODEL_SDIST_MEMBERS <= REQUIRED_SDIST_MEMBERS
