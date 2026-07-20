# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from pyamplicol._internal import versions

ROOT = Path(__file__).resolve().parents[2]
FIRST_PARTY_ROOTS = (
    ROOT / "src" / "pyamplicol",
    ROOT / "build_backend",
    ROOT / "dependencies",
    ROOT / "tools",
    ROOT / "tests",
    ROOT / "rust",
)
SOURCE_SUFFIXES = {".py", ".rs", ".h", ".hpp", ".f90"}
IGNORED_PARTS = {
    ".git",
    ".artifacts",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".trash",
    ".venv",
    "__pycache__",
    "PYPI_DEPLOYMENT_TEST",
    "build",
    "checkouts",
    "dist",
    "target",
    "venv",
    "wheelhouse",
}


def _eligible(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if IGNORED_PARTS.intersection(relative.parts):
        return False
    return not relative.is_relative_to("src/pyamplicol/assets/models/ufo")


def test_first_party_source_has_explicit_0bsd_spdx_identifier() -> None:
    missing: list[str] = []
    for source_root in FIRST_PARTY_ROOTS:
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if not _eligible(path):
                continue
            header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:5])
            if "SPDX-License-Identifier: 0BSD" not in header:
                missing.append(path.relative_to(ROOT).as_posix())
    assert not missing, "missing 0BSD SPDX identifiers:\n" + "\n".join(missing)


def test_release_source_roots_contain_no_symlinks() -> None:
    links = [
        path.relative_to(ROOT).as_posix()
        for source_root in FIRST_PARTY_ROOTS
        for path in source_root.rglob("*")
        if _eligible(path) and path.is_symlink()
    ]
    assert not links, "release source contains symlinks:\n" + "\n".join(links)


def test_model_assets_have_one_canonical_package_root() -> None:
    expected = ROOT / "src" / "pyamplicol" / "assets" / "models"
    candidates = [
        path
        for path in ROOT.rglob("models")
        if path.is_dir()
        and "assets" in path.parts
        and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    ]
    assert candidates == [expected]


def test_vendored_ufo_sources_are_not_relicensed_as_first_party_code() -> None:
    ufo_root = ROOT / "src" / "pyamplicol" / "assets" / "models" / "ufo"
    assert ufo_root.is_dir()
    assert not any(
        "SPDX-License-Identifier: 0BSD"
        in "\n".join(path.read_text(encoding="utf-8").splitlines()[:5])
        for path in ufo_root.rglob("*.py")
    )


def test_pyamplicol_symbolica_heads_are_owned_by_the_central_registry() -> None:
    package_root = ROOT / "src" / "pyamplicol"
    registry = package_root / "_internal" / "physics" / "symbols.py"
    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if path == registry:
            continue
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        if "pyamplicol::" in source:
            violations.append(f"{relative}: direct pyamplicol namespace literal")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "str"
                and node.args
                and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id == "symbols"
            ):
                violations.append(
                    f"{relative}:{node.lineno}: str(symbols.*) drops its namespace"
                )
            is_symbol_constructor = (
                isinstance(node.func, ast.Name) and node.func.id == "S"
            ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "S")
            if not is_symbol_constructor:
                continue
            literal = (
                node.args[0].value
                if node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                else None
            )
            if literal in {"spenso::f", "spenso::t"}:
                continue
            violations.append(
                f"{relative}:{node.lineno}: Symbolica S(...) bypasses symbols.py"
            )
    assert not violations, "non-central Symbolica symbol construction:\n" + "\n".join(
        violations
    )


def test_python_rust_and_release_lock_wire_versions_are_identical() -> None:
    release = tomllib.loads(
        (ROOT / "dependencies" / "release-lock.toml").read_text(encoding="utf-8")
    )["abis"]
    rust = (ROOT / "rust/crates/rusticol-core/src/lib.rs").read_text(encoding="utf-8")
    contracts = {
        "python_api": "PYTHON_API_VERSION",
        "toml": "TOML_SCHEMA_VERSION",
        "compiled_model": "COMPILED_MODEL_SCHEMA_VERSION",
        "process_artifact": "PROCESS_ARTIFACT_SCHEMA_VERSION",
        "runtime_physics": "RUNTIME_PHYSICS_SCHEMA_VERSION",
        "c_abi": "C_ABI_VERSION",
    }
    for lock_name, constant_name in contracts.items():
        python_value = getattr(versions, constant_name)
        match = re.search(
            rf"pub const {constant_name}: u32 = (\d+);",
            rust,
        )
        assert match is not None, constant_name
        assert int(match.group(1)) == python_value == release[lock_name]

    match = re.search(
        r'pub const SYMBOLICA_SERIALIZATION_ABI: &str = "([^"]+)";',
        rust,
    )
    assert match is not None
    assert (
        match.group(1)
        == versions.SYMBOLICA_SERIALIZATION_ABI
        == release["symbolica_serialization"]
    )

    from pyamplicol.generation.eager_tables import (
        EAGER_PLAN_ABI,
        EAGER_SELECTOR_DOMAINS_ABI,
    )
    from pyamplicol.models.prepared import (
        EAGER_KERNEL_ABI,
        PREPARED_KERNEL_VARIANT_ABI,
        PREPARED_MODEL_BUNDLE_SCHEMA_VERSION,
    )

    string_contracts = {
        "eager_kernel": EAGER_KERNEL_ABI,
        "eager_plan": EAGER_PLAN_ABI,
        "eager_selector_domains": EAGER_SELECTOR_DOMAINS_ABI,
    }
    rust_eager_tables = (
        ROOT / "rust/crates/rusticol-core/src/eager_tables.rs"
    ).read_text(encoding="utf-8")
    for lock_name, python_value in string_contracts.items():
        rust_name = lock_name.upper() + "_ABI"
        match = re.search(
            rf'pub const {rust_name}: &str = "([^"]+)";',
            rust_eager_tables,
        )
        assert match is not None, rust_name
        assert match.group(1) == python_value == release[lock_name]

    eager_manifest = (
        ROOT / "rust/crates/rusticol-core/src/engine/eager_manifest.rs"
    ).read_text(encoding="utf-8")
    match = re.search(
        r'const SYMJIT_APPLICATION_STORAGE_V3_ABI: &str = "([^"]+)";',
        eager_manifest,
    )
    assert match is not None
    assert (
        match.group(1)
        == versions.SYMJIT_APPLICATION_ABI
        == release["symjit_application"]
    )
    assert (
        release["prepared_model_bundle"] == PREPARED_MODEL_BUNDLE_SCHEMA_VERSION
    )
    assert release["prepared_kernel_variant"] == PREPARED_KERNEL_VARIANT_ABI
