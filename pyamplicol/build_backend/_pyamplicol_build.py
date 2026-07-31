# SPDX-License-Identifier: 0BSD
"""Read-only-source PEP 517 backend for pyAmpliCol."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import maturin  # type: ignore[import-untyped]
from native_build_identity import (
    canonical_candidate_cargo_config_bytes as _canonical_candidate_cargo_config_bytes,
)
from native_build_identity import (
    native_build_inputs_digest as _native_build_inputs_digest,
)
from package_version import (
    canonical_package_version,
    check_contributor_lock_consistency,
)
from prepared_models import (
    discard_release_packaged_prepared_model_store,
    project_release_packaged_prepared_model_store,
    stage_packaged_prepared_models,
)
from sdk import build_sdk

ROOT = Path(__file__).resolve().parents[1]
_CONTRIBUTOR_LOCK = Path("dependencies/contributor-lock.toml")
_RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)
_ARTIFACT_IDENTITY_CONTRACT = {
    "kind": "pyamplicol-runtime-payload-identity",
    "schema_version": 1,
}


def _require_artifact_identity_contract(manifest: Mapping[str, object]) -> None:
    extensions = manifest.get("extensions")
    if (
        not isinstance(extensions, dict)
        or extensions.get("artifact_identity") != _ARTIFACT_IDENTITY_CONTRACT
    ):
        raise RuntimeError(
            "process artifact lacks the current identity contract; regenerate it"
        )


def _runtime_artifact_id(manifest: Mapping[str, object]) -> str:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("process artifact payload inventory is invalid")
    records = [
        dict(payload)
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("role") in _RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES
    ]
    records.sort(key=lambda payload: str(payload.get("path", "")))
    canonical = (
        json.dumps(
            {
                "kind": "pyamplicol-runtime-payload-identity",
                "schema_version": 1,
                "payloads": records,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


ALLOWLIST = (
    ".gitattributes",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "build_backend",
    "config",
    "dependencies",
    "docs",
    "examples",
    "justfile",
    "licenses",
    "pyproject.toml",
    "release_assets",
    "rust",
    "rust-toolchain.toml",
    "schemas",
    "src",
    "tests",
    "tools/ci/__init__.py",
    "tools/ci/memory_watchdog.py",
    "tools/developer",
    "tools/performance_report",
    "tools/release",
    "tools/typing",
)
_PROFILING_CAMPAIGN_SOURCE = Path("docs/performance_reports/macbook_M3_manual")
_PROFILING_CAMPAIGN_REQUIRED_FILES = (
    Path("README.md"),
    Path("TABLE_FILLING.md"),
    Path("build_pdf.py"),
    Path("pyAmpliCol.tex"),
    Path("report-workspace.json"),
    Path("report_environment.json"),
    Path("report_environment.tex"),
    Path("result_tables.py"),
    Path("result_validation_summary.tex"),
    Path("steer_performance_campaign.py"),
)
_PROFILING_CAMPAIGN_GLOBS = (
    "result_*_table.tex",
    "section_*.tex",
    "results/*.json",
)
_PROFILING_CAMPAIGN_FILE_COUNT = 55
_RELEASE_BUILD_INFO_FIELDS = {
    "publishable",
    "schema_version",
    "selftest_fixture_bootstrap",
    "source_revision",
    "version",
}
_PERFORMANCE_REPORT_REQUIRED_MODULES = {
    "__init__.py",
    "cli.py",
    "manual_campaign.py",
    "scheduler.py",
}
_LEGACY_ORACLE_REQUIRED_MODULES = {
    "__init__.py",
    "checkout.py",
    "model.py",
    "probe.py",
}
IGNORED_NAMES = {
    "__pycache__",
    ".agent-work",
    ".artifacts",
    ".coverage",
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".trash",
    ".venv",
    "PYPI_DEPLOYMENT_TEST",
    "build",
    "checkouts",
    "dist",
    "htmlcov",
    "wheelhouse",
    "target",
    "venv",
}
_EXCLUDED_TREES = (
    Path("dependencies/patches"),
    Path("docs/.result_outputs"),
    Path("docs/archive"),
    Path("outputs"),
    Path("src/pyamplicol/_sdk/fortran"),
    Path("src/pyamplicol/_sdk/include"),
    Path("src/pyamplicol/_sdk/lib"),
)
_EXCLUDED_PATHS = {
    Path(".cargo/config.toml"),
    Path("build_backend/python_lock.py"),
    Path("dependencies/candidate-Cargo.lock"),
    Path("dependencies/candidate-cargo-config.toml"),
    Path("dependencies/contributor-lock.toml"),
    Path("dependencies/install_dependencies.py"),
    Path("dependencies/install-state.json"),
    Path("dependencies/python-runtime-lock.toml"),
    Path("dependencies/symbolica_patches.tar.gz"),
    Path("src/pyamplicol/_sdk/link.json"),
    Path("src/pyamplicol/_sdk/metadata.json"),
}
_IGNORED_SUFFIXES = (".mod", ".pyc", ".pyd", ".pyo", ".whl")
_NATIVE_EXTENSION_SUFFIXES = (".dylib", ".pyd", ".so")
_TEX_BUILD_SUFFIXES = (
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".run.xml",
    ".synctex.gz",
    ".toc",
)
# SymJIT Application v3 stores the program and MIR, then recompiles it in
# Application::load. Its wire format uses native usize fields, so this template
# is shared only by the release's 64-bit little-endian targets.
_PORTABLE_SELFTEST_TEMPLATE = "portable-64le"
_PORTABLE_SELFTEST_TARGETS = frozenset(
    {
        "aarch64-apple-darwin",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
    }
)
_SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT = "build-current-candidate-site-v1"
_RELEASE_PREPARED_MODEL_BOOTSTRAP_CONTEXT = "release-prepared-model-producer-v1"
_CANDIDATE_SOURCES = {
    "gammaloop",
    "symbolica",
    "symbolica-community",
    "symjit",
}
_INJECTION_ENVIRONMENT_NAMES = {
    "AR",
    "CARGO",
    "C_INCLUDE_PATH",
    "CC",
    "CFLAGS",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CXX",
    "CXXFLAGS",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LDFLAGS",
    "LIBRARY_PATH",
    "MACOSX_DEPLOYMENT_TARGET",
    "OBJC_INCLUDE_PATH",
    "PKG_CONFIG_PATH",
    "PYAMPLICOL_BUILD_OVERLAY",
    "PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256",
    "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP",
    "PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP",
    "PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT",
    "PYAMPLICOL_SDK_STAGING",
    "PYTHONHOME",
    "PYTHONPATH",
    "RANLIB",
    "SDKROOT",
}
_INJECTION_ENVIRONMENT_PREFIXES = (
    "CARGO_",
    "GIT_",
    "MATURIN_",
    "PYO3_",
    "RUST",
)
_Result = TypeVar("_Result")
_delegation_depth = 0


def _prepared_model_bootstrap(mode: str) -> bool:
    """Return whether this non-publishable build only bootstraps pack creation."""

    value = os.environ.get("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", "0")
    if value not in {"0", "1"}:
        raise RuntimeError(
            "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP must be either '0' or '1'"
        )
    if value == "1" and mode != "candidate":
        raise RuntimeError(
            "prepared-model bootstrap is restricted to non-publishable candidate builds"
        )
    return value == "1"


def _release_prepared_model_bootstrap(
    mode: str,
    context: str | None,
) -> bool:
    """Authorize only the dedicated release prepared-model producer."""

    if context is None:
        return False
    if mode != "release":
        raise RuntimeError(
            "release prepared-model bootstrap requires release dependency mode"
        )
    if context != _RELEASE_PREPARED_MODEL_BOOTSTRAP_CONTEXT:
        raise RuntimeError(
            "release prepared-model bootstrap requires the explicit producer context"
        )
    return True


def _selftest_fixture_bootstrap(
    mode: str,
    *,
    explicit_context: bool = False,
) -> bool:
    """Authorize the one-time candidate used to regenerate the self-test fixture."""

    value = os.environ.get("PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP", "0")
    context = os.environ.get("PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT")
    if value not in {"0", "1"}:
        raise RuntimeError(
            "PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP must be either '0' or '1'"
        )
    if value == "0":
        if context is not None:
            raise RuntimeError(
                "self-test fixture bootstrap context requires the bootstrap flag"
            )
        return False
    if mode != "candidate":
        raise RuntimeError(
            "self-test fixture bootstrap is restricted to non-publishable "
            "candidate builds"
        )
    if not explicit_context or context != _SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT:
        raise RuntimeError(
            "self-test fixture bootstrap is restricted to the explicit "
            "build-current-candidate-site context"
        )
    return True


def _strip_prepared_model_payloads(overlay: Path) -> None:
    """Remove stale bundles from a bootstrap wheel used only to create replacements."""

    root = overlay / "src" / "pyamplicol" / "assets" / "prepared_models"
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("prepared-model bootstrap input has no safe asset directory")
    for path in root.iterdir():
        if path.name == "__init__.py":
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"prepared-model bootstrap found an unsafe asset entry: {path.name}"
            )
        if path.suffix not in {".json", ".pyamplicol-model"}:
            raise RuntimeError(
                f"prepared-model bootstrap found an unexpected asset: {path.name}"
            )
        path.unlink()


def _mark_selftest_fixture_bootstrap(
    overlay: Path,
    *,
    prepared_model_recovery: bool = False,
) -> None:
    """Mark a regeneration-only build and remove its stale self-test fixture.

    The standalone self-test recovery remains restricted to an exact clean
    candidate revision. An explicit prepared-model recovery is necessarily
    coupled to self-test regeneration, however, and may operate on a dirty
    candidate checkout because its wheel is non-publishable and omits both
    generated asset families.
    """

    path = overlay / "src" / "pyamplicol" / "_build_info.json"
    try:
        build_info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "self-test fixture bootstrap has no candidate build marker"
        ) from error
    if (
        not isinstance(build_info, dict)
        or build_info.get("schema_version") != 1
        or build_info.get("publishable") is not False
        or build_info.get("selftest_fixture_bootstrap") is not False
    ):
        raise RuntimeError(
            "self-test fixture bootstrap requires an ordinary non-publishable "
            "candidate marker"
        )
    source_revision = build_info.get("source_revision")
    if prepared_model_recovery:
        release_recovery = (
            build_info.get("release_prepared_model_bootstrap") is True
            and build_info.get("candidate_fingerprint") is None
            and "source_revision" not in build_info
        )
        candidate_fingerprint = build_info.get("candidate_fingerprint")
        candidate_recovery = (
            build_info.get("release_prepared_model_bootstrap") is None
            and isinstance(candidate_fingerprint, str)
            and re.fullmatch(r"[0-9a-f]{12}", candidate_fingerprint) is not None
            and "source_revision" in build_info
            and (
                source_revision is None
                or (
                    isinstance(source_revision, str)
                    and re.fullmatch(r"[0-9a-f]{40}", source_revision) is not None
                )
            )
        )
        if not (candidate_recovery or release_recovery):
            raise RuntimeError(
                "prepared-model recovery requires a candidate or release "
                "prepared-model bootstrap marker"
            )
    elif (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise RuntimeError(
            "self-test fixture bootstrap requires an exact clean source revision"
        )
    build_info["selftest_fixture_bootstrap"] = True
    path.write_text(
        json.dumps(build_info, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fixture_root = overlay / "src" / "pyamplicol" / "assets" / "selftest"
    if not fixture_root.is_dir() or fixture_root.is_symlink():
        raise RuntimeError(
            "self-test fixture bootstrap input has no safe fixture directory"
        )
    entries = tuple(fixture_root.iterdir())
    template = fixture_root / _PORTABLE_SELFTEST_TEMPLATE
    if entries != (template,) or not template.is_dir() or template.is_symlink():
        raise RuntimeError(
            "self-test fixture bootstrap input has an unexpected fixture inventory"
        )
    shutil.rmtree(template)


def _build_mode() -> str:
    value = os.environ.get("PYAMPLICOL_BUILD_MODE", "release")
    if value not in {"candidate", "release"}:
        raise RuntimeError("PYAMPLICOL_BUILD_MODE must be 'candidate' or 'release'")
    return value


def _check_dependencies(mode: str) -> None:
    command = [
        sys.executable,
        "-I",
        str(ROOT / "tools" / "release" / "check_dependencies.py"),
    ]
    if mode == "candidate":
        command.append("--candidate")
    subprocess.run(
        command,
        cwd=ROOT,
        env=_clean_environment(),
        check=True,
    )


def _is_excluded(relative: Path) -> bool:
    if any(
        part in IGNORED_NAMES or part.endswith(".egg-info") for part in relative.parts
    ):
        return True
    if relative in _EXCLUDED_PATHS or any(
        relative.is_relative_to(tree) for tree in _EXCLUDED_TREES
    ):
        return True
    if relative.name.endswith(_IGNORED_SUFFIXES):
        return True
    if (
        relative.parent == Path("src/pyamplicol")
        and relative.name.startswith("_rusticol")
        and relative.name.endswith(_NATIVE_EXTENSION_SUFFIXES)
    ):
        return True
    return relative.is_relative_to(Path("docs")) and relative.name.endswith(
        _TEX_BUILD_SUFFIXES
    )


def _reject_symlinks(path: Path, relative: Path = Path()) -> None:
    if relative.parts and _is_excluded(relative):
        return
    if path.is_symlink():
        raise RuntimeError(f"build inputs may not be symlinks: {path}")
    if path.is_dir():
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            child_relative = relative / child.name
            if _is_excluded(child_relative):
                continue
            _reject_symlinks(child, child_relative)


def _copy_ignore(root: Path) -> Callable[[str, list[str]], set[str]]:
    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).relative_to(root)
        return {name for name in names if _is_excluded(relative / name)}

    return ignore


def _is_allowlisted(relative: Path) -> bool:
    return any(
        relative == Path(name) or Path(name) in relative.parents for name in ALLOWLIST
    )


def _git_inventory() -> list[Path] | None:
    if not os.path.lexists(ROOT / ".git"):
        return None
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "-z", "--"],
        env=_clean_environment(),
        check=True,
        capture_output=True,
    )
    files = [Path(os.fsdecode(item)) for item in completed.stdout.split(b"\0") if item]
    if not files:
        history = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
            env=_clean_environment(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if history.returncode != 0:
            # An unpacked source archive and the initial no-history bootstrap have
            # no tracked-file inventory. Their allowlisted archive bytes are the
            # complete source contract.
            return None
    inventory: list[Path] = []
    for relative in files:
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Git reported an unsafe build input: {relative}")
        if _is_allowlisted(relative) and not _is_excluded(relative):
            inventory.append(relative)
    return sorted(set(inventory), key=lambda path: path.as_posix())


def _archive_inventory() -> list[Path]:
    inventory: list[Path] = []

    def visit(directory: Path, relative: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            child_relative = relative / child.name
            if _is_excluded(child_relative):
                continue
            if child.is_symlink():
                raise RuntimeError(f"build inputs may not be symlinks: {child}")
            if child.is_dir():
                visit(child, child_relative)
            elif child.is_file():
                inventory.append(child_relative)
            else:
                raise RuntimeError(f"build inputs must be regular files: {child}")

    for name in ALLOWLIST:
        relative = Path(name)
        source = ROOT / relative
        if _is_excluded(relative) or not os.path.lexists(source):
            continue
        if source.is_symlink():
            raise RuntimeError(f"build inputs may not be symlinks: {source}")
        if source.is_dir():
            visit(source, relative)
        elif source.is_file():
            inventory.append(relative)
        else:
            raise RuntimeError(f"build inputs must be regular files: {source}")
    return inventory


def _reject_symlink_ancestors(path: Path) -> None:
    relative = path.relative_to(ROOT)
    current = ROOT
    for part in relative.parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"build inputs may not be symlinks: {current}")


def _copy_allowlisted_source(destination: Path) -> None:
    destination.mkdir(parents=True)
    inventory = _git_inventory()
    if inventory is None:
        inventory = _archive_inventory()
    for relative in inventory:
        source = ROOT / relative
        _reject_symlink_ancestors(source)
        try:
            mode = source.lstat().st_mode
        except FileNotFoundError:
            # A tracked deletion remains deleted in the overlay.
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"build inputs must be regular files: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)


def _stage_packaged_examples(overlay: Path) -> None:
    source = overlay / "examples"
    target = overlay / "src" / "pyamplicol" / "_examples"
    if not source.is_dir():
        raise RuntimeError("wheel build input has no examples directory")
    if target.exists():
        raise RuntimeError("wheel overlay already contains packaged examples")
    shutil.copytree(source, target, ignore=_copy_ignore(overlay))


def _profiling_campaign_inventory(source: Path) -> tuple[Path, ...]:
    selected = set(_PROFILING_CAMPAIGN_REQUIRED_FILES)
    for pattern in _PROFILING_CAMPAIGN_GLOBS:
        selected.update(path.relative_to(source) for path in source.glob(pattern))
    missing = tuple(
        relative
        for relative in _PROFILING_CAMPAIGN_REQUIRED_FILES
        if not (source / relative).is_file()
    )
    if missing or len(selected) != _PROFILING_CAMPAIGN_FILE_COUNT:
        detail = ", ".join(path.as_posix() for path in missing)
        raise RuntimeError(
            "wheel profiling campaign has an invalid inventory"
            + (f": missing {detail}" if detail else "")
        )
    for relative in selected:
        path = source / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"wheel profiling campaign has an unsafe member: {relative}"
            )
    return tuple(sorted(selected, key=lambda path: path.as_posix()))


def _stage_packaged_profiling_campaign(overlay: Path) -> None:
    source = overlay / _PROFILING_CAMPAIGN_SOURCE
    target = overlay / "src" / "pyamplicol" / "_profiling_campaign"
    if not source.is_dir():
        raise RuntimeError("wheel build input has no profiling campaign directory")
    if target.exists():
        raise RuntimeError("wheel overlay already contains a profiling campaign")
    target.mkdir(parents=True)
    for relative in _profiling_campaign_inventory(source):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)

    performance_source = overlay / "tools" / "performance_report"
    watchdog_source = overlay / "tools" / "ci" / "memory_watchdog.py"
    legacy_source = overlay / "tools" / "developer" / "legacy_amplicol.py"
    oracle_source = overlay / "tools" / "developer" / "legacy_oracle"
    runtime_target = overlay / "src" / "pyamplicol" / "_performance_report"
    performance_modules = tuple(sorted(performance_source.glob("*.py")))
    oracle_modules = tuple(sorted(oracle_source.glob("*.py")))
    if (
        not _PERFORMANCE_REPORT_REQUIRED_MODULES.issubset(
            {path.name for path in performance_modules}
        )
        or not _LEGACY_ORACLE_REQUIRED_MODULES.issubset(
            {path.name for path in oracle_modules}
        )
        or not watchdog_source.is_file()
        or not legacy_source.is_file()
    ):
        raise RuntimeError("wheel build input has incomplete profiling runtime sources")
    for path in (*performance_modules, watchdog_source, legacy_source, *oracle_modules):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"wheel profiling runtime has an unsafe member: {path}")
    if runtime_target.exists():
        raise RuntimeError("wheel overlay already contains a profiling runtime")
    runtime_target.mkdir(parents=True)
    for path in performance_modules:
        shutil.copy2(path, runtime_target / path.name)
    shutil.copy2(watchdog_source, runtime_target / "_memory_watchdog.py")
    developer_target = runtime_target / "_developer"
    developer_target.mkdir()
    (developer_target / "__init__.py").write_text(
        "# SPDX-License-Identifier: 0BSD\n",
        encoding="utf-8",
    )
    shutil.copy2(legacy_source, developer_target / legacy_source.name)
    oracle_target = developer_target / "legacy_oracle"
    oracle_target.mkdir()
    for path in oracle_modules:
        shutil.copy2(path, oracle_target / path.name)


def _stage_python_stub(overlay: Path) -> None:
    source = (
        overlay
        / "rust"
        / "crates"
        / "rusticol-python"
        / "stubs"
        / "pyamplicol"
        / "_rusticol.pyi"
    )
    target = overlay / "src" / "pyamplicol" / "_rusticol.pyi"
    if not source.is_file():
        raise RuntimeError("wheel build input has no maintained Rusticol stub")
    if target.exists():
        raise RuntimeError("wheel overlay already contains the Rusticol stub")
    shutil.copy2(source, target)


def _stage_runtime_resources(overlay: Path) -> None:
    """Place wheel-owned schemas below the pyamplicol package namespace."""

    package_assets = overlay / "src" / "pyamplicol" / "assets"
    sources = {
        overlay / "schemas" / "README.md": package_assets / "schemas" / "README.md",
        overlay / "schemas" / "artifact-manifest-v3.schema.json": (
            package_assets / "schemas" / "artifact-manifest-v3.schema.json"
        ),
        overlay / "schemas" / "runtime-physics-v1.schema.json": (
            package_assets / "schemas" / "runtime-physics-v1.schema.json"
        ),
    }
    for source, target in sources.items():
        if not source.is_file():
            raise RuntimeError(
                f"wheel build input is missing runtime resource: {source}"
            )
        if target.exists():
            raise RuntimeError(
                f"wheel overlay already contains runtime resource: {target}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _portable_selftest_validators(source_root: Path) -> tuple[Any, Any]:
    """Load the dependency-free release contract from the source being built."""

    source = source_root / "tools" / "release" / "prepare_selftest_fixture.py"
    spec = importlib.util.spec_from_file_location(
        "_pyamplicol_portable_selftest_contract",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the portable self-test contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact_validator = getattr(
        module,
        "validate_portable_artifact_capabilities",
        None,
    )
    execution_validator = getattr(
        module,
        "validate_portable_execution_manifest",
        None,
    )
    if not callable(artifact_validator) or not callable(execution_validator):
        raise RuntimeError("portable self-test contract has incomplete validators")
    return artifact_validator, execution_validator


def _validate_portable_selftest_executions(
    fixture: Path,
    manifest: Mapping[str, object],
    *,
    source_root: Path,
) -> None:
    """Fail the build before staging a legacy or partial compiled fixture."""

    validate_artifact, validate_execution = _portable_selftest_validators(source_root)
    validate_artifact(
        manifest,
        context="portable self-test artifact manifest",
    )

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("portable self-test payload inventory is invalid")
    execution_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("role") == "evaluator-manifest"
        and str(payload.get("path", "")).endswith("/execution.json")
    ]
    if not execution_payloads:
        raise RuntimeError("portable self-test has no compiled execution manifest")

    for payload in execution_payloads:
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("portable self-test execution manifest has no path")
        path = Path(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise RuntimeError("portable self-test execution-manifest path is unsafe")
        execution_path = fixture / "artifact" / path
        try:
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"portable self-test execution manifest is invalid: {relative}"
            ) from error
        if not isinstance(execution, dict):
            raise RuntimeError(
                f"portable self-test execution manifest is invalid: {relative}"
            )
        validate_execution(
            execution,
            context=f"portable self-test execution manifest {relative}",
        )


def _stage_selftest_fixture(overlay: Path, target: str) -> None:
    """Materialize the portable MIR fixture for the wheel's Rust target."""

    if target not in _PORTABLE_SELFTEST_TARGETS:
        raise RuntimeError(f"unsupported self-test target: {target}")

    with (overlay / "Cargo.toml").open("rb") as stream:
        cargo = tomllib.load(stream)
    try:
        cargo_version = cargo["workspace"]["package"]["version"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("build overlay has no workspace package version") from error
    if not isinstance(cargo_version, str) or not cargo_version:
        raise RuntimeError("build overlay workspace package version is invalid")
    package_version = cargo_version.replace("-dev.", ".dev")
    fixture_root = overlay / "src" / "pyamplicol" / "assets" / "selftest"
    if not fixture_root.is_dir():
        raise RuntimeError("wheel build input has no portable self-test fixture")
    template = fixture_root / _PORTABLE_SELFTEST_TEMPLATE
    if not template.is_dir() or template.is_symlink():
        raise RuntimeError("wheel build input has no portable 64-bit self-test fixture")
    for candidate in fixture_root.iterdir():
        if candidate == template:
            continue
        raise RuntimeError(f"unexpected source self-test fixture: {candidate}")
    selected = fixture_root / target
    shutil.copytree(template, selected, symlinks=False)
    shutil.rmtree(template)
    expected_path = selected / "expected.json"
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid self-test expectation: {expected_path}") from error
    if (
        not isinstance(expected, dict)
        or expected.get("target") != _PORTABLE_SELFTEST_TEMPLATE
        or expected.get("compatible_targets") != sorted(_PORTABLE_SELFTEST_TARGETS)
        or expected.get("serialization")
        != {
            "endianness": "little",
            "kind": "symjit-application-mir-v3",
            "load_behavior": "recompile-mir-for-loading-host",
            "source_optimization_level": 2,
            "word_size_bits": 64,
        }
    ):
        raise RuntimeError("portable self-test expectation is invalid")
    expected["target"] = target
    expected_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path = selected / "artifact" / "artifact.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _require_artifact_identity_contract(manifest)
        manifest["producer"]["version"] = package_version
        manifest["runtime"]["engine_version"] = package_version
        producer_target = manifest["producer"]["target"]
        if (
            not isinstance(producer_target, dict)
            or producer_target.get("triple") != _PORTABLE_SELFTEST_TEMPLATE
        ):
            raise RuntimeError("portable self-test producer target is invalid")
        producer_target["triple"] = target
        producer_target["cpu_features"] = []
        payloads = manifest["payloads"]
        if not isinstance(payloads, list):
            raise RuntimeError("portable self-test payload inventory is invalid")
        _validate_portable_selftest_executions(
            selected,
            manifest,
            source_root=overlay,
        )
        evaluator_targets = 0
        for payload in payloads:
            if not isinstance(payload, dict):
                raise RuntimeError("portable self-test payload entry is invalid")
            payload_target = payload.get("target")
            if payload_target is None:
                continue
            if (
                not isinstance(payload_target, dict)
                or payload_target.get("triple") != _PORTABLE_SELFTEST_TEMPLATE
            ):
                raise RuntimeError("portable self-test payload target is invalid")
            payload_target["triple"] = target
            payload_target["cpu_features"] = []
            evaluator_targets += 1
        if evaluator_targets == 0:
            raise RuntimeError(
                "portable self-test has no target-tagged evaluator state"
            )
        compiled_payloads = [
            payload
            for payload in payloads
            if isinstance(payload, dict) and payload.get("role") == "compiled-model"
        ]
        if len(compiled_payloads) != 1:
            raise RuntimeError("portable self-test must contain one compiled model")
        compiled_payload = compiled_payloads[0]
        compiled_relative = compiled_payload.get("path")
        if not isinstance(compiled_relative, str):
            raise RuntimeError("portable self-test compiled model has no path")
        compiled_path = Path(compiled_relative)
        if compiled_path.is_absolute() or any(
            part in {"", ".", ".."} for part in compiled_path.parts
        ):
            raise RuntimeError("portable self-test compiled-model path is unsafe")
        compiled_path = selected / "artifact" / compiled_path
        compiled_model = json.loads(compiled_path.read_text(encoding="utf-8"))
        if not isinstance(compiled_model, dict):
            raise RuntimeError("portable self-test compiled model is invalid")
        compiled_producer = compiled_model.get("producer")
        if not isinstance(compiled_producer, dict):
            raise RuntimeError("portable self-test compiled model is invalid")
        compiled_producer["pyamplicol"] = package_version
        compiled_data = (
            json.dumps(
                compiled_model,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        compiled_path.write_bytes(compiled_data)
        compiled_payload["sha256"] = hashlib.sha256(compiled_data).hexdigest()
        compiled_payload["size_bytes"] = len(compiled_data)
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid self-test artifact manifest: {path}") from error
    manifest["artifact_id"] = _runtime_artifact_id(manifest)
    path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _digest_item(digest: Any, name: str, data: bytes) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _candidate_state(
    candidate_lock: Path,
    candidate_config: Path,
    installer_state: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(installer_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid candidate installer state: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("candidate installer state must be a schema-v1 object")
    if payload.get("publishable") is not False:
        raise RuntimeError("candidate installer state must be non-publishable")
    if not candidate_lock.is_file() or not candidate_config.is_file():
        raise RuntimeError("candidate Cargo inputs are incomplete")
    with (ROOT / _CONTRIBUTOR_LOCK).open("rb") as stream:
        contributor = tomllib.load(stream)
    with (ROOT / "dependencies" / "release-lock.toml").open("rb") as stream:
        release = tomllib.load(stream)
    expected_revisions = {
        "gammaloop": str(contributor["gammaloop_candidate"]["revision"]),
        "symbolica": str(contributor["symbolica"]["candidate_revision"]),
        "symbolica-community": str(contributor["symbolica"]["community_revision"]),
        "symjit": str(release["symjit"]["revision"]),
    }
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not set(sources) >= _CANDIDATE_SOURCES:
        raise RuntimeError("candidate installer state has an incomplete source map")
    for name, expected_revision in expected_revisions.items():
        entry = sources.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError("candidate installer source entries must be objects")
        revision = entry.get("revision")
        if revision != expected_revision:
            raise RuntimeError(
                f"candidate source {name} is not at its contributor-lock revision"
            )
    return payload


def _candidate_digest(
    candidate_lock: Path,
    candidate_config: Path,
    installer_state: Path,
) -> str:
    state = _candidate_state(
        candidate_lock,
        candidate_config,
        installer_state,
    )
    digest = hashlib.sha256()
    _digest_item(
        digest,
        "contributor-lock.toml",
        (ROOT / _CONTRIBUTOR_LOCK).read_bytes(),
    )
    _digest_item(digest, "candidate-Cargo.lock", candidate_lock.read_bytes())
    _digest_item(
        digest,
        "candidate-cargo-config.toml",
        _canonical_candidate_config(candidate_config),
    )
    sources = state["sources"]
    for name in sorted(_CANDIDATE_SOURCES):
        entry = sources[name]
        _digest_item(
            digest,
            f"source/{name}",
            json.dumps(
                entry,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8"),
        )
    return digest.hexdigest()[:12]


def _canonical_candidate_config(candidate_config: Path) -> bytes:
    """Return a semantic Cargo patch identity independent of checkout paths."""

    try:
        data = candidate_config.read_bytes()
    except OSError as error:
        raise RuntimeError(f"invalid candidate Cargo config: {error}") from error
    return _canonical_candidate_cargo_config_bytes(data)


def _mark_candidate(
    overlay: Path,
    base_version: str,
    *,
    native_build_inputs_sha256: str,
) -> None:
    if not (ROOT / _CONTRIBUTOR_LOCK).is_file():
        raise RuntimeError("candidate build has no contributor dependency contract")
    check_contributor_lock_consistency(ROOT)
    candidate_lock, candidate_config, installer_state = _candidate_inputs()
    digest = _candidate_digest(
        candidate_lock,
        candidate_config,
        installer_state,
    )
    cargo_version = f"{base_version}-dev.0+candidate.{digest}"
    python_version = cargo_version.replace("-dev.", ".dev")
    shutil.copy2(candidate_lock, overlay / "Cargo.lock")
    overlay_config = overlay / ".cargo" / "config.toml"
    overlay_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_config, overlay_config)
    _rewrite_candidate_dependency_requirements(overlay)
    cargo = overlay / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    updated, count = re.subn(
        rf'(?m)^version = "{re.escape(base_version)}"$',
        f'version = "{cargo_version}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("could not derive candidate version from Cargo.toml")
    cargo.write_text(updated, encoding="utf-8")

    lock = overlay / "Cargo.lock"
    lock_text = lock.read_text(encoding="utf-8")
    for package_name in (
        "rusticol-capi",
        "rusticol-core",
        "rusticol-python",
    ):
        pattern = (
            rf'(?m)(\[\[package\]\]\nname = "{package_name}"\n)'
            rf'version = "{re.escape(base_version)}"'
        )
        lock_text, count = re.subn(
            pattern,
            rf'\g<1>version = "{cargo_version}"',
            lock_text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"could not derive candidate lock entry for {package_name}"
            )
    lock.write_text(lock_text, encoding="utf-8")

    package = overlay / "src" / "pyamplicol"
    package.mkdir(parents=True, exist_ok=True)
    source_revision = _clean_source_revision()
    (package / "_build_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": digest,
                "native_build_inputs_sha256": native_build_inputs_sha256,
                "selftest_fixture_bootstrap": False,
                "source_checkout": str(ROOT.resolve()),
                "source_revision": source_revision,
                "version": python_version,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _mark_release_prepared_model_bootstrap(
    overlay: Path,
    base_version: str,
    *,
    native_build_inputs_sha256: str,
) -> None:
    """Mark a release-version wheel as regeneration-only and non-publishable."""

    if base_version != "0.1.0":
        raise RuntimeError(
            "release prepared-model bootstrap requires package version '0.1.0'"
        )
    package = overlay / "src" / "pyamplicol"
    package.mkdir(parents=True, exist_ok=True)
    (package / "_build_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": None,
                "native_build_inputs_sha256": native_build_inputs_sha256,
                "release_prepared_model_bootstrap": True,
                "selftest_fixture_bootstrap": False,
                "source_checkout": str(ROOT.resolve()),
                "version": base_version,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _stage_release_build_info(overlay: Path, version: str) -> None:
    """Create or retain the minimal provenance marker for a published release."""

    path = overlay / "src" / "pyamplicol" / "_build_info.json"
    if (ROOT / ".git").exists():
        revision = _clean_source_revision()
        if revision is None:
            raise RuntimeError(
                "publishable release artifacts require a clean Git source revision"
            )
        payload: dict[str, object] = {
            "schema_version": 1,
            "publishable": True,
            "selftest_fixture_bootstrap": False,
            "source_revision": revision,
            "version": version,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "release source archive has no retained build provenance"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != _RELEASE_BUILD_INFO_FIELDS
        or payload.get("schema_version") != 1
        or payload.get("publishable") is not True
        or payload.get("selftest_fixture_bootstrap") is not False
        or payload.get("version") != version
        or not isinstance(payload.get("source_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", payload["source_revision"]) is None
    ):
        raise RuntimeError("release source archive has invalid build provenance")


def _clean_source_revision() -> str | None:
    """Return the exact source revision only for a clean Git checkout.

    Candidate wheels remain useful for ordinary dirty-tree development, where
    this field is ``null``. Strict reference capture rejects those wheels and
    therefore cannot accidentally certify uncommitted or ambient source.
    """

    if not (ROOT / ".git").exists():
        return None
    environment = _clean_environment()
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top_level).resolve() != ROOT.resolve():
            return None
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
        ).stdout
        if status:
            return None
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return revision if re.fullmatch(r"[a-f0-9]{40}", revision) else None


def _rewrite_candidate_dependency_requirements(overlay: Path) -> None:
    """Use managed candidate native dependencies only in the build overlay."""

    with (overlay / "dependencies" / "release-lock.toml").open("rb") as stream:
        release = tomllib.load(stream)
    with (ROOT / _CONTRIBUTOR_LOCK).open("rb") as stream:
        contributor = tomllib.load(stream)
    manifest = overlay / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    projections = (
        (
            "symbolica",
            str(release["symbolica"]["rust_version"]),
            str(contributor["symbolica"]["candidate_version"]),
        ),
    )
    for dependency, published, candidate in projections:
        pattern = (
            rf"(?m)^({dependency}\s*=\s*\{{\s*version\s*=\s*)"
            rf'"={re.escape(published)}"'
        )
        text, count = re.subn(pattern, rf'\g<1>"={candidate}"', text, count=1)
        if count != 1:
            raise RuntimeError(
                f"could not project rusticol-core {dependency} requirement "
                f"from {published} to candidate {candidate}"
            )
    manifest.write_text(text, encoding="utf-8")

    python_manifest = overlay / "pyproject.toml"
    python_text = python_manifest.read_text(encoding="utf-8")
    published_python = str(release["symbolica"]["python_version"])
    candidate_python = str(contributor["symbolica"]["candidate_version"])
    python_text, count = re.subn(
        rf'(?m)^(\s*"symbolica==){re.escape(published_python)}(",\s*)$',
        rf"\g<1>{candidate_python}\g<2>",
        python_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(
            "could not project Python Symbolica requirement "
            f"from {published_python} to candidate {candidate_python}"
        )
    python_manifest.write_text(python_text, encoding="utf-8")


def _candidate_inputs() -> tuple[Path, Path, Path]:
    lock = ROOT / "dependencies" / "candidate-Cargo.lock"
    config = ROOT / "dependencies" / "candidate-cargo-config.toml"
    state = ROOT / "dependencies" / "install-state.json"
    missing = [path for path in (lock, config, state) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "candidate build inputs are missing; run 'just dev-install': "
            + ", ".join(str(path) for path in missing)
        )
    return lock, config, state


def _stage_cargo_inputs(
    overlay: Path,
    mode: str,
    *,
    native_build_inputs_sha256: str | None,
) -> None:
    """Apply managed path overrides only to contributor builds."""

    if mode not in {"candidate", "release"}:
        raise RuntimeError(f"unsupported Cargo input mode: {mode}")
    lock = overlay / "Cargo.lock"
    if not lock.is_file():
        raise RuntimeError("build overlay has no canonical Cargo.lock")
    config = overlay / ".cargo" / "config.toml"
    if mode == "release":
        if config.exists():
            raise RuntimeError(
                "release build overlay contains an ambient Cargo patch configuration"
            )
        return
    if native_build_inputs_sha256 is None:
        raise RuntimeError("candidate build has no native source identity")
    _mark_candidate(
        overlay,
        canonical_package_version(overlay),
        native_build_inputs_sha256=native_build_inputs_sha256,
    )
    if not config.is_file():
        raise RuntimeError("candidate build overlay has no Cargo patch configuration")


@contextmanager
def _overlay(
    mode: str,
    *,
    release_prepared_model_bootstrap: bool = False,
    project_release_prepared_models: bool = False,
    temporary_directory: Path | None = None,
    cargo_target_directory: Path | None = None,
) -> Iterator[tuple[Path, Path]]:
    with TemporaryDirectory(
        prefix="pyamplicol-build-",
        dir=temporary_directory,
    ) as temporary:
        root = Path(temporary)
        source = root / "source"
        native_build_inputs_sha256 = (
            _native_build_inputs_digest(
                ROOT,
                normalize_release_cargo_lock=release_prepared_model_bootstrap,
            )
            if mode == "candidate" or release_prepared_model_bootstrap
            else None
        )
        _copy_allowlisted_source(source)
        if mode == "release" and project_release_prepared_models:
            project_release_packaged_prepared_model_store(
                source,
                require_store=os.path.lexists(ROOT / ".git"),
            )
        else:
            discard_release_packaged_prepared_model_store(source)
        _stage_cargo_inputs(
            source,
            mode,
            native_build_inputs_sha256=native_build_inputs_sha256,
        )
        if release_prepared_model_bootstrap:
            if native_build_inputs_sha256 is None:
                raise RuntimeError(
                    "release prepared-model bootstrap has no native source identity"
                )
            _mark_release_prepared_model_bootstrap(
                source,
                canonical_package_version(source),
                native_build_inputs_sha256=native_build_inputs_sha256,
            )
        yield (
            source,
            (
                root / "cargo-target"
                if cargo_target_directory is None
                else cargo_target_directory
            ),
        )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _clean_environment(
    updates: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _INJECTION_ENVIRONMENT_NAMES
        and not name.startswith(_INJECTION_ENVIRONMENT_PREFIXES)
    }
    environment["PATH"] = _build_tool_path(os.environ.get("PATH", ""))
    if updates:
        environment.update(updates)
    return environment


def _macos_native_build_updates() -> dict[str, str]:
    """Select Apple's system compilers for authenticated macOS builds.

    An absolute ``CC`` also gives native build caches such as
    ``gmp-mpfr-sys`` an unambiguous compiler identity.  Without it, that
    crate keys its cache by the unresolved name ``gcc`` and can reuse output
    produced earlier by a package-manager GNU compiler despite the clean
    build PATH.
    """

    if sys.platform != "darwin":
        return {}
    return {
        "CC": "/usr/bin/clang",
        "CXX": "/usr/bin/clang++",
        "MACOSX_DEPLOYMENT_TARGET": "11.0",
    }


def _build_tool_path(inherited: str) -> str:
    """Return a minimal build PATH that cannot select Homebrew/MacPorts tools."""

    interpreter = Path(sys.executable)
    # Keep the isolated build environment so Maturin's console script remains
    # available, but never expose the base interpreter's package-manager bin.
    # A venv Python may resolve into /opt/local or /opt/homebrew, where unrelated
    # compiler wrappers would otherwise leak non-relocatable RPATHs into wheels.
    directories: list[Path] = [interpreter.parent]
    maturin_script = _maturin_console_script()
    inherited_directories = {
        Path(os.path.abspath(directory))
        for directory in inherited.split(os.pathsep)
        if directory
    }
    if maturin_script is not None and maturin_script.parent in inherited_directories:
        directories.append(maturin_script.parent)
    for executable in ("cargo", "rustc"):
        located = shutil.which(executable, path=inherited)
        if located is None:
            raise RuntimeError(f"required build tool is unavailable: {executable}")
        path = Path(located)
        directories.append(path.parent)

    system_directories = (
        [Path(path) for path in ("/usr/bin", "/bin", "/usr/sbin", "/sbin")]
        if os.name == "posix"
        else []
    )
    directories.extend(system_directories)
    tool_search_path = os.pathsep.join(
        [*(str(path) for path in system_directories), inherited]
    )
    for executable in ("git", "cc", "clang", "ar", "ranlib", "nm"):
        located = shutil.which(executable, path=tool_search_path)
        if located is None:
            continue
        path = Path(located)
        directories.append(path.parent)
    if os.name != "posix":  # Keep unsupported Windows source hooks usable.
        directories.extend(Path(path) for path in inherited.split(os.pathsep) if path)

    unique: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        rendered = str(directory)
        if rendered in seen or not directory.is_dir():
            continue
        seen.add(rendered)
        unique.append(rendered)
    return os.pathsep.join(unique)


def _maturin_console_script() -> Path | None:
    """Locate the console script belonging to the imported Maturin package."""

    package = importlib.metadata.distribution("maturin")
    for relative in package.files or ():
        if relative.name not in {"maturin", "maturin.exe"}:
            continue
        candidate = Path(os.path.abspath(package.locate_file(relative)))
        if candidate.is_file():
            return candidate
    return None


def _pinned_rustup_toolchain() -> str:
    """Return the repository-owned Rustup toolchain selection."""

    path = ROOT / "rust-toolchain.toml"
    try:
        toolchain = tomllib.loads(path.read_text(encoding="utf-8"))["toolchain"]
        channel = toolchain["channel"]
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"invalid repository Rust toolchain contract: {path}"
        ) from error
    if not isinstance(channel, str) or not channel.strip():
        raise RuntimeError(f"invalid repository Rust toolchain channel: {path}")
    return channel


def _rust_remap_flags(overlay: Path, target_dir: Path) -> str:
    completed = subprocess.run(
        ["rustc", "--print", "sysroot"],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_environment({"RUSTUP_TOOLCHAIN": _pinned_rustup_toolchain()}),
    )
    sysroot = Path(completed.stdout.strip()).resolve()
    mappings = {
        ROOT.resolve(): "/pyamplicol/checkout",
        overlay.resolve(): "/pyamplicol/source",
        target_dir.parent.resolve(): "/pyamplicol/build",
        sysroot: "/rust/sysroot",
    }
    flags = [
        f"--remap-path-prefix={source}={destination}"
        for source, destination in sorted(
            mappings.items(), key=lambda item: len(str(item[0])), reverse=True
        )
    ]
    return "\x1f".join(flags)


@contextmanager
def _environment(updates: Mapping[str, str]) -> Iterator[None]:
    previous = dict(os.environ)
    isolated = _clean_environment(updates)
    os.environ.clear()
    os.environ.update(isolated)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


@contextmanager
def _delegating() -> Iterator[None]:
    global _delegation_depth
    if _delegation_depth or "PYAMPLICOL_BUILD_OVERLAY" in os.environ:
        raise RuntimeError("recursive PEP 517 backend delegation is not permitted")
    _delegation_depth += 1
    try:
        yield
    finally:
        _delegation_depth -= 1


def _from_overlay(
    operation: Callable[..., _Result],
    *args: Any,
    with_sdk: bool,
    validate_prepared_models: bool = False,
    retain_release_build_info: bool = False,
    release_prepared_model_bootstrap_context: str | None = None,
    **kwargs: Any,
) -> _Result:
    with _delegating():
        mode = _build_mode()
        prepared_model_bootstrap = _prepared_model_bootstrap(mode)
        release_prepared_model_bootstrap = _release_prepared_model_bootstrap(
            mode,
            release_prepared_model_bootstrap_context,
        )
        # A raw environment escape is never sufficient for a PEP 517 build.
        # Only the dedicated local regeneration helper supplies the explicit
        # context accepted by this gate.
        _selftest_fixture_bootstrap(mode)
        _check_dependencies(mode)
        if release_prepared_model_bootstrap:
            overlay_context = _overlay(
                mode,
                release_prepared_model_bootstrap=True,
            )
        elif mode == "release":
            overlay_context = _overlay(
                mode,
                project_release_prepared_models=True,
            )
        else:
            overlay_context = _overlay(mode)
        with overlay_context as (overlay, target_dir):
            if retain_release_build_info and mode == "release":
                _stage_release_build_info(
                    overlay,
                    canonical_package_version(overlay),
                )
            prepared_model_recovery = (
                prepared_model_bootstrap or release_prepared_model_bootstrap
            )
            environment = {
                "CARGO_HOME": str(target_dir.parent / "cargo-home"),
                "CARGO_ENCODED_RUSTFLAGS": _rust_remap_flags(overlay, target_dir),
                "CARGO_TARGET_DIR": str(target_dir),
                "PYAMPLICOL_BUILD_OVERLAY": str(overlay),
                "RUSTUP_TOOLCHAIN": _pinned_rustup_toolchain(),
            }
            build_info_path = overlay / "src" / "pyamplicol" / "_build_info.json"
            if build_info_path.is_file():
                try:
                    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
                    native_digest = build_info.get("native_build_inputs_sha256")
                except (OSError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        "build provenance could not be exported to Rust"
                    ) from error
                if native_digest is not None:
                    if not isinstance(native_digest, str) or not native_digest:
                        raise RuntimeError(
                            "build provenance contains an invalid native identity"
                        )
                    environment["PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256"] = native_digest
                else:
                    environment["PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256"] = (
                        _native_build_inputs_digest(
                            ROOT,
                            normalize_release_cargo_lock=mode == "release",
                        )
                    )
            else:
                environment["PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256"] = (
                    _native_build_inputs_digest(
                        ROOT,
                        normalize_release_cargo_lock=mode == "release",
                    )
                )
            environment.update(_macos_native_build_updates())
            with _environment(environment), _working_directory(overlay):
                if validate_prepared_models and not with_sdk:
                    stage_packaged_prepared_models(overlay, mode)
                if with_sdk:
                    _stage_packaged_examples(overlay)
                    _stage_packaged_profiling_campaign(overlay)
                    _stage_python_stub(overlay)
                    _stage_runtime_resources(overlay)
                    if prepared_model_recovery:
                        _strip_prepared_model_payloads(overlay)
                        _mark_selftest_fixture_bootstrap(
                            overlay,
                            prepared_model_recovery=True,
                        )
                    else:
                        stage_packaged_prepared_models(overlay, mode)
                    sdk = build_sdk(overlay, target_dir)
                    sdk_metadata = json.loads(
                        (sdk / "metadata.json").read_text(encoding="utf-8")
                    )
                    if not prepared_model_recovery:
                        _stage_selftest_fixture(
                            overlay,
                            str(sdk_metadata["target"]),
                        )
                    os.environ["PYAMPLICOL_SDK_STAGING"] = str(sdk)
                return operation(*args, **kwargs)


def build_wheel(
    wheel_directory: str,
    config_settings: Mapping[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    filename = _from_overlay(
        maturin.build_wheel,
        wheel_directory,
        config_settings,
        metadata_directory,
        with_sdk=True,
        retain_release_build_info=True,
    )
    return filename


def build_release_prepared_model_bootstrap_wheel(
    wheel_directory: str,
    *,
    bootstrap_context: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    """Build a release-version wheel usable only to regenerate prepared packs."""

    return _from_overlay(
        maturin.build_wheel,
        wheel_directory,
        config_settings,
        None,
        with_sdk=True,
        release_prepared_model_bootstrap_context=bootstrap_context,
    )


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    if _build_mode() == "candidate":
        raise RuntimeError(
            "candidate builds are wheel-only and cannot produce source distributions"
        )
    return _from_overlay(
        maturin.build_sdist,
        sdist_directory,
        config_settings,
        with_sdk=False,
        validate_prepared_models=True,
        retain_release_build_info=True,
    )


def get_requires_for_build_wheel(
    config_settings: Mapping[str, Any] | None = None,
) -> list[str]:
    return _from_overlay(
        maturin.get_requires_for_build_wheel,
        config_settings,
        with_sdk=False,
    )


def get_requires_for_build_sdist(
    config_settings: Mapping[str, Any] | None = None,
) -> list[str]:
    return _from_overlay(
        maturin.get_requires_for_build_sdist,
        config_settings,
        with_sdk=False,
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    return _from_overlay(
        maturin.prepare_metadata_for_build_wheel,
        metadata_directory,
        config_settings,
        with_sdk=False,
    )
