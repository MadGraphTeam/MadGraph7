# SPDX-License-Identifier: 0BSD
"""Read-only-source PEP 517 backend for pyAmpliCol."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
import uuid
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import maturin  # type: ignore[import-untyped]
from distribution_sbom import build_distribution_sbom
from sdk import build_sdk

ROOT = Path(__file__).resolve().parents[1]
_DISTRIBUTION_LOCK_MEMBER = "pyamplicol/assets/release/release-lock.toml"
_PYTHON_RUNTIME_LOCK_MEMBER = "pyamplicol/assets/release/python-runtime-lock.toml"
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
    "rust",
    "rust-toolchain.toml",
    "schemas",
    "src",
    "tests",
    "tools/developer",
    "tools/release",
    "tools/typing",
)
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
    Path("docs/.result_outputs"),
    Path("docs/archive"),
    Path("outputs"),
    Path("src/pyamplicol/_sdk/fortran"),
    Path("src/pyamplicol/_sdk/include"),
    Path("src/pyamplicol/_sdk/lib"),
    Path("src/pyamplicol/_sdk/sboms"),
)
_EXCLUDED_PATHS = {
    Path(".cargo/config.toml"),
    Path("dependencies/candidate-Cargo.lock"),
    Path("dependencies/candidate-cargo-config.toml"),
    Path("dependencies/install-state.json"),
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
_CANDIDATE_SOURCES = {
    "gammaloop",
    "symbolica",
    "symbolica-community",
    "symjit",
    "ufo-model-loader",
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
        "--offline",
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
    return relative.parent == Path("docs") and relative.name.endswith(
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
    """Place wheel-owned metadata below the pyamplicol package namespace."""

    package_assets = overlay / "src" / "pyamplicol" / "assets"
    sources = {
        overlay / "dependencies" / "release-lock.toml": (
            package_assets / "release" / "release-lock.toml"
        ),
        overlay / "dependencies" / "python-runtime-lock.toml": (
            package_assets / "release" / "python-runtime-lock.toml"
        ),
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


def _canonicalize_sbom_references(value: object) -> dict[str, str]:
    replacements: dict[str, str] = {}

    def visit(item: object) -> None:
        if isinstance(item, dict):
            reference = item.get("bom-ref")
            purl = item.get("purl")
            if (
                isinstance(reference, str)
                and reference.startswith("path+file:")
                and isinstance(purl, str)
                and purl.startswith("pkg:cargo/")
            ):
                base, separator, fragment = purl.partition("#")
                canonical = base.split("?", 1)[0]
                if separator:
                    canonical = f"{canonical}#{fragment}"
                replacements[reference] = canonical
                item["bom-ref"] = canonical
                item["purl"] = canonical
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return replacements


def _replace_sbom_references(value: object, replacements: Mapping[str, str]) -> object:
    if isinstance(value, dict):
        return {
            key: _replace_sbom_references(child, replacements)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_sbom_references(child, replacements) for child in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _normalize_cyclonedx(data: bytes) -> bytes:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Maturin generated an invalid CycloneDX SBOM") from error
    if not isinstance(payload, dict) or payload.get("bomFormat") != "CycloneDX":
        raise RuntimeError("Maturin generated an unsupported SBOM")
    replacements = _canonicalize_sbom_references(payload)
    payload = _replace_sbom_references(payload, replacements)
    assert isinstance(payload, dict)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Maturin SBOM has no metadata object")
    metadata["timestamp"] = "1970-01-01T00:00:00Z"
    payload["serialNumber"] = "urn:uuid:00000000-0000-4000-8000-000000000000"
    identity = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    identifier = bytearray(hashlib.sha256(identity).digest()[:16])
    identifier[6] = (identifier[6] & 0x0F) | 0x40
    identifier[8] = (identifier[8] & 0x3F) | 0x80
    payload["serialNumber"] = f"urn:uuid:{uuid.UUID(bytes=bytes(identifier))}"
    normalized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    forbidden = (b"file://", b"path+file:", b"/Users/", b"/private/var/")
    remaining = [marker for marker in forbidden if marker in normalized]
    if remaining:
        raise RuntimeError(
            "normalized CycloneDX SBOM retains local references: "
            + ", ".join(os.fsdecode(marker) for marker in remaining)
        )
    return normalized


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _normalize_built_wheel(path: Path) -> None:
    """Build the distribution SBOM and refresh wheel RECORD."""

    if not path.is_file() or path.suffix != ".whl":
        raise RuntimeError(f"Maturin did not produce the expected wheel: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        members = {info.filename: archive.read(info) for info in infos}
    sboms = [
        name
        for name in members
        if ".dist-info/sboms/" in name and name.endswith(".cyclonedx.json")
    ]
    if len(sboms) != 1:
        raise RuntimeError(
            f"Maturin wheel must contain one CycloneDX SBOM, found {len(sboms)}"
        )
    metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise RuntimeError("Maturin wheel must contain one METADATA file")
    if _DISTRIBUTION_LOCK_MEMBER not in members:
        raise RuntimeError(
            "Maturin wheel is missing the packaged dependency release lock"
        )
    metadata = BytesParser(policy=policy.default).parsebytes(members[metadata_names[0]])
    distribution = str(metadata.get("Name", ""))
    version = str(metadata.get("Version", ""))
    if not distribution or not version:
        raise RuntimeError("Maturin wheel METADATA has no distribution identity")
    mode = "candidate" if ".dev0+candidate." in version else "release"
    members[sboms[0]] = build_distribution_sbom(
        members[sboms[0]],
        members[_DISTRIBUTION_LOCK_MEMBER],
        members.get(_PYTHON_RUNTIME_LOCK_MEMBER),
        distribution_name=distribution,
        distribution_version=version,
        runtime_requirements=tuple(metadata.get_all("Requires-Dist", [])),
        mode=mode,
    )
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise RuntimeError("Maturin wheel must contain one RECORD")
    record_name = records[0]
    rows = [
        [name, _record_hash(data), str(len(data))]
        for name, data in sorted(members.items())
        if name != record_name
    ]
    rows.append([record_name, "", ""])
    record = io.StringIO(newline="")
    csv.writer(record, lineterminator="\n").writerows(rows)
    members[record_name] = record.getvalue().encode("utf-8")
    temporary = path.with_suffix(f"{path.suffix}.normalizing")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info in infos:
            archive.writestr(info, members[info.filename])
    os.replace(temporary, path)


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
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid self-test artifact manifest: {path}") from error
    content = dict(manifest)
    content.pop("artifact_id", None)
    canonical = (
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    manifest["artifact_id"] = hashlib.sha256(canonical).hexdigest()
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_item(digest: Any, name: str, data: bytes) -> None:
    encoded_name = name.encode("utf-8")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _candidate_state(
    overlay: Path,
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
    expected_digests = {
        "release_lock_sha256": _sha256(overlay / "dependencies" / "release-lock.toml"),
        "python_runtime_lock_sha256": _sha256(
            overlay / "dependencies" / "python-runtime-lock.toml"
        ),
        "candidate_lock_sha256": _sha256(candidate_lock),
        "cargo_config_sha256": _sha256(candidate_config),
    }
    for field, expected in expected_digests.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"candidate installer state {field} does not match its build input"
            )
    if payload.get("publishable") is not False:
        raise RuntimeError("candidate installer state must be non-publishable")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not set(sources) >= _CANDIDATE_SOURCES:
        raise RuntimeError("candidate installer state has an incomplete source map")
    for name, entry in sources.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise RuntimeError("candidate installer source entries must be objects")
        revision = entry.get("revision")
        worktree = entry.get("worktree_sha256")
        if not isinstance(revision, str) or not revision:
            raise RuntimeError(f"candidate source {name} has no revision")
        if (
            not isinstance(worktree, str)
            or re.fullmatch(r"[0-9a-f]{64}", worktree) is None
        ):
            raise RuntimeError(f"candidate source {name} has no valid worktree digest")
    return payload


def _candidate_digest(
    overlay: Path,
    candidate_lock: Path,
    candidate_config: Path,
    installer_state: Path,
) -> str:
    state = _candidate_state(
        overlay,
        candidate_lock,
        candidate_config,
        installer_state,
    )
    digest = hashlib.sha256()
    for path in sorted(item for item in overlay.rglob("*") if item.is_file()):
        _digest_item(
            digest,
            f"source/{path.relative_to(overlay).as_posix()}",
            path.read_bytes(),
        )
    lock = overlay / "dependencies" / "release-lock.toml"
    _digest_item(digest, "release-lock.toml", lock.read_bytes())
    runtime_lock = overlay / "dependencies" / "python-runtime-lock.toml"
    _digest_item(digest, "python-runtime-lock.toml", runtime_lock.read_bytes())
    _digest_item(digest, "candidate-Cargo.lock", candidate_lock.read_bytes())
    _digest_item(digest, "candidate-cargo-config.toml", candidate_config.read_bytes())
    patches = overlay / "dependencies" / "patches"
    if patches.is_dir():
        for path in sorted(item for item in patches.rglob("*") if item.is_file()):
            _digest_item(
                digest,
                path.relative_to(overlay).as_posix(),
                path.read_bytes(),
            )
    sources = state["sources"]
    for name in sorted(sources):
        entry = sources[name]
        _digest_item(
            digest,
            f"source/{name}",
            f"{entry['revision']}\0{entry['worktree_sha256']}".encode(),
        )
    return digest.hexdigest()[:12]


def _mark_candidate(overlay: Path) -> None:
    candidate_lock, candidate_config, installer_state = _candidate_inputs()
    digest = _candidate_digest(
        overlay,
        candidate_lock,
        candidate_config,
        installer_state,
    )
    cargo_version = f"0.1.0-dev.0+candidate.{digest}"
    shutil.copy2(candidate_lock, overlay / "Cargo.lock")
    overlay_config = overlay / ".cargo" / "config.toml"
    overlay_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_config, overlay_config)
    _rewrite_candidate_symjit_requirement(overlay)
    cargo = overlay / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^version = "0\.1\.0"$',
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
            r'version = "0\.1\.0"'
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
                "source_revision": source_revision,
                "version": f"0.1.0.dev0+candidate.{digest}",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _rewrite_candidate_symjit_requirement(overlay: Path) -> None:
    """Use the managed candidate SymJIT only inside the isolated overlay."""

    with (overlay / "dependencies" / "release-lock.toml").open("rb") as stream:
        lock = tomllib.load(stream)
    published = str(lock["symbolica"]["published_symjit_version"])
    candidate = str(lock["symjit"]["candidate_version"])
    manifest = overlay / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    pattern = rf'(?m)^(symjit\s*=\s*\{{\s*version\s*=\s*)"={re.escape(published)}"'
    updated, count = re.subn(pattern, rf'\g<1>"={candidate}"', text, count=1)
    if count != 1:
        raise RuntimeError(
            "could not project rusticol-core from the published SymJIT "
            f"requirement {published} to candidate {candidate}"
        )
    manifest.write_text(updated, encoding="utf-8")


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


def _stage_cargo_inputs(overlay: Path, mode: str) -> None:
    """Keep release and contributor Cargo resolution physically separate."""

    if mode not in {"candidate", "release"}:
        raise RuntimeError(f"unsupported Cargo input mode: {mode}")
    lock = overlay / "Cargo.lock"
    if not lock.is_file():
        raise RuntimeError("build overlay has no canonical Cargo.lock")
    config = overlay / ".cargo" / "config.toml"
    if mode == "release":
        if config.exists():
            raise RuntimeError(
                "release build overlay contains a local Cargo patch configuration"
            )
        canonical = ROOT / "Cargo.lock"
        if lock.read_bytes() != canonical.read_bytes():
            raise RuntimeError("release build overlay changed canonical Cargo.lock")
        return
    _mark_candidate(overlay)
    if not config.is_file():
        raise RuntimeError("candidate build overlay has no Cargo patch configuration")


@contextmanager
def _overlay(mode: str) -> Iterator[tuple[Path, Path]]:
    with TemporaryDirectory(prefix="pyamplicol-build-") as temporary:
        root = Path(temporary)
        source = root / "source"
        _copy_allowlisted_source(source)
        _stage_cargo_inputs(source, mode)
        yield source, root / "cargo-target"


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


def _build_tool_path(inherited: str) -> str:
    """Return a minimal build PATH that cannot select Homebrew/MacPorts tools."""

    interpreter = Path(sys.executable)
    directories: list[Path] = [interpreter.parent, interpreter.resolve().parent]
    for executable in ("cargo", "rustc"):
        located = shutil.which(executable, path=inherited)
        if located is None:
            raise RuntimeError(f"required Rust build tool is unavailable: {executable}")
        path = Path(located)
        directories.extend((path.parent, path.resolve().parent))
    if os.name == "posix":
        directories.extend(
            Path(path) for path in ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
        )
    else:  # Windows is not a 0.1.0 release target, but keep source hooks usable.
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


def _rust_remap_flags(overlay: Path, target_dir: Path) -> str:
    completed = subprocess.run(
        ["rustc", "--print", "sysroot"],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_environment(),
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
    **kwargs: Any,
) -> _Result:
    with _delegating():
        mode = _build_mode()
        _check_dependencies(mode)
        with _overlay(mode) as (overlay, target_dir):
            environment = {
                "CARGO_HOME": str(target_dir.parent / "cargo-home"),
                "CARGO_ENCODED_RUSTFLAGS": _rust_remap_flags(overlay, target_dir),
                "CARGO_TARGET_DIR": str(target_dir),
                "PYAMPLICOL_BUILD_OVERLAY": str(overlay),
            }
            if sys.platform == "darwin":
                environment["MACOSX_DEPLOYMENT_TARGET"] = "11.0"
            with _environment(environment), _working_directory(overlay):
                if with_sdk:
                    _stage_packaged_examples(overlay)
                    _stage_python_stub(overlay)
                    _stage_runtime_resources(overlay)
                    sdk = build_sdk(overlay, target_dir)
                    sdk_metadata = json.loads(
                        (sdk / "metadata.json").read_text(encoding="utf-8")
                    )
                    _stage_selftest_fixture(overlay, str(sdk_metadata["target"]))
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
    )
    _normalize_built_wheel(Path(wheel_directory) / filename)
    return filename


def build_sdist(
    sdist_directory: str,
    config_settings: Mapping[str, Any] | None = None,
) -> str:
    return _from_overlay(
        maturin.build_sdist,
        sdist_directory,
        config_settings,
        with_sdk=False,
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
