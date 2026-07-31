#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Create the isolated pinned contributor environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / "dependencies"
PYPROJECT = ROOT / "pyproject.toml"
RELEASE_LOCK = DEPENDENCIES / "release-lock.toml"
CONTRIBUTOR_LOCK = DEPENDENCIES / "contributor-lock.toml"
# Retained as the public constant used by older contributor-side callers.
LOCK = RELEASE_LOCK
PYTHON_LOCK = DEPENDENCIES / "python-runtime-lock.toml"
CHECKOUTS = DEPENDENCIES / "checkouts"
WHEELHOUSE = DEPENDENCIES / "wheelhouse"
VENV = ROOT / ".venv"
STATE = DEPENDENCIES / "install-state.json"
CANDIDATE_LOCK = DEPENDENCIES / "candidate-Cargo.lock"
CARGO_CONFIG = DEPENDENCIES / "candidate-cargo-config.toml"
ARTIFACTS = ROOT / ".artifacts" / "candidate"
TRASH = ROOT / ".trash"

sys.path.insert(0, str(ROOT / "build_backend"))
from python_lock import load_python_runtime_lock  # noqa: E402

_WORKSPACE_CRATES = frozenset(
    {
        "rusticol-capi",
        "rusticol-core",
        "rusticol-python",
    }
)
_CANDIDATE_PATH_CRATES = frozenset(
    {
        "graphica",
        "numerica",
        "symbolica",
        "symjit",
    }
)
_CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class SetupError(RuntimeError):
    """Contributor setup could not be completed safely."""


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    revision: str
    branch: str | None = None

    @property
    def path(self) -> Path:
        return CHECKOUTS / self.key


class Runner:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        capture: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        rendered = [str(item) for item in command]
        suffix = f"  # cwd={cwd}" if cwd else ""
        print(f"$ {shlex.join(rendered)}{suffix}")
        if self.dry_run:
            return subprocess.CompletedProcess(rendered, 0, "", "")
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=capture,
        )
        if check and completed.returncode != 0:
            if capture:
                print(completed.stdout, end="")
                print(completed.stderr, end="", file=sys.stderr)
            raise SetupError(
                f"command exited with {completed.returncode}: {shlex.join(rendered)}"
            )
        return completed


def _load_lock(path: Path, description: str) -> dict[str, Any]:
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    if payload.get("schema_version") != 1:
        raise SetupError(f"unsupported {description} schema")
    return payload


def _release_lock() -> dict[str, Any]:
    return _load_lock(RELEASE_LOCK, "dependency release-lock")


def _contributor_lock() -> dict[str, Any]:
    return _load_lock(CONTRIBUTOR_LOCK, "dependency contributor-lock")


def _lock() -> dict[str, Any]:
    """Return the contributor setup view without polluting release metadata."""

    release = _release_lock()
    contributor = _contributor_lock()
    payload = dict(release)
    for key, value in contributor.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            merged = dict(payload[key])
            merged.update(value)
            payload[key] = merged
        else:
            payload[key] = value
    return payload


def _sources(payload: dict[str, Any], *, with_legacy: bool) -> tuple[Source, ...]:
    symbolica = payload["symbolica"]
    symjit = payload["symjit"]
    gammaloop = payload["gammaloop_candidate"]
    sources = [
        Source(
            "symjit",
            str(symjit["repository"]),
            str(symjit["revision"]),
        ),
        Source(
            "symbolica",
            str(symbolica["source_url"]),
            str(symbolica["candidate_revision"]),
        ),
        Source(
            "symbolica-community",
            str(symbolica["community_url"]),
            str(symbolica["community_revision"]),
        ),
        Source(
            "gammaloop",
            str(gammaloop["source_url"]),
            str(gammaloop["revision"]),
        ),
        Source(
            "ratatui-ffi",
            str(payload["ratatui"]["ffi_repository"]),
            str(payload["ratatui"]["ffi_revision"]),
        ),
    ]
    if with_legacy:
        legacy = payload["legacy_amplicol"]
        sources.append(
            Source(
                "legacy-amplicol",
                str(legacy["source_url"]),
                str(legacy["revision"]),
                str(legacy["branch"]),
            )
        )
    return tuple(sources)


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["VIRTUAL_ENV"] = str(VENV)
    binary = VENV / ("Scripts" if os.name == "nt" else "bin")
    environment["PATH"] = str(binary) + os.pathsep + environment.get("PATH", "")
    return environment


def _require_tools() -> None:
    missing = [name for name in ("cargo", "git", "rustc") if shutil.which(name) is None]
    if missing:
        raise SetupError("missing contributor tools: " + ", ".join(sorted(missing)))


def _managed_trash_destination(name: str, *, workspace_root: Path) -> Path:
    """Return one new recoverable destination inside workspace-local trash."""

    relative = Path(name)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name in {"", ".", ".."}
    ):
        raise SetupError(f"invalid managed dependency trash entry: {name!r}")
    if workspace_root.is_symlink() or TRASH.is_symlink():
        raise SetupError(
            "managed dependency workspace and trash must not be symbolic links: "
            f"{workspace_root}, {TRASH}"
        )
    resolved_workspace = workspace_root.resolve(strict=False)
    resolved_trash = TRASH.resolve(strict=False)
    try:
        resolved_trash.relative_to(resolved_workspace)
    except ValueError as error:
        raise SetupError(
            f"managed dependency trash resolves outside the workspace: {TRASH}"
        ) from error
    destination = TRASH / relative
    try:
        destination.resolve(strict=False).relative_to(resolved_trash)
    except ValueError as error:
        raise SetupError(
            f"managed dependency trash entry escapes its root: {destination}"
        ) from error
    if destination.exists() or destination.is_symlink():
        raise SetupError(
            f"managed dependency trash destination already exists: {destination}"
        )
    return destination


def _archive_managed_state(runner: Runner) -> None:
    managed = (
        VENV,
        CHECKOUTS,
        WHEELHOUSE,
        STATE,
        CANDIDATE_LOCK,
        CARGO_CONFIG,
        ARTIFACTS,
    )
    present = [path for path in managed if path.exists() or path.is_symlink()]
    if not present:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = _managed_trash_destination(
        f"dependency-reset-{stamp}",
        workspace_root=ROOT,
    )
    for path in present:
        relative = path.relative_to(ROOT)
        target = destination / relative
        print(f"$ mv {shlex.quote(str(path))} {shlex.quote(str(target))}")
        if runner.dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def _git_head(runner: Runner, path: Path) -> str:
    completed = runner.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture=True,
    )
    return completed.stdout.strip()


def _managed_checkout(name: str) -> Path:
    """Return one exact managed checkout child that cannot escape by symlink."""

    relative = Path(name)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name in {"", ".", ".."}
    ):
        raise SetupError(f"invalid managed dependency checkout name: {name!r}")
    destination = CHECKOUTS / relative
    workspace_anchor = CHECKOUTS.parent.parent
    try:
        for path in (
            workspace_anchor,
            CHECKOUTS.parent,
            CHECKOUTS,
            destination,
        ):
            if path.is_symlink():
                raise SetupError(
                    "managed dependency checkout path must not be a symbolic link: "
                    f"{path}"
                )
        workspace_root = workspace_anchor.resolve(strict=False)
        checkout_root = CHECKOUTS.resolve(strict=False)
        resolved_destination = destination.resolve(strict=False)
    except OSError as error:
        raise SetupError(
            f"could not validate managed checkout path {destination}: {error}"
        ) from error
    try:
        checkout_root.relative_to(workspace_root)
        resolved_relative = resolved_destination.relative_to(checkout_root)
    except ValueError as error:
        raise SetupError(
            "managed dependency checkout resolves outside the workspace root: "
            f"{destination}"
        ) from error
    if resolved_relative != relative:
        raise SetupError(
            "managed dependency path does not resolve to the expected checkout: "
            f"{destination}"
        )
    return destination


def _managed_symjit_checkout() -> Path:
    """Return the validated managed SymJIT checkout."""

    return _managed_checkout("symjit")


def _checkout(runner: Runner, source: Source, *, update: bool) -> None:
    destination = _managed_checkout(source.key)
    if destination.exists() and not (destination / ".git").exists():
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archived = _managed_trash_destination(
            f"superseded-{source.key}-{stamp}",
            workspace_root=ROOT,
        )
        print(
            f"$ mv {shlex.quote(str(destination))} "
            f"{shlex.quote(str(archived))}"
        )
        if not runner.dry_run:
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(archived))
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone_command = [
            "git",
            "clone",
            "--filter=blob:none",
        ]
        if source.branch is not None:
            clone_command.extend(["--branch", source.branch, "--single-branch"])
        clone_command.extend(["--no-checkout", source.url, str(destination)])
        runner.run(clone_command)
        runner.run(
            ["git", "checkout", "--detach", source.revision],
            cwd=destination,
        )
        return
    if runner.dry_run:
        print(f"# verify {source.key} at {source.revision}")
        return
    head = _git_head(runner, destination)
    if head == source.revision:
        return
    if not update:
        raise SetupError(
            f"{destination} is at {head}, expected {source.revision}; "
            "rerun with --update or --reset"
        )
    fetch_ref = source.branch or source.revision
    runner.run(["git", "fetch", "origin", fetch_ref], cwd=destination)
    runner.run(
        ["git", "checkout", "--detach", source.revision],
        cwd=destination,
    )


def _replace_section(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(?ms)^\[{re.escape(name)}\]\n.*?(?=^\[[^\n]+\]\n|\Z)")
    replacement = f"[{name}]\n{body.strip()}\n\n"
    if not pattern.search(text):
        return text.rstrip() + "\n\n" + replacement
    return pattern.sub(replacement, text, count=1)


def _require_symjit_rlib_manifest(
    path: Path,
    *,
    expected_version: str | None = None,
) -> None:
    """Require the immutable upstream manifest to expose an rlib library."""

    try:
        with path.open("rb") as stream:
            manifest = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SetupError(
            f"could not validate managed SymJIT manifest: {error}"
        ) from error
    package = manifest.get("package")
    if not isinstance(package, dict) or package.get("name") != "symjit":
        raise SetupError("managed SymJIT source has the wrong package identity")
    if (
        expected_version is not None
        and str(package.get("version")) != expected_version
    ):
        raise SetupError(
            "managed SymJIT source has the wrong package version: "
            f"expected {expected_version}, got {package.get('version')!r}"
        )
    library = manifest.get("lib")
    if not isinstance(library, dict) or library.get("crate-type") != ["rlib"]:
        raise SetupError("managed SymJIT source must expose an rlib-only library")


def _configure_source_manifests(runner: Runner) -> None:
    symjit = _managed_symjit_checkout()
    symbolica = _managed_checkout("symbolica")
    community = _managed_checkout("symbolica-community")
    gammaloop = _managed_checkout("gammaloop")
    if runner.dry_run:
        print(
            "# configure candidate Cargo manifests to pinned local paths"
        )
        return
    _require_symjit_rlib_manifest(
        symjit / "Cargo.toml",
        expected_version=str(_release_lock()["symjit"]["version"]),
    )

    symbolica_cargo = symbolica / "Cargo.toml"
    text = symbolica_cargo.read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^symjit\s*=.*$",
        'symjit = { path = "../symjit" }',
        text,
        count=1,
    )
    if count != 1:
        raise SetupError("could not point Symbolica at managed SymJIT")
    symbolica_cargo.write_text(text, encoding="utf-8")

    dependencies = (
        """
example_extension = { path = "example_extension" }
idenso = { path = "../gammaloop/crates/idenso", features = ["bincode", "python"] }
spynso3 = { path = "../gammaloop/crates/spynso3" }
symbolica = { path = "../symbolica", features = ["python_export"] }
symbolica-integrate = { version = "1.0", features = ["steps"] }
pyo3 = { version = "0.28", features = ["abi3"] }
"""
        'pyo3-stub-gen = { version = "0.17", optional = true, '
        'default-features = false, features = ["numpy"] }\n'
        """
mimalloc = { version = "0.1", features = ["local_dynamic_tls"] }
vakint = { path = "../gammaloop/crates/vakint", features = [
    "symbolica_community_module",
] }
"""
    )
    patches = """
graphica = { path = "../symbolica/lib/graphica" }
idenso = { path = "../gammaloop/crates/idenso" }
linnet = { path = "../gammaloop/crates/linnet" }
numerica = { path = "../symbolica/lib/numerica" }
spenso = { path = "../gammaloop/crates/spenso" }
spenso-hep-lib = { path = "../gammaloop/crates/spenso-hep-lib" }
spenso-macros = { path = "../gammaloop/crates/spenso-macros" }
spynso3 = { path = "../gammaloop/crates/spynso3" }
symbolica = { path = "../symbolica" }
symjit = { path = "../symjit" }
"""
    community_cargo = community / "Cargo.toml"
    text = community_cargo.read_text(encoding="utf-8")
    if not re.search(r"(?m)^\[workspace\]\s*$", text):
        # Managed checkouts live below pyAmpliCol's workspace directory but are
        # independent build inputs.  An explicit empty workspace prevents Cargo
        # from adopting this package into the nearest ancestor workspace.
        text = "[workspace]\n\n" + text
    text = _replace_section(text, "dependencies", dependencies)
    text = _replace_section(text, "patch.crates-io", patches)
    text = re.sub(
        r"(?m)^numerica\s*=\s*\{[^\n]*\}\s*$",
        'numerica = { path = "../symbolica/lib/numerica" }',
        text,
        count=1,
    )
    community_cargo.write_text(text.rstrip() + "\n", encoding="utf-8")

    example = community / "example_extension" / "Cargo.toml"
    text = example.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^symbolica\s*=\s*\{[^\n]*\}\s*$",
        'symbolica = { path = "../../symbolica", features = ["python_export"] }',
        text,
        count=1,
    )
    example.write_text(text.rstrip() + "\n", encoding="utf-8")

    gammaloop_cargo = gammaloop / "Cargo.toml"
    text = gammaloop_cargo.read_text(encoding="utf-8")
    text = re.sub(
        r"(?m)^symbolica\s*=\s*\{[^\n]*\}\s*$",
        (
            'symbolica = { path = "../symbolica", '
            'default-features = false, features = ["gmp"] }'
        ),
        text,
        count=1,
    )
    text = _replace_section(
        text,
        "patch.crates-io",
        """
graphica = { path = "../symbolica/lib/graphica" }
numerica = { path = "../symbolica/lib/numerica" }
symbolica = { path = "../symbolica" }
""",
    )
    gammaloop_cargo.write_text(text.rstrip() + "\n", encoding="utf-8")

    workspace_hack = gammaloop / "crates" / "gammaloop-workspace-hack" / "Cargo.toml"
    text = workspace_hack.read_text(encoding="utf-8")
    text, symbolica_count = re.subn(
        r'(?m)^symbolica\s*=\s*\{\s*git\s*=\s*"[^"]+",\s*branch\s*=\s*"main",',
        'symbolica = { path = "../../../symbolica",',
        text,
    )
    text, numerica_count = re.subn(
        r'(?m)^numerica\s*=\s*\{\s*git\s*=\s*"[^"]+",\s*branch\s*=\s*"main",',
        'numerica = { path = "../../../symbolica/lib/numerica",',
        text,
    )
    localized_symbolica = text.count('symbolica = { path = "../../../symbolica",')
    localized_numerica = text.count(
        'numerica = { path = "../../../symbolica/lib/numerica",'
    )
    if (
        symbolica_count not in {0, 2}
        or numerica_count not in {0, 2}
        or localized_symbolica != 2
        or localized_numerica != 2
    ):
        raise SetupError("could not localize GammaLoop workspace-hack Symbolica inputs")
    workspace_hack.write_text(text, encoding="utf-8")


def _configure_sources(runner: Runner) -> None:
    _configure_source_manifests(runner)
    community = _managed_checkout("symbolica-community")
    if runner.dry_run:
        print("# restore the upstream symbolica-community Cargo.lock")
    else:
        upstream_lock = runner.run(
            ["git", "show", "HEAD:Cargo.lock"],
            cwd=community,
            capture=True,
        ).stdout
        (community / "Cargo.lock").write_text(upstream_lock, encoding="utf-8")
    # Resolve only the Git-to-path source substitutions from the exact upstream
    # lock.  This preserves every unrelated version chosen by the release that
    # the contributor build is intended to simulate.
    runner.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=community,
        capture=True,
    )
    runner.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=community,
        capture=True,
    )


def _write_cargo_config(runner: Runner) -> None:
    symjit = _managed_symjit_checkout()
    symbolica = _managed_checkout("symbolica")
    entries = {
        "graphica": symbolica / "lib" / "graphica",
        "numerica": symbolica / "lib" / "numerica",
        "symbolica": symbolica,
        "symjit": symjit,
    }
    text = ["# Generated by dependencies/install_dependencies.py", "[patch.crates-io]"]
    text.extend(
        f"{name} = {{ path = {json.dumps(str(path.resolve()))} }}"
        for name, path in entries.items()
    )
    print(f"# write {CARGO_CONFIG}")
    if runner.dry_run:
        return
    CARGO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CARGO_CONFIG.write_text("\n".join(text) + "\n", encoding="utf-8")


def _cargo_lock_packages(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SetupError(f"invalid Cargo lock file {path}: {error}") from error
    packages = payload.get("package")
    if payload.get("version") != 4 or not isinstance(packages, list):
        raise SetupError(f"Cargo lock file {path} must use format version 4")
    return packages


def _validate_release_cargo_lock(path: Path) -> None:
    """Require registry crates plus the exact immutable SymJIT Git revision."""

    symjit = _release_lock()["symjit"]
    symjit_source = (
        f"git+{symjit['repository']}?rev={symjit['revision']}#{symjit['revision']}"
    )
    invalid: list[str] = []
    for package in _cargo_lock_packages(path):
        name = str(package.get("name", "<unnamed>"))
        source = package.get("source")
        checksum = package.get("checksum")
        if name in _WORKSPACE_CRATES and source is None and checksum is None:
            continue
        if (
            name == "symjit"
            and str(package.get("version")) == str(symjit["version"])
            and source == symjit_source
            and checksum is None
        ):
            continue
        if source != _CRATES_IO_SOURCE:
            invalid.append(f"{name} has an unexpected source {source!r}")
            continue
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            invalid.append(f"{name} has no valid registry checksum")
    if invalid:
        raise SetupError(
            "canonical Cargo.lock is not release-resolved; regenerate it "
            "without the candidate Cargo patch configuration:\n  "
            + "\n  ".join(invalid)
        )


def _validate_candidate_cargo_lock(path: Path) -> None:
    """Require only the managed patch crates to use local path resolution."""

    packages = _cargo_lock_packages(path)
    local_names = {
        str(package.get("name"))
        for package in packages
        if package.get("source") is None and package.get("checksum") is None
    }
    expected_local = _WORKSPACE_CRATES | _CANDIDATE_PATH_CRATES
    missing = sorted(expected_local - local_names)
    unexpected = sorted(local_names - expected_local)
    invalid_registry: list[str] = []
    for package in packages:
        name = str(package.get("name", "<unnamed>"))
        if name in expected_local:
            continue
        source = package.get("source")
        checksum = package.get("checksum")
        if (
            source != _CRATES_IO_SOURCE
            or not isinstance(checksum, str)
            or _SHA256_PATTERN.fullmatch(checksum) is None
        ):
            invalid_registry.append(name)
    if missing or unexpected or invalid_registry:
        details = []
        if missing:
            details.append("missing local crates: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected local crates: " + ", ".join(unexpected))
        if invalid_registry:
            details.append(
                "invalid registry crates: " + ", ".join(sorted(invalid_registry))
            )
        raise SetupError("invalid candidate Cargo lock: " + "; ".join(details))


def _write_candidate_lock(runner: Runner) -> None:
    if runner.dry_run:
        print(f"# resolve {CANDIDATE_LOCK} from canonical Cargo.lock")
        return
    release_lock = ROOT / "Cargo.lock"
    release_lock_bytes = release_lock.read_bytes()
    _validate_release_cargo_lock(release_lock)
    with tempfile.TemporaryDirectory(prefix="pyamplicol-candidate-lock-") as raw:
        temporary = Path(raw)
        shutil.copy2(ROOT / "Cargo.toml", temporary / "Cargo.toml")
        shutil.copy2(release_lock, temporary / "Cargo.lock")
        shutil.copytree(ROOT / "rust", temporary / "rust")
        runner.run(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=temporary,
            capture=True,
        )
        _rewrite_candidate_requirements(temporary)
        config = temporary / ".cargo" / "config.toml"
        config.parent.mkdir(parents=True)
        shutil.copy2(CARGO_CONFIG, config)
        # Resolving from the release lock preserves every unrelated registry
        # version while replacing only the explicitly patched candidate crates.
        runner.run(
            ["cargo", "metadata", "--format-version", "1"],
            cwd=temporary,
            capture=True,
        )
        _validate_candidate_cargo_lock(temporary / "Cargo.lock")
        runner.run(
            ["cargo", "metadata", "--locked", "--format-version", "1"],
            cwd=temporary,
            capture=True,
        )
        shutil.copy2(temporary / "Cargo.lock", CANDIDATE_LOCK)
    if release_lock.read_bytes() != release_lock_bytes:
        raise SetupError("candidate lock generation modified canonical Cargo.lock")


def _rewrite_candidate_requirements(root: Path) -> None:
    """Project published release pins onto the pinned candidate sources."""

    lock = _lock()
    manifest = root / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    text = manifest.read_text(encoding="utf-8")
    projections = (
        (
            "symbolica",
            str(lock["symbolica"]["rust_version"]),
            str(lock["symbolica"]["candidate_version"]),
        ),
    )
    for dependency, published, candidate in projections:
        pattern = (
            rf"(?m)^({dependency}\s*=\s*\{{\s*version\s*=\s*)"
            rf'"={re.escape(published)}"'
        )
        text, count = re.subn(pattern, rf'\g<1>"={candidate}"', text, count=1)
        if count != 1:
            raise SetupError(
                f"could not project rusticol-core {dependency} requirement "
                f"from {published} to candidate {candidate}"
            )
    manifest.write_text(text, encoding="utf-8")


def _runtime_requirements_text() -> str:
    runtime_lock = load_python_runtime_lock(PYTHON_LOCK)
    excluded = {"symbolica"}
    lines: list[str] = []
    for package in runtime_lock.packages:
        if package.name in excluded:
            continue
        if not package.artifacts:
            raise SetupError(
                f"locked runtime package {package.name} has no wheel artifacts"
            )
        lines.append(f"{package.distribution}=={package.version} \\")
        for index, artifact in enumerate(package.artifacts):
            continuation = " \\" if index < len(package.artifacts) - 1 else ""
            lines.append(f"    --hash=sha256:{artifact.sha256}{continuation}")
    return "\n".join(lines) + "\n"


def _contributor_python_requirements() -> tuple[str, ...]:
    """Reuse the project's build, test, and documentation requirements."""

    with PYPROJECT.open("rb") as stream:
        project_file = tomllib.load(stream)
    build_system = project_file.get("build-system")
    project = project_file.get("project")
    if not isinstance(build_system, dict) or not isinstance(project, dict):
        raise SetupError("pyproject must define build-system and project tables")
    optional = project.get("optional-dependencies")
    if not isinstance(optional, dict):
        raise SetupError("pyproject must define optional dependencies")

    requirements: list[str] = []
    for description, raw in (
        ("build-system.requires", build_system.get("requires")),
        ("project.optional-dependencies.test", optional.get("test")),
        ("project.optional-dependencies.docs", optional.get("docs")),
    ):
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item for item in raw
        ):
            raise SetupError(f"pyproject {description} must be a string list")
        requirements.extend(raw)
    return tuple(dict.fromkeys(requirements))


def _venv_bootstrap_python() -> Path:
    """Return an interpreter that remains available if this venv is reset."""

    base_executable = getattr(sys, "_base_executable", None)
    if isinstance(base_executable, str) and base_executable:
        return Path(base_executable)
    return Path(sys.executable)


def _ensure_venv(runner: Runner) -> None:
    if not _venv_python().is_file():
        runner.run([_venv_bootstrap_python(), "-m", "venv", VENV])
    python = _venv_python()
    if runner.dry_run:
        print(f"# ensure pip is available in {VENV}")
    else:
        pip_probe = subprocess.run(
            [python, "-m", "pip", "--version"],
            env=_venv_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if pip_probe.returncode != 0:
            runner.run(
                [python, "-m", "ensurepip", "--upgrade"],
                env=_venv_environment(),
            )
    contributor_tools = [
        "pip",
        *_contributor_python_requirements(),
        "setuptools>=68,<81",
    ]
    runner.run(
        [python, "-m", "pip", "install", "--upgrade", *contributor_tools],
        env=_venv_environment(),
    )
    requirements = _runtime_requirements_text()
    if runner.dry_run:
        print("# install the hash-locked non-candidate Python runtime closure")
    else:
        with tempfile.TemporaryDirectory(
            prefix="pyamplicol-runtime-requirements-"
        ) as raw_directory:
            requirement_path = Path(raw_directory) / "requirements.txt"
            requirement_path.write_text(requirements, encoding="utf-8")
            runner.run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--require-hashes",
                    "--only-binary=:all:",
                    "--no-deps",
                    "--requirement",
                    requirement_path,
                ],
                env=_venv_environment(),
            )


def _ensure_just(runner: Runner) -> None:
    if shutil.which("just") is None:
        runner.run(["cargo", "install", "just", "--locked"])


def _single_wheel(directory: Path, prefix: str) -> Path:
    candidates = sorted(directory.glob(f"{prefix}*.whl"))
    if len(candidates) != 1:
        raise SetupError(
            f"expected one {prefix} wheel in {directory}, found {len(candidates)}"
        )
    return candidates[0]


def _archive_candidate_wheels(directory: Path, prefix: str) -> None:
    candidates = sorted(directory.glob(f"{prefix}*.whl"))
    if not candidates:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = _managed_trash_destination(
        f"candidate-wheel-refresh-{stamp}",
        workspace_root=ROOT,
    ) / directory.relative_to(ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in candidates:
        shutil.move(str(wheel), str(destination / wheel.name))


def _verify_candidate_python_dependencies(
    runner: Runner,
    payload: dict[str, Any],
) -> None:
    expected_version = str(payload["symbolica"]["candidate_version"])
    probe = "\n".join(
        (
            "from importlib.metadata import version",
            "import sys",
            "import symbolica",
            "from symbolica import Expression",
            "from symbolica.community.idenso import simplify_color",
            "from symbolica.community.spenso import TensorNetwork",
            'actual = version("symbolica")',
            "if actual != sys.argv[1]:",
            "    raise SystemExit(",
            '        "symbolica version mismatch: expected %s, got %s"',
            "        % (sys.argv[1], actual)",
            "    )",
        )
    )
    environment = dict(_venv_environment(), SYMBOLICA_HIDE_BANNER="1")
    runner.run(
        [_venv_python(), "-I", "-c", probe, expected_version],
        env=environment,
    )


def _materialize_ratatui_sdist(
    runner: Runner,
    payload: dict[str, Any],
) -> Path:
    ratatui = payload["ratatui"]
    version = str(ratatui["version"])
    source_url = str(ratatui["source_url"])
    expected_sha256 = str(ratatui["sdist_sha256"])
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise SetupError("contributor lock has no valid Ratatui sdist SHA-256")
    destination = WHEELHOUSE / "ratatui" / f"ratatui-{version}.tar.gz"
    if runner.dry_run:
        print(
            f"# download and verify Ratatui {version} from {source_url} "
            f"at sha256:{expected_sha256}"
        )
        return destination
    if destination.is_file():
        actual_sha256 = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SetupError(
                "cached Ratatui sdist digest mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}; "
                "rerun with --reset"
            )
        return destination
    if destination.exists():
        raise SetupError(f"invalid managed Ratatui sdist at {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".ratatui-{version}-",
        suffix=".tar.gz",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(source_url) as response:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    temporary.write(block)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        temporary_path.unlink(missing_ok=True)
        raise SetupError(
            "Ratatui sdist digest mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    os.replace(temporary_path, destination)
    return destination


def _build_ratatui_wheel(
    runner: Runner,
    payload: dict[str, Any],
) -> None:
    """Build the pinned contributor-only Ratatui wheel from pinned FFI source."""

    ratatui = payload["ratatui"]
    version = str(ratatui["version"])
    ffi_revision = str(ratatui["ffi_revision"])
    if re.fullmatch(r"[0-9a-f]{40}", ffi_revision) is None:
        raise SetupError("contributor lock has no valid Ratatui FFI revision")
    sdist = _materialize_ratatui_sdist(runner, payload)
    python = _venv_python()
    wheel_directory = WHEELHOUSE / "ratatui"
    ffi_source = CHECKOUTS / "ratatui-ffi"
    if not runner.dry_run:
        if not ffi_source.is_dir():
            raise SetupError(f"missing managed Ratatui FFI source at {ffi_source}")
        _archive_candidate_wheels(wheel_directory, "ratatui")
    with tempfile.TemporaryDirectory(
        prefix="pyamplicol-ratatui-ffi-",
    ) as temporary_name:
        external_ffi_source = Path(temporary_name) / "ratatui-ffi"
        if not runner.dry_run:
            # A checkout nested below pyAmpliCol is discovered as part of its
            # parent Cargo workspace.  Build from an exact external copy so
            # ratatui-ffi remains its own workspace without editing either
            # pinned source tree.
            shutil.copytree(
                ffi_source,
                external_ffi_source,
                ignore=shutil.ignore_patterns(".git", "target"),
            )
            # Contributor builds keep temporary outputs inside this checkout.
            # Stop Cargo from treating the private copy as a pyAmpliCol member.
            manifest = external_ffi_source / "Cargo.toml"
            manifest_text = manifest.read_text(encoding="utf-8")
            if "\n[workspace]" not in f"\n{manifest_text}":
                manifest.write_text(
                    manifest_text.rstrip() + "\n\n[workspace]\n",
                    encoding="utf-8",
                )
        environment = dict(
            _venv_environment(),
            RATATUI_FFI_SRC=str(external_ffi_source.resolve()),
            CARGO_TARGET_DIR=str((external_ffi_source / "target").resolve()),
            # Keep the exact revision visible to the upstream build even though
            # RATATUI_FFI_SRC takes precedence and prevents its fallback clone.
            RATATUI_FFI_TAG=ffi_revision,
        )
        runner.run(
            [
                python,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                wheel_directory,
                sdist,
            ],
            env=environment,
        )
    if runner.dry_run:
        return
    wheel = _single_wheel(wheel_directory, "ratatui")
    runner.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            wheel,
        ],
        env=_venv_environment(),
    )
    probe = "\n".join(
        (
            "from importlib.metadata import version",
            "import sys",
            "import ratatui",
            "import ratatui_py",
            'actual = version("ratatui")',
            "if actual != sys.argv[1]:",
            "    raise SystemExit(",
            '        "ratatui version mismatch: expected %s, got %s"',
            "        % (sys.argv[1], actual)",
            "    )",
        )
    )
    runner.run(
        [python, "-I", "-c", probe, version],
        env=_venv_environment(),
    )


def _build_candidate_dependency_wheels(
    runner: Runner,
    payload: dict[str, Any],
) -> None:
    community = _managed_checkout("symbolica-community")
    python = _venv_python()
    environment = _venv_environment()
    symbolica_wheels = WHEELHOUSE / "symbolica"
    if not runner.dry_run:
        symbolica_wheels.mkdir(parents=True, exist_ok=True)

    runner.run(
        [
            python,
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--interpreter",
            python,
            "--out",
            symbolica_wheels,
        ],
        cwd=community,
        env=environment,
    )
    if not runner.dry_run:
        runner.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                _single_wheel(symbolica_wheels, "symbolica"),
            ],
            env=environment,
        )
        _verify_candidate_python_dependencies(runner, payload)
    _build_ratatui_wheel(runner, payload)


def _build_candidate_project_wheel(runner: Runner) -> None:
    python = _venv_python()
    environment = _venv_environment()
    project_wheels = ARTIFACTS
    if not runner.dry_run:
        project_wheels.mkdir(parents=True, exist_ok=True)

    build_environment = dict(
        environment,
        PYAMPLICOL_BUILD_MODE="candidate",
    )
    if not runner.dry_run:
        _archive_candidate_wheels(project_wheels, "pyamplicol")
    runner.run(
        [
            python,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--skip-dependency-check",
            "--outdir",
            project_wheels,
        ],
        cwd=ROOT,
        env=build_environment,
    )
    if not runner.dry_run:
        runner.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                _single_wheel(project_wheels, "pyamplicol"),
            ],
            env=environment,
        )


def _build_candidate_wheels(
    runner: Runner,
    payload: dict[str, Any],
) -> None:
    _build_candidate_dependency_wheels(runner, payload)
    _build_candidate_project_wheel(runner)


def _write_state(
    runner: Runner,
    sources: tuple[Source, ...],
) -> None:
    source_checkouts = {
        source.key: _managed_checkout(source.key) for source in sources
    }
    if runner.dry_run:
        print(f"# write {STATE}")
        return
    source_state: dict[str, dict[str, str]] = {}
    for source in sources:
        checkout = source_checkouts[source.key]
        head = _git_head(runner, checkout)
        source_state[source.key] = {
            "url": source.url,
            "revision": head,
        }
        if source.branch is not None:
            source_state[source.key]["branch"] = source.branch
    state = {
        "schema_version": 1,
        "publishable": False,
        "sources": source_state,
    }
    STATE.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--without-legacy-amplicol", action="store_true")
    build_mode = parser.add_mutually_exclusive_group()
    build_mode.add_argument("--no-build", action="store_true")
    build_mode.add_argument(
        "--dependencies-only",
        action="store_true",
        help="build and install pinned candidate Python dependencies, not pyamplicol",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = _lock()
    runner = Runner(dry_run=args.dry_run)
    if args.reset:
        _archive_managed_state(runner)
    if not args.dry_run:
        _require_tools()
    sources = _sources(
        payload,
        with_legacy=not args.without_legacy_amplicol,
    )
    _ensure_just(runner)
    _ensure_venv(runner)
    for source in sources:
        _checkout(runner, source, update=args.update)
    _configure_sources(runner)
    _write_cargo_config(runner)
    _write_candidate_lock(runner)
    _write_state(runner, sources)
    if args.dependencies_only:
        _build_candidate_dependency_wheels(runner, payload)
    elif not args.no_build:
        _build_candidate_wheels(runner, payload)
    print(f"Contributor environment ready at {VENV}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
