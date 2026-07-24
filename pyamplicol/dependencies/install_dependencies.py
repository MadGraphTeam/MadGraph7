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
import tarfile
import tempfile
import tomllib
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / "dependencies"
RELEASE_LOCK = DEPENDENCIES / "release-lock.toml"
CONTRIBUTOR_LOCK = DEPENDENCIES / "contributor-lock.toml"
# Retained as the public constant used by older contributor-side callers.
LOCK = RELEASE_LOCK
PYTHON_LOCK = DEPENDENCIES / "python-runtime-lock.toml"
CHECKOUTS = DEPENDENCIES / "checkouts"
PATCHES = DEPENDENCIES / "patches"
WHEELHOUSE = DEPENDENCIES / "wheelhouse"
VENV = ROOT / ".venv"
STATE = DEPENDENCIES / "install-state.json"
CANDIDATE_LOCK = DEPENDENCIES / "candidate-Cargo.lock"
CARGO_CONFIG = DEPENDENCIES / "candidate-cargo-config.toml"
ARTIFACTS = ROOT / ".artifacts" / "candidate"
TRASH = ROOT / ".trash"

sys.path.insert(0, str(ROOT / "build_backend"))
from python_lock import load_python_runtime_lock  # noqa: E402

_SOURCE_TREE_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "target",
}
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


@dataclass(frozen=True)
class ContributorPatch:
    name: str
    target: str
    relative_path: str
    path: Path
    sha256: str
    applies_to_revision: str


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
    gammaloop = payload["gammaloop_candidate"]
    sources = [
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = TRASH / f"dependency-reset-{stamp}"
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


def _source_tree_sha256(root: Path) -> str:
    """Hash every candidate source byte outside deterministic build caches."""

    digest = hashlib.sha256()
    for raw_directory, raw_directories, raw_files in os.walk(root, topdown=True):
        directory = Path(raw_directory)
        directories = sorted(
            name for name in raw_directories if name not in _SOURCE_TREE_EXCLUDES
        )
        raw_directories[:] = [
            name for name in directories if not (directory / name).is_symlink()
        ]
        entries = [
            *(
                directory / name
                for name in directories
                if (directory / name).is_symlink()
            ),
            *(
                directory / name
                for name in sorted(raw_files)
                if name not in _SOURCE_TREE_EXCLUDES
                and not name.endswith((".pyc", ".pyo"))
            ),
        ]
        for path in entries:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            mode = path.lstat().st_mode & 0o111
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(mode.to_bytes(2, "big"))
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"L")
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
            elif path.is_file():
                digest.update(b"F")
                with path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
            else:
                digest.update(b"O")
    return digest.hexdigest()


def _contributor_patches(payload: dict[str, Any]) -> tuple[ContributorPatch, ...]:
    """Load and verify every tracked contributor patch before touching sources."""

    raw_patches = payload.get("patches")
    if not isinstance(raw_patches, list) or not raw_patches:
        raise SetupError("contributor lock must list at least one source patch")
    allowed_keys = {
        "name",
        "target",
        "path",
        "sha256",
        "applies_to_revision",
    }
    dependency_root = DEPENDENCIES.resolve()
    patches: list[ContributorPatch] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, entry in enumerate(raw_patches):
        if not isinstance(entry, dict) or set(entry) != allowed_keys:
            raise SetupError(
                f"contributor patch {index} must contain only "
                + "/".join(sorted(allowed_keys))
            )
        values = {key: entry[key] for key in allowed_keys}
        if not all(isinstance(value, str) and value for value in values.values()):
            raise SetupError(
                f"contributor patch {index} fields must be nonempty strings"
            )
        name = str(entry["name"])
        target = str(entry["target"])
        relative = str(entry["path"])
        expected_sha256 = str(entry["sha256"])
        revision = str(entry["applies_to_revision"])
        if name in seen_names:
            raise SetupError(f"contributor patch name is repeated: {name}")
        if relative in seen_paths:
            raise SetupError(f"contributor patch path is repeated: {relative}")
        if target != "symjit":
            raise SetupError(f"unsupported contributor patch target: {target}")
        target_revision = str(payload["symjit"]["candidate_revision"])
        if revision != target_revision:
            raise SetupError(
                f"contributor patch {name} targets revision {revision}, "
                f"expected {target_revision}"
            )
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise SetupError(f"contributor patch {name} has an invalid SHA-256")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or not pure.parts
            or pure.parts[0] != "patches"
            or pure.suffix != ".patch"
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise SetupError(
                f"contributor patch {name} has an unsafe dependency path: {relative}"
            )
        path = DEPENDENCIES.joinpath(*pure.parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(dependency_root)
        except (OSError, ValueError) as error:
            raise SetupError(
                f"contributor patch {name} is missing or escapes dependencies: "
                f"{relative}"
            ) from error
        current = DEPENDENCIES
        for part in pure.parts:
            current /= part
            if current.is_symlink():
                raise SetupError(f"contributor patch {name} may not use symlinks")
        if not resolved.is_file():
            raise SetupError(f"contributor patch {name} is not a regular file")
        actual_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SetupError(
                f"contributor patch {name} digest mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        patches.append(
            ContributorPatch(
                name=name,
                target=target,
                relative_path=pure.as_posix(),
                path=resolved,
                sha256=expected_sha256,
                applies_to_revision=revision,
            )
        )
        seen_names.add(name)
        seen_paths.add(relative)
    if len(patches) != 1:
        raise SetupError("contributor lock must list the one exact SymJIT patch")
    return tuple(patches)


def _patch_state(patches: Sequence[ContributorPatch]) -> list[dict[str, str]]:
    return [
        {
            "name": patch.name,
            "target": patch.target,
            "path": patch.relative_path,
            "sha256": patch.sha256,
            "applies_to_revision": patch.applies_to_revision,
        }
        for patch in patches
    ]


def _apply_contributor_patches(runner: Runner, payload: dict[str, Any]) -> None:
    """Apply each exact patch once, or verify that it is already fully applied."""

    for patch in _contributor_patches(payload):
        destination = CHECKOUTS / patch.target
        if runner.dry_run:
            print(
                f"# verify/apply {patch.name} to {patch.target} "
                f"at {patch.applies_to_revision}"
            )
            continue
        if not destination.is_dir():
            raise SetupError(f"contributor patch target is missing: {patch.target}")
        patch_environment = dict(os.environ)
        for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
            patch_environment.pop(name, None)
        # The managed archive is intentionally not a Git checkout.  Prevent Git
        # from discovering pyAmpliCol's parent repository, where the checkout is
        # ignored and `git apply` would silently skip every patch path.
        patch_environment["GIT_CEILING_DIRECTORIES"] = str(destination.parent.resolve())
        forward = [
            "git",
            "apply",
            "--check",
            "--whitespace=nowarn",
            patch.path,
        ]
        reverse = [
            "git",
            "apply",
            "--check",
            "--reverse",
            "--whitespace=nowarn",
            patch.path,
        ]
        can_apply = runner.run(
            forward,
            cwd=destination,
            env=patch_environment,
            capture=True,
            check=False,
        )
        is_applied = runner.run(
            reverse,
            cwd=destination,
            env=patch_environment,
            capture=True,
            check=False,
        )
        if can_apply.returncode == 0 and is_applied.returncode == 0:
            raise SetupError(
                f"contributor patch {patch.name} has ambiguous apply state"
            )
        if can_apply.returncode == 0:
            runner.run(
                ["git", "apply", "--whitespace=nowarn", patch.path],
                cwd=destination,
                env=patch_environment,
                capture=True,
            )
            is_applied = runner.run(
                reverse,
                cwd=destination,
                env=patch_environment,
                capture=True,
                check=False,
            )
        if is_applied.returncode != 0:
            detail = (
                can_apply.stderr.strip()
                or is_applied.stderr.strip()
                or "patch does not match the managed source"
            )
            raise SetupError(
                f"contributor patch {patch.name} is neither cleanly applicable "
                f"nor fully applied; rerun with --reset: {detail}"
            )


def _verify_symjit_tree(runner: Runner, payload: dict[str, Any]) -> str | None:
    """Verify the fully patched and configured SymJIT source-tree identity."""

    expected = payload["symjit"].get("candidate_tree_sha256")
    if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
        raise SetupError("contributor lock has no valid SymJIT candidate tree SHA-256")
    if runner.dry_run:
        print(f"# verify configured SymJIT tree at sha256:{expected}")
        return None
    actual = _source_tree_sha256(CHECKOUTS / "symjit")
    if actual != expected:
        raise SetupError(
            "configured SymJIT source tree digest mismatch: "
            f"expected {expected}, got {actual}; rerun with --reset"
        )
    return actual


def _checkout(runner: Runner, source: Source, *, update: bool) -> None:
    if not source.path.exists():
        source.path.parent.mkdir(parents=True, exist_ok=True)
        clone_command = [
            "git",
            "clone",
            "--filter=blob:none",
        ]
        if source.branch is not None:
            clone_command.extend(["--branch", source.branch, "--single-branch"])
        clone_command.extend(["--no-checkout", source.url, str(source.path)])
        runner.run(clone_command)
        runner.run(
            ["git", "checkout", "--detach", source.revision],
            cwd=source.path,
        )
        return
    if runner.dry_run:
        print(f"# verify {source.key} at {source.revision}")
        return
    head = _git_head(runner, source.path)
    if head == source.revision:
        return
    if not update:
        raise SetupError(
            f"{source.path} is at {head}, expected {source.revision}; "
            "rerun with --update or --reset"
        )
    fetch_ref = source.branch or source.revision
    runner.run(["git", "fetch", "origin", fetch_ref], cwd=source.path)
    runner.run(
        ["git", "checkout", "--detach", source.revision],
        cwd=source.path,
    )


def _materialize_symjit(runner: Runner, payload: dict[str, Any]) -> None:
    """Materialize the exact checksummed SymJIT source archive."""

    symjit = payload["symjit"]
    version = str(symjit["candidate_version"])
    expected_sha256 = str(symjit["archive_sha256"])
    archive_prefix = str(symjit["archive_prefix"])
    destination = CHECKOUTS / "symjit"
    if runner.dry_run:
        print(
            f"# download and verify SymJIT {version} from {symjit['source_url']} "
            f"at sha256:{expected_sha256}"
        )
        return

    manifest = destination / "Cargo.toml"
    if manifest.is_file():
        with manifest.open("rb") as stream:
            installed = tomllib.load(stream)
        if str(installed.get("package", {}).get("version")) != version:
            raise SetupError(
                f"{destination} is not SymJIT {version}; rerun with --reset"
            )
        return
    if destination.exists():
        raise SetupError(f"invalid managed SymJIT source at {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="symjit-source-", dir=destination.parent
    ) as raw:
        temporary = Path(raw)
        archive = temporary / f"symjit-{version}.crate"
        with urllib.request.urlopen(str(symjit["source_url"])) as response:
            archive.write_bytes(response.read())
        actual_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise SetupError(
                "SymJIT source archive digest mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        extracted = temporary / "extracted"
        extracted.mkdir()
        prefix = archive_prefix
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or not path.parts
                    or path.parts[0] != prefix
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or member.issym()
                    or member.islnk()
                ):
                    raise SetupError(f"unsafe SymJIT archive member: {member.name}")
                relative = Path(*path.parts[1:])
                target = extracted / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = source.extractfile(member)
                    if stream is None:
                        raise SetupError(
                            f"could not read SymJIT archive member: {member.name}"
                        )
                    target.write_bytes(stream.read())
                else:
                    raise SetupError(
                        f"unsupported SymJIT archive member: {member.name}"
                    )
        os.replace(extracted, destination)


def _replace_section(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(?ms)^\[{re.escape(name)}\]\n.*?(?=^\[[^\n]+\]\n|\Z)")
    replacement = f"[{name}]\n{body.strip()}\n\n"
    if not pattern.search(text):
        return text.rstrip() + "\n\n" + replacement
    return pattern.sub(replacement, text, count=1)


def _configure_source_manifests(runner: Runner) -> None:
    if runner.dry_run:
        print("# rewrite candidate Cargo manifests to pinned local paths")
        return
    symbolica = CHECKOUTS / "symbolica"
    symjit = CHECKOUTS / "symjit"
    community = CHECKOUTS / "symbolica-community"
    gammaloop = CHECKOUTS / "gammaloop"

    symjit_cargo = symjit / "Cargo.toml"
    text = symjit_cargo.read_text(encoding="utf-8")
    text, count = re.subn(
        r'(?m)^crate-type\s*=\s*\["cdylib"\]\s*$',
        'crate-type = ["rlib"]',
        text,
        count=1,
    )
    if count == 0 and 'crate-type = ["rlib"]' not in text:
        raise SetupError("could not configure SymJIT as an rlib")
    symjit_cargo.write_text(text, encoding="utf-8")

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
    community = CHECKOUTS / "symbolica-community"
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
    entries = {
        "graphica": CHECKOUTS / "symbolica" / "lib" / "graphica",
        "numerica": CHECKOUTS / "symbolica" / "lib" / "numerica",
        "symbolica": CHECKOUTS / "symbolica",
        "symjit": CHECKOUTS / "symjit",
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
    """Require the canonical lock to contain published registry crates only."""

    invalid: list[str] = []
    for package in _cargo_lock_packages(path):
        name = str(package.get("name", "<unnamed>"))
        source = package.get("source")
        checksum = package.get("checksum")
        if name in _WORKSPACE_CRATES and source is None and checksum is None:
            continue
        if source != _CRATES_IO_SOURCE:
            invalid.append(f"{name} has non-registry source {source!r}")
            continue
        if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
            invalid.append(f"{name} has no valid registry checksum")
    if invalid:
        raise SetupError(
            "canonical Cargo.lock is candidate/path-resolved; regenerate it "
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
        (
            "symjit",
            str(lock["symbolica"]["published_symjit_version"]),
            str(lock["symjit"]["candidate_version"]),
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


def _ensure_venv(runner: Runner, payload: dict[str, Any]) -> None:
    if not _venv_python().is_file():
        runner.run([sys.executable, "-m", "venv", VENV])
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
    toolchain = payload["toolchain"]
    contributor_tools = [
        "pip",
        "build>=1.2,<2",
        "jsonschema>=4.22,<5",
        f"maturin=={toolchain['maturin']}",
        "mypy>=1.13,<2",
        "pytest>=8.3,<9",
        "ruff>=0.9,<1",
        "twine>=6,<7",
        "wheel>=0.45,<1",
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = (
        TRASH / f"candidate-wheel-refresh-{stamp}" / directory.relative_to(ROOT)
    )
    destination.mkdir(parents=True, exist_ok=True)
    for wheel in candidates:
        shutil.move(str(wheel), str(destination / wheel.name))


def _build_candidate_wheels(runner: Runner) -> None:
    python = _venv_python()
    environment = _venv_environment()
    symbolica_wheels = WHEELHOUSE / "symbolica"
    project_wheels = ARTIFACTS
    for directory in (symbolica_wheels, project_wheels):
        if not runner.dry_run:
            directory.mkdir(parents=True, exist_ok=True)

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
        cwd=CHECKOUTS / "symbolica-community",
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


def _write_state(
    runner: Runner,
    payload: dict[str, Any],
    sources: tuple[Source, ...],
) -> None:
    if runner.dry_run:
        print(f"# write {STATE}")
        return
    source_state: dict[str, dict[str, str]] = {}
    for source in sources:
        head = _git_head(runner, source.path)
        source_state[source.key] = {
            "url": source.url,
            "revision": head,
            "worktree_sha256": _source_tree_sha256(source.path),
        }
        if source.branch is not None:
            source_state[source.key]["branch"] = source.branch
    patches = _contributor_patches(payload)
    state = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "publishable": False,
        "release_lock_sha256": hashlib.sha256(RELEASE_LOCK.read_bytes()).hexdigest(),
        "contributor_lock_sha256": hashlib.sha256(
            CONTRIBUTOR_LOCK.read_bytes()
        ).hexdigest(),
        "python_runtime_lock_sha256": hashlib.sha256(
            PYTHON_LOCK.read_bytes()
        ).hexdigest(),
        "candidate_lock_sha256": hashlib.sha256(
            CANDIDATE_LOCK.read_bytes()
        ).hexdigest(),
        "cargo_config_sha256": hashlib.sha256(CARGO_CONFIG.read_bytes()).hexdigest(),
        "sources": source_state,
        "patches": _patch_state(patches),
    }
    symjit = payload["symjit"]
    source_state["symjit"] = {
        "url": str(symjit["source_url"]),
        "revision": str(symjit["candidate_revision"]),
        "version": str(symjit["candidate_version"]),
        "archive_sha256": str(symjit["archive_sha256"]),
        "patch_sha256": patches[0].sha256,
        "worktree_sha256": _source_tree_sha256(CHECKOUTS / "symjit"),
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
    parser.add_argument("--no-build", action="store_true")
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
    _ensure_venv(runner, payload)
    for source in sources:
        _checkout(runner, source, update=args.update)
    _materialize_symjit(runner, payload)
    _apply_contributor_patches(runner, payload)
    _configure_sources(runner)
    _verify_symjit_tree(runner, payload)
    _write_cargo_config(runner)
    _write_candidate_lock(runner)
    _write_state(runner, payload, sources)
    if not args.no_build:
        _build_candidate_wheels(runner)
    print(f"Contributor environment ready at {VENV}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
