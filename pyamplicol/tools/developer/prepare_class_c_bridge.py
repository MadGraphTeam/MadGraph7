#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Prepare a Class-C bridge with descendant tools and the staged ancestor runtime.

The live checkout must already be at the reviewed descendant while its source
runtime and report environment still authenticate the measured ancestor.  This
helper materializes a temporary, tracked ancestor worktree, binds only the
ignored files that make up the retained source runtime, and invokes the
descendant report controller against that authenticated package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_NATIVE_EXTENSION_SUFFIXES = (".dylib", ".pyd", ".so")
_NATIVE_BUILD_INPUT_FILES = (
    Path("Cargo.lock"),
    Path("Cargo.toml"),
    Path("pyproject.toml"),
    Path("rust-toolchain.toml"),
    Path("dependencies/candidate-Cargo.lock"),
    Path("dependencies/candidate-cargo-config.toml"),
    Path("dependencies/contributor-lock.toml"),
    Path("dependencies/install-state.json"),
    Path("dependencies/release-lock.toml"),
)
_NATIVE_BUILD_INPUT_TREES = (
    Path("build_backend"),
    Path("dependencies/patches"),
    Path("rust"),
)
_NATIVE_BUILD_INPUT_SUFFIXES = {
    ".f90",
    ".h",
    ".hpp",
    ".json",
    ".patch",
    ".py",
    ".pyi",
    ".rs",
    ".toml",
}
_IGNORED_NATIVE_INPUTS = (
    Path("dependencies/candidate-Cargo.lock"),
    Path("dependencies/candidate-cargo-config.toml"),
    Path("dependencies/install-state.json"),
)
_STAGED_SDK_DIRECTORIES = (
    Path("_sdk/fortran"),
    Path("_sdk/include"),
    Path("_sdk/lib"),
)
_STAGED_SDK_FILES = (
    Path("_sdk/link.json"),
    Path("_sdk/metadata.json"),
)


class ClassCPrepareBootstrapError(RuntimeError):
    """The ancestor source runtime could not be materialized exactly."""


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def _git_commit(root: Path, revision: str) -> str:
    completed = _run_git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    value = completed.stdout.strip()
    if (
        completed.returncode != 0
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ClassCPrepareBootstrapError(
            f"cannot resolve Git revision {revision!r}"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_build_inputs_digest(root: Path) -> str:
    paths = [root / relative for relative in _NATIVE_BUILD_INPUT_FILES]
    for relative in _NATIVE_BUILD_INPUT_TREES:
        tree = root / relative
        if not tree.is_dir():
            continue
        paths.extend(
            path
            for path in tree.rglob("*")
            if path.is_file()
            and not {"__pycache__", "target"}.intersection(
                path.relative_to(tree).parts
            )
            and path.suffix in _NATIVE_BUILD_INPUT_SUFFIXES
        )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _regular_file(path: Path, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ClassCPrepareBootstrapError(f"{description} is unavailable") from error
    if path.is_symlink() or not resolved.is_file():
        raise ClassCPrepareBootstrapError(
            f"{description} must be one regular file"
        )
    return resolved


def _hardlink(source: Path, destination: Path, description: str) -> None:
    source = _regular_file(source, description)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ClassCPrepareBootstrapError(
            f"ancestor runtime destination already exists: {destination}"
        )
    os.link(source, destination)
    if destination.is_symlink() or not destination.is_file():
        raise ClassCPrepareBootstrapError(
            f"ancestor runtime member is not regular: {destination}"
        )


def _hardlink_tree(source: Path, destination: Path, description: str) -> None:
    try:
        root = source.resolve(strict=True)
    except OSError as error:
        raise ClassCPrepareBootstrapError(f"{description} is unavailable") from error
    if source.is_symlink() or not root.is_dir():
        raise ClassCPrepareBootstrapError(
            f"{description} must be one regular directory tree"
        )
    for member in sorted(root.rglob("*")):
        relative = member.relative_to(root)
        if member.is_symlink():
            raise ClassCPrepareBootstrapError(
                f"{description} contains a symlink: {relative}"
            )
        if member.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif member.is_file():
            _hardlink(member, destination / relative, description)
        else:
            raise ClassCPrepareBootstrapError(
                f"{description} contains a non-file member: {relative}"
            )


def _read_build_info(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ClassCPrepareBootstrapError(
            "retained source-runtime build info is unreadable"
        ) from error
    if not isinstance(value, dict):
        raise ClassCPrepareBootstrapError(
            "retained source-runtime build info is invalid"
        )
    return value


def _bind_ancestor_runtime(
    live_root: Path,
    ancestor_root: Path,
) -> dict[str, object]:
    """Bind the exact ignored source-runtime inputs into a tracked A worktree."""

    live_package = live_root / "src/pyamplicol"
    ancestor_package = ancestor_root / "src/pyamplicol"
    extensions = tuple(
        path
        for path in live_package.glob("_rusticol.*")
        if path.is_file()
        and not path.is_symlink()
        and path.name.endswith(_NATIVE_EXTENSION_SUFFIXES)
    )
    if len(extensions) != 1:
        raise ClassCPrepareBootstrapError(
            "live staged runtime must expose exactly one regular native extension"
        )
    extension = extensions[0]

    source_runtime = live_root / ".artifacts/source-runtime"
    try:
        source_runtime_target = source_runtime.resolve(strict=True)
    except OSError as error:
        raise ClassCPrepareBootstrapError(
            "retained source-runtime directory is unavailable"
        ) from error
    if not source_runtime_target.is_dir():
        raise ClassCPrepareBootstrapError(
            "retained source-runtime path is not a directory"
        )
    build_info = _read_build_info(source_runtime_target / "_build_info.json")
    contract = build_info.get("source_runtime")
    if not isinstance(contract, Mapping):
        raise ClassCPrepareBootstrapError(
            "retained source-runtime contract is missing"
        )
    extension_sha256 = _sha256(extension)
    if (
        contract.get("extension_name") != extension.name
        or contract.get("extension_sha256") != extension_sha256
    ):
        raise ClassCPrepareBootstrapError(
            "retained native extension differs from source-runtime build info"
        )

    _hardlink(
        extension,
        ancestor_package / extension.name,
        "retained native extension",
    )
    for relative in _STAGED_SDK_DIRECTORIES:
        _hardlink_tree(
            live_package / relative,
            ancestor_package / relative,
            f"retained SDK tree {relative}",
        )
    for relative in _STAGED_SDK_FILES:
        _hardlink(
            live_package / relative,
            ancestor_package / relative,
            f"retained SDK member {relative}",
        )
    try:
        metadata = json.loads(
            (live_package / "_sdk/metadata.json").read_text(encoding="utf-8")
        )
        target = str(metadata["target"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ClassCPrepareBootstrapError(
            "retained SDK target metadata is invalid"
        ) from error
    if not target or target in {".", ".."} or "/" in target or "\\" in target:
        raise ClassCPrepareBootstrapError(
            f"retained SDK target is unsafe: {target!r}"
        )
    _hardlink_tree(
        live_package / "assets/selftest" / target,
        ancestor_package / "assets/selftest" / target,
        "retained target self-test fixture",
    )

    ancestor_artifacts = ancestor_root / ".artifacts"
    ancestor_artifacts.mkdir()
    os.symlink(
        source_runtime_target,
        ancestor_artifacts / "source-runtime",
        target_is_directory=True,
    )
    dependencies = ancestor_root / "dependencies"
    checkouts = (live_root / "dependencies/checkouts").resolve(strict=True)
    if not checkouts.is_dir():
        raise ClassCPrepareBootstrapError(
            "retained managed dependency checkout root is unavailable"
        )
    os.symlink(checkouts, dependencies / "checkouts", target_is_directory=True)
    for relative in _IGNORED_NATIVE_INPUTS:
        source = (live_root / relative).resolve(strict=True)
        if not source.is_file():
            raise ClassCPrepareBootstrapError(
                f"retained native input is unavailable: {relative}"
            )
        destination = ancestor_root / relative
        if destination.exists() or destination.is_symlink():
            raise ClassCPrepareBootstrapError(
                f"ancestor native input unexpectedly exists: {relative}"
            )
        os.symlink(source, destination)

    observed_native_digest = _native_build_inputs_digest(ancestor_root)
    expected_native_digest = contract.get("native_build_inputs_sha256")
    if (
        not isinstance(expected_native_digest, str)
        or observed_native_digest != expected_native_digest
    ):
        raise ClassCPrepareBootstrapError(
            "materialized ancestor native inputs differ from retained build info"
        )
    return {
        "candidate_fingerprint": build_info.get("candidate_fingerprint"),
        "extension_name": extension.name,
        "extension_sha256": extension_sha256,
        "native_build_inputs_sha256": observed_native_digest,
        "package_version": build_info.get("version"),
        "target": target,
    }


def _profile(value: str) -> str:
    if _PROFILE_RE.fullmatch(value) is None or ".." in value:
        raise argparse.ArgumentTypeError("invalid performance-report profile")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--report-profile", type=_profile, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--coordination-root", type=Path)
    parser.add_argument("--ancestor-revision", required=True)
    parser.add_argument("--descendant-revision", required=True)
    parser.add_argument(
        "--impact",
        choices=("hzz-orientation-v1", "recurrence-summary-cap-v1"),
        required=True,
    )
    return parser


def _cleanup_worktree(repo_root: Path, worktree: Path) -> None:
    completed = _run_git(repo_root, "worktree", "remove", "--force", str(worktree))
    if completed.returncode != 0:
        _run_git(repo_root, "worktree", "prune")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo_root = args.repo_root.expanduser().resolve(strict=True)
        ancestor = _git_commit(repo_root, args.ancestor_revision)
        descendant = _git_commit(repo_root, args.descendant_revision)
        if ancestor == descendant or _git_commit(repo_root, "HEAD") != descendant:
            raise ClassCPrepareBootstrapError(
                "live checkout must be at the distinct requested descendant"
            )
        ancestry = _run_git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        )
        if ancestry.returncode != 0:
            raise ClassCPrepareBootstrapError(
                "Class-C descendant is not a Git descendant of its ancestor"
            )
        entrypoint = (
            repo_root
            / "docs/performance_reports"
            / args.report_profile
            / "result_tables.py"
        )
        if not entrypoint.is_file() or entrypoint.is_symlink():
            raise ClassCPrepareBootstrapError(
                "descendant report entrypoint is unavailable"
            )

        temporary_parent = Path(
            tempfile.mkdtemp(
                prefix=".pyamplicol-class-c-ancestor-runtime-",
                dir=repo_root.parent,
            )
        )
        ancestor_root = temporary_parent / "worktree"
        try:
            added = _run_git(
                repo_root,
                "worktree",
                "add",
                "--detach",
                str(ancestor_root),
                ancestor,
            )
            if added.returncode != 0:
                raise ClassCPrepareBootstrapError(
                    "cannot materialize the tracked ancestor worktree: "
                    f"{added.stderr.strip()}"
                )
            runtime = _bind_ancestor_runtime(repo_root, ancestor_root)
            print(
                json.dumps(
                    {
                        "ancestor_revision": ancestor,
                        "ancestor_runtime_root": str(ancestor_root),
                        "descendant_revision": descendant,
                        "descendant_tools_root": str(repo_root),
                        **runtime,
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            command = [
                sys.executable,
                str(entrypoint),
                "--repo-root",
                str(repo_root),
                "--report-profile",
                args.report_profile,
                "--class-c-ancestor-runtime-root",
                str(ancestor_root),
            ]
            if args.artifact_root is not None:
                command.extend(("--artifact-root", str(args.artifact_root)))
            if args.coordination_root is not None:
                command.extend(
                    ("--coordination-root", str(args.coordination_root))
                )
            command.extend(
                (
                    "prepare-class-c-bridge",
                    "--ancestor-revision",
                    ancestor,
                    "--descendant-revision",
                    descendant,
                    "--impact",
                    args.impact,
                )
            )
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise ClassCPrepareBootstrapError(
                    "descendant Class-C prepare controller failed "
                    f"with exit code {completed.returncode}"
                )
        finally:
            if ancestor_root.exists():
                _cleanup_worktree(repo_root, ancestor_root)
            shutil.rmtree(temporary_parent, ignore_errors=True)
        return 0
    except (ClassCPrepareBootstrapError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
