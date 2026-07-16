# SPDX-License-Identifier: 0BSD
"""Release-lock, Git checkout, patch, and compiler provenance helpers."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .model import (
    LOCK,
    ROOT,
    V2_ONLY_LEGACY_PATCHES,
    CompilerProvenance,
    LegacyOracleError,
)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    capture: bool = True,
    subprocess_module: Any = subprocess,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess_module.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        raise LegacyOracleError(
            f"command exited with {completed.returncode}: {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return cast(subprocess.CompletedProcess[str], completed)


def _release_lock(*, lock_path: Path = LOCK) -> dict[str, Any]:
    with lock_path.open("rb") as stream:
        return tomllib.load(stream)


def expected_revision(*, lock: Mapping[str, Any] | None = None) -> str:
    release_lock = _release_lock() if lock is None else lock
    return str(release_lock["legacy_amplicol"]["revision"])


def checkout_url(*, lock: Mapping[str, Any] | None = None) -> str:
    release_lock = _release_lock() if lock is None else lock
    source = str(release_lock["legacy_amplicol"]["source_url"])
    prefix = "git@github.com:"
    if source.startswith(prefix):
        return "https://github.com/" + source.removeprefix(prefix)
    return source


def managed_patches(
    *,
    root: Path = ROOT,
    lock: Mapping[str, Any] | None = None,
) -> tuple[Path, ...]:
    release_lock = _release_lock() if lock is None else lock
    return tuple(
        root / "dependencies" / str(entry["path"])
        for entry in release_lock["patches"]
        if entry["dependency"] == "legacy-amplicol"
    )


def managed_patch_metadata(
    *,
    fixture_schema_version: int | None = None,
    root: Path = ROOT,
    lock: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], ...]:
    release_lock = _release_lock() if lock is None else lock
    records = []
    for entry in release_lock["patches"]:
        if entry["dependency"] != "legacy-amplicol":
            continue
        if fixture_schema_version == 1 and str(entry["path"]) in V2_ONLY_LEGACY_PATCHES:
            continue
        path = root / "dependencies" / str(entry["path"])
        expected = str(entry["sha256"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise LegacyOracleError(
                f"legacy AmpliCol patch digest mismatch for {path.name}: "
                f"{actual}, expected {expected}"
            )
        records.append({"path": str(entry["path"]), "sha256": actual})
    return tuple(records)


def _managed_patch_paths(patches: Sequence[Path]) -> tuple[str, ...]:
    paths: set[str] = set()
    for patch in patches:
        for line in patch.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(?:--- a/|\+\+\+ b/)([^\t ]+)", line)
            if match is not None:
                paths.add(match.group(1))
    if not paths:
        raise LegacyOracleError("managed legacy patch series changes no files")
    return tuple(sorted(paths))


def _validate_exact_patch_state(
    repository: Path,
    patches: Sequence[Path],
    *,
    subprocess_module: Any = subprocess,
) -> None:
    expected_paths = _managed_patch_paths(patches)
    changed = subprocess_module.run(
        ["git", "diff", "--name-only", "-z", "HEAD", "--"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    changed_paths = tuple(
        sorted(path.decode("utf-8") for path in changed.split(b"\0") if path)
    )
    if changed_paths != expected_paths:
        raise LegacyOracleError(
            "legacy AmpliCol tracked changes do not exactly match the managed "
            f"patch inventory: found {changed_paths}, expected {expected_paths}"
        )

    with tempfile.TemporaryDirectory(prefix="pyamplicol-legacy-patch-state-") as raw:
        expected_root = Path(raw)
        for relative in expected_paths:
            content = subprocess_module.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=repository,
                check=True,
                capture_output=True,
            ).stdout
            destination = expected_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for patch in patches:
            completed = subprocess_module.run(
                ["git", "apply", "--whitespace=nowarn", str(patch.resolve())],
                cwd=expected_root,
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise LegacyOracleError(
                    f"managed legacy patch cannot be reconstructed: {patch.name}: "
                    f"{completed.stderr.strip()}"
                )
        mismatches = tuple(
            relative
            for relative in expected_paths
            if (repository / relative).read_bytes()
            != (expected_root / relative).read_bytes()
        )
    if mismatches:
        raise LegacyOracleError(
            "legacy AmpliCol contains edits beyond the managed patch series in "
            + ", ".join(mismatches)
        )


def validate_checkout(
    repository: Path,
    *,
    run: Any = _run,
    revision: Any = expected_revision,
    patches: Any = managed_patches,
    validate_exact: Any = _validate_exact_patch_state,
    subprocess_module: Any = subprocess,
) -> None:
    repository = repository.resolve()
    if not (repository / ".git").exists():
        raise LegacyOracleError(
            f"legacy AmpliCol checkout is absent: {repository}; run `just dev-install`"
        )
    actual_revision = run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()
    if actual_revision != revision():
        raise LegacyOracleError(
            f"legacy AmpliCol is at {actual_revision}, expected {revision()}"
        )
    managed = patches()
    for patch in managed:
        completed = subprocess_module.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=repository,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise LegacyOracleError(
                f"managed legacy patch is not applied: {patch.name}"
            )
    validate_exact(repository, managed)


def prepare_checkout(
    repository: Path,
    *,
    run: Any = _run,
    validate: Any = validate_checkout,
    url: Any = checkout_url,
    revision: Any = expected_revision,
    patches: Any = managed_patches,
) -> None:
    repository = repository.resolve()
    if repository.exists():
        validate(repository)
        return
    repository.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", str(repository)], cwd=repository.parent)
    run(["git", "remote", "add", "origin", url()], cwd=repository)
    run(
        ["git", "fetch", "--depth=1", "origin", revision()],
        cwd=repository,
        capture=False,
    )
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repository)
    for patch in patches():
        run(["git", "apply", str(patch)], cwd=repository)
    validate(repository)


def _make_database_variable(database: str, name: str) -> str:
    matches = re.findall(rf"^{re.escape(name)}\s*[:?+]?=\s*(.*?)\s*$", database, re.M)
    if not matches or not matches[-1]:
        raise LegacyOracleError(f"legacy makefile does not define {name}")
    return cast(str, matches[-1])


def _compiler_provenance(
    repository: Path,
    *,
    run: Any = _run,
    shutil_module: Any = shutil,
) -> CompilerProvenance:
    database = run(
        ["make", "-f", "makefile", "-pn", "amplicol_color_probe"],
        cwd=repository,
    ).stdout
    compiler_tokens = shlex.split(_make_database_variable(database, "FC"))
    if not compiler_tokens:
        raise LegacyOracleError("legacy makefile defines an empty Fortran compiler")
    compiler = shutil_module.which(compiler_tokens[0])
    if compiler is None:
        candidate = repository / compiler_tokens[0]
        if candidate.is_file():
            compiler = str(candidate.resolve())
    if compiler is None:
        raise LegacyOracleError(
            f"cannot resolve legacy Fortran compiler {compiler_tokens[0]!r}"
        )
    compiler_command = [compiler, *compiler_tokens[1:]]
    version_result = run([*compiler_command, "--version"], cwd=repository)
    version_lines = tuple(
        line.strip()
        for line in (version_result.stdout + "\n" + version_result.stderr).splitlines()
        if line.strip()
    )
    if not version_lines:
        raise LegacyOracleError("Fortran compiler emitted no version identity")
    target_result = run([*compiler_command, "-dumpmachine"], cwd=repository)
    target = target_result.stdout.strip()
    if not target:
        raise LegacyOracleError("Fortran compiler emitted no target triple")
    executable = repository / "amplicol_color_probe"
    if not executable.is_file():
        raise LegacyOracleError("amplicol_color_probe has not been built")
    return CompilerProvenance(
        identity=Path(compiler).name,
        version=version_lines[0],
        flags=tuple(shlex.split(_make_database_variable(database, "FFLAGS"))),
        target=target,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )
