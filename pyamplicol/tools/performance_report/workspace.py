# SPDX-License-Identifier: 0BSD
"""Architecture-scoped, tracked performance-report workspaces."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

from .publication import publication_absolute_paths
from .service import ReportPaths, ReportService, validate_profile_name
from .source_identity import (
    SOURCE_IDENTITY_SCHEMA,
    ReportSourceIdentityError,
    inspect_report_source,
)
from .standalone_build import compile_report

WORKSPACE_SCHEMA = "pyamplicol-performance-report-workspace-v1"
WORKSPACE_MANIFEST = "report-workspace.json"
ENVIRONMENT_TEX = "report_environment.tex"
STANDALONE_BUILDER = "build_pdf.py"
PROFILE_PARENT = Path("docs/performance_reports")


class ReportWorkspaceError(RuntimeError):
    """A report workspace could not be initialized or exported safely."""


def _canonical_json(payload: Mapping[str, object]) -> str:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def profile_docs_dir(repo_root: Path, profile: str) -> Path:
    root = repo_root.expanduser().resolve(strict=False)
    return root / PROFILE_PARENT / validate_profile_name(profile)


def _source_docs_dir(
    repo_root: Path,
    *,
    source_profile: str | None,
) -> Path:
    if source_profile is None:
        return repo_root / "docs"
    return profile_docs_dir(repo_root, source_profile)


def _copy_publication_members(
    source: Path,
    destination: Path,
) -> None:
    required = (source / "pyAmpliCol.tex", source / "result_tables.py")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ReportWorkspaceError(
            f"report source {source} is missing required files: {missing}"
        )
    results = source / "results"
    if not results.is_dir():
        raise ReportWorkspaceError(f"report source has no results directory: {source}")

    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.glob("*.tex")):
        if not path.is_file() or path.is_symlink():
            raise ReportWorkspaceError(f"unsafe TeX source member: {path}")
        shutil.copy2(path, destination / path.name)
    shutil.copy2(source / "result_tables.py", destination / "result_tables.py")

    output_results = destination / "results"
    output_results.mkdir()
    json_files = tuple(sorted(results.glob("*.json")))
    if not json_files:
        raise ReportWorkspaceError(f"report source has no JSON caches: {results}")
    for path in json_files:
        if not path.is_file() or path.is_symlink():
            raise ReportWorkspaceError(f"unsafe result-cache member: {path}")
        shutil.copy2(path, output_results / path.name)

def _install_standalone_builder(destination: Path) -> None:
    source = Path(__file__).with_name("standalone_build.py")
    if not source.is_file() or source.is_symlink():
        raise ReportWorkspaceError(
            f"standalone report builder is unavailable: {source}"
        )
    shutil.copy2(source, destination / STANDALONE_BUILDER)


def _assert_portable_results(docs_dir: Path) -> None:
    for path in sorted((docs_dir / "results").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ReportWorkspaceError(
                f"cannot inspect publication result {path}: {error}"
            ) from error
        absolute = publication_absolute_paths(payload)
        if absolute:
            raise ReportWorkspaceError(
                f"publication result retains absolute host paths: "
                f"{path.name}: {absolute[:3]}"
            )


def _workspace_readme(profile: str) -> str:
    return f"""<!-- SPDX-License-Identifier: 0BSD -->
# pyAmpliCol performance report: `{profile}`

This directory is the self-contained publication workspace for the
`{profile}` measurement environment. It contains the report's LaTeX sources,
canonical raw JSON measurements, generated table TeX, and the PDF when it has
been compiled. Large evaluator artifacts, worker logs, locks, and coordination
state are deliberately stored outside this tracked directory.

Compile this publication folder on any machine with Python and pdfLaTeX:

```bash
cd docs/performance_reports/{profile}
python3 {STANDALONE_BUILDER}
```

From a pyAmpliCol source checkout, regenerate tables and run the cache audit
with:

```bash
python3 docs/performance_reports/{profile}/result_tables.py render --compile
python3 docs/performance_reports/{profile}/result_tables.py audit
```

Before measuring a new profile, commit and push this complete initialized
workspace as the measured-source checkpoint, then perform the project's clean
build and native-install gate for that exact commit. Only from that clean,
exact-source checkout should you populate measurements through four
final-state particles with the standard five-second timing policy:

```bash
python3 docs/performance_reports/{profile}/result_tables.py populate \\
  --n-final 1..4 --missing-only --artifact-policy regenerate \\
  --workers 1 --cell-cores 1 --refresh-pdf end
```

The copied entry point selects this profile automatically. It still requires a
pyAmpliCol source checkout and installed native extension because measurements
exercise the public runtime APIs. After the final audit and visual review,
commit only the raw JSON caches, generated table and validation-summary TeX,
and reviewed PDF as the report-only descendant of the measured-source
checkpoint. Never add `.artifacts/`, worker attempts, logs, locks,
coordination state, or LaTeX auxiliary files.
"""


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "source checkout"


def _processor_description() -> str:
    commands = (
        ("sysctl", "-n", "machdep.cpu.brand_string"),
        ("sysctl", "-n", "hw.model"),
    )
    if platform.system() == "Darwin":
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            value = completed.stdout.strip()
            if completed.returncode == 0 and value:
                return value
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"model name", "Hardware"}:
                    rendered = value.strip()
                    if rendered:
                        return rendered
        except OSError:
            pass
    return platform.processor().strip()


def _environment_payload(profile: str) -> dict[str, str]:
    processor = _processor_description()
    machine = platform.machine().strip() or "unknown architecture"
    platform_parts = [
        f"{platform.system()} {platform.release()}".strip(),
        machine,
    ]
    if processor and processor.lower() != machine.lower():
        platform_parts.append(processor)
    return {
        "profile": profile,
        "platform": "; ".join(platform_parts),
        "machine": machine,
        "processor": processor,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyamplicol": _package_version("pyamplicol"),
        "numpy": _package_version("numpy"),
    }


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _environment_tex(environment: Mapping[str, str]) -> str:
    toolchain = (
        f"{environment['python_implementation']} "
        f"{environment['python']}; pyAmpliCol {environment['pyamplicol']}; "
        f"NumPy {environment['numpy']}"
    )
    return (
        "% SPDX-License-Identifier: 0BSD\n"
        "% Generated architecture-profile metadata; do not edit by hand.\n"
        rf"\renewcommand{{\ReportProfileName}}{{"
        f"{_tex_escape(environment['profile'])}}}\n"
        rf"\renewcommand{{\ReportPlatformSummary}}{{"
        f"{_tex_escape(environment['platform'])}}}\n"
        rf"\renewcommand{{\ReportToolchainSummary}}{{"
        f"{_tex_escape(toolchain)}}}\n"
        "\\renewcommand{\\ReportEditionStatement}{"
        "This document is an architecture-specific edition.}\n"
    )


def _workspace_manifest(
    repo_root: Path,
    *,
    profile: str,
    source_profile: str | None,
    reset_measurements: bool,
    environment: Mapping[str, str],
) -> dict[str, object]:
    paths = ReportPaths.from_repo(repo_root, profile=profile)
    try:
        source = inspect_report_source(repo_root)
        source_record: dict[str, object] = {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "revision": source.revision,
            "tree": source.tree,
            "clean": source.eligible,
            "dirty_paths": list(source.dirty_paths),
        }
    except ReportSourceIdentityError:
        source_record = {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "revision": "unknown",
            "tree": "unknown",
            "clean": False,
            "dirty_paths": [],
        }
    return {
        "schema": WORKSPACE_SCHEMA,
        "profile": profile,
        "report_source_revision": source_record["revision"],
        "report_source_tree": source_record["tree"],
        "initialized_source_identity": source_record,
        "initialized_from": (
            "docs"
            if source_profile is None
            else f"docs/performance_reports/{source_profile}"
        ),
        "measurement_state": "reset" if reset_measurements else "copied",
        "document": "pyAmpliCol.tex",
        "environment_tex": ENVIRONMENT_TEX,
        "environment": dict(environment),
        "raw_results": "results",
        "artifact_root": paths.artifact_root.relative_to(repo_root).as_posix(),
        "coordination_root": paths.coordination_root.relative_to(repo_root).as_posix(),
        "tracked_content": [
            "*.tex",
            "results/*.json",
            "README.md",
            WORKSPACE_MANIFEST,
            STANDALONE_BUILDER,
            "pyAmpliCol.pdf (when compiled and reviewed)",
        ],
        "excluded_content": [
            ".artifacts evaluator attempts",
            "worker logs",
            "locks and coordination state",
            "LaTeX auxiliary files",
        ],
    }


def initialize_profile(
    repo_root: Path,
    profile: str,
    *,
    source_profile: str | None = None,
    reset_measurements: bool = False,
) -> Path:
    """Create a new tracked report profile atomically.

    The destination must not already exist. ``reset_measurements`` rebuilds the
    caches and generated table inputs from the catalog and never carries a PDF
    from the source profile.
    """

    root = repo_root.expanduser().resolve(strict=False)
    validated = validate_profile_name(profile)
    if source_profile is not None:
        source_profile = validate_profile_name(source_profile)
        if source_profile == validated:
            raise ValueError("source and destination report profiles must differ")
    source = _source_docs_dir(root, source_profile=source_profile)
    target = profile_docs_dir(root, validated)
    if target.exists():
        raise ReportWorkspaceError(f"report profile already exists: {target}")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{validated}.init-{uuid.uuid4().hex}"
    state_root = (
        root
        / ".artifacts"
        / "performance-report-workspace-init"
        / uuid.uuid4().hex
    )
    try:
        source_paths = ReportPaths.from_repo(root, profile=source_profile)
        source_service = ReportService(source_paths)
        with source_service.store.named_lock("report-writer"):
            source_service.audit()
            _copy_publication_members(source, staging)
        _install_standalone_builder(staging)
        target_paths = ReportPaths.from_repo(root, profile=validated)
        if reset_measurements and (
            target_paths.artifact_root.exists()
            and any(target_paths.artifact_root.iterdir())
        ):
            raise ReportWorkspaceError(
                "reset profile artifact root already contains local state: "
                f"{target_paths.artifact_root}"
            )
        paths = ReportPaths.from_repo(
            root,
            docs_dir=staging,
            artifact_root=(
                target_paths.artifact_root
                if reset_measurements
                else source_paths.artifact_root
            ),
            coordination_root=state_root / "coordination",
        )
        service = ReportService(paths)
        if reset_measurements:
            service.publish(reset=True, merge_artifacts=False)
        else:
            service.publish(reset=False, merge_artifacts=False)
        service.audit()
        _assert_portable_results(staging)

        environment = _environment_payload(validated)
        (staging / ENVIRONMENT_TEX).write_text(
            _environment_tex(environment),
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _workspace_readme(validated),
            encoding="utf-8",
        )
        manifest = _workspace_manifest(
            root,
            profile=validated,
            source_profile=source_profile,
            reset_measurements=reset_measurements,
            environment=environment,
        )
        (staging / WORKSPACE_MANIFEST).write_text(
            _canonical_json(manifest),
            encoding="ascii",
        )
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(state_root, ignore_errors=True)
    return target


def _validate_workspace(
    repo_root: Path,
    profile: str,
    *,
    service: ReportService | None = None,
) -> Path:
    root = repo_root.expanduser().resolve(strict=False)
    path = profile_docs_dir(root, profile)
    manifest_path = path / WORKSPACE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportWorkspaceError(
            f"invalid or missing report workspace manifest: {manifest_path}"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != WORKSPACE_SCHEMA
        or manifest.get("profile") != profile
    ):
        raise ReportWorkspaceError(
            f"workspace manifest does not identify profile {profile!r}"
        )
    active = service or ReportService(
        ReportPaths.from_repo(root, profile=profile)
    )
    active.audit()
    return path


def export_profile(
    repo_root: Path,
    profile: str,
    destination: Path,
    *,
    include_pdf: bool = True,
) -> Path:
    """Export publication inputs without evaluator artifacts or local state."""

    root = repo_root.expanduser().resolve(strict=False)
    validated = validate_profile_name(profile)
    profile_paths = ReportPaths.from_repo(root, profile=validated)
    source_service = ReportService(profile_paths)
    target = destination.expanduser().resolve(strict=False)
    if target.exists():
        raise ReportWorkspaceError(f"export destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.export-{uuid.uuid4().hex}"
    state_root = (
        root
        / ".artifacts"
        / "performance-report-workspace-export"
        / uuid.uuid4().hex
    )
    try:
        with source_service.store.named_lock("report-writer"):
            source = _validate_workspace(
                root,
                validated,
                service=source_service,
            )
            _copy_publication_members(source, staging)
            for name in ("README.md", WORKSPACE_MANIFEST):
                shutil.copy2(source / name, staging / name)
        _install_standalone_builder(staging)
        service = ReportService(
            ReportPaths.from_repo(
                root,
                docs_dir=staging,
                artifact_root=profile_paths.artifact_root,
                coordination_root=state_root,
            )
        )
        service.publish(reset=False, merge_artifacts=False)
        service.audit()
        _assert_portable_results(staging)
        if include_pdf:
            compile_report(staging)
            for suffix in (
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
            ):
                (staging / f"pyAmpliCol{suffix}").unlink(missing_ok=True)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(state_root, ignore_errors=True)
    return target


__all__ = [
    "ENVIRONMENT_TEX",
    "PROFILE_PARENT",
    "STANDALONE_BUILDER",
    "WORKSPACE_MANIFEST",
    "WORKSPACE_SCHEMA",
    "ReportWorkspaceError",
    "export_profile",
    "initialize_profile",
    "profile_docs_dir",
]
