# SPDX-License-Identifier: 0BSD
"""Architecture-scoped, tracked performance-report workspaces."""

from __future__ import annotations

import importlib
import json
import platform
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from .publication import publication_absolute_paths
from .service import ReportPaths, ReportService, validate_profile_name
from .source_identity import (
    SOURCE_IDENTITY_SCHEMA,
    ReportSourceIdentityError,
    inspect_report_source,
)
from .standalone_build import compile_report

WORKSPACE_SCHEMA = "pyamplicol-performance-report-workspace-v2"
WORKSPACE_MANIFEST = "report-workspace.json"
ENVIRONMENT_SCHEMA = "pyamplicol-performance-report-environment-v1"
ENVIRONMENT_JSON = "report_environment.json"
ENVIRONMENT_TEX = "report_environment.tex"
STANDALONE_BUILDER = "build_pdf.py"
PROFILE_PARENT = Path("docs/performance_reports")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PENDING_RUNTIME = "pending exact-source runtime authentication"


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
    environment_json = source / ENVIRONMENT_JSON
    if environment_json.exists():
        if not environment_json.is_file() or environment_json.is_symlink():
            raise ReportWorkspaceError(
                f"unsafe environment metadata member: {environment_json}"
            )
        shutil.copy2(environment_json, destination / ENVIRONMENT_JSON)
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
    campaign = "\n".join(
        line
        for multiplicity in range(1, 5)
        for line in (
            f"python3 docs/performance_reports/{profile}/result_tables.py populate \\",
            f"  --n-final {multiplicity} --missing-only --artifact-policy reuse \\",
            "  --workers 1 --cell-cores 1 --target-runtime 5 --refresh-pdf end",
            f"python3 docs/performance_reports/{profile}/result_tables.py audit",
            "",
        )
    ).rstrip()
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

Before measuring, commit this complete initialized workspace, save and verify
its exact identity, and push the measured-source checkpoint:

```bash
git add docs/performance_reports/{profile}
git commit -m "Initialize {profile} performance report"
MEASURED_SOURCE_REVISION="$(git rev-parse HEAD)"
test "$(git rev-parse HEAD)" = "$MEASURED_SOURCE_REVISION" &&
git push origin HEAD
```

That first push must complete before any build or measurement. Keep
`MEASURED_SOURCE_REVISION` unchanged for the rest of the campaign. Perform the
project's clean build and native-install gate for that exact commit, then
authenticate and record the installed measurement runtime:

```bash
python3 docs/performance_reports/{profile}/result_tables.py \\
  refresh-profile-environment \\
  --expected-source-revision "$MEASURED_SOURCE_REVISION"
```

Initialization deliberately labels the runtime metadata as pending: the exact
installed runtime does not exist until after the checkpoint build. The refresh
command fails unless the source checkout and native runtime both match the
requested full commit SHA. It changes only generated publication metadata, so
the checkpoint remains the evaluator source identity.

Populate each multiplicity separately with the standard five-second timing
policy:

```bash
{campaign}
```

After each populate command, inspect its audit result and visually review the
newly refreshed PDF before continuing to the next multiplicity. Do not replace
these four invocations with one combined `1..4` campaign.

After all four audits and visual reviews pass, stage only the allowed
publication outputs, create the report-only descendant, save its identity, and
run the complete profile-scoped audit:

```bash
git add \\
  docs/performance_reports/{profile}/report_environment.json \\
  docs/performance_reports/{profile}/report_environment.tex \\
  docs/performance_reports/{profile}/results/*.json \\
  docs/performance_reports/{profile}/result_*_table.tex \\
  docs/performance_reports/{profile}/result_validation_summary.tex \\
  docs/performance_reports/{profile}/pyAmpliCol.pdf
git diff --cached --check
git commit -m "Publish {profile} performance report"
PUBLICATION_REVISION="$(git rev-parse HEAD)"
python3 docs/performance_reports/{profile}/result_tables.py final-audit \\
  --expected-source-revision "$MEASURED_SOURCE_REVISION" \\
  --publication-revision "$PUBLICATION_REVISION" &&
git push origin HEAD
```

The copied entry point selects this profile automatically. It still requires a
pyAmpliCol source checkout and installed native extension because measurements
exercise the public runtime APIs. Never stage profile prose, entry points,
manifests, evaluator source, `.artifacts/`, worker attempts, logs, locks,
coordination state, or LaTeX auxiliary files. Push the publication commit only
after `final-audit` succeeds.
"""


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


def _host_environment_payload(profile: str) -> dict[str, str]:
    processor = _processor_description()
    machine = platform.machine().strip() or "unknown architecture"
    platform_parts = [
        f"{platform.system()} {platform.release()}".strip(),
        machine,
    ]
    if processor and processor.lower() != machine.lower():
        platform_parts.append(processor)
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "profile": profile,
        "platform": "; ".join(platform_parts),
        "machine": machine,
        "processor": processor,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _pending_environment_payload(profile: str) -> dict[str, str]:
    return {
        **_host_environment_payload(profile),
        "status": "pending_exact_runtime",
        "source_revision": "pending",
        "pyamplicol": _PENDING_RUNTIME,
        "numpy": _PENDING_RUNTIME,
        "native_target": _PENDING_RUNTIME,
        "native_cpu_features": _PENDING_RUNTIME,
        "native_build_inputs_sha256": "pending",
        "native_extension_sha256": "pending",
        "python_package_tree_sha256": "pending",
        "candidate_fingerprint": "pending",
    }


def _required_text(
    value: object,
    context: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportWorkspaceError(f"{context} must be a non-empty string")
    return value


def _authenticated_environment_payload(
    profile: str,
    *,
    expected_source_revision: str,
    active_runtime: Mapping[str, object],
) -> dict[str, str]:
    package_version = _required_text(
        active_runtime.get("package_version"),
        "active runtime package_version",
    )
    native_target = active_runtime.get("native_target")
    if not isinstance(native_target, Mapping):
        raise ReportWorkspaceError("active runtime native_target must be an object")
    target_triple = _required_text(
        native_target.get("triple"),
        "active runtime native target triple",
    )
    raw_features = native_target.get("cpu_features")
    if not isinstance(raw_features, list) or not all(
        isinstance(feature, str) and feature for feature in raw_features
    ):
        raise ReportWorkspaceError(
            "active runtime native CPU features must be a string list"
        )
    native_digest = _required_text(
        active_runtime.get("native_build_inputs_sha256"),
        "active runtime native build-input digest",
    )
    if re.fullmatch(r"[0-9a-f]{64}", native_digest) is None:
        raise ReportWorkspaceError(
            "active runtime native build-input digest is not SHA-256"
        )
    native_extension = active_runtime.get("native_extension")
    python_package_tree = active_runtime.get("python_package_tree")
    candidate_identity = active_runtime.get("candidate_build_identity")
    if (
        not isinstance(native_extension, Mapping)
        or not isinstance(python_package_tree, Mapping)
        or not isinstance(candidate_identity, Mapping)
    ):
        raise ReportWorkspaceError(
            "active runtime file and candidate identities must be objects"
        )
    native_extension_digest = _required_text(
        native_extension.get("sha256"),
        "active runtime native extension digest",
    )
    package_tree_digest = _required_text(
        python_package_tree.get("sha256"),
        "active runtime Python package-tree digest",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in (native_extension_digest, package_tree_digest)
    ):
        raise ReportWorkspaceError(
            "active runtime file identity digest is not SHA-256"
        )
    candidate_fingerprint = _required_text(
        candidate_identity.get("candidate_fingerprint"),
        "active runtime candidate fingerprint",
    )
    numpy = importlib.import_module("numpy")
    numpy_version = _required_text(
        getattr(numpy, "__version__", None),
        "active NumPy version",
    )
    return {
        **_host_environment_payload(profile),
        "status": "authenticated",
        "source_revision": expected_source_revision,
        "pyamplicol": package_version,
        "numpy": numpy_version,
        "native_target": target_triple,
        "native_cpu_features": ", ".join(raw_features) or "baseline",
        "native_build_inputs_sha256": native_digest,
        "native_extension_sha256": native_extension_digest,
        "python_package_tree_sha256": package_tree_digest,
        "candidate_fingerprint": candidate_fingerprint,
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
        "Exact measurement-runtime metadata pending authenticated "
        "post-checkpoint build."
        if environment["status"] == "pending_exact_runtime"
        else (
            f"{environment['python_implementation']} "
            f"{environment['python']}; pyAmpliCol {environment['pyamplicol']}; "
            f"NumPy {environment['numpy']}; native {environment['native_target']}"
        )
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
        "environment_json": ENVIRONMENT_JSON,
        "environment_tex": ENVIRONMENT_TEX,
        "initialized_environment": dict(environment),
        "measurement_environment_contract": (
            "refresh generated environment JSON/TeX after the exact-source "
            "clean build and before population"
        ),
        "raw_results": "results",
        "artifact_root": paths.artifact_root.relative_to(repo_root).as_posix(),
        "coordination_root": paths.coordination_root.relative_to(repo_root).as_posix(),
        "tracked_content": [
            "*.tex",
            ENVIRONMENT_JSON,
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

        environment = _pending_environment_payload(validated)
        (staging / ENVIRONMENT_JSON).write_text(
            _canonical_json(environment),
            encoding="ascii",
        )
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


def _read_environment_payload(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportWorkspaceError(
            f"invalid or missing report environment metadata: {path}"
        ) from error
    required = {
        "schema",
        "profile",
        "status",
        "source_revision",
        "platform",
        "machine",
        "processor",
        "python",
        "python_implementation",
        "pyamplicol",
        "numpy",
        "native_target",
        "native_cpu_features",
        "native_build_inputs_sha256",
        "native_extension_sha256",
        "python_package_tree_sha256",
        "candidate_fingerprint",
    }
    if not isinstance(raw, dict) or set(raw) != required or not all(
        isinstance(value, str) for value in raw.values()
    ):
        raise ReportWorkspaceError(
            f"report environment metadata has an invalid shape: {path}"
        )
    return raw


def require_authenticated_profile_environment(
    repo_root: Path,
    profile: str,
    *,
    expected_source_revision: str,
) -> dict[str, str]:
    """Require exact-source runtime metadata before profile measurement."""

    validated = validate_profile_name(profile)
    if _GIT_SHA_RE.fullmatch(expected_source_revision) is None:
        raise ReportWorkspaceError(
            "profile environment source revision must be a full Git SHA"
        )
    path = profile_docs_dir(repo_root, validated)
    environment = _read_environment_payload(path / ENVIRONMENT_JSON)
    if (
        environment["schema"] != ENVIRONMENT_SCHEMA
        or environment["profile"] != validated
        or environment["status"] != "authenticated"
        or environment["source_revision"] != expected_source_revision
    ):
        raise ReportWorkspaceError(
            "profile environment is not authenticated for measurement source "
            f"{expected_source_revision}; run refresh-profile-environment after "
            "the exact-source clean build"
        )
    expected_tex = _environment_tex(environment)
    try:
        observed_tex = (path / ENVIRONMENT_TEX).read_text(encoding="utf-8")
    except OSError as error:
        raise ReportWorkspaceError(
            f"profile environment TeX is unavailable: {path / ENVIRONMENT_TEX}"
        ) from error
    if observed_tex != expected_tex:
        raise ReportWorkspaceError(
            "profile environment JSON and generated TeX do not match"
        )
    return environment


def refresh_profile_environment(
    repo_root: Path,
    profile: str,
    *,
    expected_source_revision: str,
    runtime_auditor: Callable[[str, Path], Mapping[str, object]] | None = None,
) -> dict[str, str]:
    """Authenticate and record the exact installed measurement runtime."""

    root = repo_root.expanduser().resolve(strict=False)
    validated = validate_profile_name(profile)
    if _GIT_SHA_RE.fullmatch(expected_source_revision) is None:
        raise ReportWorkspaceError(
            "profile environment source revision must be a full Git SHA"
        )
    _validate_workspace(root, validated)
    source = inspect_report_source(root)
    if not source.eligible or source.revision != expected_source_revision:
        raise ReportWorkspaceError(
            "profile environment refresh requires a clean evaluator source "
            "checkout at the requested full Git SHA"
        )
    if runtime_auditor is None:
        from .final_audit import _audit_active_runtime

        runtime_auditor = _audit_active_runtime
    try:
        active_runtime = runtime_auditor(expected_source_revision, root)
    except Exception as error:
        raise ReportWorkspaceError(
            "installed pyAmpliCol runtime is not authenticated for the "
            "requested measurement source"
        ) from error
    environment = _authenticated_environment_payload(
        validated,
        expected_source_revision=expected_source_revision,
        active_runtime=active_runtime,
    )
    profile_dir = profile_docs_dir(root, validated)
    json_path = profile_dir / ENVIRONMENT_JSON
    tex_path = profile_dir / ENVIRONMENT_TEX
    temporary_json = json_path.with_name(f".{json_path.name}.{uuid.uuid4().hex}")
    temporary_tex = tex_path.with_name(f".{tex_path.name}.{uuid.uuid4().hex}")
    try:
        temporary_json.write_text(_canonical_json(environment), encoding="ascii")
        temporary_tex.write_text(_environment_tex(environment), encoding="utf-8")
        temporary_json.replace(json_path)
        temporary_tex.replace(tex_path)
    finally:
        temporary_json.unlink(missing_ok=True)
        temporary_tex.unlink(missing_ok=True)
    return require_authenticated_profile_environment(
        root,
        validated,
        expected_source_revision=expected_source_revision,
    )


def require_active_profile_environment(
    repo_root: Path,
    profile: str,
    *,
    expected_source_revision: str,
    runtime_auditor: Callable[[str, Path], Mapping[str, object]] | None = None,
) -> dict[str, str]:
    """Require recorded metadata to equal the currently active exact runtime."""

    root = repo_root.expanduser().resolve(strict=False)
    validated = validate_profile_name(profile)
    recorded = require_authenticated_profile_environment(
        root,
        validated,
        expected_source_revision=expected_source_revision,
    )
    if runtime_auditor is None:
        from .final_audit import _audit_active_runtime

        runtime_auditor = _audit_active_runtime
    try:
        active_runtime = runtime_auditor(expected_source_revision, root)
    except Exception as error:
        raise ReportWorkspaceError(
            "active pyAmpliCol runtime is not authenticated for the requested "
            "measurement source"
        ) from error
    active = _authenticated_environment_payload(
        validated,
        expected_source_revision=expected_source_revision,
        active_runtime=active_runtime,
    )
    if active != recorded:
        raise ReportWorkspaceError(
            "active measurement runtime differs from the authenticated profile "
            "environment; rerun refresh-profile-environment"
        )
    return recorded


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
        or manifest.get("environment_json") != ENVIRONMENT_JSON
        or manifest.get("environment_tex") != ENVIRONMENT_TEX
    ):
        raise ReportWorkspaceError(
            f"workspace manifest does not identify profile {profile!r}"
        )
    environment = _read_environment_payload(path / ENVIRONMENT_JSON)
    if (
        environment["schema"] != ENVIRONMENT_SCHEMA
        or environment["profile"] != profile
        or environment["status"] not in {"pending_exact_runtime", "authenticated"}
        or (
            environment["status"] == "pending_exact_runtime"
            and environment["source_revision"] != "pending"
        )
        or (
            environment["status"] == "authenticated"
            and _GIT_SHA_RE.fullmatch(environment["source_revision"]) is None
        )
    ):
        raise ReportWorkspaceError(
            f"workspace environment does not identify profile {profile!r}"
        )
    try:
        environment_tex = (path / ENVIRONMENT_TEX).read_text(encoding="utf-8")
    except OSError as error:
        raise ReportWorkspaceError(
            f"invalid or missing workspace environment TeX: {path / ENVIRONMENT_TEX}"
        ) from error
    if environment_tex != _environment_tex(environment):
        raise ReportWorkspaceError(
            "workspace environment JSON and generated TeX do not match"
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
    "ENVIRONMENT_JSON",
    "ENVIRONMENT_SCHEMA",
    "ENVIRONMENT_TEX",
    "PROFILE_PARENT",
    "STANDALONE_BUILDER",
    "WORKSPACE_MANIFEST",
    "WORKSPACE_SCHEMA",
    "ReportWorkspaceError",
    "export_profile",
    "initialize_profile",
    "profile_docs_dir",
    "refresh_profile_environment",
    "require_active_profile_environment",
    "require_authenticated_profile_environment",
]
