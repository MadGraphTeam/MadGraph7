# SPDX-License-Identifier: 0BSD
"""Command-line entry point for the three-mode performance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .artifacts import ArtifactStoreError
from .cache import validate_measurement
from .campaign_policy import (
    MACBOOK_M3_POLICY_NAME,
    STRICT_POLICY,
    STRICT_POLICY_NAME,
    X86_EPYC_POLICY_NAME,
    PolicyMeasurementState,
    validate_policy_measurement,
)
from .catalog import REPORT_CATALOG
from .measurement_lineage import (
    CLASS_C_HZZ_IMPACT,
    CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
    MeasurementLineageError,
    audit_measurement_lineage,
    class_c_pending_path,
    finalize_class_c_bridge,
    load_and_audit_measurement_lineage,
    load_measurement_lineage,
    prepare_class_c_bridge,
)
from .models import (
    Accuracy,
    ArtifactPolicy,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .prepared import ensure_report_prepared_model
from .publisher import (
    DEFAULT_EXPECTED_PAGE_COUNT,
    DEFAULT_PDF_TIMEOUT_SECONDS,
    DEFAULT_PUBLICATION_INTERVAL_SECONDS,
    run_publisher,
    validate_published_snapshot,
)
from .runner import DEFAULT_TARGET_RUNTIME_SECONDS
from .scheduler import (
    CampaignScheduler,
    CampaignSettings,
    CellSelection,
    plan_campaign,
    select_cells,
)
from .service import (
    CANONICAL_REPORT_ENTRYPOINT,
    ReportPaths,
    ReportService,
    validate_profile_name,
)
from .source_identity import require_eligible_report_source
from .standalone_build import StandaloneBuildError, validate_latex_log
from .worker import write_cell_result
from .workspace import (
    ReportWorkspaceError,
    export_profile,
    initialize_profile,
    load_profile_campaign_policy,
    refresh_profile_environment,
    require_active_profile_environment,
)

_PINNED_EPYC_ORPHAN_CELL_ID = (
    "reference-amplicol-lc-n8-gg-gluons-selected-flow"
)
_PINNED_EPYC_ORPHAN_ATTEMPT_ID = "83e5c9c7-dbf6-4d61-b724-f4580df2cfa3"
_PINNED_EPYC_ORPHAN_WORKER_SHA256 = (
    "5f3a42f9e3d034efedd8b670e7acbf2b54a427449106dbabc29050f3d93afbe6"
)
_PINNED_EPYC_ORPHAN_RESOURCES = {
    "monitor": "external-cell-supervisor",
    "peak_rss_gib": None,
}


def _is_pinned_epyc_orphan_without_rss(
    *,
    profile: str,
    cell_id: str,
    attempt_id: str,
    worker_result_sha256: str,
    result: Mapping[str, object],
) -> bool:
    """Match the sole authenticated legacy worker result lacking RSS."""

    resources = result.get("resources")
    return (
        profile == "x86_EPYC"
        and cell_id == _PINNED_EPYC_ORPHAN_CELL_ID
        and attempt_id == _PINNED_EPYC_ORPHAN_ATTEMPT_ID
        and worker_result_sha256 == _PINNED_EPYC_ORPHAN_WORKER_SHA256
        and isinstance(resources, Mapping)
        and dict(resources) == _PINNED_EPYC_ORPHAN_RESOURCES
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and populate the pyAmpliCol performance report.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--report-profile",
        type=validate_profile_name,
        help=(
            "use docs/performance_reports/PROFILE with isolated evaluator "
            "artifacts and coordination state"
        ),
    )
    parser.add_argument("--docs-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--artifact-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--coordination-root", type=Path, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate", "validate canonical caches and rendered tables"),
        ("audit", "verify exact cache coverage and checked-in table rendering"),
        ("reset", "reset all report measurements to N/A"),
        ("render", "merge immutable results and render tables"),
        ("recover", "recover immutable worker results and render tables"),
    ):
        item = subparsers.add_parser(command, help=help_text)
        if command in {"reset", "render", "recover"}:
            item.add_argument(
                "--compile",
                action="store_true",
                help="compile pyAmpliCol.pdf after publishing tables",
            )

    publish_snapshot = subparsers.add_parser(
        "publish-snapshot",
        help="copy, validate, render, and compile one current-cache snapshot",
    )
    publish_snapshot.add_argument("--watch", action="store_true")
    publish_snapshot.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_PUBLICATION_INTERVAL_SECONDS,
    )
    publish_snapshot.add_argument(
        "--pdf-timeout-seconds",
        type=float,
        default=DEFAULT_PDF_TIMEOUT_SECONDS,
    )
    publish_snapshot.add_argument(
        "--expected-page-count",
        type=int,
        default=DEFAULT_EXPECTED_PAGE_COUNT,
    )
    subparsers.add_parser(
        "validate-snapshot",
        help="validate the published cache/table/PDF snapshot identities",
    )

    initialize = subparsers.add_parser(
        "init-profile",
        help="create an isolated architecture-specific report workspace",
    )
    initialize.add_argument("profile", type=validate_profile_name)
    initialize.add_argument(
        "--source-profile",
        type=validate_profile_name,
        help="copy publication inputs from another report profile instead of docs/",
    )
    initialize.add_argument(
        "--reset-measurements",
        action="store_true",
        help="replace copied measurements with canonical N/A caches and tables",
    )
    initialize.add_argument(
        "--measurement-policy",
        choices=(
            STRICT_POLICY_NAME,
            MACBOOK_M3_POLICY_NAME,
            X86_EPYC_POLICY_NAME,
        ),
        help=(
            "bind a canonical completion/resource policy; defaults to the "
            "matching architecture policy for macbook_M3 or x86_EPYC and "
            "strict completion for every other profile"
        ),
    )

    refresh_environment = subparsers.add_parser(
        "refresh-profile-environment",
        help="authenticate and record the exact installed measurement runtime",
    )
    refresh_environment.add_argument("--expected-source-revision", required=True)

    prepare_bridge = subparsers.add_parser(
        "prepare-class-c-bridge",
        help="snapshot an exact frozen campaign before one bounded source correction",
    )
    prepare_bridge.add_argument("--ancestor-revision", required=True)
    prepare_bridge.add_argument("--descendant-revision", required=True)
    prepare_bridge.add_argument(
        "--impact",
        choices=(
            CLASS_C_HZZ_IMPACT,
            CLASS_C_RECURRENCE_SUMMARY_CAP_IMPACT,
        ),
        required=True,
    )

    finalize_bridge = subparsers.add_parser(
        "finalize-class-c-bridge",
        help="authenticate the corrected runtime and publish its source lineage",
    )
    finalize_bridge.add_argument("--ancestor-revision", required=True)
    finalize_bridge.add_argument("--descendant-revision", required=True)

    audit_bridge = subparsers.add_parser(
        "audit-source-bridge",
        help="audit one finalized mixed-source profile without changing it",
    )
    audit_bridge.add_argument("--expected-active-source-revision", required=True)

    seal_orphan = subparsers.add_parser(
        "seal-existing-worker-result",
        help="authenticate and seal one completed controller-orphaned worker result",
    )
    seal_orphan.add_argument("--cell-id", required=True)
    seal_orphan.add_argument("--attempt-id", required=True)
    seal_orphan.add_argument("--worker-result-sha256", required=True)
    seal_orphan.add_argument(
        "--artifact-policy",
        choices=tuple(policy.value for policy in ArtifactPolicy),
        required=True,
    )
    seal_orphan.add_argument("--expected-source-revision", required=True)

    export = subparsers.add_parser(
        "export-profile",
        help="copy a tracked report workspace without evaluator artifacts",
    )
    export.add_argument("profile", type=validate_profile_name)
    export.add_argument("destination", type=Path)
    export.add_argument(
        "--include-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    final_audit = subparsers.add_parser(
        "final-audit",
        help=(
            "run the measured-SHA/runtime and report-only publication "
            "numerical, artifact, and PDF gate"
        ),
    )
    final_audit.add_argument("--expected-source-revision", required=True)
    final_audit.add_argument(
        "--publication-revision",
        help="require the clean publication checkout to equal this full Git SHA",
    )
    final_audit.add_argument("--max-n-final", type=int, default=9)
    final_audit.add_argument("--expected-cell-count", type=int, default=1646)
    final_audit.add_argument(
        "--structural-only",
        action="store_true",
        help="authenticate records and artifacts without numerical replay",
    )

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--cell-id", required=True)
    worker.add_argument("--attempt-root", type=Path, required=True)
    worker.add_argument("--result-json", type=Path, required=True)
    worker.add_argument("--log-path", type=Path)
    worker.add_argument("--baseline-json", type=Path)
    worker.add_argument(
        "--peer-json",
        action="append",
        nargs=2,
        metavar=("CELL_ID", "PATH"),
        default=[],
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--prepared-model", type=Path)
    worker.add_argument("--reused-measurement-json", type=Path)
    worker.add_argument("--legacy-repository", type=Path, help=argparse.SUPPRESS)
    worker.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_TARGET_RUNTIME_SECONDS,
    )
    worker.add_argument("--batch-size", type=int, default=128)
    worker.add_argument("--cell-cores", type=int, default=1)
    worker.add_argument("--phase-state-path", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--phase-state-run-id", help=argparse.SUPPRESS)
    worker.add_argument(
        "--phase-state-authentication-key",
        help=argparse.SUPPRESS,
    )

    prepare = subparsers.add_parser("_prepare", help=argparse.SUPPRESS)
    prepare.add_argument(
        "--model",
        required=True,
        choices=(ModelKey.BUILTIN_SM.value, ModelKey.UFO_SM.value),
    )
    prepare.add_argument("--result-json", type=Path, required=True)
    prepare.add_argument("--cell-cores", type=int, default=1)

    populate = subparsers.add_parser(
        "populate",
        help="run selected cells through isolated direct-API workers",
    )
    populate.add_argument("--dataset", action="append", default=[])
    populate.add_argument(
        "--mode",
        action="append",
        choices=tuple(mode.value for mode in ExecutionMode),
        default=[],
    )
    populate.add_argument(
        "--model",
        action="append",
        choices=tuple(model.value for model in ModelKey),
        default=[],
    )
    populate.add_argument(
        "--accuracy",
        action="append",
        choices=tuple(accuracy.value for accuracy in Accuracy),
        default=[],
    )
    populate.add_argument("--process-key", action="append", default=[])
    populate.add_argument("--process", action="append", default=[])
    populate.add_argument("--n-final", action="append", default=[])
    populate.add_argument("--variant", action="append", default=[])
    populate.add_argument(
        "--workload",
        choices=("selected-flow", "all-flow", "both", "contracted"),
    )
    populate.add_argument("--cell-id", action="append", default=[])
    populate.add_argument(
        "--exclude-cell-id",
        action="append",
        default=[],
        help=(
            "omit a held cell and any selected cell whose unresolved "
            "dependency closure reaches it"
        ),
    )
    populate.add_argument("--missing-only", action="store_true")
    populate.add_argument("--rerun", action="store_true")
    populate.add_argument(
        "--artifact-policy",
        choices=tuple(policy.value for policy in ArtifactPolicy),
        default=ArtifactPolicy.REGENERATE.value,
    )
    populate.add_argument("--workers", type=int, default=1)
    populate.add_argument("--cell-cores", type=int, default=1)
    populate.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_TARGET_RUNTIME_SECONDS,
    )
    populate.add_argument("--batch-size", type=int, default=128)
    populate.add_argument("--timeout-seconds", type=float)
    populate.add_argument(
        "--generation-time-limit-seconds",
        type=float,
        help="limit only the authenticated Generator.generate phase",
    )
    populate.add_argument("--max-ram-gib", type=float)
    populate.add_argument(
        "--max-ram-gb",
        type=float,
        help="decimal GB ceiling for each worker process tree",
    )
    populate.add_argument("--campaign-max-ram-gib", type=float)
    populate.add_argument(
        "--campaign-max-ram-gb",
        type=float,
        help="decimal GB ceiling for all concurrent worker process trees",
    )
    populate.add_argument(
        "--limit-gib",
        type=float,
        help="compatibility alias for --campaign-max-ram-gib",
    )
    populate.add_argument("--allow-symbolica-parallel", action="store_true")
    populate.add_argument(
        "--fast-lineage",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    populate.add_argument("--dry-run", action="store_true")
    populate.add_argument(
        "--refresh-pdf",
        choices=("never", "end"),
        default="never",
    )
    return parser


def _multiplicities(values: Sequence[str]) -> frozenset[int]:
    selected: set[int] = set()
    for value in values:
        for item in value.split(","):
            text = item.strip()
            if not text:
                continue
            if ".." in text:
                start_text, stop_text = text.split("..", 1)
                start, stop = int(start_text), int(stop_text)
                if start > stop:
                    raise ValueError("n-final range start must not exceed its end")
                selected.update(range(start, stop + 1))
            else:
                selected.add(int(text))
    if any(value < 1 for value in selected):
        raise ValueError("n-final values must be positive")
    return frozenset(selected)


def _workloads(value: str | None) -> frozenset[Workload]:
    if value is None:
        return frozenset()
    if value == "both":
        return frozenset({Workload.SELECTED_FLOW, Workload.ALL_FLOW})
    return frozenset({Workload(value)})


def _selection(args: argparse.Namespace) -> CellSelection:
    return CellSelection(
        datasets=frozenset(args.dataset),
        modes=frozenset(ExecutionMode(value) for value in args.mode),
        models=frozenset(ModelKey(value) for value in args.model),
        accuracies=frozenset(Accuracy(value) for value in args.accuracy),
        process_keys=frozenset(args.process_key),
        processes=frozenset(args.process),
        multiplicities=_multiplicities(args.n_final),
        variants=frozenset(args.variant),
        workloads=_workloads(args.workload),
        cell_ids=frozenset(args.cell_id),
    )


def _gib_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    if value <= 0.0:
        raise ValueError("RAM limits must be positive")
    return int(value * 1024**3)


def _gb_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    if value <= 0.0:
        raise ValueError("RAM limits must be positive")
    return int(value * 1_000_000_000)


def _compile_pdf(service: ReportService) -> Path:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FileNotFoundError("latexmk is required for --compile")
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C", "LC_CTYPE": "C"})
    build_root = service.paths.artifact_root / "pdf-builds"
    build_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="report-pdf-", dir=build_root))
    try:
        build_docs = staging / "docs"
        shutil.copytree(
            service.paths.docs_dir,
            build_docs,
            ignore=shutil.ignore_patterns(
                "*.aux",
                "*.fdb_latexmk",
                "*.fls",
                "*.log",
                "*.out",
                "*.toc",
                "pyAmpliCol.pdf",
                ".coordination",
            ),
        )
        completed = subprocess.run(
            (
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "pyAmpliCol.tex",
            ),
            cwd=build_docs,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            tail = "\n".join(
                (*completed.stdout.splitlines(), *completed.stderr.splitlines())[-80:]
            )
            raise RuntimeError(
                f"latexmk failed with exit {completed.returncode}:\n{tail}"
            )
        latex_log = build_docs / "pyAmpliCol.log"
        try:
            log = latex_log.read_text(encoding="utf-8", errors="replace")
            validate_latex_log(log)
        except OSError as error:
            raise RuntimeError(f"cannot read LaTeX log: {error}") from error
        except StandaloneBuildError as error:
            raise RuntimeError(str(error)) from error
        built_pdf = build_docs / "pyAmpliCol.pdf"
        if not built_pdf.is_file() or built_pdf.stat().st_size == 0:
            raise RuntimeError("latexmk did not produce a non-empty PDF")
        output = service.paths.docs_dir / "pyAmpliCol.pdf"
        temporary = output.with_name(f".{output.name}.new")
        shutil.copy2(built_pdf, temporary)
        temporary.replace(output)
        return output
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _launch_async_publication(service: ReportService) -> Path:
    """Request a one-shot report publication without waiting for it."""

    log_root = service.paths.artifact_root / "publication-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "refresh-pdf-end.log"
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        os.fspath(
            service.paths.repo_root / CANONICAL_REPORT_ENTRYPOINT
        ),
        "--repo-root",
        os.fspath(service.paths.repo_root),
        "--docs-dir",
        os.fspath(service.paths.docs_dir),
        "--artifact-root",
        os.fspath(service.paths.artifact_root),
        "--coordination-root",
        os.fspath(service.paths.coordination_root),
        "publish-snapshot",
    )
    environment = os.environ.copy()
    for name in (
        "PYAMPLICOL_EXACT_PYTHON_REEXEC",
        "PYAMPLICOL_EXACT_IMPORT_PATHS",
        "PYTHONPYCACHEPREFIX",
    ):
        environment.pop(name, None)
    with log_path.open("ab") as stream:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    return log_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    if args.command == "init-profile":
        output = initialize_profile(
            repo_root,
            args.profile,
            source_profile=args.source_profile,
            reset_measurements=args.reset_measurements,
            measurement_policy=args.measurement_policy,
        )
        print(output.relative_to(repo_root))
        return 0
    if args.command == "export-profile":
        output = export_profile(
            repo_root,
            args.profile,
            args.destination,
            include_pdf=args.include_pdf,
        )
        print(output)
        return 0

    service = ReportService(
        ReportPaths.from_repo(
            repo_root,
            profile=args.report_profile,
            docs_dir=args.docs_dir,
            artifact_root=args.artifact_root,
            coordination_root=args.coordination_root,
        )
    )

    if args.command == "publish-snapshot":
        run_publisher(
            service,
            watch=args.watch,
            interval_seconds=args.interval_seconds,
            expected_page_count=args.expected_page_count,
            pdf_timeout_seconds=args.pdf_timeout_seconds,
        )
        return 0
    if args.command == "validate-snapshot":
        print(
            json.dumps(
                validate_published_snapshot(service),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        print(json.dumps(service.validate(), sort_keys=True))
        return 0
    if args.command == "audit":
        print(json.dumps(service.audit(), sort_keys=True))
        return 0
    if args.command == "refresh-profile-environment":
        if args.report_profile is None:
            parser.error("refresh-profile-environment requires --report-profile")
        environment = refresh_profile_environment(
            repo_root,
            args.report_profile,
            expected_source_revision=args.expected_source_revision,
        )
        print(json.dumps(environment, allow_nan=False, sort_keys=True))
        return 0
    if args.command == "prepare-class-c-bridge":
        if args.report_profile is None:
            parser.error("prepare-class-c-bridge requires --report-profile")
        prepared = prepare_class_c_bridge(
            repo_root,
            service.paths.docs_dir,
            service.store,
            ancestor_revision=args.ancestor_revision,
            descendant_revision=args.descendant_revision,
            impact=args.impact,
        )
        pending = class_c_pending_path(
            service.store,
            ancestor_revision=str(prepared["ancestor_revision"]),
            descendant_revision=str(prepared["descendant_revision"]),
        )
        print(
            json.dumps(
                {
                    "pending_locator": pending.relative_to(
                        service.paths.artifact_root
                    ).as_posix(),
                    "current_snapshot_sha256": prepared[
                        "current_snapshot_sha256"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "finalize-class-c-bridge":
        if args.report_profile is None:
            parser.error("finalize-class-c-bridge requires --report-profile")
        pending = class_c_pending_path(
            service.store,
            ancestor_revision=args.ancestor_revision,
            descendant_revision=args.descendant_revision,
        )
        finalized = finalize_class_c_bridge(
            repo_root,
            service.paths.docs_dir,
            service.store,
            pending_path=pending,
            expected_active_source_revision=args.descendant_revision,
        )
        print(
            json.dumps(
                {
                    "lineage": (
                        service.paths.docs_dir / "measurement_lineage.json"
                    ).relative_to(repo_root).as_posix(),
                    "descendant_revision": finalized["descendant_revision"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "audit-source-bridge":
        if args.report_profile is None:
            parser.error("audit-source-bridge requires --report-profile")
        result = audit_measurement_lineage(
            repo_root,
            service.paths.docs_dir,
            service.store,
            expected_active_source_revision=args.expected_active_source_revision,
        )
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    if args.command == "seal-existing-worker-result":
        if args.report_profile is None:
            parser.error(
                "seal-existing-worker-result requires --report-profile"
            )
        source = require_eligible_report_source(repo_root)
        if source.revision != args.expected_source_revision:
            parser.error(
                "active source does not match --expected-source-revision"
            )
        require_active_profile_environment(
            repo_root,
            args.report_profile,
            expected_source_revision=source.revision,
        )
        try:
            cell = REPORT_CATALOG.cell(args.cell_id)
        except KeyError:
            parser.error(f"unknown --cell-id {args.cell_id!r}")
        policy = load_profile_campaign_policy(
            repo_root,
            args.report_profile,
            expected_source_revision=source.revision,
        )

        def validate_orphan_result(
            result: Mapping[str, object],
            attempt_root: Path,
        ) -> None:
            validate_measurement(result, expected_cell=cell)
            provenance = result.get("provenance")
            artifact = result.get("artifact")
            if (
                result.get("status") != ResultStatus.OK.value
                or not isinstance(provenance, Mapping)
                or provenance.get("report_source_revision") != source.revision
                or provenance.get("report_source_tree") != source.tree
                or provenance.get("report_measured_source_revision")
                != source.revision
                or provenance.get("report_measured_source_tree") != source.tree
                or not isinstance(artifact, Mapping)
                or artifact.get("path")
                != os.fspath(attempt_root / "artifact")
                or provenance.get("worker_log")
                != os.fspath(attempt_root / "worker.log")
                or not (attempt_root / "artifact").is_dir()
                or (attempt_root / "artifact").is_symlink()
                or not (attempt_root / "worker.log").is_file()
                or (attempt_root / "worker.log").is_symlink()
            ):
                raise ValueError(
                    "orphan worker result lacks exact source/artifact/log evidence"
                )
            allow_unavailable_resources = _is_pinned_epyc_orphan_without_rss(
                profile=args.report_profile,
                cell_id=args.cell_id,
                attempt_id=args.attempt_id,
                worker_result_sha256=args.worker_result_sha256,
                result=result,
            )
            state = validate_policy_measurement(
                policy,
                args.report_profile,
                cell,
                result,
                expected_source_revision=source.revision,
                expected_source_tree=source.tree,
                allow_pinned_orphan_unavailable_resources=(
                    allow_unavailable_resources
                ),
            )
            if state is not PolicyMeasurementState.SUCCESS:
                raise ValueError(
                    "orphan worker result is not a successful policy measurement"
                )

        try:
            record = service.store.seal_existing_worker_result(
                args.cell_id,
                args.attempt_id,
                worker_result_sha256=args.worker_result_sha256,
                artifact_policy=ArtifactPolicy(args.artifact_policy),
                validate_result=validate_orphan_result,
            )
        except (ValueError, OSError, ArtifactStoreError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "cell_id": record.cell_id,
                    "attempt_id": record.attempt_id,
                    "manifest_sha256": record.manifest_sha256,
                    "result_sha256": hashlib.sha256(
                        record.result_path.read_bytes()
                    ).hexdigest(),
                    "resource_monitoring": (
                        "unavailable-pinned-worker-result"
                        if _is_pinned_epyc_orphan_without_rss(
                            profile=args.report_profile,
                            cell_id=args.cell_id,
                            attempt_id=args.attempt_id,
                            worker_result_sha256=args.worker_result_sha256,
                            result=record.result,
                        )
                        else "available"
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "final-audit":
        from .final_audit import audit_final_report

        if args.report_profile is None:
            parser.error("final-audit requires --report-profile")
        require_active_profile_environment(
            repo_root,
            args.report_profile,
            expected_source_revision=args.expected_source_revision,
        )
        result = audit_final_report(
            repo_root,
            expected_source_revision=args.expected_source_revision,
            expected_publication_revision=args.publication_revision,
            max_n_final=args.max_n_final,
            expected_cell_count=args.expected_cell_count,
            replay=not args.structural_only,
            service=service,
        )
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0 if result["final_gate_complete"] is True else 2
    if args.command == "populate":
        try:
            requested = select_cells(_selection(args))
            if not requested:
                parser.error("cell filters select no report cells")
            catalog_ids = {
                cell.cell_id for cell in REPORT_CATALOG.measurement_cells()
            }
            unknown_exclusions = sorted(
                set(args.exclude_cell_id).difference(catalog_ids)
            )
            if unknown_exclusions:
                parser.error(
                    "unknown --exclude-cell-id values: "
                    + ", ".join(unknown_exclusions)
                )
            if args.max_ram_gib is not None and args.max_ram_gb is not None:
                parser.error("--max-ram-gib and --max-ram-gb are mutually exclusive")
            if args.campaign_max_ram_gb is not None and (
                args.campaign_max_ram_gib is not None
                or args.limit_gib is not None
            ):
                parser.error(
                    "--campaign-max-ram-gb is mutually exclusive with "
                    "--campaign-max-ram-gib and --limit-gib"
                )
            campaign_limit_gib = (
                args.campaign_max_ram_gib
                if args.campaign_max_ram_gib is not None
                else args.limit_gib
            )
            source_identity = require_eligible_report_source(repo_root)
            expected_revision = source_identity.revision
            if args.fast_lineage and args.report_profile is None:
                parser.error("--fast-lineage requires --report-profile")
            if args.report_profile is None:
                measurement_lineage = None
            elif args.fast_lineage:
                measurement_lineage = load_measurement_lineage(
                    repo_root,
                    service.paths.docs_dir,
                    expected_active_revision=expected_revision,
                    expected_active_tree=source_identity.tree,
                )
                if measurement_lineage is None:
                    raise MeasurementLineageError(
                        "--fast-lineage requires a finalized measurement lineage"
                    )
            else:
                measurement_lineage = load_and_audit_measurement_lineage(
                    repo_root,
                    service.paths.docs_dir,
                    service.store,
                    expected_active_source_revision=expected_revision,
                )
            if args.report_profile is None:
                campaign_policy = STRICT_POLICY
            else:
                campaign_policy = load_profile_campaign_policy(
                    repo_root,
                    args.report_profile,
                    expected_source_revision=expected_revision,
                )
            settings = CampaignSettings(
                workers=args.workers,
                cell_cores=args.cell_cores,
                target_runtime_seconds=args.target_runtime,
                batch_size=args.batch_size,
                timeout_seconds=args.timeout_seconds,
                generation_time_limit_seconds=args.generation_time_limit_seconds,
                max_rss_bytes=(
                    _gb_bytes(args.max_ram_gb)
                    if args.max_ram_gb is not None
                    else _gib_bytes(args.max_ram_gib)
                ),
                campaign_max_rss_bytes=(
                    _gb_bytes(args.campaign_max_ram_gb)
                    if args.campaign_max_ram_gb is not None
                    else _gib_bytes(campaign_limit_gib)
                ),
                artifact_policy=ArtifactPolicy(args.artifact_policy),
                missing_only=args.missing_only,
                rerun=args.rerun,
                allow_symbolica_parallel=args.allow_symbolica_parallel,
                campaign_policy=campaign_policy,
                report_profile=args.report_profile,
            )
        except (ValueError, ReportWorkspaceError, MeasurementLineageError) as error:
            parser.error(str(error))
        planned = plan_campaign(
            requested,
            store=service.store,
            settings=settings,
            expected_revision=expected_revision,
            expected_tree=source_identity.tree,
            measurement_lineage=measurement_lineage,
            excluded_cell_ids=frozenset(args.exclude_cell_id),
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "requested": len(requested),
                        "scheduled": len(planned),
                        "excluded_cell_ids": sorted(
                            set(args.exclude_cell_id)
                        ),
                        "cells": [
                            {
                                "cell_id": item.cell.cell_id,
                                "dependency": item.dependency,
                                "baseline_cell_id": item.baseline_cell_id,
                                "comparison_peer_ids": list(
                                    item.comparison_peer_ids
                                ),
                                "rank": item.rank,
                            }
                            for item in planned
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.report_profile is not None:
            require_active_profile_environment(
                repo_root,
                args.report_profile,
                expected_source_revision=expected_revision,
            )
        service.bind_measurement_lineage(measurement_lineage)
        result = CampaignScheduler(service, settings=settings).run(planned)
        if args.refresh_pdf == "end":
            _launch_async_publication(service)
        print(
            json.dumps(
                {
                    "planned": len(result.planned),
                    "completed": len(result.outcomes),
                    "failed": len(result.failed),
                    "outcomes": [
                        {
                            "cell_id": outcome.cell_id,
                            "status": outcome.status,
                            "detail": outcome.detail,
                        }
                        for outcome in result.outcomes
                    ],
                },
                sort_keys=True,
            )
        )
        return 1 if result.failed else 0
    if args.command in {"reset", "render", "recover"}:
        written = service.publish(
            reset=args.command == "reset",
            merge_artifacts=args.command != "reset",
        )
        if args.compile:
            written = (*written, _compile_pdf(service))
        for path in written:
            print(path.relative_to(repo_root))
        return 0
    if args.command == "_worker":
        if args.cell_id not in {
            cell.cell_id for cell in REPORT_CATALOG.measurement_cells()
        }:
            parser.error(f"unknown --cell-id {args.cell_id!r}")
        write_cell_result(
            args.cell_id,
            args.result_json,
            repo_root=repo_root,
            attempt_root=args.attempt_root,
            target_runtime_seconds=args.target_runtime,
            batch_size=args.batch_size,
            worker_cores=args.cell_cores,
            baseline_json=args.baseline_json,
            peer_json=tuple(
                (cell_id, Path(path)) for cell_id, path in args.peer_json
            ),
            prepared_model_path=args.prepared_model,
            reused_measurement_json=args.reused_measurement_json,
            phase_state_path=args.phase_state_path,
            phase_state_run_id=args.phase_state_run_id,
            phase_state_authentication_key=args.phase_state_authentication_key,
            legacy_repository=args.legacy_repository,
            log_path=args.log_path,
        )
        return 0
    if args.command == "_prepare":
        try:
            path, reused = ensure_report_prepared_model(
                store=service.store,
                repo_root=repo_root,
                worker_cores=args.cell_cores,
                model=ModelKey(args.model),
            )
            payload = {"path": os.fspath(path), "reused": reused}
            returncode = 0
        except Exception as error:
            payload = {
                "error": {
                    "kind": type(error).__name__,
                    "message": str(error),
                }
            }
            returncode = 1
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="ascii",
        )
        return returncode
    parser.error(f"unsupported command {args.command!r}")
    return 2


__all__ = ["main"]
