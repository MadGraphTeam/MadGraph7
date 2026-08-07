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
from .boundary import (
    authenticate_current_delta,
    load_cell_boundary,
    snapshot_cell_boundary,
)
from .cache import validate_measurement
from .campaign_policy import (
    MACBOOK_M3_POLICY_NAME,
    MACBOOK_M3_PROFILE,
    MACBOOK_M3_Z_TABLE_F_POLICY_NAME,
    STRICT_POLICY,
    STRICT_POLICY_NAME,
    X86_EPYC_POLICY_NAME,
    PolicyMeasurementState,
    validate_policy_measurement,
)
from .campaign_policy import (
    campaign_policy as resolve_campaign_policy,
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
    _report_source_copy_ignore,
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
    validate_campaign_plan,
)
from .service import (
    CANONICAL_REPORT_ENTRYPOINT,
    ReportPaths,
    ReportService,
    validate_profile_name,
)
from .source_identity import require_eligible_report_source
from .standalone_build import StandaloneBuildError, validate_latex_log
from .study_contract import (
    StudyContractError,
    audit_z_table_f_policy_projection,
    load_z_table_f_study_contract,
    require_z_table_f_explicit_cell,
    z_table_f_worker_harness_identity,
)
from .worker import _JsonlProgressSink, write_cell_result
from .worker_harness import (
    LEGACY_ADAPTER,
    POLICY_ENTRYPOINT,
    worker_harness_identity,
)
from .workspace import (
    ReportWorkspaceError,
    export_profile,
    initialize_profile,
    load_profile_campaign_policy,
    refresh_profile_environment,
    require_active_profile_environment,
)

_PINNED_EPYC_ORPHAN_CELL_ID = "reference-amplicol-lc-n8-gg-gluons-selected-flow"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authenticated_worker_harness(
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, object] | None:
    """Recheck both checkouts before an internal split worker can run."""

    names = (
        "measurement_source_root",
        "expected_measurement_source_revision",
        "expected_measurement_source_tree",
        "expected_policy_wrapper_revision",
        "expected_policy_wrapper_tree",
        "expected_policy_entrypoint_sha256",
        "expected_legacy_adapter_sha256",
        "study_contract_sha256",
    )
    values = {name: getattr(args, name) for name in names}
    if all(value is None for value in values.values()):
        return None
    if any(value is None for value in values.values()):
        raise ValueError(
            "split worker wrapper/source options must be specified together"
        )
    if args.command not in {"_prepare", "_worker"}:
        raise ValueError(
            "split worker wrapper/source options are restricted to workers"
        )
    measured_root = (
        Path(values["measurement_source_root"]).expanduser().resolve(strict=True)
    )
    if measured_root != repo_root.resolve(strict=True):
        raise ValueError("worker --repo-root must equal --measurement-source-root")
    measured = require_eligible_report_source(measured_root)
    if (
        measured.revision != values["expected_measurement_source_revision"]
        or measured.tree != values["expected_measurement_source_tree"]
    ):
        raise ValueError(
            "worker measured-source identity differs from its authorization"
        )
    wrapper_root = _repo_root().resolve(strict=True)
    wrapper = require_eligible_report_source(wrapper_root)
    if (
        wrapper.revision != values["expected_policy_wrapper_revision"]
        or wrapper.tree != values["expected_policy_wrapper_tree"]
    ):
        raise ValueError(
            "worker policy-wrapper identity differs from its authorization"
        )
    entrypoint_sha256 = _sha256_file(wrapper_root / POLICY_ENTRYPOINT)
    legacy_adapter_sha256 = _sha256_file(wrapper_root / LEGACY_ADAPTER)
    if (
        entrypoint_sha256 != values["expected_policy_entrypoint_sha256"]
        or legacy_adapter_sha256 != values["expected_legacy_adapter_sha256"]
    ):
        raise ValueError(
            "worker policy-wrapper file identity differs from its authorization"
        )
    return worker_harness_identity(
        study_contract_sha256=str(values["study_contract_sha256"]),
        policy_wrapper_revision=wrapper.revision,
        policy_wrapper_tree=wrapper.tree,
        policy_entrypoint_sha256=entrypoint_sha256,
        legacy_adapter_sha256=legacy_adapter_sha256,
        measured_source_revision=measured.revision,
        measured_source_tree=measured.tree,
    )


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
    parser.add_argument(
        "--class-c-ancestor-runtime-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--measurement-source-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-measurement-source-revision",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-measurement-source-tree",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-policy-wrapper-revision",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-policy-wrapper-tree",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-policy-entrypoint-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-legacy-adapter-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--study-contract-sha256",
        help=argparse.SUPPRESS,
    )
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

    study_audit = subparsers.add_parser(
        "audit-z-table-study",
        help=argparse.SUPPRESS,
    )
    study_audit.add_argument("--study-contract", type=Path, required=True)
    study_audit.add_argument("--maximum-n", type=int, required=True)

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
    snapshot_boundary = subparsers.add_parser(
        "snapshot-cell-boundary",
        help="snapshot one authoritative current and immutable-attempt inventory",
    )
    snapshot_boundary.add_argument("--cell-id", required=True)
    accept_boundary = subparsers.add_parser(
        "accept-cell-boundary",
        help="authenticate a new current without consulting report caches",
    )
    accept_boundary.add_argument("--cell-id", required=True)
    accept_boundary.add_argument("--expected-attempt-id", required=True)
    accept_boundary.add_argument(
        "--before-snapshot",
        type=Path,
        required=True,
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
    final_audit.add_argument("--expected-cell-count", type=int, default=1962)
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
        "--expected-authority-cell-id",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    worker.add_argument(
        "--selected-authority-cell-id",
        help=argparse.SUPPRESS,
    )
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
        "--legacy-source-repository",
        type=Path,
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--legacy-workspace", type=Path, help=argparse.SUPPRESS)
    worker.add_argument(
        "--legacy-copy-source",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--legacy-source-revision", help=argparse.SUPPRESS)
    worker.add_argument(
        "--target-runtime",
        type=float,
        default=DEFAULT_TARGET_RUNTIME_SECONDS,
    )
    worker.add_argument("--batch-size", type=int, default=128)
    worker.add_argument("--cell-cores", type=int, default=1)
    worker.add_argument("--memory-limit-bytes", type=int, help=argparse.SUPPRESS)
    worker.add_argument("--warmup-runs", type=int, default=2)
    worker.add_argument("--minimum-samples", type=int, default=5)
    worker.add_argument("--progress-jsonl", type=Path)
    worker.add_argument("--worker-wall-limit", type=float, help=argparse.SUPPRESS)
    worker.add_argument(
        "--profiling-time-limit",
        type=float,
        help=argparse.SUPPRESS,
    )
    worker.add_argument(
        "--validation-time-limit",
        type=float,
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--generation-lock-path", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--manual-source-revision", help=argparse.SUPPRESS)
    worker.add_argument("--manual-source-tree", help=argparse.SUPPRESS)
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
    prepare.add_argument("--progress-jsonl", type=Path)
    prepare.add_argument("--cell-cores", type=int, default=1)
    prepare.add_argument("--producer-revision", help=argparse.SUPPRESS)

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
    populate.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "filter named Z implementations; rows without a variant dimension "
            "remain eligible"
        ),
    )
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
    populate.add_argument(
        "--study-policy",
        choices=(MACBOOK_M3_Z_TABLE_F_POLICY_NAME,),
        help=argparse.SUPPRESS,
    )
    populate.add_argument(
        "--study-contract",
        type=Path,
        help=argparse.SUPPRESS,
    )
    populate.add_argument(
        "--reuse-cross-source-comparison-dependencies", action="store_true"
    )
    return parser


def _git_commit(root: Path, revision: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "--verify", f"{revision}^{{commit}}"),
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if (
        completed.returncode != 0
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MeasurementLineageError(
            f"cannot resolve Class-C runtime revision {revision!r}"
        )
    return value


def _class_c_ancestor_runtime_identity(
    repo_root: Path,
    runtime_root: Path,
    *,
    ancestor_revision: str,
    descendant_revision: str,
) -> dict[str, object]:
    """Authenticate the ancestor package selected by the direct bootstrap."""

    try:
        root = runtime_root.expanduser().resolve(strict=True)
        source_package = (root / "src/pyamplicol").resolve(strict=True)
    except OSError as error:
        raise MeasurementLineageError(
            "Class-C ancestor runtime root is unavailable"
        ) from error
    if root == repo_root.resolve(strict=False) or not source_package.is_dir():
        raise MeasurementLineageError(
            "Class-C ancestor runtime must be a separate source checkout"
        )
    ancestor = _git_commit(root, ancestor_revision)
    descendant = _git_commit(repo_root, descendant_revision)
    if _git_commit(root, "HEAD") != ancestor:
        raise MeasurementLineageError(
            "Class-C ancestor runtime checkout is not at the requested ancestor"
        )
    if _git_commit(repo_root, "HEAD") != descendant:
        raise MeasurementLineageError(
            "Class-C controller checkout is not at the requested descendant"
        )
    tracked = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0 or tracked.stdout:
        raise MeasurementLineageError(
            "Class-C ancestor runtime has tracked source modifications"
        )

    import pyamplicol

    try:
        package_origin = Path(str(pyamplicol.__file__)).resolve(strict=True)
    except OSError as error:
        raise MeasurementLineageError(
            "loaded Class-C ancestor package origin is unavailable"
        ) from error
    if package_origin.parent != source_package:
        raise MeasurementLineageError(
            "loaded pyamplicol package does not originate in the Class-C "
            "ancestor runtime checkout"
        )
    from .runtime_evidence import established_preimport_runtime_identity

    preimport = established_preimport_runtime_identity()
    tree = preimport.get("python_package_tree")
    native = preimport.get("native_extension")
    if (
        not isinstance(tree, Mapping)
        or tree.get("roots") != [str(source_package)]
        or not isinstance(native, Mapping)
    ):
        raise MeasurementLineageError(
            "Class-C ancestor package was not exclusively preauthenticated"
        )
    native_path = native.get("path")
    if (
        not isinstance(native_path, str)
        or Path(native_path).resolve(strict=False).parent != source_package
    ):
        raise MeasurementLineageError(
            "Class-C ancestor native extension was not preauthenticated"
        )
    return {
        "revision": ancestor,
        "root": str(root),
        "package_root": str(source_package),
        "python_package_tree_sha256": tree.get("sha256"),
        "native_extension_sha256": native.get("sha256"),
    }


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
            ignore=_report_source_copy_ignore(
                service.paths.docs_dir,
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


def _launch_async_publication(
    service: ReportService,
    *,
    entrypoint: Path | None = None,
) -> Path:
    """Request a one-shot report publication without waiting for it."""

    entrypoint_candidate = (
        service.paths.repo_root / CANONICAL_REPORT_ENTRYPOINT
        if entrypoint is None
        else entrypoint
    ).expanduser()
    if entrypoint_candidate.is_symlink():
        raise RuntimeError("asynchronous publication entrypoint is unavailable")
    selected_entrypoint = entrypoint_candidate.resolve(strict=True)
    if not selected_entrypoint.is_file():
        raise RuntimeError("asynchronous publication entrypoint is unavailable")
    log_root = service.paths.artifact_root / "publication-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "refresh-pdf-end.log"
    command = (
        sys.executable,
        "-I",
        "-S",
        "-B",
        os.fspath(selected_entrypoint),
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
    try:
        worker_harness = _authenticated_worker_harness(args, repo_root)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if (
        args.class_c_ancestor_runtime_root is not None
        and args.command != "prepare-class-c-bridge"
    ):
        parser.error(
            "--class-c-ancestor-runtime-root is restricted to prepare-class-c-bridge"
        )
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
    if args.command == "snapshot-cell-boundary":
        if args.cell_id not in {
            cell.cell_id for cell in REPORT_CATALOG.measurement_cells()
        }:
            parser.error(f"unknown --cell-id {args.cell_id!r}")
        print(
            json.dumps(
                snapshot_cell_boundary(service.store, args.cell_id),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "accept-cell-boundary":
        try:
            cell = REPORT_CATALOG.cell(args.cell_id)
        except KeyError:
            parser.error(f"unknown --cell-id {args.cell_id!r}")
        accepted = authenticate_current_delta(
            service.store,
            cell_id=cell.cell_id,
            expected_attempt_id=args.expected_attempt_id,
            before=load_cell_boundary(args.before_snapshot),
            validate_result=lambda result: validate_measurement(
                result,
                expected_cell=cell,
            ),
        )
        print(json.dumps(accepted, allow_nan=False, sort_keys=True))
        return 0
    if args.command == "validate":
        print(json.dumps(service.validate(), sort_keys=True))
        return 0
    if args.command == "audit":
        print(json.dumps(service.audit(), sort_keys=True))
        return 0
    if args.command == "audit-z-table-study":
        try:
            contract = load_z_table_f_study_contract(
                args.study_contract,
                repo_root,
                Path(__file__).resolve().parents[2],
                prior_store=service.store,
            )
            result = audit_z_table_f_policy_projection(
                contract,
                service,
                maximum_n=args.maximum_n,
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, allow_nan=False, sort_keys=True))
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
        ancestor_runtime = (
            None
            if args.class_c_ancestor_runtime_root is None
            else _class_c_ancestor_runtime_identity(
                repo_root,
                args.class_c_ancestor_runtime_root,
                ancestor_revision=args.ancestor_revision,
                descendant_revision=args.descendant_revision,
            )
        )
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
                    "current_snapshot_sha256": prepared["current_snapshot_sha256"],
                    **(
                        {}
                        if ancestor_runtime is None
                        else {"ancestor_runtime": ancestor_runtime}
                    ),
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
                    "lineage": (service.paths.docs_dir / "measurement_lineage.json")
                    .relative_to(repo_root)
                    .as_posix(),
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
            parser.error("seal-existing-worker-result requires --report-profile")
        source = require_eligible_report_source(repo_root)
        if source.revision != args.expected_source_revision:
            parser.error("active source does not match --expected-source-revision")
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
                or provenance.get("report_measured_source_revision") != source.revision
                or provenance.get("report_measured_source_tree") != source.tree
                or not isinstance(artifact, Mapping)
                or artifact.get("path") != os.fspath(attempt_root / "artifact")
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
                allow_pinned_orphan_unavailable_resources=(allow_unavailable_resources),
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
        active_study_contract: Mapping[str, object] | None = None
        try:
            requested = select_cells(_selection(args))
            if not requested:
                parser.error("cell filters select no report cells")
            catalog_ids = {cell.cell_id for cell in REPORT_CATALOG.measurement_cells()}
            unknown_exclusions = sorted(
                set(args.exclude_cell_id).difference(catalog_ids)
            )
            if unknown_exclusions:
                parser.error(
                    "unknown --exclude-cell-id values: " + ", ".join(unknown_exclusions)
                )
            if args.max_ram_gib is not None and args.max_ram_gb is not None:
                parser.error("--max-ram-gib and --max-ram-gb are mutually exclusive")
            if args.campaign_max_ram_gb is not None and (
                args.campaign_max_ram_gib is not None or args.limit_gib is not None
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
            original_amplicol_seed = (
                None
                if args.report_profile is None
                else service._original_amplicol_seed()
            )
            if args.report_profile is None:
                measurement_lineage = None
            elif args.fast_lineage:
                measurement_lineage = load_measurement_lineage(
                    repo_root,
                    service.paths.docs_dir,
                    expected_active_revision=expected_revision,
                    expected_active_tree=source_identity.tree,
                )
                if measurement_lineage is None and original_amplicol_seed is None:
                    raise MeasurementLineageError(
                        "--fast-lineage requires a finalized measurement lineage "
                        "or authenticated original-AmpliCol campaign seed"
                    )
            else:
                measurement_lineage = load_and_audit_measurement_lineage(
                    repo_root,
                    service.paths.docs_dir,
                    service.store,
                    expected_active_source_revision=expected_revision,
                )
            policy_profile = args.report_profile
            if args.study_policy is not None:
                if args.report_profile is not None:
                    raise ValueError(
                        "--study-policy cannot be combined with --report-profile"
                    )
                if args.study_contract is None:
                    raise ValueError("--study-policy requires --study-contract")
                if (
                    len(args.cell_id) != 1
                    or args.dataset
                    or args.mode
                    or args.model
                    or args.accuracy
                    or args.process_key
                    or args.process
                    or args.n_final
                    or args.variant
                    or args.workload is not None
                ):
                    raise ValueError(
                        "--study-policy requires exactly one explicit "
                        "--cell-id and no broad cell selectors"
                    )
                active_study_contract = load_z_table_f_study_contract(
                    args.study_contract,
                    repo_root,
                    Path(__file__).resolve().parents[2],
                    prior_store=service.store,
                )
                require_z_table_f_explicit_cell(
                    active_study_contract,
                    args.cell_id[0],
                )
                campaign_policy = resolve_campaign_policy(args.study_policy)
                policy_profile = MACBOOK_M3_PROFILE
            elif args.study_contract is not None:
                raise ValueError("--study-contract requires --study-policy")
            elif args.report_profile is None:
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
                report_profile=policy_profile,
                study_contract_sha256=(
                    None
                    if active_study_contract is None
                    else str(active_study_contract["sha256"])
                ),
                reuse_cross_source_comparison_dependencies=args.reuse_cross_source_comparison_dependencies,
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
            original_amplicol_seed=original_amplicol_seed,
            excluded_cell_ids=frozenset(args.exclude_cell_id),
            expected_worker_harness=(
                None
                if active_study_contract is None
                else z_table_f_worker_harness_identity(active_study_contract)
            ),
        )
        try:
            validate_campaign_plan(planned, settings)
        except StudyContractError as error:
            parser.error(str(error))
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "requested": len(requested),
                        "scheduled": len(planned),
                        "source_revision": expected_revision,
                        "source_tree": source_identity.tree,
                        "campaign_policy": (settings.campaign_policy.as_manifest()),
                        "policy_profile": settings.report_profile,
                        "study_contract_sha256": (
                            None
                            if active_study_contract is None
                            else active_study_contract["sha256"]
                        ),
                        "excluded_cell_ids": sorted(set(args.exclude_cell_id)),
                        "cells": [
                            {
                                "cell_id": item.cell.cell_id,
                                "dependency": item.dependency,
                                "baseline_cell_id": item.baseline_cell_id,
                                "comparison_peer_ids": list(item.comparison_peer_ids),
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
        if active_study_contract is None:
            scheduler = CampaignScheduler(service, settings=settings)
        else:
            scheduler = CampaignScheduler(
                service,
                settings=settings,
                study_contract=active_study_contract,
                study_contract_wrapper_root=(Path(__file__).resolve().parents[2]),
            )
        result = scheduler.run(planned)
        if args.refresh_pdf == "end":
            _launch_async_publication(
                service,
                entrypoint=scheduler.worker_entrypoint,
            )
        print(
            json.dumps(
                {
                    "planned": len(result.planned),
                    "completed": len(result.outcomes),
                    "failed": len(result.failed),
                    "source_revision": expected_revision,
                    "source_tree": source_identity.tree,
                    "campaign_policy": (settings.campaign_policy.as_manifest()),
                    "policy_profile": settings.report_profile,
                    "study_contract_sha256": (
                        None
                        if active_study_contract is None
                        else active_study_contract["sha256"]
                    ),
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
            memory_limit_bytes=args.memory_limit_bytes,
            warmup_runs=args.warmup_runs,
            minimum_samples=args.minimum_samples,
            progress_jsonl=args.progress_jsonl,
            worker_wall_limit_seconds=args.worker_wall_limit,
            profiling_time_limit_seconds=args.profiling_time_limit,
            validation_time_limit_seconds=args.validation_time_limit,
            generation_lock_path=args.generation_lock_path,
            manual_source_revision=args.manual_source_revision,
            manual_source_tree=args.manual_source_tree,
            baseline_json=args.baseline_json,
            expected_authority_cell_ids=tuple(args.expected_authority_cell_id),
            selected_authority_cell_id=args.selected_authority_cell_id,
            peer_json=tuple((cell_id, Path(path)) for cell_id, path in args.peer_json),
            prepared_model_path=args.prepared_model,
            reused_measurement_json=args.reused_measurement_json,
            phase_state_path=args.phase_state_path,
            phase_state_run_id=args.phase_state_run_id,
            phase_state_authentication_key=args.phase_state_authentication_key,
            legacy_repository=args.legacy_repository,
            legacy_source_repository=args.legacy_source_repository,
            legacy_workspace=args.legacy_workspace,
            legacy_copy_source=args.legacy_copy_source,
            legacy_source_revision=args.legacy_source_revision,
            log_path=args.log_path,
            worker_harness=worker_harness,
        )
        return 0
    if args.command == "_prepare":
        try:
            path, reused = ensure_report_prepared_model(
                store=service.store,
                repo_root=repo_root,
                worker_cores=args.cell_cores,
                model=ModelKey(args.model),
                producer_revision=args.producer_revision,
                progress=(
                    None
                    if args.progress_jsonl is None
                    else _JsonlProgressSink(args.progress_jsonl)
                ),
            )
            payload = {
                "path": os.fspath(path),
                "reused": reused,
                **(
                    {} if worker_harness is None else {"worker_harness": worker_harness}
                ),
            }
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
