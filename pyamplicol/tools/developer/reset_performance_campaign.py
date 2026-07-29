#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
# ruff: noqa: E402
"""Prepare, commit, recover, and verify one full report-campaign restart."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.cache import empty_measurement
from tools.performance_report.campaign_activity import (
    blocking_process_lines,
    is_snapshot_publisher_process,
    parse_lsof_field_output,
)
from tools.performance_report.campaign_policy import validate_policy_profile
from tools.performance_report.campaign_reset import (
    BASELINE_GATE_FILENAME,
    CAMPAIGN_MARKER_FILENAME,
    EXPECTED_AMPLICOL_CELL_COUNT,
    EXPECTED_CATALOG_CELL_COUNT,
    EXPECTED_NON_AMPLICOL_CELL_COUNT,
    CampaignResetError,
    OriginalAmplicolSeed,
    ResetTransactionPaths,
    assert_campaign_marker,
    build_seed_manifest,
    campaign_marker,
    canonical_json_bytes,
    commit_or_recover_reset,
    lightweight_archive_inventory,
    mark_campaign_ready,
    sha256_path,
    sha256_payload,
    stage_seed_store,
    temporary_staging_root,
    write_prepared_journal,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import require_eligible_report_source
from tools.performance_report.workspace import load_profile_campaign_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive an old performance campaign and seed a clean epoch with "
            "same-host authenticated original-AmpliCol currents."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    for item in (prepare,):
        item.add_argument("--source-repo-root", type=Path, required=True)
        item.add_argument("--destination-repo-root", type=Path, required=True)
        item.add_argument("--profile", required=True)
        item.add_argument("--campaign-id", required=True)
        item.add_argument("--archive-id", required=True)
        item.add_argument("--expected-source-revision", required=True)
        item.add_argument("--expected-source-tree", required=True)
        item.add_argument("--source-artifact-root", type=Path)
        item.add_argument("--source-coordination-root", type=Path)
        item.add_argument("--destination-artifact-root", type=Path)
        item.add_argument("--destination-coordination-root", type=Path)
        item.add_argument("--archive-root", type=Path)

    for command in ("commit", "recover"):
        item = subparsers.add_parser(command)
        item.add_argument("--journal", type=Path, required=True)

    verify = subparsers.add_parser("verify-baseline")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--profile", required=True)
    verify.add_argument("--campaign-id", required=True)
    verify.add_argument("--expected-source-revision", required=True)
    verify.add_argument("--expected-source-tree", required=True)
    verify.add_argument("--expected-marker-sha256")
    verify.add_argument("--artifact-root", type=Path)
    verify.add_argument("--coordination-root", type=Path)

    marker = subparsers.add_parser("assert-marker")
    marker.add_argument("--repo-root", type=Path, required=True)
    marker.add_argument("--profile", required=True)
    marker.add_argument("--campaign-id", required=True)
    marker.add_argument("--expected-source-revision", required=True)
    marker.add_argument("--expected-source-tree", required=True)
    marker.add_argument("--expected-marker-sha256")
    marker.add_argument("--artifact-root", type=Path)
    return parser


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _strict_idle(
    *,
    coordination_roots: Sequence[Path],
    entrypoints: Sequence[Path],
) -> None:
    process_output = _run(
        ("ps", "-axo", "pid=,comm=,args="),
        cwd=Path.cwd(),
    ).stdout
    blocking = list(
        blocking_process_lines(
            process_output,
            entrypoints=entrypoints,
        )
    )
    for line in process_output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and is_snapshot_publisher_process(fields[2]):
            blocking.append(line.strip())
    if blocking:
        raise CampaignResetError(
            "campaign reset requires process idleness: "
            + "; ".join(blocking[:12])
        )
    open_files: list[str] = []
    for root in coordination_roots:
        if not root.exists():
            continue
        result = _run(
            ("lsof", "-Fpcfnt", "+D", os.fspath(root)),
            cwd=Path.cwd(),
            check=False,
        )
        open_files.extend(
            f"{item.pid}:{item.command}:{item.descriptor}:{item.path}"
            for item in parse_lsof_field_output(result.stdout)
        )
    if open_files:
        raise CampaignResetError(
            "campaign reset requires every coordination FD closed: "
            + "; ".join(open_files[:12])
        )


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise CampaignResetError(f"staging destination already exists: {destination}")
    if not source.is_dir() or source.is_symlink():
        raise CampaignResetError(f"source is not a regular directory: {source}")
    shutil.copytree(source, destination, symlinks=False)


def _filesystem_device(path: Path) -> int:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise CampaignResetError(
                f"cannot resolve filesystem device for {path}"
            )
        candidate = parent
    return candidate.stat().st_dev


def _require_same_filesystem(paths: Sequence[Path]) -> None:
    devices = {_filesystem_device(path) for path in paths}
    if len(devices) != 1:
        raise CampaignResetError(
            "campaign reset roots/staging/archive are not on one filesystem"
        )


def _pinned_legacy_revision(repo_root: Path) -> str:
    path = repo_root / "dependencies" / "contributor-lock.toml"
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        legacy = payload["legacy_amplicol"]
        revision = legacy["revision"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise CampaignResetError(
            "cannot load contributor-lock original-AmpliCol revision"
        ) from error
    if not isinstance(revision, str):
        raise CampaignResetError(
            "contributor-lock original-AmpliCol revision is malformed"
        )
    return revision


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resolved_paths(arguments: argparse.Namespace) -> tuple[
    Path,
    Path,
    ReportPaths,
    ReportPaths,
    Path,
]:
    source = arguments.source_repo_root.expanduser().resolve(strict=True)
    destination = arguments.destination_repo_root.expanduser().resolve(strict=True)
    source_paths = ReportPaths.from_repo(
        source,
        profile=arguments.profile,
        artifact_root=arguments.source_artifact_root,
        coordination_root=arguments.source_coordination_root,
    )
    destination_paths = ReportPaths.from_repo(
        destination,
        profile=arguments.profile,
        artifact_root=arguments.destination_artifact_root,
        coordination_root=arguments.destination_coordination_root,
    )
    archive = (
        arguments.archive_root.expanduser().resolve(strict=False)
        if arguments.archive_root is not None
        else (
            source
            / ".artifacts"
            / "performance-report-archive"
            / arguments.archive_id
            / arguments.profile
        )
    )
    return source, destination, source_paths, destination_paths, archive


def _prepare(arguments: argparse.Namespace) -> dict[str, object]:
    (
        source,
        destination,
        source_paths,
        destination_paths,
        archive_root,
    ) = _resolved_paths(arguments)
    identity = require_eligible_report_source(destination)
    if (
        identity.revision != arguments.expected_source_revision
        or identity.tree != arguments.expected_source_tree
    ):
        raise CampaignResetError("destination source identity differs")
    if archive_root.exists():
        raise CampaignResetError(f"archive already exists: {archive_root}")
    if (
        destination_paths.artifact_root.exists()
        or destination_paths.coordination_root.exists()
    ):
        raise CampaignResetError("destination campaign roots already exist")
    _strict_idle(
        coordination_roots=(source_paths.coordination_root,),
        entrypoints=(
            source_paths.docs_dir / "result_tables.py",
            destination_paths.docs_dir / "result_tables.py",
        ),
    )
    policy = load_profile_campaign_policy(
        destination,
        arguments.profile,
        expected_source_revision=identity.revision,
    )
    validate_policy_profile(policy, arguments.profile)
    _require_same_filesystem(
        (
            source_paths.docs_dir,
            source_paths.artifact_root,
            source_paths.coordination_root,
            destination_paths.docs_dir,
            destination_paths.artifact_root.parent,
            destination_paths.coordination_root.parent,
            archive_root.parent,
        )
    )
    source_store = ArtifactStore(
        artifact_root=source_paths.artifact_root,
        lock_root=source_paths.coordination_root,
    )
    seed_manifest = build_seed_manifest(
        profile=arguments.profile,
        store=source_store,
        policy=policy,
        final_source_revision=identity.revision,
        final_source_tree=identity.tree,
        expected_legacy_revision=_pinned_legacy_revision(destination),
    )
    staging = temporary_staging_root(
        destination_paths.artifact_root.parent,
        campaign_id=arguments.campaign_id,
    )
    fresh_artifacts = staging / "fresh-artifacts"
    fresh_coordination = staging / "fresh-coordination"
    fresh_store = stage_seed_store(
        source_store=source_store,
        destination_root=fresh_artifacts,
        destination_lock_root=fresh_coordination,
        seed_manifest=seed_manifest,
    )
    archive_publication = staging / "archive-publication"
    _copy_tree(source_paths.docs_dir, archive_publication)
    fresh_publication = staging / "fresh-publication"
    _copy_tree(destination_paths.docs_dir, fresh_publication)
    inventory = lightweight_archive_inventory(
        (
            source_paths.docs_dir,
            source_paths.artifact_root,
            source_paths.coordination_root,
        )
    )
    archive_manifest: dict[str, object] = {
        "schema": "pyamplicol-performance-campaign-archive-v1",
        "archive_id": arguments.archive_id,
        "profile": arguments.profile,
        "source_repo_root": os.fspath(source),
        "source_publication": os.fspath(source_paths.docs_dir),
        "source_artifact_root": os.fspath(source_paths.artifact_root),
        "source_coordination_root": os.fspath(source_paths.coordination_root),
        "inventory": inventory,
    }
    archive_manifest["archive_manifest_sha256"] = sha256_payload(archive_manifest)
    _write_json(staging / "archive-manifest.json", archive_manifest)
    marker = campaign_marker(
        campaign_id=arguments.campaign_id,
        profile=arguments.profile,
        source_revision=identity.revision,
        source_tree=identity.tree,
        policy_sha256=sha256_payload(policy.as_manifest()),
        seed_manifest_sha256=str(seed_manifest["seed_manifest_sha256"]),
        archive_manifest_sha256=str(
            archive_manifest["archive_manifest_sha256"]
        ),
    )
    _write_json(fresh_artifacts / CAMPAIGN_MARKER_FILENAME, marker)
    staged_service = ReportService(
        ReportPaths(
            repo_root=destination,
            docs_dir=fresh_publication,
            results_dir=fresh_publication / "results",
            artifact_root=fresh_artifacts,
            coordination_root=fresh_coordination,
        )
    )
    staged_service.bind_original_amplicol_seed(
        OriginalAmplicolSeed.load(
            fresh_artifacts / "original_amplicol_seed.json",
            profile=arguments.profile,
            store=fresh_store,
            expected_final_source_revision=identity.revision,
            expected_final_source_tree=identity.tree,
            expected_manifest_sha256=str(
                seed_manifest["seed_manifest_sha256"]
            ),
        )
    )
    staged_service.publish(reset=True, merge_artifacts=False)
    staged_service.publish(reset=False, merge_artifacts=True)
    paths = ResetTransactionPaths(
        source_publication=source_paths.docs_dir,
        source_artifact_root=source_paths.artifact_root,
        source_coordination_root=source_paths.coordination_root,
        destination_publication=destination_paths.docs_dir,
        destination_artifact_root=destination_paths.artifact_root,
        destination_coordination_root=destination_paths.coordination_root,
        archive_root=archive_root,
        staging_root=staging,
        guard_path=archive_root.parent / ".campaign-reset.guard",
    )
    journal = write_prepared_journal(
        profile=arguments.profile,
        campaign_id=arguments.campaign_id,
        archive_id=arguments.archive_id,
        paths=paths,
        archive_manifest_sha256=str(
            archive_manifest["archive_manifest_sha256"]
        ),
        seed_manifest_sha256=str(seed_manifest["seed_manifest_sha256"]),
        marker_sha256=str(marker["marker_sha256"]),
    )
    return {
        "status": "PREPARED",
        "journal": os.fspath(paths.journal_path),
        "journal_sha256": journal["journal_sha256"],
        "archive_manifest_sha256": archive_manifest[
            "archive_manifest_sha256"
        ],
        "seed_manifest_sha256": seed_manifest["seed_manifest_sha256"],
        "marker_sha256": marker["marker_sha256"],
        "reused": seed_manifest["seed_count"],
    }


def _commit(arguments: argparse.Namespace) -> dict[str, object]:
    journal = json.loads(
        arguments.journal.expanduser().resolve(strict=True).read_text()
    )
    paths = journal.get("paths", {})
    if not isinstance(paths, Mapping):
        raise CampaignResetError("campaign reset journal paths are malformed")
    coordination = tuple(
        Path(str(paths[key]))
        for key in ("source_coordination_root", "destination_coordination_root")
        if Path(str(paths[key])).exists()
    )
    _strict_idle(coordination_roots=coordination, entrypoints=())
    return commit_or_recover_reset(arguments.journal.expanduser().resolve())


def _publisher_command(
    *,
    repo_root: Path,
    profile: str,
    artifact_root: Path,
    coordination_root: Path,
    command: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        os.fspath(
            repo_root / "docs" / "performance_reports" / profile / "result_tables.py"
        ),
        "--repo-root",
        os.fspath(repo_root),
        "--report-profile",
        profile,
        "--artifact-root",
        os.fspath(artifact_root),
        "--coordination-root",
        os.fspath(coordination_root),
        command,
    )


def _verify_baseline(arguments: argparse.Namespace) -> dict[str, object]:
    repo = arguments.repo_root.expanduser().resolve(strict=True)
    paths = ReportPaths.from_repo(
        repo,
        profile=arguments.profile,
        artifact_root=arguments.artifact_root,
        coordination_root=arguments.coordination_root,
    )
    marker = assert_campaign_marker(
        paths.artifact_root,
        campaign_id=arguments.campaign_id,
        profile=arguments.profile,
        source_revision=arguments.expected_source_revision,
        source_tree=arguments.expected_source_tree,
        expected_marker_sha256=arguments.expected_marker_sha256,
        require_ready=False,
    )
    store = ArtifactStore(
        artifact_root=paths.artifact_root,
        lock_root=paths.coordination_root,
    )
    seed = OriginalAmplicolSeed.load(
        paths.artifact_root / "original_amplicol_seed.json",
        profile=arguments.profile,
        store=store,
        expected_final_source_revision=arguments.expected_source_revision,
        expected_final_source_tree=arguments.expected_source_tree,
        expected_manifest_sha256=str(marker["seed_manifest_sha256"]),
    )
    records = {record.cell_id: record for record in store.recover_current_records()}
    if set(records) != set(seed.pins_by_cell):
        raise CampaignResetError("fresh current inventory differs from seed pins")
    for cell_id in records:
        if store.cell_attempt_ids(cell_id) != (records[cell_id].attempt_id,):
            raise CampaignResetError(f"{cell_id}: fresh attempt inventory differs")
    observed_attempts = {
        attempt
        for path in store.cells_root.glob("*/attempts")
        if path.is_dir()
        for attempt in path.iterdir()
        if attempt.is_dir()
    }
    expected_attempts = {record.manifest_path.parent for record in records.values()}
    if observed_attempts != expected_attempts:
        raise CampaignResetError(
            "fresh store contains a non-seed or orphan attempt"
        )
    _strict_idle(
        coordination_roots=(paths.coordination_root,),
        entrypoints=(paths.docs_dir / "result_tables.py",),
    )
    publication = _run(
        _publisher_command(
            repo_root=repo,
            profile=arguments.profile,
            artifact_root=paths.artifact_root,
            coordination_root=paths.coordination_root,
            command="publish-snapshot",
        ),
        cwd=repo,
    )
    publication_log = paths.artifact_root / "baseline-publication.log"
    publication_log.write_text(
        publication.stdout + publication.stderr,
        encoding="utf-8",
    )
    snapshot = json.loads(
        _run(
            _publisher_command(
                repo_root=repo,
                profile=arguments.profile,
                artifact_root=paths.artifact_root,
                coordination_root=paths.coordination_root,
                command="validate-snapshot",
            ),
            cwd=repo,
        ).stdout
    )
    service = ReportService(paths)
    caches = service.load_caches()
    entries = {
        str(entry["cell_id"]): entry["measurement"]
        for cache in caches.values()
        for entry in cache["entries"]
    }
    cells = REPORT_CATALOG.measurement_cells()
    if (
        len(cells) != EXPECTED_CATALOG_CELL_COUNT
        or set(entries) != {cell.cell_id for cell in cells}
    ):
        raise CampaignResetError("baseline gate does not cover exactly 1,666 cells")
    amplicol = {
        cell.cell_id
        for cell in cells
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
    }
    non_amplicol = {cell.cell_id for cell in cells} - amplicol
    if (
        len(amplicol) != EXPECTED_AMPLICOL_CELL_COUNT
        or len(non_amplicol) != EXPECTED_NON_AMPLICOL_CELL_COUNT
    ):
        raise CampaignResetError(
            "baseline gate catalog split differs from 288/1,378"
        )
    for cell_id in non_amplicol:
        if entries[cell_id] != empty_measurement():
            raise CampaignResetError(f"{cell_id}: pyAmpliCol row is not canonical N/A")
    for cell_id in amplicol:
        populated = entries[cell_id].get("status") != "not_available"
        if populated != (cell_id in seed.pins_by_cell):
            raise CampaignResetError(
                f"{cell_id}: baseline population differs from seed pins"
            )
    pdf = paths.docs_dir / "pyAmpliCol.pdf"
    info = _run(("pdfinfo", os.fspath(pdf)), cwd=repo).stdout
    page_values = [
        line.split(":", 1)[1].strip()
        for line in info.splitlines()
        if line.startswith("Pages:") and ":" in line
    ]
    if page_values != ["59"]:
        raise CampaignResetError("baseline PDF does not contain exactly 59 pages")
    reused = len(seed.pins_by_cell)
    gate: dict[str, object] = {
        "schema": "pyamplicol-performance-baseline-gate-v1",
        "campaign_id": arguments.campaign_id,
        "profile": arguments.profile,
        "source_revision": arguments.expected_source_revision,
        "source_tree": arguments.expected_source_tree,
        "prepared_marker_sha256": marker["marker_sha256"],
        "seed_manifest_sha256": seed.digest,
        "catalog_cell_count": len(cells),
        "amplicol_cell_count": len(amplicol),
        "non_amplicol_na_count": len(non_amplicol),
        "reused": reused,
        "new": 0,
        "completed": reused,
        "remaining": len(cells) - reused,
        "pdf_sha256": sha256_path(pdf),
        "pdf_mtime_ns": pdf.stat().st_mtime_ns,
        "pdf_pages": 59,
        "publisher_log_sha256": sha256_path(publication_log),
        "snapshot": snapshot,
    }
    gate["baseline_gate_sha256"] = sha256_payload(gate)
    _write_json(paths.artifact_root / BASELINE_GATE_FILENAME, gate)
    ready_marker = mark_campaign_ready(
        paths.artifact_root,
        baseline_gate_sha256=str(gate["baseline_gate_sha256"]),
    )
    assert_campaign_marker(
        paths.artifact_root,
        campaign_id=arguments.campaign_id,
        profile=arguments.profile,
        source_revision=arguments.expected_source_revision,
        source_tree=arguments.expected_source_tree,
        expected_marker_sha256=str(ready_marker["marker_sha256"]),
    )
    return {**gate, "ready_marker_sha256": ready_marker["marker_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            result = _prepare(arguments)
        elif arguments.command in {"commit", "recover"}:
            result = _commit(arguments)
        elif arguments.command == "verify-baseline":
            result = _verify_baseline(arguments)
        else:
            paths = ReportPaths.from_repo(
                arguments.repo_root.expanduser().resolve(strict=True),
                profile=arguments.profile,
                artifact_root=arguments.artifact_root,
            )
            result = assert_campaign_marker(
                paths.artifact_root,
                campaign_id=arguments.campaign_id,
                profile=arguments.profile,
                source_revision=arguments.expected_source_revision,
                source_tree=arguments.expected_source_tree,
                expected_marker_sha256=arguments.expected_marker_sha256,
            )
    except CampaignResetError as error:
        raise SystemExit(f"campaign reset failed: {error}") from error
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
