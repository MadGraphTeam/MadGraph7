# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ArtifactPolicy, CellSpec, ExecutionMode
from tools.performance_report.publication import (
    portable_publication_value,
    publication_absolute_paths,
)
from tools.performance_report.scheduler import CampaignScheduler, CampaignSettings
from tools.performance_report.service import ReportPaths


def _scheduler(*, retain_workspaces: bool = False) -> CampaignScheduler:
    scheduler = object.__new__(CampaignScheduler)
    scheduler.settings = CampaignSettings(retain_workspaces=retain_workspaces)
    return scheduler


def _cell(mode: ExecutionMode) -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is mode
    )


def test_successful_reference_artifacts_are_compacted_by_default(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler()
    legacy = tmp_path / "legacy-artifact"
    legacy.mkdir()
    (legacy / "legacy.log").write_bytes(b"a" * (128 * 1024))
    for name in (
        "contracted-generated-library",
        "legacy-structural-evidence",
        "selected-flow-generated-library",
    ):
        directory = legacy / name
        directory.mkdir()
        (directory / "evidence").write_text(name, encoding="ascii")

    legacy_result: dict[str, object] = {"artifact": {"path": str(legacy)}}
    scheduler._compact_reference_artifact(
        _cell(ExecutionMode.AMPLICOL),
        legacy,
        legacy_result,
    )

    assert (legacy / "legacy.log").stat().st_size <= 64 * 1024
    assert not (legacy / "contracted-generated-library").exists()
    # The proof inventories these compact files by digest, so they stay with it.
    assert (legacy / "legacy-structural-evidence").is_dir()
    # The compact selected-flow provider remains reusable by an all-flow cell.
    assert (legacy / "selected-flow-generated-library").is_dir()

    madgraph = tmp_path / "madgraph-artifact"
    (madgraph / "standalone").mkdir(parents=True)
    (madgraph / "standalone" / "matrix.f").write_text("x", encoding="ascii")
    (madgraph / "madgraph.log").write_bytes(b"b" * (128 * 1024))

    madgraph_result: dict[str, object] = {
        "artifact": {
            "path": str(madgraph),
            "standalone": str(madgraph / "standalone"),
            "subprocess": "standalone/SubProcesses/P0_test",
        }
    }
    scheduler._compact_reference_artifact(
        _cell(ExecutionMode.MADGRAPH),
        madgraph,
        madgraph_result,
    )

    assert (madgraph / "madgraph.log").stat().st_size <= 64 * 1024
    assert not (madgraph / "standalone").exists()
    retained_artifact = madgraph_result["artifact"]
    assert isinstance(retained_artifact, dict)
    assert "standalone" not in retained_artifact
    assert "subprocess" not in retained_artifact
    assert retained_artifact["workspace_retention"] == {
        "schema": "pyamplicol-reference-workspace-retention-v1",
        "full_workspaces_retained": False,
        "removed": ["standalone"],
    }


def test_failed_reference_keeps_only_bounded_diagnostic_tail(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler()
    artifact = tmp_path / "attempt" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "legacy.log").write_bytes(b"prefix\n" + b"z" * (128 * 1024))
    (artifact / "partial-build").mkdir()

    scheduler._discard_terminal_artifact(artifact, artifact.parent)

    diagnostic = artifact.parent / "legacy-artifact-tail.log"
    assert not artifact.exists()
    assert diagnostic.is_file()
    assert diagnostic.stat().st_size <= 64 * 1024
    assert diagnostic.read_bytes().endswith(b"z" * 1024)


def test_legacy_workspace_cleanup_is_default_and_debug_retention_is_opt_in(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "legacy-workspace"
    log = workspace / "Outputs" / "log_file.txt"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"line\n" * 20_000)
    diagnostic = tmp_path / "attempt" / "legacy-generator-tail.log"

    _scheduler()._finalize_legacy_workspace(
        workspace,
        diagnostic_path=diagnostic,
    )

    assert not workspace.exists()
    assert diagnostic.is_file()
    assert diagnostic.stat().st_size <= 64 * 1024

    retained = tmp_path / "retained-workspace"
    retained_log = retained / "Outputs" / "log_file.txt"
    retained_log.parent.mkdir(parents=True)
    retained_log.write_text("debug", encoding="ascii")
    retained_diagnostic = tmp_path / "unused-tail.log"

    _scheduler(retain_workspaces=True)._finalize_legacy_workspace(
        retained,
        diagnostic_path=retained_diagnostic,
    )

    assert retained_log.is_file()
    assert not retained_diagnostic.exists()


def test_worker_log_compaction_respects_debug_retention(tmp_path: Path) -> None:
    compact_log = tmp_path / "compact-worker.log"
    compact_log.write_bytes(b"c" * (128 * 1024))
    _scheduler()._finalize_worker_log(compact_log)
    assert compact_log.stat().st_size <= 64 * 1024

    retained_log = tmp_path / "retained-worker.log"
    retained_payload = b"d" * (128 * 1024)
    retained_log.write_bytes(retained_payload)
    _scheduler(retain_workspaces=True)._finalize_worker_log(retained_log)
    assert retained_log.read_bytes() == retained_payload


def test_compacted_madgraph_current_remains_resumable_and_portable(
    tmp_path: Path,
) -> None:
    paths = ReportPaths.from_repo(tmp_path / "source", profile="retention-test")
    store = ArtifactStore(
        artifact_root=paths.artifact_root,
        lock_root=paths.coordination_root / "locks",
    )
    cell = _cell(ExecutionMode.MADGRAPH)

    with store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE) as attempt:
        artifact = attempt.path("artifact/madgraph.log").parent
        (artifact / "madgraph.log").write_text("complete", encoding="ascii")
        (artifact / "standalone").mkdir()
        result: dict[str, object] = {
            "artifact": {
                "path": str(artifact),
                "standalone": str(artifact / "standalone"),
                "subprocess": "standalone/SubProcesses/P0_test",
            }
        }
        _scheduler()._compact_reference_artifact(cell, artifact, result)
        attempt.publish(result, artifact_paths=("artifact/madgraph.log",))

    current = store.load_current(cell.cell_id)
    assert current is not None
    current_artifact = current.result["artifact"]
    assert isinstance(current_artifact, dict)
    assert "standalone" not in current_artifact
    assert "subprocess" not in current_artifact
    portable = portable_publication_value(current.result, paths)
    assert publication_absolute_paths(portable) == ()
