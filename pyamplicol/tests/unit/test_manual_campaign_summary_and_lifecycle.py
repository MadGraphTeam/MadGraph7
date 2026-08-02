# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import errno
import json
import threading
from pathlib import Path

import pytest

import tools.performance_report.artifacts as artifacts_module
import tools.performance_report.manual_campaign as manual_campaign
from tools.performance_report.artifacts import (
    ArtifactStore,
    ArtifactStoreError,
    DiskFullError,
    ManifestValidationError,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import (
    DashboardState,
    LightweightCurrent,
    WorkerView,
)
from tools.performance_report.models import ArtifactPolicy
from tools.performance_report.scheduler import (
    CampaignResult,
    CampaignScheduler,
    CampaignSettings,
    CellOutcome,
    PlannedCell,
    _attempt_files,
    archive_cell_attempt_history,
    reconcile_attempt_history,
)
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "coordination",
    )


def _service(tmp_path: Path) -> ReportService:
    root = tmp_path / "repo"
    docs = root / "docs/performance_reports/manual"
    docs.mkdir(parents=True)
    return ReportService(
        ReportPaths.from_repo(
            root,
            docs_dir=docs,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )


def _publish_with_payload(
    store: ArtifactStore,
    cell_id: str,
    marker: str,
    *,
    artifact_path: Path | None = None,
):
    attempt = store.new_attempt(cell_id, ArtifactPolicy.REGENERATE)
    payload = attempt.path("artifact/payload.bin")
    payload.write_bytes(marker.encode("ascii"))
    attempt.path("worker.log").write_text(f"{marker}\n", encoding="utf-8")
    return attempt.publish(
        {
            "status": "ok",
            "artifact": {
                "path": str(
                    (attempt.root / "artifact")
                    if artifact_path is None
                    else artifact_path
                )
            },
        },
        artifact_paths=("artifact/payload.bin", "worker.log"),
    )


def test_cell_id_files_union_comments_and_intersect(tmp_path: Path) -> None:
    first = "scalar-contact-n2-scalar-contact-contracted"
    second = "scalar-gravity-n2-scalar-gravity-contracted"
    ids = tmp_path / "failed.txt"
    ids.write_text(f"# retry set\n{first}\n\n{second}\n{first}\n", encoding="utf-8")
    arguments = manual_campaign.build_parser().parse_args(
        (
            "inspect",
            "--cell-id-file",
            str(ids),
            "--model",
            "scalar_contact",
        )
    )

    _selection, cells = manual_campaign.selection_from_arguments(arguments)

    assert tuple(cell.cell_id for cell in cells) == (first,)


def test_repeated_cell_id_files_union_with_direct_ids(tmp_path: Path) -> None:
    contact = "scalar-contact-n2-scalar-contact-contracted"
    gravity = "scalar-gravity-n2-scalar-gravity-contracted"
    matrix = "matrix-recurrence-builtin-sm-lc-n2-dd-z-jets-selected-flow"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(f"{contact}\n", encoding="utf-8")
    second.write_text(f"{gravity}\n", encoding="utf-8")
    arguments = manual_campaign.build_parser().parse_args(
        (
            "inspect",
            "--cell-id-file",
            str(first),
            "--cell-id-file",
            str(second),
            "--cell-id",
            matrix,
            "--multiplicity",
            "2",
        )
    )

    _selection, cells = manual_campaign.selection_from_arguments(arguments)

    assert {cell.cell_id for cell in cells} == {contact, gravity, matrix}


def test_cell_id_file_reports_path_line_and_rejects_wildcard(tmp_path: Path) -> None:
    ids = tmp_path / "failed.txt"
    ids.write_text("*\n", encoding="utf-8")
    arguments = manual_campaign.build_parser().parse_args(
        ("inspect", "--cell-id-file", str(ids))
    )

    with pytest.raises(
        manual_campaign.ManualCampaignError,
        match=rf"{ids}:1: wildcards are not allowed",
    ):
        manual_campaign.selection_from_arguments(arguments)


def test_empty_cell_id_file_does_not_expand_to_all_cells(tmp_path: Path) -> None:
    ids = tmp_path / "empty.txt"
    ids.write_text("# no retries\n\n", encoding="utf-8")
    arguments = manual_campaign.build_parser().parse_args(
        ("inspect", "--cell-id-file", str(ids))
    )

    with pytest.raises(
        manual_campaign.ManualCampaignError,
        match="contain no canonical cell IDs",
    ):
        manual_campaign.selection_from_arguments(arguments)


@pytest.mark.parametrize("wildcard", ("*", "all", "ALL"))
def test_direct_cell_id_wildcard_still_intersects_other_selectors(
    wildcard: str,
) -> None:
    arguments = manual_campaign.build_parser().parse_args(
        ("inspect", "--cell-id", wildcard, "--model", "scalar_contact")
    )

    _selection, cells = manual_campaign.selection_from_arguments(arguments)

    assert cells
    assert all(cell.measurement.model is not None for cell in cells)
    assert all(cell.measurement.model.value == "scalar_contact" for cell in cells)


def test_summary_publication_replaces_stale_files_and_empty_success(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    path, counts = manual_campaign._publish_campaign_summary_ids(
        service,
        {
            "error": ["cell-b", "cell-a", "cell-a"],
            "generation_limit": {"cell-c"},
            "ok": {"ignored"},
        },
    )

    assert counts == {"error": 2, "generation_limit": 1}
    assert (path / "error.txt").read_text(encoding="utf-8") == (
        "cell-a\ncell-b\n"
    )
    assert (path / "generation_limit.txt").read_text(encoding="utf-8") == (
        "cell-c\n"
    )
    (path / ".DS_Store").write_bytes(b"stale Finder metadata")

    replacement, counts = manual_campaign._publish_campaign_summary_ids(service, {})
    assert replacement == path
    assert counts == {}
    assert tuple(path.iterdir()) == ()
    publication_root = (
        service.paths.coordination_root / "campaign-summary-publication"
    )
    assert publication_root.is_dir()
    assert tuple(publication_root.iterdir()) == ()
    assert not tuple(path.parent.glob(".campaign_summary_ids.*"))


def test_epyc_n9_codec_limit_summary_round_trips_to_exact_run_selection(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell_id = "matrix-recurrence-builtin-sm-lc-n9-gg-gluons-selected-flow"
    summary, _counts = manual_campaign._publish_campaign_summary_ids(
        service,
        {"error": {cell_id}},
    )
    arguments = manual_campaign.build_parser().parse_args(
        ("run", "--dry-run", "--cell-id-file", str(summary / "error.txt"))
    )

    _selection, cells = manual_campaign.selection_from_arguments(arguments)

    assert tuple(cell.cell_id for cell in cells) == (cell_id,)


def test_interrupted_summary_contains_observed_not_unstarted_cells() -> None:
    state = DashboardState(
        instance_id="invocation",
        selected_ids=("running", "failed", "unstarted"),
        recycled_ids=set(),
        static_na_ids=set(),
    )
    state.workers["running"] = WorkerView("running", status="running")
    state.workers["failed"] = WorkerView("failed", status="error")
    state.invocation_evidence_ids.update({"running", "failed"})

    categories = manual_campaign._campaign_summary_categories(
        state=state,
        interrupted=True,
    )

    assert categories == {"interrupted": {"running"}, "error": {"failed"}}


def test_cancelled_outcome_is_omitted_without_started_observation() -> None:
    state = DashboardState(
        instance_id="invocation",
        selected_ids=("started", "unstarted"),
        recycled_ids=set(),
        static_na_ids=set(),
    )
    state.workers["started"] = WorkerView("started", status="cancelled")
    state.invocation_evidence_ids.add("started")
    result = CampaignResult(
        planned=(),
        outcomes=(
            CellOutcome("started", "cancelled", "worker cancelled"),
            CellOutcome("unstarted", "cancelled", "not started"),
        ),
    )

    categories = manual_campaign._campaign_summary_categories(
        result=result,
        state=state,
        interrupted=True,
    )

    assert categories == {"interrupted": {"started"}}


def test_lease_finished_cancelled_not_started_is_not_interruption_evidence(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    state = DashboardState(
        instance_id="invocation",
        selected_ids=("queued", "started"),
        recycled_ids=set(),
        static_na_ids=set(),
    )
    lease = manual_campaign.LeaseManager(service, state)
    lease.observe(
        {
            "event": "finished",
            "cell_id": "queued",
            "status": "cancelled",
            "detail": "not started",
        }
    )
    lease.observe({"event": "started", "cell_id": "started"})
    lease.observe(
        {
            "event": "finished",
            "cell_id": "started",
            "status": "cancelled",
            "detail": "worker terminated by cancellation",
        }
    )
    result = CampaignResult(
        planned=(),
        outcomes=(
            CellOutcome("queued", "cancelled", "not started"),
            CellOutcome("started", "cancelled", "worker cancelled"),
        ),
    )

    categories = manual_campaign._campaign_summary_categories(
        result=result,
        state=state,
        interrupted=True,
    )
    lease.close()

    assert state.invocation_evidence_ids == {"started"}
    assert categories == {"interrupted": {"started"}}


def test_summary_omits_success_and_recycled_cells_but_keeps_future_statuses() -> None:
    state = DashboardState(
        instance_id="invocation",
        selected_ids=("recycled-cap", "ok", "future", "static"),
        recycled_ids={"recycled-cap"},
        static_na_ids={"static"},
    )
    state.workers["recycled-cap"] = WorkerView(
        "recycled-cap",
        status="memory_limit",
        recycled=True,
    )
    result = CampaignResult(
        planned=(),
        outcomes=(
            CellOutcome("ok", "ok", "complete"),
            CellOutcome("future", "Native Crash / Retry", "failed"),
        ),
    )

    categories = manual_campaign._campaign_summary_categories(
        static_na_ids=state.static_na_ids,
        result=result,
        state=state,
    )

    assert categories == {
        "native_crash_retry": {"future"},
        "static_na": {"static"},
    }


def test_cleanup_warning_is_retained_for_dashboard_and_printed_headlessly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _service(tmp_path)
    state = DashboardState(
        instance_id="invocation",
        selected_ids=("cell",),
        recycled_ids=set(),
        static_na_ids=set(),
    )
    lease = manual_campaign.LeaseManager(service, state)

    lease.observe(
        {
            "event": "cleanup-warning",
            "cell_id": "cell",
            "detail": "DiskFullError: 0 bytes available",
        }
    )
    manual_campaign._print_cleanup_warnings(
        state,
        manual_campaign.Palette(False),
    )
    lease.close()

    assert state.cleanup_warnings == [
        ("cell", "DiskFullError: 0 bytes available")
    ]
    assert state.workers["cell"].events[-1] == (
        "cleanup warning: DiskFullError: 0 bytes available"
    )
    assert capsys.readouterr().err == (
        "artifact cleanup skipped for cell: "
        "DiskFullError: 0 bytes available\n"
    )


def test_obsolete_attempt_archives_metadata_but_removes_heavy_payload(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    current = _publish_with_payload(store, "cell", "current")

    archived = store.archive_obsolete_attempts(
        "cell",
        consumer_cell_ids=("cell",),
    )

    assert archived == (old.attempt_id,)
    history = store._attempt_history_cell_root("cell") / old.attempt_id
    assert (history / "manifest.json").is_file()
    assert (history / "result.json").is_file()
    assert (history / "worker.log").read_text(encoding="utf-8") == "old\n"
    assert not (history / "artifact").exists()
    assert store.load_current("cell").attempt_id == current.attempt_id
    assert (current.manifest_path.parent / "artifact/payload.bin").is_file()


def test_equivalent_consumer_keeps_superseded_owner_until_reference_moves(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    owner = _publish_with_payload(store, "owner", "owner")
    current = _publish_with_payload(store, "owner", "current")
    consumer = store.new_attempt("consumer", ArtifactPolicy.REGENERATE)
    consumer.publish(
        {
            "status": "ok",
            "artifact": {"path": str(owner.manifest_path.parent / "artifact")},
        }
    )

    assert store.archive_obsolete_attempts(
        "owner",
        consumer_cell_ids=("owner", "consumer"),
    ) == ()
    assert (owner.manifest_path.parent / "artifact/payload.bin").is_file()

    _publish_with_payload(store, "consumer", "independent")
    assert store.archive_obsolete_attempts(
        "owner",
        consumer_cell_ids=("owner", "consumer"),
    ) == (owner.attempt_id,)
    assert store.load_current("owner").attempt_id == current.attempt_id


def test_active_unsealed_attempt_is_not_archived(tmp_path: Path) -> None:
    store = _store(tmp_path)
    current = _publish_with_payload(store, "cell", "current")
    active = store.new_attempt("cell", ArtifactPolicy.REGENERATE, based_on=current)
    active.path("worker.log").write_text("still running\n", encoding="utf-8")

    assert store.archive_obsolete_attempts(
        "cell", consumer_cell_ids=("cell",)
    ) == ()
    assert active.root.is_dir()
    assert not (active.root / "manifest.json").exists()


def test_malformed_current_pointer_prevents_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    current = _publish_with_payload(store, "cell", "current")
    pointer_path = current.manifest_path.parent.parent.parent / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["manifest_path"] = "../unsafe/manifest.json"
    pointer_path.write_text(json.dumps(pointer) + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="canonical attempt manifest"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert old.manifest_path.is_file()
    assert current.manifest_path.is_file()


def test_symlinked_cell_root_fails_closed_for_read_and_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    external = tmp_path / "escaped-cell"
    external.mkdir()
    cell_root = store._cell_root("cell")
    cell_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(ManifestValidationError, match="cell root is a symbolic link"):
        store.load_current("cell", missing_ok=True)
    with pytest.raises(ManifestValidationError, match="cell root is a symbolic link"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert tuple(external.iterdir()) == ()


def test_symlinked_current_pointer_fails_closed_for_read_and_cleanup(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    current = _publish_with_payload(store, "cell", "current")
    pointer = store._cell_root("cell") / "current.json"
    external = tmp_path / "escaped-current.json"
    pointer.replace(external)
    pointer.symlink_to(external)

    with pytest.raises(ManifestValidationError, match="pointer is a symbolic link"):
        store.load_current("cell")
    with pytest.raises(ManifestValidationError, match="pointer is a symbolic link"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert current.manifest_path.is_file()


def test_symlinked_current_manifest_fails_closed_for_read_and_cleanup(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    current = _publish_with_payload(store, "cell", "current")
    external = tmp_path / "escaped-manifest.json"
    current.manifest_path.replace(external)
    current.manifest_path.symlink_to(external)

    with pytest.raises(ManifestValidationError, match="manifest_path is not a regular"):
        store.load_current("cell")
    with pytest.raises(ManifestValidationError, match="manifest_path is not a regular"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert old.manifest_path.is_file()


def test_attempt_history_collision_prevents_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    _publish_with_payload(store, "cell", "current")
    collision = store._attempt_history_cell_root("cell") / old.attempt_id
    collision.mkdir(parents=True)

    with pytest.raises(ArtifactStoreError, match="history collision"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert old.manifest_path.is_file()


def test_catalog_reconciliation_is_bounded_and_preserves_current(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    old = _publish_with_payload(service.store, cell.cell_id, "old")
    current = _publish_with_payload(service.store, cell.cell_id, "current")

    assert reconcile_attempt_history(service, (cell,)) == ()
    assert service.store.load_current(cell.cell_id).attempt_id == current.attempt_id
    assert not old.manifest_path.exists()
    assert (
        service.store._attempt_history_cell_root(cell.cell_id)
        / old.attempt_id
        / "manifest.json"
    ).is_file()


def test_scheduler_runs_cleanup_after_each_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    old = _publish_with_payload(service.store, cell.cell_id, "old")
    current = _publish_with_payload(service.store, cell.cell_id, "current")
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            source_identity_override=ReportSourceIdentity("a" * 40, "b" * 40, ()),
            remove_heavy_attempt_artifacts=True,
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_run_cell_in_lane",
        lambda _planned: CellOutcome(cell.cell_id, "reused", current.attempt_id),
    )

    outcome = scheduler._run_cell(PlannedCell(cell, False, None, 0))

    assert outcome.status == "reused"
    assert not old.manifest_path.exists()
    assert service.store.load_current(cell.cell_id).attempt_id == current.attempt_id


def test_all_recycled_run_performs_startup_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    old = _publish_with_payload(service.store, cell.cell_id, "old")
    current = _publish_with_payload(service.store, cell.cell_id, "current")
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    lightweight_result = {
        "status": "ok",
        "provenance": {"report_source_revision": source.revision},
    }
    lightweight = LightweightCurrent(
        cell.cell_id,
        current.attempt_id,
        current.result_path,
        lightweight_result,
        True,
        True,
        "reusable",
        current,
    )
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_args, **_kwargs: {cell.cell_id: lightweight},
    )
    monkeypatch.setattr(manual_campaign, "plan_campaign", lambda *_a, **_k: ())
    monkeypatch.setattr(
        manual_campaign,
        "require_measurement_ready",
        lambda _source: None,
    )
    monkeypatch.setattr(
        manual_campaign,
        "update_source_marker",
        lambda *_args, **_kwargs: False,
    )
    arguments = manual_campaign.build_parser().parse_args(
        ("run", "--no-dashboard", "--cell-id", cell.cell_id)
    )

    exit_code = manual_campaign._run_campaign(
        arguments,
        repo_root=tmp_path,
        service=service,
        source=source,
        cells=(cell,),
        palette=manual_campaign.Palette(False),
    )

    assert exit_code == 0
    assert not old.manifest_path.exists()
    assert service.store.load_current(cell.cell_id).attempt_id == current.attempt_id


def test_immediate_scheduler_preflight_error_preserves_previous_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    summary, _counts = manual_campaign._publish_campaign_summary_ids(
        service,
        {"error": {"previous-cell"}},
    )
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        manual_campaign,
        "plan_campaign",
        lambda *_args, **_kwargs: (PlannedCell(cell, False, None, 0),),
    )
    monkeypatch.setattr(
        manual_campaign,
        "_bind_original_amplicol_if_required",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(manual_campaign, "require_measurement_ready", lambda _s: None)
    monkeypatch.setattr(manual_campaign, "update_source_marker", lambda *_a: False)
    monkeypatch.setattr(
        manual_campaign,
        "reconcile_attempt_history",
        lambda *_args, **_kwargs: (),
    )

    class ImmediateFailureScheduler:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, _planned: object) -> CampaignResult:
            raise RuntimeError("plan preflight failed before dispatch")

    monkeypatch.setattr(
        manual_campaign,
        "CampaignScheduler",
        ImmediateFailureScheduler,
    )
    arguments = manual_campaign.build_parser().parse_args(
        ("run", "--no-dashboard", "--cell-id", cell.cell_id)
    )

    with pytest.raises(RuntimeError, match="before dispatch"):
        manual_campaign._run_campaign(
            arguments,
            repo_root=tmp_path,
            service=service,
            source=source,
            cells=(cell,),
            palette=manual_campaign.Palette(False),
        )

    assert (summary / "error.txt").read_text(encoding="utf-8") == (
        "previous-cell\n"
    )
    assert tuple(path.name for path in summary.iterdir()) == ("error.txt",)


def test_concurrent_summary_publications_are_each_coherent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def publish(status: str, cell_id: str) -> None:
        try:
            barrier.wait(timeout=5)
            manual_campaign._publish_campaign_summary_ids(
                service,
                {status: {cell_id}},
            )
        except BaseException as error:
            errors.append(error)

    threads = (
        threading.Thread(target=publish, args=("error", "cell-error")),
        threading.Thread(
            target=publish,
            args=("memory_limit", "cell-memory"),
        ),
    )
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    summary = service.paths.docs_dir / "campaign_summary_ids"
    files = tuple(summary.iterdir())
    assert len(files) == 1
    assert (files[0].name, files[0].read_text(encoding="utf-8")) in {
        ("error.txt", "cell-error\n"),
        ("memory_limit.txt", "cell-memory\n"),
    }
    assert not tuple(summary.parent.glob(".campaign_summary_ids.*"))


def test_scheduler_cleanup_flag_defaults_off_and_invocation_is_validated() -> None:
    assert CampaignSettings().remove_heavy_attempt_artifacts is False
    settings = CampaignSettings(campaign_invocation_id="abc")
    assert settings.campaign_invocation_id == "abc"
    with pytest.raises(ValueError, match="campaign_invocation_id"):
        CampaignSettings(campaign_invocation_id="bad\nvalue")


def test_compact_attempt_inventory_excludes_only_heavy_artifact(tmp_path: Path) -> None:
    root = tmp_path / "attempt"
    (root / "artifact/nested").mkdir(parents=True)
    (root / "artifact/nested/data.bin").write_bytes(b"heavy")
    (root / "worker.log").write_text("diagnostic\n", encoding="utf-8")
    (root / "worker-progress.jsonl").write_text("{}\n", encoding="utf-8")

    assert _attempt_files(root, include_heavy_artifact=False) == (
        "worker-progress.jsonl",
        "worker.log",
    )
    assert "artifact/nested/data.bin" in _attempt_files(root)


def test_manual_campaign_enables_cleanup_unless_explicitly_disabled() -> None:
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    default_arguments = manual_campaign.build_parser().parse_args(("run", "--dry-run"))
    keep_arguments = manual_campaign.build_parser().parse_args(
        ("run", "--dry-run", "--no-artifacts-removal")
    )

    assert manual_campaign._campaign_settings(
        default_arguments, source
    ).remove_heavy_attempt_artifacts
    assert not manual_campaign._campaign_settings(
        default_arguments, source
    ).discard_cancelled_attempts
    assert not manual_campaign._campaign_settings(
        keep_arguments, source
    ).remove_heavy_attempt_artifacts


def test_cleanup_skips_busy_owner_use_lock(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    _publish_with_payload(store, "cell", "current")
    acquired = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with store.named_lock(f"campaign-artifact-use-{old.attempt_id}"):
            acquired.set()
            assert release.wait(timeout=5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(timeout=5)
    try:
        assert store.archive_obsolete_attempts(
            "cell", consumer_cell_ids=("cell",)
        ) == ()
        assert old.manifest_path.is_file()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_unsafe_heavy_payload_fails_closed_before_attempt_move(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    _publish_with_payload(store, "cell", "current")
    (old.manifest_path.parent / "artifact/unsafe-link").symlink_to(tmp_path)

    with pytest.raises(ArtifactStoreError, match="unsafe member"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert old.manifest_path.is_file()
    assert store.load_current("cell") is not None


def test_unsafe_root_diagnostic_fails_closed_before_attempt_move(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    _publish_with_payload(store, "cell", "current")
    worker_log = old.manifest_path.parent / "worker.log"
    worker_log.unlink()
    worker_log.symlink_to(tmp_path / "outside.log")

    with pytest.raises(ArtifactStoreError, match="unsafe member"):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert old.manifest_path.is_file()
    assert worker_log.is_symlink()
    assert store.load_current("cell") is not None


def test_atomic_report_write_normalizes_enospc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "result.json"

    def no_space(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(artifacts_module.os, "replace", no_space)
    with pytest.raises(DiskFullError, match=r"disk full while writing .*result.json"):
        artifacts_module._atomic_write_json(destination, {"status": "ok"})


def test_attempt_archive_normalizes_enospc_without_moving_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    old = _publish_with_payload(store, "cell", "old")
    _publish_with_payload(store, "cell", "current")

    def no_space(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(artifacts_module.os, "replace", no_space)
    with pytest.raises(
        DiskFullError,
        match=r"disk full while writing .*attempt-history",
    ):
        store.archive_obsolete_attempts("cell", consumer_cell_ids=("cell",))

    assert old.manifest_path.is_file()


def test_summary_publication_normalizes_enospc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)

    def no_space(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(manual_campaign.os, "replace", no_space)
    with pytest.raises(
        DiskFullError,
        match=r"disk full while writing .*campaign_summary_ids",
    ):
        manual_campaign._publish_campaign_summary_ids(
            service,
            {"error": {"cell"}},
        )

    publication_root = (
        service.paths.coordination_root / "campaign-summary-publication"
    )
    assert tuple(publication_root.iterdir()) == ()


def test_controller_atomic_json_normalizes_enospc_with_target_and_free_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "worker-result.json"

    def no_space(_source: object, _destination: object) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(manual_campaign.os, "replace", no_space)
    with pytest.raises(
        DiskFullError,
        match=(
            r"disk full while writing .*worker-result\.json; "
            r"[0-9]+ bytes available"
        ),
    ):
        manual_campaign._atomic_json(destination, {"status": "error"})

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".worker-result.json.*.tmp"))


def test_scheduler_archive_helper_uses_catalog_equivalence(tmp_path: Path) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    )
    old = _publish_with_payload(service.store, cell.cell_id, "old")
    _publish_with_payload(service.store, cell.cell_id, "current")

    assert archive_cell_attempt_history(service, cell) == (old.attempt_id,)
