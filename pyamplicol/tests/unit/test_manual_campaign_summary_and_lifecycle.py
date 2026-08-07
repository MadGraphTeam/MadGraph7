# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import errno
import json
import threading
from collections.abc import Callable
from concurrent.futures import Future
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


def test_fail_fast_parser_help_and_steering_guide_are_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = manual_campaign.build_parser()
    arguments = parser.parse_args(("run", "--fail-fast"))
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(("run", "--help"))
    run_help = capsys.readouterr().out

    assert arguments.fail_fast
    assert help_exit.value.code == 0
    assert "--fail-fast" in run_help
    assert "campaign_summary_ids/fail_fast_failure.log" in run_help
    assert "--fail-fast --workers 4 --table matrix" in manual_campaign.STEERING_GUIDE


def test_fail_fast_terminal_classification_ignores_stop_cancellation_but_not_recycled(
) -> None:
    cancellation = threading.Event()
    observed: list[dict[str, object]] = []
    observer = manual_campaign._FailFastObserver(
        lambda payload: observed.append(dict(payload)),
        cancellation,
    )

    observer.observe(
        {"event": "finished", "cell_id": "ok", "status": "skipped-current"}
    )
    observer.observe(
        {"event": "finished", "cell_id": "stopped", "status": "cancelled"}
    )
    assert observer.failure is None
    assert not cancellation.is_set()

    observer.observe(
        {
            "event": "finished",
            "cell_id": "unexpected-recycled-terminal",
            "status": "recycled",
            "detail": "not a scheduler success status",
        }
    )
    observer.observe(
        {"event": "finished", "cell_id": "later", "status": "error"}
    )

    assert cancellation.is_set()
    assert observer.failure is not None
    assert observer.failure.cell_id == "unexpected-recycled-terminal"
    assert len(observed) == 4


def test_fail_fast_all_static_selection_creates_no_failure_report(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is not None
    )
    arguments = manual_campaign.build_parser().parse_args(
        ("run", "--fail-fast", "--no-dashboard", "--cell-id", cell.cell_id)
    )

    exit_code = manual_campaign._run_campaign(
        arguments,
        repo_root=tmp_path,
        service=service,
        source=ReportSourceIdentity("a" * 40, "b" * 40, ()),
        cells=(cell,),
        palette=manual_campaign.Palette(False),
    )

    summary = service.paths.docs_dir / "campaign_summary_ids"
    assert exit_code == 0
    assert (summary / "static_na.txt").read_text(encoding="utf-8") == (
        f"{cell.cell_id}\n"
    )
    assert not (summary / manual_campaign.FAIL_FAST_FAILURE_LOG).exists()


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
    assert not (path / manual_campaign.FAIL_FAST_FAILURE_LOG).exists()
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


def test_fail_fast_report_contains_exact_cell_attempt_and_reproduction_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n2-dd-z-jets-selected-flow"
    )
    attempt = service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE)
    artifact = attempt.path("artifact")
    artifact.mkdir()
    attempt.path("artifact/payload.bin").write_bytes(b"retained failure payload")
    worker_log = attempt.path("worker.log")
    worker_log.write_text("worker failed\n", encoding="utf-8")
    worker_progress = attempt.path("worker-progress.jsonl")
    worker_progress.write_text('{"phase":"generation"}\n', encoding="utf-8")
    attempt.write_json(
        "worker-result.json",
        {
            "status": "error",
            "failure": {
                "kind": "ExactFailureClass",
                "message": "exact failure message",
            },
            "artifact": {"path": str(artifact)},
        },
    )
    attempt.mark_failed(
        "ExactFailureClass: exact failure message",
        artifact_paths=(
            "artifact/payload.bin",
            "worker-result.json",
            "worker.log",
            "worker-progress.jsonl",
        ),
    )
    state = DashboardState(
        instance_id="exact-invocation-id",
        selected_ids=(cell.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
    )
    state.workers[cell.cell_id] = WorkerView(
        cell.cell_id,
        status="error",
        attempt_id=attempt.attempt_id,
        log_path=str(worker_log),
        progress_path=str(worker_progress),
        reproduce_prepare="pyamplicol model compile --exact",
        reproduce_generate="pyamplicol generate --exact",
        reproduce_profile="pyamplicol profile --exact",
    )
    failure = manual_campaign._FailFastTerminalFailure(
        observed_at_utc="2026-08-07T12:34:56Z",
        cell_id=cell.cell_id,
        status="error",
        detail=attempt.attempt_id,
        terminal_detail="worker reported an exact terminal detail",
        completed_at_ns=1,
    )
    stale_summary = service.paths.docs_dir / "campaign_summary_ids"
    stale_summary.mkdir()
    (stale_summary / "stale.txt").write_text("stale\n", encoding="utf-8")

    summary_path = manual_campaign._publish_completion_summary(
        service,
        {"error": {cell.cell_id}},
        state=state,
        fail_fast_failure=failure,
        invocation_command=(
            "steer_performance_campaign.py run --fail-fast --cell-id "
            f"{cell.cell_id}"
        ),
        palette=manual_campaign.Palette(False),
    )

    report_path = summary_path / manual_campaign.FAIL_FAST_FAILURE_LOG
    report = report_path.read_text(encoding="utf-8")
    assert report_path.name == "fail_fast_failure.log"
    assert not (summary_path / "stale.txt").exists()
    for expected in (
        "timestamp_utc: 2026-08-07T12:34:56Z",
        "campaign_invocation_id: exact-invocation-id",
        "campaign_invocation: steer_performance_campaign.py run --fail-fast",
        f"cell_id: {cell.cell_id}",
        "process_family_id: 1",
        f"process_key: {cell.process_key}",
        f"process: {cell.process}",
        f"n_final: {cell.n_final}",
        "mode: recurrence",
        "workload: selected-flow",
        "accuracy: lc",
        "terminal_status: error",
        f"outcome_detail: {attempt.attempt_id}",
        "terminal_detail: worker reported an exact terminal detail",
        "failure_class: ExactFailureClass",
        "failure_message: exact failure message",
        f"attempt_uuid: {attempt.attempt_id}",
        f"attempt_root: {attempt.root}",
        f"artifact_path: {artifact}",
        f"worker_log_path: {worker_log}",
        f"worker_progress_path: {worker_progress}",
        "reproduce_prepare: pyamplicol model compile --exact",
        "reproduce_generate: pyamplicol generate --exact",
        "reproduce_profile: pyamplicol profile --exact",
    ):
        assert expected in report
    captured = capsys.readouterr()
    assert str(report_path) in captured.err
    assert str(artifact) in captured.err
    assert str(worker_log) in captured.err
    assert (summary_path / "error.txt").read_text(encoding="utf-8") == (
        f"{cell.cell_id}\n"
    )


def test_fail_fast_cancels_live_worker_and_never_dispatches_third_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cells = tuple(
        REPORT_CATALOG.cell(
            f"matrix-recurrence-builtin-sm-lc-n{n_final}-dd-z-jets-selected-flow"
        )
        for n_final in (1, 2, 3)
    )
    cancellation = threading.Event()
    observer = manual_campaign._FailFastObserver(lambda _payload: None, cancellation)
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            workers=2,
            source_identity_override=ReportSourceIdentity("a" * 40, "b" * 40, ()),
            progress_observer=observer.observe,
            cancellation_requested=cancellation.is_set,
        ),
    )
    live_started = threading.Event()
    live_saw_cancellation = threading.Event()
    launched: list[str] = []
    launched_guard = threading.Lock()

    def run_lane(
        _scheduler: CampaignScheduler,
        item: PlannedCell,
    ) -> CellOutcome:
        with launched_guard:
            launched.append(item.cell.cell_id)
        if item.cell.cell_id == cells[0].cell_id:
            assert live_started.wait(timeout=2.0)
            return CellOutcome(item.cell.cell_id, "error", "first failure")
        if item.cell.cell_id == cells[1].cell_id:
            live_started.set()
            assert cancellation.wait(timeout=2.0)
            live_saw_cancellation.set()
            return CellOutcome(
                item.cell.cell_id,
                "cancelled",
                "worker terminated by cancellation",
            )
        raise AssertionError("fail-fast dispatched a third cell")

    monkeypatch.setattr(CampaignScheduler, "_run_cell_in_lane", run_lane)
    result = scheduler.run(
        tuple(
            PlannedCell(cell, False, None, index)
            for index, cell in enumerate(cells)
        )
    )

    assert set(launched) == {cells[0].cell_id, cells[1].cell_id}
    assert live_saw_cancellation.is_set()
    assert cells[2].cell_id not in launched
    assert observer.failure is not None
    assert observer.failure.cell_id == cells[0].cell_id
    assert {outcome.status for outcome in result.outcomes} == {"error", "cancelled"}


def test_scheduler_rechecks_fail_fast_before_each_multiworker_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    cells = tuple(
        REPORT_CATALOG.cell(
            f"matrix-recurrence-builtin-sm-lc-n{n_final}-dd-z-jets-selected-flow"
        )
        for n_final in (1, 2, 3)
    )
    cancellation = threading.Event()
    observer = manual_campaign._FailFastObserver(lambda _payload: None, cancellation)
    submitted: list[str] = []

    class ImmediateExecutor:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> ImmediateExecutor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self,
            function: Callable[[PlannedCell], CellOutcome],
            item: PlannedCell,
        ) -> Future[CellOutcome]:
            submitted.append(item.cell.cell_id)
            future: Future[CellOutcome] = Future()
            try:
                future.set_result(function(item))
            except BaseException as error:
                future.set_exception(error)
            return future

    monkeypatch.setattr(
        "tools.performance_report.scheduler.ThreadPoolExecutor",
        ImmediateExecutor,
    )
    monkeypatch.setattr(
        CampaignScheduler,
        "_run_cell_in_lane",
        lambda _scheduler, item: CellOutcome(
            item.cell.cell_id,
            "error",
            "synchronous first failure",
        ),
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            workers=3,
            source_identity_override=ReportSourceIdentity("a" * 40, "b" * 40, ()),
            progress_observer=observer.observe,
            cancellation_requested=cancellation.is_set,
        ),
    )

    result = scheduler.run(
        tuple(
            PlannedCell(cell, False, None, index)
            for index, cell in enumerate(cells)
        )
    )

    assert submitted == [cells[0].cell_id]
    assert tuple(outcome.cell_id for outcome in result.outcomes) == (
        cells[0].cell_id,
    )


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


def test_unverified_summary_round_trips_to_exact_retry_selection(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell_id = "matrix-compiled-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    result = CampaignResult(
        planned=(),
        outcomes=(
            CellOutcome(
                cell_id,
                "unverified",
                "independent numerical authority unavailable",
            ),
        ),
    )
    categories = manual_campaign._campaign_summary_categories(result=result)
    summary, counts = manual_campaign._publish_campaign_summary_ids(
        service,
        categories,
    )

    assert counts == {"unverified": 1}
    selection_file = summary / "unverified.txt"
    assert selection_file.read_text(encoding="utf-8") == f"{cell_id}\n"
    arguments = manual_campaign.build_parser().parse_args(
        ("run", "--dry-run", "--cell-id-file", str(selection_file))
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


@pytest.mark.parametrize(
    ("extra_arguments", "cleanup_expected"),
    (((), False), (("--cleanup-artifacts",), True)),
)
def test_all_recycled_run_applies_selected_startup_cleanup_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_arguments: tuple[str, ...],
    cleanup_expected: bool,
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
        (
            "run",
            "--no-dashboard",
            "--cell-id",
            cell.cell_id,
            *extra_arguments,
        )
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
    assert old.manifest_path.exists() is not cleanup_expected
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


def test_fail_fast_first_failure_stops_later_scheduler_dispatch_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = _service(tmp_path)
    cells = tuple(
        REPORT_CATALOG.cell(
            f"matrix-recurrence-builtin-sm-lc-n{n_final}-dd-z-jets-selected-flow"
        )
        for n_final in (1, 2, 3)
    )
    planned = tuple(
        PlannedCell(cell, dependency=False, baseline_cell_id=None, rank=index)
        for index, cell in enumerate(cells)
    )
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    dispatched: list[str] = []

    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        manual_campaign,
        "plan_campaign",
        lambda *_args, **_kwargs: planned,
    )
    monkeypatch.setattr(
        manual_campaign,
        "_bind_original_amplicol_if_required",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(manual_campaign, "require_measurement_ready", lambda _s: None)
    monkeypatch.setattr(
        manual_campaign,
        "update_source_marker",
        lambda *_args, **_kwargs: False,
    )

    def terminal_failure(
        _scheduler: CampaignScheduler,
        item: PlannedCell,
    ) -> CellOutcome:
        dispatched.append(item.cell.cell_id)
        return CellOutcome(
            item.cell.cell_id,
            "validation_failed",
            "first exact failure",
            terminal_detail="independent audit mismatch",
        )

    monkeypatch.setattr(CampaignScheduler, "_run_cell_in_lane", terminal_failure)
    arguments = manual_campaign.build_parser().parse_args(
        (
            "run",
            "--fail-fast",
            "--no-dashboard",
            "--workers",
            "1",
            "--cell-id",
            *(cell.cell_id for cell in cells),
        )
    )
    arguments._campaign_invocation_command = (
        "steer_performance_campaign.py run --fail-fast --no-dashboard --workers 1"
    )

    exit_code = manual_campaign._run_campaign(
        arguments,
        repo_root=tmp_path,
        service=service,
        source=source,
        cells=cells,
        palette=manual_campaign.Palette(False),
    )

    assert exit_code == 1
    assert dispatched == [cells[0].cell_id]
    summary_path = service.paths.docs_dir / "campaign_summary_ids"
    report_path = summary_path / manual_campaign.FAIL_FAST_FAILURE_LOG
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert f"cell_id: {cells[0].cell_id}" in report
    assert "terminal_status: validation_failed" in report
    assert "outcome_detail: first exact failure" in report
    assert "terminal_detail: independent audit mismatch" in report
    assert "attempt_uuid: <unavailable>" in report
    assert not any(cell.cell_id in report for cell in cells[1:])
    assert (summary_path / "validation_failed.txt").read_text(
        encoding="utf-8"
    ) == f"{cells[0].cell_id}\n"
    captured = capsys.readouterr()
    assert "Fail-fast stopped dispatch" in captured.err
    assert str(report_path) in captured.err


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


def test_manual_campaign_retains_artifacts_unless_cleanup_is_explicit() -> None:
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    default_arguments = manual_campaign.build_parser().parse_args(("run", "--dry-run"))
    cleanup_arguments = manual_campaign.build_parser().parse_args(
        ("run", "--dry-run", "--cleanup-artifacts")
    )
    fail_fast_cleanup_arguments = manual_campaign.build_parser().parse_args(
        ("run", "--dry-run", "--fail-fast", "--cleanup-artifacts")
    )

    assert not manual_campaign._campaign_settings(
        default_arguments, source
    ).remove_heavy_attempt_artifacts
    assert not manual_campaign._campaign_settings(
        default_arguments, source
    ).discard_cancelled_attempts
    assert manual_campaign._campaign_settings(
        cleanup_arguments, source
    ).remove_heavy_attempt_artifacts
    fail_fast_settings = manual_campaign._campaign_settings(
        fail_fast_cleanup_arguments,
        source,
    )
    assert not fail_fast_settings.remove_heavy_attempt_artifacts
    assert not fail_fast_settings.discard_cancelled_attempts


@pytest.mark.parametrize(
    ("extra_arguments", "expected"),
    (
        ((), "retained (default)"),
        (("--cleanup-artifacts",), "cleanup enabled (--cleanup-artifacts)"),
        (
            ("--fail-fast", "--cleanup-artifacts"),
            "retained (--fail-fast preserves failures and cancellations)",
        ),
    ),
)
def test_dry_run_prints_effective_artifact_cleanup_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_arguments: tuple[str, ...],
    expected: str,
) -> None:
    service = _service(tmp_path)
    cell = REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(manual_campaign, "plan_campaign", lambda *_a, **_k: ())
    monkeypatch.setattr(
        manual_campaign,
        "_dry_run_recipe_blocks",
        lambda *_args, **_kwargs: (),
    )
    arguments = manual_campaign.build_parser().parse_args(
        (
            "run",
            "--dry-run",
            "--no-color",
            "--cell-id",
            cell.cell_id,
            *extra_arguments,
        )
    )

    assert (
        manual_campaign._run_campaign(
            arguments,
            repo_root=tmp_path,
            service=service,
            source=source,
            cells=(cell,),
            palette=manual_campaign.Palette(False),
        )
        == 0
    )

    assert expected in capsys.readouterr().out
    assert not (
        service.paths.docs_dir
        / "campaign_summary_ids"
        / manual_campaign.FAIL_FAST_FAILURE_LOG
    ).exists()


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
