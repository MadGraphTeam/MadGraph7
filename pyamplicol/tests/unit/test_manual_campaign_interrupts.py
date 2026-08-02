# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.performance_report.manual_campaign as manual_campaign
from tools.performance_report.agreements import DIRECT_AGREEMENT_FIELD
from tools.performance_report.cache import empty_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ArtifactPolicy, ResultStatus
from tools.performance_report.resources import (
    GenerationPhaseEvidence,
    ResourceUsage,
    SupervisedResult,
)
from tools.performance_report.scheduler import (
    CampaignScheduler,
    CampaignSettings,
    PlannedCell,
)
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity


def _service(tmp_path: Path) -> ReportService:
    root = tmp_path / "repo"
    (root / "docs/arxiv").mkdir(parents=True)
    return ReportService(
        ReportPaths.from_repo(
            root,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )


def _source() -> ReportSourceIdentity:
    return ReportSourceIdentity("a" * 40, "b" * 40, ())


def _cell():
    return REPORT_CATALOG.cell("scalar-contact-n2-scalar-contact-contracted")


def _completed_measurement(
    source: ReportSourceIdentity,
) -> dict[str, object]:
    measurement = empty_measurement()
    measurement.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {},
            "validation": {
                "status": ResultStatus.OK.value,
                DIRECT_AGREEMENT_FIELD: [],
            },
            "resources": {},
            "provenance": source.provenance(),
        }
    )
    return measurement


def _publish_completed_current(
    service: ReportService,
    source: ReportSourceIdentity,
):
    return service.store.new_attempt(
        _cell().cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(_completed_measurement(source))


def _planned() -> PlannedCell:
    return PlannedCell(
        _cell(),
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )


def test_queued_cancellation_creates_no_failed_attempt_and_preserves_current(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    source = _source()
    current = _publish_completed_current(service, source)
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            artifact_policy=ArtifactPolicy.REGENERATE,
            source_identity_override=source,
            cancellation_requested=lambda: True,
        ),
    )

    outcome = scheduler._run_cell(_planned())

    assert outcome.status == "cancelled"
    assert outcome.detail == "not started"
    assert service.store.load_current(_cell().cell_id).attempt_id == current.attempt_id
    attempts = tuple(
        service.store._cell_root(_cell().cell_id).joinpath("attempts").iterdir()
    )
    assert tuple(path.name for path in attempts) == (current.attempt_id,)


@pytest.mark.parametrize(
    ("phase", "final_phase"),
    (
        ("generation", "generation"),
        ("profiling", "post-generation"),
    ),
)
def test_running_cancellation_is_interrupted_history_and_preserves_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    final_phase: str,
) -> None:
    service = _service(tmp_path)
    source = _source()
    current = _publish_completed_current(service, source)
    cancellation = threading.Event()

    def cancelled_worker(
        _command: object,
        **arguments: object,
    ) -> SupervisedResult:
        callback = arguments["cancellation_requested"]
        assert callable(callback)
        cancellation.set()
        assert callback()
        return SupervisedResult(
            -15,
            "cancelled",
            ResourceUsage(True, 0, 0, 0, 0.0, 0.1),
            GenerationPhaseEvidence(
                configured_timeout_seconds=3600.0,
                supervisor_reason="cancelled",
                authenticated=True,
                run_id=f"{phase}-run",
                worker_pid=123,
                final_sequence=2,
                final_phase=final_phase,
                generation_started_monotonic_ns=1,
                generation_finished_monotonic_ns=2,
                generation_elapsed_seconds=1.0,
                final_state_sha256="c" * 64,
            ),
        )

    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        cancelled_worker,
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            artifact_policy=ArtifactPolicy.REGENERATE,
            source_identity_override=source,
            cancellation_requested=cancellation.is_set,
            generation_time_limit_seconds=3600.0,
        ),
    )

    outcome = scheduler._run_cell(_planned())

    assert outcome.status == "cancelled"
    assert outcome.detail == "previous valid current preserved"
    assert service.store.load_current(_cell().cell_id).attempt_id == current.attempt_id
    attempt_root = service.store._cell_root(_cell().cell_id) / "attempts"
    manifests = {
        path.parent.name: json.loads(path.read_text(encoding="ascii"))
        for path in attempt_root.glob("*/manifest.json")
    }
    assert manifests[current.attempt_id]["status"] == "ok"
    interrupted = {
        attempt_id: manifest
        for attempt_id, manifest in manifests.items()
        if attempt_id != current.attempt_id
    }
    assert len(interrupted) == 1
    manifest = next(iter(interrupted.values()))
    assert manifest["status"] == "interrupted"
    assert manifest["result_path"] is None
    assert manifest["error"] == "worker terminated by cancellation"
    assert manifest["based_on"]["attempt_id"] == current.attempt_id
    assert {path.parent.name for path in attempt_root.glob("*/result.json")} == {
        current.attempt_id
    }


def test_manual_legacy_cancellation_discards_isolated_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    source = _source()
    cell = REPORT_CATALOG.cell("reference-amplicol-full-n1-dd-z-jets-contracted")
    source_repository = tmp_path / "original-amplicol"
    source_repository.mkdir()
    workspaces: list[Path] = []

    def workspace_paths(
        _scheduler: CampaignScheduler,
        attempt_id: str,
    ) -> tuple[Path, Path]:
        workspace = tmp_path / "legacy-workspaces" / attempt_id
        workspaces.append(workspace)
        return source_repository, workspace

    def cancelled_worker(
        command: list[str],
        **_arguments: object,
    ) -> SupervisedResult:
        source_option = command.index("--legacy-source-repository")
        workspace_option = command.index("--legacy-workspace")
        assert Path(command[source_option + 1]) == source_repository
        assert Path(command[workspace_option + 1]) == workspaces[0]
        assert "--legacy-copy-source" in command
        # Workspace preparation has moved into this supervised subprocess;
        # the controller must not copy or lock the source before supervision.
        assert not workspaces[0].exists()
        workspaces[0].mkdir(parents=True)
        return SupervisedResult(
            -15,
            "cancelled",
            ResourceUsage(True, 0, 0, 0, 0.0, 0.1),
            GenerationPhaseEvidence(
                configured_timeout_seconds=3600.0,
                supervisor_reason="cancelled",
                authenticated=True,
                run_id="manual-legacy-run",
                worker_pid=123,
                final_sequence=1,
                final_phase="generation",
                generation_started_monotonic_ns=1,
                generation_finished_monotonic_ns=None,
                generation_elapsed_seconds=0.1,
                final_state_sha256="d" * 64,
            ),
        )

    monkeypatch.setattr(
        CampaignScheduler,
        "_legacy_workspace_paths",
        workspace_paths,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.supervise_worker",
        cancelled_worker,
    )
    scheduler = CampaignScheduler(
        service,
        settings=CampaignSettings(
            artifact_policy=ArtifactPolicy.REGENERATE,
            source_identity_override=source,
            generation_time_limit_seconds=3600.0,
            manual_terminal_censors=True,
            discard_cancelled_attempts=True,
        ),
    )

    outcome = scheduler._run_cell(
        PlannedCell(
            cell,
            dependency=False,
            baseline_cell_id=None,
            rank=0,
        )
    )

    assert outcome.status == "cancelled"
    assert outcome.detail == "incomplete attempt discarded"
    assert len(workspaces) == 1 and not workspaces[0].exists()
    assert service.store.cell_attempt_ids(cell.cell_id) == ()


def test_closed_lease_cannot_be_recreated_by_late_worker_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    source = _source()
    monkeypatch.setattr(
        manual_campaign,
        "_repo_head",
        lambda _repo_root: source.revision,
    )
    state = manual_campaign.DashboardState(
        instance_id="interrupt-cleanup",
        selected_ids=(_cell().cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision=source.revision,
    )
    lease = manual_campaign.LeaseManager(service, state)
    lease.publish()
    assert lease.path.is_file()

    lease.close()
    lease.observe(
        {
            "event": "started",
            "cell_id": _cell().cell_id,
        }
    )
    lease.publish()

    assert not lease.path.exists()


def test_repeated_keyboard_interrupt_waits_for_cleanup_before_closing_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cancelled_callbacks: list[object] = []
    leases: list[object] = []
    threads: list[object] = []

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    class FakeLease:
        def __init__(self, _service: object, _state: object) -> None:
            self.closed = False
            leases.append(self)

        def publish(self) -> None:
            events.append("lease-published")

        def observe(self, _payload: object) -> None:
            return None

        def close(self) -> None:
            assert threads and not threads[0].is_alive()
            self.closed = True
            events.append("lease-closed")

    class FakeThread:
        def __init__(self, **_arguments: object) -> None:
            self.join_timeouts: list[float | None] = []
            self.alive = True
            threads.append(self)

        def start(self) -> None:
            events.append("worker-started")

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)
            if len(self.join_timeouts) == 1:
                events.append("join-interrupted")
                raise KeyboardInterrupt
            if len(self.join_timeouts) == 2:
                events.append("cleanup-delayed")
            if len(self.join_timeouts) == 22:
                self.alive = False
                events.append("cleanup-finished")

        def is_alive(self) -> bool:
            return self.alive

    class FakeScheduler:
        def __init__(self, _service: object, *, settings: object) -> None:
            del settings

        def run(self, _planned: object) -> None:
            raise AssertionError("fake controller thread must not execute")

    def settings(
        _arguments: object,
        _source: object,
        *,
        observer: object = None,
        cancelled: object = None,
        campaign_invocation_id: str | None = None,
    ) -> object:
        del observer
        if cancelled is not None:
            assert campaign_invocation_id
            cancelled_callbacks.append(cancelled)
        return object()

    def dashboard(*_arguments: object, **_keywords: object) -> None:
        del _keywords
        try:
            raise KeyboardInterrupt
        finally:
            events.append("terminal-restored")

    monkeypatch.setattr(manual_campaign, "lightweight_currents", lambda *_a, **_k: {})
    monkeypatch.setattr(
        manual_campaign,
        "plan_campaign",
        lambda *_a, **_k: (_planned(),),
    )
    monkeypatch.setattr(manual_campaign, "require_measurement_ready", lambda _s: None)
    monkeypatch.setattr(manual_campaign, "update_source_marker", lambda *_a: False)
    monkeypatch.setattr(
        manual_campaign,
        "_bind_original_amplicol_if_required",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        manual_campaign,
        "reconcile_attempt_history",
        lambda *_a, **_k: (),
    )
    monkeypatch.setattr(
        manual_campaign,
        "_publish_campaign_summary_ids",
        lambda *_a, **_k: (tmp_path / "campaign_summary_ids", {}),
    )
    monkeypatch.setattr(manual_campaign, "_campaign_settings", settings)
    monkeypatch.setattr(manual_campaign, "CampaignScheduler", FakeScheduler)
    monkeypatch.setattr(manual_campaign, "LeaseManager", FakeLease)
    monkeypatch.setattr(manual_campaign.threading, "Thread", FakeThread)
    monkeypatch.setattr(manual_campaign, "_run_live_dashboard", dashboard)
    monkeypatch.setattr(manual_campaign.sys, "stdin", Tty())
    monkeypatch.setattr(manual_campaign.sys, "stdout", Tty())
    stderr = Tty()
    monkeypatch.setattr(manual_campaign.sys, "stderr", stderr)
    arguments = SimpleNamespace(
        force_refresh=False,
        dry_run=False,
        no_dashboard=False,
        generation_time_limit=3600.0,
        ram_limit=30_000_000_000,
        worker_wall_limit=None,
        cores_per_worker=1,
        target_measurement_duration=0.1,
        batch_size=1,
        warmups=0,
        minimum_samples=5,
        termination_grace=0.01,
        no_artifacts_removal=False,
    )
    service = SimpleNamespace(
        store=object(),
        paths=SimpleNamespace(docs_dir=tmp_path),
    )

    result = manual_campaign._run_campaign(
        arguments,
        repo_root=tmp_path,
        service=service,
        source=_source(),
        cells=(_cell(),),
        palette=manual_campaign.Palette(False),
    )

    assert result == 130
    assert len(cancelled_callbacks) == 1
    callback = cancelled_callbacks[0]
    assert callable(callback) and callback()
    assert len(threads) == 1
    thread = threads[0]
    assert thread.join_timeouts == [0.25] * 22
    assert len(leases) == 1 and leases[0].closed
    assert events == [
        "lease-published",
        "worker-started",
        "terminal-restored",
        "join-interrupted",
        "cleanup-delayed",
        "cleanup-finished",
        "lease-closed",
    ]
    assert "still waiting for supervised worker-tree cleanup" in stderr.getvalue()
    assert "no leases or claims will be released early" in stderr.getvalue()
