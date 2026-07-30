# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import tools.performance_report.manual_campaign as manual_campaign
from tools.performance_report.manual_campaign import (
    DashboardState,
    LeaseManager,
    render_dashboard_frame,
)
from tools.performance_report.resources import (
    ProcessRecord,
    ProcessTreeSampler,
    _parse_ps_cpu_time,
    process_snapshot,
)
from tools.performance_report.service import ReportPaths, ReportService

ROOT = Path(__file__).resolve().parents[2]


def test_darwin_process_snapshot_reports_truthful_tree_cpu_time() -> None:
    output = " 100 1 1024 00:01.25\n 101 100 2048 1:02:03.50\n"

    @dataclass
    class Completed:
        returncode: int = 0
        stdout: str = output
        stderr: str = ""

    commands: list[tuple[str, ...]] = []

    def run_ps(command: tuple[str, ...]) -> Completed:
        commands.append(tuple(command))
        return Completed()

    records = process_snapshot("Darwin", ps_runner=run_ps)
    sample = ProcessTreeSampler(100).sample(records)

    assert commands == [("ps", "-axo", "pid=,ppid=,rss=,time=")]
    assert records[100].cpu_seconds == 1.25
    assert records[101].cpu_seconds == 3_723.5
    assert sample.cpu_seconds == 3_724.75
    assert sample.rss_bytes == 3 * 1024 * 1024
    assert _parse_ps_cpu_time("2-03:04:05.50") == 183_845.5
    partial = ProcessTreeSampler(100).sample(
        {
            100: ProcessRecord(100, 1, 100, 1.0),
            101: ProcessRecord(101, 100, 100, None),
        }
    )
    assert partial.cpu_seconds is None


def test_dashboard_uses_guard_memory_and_incremental_progress_and_log_tails(
    tmp_path: Path,
) -> None:
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell_id = "observability-cell"
    progress_path = tmp_path / "progress.jsonl"
    log_path = tmp_path / "worker.log"
    initial_progress = (
        json.dumps(
            {
                "event": "update",
                "completed": 2,
                "total": 4,
                "message": "building currents",
            }
        )
        + "\n"
    )
    initial_log = "loaded model\n\x1b[31mprofiling batch 2\x1b[0m\n"
    progress_path.write_text(initial_progress, encoding="utf-8")
    log_path.write_text(initial_log, encoding="utf-8")
    state = DashboardState(
        instance_id="observability",
        selected_ids=(cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
        memory_limit_bytes=1_000,
    )
    lease = LeaseManager(service, state)
    lease.observe(
        {
            "event": "worker",
            "cell_id": cell_id,
            "attempt_id": "attempt",
            "progress_path": str(progress_path),
            "log_path": str(log_path),
        }
    )
    resource = {
        "event": "resource",
        "cell_id": cell_id,
        "phase": "profiling",
        "pid": 100,
        "member_pids": [100, 101],
        "wall_seconds": 3.5,
        "cpu_seconds": 5.25,
        "current_rss_bytes": 800,
        "peak_rss_bytes": 900,
        "current_physical_footprint_bytes": 1_200,
        "peak_physical_footprint_bytes": 1_300,
        "current_guard_bytes": 1_200,
        "peak_guard_bytes": 1_300,
        "child_count": 1,
    }
    lease.observe(resource)

    worker = state.workers[cell_id]
    assert worker.cpu_seconds == 5.25
    assert worker.current_rss_bytes == 800
    assert worker.current_physical_footprint_bytes == 1_200
    assert worker.current_guard_bytes == 1_200
    assert worker.log_tail == ["loaded model", "profiling batch 2"]
    assert worker._progress_tail_state.last_read_bytes == len(initial_progress.encode())
    assert worker._log_tail_state.last_read_bytes == len(initial_log.encode())

    appended_progress = (
        json.dumps(
            {
                "event": "update",
                "completed": 3,
                "total": 4,
                "message": "profiling batch 3",
            }
        )
        + "\n"
    )
    appended_log = "profiling batch 3\n"
    with progress_path.open("a", encoding="utf-8") as stream:
        stream.write(appended_progress)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(appended_log)
    lease.observe(resource)

    assert worker.progress_completed == 3
    assert worker._progress_tail_state.last_read_bytes == len(
        appended_progress.encode()
    )
    assert worker._log_tail_state.last_read_bytes == len(appended_log.encode())
    assert worker.log_tail[-1] == "profiling batch 3"
    lease_payload = json.loads(lease.path.read_text(encoding="utf-8"))
    leased_worker = lease_payload["workers"][cell_id]
    assert leased_worker["current_rss_bytes"] == 800
    assert leased_worker["current_physical_footprint_bytes"] == 1_200
    assert leased_worker["current_guard_bytes"] == 1_200
    assert "_progress_tail_state" not in leased_worker

    wide = render_dashboard_frame(state, width=160, height=48)
    compact = render_dashboard_frame(state, width=80, height=24)
    for frame in (wide, compact):
        assert "progress" in frame
        assert "▰" in frame
        assert "Enforcement guard" in frame
        assert "1.20 kB" in frame
        assert "Recent log" in frame
        assert "profiling batch 3" in frame
    assert "RSS current 800 B" in wide
    assert "RSS peak 900 B" in wide
    assert "Physical current 1.20 kB" in wide
    assert "Cap 1.00 kB" in wide


@pytest.mark.parametrize(
    ("arguments", "environment_color", "expected"),
    (
        (("dashboard-snapshot",), None, True),
        (("dashboard-snapshot", "--no-color"), None, False),
        (("dashboard-snapshot",), "", False),
        (("dashboard-snapshot", "--cells-json"), None, True),
        (("dashboard-snapshot", "--cells-json", "--no-color"), None, False),
        (("dashboard-snapshot", "--cells-json"), "", False),
    ),
)
def test_dashboard_cli_propagates_color_policy(
    arguments: tuple[str, ...],
    environment_color: str | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if environment_color is None:
        monkeypatch.delenv("NO_COLOR", raising=False)
    else:
        monkeypatch.setenv("NO_COLOR", environment_color)
    observed: list[bool] = []

    def capture(*_args: Any, **kwargs: Any) -> object:
        observed.append(bool(kwargs["color"]))
        return [] if kwargs["cells"] else "frame"

    monkeypatch.setattr(manual_campaign, "render_dashboard_frame", capture)

    assert manual_campaign.main(arguments, repo_root=ROOT) == 0
    capsys.readouterr()
    assert observed == [expected]


def test_ratatui_style_toggle_removes_all_cell_styles() -> None:
    state = manual_campaign._snapshot_fixture(selected=5, recycled=1, completed=1)

    colored = render_dashboard_frame(
        state,
        width=120,
        height=36,
        cells=True,
        color=True,
    )
    plain = render_dashboard_frame(
        state,
        width=120,
        height=36,
        cells=True,
        color=False,
    )

    assert isinstance(colored, list)
    assert isinstance(plain, list)
    assert any(cell.get("fg") not in (None, 0) for cell in colored)
    assert all(cell.get("fg") in (None, 0) for cell in plain)
    assert all(cell.get("bg") in (None, 0) for cell in plain)
    assert all(cell.get("mods") in (None, 0) for cell in plain)


def test_dashboard_preserves_and_displays_three_independent_recurrence_clocks() -> None:
    state = manual_campaign._snapshot_fixture(selected=5, recycled=1, completed=1)
    workers = tuple(sorted(state.workers.values(), key=lambda item: item.cell_id))
    state.selected_index = next(
        index
        for index, worker in enumerate(workers)
        if "matrix-recurrence-" in worker.cell_id
    )
    selected = state.selected_worker()
    assert selected is not None

    decoded = manual_campaign._worker_from_lease(
        selected.cell_id,
        selected.as_dict(),
        peer_instance=None,
    )
    assert decoded.generation_engine == "recurrence"
    assert decoded.published_wall_seconds_per_point == pytest.approx(218.105e-6)
    assert decoded.published_evaluator_total_seconds_per_point == pytest.approx(
        217.812e-6
    )
    assert decoded.published_recurrence_core_seconds_per_point == pytest.approx(
        205.431e-6
    )

    frame = render_dashboard_frame(state, width=120, height=36)
    assert "Outer wall" in frame
    assert "218.105 μs/pt" in frame
    assert "Evaluator total" in frame
    assert "217.812 μs/pt" in frame
    assert "Recurrence core" in frame
    assert "205.431 μs/pt" in frame

    state.selected_index = next(
        index
        for index, worker in enumerate(workers)
        if "matrix-compiled-" in worker.cell_id
    )
    compiled_frame = render_dashboard_frame(state, width=120, height=36)
    assert "Recurrence core not applicable" in compiled_frame


def test_ratatui_terminal_session_uses_native_lifecycle_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ratatui_py

    events: list[tuple[str, str | None, str | None]] = []

    class FakeTerminal:
        def __init__(self) -> None:
            events.append(
                (
                    "init",
                    os.environ.get("RATATUI_FFI_ALTSCR"),
                    os.environ.get("RATATUI_FFI_NO_RAW"),
                )
            )

        def close(self) -> None:
            events.append(("close", None, None))

    def unsafe_session(**_options: object) -> object:
        raise AssertionError("the unsafe ratatui_py.terminal_session was called")

    monkeypatch.setattr(ratatui_py, "Terminal", FakeTerminal)
    monkeypatch.setattr(ratatui_py, "terminal_session", unsafe_session)
    monkeypatch.setenv("RATATUI_FFI_ALTSCR", "existing-alt")
    monkeypatch.setenv("RATATUI_FFI_NO_RAW", "existing-no-raw")

    with (
        pytest.raises(KeyboardInterrupt),
        manual_campaign._ratatui_terminal_session(),
    ):
        events.append(
            (
                "body",
                os.environ.get("RATATUI_FFI_ALTSCR"),
                os.environ.get("RATATUI_FFI_NO_RAW"),
            )
        )
        raise KeyboardInterrupt

    assert events == [
        ("init", "1", None),
        ("body", "existing-alt", "existing-no-raw"),
        ("close", None, None),
    ]
    assert os.environ["RATATUI_FFI_ALTSCR"] == "existing-alt"
    assert os.environ["RATATUI_FFI_NO_RAW"] == "existing-no-raw"
