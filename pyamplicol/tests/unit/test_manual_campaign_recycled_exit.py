# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

import tools.performance_report.manual_campaign as manual_campaign
from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import LightweightCurrent
from tools.performance_report.models import ExecutionMode
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity

ROOT = Path(__file__).resolve().parents[2]


def test_all_recycled_run_exits_before_lease_scheduler_or_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    measurable = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and REPORT_CATALOG.static_na_reason(cell) is None
    )
    static_na = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is not None
    )
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    arguments = manual_campaign.build_parser().parse_args(
        (
            "run",
            "--cell-id",
            measurable.cell_id,
            static_na.cell_id,
            "--no-dashboard",
        )
    )
    current = LightweightCurrent(
        cell_id=measurable.cell_id,
        attempt_id="reusable-attempt",
        result_path=tmp_path / "result.json",
        result={},
        complete=True,
        reusable=True,
        reason="reusable",
    )
    events: list[str] = []

    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_args, **_kwargs: {measurable.cell_id: current},
    )
    monkeypatch.setattr(
        manual_campaign,
        "plan_campaign",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        manual_campaign,
        "require_measurement_ready",
        lambda observed: events.append(f"ready:{observed.revision}"),
    )
    monkeypatch.setattr(
        manual_campaign,
        "update_source_marker",
        lambda _service, observed: events.append(f"marker:{observed.revision}"),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("all-recycled run reached worker/attempt setup")

    monkeypatch.setattr(manual_campaign, "LeaseManager", forbidden)
    monkeypatch.setattr(manual_campaign, "CampaignScheduler", forbidden)
    monkeypatch.setattr(ArtifactStore, "new_attempt", forbidden)

    result = manual_campaign._run_campaign(
        arguments,
        repo_root=ROOT,
        service=service,
        source=source,
        cells=(measurable, static_na),
        palette=manual_campaign.Palette(enabled=True),
    )

    assert result == 0
    assert events == [f"ready:{source.revision}", f"marker:{source.revision}"]
    output = capsys.readouterr().out
    assert "\x1b[" in output
    assert "selected entries" in output
    assert "measurable entries" in output
    assert "recycled measurable entries" in output
    assert "static N/A entries" in output
    assert "workers created" in output
    assert "attempts created" in output
    assert "All selected measurable entries were recycled (1 of 1)" in output
    assert "no workers or attempts were created" in output
    assert "1 selected static N/A entry remains policy unavailable" in output
    assert not (tmp_path / "coordination" / "manual-leases").exists()
    assert not tuple((tmp_path / "artifacts" / "cells").rglob("attempts"))
