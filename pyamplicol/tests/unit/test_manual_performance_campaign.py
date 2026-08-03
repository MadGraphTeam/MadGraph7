# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ast
import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import tools.performance_report.manual_campaign as manual_campaign
from tools.performance_report.artifacts import LockCancelledError
from tools.performance_report.cache import empty_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import (
    DEFAULT_MANUAL_EXPECTED_PAGE_COUNT,
    MANUAL_STATE_SCHEMA,
    Comparison,
    DashboardState,
    LeaseManager,
    LightweightCurrent,
    ManualCampaignError,
    ReproductionSettings,
    WorkerView,
    _handle_dashboard_key,
    _index_metadata_dirty_paths,
    _install_report_snapshot,
    _live_dashboard_snapshot,
    _manual_static_na_reason,
    _merge_lightweight_snapshot,
    _progress_detail_summary,
    _result_matches_cell,
    _snapshot_fixture,
    _tail_progress,
    build_parser,
    comparison_statistics,
    render_dashboard_frame,
    reproduction_recipe,
    selection_from_arguments,
)
from tools.performance_report.manual_campaign import main as campaign_main
from tools.performance_report.models import (
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from tools.performance_report.publication import (
    PublicationPortabilityError,
    portable_publication_value,
)
from tools.performance_report.publisher import _compile_pdf
from tools.performance_report.scheduler import CellOutcome, _CoordinationDeferred
from tools.performance_report.service import ReportPaths, ReportService

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "src/pyamplicol/_profiling_campaign"
CAMPAIGN_ARTIFACT_ROOT = PROFILE / "campaign_artifacts"


def _parse(*arguments: str):
    return build_parser().parse_args(arguments)


def test_campaign_state_roots_are_local_and_isolated_by_full_destination(
    tmp_path: Path,
) -> None:
    first_docs = tmp_path / "first" / "same-name"
    second_docs = tmp_path / "second" / "same-name"
    first_docs.mkdir(parents=True)
    second_docs.mkdir(parents=True)

    first = manual_campaign._campaign_report_paths(ROOT, first_docs)
    second = manual_campaign._campaign_report_paths(ROOT, second_docs)

    assert first.docs_dir == first_docs.resolve()
    assert first.artifact_root == first_docs.resolve() / "campaign_artifacts"
    assert first.coordination_root == first.artifact_root / "coordination"
    assert second.artifact_root == second_docs.resolve() / "campaign_artifacts"
    assert first.artifact_root != second.artifact_root
    first_service = ReportService(first, portable_current_results=True)
    second_service = ReportService(second, portable_current_results=True)
    with (
        first_service.store.named_lock("same-cell", timeout=0.0),
        second_service.store.named_lock("same-cell", timeout=0.0),
    ):
        pass
    attempt = first_service.store.new_attempt(
        "same-cell",
        manual_campaign.ArtifactPolicy.REGENERATE,
    )
    attempt.publish({"status": "ok"})
    assert first_service.store.load_current("same-cell") is not None
    assert second_service.store.load_current("same-cell", missing_ok=True) is None
    assert (
        second_service.store.lightweight_current_payload(
            "same-cell",
            missing_ok=True,
        )
        is None
    )

    legacy_store = manual_campaign.ArtifactStore(
        artifact_root=tmp_path / ".artifacts/performance-report/same-name",
        lock_root=tmp_path / ".artifacts/performance-report-coordination/same-name",
    )
    legacy_attempt = legacy_store.new_attempt(
        "legacy-cell",
        manual_campaign.ArtifactPolicy.REGENERATE,
    )
    legacy_attempt.publish({"status": "ok"})
    assert first_service.store.load_current("legacy-cell", missing_ok=True) is None
    assert second_service.store.load_current("legacy-cell", missing_ok=True) is None
    assert not (first_docs / ".artifacts").exists()
    assert not (second_docs / ".artifacts").exists()


def test_campaign_state_root_is_created_fresh_when_absent(tmp_path: Path) -> None:
    docs = tmp_path / "campaign"
    docs.mkdir()
    assert not (docs / "campaign_artifacts").exists()

    paths = manual_campaign._campaign_report_paths(ROOT, docs)
    ReportService(paths)

    assert paths.artifact_root == docs / "campaign_artifacts"
    assert paths.coordination_root == docs / "campaign_artifacts/coordination"
    assert (paths.artifact_root / "cells").is_dir()
    assert paths.coordination_root.is_dir()
    assert not (docs / ".artifacts").exists()


@pytest.mark.parametrize("member", ("campaign_artifacts", "coordination"))
def test_campaign_state_rejects_symlinked_private_directories(
    tmp_path: Path,
    member: str,
) -> None:
    docs = tmp_path / "campaign"
    docs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if member == "campaign_artifacts":
        (docs / member).symlink_to(outside, target_is_directory=True)
    else:
        state = docs / "campaign_artifacts"
        state.mkdir()
        (state / member).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManualCampaignError, match="must not be a symbolic link"):
        manual_campaign._campaign_report_paths(ROOT, docs)


def test_campaign_state_rejects_special_private_member(tmp_path: Path) -> None:
    docs = tmp_path / "campaign"
    docs.mkdir()
    (docs / "campaign_artifacts").write_text("not a directory\n", encoding="ascii")

    with pytest.raises(ManualCampaignError, match="must be a directory"):
        manual_campaign._campaign_report_paths(ROOT, docs)


def test_campaign_state_rejects_symlinked_campaign_directory(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "campaign"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ManualCampaignError, match="must not be a symbolic link"):
        manual_campaign._campaign_report_paths(ROOT, linked)


def test_campaign_command_holds_shared_destination_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    docs = tmp_path / "campaign"
    docs.mkdir()
    original_fixture = manual_campaign._snapshot_fixture

    def assert_locked() -> DashboardState:
        descriptor = os.open(
            docs,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        return original_fixture()

    monkeypatch.setattr(manual_campaign, "_snapshot_fixture", assert_locked)

    assert (
        campaign_main(
            ("dashboard-snapshot", "--width", "80", "--height", "24"),
            repo_root=ROOT,
            docs_dir=docs,
        )
        == 0
    )


def test_catalog_and_fresh_profile_are_complete_but_measurement_empty() -> None:
    assert len(REPORT_CATALOG.measurement_cells()) == 1796
    assert PROFILE.is_dir()
    assert not (PROFILE / "pyAmpliCol.pdf").exists()
    assert not any(PROFILE.rglob("current.json"))
    assert not any(PROFILE.rglob("manifest.json"))
    assert not any(PROFILE.rglob("*.sha256"))
    old_profile = ROOT / "docs/performance_reports/macbook_M3"
    assert old_profile.resolve() != PROFILE.resolve()


def test_selector_repetition_wildcard_aliases_and_intersection() -> None:
    arguments = _parse(
        "run",
        "--dry-run",
        "--multiplicity",
        "3",
        "--multiplicity",
        "4",
        "--model",
        "built-in",
        "sm_ufo",
        "--color-approximation",
        "LC",
        "--generation-mode",
        "union-flow",
        "--generation-engine",
        "recurrence",
    )
    _selection, cells = selection_from_arguments(arguments)
    assert cells
    assert {cell.n_final for cell in cells} == {3, 4}
    assert {cell.measurement.model for cell in cells} == {
        ModelKey.BUILTIN_SM,
        ModelKey.UFO_SM,
    }
    assert {cell.workload for cell in cells} == {Workload.ALL_FLOW}
    assert {cell.measurement.execution_mode for cell in cells} == {
        ExecutionMode.RECURRENCE
    }

    all_arguments = _parse("inspect", "--table", "*", "--model", "all")
    _all_selection, all_cells = selection_from_arguments(all_arguments)
    assert len(all_cells) == 1796


def test_matrix_best_is_all_three_builtin_candidate_modes() -> None:
    arguments = _parse("inspect", "--table", "matrix_best")
    _selection, cells = selection_from_arguments(arguments)
    assert len(cells) == 942
    assert {cell.measurement.execution_mode for cell in cells} == {
        ExecutionMode.RECURRENCE,
        ExecutionMode.COMPILED,
        ExecutionMode.EAGER,
    }
    assert {cell.measurement.model for cell in cells} == {ModelKey.BUILTIN_SM}


def test_process_id_15_selector_targets_only_identical_quark_line_cells() -> None:
    arguments = _parse(
        "inspect",
        "--process-id",
        "15",
        "--multiplicity",
        "4",
        "--generation-engine",
        "recurrence",
        "--model",
        "builtin_sm",
    )
    _selection, cells = selection_from_arguments(arguments)

    assert len(cells) == 4
    assert {cell.process_key for cell in cells} == {"dd_3q_identical_lines"}
    assert {cell.process for cell in cells} == {"d d~ > u u~ u u~"}
    assert {cell.n_final for cell in cells} == {4}


def test_builtin_model_and_table_aliases_handle_amplicol_references() -> None:
    built_in = _parse(
        "inspect",
        "--model",
        "built_in",
        "--generation-engine",
        "amplicol",
    )
    _selection, cells = selection_from_arguments(built_in)
    assert cells
    assert all(
        cell.measurement.execution_mode is ExecutionMode.AMPLICOL for cell in cells
    )

    z_reference = _parse(
        "inspect",
        "--table",
        "z_table",
        "--generation-engine",
        "amplicol",
    )
    _selection, cells = selection_from_arguments(z_reference)
    assert cells
    assert {cell.process_key for cell in cells} == {"dd_z_jets"}
    assert {cell.workload for cell in cells} == {Workload.SELECTED_FLOW}


def test_valid_but_empty_selector_has_concise_error() -> None:
    arguments = _parse(
        "inspect",
        "--color-approximation",
        "nlc",
        "--generation-mode",
        "union-flow",
    )
    with pytest.raises(ManualCampaignError, match="selector intersection"):
        selection_from_arguments(arguments)


def test_invalid_selector_lists_allowed_values_and_suggestion() -> None:
    arguments = _parse("inspect", "--generation-engine", "recurrance")
    with pytest.raises(ManualCampaignError, match="did you mean recurrence"):
        selection_from_arguments(arguments)


def test_help_is_exhaustive_and_run_defaults_match_contract() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for fragment in (
        "Steering model",
        "pyamplicol generate",
        "pyamplicol profile",
        "dashboard-snapshot",
        "refresh-pdf",
        "matrix_recurrence_ufo_sm_lc",
        "Inspect coverage",
        "cannot fit the 1-GiB envelope",
        "C++/ASM variants above",
        "Keyboard controls",
        "current/contribution counts",
        "process-tree current/peak usage",
        "--continue-across-revisions",
        "--cell-id-file",
        "--cleanup-artifacts",
        "`unverified.txt` needs no `--force-refresh`",
        "every heavy attempt payload is retained",
        "sealed with compact diagnostics",
        "without an original-AmpliCol checkout",
        "--generation-engine recurrence compiled eager",
        "report shows its absolute",
        "quoted `*`",
        "broad/default selection includes AmpliCol",
    ):
        assert fragment in help_text
    run_help = parser._subparsers._group_actions[0].choices["run"].format_help()
    assert "recurrence, compiled, and eager selections run without" in run_help
    assert "report absolute timings" in run_help
    assert "Omitted or '*' engine" in run_help
    assert "selection means all engines" in run_help
    assert "broad/default selection" in run_help
    assert "--no-artifacts-removal" not in help_text
    arguments = _parse("run", "--dry-run")
    assert arguments.workers == 1
    assert arguments.cores_per_worker == 1
    assert arguments.generation_time_limit == 3600.0
    assert arguments.ram_limit == 30_000_000_000
    assert arguments.worker_wall_limit == 3600.0
    assert arguments.no_color is False
    assert arguments.force_refresh is False
    assert arguments.continue_across_revisions is False
    assert arguments.cleanup_artifacts is False
    assert _parse("run", "--cleanup-artifacts").cleanup_artifacts is True
    assert _parse("run", "--continue-across-revisions").continue_across_revisions
    refresh = _parse("refresh-pdf")
    assert refresh.expected_page_count is None
    assert refresh.quiet is False
    assert _parse("refresh-pdf", "--quiet").quiet is True
    assert DEFAULT_MANUAL_EXPECTED_PAGE_COUNT == 59
    underscore = _parse("inspect", "--color_approximation", "lc")
    _selection, cells = selection_from_arguments(underscore)
    assert cells
    assert {cell.measurement.accuracy.value for cell in cells} == {"lc"}


def test_colours_are_default_and_only_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    arguments = _parse("inspect")
    palette = manual_campaign.Palette(
        manual_campaign._color_enabled(arguments),
    )
    assert "\x1b[" in palette.key("profile")
    assert "\x1b[" in palette.success("ok")

    no_colour = _parse("inspect", "--no-color")
    assert not manual_campaign._color_enabled(no_colour)
    assert manual_campaign.Palette(False).key("profile") == "profile"

    monkeypatch.setenv("NO_COLOR", "")
    assert not manual_campaign._color_enabled(arguments)
    monkeypatch.delenv("NO_COLOR")
    assert not manual_campaign._color_enabled(arguments, json_output=True)


def test_reproduction_recipe_uses_public_generate_and_profile() -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.workload is Workload.ALL_FLOW
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    recipe = reproduction_recipe(
        candidate,
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
    )
    assert recipe.kind == "public-cli-template+model-compile-prerequisite"
    assert recipe.prepare is not None
    assert recipe.prepare[:5] == (
        "env",
        f"PYTHONPATH={os.fspath((ROOT / 'src').resolve())}",
        os.fspath((ROOT / ".venv/bin/pyamplicol").resolve()),
        "model",
        "compile",
    )
    assert "built-in-sm" in recipe.prepare
    assert recipe.generate is not None
    assert recipe.generate[:4] == (
        "env",
        f"PYTHONPATH={os.fspath((ROOT / 'src').resolve())}",
        os.fspath((ROOT / ".venv/bin/pyamplicol").resolve()),
        "generate",
    )
    assert "--lc-flow-layout" in recipe.generate
    assert "all-flow-union" in recipe.generate
    assert recipe.profile is not None
    assert recipe.profile[:4] == (
        "env",
        f"PYTHONPATH={os.fspath((ROOT / 'src').resolve())}",
        os.fspath((ROOT / ".venv/bin/pyamplicol").resolve()),
        "profile",
    )
    assert recipe.exact is False
    assert "<" not in " ".join(recipe.profile)

    legacy = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
    )
    legacy_recipe = reproduction_recipe(
        legacy,
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
    )
    assert legacy_recipe.kind == "legacy-report-adapter"
    assert legacy_recipe.generate is None
    assert legacy_recipe.profile is None
    assert legacy_recipe.exact is False


def test_reproduction_recipe_exposes_authenticated_reuse_off_fallback() -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
    )
    fallback = {
        "abi": "pyamplicol-numerical-current-reuse-fallback-v1",
        "requested_mode": "certified-reuse",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "reason": "evidence-envelope-fallback",
        "geometry": {
            "current_count": 4,
            "component_count": 8,
            "candidate_probe_count": 2,
            "verification_probe_count": 2,
            "runtime_parameter_count": 3,
            "scalar_count": 76,
            "row_count": 11,
        },
        "certified_relation_count": 0,
        "applied_relation_count": 0,
    }

    recipe = reproduction_recipe(
        candidate,
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
        measurement={"provenance": {"numerical_relation_fallback": fallback}},
    )

    assert recipe.generate is not None
    assert "--no-numerical-current-reuse" in recipe.generate
    assert "--numerical-current-reuse" not in recipe.generate
    assert "effective-reuse-off-fallback" in recipe.kind
    assert "evidence-envelope fallback" in recipe.note
    assert "effectively disabled" in recipe.note

    malformed = json.loads(json.dumps(fallback))
    malformed["geometry"]["scalar_count"] = 77
    with pytest.raises(ManualCampaignError, match="fallback geometry is inconsistent"):
        reproduction_recipe(
            candidate,
            repo_root=ROOT,
            artifact_root=CAMPAIGN_ARTIFACT_ROOT,
            measurement={"provenance": {"numerical_relation_fallback": malformed}},
        )


def test_reproduction_cli_prefix_runs_from_an_unrelated_directory(
    tmp_path: Path,
) -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    recipe = reproduction_recipe(
        candidate,
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
    )
    assert recipe.generate is not None
    command_prefix = recipe.generate[:3]
    completed = subprocess.run(
        (*command_prefix, "generate", "--help"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_completed_reproduction_recipe_uses_exact_selectors_and_momenta(
    tmp_path: Path,
) -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.workload is Workload.ALL_FLOW
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    recorded_momenta = [[[101.0, 1.0, 2.0, 3.0]]]
    measurement = {
        "status": "ok",
        "artifact": {"path": "/tmp/example-artifact", "process_id": "stable-process"},
        "selector_contract": {
            "selected_color_flow_ids": ["cf:1"],
            "selected_color_words": [[1, 2]],
            "all_flow_helicity_ids": ["h:+1,-1"],
            "all_flow_source_helicities": {"1": 1, "2": -1},
            "point_digest": "1" * 64,
        },
        "provenance": {
            "requested_config": {"model": {"source": "/tmp/prepared.pyamplicol-model"}},
            "report_momenta": recorded_momenta,
        },
    }
    recipe = reproduction_recipe(
        candidate,
        repo_root=tmp_path,
        artifact_root=tmp_path / "campaign_artifacts",
        measurement=measurement,
    )
    assert recipe.exact is True
    assert recipe.prepare is not None
    assert "built-in-sm" in recipe.prepare
    assert recipe.generate is not None
    assert recipe.generate[recipe.generate.index("--model") + 1].endswith(
        "/prepared-models/built-in-sm-jit-o2.pyamplicol-model"
    )
    assert recipe.profile is not None
    profile_index = recipe.profile.index("profile")
    assert recipe.profile[profile_index + 1].endswith(f"/{candidate.cell_id}/artifact")
    assert candidate.process in recipe.profile
    assert "--helicity" in recipe.profile
    assert "h:+1,-1" in recipe.profile
    assert "--momenta" in recipe.profile
    momenta_path = Path(recipe.profile[recipe.profile.index("--momenta") + 1])
    assert json.loads(momenta_path.read_text(encoding="ascii")) == recorded_momenta
    assert momenta_path.is_relative_to(tmp_path / "campaign_artifacts")
    assert all(
        ".artifacts" not in argument
        for argument in (*recipe.prepare, *recipe.generate, *recipe.profile)
    )
    assert "<" not in " ".join((*recipe.generate, *recipe.profile))


def test_ufo_recurrence_recipe_has_public_model_compile_prerequisite(
    tmp_path: Path,
) -> None:
    from pyamplicol.cli import parse_cli

    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.UFO_SM
    )
    recipe = reproduction_recipe(
        candidate,
        repo_root=tmp_path,
        artifact_root=tmp_path / "campaign_artifacts",
    )

    assert recipe.prepare is not None
    assert recipe.prepare[:5] == (
        sys.executable,
        "-m",
        "pyamplicol",
        "model",
        "compile",
    )
    prepare_index = recipe.prepare.index("model")
    assert (
        parse_cli(recipe.prepare[prepare_index:]).resolve().effective.action
        == "model-compile"
    )
    assert recipe.generate is not None
    prepared_source = recipe.generate[recipe.generate.index("--model") + 1]
    assert prepared_source.endswith(".pyamplicol-model")
    generate_index = recipe.generate.index("generate")
    assert (
        parse_cli(recipe.generate[generate_index:]).resolve().effective.action
        == "generate"
    )
    assert "model-compile-prerequisite" in recipe.kind


def test_compiled_recipe_labels_both_private_timing_exceptions() -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.COMPILED
    )
    recipe = reproduction_recipe(
        candidate,
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
    )

    assert recipe.exact is False
    assert "precompiled-generation" in recipe.kind
    assert "paired-arena" in recipe.kind
    assert "outside generation_seconds" in recipe.note
    assert "do not reproduce either timing boundary" in recipe.note


def test_dry_run_is_compact_and_labels_each_recipe(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    cell_id = "matrix-compiled-builtin-sm-lc-n1-dd-z-jets-selected-flow"

    assert (
        campaign_main(
            ("run", "--dry-run", "--cell-id", cell_id),
            repo_root=ROOT,
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "kind: public-cli+precompiled-generation+paired-arena-exceptions" in output
    assert "exact: no" in output
    assert "Diagnostic only:" in output
    assert "Generate:" in output
    assert "Profile template/diagnostic:" in output
    assert max(map(len, output.splitlines())) < 300


def test_statistics_use_ratio_of_sums_weighting() -> None:
    candidates = [
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
    ][:2]
    baselines = [REPORT_CATALOG.validation_baseline_cell(cell) for cell in candidates]
    assert all(baseline is not None for baseline in baselines)
    comparisons = (
        Comparison(candidates[0], baselines[0], 2.0, 1.0),  # type: ignore[arg-type]
        Comparison(candidates[1], baselines[1], 9.0, 3.0),  # type: ignore[arg-type]
    )
    summary = comparison_statistics(comparisons)
    assert summary["median"] == 2.5
    assert summary["mean"] == 2.5
    assert summary["weighted_mean"] == 11.0 / 4.0
    assert summary["best"] is comparisons[0]
    assert summary["worst"] is comparisons[1]


@pytest.mark.parametrize(("width", "height"), ((80, 24), (120, 36), (160, 48)))
def test_ratatui_headless_frames_are_stable_and_informative(
    width: int,
    height: int,
) -> None:
    frame = render_dashboard_frame(
        _snapshot_fixture(),
        width=width,
        height=height,
    )
    assert isinstance(frame, str)
    assert "pyAmpliCol profiling campaign" in frame
    assert "Selected" in frame
    assert "Recycled" in frame
    assert "Remaining" in frame
    assert "Ready 0" in frame
    assert "Waiting dependency 0" in frame
    assert "Waiting coordination lock 0" in frame
    assert "RAM" in frame
    assert "Caps Generation 01:00:00" in frame
    assert "RAM 30.00 GB" in frame
    assert "Total wall 01:00:00" in frame
    assert "Ctrl-C" in frame
    assert len(frame.splitlines()) == height


def test_ratatui_styled_cells_include_color_and_gauge_values() -> None:
    from ratatui import Color

    state = _snapshot_fixture()
    width = 120
    frame = render_dashboard_frame(state, width=width, height=36)
    cells = render_dashboard_frame(state, width=width, height=36, cells=True)
    assert isinstance(cells, list)
    assert cells
    assert any(item.get("fg") not in (None, 0) for item in cells)
    assert any(item.get("ch") == ord("8") for item in cells)
    header_y, header = next(
        (index, line)
        for index, line in enumerate(frame.splitlines())
        if "cell" in line and "phase / step" in line
    )
    header_cell = cells[header_y * width + header.index("cell")]
    assert header_cell["fg"] == int(Color.LightCyan)
    assert int(header_cell["mods"]) & 1


def test_dashboard_detail_columns_and_supervision_summary_are_styled() -> None:
    from ratatui import Color

    worker = WorkerView(
        "completed-demo",
        status="ok",
        phase="complete",
        step="published",
        cpu_seconds=1.0,
        current_rss_bytes=2_000,
        peak_rss_bytes=3_000,
        phase_timeline=(
            manual_campaign.PhaseTimelineRow(
                "Loading and compiling evaluator",
                wall_seconds=1.25,
                status="measured",
            ),
            manual_campaign.PhaseTimelineRow(
                "Worker supervision",
                wall_seconds=2.0,
                cpu_seconds=1.0,
                peak_memory_bytes=3_000,
                status="observed",
                detail="overall observed worker wall",
            ),
        ),
    )
    state = DashboardState(
        instance_id="aligned-details",
        selected_ids=(worker.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        workers={worker.cell_id: worker},
        terminal_outcomes={
            worker.cell_id: manual_campaign._DashboardTerminalOutcome.SUCCESS
        },
        show_completed=True,
    )
    width = 160
    frame = render_dashboard_frame(state, width=width, height=48)
    lines = frame.splitlines()
    second_keys = {}
    for first_key, key in (
        ("Phase", "Step"),
        ("Wall", "CPU"),
        ("RSS current", "RSS peak"),
        ("Guard current", "Guard peak"),
        ("Outer wall", "Evaluator total"),
        ("PID tree", "Attempt"),
    ):
        line = next(value for value in lines if first_key in value and key in value)
        second_keys[key] = line.index(key)
    assert len(set(second_keys.values())) == 1
    assert "Loading and compiling evaluator" in frame

    cells = render_dashboard_frame(state, width=width, height=48, cells=True)
    summary_y, summary_line = next(
        (index, line)
        for index, line in enumerate(lines)
        if "Worker supervision" in line
    )
    summary_cell = cells[summary_y * width + summary_line.index("Worker supervision")]
    assert summary_cell["fg"] == int(Color.LightCyan)
    assert summary_cell["bg"] == int(Color.DarkGray)
    assert int(summary_cell["mods"]) & 1


def test_dashboard_retains_and_renders_native_progress_counters(
    tmp_path: Path,
) -> None:
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        json.dumps(
            {
                "event": "update",
                "task_id": "generation:recurrence:process:rust-builder",
                "completed": 8,
                "total": 9,
                "message": "recurrence stage",
                "details": {
                    "process": "dd_z_7g",
                    "step": "recurrence stage",
                    "stage_index": 8,
                    "stage_total": 9,
                    "current_count": 38_581,
                    "contribution_count": 286_294,
                    "certified_relation_count": 0,
                    "applied_relation_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    worker = WorkerView(
        "matrix-recurrence-builtin-sm-lc-n8-dd-z-jets-all-flow",
        status="running",
        phase="generation",
        progress_path=str(progress_path),
    )

    _tail_progress(worker)

    assert worker.progress_task_id == ("generation:recurrence:process:rust-builder")
    assert worker.progress_completed == 8
    assert worker.progress_total == 9
    assert worker.progress_details["current_count"] == 38_581
    assert worker.progress_details["contribution_count"] == 286_294
    summary = _progress_detail_summary(worker.progress_details)
    assert "currents 38,581" in summary
    assert "contributions 286,294" in summary
    assert "relations certified 0" in summary
    state = DashboardState(
        instance_id="progress-details",
        selected_ids=(worker.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="6" * 40,
        workers={worker.cell_id: worker},
    )
    frame = render_dashboard_frame(state, width=160, height=48)
    assert "Progress data" in frame
    assert "currents 38,581" in frame
    assert "contributions 286,294" in frame


def test_dashboard_keys_select_scroll_help_and_interrupt() -> None:
    from ratatui import KeyCode, KeyMods

    state = _snapshot_fixture()
    cancellation = threading.Event()
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Down, "ch": 0, "mods": 0},
        cancellation,
    )
    assert state.selected_index == 1
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.PageDown, "ch": 0, "mods": 0},
        cancellation,
    )
    assert state.detail_scroll == 3
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("?"), "mods": 0},
        cancellation,
    )
    assert state.show_help
    assert _handle_dashboard_key(
        state,
        {
            "kind": "key",
            "code": KeyCode.Char,
            "ch": ord("c"),
            "mods": int(KeyMods.CTRL),
        },
        cancellation,
    )
    assert cancellation.is_set()
    assert state.interrupted


def test_dashboard_command_drawer_scrolls_and_copies_exact_command() -> None:
    from ratatui import KeyCode

    command = " ".join(
        ["pyamplicol", "generate"]
        + [f"--option-{index} 'value {index}'" for index in range(30)]
    )
    worker = WorkerView(
        "command-demo",
        status="running",
        phase="generation",
        reproduce_prepare="pyamplicol model compile built-in-sm /tmp/model",
        reproduce_generate=command,
        reproduce_profile="pyamplicol profile /tmp/artifact --process 'd d~ > z'",
    )
    state = DashboardState(
        instance_id="command-drawer",
        selected_ids=(worker.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        workers={worker.cell_id: worker},
    )
    cancellation = threading.Event()

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("2"), "mods": 0},
        cancellation,
    )
    assert state.command_stage == "generate"
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Tab, "ch": 0, "mods": 0},
        cancellation,
    )
    assert state.command_stage == "profile"
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Left, "ch": 0, "mods": 0},
        cancellation,
    )
    assert state.command_stage == "generate"
    first = render_dashboard_frame(state, width=80, height=24)
    assert "Reproduction command · generate" in first
    assert "wrapped lines 1-" in first
    assert "--option-29" not in first

    for _ in range(4):
        assert not _handle_dashboard_key(
            state,
            {"kind": "key", "code": KeyCode.PageDown, "ch": 0, "mods": 0},
            cancellation,
        )
    last = render_dashboard_frame(state, width=80, height=24)
    assert "--option-29" in last

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("y"), "mods": 0},
        cancellation,
    )
    assert state.pending_clipboard == ("generate", command)
    sequence = manual_campaign._osc52_clipboard_sequence(command)
    encoded = sequence.removeprefix("\x1b]52;c;").removesuffix("\x07")
    assert base64.b64decode(encoded).decode("utf-8") == command
    assert not cancellation.is_set()

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("p"), "mods": 0},
        cancellation,
    )
    assert state.pending_print == ("generate", command)

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Esc, "ch": 0, "mods": 0},
        cancellation,
    )
    assert state.command_stage is None
    assert not cancellation.is_set()


def test_dashboard_filters_order_and_keyboard_toggles() -> None:
    from ratatui import KeyCode

    workers = {
        "done-ok": WorkerView("done-ok", status="ok"),
        "done-reused": WorkerView("done-reused", status="reused"),
        "attention-error": WorkerView("attention-error", status="error"),
        "recycled-cap": WorkerView("recycled-cap", status="memory_limit"),
        "active-queued": WorkerView("active-queued", status="queued"),
        "active-preparing": WorkerView("active-preparing", status="preparing"),
        "active-running": WorkerView("active-running", status="running"),
    }
    state = DashboardState(
        instance_id="filters",
        selected_ids=tuple(workers),
        recycled_ids={"done-reused", "recycled-cap"},
        static_na_ids=set(),
        workers=workers,
        terminal_outcomes={
            "done-ok": manual_campaign._DashboardTerminalOutcome.SUCCESS,
            "done-reused": manual_campaign._DashboardTerminalOutcome.SUCCESS,
            "recycled-cap": manual_campaign._DashboardTerminalOutcome.CAPPED,
            "attention-error": manual_campaign._DashboardTerminalOutcome.FAILED,
        },
    )
    cancellation = threading.Event()

    assert [worker.cell_id for worker in state.visible_workers()] == [
        "active-running",
        "active-preparing",
        "active-queued",
        "attention-error",
        "recycled-cap",
    ]

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("d"), "mods": 0},
        cancellation,
    )
    assert state.show_completed
    assert [worker.cell_id for worker in state.visible_workers()] == [
        "active-running",
        "active-preparing",
        "active-queued",
        "attention-error",
        "recycled-cap",
        "done-ok",
        "done-reused",
    ]

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("e"), "mods": 0},
        cancellation,
    )
    assert not state.show_errors
    assert [worker.cell_id for worker in state.visible_workers()] == [
        "active-running",
        "active-preparing",
        "active-queued",
        "done-ok",
        "done-reused",
    ]

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("d"), "mods": 0},
        cancellation,
    )
    assert not state.show_completed
    assert [worker.cell_id for worker in state.visible_workers()] == [
        "active-running",
        "active-preparing",
        "active-queued",
    ]

    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("e"), "mods": 0},
        cancellation,
    )
    assert state.show_errors
    assert [worker.cell_id for worker in state.visible_workers()] == [
        "active-running",
        "active-preparing",
        "active-queued",
        "attention-error",
        "recycled-cap",
    ]


def test_dashboard_error_total_is_in_primary_overview_summary() -> None:
    workers = {
        "active": WorkerView("active", status="running"),
        "error": WorkerView("error", status="error"),
        "unverified": WorkerView("unverified", status="unverified"),
        "recycled-cap": WorkerView("recycled-cap", status="memory_limit"),
    }
    state = DashboardState(
        instance_id="error-counter",
        selected_ids=tuple(workers),
        recycled_ids={"recycled-cap"},
        static_na_ids=set(),
        workers=workers,
        terminal_outcomes={
            "recycled-cap": manual_campaign._DashboardTerminalOutcome.CAPPED,
            "error": manual_campaign._DashboardTerminalOutcome.FAILED,
            "unverified": manual_campaign._DashboardTerminalOutcome.UNVERIFIED,
        },
    )

    frame = render_dashboard_frame(state, width=120, height=36)
    summary = next(line for line in frame.splitlines() if "Selected " in line)
    assert "Errors 1" in summary
    assert "Unverified 1" in summary
    assert "Active 1" in summary
    assert "Capped 1" in frame


def test_finished_unverified_is_not_counted_as_an_error(tmp_path: Path) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="unverified-counter",
        selected_ids=("diagnostic",),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )

    LeaseManager(service, state).observe(
        {
            "event": "finished",
            "cell_id": "diagnostic",
            "status": ResultStatus.UNVERIFIED.value,
            "detail": "independent authority unavailable",
        }
    )

    assert state.failed_ids == set()
    assert state.unverified_ids == {"diagnostic"}
    assert state.counters()["failed"] == 0
    assert state.counters()["unverified"] == 1
    assert state.counters()["remaining"] == 0


def test_recycled_resource_cap_is_available_to_error_filter(
    tmp_path: Path,
) -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    current = LightweightCurrent(
        cell_id=cell.cell_id,
        attempt_id="recycled-cap-attempt",
        result_path=tmp_path / "worker-result.json",
        result={
            "status": ResultStatus.TIMEOUT.value,
            "resources": {
                "wall_seconds": 3601.0,
                "peak_rss_bytes": 2_000_000_000,
            },
            "provenance": {
                "manual_campaign": {
                    "generation_limit_seconds": 3600.0,
                    "memory_limit_bytes": 30_000_000_000,
                }
            },
        },
        complete=True,
        reusable=True,
        reason="resource-capped terminal",
    )

    workers = manual_campaign._recycled_attention_workers(
        (cell,),
        {cell.cell_id: current},
        {cell.cell_id},
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
        settings=ReproductionSettings(),
    )

    assert set(workers) == {cell.cell_id}
    worker = workers[cell.cell_id]
    assert worker.status == "generation_limit"
    assert worker.attempt_id == "recycled-cap-attempt"
    assert worker.wall_seconds == 3601.0
    assert worker.peak_rss_bytes == 2_000_000_000
    assert worker.reproduce_generate is not None
    assert worker.recycled

    state = DashboardState(
        instance_id="recycled-cap",
        selected_ids=(cell.cell_id,),
        recycled_ids={cell.cell_id},
        static_na_ids=set(),
        workers=workers,
        show_completed=True,
    )
    assert [row.cell_id for row in state.visible_workers()] == [cell.cell_id]
    state.show_errors = False
    assert state.visible_workers() == ()


def test_recycled_legacy_manual_memory_cap_needs_no_new_censor_record(
    tmp_path: Path,
) -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    current = LightweightCurrent(
        cell_id=cell.cell_id,
        attempt_id="legacy-memory-cap",
        result_path=tmp_path / "worker-result.json",
        result={
            "status": ResultStatus.MEMORY_LIMIT.value,
            "provenance": {"manual_campaign": {"memory_limit_bytes": 30_000_000_000}},
        },
        complete=True,
        reusable=True,
        reason="resource-capped terminal",
    )

    worker = manual_campaign._recycled_attention_workers(
        (cell,),
        {cell.cell_id: current},
        {cell.cell_id},
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
        settings=ReproductionSettings(),
    )[cell.cell_id]

    assert worker.status == "memory_limit"
    assert worker.recycled


def test_successful_recycled_result_uses_done_filter_and_persisted_timeline(
    tmp_path: Path,
) -> None:
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    result = {
        "status": ResultStatus.OK.value,
        "provenance": {
            "report_source_revision": "a" * 40,
            "manual_campaign": {
                "cell_identity": {
                    "execution_mode": "recurrence",
                    "model": "builtin_sm",
                },
                "phase_timeline": {
                    "schema": "pyamplicol-manual-campaign-phase-timeline-v1",
                    "total_worker_wall_seconds": 12.5,
                    "peak_rss_bytes": 2_048,
                    "peak_guard_bytes": 4_096,
                    "entries": [
                        {
                            "key": "preparation",
                            "label": "Preparation",
                            "seconds": 0.125,
                            "status": "ok",
                            "detail": "model ready",
                        },
                        {
                            "key": "generation",
                            "label": "Generation",
                            "seconds": 2.5,
                            "status": "ok",
                        },
                        {
                            "key": "profiling",
                            "label": "Profiling",
                            "start_seconds": 3.0,
                            "end_seconds": 9.0,
                            "status": "ok",
                        },
                    ],
                },
            },
        },
    }
    current = LightweightCurrent(
        cell_id=cell.cell_id,
        attempt_id="recycled-ok-attempt",
        result_path=tmp_path / "worker-result.json",
        result=result,
        complete=True,
        reusable=True,
        reason="matching same-source current",
    )

    workers = manual_campaign._recycled_attention_workers(
        (cell,),
        {cell.cell_id: current},
        {cell.cell_id},
        repo_root=ROOT,
        artifact_root=CAMPAIGN_ARTIFACT_ROOT,
        settings=ReproductionSettings(),
    )
    worker = workers[cell.cell_id]
    assert worker.status == "recycled"
    assert worker.recycled
    assert worker.wall_seconds == 12.5
    assert worker.peak_rss_bytes == 2_048
    assert worker.peak_guard_bytes == 4_096
    assert [row.phase for row in worker.phase_timeline] == [
        "Preparation",
        "Generation",
        "Profiling",
    ]
    assert worker.phase_timeline[2].wall_seconds is None
    assert worker.provenance_summary == (
        "source aaaaaaaaaaaa · engine recurrence · model builtin_sm"
    )

    state = DashboardState(
        instance_id="recycled-ok",
        selected_ids=(cell.cell_id,),
        recycled_ids={cell.cell_id},
        static_na_ids=set(),
        workers=workers,
    )
    assert state.visible_workers() == ()
    state.show_completed = True
    assert [row.cell_id for row in state.visible_workers()] == [cell.cell_id]
    state.show_errors = False
    assert [row.cell_id for row in state.visible_workers()] == [cell.cell_id]

    frame = render_dashboard_frame(state, width=160, height=48)
    assert "recycled" in frame
    assert "No work executed by this invocation" in frame
    assert "Reuse matching same-source current" in frame
    assert "Phase timeline" in frame
    assert "Preparation" in frame
    assert "125.0 ms" in frame
    assert "2.50 s" in frame
    assert "Generation" in frame
    assert "Profiling" in frame
    assert "unavailable" in frame
    assert "source aaaaaaaaaaaa" in frame

    decoded = manual_campaign._worker_from_lease(
        worker.cell_id,
        worker.as_dict(),
        peer_instance=None,
    )
    assert decoded.status == "recycled"
    assert decoded.recycled
    assert decoded.phase_timeline == worker.phase_timeline


def test_completed_legacy_result_marks_missing_timeline_unavailable() -> None:
    worker = WorkerView(
        "legacy-completed",
        status="ok",
        phase="completed",
        step="measurement published",
    )
    state = DashboardState(
        instance_id="legacy-completed",
        selected_ids=(worker.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        workers={worker.cell_id: worker},
        show_completed=True,
    )

    frame = render_dashboard_frame(state, width=120, height=36)
    assert "Provenance unavailable" in frame
    assert "Phase timeline unavailable (older result)" in frame


def test_selected_worker_preserves_subsecond_wall_and_cpu_durations() -> None:
    worker = WorkerView(
        "short-worker",
        status="running",
        phase="profiling",
        step="first sample",
        wall_seconds=0.125,
        cpu_seconds=0.004,
    )
    state = DashboardState(
        instance_id="short-worker",
        selected_ids=(worker.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        workers={worker.cell_id: worker},
    )

    frame = render_dashboard_frame(state, width=120, height=36)

    assert "Wall              125.0 ms" in frame
    assert "CPU               4.0 ms" in frame
    assert "Wall              00:00:00" not in frame


def test_phase_timeline_uses_only_explicit_legacy_durations() -> None:
    rows = manual_campaign._extract_phase_timeline(
        {
            "generation_seconds": 2.25,
            "provenance": {
                "runtime_profile": {
                    "measurement_phase_elapsed_seconds": 1.5,
                    "started_seconds": 3.0,
                    "finished_seconds": 99.0,
                },
            },
        }
    )

    assert [(row.phase, row.wall_seconds) for row in rows] == [
        ("Generation", 2.25),
        ("Timed headline measurement", 1.5),
    ]
    assert (
        manual_campaign._extract_phase_timeline(
            {
                "timeline": [{"label": "native core", "duration": 99.0}],
                "provenance": {
                    "manual_campaign": {
                        "phase_timeline": {
                            "schema": "unknown-timeline-v0",
                            "entries": [
                                {"label": "untrusted", "seconds": 123.0},
                            ],
                        }
                    }
                },
            }
        )
        == ()
    )


@pytest.mark.parametrize(
    ("width", "height", "expected_range"),
    ((80, 24, "14-18/18"), (120, 36, "10-18/18")),
)
def test_dashboard_worker_viewport_pans_with_selection(
    width: int,
    height: int,
    expected_range: str,
) -> None:
    from ratatui import KeyCode

    workers = {
        f"worker-{index:02d}": WorkerView(
            f"worker-{index:02d}",
            status="running",
            step=f"profiling sample {index}",
        )
        for index in range(18)
    }
    state = DashboardState(
        instance_id="viewport",
        selected_ids=tuple(workers),
        recycled_ids=set(),
        static_na_ids=set(),
        workers=workers,
    )
    cancellation = threading.Event()
    for _ in range(17):
        assert not _handle_dashboard_key(
            state,
            {"kind": "key", "code": KeyCode.Down, "ch": 0, "mods": 0},
            cancellation,
        )

    frame = render_dashboard_frame(state, width=width, height=height)
    selected_rows = [line for line in frame.splitlines() if "▶" in line]
    assert len(selected_rows) == 1
    assert "worker-17" in selected_rows[0]
    assert "worker-00" not in frame
    assert expected_range in frame
    assert "Cell worker-17" in frame
    assert "profiling sample 17" in frame


def test_dashboard_key_aliases_clamp_scroll_ignore_noise_and_escape() -> None:
    from ratatui import KeyCode

    state = _snapshot_fixture()
    cancellation = threading.Event()
    assert not _handle_dashboard_key(state, {"kind": "mouse"}, cancellation)
    assert state.selected_index == 0

    state.detail_scroll = 2
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.PageUp, "ch": 0, "mods": 0},
        cancellation,
    )
    assert state.detail_scroll == 0
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("k"), "mods": 0},
        cancellation,
    )
    assert state.selected_index == -1
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("j"), "mods": 0},
        cancellation,
    )
    assert state.selected_index == 0
    assert not _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Char, "ch": ord("c"), "mods": 0},
        cancellation,
    )
    assert not cancellation.is_set()

    assert _handle_dashboard_key(
        state,
        {"kind": "key", "code": KeyCode.Esc, "ch": 0, "mods": 0},
        cancellation,
    )
    assert cancellation.is_set()
    assert state.interrupted


def _manual_service(
    tmp_path: Path,
    *,
    initialize_source_marker: bool = True,
) -> ReportService:
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    if initialize_source_marker:
        manual_campaign.update_source_marker(
            service,
            manual_campaign.ReportSourceIdentity("a" * 40, "b" * 40, ()),
        )
    return service


def _presentation_test_cell() -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.workload is Workload.CONTRACTED
        and REPORT_CATALOG.static_na_reason(cell) is None
    )


def _presentation_outcome(
    cell_id: str,
    *,
    status: str,
    completed_at_ns: int = 1,
    invocation_id: str = "presentation-test",
    source_revision: str = "a" * 40,
    profile: str = manual_campaign._PRESENTATION_PROFILE,
    attempt_id: str | None = None,
) -> manual_campaign.LightweightPresentationOutcome:
    return manual_campaign.LightweightPresentationOutcome(
        profile=profile,
        cell_id=cell_id,
        source_revision=source_revision,
        campaign_invocation_id=invocation_id,
        attempt_id=attempt_id,
        status=status,
        label=manual_campaign._humanized_outcome_label(status),
        completed_at_ns=completed_at_ns,
    )


def test_presentation_outcome_survives_campaign_parent_and_basename_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = manual_campaign.ReportSourceIdentity("a" * 40, "b" * 40, ())
    campaign_a = tmp_path / "parent-a/original-name"
    campaign_a.mkdir(parents=True)
    service_a = ReportService(
        manual_campaign._campaign_report_paths(ROOT, campaign_a),
        portable_current_results=True,
    )
    manual_campaign.update_source_marker(service_a, source)
    cell = _presentation_test_cell()
    monkeypatch.setattr(manual_campaign, "_ACTIVE_PROFILE", "original-name")
    expected = _presentation_outcome(cell.cell_id, status="error")
    assert manual_campaign._publish_presentation_outcome(service_a, expected)

    campaign_b = tmp_path / "parent-b/renamed-campaign"
    campaign_b.parent.mkdir()
    campaign_a.rename(campaign_b)
    service_b = ReportService(
        manual_campaign._campaign_report_paths(ROOT, campaign_b),
        portable_current_results=True,
    )
    monkeypatch.setattr(manual_campaign, "_ACTIVE_PROFILE", "renamed-campaign")

    moved = manual_campaign.lightweight_presentation_outcome(
        service_b,
        cell,
        source_revision=source.revision,
    )
    assert moved == expected


def _valid_presentation_current(
    cell_id: str,
    tmp_path: Path,
) -> LightweightCurrent:
    measurement = empty_measurement()
    measurement.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-9,
            "relative_standard_error": 1.0e-3,
            "artifact": {"digest": "presentation-test"},
            "validation": {"status": "ok", "direct_agreements": []},
            "resources": {"peak_rss_bytes": 1},
            "provenance": {"method": "original-amplicol-generated-library"},
            "failure": None,
        }
    )
    return LightweightCurrent(
        cell_id=cell_id,
        attempt_id="00000000-0000-4000-8000-000000000001",
        result_path=tmp_path / "result.json",
        result=measurement,
        complete=True,
        reusable=True,
        reason="reusable",
    )


def _capped_presentation_current(
    cell_id: str,
    tmp_path: Path,
) -> LightweightCurrent:
    measurement = manual_campaign.failure_measurement(
        ResultStatus.TIMEOUT,
        "generation exceeded the configured limit",
    )
    measurement["provenance"] = {
        "manual_campaign": {"generation_limit_seconds": 3600.0}
    }
    return LightweightCurrent(
        cell_id=cell_id,
        attempt_id="00000000-0000-4000-8000-000000000002",
        result_path=tmp_path / "capped-result.json",
        result=measurement,
        complete=True,
        reusable=True,
        reason="reusable terminal cap",
    )


def _cache_measurement(
    caches: dict[str, dict[str, object]],
    *,
    dataset_id: str,
    cell_id: str,
) -> dict[str, object]:
    entries = caches[f"{dataset_id}.json"]["entries"]
    assert isinstance(entries, list)
    entry = next(item for item in entries if item["cell_id"] == cell_id)
    measurement = entry["measurement"]
    assert isinstance(measurement, dict)
    return measurement


def test_presentation_failure_overlay_never_replaces_a_valid_current(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    failure = _presentation_outcome(cell.cell_id, status="validation_failed")

    failure_caches, merged = _merge_lightweight_snapshot(
        service,
        {},
        {cell.cell_id: failure},
    )
    assert merged == 1
    failure_measurement = _cache_measurement(
        failure_caches,
        dataset_id=cell.dataset_id,
        cell_id=cell.cell_id,
    )
    assert failure_measurement["status"] == ResultStatus.VALIDATION_FAILED.value
    assert failure_measurement["failure"] == {
        "kind": "ManualCampaignOutcome:validation_failed",
        "message": "validation failed",
    }

    current = _valid_presentation_current(cell.cell_id, tmp_path)
    success_caches, merged = _merge_lightweight_snapshot(
        service,
        {cell.cell_id: current},
        {
            cell.cell_id: _presentation_outcome(
                cell.cell_id,
                status="error",
                completed_at_ns=2,
            )
        },
    )
    assert merged == 1
    assert (
        _cache_measurement(
            success_caches,
            dataset_id=cell.dataset_id,
            cell_id=cell.cell_id,
        )["status"]
        == ResultStatus.OK.value
    )


def test_presentation_success_tombstone_suppresses_failure_fallback(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    manual_campaign._publish_presentation_outcome(
        service,
        _presentation_outcome(
            cell.cell_id,
            status="error",
            completed_at_ns=1,
        ),
    )
    tombstone = _presentation_outcome(
        cell.cell_id,
        status="ok",
        completed_at_ns=2,
    )
    manual_campaign._publish_presentation_outcome(service, tombstone)
    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    assert observed == tombstone

    caches, merged = _merge_lightweight_snapshot(
        service,
        {},
        {cell.cell_id: observed},
    )

    assert merged == 0
    assert _cache_measurement(
        caches,
        dataset_id=cell.dataset_id,
        cell_id=cell.cell_id,
    ) == empty_measurement()


def test_presentation_outcome_publication_has_deterministic_last_writer(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    later_tie_breaker = _presentation_outcome(
        cell.cell_id,
        status="error",
        completed_at_ns=10,
        invocation_id="invocation-b",
    )
    earlier_tie_breaker = _presentation_outcome(
        cell.cell_id,
        status="validation_failed",
        completed_at_ns=10,
        invocation_id="invocation-a",
    )
    oldest = _presentation_outcome(
        cell.cell_id,
        status="blocked_dependency",
        completed_at_ns=9,
        invocation_id="invocation-z",
    )

    manual_campaign._publish_presentation_outcome(service, later_tie_breaker)
    manual_campaign._publish_presentation_outcome(service, oldest)
    manual_campaign._publish_presentation_outcome(service, earlier_tie_breaker)

    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    assert observed == later_tie_breaker


def test_strict_source_marker_rejects_late_old_revision_writer(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    old = _presentation_outcome(
        cell.cell_id,
        status="validation_failed",
        completed_at_ns=300,
        source_revision="a" * 40,
    )
    manual_campaign._publish_presentation_outcome(service, old)
    manual_campaign.update_source_marker(
        service,
        manual_campaign.ReportSourceIdentity("b" * 40, "c" * 40, ()),
    )
    active = _presentation_outcome(
        cell.cell_id,
        status="error",
        completed_at_ns=200,
        source_revision="b" * 40,
    )

    manual_campaign._publish_presentation_outcome(service, active)
    manual_campaign._publish_presentation_outcome(
        service,
        _presentation_outcome(
            cell.cell_id,
            status="validation_failed",
            completed_at_ns=400,
            source_revision="a" * 40,
        ),
    )

    assert manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="b" * 40,
    ) == active


def test_cross_revision_marker_retains_global_outcome_ordering(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    manual_campaign.update_source_marker(
        service,
        manual_campaign.ReportSourceIdentity("b" * 40, "c" * 40, ()),
        continue_across_revisions=True,
    )
    active = _presentation_outcome(
        cell.cell_id,
        status="error",
        completed_at_ns=200,
        source_revision="b" * 40,
    )
    historical = _presentation_outcome(
        cell.cell_id,
        status="validation_failed",
        completed_at_ns=300,
        source_revision="a" * 40,
    )

    manual_campaign._publish_presentation_outcome(service, active)
    manual_campaign._publish_presentation_outcome(service, historical)

    assert manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="b" * 40,
        accept_historical_source=True,
    ) == historical


def test_presentation_publication_requires_a_readable_source_marker(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path, initialize_source_marker=False)
    cell = _presentation_test_cell()
    outcome = _presentation_outcome(cell.cell_id, status="error")

    with pytest.raises(ManualCampaignError, match="source marker is unreadable"):
        manual_campaign._publish_presentation_outcome(service, outcome)

    marker = service.paths.coordination_root / "manual-source.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{broken", encoding="utf-8")
    with pytest.raises(ManualCampaignError, match="source marker is unreadable"):
        manual_campaign._publish_presentation_outcome(service, outcome)


def test_presentation_outcome_reader_filters_source_profile_and_malformed_data(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    path = manual_campaign._presentation_outcome_path(service, cell.cell_id)
    path.parent.mkdir(parents=True)
    historical = _presentation_outcome(
        cell.cell_id,
        status="error",
        source_revision="a" * 40,
    )
    path.write_text(json.dumps(historical.as_dict()), encoding="utf-8")

    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="b" * 40,
        )
        is None
    )
    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="b" * 40,
            accept_historical_source=True,
        )
        == historical
    )

    wrong_profile = historical.as_dict()
    wrong_profile["profile"] = "some_other_profile"
    path.write_text(json.dumps(wrong_profile), encoding="utf-8")
    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        is None
    )

    path.write_text('{"schema":"truncated"', encoding="utf-8")
    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        is None
    )


def test_presentation_outcome_reader_rejects_timestamp_above_signed_64_bit(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    path = manual_campaign._presentation_outcome_path(service, cell.cell_id)
    path.parent.mkdir(parents=True)
    malformed = _presentation_outcome(cell.cell_id, status="error").as_dict()
    malformed["completed_at_ns"] = (1 << 63)
    path.write_text(json.dumps(malformed), encoding="utf-8")

    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("campaign_invocation_id", "campaign-λ"),
        ("label", "error 🚫"),
    ),
)
def test_presentation_outcome_parser_rejects_printable_non_ascii_metadata(
    field: str,
    value: str,
) -> None:
    cell = _presentation_test_cell()
    malformed = _presentation_outcome(cell.cell_id, status="error").as_dict()
    assert value.isprintable()
    assert not value.isascii()
    malformed[field] = value

    assert (
        manual_campaign._parse_presentation_outcome(
            malformed,
            expected_profile=manual_campaign._PRESENTATION_PROFILE,
            expected_cell_id=cell.cell_id,
            source_revision="a" * 40,
            accept_historical_source=False,
        )
        is None
    )


def test_started_event_drops_stale_identity_before_generic_failure_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _capped_presentation_current(cell.cell_id, tmp_path)
    worker = WorkerView(
        cell.cell_id,
        status="generation_limit",
        attempt_id=current.attempt_id,
        log_path="stale-worker.log",
        progress_path="stale-progress.jsonl",
        progress_completed=9,
        progress_total=10,
        progress_message="stale progress",
        progress_task_id="stale-task",
        progress_details={"stale": True},
        wall_seconds=12.0,
        cpu_seconds=8.0,
        current_rss_bytes=99,
        peak_rss_bytes=101,
        published_wall_seconds_per_point=0.5,
        phase_timeline=(manual_campaign.PhaseTimelineRow("stale"),),
        blocked_prerequisite_ids=("stale-prerequisite",),
        events=["stale event"],
        log_tail=["stale cap diagnostics"],
    )
    state = DashboardState(
        instance_id="new-invocation",
        selected_ids=(cell.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
        workers={cell.cell_id: worker},
    )
    lease = LeaseManager(service, state)
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_current",
        lambda *_args, **_kwargs: current,
    )

    lease.observe({"event": "started", "cell_id": cell.cell_id})

    fresh_worker = state.workers[cell.cell_id]
    assert fresh_worker is not worker
    worker = fresh_worker
    assert worker.attempt_id is None
    assert worker.log_path is None
    assert worker.progress_path is None
    assert worker.progress_completed is None
    assert worker.progress_total is None
    assert worker.progress_message == ""
    assert worker.progress_task_id is None
    assert worker.progress_details == {}
    assert worker.log_tail == []
    assert worker.wall_seconds == 0.0
    assert worker.cpu_seconds is None
    assert worker.current_rss_bytes == 0
    assert worker.peak_rss_bytes == 0
    assert worker.published_wall_seconds_per_point is None
    assert worker.phase_timeline == ()
    assert worker.blocked_prerequisite_ids == ()
    assert worker.events == ["started: dependency preparation"]

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": "error",
            "detail": "generic worker crash without attempt records",
            "completed_at_ns": 7,
        }
    )
    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    assert observed is not None
    assert observed.status == "error"
    assert observed.attempt_id is None

    caches, merged = _merge_lightweight_snapshot(
        service,
        {cell.cell_id: current},
        {cell.cell_id: observed},
    )
    assert merged == 1
    measurement = _cache_measurement(
        caches,
        dataset_id=cell.dataset_id,
        cell_id=cell.cell_id,
    )
    assert measurement["status"] == ResultStatus.ERROR.value
    assert measurement["failure"] == {
        "kind": "ManualCampaignOutcome:error",
        "message": "error",
    }


@pytest.mark.parametrize("recycle_status", ("reused", "skipped-current"))
def test_non_ok_recycle_does_not_erase_newer_error_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recycle_status: str,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _capped_presentation_current(cell.cell_id, tmp_path)
    error = _presentation_outcome(
        cell.cell_id, status="error", completed_at_ns=2
    )
    manual_campaign._publish_presentation_outcome(service, error)
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_current",
        lambda *_args, **_kwargs: current,
    )
    lease = LeaseManager(
        service,
        DashboardState(
            instance_id="cap-recycle",
            selected_ids=(cell.cell_id,),
            recycled_ids=set(),
            static_na_ids=set(),
            source_revision="a" * 40,
        ),
    )

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": recycle_status,
            "detail": "ordinary terminal-current recycle",
            "completed_at_ns": 3,
        }
    )

    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    assert observed == error


@pytest.mark.parametrize("existing_success", (False, True))
def test_front_end_ok_recycle_does_not_write_without_existing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_success: bool,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _valid_presentation_current(cell.cell_id, tmp_path)
    if existing_success:
        manual_campaign._publish_presentation_outcome(
            service,
            _presentation_outcome(cell.cell_id, status="reused"),
        )
    writes: list[manual_campaign.LightweightPresentationOutcome] = []
    publish = manual_campaign._publish_presentation_outcome

    def capture(
        target: ReportService,
        outcome: manual_campaign.LightweightPresentationOutcome,
        **kwargs: object,
    ) -> bool:
        written = publish(target, outcome, **kwargs)
        if written:
            writes.append(outcome)
        return written

    monkeypatch.setattr(manual_campaign, "_publish_presentation_outcome", capture)
    warnings = manual_campaign._publish_recycled_presentation_outcomes(
        service,
        {cell.cell_id: current},
        (cell.cell_id,),
        source_revision="a" * 40,
        campaign_invocation_id="passive-recycle",
    )

    assert warnings == ()
    assert writes == []
    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    if existing_success:
        assert observed is not None
        assert observed.status == "reused"
    else:
        assert observed is None


@pytest.mark.parametrize(
    ("status", "existing_status", "expected_status", "expected_writes"),
    (
        ("ok", None, None, 0),
        ("reused", "reused", "reused", 0),
        ("skipped-current", "error", "skipped-current", 1),
    ),
)
def test_lease_success_with_resolved_ok_only_suppresses_existing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    existing_status: str | None,
    expected_status: str | None,
    expected_writes: int,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _valid_presentation_current(cell.cell_id, tmp_path)
    if existing_status is not None:
        manual_campaign._publish_presentation_outcome(
            service,
            _presentation_outcome(cell.cell_id, status=existing_status),
        )
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_current",
        lambda *_args, **_kwargs: current,
    )
    writes: list[manual_campaign.LightweightPresentationOutcome] = []
    publish = manual_campaign._publish_presentation_outcome

    def capture(
        target: ReportService,
        outcome: manual_campaign.LightweightPresentationOutcome,
        **kwargs: object,
    ) -> bool:
        written = publish(target, outcome, **kwargs)
        if written:
            writes.append(outcome)
        return written

    monkeypatch.setattr(manual_campaign, "_publish_presentation_outcome", capture)
    lease = LeaseManager(
        service,
        DashboardState(
            instance_id="finished-ok",
            selected_ids=(cell.cell_id,),
            recycled_ids=set(),
            static_na_ids=set(),
            source_revision="a" * 40,
        ),
    )

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": status,
            "detail": "resolved reusable OK current",
            "completed_at_ns": 2,
        }
    )

    assert len(writes) == expected_writes
    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    if expected_status is None:
        assert observed is None
    else:
        assert observed is not None
        assert observed.status == expected_status
        if expected_writes:
            assert observed.attempt_id == current.attempt_id


@pytest.mark.parametrize("continuation_at_success", (False, True))
def test_cross_revision_success_tombstones_historical_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    continuation_at_success: bool,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    manual_campaign._publish_presentation_outcome(
        service,
        _presentation_outcome(
            cell.cell_id,
            status="error",
            completed_at_ns=1,
            source_revision="a" * 40,
        ),
    )
    manual_campaign.update_source_marker(
        service,
        manual_campaign.ReportSourceIdentity("b" * 40, "c" * 40, ()),
        continue_across_revisions=continuation_at_success,
    )
    current = _valid_presentation_current(cell.cell_id, tmp_path)
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_current",
        lambda *_args, **_kwargs: current,
    )
    lease = LeaseManager(
        service,
        DashboardState(
            instance_id="cross-revision-success",
            selected_ids=(cell.cell_id,),
            recycled_ids=set(),
            static_na_ids=set(),
            source_revision="b" * 40,
        ),
    )

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": "ok",
            "detail": "new revision succeeded",
            "completed_at_ns": 2,
        }
    )

    if not continuation_at_success:
        manual_campaign.update_source_marker(
            service,
            manual_campaign.ReportSourceIdentity("b" * 40, "c" * 40, ()),
            continue_across_revisions=True,
        )

    tombstone = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="b" * 40,
        accept_historical_source=True,
    )
    assert tombstone is not None
    assert tombstone.successful
    assert tombstone.source_revision == "b" * 40
    caches, merged = _merge_lightweight_snapshot(
        service,
        {},
        {cell.cell_id: tombstone},
    )
    assert merged == 0
    assert (
        _cache_measurement(
            caches,
            dataset_id=cell.dataset_id,
            cell_id=cell.cell_id,
        )["status"]
        == ResultStatus.NOT_AVAILABLE.value
    )


def test_front_end_ok_recycle_skips_immediately_when_cell_lock_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _valid_presentation_current(cell.cell_id, tmp_path)
    calls: list[tuple[str, float | None]] = []
    writes: list[manual_campaign.LightweightPresentationOutcome] = []

    class BusyLock:
        def __enter__(self) -> None:
            raise manual_campaign.LockTimeoutError("busy")

        def __exit__(
            self,
            _error_type: object,
            _error: object,
            _traceback: object,
        ) -> None:
            return None

    def busy_lock(name: str, **kwargs: object) -> BusyLock:
        timeout = kwargs.get("timeout")
        calls.append((name, timeout if isinstance(timeout, float) else None))
        return BusyLock()

    monkeypatch.setattr(service.store, "named_lock", busy_lock)
    monkeypatch.setattr(
        manual_campaign,
        "_publish_presentation_outcome",
        lambda _service, outcome, **_kwargs: writes.append(outcome),
    )

    warnings = manual_campaign._publish_recycled_presentation_outcomes(
        service,
        {cell.cell_id: current},
        (cell.cell_id,),
        source_revision="a" * 40,
        campaign_invocation_id="no-ledger",
    )

    assert warnings == ()
    assert calls == [(f"campaign-cell-{cell.cell_id}", 0.0)]
    assert writes == []


def test_front_end_recycle_reconciles_outcome_under_campaign_cell_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _valid_presentation_current(cell.cell_id, tmp_path)
    manual_campaign._publish_presentation_outcome(
        service,
        _presentation_outcome(cell.cell_id, status="error"),
    )
    campaign_lock = f"campaign-cell-{cell.cell_id}"
    held: list[str] = []

    class TrackingLock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            held.append(self.name)

        def __exit__(
            self,
            _error_type: object,
            _error: object,
            _traceback: object,
        ) -> None:
            assert held.pop() == self.name

    monkeypatch.setattr(
        service.store,
        "named_lock",
        lambda name, **_kwargs: TrackingLock(name),
    )
    read = manual_campaign.lightweight_presentation_outcome
    publish = manual_campaign._publish_presentation_outcome
    operations: list[tuple[str, bool]] = []

    def tracked_read(*args: object, **kwargs: object):
        operations.append(("read", campaign_lock in held))
        return read(*args, **kwargs)

    def tracked_publish(*args: object, **kwargs: object) -> None:
        assert campaign_lock in held
        operations.append(("write", True))
        publish(*args, **kwargs)

    monkeypatch.setattr(
        manual_campaign,
        "lightweight_presentation_outcome",
        tracked_read,
    )
    monkeypatch.setattr(
        manual_campaign,
        "_publish_presentation_outcome",
        tracked_publish,
    )

    warnings = manual_campaign._publish_recycled_presentation_outcomes(
        service,
        {cell.cell_id: current},
        (cell.cell_id,),
        source_revision="a" * 40,
        campaign_invocation_id="lock-order",
    )

    assert warnings == ()
    assert operations == [("read", False), ("write", True)]
    assert held == []


def test_front_end_recycle_honours_marker_switch_to_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    manual_campaign._publish_presentation_outcome(
        service,
        _presentation_outcome(
            cell.cell_id,
            status="error",
            source_revision="a" * 40,
        ),
    )
    source_b = manual_campaign.ReportSourceIdentity("b" * 40, "c" * 40, ())
    manual_campaign.update_source_marker(service, source_b)
    current = _valid_presentation_current(cell.cell_id, tmp_path)
    read = manual_campaign.lightweight_presentation_outcome
    switched = False

    def read_then_switch(*args: object, **kwargs: object):
        nonlocal switched
        outcome = read(*args, **kwargs)
        if not switched:
            switched = True
            manual_campaign.update_source_marker(
                service,
                source_b,
                continue_across_revisions=True,
            )
        return outcome

    monkeypatch.setattr(
        manual_campaign,
        "lightweight_presentation_outcome",
        read_then_switch,
    )

    warnings = manual_campaign._publish_recycled_presentation_outcomes(
        service,
        {cell.cell_id: current},
        (cell.cell_id,),
        source_revision=source_b.revision,
        campaign_invocation_id="marker-switch",
        accept_historical_source=False,
    )

    assert warnings == ()
    outcome = read(
        service,
        cell,
        source_revision=source_b.revision,
        accept_historical_source=True,
    )
    assert outcome is not None
    assert outcome.successful
    assert outcome.source_revision == source_b.revision


def _stub_run_campaign_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    currents: dict[str, LightweightCurrent],
    planned: tuple[manual_campaign.PlannedCell, ...],
    outcomes: tuple[CellOutcome, ...] = (),
) -> None:
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_args, **_kwargs: dict(currents),
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
    monkeypatch.setattr(
        manual_campaign,
        "require_measurement_ready",
        lambda _source: None,
    )
    monkeypatch.setattr(
        manual_campaign,
        "_publish_campaign_summary_ids",
        lambda *_args, **_kwargs: (tmp_path / "summary", {}),
    )
    monkeypatch.setattr(
        manual_campaign,
        "_print_campaign_summary_ids",
        lambda *_args, **_kwargs: None,
    )

    class StubScheduler:
        def __init__(self, _service: ReportService, *, settings: object) -> None:
            self.settings = settings

        def run(
            self,
            observed: tuple[manual_campaign.PlannedCell, ...],
        ) -> manual_campaign.CampaignResult:
            assert tuple(observed) == planned
            return manual_campaign.CampaignResult(tuple(observed), outcomes)

    monkeypatch.setattr(manual_campaign, "CampaignScheduler", StubScheduler)


@pytest.mark.parametrize("mixed", (False, True))
def test_run_campaign_recycle_paths_replace_failure_with_known_ok_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mixed: bool,
) -> None:
    service = _manual_service(tmp_path)
    measurable = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is None
    )
    recycled_cell = measurable[0]
    work_cell = measurable[1]
    current = _valid_presentation_current(recycled_cell.cell_id, tmp_path)
    manual_campaign._publish_presentation_outcome(
        service,
        _presentation_outcome(recycled_cell.cell_id, status="error"),
    )
    planned = (
        (
            manual_campaign.PlannedCell(
                work_cell,
                dependency=False,
                baseline_cell_id=None,
                rank=1,
            ),
        )
        if mixed
        else ()
    )
    outcomes = (
        (CellOutcome(work_cell.cell_id, "ok", "completed"),) if mixed else ()
    )
    _stub_run_campaign_boundaries(
        monkeypatch,
        tmp_path,
        currents={recycled_cell.cell_id: current},
        planned=planned,
        outcomes=outcomes,
    )
    arguments = _parse("run", "--no-dashboard")
    source = manual_campaign.ReportSourceIdentity("a" * 40, "b" * 40, ())

    return_code = manual_campaign._run_campaign(
        arguments,
        repo_root=ROOT,
        service=service,
        source=source,
        cells=(recycled_cell, work_cell) if mixed else (recycled_cell,),
        palette=manual_campaign.Palette(False),
    )

    assert return_code == 0
    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        recycled_cell,
        source_revision=source.revision,
    )
    assert observed is not None
    assert observed.status == "reused"
    assert observed.attempt_id == current.attempt_id


def test_cross_revision_continuation_recycles_ok_and_replans_blocked_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    old_revision = "a" * 40
    active_source = manual_campaign.ReportSourceIdentity("b" * 40, "c" * 40, ())
    ok_cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n3-dd-z-jets-selected-flow"
    )
    blocked_cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n4-dd-ttzh-jets-selected-flow"
    )

    def historical(
        cell: CellSpec,
        *,
        status: str,
        reusable: bool,
    ) -> LightweightCurrent:
        return LightweightCurrent(
            cell_id=cell.cell_id,
            attempt_id=f"historical-{status}",
            result_path=tmp_path / f"{cell.cell_id}.json",
            result={
                "status": status,
                "provenance": {"report_source_revision": old_revision},
            },
            complete=reusable,
            reusable=reusable,
            reason=("historical source current" if reusable else "incomplete status"),
        )

    currents = {
        ok_cell.cell_id: historical(ok_cell, status="ok", reusable=True),
        blocked_cell.cell_id: historical(
            blocked_cell,
            status="skip",
            reusable=False,
        ),
    }
    planned = (
        manual_campaign.PlannedCell(
            blocked_cell,
            dependency=False,
            baseline_cell_id=None,
            rank=0,
            optional_baseline_cell_id=(
                "reference-amplicol-lc-n4-dd-ttzh-jets-selected-flow"
            ),
        ),
    )
    _stub_run_campaign_boundaries(
        monkeypatch,
        tmp_path,
        currents=currents,
        planned=planned,
        outcomes=(CellOutcome(blocked_cell.cell_id, "ok", "completed"),),
    )
    monkeypatch.setattr(
        manual_campaign,
        "lightweight_currents",
        lambda *_args, **kwargs: (
            dict(currents) if kwargs.get("accept_historical_source") else {}
        ),
    )
    requested: list[tuple[str, ...]] = []

    def capture_plan(cells: tuple[CellSpec, ...], **_kwargs: object):
        requested.append(tuple(cell.cell_id for cell in cells))
        return planned

    monkeypatch.setattr(manual_campaign, "plan_campaign", capture_plan)

    return_code = manual_campaign._run_campaign(
        _parse("run", "--continue-across-revisions", "--no-dashboard"),
        repo_root=ROOT,
        service=service,
        source=active_source,
        cells=(ok_cell, blocked_cell),
        palette=manual_campaign.Palette(False),
    )

    assert return_code == 0
    assert requested == [(blocked_cell.cell_id,)]


def test_run_campaign_recycled_cap_does_not_erase_newer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    current = _capped_presentation_current(cell.cell_id, tmp_path)
    error = _presentation_outcome(cell.cell_id, status="error")
    manual_campaign._publish_presentation_outcome(service, error)
    _stub_run_campaign_boundaries(
        monkeypatch,
        tmp_path,
        currents={cell.cell_id: current},
        planned=(),
    )

    return_code = manual_campaign._run_campaign(
        _parse("run", "--no-dashboard"),
        repo_root=ROOT,
        service=service,
        source=manual_campaign.ReportSourceIdentity("a" * 40, "b" * 40, ()),
        cells=(cell,),
        palette=manual_campaign.Palette(False),
    )

    assert return_code == 0
    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        == error
    )


def test_catalog_static_na_takes_precedence_over_presentation_outcome(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if REPORT_CATALOG.static_na_reason(cell) is not None
    )
    caches, merged = _merge_lightweight_snapshot(
        service,
        {},
        {cell.cell_id: _presentation_outcome(cell.cell_id, status="error")},
    )

    assert merged == 0
    assert _cache_measurement(
        caches,
        dataset_id=cell.dataset_id,
        cell_id=cell.cell_id,
    ) == empty_measurement()


@pytest.mark.parametrize(
    "status",
    (
        "generation_limit",
        "memory_limit",
        "worker_timeout",
        "profiling_timeout",
        "validation_timeout",
        "dependency",
        "resource_frontier",
        "error",
        "validation_failed",
        "blocked_dependency",
        "skip",
        "preparation_error",
        "failed",
        "unsupported",
        "cancelled",
    ),
)
def test_lease_manager_routes_every_started_terminal_status_to_presentation(
    tmp_path: Path,
    status: str,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    state = DashboardState(
        instance_id=f"terminal-{status}",
        selected_ids=(cell.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(service, state)

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": status,
            "detail": (
                "worker terminated by cancellation"
                if status == "cancelled"
                else "terminal outcome"
            ),
            "completed_at_ns": 123,
        }
    )

    observed = manual_campaign.lightweight_presentation_outcome(
        service,
        cell,
        source_revision="a" * 40,
    )
    assert observed is not None
    assert observed.status == status
    assert observed.label == manual_campaign._humanized_outcome_label(status)
    assert observed.attempt_id is None
    assert observed.completed_at_ns == 123


@pytest.mark.parametrize("status", ("ok", "reused", "skipped-current"))
def test_lease_manager_does_not_publish_success_without_reusable_ok_current(
    tmp_path: Path,
    status: str,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    lease = LeaseManager(
        service,
        DashboardState(
            instance_id=f"terminal-{status}",
            selected_ids=(cell.cell_id,),
            recycled_ids=set(),
            static_na_ids=set(),
            source_revision="a" * 40,
        ),
    )

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": status,
            "detail": "no reusable OK current exists",
            "completed_at_ns": 123,
        }
    )

    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        is None
    )


@pytest.mark.parametrize("raises", (False, True))
def test_scheduler_finished_event_has_positive_completion_timestamp_after_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raises: bool,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    events: list[dict[str, object]] = []
    campaign_lock = f"campaign-cell-{cell.cell_id}"
    held: list[str] = []

    class TrackingLock:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            held.append(self.name)

        def __exit__(
            self,
            _error_type: object,
            _error: object,
            _traceback: object,
        ) -> None:
            assert held.pop() == self.name

    monkeypatch.setattr(
        service.store,
        "named_lock",
        lambda name, **_kwargs: TrackingLock(name),
    )

    def observe(payload: object) -> None:
        assert isinstance(payload, dict)
        if payload.get("event") == "finished":
            assert campaign_lock in held
        events.append(dict(payload))

    scheduler = manual_campaign.CampaignScheduler(
        service,
        settings=manual_campaign.CampaignSettings(
            source_identity_override=manual_campaign.ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
            progress_observer=observe,
        ),
    )
    planned = manual_campaign.PlannedCell(
        cell,
        dependency=False,
        baseline_cell_id=None,
        rank=0,
    )
    def run_in_lane(_planned: manual_campaign.PlannedCell) -> CellOutcome:
        scheduler._observe("started", cell, dependency=False)
        if raises:
            raise RuntimeError("synthetic worker failure")
        return CellOutcome(cell.cell_id, "ok", "completed")

    monkeypatch.setattr(scheduler, "_run_cell_in_lane", run_in_lane)
    before = time.time_ns()
    if raises:
        with pytest.raises(RuntimeError, match="synthetic worker failure"):
            scheduler._run_cell(planned)
    else:
        assert scheduler._run_cell(planned).status == "ok"
    after = time.time_ns()

    assert [event["event"] for event in events] == ["started", "finished"]
    finished = events[-1]
    assert isinstance(finished["completed_at_ns"], int)
    assert before <= finished["completed_at_ns"] <= after
    assert finished["status"] == ("error" if raises else "ok")


def test_scheduler_inner_coordination_deferred_emits_no_finished_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    scheduler = manual_campaign.CampaignScheduler(
        _manual_service(tmp_path),
        settings=manual_campaign.CampaignSettings(
            source_identity_override=manual_campaign.ReportSourceIdentity(
                "a" * 40, "b" * 40, ()
            ),
            progress_observer=lambda payload: events.append(dict(payload)),
        ),
    )
    planned = manual_campaign.PlannedCell(
        _presentation_test_cell(), False, None, 0
    )
    monkeypatch.setattr(scheduler, "_coordination_lock_names", lambda _cell: ())
    monkeypatch.setattr(
        scheduler,
        "_run_cell_in_lane",
        lambda _planned: (_ for _ in ()).throw(_CoordinationDeferred("inner")),
    )

    with pytest.raises(_CoordinationDeferred, match="inner"):
        scheduler._run_cell(planned)

    assert [event for event in events if event["event"] == "finished"] == []


def test_scheduler_inner_lock_cancel_emits_one_cancelled_and_no_presentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    events: list[dict[str, object]] = []
    lease = LeaseManager(
        service,
        DashboardState(
            instance_id="inner-lock-cancel",
            selected_ids=(cell.cell_id,),
            recycled_ids=set(),
            static_na_ids=set(),
            source_revision="a" * 40,
        ),
    )

    def observe(payload: object) -> None:
        assert isinstance(payload, dict)
        events.append(dict(payload))
        lease.observe(payload)

    scheduler = manual_campaign.CampaignScheduler(
        service,
        settings=manual_campaign.CampaignSettings(
            source_identity_override=manual_campaign.ReportSourceIdentity(
                "a" * 40, "b" * 40, ()
            ),
            progress_observer=observe,
        ),
    )
    planned = manual_campaign.PlannedCell(cell, False, None, 0)
    monkeypatch.setattr(scheduler, "_coordination_lock_names", lambda _cell: ())
    monkeypatch.setattr(
        scheduler,
        "_run_cell_in_lane",
        lambda _planned: (_ for _ in ()).throw(LockCancelledError("cancelled")),
    )

    outcome = scheduler._run_cell(planned)

    finished = [event for event in events if event["event"] == "finished"]
    assert [(event["status"], event["detail"]) for event in finished] == [
        ("cancelled", "lock wait cancelled")
    ]
    assert not any(event.get("status") == "error" for event in events)
    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        is None
    )
    assert outcome == CellOutcome(cell.cell_id, "cancelled", "lock wait cancelled")


@pytest.mark.parametrize(
    ("status", "detail"),
    (
        (ResultStatus.NOT_AVAILABLE.value, "canonical reset"),
        ("cancelled", "not started"),
        ("cancelled", "lock wait cancelled"),
    ),
)
def test_lease_manager_does_not_publish_non_attempt_outcomes(
    tmp_path: Path,
    status: str,
    detail: str,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    lease = LeaseManager(
        service,
        DashboardState(
            instance_id="non-attempt",
            selected_ids=(cell.cell_id,),
            recycled_ids=set(),
            static_na_ids=set(),
            source_revision="a" * 40,
        ),
    )

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell.cell_id,
            "status": status,
            "detail": detail,
            "completed_at_ns": 123,
        }
    )

    assert (
        manual_campaign.lightweight_presentation_outcome(
            service,
            cell,
            source_revision="a" * 40,
        )
        is None
    )


def test_lease_publication_uses_runtime_revision_without_checkout_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="installed",
        selected_ids=(),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )

    def forbidden(_repo_root: Path) -> str:
        raise AssertionError("installed campaign lease inspected checkout Git")

    monkeypatch.setattr(manual_campaign, "_repo_head", forbidden)
    lease = LeaseManager(service, state)
    lease.publish()

    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["source_revision"] == "a" * 40


def test_read_only_views_use_recorded_measurement_source_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path, initialize_source_marker=False)
    checkout_revision = "b" * 40
    measurement_revision = "a" * 40
    assert (
        manual_campaign.recorded_measurement_source_revision(
            service,
            checkout_revision=checkout_revision,
        )
        == checkout_revision
    )

    marker = service.paths.coordination_root / "manual-source.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema": manual_campaign.SOURCE_MARKER_SCHEMA,
                "source_revision": measurement_revision,
            }
        ),
        encoding="utf-8",
    )
    assert (
        manual_campaign.recorded_measurement_source_revision(
            service,
            checkout_revision=checkout_revision,
        )
        == measurement_revision
    )

    observed: list[tuple[str, bool]] = []

    def stop_after_source_selection(
        _service: ReportService,
        *,
        source_revision: str,
        accept_historical_source: bool = False,
    ) -> object:
        observed.append((source_revision, accept_historical_source))
        raise RuntimeError("snapshot stopped after source selection")

    monkeypatch.setattr(
        manual_campaign,
        "_capture_lightweight_snapshot",
        stop_after_source_selection,
    )
    source = manual_campaign.ReportSourceIdentity(
        checkout_revision,
        "c" * 40,
        (),
    )
    with pytest.raises(RuntimeError, match="snapshot stopped"):
        manual_campaign._refresh_pdf(
            _parse("refresh-pdf", "--quiet"),
            service=service,
            source=source,
            palette=manual_campaign.Palette(False),
        )
    assert observed == [(measurement_revision, False)]


def test_cross_revision_policy_persists_for_inspect_and_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path, initialize_source_marker=False)
    source = manual_campaign.ReportSourceIdentity(
        "a" * 40,
        "c" * 40,
        (),
    )

    manual_campaign.update_source_marker(
        service,
        source,
        continue_across_revisions=True,
    )
    policy = manual_campaign.recorded_measurement_source_policy(
        service,
        checkout_revision="b" * 40,
    )

    assert policy.source_revision == source.revision
    assert policy.continue_across_revisions
    marker = json.loads(
        (service.paths.coordination_root / "manual-source.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["continue_across_revisions"] is True

    observed: list[tuple[str, bool]] = []

    def stop_after_source_selection(
        _service: ReportService,
        *,
        source_revision: str,
        accept_historical_source: bool = False,
    ) -> object:
        observed.append((source_revision, accept_historical_source))
        raise RuntimeError("mixed snapshot selected")

    monkeypatch.setattr(
        manual_campaign,
        "_capture_lightweight_snapshot",
        stop_after_source_selection,
    )
    with pytest.raises(RuntimeError, match="mixed snapshot selected"):
        manual_campaign._refresh_pdf(
            _parse("refresh-pdf", "--quiet"),
            service=service,
            source=manual_campaign.ReportSourceIdentity(
                "b" * 40,
                "d" * 40,
                (),
            ),
            palette=manual_campaign.Palette(False),
        )
    assert observed == [(source.revision, True)]


def test_refresh_renders_outcomes_without_serializing_them_as_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _manual_service(tmp_path)
    cell = _presentation_test_cell()
    outcome = _presentation_outcome(cell.cell_id, status="error")
    observed: dict[str, str] = {}
    original_render = ReportService._render_tables

    monkeypatch.setattr(
        manual_campaign,
        "_capture_lightweight_snapshot",
        lambda *_args, **_kwargs: ({}, {cell.cell_id: outcome}, ()),
    )
    monkeypatch.setattr(
        manual_campaign,
        "_copy_report_sources",
        lambda _source, destination: destination.mkdir(parents=True),
    )
    monkeypatch.setattr(
        manual_campaign,
        "_stage_copied_profile_identity",
        lambda *_args, **_kwargs: None,
    )

    def capture_render(
        staging_service: ReportService,
        caches: dict[str, dict[str, object]],
    ) -> dict[str, str]:
        observed["render_caches"] = json.dumps(caches, sort_keys=True)
        tables = original_render(staging_service, caches)
        observed["tables"] = "\n".join(tables.values())
        return tables

    def capture_snapshot(
        _staging_service: ReportService,
        caches: dict[str, dict[str, object]],
        tables: dict[str, str],
    ) -> tuple[Path, ...]:
        observed["snapshot_caches"] = json.dumps(caches, sort_keys=True)
        observed["snapshot_tables"] = "\n".join(tables.values())
        return ()

    monkeypatch.setattr(ReportService, "_render_tables", capture_render)
    monkeypatch.setattr(ReportService, "_snapshot_files", capture_snapshot)
    monkeypatch.setattr(manual_campaign, "_compile_pdf", lambda *_a, **_k: 59)
    monkeypatch.setattr(
        manual_campaign,
        "_install_report_snapshot",
        lambda *_args, **_kwargs: None,
    )

    assert (
        manual_campaign._refresh_pdf(
            _parse("refresh-pdf", "--quiet"),
            service=service,
            source=manual_campaign.ReportSourceIdentity(
                "a" * 40,
                "b" * 40,
                (),
            ),
            palette=manual_campaign.Palette(False),
        )
        == 0
    )

    failure_kind = "ManualCampaignOutcome:error"
    status_tex = r"\matrixstatus{ReportRed}{error}"
    assert failure_kind in observed["render_caches"]
    assert status_tex in observed["tables"]
    assert failure_kind not in observed["snapshot_caches"]
    assert status_tex in observed["snapshot_tables"]


def test_malformed_measurement_source_marker_is_not_guessed(tmp_path: Path) -> None:
    service = _manual_service(tmp_path, initialize_source_marker=False)
    marker = service.paths.coordination_root / "manual-source.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema": manual_campaign.SOURCE_MARKER_SCHEMA,
                "source_revision": "short",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ManualCampaignError, match="source marker is malformed"):
        manual_campaign.recorded_measurement_source_revision(
            service,
            checkout_revision="b" * 40,
        )


def _write_lease(
    service: ReportService,
    *,
    instance_id: str,
    source_revision: str,
    updated_at: float,
    workers: dict[str, dict[str, object]],
    counters: dict[str, int] | None = None,
    limits: dict[str, object] | None = None,
    started_at: float | None = None,
    schema: object = MANUAL_STATE_SCHEMA,
) -> None:
    path = service.paths.coordination_root / "manual-leases" / f"{instance_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if counters is None:
        active = sum(
            worker.get("status") in {"queued", "preparing", "running"}
            for worker in workers.values()
        )
        counters = {
            "selected": len(workers),
            "recycled": 0,
            "active": active,
            "completed": 0,
            "remaining": max(0, len(workers) - active),
            "static_na": 0,
            "capped": 0,
            "failed": 0,
            "dependency_only": 0,
        }
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "instance_id": instance_id,
                "source_revision": source_revision,
                "started_at": updated_at if started_at is None else started_at,
                "updated_at": updated_at,
                "counters": counters,
                **({} if limits is None else {"limits": limits}),
                "workers": workers,
            }
        ),
        encoding="utf-8",
    )


def test_peer_leases_merge_by_sha_without_accumulating_or_double_counting(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="local",
        selected_ids=("cell-a", "cell-b"),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
        workers={"cell-a": WorkerView("cell-a", status="running")},
    )
    lease = LeaseManager(service, state)
    now = time.time()
    active_worker = {
        "cell_id": "cell-a",
        "status": "running",
        "phase": "generation",
        "step": "generating",
        "progress_task_id": "generation:recurrence:rust-builder",
        "progress_details": {
            "current_count": 38_581,
            "contribution_count": 286_294,
        },
        "updated_at": now,
    }
    _write_lease(
        service,
        instance_id="peer-one",
        source_revision=state.source_revision,
        updated_at=now,
        workers={"cell-a": active_worker},
    )
    _write_lease(
        service,
        instance_id="peer-two",
        source_revision=state.source_revision,
        updated_at=now,
        workers={"cell-b": active_worker | {"cell_id": "cell-b"}},
    )
    _write_lease(
        service,
        instance_id="wrong-source",
        source_revision="b" * 40,
        updated_at=now,
        workers={"cell-b": active_worker | {"cell_id": "cell-b"}},
    )
    _write_lease(
        service,
        instance_id="stale",
        source_revision=state.source_revision,
        updated_at=0.0,
        workers={"cell-b": active_worker | {"cell_id": "cell-b"}},
    )

    first = lease.dashboard_snapshot()
    second = lease.dashboard_snapshot()

    expected_keys = {
        "cell-a",
        "peer:peer-one:cell-a",
        "peer:peer-two:cell-b",
    }
    assert set(first.workers) == expected_keys
    assert set(second.workers) == expected_keys
    peer = first.workers["peer:peer-one:cell-a"]
    assert peer.progress_task_id == "generation:recurrence:rust-builder"
    assert peer.progress_details == {
        "current_count": 38_581,
        "contribution_count": 286_294,
    }
    assert first.counters()["active"] == 1
    assert first.counters()["remaining"] == 1
    assert len(first.peer_active) == 2
    assert "Peer active 2" in render_dashboard_frame(first, width=120, height=36)

    lease.publish()
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["counters"]["active"] == 1


def test_lease_manager_publishes_effective_invocation_caps(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="bounded-smoke",
        selected_ids=("cell-a",),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
        generation_time_limit_seconds=600.0,
        memory_limit_bytes=30_000_000_000,
        worker_wall_limit_seconds=1200.0,
    )
    lease = LeaseManager(service, state)

    lease.publish()

    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["limits"] == {
        "generation_time_limit_seconds": 600.0,
        "memory_limit_bytes": 30_000_000_000,
        "worker_wall_limit_seconds": 1200.0,
    }


def test_preflight_completion_removes_only_matching_transient_worker(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    state = DashboardState(
        instance_id="preflight-lifecycle",
        selected_ids=(cell.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(service, state)
    attempt_id = "prepared-model-builtin_sm"

    lease.observe({"event": "started", "cell_id": cell.cell_id})
    lease.observe(
        {
            "event": "worker",
            "cell_id": cell.cell_id,
            "attempt_id": attempt_id,
        }
    )

    assert state.counters()["active"] == 1
    lease.observe(
        {
            "event": "preflight-finished",
            "cell_id": cell.cell_id,
            "attempt_id": "different-attempt",
        }
    )
    assert cell.cell_id in state.workers

    lease.observe(
        {
            "event": "preflight-finished",
            "cell_id": cell.cell_id,
            "attempt_id": attempt_id,
        }
    )

    assert cell.cell_id not in state.workers
    assert state.completed_ids == set()
    assert state.recycled_ids == set()
    assert state.failed_ids == set()
    assert state.counters()["active"] == 0
    assert state.counters()["completed"] == 0
    assert state.counters()["remaining"] == 1
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["workers"] == {}


def test_scheduler_state_is_visible_and_persisted_without_creating_worker_rows(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="scheduler-state",
        selected_ids=("cell-a", "cell-b", "cell-c"),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(service, state)

    lease.observe(
        {
            "event": "scheduler-state",
            "ready": 1,
            "waiting_dependency": 1,
            "waiting_coordination_lock": 1,
        }
    )

    assert state.workers == {}
    assert (state.ready_count, state.waiting_dependency_count) == (1, 1)
    assert state.waiting_coordination_lock_count == 1
    frame = render_dashboard_frame(state, width=120, height=36)
    assert "Ready 1" in frame
    assert "Waiting dependency 1" in frame
    assert "Waiting coordination lock 1" in frame
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["scheduler"] == {
        "ready": 1,
        "waiting_dependency": 1,
        "waiting_coordination_lock": 1,
    }


def test_blocked_dependency_dashboard_retains_exact_prerequisite_ids(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell_id = "matrix-compiled-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    prerequisite = "matrix-recurrence-builtin-sm-lc-n1-dd-z-jets-selected-flow"
    state = DashboardState(
        instance_id="blocked-dependency",
        selected_ids=(cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(service, state)

    lease.observe(
        {
            "event": "finished",
            "cell_id": cell_id,
            "status": "blocked_dependency",
            "detail": "blocked by dependency",
            "prerequisite_cell_ids": (prerequisite,),
        }
    )

    worker = state.workers[cell_id]
    assert worker.blocked_prerequisite_ids == (prerequisite,)
    assert prerequisite in render_dashboard_frame(state, width=160, height=48)
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["workers"][cell_id]["blocked_prerequisite_ids"] == [prerequisite]


def test_active_lease_recipe_uses_effective_invocation_settings(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    defaults = ReproductionSettings()
    assert (
        defaults.cores,
        defaults.target_runtime,
        defaults.batch_size,
        defaults.warmups,
        defaults.minimum_samples,
    ) == (1, 5.0, 128, 2, 5)
    state = DashboardState(
        instance_id="recipe-settings",
        selected_ids=(cell.cell_id,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
        reproduction_settings=ReproductionSettings(
            cores=3,
            target_runtime=0.5,
            batch_size=16,
            warmups=1,
            minimum_samples=7,
        ),
    )
    lease = LeaseManager(service, state)

    lease.observe({"event": "started", "cell_id": cell.cell_id})

    worker = lease.dashboard_snapshot().workers[cell.cell_id]
    assert worker.reproduce_generate is not None
    assert "--workers 3" in worker.reproduce_generate
    assert "--cores 3" in worker.reproduce_generate
    assert "--batch-size 16" in worker.reproduce_generate
    assert worker.reproduce_profile is not None
    assert "--target-runtime 0.5" in worker.reproduce_profile
    assert "--batch-size 16" in worker.reproduce_profile
    assert "--warmup-runs 1" in worker.reproduce_profile
    assert "--minimum-samples 7" in worker.reproduce_profile
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert "reproduction_settings" not in payload
    assert (
        payload["workers"][cell.cell_id]["reproduce_profile"]
        == worker.reproduce_profile
    )


def test_live_dashboard_snapshot_selects_instance_and_shows_same_source_peers(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    now = 50_000.0
    source_revision = "a" * 40
    counters = {
        "selected": 9,
        "recycled": 3,
        "active": 2,
        "completed": 1,
        "remaining": 2,
        "static_na": 1,
        "capped": 1,
        "failed": 0,
        "unverified": 0,
        "dependency_only": 2,
    }
    _write_lease(
        service,
        instance_id="primary-7f91c2",
        source_revision=source_revision,
        started_at=now - 65.0,
        updated_at=now - 1.0,
        counters=counters,
        workers={
            "matrix-primary": {
                "cell_id": "matrix-primary",
                "status": "running",
                "phase": "profiling",
                "step": "sample 4/8",
                "pid": 4100,
                "member_pids": [4100, 4101],
                "wall_seconds": 64.0,
                "cpu_seconds": 91.0,
                "current_rss_bytes": 2_500_000_000,
                "peak_rss_bytes": 3_000_000_000,
                "progress_completed": 4,
                "progress_total": 8,
                "events": ["profiling: sample 4/8"],
            }
        },
    )
    _write_lease(
        service,
        instance_id="peer-81ab",
        source_revision=source_revision,
        updated_at=now - 2.0,
        workers={
            "scalar-peer": {
                "cell_id": "scalar-peer",
                "status": "running",
                "phase": "generation",
                "step": "gravity model generation",
                "pid": 4200,
            },
            "finished-peer": {
                "cell_id": "finished-peer",
                "status": "ok",
                "phase": "completed",
                "step": "published",
            },
        },
    )
    _write_lease(
        service,
        instance_id="foreign-newer",
        source_revision="b" * 40,
        updated_at=now - 0.5,
        workers={
            "foreign": {
                "cell_id": "foreign",
                "status": "running",
            }
        },
    )

    state = _live_dashboard_snapshot(
        service.paths.coordination_root,
        instance="primary-7",
        stale_after_seconds=15.0,
        now=now,
    )

    assert state.live_snapshot
    assert state.instance_id == "primary-7f91c2"
    assert state.source_revision == source_revision
    assert state.started_at == now - 65.0
    assert state.lease_updated_at == now - 1.0
    assert state.counters() == counters
    assert set(state.workers) == {
        "matrix-primary",
        "peer:peer-81ab:scalar-peer",
    }
    peer = state.workers["peer:peer-81ab:scalar-peer"]
    assert peer.peer_instance == "peer-81ab"
    frame = render_dashboard_frame(state, width=160, height=48)
    assert "LIVE pyAmpliCol profiling campaign" in frame
    assert "primary-7f91" in frame
    assert "Selected 9" in frame
    assert "Recycled 3" in frame
    assert "Remaining 2" in frame
    assert "matrix-primary" in frame
    assert "scalar-peer" in frame


@pytest.mark.parametrize(("width", "height"), ((80, 24), (120, 36), (160, 48)))
def test_live_dashboard_snapshot_renders_effective_invocation_caps(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    service = _manual_service(tmp_path)
    now = 60_000.0
    _write_lease(
        service,
        instance_id="bounded-smoke",
        source_revision="b" * 40,
        updated_at=now - 1.0,
        limits={
            "generation_time_limit_seconds": 600.0,
            "memory_limit_bytes": 30_000_000_000,
            "worker_wall_limit_seconds": 1200.0,
        },
        workers={
            "scalar-gravity": {
                "cell_id": "scalar-gravity",
                "status": "running",
                "phase": "generation",
                "step": "building scalar-gravity evaluator",
            }
        },
    )

    state = _live_dashboard_snapshot(
        service.paths.coordination_root,
        instance="bounded",
        stale_after_seconds=15.0,
        now=now,
    )

    assert state.generation_time_limit_seconds == 600.0
    assert state.memory_limit_bytes == 30_000_000_000
    assert state.worker_wall_limit_seconds == 1200.0
    frame = render_dashboard_frame(state, width=width, height=height)
    assert "Caps Generation 00:10:00" in frame
    assert "RAM 30.00 GB" in frame
    assert "Total wall 00:20:00" in frame


def test_live_snapshot_command_is_read_only_and_lease_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs/performance_reports/macbook_M3_manual"
    docs.mkdir(parents=True)
    service = ReportService(manual_campaign._campaign_report_paths(tmp_path, docs))
    now = time.time()
    _write_lease(
        service,
        instance_id="lease-only-instance",
        source_revision="c" * 40,
        started_at=now - 3.0,
        updated_at=now,
        counters={
            "selected": 3,
            "recycled": 1,
            "active": 1,
            "completed": 0,
            "remaining": 1,
            "static_na": 0,
            "capped": 0,
            "failed": 0,
            "dependency_only": 1,
        },
        workers={
            "gravity-cell": {
                "cell_id": "gravity-cell",
                "status": "running",
                "phase": "generation",
                "step": "building scalar-gravity evaluator",
                "pid": 4300,
            }
        },
    )
    lease_files = tuple(
        sorted((service.paths.coordination_root / "manual-leases").glob("*.json"))
    )
    before = {path: path.read_bytes() for path in lease_files}

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("live snapshot escaped the lease-only read path")

    monkeypatch.setattr(manual_campaign, "_repo_head", forbidden)
    monkeypatch.setattr(manual_campaign, "lightweight_source_identity", forbidden)
    monkeypatch.setattr(manual_campaign, "lightweight_currents", forbidden)
    monkeypatch.setattr(manual_campaign, "_capture_lightweight_snapshot", forbidden)

    result = campaign_main(
        (
            "dashboard-snapshot",
            "--live",
            "--instance",
            "lease-only",
            "--width",
            "120",
            "--height",
            "36",
        ),
        repo_root=tmp_path,
        docs_dir=docs,
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "LIVE pyAmpliCol profiling campaign" in output
    assert "gravity-cell" in output
    assert {path: path.read_bytes() for path in lease_files} == before


def test_live_snapshot_rejects_stale_malformed_and_ambiguous_leases(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    now = 70_000.0
    worker = {"cell": {"cell_id": "cell", "status": "running"}}
    _write_lease(
        service,
        instance_id="stale",
        source_revision="a" * 40,
        updated_at=now - 20.0,
        workers=worker,
    )
    _write_lease(
        service,
        instance_id="wrong-schema",
        source_revision="a" * 40,
        updated_at=now,
        workers=worker,
        schema="unknown-state-v0",
    )
    with pytest.raises(ManualCampaignError, match="no active manual-campaign lease"):
        _live_dashboard_snapshot(
            service.paths.coordination_root,
            instance=None,
            stale_after_seconds=15.0,
            now=now,
        )

    for suffix in ("one", "two"):
        _write_lease(
            service,
            instance_id=f"shared-{suffix}",
            source_revision="a" * 40,
            updated_at=now,
            workers=worker,
        )
    with pytest.raises(ManualCampaignError, match="ambiguous"):
        _live_dashboard_snapshot(
            service.paths.coordination_root,
            instance="shared-",
            stale_after_seconds=15.0,
            now=now,
        )
    with pytest.raises(ManualCampaignError, match="positive finite"):
        _live_dashboard_snapshot(
            service.paths.coordination_root,
            instance=None,
            stale_after_seconds=0.0,
            now=now,
        )


def test_dashboard_snapshot_help_explains_synthetic_and_live_modes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(("dashboard-snapshot", "--help"))
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for fragment in (
        "deterministic synthetic fixture",
        "--live",
        "--instance",
        "unique prefix",
        "--stale-after",
        "does not inspect results, artifacts, source identity, or Git",
        "never guessed or double-counted",
    ):
        assert fragment in output


def test_dashboard_snapshot_default_does_not_read_live_leases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> DashboardState:
        raise AssertionError("deterministic snapshot read a live lease")

    monkeypatch.setattr(manual_campaign, "_live_dashboard_snapshot", forbidden)
    result = campaign_main(
        ("dashboard-snapshot", "--width", "80", "--height", "24"),
        repo_root=ROOT,
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "pyAmpliCol profiling campaign" in output
    assert "LIVE Manual" not in output


def test_installed_run_disables_dashboard_when_optional_bindings_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        manual_campaign,
        "_missing_ratatui_bindings",
        lambda: ("ratatui", "ratatui_py"),
    )
    arguments = _parse("run", "--dry-run")

    changed = manual_campaign._configure_dashboard_capability(
        arguments,
        installed=True,
    )

    assert changed is True
    assert arguments.no_dashboard is True


def test_installed_campaign_uses_copied_local_amplicol_and_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "campaign"
    docs_dir.mkdir()
    configured = (tmp_path / "configured-amplicol").resolve()
    override = (tmp_path / "override-amplicol").resolve()
    (docs_dir / ".pyamplicol-original-amplicol").write_text(
        f"{configured}\n",
        encoding="utf-8",
    )
    observed: list[Path] = []

    def validate(path: Path) -> tuple[Path, str]:
        observed.append(path)
        return path, "a" * 40

    monkeypatch.setattr(
        manual_campaign,
        "_validated_original_amplicol_checkout",
        validate,
    )

    legacy = next(
        candidate
        for candidate in REPORT_CATALOG.measurement_cells()
        if candidate.measurement.execution_mode is ExecutionMode.AMPLICOL
    )
    planned = (manual_campaign.PlannedCell(legacy, False, None, 0),)
    arguments = _parse("run")
    manual_campaign._bind_original_amplicol_if_required(
        arguments,
        planned,
        installed=True,
        root=tmp_path,
        docs_dir=docs_dir,
    )
    assert arguments.original_amplicol == configured
    assert arguments.original_amplicol_revision == "a" * 40

    arguments.original_amplicol = override
    manual_campaign._bind_original_amplicol_if_required(
        arguments,
        planned,
        installed=True,
        root=tmp_path,
        docs_dir=docs_dir,
    )
    assert arguments.original_amplicol == override
    assert arguments.original_amplicol_revision == "a" * 40
    assert observed == [configured, override]


@pytest.mark.parametrize(
    "cell_id",
    (
        "matrix-recurrence-builtin-sm-lc-n4-dd-ttzh-jets-all-flow",
        "matrix-compiled-builtin-sm-lc-n4-dd-ttzh-jets-selected-flow",
    ),
)
def test_pyamplicol_only_plan_and_binding_need_no_original_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_id: str,
) -> None:
    docs_dir = tmp_path / "campaign"
    docs_dir.mkdir()
    service = _manual_service(tmp_path)
    selected = REPORT_CATALOG.cell(cell_id)
    planned = manual_campaign.plan_campaign(
        (selected,),
        store=service.store,
        settings=manual_campaign.CampaignSettings(),
    )

    assert planned
    assert all(
        item.cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
        for item in planned
    )
    assert any(
        item.optional_baseline_cell_id is not None
        or item.optional_comparison_peer_ids
        for item in planned
    )
    monkeypatch.setattr(
        manual_campaign,
        "_validated_original_amplicol_checkout",
        lambda _path: pytest.fail("pyAmpliCol-only plan validated legacy checkout"),
    )
    arguments = _parse(
        "run",
        "--generation-engine",
        selected.measurement.execution_mode.value,
    )

    manual_campaign._bind_original_amplicol_if_required(
        arguments,
        planned,
        installed=True,
        root=tmp_path,
        docs_dir=docs_dir,
    )

    assert arguments.original_amplicol is None
    assert arguments.original_amplicol_revision is None


def test_direct_or_wildcard_amplicol_selection_still_requires_checkout(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "campaign"
    docs_dir.mkdir()
    legacy = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and REPORT_CATALOG.static_na_reason(cell) is None
    )
    planned = (manual_campaign.PlannedCell(legacy, False, None, 0),)

    with pytest.raises(
        ManualCampaignError,
        match="selection requires original AmpliCol",
    ):
        manual_campaign._bind_original_amplicol_if_required(
            _parse("run"),
            planned,
            installed=True,
            root=tmp_path,
            docs_dir=docs_dir,
        )
    _selection, wildcard_cells = selection_from_arguments(_parse("run", "--dry-run"))
    assert any(
        cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        for cell in wildcard_cells
    )


def test_saved_local_amplicol_is_not_validated_for_pyamplicol_only_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "campaign"
    docs_dir.mkdir()
    missing = (tmp_path / "missing-amplicol").resolve()
    (docs_dir / ".pyamplicol-original-amplicol").write_text(
        f"{missing}\n",
        encoding="utf-8",
    )
    cell = next(
        candidate
        for candidate in REPORT_CATALOG.measurement_cells()
        if candidate.measurement.execution_mode is ExecutionMode.RECURRENCE
    )
    planned = (manual_campaign.PlannedCell(cell, False, None, 1),)

    monkeypatch.setattr(
        manual_campaign,
        "_validated_original_amplicol_checkout",
        lambda _path: pytest.fail("unused legacy checkout was validated"),
    )
    arguments = _parse("run")

    manual_campaign._bind_original_amplicol_if_required(
        arguments,
        planned,
        installed=True,
        root=tmp_path,
        docs_dir=docs_dir,
    )

    assert arguments.original_amplicol is None
    assert arguments.original_amplicol_revision is None


def test_source_run_does_not_probe_or_change_dashboard_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> tuple[str, ...]:
        raise AssertionError("source mode probed installed-wheel bindings")

    monkeypatch.setattr(manual_campaign, "_missing_ratatui_bindings", forbidden)
    arguments = _parse("run", "--dry-run")

    changed = manual_campaign._configure_dashboard_capability(
        arguments,
        installed=False,
    )

    assert changed is False
    assert arguments.no_dashboard is False


def test_installed_dashboard_snapshot_explains_optional_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        manual_campaign,
        "_missing_ratatui_bindings",
        lambda: ("ratatui_py",),
    )
    (tmp_path / "campaign").mkdir()

    result = campaign_main(
        ("dashboard-snapshot", "--no-color"),
        repo_root=tmp_path,
        docs_dir=tmp_path / "campaign",
        installed=True,
    )

    assert result == 2
    error = capsys.readouterr().err
    assert "contributor feature" in error
    assert "ratatui_py" in error
    assert "run --no-dashboard" in error
    assert "just dev-install" in error


def test_finished_reuse_stays_recycled_while_work_and_caps_complete(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="local",
        selected_ids=("reused", "skipped", "fresh", "capped"),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(service, state)

    for cell_id, status in (
        ("reused", "reused"),
        ("skipped", "skipped-current"),
        ("fresh", "ok"),
        ("capped", "generation_limit"),
    ):
        lease.observe(
            {
                "event": "finished",
                "cell_id": cell_id,
                "status": status,
                "detail": status,
            }
        )

    assert state.completed_ids == {"fresh", "capped"}
    assert state.capped_ids == {"capped"}
    assert state.workers["reused"].status == "recycled"
    assert state.workers["reused"].recycled
    assert state.workers["skipped"].status == "recycled"
    assert state.workers["skipped"].recycled
    assert state.counters() == {
        "selected": 4,
        "recycled": 2,
        "active": 0,
        "completed": 2,
        "remaining": 0,
        "static_na": 0,
        "capped": 1,
        "failed": 0,
        "unverified": 0,
        "dependency_only": 0,
    }


def test_active_counter_includes_dependency_work_without_consuming_remaining() -> None:
    state = DashboardState(
        instance_id="dependency-active",
        selected_ids=("direct-cell",),
        recycled_ids=set(),
        static_na_ids=set(),
        dependency_ids={"dependency-cell"},
        workers={
            "dependency-cell": WorkerView(
                "dependency-cell",
                dependency=True,
                status="running",
            )
        },
    )

    counters = state.counters()

    assert counters["active"] == 1
    assert counters["remaining"] == 1
    assert counters["dependency_only"] == 1
    assert "Active 1" in render_dashboard_frame(state, width=120, height=36)


def test_worker_status_counters_include_dependency_only_rows_and_lease(
    tmp_path: Path,
) -> None:
    dependency_ids = {
        "dependency-ok",
        "dependency-cap",
        "dependency-error",
        "dependency-unverified",
        "dependency-running",
    }
    state = DashboardState(
        instance_id="dependency-status-counters",
        selected_ids=("direct-cell",),
        recycled_ids=set(),
        static_na_ids=set(),
        dependency_ids=dependency_ids,
        source_revision="a" * 40,
    )
    lease = LeaseManager(_manual_service(tmp_path), state)
    for cell_id, status in (
        ("dependency-ok", "ok"),
        ("dependency-cap", "memory_limit"),
        ("dependency-error", "error"),
        ("dependency-unverified", ResultStatus.UNVERIFIED.value),
    ):
        lease.observe({"event": "started", "cell_id": cell_id, "dependency": True})
        lease.observe(
            {
                "event": "finished",
                "cell_id": cell_id,
                "status": status,
                "detail": status,
            }
        )
    lease.observe(
        {
            "event": "started",
            "cell_id": "dependency-running",
            "dependency": True,
        }
    )

    counters = state.counters()

    assert counters == {
        "selected": 1,
        "recycled": 0,
        "active": 1,
        "completed": 2,
        "remaining": 1,
        "static_na": 0,
        "capped": 1,
        "failed": 1,
        "unverified": 1,
        "dependency_only": 5,
    }
    frame = render_dashboard_frame(state, width=160, height=40)
    assert "Active 1" in frame
    assert "Errors 1" in frame
    assert "Unverified 1" in frame
    assert "Completed 2" in frame
    assert "Capped 1" in frame
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["counters"] == counters


def test_dashboard_reducer_replaces_prior_outcomes_across_retries(
    tmp_path: Path,
) -> None:
    state = DashboardState(
        instance_id="retry-transitions",
        selected_ids=("retry-error", "retry-cap", "retry-recycled"),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(_manual_service(tmp_path), state)

    lease.observe(
        {
            "event": "finished",
            "cell_id": "retry-error",
            "status": "error",
        }
    )
    assert state.counters()["failed"] == 1
    assert state.counters()["remaining"] == 2
    lease.observe({"event": "started", "cell_id": "retry-error"})
    assert state.failed_ids == set()
    assert state.counters()["active"] == 1
    lease.observe({"event": "finished", "cell_id": "retry-error", "status": "ok"})
    assert state.completed_ids == {"retry-error"}
    assert state.failed_ids == set()

    lease.observe(
        {
            "event": "finished",
            "cell_id": "retry-cap",
            "status": "memory_limit",
        }
    )
    assert state.capped_ids == {"retry-cap"}
    lease.observe({"event": "started", "cell_id": "retry-cap"})
    lease.observe(
        {
            "event": "finished",
            "cell_id": "retry-cap",
            "status": ResultStatus.UNVERIFIED.value,
        }
    )
    assert state.capped_ids == set()
    assert state.completed_ids == {"retry-error"}
    assert state.unverified_ids == {"retry-cap"}

    lease.observe(
        {
            "event": "finished",
            "cell_id": "retry-recycled",
            "status": "reused",
        }
    )
    assert state.recycled_ids == {"retry-recycled"}
    lease.observe({"event": "started", "cell_id": "retry-recycled"})
    lease.observe(
        {
            "event": "finished",
            "cell_id": "retry-recycled",
            "status": "error",
        }
    )
    assert state.recycled_ids == set()
    assert state.failed_ids == {"retry-recycled"}
    assert state.counters() == {
        "selected": 3,
        "recycled": 0,
        "active": 0,
        "completed": 1,
        "remaining": 0,
        "static_na": 0,
        "capped": 0,
        "failed": 1,
        "unverified": 1,
        "dependency_only": 0,
    }
    payload = json.loads(lease.path.read_text(encoding="utf-8"))
    assert payload["counters"] == state.counters()


def test_cancelled_cell_remains_pending_in_dashboard_progress(tmp_path: Path) -> None:
    state = DashboardState(
        instance_id="cancelled-transition",
        selected_ids=("cancelled-cell",),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(_manual_service(tmp_path), state)
    lease.observe({"event": "started", "cell_id": "cancelled-cell"})
    lease.observe(
        {
            "event": "finished",
            "cell_id": "cancelled-cell",
            "status": "cancelled",
        }
    )

    assert state.terminal_outcomes == {}
    assert state.counters()["active"] == 0
    assert state.counters()["remaining"] == 1


def test_late_resource_event_cannot_reactivate_terminal_worker(tmp_path: Path) -> None:
    state = DashboardState(
        instance_id="late-resource",
        selected_ids=("cell",),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    lease = LeaseManager(_manual_service(tmp_path), state)
    lease.observe({"event": "started", "cell_id": "cell"})
    lease.observe(
        {
            "event": "worker",
            "cell_id": "cell",
            "attempt_id": "new-attempt",
        }
    )
    lease.observe(
        {
            "event": "resource",
            "cell_id": "cell",
            "attempt_id": "old-attempt",
            "wall_seconds": 50.0,
        }
    )
    assert state.workers["cell"].wall_seconds == 0.0
    lease.observe(
        {
            "event": "resource",
            "cell_id": "cell",
            "attempt_id": "new-attempt",
            "wall_seconds": 2.0,
        }
    )
    lease.observe({"event": "finished", "cell_id": "cell", "status": "ok"})
    lease.observe(
        {
            "event": "resource",
            "cell_id": "cell",
            "attempt_id": "new-attempt",
            "wall_seconds": 99.0,
        }
    )

    assert state.workers["cell"].status == "ok"
    assert state.workers["cell"].wall_seconds == 2.0
    assert state.counters()["active"] == 0
    assert state.counters()["completed"] == 1


@pytest.mark.parametrize(
    "status",
    ("worker_timeout", "profiling_timeout", "validation_timeout"),
)
def test_manual_stage_timeouts_are_addressed_caps_in_dashboard_state(
    tmp_path: Path,
    status: str,
) -> None:
    service = _manual_service(tmp_path)
    state = DashboardState(
        instance_id="local",
        selected_ids=(status,),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision="a" * 40,
    )
    LeaseManager(service, state).observe(
        {
            "event": "finished",
            "cell_id": status,
            "status": status,
            "detail": status,
        }
    )

    assert state.completed_ids == {status}
    assert state.capped_ids == {status}
    assert state.failed_ids == set()
    assert state.counters()["remaining"] == 0


def test_n8_all_flow_recurrence_is_runnable_but_native_n7_stays_static_na() -> None:
    recurrence = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.workload is Workload.ALL_FLOW
        and cell.n_final == 8
    )
    assert _manual_static_na_reason(recurrence) is None

    native = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id.startswith("z_")
        and cell.variant in {"cpp_o3", "asm_o3"}
        and cell.n_final == 7
    )
    assert _manual_static_na_reason(native) == ("native-backend-generation-cap-n6-v1")


def _write_minimal_index(repo: Path, relative: str) -> None:
    path = repo / relative
    observed = path.stat()
    encoded = relative.encode()
    fields = struct.pack(
        "!10I20sH",
        int(observed.st_ctime_ns // 1_000_000_000),
        int(observed.st_ctime_ns % 1_000_000_000),
        int(observed.st_mtime_ns // 1_000_000_000),
        int(observed.st_mtime_ns % 1_000_000_000),
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_uid),
        int(observed.st_gid),
        int(observed.st_size),
        b"\0" * 20,
        len(encoded),
    )
    entry = fields + encoded + b"\0"
    entry += b"\0" * ((8 - len(entry) % 8) % 8)
    (repo / ".git/index").write_bytes(struct.pack("!4sII", b"DIRC", 2, 1) + entry)


def test_source_cleanliness_is_index_metadata_only(tmp_path: Path) -> None:
    (tmp_path / ".git/objects").mkdir(parents=True)
    source = tmp_path / "tracked.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _write_minimal_index(tmp_path, "tracked.py")
    assert _index_metadata_dirty_paths(tmp_path) == ()
    source.write_text("value = 200\n", encoding="utf-8")
    assert _index_metadata_dirty_paths(tmp_path) == ("tracked.py",)


def test_repo_head_resolves_loose_branch_from_linked_worktree_common_dir(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "linked"
    common = tmp_path / "common.git"
    worktree_git = common / "worktrees/linked"
    reference = "refs/heads/codex/manual-campaign-smoke"
    revision = "a" * 40
    repo.mkdir()
    worktree_git.mkdir(parents=True)
    (repo / ".git").write_text(
        f"gitdir: {worktree_git}\n",
        encoding="ascii",
    )
    (worktree_git / "HEAD").write_text(
        f"ref: {reference}\n",
        encoding="ascii",
    )
    (worktree_git / "commondir").write_text("../..\n", encoding="ascii")
    loose = common / reference
    loose.parent.mkdir(parents=True)
    loose.write_text(f"{revision}\n", encoding="ascii")

    assert manual_campaign._repo_head(repo) == revision


def test_lightweight_result_identity_must_match_selected_cell() -> None:
    cell = REPORT_CATALOG.measurement_cells()[0]
    identity = {
        "cell_id": cell.cell_id,
        "dataset_id": cell.dataset_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "workload": cell.workload.value,
        "execution_mode": cell.measurement.execution_mode.value,
        "model": None,
        "accuracy": cell.measurement.accuracy.value,
        "backend": cell.measurement.backend,
        "variant": cell.variant,
    }
    result = {"provenance": {"manual_campaign": {"cell_identity": identity}}}
    assert _result_matches_cell(result, cell)
    identity["cell_id"] = "wrong-cell"
    assert not _result_matches_cell(result, cell)


def test_manual_refresh_projects_artifact_output_and_reproduction_commands(
    tmp_path: Path,
) -> None:
    service = _manual_service(tmp_path)
    cell = next(
        cell
        for cell in service.catalog.measurement_cells()
        if (
            cell.measurement.execution_mode is ExecutionMode.RECURRENCE
            and cell.measurement.model is ModelKey.UFO_SM
            and service.catalog.static_na_reason(cell) is None
        )
    )
    artifact = (
        service.paths.artifact_root
        / "cells"
        / cell.cell_id
        / "attempts"
        / "attempt-one"
        / "artifact"
    )
    reproduction = reproduction_recipe(
        cell,
        repo_root=service.paths.repo_root,
        artifact_root=service.paths.artifact_root,
        artifact_path=str(artifact),
    ).as_dict()
    assert all(
        isinstance(reproduction[field], list)
        for field in ("prepare", "generate", "profile")
    )

    def merged_cache(
        output: Path,
        recipe: dict[str, object] = reproduction,
    ) -> dict[str, dict[str, object]]:
        measurement = empty_measurement()
        measurement.update(
            {
                "status": ResultStatus.TIMEOUT.value,
                "failure": {"kind": "generation_timeout"},
                "provenance": {
                    "effective_config": {
                        "generation": {"output": str(output)},
                    },
                    "manual_campaign": {
                        "public_cli_reproduction": recipe,
                    },
                },
            }
        )
        current = LightweightCurrent(
            cell_id=cell.cell_id,
            attempt_id="attempt-one",
            result_path=output.parent / "result.json",
            result=measurement,
            complete=True,
            reusable=True,
            reason="reusable",
        )
        caches, merged = _merge_lightweight_snapshot(
            service,
            {cell.cell_id: current},
        )
        assert merged == 1
        return caches

    caches = merged_cache(artifact)
    portable = portable_publication_value(
        caches[f"{cell.dataset_id}.json"],
        service.paths,
    )
    assert isinstance(portable, dict)
    entries = portable["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if isinstance(item, dict) and item.get("cell_id") == cell.cell_id
    )
    output = entry["measurement"]["provenance"]["effective_config"][  # type: ignore[index]
        "generation"
    ]["output"]
    assert output == (
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
        f"cells/{cell.cell_id}/attempts/attempt-one/artifact"
    )
    provenance = entry["measurement"]["provenance"]  # type: ignore[index]
    published_recipe = provenance["manual_campaign"][  # type: ignore[index]
        "public_cli_reproduction"
    ]
    for field in ("prepare", "generate", "profile"):
        argv = published_recipe[field]  # type: ignore[index]
        assert isinstance(argv, list)
        assert any("${PYAMPLICOL_SOURCE_ROOT}" in argument for argument in argv)
    rendered = json.dumps(portable)
    assert str(service.paths.artifact_root) not in rendered
    assert str(service.paths.repo_root) not in rendered
    assert portable_publication_value(portable, service.paths) == portable

    external_recipe = dict(reproduction)
    external_trace = (
        service.paths.repo_root.parent / "pyamplicol-external-test" / "trace.json"
    )
    external_profile = external_recipe["profile"]
    assert isinstance(external_profile, list)
    external_recipe["profile"] = [*external_profile, "--trace", str(external_trace)]
    external_diagnostic = merged_cache(artifact, external_recipe)
    redacted = portable_publication_value(
        external_diagnostic[f"{cell.dataset_id}.json"],
        service.paths,
    )
    redacted_entry = next(
        item
        for item in redacted["entries"]  # type: ignore[index]
        if isinstance(item, dict) and item.get("cell_id") == cell.cell_id
    )
    redacted_recipe = redacted_entry["measurement"]["provenance"][  # type: ignore[index]
        "manual_campaign"
    ]["public_cli_reproduction"]
    assert redacted_recipe["profile"][-1] == "${LOCAL_PATH_REDACTED}"
    assert str(tmp_path) not in json.dumps(redacted)
    assert portable_publication_value(redacted, service.paths) == redacted

    with pytest.raises(
        PublicationPortabilityError,
        match=r"/measurement/provenance/effective_config/generation/output",
    ):
        external = merged_cache(tmp_path / "external" / "artifact")
        portable_publication_value(
            external[f"{cell.dataset_id}.json"],
            service.paths,
        )


def test_copied_campaign_rebinds_private_pdf_build_profile(tmp_path: Path) -> None:
    staging = tmp_path / "copied_campaign"
    staging.mkdir()
    (staging / "report_environment.json").write_text(
        json.dumps({"profile": "macbook_M3_manual"}),
        encoding="utf-8",
    )
    (staging / "report-workspace.json").write_text(
        json.dumps(
            {
                "profile": "macbook_M3_manual",
                "artifact_root": ".artifacts/performance-report/macbook_M3_manual",
                "coordination_root": (
                    ".artifacts/performance-report-coordination/macbook_M3_manual"
                ),
                "initialized_environment": {"profile": "macbook_M3_manual"},
            }
        ),
        encoding="utf-8",
    )
    (staging / "report_environment.tex").write_text(
        "\\renewcommand{\\ReportProfileName}{macbook\\_M3\\_manual}\n",
        encoding="utf-8",
    )

    manual_campaign._stage_copied_profile_identity(staging, "independent_run")

    environment = json.loads(
        (staging / "report_environment.json").read_text(encoding="utf-8")
    )
    workspace = json.loads(
        (staging / "report-workspace.json").read_text(encoding="utf-8")
    )
    assert environment["profile"] == "independent_run"
    assert workspace["profile"] == "independent_run"
    assert workspace["initialized_environment"]["profile"] == "independent_run"
    assert workspace["artifact_root"] == "campaign_artifacts"
    assert workspace["coordination_root"] == "campaign_artifacts/coordination"
    assert "{independent\\_run}" in (staging / "report_environment.tex").read_text(
        encoding="utf-8"
    )


def test_manual_refresh_source_copy_excludes_only_top_level_campaign_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyAmpliCol.tex").write_text("report\n", encoding="ascii")
    state = source / "campaign_artifacts"
    state.mkdir()
    (state / "large-sentinel.bin").write_bytes(b"private")
    nested = source / "appendix/campaign_artifacts"
    nested.mkdir(parents=True)
    (nested / "published.txt").write_text("keep\n", encoding="ascii")
    destination = tmp_path / "staging"

    manual_campaign._copy_report_sources(source, destination)

    assert not (destination / "campaign_artifacts").exists()
    assert (destination / "appendix/campaign_artifacts/published.txt").is_file()


def test_refresh_pdf_does_not_stage_local_campaign_state_recursively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "portable-campaign"
    docs.mkdir()
    (docs / "pyAmpliCol.tex").write_text("report\n", encoding="ascii")
    service = ReportService(
        manual_campaign._campaign_report_paths(ROOT, docs),
        portable_current_results=True,
    )
    sentinel = service.paths.artifact_root / "large-private-sentinel.bin"
    sentinel.write_bytes(b"private-state" * 1024)
    manual_campaign.update_source_marker(
        service,
        manual_campaign.ReportSourceIdentity("a" * 40, "b" * 40, ()),
    )
    observed: dict[str, Path] = {}

    monkeypatch.setattr(
        manual_campaign,
        "_capture_lightweight_snapshot",
        lambda *_args, **_kwargs: ({}, {}, ()),
    )
    monkeypatch.setattr(
        manual_campaign,
        "_merge_lightweight_snapshot",
        lambda *_args, **_kwargs: ({}, 0),
    )
    monkeypatch.setattr(ReportService, "_render_tables", lambda *_args: {})
    monkeypatch.setattr(ReportService, "_snapshot_files", lambda *_args: ())

    def compile_staged(staging_docs: Path, **_kwargs: object) -> int:
        observed["staging_docs"] = staging_docs
        assert not (staging_docs / "campaign_artifacts").exists()
        assert not any(
            path.name == sentinel.name for path in staging_docs.rglob("*")
        )
        (staging_docs / "pyAmpliCol.pdf").write_bytes(b"%PDF-1.4\n")
        return 1

    monkeypatch.setattr(manual_campaign, "_compile_pdf", compile_staged)
    monkeypatch.setattr(
        manual_campaign,
        "_install_report_snapshot",
        lambda *_args, **_kwargs: None,
    )

    assert (
        manual_campaign._refresh_pdf(
            _parse("refresh-pdf", "--quiet", "--expected-page-count", "1"),
            service=service,
            source=manual_campaign.ReportSourceIdentity("a" * 40, "b" * 40, ()),
            palette=manual_campaign.Palette(False),
        )
        == 0
    )
    assert "staging_docs" in observed
    assert not observed["staging_docs"].exists()
    assert sentinel.read_bytes() == b"private-state" * 1024
    build_root = service.paths.artifact_root / "manual-publication-builds"
    assert build_root.is_dir()
    assert not tuple(build_root.iterdir())


def test_controller_sources_only_use_read_only_git_for_external_checkout() -> None:
    source_paths = (
        PROFILE / "steer_performance_campaign.py",
        ROOT / "tools/performance_report/manual_campaign.py",
    )
    sources = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    for forbidden in (
        "subprocess.Popen(",
        "os.system(",
        "shell=True",
        "pip install",
        "git pull",
        "git fetch",
    ):
        assert forbidden not in sources
    commands: list[tuple[str | None, ...]] = []
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
            ):
                continue
            assert node.args and isinstance(node.args[0], (ast.Tuple, ast.List))
            command = tuple(
                element.value if isinstance(element, ast.Constant) else None
                for element in node.args[0].elts
            )
            assert command[:2] in {("git", "rev-parse"), ("git", "status")}
            assert not {
                "add",
                "apply",
                "checkout",
                "clean",
                "commit",
                "fetch",
                "merge",
                "pull",
                "push",
                "reset",
                "stash",
            }.intersection(value for value in command if value is not None)
            commands.append(command)
    assert ("git", "rev-parse", "HEAD") in commands
    assert (
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ) in commands
    assert any(command[:3] == ("git", "rev-parse", "--verify") for command in commands)
    assert os.access(PROFILE / "steer_performance_campaign.py", os.X_OK)


def test_report_snapshot_install_is_atomic_and_rolls_back_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = tmp_path / "docs"
    staging = tmp_path / "staging"
    for root, prefix in ((docs, "old"), (staging, "new")):
        (root / "results").mkdir(parents=True)
        (root / "results/cache.json").write_text(
            f"{prefix}-cache\n",
            encoding="ascii",
        )
        (root / "result_demo_table.tex").write_text(
            f"{prefix}-table\n",
            encoding="utf-8",
        )
        (root / "pyAmpliCol.pdf").write_bytes(f"{prefix}-pdf".encode("ascii"))
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            docs_dir=docs,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    real_replace = os.replace

    def fail_table_install(
        source: os.PathLike[str],
        destination: os.PathLike[str],
    ) -> None:
        if (
            Path(destination).name == "result_demo_table.tex"
            and ".manual-" in Path(source).name
        ):
            raise OSError("injected table install interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_table_install)
    with pytest.raises(OSError, match="injected table install"):
        _install_report_snapshot(
            service,
            staging,
            ("result_demo_table.tex",),
        )

    assert (docs / "results/cache.json").read_text(encoding="ascii") == "old-cache\n"
    assert (docs / "result_demo_table.tex").read_text(encoding="utf-8") == "old-table\n"
    assert (docs / "pyAmpliCol.pdf").read_bytes() == b"old-pdf"


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is unavailable")
def test_fresh_manual_report_compiles_with_the_manual_page_contract(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    shutil.copytree(
        PROFILE,
        docs,
        ignore=shutil.ignore_patterns("pyAmpliCol.pdf", "*.aux", "*.log"),
    )
    pages = _compile_pdf(
        docs,
        expected_page_count=DEFAULT_MANUAL_EXPECTED_PAGE_COUNT,
        timeout_seconds=900.0,
    )
    assert pages == 59
    assert (docs / "pyAmpliCol.pdf").stat().st_size > 100_000
