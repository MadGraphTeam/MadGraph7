# SPDX-License-Identifier: 0BSD
from __future__ import annotations

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
from tools.performance_report.service import ReportPaths, ReportService

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "docs/performance_reports/macbook_M3_manual"


def _parse(*arguments: str):
    return build_parser().parse_args(arguments)


def test_catalog_and_fresh_profile_are_complete_but_measurement_empty() -> None:
    assert len(REPORT_CATALOG.measurement_cells()) == 1706
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
    assert len(all_cells) == 1706


def test_matrix_best_is_all_three_builtin_candidate_modes() -> None:
    arguments = _parse("inspect", "--table", "matrix_best")
    _selection, cells = selection_from_arguments(arguments)
    assert len(cells) == 888
    assert {cell.measurement.execution_mode for cell in cells} == {
        ExecutionMode.RECURRENCE,
        ExecutionMode.COMPILED,
        ExecutionMode.EAGER,
    }
    assert {cell.measurement.model for cell in cells} == {ModelKey.BUILTIN_SM}


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
        "bounded spooled/compressed numerical-current evidence",
        "C++/ASM variants above",
        "Keyboard controls",
        "current/contribution counts",
        "process-tree current/peak usage",
    ):
        assert fragment in help_text
    arguments = _parse("run", "--dry-run")
    assert arguments.workers == 1
    assert arguments.cores_per_worker == 1
    assert arguments.generation_time_limit == 3600.0
    assert arguments.ram_limit == 30_000_000_000
    assert arguments.worker_wall_limit is None
    assert arguments.no_color is False
    assert arguments.force_refresh is False
    refresh = _parse("refresh-pdf")
    assert refresh.expected_page_count is None
    assert refresh.quiet is False
    assert _parse("refresh-pdf", "--quiet").quiet is True
    assert DEFAULT_MANUAL_EXPECTED_PAGE_COUNT == 60
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
    recipe = reproduction_recipe(candidate, repo_root=ROOT)
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
    legacy_recipe = reproduction_recipe(legacy, repo_root=ROOT)
    assert legacy_recipe.kind == "legacy-report-adapter"
    assert legacy_recipe.generate is None
    assert legacy_recipe.profile is None
    assert legacy_recipe.exact is False


def test_reproduction_cli_prefix_runs_from_an_unrelated_directory(
    tmp_path: Path,
) -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
    )
    recipe = reproduction_recipe(candidate, repo_root=ROOT)
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
    recipe = reproduction_recipe(candidate, repo_root=tmp_path)

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
    recipe = reproduction_recipe(candidate, repo_root=ROOT)

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
    assert "Manual MacBook M3 campaign" in frame
    assert "Selected" in frame
    assert "Recycled" in frame
    assert "Remaining" in frame
    assert "RAM" in frame
    assert "Caps Generation 01:00:00" in frame
    assert "RAM 30.00 GB" in frame
    assert "Total wall disabled" in frame
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
        completed_ids={worker.cell_id},
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
        completed_ids={"done-ok", "done-reused", "recycled-cap"},
        capped_ids={"recycled-cap"},
        failed_ids={"attention-error"},
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
        "recycled-cap": WorkerView("recycled-cap", status="memory_limit"),
    }
    state = DashboardState(
        instance_id="error-counter",
        selected_ids=tuple(workers),
        recycled_ids={"recycled-cap"},
        static_na_ids=set(),
        workers=workers,
        completed_ids={"recycled-cap"},
        capped_ids={"recycled-cap"},
        failed_ids={"error"},
    )

    frame = render_dashboard_frame(state, width=120, height=36)
    summary = next(line for line in frame.splitlines() if "Selected " in line)
    assert "Errors 1" in summary
    assert "Active 1" in summary
    assert "Capped 1" in frame


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


def _manual_service(tmp_path: Path) -> ReportService:
    return ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
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
    service = _manual_service(tmp_path)
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

    observed: list[str] = []

    def stop_after_source_selection(
        _service: ReportService,
        *,
        source_revision: str,
    ) -> object:
        observed.append(source_revision)
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
    assert observed == [measurement_revision]


def test_malformed_measurement_source_marker_is_not_guessed(tmp_path: Path) -> None:
    service = _manual_service(tmp_path)
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
    assert "LIVE Manual MacBook M3 campaign" in frame
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
    service = ReportService(
        ReportPaths.from_repo(tmp_path, profile="macbook_M3_manual")
    )
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
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "LIVE Manual MacBook M3 campaign" in output
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
    assert "Manual MacBook M3 campaign" in output
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

    arguments = _parse("run", "--dry-run")
    assert manual_campaign._resolve_original_amplicol(
        arguments,
        installed=True,
        root=tmp_path,
        docs_dir=docs_dir,
    ) == (configured, "a" * 40)

    arguments.original_amplicol = override
    assert manual_campaign._resolve_original_amplicol(
        arguments,
        installed=True,
        root=tmp_path,
        docs_dir=docs_dir,
    ) == (override, "a" * 40)
    assert observed == [configured, override]


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
        "dependency_only": 0,
    }


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
        artifact_path=str(artifact),
    ).as_dict()
    assert all(
        isinstance(reproduction[field], str)
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
        command = published_recipe[field]  # type: ignore[index]
        assert isinstance(command, str)
        assert "${PYAMPLICOL_SOURCE_ROOT}" in command
    rendered = json.dumps(portable)
    assert str(service.paths.artifact_root) not in rendered
    assert str(service.paths.repo_root) not in rendered
    assert portable_publication_value(portable, service.paths) == portable

    external_recipe = dict(reproduction)
    external_trace = (
        service.paths.repo_root.parent / "pyamplicol-external-test" / "trace.json"
    )
    external_recipe["profile"] = (
        f"{external_recipe['profile']} --trace {external_trace}"
    )
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
    assert redacted_recipe["profile"] == "${LOCAL_PATH_REDACTED}"
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
    assert workspace["artifact_root"].endswith("/independent_run")
    assert workspace["coordination_root"].endswith("/independent_run")
    assert "{independent\\_run}" in (staging / "report_environment.tex").read_text(
        encoding="utf-8"
    )


def test_executable_reexecutes_from_base_python_into_repository_venv() -> None:
    base_python = getattr(sys, "_base_executable", sys.executable)
    completed = subprocess.run(
        (
            base_python,
            str(PROFILE / "steer_performance_campaign.py"),
            "dashboard-snapshot",
            "--width",
            "80",
            "--height",
            "24",
        ),
        cwd=PROFILE,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Manual MacBook M3 campaign" in completed.stdout


def test_controller_sources_only_use_read_only_git_for_external_checkout() -> None:
    sources = (PROFILE / "steer_performance_campaign.py").read_text(
        encoding="utf-8"
    ) + (ROOT / "tools/performance_report/manual_campaign.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "subprocess.Popen(",
        "os.system(",
        "shell=True",
        "pip install",
        "git pull",
        "git fetch",
    ):
        assert forbidden not in sources
    assert sources.count("subprocess.run(") == 2
    assert '("git", "rev-parse", "HEAD")' in sources
    assert '("git", "status", "--porcelain=v1", "--untracked-files=all")' in sources
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
    assert pages == 60
    assert (docs / "pyAmpliCol.pdf").stat().st_size > 100_000
