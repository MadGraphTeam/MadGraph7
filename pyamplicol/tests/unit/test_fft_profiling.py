# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import fft_scaling_study as study  # noqa: E402
from tools.fft_profiling import fft_profiling as profiling  # noqa: E402


def _arguments(*values: str):
    return profiling._parser().parse_args(values)


@pytest.fixture(autouse=True)
def _redirect_canonical_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        profiling, "CANONICAL_RESULTS_ROOT", tmp_path / "canonical-results"
    )


def test_dry_run_sparse_custom_caps_is_deterministic_and_write_free(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cluster run"
    arguments = _arguments(
        "--dry-run",
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--cores",
        "8",
        "--candidate-cores",
        "2",
        "--memory-limit-gib",
        "64",
        "--time-limit-seconds",
        "14400",
        "--madgraph-root",
        str(tmp_path / "mg5"),
    )
    profiling._validate_arguments(arguments)

    first = profiling.dry_run_plan(arguments)
    second = profiling.dry_run_plan(arguments)

    assert first == second
    assert not output.exists()
    assert first["identity"]["scan"]["multiplicity_universe"] == list(range(2, 10))
    assert first["batch_size"] == 128
    assert first["identity"]["scan"]["batch_size"] == 128
    assert first["requested_fill_multiplicities"] == [2, 3]
    assert first["identity"]["resources"] == {
        "candidate_optimization_cores": 2,
        "per_cell_generation_timeout_seconds": 14400.0,
        "per_cell_memory_limit_gib": 64.0,
        "per_cell_runtime_timeout_seconds": 14400.0,
    }
    assert set(first["shards"]) == {shard.name for shard in profiling.SHARDS}
    for shard in first["shards"].values():
        if shard["argv"] is not None:
            batch_index = shard["argv"].index("--batch-size")
            assert shard["argv"][batch_index + 1] == "128"
    assert first["shards"]["gg-otf"]["owned_modes"] == [
        "otf-direct",
        "otf-fft",
    ]
    mg_argv = first["madgraph"]["argv"]
    assert mg_argv is not None
    assert mg_argv.count("--multiplicity") == 2
    assert mg_argv[mg_argv.index("--memory-limit-gib") + 1] == "64"
    assert mg_argv[mg_argv.index("--generation-timeout-seconds") + 1] == "14400"


def test_helicity_sum_dry_run_generates_a_distinct_summed_madgraph_series(
    tmp_path: Path,
) -> None:
    arguments = _arguments(
        "--dry-run",
        "--compare-helicity-sums",
        "--output",
        str(tmp_path / "summed"),
        "--multiplicities",
        "2",
        "3",
        "--memory-limit-gib",
        "64",
        "--time-limit-seconds",
        "14400",
        "--madgraph-root",
        str(tmp_path / "mg5"),
    )
    profiling._validate_arguments(arguments)

    plan = profiling.dry_run_plan(arguments)

    assert plan["helicity_workload"] == "sum"
    assert plan["batch_size"] == 128
    assert plan["identity"]["scan"]["helicity_workload"] == "sum"
    assert plan["identity"]["tools"]["madgraph_root"] == str(tmp_path / "mg5")
    assert plan["madgraph"] == {
        "phase": 4,
        "applicable": True,
        "helicity_workload": "sum",
        "measurement_multiplicities": [2, 3],
        "protocol_scope_multiplicities": [],
        "dependency": "completed pyAmpliCol cells for the requested fill",
        "not_applicable_reason": None,
        "report": str(tmp_path / "summed" / "madgraph" / "overlay.json"),
        "argv": plan["madgraph"]["argv"],
        "shell_command": plan["madgraph"]["shell_command"],
    }
    assert plan["madgraph"]["argv"] is not None
    assert "--compare-helicity-sums" in plan["madgraph"]["argv"]
    assert plan["outputs"]["final_pdf"].endswith(
        "/summary_plots_final_helicity_sum.pdf"
    )
    for shard in plan["shards"].values():
        if shard["argv"] is not None:
            assert "--compare-helicity-sums" in shard["argv"]
    master = profiling._master_arguments(arguments, tmp_path / "summed")
    assert master.compare_helicity_sums is True
    assert study.dry_run_plan(master)["measurement"]["helicity_workload"] == "sum"


def test_madgraph_command_protocol_scopes_n_above_six(tmp_path: Path) -> None:
    arguments = _arguments(
        "--dry-run",
        "--output",
        str(tmp_path / "fixed"),
        "--multiplicities",
        "6",
        "7",
        "8",
        "--madgraph-root",
        str(tmp_path / "mg5"),
    )

    plan = profiling.dry_run_plan(arguments)
    argv = plan["madgraph"]["argv"]

    assert plan["madgraph"]["measurement_multiplicities"] == [6]
    assert plan["madgraph"]["protocol_scope_multiplicities"] == [7, 8]
    assert argv is not None
    assert argv[argv.index("--multiplicity") + 1] == "6"
    assert [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--protocol-scope-multiplicity"
    ] == ["7", "8"]


def test_canonical_authenticated_madgraph_overlay_has_resume_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fixed"
    overlay = tmp_path / "overlay.json"
    overlay.write_text("{}", encoding="ascii")
    arguments = _arguments("--output", str(output))
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(profiling, "_report_publication_profile", lambda _report: True)
    monkeypatch.setattr(
        profiling,
        "_matching_madgraph_overlay",
        lambda *_args, **_kwargs: overlay,
    )
    monkeypatch.setattr(
        profiling.publication,
        "build_final_report",
        lambda *, campaign_path, madgraph_overlay_path: calls.append(
            (campaign_path, madgraph_overlay_path)
        ),
    )

    assert profiling._canonical_madgraph_overlay_authenticated(
        arguments, output, {"status": "complete"}
    )
    assert calls == [(profiling._master_report_path(output), overlay)]


def test_workload_defaults_are_separate_and_explicit_output_is_exact(
    tmp_path: Path,
) -> None:
    fixed = _arguments()
    summed = _arguments("--compare-helicity-sums")
    explicit_path = tmp_path / "chosen"
    explicit = _arguments(
        "--compare-helicity-sums", "--output", str(explicit_path)
    )

    assert profiling._run_directory(fixed) == profiling.DEFAULT_FIXED_OUTPUT
    assert profiling._run_directory(summed) == profiling.DEFAULT_SUM_OUTPUT
    assert profiling._run_directory(explicit) == explicit_path
    assert fixed.batch_size == 128
    assert summed.batch_size == 128
    assert profiling._pdf_filename(fixed) == "summary_plots_final.pdf"
    assert (
        profiling._pdf_filename(summed)
        == "summary_plots_final_helicity_sum.pdf"
    )


def test_dashboard_headline_displays_workload_batch_cores_and_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headlines: list[str] = []

    class FakeBar:
        def start(self) -> None:
            pass

        def update(self, _value: int, **values: object) -> None:
            headlines.append(str(values["headline"]))

        def finish(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        profiling.progressbar,
        "ProgressBar",
        lambda **_kwargs: FakeBar(),
    )
    monkeypatch.setattr(profiling, "_aggregate_rss", lambda _active: 3 * 1024**3)

    class FakeJob:
        claimed_cores = 2
        detail = "gg/recurrence-fft/n4"

    dashboard = profiling.Dashboard(
        total=2,
        core_budget=4,
        helicity_workload="sum",
        batch_size=128,
    )

    dashboard.update(1, phase="phase 2", active=(FakeJob(),))

    assert "1/2 cells | sum batch=128" in headlines[-1]
    assert "phase 2: gg/recurrence-fft/n4" in headlines[-1]
    assert "cores active=2/4" in headlines[-1]
    assert "RSS 3.00 GiB" in headlines[-1]
    assert "elapsed " in headlines[-1]
    assert "ETA" not in headlines[-1]


def test_authoritative_study_supports_isolated_roots_and_custom_caps(
    tmp_path: Path,
) -> None:
    arguments = study._parser().parse_args(
        (
            "--study-root",
            str(tmp_path / "one"),
            "--run-id",
            "cell",
            "--multiplicity",
            "2",
            "--generation-timeout",
            "14400",
            "--runtime-timeout",
            "14400",
            "--memory-limit-gib",
            "64",
            "--optimization-cores",
            "3",
        )
    )
    study._validate_arguments(arguments)
    plan = study.dry_run_plan(arguments)
    measurement = plan["measurement"]
    assert measurement["requested_memory_ceiling_gib"] == 64.0
    assert measurement["memory_watchdog_gib"] == 64.0
    assert measurement["candidate_optimization_cores"] == 3
    assert measurement["generation_timeout_seconds"] == 14400.0

    first_root = tmp_path / "lock-one"
    second_root = tmp_path / "lock-two"
    first_root.mkdir()
    second_root.mkdir()
    first = study._acquire_campaign_lock(first_root)
    second = study._acquire_campaign_lock(second_root)
    first.close()
    second.close()


def test_authoritative_fill_selector_does_not_change_report_policy() -> None:
    base = study._parser().parse_args(("--multiplicity", "2", "--multiplicity", "3"))
    fill = study._parser().parse_args(
        (
            "--multiplicity",
            "2",
            "--multiplicity",
            "3",
            "--fill-multiplicity",
            "3",
        )
    )
    study._validate_arguments(fill)

    assert study.dry_run_plan(fill) == study.dry_run_plan(base)
    assert study._fill_multiplicities(fill) == (3,)


def test_otf_protocol_scope_keeps_fixed_n6_and_skips_n7_n8_n9() -> None:
    arguments = _arguments("--multiplicities", "6", "7", "8", "9")
    shard = profiling.SHARD_BY_NAME["gg-otf"]
    namespace = profiling._shard_arguments(arguments, Path("/tmp/otf-scope"), shard)
    report = study.compose_report(namespace, {})

    changed = study.apply_protocol_scope_cells(
        report,
        family="gg",
        modes=shard.owned_modes,
        multiplicities=(6, 7, 8, 9),
    )

    assert changed is True
    for mode in shard.owned_modes:
        assert "6" not in report["cells"]["gg"][mode]
        assert {
            report["cells"]["gg"][mode][str(n)]["status"]
            for n in (7, 8, 9)
        } == {"skipped"}
    assert not profiling._selected_shard_complete(arguments, shard, report)

    plan = profiling.dry_run_plan(arguments)
    assert plan["shards"]["gg-otf"]["measurement_multiplicities"] == [6]
    assert plan["shards"]["gg-otf"]["protocol_skip_multiplicities"] == [7, 8, 9]
    assert plan["shards"]["gg-otf"]["argv"] is not None


def test_composer_refuses_terminal_status_after_out_of_order_frontier() -> None:
    arguments = study._parser().parse_args(
        (
            "--multiplicity",
            "2",
            "--multiplicity",
            "8",
            "--family",
            "gg",
            "--mode",
            "reference-fft",
        )
    )
    failed = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 2) | {
        "status": "failed",
        "censors_higher_multiplicities": True,
    }
    measured = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 8) | {
        "status": "measured"
    }

    report = study.compose_report(
        arguments,
        {"gg": {"reference-fft": {"2": failed, "8": measured}}},
    )

    assert report["status"] == "stopped-protocol-investigation"
    assert report["cells"]["gg"]["reference-fft"]["8"] == measured


def test_candidate_core_setting_is_passed_to_generation() -> None:
    command = study._candidate_generation_command(
        python="python",
        family="gg",
        final_multiplicity=3,
        mode=study.MODE_BY_KEY["otf-fft"],
        artifact=Path("artifact"),
        batch_size=1,
        optimization_cores=4,
    )
    assert "evaluator.optimization.cores=4" in command


def test_matching_manifest_resumes_automatically_and_rejects_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments("--output", str(output), "--multiplicities", "2", "3")
    first = profiling._create_or_resume_manifest(arguments, output)
    assert profiling._create_or_resume_manifest(arguments, output) == first

    later = _arguments("--output", str(output), "--multiplicities", "4", "5")
    expanded = profiling._create_or_resume_manifest(later, output)
    assert expanded["requested_multiplicities"] == [2, 3, 4, 5]
    shard_arguments = profiling._shard_arguments(
        later, output, profiling.SHARD_BY_NAME["ddbar-recurrence"]
    )
    assert shard_arguments.fill_multiplicities == [2, 3, 4, 5]

    changed = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--memory-limit-gib",
        "64",
    )
    with pytest.raises(profiling.ProfilingError, match="measurement identity"):
        profiling._create_or_resume_manifest(changed, output)

    summed = _arguments(
        "--compare-helicity-sums",
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
    )
    with pytest.raises(profiling.ProfilingError, match="helicity_workload"):
        profiling._create_or_resume_manifest(summed, output)

    scalar = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--batch-size",
        "1",
    )
    with pytest.raises(profiling.ProfilingError, match="batch_size"):
        profiling._create_or_resume_manifest(scalar, output)

    other_host = dict(first["identity"]["measurement_host"])
    other_host["node_sha256"] = "f" * 64
    monkeypatch.setattr(
        profiling.madgraph, "measurement_host_identity", lambda: other_host
    )
    with pytest.raises(profiling.ProfilingError, match="measurement_host"):
        profiling._create_or_resume_manifest(arguments, output)


def test_status_rejects_workload_mismatch_with_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "fixed-run"
    fixed = _arguments("--output", str(output), "--multiplicities", "2")
    profiling._create_or_resume_manifest(fixed, output)
    summed = _arguments(
        "--status",
        "--compare-helicity-sums",
        "--output",
        str(output),
        "--multiplicities",
        "2",
    )

    with pytest.raises(profiling.ProfilingError, match="identity differs"):
        profiling.status_payload(summed)


def test_refresh_is_confined_to_recognized_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    arguments = _arguments("--output", str(output))
    profiling._create_or_resume_manifest(arguments, output)
    retained = output / "owned.txt"
    retained.write_text("owned", encoding="ascii")

    profiling._safe_refresh(output)

    assert not output.exists()
    with pytest.raises(profiling.ProfilingError, match="ambiguous"):
        profiling._safe_refresh(Path.home())
    unrecognized = tmp_path / "not-a-run"
    unrecognized.mkdir()
    with pytest.raises(profiling.ProfilingError, match="recognized"):
        profiling._safe_refresh(unrecognized)


def test_first_refresh_of_a_missing_output_starts_cleanly(tmp_path: Path) -> None:
    output = tmp_path / "new-run"

    assert profiling._safe_refresh(output) is False
    assert not output.exists()


def test_refresh_lock_survives_output_deletion(tmp_path: Path) -> None:
    output = tmp_path / "run"
    arguments = _arguments("--output", str(output))
    profiling._create_or_resume_manifest(arguments, output)
    first = profiling._acquire_execution_lock(output)
    try:
        profiling._safe_refresh(output)
        assert not output.exists()
        with pytest.raises(profiling.ProfilingError, match="another profiling"):
            profiling._acquire_execution_lock(output)
    finally:
        first.close()


def test_refresh_rejects_an_active_madgraph_cache_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    arguments = _arguments("--output", str(output))
    profiling._create_or_resume_manifest(arguments, output)
    retained = output / "owned.txt"
    retained.write_text("owned", encoding="ascii")
    monkeypatch.setattr(profiling.madgraph, "CACHE_LOCK_ROOT", tmp_path / "locks")

    cache_lock = profiling.madgraph._acquire_cache_lock(
        profiling._madgraph_cache_path(output)
    )
    try:
        with pytest.raises(
            profiling.ProfilingError,
            match="cannot refresh while another MadGraph profiler holds",
        ):
            profiling._safe_refresh(output)
    finally:
        cache_lock.close()

    assert retained.read_text(encoding="ascii") == "owned"


def test_render_wrapper_holds_generation_lock_until_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    arguments = _arguments("--output", str(output))

    class Lock:
        closed = False

        def close(self) -> None:
            self.closed = True

    lock = Lock()
    monkeypatch.setattr(profiling, "_acquire_render_lock", lambda _output: lock)

    def render_locked(*_args, **_kwargs):
        assert lock.closed is False
        return output / "published.pdf"

    monkeypatch.setattr(profiling, "_render_snapshot_locked", render_locked)

    assert profiling.render_snapshot(arguments) == output / "published.pdf"
    assert lock.closed is True


def test_refresh_rejects_symlinked_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-run"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(profiling.ProfilingError, match="symlinked"):
        profiling._safe_refresh(link)

    assert target.is_dir()


def _partial_report(output: Path, *, summed: bool = False) -> dict[str, object]:
    values = ["--output", str(output), "--multiplicities", "2"]
    if summed:
        values.append("--compare-helicity-sums")
    arguments = _arguments(*values)
    report = study._empty_report(profiling._master_arguments(arguments, output))
    report["status"] = "running"
    return report


def test_render_freezes_once_and_publishes_only_complete_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "render-run"
    source = tmp_path / "live.json"
    source.write_text(json.dumps(_partial_report(output)), encoding="utf-8")
    original = source.read_bytes()
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        command = tuple(command)
        calls.append(command)
        if Path(command[1]) == profiling.PLOT_TOOL:
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(b"png")
        elif Path(command[1]) == profiling.PDF_TOOL:
            Path(command[command.index("--output") + 1]).write_bytes(b"pdf")

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_madgraph_render_source",
        lambda *_args: pytest.fail("external report must not inherit output MG data"),
    )
    arguments = _arguments(
        "--render",
        "--output",
        str(output),
        "--campaign-report",
        str(source),
    )

    pdf = profiling.render_snapshot(arguments)

    assert source.read_bytes() == original
    assert pdf.read_bytes() == b"pdf"
    assert not profiling._canonical_pdf_path(arguments).exists()
    assert (output / "render" / "current").is_symlink()
    assert [Path(command[1]) for command in calls] == [
        profiling.PLOT_TOOL,
        profiling.PDF_TOOL,
    ]
    assert calls[0][2] == calls[1][calls[1].index("--campaign-report") + 1]


def test_default_campaign_render_atomically_publishes_canonical_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "default-run"
    monkeypatch.setattr(profiling, "DEFAULT_FIXED_OUTPUT", output)
    source = profiling._master_report_path(output)
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_partial_report(output)), encoding="utf-8")
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)

    def fake_run(command, **_kwargs):
        command = tuple(command)
        if Path(command[1]) == profiling.PLOT_TOOL:
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(
                        b"png"
                    )
        else:
            Path(command[command.index("--output") + 1]).write_bytes(b"pdf")

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    arguments = _arguments("--render")

    profiling.render_snapshot(arguments)

    assert profiling._canonical_pdf_path(arguments).read_bytes() == b"pdf"


def test_bare_render_falls_back_to_richest_compatible_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "missing-default-run"
    monkeypatch.setattr(profiling, "DEFAULT_FIXED_OUTPUT", output)
    report_root = (
        profiling.CANONICAL_RESULTS_ROOT / "fft-scaling-study" / "raw" / "runs"
    )
    sparse = report_root / "sparse" / "report.json"
    richer = report_root / "richer" / "report.json"
    summed = report_root / "summed" / "report.json"
    for path, report in (
        (sparse, _partial_report(output)),
        (richer, _partial_report(output)),
        (summed, _partial_report(output, summed=True)),
    ):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report), encoding="utf-8")
    richer_payload = json.loads(richer.read_text(encoding="utf-8"))
    richer_payload["cells"]["gg"]["reference-fft"]["2"] = {
        "status": "measured",
        "generation_seconds": 1.0,
        "warm_seconds_per_point": 1.0e-6,
        "max_rss_kib": 1024,
    }
    richer.write_text(json.dumps(richer_payload), encoding="utf-8")

    arguments = _arguments("--render")

    assert profiling._render_source(arguments, output) == richer.resolve()


def test_bare_render_compares_existing_primary_with_isolated_composite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "default-run"
    monkeypatch.setattr(profiling, "DEFAULT_FIXED_OUTPUT", output)
    primary = profiling._master_report_path(output)
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(_partial_report(output)), encoding="utf-8")
    composite = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-profiling"
        / "runs"
        / "extension"
        / "composites"
        / "selected"
        / "report.json"
    )
    richer = _partial_report(output)
    richer["cells"]["ddbar"]["recurrence-direct"]["2"] = {
        "status": "measured",
        "generation_seconds": 1.0,
        "warm_seconds_per_point": 1.0e-6,
        "max_rss_kib": 1024,
    }
    composite.parent.mkdir(parents=True)
    composite.write_text(json.dumps(richer), encoding="utf-8")
    rendered_cell: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        command = tuple(command)
        if Path(command[1]) == profiling.PLOT_TOOL:
            frozen = json.loads(Path(command[2]).read_text(encoding="utf-8"))
            rendered_cell.update(
                frozen["cells"]["ddbar"]["recurrence-direct"]["2"]
            )
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(
                        b"png"
                    )
        else:
            Path(command[command.index("--output") + 1]).write_bytes(b"pdf")

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    arguments = _arguments("--render")

    assert profiling._render_source(arguments, output) == composite.resolve()
    profiling.render_snapshot(arguments)

    assert rendered_cell["status"] == "measured"
    assert profiling._canonical_pdf_path(arguments).read_bytes() == b"pdf"


def test_implicit_render_selects_only_workload_compatible_global_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "summed-default"
    monkeypatch.setattr(profiling, "DEFAULT_SUM_OUTPUT", output)
    primary = profiling._master_report_path(output)
    report = _partial_report(output, summed=True)
    report["runtime_series"] = {}
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(report), encoding="utf-8")
    fixed_overlay = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-scaling-study"
        / "data"
        / "selected-scalar-madgraph-runtime-series-overlay.json"
    )
    summed_overlay = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-profiling"
        / "runs"
        / "summed-extension"
        / "madgraph"
        / "overlay.json"
    )
    for path in (fixed_overlay, summed_overlay):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": profiling.madgraph.KIND}), encoding="utf-8")

    def attach(report_payload: dict[str, object], path: Path) -> None:
        if path == fixed_overlay.resolve():
            raise profiling.ProfilingError("fixed-helicity overlay")
        assert path == summed_overlay.resolve()
        report_payload["runtime_series"] = {
            "gg": {
                "madgraph-standalone": {
                    "2": {
                        "status": "measured",
                        "warm_seconds_per_point": 1.0e-6,
                    }
                }
            }
        }

    monkeypatch.setattr(profiling, "_attach_partial_overlay", attach)
    arguments = _arguments("--render", "--compare-helicity-sums")

    selection = profiling._implicit_render_selection(arguments, output)

    assert selection == profiling.RenderSelection(
        source=primary,
        overlay=summed_overlay.resolve(),
    )


def test_summed_bare_render_preserves_n6_extension_and_n2_n5_madgraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "summed-default"
    monkeypatch.setattr(profiling, "DEFAULT_SUM_OUTPUT", output)
    primary = profiling._master_report_path(output)
    base = _partial_report(output, summed=True)
    for family in ("gg", "ddbar"):
        for final_multiplicity in range(2, 6):
            base["cells"][family]["recurrence-direct"][str(final_multiplicity)] = {
                "status": "measured",
                "warm_seconds_per_point": 1.0e-6,
            }
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(base), encoding="utf-8")
    composite = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-profiling"
        / "runs"
        / "summed-ddbar-n6"
        / "composites"
        / "selected"
        / "report.json"
    )
    extended = json.loads(json.dumps(base))
    extended["cells"]["ddbar"]["recurrence-direct"]["6"] = {
        "status": "measured",
        "warm_seconds_per_point": 2.0e-6,
    }
    composite.parent.mkdir(parents=True)
    composite.write_text(json.dumps(extended), encoding="utf-8")
    overlay = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-profiling"
        / "runs"
        / "summed-madgraph-n2-n5"
        / "madgraph"
        / "overlay.json"
    )
    overlay.parent.mkdir(parents=True)
    overlay.write_text(json.dumps({"kind": profiling.madgraph.KIND}), encoding="utf-8")

    def attach(report_payload: dict[str, object], path: Path) -> None:
        assert path == overlay.resolve()
        report_payload["runtime_series"] = {
            family: {
                "madgraph-standalone": {
                    str(final_multiplicity): {
                        "status": "measured",
                        "warm_seconds_per_point": 3.0e-6,
                    }
                    for final_multiplicity in range(2, 6)
                }
            }
            for family in ("gg", "ddbar")
        }

    rendered: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        command = tuple(command)
        if Path(command[1]) == profiling.PLOT_TOOL:
            rendered.update(json.loads(Path(command[2]).read_text(encoding="utf-8")))
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(
                        b"png"
                    )
        else:
            Path(command[command.index("--output") + 1]).write_bytes(b"pdf")

    monkeypatch.setattr(profiling, "_attach_partial_overlay", attach)
    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    arguments = _arguments("--render", "--compare-helicity-sums")

    assert profiling._implicit_render_selection(
        arguments, output
    ) == profiling.RenderSelection(
        source=composite.resolve(),
        overlay=overlay.resolve(),
    )
    profiling.render_snapshot(arguments)

    assert rendered["cells"]["ddbar"]["recurrence-direct"]["6"]["status"] == (
        "measured"
    )
    assert set(rendered["runtime_series"]["gg"]["madgraph-standalone"]) == {
        "2",
        "3",
        "4",
        "5",
    }
    assert profiling._canonical_pdf_path(arguments).read_bytes() == b"pdf"


def test_fixed_bare_render_combines_otf_frontier_with_valid_madgraph_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fixed-default"
    monkeypatch.setattr(profiling, "DEFAULT_FIXED_OUTPUT", output)
    composite_report = _partial_report(output)
    for family, modes in profiling.publication.FAMILY_MODES.items():
        for mode in modes:
            for final_multiplicity in range(2, 8):
                if (family, mode, final_multiplicity) == ("gg", "otf-fft", 7):
                    continue
                composite_report["cells"][family][mode][str(final_multiplicity)] = {
                    "status": "measured",
                    "warm_seconds_per_point": 1.0e-6,
                }
    primary_report = json.loads(json.dumps(composite_report))
    for mode in ("otf-direct", "otf-fft"):
        for final_multiplicity in (6, 7):
            del primary_report["cells"]["ddbar"][mode][str(final_multiplicity)]

    def runtime_series(*, protocol_scoped: bool) -> dict[str, object]:
        series: dict[str, object] = {}
        for family in ("gg", "ddbar"):
            measured_max = 5 if family == "gg" else 6
            cells: dict[str, object] = {}
            for final_multiplicity in range(2, 10):
                if final_multiplicity <= measured_max:
                    status = "measured"
                elif protocol_scoped and final_multiplicity > 6:
                    status = "not-applicable"
                elif final_multiplicity == measured_max + 1:
                    status = "failed"
                else:
                    status = "skipped"
                cells[str(final_multiplicity)] = {"status": status}
            series[family] = {"madgraph-standalone": cells}
        return series

    primary_report["runtime_series"] = runtime_series(protocol_scoped=False)
    primary = profiling._master_report_path(output)
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(primary_report), encoding="utf-8")
    composite = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-profiling"
        / "runs"
        / "fixed-otf-extension"
        / "composites"
        / "selected"
        / "report.json"
    )
    composite.parent.mkdir(parents=True)
    composite.write_text(json.dumps(composite_report), encoding="utf-8")
    overlay_series = runtime_series(protocol_scoped=True)
    overlay = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-scaling-study"
        / "data"
        / "selected-fixed-madgraph-overlay.json"
    )
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        json.dumps(
            {
                "kind": profiling.madgraph.KIND,
                "policy": {
                    "final_state_multiplicities": list(range(2, 10)),
                    "helicity_workload": "fixed",
                    "warm_fixed_helicity": True,
                    "warm_helicity_sum": False,
                    "maximum_measured_multiplicity": 6,
                    "higher_multiplicity_policy": (
                        "not-applicable-protocol-scope"
                    ),
                },
                "runtime_series": overlay_series,
            }
        ),
        encoding="utf-8",
    )

    def attach(report_payload: dict[str, object], path: Path) -> None:
        assert path == overlay.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["policy"]["helicity_workload"] == "fixed"
        for family_cells in payload["runtime_series"].values():
            mode_cells = family_cells["madgraph-standalone"]
            assert {
                mode_cells[str(final_multiplicity)]["status"]
                for final_multiplicity in range(7, 10)
            } == {"not-applicable"}
        report_payload["runtime_series"] = json.loads(
            json.dumps(payload["runtime_series"])
        )

    rendered: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        command = tuple(command)
        if Path(command[1]) == profiling.PLOT_TOOL:
            rendered.update(json.loads(Path(command[2]).read_text(encoding="utf-8")))
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(
                        b"png"
                    )
        else:
            Path(command[command.index("--output") + 1]).write_bytes(b"pdf")

    monkeypatch.setattr(profiling, "_attach_partial_overlay", attach)
    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    arguments = _arguments("--render")
    requested = set(profiling._selection(arguments))

    assert profiling._render_inventory(
        primary_report, requested=requested
    )[0] == 70
    assert profiling._render_inventory(
        composite_report, requested=requested
    )[0] == 65
    assert profiling._implicit_render_selection(
        arguments, output
    ) == profiling.RenderSelection(
        source=composite.resolve(),
        overlay=overlay.resolve(),
    )
    profiling.render_snapshot(arguments)

    for mode in ("otf-direct", "otf-fft"):
        assert rendered["cells"]["ddbar"][mode]["6"]["status"] == "measured"
    measured_madgraph = {
        (family, int(final_multiplicity))
        for family, family_cells in rendered["runtime_series"].items()
        for final_multiplicity, cell in family_cells[
            "madgraph-standalone"
        ].items()
        if cell["status"] == "measured"
    }
    expected_madgraph = {
        ("gg", final_multiplicity) for final_multiplicity in range(2, 6)
    } | {("ddbar", final_multiplicity) for final_multiplicity in range(2, 7)}
    assert measured_madgraph == expected_madgraph
    assert profiling._canonical_pdf_path(arguments).read_bytes() == b"pdf"


def test_explicit_existing_output_ignores_richer_implicit_fallback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "explicit"
    primary = profiling._master_report_path(output)
    primary.parent.mkdir(parents=True)
    primary.write_text(json.dumps(_partial_report(output)), encoding="utf-8")
    fallback = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-profiling"
        / "runs"
        / "extension"
        / "composites"
        / "selected"
        / "report.json"
    )
    richer = _partial_report(output)
    richer["cells"]["gg"]["reference-fft"]["2"] = {"status": "measured"}
    fallback.parent.mkdir(parents=True)
    fallback.write_text(json.dumps(richer), encoding="utf-8")
    arguments = _arguments("--render", "--output", str(output))

    assert profiling._render_source(arguments, output) == primary


def test_explicit_output_does_not_use_implicit_render_fallback(
    tmp_path: Path,
) -> None:
    fallback = (
        profiling.CANONICAL_RESULTS_ROOT
        / "fft-scaling-study"
        / "data"
        / "campaign-report-existing.json"
    )
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        json.dumps(_partial_report(tmp_path / "unrelated")), encoding="utf-8"
    )
    output = tmp_path / "explicit-missing"
    arguments = _arguments("--render", "--output", str(output))

    with pytest.raises(profiling.ProfilingError, match="no campaign snapshot yet"):
        profiling.render_snapshot(arguments)

    assert not output.exists()


def test_missing_render_snapshot_fails_without_waiting_or_writing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing"
    arguments = _arguments("--render", "--output", str(output))
    with pytest.raises(profiling.ProfilingError, match="no campaign snapshot yet"):
        profiling.render_snapshot(arguments)
    assert not output.exists()


@pytest.mark.parametrize(
    ("report_summed", "argument_summed"), ((False, True), (True, False))
)
def test_render_rejects_report_workload_filename_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_summed: bool,
    argument_summed: bool,
) -> None:
    output = tmp_path / "mismatch"
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(_partial_report(output, summed=report_summed)), encoding="utf-8"
    )
    values = ["--render", "--output", str(output), "--campaign-report", str(source)]
    if argument_summed:
        values.append("--compare-helicity-sums")
    arguments = _arguments(*values)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_run_checked",
        lambda *_args, **_kwargs: pytest.fail("mismatched workload must not render"),
    )

    with pytest.raises(profiling.ProfilingError, match="helicity workload"):
        profiling.render_snapshot(arguments)

    assert not profiling._canonical_pdf_path(arguments).exists()


def test_workload_validation_does_not_import_matplotlib_plotter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _partial_report(tmp_path / "fixed")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.endswith("fft_scaling_study_plots"):
            raise AssertionError("workload validation imported the renderer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    profiling._validate_render_workload(_arguments(), report)


def test_render_prefers_matching_running_madgraph_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "render-progress"
    source = profiling._master_report_path(output)
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(_partial_report(output)), encoding="utf-8")
    progress_path = profiling._madgraph_progress_path(output)
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text("{}", encoding="ascii")
    attached: list[Path] = []

    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_matching_madgraph_progress",
        lambda _arguments, _output: {"status": "running"},
    )
    monkeypatch.setattr(
        profiling,
        "_attach_partial_overlay",
        lambda _report, path: attached.append(path),
    )

    def fake_run(command, **_kwargs):
        command = tuple(command)
        if Path(command[1]) == profiling.PLOT_TOOL:
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(b"png")
        else:
            Path(command[command.index("--output") + 1]).write_bytes(b"pdf")

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    arguments = _arguments("--render", "--output", str(output))

    profiling.render_snapshot(arguments)

    assert attached == [progress_path]


def test_renderer_preflight_has_actionable_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiling, "_command_available", lambda _command: True)
    monkeypatch.setattr(
        profiling.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})(),
    )
    with pytest.raises(profiling.ProfilingError, match="fft-profiling"):
        profiling._preflight_renderer(_arguments())


def test_renderer_preflight_rejects_selected_python_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiling, "_command_available", lambda _command: True)
    monkeypatch.setattr(
        profiling.subprocess,
        "run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 0, "stdout": "0.0-incompatible\n"}
        )(),
    )

    with pytest.raises(profiling.ProfilingError, match="profiling driver uses"):
        profiling._preflight_renderer(_arguments())


def test_executable_normalization_preserves_virtualenv_symlink(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(Path(sys.executable))
    arguments = _arguments("--python", str(launcher))

    profiling._normalize_executables(arguments)

    assert arguments.python == str(launcher.absolute())


def test_progress_attachment_validates_one_immutable_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "progress.json"
    current = profiling.madgraph.measurement_host_identity()
    payload = {
        "kind": profiling.madgraph.PROGRESS_KIND,
        "host": {key: current[key] for key in ("system", "machine", "python")},
    }
    raw = json.dumps(payload).encode("utf-8")
    live.write_bytes(raw)
    validated_paths: list[Path] = []
    attached: list[profiling.selected.SourceReport] = []

    def validate(path: Path):
        validated_paths.append(path)
        assert path != live
        assert path.read_bytes() == raw
        return payload

    monkeypatch.setattr(profiling.madgraph, "load_runtime_progress", validate)
    monkeypatch.setattr(
        profiling.selected,
        "apply_runtime_series_source",
        lambda _report, source: attached.append(source),
    )

    profiling._attach_partial_overlay({}, live)

    assert len(validated_paths) == 1
    assert attached[0].path == live
    assert attached[0].sha256 == profiling.hashlib.sha256(raw).hexdigest()


def test_summed_overlay_replaces_stale_madgraph_omission_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = tmp_path / "summed-overlay.json"
    current = profiling.madgraph.measurement_host_identity()
    overlay.write_text(
        json.dumps(
            {
                "kind": profiling.madgraph.KIND,
                "host": {
                    key: current[key] for key in ("system", "machine", "python")
                },
                "policy": {
                    "helicity_workload": "sum",
                    "warm_fixed_helicity": False,
                    "warm_helicity_sum": True,
                },
            }
        ),
        encoding="utf-8",
    )
    report = {
        "policy": {
            "plot": {
                "notes": [
                    "Complete helicity sum; MadGraph is omitted because its "
                    "available series is fixed-helicity."
                ]
            }
        }
    }
    monkeypatch.setattr(
        profiling.selected, "apply_runtime_series_source", lambda *_args: None
    )

    profiling._attach_partial_overlay(report, overlay)

    notes = report["policy"]["plot"]["notes"]
    assert all("omitted" not in note.lower() for note in notes)
    assert notes == [
        profiling.LEGACY_MADGRAPH_NOTE,
        "MadGraph standalone uses generated SMATRIX with USERHEL=-1; warmed "
        "GOODHEL pruning remains enabled."
    ]


def test_helicity_sum_n2_n3_campaign_runs_summed_madgraph_and_renders_sum_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "sum-run"
    arguments = _arguments(
        "--compare-helicity-sums",
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
    )
    report = {"status": "running", "cells": {}}
    phases: list[int] = []

    class FakeDashboard:
        def __init__(self, **_kwargs) -> None:
            pass

        def update(self, *_args, **_kwargs) -> None:
            pass

        def finish(self) -> None:
            pass

    monkeypatch.setattr(profiling, "Dashboard", FakeDashboard)
    monkeypatch.setattr(profiling, "_preflight", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_publish_master",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(
        profiling,
        "_phase",
        lambda _arguments, _output, phase, _dashboard: phases.append(phase),
    )
    monkeypatch.setattr(profiling, "_completed_cells", lambda _report: 0)
    monkeypatch.setattr(
        profiling, "_selected_pending_cells", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(
        profiling, "_selected_master_complete", lambda *_args, **_kwargs: True
    )
    madgraph_steps: list[str] = []
    monkeypatch.setattr(
        profiling,
        "_freeze_madgraph_source",
        lambda *_args: madgraph_steps.append("freeze"),
    )
    monkeypatch.setattr(
        profiling,
        "_run_madgraph",
        lambda *_args: madgraph_steps.append("run"),
    )
    expected = output / "render" / "current" / "summary_plots_final_helicity_sum.pdf"
    render_calls: list[bool] = []

    def fake_render(_arguments, *, renderer_preflight=True):
        render_calls.append(renderer_preflight)
        return expected

    monkeypatch.setattr(profiling, "render_snapshot", fake_render)

    result = profiling.run_campaign(arguments)

    assert result == expected
    assert phases == [1, 2, 3]
    assert madgraph_steps == ["freeze", "run"]
    assert render_calls == [False, False, False, False]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity"]["scan"]["helicity_workload"] == "sum"


def test_partial_render_can_reuse_ordered_subset_madgraph_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "expanded"
    overlay = profiling._madgraph_overlay_path(output)
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        json.dumps(
            {
                "host": profiling.madgraph.measurement_host_identity(),
                "policy": {
                    "final_state_multiplicities": [2, 3],
                    "maximum_measured_multiplicity": 6,
                    "higher_multiplicity_policy": (
                        "not-applicable-protocol-scope"
                    ),
                },
                "runtime_series": {},
            }
        ),
        encoding="utf-8",
    )
    arguments = _arguments("--output", str(output), "--multiplicities", "4")
    monkeypatch.setattr(
        profiling,
        "_requested_multiplicities",
        lambda _arguments, _output: (2, 3, 4),
    )

    assert profiling._matching_madgraph_overlay(arguments, output) is None
    assert (
        profiling._matching_madgraph_overlay(arguments, output, require_exact=False)
        == overlay
    )
    assert profiling._madgraph_render_source(arguments, output) == overlay

    legacy_payload = json.loads(overlay.read_text(encoding="utf-8"))
    current = profiling.madgraph.measurement_host_identity()
    legacy_payload["host"] = {
        key: current[key] for key in ("system", "machine", "python")
    }
    overlay.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert profiling._matching_madgraph_overlay(arguments, output) is None
    assert (
        profiling._matching_madgraph_overlay(
            arguments, output, require_exact=False
        )
        == overlay
    )
    assert profiling._madgraph_render_source(arguments, output) == overlay

    legacy_payload["host"]["machine"] = "foreign-machine"
    overlay.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert (
        profiling._matching_madgraph_overlay(
            arguments, output, require_exact=False
        )
        is None
    )

    legacy_payload["host"] = {
        key: current[key] for key in ("system", "machine", "python")
    } | {"node_sha256": "malformed-modern-fingerprint"}
    overlay.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert (
        profiling._matching_madgraph_overlay(
            arguments, output, require_exact=False
        )
        is None
    )
