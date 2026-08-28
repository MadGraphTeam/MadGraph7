# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import sys
import tomllib
from copy import deepcopy
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
        "process_families": ["gg", "ddbar"],
        "measurement_multiplicities": [2, 3],
        "protocol_scope_multiplicities": [],
        "measurement_multiplicities_by_family": {
            "gg": [2, 3],
            "ddbar": [2, 3],
        },
        "protocol_scope_multiplicities_by_family": {"gg": [], "ddbar": []},
        "dependency": (
            "completed pyAmpliCol source cells for the requested MadGraph fill"
        ),
        "not_applicable_reason": None,
        "report": str(tmp_path / "summed" / "madgraph" / "overlay.json"),
        "argv": plan["madgraph"]["argv"],
        "shell_command": plan["madgraph"]["shell_command"],
    }
    assert plan["madgraph"]["argv"] is not None
    assert "--compare-helicity-sums" in plan["madgraph"]["argv"]
    assert plan["madgraph"]["argv"][
        plan["madgraph"]["argv"].index("--target-seconds") + 1
    ] == "180"
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
    assert plan["madgraph"]["measurement_multiplicities_by_family"] == {
        "gg": [],
        "ddbar": [6],
    }
    assert plan["madgraph"]["protocol_scope_multiplicities_by_family"] == {
        "gg": [6, 7, 8],
        "ddbar": [7, 8],
    }
    assert argv is not None
    assert argv[argv.index("--multiplicity") + 1] == "6"
    assert argv[argv.index("--target-seconds") + 1] == "180"
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
    explicit = _arguments("--compare-helicity-sums", "--output", str(explicit_path))

    assert profiling._run_directory(fixed) == profiling.DEFAULT_FIXED_OUTPUT
    assert profiling._run_directory(summed) == profiling.DEFAULT_SUM_OUTPUT
    assert profiling._run_directory(explicit) == explicit_path
    assert fixed.batch_size == 128
    assert summed.batch_size == 128
    assert profiling._pdf_filename(fixed) == "summary_plots_final.pdf"
    assert profiling._pdf_filename(summed) == "summary_plots_final_helicity_sum.pdf"


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
        assert {report["cells"]["gg"][mode][str(n)]["status"] for n in (7, 8, 9)} == {
            "skipped"
        }
    assert not profiling._selected_shard_complete(arguments, shard, report)

    plan = profiling.dry_run_plan(arguments)
    assert plan["shards"]["gg-otf"]["measurement_multiplicities"] == [6]
    assert plan["shards"]["gg-otf"]["protocol_skip_multiplicities"] == [7, 8, 9]
    assert plan["shards"]["gg-otf"]["argv"] is not None


def test_otf_protocol_scope_can_be_extended_for_selective_top_up() -> None:
    arguments = _arguments(
        "--multiplicities",
        "7",
        "8",
        "--families",
        "ddbar",
        "--lines",
        "pyamplicol-otf",
        "--otf-max-multiplicity",
        "8",
    )

    plan = profiling.dry_run_plan(arguments)
    otf = plan["shards"]["ddbar-otf"]

    assert otf["measurement_multiplicities"] == [7, 8]
    assert otf["protocol_skip_multiplicities"] == []
    assert [(job["n"], job["mode"]) for job in otf["jobs"]] == [
        (7, "otf-direct"),
        (7, "otf-fft"),
        (8, "otf-direct"),
        (8, "otf-fft"),
    ]
    for job in otf["jobs"]:
        option = job["argv"].index("--otf-max-multiplicity")
        assert job["argv"][option + 1] == "8"


def test_candidate_direct_and_fft_modes_are_distinct_work_items(tmp_path: Path) -> None:
    output = tmp_path / "split"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "6",
        "--families",
        "gg",
        "--lines",
        "pyamplicol-otf",
    )

    works = profiling._phase_work_items(arguments, output, 2)

    assert [
        (work.shard.name, work.final_multiplicity, work.owned_mode)
        for work in works
    ] == [
        ("gg-otf", 6, "otf-direct"),
        ("gg-otf", 6, "otf-fft"),
    ]
    assert len({profiling._work_report_path(output, work) for work in works}) == 2
    for work in works:
        command = profiling._child_command(arguments, output, work)
        assert command.count("--fill-mode") == 1
        assert command[command.index("--fill-mode") + 1] == work.owned_mode


def test_helicity_sum_amplicol_work_items_do_not_overlap_shared_overlay() -> None:
    arguments = _arguments("--compare-helicity-sums")
    first = profiling.ShardWork(
        profiling.SHARD_BY_NAME["ddbar-amplicol"], 2, "amplicol"
    )
    second = profiling.ShardWork(
        profiling.SHARD_BY_NAME["gg-amplicol"], 2, "amplicol"
    )
    active = [
        profiling.ActiveJob(
            shard=first.shard,
            final_multiplicity=first.final_multiplicity,
            owned_mode=first.owned_mode,
            command=(),
            process=object(),
            stdout=io.BytesIO(),
            stderr=io.BytesIO(),
            sampler=object(),
            claimed_cores=1,
            detail="",
        )
    ]

    assert profiling._work_launch_conflicts(arguments, second, active) is True
    assert profiling._work_launch_conflicts(arguments, first, ()) is False


def test_composer_retains_measured_cells_above_out_of_order_frontier() -> None:
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

    assert report["status"] == "complete-with-failures"
    assert report["cells"]["gg"]["reference-fft"]["8"] == measured
    assert report["resource_frontier_inversions"] == [["gg", "reference-fft", 2, 8]]


def test_dry_run_exposes_per_multiplicity_workers(tmp_path: Path) -> None:
    arguments = _arguments(
        "--dry-run",
        "--output",
        str(tmp_path / "run"),
        "--multiplicities",
        "2",
        "3",
        "4",
        "--lines",
        "reference-fft",
        "--cores",
        "3",
    )

    plan = profiling.dry_run_plan(arguments)
    reference = plan["shards"]["gg-reference"]
    jobs = reference["jobs"]

    assert plan["scheduler"]["parallelism"] == (
        "weighted dependency-aware shard-multiplicity workers"
    )
    assert [job["n"] for job in jobs] == [2, 3, 4]
    assert len({job["study_root"] for job in jobs}) == 3
    assert reference["argv"] == jobs[0]["argv"]
    assert [
        job["argv"][index + 1]
        for job in jobs
        for index, value in enumerate(job["argv"])
        if value == "--fill-multiplicity"
    ] == ["2", "3", "4"]


def test_isolated_shard_reports_merge_and_retain_started_higher_measurement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--lines",
        "reference-fft",
    )
    profiling._create_or_resume_manifest(arguments, output)
    shard = profiling.SHARD_BY_NAME["gg-reference"]
    failed = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 2) | {
        "status": "failed",
        "failure_category": "memory-limit",
        "failure_reason": "memory cap",
        "censors_higher_multiplicities": True,
    }
    measured = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 3) | {
        "status": "measured",
        "generation_seconds": 1.0,
        "warm_seconds_per_point": 1.0e-6,
        "max_rss_kib": 1024,
    }
    for final_multiplicity, cell in ((2, failed), (3, measured)):
        report = study.compose_report(
            profiling._shard_arguments(arguments, output, shard, final_multiplicity),
            {"gg": {"reference-fft": {str(final_multiplicity): cell}}},
        )
        profiling._write_json_atomic(
            profiling._shard_report_path(output, shard, final_multiplicity),
            report,
        )

    report = profiling._load_shard_report(arguments, output, shard, required=True)

    assert report["status"] == "running"
    curve = report["cells"]["gg"]["reference-fft"]
    assert curve["2"]["status"] == "failed"
    assert curve["3"] == measured
    assert study.resource_frontier_inversions(report) == (
        ("gg", "reference-fft", 2, 3),
    )


def test_isolated_shard_reports_preserve_prior_measurement_over_later_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "9",
        "--lines",
        "amplicol",
    )
    profiling._create_or_resume_manifest(arguments, output)
    shard = profiling.SHARD_BY_NAME["ddbar-amplicol"]
    measured = study._cell_base("ddbar", study.MODE_BY_KEY["amplicol"], 9) | {
        "status": "measured",
        "generation_seconds": 1.0,
        "warm_seconds_per_point": 1.0e-6,
        "max_rss_kib": 1024,
    }
    failed = study._cell_base("ddbar", study.MODE_BY_KEY["amplicol"], 9) | {
        "status": "failed",
        "failure_category": "legacy-amplicol-structural-limit",
        "failure_reason": "legacy crash",
        "censors_higher_multiplicities": True,
    }
    aggregate = study.compose_report(
        profiling._shard_arguments(arguments, output, shard),
        {"ddbar": {"amplicol": {"9": measured}}},
    )
    retry = study.compose_report(
        profiling._shard_arguments(arguments, output, shard, 9),
        {"ddbar": {"amplicol": {"9": failed}}},
    )
    profiling._write_json_atomic(profiling._shard_report_path(output, shard), aggregate)
    profiling._write_json_atomic(profiling._shard_report_path(output, shard, 9), retry)

    report = profiling._load_shard_report(arguments, output, shard, required=True)

    assert report["cells"]["ddbar"]["amplicol"]["9"] == measured


def test_shard_policy_compatibility_accepts_sampling_policy_updates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "5",
        "--lines",
        "pyamplicol-otf",
        "--compare-helicity-sums",
        "--memory-limit-gib",
        "100",
        "--time-limit-seconds",
        "3600",
    )
    shard = profiling.SHARD_BY_NAME["gg-otf"]
    modes = ("reference-fft", "otf-direct")
    expected = profiling._expected_shard_policy(
        arguments,
        output,
        shard,
        5,
        modes=modes,
    )
    stale = deepcopy(expected)
    measurement = stale["measurement"]
    measurement["calibration_target_seconds"] = 0.25
    measurement["candidate_profile_target_runtime_seconds"] = 0.25
    measurement["candidate_profile_warmup_runs"] = 2
    measurement["candidate_profile_minimum_samples"] = 10
    measurement["warm_samples"] = 10
    measurement["warm_sample_count"] = 10
    measurement["warm_timing_metric"] = {"pyamplicol": "old short sampling"}
    measurement["generation_timeout_seconds"] = 60.0
    measurement["runtime_timeout_seconds"] = 60.0
    measurement["requested_memory_ceiling_gib"] = 10.0
    measurement["memory_watchdog_gib"] = 10.0
    measurement["cell_admission_limits"] = {}

    assert profiling._shard_policy_matches_requested_modes(
        stale,
        expected,
        family="gg",
        modes=modes,
    )

    measurement["alpha_s"] = 0.2
    assert not profiling._shard_policy_matches_requested_modes(
        stale,
        expected,
        family="gg",
        modes=modes,
    )


def test_phase_refresh_does_not_queue_higher_work_after_lower_resource_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "4",
        "--lines",
        "reference-fft",
    )
    profiling._create_or_resume_manifest(arguments, output)
    shard = profiling.SHARD_BY_NAME["gg-reference"]
    failed = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 2) | {
        "status": "failed",
        "failure_category": "runtime-time-limit",
        "failure_reason": "runtime cap",
        "censors_higher_multiplicities": True,
    }
    report = study.compose_report(
        profiling._shard_arguments(arguments, output, shard, 2),
        {"gg": {"reference-fft": {"2": failed}}},
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard, 2),
        report,
    )

    pending = profiling._refresh_phase_pending(
        arguments,
        output,
        profiling._phase_work_items(arguments, output, 1),
        active=(),
    )
    merged = profiling._load_shard_report(arguments, output, shard, required=True)

    assert pending == []
    assert {
        n: merged["cells"]["gg"]["reference-fft"][str(n)]["status"] for n in (2, 3, 4)
    } == {2: "failed", 3: "skipped", 4: "skipped"}
    assert profiling._shard_report_path(output, shard).is_file()


def test_overwrite_retains_queued_higher_result_after_lower_resource_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "overwrite-frontier"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--lines",
        "reference-fft",
        "--overwrite",
    )
    profiling._create_or_resume_manifest(arguments, output)
    shard = profiling.SHARD_BY_NAME["gg-reference"]
    failed = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 2) | {
        "status": "failed",
        "failure_category": "runtime-time-limit",
        "failure_reason": "runtime cap",
        "censors_higher_multiplicities": True,
    }
    measured = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 3) | {
        "status": "measured",
        "generation_seconds": 2.0,
        "warm_seconds_per_point": 2.0e-6,
        "max_rss_kib": 2048,
    }
    report = study.compose_report(
        profiling._shard_arguments(arguments, output, shard),
        {"gg": {"reference-fft": {"2": failed, "3": measured}}},
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard), report
    )
    works = profiling._phase_work_items(arguments, output, 1)
    overwrite_pending = {profiling._work_identity(work) for work in works}

    pending = profiling._refresh_phase_pending(
        arguments,
        output,
        works,
        active=(),
        overwrite_pending=overwrite_pending,
    )

    assert [work.final_multiplicity for work in pending] == [2]
    assert overwrite_pending == {("gg", "reference-fft", 2)}
    retained = profiling._load_shard_report(arguments, output, shard, required=True)
    assert retained["cells"]["gg"]["reference-fft"]["3"] == measured


def test_phase_launches_independent_multiplicity_workers_when_cores_allow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "4",
        "--lines",
        "reference-fft",
        "--cores",
        "3",
        "--poll-seconds",
        "0.001",
    )
    profiling._create_or_resume_manifest(arguments, output)
    shard = profiling.SHARD_BY_NAME["gg-reference"]
    launched: list[int] = []
    active_counts: list[int] = []

    class FakeProcess:
        def __init__(self, work: profiling.ShardWork) -> None:
            self.pid = 12345 + work.final_multiplicity
            self.returncode: int | None = None
            self.work = work

        def poll(self) -> int | None:
            if self.returncode is None:
                cell = study._cell_base(
                    self.work.shard.family,
                    study.MODE_BY_KEY["reference-fft"],
                    self.work.final_multiplicity,
                ) | {"status": "measured"}
                report = study.compose_report(
                    profiling._shard_arguments(
                        arguments,
                        output,
                        self.work.shard,
                        self.work.final_multiplicity,
                    ),
                    {
                        self.work.shard.family: {
                            "reference-fft": {str(self.work.final_multiplicity): cell}
                        }
                    },
                )
                profiling._write_json_atomic(
                    profiling._shard_report_path(
                        output,
                        self.work.shard,
                        self.work.final_multiplicity,
                    ),
                    report,
                )
                self.returncode = 0
            return self.returncode

    class FakeDashboard:
        def update(self, _completed, *, phase, active) -> None:
            assert phase == "phase 1"
            active_counts.append(len(active))

    def launch(_arguments, _output, work: profiling.ShardWork):
        launched.append(work.final_multiplicity)
        return profiling.ActiveJob(
            shard=work.shard,
            final_multiplicity=work.final_multiplicity,
            owned_mode=work.owned_mode,
            command=(),
            process=FakeProcess(work),
            stdout=io.BytesIO(),
            stderr=io.BytesIO(),
            sampler=object(),
            claimed_cores=profiling._claimed_cores(arguments, work),
            detail="",
        )

    monkeypatch.setattr(profiling, "_launch_shard", launch)

    profiling._phase(arguments, output, 1, FakeDashboard())

    assert launched[:3] == [2, 3, 4]
    assert max(active_counts) == 3
    merged = profiling._load_shard_report(arguments, output, shard, required=True)
    assert study.selected_cells_complete(
        merged,
        family="gg",
        modes=("reference-fft",),
        multiplicities=(2, 3, 4),
    )


def test_phase_accepts_authoritative_failed_cell_from_nonzero_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    arguments = _arguments(
        "--output",
        str(output),
        "--families",
        "ddbar",
        "--multiplicities",
        "9",
        "--lines",
        "amplicol",
        "--cores",
        "1",
        "--poll-seconds",
        "0.001",
    )
    profiling._create_or_resume_manifest(arguments, output)

    class FakeProcess:
        def __init__(self, work: profiling.ShardWork) -> None:
            self.pid = 45678
            self.returncode: int | None = None
            self.work = work

        def poll(self) -> int | None:
            if self.returncode is None:
                cell = study._cell_base(
                    self.work.shard.family,
                    study.MODE_BY_KEY[self.work.owned_mode],
                    self.work.final_multiplicity,
                ) | {
                    "status": "failed",
                    "failure_category": "error",
                    "failure_reason": "legacy checkout contains tracked edits",
                    "censors_higher_multiplicities": False,
                }
                report = study.compose_report(
                    profiling._work_arguments(arguments, output, self.work),
                    {
                        self.work.shard.family: {
                            self.work.owned_mode: {
                                str(self.work.final_multiplicity): cell
                            }
                        }
                    },
                )
                report["status"] = "stopped-correctness-failure"
                profiling._write_json_atomic(
                    profiling._work_report_path(output, self.work),
                    report,
                )
                self.returncode = 1
            return self.returncode

    class FakeDashboard:
        def update(self, _completed, *, phase, active) -> None:
            assert phase == "phase 1"

    def launch(_arguments, _output, work: profiling.ShardWork):
        return profiling.ActiveJob(
            shard=work.shard,
            final_multiplicity=work.final_multiplicity,
            owned_mode=work.owned_mode,
            command=(),
            process=FakeProcess(work),
            stdout=io.BytesIO(),
            stderr=io.BytesIO(),
            sampler=object(),
            claimed_cores=1,
            detail="",
        )

    monkeypatch.setattr(profiling, "_launch_shard", launch)

    profiling._phase(arguments, output, 1, FakeDashboard())
    report = profiling._publish_master(arguments, output)

    assert profiling._selected_completed_cells(
        arguments, report, multiplicities=(9,)
    ) == 1
    assert (
        profiling._selected_pending_cells(arguments, report, multiplicities=(9,)) == 0
    )
    assert (
        report["cells"]["ddbar"]["amplicol"]["9"]["failure_category"] == "error"
    )


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
    assert expanded["requested_multiplicities"] == [4, 5]
    assert expanded["active_multiplicities"] == [4, 5]
    assert expanded["fill_history_multiplicities"] == [2, 3, 4, 5]
    shard_arguments = profiling._shard_arguments(
        later, output, profiling.SHARD_BY_NAME["ddbar-recurrence"]
    )
    assert shard_arguments.fill_multiplicities == [4, 5]

    resource_changed = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--memory-limit-gib",
        "64",
        "--time-limit-seconds",
        "7200",
    )
    resource_updated = profiling._create_or_resume_manifest(resource_changed, output)
    assert resource_updated["identity"]["resources"] == {
        "candidate_optimization_cores": 1,
        "per_cell_generation_timeout_seconds": 7200.0,
        "per_cell_memory_limit_gib": 64.0,
        "per_cell_runtime_timeout_seconds": 7200.0,
    }

    target_changed = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--memory-limit-gib",
        "64",
        "--time-limit-seconds",
        "7200",
        "--target-seconds",
        "30",
    )
    target_updated = profiling._create_or_resume_manifest(target_changed, output)
    assert target_updated["identity"]["scan"]["target_seconds"] == 30.0

    summed = _arguments(
        "--compare-helicity-sums",
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--memory-limit-gib",
        "64",
        "--time-limit-seconds",
        "7200",
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
        "--memory-limit-gib",
        "64",
        "--time-limit-seconds",
        "7200",
    )
    with pytest.raises(profiling.ProfilingError, match="batch_size"):
        profiling._create_or_resume_manifest(scalar, output)

    other_host = dict(first["identity"]["measurement_host"])
    other_host["node_sha256"] = "f" * 64
    monkeypatch.setattr(
        profiling.madgraph, "measurement_host_identity", lambda: other_host
    )
    with pytest.raises(profiling.ProfilingError, match="measurement_host"):
        profiling._create_or_resume_manifest(resource_changed, output)


def test_render_arguments_inherit_helicity_workload_from_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "summed"
    profiling._create_or_resume_manifest(
        _arguments(
            "--output",
            str(output),
            "--compare-helicity-sums",
            "--multiplicities",
            "2",
        ),
        output,
    )

    rendered = profiling._composition_arguments(
        _arguments("--render", "--output", str(output)), output
    )

    assert rendered.compare_helicity_sums is True
    assert profiling._helicity_workload(rendered) == "sum"
    assert profiling._pdf_filename(rendered) == "summary_plots_final_helicity_sum.pdf"


def test_resume_manifest_allows_midrun_resource_limit_decreases(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    initial = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--memory-limit-gib",
        "800",
        "--time-limit-seconds",
        "360000",
    )
    lowered = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--memory-limit-gib",
        "200",
        "--time-limit-seconds",
        "36000",
    )
    status = _arguments(
        "--status",
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--memory-limit-gib",
        "100",
        "--time-limit-seconds",
        "18000",
    )

    profiling._create_or_resume_manifest(initial, output)
    manifest = profiling._create_or_resume_manifest(lowered, output)

    assert manifest["identity"]["resources"] == {
        "candidate_optimization_cores": 1,
        "per_cell_generation_timeout_seconds": 36000.0,
        "per_cell_memory_limit_gib": 200.0,
        "per_cell_runtime_timeout_seconds": 36000.0,
    }
    assert profiling.status_payload(status)["requested_multiplicities"] == [2]


def test_existing_manifest_history_does_not_expand_current_multiplicity_request(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    broad = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "--lines",
        "reference-fft",
    )
    profiling._create_or_resume_manifest(broad, output)
    lower = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "3",
        "--lines",
        "reference-fft",
    )

    manifest = profiling._create_or_resume_manifest(lower, output)
    work = profiling._phase_work_items(lower, output, 1)
    plan = profiling.dry_run_plan(lower)

    assert manifest["requested_multiplicities"] == [2, 3]
    assert manifest["fill_history_multiplicities"] == list(range(2, 10))
    assert [item.final_multiplicity for item in work] == [2, 3]
    assert [job["n"] for job in plan["shards"]["gg-reference"]["jobs"]] == [2, 3]


def test_line_groups_track_history_but_schedule_only_requested_work(
    tmp_path: Path,
) -> None:
    output = tmp_path / "line-groups"
    common = ("--output", str(output), "--madgraph-root", str(tmp_path / "mg5"))
    reference = _arguments(
        *common,
        "--multiplicities",
        "2",
        "--lines",
        "reference-fft",
    )
    first = profiling._create_or_resume_manifest(reference, output)
    assert first["requested_line_groups"] == ["reference-fft"]
    reference_plan = profiling.dry_run_plan(reference)
    assert {
        name for name, shard in reference_plan["shards"].items() if shard["scheduled"]
    } == {"gg-reference"}
    assert reference_plan["madgraph"]["applicable"] is False

    otf = _arguments(
        *common,
        "--multiplicities",
        "3",
        "--lines",
        "pyamplicol-otf",
    )
    expanded = profiling._create_or_resume_manifest(otf, output)
    assert expanded["requested_multiplicities"] == [3]
    assert expanded["requested_line_groups"] == ["pyamplicol-otf"]
    assert expanded["fill_history_multiplicities"] == [2, 3]
    assert expanded["line_group_history"] == [
        "reference-fft",
        "pyamplicol-otf",
    ]
    otf_plan = profiling.dry_run_plan(otf)
    assert {
        name for name, shard in otf_plan["shards"].items() if shard["scheduled"]
    } == {"gg-reference", "gg-otf", "ddbar-amplicol", "ddbar-otf"}
    assert otf_plan["madgraph"]["applicable"] is False
    assert otf_plan["shards"]["gg-recurrence"]["argv"] is None

    with_madgraph = _arguments(
        *common,
        "--multiplicities",
        "4",
        "--lines",
        "madgraph",
    )
    final = profiling._create_or_resume_manifest(with_madgraph, output)
    assert final["requested_multiplicities"] == [4]
    assert final["requested_line_groups"] == ["madgraph"]
    assert final["fill_history_multiplicities"] == [2, 3, 4]
    assert final["line_group_history"] == [
        "reference-fft",
        "pyamplicol-otf",
        "madgraph",
    ]
    madgraph_plan = profiling.dry_run_plan(with_madgraph)
    assert madgraph_plan["madgraph"]["applicable"] is True
    assert madgraph_plan["madgraph"]["argv"] is not None
    assert madgraph_plan["shards"]["ddbar-recurrence"]["scheduled"] is True


def test_concrete_line_selector_schedules_only_requested_fft_variant(
    tmp_path: Path,
) -> None:
    arguments = _arguments(
        "--dry-run",
        "--output",
        str(tmp_path / "line-mode"),
        "--multiplicities",
        "2",
        "--lines",
        "recurrence-fft",
    )

    plan = profiling.dry_run_plan(arguments)
    scheduled = {name for name, shard in plan["shards"].items() if shard["scheduled"]}
    gg = plan["shards"]["gg-recurrence"]
    ddbar = plan["shards"]["ddbar-recurrence"]

    assert plan["requested_line_groups"] == ["recurrence-fft"]
    assert scheduled == {
        "gg-reference",
        "ddbar-amplicol",
        "gg-recurrence",
        "ddbar-recurrence",
    }
    assert gg["modes"] == ["reference-fft", "recurrence-fft"]
    assert gg["owned_modes"] == ["recurrence-fft"]
    assert ddbar["modes"] == ["amplicol", "recurrence-fft"]
    assert ddbar["owned_modes"] == ["recurrence-fft"]
    assert gg["argv"] is not None
    assert "--mode recurrence-fft" in gg["shell_command"]
    assert "--mode recurrence-direct" not in gg["shell_command"]
    assert plan["shards"]["gg-otf"]["scheduled"] is False


def test_process_family_selector_limits_scheduled_shards_and_madgraph(
    tmp_path: Path,
) -> None:
    arguments = _arguments(
        "--dry-run",
        "--output",
        str(tmp_path / "family-mode"),
        "--multiplicities",
        "2",
        "--families",
        "gg",
        "--lines",
        "recurrence-fft",
        "madgraph",
        "--madgraph-root",
        str(tmp_path / "mg5"),
    )

    plan = profiling.dry_run_plan(arguments)
    scheduled = {name for name, shard in plan["shards"].items() if shard["scheduled"]}
    madgraph_argv = plan["madgraph"]["argv"]

    assert plan["requested_process_families"] == ["gg"]
    assert scheduled == {"gg-reference", "gg-recurrence"}
    assert plan["shards"]["ddbar-amplicol"]["scheduled"] is False
    assert plan["madgraph"]["process_families"] == ["gg"]
    assert madgraph_argv is not None
    assert madgraph_argv[madgraph_argv.index("--family") + 1] == "gg"


def test_process_family_selector_rejects_inapplicable_line() -> None:
    arguments = _arguments(
        "--families",
        "ddbar",
        "--lines",
        "reference-fft",
    )
    with pytest.raises(profiling.ProfilingError, match="no applicable"):
        profiling._validate_arguments(arguments)


def test_madgraph_completed_ignores_overlay_when_not_requested(tmp_path: Path) -> None:
    output = tmp_path / "reference-only-with-overlay"
    overlay = profiling._madgraph_overlay_path(output)
    overlay.parent.mkdir(parents=True)
    overlay.write_text(
        json.dumps(
            {
                "host": profiling.madgraph.measurement_host_identity(),
                "policy": {
                    "final_state_multiplicities": [2],
                    "maximum_measured_multiplicity": 6,
                    "family_maximum_measured_multiplicity": {
                        "gg": 5,
                        "ddbar": 6,
                    },
                    "higher_multiplicity_policy": "not-applicable-protocol-scope",
                },
                "runtime_series": {
                    "gg": {
                        "madgraph-standalone": {
                            "2": {
                                "status": "measured",
                                "warm_seconds_per_point": 1.0e-6,
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--lines",
        "reference-fft",
    )

    assert profiling._madgraph_completed(arguments, output) == 0


def test_retry_invalidates_failed_selected_cell_without_dependency_count(
    tmp_path: Path,
) -> None:
    output = tmp_path / "retry"
    initial = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "7",
        "--families",
        "gg",
        "--lines",
        "recurrence-direct",
        "--memory-limit-gib",
        "100",
    )
    retry = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "7",
        "--families",
        "gg",
        "--lines",
        "recurrence-direct",
        "--memory-limit-gib",
        "160",
        "--retry",
    )
    profiling._create_or_resume_manifest(initial, output)
    reference = profiling.SHARD_BY_NAME["gg-reference"]
    recurrence = profiling.SHARD_BY_NAME["gg-recurrence"]
    reference_cell = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 7) | {
        "status": "measured",
        "generation_seconds": 1.0,
        "warm_seconds_per_point": 1.0e-6,
        "max_rss_kib": 1024,
    }
    failed_cell = study._cell_base("gg", study.MODE_BY_KEY["recurrence-direct"], 7) | {
        "status": "failed",
        "failure_category": "memory-limit",
        "failure_reason": "memory cap",
        "censors_higher_multiplicities": True,
    }
    reference_report = study.compose_report(
        profiling._shard_arguments(initial, output, reference),
        {"gg": {"reference-fft": {"7": reference_cell}}},
    )
    recurrence_cells = {
        "gg": {
            "reference-fft": {"7": reference_cell},
            "recurrence-direct": {"7": failed_cell},
        }
    }
    recurrence_aggregate = study.compose_report(
        profiling._shard_arguments(initial, output, recurrence),
        recurrence_cells,
    )
    recurrence_cell_report = study.compose_report(
        profiling._shard_arguments(initial, output, recurrence, 7),
        recurrence_cells,
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, reference),
        reference_report,
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, recurrence),
        recurrence_aggregate,
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, recurrence, 7),
        recurrence_cell_report,
    )

    manifest = profiling._create_or_resume_manifest(retry, output)
    removed = profiling._apply_retry_invalidation(retry, output)
    master = profiling._publish_master(retry, output)
    per_cell = json.loads(
        profiling._shard_report_path(output, recurrence, 7).read_text(
            encoding="utf-8"
        )
    )
    plan = profiling.dry_run_plan(retry)

    assert manifest["identity"]["resources"]["per_cell_memory_limit_gib"] == 160.0
    assert removed == 1
    assert profiling._phase_work_items(retry, output, 1) == ()
    assert [
        (work.shard.name, work.final_multiplicity)
        for work in profiling._phase_work_items(retry, output, 2)
    ] == [("gg-recurrence", 7)]
    assert profiling._selected_completed_cells(retry, master, multiplicities=(7,)) == 0
    assert profiling._selected_pending_cells(retry, master, multiplicities=(7,)) == 1
    assert (
        per_cell["cells"]["gg"]["reference-fft"]["7"]["status"] == "measured"
    )
    assert "7" not in per_cell["cells"]["gg"]["recurrence-direct"]
    assert plan["shards"]["gg-reference"]["scheduled"] is False
    assert plan["shards"]["gg-reference"]["jobs"] == []
    assert [job["n"] for job in plan["shards"]["gg-recurrence"]["jobs"]] == [7]


def test_overwrite_group_replaces_each_mode_only_when_its_worker_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "overwrite"
    initial = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "6",
        "--families",
        "ddbar",
        "--lines",
        "pyamplicol-recurrence",
    )
    overwrite = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "6",
        "--families",
        "ddbar",
        "--lines",
        "pyamplicol-recurrence",
        "--cores",
        "2",
        "--poll-seconds",
        "0.001",
        "--overwrite",
    )
    profiling._create_or_resume_manifest(initial, output)
    profiling._create_or_resume_manifest(overwrite, output)
    dependency = profiling.SHARD_BY_NAME["ddbar-amplicol"]
    recurrence = profiling.SHARD_BY_NAME["ddbar-recurrence"]

    def measured(mode: str, generation_seconds: float) -> dict[str, object]:
        return study._cell_base("ddbar", study.MODE_BY_KEY[mode], 6) | {
            "status": "measured",
            "generation_seconds": generation_seconds,
            "warm_seconds_per_point": generation_seconds / 1000.0,
            "max_rss_kib": 1024,
        }

    amplicol = measured("amplicol", 1.0)
    old = {
        "recurrence-direct": measured("recurrence-direct", 11.0),
        "recurrence-fft": measured("recurrence-fft", 12.0),
    }
    dependency_report = study.compose_report(
        profiling._shard_arguments(initial, output, dependency),
        {"ddbar": {"amplicol": {"6": amplicol}}},
    )
    aggregate = study.compose_report(
        profiling._shard_arguments(initial, output, recurrence),
        {
            "ddbar": {
                "amplicol": {"6": amplicol},
                **{mode: {"6": cell} for mode, cell in old.items()},
            }
        },
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, dependency), dependency_report
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, recurrence), aggregate
    )
    legacy = study.compose_report(
        profiling._shard_arguments(initial, output, recurrence, 6),
        {
            "ddbar": {
                "amplicol": {"6": amplicol},
                **{mode: {"6": cell} for mode, cell in old.items()},
            }
        },
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, recurrence, 6), legacy
    )
    works = profiling._phase_work_items(overwrite, output, 2)
    assert [work.owned_mode for work in works] == [
        "recurrence-direct",
        "recurrence-fft",
    ]
    for work in works:
        per_mode = study.compose_report(
            profiling._work_arguments(initial, output, work),
            {
                "ddbar": {
                    "amplicol": {"6": amplicol},
                    work.owned_mode: {"6": old[work.owned_mode]},
                }
            },
        )
        profiling._write_json_atomic(
            profiling._work_report_path(output, work), per_mode
        )

    overwrite_pending = {profiling._work_identity(work) for work in works}
    initial_master = profiling._publish_master(overwrite, output)
    assert (
        profiling._selected_completed_cells(
            overwrite,
            initial_master,
            multiplicities=(6,),
            ignored_cells=overwrite_pending,
        )
        == 0
    )
    assert (
        profiling._selected_pending_cells(
            overwrite,
            initial_master,
            multiplicities=(6,),
            ignored_cells=overwrite_pending,
        )
        == 2
    )
    assert all(profiling._work_report_path(output, work).is_file() for work in works)

    launched: list[str] = []

    class FakeProcess:
        def __init__(self, work: profiling.ShardWork) -> None:
            self.pid = 23456 + len(launched)
            self.returncode: int | None = None
            self.work = work

        def poll(self) -> int | None:
            if self.returncode is None:
                replacement = measured(
                    self.work.owned_mode,
                    101.0 if self.work.owned_mode == "recurrence-direct" else 102.0,
                )
                report = study.compose_report(
                    profiling._work_arguments(overwrite, output, self.work),
                    {
                        "ddbar": {
                            "amplicol": {"6": amplicol},
                            self.work.owned_mode: {"6": replacement},
                        }
                    },
                )
                profiling._write_json_atomic(
                    profiling._work_report_path(output, self.work), report
                )
                self.returncode = 0
            return self.returncode

    class FakeDashboard:
        def update(self, _completed, *, phase, active) -> None:
            assert phase == "phase 2"

    def launch(_arguments, _output, work: profiling.ShardWork):
        seeded = json.loads(
            profiling._work_report_path(output, work).read_text(encoding="utf-8")
        )
        assert "6" not in seeded["cells"]["ddbar"][work.owned_mode]
        if work.owned_mode == "recurrence-direct":
            fft_work = next(
                candidate
                for candidate in works
                if candidate.owned_mode == "recurrence-fft"
            )
            fft_report = json.loads(
                profiling._work_report_path(output, fft_work).read_text(
                    encoding="utf-8"
                )
            )
            assert (
                fft_report["cells"]["ddbar"]["recurrence-fft"]["6"][
                    "generation_seconds"
                ]
                == 12.0
            )
            legacy_report = json.loads(
                profiling._shard_report_path(output, recurrence, 6).read_text(
                    encoding="utf-8"
                )
            )
            assert "6" not in legacy_report["cells"]["ddbar"]["recurrence-direct"]
            assert (
                legacy_report["cells"]["ddbar"]["recurrence-fft"]["6"][
                    "generation_seconds"
                ]
                == 12.0
            )
        launched.append(work.owned_mode)
        return profiling.ActiveJob(
            shard=work.shard,
            final_multiplicity=work.final_multiplicity,
            owned_mode=work.owned_mode,
            command=(),
            process=FakeProcess(work),
            stdout=io.BytesIO(),
            stderr=io.BytesIO(),
            sampler=object(),
            claimed_cores=profiling._claimed_cores(overwrite, work),
            detail="",
        )

    original_prepare = profiling._prepare_overwrite_work
    original_seed = profiling._seed_cell_inputs

    def prepare(_arguments, _output, work: profiling.ShardWork):
        removed = original_prepare(_arguments, _output, work)
        aggregate_report = json.loads(
            profiling._shard_report_path(output, recurrence).read_text(
                encoding="utf-8"
            )
        )
        assert "6" not in aggregate_report["cells"]["ddbar"][work.owned_mode]
        legacy_report = json.loads(
            profiling._shard_report_path(output, recurrence, 6).read_text(
                encoding="utf-8"
            )
        )
        assert "6" not in legacy_report["cells"]["ddbar"][work.owned_mode]
        assert not profiling._work_cell_root(output, work).exists()
        stale_sources = []
        for path in (output / "shards" / recurrence.name).rglob("report.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            cell = payload.get("cells", {}).get("ddbar", {}).get(
                work.owned_mode, {}
            ).get("6")
            if cell is not None:
                stale_sources.append((str(path), cell.get("generation_seconds")))
        assert stale_sources == []
        merged_report = profiling._load_shard_report(
            overwrite, output, recurrence, required=True
        )
        merged_cell = merged_report["cells"]["ddbar"][work.owned_mode].get("6")
        assert merged_cell is None, (
            merged_cell.get("status"),
            merged_cell.get("generation_seconds"),
            merged_cell.get("failure_reason"),
        )
        return removed

    def seed(_arguments, _output, work: profiling.ShardWork):
        original_seed(_arguments, _output, work)
        seeded = json.loads(
            profiling._work_report_path(output, work).read_text(encoding="utf-8")
        )
        assert "6" not in seeded["cells"]["ddbar"][work.owned_mode]

    monkeypatch.setattr(profiling, "_prepare_overwrite_work", prepare)
    monkeypatch.setattr(profiling, "_seed_cell_inputs", seed)
    monkeypatch.setattr(profiling, "_launch_shard", launch)

    profiling._phase(
        overwrite,
        output,
        2,
        FakeDashboard(),
        overwrite_pending=overwrite_pending,
    )

    assert launched == ["recurrence-direct", "recurrence-fft"]
    assert overwrite_pending == set()
    merged = profiling._load_shard_report(
        overwrite, output, recurrence, required=True
    )
    assert (
        merged["cells"]["ddbar"]["recurrence-direct"]["6"][
            "generation_seconds"
        ]
        == 101.0
    )
    assert (
        merged["cells"]["ddbar"]["recurrence-fft"]["6"][
            "generation_seconds"
        ]
        == 102.0
    )


def test_overwrite_and_retry_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _arguments("--overwrite", "--retry")


def test_narrow_line_selector_reuses_existing_broader_shard_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "line-mode-resume"
    shard = profiling.SHARD_BY_NAME["gg-recurrence"]
    broad = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--lines",
        "pyamplicol-recurrence",
    )
    narrow = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--lines",
        "recurrence-fft",
    )
    cells = {"gg": {}}
    for mode in ("reference-fft", "recurrence-direct", "recurrence-fft"):
        cells["gg"][mode] = {
            "2": study._cell_base("gg", study.MODE_BY_KEY[mode], 2)
            | {
                "status": "measured",
                "generation_seconds": 1.0,
                "warm_seconds_per_point": 1.0e-6,
                "max_rss_kib": 1024,
            }
        }
    report = study.compose_report(
        profiling._shard_arguments(broad, output, shard, 2),
        cells,
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard, 2),
        report,
    )

    loaded = profiling._load_shard_report(narrow, output, shard, required=True)
    family = loaded["cells"]["gg"]

    assert set(family) == {"reference-fft", "recurrence-fft"}
    assert family["recurrence-fft"]["2"]["status"] == "measured"


def test_broad_line_selector_tops_up_existing_narrow_shard_reports(
    tmp_path: Path,
) -> None:
    output = tmp_path / "line-mode-top-up"
    shard = profiling.SHARD_BY_NAME["gg-recurrence"]
    direct = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--families",
        "gg",
        "--lines",
        "recurrence-direct",
    )
    fft = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--families",
        "gg",
        "--lines",
        "recurrence-fft",
    )
    broad = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--families",
        "gg",
        "--lines",
        "pyamplicol-recurrence",
    )
    cells = {"gg": {}}
    for mode in ("reference-fft", "recurrence-direct", "recurrence-fft"):
        cells["gg"][mode] = {
            "2": study._cell_base("gg", study.MODE_BY_KEY[mode], 2)
            | {
                "status": "measured",
                "generation_seconds": 1.0,
                "warm_seconds_per_point": 1.0e-6,
                "max_rss_kib": 1024,
            }
        }
    direct_aggregate = study.compose_report(
        profiling._shard_arguments(direct, output, shard),
        {
            "gg": {
                mode: cells["gg"][mode]
                for mode in ("reference-fft", "recurrence-direct")
            }
        },
    )
    fft_per_cell = study.compose_report(
        profiling._shard_arguments(fft, output, shard, 2),
        {
            "gg": {
                mode: cells["gg"][mode]
                for mode in ("reference-fft", "recurrence-fft")
            }
        },
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard),
        direct_aggregate,
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard, 2),
        fft_per_cell,
    )

    loaded = profiling._load_shard_report(broad, output, shard, required=True)
    family = loaded["cells"]["gg"]

    assert set(family) == {"reference-fft", "recurrence-direct", "recurrence-fft"}
    assert family["recurrence-direct"]["2"]["status"] == "measured"
    assert family["recurrence-fft"]["2"]["status"] == "measured"


def test_mode_cell_report_run_root_does_not_block_shard_merge(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mode-cell-run-root"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--families",
        "ddbar",
        "--lines",
        "otf-direct",
        "--compare-helicity-sums",
    )
    shard = profiling.SHARD_BY_NAME["ddbar-otf"]
    work = profiling.ShardWork(shard, 2, "otf-direct")
    amplicol_cell = study._cell_base(
        "ddbar", study.MODE_BY_KEY["amplicol"], 2, sum_helicities=True
    ) | {
        "status": "measured",
    }
    otf_cell = study._cell_base(
        "ddbar", study.MODE_BY_KEY["otf-direct"], 2, sum_helicities=True
    ) | {
        "status": "measured",
    }
    report = study.compose_report(
        profiling._work_arguments(arguments, output, work),
        {"ddbar": {"amplicol": {"2": amplicol_cell}, "otf-direct": {"2": otf_cell}}},
    )
    profiling._write_json_atomic(profiling._work_report_path(output, work), report)

    loaded = profiling._load_shard_report(arguments, output, shard, required=True)

    assert loaded["cells"]["ddbar"]["otf-direct"]["2"]["status"] == "measured"


def test_output_render_rebuilds_master_from_manifest_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "render-history"
    shard = profiling.SHARD_BY_NAME["gg-recurrence"]
    broad = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--families",
        "gg",
        "--lines",
        "pyamplicol-recurrence",
    )
    narrow = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--families",
        "gg",
        "--lines",
        "recurrence-direct",
    )
    profiling._create_or_resume_manifest(broad, output)
    profiling._create_or_resume_manifest(narrow, output)
    cells = {"gg": {}}
    for mode in ("reference-fft", "recurrence-direct", "recurrence-fft"):
        cells["gg"][mode] = {
            "2": study._cell_base("gg", study.MODE_BY_KEY[mode], 2)
            | {
                "status": "measured",
                "generation_seconds": 1.0,
                "warm_seconds_per_point": 1.0e-6,
                "max_rss_kib": 1024,
            }
        }
    per_cell = study.compose_report(
        profiling._shard_arguments(broad, output, shard, 2),
        cells,
    )
    partial_aggregate = study.compose_report(
        profiling._shard_arguments(narrow, output, shard),
        {
            "gg": {
                mode: cells["gg"][mode]
                for mode in ("reference-fft", "recurrence-direct")
            }
        },
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard, 2),
        per_cell,
    )
    profiling._write_json_atomic(
        profiling._shard_report_path(output, shard),
        partial_aggregate,
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
            _write_fake_pdf_output(command)

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)

    profiling.render_snapshot(_arguments("--render", "--output", str(output)))

    assert rendered["cells"]["gg"]["recurrence-direct"]["2"]["status"] == "measured"
    assert rendered["cells"]["gg"]["recurrence-fft"]["2"]["status"] == "measured"


def test_line_groups_reject_duplicates() -> None:
    arguments = _arguments(
        "--lines",
        "pyamplicol-otf",
        "pyamplicol-otf",
    )
    with pytest.raises(profiling.ProfilingError, match="--lines"):
        profiling._validate_arguments(arguments)


def test_reference_fft_root_is_explicit_in_identity_and_child_commands(
    tmp_path: Path,
) -> None:
    reference_root = tmp_path / "reference-fft"
    arguments = _arguments(
        "--output",
        str(tmp_path / "run"),
        "--multiplicities",
        "2",
        "--lines",
        "reference-fft",
        "--reference-fft-root",
        str(reference_root),
    )

    plan = profiling.dry_run_plan(arguments)
    assert plan["identity"]["tools"]["reference_fft_root"] == str(reference_root)
    command = plan["shards"]["gg-reference"]["argv"]
    assert command is not None
    option = command.index("--reference-fft-root")
    assert command[option + 1] == str(reference_root)


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


def _write_fake_pdf_output(command: tuple[str, ...]) -> None:
    output = Path(command[command.index("--output") + 1])
    output.write_bytes(b"pdf")
    output.with_suffix(".json").write_text("{}\n", encoding="utf-8")


def test_render_forwards_recola_results_to_plotter_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "render-recola"
    source = tmp_path / "live.json"
    recola_results = tmp_path / "recola.json"
    source.write_text(json.dumps(_partial_report(output)), encoding="utf-8")
    recola_results.write_text("{}\n", encoding="utf-8")
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
            _write_fake_pdf_output(command)

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_madgraph_render_source",
        lambda *_args: pytest.fail("external report must not inherit output MG data"),
    )

    profiling.render_snapshot(
        _arguments(
            "--render",
            "--output",
            str(output),
            "--campaign-report",
            str(source),
            "--recola-results",
            str(recola_results),
            "--main-y-range",
            "1e-6",
            "1e3",
            "--ratio-y-range",
            "0",
            "5",
            "--ratio-y-scale",
            "linear",
            "--main-include-lines",
            "recola",
            "madgraph",
            "--main-veto-lines",
            "pyamplicol-recurrence",
            "--ratio-include-lines",
            "recola",
            "--ratio-veto-lines",
            "madgraph",
        )
    )

    plot_command, pdf_command = calls
    assert plot_command[plot_command.index("--recola-results") + 1] == str(
        recola_results.resolve(strict=False)
    )
    assert plot_command[plot_command.index("--main-y-range") + 1 :][0:2] == (
        "1e-06",
        "1000.0",
    )
    assert plot_command[plot_command.index("--ratio-y-range") + 1 :][0:2] == (
        "0.0",
        "5.0",
    )
    assert plot_command[plot_command.index("--ratio-y-scale") + 1] == "linear"
    assert plot_command[plot_command.index("--main-include-lines") + 1 :][0:2] == (
        "recola",
        "madgraph",
    )
    assert plot_command[plot_command.index("--main-veto-lines") + 1] == (
        "pyamplicol-recurrence"
    )
    assert plot_command[plot_command.index("--ratio-include-lines") + 1] == "recola"
    assert plot_command[plot_command.index("--ratio-veto-lines") + 1] == "madgraph"
    assert "--recola-results" not in pdf_command
    assert "--main-y-range" not in pdf_command
    assert "--ratio-y-range" not in pdf_command
    assert "--ratio-y-scale" not in pdf_command
    assert "--main-include-lines" not in pdf_command
    assert "--main-veto-lines" not in pdf_command
    assert "--ratio-include-lines" not in pdf_command
    assert "--ratio-veto-lines" not in pdf_command


def test_render_axis_options_are_validated() -> None:
    arguments = _arguments("--render", "--main-y-range", "0", "1")

    with pytest.raises(profiling.ProfilingError, match="logarithmic y-axis"):
        profiling._validate_arguments(arguments)

    arguments = _arguments(
        "--render",
        "--ratio-y-scale",
        "linear",
        "--ratio-y-range",
        "0",
        "2",
    )

    profiling._validate_arguments(arguments)


def test_render_line_filters_are_render_only_and_deduplicated() -> None:
    arguments = _arguments("--main-include-lines", "recola")
    with pytest.raises(profiling.ProfilingError, match="valid only with --render"):
        profiling._validate_arguments(arguments)

    arguments = _arguments("--render", "--main-include-lines", "recola", "recola")
    with pytest.raises(profiling.ProfilingError, match="duplicates"):
        profiling._validate_arguments(arguments)


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
            _write_fake_pdf_output(command)

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


def test_run_checked_suppresses_success_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    profiling._run_checked(
        (
            sys.executable,
            "-c",
            "print('generated/path.png')",
        )
    )

    assert capsys.readouterr().out == ""


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
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(b"png")
        else:
            _write_fake_pdf_output(command)

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    arguments = _arguments("--render")

    profiling.render_snapshot(arguments)

    assert profiling._canonical_pdf_path(arguments).read_bytes() == b"pdf"
    assert profiling._canonical_pdf_path(arguments).with_suffix(".json").is_file()


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
            rendered_cell.update(frozen["cells"]["ddbar"]["recurrence-direct"]["2"])
            plot_root = Path(command[3])
            plot_root.mkdir(parents=True)
            for family in ("gg", "ddbar"):
                for metric in ("generation", "warm-runtime", "rss"):
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(b"png")
        else:
            _write_fake_pdf_output(command)

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
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(b"png")
        else:
            _write_fake_pdf_output(command)

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
                    "family_maximum_measured_multiplicity": {
                        "gg": 5,
                        "ddbar": 6,
                    },
                    "higher_multiplicity_policy": ("not-applicable-protocol-scope"),
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
                    (plot_root / f"fullcolor-{family}-{metric}.png").write_bytes(b"png")
        else:
            _write_fake_pdf_output(command)

    monkeypatch.setattr(profiling, "_attach_partial_overlay", attach)
    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    arguments = _arguments("--render")
    requested = set(profiling._selection(arguments))

    assert profiling._render_inventory(primary_report, requested=requested)[0] == 70
    assert profiling._render_inventory(composite_report, requested=requested)[0] == 65
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
        for final_multiplicity, cell in family_cells["madgraph-standalone"].items()
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
            _write_fake_pdf_output(command)

    monkeypatch.setattr(profiling, "_run_checked", fake_run)
    arguments = _arguments("--render", "--output", str(output))

    profiling.render_snapshot(arguments)

    assert attached == [progress_path]


def test_renderer_preflight_has_actionable_direct_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(profiling, "_command_available", lambda _command: True)
    monkeypatch.setattr(
        profiling.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})(),
    )
    with pytest.raises(profiling.ProfilingError) as raised:
        profiling._preflight_renderer(_arguments())
    message = str(raised.value)
    assert "pip install" in message
    assert "-e" not in message
    for requirement in profiling.RENDER_REQUIREMENTS:
        assert requirement in message


def test_renderer_requirements_match_the_project_extra() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)

    assert (
        tuple(project["project"]["optional-dependencies"]["fft-profiling"])
        == profiling.RENDER_REQUIREMENTS
    )


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


def _available_preflight_status(tmp_path: Path) -> dict[str, object]:
    return {
        "python": {"available": True, "path": sys.executable},
        "cxx": {"available": True, "path": "c++"},
        "fortran": {"available": True, "path": "gfortran"},
        "amplicol_root": {
            "compatible": True,
            "path": str(tmp_path / "amplicol"),
            "probe_available": True,
            "probe": str(tmp_path / "amplicol" / "amplicol_color_probe"),
        },
        "reference_fft_root": {
            "compatible": True,
            "path": str(tmp_path / "reference-fft"),
        },
        "madgraph_root": {
            "compatible": True,
            "path": str(tmp_path / "mg5"),
        },
    }


def test_preflight_skips_pyamplicol_runtime_for_reference_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(
        "--output",
        str(tmp_path / "run"),
        "--lines",
        "reference-fft",
    )
    monkeypatch.setattr(
        profiling,
        "_dependency_status",
        lambda _arguments: _available_preflight_status(tmp_path),
    )
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_preflight_pyamplicol_runtime",
        lambda _arguments: pytest.fail(
            "reference-only scan should not import pyAmpliCol"
        ),
    )

    profiling._preflight(arguments)


def test_preflight_checks_pyamplicol_runtime_for_ddbar_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(
        "--output",
        str(tmp_path / "run"),
        "--lines",
        "amplicol",
    )
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(
        profiling,
        "_dependency_status",
        lambda _arguments: _available_preflight_status(tmp_path),
    )
    monkeypatch.setattr(profiling, "_preflight_renderer", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_preflight_pyamplicol_runtime",
        lambda runtime_arguments: calls.append(runtime_arguments),
    )

    profiling._preflight(arguments)

    assert calls == [arguments]


def test_pyamplicol_preflight_uses_checkout_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/already/configured")
    arguments = _arguments()

    environment = profiling._python_source_environment(arguments)

    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(ROOT / "src"),
        "/already/configured",
    ]


def test_pyamplicol_preflight_reports_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_environment: dict[str, str] = {}

    def fake_run(*_args, **kwargs):
        captured_environment.update(kwargs["env"])
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "candidate wheel source checkout is unavailable",
            },
        )()

    monkeypatch.setattr(profiling.subprocess, "run", fake_run)

    with pytest.raises(profiling.ProfilingError) as raised:
        profiling._preflight_pyamplicol_runtime(_arguments())

    message = str(raised.value)
    assert "just dev-install" in message
    assert "candidate wheel source checkout is unavailable" in message
    assert captured_environment["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")


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
                "host": {key: current[key] for key in ("system", "machine", "python")},
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
        "GOODHEL pruning remains enabled.",
    ]


def test_madgraph_source_freeze_depends_only_on_source_cells(tmp_path: Path) -> None:
    output = tmp_path / "mg-source"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--lines",
        "madgraph",
    )
    gg_source = study._cell_base("gg", study.MODE_BY_KEY["reference-fft"], 2) | {
        "status": "measured"
    }
    ddbar_source = study._cell_base(
        "ddbar", study.MODE_BY_KEY["recurrence-fft"], 2
    ) | {"status": "measured"}
    unrelated = study._cell_base("gg", study.MODE_BY_KEY["otf-direct"], 2) | {
        "status": "measured",
        "generation_seconds": 1.0,
    }
    cells = {
        "gg": {
            "reference-fft": {"2": gg_source},
            "otf-direct": {"2": unrelated},
        },
        "ddbar": {"recurrence-fft": {"2": ddbar_source}},
    }
    master = study.compose_report(
        profiling._master_arguments(arguments, output),
        cells,
    )
    master["measurement_host"] = profiling.madgraph.measurement_host_identity()
    profiling._write_json_atomic(profiling._master_report_path(output), master)

    path = profiling._freeze_madgraph_source(arguments, output)
    frozen = json.loads(path.read_text(encoding="utf-8"))

    assert frozen["kind"] == profiling.madgraph.SOURCE_KIND
    assert set(frozen["cells"]["gg"]) == {"reference-fft"}
    assert set(frozen["cells"]["ddbar"]) == {"recurrence-fft"}

    master["cells"]["gg"]["otf-direct"]["2"]["generation_seconds"] = 2.0
    profiling._write_json_atomic(profiling._master_report_path(output), master)

    assert profiling._freeze_madgraph_source(arguments, output) == path


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
    steps: list[str] = []

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
    def fake_phase(_arguments, _output, phase, _dashboard, **_kwargs):
        phases.append(phase)
        steps.append(f"phase-{phase}")

    monkeypatch.setattr(profiling, "_phase", fake_phase)
    monkeypatch.setattr(profiling, "_completed_cells", lambda _report: 0)
    monkeypatch.setattr(
        profiling, "_selected_pending_cells", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(
        profiling, "_selected_master_complete", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(profiling, "_madgraph_source_ready", lambda *_args: True)
    madgraph_steps: list[str] = []

    def fake_start_madgraph(*_args):
        steps.append("madgraph-start")
        madgraph_steps.append("start")
        return None

    monkeypatch.setattr(
        profiling,
        "_start_madgraph_overlay",
        fake_start_madgraph,
    )
    monkeypatch.setattr(
        profiling,
        "_run_madgraph",
        lambda *_args: pytest.fail("MadGraph should not block the campaign"),
    )
    expected = output / "render" / "current" / "summary_plots_final_helicity_sum.pdf"
    render_calls: list[bool] = []

    def fake_render(_arguments, *, renderer_preflight=True):
        render_calls.append(renderer_preflight)
        return expected

    monkeypatch.setattr(profiling, "render_snapshot", fake_render)

    result = profiling.run_campaign(arguments)

    assert result == expected
    assert phases == [1, 2]
    assert steps == ["phase-1", "madgraph-start", "phase-2"]
    assert madgraph_steps == ["start"]
    assert render_calls == [False, False, False]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity"]["scan"]["helicity_workload"] == "sum"


def test_campaign_without_madgraph_line_group_skips_madgraph_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "reference-only"
    arguments = _arguments(
        "--output",
        str(output),
        "--multiplicities",
        "2",
        "--lines",
        "reference-fft",
    )
    report = {"status": "running", "cells": {}}
    monkeypatch.setattr(profiling, "_preflight", lambda _arguments: None)
    monkeypatch.setattr(
        profiling,
        "_publish_master",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(profiling, "_phase", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(profiling, "_completed_cells", lambda _report: 0)
    monkeypatch.setattr(
        profiling, "_selected_pending_cells", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(
        profiling, "_selected_master_complete", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        profiling,
        "_freeze_madgraph_source",
        lambda *_args: pytest.fail("MadGraph source should not be frozen"),
    )
    monkeypatch.setattr(
        profiling,
        "_run_madgraph",
        lambda *_args: pytest.fail("MadGraph should not run"),
    )
    expected = output / "render" / "current" / "summary_plots_final.pdf"
    monkeypatch.setattr(
        profiling,
        "render_snapshot",
        lambda *_args, **_kwargs: expected,
    )

    assert profiling.run_campaign(arguments) == expected


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
                    "family_maximum_measured_multiplicity": {
                        "gg": 5,
                        "ddbar": 6,
                    },
                    "higher_multiplicity_policy": ("not-applicable-protocol-scope"),
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
        profiling._matching_madgraph_overlay(arguments, output, require_exact=False)
        == overlay
    )
    assert profiling._madgraph_render_source(arguments, output) == overlay

    legacy_payload["host"]["machine"] = "foreign-machine"
    overlay.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert (
        profiling._matching_madgraph_overlay(arguments, output, require_exact=False)
        is None
    )

    legacy_payload["host"] = {
        key: current[key] for key in ("system", "machine", "python")
    } | {"node_sha256": "malformed-modern-fingerprint"}
    overlay.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert (
        profiling._matching_madgraph_overlay(arguments, output, require_exact=False)
        is None
    )


def test_partial_render_can_reuse_ordered_subset_madgraph_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "expanded-progress"
    progress = profiling._madgraph_progress_path(output)
    progress.parent.mkdir(parents=True)
    progress.write_text("{}", encoding="utf-8")
    payload = {
        "status": "running",
        "host": profiling.madgraph.measurement_host_identity(),
        "policy": {
            "final_state_multiplicities": [2, 3, 4, 5, 6],
            "helicity_workload": "fixed",
            "selected_process_families": ["gg", "ddbar"],
            "family_maximum_measured_multiplicity": {
                "gg": 5,
                "ddbar": 6,
            },
        },
        "runtime_series": {},
    }
    arguments = _arguments("--output", str(output), "--multiplicities", "4")
    monkeypatch.setattr(
        profiling,
        "_requested_multiplicities",
        lambda _arguments, _output: (2, 3, 4, 5, 6, 7, 8, 9),
    )
    monkeypatch.setattr(
        profiling.madgraph,
        "load_runtime_progress",
        lambda _path: payload,
    )

    assert profiling._matching_madgraph_progress(arguments, output) == payload
    assert profiling._madgraph_render_source(arguments, output) == progress

    payload["policy"]["final_state_multiplicities"] = [6, 5, 4]
    assert profiling._matching_madgraph_progress(arguments, output) is None
