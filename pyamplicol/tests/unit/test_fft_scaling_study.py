# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "fft_scaling_study.py"
SPEC = importlib.util.spec_from_file_location("test_fft_scaling_study_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scaling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scaling
SPEC.loader.exec_module(scaling)


def test_watchdog_peak_rss_accepts_a_rounded_zero_for_short_children() -> None:
    stderr = (
        "memory-watchdog: command finished exit=0 peak_rss=0.000 GiB "
        "peak_physical_footprint=0.001 GiB peak_guard=0.001 GiB"
    )

    assert scaling._parse_watchdog_peak_rss_kib(stderr) == 0


def _pinned_summed_probe_source() -> bytes:
    return (
        "program probe\n"
        + scaling._SUMMED_AMPLICOL_FERMION_GUARD
        + "             continue\n"
        + scaling._SUMMED_AMPLICOL_FALLBACK_GUARD
        + "             continue\n"
        + scaling._SUMMED_AMPLICOL_HELPER_ANCHOR
        + "end program probe\n"
    ).encode("utf-8")


@pytest.mark.parametrize("body_raises", (False, True))
def test_summed_amplicol_probe_overlay_restores_clean_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_raises: bool,
) -> None:
    repository = tmp_path / "amplicol"
    repository.mkdir()
    source = repository / "amplicol_color_library_probe.f03"
    original = _pinned_summed_probe_source()
    source.write_bytes(original)
    source.chmod(0o640)
    original_stat = source.stat()
    monkeypatch.setattr(
        scaling.legacy_amplicol,
        "validate_checkout",
        lambda checked: checked.samefile(repository),
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol,
        "expected_revision",
        lambda: "pinned-test-revision",
    )
    evidence = tmp_path / "evidence"
    captured: dict[str, object] = {}

    def exercise() -> None:
        with scaling._summed_amplicol_probe_source_overlay(
            repository, evidence
        ) as record:
            captured.update(record)
            patched = source.read_bytes()
            assert patched != original
            assert b"fermion_relabel_is_unique" in patched
            snapshot = evidence / "amplicol_color_library_probe.f03"
            assert snapshot.read_bytes() == patched
            if body_raises:
                raise RuntimeError("body failed")

    if body_raises:
        with pytest.raises(RuntimeError, match="body failed"):
            exercise()
    else:
        exercise()

    restored_stat = source.stat()
    assert source.read_bytes() == original
    assert restored_stat.st_mode == original_stat.st_mode
    assert restored_stat.st_size == original_stat.st_size
    assert restored_stat.st_mtime_ns == original_stat.st_mtime_ns
    manifest = json.loads((evidence / "provenance.json").read_text())
    assert manifest["revision"] == "pinned-test-revision"
    assert manifest["restoration"]["status"] == "restored"
    assert manifest["restoration"]["body_exit"] == (
        "exception" if body_raises else "success"
    )
    assert captured["original_sha256"] != captured["patched_sha256"]


def _amplicol_timing_output(points: int, total_seconds: float) -> str:
    return (
        "AmpliCol colour probe\n"
        f"points {points}\n"
        "generation setup 0.125 50.0% outside-runtime\n"
        f"total {total_seconds:.17g} 100.0%\n"
    )


def test_amplicol_warm_sampling_grows_and_restarts_one_fixed_repetition_set() -> None:
    calls: list[tuple[int, int, int]] = []

    def run_sample(sample_set: int, sample: int, repetitions: int) -> str:
        calls.append((sample_set, sample, repetitions))
        if sample_set == 1:
            total = 0.30 if sample == 1 else 0.20
        else:
            total = 0.36 + sample * 0.001
        return _amplicol_timing_output(repetitions, total)

    measured = scaling._collect_amplicol_warm_samples(
        seed_points=10,
        seed_total_seconds=0.10,
        target_seconds=0.25,
        run_sample=run_sample,
    )

    assert calls[:2] == [(1, 1, 27), (1, 2, 27)]
    assert calls[2:] == [(2, sample, 36) for sample in range(1, 11)]
    assert measured.repetitions == 36
    assert measured.sample_set_attempts == 2
    assert len(measured.samples_seconds) == 10
    assert min(measured.sample_totals_seconds) >= 0.25
    assert statistics.median(measured.samples_seconds) == pytest.approx(0.3655 / 36)


def test_amplicol_timing_parser_retains_the_aggregate_duration() -> None:
    output = _amplicol_timing_output(40, 0.4)

    assert scaling._parse_amplicol_timing_aggregate(output) == (0.125, 0.4, 40)
    assert scaling._parse_amplicol_timing(output) == (0.125, 0.01, 40)


def test_amplicol_cell_uses_fresh_fixed_samples_and_program_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "amplicol"
    repository.mkdir()
    event = tmp_path / "event"
    event.write_text("mock event\n", encoding="utf-8")
    helicity = (-1, 1, -1, 1)
    momenta = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (500.0, 500.0, 0.0, 0.0),
        (500.0, -500.0, 0.0, 0.0),
    )
    entry = scaling.legacy_amplicol.ProcessEntry(1, 1, (1, -1, 1, -1), (1, 2, 3, 4))
    monkeypatch.setattr(scaling, "_read_event", lambda _path: (momenta, helicity))
    monkeypatch.setattr(
        scaling.legacy_amplicol, "parse_process_file", lambda _path: [entry]
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol, "process_pdgs", lambda _process: entry.process_pdgs
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol,
        "select_generated_process_entry",
        lambda *_args, **_kwargs: (entry, [entry]),
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol,
        "_permutation",
        lambda source, _target: tuple(range(len(source))),
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol,
        "_ordered_binary64_momenta",
        lambda _source, _target, values: values,
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol,
        "_parse_probe_output",
        lambda _output: SimpleNamespace(value=2.0),
    )

    calls: list[dict[str, object]] = []

    def fake_run_watched(
        command: tuple[str, ...], **kwargs: object
    ) -> scaling.performance.WatchedCompletedProcess:
        log_path = Path(str(kwargs["log_path"]))
        environment = dict(kwargs["environment"])
        watchdog_report = Path(str(kwargs["watchdog_report_path"]))
        calls.append(
            {
                "log_path": log_path,
                "environment": environment,
                "watchdog_report": watchdog_report,
            }
        )
        if log_path.name == "process-generation.json":
            return scaling.performance.WatchedCompletedProcess(
                command,
                0,
                "",
                "memory-watchdog: peak_rss=0.001 GiB peak_guard=0.009 GiB\n",
                elapsed_seconds=1.0,
                log_write_seconds=0.0,
                timed_out=False,
                timeout_cleanup="not-required",
                peak_rss_kib=1_500,
                peak_guard_kib=9_000,
            )

        if log_path.name == "probe.json":
            points, total, child_rss = 10, 0.10, 1_100
        elif "warm-sample-set-01" in log_path.name:
            sample = int(log_path.stem.rsplit("-", 1)[1])
            points = 27
            total = 0.30 if sample == 1 else 0.20
            child_rss = 1_200 + 100 * sample
        else:
            sample = int(log_path.stem.rsplit("-", 1)[1])
            points = 36
            total = 0.36 + sample * 0.001
            child_rss = 1_500 + 100 * sample
        stderr = (
            f"{scaling.RSS_MARKER} {child_rss}\n"
            "memory-watchdog: peak_rss=0.008 GiB peak_guard=0.010 GiB\n"
        )
        return scaling.performance.WatchedCompletedProcess(
            command,
            0,
            _amplicol_timing_output(points, total),
            stderr,
            elapsed_seconds=0.5,
            log_write_seconds=0.0,
            timed_out=False,
            timeout_cleanup="not-required",
            peak_rss_kib=8_000,
            peak_guard_kib=10_000,
        )

    monkeypatch.setattr(scaling.performance, "_run_watched", fake_run_watched)
    arguments = scaling._parser().parse_args(
        (
            "--run-id",
            "amplicol-protocol",
            "--family",
            "ddbar",
            "--mode",
            "amplicol",
            "--batch-size",
            "1",
            "--python",
            sys.executable,
            "--amplicol-repository",
            str(repository),
        )
    )

    cell = scaling._amplicol_cell(
        arguments=arguments,
        family="ddbar",
        final_multiplicity=2,
        events=(event,),
        helicity=helicity,
        baseline=None,
        run_root=tmp_path / "run",
        environment={},
    )

    probe_calls = [
        call
        for call in calls
        if Path(call["log_path"]).name != "process-generation.json"
    ]
    assert len(calls) == 14
    assert len(probe_calls) == 13
    assert len({Path(call["log_path"]) for call in calls}) == len(calls)
    assert len({Path(call["watchdog_report"]) for call in calls}) == len(calls)
    assert probe_calls[0]["environment"] == {
        "AMPICOL_COLOR_PROBE_TARGET_RUNTIME_S": "0.25"
    }
    assert all(
        "AMPICOL_COLOR_PROBE_TARGET_RUNTIME_S" not in call["environment"]
        for call in probe_calls[1:]
    )
    assert cell["warm_repetitions"] == [36] * 10
    assert cell["warm_sample_set_attempts"] == 2
    assert min(cell["warm_sample_totals_seconds"]) >= 0.25
    assert cell["metrics"]["warm_seconds_per_point"] == pytest.approx(0.3655 / 36)
    assert cell["metrics"]["max_rss_kib"] == 2_500
    assert cell["resource_peaks_kib"] == {
        "generation": 1_500,
        "generation_guard": 9_000,
        "runtime_self": 2_500,
        "runtime_watchdog": 8_000,
        "runtime_guard": 10_000,
        "runtime_watchdog_included_in_plotted_rss": False,
    }


def test_publication_defaults_use_strict_one_hour_and_30_gib_frontiers() -> None:
    arguments = scaling._parser().parse_args(())
    plan = scaling.dry_run_plan(arguments)

    assert arguments.generation_timeout == 3600.0
    assert arguments.runtime_timeout == 3600.0
    assert arguments.memory_limit_gib == 30.0
    assert (
        "median of 10 independent process-CPU aggregates"
        in plan["measurement"]["warm_timing_metric"]["amplicol"]
    )
    assert (
        "Fortran runtime child peak RSS" in plan["measurement"]["amplicol_rss_metric"]
    )
    assert scaling._require_bounded_rss(30 * 1024**2 - 1, 30.0) > 0
    with pytest.raises(scaling.StudyError, match=r"strict <30 GiB"):
        scaling._require_bounded_rss(30 * 1024**2, 30.0)


def test_publication_generation_uses_normal_optimized_defaults() -> None:
    command = scaling._candidate_generation_command(
        python="python-test",
        family="ddbar",
        final_multiplicity=7,
        mode=scaling.MODE_BY_KEY["compiled-direct"],
        artifact=Path("artifact"),
        batch_size=1,
    )

    assert "--no-numerical-current-reuse" not in command
    assert "--no-emit-api-bundle" not in command
    assert "--workers" not in command
    assert "--progress" not in command
    assert "--log-level" not in command
    assert "generation.validation.enabled=false" not in command
    assert "generation.validation.post_build_validation=false" not in command
    assert "model.cache=false" not in command
    assert "evaluator.optimization.cores=1" not in command
    assert "evaluator.jit.optimization_level=2" not in command
    assert not any(
        value.startswith("process.selected_source_helicities=") for value in command
    )
    assert command[command.index("--set") + 1] == "evaluator.batch_size=1"


def test_timed_command_uses_python_getrusage_wrapper_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scaling.sys, "platform", "linux")

    command, normalizer = scaling._timed_command(("payload",), python="python-test")

    assert command == (
        "python-test",
        str(scaling.Path(scaling.__file__).resolve()),
        "_time-rss",
        "payload",
    )
    assert normalizer is None


def _higher_n_arguments() -> object:
    return scaling._parser().parse_args(
        (
            "--run-id",
            "higher-n-scalar",
            "--min-n",
            "7",
            "--max-n",
            "10",
            "--family",
            "gg",
            "--family",
            "ddbar",
            "--mode",
            "reference-fft",
            "--mode",
            "amplicol",
            "--mode",
            "recurrence-fft",
            "--batch-size",
            "1",
            "--generation-timeout",
            "3600",
            "--runtime-timeout",
            "3600",
            "--memory-limit-gib",
            "20",
        )
    )


def test_higher_n_scalar_policy_selects_only_requested_curves() -> None:
    arguments = _higher_n_arguments()
    scaling._validate_arguments(arguments)

    plan = scaling.dry_run_plan(arguments)
    assert plan["final_state_multiplicities"] == [7, 8, 9, 10]
    assert plan["total_external_particles"] == [9, 10, 11, 12]
    assert plan["measurement"]["warm_benchmark_batch_size"] == 1
    assert plan["process_families"]["gg"]["modes"] == [
        "reference-fft",
        "amplicol",
        "recurrence-fft",
    ]
    assert plan["process_families"]["ddbar"]["modes"] == [
        "amplicol",
        "recurrence-fft",
    ]
    assert set(scaling._empty_report(arguments)["cells"]) == {"gg", "ddbar"}


def test_fft_steering_adds_applicable_pairs_and_records_provenance() -> None:
    arguments = scaling._parser().parse_args(
        (
            "--family",
            "gg",
            "--mode",
            "reference-fft",
            "--mode",
            "recurrence-direct",
            "--mode",
            "compiled-direct",
            "--fft",
        )
    )
    scaling._validate_arguments(arguments)

    plan = scaling.dry_run_plan(arguments)
    assert plan["fft_enabled"] is True
    assert plan["process_families"]["gg"]["modes"] == [
        "reference-fft",
        "recurrence-direct",
        "recurrence-fft",
        "compiled-direct",
    ]
    assert plan["selected_pyamplicol_color_contractions"] == {
        "recurrence": ["direct", "symmetric-group-fft"],
        "compiled": ["direct"],
    }
    assert plan["measurement"]["generation_helicity_coverage"] == "all"
    assert plan["measurement"]["warm_fixed_helicity"] is True
    assert "fixed_helicity" not in plan["measurement"]
    assert plan["measurement"]["compiled_fft_enabled"] is False


def test_helicity_sum_policy_has_separate_identity_and_count_semantics() -> None:
    fixed_arguments = scaling._parser().parse_args(("--multiplicity", "2"))
    sum_arguments = scaling._parser().parse_args(
        ("--multiplicity", "2", "--compare-helicity-sums")
    )

    fixed = scaling.dry_run_plan(fixed_arguments)["measurement"]
    summed = scaling.dry_run_plan(sum_arguments)["measurement"]

    assert fixed["warm_fixed_helicity"] is True
    assert "helicity_workload" not in fixed
    assert summed["helicity_workload"] == "sum"
    assert summed["warm_fixed_helicity"] is False
    assert summed["warm_helicity_sum"] is True
    assert "candidate_timed_helicity_contract" in summed
    semantics = summed["helicity_count_semantics"]
    assert "complete physical API axis" in semantics["candidate"]
    assert "analytic-nonzero subset" in semantics["reference_fft"]
    assert "does not expose that retained count" in semantics["amplicol"]
    assert "hel_fac filter" in semantics["amplicol"]
    assert "generated-library" in summed["generation_metric"]["amplicol"]


def test_compiled_fft_is_explicitly_not_applicable_to_publication() -> None:
    cell = scaling._compiled_fft_not_applicable_cell(
        family="gg",
        mode=scaling.MODE_BY_KEY["compiled-fft"],
        final_multiplicity=6,
    )

    assert cell is not None
    assert cell["status"] == "not-applicable"
    assert cell["censors_higher_multiplicities"] is False
    assert cell["applicability"] == {
        "compiled_direct": True,
        "compiled_fft": False,
        "reason_code": "publication-requires-all-runtime-helicities",
    }
    assert "all runtime helicities" in cell["failure_reason"]
    assert "compiled-direct remains supported" in cell["failure_reason"]


def test_sum_nonmeasurement_cells_keep_summed_workload_markers() -> None:
    compiled = scaling._compiled_fft_not_applicable_cell(
        family="gg",
        mode=scaling.MODE_BY_KEY["compiled-fft"],
        final_multiplicity=6,
        sum_helicities=True,
    )
    otf = scaling.otf_protocol_scope_cell(
        family="gg",
        mode=scaling.MODE_BY_KEY["otf-direct"],
        final_multiplicity=7,
        sum_helicities=True,
    )

    for cell in (compiled, otf):
        assert cell is not None
        assert cell["helicity_workload"] == "sum"
        assert cell["warm_fixed_helicity"] is False
        assert cell["warm_helicity_sum"] is True


def test_fixed_helicity_otf_frontier_matches_summed_n6_limit() -> None:
    mode = scaling.MODE_BY_KEY["otf-direct"]

    for sum_helicities in (False, True):
        assert (
            scaling.otf_protocol_scope_cell(
                family="gg",
                mode=mode,
                final_multiplicity=6,
                sum_helicities=sum_helicities,
            )
            is None
        )
    fixed_n7 = scaling.otf_protocol_scope_cell(
        family="gg",
        mode=mode,
        final_multiplicity=7,
    )
    assert fixed_n7 is not None
    assert fixed_n7["failed_at_n"] == 7
    assert "n<=6" in fixed_n7["failure_reason"]


def test_publication_artifact_requires_complete_reusable_helicities(
    tmp_path: Path,
) -> None:
    physics_path = tmp_path / "processes" / "gg_n2" / "physics.json"
    physics_path.parent.mkdir(parents=True)
    physics = {
        "coverage": {"helicities": "complete"},
        "extensions": {
            "runtime_selectors": {
                "axes": {
                    "helicity": {
                        "generation_coverage": "complete",
                        "generation_selection": {},
                        "runtime_contract": "complete-reusable",
                    }
                },
                "generation_specialized_axes": [],
            }
        },
        "helicities": [{"id": f"h:{index}"} for index in range(16)],
    }
    physics_path.write_text(json.dumps(physics), encoding="utf-8")

    assert scaling._validate_publication_candidate_artifact(tmp_path, "gg_n2") == 16

    physics["extensions"]["runtime_selectors"]["axes"]["helicity"] = {
        "generation_coverage": "selected",
        "generation_selection": {"1": 1},
        "runtime_contract": "generation-specialized",
    }
    physics_path.write_text(json.dumps(physics), encoding="utf-8")
    with pytest.raises(scaling.StudyError, match="not reusable"):
        scaling._validate_publication_candidate_artifact(tmp_path, "gg_n2")


def _publication_otf_execution() -> dict[str, object]:
    return {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-on-the-fly-execution",
        "selector_policy": {
            "color_coverage": "contracted",
            "selector_census": {"physical_helicity_count": 4},
        },
        "runtime_metadata": {
            "process_seed_identity": {
                "external_sources": [
                    {
                        "source_slot": 0,
                        "states": [
                            {"state_index": 0, "public_helicity": -1},
                            {"state_index": 1, "public_helicity": 1},
                        ],
                    },
                    {
                        "source_slot": 1,
                        "states": [
                            {"state_index": 0, "public_helicity": -1},
                            {"state_index": 1, "public_helicity": 1},
                        ],
                    },
                ]
            }
        },
    }


def _write_publication_otf_execution(
    root: Path, execution: object, *, raw: str | None = None
) -> None:
    path = root / "processes" / "gg_n2" / "execution.json"
    path.parent.mkdir(parents=True)
    path.write_text(raw if raw is not None else json.dumps(execution), encoding="utf-8")


def test_publication_artifact_accepts_otf_seed_helicity_census(tmp_path: Path) -> None:
    _write_publication_otf_execution(tmp_path, _publication_otf_execution())

    assert scaling._validate_publication_candidate_artifact(tmp_path, "gg_n2") == 4


@pytest.mark.parametrize(
    "corruption",
    (
        "schema",
        "kind",
        "coverage",
        "small-census",
        "census-mismatch",
        "source-slot",
        "state-index",
        "public-helicity",
    ),
)
def test_publication_artifact_rejects_corrupt_otf_helicity_contract(
    tmp_path: Path, corruption: str
) -> None:
    execution = deepcopy(_publication_otf_execution())
    selector = execution["selector_policy"]
    assert isinstance(selector, dict)
    census = selector["selector_census"]
    assert isinstance(census, dict)
    runtime_metadata = execution["runtime_metadata"]
    assert isinstance(runtime_metadata, dict)
    identity = runtime_metadata["process_seed_identity"]
    assert isinstance(identity, dict)
    sources = identity["external_sources"]
    assert isinstance(sources, list)
    first_source = sources[0]
    second_source = sources[1]
    assert isinstance(first_source, dict) and isinstance(second_source, dict)
    first_states = first_source["states"]
    assert isinstance(first_states, list)

    if corruption == "schema":
        execution["schema_version"] = 2
    elif corruption == "kind":
        execution["kind"] = "pyamplicol-runtime-recurrence-execution"
    elif corruption == "coverage":
        selector["color_coverage"] = "complete"
    elif corruption == "small-census":
        census["physical_helicity_count"] = 1
    elif corruption == "census-mismatch":
        census["physical_helicity_count"] = 8
    elif corruption == "source-slot":
        second_source["source_slot"] = 0
    elif corruption == "state-index":
        assert isinstance(first_states[1], dict)
        first_states[1]["state_index"] = 0
    elif corruption == "public-helicity":
        assert isinstance(first_states[1], dict)
        first_states[1]["public_helicity"] = -1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(corruption)

    _write_publication_otf_execution(tmp_path, execution)
    with pytest.raises(scaling.StudyError):
        scaling._validate_publication_candidate_artifact(tmp_path, "gg_n2")


def test_publication_artifact_rejects_malformed_otf_execution(tmp_path: Path) -> None:
    _write_publication_otf_execution(tmp_path, {}, raw="{")

    with pytest.raises(scaling.StudyError):
        scaling._validate_publication_candidate_artifact(tmp_path, "gg_n2")


def test_higher_n_process_and_reference_seed_are_unbounded_above_n7() -> None:
    assert scaling.process_expression("gg", 8) == "g g > g g g g g g g g"
    assert scaling.process_expression("ddbar", 8) == "d d~ > d d~ g g g g g g"
    assert scaling.performance.generator_seed(12) == scaling.performance.BASE_SEED + 12
    with pytest.raises(scaling.performance.AcceptanceError):
        scaling.performance.generator_seed(3)


def test_only_resource_failures_censor_a_curve(tmp_path: Path) -> None:
    base = scaling._cell_base("gg", scaling.MODE_BY_KEY["recurrence-fft"], 8)
    resource = scaling._failure_cell(
        base,
        scaling._CellResourceLimitError(
            "candidate artifact generation exhausted the cold cap",
            category="generation-time-limit",
        ),
        tmp_path,
    )
    correctness = scaling._failure_cell(
        base,
        scaling.StudyError("candidate numerical comparison failed"),
        tmp_path,
    )
    legacy_runtime = scaling._failure_cell(
        base,
        scaling.legacy_report.ProfilingTimeLimitError(
            "legacy profiling stage exhausted its remaining budget"
        ),
        tmp_path,
    )

    assert resource["failure_category"] == "generation-time-limit"
    assert resource["censors_higher_multiplicities"] is True
    assert correctness["failure_category"] == "error"
    assert correctness["censors_higher_multiplicities"] is False
    assert legacy_runtime["failure_category"] == "runtime-time-limit"
    assert legacy_runtime["censors_higher_multiplicities"] is True


def _write_watched_command_record(
    path: Path,
    *,
    timed_out: bool,
    returncode: int,
    watchdog_report: Path | None = None,
    command: list[str] | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> None:
    command = command or ["amplicol_color_library_probe", "1"]
    watchdog_command = [
        sys.executable,
        str(scaling.performance.WATCHDOG),
        "--limit-gib",
        "30",
        "--",
        *command,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "command": command,
                "watchdog_command": watchdog_command,
                "elapsed_seconds": 0.75,
                "returncode": returncode,
                "stdout": stdout
                if stdout is not None
                else (
                    "Could not map colour row to generated-library integral 2\n"
                    "row permutation: 3 1 2 4\n"
                ),
                "stderr": stderr
                if stderr is not None
                else "timing metadata: runtime was below its deadline",
                "timed_out": timed_out,
                "timeout_cleanup": "not-required",
                "watchdog_report": (
                    None if watchdog_report is None else str(watchdog_report)
                ),
            }
        ),
        encoding="utf-8",
    )


def test_structural_command_failure_is_not_misclassified_as_a_resource_limit(
    tmp_path: Path,
) -> None:
    command_log = tmp_path / "commands" / "008-amplicol_color_library_probe.json"
    _write_watched_command_record(
        command_log,
        timed_out=False,
        returncode=1,
    )
    _write_watched_command_record(
        tmp_path / "commands" / "stale-timeout.json",
        timed_out=True,
        returncode=-15,
    )
    base = scaling._cell_base("ddbar", scaling.MODE_BY_KEY["amplicol"], 2)
    cell = scaling._failure_cell(
        base,
        scaling._CellCommandError(
            "AmpliCol generated-library runtime command failed: "
            f"command failed with status 1; see {command_log}",
            timeout_category="runtime-time-limit",
        ),
        tmp_path,
    )

    assert cell["failure_category"] == "error"
    assert cell["censors_higher_multiplicities"] is False


def test_legacy_amplicol_colour_init_segfault_censors_the_curve(
    tmp_path: Path,
) -> None:
    command_log = tmp_path / "amplicol" / "ddbar" / "n9" / "probe.json"
    _write_watched_command_record(
        command_log,
        command=["amplicol_color_probe", "1000000000", "2", "1", "full"],
        timed_out=False,
        returncode=245,
        stdout="Initialising amplitude for:\n",
        stderr=(
            "Program received signal SIGSEGV: Segmentation fault - invalid memory "
            "reference.\n"
            "#1 0x000000 in __amplitude_qcd_mod_MOD_init_col\n"
        ),
    )
    base = scaling._cell_base("ddbar", scaling.MODE_BY_KEY["amplicol"], 9)
    cell = scaling._failure_cell(
        base,
        scaling._CellCommandError(
            "AmpliCol setup/runtime probe failed: "
            f"command failed with status 245; see {command_log}",
            timeout_category="setup-or-runtime-time-limit",
        ),
        tmp_path,
    )

    assert (
        cell["failure_category"] == scaling.LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY
    )
    assert (
        cell["failure_reason_short"]
        == "legacy AmpliCol structural colour-init limit"
    )
    assert cell["censors_higher_multiplicities"] is True


def test_loaded_legacy_amplicol_colour_init_segfault_report_is_normalized(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "historical-aggregate"
    command_log = tmp_path / "isolated" / "cells" / "n9" / "probe.json"
    _write_watched_command_record(
        command_log,
        command=["amplicol_color_probe", "1000000000", "2", "1", "full"],
        timed_out=False,
        returncode=245,
        stdout="Initialising amplitude for:\n",
        stderr=(
            "Program received signal SIGSEGV: Segmentation fault - invalid memory "
            "reference.\n"
            "#1 0x000000 in __amplitude_qcd_mod_MOD_init_col\n"
        ),
    )
    report: dict[str, object] = {
        "cells": {
            "ddbar": {
                "amplicol": {
                    "9": {
                        "status": "failed",
                        "censors_higher_multiplicities": False,
                        "failure_category": "error",
                        "failure_reason": (
                            "AmpliCol setup/runtime probe failed: "
                            f"command failed with status 245; see {command_log}"
                        ),
                    }
                }
            }
        }
    }

    assert scaling.normalize_loaded_failure_cells(report, run_root) is True

    cell = report["cells"]["ddbar"]["amplicol"]["9"]  # type: ignore[index]
    assert (
        cell["failure_category"] == scaling.LEGACY_AMPLICOL_STRUCTURAL_LIMIT_CATEGORY
    )
    assert cell["censors_higher_multiplicities"] is True


def test_typed_watched_command_timeout_censors_the_curve(tmp_path: Path) -> None:
    command_log = tmp_path / "commands" / "probe.json"
    _write_watched_command_record(
        command_log,
        timed_out=True,
        returncode=-15,
    )
    base = scaling._cell_base("ddbar", scaling.MODE_BY_KEY["amplicol"], 2)
    cell = scaling._failure_cell(
        base,
        scaling._CellCommandError(
            f"runtime command failed; see {command_log}",
            timeout_category="runtime-time-limit",
        ),
        tmp_path,
    )

    assert cell["failure_category"] == "runtime-time-limit"
    assert cell["censors_higher_multiplicities"] is True


def test_candidate_generation_retries_symbolica_single_instance_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = scaling._parser().parse_args(
        ("--generation-timeout", "60", "--memory-limit-gib", "100")
    )
    mode = scaling.MODE_BY_KEY["recurrence-direct"]
    environment = {
        scaling.SYMBOLICA_GENERATION_LOCK_ENV: str(tmp_path / "symbolica.lock")
    }
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    def fake_run_watched(
        command: tuple[str, ...], **kwargs: object
    ) -> scaling.performance.WatchedCompletedProcess:
        calls.append(tuple(command))
        log_path = Path(str(kwargs["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            log_path.write_text(
                json.dumps(
                    {
                        "command": list(command),
                        "elapsed_seconds": 1.0,
                        "returncode": 134,
                        "stdout": scaling.SYMBOLICA_LICENSE_BUSY_FRAGMENT,
                        "stderr": "",
                        "timed_out": False,
                        "timeout_cleanup": "not-required",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raise scaling.performance.AcceptanceError("generation failed")
        log_path.write_text(
            json.dumps(
                {
                    "command": list(command),
                    "elapsed_seconds": 1.0,
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "timeout_cleanup": "not-required",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return scaling.performance.WatchedCompletedProcess(
            command,
            0,
            "",
            "",
            elapsed_seconds=1.0,
            log_write_seconds=0.0,
            timed_out=False,
            timeout_cleanup="not-required",
        )

    monkeypatch.setattr(scaling.performance, "_run_watched", fake_run_watched)
    monkeypatch.setattr(scaling.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = scaling._run_candidate_generation(
        arguments=arguments,
        family="ddbar",
        final_multiplicity=2,
        mode=mode,
        artifact=tmp_path / "cell" / "artifact",
        cell_root=tmp_path / "cell",
        environment=environment,
    )

    assert result.returncode == 0
    assert len(calls) == 2
    assert sleeps == [scaling.SYMBOLICA_BUSY_RETRY_SECONDS]


def test_symbolica_generation_lock_is_skipped_for_explicit_license() -> None:
    assert (
        scaling._symbolica_generation_lock_path({"SYMBOLICA_LICENSE": "configured"})
        is None
    )
    assert (
        scaling._symbolica_generation_lock_path(
            {"SYMBOLICA_LICENSE": "configured"}, force=True
        )
        is not None
    )


@pytest.mark.parametrize("with_report", [False, True])
def test_watchdog_memory_limit_outcome_censors_the_curve(
    tmp_path: Path, with_report: bool
) -> None:
    watchdog_report = tmp_path / "commands" / "probe.watchdog.json"
    if with_report:
        watchdog_report.parent.mkdir(parents=True)
        watchdog_report.write_text(
            json.dumps(
                {
                    "kind": scaling.memory_watchdog.WATCHDOG_REPORT_KIND,
                    "schema_version": scaling.memory_watchdog.WATCHDOG_REPORT_SCHEMA,
                    "complete": True,
                    "execution": {"outcome": "memory-limit-exceeded"},
                }
            ),
            encoding="utf-8",
        )
    command_log = tmp_path / "commands" / "probe.json"
    _write_watched_command_record(
        command_log,
        timed_out=False,
        returncode=scaling.memory_watchdog.MEMORY_LIMIT_EXIT_CODE,
        watchdog_report=watchdog_report if with_report else None,
    )
    base = scaling._cell_base("gg", scaling.MODE_BY_KEY["recurrence-fft"], 8)
    cell = scaling._failure_cell(
        base,
        scaling.StudyError(f"command failed; see {command_log}"),
        tmp_path,
    )

    assert cell["failure_category"] == "memory-limit"
    assert cell["censors_higher_multiplicities"] is True


def test_selection_validation_requires_oracles_and_supported_scalar_modes() -> None:
    missing_oracle = scaling._parser().parse_args(
        ("--family", "ddbar", "--mode", "recurrence-fft")
    )
    with pytest.raises(scaling.StudyError, match="require --mode amplicol"):
        scaling._validate_arguments(missing_oracle)

    compiled_fft_scalar = scaling._parser().parse_args(
        (
            "--family",
            "gg",
            "--mode",
            "reference-fft",
            "--mode",
            "compiled-fft",
            "--batch-size",
            "1",
        )
    )
    scaling._validate_arguments(compiled_fft_scalar)

    compiled_scalar = scaling._parser().parse_args(
        (
            "--family",
            "gg",
            "--mode",
            "reference-fft",
            "--mode",
            "compiled-direct",
            "--batch-size",
            "1",
        )
    )
    with pytest.raises(scaling.StudyError, match="not supported for compiled"):
        scaling._validate_arguments(compiled_scalar)


def test_correctness_failure_is_recorded_then_retried_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_root = tmp_path / "study"
    monkeypatch.setattr(scaling, "STUDY_ROOT", study_root)
    monkeypatch.setattr(scaling, "_study_environment", lambda _python: {})
    monkeypatch.setattr(
        scaling,
        "_acquire_campaign_lock",
        lambda: (tmp_path / "campaign.lock").open("a+", encoding="utf-8"),
    )
    monkeypatch.setattr(scaling.performance, "_load_reference_module", lambda: object())
    attempts = 0

    def fake_reference_cell(**kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise scaling.StudyError("reference numerical comparison failed")
        final_multiplicity = int(kwargs["final_multiplicity"])
        return scaling._cell_base(
            "gg", scaling.MODE_BY_KEY["reference-fft"], final_multiplicity
        ) | {"status": "measured"}

    monkeypatch.setattr(scaling, "_reference_cell", fake_reference_cell)
    common = (
        "--run-id",
        "retry-correctness",
        "--min-n",
        "7",
        "--max-n",
        "7",
        "--family",
        "gg",
        "--mode",
        "reference-fft",
    )
    first = scaling._parser().parse_args(common)
    scaling._validate_arguments(first)
    with pytest.raises(scaling.StudyError, match="repair it, then resume"):
        scaling._campaign(first)

    report_path = study_root / "runs" / "retry-correctness" / "report.json"
    stopped = json.loads(report_path.read_text(encoding="utf-8"))
    failed = stopped["cells"]["gg"]["reference-fft"]["7"]
    assert stopped["status"] == "stopped-correctness-failure"
    assert failed["status"] == "failed"
    assert failed["censors_higher_multiplicities"] is False

    resumed = scaling._parser().parse_args((*common, "--resume"))
    scaling._validate_arguments(resumed)
    completed = scaling._campaign(resumed)
    assert attempts == 2
    assert completed["status"] == "complete"
    assert completed["failure_count"] == 0
    assert completed["cells"]["gg"]["reference-fft"]["7"]["status"] == "measured"


def test_resume_retries_lower_resource_failure_below_retained_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study_root = tmp_path / "study"
    monkeypatch.setattr(scaling, "STUDY_ROOT", study_root)
    monkeypatch.setattr(scaling, "_study_environment", lambda _python: {})
    monkeypatch.setattr(
        scaling,
        "_acquire_campaign_lock",
        lambda: (tmp_path / "campaign.lock").open("a+", encoding="utf-8"),
    )
    monkeypatch.setattr(scaling.performance, "_load_reference_module", lambda: object())
    common = (
        "--run-id",
        "retry-lower-frontier",
        "--multiplicity",
        "2",
        "--multiplicity",
        "8",
        "--fill-multiplicity",
        "2",
        "--family",
        "gg",
        "--mode",
        "reference-fft",
        "--resume",
    )
    arguments = scaling._parser().parse_args(common)
    scaling._validate_arguments(arguments)
    report = scaling._empty_report(arguments)
    curve = report["cells"]["gg"]["reference-fft"]
    curve["2"] = scaling._cell_base(
        "gg", scaling.MODE_BY_KEY["reference-fft"], 2
    ) | {
        "status": "failed",
        "failure_category": "memory-limit",
        "failure_reason": "old lower-n resource observation",
        "censors_higher_multiplicities": True,
    }
    curve["8"] = scaling._cell_base(
        "gg", scaling.MODE_BY_KEY["reference-fft"], 8
    ) | {"status": "measured"}
    report_path = study_root / "runs" / "retry-lower-frontier" / "report.json"
    scaling.performance._write_report(report_path, report)
    attempts = 0

    def fake_reference_cell(**kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        final_multiplicity = int(kwargs["final_multiplicity"])
        assert final_multiplicity == 2
        return scaling._cell_base(
            "gg", scaling.MODE_BY_KEY["reference-fft"], final_multiplicity
        ) | {"status": "measured"}

    monkeypatch.setattr(scaling, "_reference_cell", fake_reference_cell)
    completed = scaling._campaign(arguments)

    assert attempts == 1
    assert completed["status"] == "complete"
    assert completed["cells"]["gg"]["reference-fft"]["2"]["status"] == "measured"
    assert completed["cells"]["gg"]["reference-fft"]["8"]["status"] == "measured"
    assert scaling.resource_frontier_inversions(completed) == ()


def test_candidate_public_cli_commands_and_profile_json_are_authoritative(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    momenta = tmp_path / "momenta.json"
    parameters = tmp_path / "parameters.json"
    helicity = (-1, -1, 1, 1)
    fixed_evaluate = scaling._candidate_evaluate_command(
        python="python",
        artifact=artifact,
        process="gg_n2",
        momenta=momenta,
        model_parameters=parameters,
        helicity=helicity,
        sum_helicities=False,
    )
    fixed_profile = scaling._candidate_profile_command(
        python="python",
        artifact=artifact,
        process="gg_n2",
        momenta=momenta,
        helicity=helicity,
        sum_helicities=False,
        target_seconds=0.25,
        batch_size=128,
    )
    summed_profile = scaling._candidate_profile_command(
        python="python",
        artifact=artifact,
        process="gg_n2",
        momenta=momenta,
        helicity=helicity,
        sum_helicities=True,
        target_seconds=0.25,
        batch_size=128,
    )

    assert fixed_evaluate[fixed_evaluate.index("--model-parameters") + 1] == str(
        parameters
    )
    assert fixed_evaluate[fixed_evaluate.index("--helicity") + 1] == "h:-1,-1,+1,+1"
    assert fixed_profile[fixed_profile.index("--helicity") + 1] == "h:-1,-1,+1,+1"
    assert "--helicity" not in summed_profile
    assert fixed_profile[fixed_profile.index("--target-runtime") + 1] == "0.25"
    assert fixed_profile[fixed_profile.index("--batch-size") + 1] == "128"
    assert fixed_profile[fixed_profile.index("--warmup-runs") + 1] == "2"
    assert fixed_profile[fixed_profile.index("--minimum-samples") + 1] == "10"
    assert all("fft_gluon_candidate_probe" not in token for token in fixed_profile)

    values = scaling._parse_candidate_evaluate_json(
        json.dumps([{"real": index + 1.0, "imag": 0.0} for index in range(10)])
    )
    assert values == [float(index) for index in range(1, 11)]

    config = {
        "batch_size": 128,
        "color_flow_ids": [],
        "helicity_ids": [],
        "minimum_samples": 10,
        "precision": 16,
        "target_runtime": 0.25,
        "warmup_runs": 2,
    }
    wall = 0.0005343891273355439
    payload = {
        "requested_config": config,
        "effective_config": config,
        "sample_count": 10,
        "repetitions_per_sample": 1,
        "wall_time_per_point": wall,
        "interrupted": False,
        "process_id": "gg_n2",
        "process_expression": scaling.process_expression("gg", 2),
        "environment": {
            "target": str(artifact),
            "execution_mode": "recurrence",
            "color_accuracy": "full",
            "batch_size": 128,
            "precision": 16,
            "selected_color_ids": [],
            "selected_helicity_ids": [],
            "completed_sample_count": 10,
            "measured_point_count": 1280,
            "interrupted": False,
        },
        "retained_full_cli_provenance": {"sentinel": True},
    }
    parsed = scaling._parse_candidate_profile_json(
        json.dumps(payload),
        artifact=artifact,
        family="gg",
        final_multiplicity=2,
        execution_mode="recurrence",
        helicity=helicity,
        sum_helicities=True,
        target_seconds=0.25,
        batch_size=128,
    )

    assert parsed["wall_time_per_point"] == wall
    assert parsed["retained_full_cli_provenance"] == {"sentinel": True}


def test_summed_reference_distinguishes_shared_and_timed_event_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tuple(tmp_path / f"shared-{index}.event" for index in range(10))
    timed = tuple(tmp_path / f"timed-{index}.event" for index in range(10))
    for path in (*shared, *timed):
        path.write_text("event\n", encoding="utf-8")
    metrics = scaling.performance.ReferenceMetrics(
        total_gluons=4,
        generator_seed=scaling.performance.generator_seed(4),
        backend=scaling.performance.REFERENCE_BACKEND,
        clean_build_scope=scaling.performance.REFERENCE_CLEAN_BUILD_SCOPE,
        clean_build_command_count=1,
        clean_build_seconds=1.0,
        setup_to_driver_seconds=1.0,
        initialization_seconds=0.1,
        first_pass_seconds=0.1,
        warm_samples_seconds=(0.01,) * 10,
        warm_median_seconds=0.01,
        max_rss_kib=1024,
        selected_helicity=(-1, -1, 1, 1),
        selected_path="mhv",
        event_paths=tuple(str(path) for path in shared),
        matrix_elements=(1.0,) * 10,
        helicity_workload="sum",
        helicity_coverage_count=16,
        timed_helicity_count=6,
        active_helicity_count=6,
        exhaustive_event_paths=tuple(str(path) for path in timed),
    )
    monkeypatch.setattr(
        scaling.performance,
        "_run_reference",
        lambda **_kwargs: scaling.performance._ReferenceRun(
            metrics=metrics, event_paths=shared
        ),
    )
    arguments = scaling._parser().parse_args(("--compare-helicity-sums",))

    cell = scaling._reference_cell(
        arguments=arguments,
        reference=object(),
        run_root=tmp_path,
        environment={},
        final_multiplicity=2,
    )

    assert cell["event_paths"] == [str(path) for path in shared]
    assert cell["shared_event_paths"] == [str(path) for path in shared]
    assert cell["timed_event_paths"] == [str(path) for path in timed]
    assert cell["reference"]["exhaustive_event_paths"] == tuple(
        str(path) for path in timed
    )
    report = {"cells": {"gg": {"reference-fft": {"2": cell}}}}
    events, helicity = scaling._inputs(report, tmp_path, "gg", 2)
    assert events == shared
    assert helicity == metrics.selected_helicity


@pytest.mark.parametrize(
    ("failed_phase", "expected_generation_calls", "expected_cli_calls"),
    (("interruption", 1, 3), ("profile", 1, 4), ("generation", 2, 2)),
)
def test_candidate_resume_reuses_only_a_successful_generation_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_phase: str,
    expected_generation_calls: int,
    expected_cli_calls: int,
) -> None:
    study_root = tmp_path / "study"
    final_multiplicity = 8
    run_id = f"resume-{failed_phase}"
    monkeypatch.setattr(scaling, "STUDY_ROOT", study_root)
    monkeypatch.setattr(scaling, "_study_environment", lambda _python: {})
    monkeypatch.setattr(
        scaling,
        "_acquire_campaign_lock",
        lambda: (tmp_path / "campaign.lock").open("a+", encoding="utf-8"),
    )
    monkeypatch.setattr(scaling.performance, "_load_reference_module", lambda: object())
    monkeypatch.setattr(
        scaling.performance,
        "_build_probe",
        lambda **_kwargs: pytest.fail(
            "publication candidate cells must not build the custom C++ probe"
        ),
    )
    monkeypatch.setattr(
        scaling,
        "_validate_publication_candidate_artifact",
        lambda _artifact, _process: 16,
    )
    factor = (
        scaling.performance.INITIAL_GLUON_AVERAGE_FACTOR
        * scaling.math.factorial(final_multiplicity)
        / (4.0 * scaling.math.pi * scaling.DEFAULT_ALPHA_S) ** final_multiplicity
    )
    event_paths = tuple(tmp_path / f"event-{index}" for index in range(10))
    for path in event_paths:
        path.write_text("mock event\n", encoding="utf-8")

    def fake_reference_cell(**_kwargs: object) -> dict[str, object]:
        return scaling._cell_base(
            "gg", scaling.MODE_BY_KEY["reference-fft"], final_multiplicity
        ) | {
            "status": "measured",
            "event_paths": [str(path) for path in event_paths],
            "helicity": [1] * (final_multiplicity + 2),
            "point_values": [factor] * 10,
        }

    generation_calls = 0
    cli_calls = 0
    failure_triggered = False
    evaluate_timeouts: list[float] = []
    watchdog_stderr = (
        "memory-watchdog: command finished exit=0 peak_rss=0.001 GiB "
        "peak_physical_footprint=0.001 GiB peak_guard=0.001 GiB\n"
    )

    def fake_run_watched(
        command: tuple[str, ...], **kwargs: object
    ) -> scaling.performance.WatchedCompletedProcess:
        nonlocal generation_calls, cli_calls, failure_triggered
        assert all("fft_gluon_candidate_probe" not in token for token in command)
        log_path = Path(str(kwargs["log_path"]))
        if log_path.name == "generation.json":
            generation_calls += 1
            artifact = log_path.parent / "artifact"
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "artifact.json").write_text("{}\n", encoding="utf-8")
            returncode = int(failed_phase == "generation" and not failure_triggered)
            failure_triggered = failure_triggered or returncode != 0
            log_path.write_text(
                json.dumps(
                    {
                        "command": list(command),
                        "elapsed_seconds": 1.0,
                        "returncode": returncode,
                        "stdout": "",
                        "stderr": watchdog_stderr,
                        "timed_out": False,
                        "timeout_cleanup": "not-required",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            if returncode != 0:
                raise scaling.performance.AcceptanceError("generation failed")
            output = ""
        elif log_path.name == "evaluate.json":
            cli_calls += 1
            evaluate_timeouts.append(float(kwargs["timeout_seconds"]))
            if failed_phase == "interruption" and not failure_triggered:
                failure_triggered = True
                raise KeyboardInterrupt
            output = json.dumps([{"real": 1.0, "imag": 0.0}] * 10)
        else:
            assert log_path.name == "profile.json"
            cli_calls += 1
            if failed_phase == "profile" and not failure_triggered:
                failure_triggered = True
                raise scaling.performance.AcceptanceError("profile failed")
            config = {
                "batch_size": 1,
                "color_flow_ids": [],
                "helicity_ids": ["h:" + ",".join(["+1"] * 10)],
                "minimum_samples": 10,
                "precision": 16,
                "target_runtime": 0.25,
                "warmup_runs": 2,
            }
            output = json.dumps(
                {
                    "requested_config": config,
                    "effective_config": config,
                    "sample_count": 10,
                    "repetitions_per_sample": 1,
                    "wall_time_per_point": 0.001,
                    "interrupted": False,
                    "process_id": "gg_n8",
                    "process_expression": scaling.process_expression("gg", 8),
                    "environment": {
                        "target": str(log_path.parent / "artifact"),
                        "execution_mode": "recurrence",
                        "color_accuracy": "full",
                        "batch_size": 1,
                        "precision": 16,
                        "selected_color_ids": [],
                        "selected_helicity_ids": config["helicity_ids"],
                        "completed_sample_count": 10,
                        "measured_point_count": 10,
                        "interrupted": False,
                    },
                }
            )
        return scaling.performance.WatchedCompletedProcess(
            command,
            0,
            output,
            watchdog_stderr,
            elapsed_seconds=1.0,
            log_write_seconds=0.0,
            timed_out=False,
            timeout_cleanup="not-required",
        )

    monkeypatch.setattr(scaling, "_reference_cell", fake_reference_cell)
    monkeypatch.setattr(
        scaling,
        "_write_candidate_cli_inputs",
        lambda *, cell_root, **_kwargs: (
            cell_root / "cli-momenta.json",
            cell_root / "cli-model-parameters.json",
        ),
    )
    monkeypatch.setattr(scaling.performance, "_run_watched", fake_run_watched)
    common = (
        "--run-id",
        run_id,
        "--min-n",
        str(final_multiplicity),
        "--max-n",
        str(final_multiplicity),
        "--family",
        "gg",
        "--mode",
        "reference-fft",
        "--mode",
        "recurrence-fft",
        "--batch-size",
        "1",
    )
    first = scaling._parser().parse_args(common)
    scaling._validate_arguments(first)
    if failed_phase == "interruption":
        with pytest.raises(KeyboardInterrupt):
            scaling._campaign(first)
    else:
        with pytest.raises(scaling.StudyError, match="repair it, then resume"):
            scaling._campaign(first)

    resumed = scaling._parser().parse_args((*common, "--resume"))
    scaling._validate_arguments(resumed)
    report = scaling._campaign(resumed)

    candidate_root = (
        study_root / "runs" / run_id / "candidate" / "recurrence-fft" / "gg"
    )
    assert generation_calls == expected_generation_calls
    assert cli_calls == expected_cli_calls
    assert evaluate_timeouts
    assert all(timeout == pytest.approx(3599.0) for timeout in evaluate_timeouts)
    assert not (candidate_root / "n8.interrupted-1").exists()
    assert not (candidate_root / ".n8.staging").exists()
    assert (candidate_root / "n8" / "artifact" / "artifact.json").is_file()
    measured = report["cells"]["gg"]["recurrence-fft"]["8"]
    assert measured["status"] == "measured"
    assert measured["metrics"]["warm_seconds_per_point"] == 0.001
    assert "runtime_profile" in measured
    assert "probe" not in measured


def test_disk_exhaustion_is_not_misreported_as_an_rss_censor(tmp_path: Path) -> None:
    cell = scaling._failure_cell(
        scaling._cell_base("gg", scaling.MODE_BY_KEY["recurrence-fft"], 8),
        OSError(28, "No space left on device"),
        tmp_path,
    )

    assert cell["failure_category"] == "disk-infrastructure-failure"
    assert cell["censors_higher_multiplicities"] is False


def _prepare_mocked_selected_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    study_root = tmp_path / "study"
    repository = tmp_path / "amplicol"
    repository.mkdir()
    (repository / "amplicol_color_probe").write_text("test probe\n", encoding="utf-8")
    monkeypatch.setattr(scaling, "STUDY_ROOT", study_root)
    monkeypatch.setattr(scaling, "_study_environment", lambda _python: {})
    monkeypatch.setattr(
        scaling,
        "_acquire_campaign_lock",
        lambda: (tmp_path / "campaign.lock").open("a+", encoding="utf-8"),
    )
    monkeypatch.setattr(scaling.performance, "_load_reference_module", lambda: object())
    monkeypatch.setattr(
        scaling.performance,
        "_build_probe",
        lambda **_kwargs: pytest.fail(
            "publication candidate cells must not build the custom C++ probe"
        ),
    )
    monkeypatch.setattr(
        scaling.legacy_amplicol, "validate_checkout", lambda _repository: None
    )
    return repository


def test_reference_resource_censor_skips_gg_dependencies_but_runs_ddbar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _prepare_mocked_selected_campaign(tmp_path, monkeypatch)
    calls = {"reference": 0, "amplicol-ddbar": 0, "candidate-ddbar": 0}

    def fake_reference_cell(**_kwargs: object) -> dict[str, object]:
        calls["reference"] += 1
        raise scaling._CellResourceLimitError(
            "reference generation reached its cell deadline",
            category="generation-time-limit",
        )

    def fake_inputs(
        _report: object, _run_root: Path, family: str, final_multiplicity: int
    ) -> tuple[tuple[Path, ...], tuple[int, ...]]:
        assert family == "ddbar"
        return (Path("event"),), scaling.fixed_ddbar_helicity(final_multiplicity)

    def fake_amplicol_cell(**kwargs: object) -> dict[str, object]:
        family = str(kwargs["family"])
        assert family == "ddbar"
        calls["amplicol-ddbar"] += 1
        final_multiplicity = int(kwargs["final_multiplicity"])
        return scaling._cell_base(
            family, scaling.MODE_BY_KEY["amplicol"], final_multiplicity
        ) | {"status": "measured"}

    def fake_candidate_cell(**kwargs: object) -> dict[str, object]:
        family = str(kwargs["family"])
        assert family == "ddbar"
        calls["candidate-ddbar"] += 1
        final_multiplicity = int(kwargs["final_multiplicity"])
        mode = kwargs["mode"]
        return scaling._cell_base(family, mode, final_multiplicity) | {
            "status": "measured"
        }

    monkeypatch.setattr(scaling, "_reference_cell", fake_reference_cell)
    monkeypatch.setattr(scaling, "_inputs", fake_inputs)
    monkeypatch.setattr(scaling, "_amplicol_cell", fake_amplicol_cell)
    monkeypatch.setattr(scaling, "_candidate_cell", fake_candidate_cell)
    common = (
        "--run-id",
        "dependency-censor",
        "--min-n",
        "7",
        "--max-n",
        "8",
        "--family",
        "gg",
        "--family",
        "ddbar",
        "--mode",
        "reference-fft",
        "--mode",
        "amplicol",
        "--mode",
        "recurrence-fft",
        "--amplicol-repository",
        str(repository),
    )
    first = scaling._parser().parse_args(common)
    scaling._validate_arguments(first)
    report = scaling._campaign(first)

    assert calls == {"reference": 1, "amplicol-ddbar": 2, "candidate-ddbar": 2}
    for n in ("7", "8"):
        for mode in ("amplicol", "recurrence-fft"):
            dependency = report["cells"]["gg"][mode][n]
            assert dependency["status"] == "skipped"
            assert dependency["failure_category"] == "dependency-unavailable"
            assert dependency["censors_higher_multiplicities"] is False
        assert report["cells"]["ddbar"]["amplicol"][n]["status"] == "measured"
        assert report["cells"]["ddbar"]["recurrence-fft"][n]["status"] == "measured"

    resumed = scaling._parser().parse_args((*common, "--resume"))
    scaling._validate_arguments(resumed)
    scaling._campaign(resumed)
    assert calls == {"reference": 1, "amplicol-ddbar": 2, "candidate-ddbar": 2}


def test_amplicol_gg_dense_index_preflight_censors_only_that_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _prepare_mocked_selected_campaign(tmp_path, monkeypatch)
    calls = {"reference": 0, "amplicol": 0, "candidate": 0}

    def fake_reference_cell(**kwargs: object) -> dict[str, object]:
        calls["reference"] += 1
        final_multiplicity = int(kwargs["final_multiplicity"])
        return scaling._cell_base(
            "gg", scaling.MODE_BY_KEY["reference-fft"], final_multiplicity
        ) | {"status": "measured"}

    def fake_inputs(
        _report: object, _run_root: Path, family: str, final_multiplicity: int
    ) -> tuple[tuple[Path, ...], tuple[int, ...]]:
        assert family == "gg"
        return (Path("event"),), (1,) * (final_multiplicity + 2)

    def fake_amplicol_cell(**_kwargs: object) -> dict[str, object]:
        calls["amplicol"] += 1
        raise AssertionError("structurally infeasible AmpliCol cell was launched")

    def fake_candidate_cell(**kwargs: object) -> dict[str, object]:
        calls["candidate"] += 1
        final_multiplicity = int(kwargs["final_multiplicity"])
        mode = kwargs["mode"]
        return scaling._cell_base("gg", mode, final_multiplicity) | {
            "status": "measured"
        }

    monkeypatch.setattr(scaling, "_reference_cell", fake_reference_cell)
    monkeypatch.setattr(scaling, "_inputs", fake_inputs)
    monkeypatch.setattr(scaling, "_amplicol_cell", fake_amplicol_cell)
    monkeypatch.setattr(scaling, "_candidate_cell", fake_candidate_cell)
    arguments = scaling._parser().parse_args(
        (
            "--run-id",
            "amplicol-preflight",
            "--min-n",
            "8",
            "--max-n",
            "9",
            "--family",
            "gg",
            "--mode",
            "reference-fft",
            "--mode",
            "amplicol",
            "--mode",
            "recurrence-fft",
            "--memory-limit-gib",
            "20",
            "--amplicol-repository",
            str(repository),
        )
    )
    scaling._validate_arguments(arguments)
    report = scaling._campaign(arguments)

    n7 = scaling._amplicol_gg_dense_index_preflight(7, 20.0)
    n8 = scaling._amplicol_gg_dense_index_preflight(8, 20.0)
    assert n7["color_orders"] == 40_320
    assert n7["feasible"] is True
    assert n8["color_orders"] == 362_880
    assert n8["lower_bound_bytes"] == 263_363_788_800
    assert n8["feasible"] is False
    assert calls == {"reference": 2, "amplicol": 0, "candidate": 2}
    first_skip = report["cells"]["gg"]["amplicol"]["8"]
    assert first_skip["status"] == "skipped"
    assert first_skip["failure_category"] == "structural-memory-limit"
    assert first_skip["censors_higher_multiplicities"] is True
    higher_skip = report["cells"]["gg"]["amplicol"]["9"]
    assert higher_skip["status"] == "skipped"
    assert higher_skip["failed_at_n"] == 8
    assert report["cells"]["gg"]["recurrence-fft"]["9"]["status"] == "measured"
