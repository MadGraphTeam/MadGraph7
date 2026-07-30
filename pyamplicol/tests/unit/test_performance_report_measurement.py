# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import (
    _baseline_matrix_element,
    _baseline_selector_contract,
    _require_nonzero_lc_all_flow_baseline,
    _reuse_artifact_for_measurement,
    _stable_runtime_identity,
    _validate_runtime_identity_postflight,
    failure_measurement,
    generated_artifact_from_measurement,
    measure_pyamplicol_cell,
    source_revision,
)
from tools.performance_report.models import Accuracy, ExecutionMode, ResultStatus
from tools.performance_report.phase_state import (
    WorkerPhaseChannel,
    WorkerPhaseReporter,
    read_worker_phase_state,
)
from tools.performance_report.runner import (
    LOADED_RUNTIME_PROFILE_COMMAND_PATH,
    PUBLIC_CLI_COMMAND_PATH,
    GeneratedArtifact,
    RunnerError,
    RunnerSettings,
    SelectorContract,
)


def _contract() -> SelectorContract:
    return SelectorContract(
        selected_color_flow_ids=("flow:1,2,3",),
        selected_color_words=((1, 2, 3),),
        all_flow_helicity_ids=("h:-1,+1,-1",),
        all_flow_source_helicities=((1, -1), (2, 1), (3, -1)),
        point_digest="a" * 64,
    )


def test_baseline_contract_and_matrix_element_are_strict() -> None:
    baseline = {
        "status": "ok",
        "matrix_element": 2.0,
        "selector_contract": _contract().as_dict(),
    }
    assert _baseline_selector_contract(baseline) == _contract()
    assert _baseline_matrix_element(baseline) == 2.0

    with pytest.raises(RunnerError, match="not a valid completed"):
        _baseline_matrix_element({"status": "error"})
    with pytest.raises(RunnerError, match="no matrix element"):
        _baseline_matrix_element({"status": "ok", "matrix_element": None})


def test_lc_all_flow_baseline_must_authenticate_nonzero_common_component() -> None:
    cell = REPORT_CATALOG.cell("matrix-compiled-builtin-sm-lc-n2-ud-epve-jets-all-flow")
    baseline = {
        "validation": {
            "lc_common_component": {
                "value": 1.0,
            }
        }
    }

    _require_nonzero_lc_all_flow_baseline(cell, baseline)

    for value in (0.0, float("nan"), None):
        baseline["validation"]["lc_common_component"]["value"] = value
        with pytest.raises(RunnerError, match="baseline selector is structural zero"):
            _require_nonzero_lc_all_flow_baseline(cell, baseline)


def test_failure_measurement_preserves_compact_cache_shape() -> None:
    measurement = failure_measurement(
        ResultStatus.MEMORY_LIMIT,
        RuntimeError("over limit"),
        resources={"peak_rss_bytes": 42},
    )

    assert measurement["status"] == "memory_limit"
    assert measurement["generation_seconds"] is None
    assert measurement["resources"] == {"peak_rss_bytes": 42}
    assert measurement["failure"] == {
        "kind": "RuntimeError",
        "message": "over limit",
    }


def test_reused_artifact_closes_current_worker_generation_phase(
    tmp_path: Path,
) -> None:
    channel = WorkerPhaseChannel.create(tmp_path / "phase.json")
    ticks = iter((100, 200, 300))
    reporter = WorkerPhaseReporter(
        channel,
        worker_pid=42,
        clock_ns=lambda: next(ticks),
    )
    artifact = GeneratedArtifact(
        path=tmp_path / "artifact",
        process_id="process",
        generation_seconds=6.5,
        model_preparation_seconds=1.0,
        model_preparation_reused=True,
        requested_config={},
        effective_config={},
    )

    assert (
        _reuse_artifact_for_measurement(
            artifact,
            phase_reporter=reporter,
        )
        is artifact
    )
    state = read_worker_phase_state(channel, expected_pid=42)

    assert state.phase == "post-generation"
    assert state.sequence == 2
    assert state.generation_started_monotonic_ns == 200
    assert state.generation_finished_monotonic_ns == 300
    assert state.generation_elapsed_seconds(now_seconds=1.0) == 1.0e-7


def test_measurement_routes_reused_artifact_through_phase_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.performance_report.measurement as report_measurement

    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
    )
    artifact = GeneratedArtifact(
        path=tmp_path / "artifact",
        process_id="process",
        generation_seconds=6.5,
        model_preparation_seconds=1.0,
        model_preparation_reused=True,
        requested_config={},
        effective_config={},
    )
    reporter = object()
    observed: list[tuple[GeneratedArtifact, object]] = []

    def reuse(
        value: GeneratedArtifact,
        *,
        phase_reporter: object,
    ) -> GeneratedArtifact:
        observed.append((value, phase_reporter))
        return value

    def stop_after_selection(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("selection observed")

    monkeypatch.setattr(report_measurement, "_reuse_artifact_for_measurement", reuse)
    monkeypatch.setattr(
        report_measurement,
        "validate_artifact_contract",
        stop_after_selection,
    )

    with pytest.raises(RuntimeError, match="selection observed"):
        measure_pyamplicol_cell(
            cell,
            artifact_path=tmp_path / "unused",
            settings=RunnerSettings(),
            repo_root=tmp_path,
            baseline=None,
            reused_artifact=artifact,
            phase_reporter=reporter,  # type: ignore[arg-type]
        )

    assert observed == [(artifact, reporter)]


def test_reusable_artifact_retains_generation_command_path(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    measurement = {
        "status": ResultStatus.OK.value,
        "generation_seconds": 2.5,
        "artifact": {"path": str(artifact_path), "process_id": "process"},
        "provenance": {
            "requested_config": {},
            "effective_config": {},
            "model_preparation_seconds": 0.25,
            "model_preparation_reused": True,
            "generation_command_path": PUBLIC_CLI_COMMAND_PATH,
        },
    }

    artifact = generated_artifact_from_measurement(measurement)

    assert artifact.generation_command_path == PUBLIC_CLI_COMMAND_PATH

    del measurement["provenance"]["generation_command_path"]
    legacy = generated_artifact_from_measurement(measurement)
    assert legacy.generation_command_path is None

    measurement["provenance"]["generation_command_path"] = 42
    with pytest.raises(RunnerError, match="reusable artifact metadata is malformed"):
        generated_artifact_from_measurement(measurement)


def test_measurement_persists_generation_and_runtime_command_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.performance_report.measurement as report_measurement

    cell = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.accuracy is Accuracy.NLC
    )
    artifact_path = tmp_path / "artifact"
    artifact = GeneratedArtifact(
        path=artifact_path,
        process_id="process",
        generation_seconds=2.5,
        model_preparation_seconds=0.25,
        model_preparation_reused=True,
        requested_config={},
        effective_config={},
        generation_command_path=PUBLIC_CLI_COMMAND_PATH,
    )

    class Runtime:
        def evaluate(self, *_args: object, **_kwargs: object) -> list[float]:
            return [1.0]

    identity = {
        "loaded_module_origin_policy": {
            "observations": [],
        }
    }
    profile = {
        "status": ResultStatus.OK.value,
        "wall_seconds_per_point": 1.0e-6,
        "execution_seconds_per_point": 0.5e-6,
        "matrix_element": 1.0,
        "sample_count": 5,
        "standard_error_seconds_per_point": 0.1e-6,
        "relative_standard_error": 0.1,
        "execution_timing": {},
        "arena_profile_evidence": {},
        "benchmark_evidence": {
            "report_command_path": LOADED_RUNTIME_PROFILE_COMMAND_PATH,
        },
        "resolved_sum_validation": {"status": ResultStatus.OK.value},
    }
    monkeypatch.setattr(
        report_measurement,
        "generate_artifact",
        lambda *_args, **_kwargs: artifact,
    )
    monkeypatch.setattr(
        report_measurement,
        "validate_artifact_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        report_measurement,
        "_load_runtime",
        lambda *_args, **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        report_measurement,
        "runtime_identity_payload",
        lambda *_args, **_kwargs: identity,
    )
    monkeypatch.setattr(
        report_measurement,
        "shared_validation_points",
        lambda _process: ((1.0,),),
    )
    monkeypatch.setattr(
        report_measurement,
        "_resolution_benchmark_config",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        report_measurement,
        "profile_runtime",
        lambda *_args, **_kwargs: deepcopy(profile),
    )

    measurement = measure_pyamplicol_cell(
        cell,
        artifact_path=artifact_path,
        settings=RunnerSettings(source_revision_override="a" * 40),
        repo_root=tmp_path,
        baseline=None,
    )

    provenance = measurement["provenance"]
    assert provenance["generation_command_path"] == PUBLIC_CLI_COMMAND_PATH
    assert provenance["report_momenta"] == [[1.0]]
    assert (
        provenance["runtime_profile"]["report_command_path"]
        == LOADED_RUNTIME_PROFILE_COMMAND_PATH
    )


def test_catalog_contains_no_amplicol_candidate_matrix_cell() -> None:
    assert all(
        cell.measurement.execution_mode.value != "amplicol"
        for cell in REPORT_CATALOG.matrix_cells()
    )


def test_source_revision_rejects_source_dirt_but_allows_report_outputs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "report-test@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Report Test"),
        cwd=repo,
        check=True,
    )
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "source.py"), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-q", "-m", "initial"), cwd=repo, check=True)

    revision = source_revision(repo, require_clean=True)
    assert len(revision) == 40

    cache = repo / "docs/arxiv/results/z_builtin_sm.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{}\n", encoding="ascii")
    table = repo / "docs/arxiv/result_z_builtin_sm_table.tex"
    table.write_text("% generated\n", encoding="ascii")
    assert source_revision(repo, require_clean=True) == revision

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="outside generated report outputs"):
        source_revision(repo, require_clean=True)


def _runtime_identity() -> dict[str, object]:
    observations = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 10,
            "sha256": "1" * 64,
        }
    ]
    return {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "loaded_module_origin_policy": {
            "kind": "pyamplicol-loaded-module-origin-policy-v1",
            "all_loaded_origins_authenticated": True,
            "native_image_origin_bound": True,
            "loaded_bytecode_eligible": False,
            "observed_module_count": len(observations),
            "observations": observations,
            "observations_sha256": "a" * 64,
        },
    }


def test_runtime_identity_postflight_is_stable_and_monotonic() -> None:
    initial = _runtime_identity()
    postflight = deepcopy(initial)
    policy = postflight["loaded_module_origin_policy"]
    assert isinstance(policy, dict)
    observations = policy["observations"]
    assert isinstance(observations, list)
    observations.append(
        {
            "module": "pyamplicol.runtime",
            "kind": "package-member",
            "root_index": 0,
            "path": "runtime/__init__.py",
            "size": 20,
            "sha256": "2" * 64,
        }
    )
    policy["observed_module_count"] = len(observations)
    _validate_runtime_identity_postflight(initial, postflight)
    assert _stable_runtime_identity(initial) == _stable_runtime_identity(postflight)

    changed = deepcopy(postflight)
    changed["native_build_inputs_sha256"] = "3" * 64
    with pytest.raises(RunnerError, match="changed during report measurement"):
        _validate_runtime_identity_postflight(initial, changed)

    lost = deepcopy(postflight)
    lost_policy = lost["loaded_module_origin_policy"]
    assert isinstance(lost_policy, dict)
    lost_observations = lost_policy["observations"]
    assert isinstance(lost_observations, list)
    lost_observations.pop(0)
    lost_policy["observed_module_count"] = len(lost_observations)
    with pytest.raises(RunnerError, match="lost an authenticated"):
        _validate_runtime_identity_postflight(initial, lost)
