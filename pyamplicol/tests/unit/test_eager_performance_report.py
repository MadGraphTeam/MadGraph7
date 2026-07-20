# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
sys.path.insert(0, str(DOCS))

import result_tables as report  # noqa: E402


def _ok_measurement(
    *,
    generation: float = 1.0,
    wall: float = 2.0e-6,
    evaluator: float = 1.0e-6,
    matrix_element: float = 3.0,
    old_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    measurement = report._empty_measurement()
    measurement.update(
        {
            "status": report.ResultStatus.OK.value,
            "generation_seconds": generation,
            "sample_count": 5,
            "wall_seconds_per_point": wall,
            "evaluator_seconds_per_point": evaluator,
            "standard_deviation_seconds_per_point": 0.0,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "matrix_element": matrix_element,
            "requested_config": {},
            "effective_config": {"evaluator": {"execution_mode": "eager"}},
            "environment": {
                "wall_time_source": "runtime_core_repeated_wall_time",
                "evaluator_time_source": (
                    "runtime_profile_core_evaluator_call_time"
                ),
            },
            "metadata": {
                "old_matrix_format": dict(old_fields or {}),
                "model_precompile_policy": (
                    report.PYAMPLICOL_GENERATION_PROFILE_POLICY
                ),
                "generation_timer_excludes_model_compile": True,
            },
        }
    )
    return measurement


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _eager_cell(dataset_id: str, *, n_final: int = 1) -> report.CampaignCell:
    return next(
        cell
        for cell in report._campaign_cells()
        if cell.dataset_id == dataset_id
        and cell.n_final == n_final
        and cell.process_key == "dd_z_jets"
    )


def _publish_reset_report(docs: Path) -> report.ReportPaths:
    docs.mkdir()
    paths = report.ReportPaths(docs)
    report.ReportService(paths)._refresh(
        report.build_reset_caches(),
        compile_pdf=False,
    )
    return paths


def _start_barrier_processes(
    script: str,
    argument_sets: list[list[str]],
    tmp_path: Path,
) -> list[subprocess.CompletedProcess[str]]:
    start = tmp_path / "start"
    processes: list[subprocess.Popen[str]] = []
    ready_paths: list[Path] = []
    for index, arguments in enumerate(argument_sets):
        ready = tmp_path / f"ready-{index}"
        ready_paths.append(ready)
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", script, str(ready), str(start), *arguments],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    deadline = time.monotonic() + 15.0
    while not all(path.is_file() for path in ready_paths):
        if time.monotonic() >= deadline:
            for process in processes:
                process.kill()
            raise AssertionError("child-process barrier was not reached")
        time.sleep(0.01)
    start.write_text("go\n", encoding="utf-8")
    completed: list[subprocess.CompletedProcess[str]] = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        completed.append(
            subprocess.CompletedProcess(
                process.args,
                process.returncode,
                stdout,
                stderr,
            )
        )
    return completed


def test_eager_cache_contract_is_additive_and_z_row_is_after_compiled_o3() -> None:
    assert [variant.key for variant in report.Z_VARIANTS][-2:] == [
        "jit_o3",
        "eager_jit_o3",
    ]
    eager_variant = report.Z_VARIANTS[-1]
    assert report._z_variant_setup(eager_variant) == (
        r"eager-DAG JIT \(\mathrm{O}3\)"
    )
    for spec in report.EAGER_MATRIX_SPECS:
        payload = report.build_eager_matrix_cache(spec)
        assert payload["kind"] == report.CacheKind.EAGER_PROCESS_MATRIX.value
        assert payload["reference"] == {
            "dataset_id": spec.reference_dataset_id,
            "cache_name": spec.reference_cache_name,
            "measurement_field": "pyamplicol_jit_o3",
            "setup": "compiled JIT O3",
        }
        for entry in payload["entries"]:
            assert "eager_jit_o3" in entry
            assert "legacy_amplicol" not in entry
            assert "pyamplicol_jit_o3" not in entry


def test_z_cache_migration_preserves_existing_measurement_payloads() -> None:
    payload = report.build_ladder_cache(report.LADDER_SPECS[0])
    payload["variants"] = [
        variant
        for variant in payload["variants"]
        if variant["key"] != "eager_jit_o3"
    ]
    payload["entries"] = [
        entry
        for entry in payload["entries"]
        if entry["variant"] != "eager_jit_o3"
    ]
    compiled = next(
        entry for entry in payload["entries"] if entry["variant"] == "jit_o3"
    )
    compiled["measurement"] = _ok_measurement(
        old_fields={"arbitrary_nested_fixture": {"labels": [4, 2, 1]}}
    )
    before = _canonical_digest(compiled["measurement"])

    migrated = report.normalize_cache_payload(payload)
    migrated_compiled = next(
        entry for entry in migrated["entries"] if entry["variant"] == "jit_o3"
    )
    eager = next(
        entry for entry in migrated["entries"] if entry["variant"] == "eager_jit_o3"
    )

    assert _canonical_digest(migrated_compiled["measurement"]) == before
    assert eager["status"] == report.NA_STATUS
    assert eager["measurement"] == report._empty_measurement()


def test_eager_lc_renderer_joins_compiled_reference_without_copying_it() -> None:
    spec = report.EAGER_MATRIX_SPECS[0]
    eager_payload = report.build_eager_matrix_cache(spec)
    reference_spec = next(
        candidate
        for candidate in report.MATRIX_SPECS
        if candidate.dataset_id == spec.reference_dataset_id
    )
    reference_payload = report.build_matrix_cache(reference_spec)
    reference_entry = reference_payload["entries"][0]
    eager_entry = eager_payload["entries"][0]
    compiled = _ok_measurement(
        generation=2.0,
        wall=10.0e-6,
        evaluator=8.0e-6,
        matrix_element=7.0,
        old_fields={
            "status": "ok",
            "selected_generation_s": 2.0,
            "all_flow_generation_s": 4.0,
            "wall_us_per_point": 10.0,
            "runtime_us_per_point": 8.0,
            "all_flow_wall_us_per_point": 20.0,
            "all_flow_runtime_us_per_point": 16.0,
            "all_flow_matrix_element": 11.0,
        },
    )
    eager = _ok_measurement(
        generation=0.5,
        wall=5.0e-6,
        evaluator=3.0e-6,
        matrix_element=7.0,
        old_fields={
            "status": "ok",
            "selected_generation_s": 0.5,
            "all_flow_generation_s": 0.5,
            "wall_us_per_point": 5.0,
            "runtime_us_per_point": 3.0,
            "all_flow_wall_us_per_point": 6.0,
            "all_flow_runtime_us_per_point": 4.0,
            "all_flow_matrix_element": 11.0,
        },
    )
    reference_entry["pyamplicol_jit_o3"] = compiled
    report._refresh_matrix_derived_fields(reference_entry)
    eager_entry.update(
        {
            "eager_jit_o3": eager,
            "pointwise_validation": {
                **report._empty_validation(),
                "status": "ok",
                "reference_matrix_element": 7.0,
                "pyamplicol_matrix_element": 7.0,
                "absolute_difference": 0.0,
                "relative_difference": 0.0,
            },
            "selector_contract": {
                **report._empty_eager_selector_contract(),
                "status": "ok",
            },
        }
    )
    report._refresh_eager_matrix_derived_fields(eager_entry)

    table = report.render_matrix_table(
        spec,
        eager_payload,
        reference_payload=reference_payload,
    )

    assert r"\matrixrefpair{\texttt{2}}{\texttt{4}}" in table
    assert r"\matrixratio{ReportGreen}{0.25}" in table
    assert r"\matrixratio{ReportGreen}{0.125}" in table
    assert r"\matrixrefpair{\texttt{10}}{\texttt{20}}" in table
    assert r"\matrixratiopair{ReportGreen}{x0.5}{ReportGreen}{0.3}" in table
    assert r"\matrixratiopair{ReportGreen}{x0.3}{ReportGreen}{0.2}" in table
    assert "legacy_amplicol" not in eager_entry
    assert "pyamplicol_jit_o3" not in eager_entry


def test_eager_reference_digest_ignores_retiming_but_tracks_physics_contract() -> None:
    cell = _eager_cell("matrix_external_sm_eager_lc")
    original = _ok_measurement(
        matrix_element=4.0,
        old_fields={
            "reference_color_order": [2, 1, 3],
            "all_flow_source_helicities": {"1": -1, "2": 1, "3": 1},
            "all_flow_matrix_element": 5.0,
        },
    )
    retimed = json.loads(json.dumps(original))
    retimed["generation_seconds"] = 99.0
    retimed["wall_seconds_per_point"] = 123.0e-6
    changed = json.loads(json.dumps(retimed))
    changed["metadata"]["old_matrix_format"]["reference_color_order"] = [1, 2, 3]

    assert report._eager_reference_digest(
        cell, original
    ) == report._eager_reference_digest(cell, retimed)
    assert report._eager_reference_digest(
        cell, original
    ) != report._eager_reference_digest(cell, changed)


def test_eager_lc_reference_uses_cached_selector_without_mutating_history() -> None:
    cell = _eager_cell("matrix_external_sm_eager_lc")
    spec = report.EAGER_MATRIX_SPECS[0]
    reference_spec = next(
        candidate
        for candidate in report.MATRIX_SPECS
        if candidate.dataset_id == spec.reference_dataset_id
    )
    reference_payload = report.build_matrix_cache(reference_spec)
    entry = next(
        candidate
        for candidate in reference_payload["entries"]
        if candidate["process_key"] == cell.process_key
        and candidate["n_final"] == cell.n_final
    )
    compiled = _ok_measurement(
        old_fields={
            "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
            "all_flow_matrix_element": 5.0,
        }
    )
    entry["pyamplicol_jit_o3"] = compiled
    entry["legacy_amplicol"] = _ok_measurement(
        old_fields={"reference_color_order": [2, 1, 3]}
    )
    before = _canonical_digest(compiled)
    caches = {spec.reference_cache_name: reference_payload}

    enriched = report._eager_reference_measurement(cell, caches)

    assert isinstance(enriched, dict)
    assert enriched is not compiled
    assert report._measurement_old_matrix_fields(enriched)[
        "reference_color_order"
    ] == [2, 1, 3]
    assert _canonical_digest(compiled) == before
    report._preflight_eager_campaign_references((cell,), caches)


@pytest.mark.parametrize(
    ("process", "reference_order", "partition_word", "source_helicities"),
    (
        (
            "d d~ > z g",
            [2, 4, 1, 3],
            (2, 4, 1),
            {1: -1, 2: 1, 3: -1, 4: 1},
        ),
        (
            "d d~ > u u~ s s~",
            [2, 1, 3, 4, 5, 6],
            (2, 1, 3, 4, 5, 6),
            {1: -1, 2: 1, 3: -1, 4: 1, 5: -1, 6: 1},
        ),
    ),
)
def test_eager_lc_selector_resolution_uses_physical_flows_and_source_helicities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    process: str,
    reference_order: list[int],
    partition_word: tuple[int, ...],
    source_helicities: dict[int, int],
) -> None:
    cell = report.CampaignCell(
        kind="eager_matrix",
        cache_name="matrix_external_sm_eager_lc.json",
        dataset_id="matrix_external_sm_eager_lc",
        n_final=len(source_helicities) - 2,
        process=process,
        process_key="dd_z_jets",
    )
    reference = _ok_measurement(
        old_fields={
            "reference_color_order": reference_order,
            "all_flow_source_helicities": {
                str(label): value for label, value in source_helicities.items()
            },
            "all_flow_matrix_element": 1.0,
        }
    )
    physics = SimpleNamespace(
        color_flows=(SimpleNamespace(id="flow:selected", word=partition_word),),
        external_particles=tuple(
            SimpleNamespace(label=label) for label in sorted(source_helicities)
        ),
        helicities=(
            SimpleNamespace(
                id="helicity:selected",
                values=tuple(
                    source_helicities[label] for label in sorted(source_helicities)
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        report,
        "_selected_lc_reference_partition_words",
        lambda *args, **kwargs: {partition_word},
    )

    contract = report._eager_lc_selector_contract(
        cell=cell,
        spec=report.EAGER_MATRIX_SPECS[0],
        reference_measurement=reference,
        physics=physics,
        artifact_root=tmp_path,
    )

    assert contract["selected_color_flow_ids"] == ["flow:selected"]
    assert contract["all_flow_helicity_ids"] == ["helicity:selected"]
    assert contract["selected_reference_color_order"] == reference_order


def test_eager_preflight_reports_exact_missing_compiled_reference() -> None:
    caches = report.build_reset_caches()
    cell = _eager_cell("matrix_external_sm_eager_lc")

    with pytest.raises(
        ValueError,
        match=(
            "dataset='matrix_external_sm_lc', "
            "process_key='dd_z_jets', n_final=1"
        ),
    ):
        report._preflight_eager_campaign_references((cell,), caches)


def test_missing_only_eager_filters_schedule_no_reference_workloads() -> None:
    caches = report.load_caches(report.ReportPaths(DOCS))
    matrix_datasets = {spec.dataset_id for spec in report.EAGER_MATRIX_SPECS}
    matrix_cells = report._select_cells(
        datasets=matrix_datasets,
        missing_only=True,
        caches=caches,
    )
    z_cells = report._select_cells(
        datasets={"z_builtin_sm", "z_external_sm"},
        variants={"eager_jit_o3"},
        missing_only=True,
        caches=caches,
    )

    assert matrix_cells
    assert all(cell.kind == "eager_matrix" for cell in matrix_cells)
    assert z_cells
    assert all(
        cell.kind == "performance_ladder" and cell.variant == "eager_jit_o3"
        for cell in z_cells
    )


def test_strict_eager_validation_checks_selected_and_all_flow() -> None:
    compiled = _ok_measurement(
        matrix_element=1.0,
        old_fields={"all_flow_matrix_element": 2.0},
    )
    eager = _ok_measurement(
        matrix_element=1.0 + 5.0e-13,
        old_fields={"all_flow_matrix_element": 2.0 + 5.0e-13},
    )
    passed = report._eager_pointwise_validation(
        compiled,
        eager,
        require_all_flow=True,
    )
    assert passed["status"] == report.ResultStatus.OK.value
    assert passed["all_flow_status"] == report.ResultStatus.OK.value

    eager["metadata"]["old_matrix_format"]["all_flow_matrix_element"] = 2.01
    failed = report._eager_pointwise_validation(
        compiled,
        eager,
        require_all_flow=True,
    )
    assert failed["status"] == report.ResultStatus.VALIDATION_FAILED.value
    assert failed["all_flow_status"] == report.ResultStatus.VALIDATION_FAILED.value


def test_lc_eager_measurement_generates_once_and_profiles_two_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol.api as api
    import pyamplicol.config.resolver as resolver

    generated: list[Path] = []
    profiles: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    class FakeGenerator:
        def __init__(self, resolution: object) -> None:
            del resolution

        def generate(
            self,
            process: str,
            output: Path,
            *,
            model: object,
            mode: str,
        ) -> None:
            del process, model, mode
            output.mkdir(parents=True)
            generated.append(output)

    class FakeRuntime:
        physics = object()

        @classmethod
        def load(cls, artifact: Path, *, process: str) -> FakeRuntime:
            del artifact, process
            return cls()

    def fake_profile(
        runtime: object,
        *,
        benchmark_config: object,
        points: object,
        helicity_ids: tuple[str, ...] = (),
        color_flow_ids: tuple[str, ...] = (),
    ) -> dict[str, object]:
        del runtime, benchmark_config, points
        profiles.append((tuple(helicity_ids), tuple(color_flow_ids)))
        return _ok_measurement(matrix_element=5.0)

    monkeypatch.setattr(api, "Generator", FakeGenerator)
    monkeypatch.setattr(api, "Runtime", FakeRuntime)
    monkeypatch.setattr(
        resolver,
        "resolve_config",
        lambda *args, **kwargs: SimpleNamespace(
            requested=object(),
            effective=SimpleNamespace(benchmark=object()),
        ),
    )
    monkeypatch.setattr(
        resolver,
        "config_to_dict",
        lambda value: {"evaluator": {"execution_mode": "eager"}},
    )
    monkeypatch.setattr(report, "_run_config_values", lambda **kwargs: {})
    monkeypatch.setattr(
        report,
        "_prepared_model_source_for_eager",
        lambda *args, **kwargs: (
            tmp_path / "prepared.pyamplicol-model",
            {"preparation_seconds": 12.0},
        ),
    )
    monkeypatch.setattr(
        report,
        "_precompile_model_for_generation",
        lambda *args, **kwargs: (
            object(),
            {
                "model_precompile_policy": (
                    report.PYAMPLICOL_GENERATION_PROFILE_POLICY
                ),
                "generation_timer_excludes_model_compile": True,
            },
        ),
    )
    monkeypatch.setattr(
        report,
        "_reusable_pyamplicol_generation_seconds",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(report, "_single_artifact_process_id", lambda *a, **k: "p")
    monkeypatch.setattr(report, "_profile_eager_runtime", fake_profile)
    monkeypatch.setattr(
        report,
        "_eager_lc_selector_contract",
        lambda **kwargs: {
            **report._empty_eager_selector_contract(),
            "status": "ok",
            "selected_reference_color_order": [2, 4, 1, 3],
            "selected_color_flow_ids": ["flow:2,4,1"],
            "all_flow_source_helicities": {
                "1": -1,
                "2": 1,
                "3": -1,
                "4": 1,
            },
            "all_flow_helicity_ids": ["helicity:-1,1,-1,1"],
        },
    )
    cell = report.CampaignCell(
        kind="eager_matrix",
        cache_name="matrix_external_sm_eager_lc.json",
        dataset_id="matrix_external_sm_eager_lc",
        n_final=2,
        process="d d~ > z g",
        process_key="dd_z_jets",
    )

    measurement, _, _ = report._measure_pyamplicol_eager_complete(
        cell=cell,
        spec=report.EAGER_MATRIX_SPECS[0],
        reference_measurement=_ok_measurement(),
        artifact_root=tmp_path / "artifacts",
        generation_timeout_seconds=30.0,
        target_runtime=0.01,
        cell_cores=1,
        points=(((1.0, 0.0, 0.0, 1.0),),),
    )

    assert len(generated) == 1
    assert profiles == [
        ((), ("flow:2,4,1",)),
        (("helicity:-1,1,-1,1",), ()),
    ]
    old = measurement["metadata"]["old_matrix_format"]
    assert old["selected_generation_s"] == old["all_flow_generation_s"]
    assert measurement["metadata"]["prepared_model_creation_excluded_from_generation"]


def test_separate_processes_merge_disjoint_eager_cells_without_lost_updates(
    tmp_path: Path,
) -> None:
    paths = _publish_reset_report(tmp_path / "report-docs")
    cells = (
        _eager_cell("matrix_external_sm_eager_lc", n_final=1),
        _eager_cell("matrix_external_sm_eager_lc", n_final=2),
    )
    argument_sets: list[list[str]] = []
    for index, cell in enumerate(cells):
        cell_path = tmp_path / f"cell-{index}.json"
        entry_path = tmp_path / f"entry-{index}.json"
        cell_path.write_text(json.dumps(cell.as_json()), encoding="utf-8")
        entry_path.write_text(
            json.dumps(
                report._failure_entry_for_cell(
                    cell,
                    status=report.ResultStatus.ERROR,
                    message=f"failure-{index}",
                    artifact_root=tmp_path / "artifacts",
                    limit_gib=30,
                    timeout_seconds=30,
                )
            ),
            encoding="utf-8",
        )
        argument_sets.append(
            [str(DOCS), str(paths.docs_dir), str(cell_path), str(entry_path)]
        )
    script = r"""
import json, sys, time
from pathlib import Path
ready, start, source_docs, report_docs, cell_file, entry_file = map(Path, sys.argv[1:])
sys.path.insert(0, str(source_docs))
import result_tables as report
ready.write_text('ready')
while not start.exists(): time.sleep(0.01)
cell = report._cell_from_json(json.loads(cell_file.read_text()))
entry = json.loads(entry_file.read_text())
report.ReportService(report.ReportPaths(report_docs))._merge_and_refresh(
    cell, entry, compile_pdf=False
)
"""

    completed = _start_barrier_processes(script, argument_sets, tmp_path)
    assert all(result.returncode == 0 for result in completed), [
        result.stderr for result in completed
    ]
    caches = report.load_caches(paths)
    for index, cell in enumerate(cells):
        entry = report._cache_entry_for_cell(cell, caches)
        assert entry is not None
        assert entry["eager_jit_o3"]["failure_message"] == f"failure-{index}"


def test_duplicate_cell_processes_invoke_worker_once(
    tmp_path: Path,
) -> None:
    paths = _publish_reset_report(tmp_path / "report-docs")
    cell = _eager_cell("matrix_external_sm_eager_lc", n_final=1)
    cell_path = tmp_path / "cell.json"
    payload_path = tmp_path / "payload.json"
    counter_path = tmp_path / "worker-count.txt"
    cell_path.write_text(json.dumps(cell.as_json()), encoding="utf-8")
    payload_path.write_text(
        json.dumps(
            {
                "cell": cell.as_json(),
                "cache_name": cell.cache_name,
                "entry": report._failure_entry_for_cell(
                    cell,
                    status=report.ResultStatus.ERROR,
                    message="synthetic worker result",
                    artifact_root=tmp_path / "artifacts",
                    limit_gib=30,
                    timeout_seconds=30,
                ),
            }
        ),
        encoding="utf-8",
    )
    output_paths = [tmp_path / f"result-{index}.json" for index in range(2)]
    argument_sets = [
        [
            str(DOCS),
            str(paths.docs_dir),
            str(tmp_path / "artifacts"),
            str(cell_path),
            str(payload_path),
            str(counter_path),
            str(output_path),
        ]
        for output_path in output_paths
    ]
    script = r"""
import json, sys, time
from pathlib import Path
(ready, start, source_docs, report_docs, artifacts, cell_file,
 payload_file, counter, output) = map(Path, sys.argv[1:])
sys.path.insert(0, str(source_docs))
import result_tables as report
payload = json.loads(payload_file.read_text())
def fake_run(command, **kwargs):
    del kwargs
    with counter.open('a') as stream: stream.write('worker\n')
    time.sleep(0.35)
    result_path = Path(command[command.index('--result-json') + 1])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload))
    return 0
report._run_worker_command = fake_run
cell = report._cell_from_json(json.loads(cell_file.read_text()))
ready.write_text('ready')
while not start.exists(): time.sleep(0.01)
result = report._execute_campaign_cell(
    cell,
    python=Path(sys.executable),
    artifact_root=artifacts,
    limit_gib=30,
    generation_timeout_seconds=30,
    jit_o3_generation_timeout_seconds=30,
    reference_timeout_seconds=30,
    target_runtime=0.01,
    cell_cores=1,
    report_paths=report.ReportPaths(report_docs),
)
output.write_text(json.dumps(result))
"""

    completed = _start_barrier_processes(script, argument_sets, tmp_path)
    assert all(result.returncode == 0 for result in completed), [
        result.stderr for result in completed
    ]
    assert counter_path.read_text(encoding="utf-8").splitlines() == ["worker"]
    results = [json.loads(path.read_text(encoding="utf-8")) for path in output_paths]
    assert (
        sum(bool(result.get("skipped_after_cell_completion")) for result in results)
        == 1
    )


def test_concurrent_prepared_pack_requests_publish_one_bundle(
    tmp_path: Path,
) -> None:
    model_source = tmp_path / "model.json"
    model_source.write_text("{}\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    counter = tmp_path / "prepare-count.txt"
    output_paths = [tmp_path / f"prepared-{index}.json" for index in range(2)]
    argument_sets = [
        [str(DOCS), str(artifacts), str(model_source), str(counter), str(output)]
        for output in output_paths
    ]
    script = r"""
import json, sys, time
from pathlib import Path
(ready, start, source_docs, artifacts, model_source,
 counter, output) = map(Path, sys.argv[1:])
sys.path.insert(0, str(source_docs))
import result_tables as report
identity = {'schema': 9, 'compiler': 13, 'backend': 'jit', 'target': 'test'}
report._report_prepared_pack_identity = lambda: identity
report._model_source_path = lambda model: model_source
report._validate_report_prepared_pack = lambda path: None
def fake_run(command, **kwargs):
    del kwargs
    with counter.open('a') as stream: stream.write('prepare\n')
    time.sleep(0.35)
    bundle = next(
        Path(part)
        for part in command
        if str(part).endswith('.pyamplicol-model')
    )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b'prepared')
    return 0
report._run_worker_command = fake_run
ready.write_text('ready')
while not start.exists(): time.sleep(0.01)
bundle, metadata = report._ensure_report_ufo_sm_prepared_pack(
    artifacts,
    python=Path(sys.executable),
    limit_gib=30,
    timeout_seconds=30,
)
output.write_text(json.dumps({'bundle': str(bundle), 'identity': metadata['identity']}))
"""

    completed = _start_barrier_processes(script, argument_sets, tmp_path)
    assert all(result.returncode == 0 for result in completed), [
        result.stderr for result in completed
    ]
    assert counter.read_text(encoding="utf-8").splitlines() == ["prepare"]
    results = [json.loads(path.read_text(encoding="utf-8")) for path in output_paths]
    assert results[0] == results[1]
    assert Path(results[0]["bundle"]).read_bytes() == b"prepared"
