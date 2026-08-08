# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.config import BenchmarkConfig
from tools.performance_report.agreements import (
    LC_CROSS_LAYOUT_COMPONENT,
    incoming_agreement_edges,
)
from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.cache import build_reset_caches
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import reproduction_recipe
from tools.performance_report.models import Accuracy, ExecutionMode, Workload
from tools.performance_report.render import _BEST_MODE_CODES, render_all_matrix_tables
from tools.performance_report.runner import (
    CONVENTIONAL_WARMUP_TIMING_SCOPE,
    LOADED_RUNTIME_PROFILE_COMMAND_PATH,
    OTF_ACTIVE_FAMILY_COUNT_FIELDS,
    OTF_COLD_WARMUP_RUNTIME_FRESHNESS,
    OTF_COLD_WARMUP_RUNTIME_STATE_EVIDENCE,
    OTF_COLD_WARMUP_TIMING_SCOPE,
    OTF_RUNTIME_STATE_CENSUS_KIND,
    OTF_RUNTIME_STATE_COUNT_FIELDS,
    OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT,
    OTF_RUNTIME_STATE_FAMILY_CACHE_POLICY,
    OTF_RUNTIME_STATE_RETAINED_BASE_POSITIVE_FIELDS,
    OTF_RUNTIME_STATE_RETAINED_EXECUTABLE_FIELDS,
    PUBLIC_CLI_COMMAND_PATH,
    WARMUP_TIMER_SOURCE,
    RunnerError,
    RunnerSettings,
    SelectorContract,
    _benchmark_measurement,
    config_values,
    point_digest,
    profile_runtime,
    validate_artifact_contract,
    validate_runtime_contract,
)
from tools.performance_report.scheduler import CampaignSettings, plan_campaign


def _otf_warmup_runtime_state(*, retained: bool) -> dict[str, object]:
    counts = {field: 0 for field in OTF_RUNTIME_STATE_COUNT_FIELDS}
    active: dict[str, object] | None = None
    if retained:
        counts.update(
            {
                field: 1
                for field in (
                    *OTF_RUNTIME_STATE_RETAINED_BASE_POSITIVE_FIELDS,
                    *OTF_RUNTIME_STATE_RETAINED_EXECUTABLE_FIELDS,
                )
            }
        )
        active = {
            "basis": "shared-query-family-union-v1",
            "scope": "active-family-union",
            **{field: 1 for field in OTF_ACTIVE_FAMILY_COUNT_FIELDS},
        }
    return {
        "kind": OTF_RUNTIME_STATE_CENSUS_KIND,
        "process_id": "otf_campaign_process",
        "family_cache_policy": OTF_RUNTIME_STATE_FAMILY_CACHE_POLICY,
        "family_cache_limit": OTF_RUNTIME_STATE_FAMILY_CACHE_LIMIT,
        **counts,
        "active_family_union_census": active,
    }


def _otf_cells():
    return tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
    )


def test_otf_matrix_surface_is_measurable_for_every_color_accuracy() -> None:
    cells = _otf_cells()
    by_accuracy = Counter(cell.measurement.accuracy for cell in cells)
    lc = tuple(cell for cell in cells if cell.measurement.accuracy is Accuracy.LC)

    assert by_accuracy == {
        Accuracy.LC: 66,
        Accuracy.NLC: 50,
        Accuracy.FULL: 100,
    }
    assert len({(cell.process_key, cell.n_final) for cell in lc}) == 33
    assert {cell.n_final for cell in lc} == {1, 2, 3, 4}
    assert all(REPORT_CATALOG.static_na_reason(cell) is None for cell in cells)
    assert _BEST_MODE_CODES[ExecutionMode.ON_THE_FLY] == "o"


def test_otf_uses_amplicol_for_display_and_recurrence_for_correctness() -> None:
    all_flow = next(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is Accuracy.LC
        and cell.process_key == "dd_tt_jets"
        and cell.n_final == 4
        and cell.workload is Workload.ALL_FLOW
    )
    display = REPORT_CATALOG.baseline_cell(all_flow)
    validation = REPORT_CATALOG.validation_baseline_cell(all_flow)
    edges = incoming_agreement_edges(all_flow)

    assert display is not None
    assert display.measurement.execution_mode is ExecutionMode.AMPLICOL
    assert validation is not None
    assert validation.measurement.execution_mode is ExecutionMode.RECURRENCE
    assert all(
        edge.baseline.measurement.execution_mode is not ExecutionMode.COMPILED
        for edge in edges
    )
    selected = next(
        edge.baseline for edge in edges if edge.kind == LC_CROSS_LAYOUT_COMPONENT
    )
    assert selected.measurement.execution_mode is ExecutionMode.ON_THE_FLY
    assert REPORT_CATALOG.equivalent_cells(all_flow) == (selected,)


def test_otf_plan_orders_lc_and_contracted_authorities(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    selected = next(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is Accuracy.LC
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is Workload.SELECTED_FLOW
    )
    planned = plan_campaign(
        (selected,),
        store=store,
        settings=CampaignSettings(),
    )

    assert tuple(item.cell.measurement.execution_mode for item in planned) == (
        ExecutionMode.RECURRENCE,
        ExecutionMode.ON_THE_FLY,
    )
    nlc = next(
        cell for cell in _otf_cells() if cell.measurement.accuracy is Accuracy.NLC
    )
    assert nlc.workload is Workload.CONTRACTED
    assert tuple(
        item.cell.measurement.execution_mode
        for item in plan_campaign(
            (nlc,),
            store=store,
            settings=CampaignSettings(),
        )
    ) == (
        ExecutionMode.RECURRENCE,
        ExecutionMode.ON_THE_FLY,
    )

    ufo_full = next(
        cell
        for cell in _otf_cells()
        if cell.dataset_id == "matrix_on_the_fly_ufo_sm_full"
    )
    assert tuple(
        item.cell.measurement.execution_mode
        for item in plan_campaign(
            (ufo_full,),
            store=store,
            settings=CampaignSettings(),
        )
    ) == (
        ExecutionMode.MADGRAPH,
        ExecutionMode.ON_THE_FLY,
    )


def test_otf_config_reuses_one_topology_replay_artifact_for_both_workloads(
    tmp_path: Path,
) -> None:
    cells = tuple(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is Accuracy.LC
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
    )

    assert {cell.workload for cell in cells} == {
        Workload.SELECTED_FLOW,
        Workload.ALL_FLOW,
    }
    configurations = tuple(
        config_values(cell, RunnerSettings(), repo_root=tmp_path) for cell in cells
    )
    assert {
        config["color"]["lc_flow_layout"]  # type: ignore[index]
        for config in configurations
    } == {"topology-replay"}
    for config in configurations:
        assert config["generation"]["relation_discovery"] == {  # type: ignore[index]
            "mode": "off"
        }
        assert config["generation"]["validation"]["enabled"] is False  # type: ignore[index]
        assert (  # type: ignore[index]
            config["generation"]["validation"]["post_build_validation"] is False
        )
        assert config["evaluator"]["backend"] == "jit"  # type: ignore[index]
        assert config["evaluator"]["jit"]["optimization_level"] == 2  # type: ignore[index]


@pytest.mark.parametrize("workload", [Workload.SELECTED_FLOW, Workload.ALL_FLOW])
def test_otf_reproduction_recipe_uses_the_compact_public_generation_contract(
    tmp_path: Path,
    workload: Workload,
) -> None:
    cell = next(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is Accuracy.LC
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is workload
    )

    recipe = reproduction_recipe(
        cell,
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifacts",
    )

    assert recipe.generate is not None
    generate = recipe.generate
    assert generate[generate.index("--execution-mode") + 1] == "on-the-fly"
    assert generate[generate.index("--lc-flow-layout") + 1] == "topology-replay"
    assert "--no-emit-api-bundle" in generate
    assert "--emit-api-bundle" not in generate
    assert "--no-validation" in generate
    assert "--validation" not in generate
    assert "--no-post-build-validation" in generate
    assert "--post-build-validation" not in generate
    assert "--no-numerical-current-reuse" in generate
    assert "--numerical-current-reuse" not in generate


@pytest.mark.parametrize("accuracy", (Accuracy.LC, Accuracy.NLC, Accuracy.FULL))
def test_otf_artifact_requires_accuracy_specific_compact_runtime_capability(
    monkeypatch: pytest.MonkeyPatch,
    accuracy: Accuracy,
) -> None:
    cell = next(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is accuracy
        and cell.workload
        is (Workload.SELECTED_FLOW if accuracy is Accuracy.LC else Workload.CONTRACTED)
    )
    process = SimpleNamespace(
        execution_mode="on-the-fly",
        generation_specialized_axes=(),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_flow_layout="compact/query-local",
        recurrence_color_accuracy=accuracy.value,
        recurrence_color_storage="expanded",
        recurrence_color_component_count=1,
        recurrence_color_group_count=2,
        recurrence_color_destination_count=2,
    )
    color_capability = (
        "rusticol.on-the-fly.lc-color.v1"
        if accuracy is Accuracy.LC
        else "rusticol.on-the-fly.contracted-color.v1"
    )
    inspection = SimpleNamespace(
        processes=(process,),
        runtime_capabilities=(
            "rusticol.on-the-fly.complex-f64.v1",
            color_capability,
        ),
    )
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspect_artifact",
        lambda _path: inspection,
    )

    validate_artifact_contract(cell, Path("/artifact"))
    if accuracy is not Accuracy.LC:
        process.recurrence_color_component_count = 2
        with pytest.raises(RunnerError, match="expanded color payload"):
            validate_artifact_contract(cell, Path("/artifact"))
        process.recurrence_color_component_count = 1
        inspection.runtime_capabilities = (
            "rusticol.on-the-fly.complex-f64.v1",
            "rusticol.on-the-fly.lc-color.v1",
        )
        with pytest.raises(RunnerError, match="accuracy-specific"):
            validate_artifact_contract(cell, Path("/artifact"))
        return
    process.lc_flow_layout = "topology-replay"
    with pytest.raises(RunnerError, match="compact/query-local"):
        validate_artifact_contract(cell, Path("/artifact"))
    process.lc_flow_layout = "compact/query-local"
    inspection.runtime_capabilities += ("unexpected-capability",)
    with pytest.raises(RunnerError, match="exactly its two accuracy-specific"):
        validate_artifact_contract(cell, Path("/artifact"))


@pytest.mark.parametrize("accuracy", (Accuracy.NLC, Accuracy.FULL))
def test_otf_contracted_runtime_uses_compact_context_without_flow_selectors(
    accuracy: Accuracy,
) -> None:
    cell = next(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is accuracy
        and cell.workload is Workload.CONTRACTED
    )

    process_expression = cell.process.upper()

    class Backend:
        @staticmethod
        def _on_the_fly_benchmark_context(
            requested: tuple[str, ...],
        ) -> dict[str, object]:
            assert requested == ()
            return {
                "process_id": "contracted-otf",
                "process_expression": process_expression,
                "color_accuracy": accuracy.value,
                "helicity_count": 4,
                "color_count": 1,
                "selected_color_ids": [],
            }

    runtime = SimpleNamespace(execution_mode="on-the-fly", _backend=Backend())
    validate_runtime_contract(cell, runtime)

    process_expression = "g g > g"
    with pytest.raises(RunnerError, match="compact selector identity"):
        validate_runtime_contract(cell, runtime)
    process_expression = cell.process.upper()

    class DensePhysicsTrap:
        @property
        def physics(self) -> object:
            raise AssertionError("contracted OTF validation opened dense physics")

    runtime._backend = DensePhysicsTrap()
    with pytest.raises(RunnerError, match="compact selector context"):
        validate_runtime_contract(cell, runtime)


@pytest.mark.parametrize("workload", (Workload.SELECTED_FLOW, Workload.ALL_FLOW))
def test_otf_public_profile_owns_first_evaluation_without_dense_physics(
    monkeypatch: pytest.MonkeyPatch,
    workload: Workload,
) -> None:
    import pyamplicol.api

    cell = next(
        cell
        for cell in _otf_cells()
        if cell.measurement.accuracy is Accuracy.LC and cell.workload is workload
    )
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = SelectorContract(
        selected_color_flow_ids=("flow:1,2",),
        selected_color_words=((1, 2),),
        all_flow_helicity_ids=("h:+1,-1",),
        all_flow_source_helicities=((1, 1), (2, -1)),
        point_digest=point_digest(points),
    )
    events: list[str] = []
    selector_calls: list[tuple[str, tuple[str, ...]]] = []
    dense_physics_accesses = 0

    class Backend:
        @staticmethod
        def _on_the_fly_benchmark_context(
            requested: tuple[str, ...],
        ) -> dict[str, object]:
            assert requested == ()
            return {
                "process_id": "compact-profile-candidate",
                "process_expression": cell.process,
                "color_accuracy": "lc",
                "helicity_count": 2,
                "color_count": 2,
                "selected_color_ids": [],
            }

        @staticmethod
        def _point_selector_indices(
            values: tuple[str, ...],
            name: str,
        ) -> tuple[int, ...]:
            selector_calls.append((name, tuple(values)))
            return (0,)

    class Runtime:
        execution_mode = "on-the-fly"
        _backend = Backend()

        @property
        def physics(self) -> object:
            nonlocal dense_physics_accesses
            dense_physics_accesses += 1
            raise AssertionError("OTF profile opened dense physics")

        def evaluate(self, *_args: object, **_kwargs: object) -> list[float]:
            events.append("campaign-evaluate")
            expected = (
                {"helicities": None, "color_flows": ("flow:1,2",)}
                if workload is Workload.SELECTED_FLOW
                else {"helicities": ("h:+1,-1",), "color_flows": None}
            )
            assert _kwargs == expected
            return [2.0]

    @dataclass(frozen=True)
    class Result:
        environment: dict[str, object]

    class BenchmarkRunner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def run(self, runtime: object, *, points: object) -> Result:
            events.append("public-profile")
            return Result(
                environment={
                    "execution_mode": "on-the-fly",
                    "cold_warmup_elapsed_seconds": 0.25,
                }
            )

    def benchmark_measurement(benchmark: Result, *, matrix_element: float):
        assert matrix_element == 2.0
        assert benchmark.environment["cold_warmup_elapsed_seconds"] == 0.25
        assert benchmark.environment["report_command_path"] == (
            LOADED_RUNTIME_PROFILE_COMMAND_PATH
        )
        assert benchmark.environment["report_public_cli_path"] == (
            PUBLIC_CLI_COMMAND_PATH
        )
        return {"status": "ok"}

    monkeypatch.setattr(pyamplicol.api, "BenchmarkRunner", BenchmarkRunner)
    monkeypatch.setattr(
        "tools.performance_report.runner._benchmark_measurement",
        benchmark_measurement,
    )
    monkeypatch.setattr(
        "tools.performance_report.runner.resolved_sum_validation",
        lambda *_args, **_kwargs: {"status": "ok"},
    )

    result = profile_runtime(
        Runtime(),
        points,
        cell=cell,
        benchmark_config=BenchmarkConfig(),
        selector_contract=contract,
    )

    assert result["status"] == "ok"
    assert events == ["public-profile", "campaign-evaluate"]
    assert dense_physics_accesses == 0
    assert selector_calls == [
        ("color_flow_by_point", ("flow:1,2",)),
        ("helicity_by_point", ("h:+1,-1",)),
    ]


def test_otf_benchmark_measurement_requires_and_stores_cold_warmup() -> None:
    environment = {
        "execution_mode": "on-the-fly",
        "batch_size": 128,
        "evaluator_time_raw_seconds_per_point": 8.0e-7,
        "evaluator_time_status": "measured",
        "evaluator_time_ratio_eligible": True,
        "evaluator_time_source": "runtime_profile_core_recurrence_schedule_time",
        "native_profile_points_per_sample": 128,
        "native_profile_repetitions_per_sample": 1,
        "native_profile_batch_size": 128,
        "timing_sample_contract": "paired-native-repeated-profile-v1",
        "elapsed_seconds": 1.0,
        "measured_point_count": 640,
        "evaluator_total_time_raw_seconds_per_point": 9.0e-7,
        "evaluator_total_time_status": "measured",
        "evaluator_total_time_ratio_eligible": False,
        "evaluator_total_time_source": ("runtime._benchmark_f64_wall_time.accumulated"),
        "evaluator_total_time_sample_contract": (
            "accumulated-repeated-warmed-evaluator-total-v1"
        ),
        "evaluator_total_accumulated_seconds": 5.76e-4,
        "cold_warmup_elapsed_seconds": 0.25,
        "cold_warmup_run_count": 1,
        "cold_warmup_batch_size": 128,
        "cold_warmup_point_count": 128,
        "cold_warmup_timer_source": WARMUP_TIMER_SOURCE,
        "cold_warmup_timing_scope": OTF_COLD_WARMUP_TIMING_SCOPE,
        "cold_warmup_runtime_freshness": OTF_COLD_WARMUP_RUNTIME_FRESHNESS,
        "cold_warmup_runtime_state_evidence": (OTF_COLD_WARMUP_RUNTIME_STATE_EVIDENCE),
        "cold_warmup_runtime_state_before": _otf_warmup_runtime_state(retained=False),
        "cold_warmup_runtime_state_after": _otf_warmup_runtime_state(retained=True),
        "cold_warmup_runtime_cold_before_first_evaluation": True,
        "cold_warmup_runtime_retained_before_first_evaluation": False,
        "cold_warmup_runtime_retained_after_first_evaluation": True,
        "cold_warmup_ratio_eligible": False,
        "cold_warmup_acceptance_eligible": False,
        "warmup_elapsed_seconds": 0.2,
        "warmup_configured_run_count": 2,
        "warmup_batch_size": 128,
        "warmup_point_count": 256,
        "warmup_run_outer_wall_seconds": (0.1, 0.1),
        "first_warmup_run_outer_wall_seconds": 0.1,
        "warmup_timer_source": WARMUP_TIMER_SOURCE,
        "warmup_timing_scope": CONVENTIONAL_WARMUP_TIMING_SCOPE,
    }
    benchmark = SimpleNamespace(
        uncertainty=SimpleNamespace(
            standard_error=1.0e-8,
            relative_standard_error=0.01,
        ),
        environment=environment,
        evaluator_time_per_point=8.0e-7,
        evaluator_total_time_per_point=9.0e-7,
        wall_time_per_point=1.0e-6,
        sample_count=5,
        effective_config=SimpleNamespace(
            target_runtime=1.0,
            batch_size=128,
            warmup_runs=2,
        ),
    )

    measurement = _benchmark_measurement(benchmark, matrix_element=2.0)
    assert measurement["benchmark_evidence"]["cold_warmup_elapsed_seconds"] == 0.25
    assert (
        measurement["benchmark_evidence"]["cold_warmup_runtime_state_evidence"]
        == OTF_COLD_WARMUP_RUNTIME_STATE_EVIDENCE
    )
    assert (
        measurement["benchmark_evidence"]["cold_warmup_runtime_state_before"][
            "retained_family_count"
        ]
        == 0
    )
    assert (
        measurement["benchmark_evidence"]["cold_warmup_runtime_state_after"][
            "retained_family_count"
        ]
        == 1
    )
    assert measurement["evaluator_total_timing"] == {
        "abi": "pyamplicol-report-evaluator-total-timing-v1",
        "status": "measured",
        "ratio_eligible": False,
        "raw_seconds_per_point": 9.0e-7,
        "source": "runtime._benchmark_f64_wall_time.accumulated",
        "execution_mode": "on-the-fly",
        "sample_contract": "accumulated-repeated-warmed-evaluator-total-v1",
        "sample_count": 5,
        "repetitions_per_sample": 1,
        "batch_size": 128,
        "points_per_sample": 128,
        "measured_point_count": 640,
        "accumulated_seconds": 5.76e-4,
    }

    del environment["cold_warmup_elapsed_seconds"]
    with pytest.raises(RunnerError, match="incomplete cold warm-up evidence"):
        _benchmark_measurement(benchmark, matrix_element=2.0)


def test_otf_tables_are_in_the_normal_matrix_render_path() -> None:
    tables = render_all_matrix_tables(build_reset_caches())

    assert {
        "result_matrix_on_the_fly_builtin_sm_lc_table.tex",
        "result_matrix_on_the_fly_builtin_sm_nlc_table.tex",
        "result_matrix_on_the_fly_builtin_sm_full_table.tex",
        "result_matrix_on_the_fly_ufo_sm_full_vs_madgraph_table.tex",
    } <= tables.keys()
