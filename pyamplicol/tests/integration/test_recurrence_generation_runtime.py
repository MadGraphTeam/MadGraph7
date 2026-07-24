# SPDX-License-Identifier: 0BSD
"""Public generation/load canary for compact LC recurrence execution."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol import CompiledModel, Generator, ModelSource, Runtime
from pyamplicol.artifacts import inspect_artifact, load_manifest
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point
from pyamplicol.reporting import CallbackProgressSink, ProgressUpdate
from pyamplicol.runtime.recurrence_exact._plan import _validate_execution

_PROCESS = "d d~ > z g g"
_TOPOLOGY_REPLAY_PROCESSES = (
    "d d~ > z g",
    _PROCESS,
)
_TOPOLOGY_REPLAY_STRUCTURE = {
    "d d~ > z g": (31, 34, 12),
    _PROCESS: (69, 126, 24),
}
_RECURRENCE_KIND = "pyamplicol-runtime-recurrence-execution"
_RECURRENCE_CAPABILITIES = {
    "rusticol.recurrence-color.lc.v1",
    "rusticol.recurrence-direct-arena.complex-f64.v1",
}
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)

_Point = tuple[tuple[float, ...], ...]
_Points = tuple[_Point, ...]


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_recurrence() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        _unavailable("Symbolica is unavailable")
    rusticol = importlib.import_module("pyamplicol._rusticol")
    if not hasattr(rusticol, "_lower_recurrence_direct_v2"):
        _unavailable("the installed Rusticol extension lacks recurrence lowering")


def _generation_config(
    execution_mode: str,
    *,
    lc_flow_layout: str = "topology-replay",
) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(
            accuracy="lc",
            lc_flow_layout=lc_flow_layout,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(
                optimization_level=2 if execution_mode == "recurrence" else 1
            ),
        ),
    )


def _flatten(
    values: tuple[
        tuple[tuple[complex | Decimal, ...], ...],
        ...,
    ],
) -> tuple[complex, ...]:
    return tuple(
        complex(value) for point in values for helicity in point for value in helicity
    )


def _validation_points(process_expression: str) -> _Points:
    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(process_expression)
        ),
    )


def _assert_topology_replay_structure(
    artifact: Path,
    process_expression: str,
) -> None:
    expected_currents, expected_contributions, expected_closures = (
        _TOPOLOGY_REPLAY_STRUCTURE[process_expression]
    )
    manifest = load_manifest(artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (artifact / "processes" / process_id / "execution.json").read_text(
            encoding="utf-8"
        )
    )
    summary = execution["recurrence_summary"]
    assert (
        summary["current_count"],
        summary["contribution_count"],
        summary["closure_term_count"],
    ) == (
        expected_currents,
        expected_contributions,
        expected_closures,
    )

    inspection = inspect_artifact(artifact).processes[0]
    assert inspection.invocation_count == expected_contributions
    assert inspection.direct_contribution_row_count == expected_contributions
    assert inspection.closure_count == expected_closures
    assert inspection.direct_closure_row_count == expected_closures


def _assert_topology_replay_artifacts_match(
    recurrence_artifact: Path,
    compiled_artifact: Path,
    process_expression: str,
) -> tuple[Runtime, Runtime, _Points]:
    points = _validation_points(process_expression)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)

    recurrence_total = recurrence.evaluate(points)
    recurrence_resolved = recurrence.evaluate_resolved(points)
    compiled_total = compiled.evaluate(points)
    compiled_resolved = compiled.evaluate_resolved(points)

    assert recurrence_resolved.total() == pytest.approx(
        recurrence_total,
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert compiled_resolved.total() == pytest.approx(
        compiled_total,
        rel=1.0e-13,
        abs=1.0e-15,
    )
    assert recurrence_resolved.helicity_ids == compiled_resolved.helicity_ids
    assert recurrence_resolved.color_ids == compiled_resolved.color_ids
    assert recurrence_resolved.shape == compiled_resolved.shape
    assert _flatten(recurrence_resolved.values) == pytest.approx(
        _flatten(compiled_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_total == pytest.approx(
        compiled_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )

    # Public flow IDs are not recurrence construction-sector IDs. Exercise
    # every public selector through both resolved and optimized runtime paths.
    for color_id in recurrence_resolved.color_ids:
        recurrence_selected = recurrence.evaluate(points, color_flows=(color_id,))
        compiled_selected = compiled.evaluate(points, color_flows=(color_id,))
        recurrence_selected_resolved = recurrence.evaluate_resolved(
            points,
            color_flows=(color_id,),
        )
        compiled_selected_resolved = compiled.evaluate_resolved(
            points,
            color_flows=(color_id,),
        )
        assert recurrence_selected_resolved.helicity_ids == (
            compiled_selected_resolved.helicity_ids
        )
        assert recurrence_selected_resolved.color_ids == (
            compiled_selected_resolved.color_ids
        )
        assert recurrence_selected_resolved.shape == compiled_selected_resolved.shape
        assert _flatten(recurrence_selected_resolved.values) == pytest.approx(
            _flatten(compiled_selected_resolved.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        assert recurrence_selected == pytest.approx(
            compiled_selected,
            rel=1.0e-12,
            abs=1.0e-15,
        )

    return recurrence, compiled, points


def _assert_decimal_values_match(
    actual: object,
    expected: object,
    precision: int,
) -> None:
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    assert len(actual) == len(expected)
    relative_tolerance = Decimal("1e-12")
    absolute_tolerance = Decimal("1e-15")
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if isinstance(actual_value, tuple):
            _assert_decimal_values_match(actual_value, expected_value, precision)
            continue
        assert isinstance(actual_value, Decimal)
        assert isinstance(expected_value, Decimal)
        assert abs(actual_value - expected_value) <= (
            absolute_tolerance + relative_tolerance * abs(expected_value)
        )


def _assert_topology_replay_exact_matches_compiled(
    recurrence: Runtime,
    compiled: Runtime,
    points: _Points,
) -> None:
    for precision in (32, 50):
        recurrence_resolved = recurrence.evaluate_resolved(
            points,
            precision=precision,
        )
        compiled_resolved = compiled.evaluate_resolved(
            points,
            precision=precision,
        )
        assert recurrence_resolved.helicity_ids == compiled_resolved.helicity_ids
        assert recurrence_resolved.color_ids == compiled_resolved.color_ids
        assert recurrence_resolved.shape == compiled_resolved.shape
        _assert_decimal_values_match(
            recurrence_resolved.values,
            compiled_resolved.values,
            precision,
        )
        _assert_decimal_values_match(
            recurrence_resolved.total(),
            recurrence.evaluate(points, precision=precision),
            precision,
        )

        # Requesting every public ID explicitly must preserve the complete
        # resolved result, and each physical flow remains independently usable.
        explicit = recurrence.evaluate_resolved(
            points,
            helicities=recurrence_resolved.helicity_ids,
            color_flows=recurrence_resolved.color_ids,
            precision=precision,
        )
        _assert_decimal_values_match(
            explicit.values,
            recurrence_resolved.values,
            precision,
        )
        for color_id in recurrence_resolved.color_ids:
            actual = recurrence.evaluate_resolved(
                points,
                color_flows=(color_id,),
                precision=precision,
            )
            expected = compiled.evaluate_resolved(
                points,
                color_flows=(color_id,),
                precision=precision,
            )
            _assert_decimal_values_match(actual.values, expected.values, precision)

        sampled_helicities = tuple(
            recurrence_resolved.helicity_ids[index]
            for index in sorted(
                {
                    0,
                    len(recurrence_resolved.helicity_ids) // 2,
                    len(recurrence_resolved.helicity_ids) - 1,
                }
            )
        )
        actual = recurrence.evaluate_resolved(
            points,
            helicities=sampled_helicities,
            precision=precision,
        )
        expected = compiled.evaluate_resolved(
            points,
            helicities=sampled_helicities,
            precision=precision,
        )
        _assert_decimal_values_match(actual.values, expected.values, precision)

        doubled_points = points + points
        helicity_by_point = (
            recurrence_resolved.helicity_ids[0],
            recurrence_resolved.helicity_ids[-1],
        )
        color_by_point = (
            recurrence_resolved.color_ids[0],
            recurrence_resolved.color_ids[-1],
        )
        _assert_decimal_values_match(
            recurrence.evaluate(
                doubled_points,
                helicity_by_point=helicity_by_point,
                color_flow_by_point=color_by_point,
                precision=precision,
            ),
            compiled.evaluate(
                doubled_points,
                helicity_by_point=helicity_by_point,
                color_flow_by_point=color_by_point,
                precision=precision,
            ),
            precision,
        )


@pytest.fixture(scope="module")
def ufo_sm_recurrence_jit_o2_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompiledModel:
    """Prepare one reusable UFO-SM recurrence pack for both public canaries."""

    _require_native_recurrence()
    root = tmp_path_factory.mktemp("ufo-sm-recurrence-jit-o2")
    model = ModelSource.from_path(
        _UFO_SM_ROOT / "sm.json",
        restriction=_UFO_SM_ROOT / "restrict_default.json",
    ).compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "ufo-sm-jit-o2.pyamplicol-model",
        evaluator=_generation_config("recurrence").evaluator,
    )
    assert model.is_prepared
    assert model.prepared_backend == "jit"
    return model


@pytest.mark.parametrize("process_expression", _TOPOLOGY_REPLAY_PROCESSES)
def test_builtin_lc_recurrence_artifact_loads_and_matches_compiled(
    tmp_path: Path,
    process_expression: str,
) -> None:
    """Exercise the first public topology-replay artifact end to end."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence"
    compiled_artifact = tmp_path / "compiled"
    progress_events: list[object] = []

    # Keep the packaged resource alive throughout generation without writing a
    # user-cache copy. Its compiled-model identity is still the built-in SM.
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(
            _generation_config("recurrence"),
            progress=CallbackProgressSink(progress_events.append),
        ).generate(
            process_expression,
            recurrence_artifact,
            model=ModelSource.from_path(prepared_model),
        )
    Generator(_generation_config("compiled")).generate(
        process_expression,
        compiled_artifact,
    )
    _assert_topology_replay_structure(recurrence_artifact, process_expression)
    native_progress = [
        event
        for event in progress_events
        if isinstance(event, ProgressUpdate) and event.task_id.endswith(":rust-builder")
    ]
    assert native_progress
    assert any(
        event.details.get("step") == "recurrence stage" for event in native_progress
    )
    assert any(
        int(event.details.get("current_count", 0)) > 0 for event in native_progress
    )

    manifest = load_manifest(recurrence_artifact)
    assert len(manifest.processes) == 1
    process = manifest.processes[0]
    process_id = str(process["id"])
    assert set(process["required_runtime_capabilities"]) == _RECURRENCE_CAPABILITIES
    assert (
        set(manifest.runtime["required_runtime_capabilities"])
        == _RECURRENCE_CAPABILITIES
    )
    inspection = inspect_artifact(recurrence_artifact).processes[0]
    assert inspection.execution_mode == "recurrence"
    assert inspection.prepared_backend == "jit"
    assert inspection.prepared_kernel_count
    assert inspection.invocation_count
    assert inspection.finalization_count
    assert inspection.closure_count
    assert inspection.native_profile_phases == (
        "selector-plan",
        "source-fill",
        "momentum-form-fill",
        "recurrence-direct-contribution",
        "recurrence-direct-finalization",
        "recurrence-direct-closure",
        "reduction",
    )

    process_root = recurrence_artifact / "processes" / process_id
    execution_path = process_root / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["plan"]["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_summary"]["lc_flow_layout"] == "topology-replay"
    runtime_container = execution["plan"]["runtime_container"]
    assert runtime_container["storage_abi"] == "pacbin-v1"
    assert runtime_container["member_count"] >= 1
    z_masses = {
        row["outgoing_pdg"]: row["mass"]
        for row in execution["runtime_metadata"]["particle_masses"]
    }
    assert z_masses[23] == pytest.approx(91.188)
    z_sources = [
        row
        for row in execution["runtime_metadata"]["source_templates"]
        if row["source_ir"]["identity"]["pdg_label"] == 23
    ]
    assert z_sources
    assert {row["source_ir"]["mass_parameter"] for row in z_sources} == {
        "particle.23.mass"
    }

    runtime_path = process_root / runtime_container["path"]
    assert runtime_path == process_root / "recurrence-runtime.pacbin"
    assert runtime_path.is_file()
    assert runtime_path.stat().st_size == runtime_container["size_bytes"]
    payloads = {record.path: record for record in manifest.payloads}
    runtime_payload = payloads[runtime_path.relative_to(recurrence_artifact).as_posix()]
    assert runtime_payload.size_bytes == runtime_path.stat().st_size
    assert runtime_payload.sha256 == runtime_container["sha256"]
    assert (recurrence_artifact / "evaluators.pacbin").is_file()

    recurrence, compiled, points = _assert_topology_replay_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
    )
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )

    # Prepared packs own the model-parameter derivation kernel. Recurrence must
    # refresh derived parameters after an independent runtime update exactly as
    # compiled mode does.
    exact_before_update = recurrence.evaluate(points, precision=50)
    recurrence.set_model_parameters({"particle.23.mass": 100.0})
    compiled.set_model_parameters({"particle.23.mass": 100.0})
    assert recurrence.evaluate(points, precision=50) != exact_before_update
    assert recurrence.evaluate(points) == pytest.approx(
        compiled.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )


@pytest.mark.parametrize("process_expression", _TOPOLOGY_REPLAY_PROCESSES)
def test_ufo_sm_lc_recurrence_artifact_loads_and_matches_compiled(
    tmp_path: Path,
    process_expression: str,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise public topology-replay artifacts with the prepared UFO-SM."""

    recurrence_artifact = tmp_path / "recurrence-ufo-sm"
    compiled_artifact = tmp_path / "compiled-ufo-sm"
    Generator(_generation_config("recurrence")).generate(
        process_expression,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(_generation_config("compiled")).generate(
        process_expression,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    _assert_topology_replay_structure(recurrence_artifact, process_expression)
    recurrence, compiled, points = _assert_topology_replay_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
    )
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )
    exact_before_update = recurrence.evaluate(points, precision=50)
    recurrence.set_model_parameters({"MZ": 100.0})
    compiled.set_model_parameters({"MZ": 100.0})
    assert recurrence.evaluate(points, precision=50) != exact_before_update
    _assert_topology_replay_exact_matches_compiled(
        recurrence,
        compiled,
        points,
    )


def test_builtin_lc_all_flow_union_recurrence_matches_compiled(
    tmp_path: Path,
) -> None:
    """Exercise all-flow union with runtime-selected helicity end to end."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence-union"
    compiled_artifact = tmp_path / "compiled-union"
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(
            _generation_config(
                "recurrence",
                lc_flow_layout="all-flow-union",
            )
        ).generate(
            _PROCESS,
            recurrence_artifact,
            model=ModelSource.from_path(prepared_model),
        )
    Generator(
        _generation_config(
            "compiled",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        _PROCESS,
        compiled_artifact,
    )

    point = tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in generic_validation_point(_PROCESS)
    )
    points = (point,)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)
    assert recurrence.physics.color_ids == compiled.physics.color_ids
    assert recurrence.physics.helicity_ids == compiled.physics.helicity_ids

    helicity_ids = recurrence.physics.helicity_ids
    selected_ids = tuple(
        dict.fromkeys(
            (
                helicity_ids[0],
                helicity_ids[len(helicity_ids) // 2],
                helicity_ids[-1],
                "h:-1,+1,-1,+1,-1",
            )
        )
    )
    assert set(selected_ids) <= set(helicity_ids)
    for helicity_id in selected_ids:
        recurrence_resolved = recurrence.evaluate_resolved(
            points,
            helicities=(helicity_id,),
        )
        compiled_resolved = compiled.evaluate_resolved(
            points,
            helicities=(helicity_id,),
        )
        assert recurrence_resolved.shape == compiled_resolved.shape
        assert _flatten(recurrence_resolved.values) == pytest.approx(
            _flatten(compiled_resolved.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        assert recurrence.evaluate(
            points,
            helicities=(helicity_id,),
        ) == pytest.approx(
            compiled.evaluate(points, helicities=(helicity_id,)),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        recurrence_exact = recurrence.evaluate_resolved(
            points,
            helicities=(helicity_id,),
            precision=32,
        )
        compiled_exact = compiled.evaluate_resolved(
            points,
            helicities=(helicity_id,),
            precision=32,
        )
        assert recurrence_exact.color_ids == compiled_exact.color_ids
        assert recurrence_exact.helicity_ids == compiled_exact.helicity_ids
        _assert_decimal_values_match(
            recurrence_exact.values,
            compiled_exact.values,
            32,
        )
        _assert_decimal_values_match(
            recurrence_exact.total(),
            recurrence.evaluate(
                points,
                helicities=(helicity_id,),
                precision=32,
            ),
            32,
        )


def test_recurrence_exact_accepts_all_flow_union_layout() -> None:
    """Accept both complete LC recurrence strategies."""

    _validate_execution(
        {
            "schema_version": 3,
            "kind": _RECURRENCE_KIND,
            "key": "d_dbar_to_z_g_g",
            "recurrence_plan_abi": "pyamplicol-recurrence-plan-v2",
            "runtime_layout_abi": "pyamplicol-recurrence-runtime-layout-v2",
            "recurrence_summary": {"lc_flow_layout": "all-flow-union"},
            "required_runtime_capabilities": sorted(_RECURRENCE_CAPABILITIES),
        },
        "d_dbar_to_z_g_g",
    )
