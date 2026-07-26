# SPDX-License-Identifier: 0BSD
"""Public generation/load canary for compact LC recurrence execution."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import CompiledModel, Generator, ModelSource, Runtime
from pyamplicol.api.errors import EvaluationError
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
_THREE_LINE_PROCESS = "d d~ > u u~ s s~"
_PURE_GLUON_PROCESS = "g g > g g"
_SAME_FLAVOUR_PROCESS = "d d~ > d d~"
_NEUTRAL_CURRENT_PROCESS = "d d~ > e+ e-"
_CHARGED_CURRENT_PROCESS = "u d~ > e+ ve"
_TWO_QUARK_LINE_PROCESS = "d d~ > t t~"
_CONTRACTED_COLOR_PROCESSES = (
    _PROCESS,
    _THREE_LINE_PROCESS,
    _PURE_GLUON_PROCESS,
    _SAME_FLAVOUR_PROCESS,
)
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
_CONTRACTED_RECURRENCE_CAPABILITIES = {
    "rusticol.recurrence-color.contracted.v1",
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
_REFERENCE_PAYLOAD = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "reference"
        / "physics-v2.json"
    ).read_text(encoding="utf-8")
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
    color_accuracy: str = "lc",
    lc_flow_layout: str = "topology-replay",
) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(
            accuracy=color_accuracy,
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


def _reference_case(case_id: str) -> dict[str, Any]:
    return next(case for case in _REFERENCE_PAYLOAD["cases"] if case["id"] == case_id)


def _reference_point(point_id: str) -> _Point:
    point = next(
        item for item in _REFERENCE_PAYLOAD["points"] if item["id"] == point_id
    )
    return tuple(
        tuple(float(component) for component in momentum)
        for momentum in point["momenta"]
    )


def _assert_recurrence_per_point_selector_patterns(
    artifact: Path,
    point: _Point,
    *,
    expected_layout: str,
) -> None:
    """Exercise recurrence-native selector grouping without a compiled fallback."""

    execution_files = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_files) == 1
    execution = json.loads(execution_files[0].read_text(encoding="utf-8"))
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_plan_abi"] == "pyamplicol-recurrence-plan-v2"
    assert execution["runtime_layout_abi"] == "pyamplicol-recurrence-runtime-layout-v2"
    assert execution["recurrence_summary"]["lc_flow_layout"] == expected_layout
    assert set(execution["required_runtime_capabilities"]) == _RECURRENCE_CAPABILITIES
    assert inspect_artifact(artifact).processes[0].execution_mode == "recurrence"

    runtime = Runtime.load(artifact)
    resolved = runtime.evaluate_resolved((point,))
    helicity_ids = tuple(
        helicity.id
        for helicity in runtime.physics.helicities
        if not helicity.structural_zero
    )
    color_ids = resolved.color_ids
    assert len(helicity_ids) >= 2
    assert len(color_ids) >= 2
    selector_pairs = (
        (helicity_ids[0], color_ids[0]),
        (helicity_ids[-1], color_ids[-1]),
    )
    patterns = {
        "homogeneous": (selector_pairs[0],) * 8,
        "alternating": selector_pairs * 4,
        "random": tuple(selector_pairs[index] for index in (1, 0, 1, 1, 0, 0, 1, 0)),
        "pre-grouped": (selector_pairs[0],) * 4 + (selector_pairs[1],) * 4,
    }

    for name, selectors in patterns.items():
        points = (point,) * len(selectors)
        actual = runtime.evaluate(
            points,
            helicity_by_point=tuple(selector[0] for selector in selectors),
            color_flow_by_point=tuple(selector[1] for selector in selectors),
        )
        expected = tuple(
            runtime.evaluate(
                (point,),
                helicities=(helicity_id,),
                color_flows=(color_id,),
            )[0]
            for helicity_id, color_id in selectors
        )
        assert actual == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-15,
        ), name


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


def _assert_contracted_color_artifacts_match(
    recurrence_artifact: Path,
    compiled_artifact: Path,
    process_expression: str,
    color_accuracy: str,
    *,
    parameter_update: tuple[str, float] | None = None,
) -> None:
    points = _validation_points(process_expression)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)
    assert recurrence.physics.color_accuracy == color_accuracy
    assert recurrence.physics.color_ids == ("color:contracted",)
    assert compiled.physics.color_ids == recurrence.physics.color_ids
    assert compiled.physics.helicity_ids == recurrence.physics.helicity_ids

    recurrence_total = recurrence.evaluate(points)
    compiled_total = compiled.evaluate(points)
    recurrence_resolved = recurrence.evaluate_resolved(points)
    compiled_resolved = compiled.evaluate_resolved(points)
    assert recurrence_resolved.total() == pytest.approx(
        recurrence_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_total == pytest.approx(
        compiled_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_resolved.shape == compiled_resolved.shape
    assert _flatten(recurrence_resolved.values) == pytest.approx(
        _flatten(compiled_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    nonzero_helicities = tuple(
        helicity.id
        for helicity in recurrence.physics.helicities
        if not helicity.structural_zero
    )
    assert nonzero_helicities
    sampled_helicities = tuple(
        dict.fromkeys(
            (
                nonzero_helicities[0],
                nonzero_helicities[len(nonzero_helicities) // 2],
                nonzero_helicities[-1],
            )
        )
    )
    recurrence_selected = recurrence.evaluate_resolved(
        points,
        helicities=sampled_helicities,
    )
    compiled_selected = compiled.evaluate_resolved(
        points,
        helicities=sampled_helicities,
    )
    assert _flatten(recurrence_selected.values) == pytest.approx(
        _flatten(compiled_selected.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    repeated_points = points * len(sampled_helicities)
    assert recurrence.evaluate(
        repeated_points,
        helicity_by_point=sampled_helicities,
    ) == pytest.approx(
        compiled.evaluate(
            repeated_points,
            helicity_by_point=sampled_helicities,
        ),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    with pytest.raises(EvaluationError, match="color-flow selection"):
        recurrence.evaluate(points, color_flows=("color:contracted",))

    for precision in (32, 50):
        recurrence_exact = recurrence.evaluate_resolved(
            points,
            helicities=sampled_helicities,
            precision=precision,
        )
        recurrence_f64 = recurrence.evaluate_resolved(
            points,
            helicities=sampled_helicities,
        )
        assert _flatten(recurrence_exact.values) == pytest.approx(
            _flatten(recurrence_f64.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        _assert_decimal_values_match(
            recurrence_exact.total(),
            recurrence.evaluate(
                points,
                helicities=sampled_helicities,
                precision=precision,
            ),
            precision,
        )

    manifest = load_manifest(recurrence_artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (
            recurrence_artifact
            / "processes"
            / process_id
            / "execution.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        execution["recurrence_summary"]["lc_flow_layout"]
        == "contracted-color-union"
    )
    assert (
        set(execution["required_runtime_capabilities"])
        == _CONTRACTED_RECURRENCE_CAPABILITIES
    )
    color_reference = execution["runtime_metadata"]["color_contraction"]
    inspection = inspect_artifact(recurrence_artifact).processes[0]
    assert inspection.execution_mode == "recurrence"
    assert inspection.recurrence_color_accuracy == color_accuracy
    assert inspection.recurrence_color_storage == color_reference["storage"]
    assert (
        inspection.recurrence_color_sector_count
        == color_reference["sector_count"]
    )
    assert (
        inspection.recurrence_color_active_sector_count
        == color_reference["active_sector_count"]
    )
    assert (
        inspection.recurrence_color_component_count
        == color_reference["component_count"]
    )
    assert inspection.recurrence_color_group_count == color_reference["group_count"]
    assert (
        inspection.recurrence_color_entry_count
        == color_reference["entry_count"]
    )
    assert (
        inspection.recurrence_color_logical_entry_count
        == color_reference["logical_entry_count"]
    )
    if process_expression == _THREE_LINE_PROCESS:
        # Six endpoint pairings are physical. The six orderings of each
        # disconnected open-string forest are coherent aliases and must have
        # one deterministic owner, not six duplicated amplitude destinations.
        assert color_reference["sector_count"] == 36
        assert color_reference["active_sector_count"] == 6
        assert color_reference["group_count"] == (
            6 * color_reference["component_count"]
        )
        assert color_reference["storage"] == "repeated"
    color_path = (
        recurrence_artifact
        / "processes"
        / process_id
        / color_reference["path"]
    )
    assert color_path.is_file()
    assert color_path.stat().st_size == color_reference["size_bytes"]
    payloads = {
        record.path: record
        for record in manifest.payloads
        if record.process_id == process_id
    }
    relative_color_path = color_path.relative_to(recurrence_artifact).as_posix()
    assert payloads[relative_color_path].sha256 == color_reference["sha256"]

    if parameter_update is not None:
        parameter_name, parameter_value = parameter_update
        exact_before_update = recurrence.evaluate(points, precision=50)
        recurrence.set_model_parameters({parameter_name: parameter_value})
        compiled.set_model_parameters({parameter_name: parameter_value})
        assert recurrence.evaluate(points, precision=50) != exact_before_update
        assert recurrence.evaluate(points) == pytest.approx(
            compiled.evaluate(points),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        recurrence_exact = recurrence.evaluate_resolved(points, precision=50)
        recurrence_f64 = recurrence.evaluate_resolved(points)
        assert _flatten(recurrence_exact.values) == pytest.approx(
            _flatten(recurrence_f64.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        _assert_decimal_values_match(
            recurrence_exact.total(),
            recurrence.evaluate(points, precision=50),
            50,
        )


def _contracted_structure_signature(artifact: Path) -> dict[str, object]:
    manifest = load_manifest(artifact)
    assert len(manifest.processes) == 1
    process_id = str(manifest.processes[0]["id"])
    execution = json.loads(
        (
            artifact / "processes" / process_id / "execution.json"
        ).read_text(encoding="utf-8")
    )
    summary = execution["recurrence_summary"]
    inspection = execution["plan"]["inspection_summary"]
    schedule = inspection["schedule"]
    construction = inspection["construction"]
    color = execution["runtime_metadata"]["color_contraction"]
    return {
        "layout": summary["lc_flow_layout"],
        "currents": summary["current_count"],
        "contributions": summary["contribution_count"],
        "closures": summary["closure_term_count"],
        "sources": schedule["source_row_count"],
        "finalizations": schedule["finalization_count"],
        "destinations": schedule["amplitude_destination_count"],
        "resolved_helicities": schedule["resolved_helicity_count"],
        "dynamic_color_states": construction["peak_dynamic_color_state_count"],
        "color_storage": color["storage"],
        "color_sectors": color["sector_count"],
        "active_color_sectors": color["active_sector_count"],
        "color_components": color["component_count"],
        "color_groups": color["group_count"],
        "color_entries": color["entry_count"],
        "logical_color_entries": color["logical_entry_count"],
        "factorization": color["factorization"],
    }


@pytest.fixture(scope="module")
def builtin_sm_recurrence_jit_o2_model() -> Iterator[ModelSource]:
    """Use a source-current prepared pack without weakening packaged checks."""

    override = os.environ.get("PYAMPLICOL_RECURRENCE_TEST_PREPARED_MODEL")
    if override:
        yield ModelSource.from_path(Path(override))
        return
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        yield ModelSource.from_path(prepared_model)


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
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Exercise the first public topology-replay artifact end to end."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence"
    compiled_artifact = tmp_path / "compiled"
    progress_events: list[object] = []

    Generator(
        _generation_config("recurrence"),
        progress=CallbackProgressSink(progress_events.append),
    ).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
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
    runtime_schedule = execution["plan"]["runtime_schedule"]
    assert runtime_schedule["storage_abi"] == "pacbin-v1"
    assert runtime_schedule["member_count"] >= 1
    assert execution["plan"]["process_binding"]["path"] == "recurrence-binding.bin"
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

    runtime_path = recurrence_artifact / runtime_schedule["path"]
    assert runtime_path.parent.parent.parent == recurrence_artifact / "recurrence"
    assert runtime_path.is_file()
    assert runtime_path.stat().st_size == runtime_schedule["size_bytes"]
    payloads = {record.path: record for record in manifest.payloads}
    runtime_payload = payloads[runtime_path.relative_to(recurrence_artifact).as_posix()]
    assert runtime_payload.size_bytes == runtime_path.stat().st_size
    assert runtime_payload.sha256 == runtime_schedule["sha256"]
    binding_path = process_root / execution["plan"]["process_binding"]["path"]
    assert binding_path.is_file()
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
    if process_expression == _PROCESS:
        _assert_recurrence_per_point_selector_patterns(
            recurrence_artifact,
            points[0],
            expected_layout="topology-replay",
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


@pytest.mark.parametrize(
    ("color_accuracy", "lc_flow_layout", "case_id"),
    (
        ("lc", "topology-replay", "case:sm_ddbar_ee:lc"),
        ("lc", "all-flow-union", "case:sm_ddbar_ee:lc"),
        ("nlc", "topology-replay", "case:sm_ddbar_ee:nlc"),
        ("full", "topology-replay", "case:sm_ddbar_ee:full"),
    ),
)
def test_builtin_neutral_current_recurrence_matches_legacy_oracle(
    tmp_path: Path,
    color_accuracy: str,
    lc_flow_layout: str,
    case_id: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Guard the incoming-spin average and mirrored fermion-pair orientation."""

    _require_native_recurrence()
    artifact = tmp_path / f"{color_accuracy}-{lc_flow_layout}"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy=color_accuracy,
            lc_flow_layout=lc_flow_layout,
        )
    ).generate(
        _NEUTRAL_CURRENT_PROCESS,
        artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )

    reference = _reference_case(case_id)
    observation = reference["observations"][0]
    reference_point = _reference_point(observation["point_id"])
    point = (
        reference_point[0],
        reference_point[1],
        reference_point[3],
        reference_point[2],
    )
    runtime = Runtime.load(artifact)
    resolved = runtime.evaluate_resolved((point,))
    helicity_values = {
        helicity.id: helicity.values for helicity in runtime.physics.helicities
    }
    actual = {
        (helicity_values[helicity_id], color_id): complex(
            resolved.values[0][helicity_index][color_index]
        )
        for helicity_index, helicity_id in enumerate(resolved.helicity_ids)
        for color_index, color_id in enumerate(resolved.color_ids)
    }
    expected = {
        (
            (
                helicity["values"][0],
                helicity["values"][1],
                helicity["values"][3],
                helicity["values"][2],
            ),
            color["id"],
        ): float(
            observation["values"][helicity_index][color_index]
        )
        for helicity_index, helicity in enumerate(reference["axes"]["helicities"])
        for color_index, color in enumerate(reference["axes"]["colors"])
    }

    assert set(actual) == set(expected)
    assert {key: value.real for key, value in actual.items()} == pytest.approx(
        expected,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert {key: value.imag for key, value in actual.items()} == pytest.approx(
        dict.fromkeys(actual, 0.0),
        abs=1.0e-15,
    )
    expected_total = float(observation["total"])
    assert runtime.evaluate((point,))[0] == pytest.approx(
        complex(expected_total),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert resolved.total()[0] == pytest.approx(
        complex(expected_total),
        rel=1.0e-12,
        abs=1.0e-15,
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


@pytest.mark.parametrize("process_expression", _CONTRACTED_COLOR_PROCESSES)
@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_builtin_contracted_color_recurrence_matches_compiled(
    tmp_path: Path,
    color_accuracy: str,
    process_expression: str,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Exercise built-in NLC/full generation and Direct-Arena contraction."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / f"recurrence-{color_accuracy}"
    compiled_artifact = tmp_path / f"compiled-{color_accuracy}"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        compiled_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    _assert_contracted_color_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
        color_accuracy,
        parameter_update=(
            ("particle.23.mass", 100.0)
            if process_expression == _PROCESS
            else None
        ),
    )


@pytest.mark.parametrize("process_expression", _CONTRACTED_COLOR_PROCESSES)
@pytest.mark.parametrize("color_accuracy", ("nlc", "full"))
def test_ufo_sm_contracted_color_recurrence_matches_compiled(
    tmp_path: Path,
    color_accuracy: str,
    process_expression: str,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise model-generic NLC/full recurrence with the UFO-SM pack."""

    recurrence_artifact = tmp_path / f"recurrence-ufo-{color_accuracy}"
    compiled_artifact = tmp_path / f"compiled-ufo-{color_accuracy}"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            color_accuracy=color_accuracy,
        )
    ).generate(
        process_expression,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    _assert_contracted_color_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
        color_accuracy,
        parameter_update=(
            ("MZ", 100.0) if process_expression == _PROCESS else None
        ),
    )


def test_builtin_and_ufo_contracted_recurrence_have_matching_structure(
    tmp_path: Path,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Equivalent SM models must construct the same generic color schedule."""

    artifacts = []
    for label, model in (
        ("builtin", builtin_sm_recurrence_jit_o2_model),
        ("ufo", ufo_sm_recurrence_jit_o2_model),
    ):
        artifact = tmp_path / label
        Generator(
            _generation_config(
                "recurrence",
                color_accuracy="full",
            )
        ).generate(
            _THREE_LINE_PROCESS,
            artifact,
            model=model,
        )
        artifacts.append(artifact)

    assert _contracted_structure_signature(
        artifacts[0]
    ) == _contracted_structure_signature(artifacts[1])


@pytest.mark.parametrize(
    ("process_expression", "required_color_id"),
    (
        (_PROCESS, None),
        (_CHARGED_CURRENT_PROCESS, None),
        (_TWO_QUARK_LINE_PROCESS, "flow:3,1,2,4"),
    ),
)
def test_builtin_lc_all_flow_union_recurrence_matches_compiled(
    tmp_path: Path,
    process_expression: str,
    required_color_id: str | None,
    builtin_sm_recurrence_jit_o2_model: ModelSource,
) -> None:
    """Exercise all-flow union report canaries through numerical execution."""

    _require_native_recurrence()
    recurrence_artifact = tmp_path / "recurrence-union"
    compiled_artifact = tmp_path / "compiled-union"
    Generator(
        _generation_config(
            "recurrence",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        process_expression,
        recurrence_artifact,
        model=builtin_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        process_expression,
        compiled_artifact,
    )
    _assert_all_flow_union_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        process_expression,
        parameter_update=(
            ("particle.23.mass", 100.0) if process_expression == _PROCESS else None
        ),
        required_color_id=required_color_id,
    )
    if process_expression == _PROCESS:
        _assert_recurrence_per_point_selector_patterns(
            recurrence_artifact,
            _validation_points(_PROCESS)[0],
            expected_layout="all-flow-union",
        )


def test_ufo_sm_lc_all_flow_union_recurrence_matches_compiled(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise the model-generic union lane with the prepared UFO-SM."""

    recurrence_artifact = tmp_path / "recurrence-ufo-union"
    compiled_artifact = tmp_path / "compiled-ufo-union"
    Generator(
        _generation_config(
            "recurrence",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        _PROCESS,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            lc_flow_layout="all-flow-union",
        )
    ).generate(
        _PROCESS,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    _assert_all_flow_union_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        _PROCESS,
        parameter_update=("MZ", 100.0),
    )


def test_ufo_sm_full_neutral_current_recurrence_executes_prepared_mass_slot(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Exercise the report's UFO full-colour source-only ``Me`` slot."""

    recurrence_artifact = tmp_path / "recurrence-ufo-full-neutral-current"
    compiled_artifact = tmp_path / "compiled-ufo-full-neutral-current"
    Generator(
        _generation_config(
            "recurrence",
            color_accuracy="full",
        )
    ).generate(
        _NEUTRAL_CURRENT_PROCESS,
        recurrence_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    Generator(
        _generation_config(
            "compiled",
            color_accuracy="full",
        )
    ).generate(
        _NEUTRAL_CURRENT_PROCESS,
        compiled_artifact,
        model=ufo_sm_recurrence_jit_o2_model,
    )
    runtimes = (
        Runtime.load(recurrence_artifact),
        Runtime.load(compiled_artifact),
    )
    for runtime in runtimes:
        electron_mass = next(
            parameter
            for parameter in runtime.physics.model_parameters
            if parameter.name == "Me"
        )
        assert electron_mass.kind == "derived"
        assert electron_mass.mutable is False
        assert electron_mass.default_real == 0.0
        assert electron_mass.default_imaginary == 0.0

    execution_path = next((recurrence_artifact / "processes").glob("*/execution.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    electron_mass_projection = next(
        row
        for row in execution["runtime_metadata"]["parameter_projection"]
        if row["runtime_name"] == "Me"
    )
    prepared_parameter_id = electron_mass_projection["prepared_parameter_id"]
    assert prepared_parameter_id is not None
    assert execution["runtime_metadata"]["prepared_parameter_defaults"][
        prepared_parameter_id
    ] == [0.0, 0.0]

    _assert_contracted_color_artifacts_match(
        recurrence_artifact,
        compiled_artifact,
        _NEUTRAL_CURRENT_PROCESS,
        "full",
        parameter_update=("aEWM1", 128.0),
    )


def _assert_all_flow_union_artifacts_match(
    recurrence_artifact: Path,
    compiled_artifact: Path,
    process_expression: str,
    *,
    parameter_update: tuple[str, float] | None = None,
    required_color_id: str | None = None,
) -> None:
    point = tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in generic_validation_point(process_expression)
    )
    points = (point,)
    recurrence = Runtime.load(recurrence_artifact)
    compiled = Runtime.load(compiled_artifact)
    assert recurrence.physics.color_ids == compiled.physics.color_ids
    assert recurrence.physics.helicity_ids == compiled.physics.helicity_ids

    recurrence_resolved = recurrence.evaluate_resolved(points)
    compiled_resolved = compiled.evaluate_resolved(points)
    assert recurrence_resolved.total() == pytest.approx(
        recurrence.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert compiled_resolved.total() == pytest.approx(
        compiled.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence_resolved.shape == compiled_resolved.shape
    assert _flatten(recurrence_resolved.values) == pytest.approx(
        _flatten(compiled_resolved.values),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert recurrence.evaluate(points) == pytest.approx(
        compiled.evaluate(points),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    helicity_ids = tuple(
        helicity.id
        for helicity in recurrence.physics.helicities
        if not helicity.structural_zero
    )
    assert helicity_ids
    selected_ids = tuple(
        dict.fromkeys(
            (
                helicity_ids[0],
                helicity_ids[len(helicity_ids) // 2],
                helicity_ids[-1],
                *(
                    ("h:-1,+1,-1,+1,-1",)
                    if process_expression == _PROCESS
                    else ()
                ),
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
    if required_color_id is not None:
        assert required_color_id in recurrence.physics.color_ids
        assert recurrence.evaluate(
            points,
            color_flows=(required_color_id,),
        ) == pytest.approx(
            compiled.evaluate(
                points,
                color_flows=(required_color_id,),
            ),
            rel=1.0e-12,
            abs=1.0e-15,
        )
        recurrence_selected = recurrence.evaluate_resolved(
            points,
            color_flows=(required_color_id,),
        )
        compiled_selected = compiled.evaluate_resolved(
            points,
            color_flows=(required_color_id,),
        )
        assert _flatten(recurrence_selected.values) == pytest.approx(
            _flatten(compiled_selected.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )
    if parameter_update is not None:
        parameter_name, parameter_value = parameter_update
        before = recurrence.evaluate(points, precision=50)
        recurrence.set_model_parameters({parameter_name: parameter_value})
        compiled.set_model_parameters({parameter_name: parameter_value})
        assert recurrence.evaluate(points, precision=50) != before
        for helicity_id in selected_ids:
            assert recurrence.evaluate(
                points,
                helicities=(helicity_id,),
            ) == pytest.approx(
                compiled.evaluate(points, helicities=(helicity_id,)),
                rel=1.0e-12,
                abs=1.0e-15,
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
