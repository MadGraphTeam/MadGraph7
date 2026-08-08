# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.agreements import independent_numerical_authorities
from tools.performance_report.cache import _validate_independent_authority_record
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import (
    _attach_baseline_validation,
    _baseline_matrix_element,
    _baseline_selector_contract,
    _measurement_selector_contract,
    _require_nonzero_lc_all_flow_baseline,
    _requires_high_precision_validation,
    _reuse_artifact_for_measurement,
    _stable_runtime_identity,
    _validate_runtime_identity_postflight,
    _validation_failure_precision_diagnostic,
    attach_validation_failure_precision_diagnostic,
    baseline_uses_lc_common_component_authority,
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


def _unchanged_diagnostic_projection(
    points: object, *, precision: int
) -> tuple[object, dict[str, object]]:
    assert precision == 200
    return points, {
        "abi": "pyamplicol-diagnostic-onshell-projection-v1",
        "kind": "per-leg-onshell-energy-v1",
        "precision_digits": precision,
        "unchanged": True,
        "projected_digest": "b" * 64,
    }


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


def test_lc_selector_uses_baseline_then_selected_flow_provider() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n7-dd-4q-lines-all-flow"
    )
    baseline_contract = _contract()
    provider_contract = SelectorContract(
        selected_color_flow_ids=("flow:3,2,1",),
        selected_color_words=((3, 2, 1),),
        all_flow_helicity_ids=("h:+1,-1,+1",),
        all_flow_source_helicities=((1, 1), (2, -1), (3, 1)),
        point_digest="b" * 64,
    )
    baseline = {"selector_contract": baseline_contract.as_dict()}
    matching_provider = {
        "status": ResultStatus.OK.value,
        "selector_contract": baseline_contract.as_dict(),
    }
    provider = {
        "status": ResultStatus.OK.value,
        "selector_contract": provider_contract.as_dict(),
    }

    assert _measurement_selector_contract(cell, baseline, matching_provider) == (
        baseline_contract
    )
    assert _measurement_selector_contract(cell, None, provider) == provider_contract


@pytest.mark.parametrize(
    "provider",
    (
        {"status": ResultStatus.ERROR.value},
        {"status": ResultStatus.OK.value},
        {
            "status": ResultStatus.OK.value,
            "selector_contract": {"selected_color_flow_ids": []},
        },
    ),
)
def test_invalid_lc_selector_provider_fails_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: dict[str, object],
) -> None:
    import tools.performance_report.measurement as report_measurement

    cell = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-lc-n7-dd-4q-lines-all-flow"
    )
    generation_called = False

    def unexpected_generation(*_args: object, **_kwargs: object) -> None:
        nonlocal generation_called
        generation_called = True

    monkeypatch.setattr(report_measurement, "generate_artifact", unexpected_generation)

    with pytest.raises((RunnerError, ValueError)):
        measure_pyamplicol_cell(
            cell,
            artifact_path=tmp_path / "artifact",
            settings=RunnerSettings(),
            repo_root=tmp_path,
            baseline=None,
            selector_provider=provider,
        )

    assert not generation_called


def test_differing_baseline_and_peer_selectors_fail_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.performance_report.measurement as report_measurement

    cell = REPORT_CATALOG.cell("matrix-compiled-builtin-sm-lc-n2-ud-epve-jets-all-flow")
    provider_contract = SelectorContract(
        selected_color_flow_ids=("flow:3,2,1",),
        selected_color_words=((3, 2, 1),),
        all_flow_helicity_ids=("h:+1,-1,+1",),
        all_flow_source_helicities=((1, 1), (2, -1), (3, 1)),
        point_digest="b" * 64,
    )
    baseline = {
        "status": ResultStatus.OK.value,
        "selector_contract": _contract().as_dict(),
        "validation": {"lc_common_component": {"value": 1.0}},
    }
    provider = {
        "status": ResultStatus.OK.value,
        "selector_contract": provider_contract.as_dict(),
    }
    generation_called = False

    def unexpected_generation(*_args: object, **_kwargs: object) -> None:
        nonlocal generation_called
        generation_called = True

    monkeypatch.setattr(report_measurement, "generate_artifact", unexpected_generation)

    with pytest.raises(RunnerError, match=r"baseline and selected-flow.*disagree"):
        measure_pyamplicol_cell(
            cell,
            artifact_path=tmp_path / "artifact",
            settings=RunnerSettings(),
            repo_root=tmp_path,
            baseline=baseline,
            expected_authority_cell_ids=tuple(
                authority.cell_id
                for authority in independent_numerical_authorities(cell)
            ),
            selected_authority_cell_id=independent_numerical_authorities(
                cell
            )[0].cell_id,
            selector_provider=provider,
        )

    assert not generation_called


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


def _legacy_lc_component_baseline() -> dict[str, object]:
    contract = _contract().as_dict()
    return {
        "status": ResultStatus.OK.value,
        "matrix_element": 3.736057085488201e-8,
        "selector_contract": contract,
        "validation": {
            "legacy_numerical_authority": {
                "abi": "pyamplicol-report-legacy-numerical-authority-v1",
                "source": "all-flow-selected-provider-replay",
            },
            "lc_common_component": {
                "abi": "pyamplicol-report-lc-common-component-v1",
                "cell_id": "reference-amplicol-lc-n4-dd-tt-jets-all-flow",
                "value": 2.0402809302299425e-10,
                "point_digest": contract["point_digest"],
                "helicity_ids": contract["all_flow_helicity_ids"],
                "color_flow_ids": contract["selected_color_flow_ids"],
            },
            "legacy_imode2_diagnostic": {
                "authoritative_source": "selected-flow-generated-library-component",
                "authoritative_value": 2.0402809302299425e-10,
                "imode2_value": 2.0400817525978893e-10,
            },
        },
    }


def test_lc_provider_replay_authenticates_component_not_aggregate() -> None:
    cell = REPORT_CATALOG.cell("matrix-recurrence-builtin-sm-lc-n4-dd-tt-jets-all-flow")
    baseline = _legacy_lc_component_baseline()
    validation: dict[str, object] = {}

    assert baseline_uses_lc_common_component_authority(cell, baseline)
    _attach_baseline_validation(
        cell,
        validation,
        candidate_matrix_element=3.736053706261443e-8,
        baseline=baseline,
    )

    assert "pointwise" not in validation


@pytest.mark.parametrize(
    ("cell_id", "candidate_text", "generated_library", "direct_imode2"),
    (
        (
            "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted",
            "1.0521884081060938437e-7",
            1.0521886363225842e-7,
            1.0521884081060942e-7,
        ),
        (
            "matrix-compiled-builtin-sm-nlc-n4-dd-tt-jets-contracted",
            "1.0434229425762874817e-7",
            1.0434231717649298e-7,
            1.0434229425762883e-7,
        ),
    ),
)
def test_n4_precision_diagnostic_retains_point_and_independent_authorities(
    cell_id: str,
    candidate_text: str,
    generated_library: float,
    direct_imode2: float,
) -> None:
    from decimal import Decimal

    cell = REPORT_CATALOG.cell(cell_id)
    candidate = Decimal(candidate_text)
    calls: list[int] = []

    class Resolved:
        def total(self) -> tuple[Decimal]:
            return (candidate,)

    class Runtime:
        _diagnostic_project_onshell = staticmethod(_unchanged_diagnostic_projection)

        def evaluate(
            self, _points: object, *, precision: int, **_selectors: object
        ) -> tuple[Decimal]:
            calls.append(precision)
            return (candidate,)

        def evaluate_resolved(
            self, _points: object, *, precision: int, **_selectors: object
        ) -> Resolved:
            assert precision in {32, 200}
            return Resolved()

    baseline = {
        "status": ResultStatus.OK.value,
        "matrix_element": generated_library,
        "validation": {
            "legacy_numerical_authority": {
                "abi": "pyamplicol-report-legacy-numerical-authority-v1",
                "source": "contracted-generated-library",
            },
            "legacy_imode2_diagnostic": {
                "authoritative_source": "dedicated-generated-library-probe",
                "authoritative_value": generated_library,
                "imode2_value": direct_imode2,
            },
        },
    }
    points = ((500.0, 0.0, 0.0, 500.0),)

    diagnostic = _validation_failure_precision_diagnostic(
        Runtime(),
        points,
        cell=cell,
        contract=None,
        baseline=baseline,
        measurement_context={
            "wall_seconds_per_point": 1.5e-6,
            "provenance": {
                "evaluator_total_timing": {"raw_seconds_per_point": 1.4e-6},
                "execution_timing": {"raw_seconds_per_point": 1.3e-6},
            },
        },
    )

    assert diagnostic["status"] == "diagnostic-only"
    assert diagnostic["promotes_measurement"] is False
    assert diagnostic["timings_unchanged"] is True
    assert diagnostic["execution_identity"] == {
        "cell_id": cell_id,
        "process": "d d~ > t t~ g g",
        "multiplicity": 4,
        "workload": "contracted",
        "layout": "contracted",
        "execution_mode": "compiled",
        "backend": "jit",
        "model": "builtin_sm",
        "accuracy": cell.measurement.accuracy.value,
        "variant": None,
    }
    assert diagnostic["measurement_timing_context"] == {
        "source": "copied-from-measurement",
        "recomputed": False,
        "outer_wall_seconds_per_point": 1.5e-6,
        "evaluator_total_timing": {"raw_seconds_per_point": 1.4e-6},
        "recurrence_core_timing": None,
    }
    assert diagnostic["retained_point"]["momenta"] == [[500.0, 0.0, 0.0, 500.0]]
    assert calls == [32, 200]
    assert [attempt["precision_digits"] for attempt in diagnostic["attempts"]] == [
        32,
        200,
    ]
    comparisons = diagnostic["attempts"][-1]["comparisons"]
    generated = [
        item
        for item in comparisons
        if item["authority"] == "baseline:generated-library"
    ]
    direct = [
        item for item in comparisons if item["authority"] == "baseline:direct-imode2"
    ]
    assert {item["candidate_value_kind"] for item in generated} == {
        "total",
        "resolved_sum",
    }
    assert {item["status"] for item in generated} == {
        ResultStatus.VALIDATION_FAILED.value
    }
    assert {item["status"] for item in direct} == {ResultStatus.OK.value}


def test_lc_precision_diagnostic_compares_shared_component() -> None:
    from decimal import Decimal

    cell = REPORT_CATALOG.cell("matrix-recurrence-builtin-sm-lc-n4-dd-tt-jets-all-flow")
    contract = _contract()
    aggregate = Decimal("3.7360537062614512535e-8")
    component = Decimal("2.0402809302299355e-10")
    points = ((500.0, 0.0, 0.0, 500.0),)
    received: list[object] = []
    projection_calls: list[tuple[object, int]] = []

    class Resolved:
        def __init__(self, component_only: bool) -> None:
            self.helicity_ids = (
                contract.runtime_all_flow_helicity_ids if component_only else ()
            )
            self.color_ids = contract.selected_color_flow_ids if component_only else ()
            self.values = (((component,),),) if component_only else ()

        def total(self) -> tuple[Decimal]:
            return (aggregate,)

    class Runtime:
        def _diagnostic_project_onshell(
            self, raw_points: object, *, precision: int
        ) -> tuple[object, dict[str, object]]:
            projection_calls.append((raw_points, precision))
            return _unchanged_diagnostic_projection(
                raw_points,
                precision=precision,
            )

        def evaluate(self, raw_points: object, **_selectors: object):
            received.append(raw_points)
            return (aggregate,)

        def evaluate_resolved(
            self,
            raw_points: object,
            *,
            color_flows: object = None,
            **_selectors: object,
        ) -> Resolved:
            received.append(raw_points)
            return Resolved(color_flows is not None)

    diagnostic = _validation_failure_precision_diagnostic(
        Runtime(),
        points,
        cell=cell,
        contract=contract,
        baseline=_legacy_lc_component_baseline(),
        measurement_context={
            "wall_seconds_per_point": 2.5e-6,
            "provenance": {
                "evaluator_total_timing": {"raw_seconds_per_point": 2.4e-6},
                "execution_timing": {"raw_seconds_per_point": 2.3e-6},
            },
        },
    )

    final = diagnostic["attempts"][-1]
    assert projection_calls == [(points, 200)]
    assert len(received) == 6
    assert all(raw_points is points for raw_points in received)
    assert diagnostic["measurement_timing_context"] == {
        "source": "copied-from-measurement",
        "recomputed": False,
        "outer_wall_seconds_per_point": 2.5e-6,
        "evaluator_total_timing": {"raw_seconds_per_point": 2.4e-6},
        "recurrence_core_timing": {"raw_seconds_per_point": 2.3e-6},
    }
    assert final["candidate"]["lc_common_component"]["value"] == str(component)
    comparisons = {item["authority"]: item for item in final["comparisons"]}
    assert comparisons["baseline:generated-library"]["status"] == (
        ResultStatus.OK.value
    )
    assert comparisons["baseline:direct-imode2"]["status"] == (
        ResultStatus.VALIDATION_FAILED.value
    )


def test_precision_diagnostic_stops_after_resolved_p32() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted"
    )
    calls: list[int] = []

    class Resolved:
        def total(self) -> tuple[float]:
            return (1.0,)

    class Runtime:
        _diagnostic_project_onshell = staticmethod(_unchanged_diagnostic_projection)

        def evaluate(self, _points: object, *, precision: int, **_selectors: object):
            calls.append(precision)
            return (1.0,)

        def evaluate_resolved(
            self, _points: object, *, precision: int, **_selectors: object
        ) -> Resolved:
            return Resolved()

    diagnostic = _validation_failure_precision_diagnostic(
        Runtime(),
        ((500.0, 0.0, 0.0, 500.0),),
        cell=cell,
        contract=None,
        baseline={"status": ResultStatus.OK.value, "matrix_element": 1.0},
    )

    assert calls == [32]
    assert [attempt["precision_digits"] for attempt in diagnostic["attempts"]] == [32]


def test_precision_diagnostic_records_p32_error_then_attempts_p200() -> None:
    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted"
    )
    calls: list[int] = []

    class Resolved:
        def total(self) -> tuple[float]:
            return (1.0,)

    class Runtime:
        _diagnostic_project_onshell = staticmethod(_unchanged_diagnostic_projection)

        def evaluate(self, _points: object, *, precision: int, **_selectors: object):
            calls.append(precision)
            if precision == 32:
                raise RuntimeError("p32 diagnostic failure")
            return (1.0,)

        def evaluate_resolved(
            self, _points: object, *, precision: int, **_selectors: object
        ) -> Resolved:
            return Resolved()

    diagnostic = _validation_failure_precision_diagnostic(
        Runtime(),
        ((500.0, 0.0, 0.0, 500.0),),
        cell=cell,
        contract=None,
        baseline={"status": ResultStatus.OK.value, "matrix_element": 1.0},
    )

    assert calls == [32, 200]
    assert diagnostic["attempts"][0]["status"] == "unavailable"
    assert diagnostic["attempts"][0]["error"]["message"] == ("p32 diagnostic failure")
    assert diagnostic["attempts"][1]["status"] == "evaluated"


def test_precision_diagnostic_reuses_projection_and_contextualizes_peers() -> None:
    from decimal import Decimal

    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted"
    )
    original = ((500.0, 0.0, 0.0, 500.0),)
    projected = (
        (Decimal("500.0000000000000001"), Decimal(0), Decimal(0), Decimal(500)),
    )
    received: list[object] = []
    projection_calls: list[tuple[object, int]] = []

    class Resolved:
        def total(self) -> tuple[Decimal]:
            return (Decimal("2"),)

    class Runtime:
        def _diagnostic_project_onshell(
            self, points: object, *, precision: int
        ) -> tuple[object, dict[str, object]]:
            projection_calls.append((points, precision))
            return projected, {
                "unchanged": False,
                "projected_digest": "c" * 64,
            }

        def evaluate(
            self, points: object, *, precision: int, **_selectors: object
        ) -> tuple[Decimal]:
            assert precision in {32, 200}
            received.append(points)
            return (Decimal("2"),)

        def evaluate_resolved(
            self, points: object, *, precision: int, **_selectors: object
        ) -> Resolved:
            assert precision in {32, 200}
            received.append(points)
            return Resolved()

    diagnostic = _validation_failure_precision_diagnostic(
        Runtime(),
        original,
        cell=cell,
        contract=None,
        baseline={"status": ResultStatus.OK.value, "matrix_element": 1.0},
    )

    assert projection_calls == [(original, 200)]
    assert len(received) == 4
    assert all(points is projected for points in received)
    assert (
        diagnostic["kinematic_projection"]["original_digest"]
        == (diagnostic["retained_point"]["digest"])
    )
    assert diagnostic["kinematic_projection"]["projected_digest"] == "c" * 64
    assert [attempt["precision_digits"] for attempt in diagnostic["attempts"]] == [
        32,
        200,
    ]
    for attempt in diagnostic["attempts"]:
        assert attempt["candidate"]["point_context"] == "projected-onshell-point"
        for comparison in attempt["comparisons"]:
            assert comparison["authority_point_context"] == "original-retained-point"
            assert comparison["same_kinematic_point"] is False
            assert comparison["certifying"] is False
            assert comparison["context_only"] is True
            assert "status" not in comparison


def test_precision_diagnostic_is_bounded_and_non_promoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-lc-n5-dd-z-jets-selected-flow"
    )
    measurement = {
        "status": ResultStatus.VALIDATION_FAILED.value,
        "validation": {"status": ResultStatus.VALIDATION_FAILED.value},
    }
    attach_validation_failure_precision_diagnostic(outside, measurement, baseline=None)
    assert "precision_diagnostic" not in measurement["validation"]

    cell = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n4-dd-tt-jets-contracted"
    )
    monkeypatch.setattr(
        "tools.performance_report.measurement.point_digest",
        lambda _points: (_ for _ in ()).throw(RuntimeError("diagnostic only")),
    )
    diagnostic = _validation_failure_precision_diagnostic(
        object(), object(), cell=cell, contract=None, baseline=None
    )
    assert diagnostic["status"] == "unavailable"
    assert diagnostic["promotes_measurement"] is False


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
            "numerical_relation_correctness": {
                "abi": "pyamplicol-numerical-current-relation-correctness-v1",
                "state": "no-applied-relations",
                "applied_relation_count": 0,
            },
            "numerical_relation_fallback": {
                "abi": "pyamplicol-numerical-current-reuse-fallback-v1",
                "requested_mode": "certified-reuse",
                "effective_mode": "off",
                "effective_reuse_state": "disabled",
                "reason": "evidence-envelope-fallback",
                "geometry": {"current_count": 4},
                "certified_relation_count": 0,
                "applied_relation_count": 0,
            },
        },
    }

    artifact = generated_artifact_from_measurement(measurement)

    assert artifact.generation_command_path == PUBLIC_CLI_COMMAND_PATH
    assert artifact.numerical_relation_correctness == {
        "abi": "pyamplicol-numerical-current-relation-correctness-v1",
        "state": "no-applied-relations",
        "applied_relation_count": 0,
    }
    assert artifact.numerical_relation_fallback == {
        "abi": "pyamplicol-numerical-current-reuse-fallback-v1",
        "requested_mode": "certified-reuse",
        "effective_mode": "off",
        "effective_reuse_state": "disabled",
        "reason": "evidence-envelope-fallback",
        "geometry": {"current_count": 4},
        "certified_relation_count": 0,
        "applied_relation_count": 0,
    }

    del measurement["provenance"]["generation_command_path"]
    legacy = generated_artifact_from_measurement(measurement)
    assert legacy.generation_command_path is None

    measurement["provenance"]["generation_command_path"] = 42
    with pytest.raises(RunnerError, match="reusable artifact metadata is malformed"):
        generated_artifact_from_measurement(measurement)

    measurement["provenance"]["generation_command_path"] = PUBLIC_CLI_COMMAND_PATH
    measurement["provenance"]["numerical_relation_fallback"] = 42
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
        numerical_relation_correctness={
            "abi": "pyamplicol-numerical-current-relation-correctness-v1",
            "state": "no-applied-relations",
            "applied_relation_count": 0,
        },
        numerical_relation_fallback={
            "abi": "pyamplicol-numerical-current-reuse-fallback-v1",
            "requested_mode": "certified-reuse",
            "effective_mode": "off",
            "effective_reuse_state": "disabled",
            "reason": "evidence-envelope-fallback",
            "geometry": {"current_count": 4},
            "certified_relation_count": 0,
            "applied_relation_count": 0,
        },
    )

    class Runtime:
        def evaluate(self, *_args: object, **_kwargs: object) -> list[float]:
            return [1.0]

    identity = {
        "loaded_module_origin_policy": {
            "observations": [],
        }
    }

    def resolved_record(precision: int) -> dict[str, object]:
        comparison = report_measurement.pointwise_validation(
            1.0,
            1.0,
            candidate_scale=1.0,
            baseline_scale=1.0,
            comparison_binding={
                "abi": "pyamplicol-report-resolved-component-scale-v1",
                "point_digest": "f" * 64,
                "helicity_ids": [],
                "color_flow_ids": [],
                "resolved_ordering_sha256": "a" * 64,
                "resolved_source_sha256": ("b" if precision == 16 else "c") * 64,
                "point_index": 0,
            },
        )
        return {
            "abi": "pyamplicol-report-resolved-sum-validation-v2",
            "status": ResultStatus.OK.value,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "maximum_conditioned_residual": 0.0,
            "relative_tolerance": 1.0e-12,
            "point_digest": "f" * 64,
            "helicity_ids": [],
            "color_flow_ids": [],
            "resolved_ordering_sha256": "a" * 64,
            "resolved_source_sha256": ("b" if precision == 16 else "c") * 64,
            "scale_source": "resolved-component-l1",
            "precision_digits": precision,
            "points": [comparison],
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
        "resolved_sum_validation": resolved_record(16),
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
    monkeypatch.setattr(
        report_measurement,
        "resolved_sum_validation",
        lambda *_args, precision=16, **_kwargs: resolved_record(precision),
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
    assert provenance["numerical_relation_fallback"]["reason"] == (
        "evidence-envelope-fallback"
    )
    assert provenance["report_momenta"] == [[1.0]]
    assert (
        provenance["runtime_profile"]["report_command_path"]
        == LOADED_RUNTIME_PROFILE_COMMAND_PATH
    )


@pytest.mark.parametrize(
    (
        "cell_id",
        "authority_available",
        "standalone",
        "p32_matches",
        "custom_authority",
    ),
    (
        (
            "matrix-compiled-builtin-sm-nlc-n1-dd-z-jets-contracted",
            True,
            False,
            True,
            False,
        ),
        (
            "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted",
            True,
            False,
            True,
            False,
        ),
        (
            "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted",
            False,
            False,
            True,
            False,
        ),
        (
            "scalar-contact-n2-scalar-contact-contracted",
            False,
            True,
            True,
            False,
        ),
        (
            "scalar-gravity-n2-scalar-gravity-contracted",
            False,
            True,
            True,
            False,
        ),
        (
            "scalar-contact-n2-scalar-contact-contracted",
            False,
            True,
            False,
            False,
        ),
        (
            "scalar-contact-n2-scalar-contact-contracted",
            True,
            False,
            True,
            True,
        ),
    ),
)
def test_initial_compiled_authority_uses_nested_resolved_point_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_id: str,
    authority_available: bool,
    standalone: bool,
    p32_matches: bool,
    custom_authority: bool,
) -> None:
    import tools.performance_report.measurement as report_measurement

    cell = REPORT_CATALOG.cell(cell_id)

    class CustomAuthorityCatalog:
        """Make one scalar use the ordinary matrix authority chain."""

        def measurement_cells(self):
            return REPORT_CATALOG.measurement_cells()

        def validation_baseline_cell(self, candidate):
            if candidate == cell:
                return REPORT_CATALOG.cell(
                    "matrix-recurrence-builtin-sm-full-n1-dd-z-jets-contracted"
                )
            return REPORT_CATALOG.validation_baseline_cell(candidate)

    catalog = CustomAuthorityCatalog() if custom_authority else REPORT_CATALOG
    validation_points = ((1.0,),)
    point_identity = report_measurement.point_digest(validation_points)
    source = "b" * 64
    resolved_point = report_measurement.pointwise_validation(
        1.0,
        1.0,
        candidate_scale=1.0,
        baseline_scale=1.0,
        comparison_binding={
            "abi": "pyamplicol-report-resolved-component-scale-v1",
            "point_digest": point_identity,
            "helicity_ids": [],
            "color_flow_ids": [],
            "resolved_ordering_sha256": "a" * 64,
            "resolved_source_sha256": source,
            "point_index": 0,
        },
    )
    resolved = {
        "abi": "pyamplicol-report-resolved-sum-validation-v2",
        "status": ResultStatus.OK.value,
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
        "maximum_conditioned_residual": 0.0,
        "relative_tolerance": 1.0e-12,
        "point_digest": point_identity,
        "helicity_ids": [],
        "color_flow_ids": [],
        "resolved_ordering_sha256": "a" * 64,
        "resolved_source_sha256": source,
        "scale_source": "resolved-component-l1",
        "precision_digits": 16,
        "points": [resolved_point],
    }
    p32_resolved = {**deepcopy(resolved), "precision_digits": 32}
    if not p32_matches:
        p32_resolved["points"] = [
            report_measurement.pointwise_validation(
                2.0,
                2.0,
                candidate_scale=2.0,
                baseline_scale=2.0,
                comparison_binding={
                    "abi": "pyamplicol-report-resolved-component-scale-v1",
                    "point_digest": point_identity,
                    "helicity_ids": [],
                    "color_flow_ids": [],
                    "resolved_ordering_sha256": "a" * 64,
                    "resolved_source_sha256": source,
                    "point_index": 0,
                },
            )
        ]
    artifact_path = tmp_path / "artifact"
    artifact = GeneratedArtifact(
        path=artifact_path,
        process_id="process",
        generation_seconds=1.0,
        model_preparation_seconds=0.0,
        model_preparation_reused=True,
        requested_config={},
        effective_config={},
    )
    profile = {
        "status": ResultStatus.OK.value,
        "wall_seconds_per_point": 1.0e-6,
        "execution_seconds_per_point": 0.8e-6,
        "matrix_element": 1.0,
        "sample_count": 5,
        "standard_error_seconds_per_point": 0.0,
        "relative_standard_error": 0.0,
        "execution_timing": {},
        "arena_profile_evidence": {},
        "benchmark_evidence": {},
        "resolved_sum_validation": resolved,
    }
    identity = {"loaded_module_origin_policy": {"observations": []}}

    class Runtime:
        def _diagnostic_project_onshell(
            self,
            points: object,
            *,
            precision: int,
        ) -> tuple[object, dict[str, object]]:
            assert precision == 200
            return points, {"unchanged": True, "projected_digest": point_identity}

        def evaluate(self, *_args: object, **_kwargs: object) -> list[float]:
            return [1.0]

        def evaluate_resolved(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                total=lambda: (1.0,),
                values=(((1.0,),),),
                helicity_ids=(),
                color_ids=(),
            )
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
        lambda _process: validation_points,
    )
    if standalone or cell.measurement.model in {
        report_measurement.ModelKey.SCALAR_CONTACT,
        report_measurement.ModelKey.SCALAR_GRAVITY,
    }:
        monkeypatch.setattr(
            report_measurement,
            "runtime_validation_points",
            lambda _runtime: validation_points,
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
    monkeypatch.setattr(
        report_measurement,
        "resolved_sum_validation",
        lambda *_args, **_kwargs: (
            pytest.fail("p32 is diagnostic-only with authority")
            if authority_available and not custom_authority
            else deepcopy(p32_resolved)
        ),
    )
    authorities = independent_numerical_authorities(cell, catalog=catalog)  # type: ignore[arg-type]
    baseline = {
        "status": ResultStatus.OK.value,
        "matrix_element": 1.0,
        "selector_contract": None,
        "validation": {"point_digest": point_identity},
        "provenance": {},
    }

    measurement = measure_pyamplicol_cell(
        cell,
        artifact_path=artifact_path,
        settings=RunnerSettings(source_revision_override="a" * 40),
        repo_root=tmp_path,
        baseline=baseline if authority_available else None,
        expected_authority_cell_ids=tuple(item.cell_id for item in authorities),
        selected_authority_cell_id=(
            authorities[0].cell_id if authority_available else None
        ),
        catalog=catalog,  # type: ignore[arg-type]
    )

    if custom_authority:
        validation = measurement["validation"]
        assert isinstance(validation, dict)
        _validate_independent_authority_record(
            validation,
            expected_cell=cell,
            expected_status="verified",
            catalog=catalog,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="not applicable"):
            _validate_independent_authority_record(
                validation,
                expected_cell=cell,
                expected_status="verified",
                catalog=REPORT_CATALOG,
            )

    validation = measurement["validation"]
    if authority_available:
        assert measurement["status"] == ResultStatus.OK.value
        assert validation["pointwise"]["status"] == ResultStatus.OK.value
        assert validation["independent_authority"]["selected_cell_id"] == (
            authorities[0].cell_id
        )
    elif standalone and p32_matches:
        assert authorities == ()
        assert measurement["status"] == ResultStatus.OK.value
        assert measurement["failure"] is None
        assert validation["high_precision"]["status"] == ResultStatus.OK.value
        assert "independent_authority" not in validation
        assert "precision_diagnostic" not in validation
    elif standalone:
        assert authorities == ()
        assert measurement["status"] == ResultStatus.VALIDATION_FAILED.value
        assert measurement["failure"]["kind"] == "MeasurementValidationError"
        assert validation["high_precision"]["status"] == (
            ResultStatus.VALIDATION_FAILED.value
        )
        assert "independent_authority" not in validation
    else:
        assert measurement["status"] == ResultStatus.UNVERIFIED.value
        assert measurement["failure"]["kind"] == "IndependentAuthorityUnavailable"
        assert validation["independent_authority"]["status"] == "unavailable"
        assert [
            attempt["precision_digits"]
            for attempt in validation["precision_diagnostic"]["attempts"]
        ] == [32, 200]


def test_catalog_contains_no_amplicol_candidate_matrix_cell() -> None:
    assert all(
        cell.measurement.execution_mode.value != "amplicol"
        for cell in REPORT_CATALOG.matrix_cells()
    )


def test_legacy_amplicol_diagnostic_does_not_replace_p32_certification() -> None:
    recurrence = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    legacy = {"status": ResultStatus.OK.value, "matrix_element": 1.0}

    assert _requires_high_precision_validation(
        recurrence,
        baseline=legacy,
        selected_authority_cell_id=None,
        catalog=REPORT_CATALOG,
    ) == (True, True)

    compiled = REPORT_CATALOG.cell(
        "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    recurrence_authority = {"status": ResultStatus.OK.value, "matrix_element": 1.0}
    assert _requires_high_precision_validation(
        compiled,
        baseline=recurrence_authority,
        selected_authority_cell_id=recurrence.cell_id,
        catalog=REPORT_CATALOG,
    ) == (False, False)


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
