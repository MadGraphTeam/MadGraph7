# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from weakref import ReferenceType, ref

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tools.performance_report import cache as report_cache
from tools.performance_report import measurement as report_measurement
from tools.performance_report import runner as report_runner
from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_FIELD,
    OTF_COMPILED_CROSS_MODE,
    incoming_agreement_edges,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import Accuracy, CellSpec, ExecutionMode, Workload
from tools.performance_report.runner import (
    OTF_DUAL_AUTHORITY_VALIDATION_FIELD,
    GeneratedArtifact,
    RunnerError,
    RunnerSettings,
    SelectorContract,
    config_values,
    derive_selector_contract,
    on_the_fly_dual_authority_validation,
    point_digest,
    runtime_identity_payload,
    validate_runtime_contract,
    validate_selector_contract,
)

POINTS = (((1.0,),), ((2.0,),))
FLOW_IDS = ("flow:1,2", "flow:2,1")
HELICITY_IDS = ("h:+1,-1", "h:-1,+1")


def _otf_cell(workload: Workload):
    return next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
        and cell.measurement.accuracy is Accuracy.LC
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is workload
    )


def _authorities(cell):
    recurrence = REPORT_CATALOG.validation_baseline_cell(cell)
    assert recurrence is not None
    compiled = next(
        edge.baseline
        for edge in incoming_agreement_edges(cell)
        if edge.kind == OTF_COMPILED_CROSS_MODE
    )
    return recurrence, compiled


def _contract() -> SelectorContract:
    return SelectorContract(
        selected_color_flow_ids=(FLOW_IDS[0],),
        selected_color_words=((1, 2),),
        all_flow_helicity_ids=(HELICITY_IDS[0],),
        all_flow_source_helicities=((1, 1), (2, -1)),
        point_digest=point_digest(POINTS),
    )


class _Runtime:
    def __init__(
        self,
        mode: ExecutionMode,
        artifact_character: str,
        *,
        reverse_resolved_axis: bool = False,
        mutate_nonfirst_component: bool = False,
        inconsistent_total: bool = False,
        structural_zero_residue: float | None = None,
    ) -> None:
        self.execution_mode = mode.value
        self.artifact_id = artifact_character * 64
        self.reverse_resolved_axis = reverse_resolved_axis
        self.mutate_nonfirst_component = mutate_nonfirst_component
        self.inconsistent_total = inconsistent_total
        self.structural_zero_residue = structural_zero_residue
        self.events: list[tuple[object, ...]] = []
        self.physics = SimpleNamespace(
            color_accuracy="lc",
            selector_capabilities=("helicity", "color_flow"),
            color_flows=(
                SimpleNamespace(id=FLOW_IDS[0], word=(1, 2)),
                SimpleNamespace(id=FLOW_IDS[1], word=(2, 1)),
            ),
            helicities=(
                SimpleNamespace(id=HELICITY_IDS[0], values=(1, -1)),
                SimpleNamespace(id=HELICITY_IDS[1], values=(-1, 1)),
            ),
            external_particles=(
                SimpleNamespace(label=1),
                SimpleNamespace(label=2),
            ),
        )

    def _on_the_fly_benchmark_context(
        self,
        requested: tuple[str, ...],
    ) -> dict[str, object]:
        assert self.execution_mode == ExecutionMode.ON_THE_FLY.value
        return {
            "process_id": "compact-candidate",
            "process_expression": "d d~ > z",
            "color_accuracy": "lc",
            "helicity_count": len(HELICITY_IDS),
            "color_count": len(FLOW_IDS),
            "selected_color_ids": list(requested),
        }

    def _point_selector_indices(
        self,
        values: tuple[str, ...],
        name: str,
    ) -> tuple[int, ...]:
        assert self.execution_mode == ExecutionMode.ON_THE_FLY.value
        available = HELICITY_IDS if name == "helicity_by_point" else FLOW_IDS
        return tuple(available.index(identifier) for identifier in values)

    def evaluate(
        self,
        points: object,
        *,
        helicities: tuple[str, ...] | None = None,
        color_flows: tuple[str, ...] | None = None,
        precision: int = 16,
    ) -> tuple[float, ...]:
        assert precision == 16
        self.events.append(("evaluate", helicities, color_flows))
        _selected_helicities, _selected_colors, values = self._resolved_payload(
            points,
            helicities=helicities,
            color_flows=color_flows,
        )
        totals = [
            sum(component for row in point for component in row) for point in values
        ]
        if self.inconsistent_total:
            totals[-1] += 1.0
        return tuple(totals)

    def _resolved_payload(
        self,
        points: object,
        *,
        helicities: tuple[str, ...] | None,
        color_flows: tuple[str, ...] | None,
    ) -> tuple[list[str], list[str], list[list[list[float]]]]:
        selected_helicities = list(HELICITY_IDS if helicities is None else helicities)
        selected_colors = list(FLOW_IDS if color_flows is None else color_flows)
        if self.reverse_resolved_axis:
            if len(selected_helicities) > 1:
                selected_helicities.reverse()
            elif len(selected_colors) > 1:
                selected_colors.reverse()
        first_component = (selected_helicities[0], selected_colors[0])
        last_component = (selected_helicities[-1], selected_colors[-1])
        values: list[list[list[float]]] = []
        for point_index, _point in enumerate(points):  # type: ignore[arg-type]
            point_rows: list[list[float]] = []
            for helicity_id in selected_helicities:
                row: list[float] = []
                for color_id in selected_colors:
                    value = (
                        1.0
                        + 10.0 * point_index
                        + 2.0 * HELICITY_IDS.index(helicity_id)
                        + FLOW_IDS.index(color_id)
                    )
                    component = (helicity_id, color_id)
                    if (
                        self.structural_zero_residue is not None
                        and point_index == 0
                        and component == (HELICITY_IDS[0], FLOW_IDS[0])
                    ):
                        value = self.structural_zero_residue
                    if self.mutate_nonfirst_component and point_index == 1:
                        if component == first_component:
                            value -= 1.0
                        if component == last_component:
                            value += 1.0
                    row.append(value)
                point_rows.append(row)
            values.append(point_rows)
        return selected_helicities, selected_colors, values

    def evaluate_resolved(
        self,
        points: object,
        *,
        helicities: tuple[str, ...] | None = None,
        color_flows: tuple[str, ...] | None = None,
        precision: int = 16,
    ) -> SimpleNamespace:
        assert precision == 16
        self.events.append(("resolved", helicities, color_flows))
        selected_helicities, selected_colors, values = self._resolved_payload(
            points,
            helicities=helicities,
            color_flows=color_flows,
        )
        frozen_values = tuple(
            tuple(tuple(row) for row in point_rows) for point_rows in values
        )

        def total() -> tuple[float, ...]:
            self.events.append(("resolved-total", helicities, color_flows))
            return tuple(
                sum(component for row in point for component in row)
                for point in frozen_values
            )

        return SimpleNamespace(
            helicity_ids=tuple(selected_helicities),
            color_ids=tuple(selected_colors),
            values=frozen_values,
            total=total,
        )

    def clear(self) -> None:
        self.events.append(("clear",))


class _CompactBackend:
    def __init__(
        self,
        cell: CellSpec,
        *,
        context_updates: dict[str, object] | None = None,
        ordinal_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self.cell = cell
        self.context_updates = context_updates or {}
        self.ordinal_overrides = ordinal_overrides or {}
        self.context_requests: list[tuple[str, ...]] = []
        self.ordinal_requests: list[tuple[str, tuple[str, ...]]] = []

    def _on_the_fly_benchmark_context(
        self,
        requested: tuple[str, ...],
    ) -> dict[str, object]:
        self.context_requests.append(requested)
        value = {
            "process_id": "compact-candidate",
            "process_expression": self.cell.process,
            "color_accuracy": "lc",
            "helicity_count": len(HELICITY_IDS),
            "color_count": len(FLOW_IDS),
            "selected_color_ids": list(requested),
        }
        value.update(self.context_updates)
        return value

    def _point_selector_indices(
        self,
        values: tuple[str, ...],
        name: str,
    ) -> tuple[int, ...]:
        frozen = tuple(values)
        self.ordinal_requests.append((name, frozen))
        override = self.ordinal_overrides.get(name)
        if override is not None:
            return override
        available = HELICITY_IDS if name == "helicity_by_point" else FLOW_IDS
        return tuple(available.index(identifier) for identifier in frozen)


class _CompactCandidate:
    execution_mode = ExecutionMode.ON_THE_FLY.value
    artifact_id = "a" * 64

    def __init__(
        self,
        cell: CellSpec,
        *,
        context_updates: dict[str, object] | None = None,
        ordinal_overrides: dict[str, tuple[int, ...]] | None = None,
    ) -> None:
        self._delegate = _Runtime(ExecutionMode.ON_THE_FLY, "a")
        self._backend = _CompactBackend(
            cell,
            context_updates=context_updates,
            ordinal_overrides=ordinal_overrides,
        )
        self.dense_physics_access_count = 0

    @property
    def physics(self) -> object:
        self.dense_physics_access_count += 1
        raise AssertionError("compact candidate opened dense physics")

    @property
    def events(self) -> list[tuple[object, ...]]:
        return self._delegate.events

    def evaluate(self, *args: object, **kwargs: object) -> tuple[float, ...]:
        return self._delegate.evaluate(*args, **kwargs)

    def evaluate_resolved(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return self._delegate.evaluate_resolved(*args, **kwargs)

    def clear(self) -> None:
        self._delegate.clear()


def _valid_gate(workload: Workload):
    cell = _otf_cell(workload)
    recurrence_cell, compiled_cell = _authorities(cell)
    candidate = _Runtime(ExecutionMode.ON_THE_FLY, "a")
    recurrence = _Runtime(
        ExecutionMode.RECURRENCE,
        "b",
        reverse_resolved_axis=True,
    )
    compiled = _Runtime(ExecutionMode.COMPILED, "c")
    contract = _contract()
    record = on_the_fly_dual_authority_validation(
        candidate,
        POINTS,
        cell=cell,
        selector_contract=contract,
        authorities=(
            ("recurrence", recurrence_cell, recurrence),
            ("compiled", compiled_cell, compiled),
        ),
    )
    return cell, contract, record, candidate, recurrence, compiled


@pytest.mark.parametrize("workload", (Workload.SELECTED_FLOW, Workload.ALL_FLOW))
def test_dual_authority_candidate_uses_only_compact_selector_contract(
    workload: Workload,
) -> None:
    cell = _otf_cell(workload)
    recurrence_cell, compiled_cell = _authorities(cell)
    candidate = _CompactCandidate(cell)

    record = on_the_fly_dual_authority_validation(
        candidate,
        POINTS,
        cell=cell,
        selector_contract=_contract(),
        authorities=(
            (
                "recurrence",
                recurrence_cell,
                _Runtime(ExecutionMode.RECURRENCE, "b"),
            ),
            (
                "compiled",
                compiled_cell,
                _Runtime(ExecutionMode.COMPILED, "c"),
            ),
        ),
    )

    assert record["status"] == "ok"
    assert candidate.dense_physics_access_count == 0
    assert candidate._backend.context_requests == [(), ()]
    assert candidate._backend.ordinal_requests == [
        ("color_flow_by_point", (FLOW_IDS[0],)),
        ("helicity_by_point", (HELICITY_IDS[0],)),
    ]


@pytest.mark.parametrize(
    ("context_updates", "message"),
    (
        ({"process_expression": "g g > g"}, "identity differs"),
        ({"helicity_count": 0}, "helicity_count is invalid"),
        ({"color_count": False}, "color_count is invalid"),
    ),
)
def test_compact_runtime_contract_rejects_invalid_identity_or_counts(
    context_updates: dict[str, object],
    message: str,
) -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    candidate = _CompactCandidate(cell, context_updates=context_updates)

    with pytest.raises(RunnerError, match=message):
        validate_runtime_contract(cell, candidate)

    assert candidate.dense_physics_access_count == 0


@pytest.mark.parametrize(
    ("name", "count"),
    (
        ("color_flow_by_point", len(FLOW_IDS)),
        ("helicity_by_point", len(HELICITY_IDS)),
    ),
)
def test_compact_selector_contract_rejects_out_of_range_ordinal(
    name: str,
    count: int,
) -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    candidate = _CompactCandidate(
        cell,
        ordinal_overrides={name: (count,)},
    )

    with pytest.raises(RunnerError, match=rf"invalid {name} ordinals"):
        validate_selector_contract(
            candidate,
            _contract(),
            POINTS,
            cell=cell,
        )

    assert candidate.dense_physics_access_count == 0


def test_selector_derivation_rejects_otf_before_dense_physics_access() -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    candidate = _CompactCandidate(cell)

    with pytest.raises(RunnerError, match="must be inherited from a dense authority"):
        derive_selector_contract(candidate, POINTS)

    assert candidate.dense_physics_access_count == 0
    assert candidate._backend.context_requests == []


@pytest.mark.parametrize("validation", ("runtime", "selector"))
def test_compact_runtime_rejects_non_otf_cell_before_dense_physics_access(
    validation: str,
) -> None:
    otf_cell = _otf_cell(Workload.SELECTED_FLOW)
    recurrence_cell, _compiled_cell = _authorities(otf_cell)
    candidate = _CompactCandidate(otf_cell)

    with pytest.raises(RunnerError, match="wrong execution mode"):
        if validation == "runtime":
            validate_runtime_contract(recurrence_cell, candidate)
        else:
            validate_selector_contract(
                candidate,
                _contract(),
                POINTS,
                cell=recurrence_cell,
            )

    assert candidate.dense_physics_access_count == 0
    assert candidate._backend.context_requests == []
    assert candidate._backend.ordinal_requests == []


@pytest.mark.parametrize("workload", (Workload.SELECTED_FLOW, Workload.ALL_FLOW))
def test_dual_authority_gate_aligns_semantic_axes_and_resets_before_profile(
    workload: Workload,
) -> None:
    _cell, _contract_value, record, candidate, recurrence, compiled = _valid_gate(
        workload
    )
    expected_selectors = (
        (None, (FLOW_IDS[0],))
        if workload is Workload.SELECTED_FLOW
        else ((HELICITY_IDS[0],), None)
    )

    assert candidate.events == [
        ("evaluate", *expected_selectors),
        ("resolved", *expected_selectors),
        ("resolved-total", *expected_selectors),
        ("clear",),
        ("evaluate", *expected_selectors),
        ("resolved", *expected_selectors),
        ("resolved-total", *expected_selectors),
        ("clear",),
    ]
    assert recurrence.events == [
        ("evaluate", *expected_selectors),
        ("resolved", *expected_selectors),
        ("resolved-total", *expected_selectors),
    ]
    assert compiled.events == recurrence.events
    assert record["lifecycle"]["clear_call_count"] == 2
    assert record["lifecycle"]["final_clear_before_profile"] is True
    assert record["resolved_check_count"] == 4
    assert all(
        authority["resolved_sum"]["maximum_conditioned_residual"] == 0.0
        for authority in record["authorities"]
    )
    assert all(
        stage["candidate_resolved_sum"]["maximum_conditioned_residual"] == 0.0
        for stage in (record["before_clear"], record["after_clear"])
    )
    assert (
        record["authorities"][0]["resolved_ordering_sha256"]
        != record["before_clear"]["candidate_resolved_ordering_sha256"]
    )
    assert all(
        comparison["resolved_components"]["maximum_conditioned_residual"] == 0.0
        for stage in (record["before_clear"], record["after_clear"])
        for comparison in stage["comparisons"]
    )


def test_mutated_nonfirst_component_fails_before_public_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    recurrence_cell, compiled_cell = _authorities(cell)
    candidate = _Runtime(
        ExecutionMode.ON_THE_FLY,
        "a",
        mutate_nonfirst_component=True,
    )
    recurrence = _Runtime(ExecutionMode.RECURRENCE, "b")
    compiled = _Runtime(ExecutionMode.COMPILED, "c")
    contract = _contract()
    generated = GeneratedArtifact(
        path=tmp_path / "candidate",
        process_id="process",
        generation_seconds=1.0,
        model_preparation_seconds=0.0,
        model_preparation_reused=True,
        requested_config={},
        effective_config={},
    )
    profile_calls: list[object] = []

    monkeypatch.setattr(
        report_measurement, "generate_artifact", lambda *_a, **_k: generated
    )
    monkeypatch.setattr(
        report_measurement, "validate_artifact_contract", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        report_measurement, "_load_runtime", lambda *_a, **_k: candidate
    )
    monkeypatch.setattr(
        report_measurement,
        "runtime_identity_payload",
        lambda *_a, **_k: {"loaded_module_origin_policy": {"observations": []}},
    )
    monkeypatch.setattr(
        report_measurement, "shared_validation_points", lambda _p: POINTS
    )
    monkeypatch.setattr(
        report_measurement,
        "_load_on_the_fly_authority_runtimes",
        lambda *_a, **_k: (
            ("recurrence", recurrence_cell, recurrence),
            ("compiled", compiled_cell, compiled),
        ),
    )
    monkeypatch.setattr(
        report_measurement,
        "profile_runtime",
        lambda *_a, **_k: profile_calls.append(object()),
    )
    baseline = {
        "status": "ok",
        "matrix_element": 1.0,
        "selector_contract": contract.as_dict(),
    }

    with pytest.raises(RunnerError, match="resolved components disagrees"):
        report_measurement.measure_pyamplicol_cell(
            cell,
            artifact_path=tmp_path / "artifact",
            settings=RunnerSettings(source_revision_override="d" * 40),
            repo_root=tmp_path,
            baseline=baseline,
        )

    assert profile_calls == []
    assert ("clear",) not in candidate.events


def test_pure_gluon_structural_zero_residue_uses_resolved_slice_scale() -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    recurrence_cell, compiled_cell = _authorities(cell)
    record = on_the_fly_dual_authority_validation(
        _Runtime(
            ExecutionMode.ON_THE_FLY,
            "a",
            structural_zero_residue=1.0e-30,
        ),
        POINTS,
        cell=cell,
        selector_contract=_contract(),
        authorities=(
            (
                "recurrence",
                recurrence_cell,
                _Runtime(
                    ExecutionMode.RECURRENCE,
                    "b",
                    reverse_resolved_axis=True,
                    structural_zero_residue=0.0,
                ),
            ),
            (
                "compiled",
                compiled_cell,
                _Runtime(
                    ExecutionMode.COMPILED,
                    "c",
                    structural_zero_residue=0.0,
                ),
            ),
        ),
    )

    comparisons = tuple(
        comparison
        for stage in (record["before_clear"], record["after_clear"])
        for comparison in stage["comparisons"]
    )
    assert all(
        comparison["resolved_components"]["maximum_absolute_delta"]
        == pytest.approx(1.0e-30)
        for comparison in comparisons
    )
    assert all(
        comparison["resolved_components"]["maximum_conditioned_residual"]
        < report_runner.RELATIVE_TOLERANCE
        for comparison in comparisons
    )


def test_inconsistent_optimized_and_resolved_sum_fails_before_clear() -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    recurrence_cell, compiled_cell = _authorities(cell)
    candidate = _Runtime(
        ExecutionMode.ON_THE_FLY,
        "a",
        inconsistent_total=True,
    )

    with pytest.raises(RunnerError, match="optimized/resolved sum disagrees"):
        on_the_fly_dual_authority_validation(
            candidate,
            POINTS,
            cell=cell,
            selector_contract=_contract(),
            authorities=(
                (
                    "recurrence",
                    recurrence_cell,
                    _Runtime(ExecutionMode.RECURRENCE, "b"),
                ),
                (
                    "compiled",
                    compiled_cell,
                    _Runtime(ExecutionMode.COMPILED, "c"),
                ),
            ),
        )

    assert ("resolved-total", None, (FLOW_IDS[0],)) in candidate.events
    assert ("clear",) not in candidate.events


def test_measurement_releases_authority_runtimes_before_public_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProfileReached(Exception):
        pass

    cell = _otf_cell(Workload.SELECTED_FLOW)
    recurrence_cell, compiled_cell = _authorities(cell)
    candidate = _Runtime(ExecutionMode.ON_THE_FLY, "a")
    contract = _contract()
    generated = GeneratedArtifact(
        path=tmp_path / "candidate",
        process_id="process",
        generation_seconds=1.0,
        model_preparation_seconds=0.0,
        model_preparation_reused=True,
        requested_config={},
        effective_config={},
    )
    authority_refs: list[ReferenceType[_Runtime]] = []

    def load_authorities(*_args: object, **_kwargs: object):
        recurrence = _Runtime(ExecutionMode.RECURRENCE, "b")
        compiled = _Runtime(ExecutionMode.COMPILED, "c")
        authority_refs.extend((ref(recurrence), ref(compiled)))
        return (
            ("recurrence", recurrence_cell, recurrence),
            ("compiled", compiled_cell, compiled),
        )

    def profile(*_args: object, **_kwargs: object):
        gc.collect()
        assert authority_refs and all(item() is None for item in authority_refs)
        raise _ProfileReached

    monkeypatch.setattr(
        report_measurement, "generate_artifact", lambda *_a, **_k: generated
    )
    monkeypatch.setattr(
        report_measurement, "validate_artifact_contract", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        report_measurement, "_load_runtime", lambda *_a, **_k: candidate
    )
    monkeypatch.setattr(
        report_measurement,
        "runtime_identity_payload",
        lambda *_a, **_k: {"loaded_module_origin_policy": {"observations": []}},
    )
    monkeypatch.setattr(
        report_measurement, "shared_validation_points", lambda _process: POINTS
    )
    monkeypatch.setattr(
        report_measurement,
        "_load_on_the_fly_authority_runtimes",
        load_authorities,
    )
    monkeypatch.setattr(
        report_measurement,
        "_resolution_benchmark_config",
        lambda _config: object(),
    )
    monkeypatch.setattr(report_measurement, "profile_runtime", profile)
    baseline = {
        "status": "ok",
        "matrix_element": 1.0,
        "selector_contract": contract.as_dict(),
    }

    with pytest.raises(_ProfileReached):
        report_measurement.measure_pyamplicol_cell(
            cell,
            artifact_path=tmp_path / "artifact",
            settings=RunnerSettings(source_revision_override="d" * 40),
            repo_root=tmp_path,
            baseline=baseline,
        )


def test_authorities_are_loaded_from_measurements_without_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    recurrence_cell, compiled_cell = _authorities(cell)
    recurrence_path = tmp_path / "recurrence"
    compiled_path = tmp_path / "compiled"
    recurrence_path.mkdir()
    compiled_path.mkdir()

    def measurement(path: Path, process_id: str) -> dict[str, object]:
        return {
            "status": "ok",
            "generation_seconds": 1.0,
            "artifact": {"path": str(path), "process_id": process_id},
            "provenance": {
                "requested_config": {},
                "effective_config": {},
                "model_preparation_seconds": 0.0,
                "model_preparation_reused": True,
            },
        }

    baseline = measurement(recurrence_path, "recurrence-process")
    compiled_measurement = measurement(compiled_path, "compiled-process")
    loads: list[tuple[Path, str]] = []
    validations: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        report_measurement,
        "validate_artifact_contract",
        lambda authority_cell, path: validations.append((authority_cell.cell_id, path)),
    )
    monkeypatch.setattr(
        report_measurement,
        "_load_runtime",
        lambda path, process_id: loads.append((path, process_id)) or object(),
    )
    monkeypatch.setattr(
        report_measurement,
        "generate_artifact",
        lambda *_a, **_k: pytest.fail("authority generation must not run"),
    )

    loaded = report_measurement._load_on_the_fly_authority_runtimes(
        cell,
        baseline=baseline,
        validation_peers={compiled_cell.cell_id: compiled_measurement},
        catalog=REPORT_CATALOG,
    )

    assert [(role, authority_cell.cell_id) for role, authority_cell, _ in loaded] == [
        ("recurrence", recurrence_cell.cell_id),
        ("compiled", compiled_cell.cell_id),
    ]
    assert loads == [
        (recurrence_path.resolve(), "recurrence-process"),
        (compiled_path.resolve(), "compiled-process"),
    ]
    assert validations == [
        (recurrence_cell.cell_id, recurrence_path.resolve()),
        (compiled_cell.cell_id, compiled_path.resolve()),
    ]


def test_otf_cache_requires_and_authenticates_compact_gate_record() -> None:
    cell, contract, record, *_runtimes = _valid_gate(Workload.SELECTED_FLOW)
    measurement = {
        "selector_contract": contract.as_dict(),
        "provenance": {
            "runtime_identity": {"artifact_id": record["candidate_artifact_id"]}
        },
    }
    validation = {OTF_DUAL_AUTHORITY_VALIDATION_FIELD: record}
    report_cache._validate_successful_otf_dual_authority(
        measurement,
        validation,
        expected_cell=cell,
        catalog=REPORT_CATALOG,
    )

    with pytest.raises(ValueError, match="must be an object"):
        report_cache._validate_successful_otf_dual_authority(
            measurement,
            {},
            expected_cell=cell,
            catalog=REPORT_CATALOG,
        )

    for mutation in ("digest", "count", "resolved_sum", "lifecycle", "authority"):
        corrupted = deepcopy(record)
        if mutation == "digest":
            corrupted["before_clear"]["candidate_resolved_source_sha256"] = "bad"
        elif mutation == "count":
            corrupted["resolved_check_count"] += 1
        elif mutation == "resolved_sum":
            corrupted["before_clear"]["candidate_resolved_sum"][
                "maximum_conditioned_residual"
            ] = 1.0e-6
        elif mutation == "lifecycle":
            corrupted["lifecycle"]["clear_call_count"] = 1
        else:
            corrupted["authorities"][1]["cell_id"] = "wrong-compiled-cell"
        with pytest.raises(ValueError):
            report_cache._validate_successful_otf_dual_authority(
                measurement,
                {OTF_DUAL_AUTHORITY_VALIDATION_FIELD: corrupted},
                expected_cell=cell,
                catalog=REPORT_CATALOG,
            )


def _warmup_evidence() -> dict[str, object]:
    return {
        "cold_warmup_elapsed_seconds": 4.767252833,
        "cold_warmup_run_count": 1,
        "cold_warmup_batch_size": 128,
        "cold_warmup_point_count": 128,
        "cold_warmup_timer_source": report_runner.WARMUP_TIMER_SOURCE,
        "cold_warmup_timing_scope": report_runner.OTF_COLD_WARMUP_TIMING_SCOPE,
        "cold_warmup_runtime_freshness": (
            report_runner.OTF_COLD_WARMUP_RUNTIME_FRESHNESS
        ),
        "cold_warmup_ratio_eligible": False,
        "cold_warmup_acceptance_eligible": False,
        "warmup_elapsed_seconds": 0.021,
        "warmup_configured_run_count": 2,
        "warmup_batch_size": 128,
        "warmup_point_count": 256,
        "warmup_run_outer_wall_seconds": [0.010, 0.011],
        "first_warmup_run_outer_wall_seconds": 0.010,
        "warmup_timer_source": report_runner.WARMUP_TIMER_SOURCE,
        "warmup_timing_scope": report_runner.CONVENTIONAL_WARMUP_TIMING_SCOPE,
    }


def test_otf_cache_requires_complete_cold_and_conventional_warmups() -> None:
    cell = _otf_cell(Workload.SELECTED_FLOW)
    measurement = {
        "provenance": {
            "runtime_profile": _warmup_evidence(),
            "runtime_load_included_in_cold_warmup": False,
            "generation_timer_excludes_model_preparation": True,
            "effective_config": {"benchmark": {"batch_size": 128, "warmup_runs": 2}},
        }
    }

    report_cache._validate_successful_otf_warmup_evidence(
        measurement,
        expected_cell=cell,
    )

    missing = deepcopy(measurement)
    del missing["provenance"]["runtime_profile"]["warmup_timer_source"]
    with pytest.raises(ValueError, match="incomplete conventional warm-up"):
        report_cache._validate_successful_otf_warmup_evidence(
            missing,
            expected_cell=cell,
        )

    eligible = deepcopy(measurement)
    eligible["provenance"]["runtime_profile"]["cold_warmup_ratio_eligible"] = True
    with pytest.raises(ValueError, match="ineligible for ratios and acceptance"):
        report_cache._validate_successful_otf_warmup_evidence(
            eligible,
            expected_cell=cell,
        )

    for field, invalid in (
        ("runtime_load_included_in_cold_warmup", True),
        ("generation_timer_excludes_model_preparation", False),
    ):
        contradictory = deepcopy(measurement)
        contradictory["provenance"][field] = invalid
        with pytest.raises(ValueError, match="exclude"):
            report_cache._validate_successful_otf_warmup_evidence(
                contradictory,
                expected_cell=cell,
            )

    coordinated_batch = deepcopy(measurement)
    batch_profile = coordinated_batch["provenance"]["runtime_profile"]
    batch_profile["cold_warmup_batch_size"] = 64
    batch_profile["cold_warmup_point_count"] = 64
    batch_profile["warmup_batch_size"] = 64
    batch_profile["warmup_point_count"] = 128
    with pytest.raises(ValueError, match="effective benchmark configuration"):
        report_cache._validate_successful_otf_warmup_evidence(
            coordinated_batch,
            expected_cell=cell,
        )

    coordinated_runs = deepcopy(measurement)
    run_profile = coordinated_runs["provenance"]["runtime_profile"]
    run_profile["warmup_configured_run_count"] = 1
    run_profile["warmup_point_count"] = 128
    run_profile["warmup_run_outer_wall_seconds"] = [0.010]
    run_profile["first_warmup_run_outer_wall_seconds"] = 0.010
    run_profile["warmup_elapsed_seconds"] = 0.010
    with pytest.raises(ValueError, match="effective benchmark configuration"):
        report_cache._validate_successful_otf_warmup_evidence(
            coordinated_runs,
            expected_cell=cell,
        )


def test_report_cache_schema_carries_bounded_warmup_field_contracts() -> None:
    schema = report_cache.schema_document()
    measurement = schema["properties"]["entries"]["items"]["properties"]["measurement"]
    provenance = measurement["properties"]["provenance"]["oneOf"][1]
    runtime_profile = provenance["properties"]["runtime_profile"]

    assert runtime_profile["properties"]["cold_warmup_run_count"] == {"const": 1}
    assert runtime_profile["properties"]["cold_warmup_ratio_eligible"] == {
        "const": False
    }
    assert runtime_profile["properties"]["warmup_timer_source"] == {
        "const": report_runner.WARMUP_TIMER_SOURCE
    }
    assert provenance["properties"]["runtime_load_included_in_cold_warmup"] == {
        "const": False
    }
    assert provenance["properties"]["generation_timer_excludes_model_preparation"] == {
        "const": True
    }
    assert set(runtime_profile["dependentRequired"]["cold_warmup_elapsed_seconds"]) == (
        report_runner.OTF_COLD_WARMUP_FIELDS | report_runner.CONVENTIONAL_WARMUP_FIELDS
    )
    assert "warmup_elapsed_seconds" not in runtime_profile["dependentRequired"]

    validator = Draft202012Validator(runtime_profile)
    validator.validate({"warmup_elapsed_seconds": 0.2})
    with pytest.raises(ValidationError):
        validator.validate({"cold_warmup_elapsed_seconds": 0.2})

    compiled = _authorities(_otf_cell(Workload.SELECTED_FLOW))[1]
    historical = report_cache.build_reset_cache(compiled.dataset_id, (compiled,))
    historical_measurement = historical["entries"][0]["measurement"]
    historical_measurement["status"] = "ok"
    historical_measurement["validation"] = {DIRECT_AGREEMENT_FIELD: []}
    historical_measurement["provenance"] = {
        "runtime_profile": {"warmup_elapsed_seconds": 0.2}
    }
    Draft202012Validator(schema).validate(historical)


@pytest.mark.parametrize("count_field", ("check_count", "component_count"))
def test_otf_compact_summary_rejects_boolean_counts(count_field: str) -> None:
    summary = {
        "check_count": 1,
        "component_count": 1,
        "maximum_conditioned_residual": 0.0,
        "maximum_absolute_delta": 0.0,
    }
    summary[count_field] = True

    with pytest.raises(ValueError, match="count"):
        report_runner._validate_otf_compact_summary(
            summary,
            field="test-summary",
            expected_checks=1,
            expected_components=1,
            relative_tolerance=1.0e-12,
        )


def test_otf_config_omits_api_bundle_and_materialized_generation_work(
    tmp_path: Path,
) -> None:
    values = config_values(
        _otf_cell(Workload.ALL_FLOW),
        RunnerSettings(),
        repo_root=tmp_path,
    )

    assert values["generation"]["emit_api_bundle"] is False
    assert values["generation"]["relation_discovery"] == {"mode": "off"}
    assert values["color"]["lc_flow_layout"] == "topology-replay"


def test_otf_runtime_identity_uses_the_real_prepared_jit_source_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol._internal.versions as versions
    import pyamplicol.artifacts

    cell = _otf_cell(Workload.SELECTED_FLOW)
    artifact_id = "a" * 64
    revision = "d" * 40
    native_path = tmp_path / "_rusticol.so"
    native_path.write_bytes(b"native")
    manifest = SimpleNamespace(
        artifact_id=artifact_id,
        processes=(
            {
                "id": "process",
                "required_runtime_capabilities": [
                    "rusticol.on-the-fly.complex-f64.v1",
                    "rusticol.on-the-fly.lc-color.v1",
                ],
            },
        ),
        runtime={"engine_version": "test"},
    )
    native = SimpleNamespace(
        __file__=str(native_path),
        native_build_inputs_sha256=lambda: "e" * 64,
        target_info=lambda: SimpleNamespace(triple="test", cpu_features=()),
        package_version=lambda: "test",
    )
    package_tree = {"kind": "tree"}
    native_identity = {"kind": "native"}
    monkeypatch.setattr(
        pyamplicol.artifacts, "load_manifest", lambda *_a, **_k: manifest
    )
    monkeypatch.setattr(versions, "verify_native_module", lambda _native: None)
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "schema_version": 1,
            "version": "test",
            "source_revision": revision,
            "publishable": True,
            "selftest_fixture_bootstrap": False,
        },
    )
    monkeypatch.setattr(report_runner.importlib, "import_module", lambda _name: native)
    monkeypatch.setattr(
        report_runner,
        "established_preimport_runtime_identity",
        lambda: {
            "kind": "pyamplicol-preimport-runtime-identity-v1",
            "python_package_tree": package_tree,
            "native_extension": native_identity,
        },
    )
    monkeypatch.setattr(
        report_runner,
        "python_package_tree_identity",
        lambda _roots: package_tree,
    )
    monkeypatch.setattr(
        report_runner,
        "loaded_pyamplicol_origin_policy",
        lambda *_a, **_k: {"kind": "loaded-origin"},
    )

    identity = runtime_identity_payload(
        cell,
        SimpleNamespace(artifact_id=artifact_id, execution_mode="on-the-fly"),
        tmp_path,
        "process",
        expected_source_revision=revision,
    )

    assert identity["expected_evaluator_abi"] == (
        "pyamplicol-runtime-on-the-fly-execution"
    )
    assert identity["expected_source_evaluator_abi"] == (
        "pyamplicol-symjit-plane-application-v2"
    )
    assert identity["expected_source_evaluator_runtime_capability"] == (
        "symjit.application.complex-f64.v1"
    )
