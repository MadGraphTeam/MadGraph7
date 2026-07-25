# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ExecutionMode,
    Workload,
)
from tools.performance_report.runner import (
    RunnerError,
    RunnerSettings,
    SelectorContract,
    config_values,
    derive_selector_contract,
    point_digest,
    pointwise_validation,
    resolved_sum_validation,
    validate_artifact_contract,
    validate_runtime_contract,
    validate_selector_contract,
)


@dataclass(frozen=True)
class Flow:
    id: str
    word: tuple[int, ...]


@dataclass(frozen=True)
class Helicity:
    id: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class Particle:
    label: int


class Resolved:
    def __init__(self, values: object, totals: tuple[complex, ...]) -> None:
        self.values = values
        self._totals = totals

    def total(self) -> tuple[complex, ...]:
        return self._totals


class FakeRuntime:
    def __init__(self) -> None:
        self.physics = SimpleNamespace(
            color_accuracy="lc",
            selector_capabilities=("helicity", "color_flow"),
            external_particles=(Particle(1), Particle(2), Particle(3)),
            helicities=(
                Helicity("h:-1,-1,-1", (-1, -1, -1)),
                Helicity("h:-1,+1,-1", (-1, 1, -1)),
            ),
            color_flows=(
                Flow("flow:2,1,3", (2, 1, 3)),
                Flow("flow:1,2,3", (1, 2, 3)),
            ),
        )
        self.optimized = (3.0 + 0.0j,)
        self.resolved_total = (3.0 + 0.0j,)

    def evaluate(self, _points: object, **_selectors: object) -> tuple[complex, ...]:
        return self.optimized

    def evaluate_resolved(
        self,
        _points: object,
        **_selectors: object,
    ) -> Resolved:
        return Resolved(
            (
                (
                    (0.0 + 0.0j,),
                    (3.0 + 0.0j,),
                ),
            ),
            self.resolved_total,
        )


def _cell(
    mode: ExecutionMode,
    accuracy: Accuracy,
    workload: Workload,
):
    return next(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode is mode
        and cell.measurement.accuracy is accuracy
        and cell.workload is workload
    )


@pytest.mark.parametrize(
    ("mode", "workload", "expected_layout", "expected_level"),
    (
        (
            ExecutionMode.RECURRENCE,
            Workload.SELECTED_FLOW,
            "topology-replay",
            2,
        ),
        (
            ExecutionMode.RECURRENCE,
            Workload.ALL_FLOW,
            "all-flow-union",
            2,
        ),
        (
            ExecutionMode.COMPILED,
            Workload.SELECTED_FLOW,
            "topology-replay",
            3,
        ),
        (
            ExecutionMode.EAGER,
            Workload.ALL_FLOW,
            "all-flow-union",
            2,
        ),
    ),
)
def test_config_steers_complete_coverage_and_layout_only(
    mode: ExecutionMode,
    workload: Workload,
    expected_layout: str,
    expected_level: int,
) -> None:
    cell = _cell(mode, Accuracy.LC, workload)
    values = config_values(
        cell,
        RunnerSettings(worker_cores=1),
        repo_root=Path("/repo"),
    )

    assert values["color"]["lc_flow_layout"] == expected_layout  # type: ignore[index]
    assert values["evaluator"]["execution_mode"] == mode.value  # type: ignore[index]
    assert (
        values["evaluator"]["jit"]["optimization_level"]  # type: ignore[index]
        == expected_level
    )
    serialized = repr(values)
    assert "selected_color_sector_ids" not in serialized
    assert "selected_source_helicities" not in serialized
    assert "reference_color_order" not in serialized


def test_nlc_and_full_use_contracted_topology_replay_configuration() -> None:
    for accuracy in (Accuracy.NLC, Accuracy.FULL):
        cell = _cell(ExecutionMode.RECURRENCE, accuracy, Workload.CONTRACTED)
        values = config_values(
            cell,
            RunnerSettings(),
            repo_root=Path("/repo"),
        )
        assert values["color"] == {
            "accuracy": accuracy.value,
            "lc_flow_layout": "topology-replay",
        }


def test_selector_contract_uses_first_flow_and_first_nonzero_helicity() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)

    contract = derive_selector_contract(runtime, points)

    assert contract.selected_color_flow_ids == ("flow:2,1,3",)
    assert contract.selected_color_words == ((2, 1, 3),)
    assert contract.all_flow_helicity_ids == ("h:-1,+1,-1",)
    assert contract.all_flow_source_helicities == ((1, -1), (2, 1), (3, -1))
    assert contract.point_digest == point_digest(points)
    assert SelectorContract.from_mapping(contract.as_dict()) == contract
    validate_selector_contract(runtime, contract, points)


def test_selector_contract_rejects_changed_point_or_axis() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = derive_selector_contract(runtime, points)

    with pytest.raises(RunnerError, match="measurement point differ"):
        validate_selector_contract(
            runtime,
            contract,
            (((2.0, 0.0, 0.0, 2.0),),),
        )

    runtime.physics.color_flows = (Flow("different", (2, 1, 3)),)
    with pytest.raises(RunnerError, match="selected physical flow"):
        validate_selector_contract(runtime, contract, points)


def test_runtime_contract_requires_both_lc_selector_axes() -> None:
    runtime = FakeRuntime()
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )
    validate_runtime_contract(cell, runtime)

    runtime.physics.selector_capabilities = ("helicity",)
    with pytest.raises(RunnerError, match="color_flow"):
        validate_runtime_contract(cell, runtime)


def test_artifact_contract_rejects_generation_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )
    process = SimpleNamespace(
        execution_mode="recurrence",
        generation_specialized_axes=(),
        selected_source_helicities=(),
        selected_color_sector_ids=(),
        lc_flow_layout="topology-replay",
    )
    inspection = SimpleNamespace(processes=(process,))
    monkeypatch.setattr(
        "pyamplicol.artifacts.inspect_artifact",
        lambda _path: inspection,
    )
    validate_artifact_contract(cell, Path("/artifact"))

    process.generation_specialized_axes = ("color_flow",)
    with pytest.raises(RunnerError, match="complete runtime coverage"):
        validate_artifact_contract(cell, Path("/artifact"))


def test_resolved_sum_validation_and_pointwise_tolerances() -> None:
    runtime = FakeRuntime()
    points = (((1.0, 0.0, 0.0, 1.0),),)
    contract = derive_selector_contract(runtime, points)
    cell = _cell(
        ExecutionMode.RECURRENCE,
        Accuracy.LC,
        Workload.SELECTED_FLOW,
    )

    assert (
        resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
        )["status"]
        == "ok"
    )
    runtime.resolved_total = (2.0 + 0.0j,)
    assert (
        resolved_sum_validation(
            runtime,
            points,
            cell=cell,
            selector_contract=contract,
        )["status"]
        == "validation_failed"
    )

    assert pointwise_validation(1.0 + 1.0e-13, 1.0)["status"] == "ok"
    assert pointwise_validation(2.0, 1.0)["status"] == "validation_failed"
