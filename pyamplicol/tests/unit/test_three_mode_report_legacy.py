# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.cache import validate_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.legacy import (
    CommandResult,
    LegacyAdapterError,
    LegacyMeasurementAdapter,
    LegacySettings,
    TimingRow,
    _canonical_mapped_color_word,
    adaptive_profile_points,
)
from tools.performance_report.models import Accuracy, ExecutionMode, Workload
from tools.performance_report.runner import SelectorContract


@dataclass(frozen=True)
class FakeEntry:
    group: int = 1
    integral: int = 2
    process_pdgs: tuple[int, ...] = (1, -1, 23, 21)
    color_order: tuple[int, ...] = (2, 4, 1, 3)


class FakeApi:
    default_repository = Path("/fake/legacy")

    def __init__(self, pdgs: tuple[int, ...] = (1, -1, 23, 21)) -> None:
        self.pdgs = pdgs
        self.entry = FakeEntry(process_pdgs=pdgs)
        self.selected_calls: list[int] = []
        self.color_calls: list[tuple[str, tuple[int, ...] | None]] = []

    def expected_revision(self) -> str:
        return "a" * 40

    def validate_checkout(self, repository: Path) -> None:
        pass

    def compiler_provenance(self, repository: Path) -> object:
        return {"identity": "gfortran", "version": "test"}

    def process_pdgs(self, process: str) -> tuple[int, ...]:
        return self.pdgs

    def parse_process_file(self, path: Path) -> tuple[object, ...]:
        return (self.entry,)

    def select_generated_process_entry(
        self,
        entries: tuple[object, ...],
        *,
        generated_process: str,
        wanted_pdgs: tuple[int, ...],
    ) -> tuple[object, tuple[object, ...]]:
        assert wanted_pdgs == self.pdgs
        return entries[0], entries

    def source_mapped_color_order(
        self,
        entry: object,
        *,
        source_pdgs: tuple[int, ...],
    ) -> tuple[int, ...]:
        return self.entry.color_order

    def ordered_momenta(
        self,
        source_pdgs: tuple[int, ...],
        target_pdgs: tuple[int, ...],
        momenta: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return momenta

    def source_to_row_permutation(
        self,
        source_pdgs: tuple[int, ...],
        target_pdgs: tuple[int, ...],
    ) -> tuple[int, ...]:
        return tuple(range(len(source_pdgs)))

    def parse_probe_output(self, output: str) -> object:
        return SimpleNamespace(value=12.5)

    def run_selected_flow_probe(self, repository: Path, **kwargs: object) -> object:
        self.selected_calls.append(int(kwargs["points"]))
        return SimpleNamespace(value=3.25)

    def run_color_probe(self, repository: Path, **kwargs: object) -> object:
        accuracy = str(kwargs["color_accuracy"])
        helicities = kwargs["helicities"]
        self.color_calls.append(
            (accuracy, None if helicities is None else tuple(helicities))
        )
        return SimpleNamespace(value=9.75)


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.profile_calls = 0

    def run(
        self,
        args: object,
        *,
        cwd: Path,
        environment: object = None,
    ) -> CommandResult:
        rendered = tuple(str(item) for item in args)
        self.commands.append(rendered)
        if any(item.endswith("process_list.py") for item in rendered):
            cwd.mkdir(parents=True, exist_ok=True)
            (cwd / "processes.txt").write_text("fake\n", encoding="utf-8")
        points = 1
        is_profile = bool(
            rendered
            and (
                rendered[0]
                in {
                    "./amplicol_library_benchmark",
                    "./amplicol_color_library_probe",
                }
                or rendered[0].endswith("amplicol_color_probe")
            )
        )
        if is_profile:
            points = int(rendered[1])
            rate_factors = (
                1.0,
                0.99999,
                1.00001,
                1.0,
                1.000005,
                0.999995,
            )
            factor = rate_factors[self.profile_calls % len(rate_factors)]
            self.profile_calls += 1
        else:
            factor = 1.0
        evaluation = points * 0.001 * factor
        output = (
            "Timing summary\n"
            "generation setup 2.5\n"
            f"amplitude evaluation {evaluation}\n"
            f"total {evaluation * 1.1}\n"
        )
        elapsed = 0.25 if rendered and rendered[0] != "make" else 0.1
        return CommandResult(
            args=rendered,
            cwd=cwd,
            elapsed_seconds=elapsed,
            returncode=0,
            stdout=output,
            stderr="",
            environment={} if environment is None else dict(environment),
        )


class FakeSnapshotter:
    def snapshot(
        self,
        repository: Path,
        destination: Path,
        *,
        executables: tuple[str, ...],
        process_file: Path,
    ) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        return destination


def _cell(
    accuracy: Accuracy,
    workload: Workload,
    *,
    process_key: str = "dd_z_jets",
):
    return next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.measurement.accuracy is accuracy
        and cell.workload is workload
        and cell.process_key == process_key
        and cell.n_final == (1 if process_key == "dd_z_jets" else cell.n_final)
    )


def _adapter(api: FakeApi | None = None):
    fake_api = FakeApi() if api is None else api
    executor = FakeExecutor()
    return (
        LegacyMeasurementAdapter(
            api=fake_api,
            executor=executor,
            snapshotter=FakeSnapshotter(),
        ),
        fake_api,
        executor,
    )


def _settings(repository: Path) -> LegacySettings:
    return LegacySettings(
        target_runtime_seconds=1.0,
        warmup_points=100,
        minimum_points=100,
        maximum_points=10_000,
        repository=repository,
    )


def test_adaptive_profile_points_are_bounded() -> None:
    assert adaptive_profile_points(
        0.0, target_runtime_seconds=20.0, minimum_points=17
    ) == 17
    assert adaptive_profile_points(
        10.0,
        target_runtime_seconds=20.0,
        warmup_points=100,
        minimum_points=100,
        maximum_points=1_000,
    ) == 200
    assert adaptive_profile_points(
        0.001,
        target_runtime_seconds=20.0,
        warmup_points=100,
        maximum_points=1_000,
    ) == 1_000


@pytest.mark.parametrize(
    ("source_pdgs", "raw_word", "canonical_word"),
    (
        (
            (1, -1, 6, -6),
            (3, 1, 2, 4),
            (2, 4, 3, 1),
        ),
        (
            (1, -1, 6, -6, 21),
            (3, 1, 2, 5, 4),
            (2, 5, 4, 3, 1),
        ),
        (
            (1, -1, 6, -6, 21, 21),
            (3, 1, 2, 5, 6, 4),
            (2, 5, 6, 4, 3, 1),
        ),
        (
            (21, 21, 6, -6),
            (3, 1, 2, 4),
            (3, 1, 2, 4),
        ),
    ),
)
def test_legacy_open_string_blocks_match_canonical_public_selector_axis(
    source_pdgs: tuple[int, ...],
    raw_word: tuple[int, ...],
    canonical_word: tuple[int, ...],
) -> None:
    word = _canonical_mapped_color_word(
        source_pdgs,
        raw_word,
        initial_state_count=2,
    )

    assert word == canonical_word
    helicities = tuple(-1 if label % 2 else 1 for label in range(1, len(word) + 1))
    helicity_id = "h:" + ",".join(f"{value:+d}" for value in helicities)
    source_helicities = tuple(enumerate(helicities, start=1))
    legacy_contract = SelectorContract(
        selected_color_flow_ids=("flow:" + ",".join(str(label) for label in word),),
        selected_color_words=(word,),
        all_flow_helicity_ids=(helicity_id,),
        all_flow_source_helicities=source_helicities,
        point_digest="a" * 64,
    )
    candidate_contract = SelectorContract(
        selected_color_flow_ids=(
            "flow:" + ",".join(str(label) for label in canonical_word),
        ),
        selected_color_words=(canonical_word,),
        all_flow_helicity_ids=(helicity_id,),
        all_flow_source_helicities=source_helicities,
        point_digest="a" * 64,
    )
    assert legacy_contract == candidate_contract


def test_legacy_closed_adjoint_word_keeps_generated_row_order() -> None:
    assert _canonical_mapped_color_word(
        (21, 21, 21, 21),
        (1, 3, 2, 4),
        initial_state_count=2,
    ) == (1, 3, 2, 4)


@pytest.mark.parametrize(
    "raw_word",
    (
        (3, 2, 1, 4),
        (1, 3, 2, 4),
        (3, 1, 4, 2),
        (3, 1, 2, 2),
    ),
)
def test_legacy_selector_canonicalization_rejects_non_block_rows(
    raw_word: tuple[int, ...],
) -> None:
    with pytest.raises(
        LegacyAdapterError,
        match=r"permutation|concatenation",
    ):
        _canonical_mapped_color_word(
            (1, -1, 6, -6),
            raw_word,
            initial_state_count=2,
        )


def test_profile_rejects_exactly_identical_bounded_chunk_rates(
    tmp_path: Path,
) -> None:
    adapter, _api, _executor = _adapter()
    calls: list[int] = []

    def invoke(points: int):
        calls.append(points)
        seconds = points * 1.0e-5
        result = CommandResult(
            args=("probe", str(points)),
            cwd=tmp_path,
            elapsed_seconds=seconds,
            returncode=0,
            stdout="",
            stderr="",
            environment={},
        )
        return (
            result,
            (TimingRow("amplitude evaluation", seconds),),
            None,
        )

    with pytest.raises(
        LegacyAdapterError,
        match="positive measured uncertainty",
    ):
        adapter._profile(
            invoke,
            settings=LegacySettings(
                target_runtime_seconds=0.005,
                warmup_points=10,
                minimum_points=10,
                maximum_points=100,
                maximum_profile_chunks=8,
                repository=tmp_path,
            ),
            timing_labels=("amplitude evaluation",),
        )

    assert calls == [10, 100, 100, 100, 100, 100, 10, 10, 10]


def test_profile_rejects_a_bound_below_five_timed_chunks(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="maximum_profile_chunks must not be below",
    ):
        LegacySettings(
            target_runtime_seconds=0.005,
            warmup_points=10,
            minimum_points=10,
            maximum_points=100,
            maximum_profile_chunks=2,
            repository=tmp_path,
        )


def test_selected_flow_uses_generated_mode_one_and_compact_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, executor = _adapter()
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    measurement = adapter.measure(
        _cell(Accuracy.LC, Workload.SELECTED_FLOW),
        artifact_path=tmp_path / "selected",
        settings=_settings(tmp_path / "repository"),
    )

    validate_measurement(measurement)
    assert measurement["status"] == "ok"
    assert measurement["matrix_element"] == 3.25
    assert measurement["generation_seconds"] == pytest.approx(0.55)
    assert measurement["wall_seconds_per_point"] == pytest.approx(1.0e-3)
    contract = SelectorContract.from_mapping(measurement["selector_contract"])
    assert contract.selected_color_flow_ids == ("flow:2,4,1",)
    assert contract.all_flow_helicity_ids == ("h:-1,+1,-1,+1",)
    assert api.selected_calls == [1]
    flattened = [" ".join(command) for command in executor.commands]
    assert any("--library=create" in command for command in flattened)
    assert any("--amplicol_momenta_probe=10" in command for command in flattened)
    assert not any("--library=create-raw" in command for command in flattened)
    momenta_file = (
        tmp_path / "repository/Utilities/ME_checks/momenta_1_2.txt"
    )
    assert momenta_file.is_file()
    assert len(momenta_file.read_text(encoding="utf-8").splitlines()) == 4


def test_all_flow_uses_direct_fixed_helicity_and_its_own_generation_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, executor = _adapter()
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    measurement = adapter.measure(
        _cell(Accuracy.LC, Workload.ALL_FLOW),
        artifact_path=tmp_path / "all",
        settings=_settings(tmp_path / "repository"),
    )

    validate_measurement(measurement)
    assert measurement["matrix_element"] == 12.5
    assert measurement["generation_seconds"] == 2.5
    assert measurement["wall_seconds_per_point"] == pytest.approx(1.0e-3)
    provenance = measurement["provenance"]
    assert provenance["generation_timing_is_workload_specific"] is True
    assert provenance["raw_mapped_color_order"] == [2, 4, 1, 3]
    assert (
        provenance["selector_color_word_policy"]
        == "outgoing-open-string-blocks-by-fundamental-source-label-v1"
    )
    assert (
        provenance["generation_source"] == "direct-imode2-generation-setup"
    )
    flattened = [" ".join(command) for command in executor.commands]
    assert not any("--library=create" in command for command in flattened)
    assert any(command.endswith("amplicol_color_probe") for command in flattened)


@pytest.mark.parametrize("accuracy", [Accuracy.NLC, Accuracy.FULL])
def test_contracted_uses_raw_library_and_direct_oracle_value(
    accuracy: Accuracy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, executor = _adapter()
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    measurement = adapter.measure(
        _cell(accuracy, Workload.CONTRACTED),
        artifact_path=tmp_path / accuracy.value,
        settings=_settings(tmp_path / "repository"),
    )

    validate_measurement(measurement)
    assert measurement["matrix_element"] == 9.75
    assert measurement["selector_contract"] is None
    assert api.color_calls == [(accuracy.value, None)]
    flattened = [" ".join(command) for command in executor.commands]
    assert any("--library=create-raw" in command for command in flattened)


def test_more_than_three_open_quark_lines_is_preserved_as_unsupported(
    tmp_path: Path,
) -> None:
    api = FakeApi((1, -1, 2, -2, 3, -3, 4, -4))
    adapter, _api, executor = _adapter(api)
    cell = next(
        item
        for item in REPORT_CATALOG.reference_cells()
        if item.process_key == "dd_4q_lines"
        and item.measurement.accuracy is Accuracy.LC
        and item.workload is Workload.SELECTED_FLOW
    )
    measurement = adapter.measure(
        cell,
        artifact_path=tmp_path / "unsupported",
        settings=_settings(tmp_path / "repository"),
    )

    validate_measurement(measurement)
    assert measurement["status"] == "unsupported"
    assert "at most 3 open quark lines" in measurement["failure"]["message"]
    assert executor.commands == []


def test_profile_never_reuses_warmup_as_the_only_timed_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, executor = _adapter()
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    settings = LegacySettings(
        target_runtime_seconds=0.05,
        warmup_points=100,
        minimum_points=10,
        maximum_points=1_000,
        repository=tmp_path / "repository",
    )
    measurement = adapter.measure(
        _cell(Accuracy.LC, Workload.SELECTED_FLOW),
        artifact_path=tmp_path / "reuse",
        settings=settings,
    )

    benchmark_commands = [
        command
        for command in executor.commands
        if command and command[0] == "./amplicol_library_benchmark"
    ]
    assert len(benchmark_commands) == 6
    assert measurement["sample_count"] >= 50
    assert (
        measurement["provenance"]["runtime_profile"]["measurement"][
            "profile_phase"
        ]
        == "measurement_chunks"
    )
    assert (
        measurement["provenance"]["runtime_profile"]["measurement"][
            "chunk_count"
        ]
        == 5
    )


def test_fast_warmup_still_produces_five_samples_and_measured_rse(
    tmp_path: Path,
) -> None:
    adapter, _api, _executor = _adapter()
    calls: list[int] = []
    rates = iter((0.09, 0.11, 0.10, 0.105, 0.095))

    def invoke(points: int):
        calls.append(points)
        seconds = 5.0 if len(calls) == 1 else points * next(rates)
        result = CommandResult(
            args=("probe", str(points)),
            cwd=tmp_path,
            elapsed_seconds=seconds,
            returncode=0,
            stdout="",
            stderr="",
            environment={},
        )
        return (
            result,
            (TimingRow("amplitude evaluation", seconds),),
            None,
        )

    profile = adapter._profile(
        invoke,
        settings=LegacySettings(
            target_runtime_seconds=5.0,
            warmup_points=10,
            minimum_points=10,
            maximum_points=10,
            repository=tmp_path,
        ),
        timing_labels=("amplitude evaluation",),
    )

    assert len(calls) == 6
    assert profile.record["chunk_count"] == 5
    assert profile.seconds == pytest.approx(5.0)
    assert profile.standard_error_seconds_per_point > 0.0
    assert profile.relative_standard_error > 0.0
