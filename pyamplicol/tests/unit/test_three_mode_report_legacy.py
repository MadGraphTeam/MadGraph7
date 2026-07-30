# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from contextlib import contextmanager
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
    _fixed_helicity,
    _helicity_id,
    adaptive_profile_points,
)
from tools.performance_report.models import Accuracy, ExecutionMode, Workload
from tools.performance_report.runner import (
    RunnerError,
    SelectorContract,
    point_digest,
    validate_selector_contract,
)


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
        self.entry = FakeEntry(
            process_pdgs=pdgs,
            color_order=(
                (2, 1, *range(3, len(pdgs) + 1))
                if len(pdgs) >= 6
                else (2, 4, 1, 3)
            ),
        )
        self.selected_calls: list[int] = []
        self.color_calls: list[tuple[str, tuple[int, ...] | None]] = []
        self.lc_probe_result: object | None = None

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
        if accuracy == Accuracy.LC.value and self.lc_probe_result is not None:
            return self.lc_probe_result
        aggregate = 2.5 if accuracy == Accuracy.LC.value else 9.75
        return SimpleNamespace(
            value=aggregate,
            lc_row_partitions=(
                SimpleNamespace(row=1, value=2.5, permutation=(2, 4, 1)),
            ),
            lc_partition_sum=2.5,
        )


class FakeExecutor:
    def __init__(self, phase_reporter: FakePhaseReporter | None = None) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.command_generation_phase: list[bool] = []
        self.profile_calls = 0
        self.phase_reporter = phase_reporter

    def run(
        self,
        args: object,
        *,
        cwd: Path,
        environment: object = None,
    ) -> CommandResult:
        rendered = tuple(str(item) for item in args)
        self.commands.append(rendered)
        self.command_generation_phase.append(
            bool(
                self.phase_reporter is not None
                and self.phase_reporter.active
            )
        )
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
            if self.phase_reporter is not None and self.phase_reporter.active:
                # A dedicated generation-only probe must not consume a sample
                # from the subsequent adaptive runtime profile.
                factor = 1.0
            else:
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


class FakePhaseReporter:
    def __init__(self) -> None:
        self.active = False
        self.interval_count = 0

    @contextmanager
    def generation(self):
        assert not self.active
        self.interval_count += 1
        self.active = True
        try:
            yield
        finally:
            self.active = False


def _cell(
    accuracy: Accuracy,
    workload: Workload,
    *,
    process_key: str = "dd_z_jets",
    n_final: int | None = None,
):
    return next(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.measurement.accuracy is accuracy
        and cell.workload is workload
        and cell.process_key == process_key
        and cell.n_final
        == (
            n_final
            if n_final is not None
            else 1
            if process_key == "dd_z_jets"
            else cell.n_final
        )
    )


def _adapter(
    api: FakeApi | None = None,
    *,
    phase_reporter: FakePhaseReporter | None = None,
):
    fake_api = FakeApi() if api is None else api
    executor = FakeExecutor(phase_reporter)
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


@pytest.mark.parametrize(
    "workload",
    (Workload.SELECTED_FLOW, Workload.ALL_FLOW),
)
def test_legacy_generation_phase_excludes_adaptive_runtime_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workload: Workload,
) -> None:
    repository = tmp_path / "legacy"
    repository.mkdir()
    phase_reporter = FakePhaseReporter()
    adapter, _api, executor = _adapter(
        phase_reporter=phase_reporter,
    )
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            _api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in _api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in _api.pdgs),),
        ),
    )

    result = adapter.measure(
        _cell(Accuracy.LC, workload, n_final=2),
        artifact_path=tmp_path / "artifact",
        settings=_settings(repository),
        phase_reporter=phase_reporter,  # type: ignore[arg-type]
    )

    assert result["status"] == "ok"
    assert phase_reporter.interval_count == 1
    assert not phase_reporter.active
    profiled = tuple(
        in_generation
        for command, in_generation in zip(
            executor.commands,
            executor.command_generation_phase,
            strict=True,
        )
        if command
        and (
            command[0] == "./amplicol_library_benchmark"
            or command[0].endswith("amplicol_color_probe")
        )
    )
    assert profiled
    if workload is Workload.SELECTED_FLOW:
        assert not any(profiled)
        generated = tuple(
            in_generation
            for command, in_generation in zip(
                executor.commands,
                executor.command_generation_phase,
                strict=True,
            )
            if command and command[0] == "./amplicol_generate"
        )
        assert generated == (True,)
    else:
        # Direct all-flow generation is authenticated by one dedicated
        # one-point probe.  Warm-up and every adaptive timing chunk follow it
        # outside the generation interval.
        assert profiled[0] is True
        assert not any(profiled[1:])
        assert result["generation_seconds"] == pytest.approx(2.5)


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
    common = measurement["validation"]["lc_common_component"]
    assert common["value"] == 2.5
    flattened = [" ".join(command) for command in executor.commands]
    assert any("--library=create" in command for command in flattened)
    assert any("--amplicol_momenta_probe=10" in command for command in flattened)
    assert not any("--library=create-raw" in command for command in flattened)
    momenta_file = (
        tmp_path / "repository/Utilities/ME_checks/momenta_1_2.txt"
    )
    assert momenta_file.is_file()
    assert len(momenta_file.read_text(encoding="utf-8").splitlines()) == 4


def test_selected_flow_excludes_cold_generator_bootstrap_from_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, executor = _adapter()
    original_run = executor.run
    generator_builds = 0

    def run_with_cold_bootstrap(
        args: object,
        *,
        cwd: Path,
        environment: object = None,
    ) -> CommandResult:
        nonlocal generator_builds
        result = original_run(args, cwd=cwd, environment=environment)
        rendered = tuple(str(item) for item in args)
        if (
            rendered
            and rendered[0] == "make"
            and "amplicol_generate" in rendered
        ):
            generator_builds += 1
            if generator_builds == 1:
                return CommandResult(
                    args=result.args,
                    cwd=result.cwd,
                    elapsed_seconds=17.0,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    environment=result.environment,
                )
        return result

    monkeypatch.setattr(executor, "run", run_with_cold_bootstrap)
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

    assert generator_builds == 2
    assert measurement["generation_seconds"] == pytest.approx(0.55)
    commands = measurement["provenance"]["commands"]
    assert commands[1]["args"] == ["make", "-j1", "amplicol_generate"]
    assert commands[1]["elapsed_seconds"] == 17.0


@pytest.mark.parametrize(
    ("final_pdgs", "n_final", "expected_id"),
    (
        ((-11, 11, 23, 25), 4, "h:-1,+1,-1,+1,-1,+0"),
        ((-11, 11, 23, 25), 5, "h:-1,+1,-1,+1,-1,+0,-1"),
        ((-11, 11, 23, 25), 6, "h:-1,+1,-1,+1,-1,+0,-1,+1"),
        ((-11, 11, 23, 25), 7, "h:-1,+1,-1,+1,-1,+0,-1,+1,-1"),
        ((6, -6, 23, 25), 4, "h:-1,+1,-1,+1,-1,+0"),
        ((6, -6, 23, 25), 5, "h:-1,+1,-1,+1,-1,+0,-1"),
        ((6, -6, 23, 25), 6, "h:-1,+1,-1,+1,-1,+0,-1,+1"),
        ((6, -6, 23, 25), 7, "h:-1,+1,-1,+1,-1,+0,-1,+1,-1"),
    ),
)
def test_higgs_selector_ids_match_runtime_signed_zero_axis(
    final_pdgs: tuple[int, ...],
    n_final: int,
    expected_id: str,
) -> None:
    pdgs = (1, -1, *final_pdgs, *(21 for _ in range(n_final - 4)))
    helicities = _fixed_helicity(pdgs)
    points = (((1.0, 0.0, 0.0, 1.0),),)
    labels = tuple(range(1, len(pdgs) + 1))
    flow_id = "flow:2,1"
    runtime = SimpleNamespace(
        physics=SimpleNamespace(
            color_flows=(SimpleNamespace(id=flow_id, word=(2, 1)),),
            helicities=(
                SimpleNamespace(id=expected_id, values=helicities),
            ),
            external_particles=tuple(
                SimpleNamespace(label=label) for label in labels
            ),
        )
    )
    contract = SelectorContract(
        selected_color_flow_ids=(flow_id,),
        selected_color_words=((2, 1),),
        all_flow_helicity_ids=(_helicity_id(helicities),),
        all_flow_source_helicities=tuple(
            zip(labels, helicities, strict=True)
        ),
        point_digest=point_digest(points),
    )

    assert helicities[5] == 0
    assert contract.all_flow_helicity_ids == (expected_id,)
    validate_selector_contract(runtime, contract, points)


def test_higgs_selector_keeps_nonzero_signs_fail_closed() -> None:
    points = (((1.0, 0.0, 0.0, 1.0),),)
    helicities = (-1, 1, -1, 1, -1, 0)
    identifier = _helicity_id(helicities)
    runtime = SimpleNamespace(
        physics=SimpleNamespace(
            color_flows=(SimpleNamespace(id="flow:2,1", word=(2, 1)),),
            helicities=(
                SimpleNamespace(
                    id=identifier,
                    values=(1, 1, -1, 1, -1, 0),
                ),
            ),
            external_particles=tuple(
                SimpleNamespace(label=label)
                for label in range(1, len(helicities) + 1)
            ),
        )
    )
    contract = SelectorContract(
        selected_color_flow_ids=("flow:2,1",),
        selected_color_words=((2, 1),),
        all_flow_helicity_ids=(identifier,),
        all_flow_source_helicities=tuple(
            enumerate(helicities, start=1)
        ),
        point_digest=point_digest(points),
    )

    with pytest.raises(RunnerError, match="selected physical helicity"):
        validate_selector_contract(runtime, contract, points)


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
    common = measurement["validation"]["lc_common_component"]
    assert common["value"] == 2.5
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


@pytest.mark.parametrize(
    ("pdgs", "expected", "expected_id"),
    (
        (
            (2, -1, -11, 12),
            (-1, 1, 1, -1),
            "h:-1,+1,+1,-1",
        ),
        (
            (2, -1, -11, 12, 21),
            (-1, 1, 1, -1, -1),
            "h:-1,+1,+1,-1,-1",
        ),
    ),
)
def test_fixed_helicity_uses_nonzero_chiral_charged_current(
    pdgs: tuple[int, ...],
    expected: tuple[int, ...],
    expected_id: str,
) -> None:
    helicities = _fixed_helicity(pdgs)

    assert helicities == expected
    assert _helicity_id(helicities) == expected_id


def test_all_flow_rejects_structural_zero_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = FakeApi()
    api.lc_probe_result = SimpleNamespace(
        value=0.0,
        lc_row_partitions=(
            SimpleNamespace(row=1, value=0.0, permutation=(2, 4, 1)),
        ),
        lc_partition_sum=0.0,
    )
    adapter, _api, _executor = _adapter(api)
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )

    with pytest.raises(LegacyAdapterError, match="structural-zero helicity"):
        adapter.measure(
            _cell(Accuracy.LC, Workload.ALL_FLOW),
            artifact_path=tmp_path / "all",
            settings=_settings(tmp_path / "repository"),
        )


def _measure_with_lc_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: object,
) -> dict[str, object]:
    api = FakeApi()
    api.lc_probe_result = probe
    adapter, _api, _executor = _adapter(api)
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    return adapter.measure(
        _cell(Accuracy.LC, Workload.ALL_FLOW),
        artifact_path=tmp_path / "adversarial",
        settings=_settings(tmp_path / "repository"),
    )


@pytest.mark.parametrize(
    "permutation",
    (
        (2, 4),
        (2, 4, 4),
        (0, 4, 1),
        (2, 5, 1),
        (2, "4", 1),
    ),
)
def test_lc_common_probe_rejects_malformed_row_permutations(
    permutation: tuple[object, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = SimpleNamespace(
        value=2.5,
        lc_row_partitions=(
            SimpleNamespace(row=1, value=2.5, permutation=permutation),
        ),
        lc_partition_sum=2.5,
    )

    with pytest.raises(
        LegacyAdapterError,
        match=r"row permutation|every colored source label",
    ):
        _measure_with_lc_probe(tmp_path, monkeypatch, probe)


@pytest.mark.parametrize(
    "partitions",
    (
        (
            SimpleNamespace(row=1, value=1.0, permutation=(2, 4, 1)),
            SimpleNamespace(row=1, value=1.5, permutation=(2, 1, 4)),
        ),
        (
            SimpleNamespace(row=1, value=1.0, permutation=(2, 4, 1)),
            SimpleNamespace(row=2, value=1.5, permutation=(2, 4, 1)),
        ),
    ),
)
def test_lc_common_probe_rejects_duplicate_rows(
    partitions: tuple[object, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = SimpleNamespace(
        value=2.5,
        lc_row_partitions=partitions,
        lc_partition_sum=2.5,
    )

    with pytest.raises(LegacyAdapterError, match=r"duplicate"):
        _measure_with_lc_probe(tmp_path, monkeypatch, probe)


@pytest.mark.parametrize(
    ("partition_sum", "aggregate", "match"),
    (
        (3.0, 3.0, "resolved partitions do not match"),
        (2.5, 3.0, "partition sum does not match"),
        (float("nan"), 2.5, "aggregate evidence is not finite"),
    ),
)
def test_lc_common_probe_authenticates_partition_sum_and_aggregate(
    partition_sum: float,
    aggregate: float,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = SimpleNamespace(
        value=aggregate,
        lc_row_partitions=(
            SimpleNamespace(row=1, value=2.5, permutation=(2, 4, 1)),
        ),
        lc_partition_sum=partition_sum,
    )

    with pytest.raises(LegacyAdapterError, match=match):
        _measure_with_lc_probe(tmp_path, monkeypatch, probe)


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
    probe_build = (
        "make",
        "-j1",
        "amplicol_color_library_probe",
        "amplicol_color_probe",
    )
    assert probe_build in executor.commands
    assert executor.commands.index(probe_build) < next(
        index
        for index, command in enumerate(executor.commands)
        if command[0] == "./amplicol_color_library_probe"
    )


@pytest.mark.parametrize("accuracy", (Accuracy.NLC, Accuracy.FULL))
def test_three_quark_line_contracted_uses_exact_direct_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accuracy: Accuracy,
) -> None:
    api = FakeApi((1, -1, 2, -2, 3, -3, 21, 21))
    adapter, _api, executor = _adapter(api)
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )

    measurement = adapter.measure(
        _cell(
            accuracy,
            Workload.CONTRACTED,
            process_key="dd_3q_lines",
            n_final=6,
        ),
        artifact_path=tmp_path / accuracy.value,
        settings=_settings(tmp_path / "repository"),
    )

    validate_measurement(measurement)
    assert measurement["matrix_element"] == 12.5
    assert measurement["generation_seconds"] == 2.5
    assert api.color_calls == []
    assert (
        measurement["provenance"]["generation_source"]
        == "direct-imode2-three-quark-line-setup"
    )
    flattened = [" ".join(command) for command in executor.commands]
    assert any("amplicol_color_probe" in command for command in flattened)
    assert not any("amplicol_generate" in command for command in flattened)
    assert not any("amplicol_color_library_probe" in command for command in flattened)


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
