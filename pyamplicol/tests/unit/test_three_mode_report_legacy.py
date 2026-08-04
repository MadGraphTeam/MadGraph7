# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.cache import validate_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.legacy import (
    LEGACY_IMODE2_DIAGNOSTIC_ABI,
    LEGACY_IMODE2_DIAGNOSTIC_FIELD,
    LEGACY_NUMERICAL_AUTHORITY_ABI,
    LEGACY_NUMERICAL_AUTHORITY_FIELD,
    CommandResult,
    LegacyAdapterError,
    LegacyMeasurementAdapter,
    LegacySettings,
    MaintainedLegacyApi,
    TimingRow,
    _canonical_mapped_color_word,
    _canonical_process_entry,
    _fixed_helicity,
    _helicity_id,
    _parse_generated_library_color_probe_output,
    adaptive_profile_points,
)
from tools.performance_report.models import Accuracy, ExecutionMode, Workload
from tools.performance_report.runner import (
    ProfilingTimeLimitError,
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
                # Three open strings with any adjoints attached to the final
                # string: [fundamental, adjoints..., antifundamental].
                (2, 1, 3, 4, 5, *range(7, len(pdgs) + 1), 6)
                if len(pdgs) >= 6
                else (2, 4, 1, 3)
            ),
        )
        self.selected_calls: list[tuple[int, tuple[int, ...]]] = []
        self.color_calls: list[tuple[str, tuple[int, ...] | None]] = []
        self.lc_probe_result: object | None = None
        self.generated_library_probe_value = 9.75
        self.direct_color_probe_value = 9.75
        self.selected_flow_value = 3.25
        self.selected_flow_fixed_helicity_value = 2.5
        self.parsed_probe_count = 0

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
        self.parsed_probe_count += 1
        marker = "FAKE_PROBE_VALUE "
        values = [
            float(line.removeprefix(marker))
            for line in output.splitlines()
            if line.startswith(marker)
        ]
        assert len(values) == 1
        return SimpleNamespace(value=values[0])

    def run_selected_flow_probe(self, repository: Path, **kwargs: object) -> object:
        self.selected_calls.append((int(kwargs["points"]), tuple(kwargs["helicities"])))
        return SimpleNamespace(
            value=self.selected_flow_value,
            fixed_helicity_value=self.selected_flow_fixed_helicity_value,
        )

    def run_color_probe(self, repository: Path, **kwargs: object) -> object:
        accuracy = str(kwargs["color_accuracy"])
        helicities = kwargs["helicities"]
        self.color_calls.append(
            (accuracy, None if helicities is None else tuple(helicities))
        )
        if accuracy == Accuracy.LC.value and self.lc_probe_result is not None:
            return self.lc_probe_result
        aggregate = (
            2.5
            if accuracy == Accuracy.LC.value
            else self.direct_color_probe_value
        )
        return SimpleNamespace(
            value=aggregate,
            lc_row_partitions=(
                SimpleNamespace(row=1, value=2.5, permutation=(2, 4, 1)),
            ),
            lc_partition_sum=2.5,
        )


class EntryMappedFakeApi(FakeApi):
    """Expose distinct generated entries so selector tests cannot use a stub row."""

    def __init__(
        self,
        pdgs: tuple[int, ...],
        entries: tuple[FakeEntry, ...],
        *,
        selected_entry: FakeEntry,
    ) -> None:
        super().__init__(pdgs)
        self.entries = entries
        self.selected_entry = selected_entry
        self.mapped_entries: list[FakeEntry] = []
        self.probed_entries: list[FakeEntry] = []

    def parse_process_file(self, path: Path) -> tuple[object, ...]:
        return self.entries

    def select_generated_process_entry(
        self,
        entries: tuple[object, ...],
        *,
        generated_process: str,
        wanted_pdgs: tuple[int, ...],
    ) -> tuple[object, tuple[object, ...]]:
        assert entries == self.entries
        assert wanted_pdgs == self.pdgs
        return self.selected_entry, self.entries

    def source_mapped_color_order(
        self,
        entry: object,
        *,
        source_pdgs: tuple[int, ...],
    ) -> tuple[int, ...]:
        assert source_pdgs == self.pdgs
        assert isinstance(entry, FakeEntry)
        self.mapped_entries.append(entry)
        return entry.color_order

    def run_color_probe(self, repository: Path, **kwargs: object) -> object:
        entry = kwargs["entry"]
        assert isinstance(entry, FakeEntry)
        self.probed_entries.append(entry)
        return super().run_color_probe(repository, **kwargs)


class FakeExecutor:
    def __init__(
        self,
        api: FakeApi,
        phase_reporter: FakePhaseReporter | None = None,
    ) -> None:
        self.api = api
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
            "FAKE_PROBE_VALUE "
            + (
                f"{self.api.generated_library_probe_value}\n"
                if rendered and rendered[0] == "./amplicol_color_library_probe"
                else "12.5\n"
            )
        )
        if rendered and rendered[0] == "./amplicol_color_library_probe":
            accuracy = rendered[4]
            value = self.api.generated_library_probe_value
            output += (
                f"AMPICOL_COLOR_PROBE_COMPONENTS {value} {value} {value}\n"
                f"AMPICOL_COLOR_PROBE_VALUE {accuracy} {rendered[2]} "
                f"{rendered[3]} {value}\n"
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
        (destination / "Library").mkdir(exist_ok=True)
        (destination / "processes.txt").write_text("fake\n", encoding="utf-8")
        for executable in executables:
            (destination / executable).touch()
            (destination / executable).chmod(0o755)
        return destination


class FakePhaseReporter:
    def __init__(self) -> None:
        self.active = False
        self.interval_count = 0
        self.phase = "pre-generation"
        self.transitions: list[str] = []

    @contextmanager
    def generation(self):
        assert not self.active
        assert self.phase == "pre-generation"
        self.interval_count += 1
        self.active = True
        self.phase = "generation"
        try:
            yield
        finally:
            self.active = False
            self.phase = "post-generation"

    def profiling_started(self) -> None:
        assert self.phase == "post-generation"
        self.phase = "profiling"
        self.transitions.append(self.phase)

    def validation_started(self) -> None:
        assert self.phase == "profiling"
        self.phase = "validation"
        self.transitions.append(self.phase)

    def complete(self) -> None:
        assert self.phase == "validation"
        self.phase = "complete"
        self.transitions.append(self.phase)


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
    executor = FakeExecutor(fake_api, phase_reporter)
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


def test_canonical_process_entry_retains_preferred_duplicate_word() -> None:
    pdgs = (1, -1, 11, -11)
    first = FakeEntry(group=1, integral=1, process_pdgs=pdgs, color_order=(2, 1, 3, 4))
    preferred = FakeEntry(
        group=2,
        integral=1,
        process_pdgs=pdgs,
        color_order=(2, 1, 4, 3),
    )
    api = EntryMappedFakeApi(pdgs, (first, preferred), selected_entry=preferred)

    entry, _mapped, word = _canonical_process_entry(
        api,
        (first, preferred),
        preferred_entry=preferred,
        source_pdgs=pdgs,
        initial_state_count=2,
    )

    assert entry is preferred
    assert word == (2, 1)


@pytest.mark.parametrize(
    "pdgs",
    (
        (1, -1, 2, -2, 3, -3, 21),
        (1, -1, 2, -2, 2, -2, 21),
    ),
)
def test_real_three_line_process_file_keeps_exact_source_carrier(
    pdgs: tuple[int, ...],
    tmp_path: Path,
) -> None:
    process_file = tmp_path / "processes.txt"
    pdg_row = " ".join(str(pdg) for pdg in pdgs)
    process_file.write_text(
        "\n".join(
            (
                "7 1",
                pdg_row,
                "2",
                "4 1 1 1 3 4 5 6 2 7",
                f"1 4 {pdg_row} 2 7 1 3 4 5 6 0.5",
                "5 1 1 1 3 4 5 6 7 2",
                f"1 5 {pdg_row} 2 1 3 4 5 7 6 0.5",
                "",
            )
        ),
        encoding="utf-8",
    )
    api = MaintainedLegacyApi()
    entries = api.parse_process_file(process_file)
    preferred, matches = api.select_generated_process_entry(
        entries,
        generated_process="d d~ > three open lines plus g",
        wanted_pdgs=pdgs,
    )

    entry, mapped, word = _canonical_process_entry(
        api,
        matches,
        preferred_entry=preferred,
        source_pdgs=pdgs,
        initial_state_count=2,
    )

    assert int(entry.group) == 4
    assert mapped == (2, 7, 1, 3, 4, 5, 6)
    assert word == (2, 7, 1, 3, 4, 5, 6)


def test_real_two_line_process_file_keeps_cross_paired_source_carrier(
    tmp_path: Path,
) -> None:
    process_file = tmp_path / "processes.txt"
    process_file.write_text(
        """6 4
1 6 -1 -6 21 21
1 6 -6 -1 21 21
6 1 -1 -6 21 21
6 1 -6 -1 21 21


2

1   1   1   1 2 3 4 6 5
1   1   1 -1 21 21 6 -6   5 1 2 3 4 6   2.0



2   1   1   1 2 3 6 5 4
1   2   1 -1 21 21 6 -6   5 4 1 2 3 6   2.0
""",
        encoding="utf-8",
    )
    pdgs = (1, -1, 6, -6, 21, 21)
    api = MaintainedLegacyApi()
    entries = api.parse_process_file(process_file)
    preferred, matches = api.select_generated_process_entry(
        entries,
        generated_process="d d~ > t t~ g g",
        wanted_pdgs=pdgs,
    )

    entry, mapped, word = _canonical_process_entry(
        api,
        matches,
        preferred_entry=preferred,
        source_pdgs=pdgs,
        initial_state_count=2,
    )

    assert int(entry.group) == 1
    assert mapped == (3, 1, 2, 5, 6, 4)
    assert word == (2, 5, 6, 4, 3, 1)


@pytest.mark.parametrize(
    ("n_final", "pdgs", "selected_word", "other_word", "reverse_entries"),
    (
        (
            4,
            (1, -1, 2, -2, 2, -2),
            (2, 1, 3, 4, 5, 6),
            (2, 4, 3, 1, 5, 6),
            False,
        ),
        (
            4,
            (1, -1, 2, -2, 2, -2),
            (2, 1, 3, 4, 5, 6),
            (2, 4, 3, 1, 5, 6),
            True,
        ),
        (
            5,
            (1, -1, 2, -2, 2, -2, 21),
            (2, 7, 1, 3, 4, 5, 6),
            (2, 1, 3, 4, 5, 7, 6),
            False,
        ),
        (
            5,
            (1, -1, 2, -2, 2, -2, 21),
            (2, 7, 1, 3, 4, 5, 6),
            (2, 1, 3, 4, 5, 7, 6),
            True,
        ),
    ),
)
def test_identical_three_line_lc_uses_selected_entry_mapped_row(
    n_final: int,
    pdgs: tuple[int, ...],
    selected_word: tuple[int, ...],
    other_word: tuple[int, ...],
    reverse_entries: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_entry = FakeEntry(group=2, process_pdgs=pdgs, color_order=selected_word)
    other_entry = FakeEntry(group=1, process_pdgs=pdgs, color_order=other_word)
    entries = (
        (selected_entry, other_entry)
        if reverse_entries
        else (other_entry, selected_entry)
    )
    # The exact-source process-file row is the generated-library carrier.
    # Reordering the surrounding candidate rows must not replace it.
    api = EntryMappedFakeApi(pdgs, entries, selected_entry=selected_entry)
    api.lc_probe_result = SimpleNamespace(
        value=4.0,
        lc_row_partitions=(
            SimpleNamespace(row=1, value=1.5, permutation=other_word),
            SimpleNamespace(row=2, value=2.5, permutation=selected_word),
        ),
        lc_partition_sum=4.0,
    )
    adapter, _api, _executor = _adapter(api)
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in pdgs),),
        ),
    )

    measurement = adapter.measure(
        _cell(
            Accuracy.LC,
            Workload.ALL_FLOW,
            process_key="dd_3q_identical_lines",
            n_final=n_final,
        ),
        artifact_path=tmp_path / f"identical-n{n_final}",
        settings=_settings(tmp_path / "repository"),
    )

    contract = SelectorContract.from_mapping(measurement["selector_contract"])
    assert api.mapped_entries == [selected_entry]
    assert api.probed_entries == [selected_entry]
    assert contract.selected_color_flow_ids == (
        "flow:" + ",".join(str(label) for label in selected_word),
    )
    assert contract.selected_color_words == (selected_word,)
    assert measurement["validation"]["lc_common_component"]["value"] == 2.5


@pytest.mark.parametrize(
    (
        "n_final",
        "pdgs",
        "selected_word",
        "same_canonical_word",
        "other_word",
    ),
    (
        (
            4,
            (1, -1, 2, -2, 2, -2),
            (2, 1, 3, 4, 5, 6),
            (3, 4, 2, 1, 5, 6),
            (2, 4, 3, 1, 5, 6),
        ),
        (
            5,
            (1, -1, 2, -2, 2, -2, 21),
            (2, 7, 1, 3, 4, 5, 6),
            (3, 4, 2, 7, 1, 5, 6),
            (2, 1, 3, 4, 5, 7, 6),
        ),
    ),
)
def test_identical_three_line_lc_rejects_duplicate_canonical_probe_rows(
    n_final: int,
    pdgs: tuple[int, ...],
    selected_word: tuple[int, ...],
    same_canonical_word: tuple[int, ...],
    other_word: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_entry = FakeEntry(
        group=2,
        process_pdgs=pdgs,
        color_order=selected_word,
    )
    api = EntryMappedFakeApi(
        pdgs,
        (
            FakeEntry(group=1, process_pdgs=pdgs, color_order=other_word),
            selected_entry,
        ),
        selected_entry=selected_entry,
    )
    api.lc_probe_result = SimpleNamespace(
        value=3.0,
        lc_row_partitions=(
            SimpleNamespace(row=1, value=1.0, permutation=selected_word),
            SimpleNamespace(row=2, value=2.0, permutation=same_canonical_word),
        ),
        lc_partition_sum=3.0,
    )
    adapter, _api, _executor = _adapter(api)
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in pdgs),),
        ),
    )

    with pytest.raises(
        LegacyAdapterError,
        match="did not identify exactly one selected physical row",
    ):
        adapter.measure(
            _cell(
                Accuracy.LC,
                Workload.ALL_FLOW,
                process_key="dd_3q_identical_lines",
                n_final=n_final,
            ),
            artifact_path=tmp_path / f"identical-duplicate-n{n_final}",
            settings=_settings(tmp_path / "repository"),
        )


def test_generated_library_color_probe_parser_accepts_real_counterless_stdout() -> None:
    stdout = "\n".join(
        (
            " Initialising amplitude for:",
            "    - a single polarisation/helicity configuration",
            "    - all colour orders",
            "AmpliCol generated-library colour probe",
            "points 100",
            "group 1",
            "integral 1",
            "color_accuracy full",
            "color_orders 2",
            (
                "AMPICOL_COLOR_PROBE_COMPONENTS   4.7180245588895062E-04"
                "   3.6371847402940423E-04   3.6371847402940423E-04"
            ),
            (
                "AMPICOL_COLOR_PROBE_RAW_COMPONENTS   8.1453155359829235E-02"
                "   6.2793266551648116E-02   6.2793266551648116E-02"
            ),
            "AMPICOL_COLOR_PROBE_VALUE full 1 1   3.6371847402940423E-04",
            "------------------------------------------------------------------------------",
            "Timing summary                           seconds    percent  note",
            "------------------------------------------------------------------------------",
            (
                "amplitude evaluation                    0.000056     56.57%  "
                "outer-loop-diagnostic"
            ),
            (
                "colour contraction                      0.000031     31.31%  "
                "outer-loop-diagnostic"
            ),
            "total                                   0.000099    100.00%  ",
            "------------------------------------------------------------------------------",
        )
    )

    value = _parse_generated_library_color_probe_output(
        stdout,
        expected_accuracy="full",
        expected_group=1,
        expected_integral=1,
    )

    assert "AMPICOL_COLOR_PROBE_CURRENTS" not in stdout
    assert value == 3.6371847402940423e-4


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
    assert phase_reporter.transitions == ["profiling", "validation"]
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


def test_legacy_command_is_flushed_before_launch(tmp_path: Path) -> None:
    log_path = tmp_path / "legacy.log"
    commands: list[dict[str, object]] = []

    class InspectingExecutor:
        def run(self, args, *, cwd: Path, environment=None) -> CommandResult:
            assert cwd == tmp_path
            assert environment is None
            assert log_path.read_text(encoding="utf-8") == (
                "$ probe 1\n[launch] intended_points=1\n"
            )
            assert commands == [
                {
                    "args": ["probe", "1"],
                    "cwd": str(tmp_path.resolve()),
                    "environment": {},
                    "status": "launching",
                    "intended_points": 1,
                }
            ]
            return CommandResult(
                args=tuple(args),
                cwd=cwd,
                elapsed_seconds=0.25,
                returncode=0,
                stdout="done\n",
                stderr="",
                environment={},
            )

    adapter = LegacyMeasurementAdapter(
        api=FakeApi(),
        executor=InspectingExecutor(),
        snapshotter=FakeSnapshotter(),
    )

    adapter._run(
        ("probe", "1"),
        cwd=tmp_path,
        commands=commands,
        log_path=log_path,
    )

    assert commands[0]["returncode"] == 0
    assert "status" not in commands[0]
    assert log_path.read_text(encoding="utf-8").endswith("done\n")


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

    assert calls == [1, 100, 100, 100, 100, 100, 10, 10, 10]


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
    assert api.selected_calls == [(1, (-1, 1, -1, 1))]
    assert api.color_calls == []
    common = measurement["validation"]["lc_common_component"]
    assert common["value"] == 2.5
    assert measurement["validation"][LEGACY_NUMERICAL_AUTHORITY_FIELD] == {
        "abi": LEGACY_NUMERICAL_AUTHORITY_ABI,
        "source": "selected-flow-generated-library",
    }
    flattened = [" ".join(command) for command in executor.commands]
    assert any("--library=create" in command for command in flattened)
    assert any("--amplicol_momenta_probe=10" in command for command in flattened)
    assert not any("--library=create-raw" in command for command in flattened)
    assert not any("amplicol_color_probe" in command for command in flattened)
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
    make_commands = [command for command in commands if command["args"][0] == "make"]
    assert make_commands
    assert all(
        command["environment"] == {"PDF_BACKEND": "internal"}
        for command in make_commands
    )


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
    assert LEGACY_NUMERICAL_AUTHORITY_FIELD not in measurement["validation"]
    provenance = measurement["provenance"]
    assert provenance["generation_timing_is_workload_specific"] is True
    assert provenance["row_selection_policy"] == (
        "exact-external-pdg-order-then-process-file-order-v1"
    )
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


def test_all_flow_replays_selected_flow_provider_as_authoritative_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, _executor = _adapter()
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    provider = adapter.measure(
        _cell(Accuracy.LC, Workload.SELECTED_FLOW),
        artifact_path=tmp_path / "selected",
        settings=_settings(tmp_path / "repository"),
    )
    api.lc_probe_result = SimpleNamespace(
        value=2.75,
        lc_row_partitions=(
            SimpleNamespace(row=1, value=2.75, permutation=(2, 4, 1)),
        ),
        lc_partition_sum=2.75,
    )

    measurement = adapter.measure(
        _cell(Accuracy.LC, Workload.ALL_FLOW),
        artifact_path=tmp_path / "all",
        settings=_settings(tmp_path / "repository"),
        selector_provider=provider,
    )

    validate_measurement(measurement)
    assert measurement["status"] == "ok"
    assert measurement["matrix_element"] == 12.5
    assert measurement["validation"]["lc_common_component"]["value"] == 2.5
    assert measurement["validation"][LEGACY_NUMERICAL_AUTHORITY_FIELD] == {
        "abi": LEGACY_NUMERICAL_AUTHORITY_ABI,
        "source": "all-flow-selected-provider-replay",
    }
    diagnostic = measurement["validation"][LEGACY_IMODE2_DIAGNOSTIC_FIELD]
    assert diagnostic == {
        "abi": LEGACY_IMODE2_DIAGNOSTIC_ABI,
        "certifying": False,
        "authoritative_source": "selected-flow-generated-library-component",
        "authoritative_value": 2.5,
        "imode2_value": 2.75,
        "absolute_difference": 0.25,
        "relative_difference": 0.1,
    }
    assert api.selected_calls == [
        (1, (-1, 1, -1, 1)),
        (1, (-1, 1, -1, 1)),
    ]


def test_high_flow_provider_omits_out_of_scope_imode2_component_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdgs = (1, -1, 2, -2, 3, -3, 21, 21, 21, 21, 21)
    api = FakeApi(pdgs)
    adapter, _api, _executor = _adapter(api)
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in pdgs),),
        ),
    )
    provider = adapter.measure(
        _cell(
            Accuracy.LC,
            Workload.SELECTED_FLOW,
            process_key="dd_3q_lines",
            n_final=9,
        ),
        artifact_path=tmp_path / "selected",
        settings=_settings(tmp_path / "repository"),
    )

    measurement = adapter.measure(
        _cell(
            Accuracy.LC,
            Workload.ALL_FLOW,
            process_key="dd_3q_lines",
            n_final=9,
        ),
        artifact_path=tmp_path / "all",
        settings=_settings(tmp_path / "repository"),
        selector_provider=provider,
    )

    validate_measurement(measurement)
    assert measurement["status"] == "ok"
    assert measurement["validation"]["lc_common_component"]["value"] == 2.5
    assert measurement["validation"][LEGACY_NUMERICAL_AUTHORITY_FIELD] == {
        "abi": LEGACY_NUMERICAL_AUTHORITY_ABI,
        "source": "all-flow-selected-provider-replay",
    }
    assert LEGACY_IMODE2_DIAGNOSTIC_FIELD not in measurement["validation"]
    assert api.color_calls == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("status", "not a successful measurement"),
        ("selector", "contract does not match"),
        ("component", "invalid common component"),
    ),
)
def test_all_flow_rejects_invalid_selected_flow_provider_before_measurement(
    mutation: str,
    match: str,
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
    provider = adapter.measure(
        _cell(Accuracy.LC, Workload.SELECTED_FLOW),
        artifact_path=tmp_path / "selected",
        settings=_settings(tmp_path / "repository"),
    )
    provider = copy.deepcopy(provider)
    if mutation == "status":
        provider["status"] = "validation_failed"
    elif mutation == "selector":
        provider["selector_contract"]["point_digest"] = "b" * 64
    else:
        provider["validation"]["lc_common_component"]["value"] = float("nan")
    commands_before = len(executor.commands)

    with pytest.raises(LegacyAdapterError, match=match):
        adapter.measure(
            _cell(Accuracy.LC, Workload.ALL_FLOW),
            artifact_path=tmp_path / "all",
            settings=_settings(tmp_path / "repository"),
            selector_provider=provider,
        )

    new_commands = executor.commands[commands_before:]
    assert not any(
        command and command[0].endswith("amplicol_color_probe")
        for command in new_commands
    )


def test_all_flow_rejects_provider_replay_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, _executor = _adapter()
    monkeypatch.setattr(
        "tools.performance_report.legacy._shared_point",
        lambda _process: (
            api.pdgs,
            tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),
            (tuple((1.0, 0.0, 0.0, 0.0) for _ in api.pdgs),),
        ),
    )
    provider = adapter.measure(
        _cell(Accuracy.LC, Workload.SELECTED_FLOW),
        artifact_path=tmp_path / "selected",
        settings=_settings(tmp_path / "repository"),
    )
    api.selected_flow_fixed_helicity_value = 2.5001

    with pytest.raises(LegacyAdapterError, match="replay disagrees"):
        adapter.measure(
            _cell(Accuracy.LC, Workload.ALL_FLOW),
            artifact_path=tmp_path / "all",
            settings=_settings(tmp_path / "repository"),
            selector_provider=provider,
        )


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
def test_contracted_publishes_dedicated_library_probe_with_imode2_diagnostic(
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
    assert measurement["status"] == "ok"
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
    assert api.parsed_probe_count == 0
    diagnostic = measurement["validation"][LEGACY_IMODE2_DIAGNOSTIC_FIELD]
    assert diagnostic == {
        "abi": LEGACY_IMODE2_DIAGNOSTIC_ABI,
        "certifying": False,
        "authoritative_source": "dedicated-generated-library-probe",
        "authoritative_value": 9.75,
        "imode2_value": 9.75,
        "absolute_difference": 0.0,
        "relative_difference": 0.0,
    }
    assert measurement["validation"][LEGACY_NUMERICAL_AUTHORITY_FIELD] == {
        "abi": LEGACY_NUMERICAL_AUTHORITY_ABI,
        "source": "contracted-generated-library",
    }


@pytest.mark.parametrize("accuracy", [Accuracy.NLC, Accuracy.FULL])
def test_contracted_keeps_known_imode2_mismatch_non_certifying(
    accuracy: Accuracy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, api, _executor = _adapter()
    api.direct_color_probe_value = 10.0
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
    assert measurement["status"] == "ok"
    assert measurement["matrix_element"] == 9.75
    assert measurement["failure"] is None
    diagnostic = measurement["validation"][LEGACY_IMODE2_DIAGNOSTIC_FIELD]
    assert diagnostic["certifying"] is False
    assert diagnostic["authoritative_value"] == 9.75
    assert diagnostic["imode2_value"] == 10.0
    assert diagnostic["absolute_difference"] == 0.25
    assert diagnostic["relative_difference"] == pytest.approx(0.25 / 9.75)


@pytest.mark.parametrize("accuracy", (Accuracy.NLC, Accuracy.FULL))
def test_three_quark_line_contracted_uses_exact_direct_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accuracy: Accuracy,
) -> None:
    api = FakeApi((1, -1, 2, -2, 3, -3, 21, 21))
    api.entry = FakeEntry(
        process_pdgs=api.pdgs,
        color_order=(2, 7, 8, 1, 3, 4, 5, 6),
    )
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
    assert measurement["validation"][LEGACY_NUMERICAL_AUTHORITY_FIELD] == {
        "abi": LEGACY_NUMERICAL_AUTHORITY_ABI,
        "source": "direct-imode2-three-quark-line",
    }
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


def test_legacy_profile_does_not_launch_chunk_larger_than_remaining_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _api, _executor = _adapter()
    calls: list[int] = []
    clock = iter((0.0, 0.0, 0.0))
    monkeypatch.setattr(
        "tools.performance_report.legacy.time.monotonic",
        lambda: next(clock),
    )

    def invoke(points: int):
        calls.append(points)
        result = CommandResult(
            args=("probe", str(points)),
            cwd=tmp_path,
            elapsed_seconds=0.2,
            returncode=0,
            stdout="",
            stderr="",
            environment={},
        )
        return result, (TimingRow("total", 0.2),), None

    with pytest.raises(ProfilingTimeLimitError, match="legacy 10-point chunk"):
        adapter._profile(
            invoke,
            settings=LegacySettings(
                target_runtime_seconds=1.0,
                minimum_points=10,
                maximum_points=100,
                profiling_time_limit_seconds=1.0,
                repository=tmp_path,
            ),
            timing_labels=("total",),
        )

    assert calls == [1]
