# SPDX-License-Identifier: 0BSD
"""Independent original-AmpliCol adapter for performance-report cells.

This module deliberately depends on the maintained developer oracle rather
than the historical report driver.  It owns only campaign orchestration:
process-row selection, the three legacy measurement paths, bounded adaptive
sampling, compact result shaping, and provenance capture.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from .agreements import (
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_FIELD,
    legacy_lc_common_component,
)
from .cache import empty_measurement
from .catalog import REPORT_CATALOG
from .models import (
    LEGACY_AMPLICOL_MAX_OPEN_QUARK_LINES,
    Accuracy,
    CellSpec,
    ExecutionMode,
    ResultStatus,
    Workload,
)
from .phase_state import WorkerPhaseReporter
from .runner import (
    DEFAULT_TARGET_RUNTIME_SECONDS,
    SelectorContract,
    point_digest,
)

DEFAULT_WARMUP_POINTS = 100
DEFAULT_MIN_POINTS = 100
DEFAULT_MAX_POINTS = 10_000_000
DEFAULT_MIN_PROFILE_CHUNKS = 5
DEFAULT_MAX_PROFILE_CHUNKS = 512
MAX_OPEN_QUARK_LINES = LEGACY_AMPLICOL_MAX_OPEN_QUARK_LINES


class LegacyAdapterError(RuntimeError):
    """The original-AmpliCol measurement contract could not be satisfied."""


@dataclass(frozen=True, slots=True)
class LegacySettings:
    target_runtime_seconds: float = DEFAULT_TARGET_RUNTIME_SECONDS
    warmup_points: int = DEFAULT_WARMUP_POINTS
    minimum_points: int = DEFAULT_MIN_POINTS
    maximum_points: int = DEFAULT_MAX_POINTS
    minimum_profile_chunks: int = DEFAULT_MIN_PROFILE_CHUNKS
    maximum_profile_chunks: int = DEFAULT_MAX_PROFILE_CHUNKS
    jobs: int = 1
    repository: Path | None = None
    validate_checkout: bool = True

    def __post_init__(self) -> None:
        if self.target_runtime_seconds <= 0.0:
            raise ValueError("target_runtime_seconds must be positive")
        if self.warmup_points < 1:
            raise ValueError("warmup_points must be positive")
        if self.minimum_points < 1:
            raise ValueError("minimum_points must be positive")
        if self.maximum_points < self.minimum_points:
            raise ValueError("maximum_points must not be below minimum_points")
        if self.minimum_profile_chunks < DEFAULT_MIN_PROFILE_CHUNKS:
            raise ValueError(
                f"minimum_profile_chunks must be at least {DEFAULT_MIN_PROFILE_CHUNKS}"
            )
        if self.maximum_profile_chunks < self.minimum_profile_chunks:
            raise ValueError(
                "maximum_profile_chunks must not be below minimum_profile_chunks"
            )
        if self.jobs < 1:
            raise ValueError("jobs must be positive")


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    cwd: Path
    elapsed_seconds: float
    returncode: int
    stdout: str
    stderr: str
    environment: Mapping[str, str]

    def as_record(self) -> dict[str, object]:
        return {
            "args": list(self.args),
            "cwd": os.fspath(self.cwd.resolve(strict=False)),
            "elapsed_seconds": self.elapsed_seconds,
            "returncode": self.returncode,
            "environment": dict(self.environment),
        }


class CommandExecutor(Protocol):
    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ) -> CommandResult: ...


class SubprocessExecutor:
    """Small command seam used by unit tests and campaign supervision."""

    def run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
    ) -> CommandResult:
        rendered = tuple(os.fspath(item) for item in args)
        started = time.perf_counter()
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            env=(None if environment is None else {**os.environ, **dict(environment)}),
            capture_output=True,
            check=False,
            text=True,
        )
        result = CommandResult(
            args=rendered,
            cwd=cwd,
            elapsed_seconds=time.perf_counter() - started,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            environment={} if environment is None else dict(environment),
        )
        if completed.returncode != 0:
            raise LegacyAdapterError(
                f"command exited with {completed.returncode}: "
                f"{' '.join(rendered)}\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return result


class LegacyApi(Protocol):
    default_repository: Path

    def expected_revision(self) -> str: ...

    def validate_checkout(self, repository: Path) -> None: ...

    def compiler_provenance(self, repository: Path) -> object: ...

    def process_pdgs(self, process: str) -> tuple[int, ...]: ...

    def parse_process_file(self, path: Path) -> tuple[object, ...]: ...

    def select_generated_process_entry(
        self,
        entries: Sequence[object],
        *,
        generated_process: str,
        wanted_pdgs: Sequence[int],
    ) -> tuple[object, tuple[object, ...]]: ...

    def source_mapped_color_order(
        self,
        entry: object,
        *,
        source_pdgs: Sequence[int],
    ) -> tuple[int, ...]: ...

    def ordered_momenta(
        self,
        source_pdgs: Sequence[int],
        target_pdgs: Sequence[int],
        momenta: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]: ...

    def source_to_row_permutation(
        self,
        source_pdgs: Sequence[int],
        target_pdgs: Sequence[int],
    ) -> tuple[int, ...]: ...

    def parse_probe_output(self, output: str) -> object: ...

    def run_selected_flow_probe(
        self,
        repository: Path,
        *,
        entry: object,
        source_pdgs: Sequence[int],
        momenta: Sequence[Sequence[float]],
        points: int,
    ) -> object: ...

    def run_color_probe(
        self,
        repository: Path,
        *,
        process_file: Path,
        entry: object,
        source_pdgs: Sequence[int],
        momenta: Sequence[Sequence[float]],
        color_accuracy: str,
        helicities: Sequence[int] | None,
    ) -> object: ...


class MaintainedLegacyApi:
    """Thin adapter over ``tools.developer.legacy_amplicol``."""

    def __init__(self) -> None:
        from tools.developer import legacy_amplicol

        self._api = legacy_amplicol
        self.default_repository = legacy_amplicol.DEFAULT_REPOSITORY

    def expected_revision(self) -> str:
        return str(self._api.expected_revision())

    def validate_checkout(self, repository: Path) -> None:
        self._api.validate_checkout(repository)

    def compiler_provenance(self, repository: Path) -> object:
        return self._api._compiler_provenance(repository)

    def process_pdgs(self, process: str) -> tuple[int, ...]:
        return tuple(self._api.process_pdgs(process))

    def parse_process_file(self, path: Path) -> tuple[object, ...]:
        return tuple(self._api.parse_process_file(path))

    def select_generated_process_entry(
        self,
        entries: Sequence[object],
        *,
        generated_process: str,
        wanted_pdgs: Sequence[int],
    ) -> tuple[object, tuple[object, ...]]:
        return self._api.select_generated_process_entry(
            entries,
            generated_process=generated_process,
            wanted_pdgs=wanted_pdgs,
        )

    def source_mapped_color_order(
        self,
        entry: object,
        *,
        source_pdgs: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(
            self._api.source_mapped_color_order(
                entry,
                source_pdgs=source_pdgs,
            )
        )

    def ordered_momenta(
        self,
        source_pdgs: Sequence[int],
        target_pdgs: Sequence[int],
        momenta: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(component) for component in vector)
            for vector in self._api._ordered_binary64_momenta(
                source_pdgs,
                target_pdgs,
                momenta,
            )
        )

    def source_to_row_permutation(
        self,
        source_pdgs: Sequence[int],
        target_pdgs: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(self._api._permutation(source_pdgs, target_pdgs))

    def parse_probe_output(self, output: str) -> object:
        return self._api._parse_probe_output(output)

    def run_selected_flow_probe(
        self,
        repository: Path,
        *,
        entry: object,
        source_pdgs: Sequence[int],
        momenta: Sequence[Sequence[float]],
        points: int,
    ) -> object:
        return self._api.run_selected_flow_library_probe(
            repository,
            entry=entry,
            source_pdgs=source_pdgs,
            momenta=momenta,
            points=points,
        )

    def run_color_probe(
        self,
        repository: Path,
        *,
        process_file: Path,
        entry: object,
        source_pdgs: Sequence[int],
        momenta: Sequence[Sequence[float]],
        color_accuracy: str,
        helicities: Sequence[int] | None,
    ) -> object:
        return self._api.run_color_probe(
            repository,
            process_file=process_file,
            entry=entry,
            source_pdgs=source_pdgs,
            momenta=momenta,
            color_accuracy=color_accuracy,
            helicities=helicities,
        )


class ArtifactSnapshotter(Protocol):
    def snapshot(
        self,
        repository: Path,
        destination: Path,
        *,
        executables: Sequence[str],
        process_file: Path,
    ) -> Path: ...


class GeneratedLibrarySnapshotter:
    """Copy a generated legacy library into its immutable cell attempt."""

    def snapshot(
        self,
        repository: Path,
        destination: Path,
        *,
        executables: Sequence[str],
        process_file: Path,
    ) -> Path:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        libraries = tuple(sorted(repository.glob("libamp*.so")))
        if not libraries:
            raise LegacyAdapterError("legacy generation produced no libamp*.so")
        for source in libraries:
            shutil.copy2(source, destination / source.name)
        source_library = repository / "Library"
        if not source_library.is_dir():
            raise LegacyAdapterError("legacy generation produced no Library directory")
        shutil.copytree(source_library, destination / "Library")
        for executable in executables:
            source = repository / executable
            if not source.is_file():
                raise LegacyAdapterError(
                    f"legacy generated-library probe is missing: {source}"
                )
            shutil.copy2(source, destination / executable)
        shutil.copy2(process_file, destination / "processes.txt")
        return destination


@dataclass(frozen=True, slots=True)
class TimingRow:
    label: str
    seconds: float


@dataclass(frozen=True, slots=True)
class ProfileResult:
    points: int
    seconds: float
    rows: tuple[TimingRow, ...]
    record: Mapping[str, object]
    probe: object | None
    warmup_record: Mapping[str, object]
    standard_error_seconds_per_point: float
    relative_standard_error: float


@dataclass(frozen=True, slots=True)
class _ProfileChunk:
    points: int
    seconds: float
    rows: tuple[TimingRow, ...]
    result: CommandResult
    probe: object | None


def _profile_rate_uncertainty(
    chunks: Sequence[_ProfileChunk],
) -> tuple[float, float]:
    if len(chunks) < 2:
        return 0.0, 0.0
    rates = tuple(chunk.seconds / chunk.points for chunk in chunks)
    mean_rate = statistics.fmean(rates)
    standard_error = statistics.stdev(rates) / math.sqrt(len(rates))
    relative_standard_error = standard_error / mean_rate if mean_rate > 0.0 else 0.0
    return standard_error, relative_standard_error


@dataclass(frozen=True, slots=True)
class _ProcessContext:
    process_file: Path
    entries: tuple[object, ...]
    entry: object
    matching_rows: int
    source_pdgs: tuple[int, ...]
    momenta: tuple[tuple[float, ...], ...]
    points: tuple[tuple[tuple[float, ...], ...], ...]
    mapped_color_order: tuple[int, ...]
    selector_contract: SelectorContract


def _initial_state_count(process: str) -> int:
    initial, separator, final = process.partition(">")
    initial_count = len(initial.split())
    final_count = len(final.split())
    if (
        separator != ">"
        or process.count(">") != 1
        or initial_count < 1
        or final_count < 1
    ):
        raise LegacyAdapterError(
            f"cannot identify initial and final legs in process {process!r}"
        )
    return initial_count


def _colored_roles(
    source_pdgs: Sequence[int],
    *,
    initial_state_count: int,
) -> dict[int, str]:
    roles: dict[int, str] = {}
    for label, physical_pdg in enumerate(source_pdgs, start=1):
        absolute = abs(int(physical_pdg))
        if absolute == 21:
            roles[label] = "adjoint"
            continue
        if not 1 <= absolute <= 6:
            continue
        outgoing_pdg = (
            -int(physical_pdg) if label <= initial_state_count else int(physical_pdg)
        )
        roles[label] = "fundamental" if outgoing_pdg > 0 else "antifundamental"
    return roles


def _canonical_mapped_color_word(
    source_pdgs: Sequence[int],
    mapped_color_order: Sequence[int],
    *,
    initial_state_count: int,
) -> tuple[int, ...]:
    """Project a generated-library row onto the canonical public LC axis."""

    pdgs = tuple(int(pdg) for pdg in source_pdgs)
    mapped = tuple(int(label) for label in mapped_color_order)
    expected_labels = tuple(range(1, len(pdgs) + 1))
    if tuple(sorted(mapped)) != expected_labels:
        raise LegacyAdapterError(
            "legacy mapped color row must be a permutation of source labels "
            f"1..{len(pdgs)}, got {mapped}"
        )
    if not 0 < initial_state_count < len(pdgs):
        raise LegacyAdapterError(
            "legacy selector canonicalization requires nonempty initial and "
            "final states"
        )
    roles = _colored_roles(pdgs, initial_state_count=initial_state_count)
    return _canonical_colored_word(
        pdgs,
        tuple(label for label in mapped if label in roles),
        initial_state_count=initial_state_count,
    )


def _canonical_colored_word(
    source_pdgs: Sequence[int],
    colored_word: Sequence[int],
    *,
    initial_state_count: int,
) -> tuple[int, ...]:
    """Canonicalize one exact colored-label permutation without padding it."""

    pdgs = tuple(int(pdg) for pdg in source_pdgs)
    word = tuple(colored_word)
    if not 0 < initial_state_count < len(pdgs):
        raise LegacyAdapterError(
            "legacy selector canonicalization requires nonempty initial and "
            "final states"
        )

    roles = _colored_roles(pdgs, initial_state_count=initial_state_count)

    if not roles:
        raise LegacyAdapterError("selected legacy LC row has no colored word")
    if (
        any(isinstance(label, bool) or not isinstance(label, int) for label in word)
        or len(word) != len(roles)
        or len(set(word)) != len(word)
        or set(word) != set(roles)
    ):
        raise LegacyAdapterError(
            "legacy mapped color word must contain every colored source label "
            "exactly once"
        )

    fundamental_count = sum(role == "fundamental" for role in roles.values())
    antifundamental_count = sum(role == "antifundamental" for role in roles.values())
    if fundamental_count == antifundamental_count == 0:
        return word
    if fundamental_count != antifundamental_count:
        raise LegacyAdapterError(
            "legacy mapped color word has unbalanced fundamental endpoints"
        )

    blocks: list[tuple[int, ...]] = []
    cursor = 0
    while cursor < len(word):
        first = word[cursor]
        if roles[first] != "fundamental":
            raise LegacyAdapterError(
                "legacy mapped color word is not a concatenation of "
                "[fundamental, adjoints..., antifundamental] blocks"
            )
        block = [first]
        cursor += 1
        while cursor < len(word) and roles[word[cursor]] == "adjoint":
            block.append(word[cursor])
            cursor += 1
        if cursor >= len(word) or roles[word[cursor]] != "antifundamental":
            raise LegacyAdapterError(
                "legacy mapped color word is not a concatenation of "
                "[fundamental, adjoints..., antifundamental] blocks"
            )
        block.append(word[cursor])
        cursor += 1
        blocks.append(tuple(block))

    blocks.sort(key=lambda block: block[0])
    return tuple(label for block in blocks for label in block)


def adaptive_profile_points(
    warmup_seconds: float,
    *,
    target_runtime_seconds: float,
    warmup_points: int = DEFAULT_WARMUP_POINTS,
    minimum_points: int = DEFAULT_MIN_POINTS,
    maximum_points: int = DEFAULT_MAX_POINTS,
) -> int:
    """Choose a bounded integer sample count from one warmup."""

    if warmup_points < 1 or minimum_points < 1:
        raise ValueError("profile point bounds must be positive")
    if maximum_points < minimum_points:
        raise ValueError("maximum_points must not be below minimum_points")
    if not math.isfinite(warmup_seconds) or warmup_seconds <= 0.0:
        return minimum_points
    estimate = math.ceil(float(target_runtime_seconds) * warmup_points / warmup_seconds)
    return max(minimum_points, min(maximum_points, int(estimate)))


def _timing_rows(output: str) -> tuple[TimingRow, ...]:
    rows: list[TimingRow] = []
    in_summary = False
    for line in output.splitlines():
        if "Timing summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        seconds_index: int | None = None
        seconds: float | None = None
        for index in range(1, len(tokens)):
            try:
                candidate = float(tokens[index])
            except ValueError:
                continue
            seconds_index = index
            seconds = candidate
            break
        if seconds_index is None or seconds is None:
            continue
        rows.append(TimingRow(" ".join(tokens[:seconds_index]).lower(), seconds))
    return tuple(rows)


def _timing_seconds(rows: Sequence[TimingRow], *labels: str) -> float | None:
    for label in labels:
        wanted = label.strip().lower()
        for row in rows:
            if row.label == wanted:
                return row.seconds
        for row in rows:
            if wanted in row.label:
                return row.seconds
    return None


def _quark_line_count(pdgs: Sequence[int]) -> int:
    return sum(1 for pdg in pdgs if 1 <= abs(int(pdg)) <= 6) // 2


def _preferred_helicities(pdg: int) -> tuple[int, ...]:
    absolute = abs(int(pdg))
    if absolute in {
        1,
        2,
        3,
        4,
        5,
        6,
        11,
        12,
        13,
        14,
        15,
        16,
        21,
        22,
    }:
        return (-1, 1)
    if absolute in {23, 24}:
        return (-1, 0, 1)
    return (0,)


def _fixed_helicity(pdgs: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    charged_current_fermions = any(abs(int(pdg)) in {12, 14, 16} for pdg in pdgs)
    for index, pdg in enumerate(pdgs, start=1):
        domain = _preferred_helicities(pdg)
        if -1 in domain and 1 in domain:
            # Prefer the left-chiral particle/right-chiral antiparticle state
            # for charged-current fermion chains.  The former position-parity
            # heuristic chose the structural-zero e+ nu_e state in families
            # such as u d~ > e+ ve, then propagated that zero selector through
            # every pyAmpliCol baseline.  Preserve the established neutral-
            # current/vector policy outside this focused correction.
            if charged_current_fermions and (
                1 <= abs(int(pdg)) <= 6 or 11 <= abs(int(pdg)) <= 16
            ):
                result.append(-1 if int(pdg) > 0 else 1)
            else:
                result.append(-1 if index % 2 else 1)
        elif 0 in domain:
            result.append(0)
        else:
            result.append(domain[0])
    return tuple(result)


def _helicity_id(values: Sequence[int]) -> str:
    return "h:" + ",".join(f"{int(value):+d}" for value in values)


def _compiler_payload(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def _shared_point(
    process: str,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[tuple[float, ...], ...], ...],
]:
    from pyamplicol.models.builtin.validation import generic_validation_point

    particles = tuple(generic_validation_point(process))
    pdgs = tuple(int(particle.pdg) for particle in particles)
    momenta = tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in particles
    )
    return pdgs, momenta, (momenta,)


def _library_environment(path: Path) -> dict[str, str]:
    existing = os.environ.get("LD_LIBRARY_PATH")
    root = os.fspath(path.resolve(strict=False))
    return {"LD_LIBRARY_PATH": root if not existing else f"{root}:{existing}"}


@contextmanager
def _staged_process_file(repository: Path, process_file: Path):
    repository.mkdir(parents=True, exist_ok=True)
    target = repository / "processes.txt"
    backup = repository / ".processes.txt.pyamplicol-report-backup"
    existed = target.exists()
    if existed:
        shutil.copy2(target, backup)
    shutil.copy2(process_file, target)
    try:
        yield target.name
    finally:
        if existed:
            shutil.move(backup, target)
        else:
            target.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


class LegacyMeasurementAdapter:
    """Measure one original-AmpliCol report workload."""

    def __init__(
        self,
        *,
        api: LegacyApi | None = None,
        executor: CommandExecutor | None = None,
        snapshotter: ArtifactSnapshotter | None = None,
        structural_proof: bool | None = None,
    ) -> None:
        use_default_api = api is None
        self.api = MaintainedLegacyApi() if api is None else api
        self.executor = SubprocessExecutor() if executor is None else executor
        self.snapshotter = (
            GeneratedLibrarySnapshotter() if snapshotter is None else snapshotter
        )
        # Injected APIs are unit-test/developer seams and preserve the historical
        # adapter surface unless the caller explicitly supplies real probe
        # sources.  The production worker constructs the default maintained API,
        # for which structural evidence is mandatory and fail-closed.
        self.structural_proof = (
            use_default_api if structural_proof is None else structural_proof
        )

    def measure(
        self,
        cell: CellSpec,
        *,
        artifact_path: Path,
        settings: LegacySettings,
        phase_reporter: WorkerPhaseReporter | None = None,
    ) -> dict[str, object]:
        if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL:
            raise LegacyAdapterError("legacy adapter requires an AmpliCol cell")
        source_pdgs = self.api.process_pdgs(cell.process)
        if _quark_line_count(source_pdgs) > MAX_OPEN_QUARK_LINES:
            if phase_reporter is not None:
                # Unsupported catalog declarations perform no generation, but
                # a supervised worker must still close its authenticated phase
                # channel rather than remain indefinitely in pre-generation.
                with phase_reporter.generation():
                    pass
            result = self._unsupported_measurement(
                f"original AmpliCol supports at most {MAX_OPEN_QUARK_LINES} "
                "open quark lines in this report"
            )
            if self.structural_proof:
                from .legacy_structure import (
                    emit_legacy_scope_unavailable_proof,
                )

                artifact_path.mkdir(parents=True, exist_ok=True)
                revision = self.api.expected_revision()
                proof = emit_legacy_scope_unavailable_proof(
                    cell,
                    artifact_path=artifact_path,
                    source_revision=revision,
                    maximum_open_quark_lines=MAX_OPEN_QUARK_LINES,
                    observed_open_quark_lines=_quark_line_count(source_pdgs),
                )
                result["artifact"] = {
                    "path": os.fspath(artifact_path),
                    "legacy_structural_proof": os.fspath(proof),
                }
                result["provenance"] = {
                    "method": "original-amplicol-scope-boundary",
                    "revision": revision,
                }
            return result

        repository = settings.repository or self.api.default_repository
        repository_context: Any = nullcontext()
        if self.structural_proof:
            from .legacy_structure import legacy_structural_probe_lock

            repository_context = legacy_structural_probe_lock(repository)
        with repository_context:
            if settings.validate_checkout:
                self.api.validate_checkout(repository)
            artifact_path.mkdir(parents=True, exist_ok=True)
            instrumentation_context: Any = nullcontext(None)
            if self.structural_proof:
                from .legacy_structure import instrument_legacy_structural_probes

                instrumentation_context = instrument_legacy_structural_probes(
                    repository,
                    artifact_path,
                )
            with instrumentation_context as instrumentation:
                return self._measure_supported(
                    cell,
                    repository=repository,
                    artifact_path=artifact_path,
                    settings=settings,
                    instrumentation=instrumentation,
                    phase_reporter=phase_reporter,
                )

    def _measure_supported(
        self,
        cell: CellSpec,
        *,
        repository: Path,
        artifact_path: Path,
        settings: LegacySettings,
        instrumentation: object | None,
        phase_reporter: WorkerPhaseReporter | None,
    ) -> dict[str, object]:
        log_path = artifact_path / "legacy.log"
        commands: list[dict[str, object]] = []
        context = self._prepare_process(
            cell,
            repository=repository,
            artifact_path=artifact_path,
            commands=commands,
            log_path=log_path,
        )
        if cell.measurement.accuracy is Accuracy.LC:
            if cell.workload is Workload.SELECTED_FLOW:
                result = self._measure_selected_flow(
                    cell,
                    context=context,
                    repository=repository,
                    artifact_path=artifact_path,
                    settings=settings,
                    commands=commands,
                    log_path=log_path,
                    phase_reporter=phase_reporter,
                )
            elif cell.workload is Workload.ALL_FLOW:
                result = self._measure_all_flow(
                    cell,
                    context=context,
                    repository=repository,
                    artifact_path=artifact_path,
                    settings=settings,
                    commands=commands,
                    log_path=log_path,
                    phase_reporter=phase_reporter,
                )
            else:
                raise LegacyAdapterError("LC AmpliCol cell requires an LC workload")
        elif cell.workload is Workload.CONTRACTED:
            result = self._measure_contracted(
                cell,
                context=context,
                repository=repository,
                artifact_path=artifact_path,
                settings=settings,
                commands=commands,
                log_path=log_path,
                phase_reporter=phase_reporter,
            )
        else:
            raise LegacyAdapterError("NLC/full AmpliCol cells must be contracted")

        process_row = (
            f"group:{int(context.entry.group)}:integral:{int(context.entry.integral)}"
        )
        result["artifact"] = {
            "path": os.fspath(artifact_path),
            "process_row": process_row,
            "log_path": os.fspath(log_path),
        }
        result["selector_contract"] = (
            context.selector_contract.as_dict()
            if cell.measurement.accuracy is Accuracy.LC
            else None
        )
        result["validation"] = {
            "status": ResultStatus.OK.value,
            "method": "independent-original-amplicol-oracle",
            "point_digest": context.selector_contract.point_digest,
            DIRECT_AGREEMENT_FIELD: [],
        }
        if cell.measurement.accuracy is Accuracy.LC:
            common_component = self._measure_lc_common_component(
                cell,
                context=context,
                repository=repository,
                commands=commands,
                log_path=log_path,
            )
            result["validation"][LC_COMMON_COMPONENT_FIELD] = common_component
            if cell.workload is Workload.ALL_FLOW and (
                float(result["matrix_element"]) <= 0.0
                or float(common_component["value"]) <= 0.0
            ):
                raise LegacyAdapterError(
                    "LC all-flow selector resolved to a structural-zero "
                    "helicity; choose and remeasure a nonzero fixed-helicity "
                    "selector before publishing dependent cells"
                )
        result["resources"] = {
            "monitor": "external-cell-supervisor",
            "peak_rss_gib": None,
        }
        try:
            compiler = _compiler_payload(self.api.compiler_provenance(repository))
        except Exception as error:
            compiler = {
                "status": "unavailable",
                "error": str(error),
            }
        revision = self.api.expected_revision()
        result["provenance"] = {
            **dict(result.get("provenance") or {}),
            "method": "original-amplicol-generated-library",
            "revision": revision,
            "compiler": compiler,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "repository": os.fspath(repository.resolve(strict=False)),
            "row_selection_policy": (
                "exact-external-pdg-order-then-process-file-order-v1"
            ),
            "matching_row_count": context.matching_rows,
            "raw_mapped_color_order": list(context.mapped_color_order),
            "selector_color_word_policy": (
                "outgoing-open-string-blocks-by-fundamental-source-label-v1"
            ),
            "commands": commands,
            "target_runtime_seconds": settings.target_runtime_seconds,
            "warmup_points": settings.warmup_points,
            "minimum_points": settings.minimum_points,
            "maximum_points": settings.maximum_points,
            "minimum_profile_chunks": settings.minimum_profile_chunks,
            "maximum_profile_chunks": settings.maximum_profile_chunks,
            "generation_timing_is_workload_specific": True,
        }
        if instrumentation is not None:
            from .legacy_structure import emit_legacy_structural_proof

            proof = emit_legacy_structural_proof(
                cell,
                artifact_path=artifact_path,
                process_row=process_row,
                source_revision=revision,
                repository=repository,
                instrumentation=instrumentation,
            )
            result["artifact"]["legacy_structural_proof"] = os.fspath(proof)
        result["failure"] = None
        return result

    def _measure_lc_common_component(
        self,
        cell: CellSpec,
        *,
        context: _ProcessContext,
        repository: Path,
        commands: list[dict[str, object]],
        log_path: Path,
    ) -> dict[str, object]:
        """Probe the fixed-helicity/fixed-flow component common to both lanes."""

        if cell.workload is Workload.SELECTED_FLOW:
            self._run(
                ("make", "amplicol_color_probe"),
                cwd=repository,
                commands=commands,
                log_path=log_path,
            )
        helicities = tuple(
            value
            for _label, value in (context.selector_contract.all_flow_source_helicities)
        )
        started = time.perf_counter()
        probe = self.api.run_color_probe(
            repository,
            process_file=context.process_file,
            entry=context.entry,
            source_pdgs=context.source_pdgs,
            momenta=context.momenta,
            color_accuracy=Accuracy.LC.value,
            helicities=helicities,
        )
        commands.append(
            {
                "args": [
                    "legacy_amplicol.run_color_probe",
                    Accuracy.LC.value,
                    "fixed-helicity",
                    "points=1",
                ],
                "cwd": os.fspath(repository),
                "elapsed_seconds": time.perf_counter() - started,
                "returncode": 0,
            }
        )
        partitions = tuple(getattr(probe, "lc_row_partitions", ()))
        if not partitions:
            raise LegacyAdapterError(
                "direct LC probe emitted no resolved row partitions"
            )
        source_to_row = self.api.source_to_row_permutation(
            context.source_pdgs,
            context.entry.process_pdgs,
        )
        if (
            len(source_to_row) != len(context.source_pdgs)
            or any(
                isinstance(position, bool) or not isinstance(position, int)
                for position in source_to_row
            )
            or set(source_to_row) != set(range(len(context.source_pdgs)))
        ):
            raise LegacyAdapterError(
                "direct LC probe source-to-row mapping is not a permutation"
            )
        wanted_word = context.selector_contract.selected_color_words[0]
        matching: list[object] = []
        seen_rows: set[int] = set()
        seen_words: set[tuple[int, ...]] = set()
        resolved_values: list[float] = []
        for partition in partitions:
            raw_permutation = tuple(getattr(partition, "permutation", ()))
            row = getattr(partition, "row", None)
            if (
                isinstance(row, bool)
                or not isinstance(row, int)
                or row < 1
                or row in seen_rows
            ):
                raise LegacyAdapterError(
                    "direct LC probe emitted an invalid or duplicate row index"
                )
            seen_rows.add(row)
            if (
                any(
                    isinstance(position, bool) or not isinstance(position, int)
                    for position in raw_permutation
                )
                or len(set(raw_permutation)) != len(raw_permutation)
                or any(
                    position < 1 or position > len(source_to_row)
                    for position in raw_permutation
                )
            ):
                raise LegacyAdapterError(
                    "direct LC probe emitted an invalid row permutation"
                )
            source_word = tuple(
                source_to_row[position - 1] + 1 for position in raw_permutation
            )
            if source_word in seen_words:
                raise LegacyAdapterError(
                    "direct LC probe emitted a duplicate physical row"
                )
            seen_words.add(source_word)
            canonical_word = _canonical_colored_word(
                context.source_pdgs,
                source_word,
                initial_state_count=_initial_state_count(cell.process),
            )
            raw_value = getattr(partition, "value", None)
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
            ):
                raise LegacyAdapterError(
                    "direct LC probe emitted a non-finite partition value"
                )
            resolved_values.append(float(raw_value))
            if canonical_word == wanted_word:
                matching.append(partition)
        if sorted(seen_rows) != list(range(1, len(partitions) + 1)):
            raise LegacyAdapterError(
                "direct LC probe row indices are not a complete contiguous axis"
            )
        raw_partition_sum = getattr(probe, "lc_partition_sum", None)
        raw_aggregate = getattr(probe, "value", None)
        if (
            isinstance(raw_partition_sum, bool)
            or not isinstance(raw_partition_sum, (int, float))
            or not math.isfinite(float(raw_partition_sum))
            or isinstance(raw_aggregate, bool)
            or not isinstance(raw_aggregate, (int, float))
            or not math.isfinite(float(raw_aggregate))
        ):
            raise LegacyAdapterError("direct LC probe aggregate evidence is not finite")
        resolved_sum = math.fsum(resolved_values)
        if not math.isclose(
            resolved_sum,
            float(raw_partition_sum),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise LegacyAdapterError(
                "direct LC probe resolved partitions do not match their sum"
            )
        if not math.isclose(
            float(raw_partition_sum),
            float(raw_aggregate),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise LegacyAdapterError(
                "direct LC probe partition sum does not match its aggregate"
            )
        if len(matching) != 1:
            raise LegacyAdapterError(
                "direct LC probe did not identify exactly one selected physical row"
            )
        raw_value = getattr(matching[0], "value", None)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            raise LegacyAdapterError("direct LC probe selected row has no finite value")
        return legacy_lc_common_component(
            cell,
            context.selector_contract,
            float(raw_value),
        )

    def _prepare_process(
        self,
        cell: CellSpec,
        *,
        repository: Path,
        artifact_path: Path,
        commands: list[dict[str, object]],
        log_path: Path,
    ) -> _ProcessContext:
        source_pdgs, momenta, points = _shared_point(cell.process)
        declared_pdgs = self.api.process_pdgs(cell.process)
        if source_pdgs != declared_pdgs:
            raise LegacyAdapterError(
                "shared validation point external order differs from the "
                "legacy process expression"
            )
        family = next(
            (
                item
                for item in REPORT_CATALOG.process_families
                if item.key == cell.process_key
            ),
            None,
        )
        flags: list[str] = []
        if family is not None:
            if family.include_3qqbar:
                flags.append("-3")
            if family.include_cc:
                flags.append("-cc")
            if family.include_resonance:
                flags.append("-res")
        process_result = self._run(
            [
                sys.executable,
                repository / "process_list.py",
                "--serial",
                *flags,
                cell.process,
            ],
            cwd=artifact_path,
            commands=commands,
            log_path=log_path,
        )
        process_file = artifact_path / "processes.txt"
        if not process_file.is_file():
            raise LegacyAdapterError(
                "legacy process_list.py did not produce processes.txt; "
                f"output={process_result.stdout[-1000:]!r}"
            )
        entries = self.api.parse_process_file(process_file)
        entry, matches = self.api.select_generated_process_entry(
            entries,
            generated_process=cell.process,
            wanted_pdgs=source_pdgs,
        )
        mapped = self.api.source_mapped_color_order(
            entry,
            source_pdgs=source_pdgs,
        )
        color_word = _canonical_mapped_color_word(
            source_pdgs,
            mapped,
            initial_state_count=_initial_state_count(cell.process),
        )
        helicities = _fixed_helicity(source_pdgs)
        contract = SelectorContract(
            selected_color_flow_ids=(
                "flow:" + ",".join(str(label) for label in color_word),
            ),
            selected_color_words=(color_word,),
            all_flow_helicity_ids=(_helicity_id(helicities),),
            all_flow_source_helicities=tuple(
                (index, value) for index, value in enumerate(helicities, start=1)
            ),
            point_digest=point_digest(points),
        )
        return _ProcessContext(
            process_file=process_file,
            entries=entries,
            entry=entry,
            matching_rows=len(matches),
            source_pdgs=source_pdgs,
            momenta=momenta,
            points=points,
            mapped_color_order=mapped,
            selector_contract=contract,
        )

    def _generate_library(
        self,
        *,
        context: _ProcessContext,
        repository: Path,
        raw_color: bool,
        settings: LegacySettings,
        commands: list[dict[str, object]],
        log_path: Path,
        phase_reporter: WorkerPhaseReporter | None = None,
    ) -> float:
        # Building the reusable generator and its object graph is campaign
        # bootstrap, not per-process generation.  Resolve that one-time cost
        # before opening the generation timing boundary.  ``cleanlib`` updates
        # dummy.o, so the ordinary per-process relink below remains measured.
        self._run(
            ("make", f"-j{settings.jobs}", "amplicol_generate"),
            cwd=repository,
            commands=commands,
            log_path=log_path,
        )
        started_index = len(commands)
        generation_context = (
            nullcontext() if phase_reporter is None else phase_reporter.generation()
        )
        with generation_context:
            momenta_directory = repository / "Utilities" / "ME_checks"
            momenta_directory.mkdir(parents=True, exist_ok=True)
            for entry in context.entries:
                ordered = self.api.ordered_momenta(
                    context.source_pdgs,
                    entry.process_pdgs,
                    context.momenta,
                )
                momenta_path = (
                    momenta_directory / f"momenta_{entry.group}_{entry.integral}.txt"
                )
                momenta_path.write_text(
                    "\n".join(
                        " ".join(f"{component:.17e}" for component in vector)
                        for vector in ordered
                    )
                    + "\n",
                    encoding="utf-8",
                )
            with _staged_process_file(
                repository,
                context.process_file,
            ) as process_arg:
                for args in (
                    ("make", "cleanlib"),
                    ("make", f"-j{settings.jobs}", "amplicol_generate"),
                    (
                        "./amplicol_generate",
                        f"--library={'create-raw' if raw_color else 'create'}",
                        f"--process={process_arg}",
                        "--amplicol_momenta_probe=10",
                        "--amplicol_probe_quiet",
                        "--timing=none",
                    ),
                    ("make", f"-j{settings.jobs}", "amplicol_generate_library"),
                ):
                    self._run(
                        args,
                        cwd=repository,
                        commands=commands,
                        log_path=log_path,
                    )
        return math.fsum(
            float(record["elapsed_seconds"]) for record in commands[started_index:]
        )

    def _measure_selected_flow(
        self,
        cell: CellSpec,
        *,
        context: _ProcessContext,
        repository: Path,
        artifact_path: Path,
        settings: LegacySettings,
        commands: list[dict[str, object]],
        log_path: Path,
        phase_reporter: WorkerPhaseReporter | None,
    ) -> dict[str, object]:
        generation_seconds = self._generate_library(
            context=context,
            repository=repository,
            raw_color=False,
            settings=settings,
            commands=commands,
            log_path=log_path,
            phase_reporter=phase_reporter,
        )
        self._run(
            ("make", f"-j{settings.jobs}", "amplicol_library_benchmark"),
            cwd=repository,
            commands=commands,
            log_path=log_path,
        )
        generated = self.snapshotter.snapshot(
            repository,
            artifact_path / "selected-flow-generated-library",
            executables=("amplicol_library_benchmark",),
            process_file=context.process_file,
        )
        environment = _library_environment(generated)
        profile = self._profile(
            lambda count: self._invoke_command(
                (
                    "./amplicol_library_benchmark",
                    str(count),
                    str(context.entry.group),
                    str(context.entry.integral),
                ),
                cwd=generated,
                environment=environment,
                commands=commands,
                log_path=log_path,
            ),
            settings=settings,
            timing_labels=("amplitude evaluation", "total"),
        )
        with _temporary_environment(environment):
            probe_started = time.perf_counter()
            probe = self.api.run_selected_flow_probe(
                generated,
                entry=context.entry,
                source_pdgs=context.source_pdgs,
                momenta=context.momenta,
                points=1,
            )
            probe_record = {
                "args": [
                    "legacy_amplicol.run_selected_flow_library_probe",
                    os.fspath(generated),
                    "points=1",
                ],
                "cwd": os.fspath(generated),
                "elapsed_seconds": time.perf_counter() - probe_started,
                "returncode": 0,
            }
            commands.append(probe_record)
        return self._success_measurement(
            generation_seconds=generation_seconds,
            profile=profile,
            matrix_element=float(probe.value),
            generation_source="generated-library-create-mode-1",
        )

    def _measure_all_flow(
        self,
        cell: CellSpec,
        *,
        context: _ProcessContext,
        repository: Path,
        artifact_path: Path,
        settings: LegacySettings,
        commands: list[dict[str, object]],
        log_path: Path,
        phase_reporter: WorkerPhaseReporter | None,
    ) -> dict[str, object]:
        self._run(
            ("make", f"-j{settings.jobs}", "amplicol_color_probe"),
            cwd=repository,
            commands=commands,
            log_path=log_path,
        )
        helicities = tuple(
            value
            for _label, value in (context.selector_contract.all_flow_source_helicities)
        )
        with tempfile.TemporaryDirectory(prefix="pac-", dir="/tmp") as raw:
            work = Path(raw)
            process_copy = work / "processes.txt"
            momenta_path = work / "momenta.dat"
            shutil.copy2(context.process_file, process_copy)
            ordered = self.api.ordered_momenta(
                context.source_pdgs,
                context.entry.process_pdgs,
                context.momenta,
            )
            momenta_path.write_text(
                "\n".join(
                    " ".join(format(component, ".17g") for component in vector)
                    for vector in ordered
                )
                + "\n",
                encoding="utf-8",
            )
            permutation = self.api.source_to_row_permutation(
                context.source_pdgs,
                context.entry.process_pdgs,
            )
            ordered_helicities = tuple(helicities[index] for index in permutation)
            probe_args = (
                repository / "amplicol_color_probe",
                str(1),
                str(context.entry.group),
                str(context.entry.integral),
                "lc",
                process_copy,
                momenta_path,
                *(str(value) for value in ordered_helicities),
            )
            if phase_reporter is not None:
                _generation_result, generation_rows, generation_probe = (
                    self._invoke_generation_probe(
                        probe_args,
                        phase_reporter=phase_reporter,
                        cwd=work,
                        commands=commands,
                        log_path=log_path,
                    )
                )
            profile = self._profile(
                lambda count: self._invoke_probe_command(
                    (
                        repository / "amplicol_color_probe",
                        str(count),
                        str(context.entry.group),
                        str(context.entry.integral),
                        "lc",
                        process_copy,
                        momenta_path,
                        *(str(value) for value in ordered_helicities),
                    ),
                    cwd=work,
                    commands=commands,
                    log_path=log_path,
                ),
                settings=settings,
                timing_labels=("amplitude evaluation", "total"),
            )
            if phase_reporter is None:
                # Preserve the unsupervised adapter's historical single-profile
                # behavior.  Campaign workers always supply a reporter and use
                # the dedicated authenticated generation probe above.
                generation_rows = profile.rows
                generation_probe = profile.probe
            if self.structural_proof:
                from .legacy_structure import (
                    STRUCTURAL_PROBE_ENVIRONMENT,
                    write_legacy_structural_probe_output,
                )

                structural, _rows, _probe = self._invoke_probe_command(
                    probe_args,
                    cwd=work,
                    commands=commands,
                    log_path=log_path,
                    environment={STRUCTURAL_PROBE_ENVIRONMENT: "1"},
                )
                write_legacy_structural_probe_output(
                    artifact_path
                    / "legacy-structural-evidence"
                    / "direct-structural-probe.stdout",
                    stdout=structural.stdout,
                    stderr=structural.stderr,
                )
        generation_seconds = _timing_seconds(
            generation_rows,
            "generation setup",
        )
        if generation_seconds is None:
            raise LegacyAdapterError(
                "direct all-flow probe did not report generation setup"
            )
        if generation_probe is None:
            raise LegacyAdapterError("direct all-flow probe emitted no value")
        return self._success_measurement(
            generation_seconds=generation_seconds,
            profile=profile,
            matrix_element=float(generation_probe.value),
            generation_source="direct-imode2-generation-setup",
        )

    def _measure_contracted(
        self,
        cell: CellSpec,
        *,
        context: _ProcessContext,
        repository: Path,
        artifact_path: Path,
        settings: LegacySettings,
        commands: list[dict[str, object]],
        log_path: Path,
        phase_reporter: WorkerPhaseReporter | None,
    ) -> dict[str, object]:
        if _quark_line_count(context.source_pdgs) == MAX_OPEN_QUARK_LINES:
            return self._measure_direct_contracted(
                cell,
                context=context,
                repository=repository,
                artifact_path=artifact_path,
                settings=settings,
                commands=commands,
                log_path=log_path,
                phase_reporter=phase_reporter,
            )
        generation_seconds = self._generate_library(
            context=context,
            repository=repository,
            raw_color=True,
            settings=settings,
            commands=commands,
            log_path=log_path,
            phase_reporter=phase_reporter,
        )
        self._run(
            (
                "make",
                f"-j{settings.jobs}",
                "amplicol_color_library_probe",
                "amplicol_color_probe",
            ),
            cwd=repository,
            commands=commands,
            log_path=log_path,
        )
        generated = self.snapshotter.snapshot(
            repository,
            artifact_path / "contracted-generated-library",
            executables=("amplicol_color_library_probe",),
            process_file=context.process_file,
        )
        ordered = self.api.ordered_momenta(
            context.source_pdgs,
            context.entry.process_pdgs,
            context.momenta,
        )
        momenta_path = generated / "momenta.dat"
        momenta_path.write_text(
            "\n".join(
                " ".join(format(component, ".17g") for component in vector)
                for vector in ordered
            )
            + "\n",
            encoding="utf-8",
        )
        environment = _library_environment(generated)
        profile = self._profile(
            lambda count: self._invoke_command(
                (
                    "./amplicol_color_library_probe",
                    str(count),
                    str(context.entry.group),
                    str(context.entry.integral),
                    cell.measurement.accuracy.value,
                    momenta_path.name,
                ),
                cwd=generated,
                environment=environment,
                commands=commands,
                log_path=log_path,
            ),
            settings=settings,
            timing_labels=("total",),
        )
        if self.structural_proof:
            from .legacy_structure import (
                STRUCTURAL_PROBE_ENVIRONMENT,
                write_legacy_structural_probe_output,
            )

            structural, _rows, _probe = self._invoke_command(
                (
                    "./amplicol_color_library_probe",
                    "1",
                    str(context.entry.group),
                    str(context.entry.integral),
                    cell.measurement.accuracy.value,
                    momenta_path.name,
                ),
                cwd=generated,
                environment={
                    **environment,
                    STRUCTURAL_PROBE_ENVIRONMENT: "1",
                },
                commands=commands,
                log_path=log_path,
            )
            write_legacy_structural_probe_output(
                artifact_path
                / "legacy-structural-evidence"
                / "contracted-structural-probe.stdout",
                stdout=structural.stdout,
                stderr=structural.stderr,
            )
        probe_started = time.perf_counter()
        probe = self.api.run_color_probe(
            repository,
            process_file=context.process_file,
            entry=context.entry,
            source_pdgs=context.source_pdgs,
            momenta=context.momenta,
            color_accuracy=cell.measurement.accuracy.value,
            helicities=None,
        )
        commands.append(
            {
                "args": [
                    "legacy_amplicol.run_color_probe",
                    cell.measurement.accuracy.value,
                    "points=1",
                ],
                "cwd": os.fspath(repository),
                "elapsed_seconds": time.perf_counter() - probe_started,
                "returncode": 0,
            }
        )
        return self._success_measurement(
            generation_seconds=generation_seconds,
            profile=profile,
            matrix_element=float(probe.value),
            generation_source="generated-library-create-raw",
        )

    def _measure_direct_contracted(
        self,
        cell: CellSpec,
        *,
        context: _ProcessContext,
        repository: Path,
        artifact_path: Path,
        settings: LegacySettings,
        commands: list[dict[str, object]],
        log_path: Path,
        phase_reporter: WorkerPhaseReporter | None,
    ) -> dict[str, object]:
        """Measure the exact direct path for three independent quark lines."""

        self._run(
            ("make", f"-j{settings.jobs}", "amplicol_color_probe"),
            cwd=repository,
            commands=commands,
            log_path=log_path,
        )
        with tempfile.TemporaryDirectory(prefix="pac-", dir="/tmp") as raw:
            work = Path(raw)
            process_copy = work / "processes.txt"
            momenta_path = work / "momenta.dat"
            shutil.copy2(context.process_file, process_copy)
            ordered = self.api.ordered_momenta(
                context.source_pdgs,
                context.entry.process_pdgs,
                context.momenta,
            )
            momenta_path.write_text(
                "\n".join(
                    " ".join(format(component, ".17g") for component in vector)
                    for vector in ordered
                )
                + "\n",
                encoding="utf-8",
            )
            generation_args = (
                repository / "amplicol_color_probe",
                "1",
                str(context.entry.group),
                str(context.entry.integral),
                cell.measurement.accuracy.value,
                process_copy,
                momenta_path,
            )
            if phase_reporter is not None:
                _generation_result, generation_rows, generation_probe = (
                    self._invoke_generation_probe(
                        generation_args,
                        phase_reporter=phase_reporter,
                        cwd=work,
                        commands=commands,
                        log_path=log_path,
                    )
                )
            profile = self._profile(
                lambda count: self._invoke_probe_command(
                    (
                        repository / "amplicol_color_probe",
                        str(count),
                        str(context.entry.group),
                        str(context.entry.integral),
                        cell.measurement.accuracy.value,
                        process_copy,
                        momenta_path,
                    ),
                    cwd=work,
                    commands=commands,
                    log_path=log_path,
                ),
                settings=settings,
                timing_labels=("total",),
            )
            if phase_reporter is None:
                generation_rows = profile.rows
                generation_probe = profile.probe
            if self.structural_proof:
                from .legacy_structure import (
                    STRUCTURAL_PROBE_ENVIRONMENT,
                    write_legacy_structural_probe_output,
                )

                structural, _rows, _probe = self._invoke_probe_command(
                    (
                        repository / "amplicol_color_probe",
                        "1",
                        str(context.entry.group),
                        str(context.entry.integral),
                        cell.measurement.accuracy.value,
                        process_copy,
                        momenta_path,
                    ),
                    cwd=work,
                    commands=commands,
                    log_path=log_path,
                    environment={STRUCTURAL_PROBE_ENVIRONMENT: "1"},
                )
                write_legacy_structural_probe_output(
                    artifact_path
                    / "legacy-structural-evidence"
                    / "direct-structural-probe.stdout",
                    stdout=structural.stdout,
                    stderr=structural.stderr,
                )
        generation_seconds = _timing_seconds(
            generation_rows,
            "generation setup",
        )
        if generation_seconds is None:
            raise LegacyAdapterError(
                "direct three-quark-line probe did not report generation setup"
            )
        if generation_probe is None:
            raise LegacyAdapterError("direct three-quark-line probe emitted no value")
        return self._success_measurement(
            generation_seconds=generation_seconds,
            profile=profile,
            matrix_element=float(generation_probe.value),
            generation_source="direct-imode2-three-quark-line-setup",
        )

    def _profile(
        self,
        invoke: Any,
        *,
        settings: LegacySettings,
        timing_labels: Sequence[str],
    ) -> ProfileResult:
        warmup_result, warmup_rows, _warmup_probe = invoke(settings.warmup_points)
        warmup_seconds = _timing_seconds(warmup_rows, *timing_labels)
        if warmup_seconds is None or warmup_seconds <= 0.0:
            warmup_seconds = warmup_result.elapsed_seconds
        chunks: list[_ProfileChunk] = []
        phase = "measurement_chunks"
        points = adaptive_profile_points(
            warmup_seconds,
            target_runtime_seconds=(
                settings.target_runtime_seconds / settings.minimum_profile_chunks
            ),
            warmup_points=settings.warmup_points,
            minimum_points=settings.minimum_points,
            maximum_points=settings.maximum_points,
        )
        measured_seconds = 0.0
        for _index in range(settings.maximum_profile_chunks):
            result, rows, probe = invoke(points)
            seconds = _timing_seconds(rows, *timing_labels)
            if seconds is None or seconds <= 0.0:
                seconds = result.elapsed_seconds
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise LegacyAdapterError(
                    "legacy timing chunk did not report a positive duration"
                )
            chunks.append(
                _ProfileChunk(
                    points,
                    seconds,
                    tuple(rows),
                    result,
                    probe,
                )
            )
            measured_seconds = math.fsum(chunk.seconds for chunk in chunks)
            minimum_complete = len(chunks) >= settings.minimum_profile_chunks
            target_complete = measured_seconds >= settings.target_runtime_seconds
            standard_error, relative_standard_error = _profile_rate_uncertainty(chunks)
            uncertainty_complete = (
                math.isfinite(standard_error)
                and standard_error > 0.0
                and math.isfinite(relative_standard_error)
                and relative_standard_error > 0.0
            )
            if minimum_complete and target_complete and uncertainty_complete:
                break
            remaining = max(
                settings.target_runtime_seconds - measured_seconds,
                0.0,
            )
            remaining_minimum_chunks = max(
                settings.minimum_profile_chunks - len(chunks),
                1,
            )
            next_chunk_target = remaining / remaining_minimum_chunks
            estimated = math.ceil(next_chunk_target * points / seconds)
            points = max(
                settings.minimum_points,
                min(settings.maximum_points, int(estimated)),
            )
        else:
            raise LegacyAdapterError(
                "legacy timing did not reach its target, minimum sample "
                "count, and positive measured uncertainty within "
                f"{settings.maximum_profile_chunks} bounded chunks"
            )

        points = sum(chunk.points for chunk in chunks)
        seconds = math.fsum(chunk.seconds for chunk in chunks)
        if seconds < settings.target_runtime_seconds:
            raise LegacyAdapterError(
                "legacy timing completed without reaching its target runtime"
            )
        standard_error, relative_standard_error = _profile_rate_uncertainty(chunks)
        representative = chunks[0]
        final = chunks[-1]
        return ProfileResult(
            points=points,
            seconds=seconds,
            rows=representative.rows,
            record={
                **final.result.as_record(),
                "profile_phase": phase,
                "target_runtime_seconds": settings.target_runtime_seconds,
                "achieved_runtime_seconds": seconds,
                "target_runtime_achieved": True,
                "chunk_count": len(chunks),
                "total_points": points,
                "chunks": [
                    {
                        **chunk.result.as_record(),
                        "points": chunk.points,
                        "profile_seconds": chunk.seconds,
                    }
                    for chunk in chunks
                ],
            },
            probe=final.probe,
            warmup_record=warmup_result.as_record(),
            standard_error_seconds_per_point=standard_error,
            relative_standard_error=relative_standard_error,
        )

    def _invoke_command(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        commands: list[dict[str, object]],
        log_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> tuple[CommandResult, tuple[TimingRow, ...], None]:
        result = self._run(
            args,
            cwd=cwd,
            environment=environment,
            commands=commands,
            log_path=log_path,
        )
        return result, _timing_rows(result.stdout + "\n" + result.stderr), None

    def _invoke_probe_command(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        commands: list[dict[str, object]],
        log_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> tuple[CommandResult, tuple[TimingRow, ...], object]:
        result = self._run(
            args,
            cwd=cwd,
            environment=environment,
            commands=commands,
            log_path=log_path,
        )
        output = result.stdout + "\n" + result.stderr
        return result, _timing_rows(output), self.api.parse_probe_output(output)

    def _invoke_generation_probe(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        phase_reporter: WorkerPhaseReporter | None,
        cwd: Path,
        commands: list[dict[str, object]],
        log_path: Path,
    ) -> tuple[CommandResult, tuple[TimingRow, ...], object]:
        """Run one direct generation setup inside the authenticated phase.

        Direct original-AmpliCol probes combine their setup and a single
        evaluation in one process.  A dedicated one-point invocation gives the
        supervisor one bounded generation interval, while the subsequent
        adaptive runtime profile remains strictly post-generation.
        """

        generation_context = (
            nullcontext() if phase_reporter is None else phase_reporter.generation()
        )
        with generation_context:
            return self._invoke_probe_command(
                args,
                cwd=cwd,
                commands=commands,
                log_path=log_path,
            )

    def _run(
        self,
        args: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path,
        commands: list[dict[str, object]],
        log_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> CommandResult:
        result = self.executor.run(args, cwd=cwd, environment=environment)
        commands.append(result.as_record())
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"$ {' '.join(result.args)}\n")
            stream.write(result.stdout)
            if result.stdout and not result.stdout.endswith("\n"):
                stream.write("\n")
            stream.write(result.stderr)
            if result.stderr and not result.stderr.endswith("\n"):
                stream.write("\n")
        return result

    @staticmethod
    def _success_measurement(
        *,
        generation_seconds: float,
        profile: ProfileResult,
        matrix_element: float,
        generation_source: str,
    ) -> dict[str, object]:
        measurement = empty_measurement()
        measurement.update(
            {
                "status": ResultStatus.OK.value,
                "generation_seconds": float(generation_seconds),
                "wall_seconds_per_point": profile.seconds / profile.points,
                "execution_seconds_per_point": profile.seconds / profile.points,
                "matrix_element": abs(float(matrix_element)),
                "sample_count": profile.points,
                "standard_error_seconds_per_point": (
                    profile.standard_error_seconds_per_point
                ),
                "relative_standard_error": profile.relative_standard_error,
                "provenance": {
                    "generation_source": generation_source,
                    "runtime_profile": {
                        "measurement": dict(profile.record),
                        "warmup": dict(profile.warmup_record),
                        "timing_rows": [
                            {"label": row.label, "seconds": row.seconds}
                            for row in profile.rows
                        ],
                    },
                },
            }
        )
        return measurement

    @staticmethod
    def _unsupported_measurement(message: str) -> dict[str, object]:
        measurement = empty_measurement()
        measurement.update(
            {
                "status": ResultStatus.UNSUPPORTED.value,
                "failure": {
                    "kind": "LegacyOracleScopeError",
                    "message": message,
                },
            }
        )
        return measurement


@contextmanager
def _temporary_environment(values: Mapping[str, str]):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def measurement_digest(measurement: Mapping[str, object]) -> str:
    """Stable helper used when linking a legacy reference from another cache."""

    payload = json.dumps(
        measurement,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CommandExecutor",
    "CommandResult",
    "LegacyAdapterError",
    "LegacyMeasurementAdapter",
    "LegacySettings",
    "MaintainedLegacyApi",
    "SubprocessExecutor",
    "adaptive_profile_points",
    "measurement_digest",
]
