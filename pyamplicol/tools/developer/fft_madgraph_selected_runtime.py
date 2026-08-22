#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Generate and benchmark MadGraph standalone FullColor processes.

The campaign is deliberately process-by-process and resumable. For each
selected process family and final-state multiplicity ``n>=2`` it asks the
user's MadGraph installation to generate an ordinary, helicity-general
standalone process under configured per-cell wall-time and process-tree memory
guards. A generated check program then calls the
direct ``MATRIX(P,NHEL,IC)`` kernel at ten authenticated phase-space points
used by the pyAmpliCol campaign.  The default workload is one fixed selected
helicity.  ``--compare-helicity-sums`` instead calls the generated
``SMATRIX(P,ANS)`` with ``USERHEL=-1``.  MadGraph owns the complete helicity
sum, its IDEN normalization, and the warmed ``GOODHEL`` pruning; summed runs
never reuse fixed-helicity measurements.

The final-plot protocol measures MadGraph only through ``n=6``.  Requested
higher multiplicities are retained as explicit not-applicable cells without
loading their event data or invoking MadGraph.

Each measured overlay cell contains cold-to-ready wall time, a conservative
peak RSS across generation/build/runtime, and the median of ten warm point-01
CPU-time samples. A resource/time failure is the honest frontier for that
process family; higher multiplicities are recorded as skipped. Completed cells
are checkpointed before campaign-owned generated intermediates are pruned, so
interrupted campaigns resume cheaply.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "tools" / "ci" / "memory_watchdog.py"

KIND = "pyamplicol-fullcolor-selected-scalar-runtime-series-overlay"
PROGRESS_KIND = "pyamplicol-fullcolor-selected-scalar-runtime-series-progress"
SOURCE_KIND = "pyamplicol-fullcolor-selected-scalar-composite"
SCALING_STUDY_KIND = "pyamplicol-fullcolor-fft-scaling-study"
SCHEMA_VERSION = 1
MODE = "madgraph-standalone"
FIXED_LABEL = "MadGraph standalone (fixed h)"
SUM_LABEL = "MadGraph standalone (helicity sum)"
LABEL = FIXED_LABEL
FAMILIES = ("gg", "ddbar")
FINAL_MULTIPLICITIES = tuple(range(2, 10))
MAX_PROTOCOL_MEASURED_MULTIPLICITY = 6
POINT_COUNT = 10
WARM_SAMPLE_COUNT = 10
TIMING_DRIVER_BATCH_COUNT = WARM_SAMPLE_COUNT + 1
TARGET_SECONDS = 0.25
RELATIVE_TOLERANCE = 1.0e-10
MAX_GENERATION_SECONDS = 3600.0
DEFAULT_GENERATION_TIMEOUT_SECONDS = 3595.0
MAX_MEMORY_GIB = 30.0
SOURCE_MODE = {"gg": "reference-fft", "ddbar": "recurrence-fft"}
SUM_SOURCE_MODE = {"gg": "recurrence-fft", "ddbar": "recurrence-fft"}
TIMEOUT_EXIT_CODE = 124
TIMEOUT_MARKER = "madgraph-benchmark: bounded command reached its time limit"
CACHE_LOCK_ROOT = (
    Path(tempfile.gettempdir())
    / f"pyamplicol-{os.getuid()}"
    / "madgraph-cache-locks"
)
RESOURCE_FRONTIER_CATEGORIES = frozenset(
    {
        "dependency-unavailable",
        "generation-memory-limit",
        "generation-time-limit",
        "memory-limit",
        "resource-unavailable",
        "runtime-time-limit",
        "structural-memory-limit",
        "structural-unavailable",
    }
)


class SelectedMadGraphError(RuntimeError):
    """The selected-event MadGraph campaign could not be authenticated."""


class ResourceFrontierError(SelectedMadGraphError):
    """A measured resource limit establishes an honest availability frontier."""

    def __init__(self, category: str, reason: str) -> None:
        if category not in RESOURCE_FRONTIER_CATEGORIES:
            raise ValueError(f"non-resource frontier category: {category}")
        super().__init__(reason)
        self.category = category


@dataclass(frozen=True, slots=True)
class EventData:
    path: Path
    sha256: str
    final_gluons: int
    strong_coupling: float
    helicity: tuple[int, ...]
    momenta: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class SelectedCell:
    family: str
    n: int
    total_external: int
    process: str
    helicity: tuple[int, ...]
    events: tuple[EventData, ...]
    reference_point_values: tuple[float, ...]
    source_mode: str
    helicity_workload: str
    source_alpha_s: float


@dataclass(frozen=True, slots=True)
class SourceSelection:
    path: Path
    sha256: str
    alpha_s: float
    cells: Mapping[str, Mapping[int, SelectedCell]]
    unavailable: Mapping[str, Mapping[int, str]]
    helicity_workload: str


@dataclass(frozen=True, slots=True)
class DriverResult:
    total_external: int
    initialization_seconds: float
    first_pass_seconds: float
    point_values: tuple[float, ...]
    cell_seconds: tuple[tuple[float, ...], ...]
    checksum: float
    evaluations_per_sweep: int
    helicity_coverage_count: int


def _workload_label(helicity_workload: str) -> str:
    if helicity_workload == "fixed":
        return FIXED_LABEL
    if helicity_workload == "sum":
        return SUM_LABEL
    raise SelectedMadGraphError(
        f"unsupported helicity workload {helicity_workload!r}"
    )


def source_mode(family: str, helicity_workload: str) -> str:
    _workload_label(helicity_workload)
    modes = SUM_SOURCE_MODE if helicity_workload == "sum" else SOURCE_MODE
    try:
        return modes[family]
    except KeyError as error:
        raise SelectedMadGraphError(f"unsupported family {family!r}") from error


def _declared_helicity_workload(
    policy: Mapping[str, Any], *, legacy_default: str = "fixed"
) -> str:
    measurement = _mapping(
        policy.get("measurement", policy), context="helicity workload policy"
    )
    declared = measurement.get("helicity_workload")
    warm_sum = measurement.get("warm_helicity_sum")
    warm_fixed = measurement.get("warm_fixed_helicity")
    markers: set[str] = set()
    if declared is not None:
        markers.add(str(declared))
    if warm_sum is True:
        markers.add("sum")
    if warm_fixed is True:
        markers.add("fixed")
    if warm_sum is False and warm_fixed is False:
        raise SelectedMadGraphError("helicity workload policy disables both workloads")
    if len(markers) > 1 or any(value not in {"fixed", "sum"} for value in markers):
        raise SelectedMadGraphError("helicity workload policy is contradictory")
    return next(iter(markers), legacy_default)


def process_expression(family: str, n: int) -> str:
    if family not in FAMILIES:
        raise SelectedMadGraphError(f"unsupported process family {family!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise SelectedMadGraphError(f"unsupported final-state multiplicity {n}")
    extra = " ".join("g" for _ in range(n - 2))
    base = "g g > g g" if family == "gg" else "d d~ > d d~"
    return base + (f" {extra}" if extra else "")


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectedMadGraphError(f"{context} must be an object")
    return value


def _finite_number(value: object, *, context: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SelectedMadGraphError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise SelectedMadGraphError(f"{context} must be a finite number")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _event_path(raw_path: object, *, context: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SelectedMadGraphError(f"{context} must be a nonempty path string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=False)
    if not path.is_file() and "IMPLEMENTATION_DOCS" in path.parts:
        marker = path.parts.index("IMPLEMENTATION_DOCS")
        rebased = ROOT.joinpath(*path.parts[marker:]).resolve(strict=False)
        if rebased.is_file():
            path = rebased
    if not path.is_file():
        raise SelectedMadGraphError(f"selected event does not exist: {path}")
    return path


def _single_block(
    lines: Sequence[str], begin_marker: str, end_marker: str, *, path: Path
) -> list[str]:
    if lines.count(begin_marker) != 1 or lines.count(end_marker) != 1:
        raise SelectedMadGraphError(
            f"{path} must contain one {begin_marker}/{end_marker} block"
        )
    begin = lines.index(begin_marker)
    end = lines.index(end_marker)
    if end <= begin:
        raise SelectedMadGraphError(f"{path} has a malformed {begin_marker} block")
    return list(lines[begin + 1 : end])


def _read_event(
    path: Path, *, expected_n: int, default_strong_coupling: float = 1.0
) -> EventData:
    try:
        raw = path.read_bytes()
        lines = [
            line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise SelectedMadGraphError(
            f"cannot read selected event {path}: {error}"
        ) from error
    if not lines or lines[0] not in {
        "AMPLIGLUON_EVENT_V1",
        "PYAMPLICOL_SCALING_EVENT_V1",
    }:
        raise SelectedMadGraphError(f"{path} has an unsupported event format")

    strong_coupling = default_strong_coupling
    if lines[0] == "AMPLIGLUON_EVENT_V1":
        final_rows = [
            line.split() for line in lines if line.startswith("FINAL_GLUONS ")
        ]
        coupling_rows = [
            line.split() for line in lines if line.startswith("STRONG_COUPLING ")
        ]
        if len(final_rows) != 1 or final_rows[0] != [
            "FINAL_GLUONS",
            str(expected_n),
        ]:
            raise SelectedMadGraphError(f"{path} has the wrong FINAL_GLUONS")
        if len(coupling_rows) != 1 or len(coupling_rows[0]) != 2:
            raise SelectedMadGraphError(f"{path} has an invalid STRONG_COUPLING")
        try:
            strong_coupling = float(coupling_rows[0][1])
        except ValueError as error:
            raise SelectedMadGraphError(
                f"{path} has an invalid STRONG_COUPLING"
            ) from error
    if not math.isfinite(strong_coupling) or strong_coupling <= 0.0:
        raise SelectedMadGraphError(f"{path} has an invalid STRONG_COUPLING")

    momentum_rows = _single_block(lines, "BEGIN_MOMENTA", "END_MOMENTA", path=path)
    if len(momentum_rows) != expected_n + 2:
        raise SelectedMadGraphError(f"{path} has the wrong number of momenta")
    try:
        momenta = tuple(
            tuple(float(value) for value in row.split()) for row in momentum_rows
        )
    except ValueError as error:
        raise SelectedMadGraphError(f"{path} has invalid momentum values") from error
    if any(
        len(momentum) != 4 or any(not math.isfinite(value) for value in momentum)
        for momentum in momenta
    ):
        raise SelectedMadGraphError(f"{path} has invalid momentum rows")

    count_rows = [line.split() for line in lines if line.startswith("NHELICITIES ")]
    if len(count_rows) != 1 or count_rows[0] != ["NHELICITIES", "1"]:
        raise SelectedMadGraphError(f"{path} must contain exactly one helicity")
    helicity_rows = _single_block(
        lines, "BEGIN_HELICITIES", "END_HELICITIES", path=path
    )
    if len(helicity_rows) != 1:
        raise SelectedMadGraphError(f"{path} must contain one helicity row")
    try:
        helicity = tuple(int(value) for value in helicity_rows[0].split())
    except ValueError as error:
        raise SelectedMadGraphError(f"{path} has an invalid helicity row") from error
    if len(helicity) != expected_n + 2 or any(
        value not in (-1, 1) for value in helicity
    ):
        raise SelectedMadGraphError(f"{path} has an invalid helicity row")
    return EventData(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        final_gluons=expected_n,
        strong_coupling=strong_coupling,
        helicity=helicity,
        momenta=momenta,  # type: ignore[arg-type]
    )


def load_source_selection(
    path: Path,
    *,
    multiplicities: Sequence[int] = FINAL_MULTIPLICITIES,
    helicity_workload: str = "fixed",
) -> SourceSelection:
    _workload_label(helicity_workload)
    path = path.expanduser().resolve(strict=False)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SelectedMadGraphError(
            f"cannot read source report {path}: {error}"
        ) from error
    if not isinstance(payload, dict) or payload.get("kind") not in {
        SOURCE_KIND,
        SCALING_STUDY_KIND,
    }:
        raise SelectedMadGraphError("source report has the wrong kind")
    policy = _mapping(payload.get("policy"), context="source report policy")
    observed_workload = _declared_helicity_workload(policy)
    if observed_workload != helicity_workload:
        raise SelectedMadGraphError(
            "source report helicity workload differs from the requested "
            f"MadGraph workload: {observed_workload!r} != {helicity_workload!r}"
        )
    measurement = _mapping(
        policy.get("measurement"), context="source report policy.measurement"
    )
    alpha_s = _finite_number(
        measurement.get("alpha_s"),
        context="source report policy.measurement.alpha_s",
    )
    if alpha_s <= 0.0:
        raise SelectedMadGraphError("source report alpha_s must be positive")
    default_strong_coupling = math.sqrt(4.0 * math.pi * alpha_s)
    cells = _mapping(payload.get("cells"), context="source report cells")
    selected: dict[str, dict[int, SelectedCell]] = {}
    unavailable: dict[str, dict[int, str]] = {}
    for family in FAMILIES:
        mode = source_mode(family, helicity_workload)
        raw_family = _mapping(
            cells.get(family), context=f"source report cells.{family}"
        )
        raw_mode = _mapping(
            raw_family.get(mode), context=f"source report cells.{family}.{mode}"
        )
        selected[family] = {}
        unavailable[family] = {}
        for n in multiplicities:
            context = f"source report cells.{family}.{mode}.{n}"
            cell = _mapping(raw_mode.get(str(n)), context=context)
            expected_process = process_expression(family, n)
            for key, expected in (
                ("family", family),
                ("mode", mode),
                ("n", n),
                ("total_external", n + 2),
                ("process", expected_process),
            ):
                if cell.get(key) != expected:
                    raise SelectedMadGraphError(
                        f"{context}.{key} has the wrong identity"
                    )
            if cell.get("status") != "measured":
                if (
                    cell.get("status") in {"failed", "skipped"}
                    and cell.get("censors_higher_multiplicities") is True
                    and isinstance(cell.get("failure_reason"), str)
                    and cell["failure_reason"].strip()
                ):
                    unavailable[family][n] = str(cell["failure_reason"])
                    continue
                raise SelectedMadGraphError(
                    f"{context}.status is not a measured or resource-frontier cell"
                )
            cell_workload = _declared_helicity_workload(cell)
            if cell_workload != helicity_workload:
                raise SelectedMadGraphError(
                    f"{context} helicity workload differs from the source policy"
                )
            raw_helicity = cell.get("helicity")
            if (
                not isinstance(raw_helicity, Sequence)
                or isinstance(raw_helicity, str)
                or len(raw_helicity) != n + 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value not in (-1, 1)
                    for value in raw_helicity
                )
            ):
                raise SelectedMadGraphError(f"{context}.helicity is invalid")
            helicity = tuple(int(value) for value in raw_helicity)
            raw_paths = cell.get("event_paths")
            if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, str):
                raise SelectedMadGraphError(f"{context}.event_paths is invalid")
            paths = tuple(
                _event_path(value, context=f"{context}.event_paths")
                for value in raw_paths
            )
            if len(paths) != POINT_COUNT or len(set(paths)) != POINT_COUNT:
                raise SelectedMadGraphError(
                    f"{context}.event_paths must contain ten unique paths"
                )
            events = tuple(
                _read_event(
                    event,
                    expected_n=n,
                    default_strong_coupling=default_strong_coupling,
                )
                for event in paths
            )
            if any(event.helicity != helicity for event in events):
                raise SelectedMadGraphError(f"{context} event helicities differ")
            raw_points = cell.get("point_values")
            if not isinstance(raw_points, Sequence) or isinstance(raw_points, str):
                raise SelectedMadGraphError(f"{context}.point_values is invalid")
            points = tuple(
                _finite_number(
                    value, context=f"{context}.point_values", nonnegative=True
                )
                for value in raw_points
            )
            if len(points) != POINT_COUNT or not any(value > 0.0 for value in points):
                raise SelectedMadGraphError(
                    f"{context}.point_values must contain ten nonzero data"
                )
            selected[family][n] = SelectedCell(
                family=family,
                n=n,
                total_external=n + 2,
                process=expected_process,
                helicity=helicity,
                events=events,
                reference_point_values=points,
                source_mode=mode,
                helicity_workload=helicity_workload,
                source_alpha_s=alpha_s,
            )
    return SourceSelection(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        alpha_s=alpha_s,
        cells=selected,
        unavailable=unavailable,
        helicity_workload=helicity_workload,
    )


def parse_driver_output(
    output: str,
    *,
    batches: int,
    expected_total_external: int,
    expected_events: int,
    helicity_workload: str = "fixed",
) -> DriverResult:
    _workload_label(helicity_workload)
    scalar: dict[str, str] = {}
    points: dict[int, float] = {}
    cells: dict[tuple[int, int], float] = {}
    allowed_scalars = {
        "BACKEND",
        "HELICITY_EVALUATOR",
        "TOTAL_EXTERNAL",
        "INITIALIZATION_SECONDS",
        "FIRST_SAMPLE_PASS_SECONDS",
        "EVALUATIONS_PER_SWEEP",
        "HELICITY_COVERAGE_COUNT",
        "CHECKSUM",
    }
    try:
        for line in output.splitlines():
            fields = line.split()
            if not fields:
                continue
            key = fields[0]
            if key == "MATRIX_ELEMENT":
                if len(fields) != 4 or fields[2] != "1":
                    raise SelectedMadGraphError("malformed MadGraph matrix-element row")
                index = int(fields[1])
                if index in points:
                    raise SelectedMadGraphError("duplicate MadGraph matrix-element row")
                points[index] = float(fields[3].replace("D", "E").replace("d", "e"))
            elif key == "EVALUATION_CELL_SECONDS":
                if len(fields) != 5 or fields[2:4] != ["1", "1"]:
                    raise SelectedMadGraphError("malformed MadGraph timing-cell row")
                index = (int(fields[1]), 1)
                if index in cells:
                    raise SelectedMadGraphError("duplicate MadGraph timing-cell row")
                cells[index] = float(fields[4].replace("D", "E").replace("d", "e"))
            elif key == "EVALUATION_SWEEP_SECONDS":
                continue
            elif key in allowed_scalars:
                if len(fields) != 2 or key in scalar:
                    raise SelectedMadGraphError(
                        f"malformed or duplicate MadGraph {key}"
                    )
                scalar[key] = fields[1]
            else:
                raise SelectedMadGraphError(f"unexpected MadGraph output row: {line!r}")
    except ValueError as error:
        raise SelectedMadGraphError(
            "invalid numeric value in MadGraph output"
        ) from error
    if set(scalar) != allowed_scalars:
        raise SelectedMadGraphError("MadGraph output has incomplete scalar metadata")
    expected_backend = (
        "MadGraph5_aMCatNLOFixedHelicity"
        if helicity_workload == "fixed"
        else "MadGraph5_aMCatNLOHelicitySum"
    )
    expected_evaluator = (
        "MATRIX_DIRECT_VECTOR"
        if helicity_workload == "fixed"
        else "SMATRIX_GENERATED_COMPLETE_HELICITY_SUM"
    )
    if scalar["BACKEND"] != expected_backend:
        raise SelectedMadGraphError("unexpected MadGraph backend identity")
    if scalar["HELICITY_EVALUATOR"] != expected_evaluator:
        raise SelectedMadGraphError(
            "MadGraph used the wrong workload-specific evaluator"
        )
    total_external = int(scalar["TOTAL_EXTERNAL"])
    initialization = float(scalar["INITIALIZATION_SECONDS"].replace("D", "E"))
    first_pass = float(scalar["FIRST_SAMPLE_PASS_SECONDS"].replace("D", "E"))
    checksum = float(scalar["CHECKSUM"].replace("D", "E"))
    expected_evaluations = 1
    expected_coverage = (
        1 if helicity_workload == "fixed" else 2**expected_total_external
    )
    evaluations_per_sweep = int(scalar["EVALUATIONS_PER_SWEEP"])
    helicity_coverage_count = int(scalar["HELICITY_COVERAGE_COUNT"])
    if (
        total_external != expected_total_external
        or evaluations_per_sweep != expected_evaluations
        or helicity_coverage_count != expected_coverage
    ):
        raise SelectedMadGraphError("MadGraph output has the wrong process dimensions")
    if set(points) != set(range(1, expected_events + 1)):
        raise SelectedMadGraphError(
            "MadGraph output did not report every validation point"
        )
    if set(cells) != {(batch, 1) for batch in range(1, batches + 1)}:
        raise SelectedMadGraphError("MadGraph output has incomplete timing cells")
    values = (initialization, first_pass, checksum, *points.values(), *cells.values())
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise SelectedMadGraphError("MadGraph output contains invalid numeric data")
    if checksum <= 0.0:
        raise SelectedMadGraphError("MadGraph checksum must be positive")
    return DriverResult(
        total_external=total_external,
        initialization_seconds=initialization,
        first_pass_seconds=first_pass,
        point_values=tuple(points[index] for index in range(1, expected_events + 1)),
        cell_seconds=tuple((cells[(batch, 1)],) for batch in range(1, batches + 1)),
        checksum=checksum,
        evaluations_per_sweep=evaluations_per_sweep,
        helicity_coverage_count=helicity_coverage_count,
    )


def _fortran_double(value: float) -> str:
    return format(value, ".17e").replace("e", "D")


def render_check_source(selected: SelectedCell) -> str:
    """Render a check driver; the generated MATRIX remains helicity-general."""

    n_external = selected.total_external
    summed = selected.helicity_workload == "sum"
    _workload_label(selected.helicity_workload)
    backend_identity = (
        "MadGraph5_aMCatNLOHelicitySum"
        if summed
        else "MadGraph5_aMCatNLOFixedHelicity"
    )
    evaluator_identity = (
        "SMATRIX_GENERATED_COMPLETE_HELICITY_SUM"
        if summed
        else "MATRIX_DIRECT_VECTOR"
    )
    helicity_coverage_count = 2**n_external if summed else 1
    lines = [
        "      PROGRAM DRIVER",
        "      IMPLICIT NONE",
        "      INTEGER NEXTERNAL,NEVENTS,NSAMPLES",
        f"      PARAMETER (NEXTERNAL={n_external},NEVENTS={POINT_COUNT})",
        f"      PARAMETER (NSAMPLES={TIMING_DRIVER_BATCH_COUNT})",
        "      INTEGER E,B,R,NREP,CLOCK0,CLOCK1,CLOCKRATE",
        "      INTEGER HEL(NEXTERNAL),IC(NEXTERNAL)",
        "      DOUBLE PRECISION P(0:3,NEXTERNAL,NEVENTS)",
        (
            "      DOUBLE PRECISION V(NEVENTS),VALUE,BEFORE,AFTER"
            if summed
            else "      DOUBLE PRECISION V(NEVENTS),MATRIX,BEFORE,AFTER"
        ),
        "      DOUBLE PRECISION TARGET,ELAPSED,CHECKSUM,FIRST0,FIRST1",
        "      DOUBLE PRECISION PI,ASVALUE",
        "      PARAMETER (PI=3.141592653589793D0)",
    ]
    if summed:
        lines.extend(
            [
                "      INTEGER USERHEL",
                "      COMMON/HELUSERCHOICE/USERHEL",
            ]
        )
    else:
        lines.append("      EXTERNAL MATRIX")
    lines.extend(
        [
            f"      DATA HEL /{','.join(str(value) for value in selected.helicity)}/",
            f"      DATA IC /{n_external}*1/",
        ]
    )
    for event_index, event in enumerate(selected.events, start=1):
        for leg_index, momentum in enumerate(event.momenta, start=1):
            for component, value in enumerate(momentum):
                lines.append(
                    f"      P({component},{leg_index},{event_index})="
                    f"{_fortran_double(value)}"
                )
    if summed:
        as_value = _fortran_double(selected.source_alpha_s)
    else:
        coupling = selected.events[0].strong_coupling
        if any(
            not math.isclose(
                event.strong_coupling, coupling, rel_tol=1.0e-14, abs_tol=0.0
            )
            for event in selected.events[1:]
        ):
            raise SelectedMadGraphError(
                "selected events use different strong couplings"
            )
        as_value = f"{_fortran_double(coupling * coupling)}/(4D0*PI)"
    validation_evaluation = (
        "        CALL SMATRIX(P(0,1,E),V(E))"
        if summed
        else "        V(E)=MATRIX(P(0,1,E),HEL,IC)"
    )
    timed_evaluation = (
        (
            "        CALL SMATRIX(P(0,1,1),VALUE)",
            "        CHECKSUM=CHECKSUM+VALUE",
        )
        if summed
        else ("        CHECKSUM=CHECKSUM+MATRIX(P(0,1,1),HEL,IC)",)
    )
    repeated_evaluation = tuple(f"  {line}" for line in timed_evaluation)
    lines.extend(
        [
            f"      TARGET={_fortran_double(TARGET_SECONDS)}",
            f"      ASVALUE={as_value}",
            *(("      USERHEL=-1",) if summed else ()),
            "      CALL SYSTEM_CLOCK(CLOCK0,CLOCKRATE)",
            "      CALL SETPARA('../../Cards/param_card.dat')",
            "      CALL UPDATE_AS_PARAM2(1D0,ASVALUE)",
            "      CALL SYSTEM_CLOCK(CLOCK1)",
            "      WRITE(*,'(A,1X,A)') 'BACKEND',",
            f"     $ '{backend_identity}'",
            "      WRITE(*,'(A,1X,A)') 'HELICITY_EVALUATOR',",
            f"     $ '{evaluator_identity}'",
            "      WRITE(*,'(A,1X,I0)') 'TOTAL_EXTERNAL',NEXTERNAL",
            "      WRITE(*,'(A,1X,ES24.16)') 'INITIALIZATION_SECONDS',",
            "     $ DBLE(CLOCK1-CLOCK0)/DBLE(CLOCKRATE)",
            "      CALL CPU_TIME(FIRST0)",
            "      DO E=1,NEVENTS",
            validation_evaluation,
            "      ENDDO",
            "      CALL CPU_TIME(FIRST1)",
            "      WRITE(*,'(A,1X,ES24.16)') 'FIRST_SAMPLE_PASS_SECONDS',",
            "     $ FIRST1-FIRST0",
            "      DO E=1,NEVENTS",
            "        WRITE(*,'(A,2(1X,I0),1X,ES24.16)')",
            "     $   'MATRIX_ELEMENT',E,1,V(E)",
            "      ENDDO",
            "      WRITE(*,'(A,1X,I0)') 'EVALUATIONS_PER_SWEEP',",
            "     $ 1",
            "      WRITE(*,'(A,1X,I0)') 'HELICITY_COVERAGE_COUNT',",
            f"     $ {helicity_coverage_count}",
            "      NREP=1",
            "      CHECKSUM=0D0",
            " 10   CONTINUE",
            "      CALL CPU_TIME(BEFORE)",
            "      DO R=1,NREP",
            *timed_evaluation,
            "      ENDDO",
            "      CALL CPU_TIME(AFTER)",
            "      ELAPSED=AFTER-BEFORE",
            "      IF (ELAPSED.LT.TARGET) THEN",
            "        IF (NREP.GT.1000000000) STOP 91",
            "        NREP=2*NREP",
            "        GOTO 10",
            "      ENDIF",
            "      WRITE(*,'(A,1X,I0,1X,ES24.16)')",
            "     $ 'EVALUATION_SWEEP_SECONDS',1,ELAPSED/DBLE(NREP)",
            "      WRITE(*,'(A,3(1X,I0),1X,ES24.16)')",
            "     $ 'EVALUATION_CELL_SECONDS',1,1,1,ELAPSED/DBLE(NREP)",
            "      DO B=2,NSAMPLES",
            "        CHECKSUM=0D0",
            "        CALL CPU_TIME(BEFORE)",
            "        DO R=1,NREP",
            *repeated_evaluation,
            "        ENDDO",
            "        CALL CPU_TIME(AFTER)",
            "        ELAPSED=(AFTER-BEFORE)/DBLE(NREP)",
            "        WRITE(*,'(A,1X,I0,1X,ES24.16)')",
            "     $   'EVALUATION_SWEEP_SECONDS',B,ELAPSED",
            "        WRITE(*,'(A,3(1X,I0),1X,ES24.16)')",
            "     $   'EVALUATION_CELL_SECONDS',B,1,1,ELAPSED",
            "      ENDDO",
            "      WRITE(*,'(A,1X,ES24.16)') 'CHECKSUM',CHECKSUM",
            "      END",
        ]
    )
    if any(len(line) > 132 for line in lines):
        raise SelectedMadGraphError(
            "generated fixed-form check source exceeds 132 columns"
        )
    body = "\n".join(lines) + "\n"
    if summed:
        if body.count("CALL SMATRIX(") != 3 or re.search(
            r"(?<!S)\bMATRIX\s*\(", body, re.IGNORECASE
        ):
            raise SelectedMadGraphError(
                "summed check source must call generated SMATRIX only"
            )
    elif re.search(r"\bSMATRIX(?:HEL)?\b", body, re.IGNORECASE):
        raise SelectedMadGraphError("fixed check source must call direct MATRIX only")
    return body


def _json_atomic(
    path: Path, payload: Mapping[str, Any], *, replace: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    if not replace and path.exists():
        temporary.unlink()
        raise SelectedMadGraphError(f"refusing to replace {path}")
    temporary.replace(path)


def _cache_lock_path(cache_dir: Path) -> Path:
    resolved = cache_dir.expanduser().resolve(strict=False)
    digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()
    return CACHE_LOCK_ROOT / f"{digest}.lock"


def _acquire_cache_lock(cache_dir: Path) -> Any:
    """Reject overlapping writers even if the cache tree is replaced."""

    path = _cache_lock_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise SelectedMadGraphError(
            f"another MadGraph profiler holds cache lock {path}"
        ) from error
    return handle


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _generation_worker(arguments: argparse.Namespace) -> int:
    output = arguments.generated_output.resolve(strict=False)
    result_path = arguments.worker_result.resolve(strict=False)
    cell_root = result_path.parent.parent.parent.resolve(strict=False)
    if output.parent != cell_root:
        raise SelectedMadGraphError(
            "generation output is outside its dedicated cell directory"
        )
    if output.exists():
        shutil.rmtree(output)
    disk_free_before = shutil.disk_usage(cell_root).free
    card = result_path.parent / "generate.mg5"
    card.write_text(
        "\n".join(
            (
                "set automatic_html_opening False",
                "import model sm-default",
                f"generate {arguments.process} QED=0",
                f"output standalone {output} -f",
                "",
            )
        ),
        encoding="utf-8",
    )
    stdout_path = result_path.parent / "mg5.stdout.log"
    stderr_path = result_path.parent / "mg5.stderr.log"
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    error_text: str | None = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                [str(arguments.mg5_root / "bin" / "mg5_aMC"), str(card)],
                cwd=arguments.mg5_root,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=arguments.generation_timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process(process)
                returncode = process.returncode
        except OSError as error:
            error_text = str(error)
    elapsed = time.monotonic() - started
    process_dirs = sorted(
        str(path)
        for path in (output / "SubProcesses").glob("P*")
        if (path / "matrix.f").is_file()
    )
    status = "measured" if returncode == 0 and len(process_dirs) == 1 else "failed"
    reason = None
    category = None
    if timed_out:
        category = "generation-time-limit"
        reason = (
            "MadGraph generation exceeded "
            f"{arguments.generation_timeout_seconds:g} seconds"
        )
    elif error_text is not None:
        category = "generation-launch-error"
        reason = error_text
    elif returncode != 0:
        log_tail = ""
        with suppress(OSError):
            log_tail = (
                stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                + stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            )
        if "No space left on device" in log_tail:
            category = "infrastructure-disk-exhaustion"
            reason = "campaign filesystem ran out of space during MadGraph generation"
        else:
            category = "generation-error"
            reason = f"MadGraph exited with status {returncode}"
    elif len(process_dirs) != 1:
        category = "generation-structure-error"
        reason = f"expected one standalone subprocess, found {len(process_dirs)}"
    _json_atomic(
        result_path,
        {
            "status": status,
            "failure_category": category,
            "failure_reason": reason,
            "elapsed_wall_seconds": elapsed,
            "returncode": returncode,
            "timed_out": timed_out,
            "process": arguments.process,
            "output": str(output),
            "process_dirs": process_dirs,
            "card": str(card),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "disk_free_bytes_before": disk_free_before,
            "disk_free_bytes_after": shutil.disk_usage(cell_root).free,
        },
    )
    return 0


def _timeout_worker(arguments: argparse.Namespace) -> int:
    if not math.isfinite(arguments.timeout_seconds) or arguments.timeout_seconds <= 0.0:
        raise SelectedMadGraphError("timeout must be a positive finite number")
    command = list(arguments.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SelectedMadGraphError("timeout worker requires a command")
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except OSError as error:
        raise SelectedMadGraphError(
            f"bounded command could not start: {error}"
        ) from error
    try:
        return process.wait(timeout=arguments.timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process(process)
        print(TIMEOUT_MARKER, file=sys.stderr, flush=True)
        return TIMEOUT_EXIT_CODE


def _run_watchdog(
    command: Sequence[str], *, cwd: Path, report: Path, limit_gib: float
) -> subprocess.CompletedProcess[str]:
    if report.exists():
        raise SelectedMadGraphError(f"watchdog report already exists: {report}")
    return subprocess.run(
        [
            str(sys.executable),
            str(WATCHDOG),
            "--limit-gib",
            format(limit_gib, ".17g"),
            "--report-json",
            str(report),
            "--",
            *command,
        ],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )


def _read_json(path: Path, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectedMadGraphError(f"cannot read {context} {path}: {error}") from error
    return _mapping(value, context=context)


def _new_attempt_directory(cell_dir: Path, stage: str) -> Path:
    if stage not in {"generation", "measurement"}:
        raise SelectedMadGraphError(f"unsupported attempt stage {stage!r}")
    attempts = cell_dir / f"{stage}-attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices = [
        int(match.group(1))
        for path in attempts.iterdir()
        if path.is_dir()
        and (match := re.fullmatch(r"attempt-(\d+)", path.name)) is not None
    ]
    index = max(indices, default=0) + 1
    while True:
        attempt = attempts / f"attempt-{index:03d}"
        try:
            attempt.mkdir()
        except FileExistsError:
            index += 1
            continue
        return attempt


def _prune_cell_generated(cell_dir: Path) -> bool:
    cell_root = cell_dir.resolve(strict=True)
    generated = cell_root / "generated"
    if not os.path.lexists(generated):
        return False
    if generated.is_symlink():
        raise SelectedMadGraphError(
            f"refusing generated-output symlink during cleanup: {generated}"
        )
    target = generated.resolve(strict=True)
    if target != generated or target.parent != cell_root or target.name != "generated":
        raise SelectedMadGraphError(
            f"refusing unexpected generated-output cleanup target: {target}"
        )
    if not target.is_dir():
        raise SelectedMadGraphError(
            f"generated-output cleanup target is not a directory: {target}"
        )
    try:
        shutil.rmtree(target)
    except OSError as error:
        raise SelectedMadGraphError(
            f"cannot prune campaign-owned generated output {target}: {error}"
        ) from error
    return True


def _load_checkpoint_cell(
    checkpoint: Path,
    *,
    identity: Mapping[str, Any],
    cell_dir: Path,
    keep_generated: bool,
) -> dict[str, Any]:
    saved = _read_json(checkpoint, context="MadGraph cell checkpoint")
    saved_identity = saved.get("checkpoint_identity")
    identity_matches = saved_identity == identity
    if not identity_matches and isinstance(saved_identity, Mapping):
        compatible = dict(saved_identity)
        expected = dict(identity)
        if compatible.get("source_cell_sha256") == expected.get(
            "source_cell_sha256"
        ) and isinstance(compatible.get("source_cell_sha256"), str):
            # Expanding a sparse source report does not change this cell's inputs.
            compatible.pop("source_report_sha256", None)
            expected.pop("source_report_sha256", None)
        identity_matches = compatible == expected
    if not identity_matches:
        raise SelectedMadGraphError(
            f"checkpoint identity changed at {cell_dir.parent.name}/{cell_dir.name}; "
            "use a fresh cache directory"
        )
    cell = dict(_mapping(saved.get("cell"), context="checkpoint.cell"))
    status = cell.get("status")
    if status == "failed":
        category = str(cell.get("failure_category"))
        if category not in RESOURCE_FRONTIER_CATEGORIES:
            raise SelectedMadGraphError(
                f"checkpoint contains a non-resource failed frontier: {category}"
            )
    elif status != "measured":
        raise SelectedMadGraphError(
            f"checkpoint contains unsupported cell status {status!r}"
        )
    if not keep_generated:
        _prune_cell_generated(cell_dir)
    return cell


def _generation_attempt(
    *,
    selected: SelectedCell,
    cell_dir: Path,
    mg5_root: Path,
    timeout: float,
    limit_gib: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    attempt = _new_attempt_directory(cell_dir, "generation")
    worker_result = attempt / "result.json"
    watchdog_report = attempt / "watchdog.json"
    command = [
        str(Path(__file__).resolve()),
        "_generate-cell",
        "--mg5-root",
        str(mg5_root),
        "--generated-output",
        str(cell_dir / "generated"),
        "--worker-result",
        str(worker_result),
        "--process",
        selected.process,
        "--generation-timeout-seconds",
        format(timeout, ".17g"),
    ]
    completed = _run_watchdog(
        command, cwd=ROOT, report=watchdog_report, limit_gib=limit_gib
    )
    watchdog = _read_json(watchdog_report, context="generation watchdog")
    if worker_result.is_file():
        generation = _read_json(worker_result, context="generation worker result")
    else:
        enforcement = _mapping(
            watchdog.get("enforcement"), context="watchdog.enforcement"
        )
        execution = _mapping(watchdog.get("execution"), context="watchdog.execution")
        generation = {
            "status": "failed",
            "failure_category": (
                "infrastructure-disk-exhaustion"
                if _is_disk_exhaustion(completed.stderr)
                else (
                    "generation-memory-limit"
                    if execution.get("outcome") == "memory-limit-exceeded"
                    else "generation-watchdog-error"
                )
            ),
            "failure_reason": (
                "campaign filesystem ran out of space during MadGraph generation"
                if _is_disk_exhaustion(completed.stderr)
                else (
                    "MadGraph generation reached the "
                    f"{limit_gib:g}-GiB process-tree memory limit"
                    if execution.get("outcome") == "memory-limit-exceeded"
                    else f"generation watchdog failed: {execution.get('outcome')}"
                )
            ),
            "elapsed_wall_seconds": execution.get("elapsed_wall_seconds"),
            "returncode": completed.returncode,
            "process_dirs": [],
            "output": str(cell_dir / "generated"),
            "watchdog_peak_guard_bytes": enforcement.get("peak_guard_bytes"),
        }
    return generation, watchdog


def _extract_integer(source: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*(\d+)\b", source, re.IGNORECASE)
    if match is None or int(match.group(1)) < 1:
        raise SelectedMadGraphError(f"generated MATRIX source lacks positive {name}")
    return int(match.group(1))


def _expected_smatrix_iden(family: str, final_multiplicity: int) -> int:
    if family == "gg":
        return 256 * math.factorial(final_multiplicity)
    if family == "ddbar":
        return 36 * math.factorial(final_multiplicity - 2)
    raise SelectedMadGraphError(f"unsupported family {family!r}")


def _validate_matrix_source(
    path: Path,
    *,
    family: str,
    final_multiplicity: int,
) -> dict[str, Any]:
    expected_external = final_multiplicity + 2
    text = path.read_text(encoding="utf-8", errors="strict")
    match = re.search(r"(?ims)^\s*REAL\*8 FUNCTION MATRIX\b(.*?)(?=^\s*END\s*$)", text)
    if match is None:
        raise SelectedMadGraphError("generated source has no direct MATRIX function")
    body = match.group(1)
    if re.search(r"\bSMATRIX(?:HEL)?\b|\bIHEL\b|\bNCOMB\b", body, re.IGNORECASE):
        raise SelectedMadGraphError("generated MATRIX is not a direct-helicity kernel")
    if not re.search(r"INTEGER\s+NHEL\s*\(\s*NEXTERNAL\s*\)", body, re.IGNORECASE):
        raise SelectedMadGraphError("generated MATRIX lacks its NHEL vector")
    n_external = _extract_integer(text, "NEXTERNAL")
    if n_external != expected_external:
        raise SelectedMadGraphError(
            "generated MATRIX has the wrong external multiplicity"
        )
    smatrix = re.search(r"(?ims)^\s*SUBROUTINE SMATRIX\b(.*?)(?=^\s*END\s*$)", text)
    if smatrix is None or not re.search(
        r"\bNCOMB\b|\bNHEL\b", smatrix.group(1), re.IGNORECASE
    ):
        raise SelectedMadGraphError(
            "generated standalone source is not helicity-general"
        )
    iden_matches = re.findall(
        r"^[ \t]{6,}DATA[ \t]+IDEN[ \t]*/[ \t]*(\d+)[ \t]*/[ \t]*$",
        smatrix.group(1),
        re.IGNORECASE | re.MULTILINE,
    )
    if len(iden_matches) != 1 or int(iden_matches[0]) < 1:
        raise SelectedMadGraphError(
            "generated SMATRIX lacks one positive IDEN denominator"
        )
    iden_uses = re.findall(
        r"^[ \t]{6,}ANS[ \t]*=[ \t]*ANS[ \t]*/[ \t]*"
        r"DBLE[ \t]*\([ \t]*IDEN[ \t]*\)[ \t]*$",
        smatrix.group(1),
        re.IGNORECASE | re.MULTILINE,
    )
    if len(iden_uses) != 1:
        raise SelectedMadGraphError(
            "generated SMATRIX does not apply its IDEN denominator exactly once"
        )
    smatrix_iden = int(iden_matches[0])
    expected_iden = _expected_smatrix_iden(family, final_multiplicity)
    if smatrix_iden != expected_iden:
        raise SelectedMadGraphError(
            "generated SMATRIX has the wrong IDEN denominator: "
            f"expected {expected_iden}, observed {smatrix_iden}"
        )
    return {
        "matrix_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "generated_matrix_graphs": _extract_integer(body, "NGRAPHS"),
        "colour_flows": _extract_integer(body, "NCOLOR"),
        "smatrix_iden": smatrix_iden,
        "generation_helicity_coverage": "all",
    }


def _parse_time_max_rss(stderr: str) -> int | None:
    system = platform.system()
    if system == "Darwin":
        matches = re.findall(
            r"^\s*(\d+)\s+maximum resident set size\s*$", stderr, re.MULTILINE
        )
        if len(matches) != 1:
            return None
        return math.ceil(int(matches[0]) / 1024)
    if system == "Linux":
        matches = re.findall(
            r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$",
            stderr,
            re.MULTILINE,
        )
        if len(matches) != 1:
            return None
        return int(matches[0])
    return None


def _external_time_command(command: Sequence[str]) -> list[str]:
    """Add the host's external time utility when its dialect is known."""

    system = platform.system()
    if system == "Darwin":
        time_path = Path("/usr/bin/time")
        prefix = [str(time_path), "-l"] if time_path.is_file() else []
    elif system == "Linux":
        located = shutil.which("time")
        prefix = [located, "-v"] if located is not None else []
    else:
        prefix = []
    return [*prefix, *command]


def _watchdog_resource(report: Mapping[str, Any]) -> dict[str, Any]:
    enforcement = _mapping(report.get("enforcement"), context="watchdog.enforcement")
    execution = _mapping(report.get("execution"), context="watchdog.execution")
    peak_guard = int(enforcement.get("peak_guard_bytes", 0))
    return {
        "elapsed_wall_seconds": _finite_number(
            execution.get("elapsed_wall_seconds"),
            context="watchdog elapsed",
            nonnegative=True,
        ),
        "peak_guard_bytes": peak_guard,
        "peak_guard_kib": math.ceil(peak_guard / 1024),
        "peak_rss_bytes": int(enforcement.get("peak_rss_bytes", 0)),
        "peak_physical_footprint_bytes": enforcement.get(
            "peak_physical_footprint_bytes"
        ),
        "metric": enforcement.get("metric"),
        "limit_bytes": enforcement.get("limit_bytes"),
        "outcome": execution.get("outcome"),
        "passes": report.get("passes") is True,
    }


def _run_bounded_command(
    command: Sequence[str],
    *,
    cwd: Path,
    report: Path,
    limit_gib: float,
    timeout_seconds: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], Mapping[str, Any]]:
    bounded = list(command)
    if timeout_seconds is not None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise SelectedMadGraphError("cold-to-ready time budget is exhausted")
        bounded = [
            str(Path(__file__).resolve()),
            "_run-with-timeout",
            "--timeout-seconds",
            format(timeout_seconds, ".17g"),
            "--",
            *bounded,
        ]
    completed = _run_watchdog(
        _external_time_command(bounded),
        cwd=cwd,
        report=report,
        limit_gib=limit_gib,
    )
    report.with_name(f"{report.stem}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    report.with_name(f"{report.stem}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    return completed, _read_json(report, context="bounded command watchdog")


def _bounded_command_timed_out(
    completed: subprocess.CompletedProcess[str],
) -> bool:
    return (
        completed.returncode == TIMEOUT_EXIT_CODE and TIMEOUT_MARKER in completed.stderr
    )


def _remaining_cold_to_ready_budget(
    limit_seconds: float, *resources: Mapping[str, Any]
) -> float:
    used = sum(float(resource["elapsed_wall_seconds"]) for resource in resources)
    remaining = limit_seconds - used
    if not math.isfinite(remaining) or remaining <= 0.0:
        raise ResourceFrontierError(
            "generation-time-limit",
            "cold-to-ready generation reached the strict time limit",
        )
    return remaining


def _relative_error(observed: float, reference: float) -> float:
    if observed == reference:
        return 0.0
    return abs(observed - reference) / max(abs(observed), abs(reference), 1.0e-300)


def _numerical_normalization_factor(
    family: str, *, smatrix_iden: int, helicity_workload: str = "fixed"
) -> float:
    if family not in FAMILIES:
        raise SelectedMadGraphError(f"unsupported family {family!r}")
    if smatrix_iden < 1:
        raise SelectedMadGraphError("SMATRIX IDEN denominator must be positive")
    _workload_label(helicity_workload)
    if helicity_workload == "sum":
        return 1.0
    return 1.0 if family == "gg" else 1.0 / float(smatrix_iden)


def _is_disk_exhaustion(error: BaseException | str) -> bool:
    text = str(error)
    return "No space left on device" in text or "[Errno 28]" in text


def _directory_size(path: Path) -> int:
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def _measure_generated_cell(
    *,
    source: SourceSelection,
    selected: SelectedCell,
    generation: Mapping[str, Any],
    generation_watchdog: Mapping[str, Any],
    cell_dir: Path,
    fc: str,
    fflags: str,
    limit_gib: float,
    mg5_root: Path,
    cold_to_ready_limit_seconds: float,
) -> dict[str, Any]:
    process_dirs = generation.get("process_dirs")
    if (
        not isinstance(process_dirs, Sequence)
        or isinstance(process_dirs, str)
        or len(process_dirs) != 1
    ):
        raise SelectedMadGraphError(
            "generation result has the wrong subprocess inventory"
        )
    generated_root = Path(str(generation.get("output"))).resolve(strict=True)
    process_dir = Path(str(process_dirs[0])).resolve(strict=True)
    matrix_source = process_dir / "matrix.f"
    matrix = _validate_matrix_source(
        matrix_source,
        family=selected.family,
        final_multiplicity=selected.n,
    )
    generation_resource = _watchdog_resource(generation_watchdog)
    if not generation_resource["passes"]:
        if generation_resource["outcome"] == "memory-limit-exceeded":
            raise ResourceFrontierError(
                "memory-limit",
                f"MadGraph generation reached the strict {limit_gib:g}-GiB "
                "memory limit",
            )
        raise SelectedMadGraphError(
            "generation memory enforcement did not pass: "
            f"{generation_resource['outcome']}"
        )
    build_budget_seconds = _remaining_cold_to_ready_budget(
        cold_to_ready_limit_seconds, generation_resource
    )
    check_source = process_dir.parent / "check_sa.f"
    check_body = render_check_source(selected)
    check_source.write_text(check_body, encoding="ascii")

    makefile = process_dir / "makefile"
    if not makefile.exists():
        raise SelectedMadGraphError("generated subprocess has no makefile")
    measurement_attempt = _new_attempt_directory(cell_dir, "measurement")
    build_report = measurement_attempt / "build-watchdog.json"
    retained = measurement_attempt / "retained"
    retained.mkdir()
    retained_matrix = retained / "matrix.f"
    retained_check = retained / "check_sa.f"
    retained_executable = retained / "check"
    shutil.copy2(matrix_source, retained_matrix)
    shutil.copy2(check_source, retained_check)
    build_command = ["make", "check", f"FC={fc}", f"FFLAGS={fflags}"]
    build, build_watchdog = _run_bounded_command(
        build_command,
        cwd=process_dir,
        report=build_report,
        limit_gib=limit_gib,
        timeout_seconds=build_budget_seconds,
    )
    build_resource = _watchdog_resource(build_watchdog)
    if build.returncode != 0:
        detail = (build.stdout + "\n" + build.stderr)[-8000:]
        if build_resource["outcome"] == "memory-limit-exceeded":
            raise ResourceFrontierError(
                "memory-limit",
                f"cold-to-ready generation reached the strict {limit_gib:g}-GiB "
                "memory limit",
            )
        if _bounded_command_timed_out(build):
            raise ResourceFrontierError(
                "generation-time-limit",
                "cold-to-ready generation reached the strict time limit",
            )
        raise SelectedMadGraphError(f"generated standalone build failed:\n{detail}")
    if not build_resource["passes"]:
        raise SelectedMadGraphError(
            f"build memory enforcement did not pass: {build_resource['outcome']}"
        )
    executable = process_dir / "check"
    if not executable.is_file():
        raise SelectedMadGraphError(
            "generated standalone build produced no check executable"
        )
    generated_size_bytes = _directory_size(generated_root)

    shutil.copy2(executable, retained_executable)

    runtime_budget_seconds = _remaining_cold_to_ready_budget(
        cold_to_ready_limit_seconds, generation_resource, build_resource
    )
    runtime_report = measurement_attempt / "runtime-watchdog.json"
    runtime, runtime_watchdog = _run_bounded_command(
        [str(executable)],
        cwd=process_dir,
        report=runtime_report,
        limit_gib=limit_gib,
        timeout_seconds=runtime_budget_seconds,
    )
    runtime_resource = _watchdog_resource(runtime_watchdog)
    if runtime.returncode != 0:
        detail = (runtime.stdout + "\n" + runtime.stderr)[-8000:]
        if runtime_resource["outcome"] == "memory-limit-exceeded":
            raise ResourceFrontierError(
                "memory-limit",
                f"initialized runtime reached the strict {limit_gib:g}-GiB "
                "memory limit",
            )
        if _bounded_command_timed_out(runtime):
            raise ResourceFrontierError(
                "generation-time-limit",
                "initialization/check execution exhausted the remaining "
                "cold-to-ready time budget",
            )
        raise SelectedMadGraphError(f"generated standalone runtime failed:\n{detail}")
    if not runtime_resource["passes"]:
        raise SelectedMadGraphError(
            f"runtime memory enforcement did not pass: {runtime_resource['outcome']}"
        )
    parsed = parse_driver_output(
        runtime.stdout,
        batches=TIMING_DRIVER_BATCH_COUNT,
        expected_total_external=selected.total_external,
        expected_events=POINT_COUNT,
        helicity_workload=selected.helicity_workload,
    )
    # Fixed MATRIX is deliberately unaveraged and retains the historical
    # family-specific reference convention.  Summed mode calls SMATRIX, which
    # already applies IDEN exactly once, so its normalized source points need
    # no further factor.
    normalization_factor = _numerical_normalization_factor(
        selected.family,
        smatrix_iden=int(matrix["smatrix_iden"]),
        helicity_workload=selected.helicity_workload,
    )
    relative_errors = tuple(
        _relative_error(observed * normalization_factor, reference)
        for observed, reference in zip(
            parsed.point_values, selected.reference_point_values, strict=True
        )
    )
    maximum_error = max(relative_errors)
    if maximum_error > RELATIVE_TOLERANCE:
        raise SelectedMadGraphError(
            f"MadGraph points disagree with {selected.source_mode} "
            f"(maximum relative error {maximum_error:.6e})"
        )
    calibration = parsed.cell_seconds[0][0]
    warm_samples = tuple(row[0] for row in parsed.cell_seconds[1:])
    if len(warm_samples) != WARM_SAMPLE_COUNT or any(
        value <= 0.0 for value in warm_samples
    ):
        raise SelectedMadGraphError(
            "MadGraph did not provide ten positive warm samples"
        )
    warm = statistics.median(warm_samples)
    cold_to_ready_seconds = (
        generation_resource["elapsed_wall_seconds"]
        + build_resource["elapsed_wall_seconds"]
        + parsed.initialization_seconds
    )
    cold_to_ready_peak_kib = max(
        generation_resource["peak_guard_kib"],
        build_resource["peak_guard_kib"],
        runtime_resource["peak_guard_kib"],
    )
    if cold_to_ready_seconds >= cold_to_ready_limit_seconds:
        raise ResourceFrontierError(
            "generation-time-limit",
            "cold-to-ready generation reached the strict time limit",
        )
    runtime_self_rss = _parse_time_max_rss(runtime.stderr)
    event_paths = [str(event.path) for event in selected.events]
    event_hashes = [event.sha256 for event in selected.events]
    mg_version = (mg5_root / "VERSION").read_text(encoding="utf-8").strip()
    summed = selected.helicity_workload == "sum"
    label = _workload_label(selected.helicity_workload)
    timed_helicity_count = parsed.helicity_coverage_count
    return {
        "status": "measured",
        "family": selected.family,
        "mode": MODE,
        "label": label,
        "n": selected.n,
        "total_external": selected.total_external,
        "process": selected.process,
        "alpha_s": (
            selected.source_alpha_s
            if summed
            else selected.events[0].strong_coupling ** 2 / (4.0 * math.pi)
        ),
        "helicity": list(selected.helicity),
        "helicity_workload": selected.helicity_workload,
        "warm_fixed_helicity": not summed,
        "warm_helicity_sum": summed,
        "timed_helicity_count": timed_helicity_count,
        "event_paths": event_paths,
        "event_sha256": event_hashes,
        "point_values": list(parsed.point_values),
        "matrix_element": parsed.point_values[0],
        "generation_seconds": cold_to_ready_seconds,
        "warm_seconds_per_point": warm,
        "max_rss_kib": cold_to_ready_peak_kib,
        "warm_samples_seconds": list(warm_samples),
        "metrics": {
            "generation_seconds": cold_to_ready_seconds,
            "warm_seconds_per_point": warm,
            "max_rss_kib": cold_to_ready_peak_kib,
        },
        "probe": {
            "timer_source": "process-cpu-time",
            "warm_median_seconds": warm,
            "warm_samples_seconds": list(warm_samples),
            "warm_sample_count": WARM_SAMPLE_COUNT,
            "calibration_target_seconds": TARGET_SECONDS,
            "calibration_sample_seconds": calibration,
            "published_driver_batches": list(range(2, TIMING_DRIVER_BATCH_COUNT + 1)),
            "timed_event_index": 1,
            "validation_event_count": POINT_COUNT,
            "validation_first_pass_seconds": parsed.first_pass_seconds,
            "runtime_self_max_rss_kib": runtime_self_rss,
            "runtime_process_tree_peak_guard_kib": runtime_resource["peak_guard_kib"],
        },
        "numerical": {
            "reference_mode": selected.source_mode,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "normalization_factor_reference_per_madgraph": normalization_factor,
            "normalization_contract": (
                (
                    "generated SMATRIX applies its IDEN denominator exactly once; "
                    "no additional normalization"
                )
                if summed
                else "identity"
                if selected.family == "gg"
                else (
                    "generated SMATRIX IDEN denominator applied to unaveraged "
                    "direct MATRIX; includes incoming spin/colour averaging "
                    "and identical-final-state symmetry"
                )
            ),
            "maximum_relative_error": maximum_error,
            "relative_errors": list(relative_errors),
            "passes": True,
        },
        "protocol": {
            "evaluator": (
                "SMATRIX(P,ANS)-generated-complete-helicity-sum"
                if summed
                else "MATRIX(P,NHEL,IC)-direct"
            ),
            "generation_helicity_coverage": "all",
            "helicity_workload": selected.helicity_workload,
            "warm_fixed_helicity": not summed,
            "warm_helicity_sum": summed,
            "helicity_summed": summed,
            "timed_helicity_count": timed_helicity_count,
            "helicity_sum_implementation": (
                "generated-SMATRIX-with-USERHEL-minus-one"
                if summed
                else "not-applicable"
            ),
            "warmed_native_call_pruning": (
                "generated GOODHEL cache may skip structurally zero helicities"
                if summed
                else "not-applicable"
            ),
            "color_sum": (
                "generated-SMATRIX-summed-and-averaged"
                if summed
                else "full-unaveraged"
            ),
            "incoming_color_average": summed,
            "incoming_helicity_average": summed,
            "final_state_symmetry_factor": summed,
            "timer_source": "process-cpu-time",
            "batch_size": 1,
            "warm_sample_count": WARM_SAMPLE_COUNT,
            "validated_point_count": POINT_COUNT,
            "timed_point_index": 1,
            "calibration_target_seconds": TARGET_SECONDS,
            "fixed_repetitions_after_calibration": True,
            "calibration_sample_included": False,
            "initialization_included": False,
            "first_pass_included": False,
            "build_included": False,
        },
        "runtime_refresh": {
            "accepted": True,
            "fresh_process": True,
            "scope": "generation-resource-and-warm-runtime",
        },
        "provenance": {
            "source_report": {
                "path": _display_path(source.path),
                "sha256": source.sha256,
                "cell": (
                    f"cells.{selected.family}.{selected.source_mode}.{selected.n}"
                ),
            },
            "source_alpha_s": source.alpha_s,
            "runtime_alpha_s": (
                source.alpha_s
                if summed
                else selected.events[0].strong_coupling ** 2 / (4.0 * math.pi)
            ),
            "madgraph_root": str(mg5_root),
            "madgraph_version": mg_version,
            "matrix_sha256": matrix["matrix_sha256"],
            "generated_matrix_graphs": matrix["generated_matrix_graphs"],
            "colour_flows": matrix["colour_flows"],
            "smatrix_iden": matrix["smatrix_iden"],
            "generation_helicity_coverage": "all",
            "helicity_workload": selected.helicity_workload,
            "warm_fixed_helicity": not summed,
            "warm_helicity_sum": summed,
            "timed_helicity_count": timed_helicity_count,
            "compiler": fc,
            "fflags": fflags,
            "check_source_sha256": hashlib.sha256(
                check_body.encode("ascii")
            ).hexdigest(),
            "generated_output_size_bytes": generated_size_bytes,
            "cold_to_ready": {
                "elapsed_wall_seconds": cold_to_ready_seconds,
                "peak_guard_kib": cold_to_ready_peak_kib,
                "included_stages": [
                    "mg5-output",
                    "standalone-check-build",
                    "required-model-initialization",
                ],
                "initialization_seconds": parsed.initialization_seconds,
            },
            "measurement_attempt": str(measurement_attempt),
            "build_watchdog_report": str(build_report),
            "build_stdout_log": str(
                build_report.with_name(f"{build_report.stem}.stdout.log")
            ),
            "build_stderr_log": str(
                build_report.with_name(f"{build_report.stem}.stderr.log")
            ),
            "runtime_watchdog_report": str(runtime_report),
            "runtime_stdout_log": str(
                runtime_report.with_name(f"{runtime_report.stem}.stdout.log")
            ),
            "runtime_stderr_log": str(
                runtime_report.with_name(f"{runtime_report.stem}.stderr.log")
            ),
            "retained": {
                "matrix_source": str(retained_matrix),
                "matrix_sha256": _sha256(retained_matrix),
                "check_source": str(retained_check),
                "check_source_sha256": _sha256(retained_check),
                "executable": str(retained_executable),
                "executable_sha256": _sha256(retained_executable),
            },
            "generation": generation_resource,
            "build": build_resource,
            "runtime": runtime_resource,
            "runtime_self_max_rss_kib": runtime_self_rss,
        },
    }


def _failure_cell(selected: SelectedCell, category: str, reason: str) -> dict[str, Any]:
    if category not in RESOURCE_FRONTIER_CATEGORIES:
        raise SelectedMadGraphError(
            f"refusing non-resource failure as a physics frontier: {category}"
        )
    return {
        "status": "failed",
        "family": selected.family,
        "mode": MODE,
        "label": _workload_label(selected.helicity_workload),
        "n": selected.n,
        "total_external": selected.total_external,
        "process": selected.process,
        "helicity_workload": selected.helicity_workload,
        "failure_category": category,
        "failure_reason": reason,
        "censors_higher_multiplicities": True,
        "availability_frontier_n": selected.n - 1,
    }


def _dependency_failure_cell(
    family: str, n: int, reason: str, *, helicity_workload: str = "fixed"
) -> dict[str, Any]:
    return {
        "status": "failed",
        "family": family,
        "mode": MODE,
        "label": _workload_label(helicity_workload),
        "n": n,
        "total_external": n + 2,
        "process": process_expression(family, n),
        "helicity_workload": helicity_workload,
        "failure_category": "dependency-unavailable",
        "failure_reason": (
            f"source {source_mode(family, helicity_workload)} cell is unavailable: "
            f"{reason}"
        ),
        "censors_higher_multiplicities": True,
        "availability_frontier_n": n - 1,
    }


def _frontier_cell(
    family: str,
    n: int,
    *,
    failure_n: int,
    reason: str,
    helicity_workload: str = "fixed",
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "family": family,
        "mode": MODE,
        "label": _workload_label(helicity_workload),
        "n": n,
        "total_external": n + 2,
        "process": process_expression(family, n),
        "helicity_workload": helicity_workload,
        "failure_category": "skipped-after-frontier",
        "failure_reason": reason,
        "censors_higher_multiplicities": True,
        "availability_frontier_n": failure_n - 1,
        "skipped_after_frontier": True,
    }


def _diagnostic_skip_cell(
    family: str, n: int, reason: str, *, helicity_workload: str = "fixed"
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "family": family,
        "mode": MODE,
        "label": _workload_label(helicity_workload),
        "n": n,
        "total_external": n + 2,
        "process": process_expression(family, n),
        "helicity_workload": helicity_workload,
        "failure_category": "diagnostic-selection-omitted",
        "failure_reason": reason,
        "censors_higher_multiplicities": False,
        "skipped_after_frontier": False,
    }


def _protocol_scope_cell(
    family: str, n: int, *, helicity_workload: str = "fixed"
) -> dict[str, Any]:
    if n <= MAX_PROTOCOL_MEASURED_MULTIPLICITY:
        raise SelectedMadGraphError(
            f"MadGraph protocol scope does not apply at n={n}"
        )
    return {
        "status": "not-applicable",
        "family": family,
        "mode": MODE,
        "label": _workload_label(helicity_workload),
        "n": n,
        "total_external": n + 2,
        "process": process_expression(family, n),
        "helicity_workload": helicity_workload,
        "failure_category": "protocol-scope-n>6",
        "failure_reason": (
            "final-plot protocol measures MadGraph only through n=6; higher "
            "candidate and reference frontiers are profiled independently"
        ),
        "censors_higher_multiplicities": False,
        "protocol_scope": {
            "maximum_measured_multiplicity": MAX_PROTOCOL_MEASURED_MULTIPLICITY,
            "disposition": "not-applicable",
        },
    }


def _checkpoint_identity(
    source: SourceSelection,
    selected: SelectedCell,
    mg5_root: Path,
    fc: str,
    fflags: str,
    timeout_seconds: float,
    memory_limit_gib: float,
) -> dict[str, Any]:
    source_cell_payload = {
        "schema_version": 1,
        "alpha_s": source.alpha_s,
        "family": selected.family,
        "n": selected.n,
        "process": selected.process,
        "source_mode": selected.source_mode,
        "helicity_workload": selected.helicity_workload,
        "helicity": list(selected.helicity),
        "event_sha256": [event.sha256 for event in selected.events],
        "reference_point_values": list(selected.reference_point_values),
    }
    source_cell_sha256 = hashlib.sha256(
        json.dumps(
            source_cell_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "source_report_sha256": source.sha256,
        "source_cell_sha256": source_cell_sha256,
        "family": selected.family,
        "n": selected.n,
        "process": selected.process,
        "helicity_workload": selected.helicity_workload,
        "helicity": list(selected.helicity),
        "event_sha256": [event.sha256 for event in selected.events],
        "mg5_root": str(mg5_root),
        "mg5_version_sha256": _sha256(mg5_root / "VERSION"),
        "fc": fc,
        "fflags": fflags,
        "generation_timeout_seconds": timeout_seconds,
        "memory_limit_gib": memory_limit_gib,
    }


def _rebind_source_provenance(
    cell: Mapping[str, Any], *, source: SourceSelection, selected: SelectedCell
) -> tuple[dict[str, Any], bool]:
    copied = dict(cell)
    if copied.get("status") != "measured":
        return copied, False
    raw_provenance = _mapping(
        copied.get("provenance"), context="checkpoint cell provenance"
    )
    provenance = dict(raw_provenance)
    expected_source = {
        "path": _display_path(source.path),
        "sha256": source.sha256,
        "cell": f"cells.{selected.family}.{selected.source_mode}.{selected.n}",
    }
    changed = provenance.get("source_report") != expected_source
    provenance["source_report"] = expected_source
    copied["provenance"] = provenance
    return copied, changed


def _runtime_policy(
    multiplicities: Sequence[int],
    *,
    timeout_seconds: float,
    memory_limit_gib: float,
    helicity_workload: str = "fixed",
) -> dict[str, Any]:
    summed = helicity_workload == "sum"
    _workload_label(helicity_workload)
    return {
        "final_state_multiplicities": list(multiplicities),
        "process_families": list(FAMILIES),
        "point_validation_count": POINT_COUNT,
        "warm_timed_point_index": 1,
        "warm_sample_count": WARM_SAMPLE_COUNT,
        "calibration_target_seconds": TARGET_SECONDS,
        "generation_timeout_seconds": timeout_seconds,
        "outer_memory_watchdog_gib": memory_limit_gib,
        "watchdog_enforced_per_generation_build_and_runtime": True,
        "standalone_source": "fresh helicity-general MadGraph standalone",
        "evaluator": (
            "generated SMATRIX(P,ANS) with USERHEL=-1 and warmed GOODHEL pruning"
            if summed
            else "direct MATRIX(P,NHEL,IC)"
        ),
        "generation_helicity_coverage": "all",
        "helicity_workload": helicity_workload,
        "warm_fixed_helicity": not summed,
        "warm_helicity_sum": summed,
        "maximum_measured_multiplicity": MAX_PROTOCOL_MEASURED_MULTIPLICITY,
        "higher_multiplicity_policy": "not-applicable-protocol-scope",
        "metric_scope": {
            "generation_seconds": (
                "cold-to-ready MG5 generation, check build, and required "
                "model initialization"
            ),
            "max_rss_kib": (
                "maximum conservative process-tree peak guard across "
                "generation, build, and initialized runtime"
            ),
            "warm_seconds_per_point": (
                "median of ten post-calibration point-01 CPU samples of one "
                + (
                    "complete physical-helicity sum"
                    if summed
                    else "fixed-helicity evaluation"
                )
            ),
        },
    }


def _progress_cell_identity(
    family: str, n: int, *, helicity_workload: str = "fixed"
) -> dict[str, Any]:
    return {
        "family": family,
        "mode": MODE,
        "n": n,
        "total_external": n + 2,
        "process": process_expression(family, n),
        "helicity_workload": helicity_workload,
    }


def _runtime_progress_report(
    *,
    source: SourceSelection,
    runtime_series: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    selected_multiplicities: Sequence[int],
    selected_families: Sequence[str],
    timeout_seconds: float,
    memory_limit_gib: float,
    started_at_utc: str,
    current_cell: Mapping[str, Any] | None,
) -> dict[str, Any]:
    counts = Counter(
        str(cell.get("status"))
        for family_series in runtime_series.values()
        for mode_cells in family_series.values()
        for cell in mode_cells.values()
    )
    completed = sum(counts.values())
    total = len(FAMILIES) * len(selected_multiplicities)
    pending = [
        _progress_cell_identity(
            family, n, helicity_workload=source.helicity_workload
        )
        for family in FAMILIES
        for n in selected_multiplicities
        if str(n) not in runtime_series[family][MODE]
    ]
    if completed == total:
        status = "complete-with-failures" if counts["failed"] else "complete"
        current: Mapping[str, Any] | None = None
    else:
        status = "running"
        current = current_cell
    return {
        "kind": PROGRESS_KIND,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at_utc,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "source_report": {
            "path": _display_path(source.path),
            "sha256": source.sha256,
        },
        "producer": {
            "path": _display_path(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "policy": _runtime_policy(
            selected_multiplicities,
            timeout_seconds=timeout_seconds,
            memory_limit_gib=memory_limit_gib,
            helicity_workload=source.helicity_workload,
        )
        | {"selected_process_families": list(selected_families)},
        "current_cell": dict(current) if current is not None else None,
        "pending_cells": pending,
        "summary": {
            "completed_cell_count": completed,
            "pending_cell_count": len(pending),
            "total_cell_count": total,
            "runtime_series_status_counts": dict(sorted(counts.items())),
        },
        "runtime_series": runtime_series,
    }


def load_runtime_progress(path: Path) -> dict[str, Any]:
    """Read and authenticate one atomically published sparse progress report."""

    payload = _read_json(path, context="MadGraph runtime progress")
    if (
        payload.get("kind") != PROGRESS_KIND
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise SelectedMadGraphError("MadGraph runtime progress has the wrong schema")
    status = payload.get("status")
    if status not in {"running", "complete", "complete-with-failures"}:
        raise SelectedMadGraphError("MadGraph runtime progress has an invalid status")
    source = _mapping(
        payload.get("source_report"), context="MadGraph runtime progress source"
    )
    source_path = source.get("path")
    source_sha256 = source.get("sha256")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise SelectedMadGraphError(
            "MadGraph runtime progress has invalid source-report provenance"
        )
    policy = _mapping(
        payload.get("policy"), context="MadGraph runtime progress policy"
    )
    helicity_workload = _declared_helicity_workload(policy)
    raw_multiplicities = policy.get("final_state_multiplicities")
    if (
        not isinstance(raw_multiplicities, Sequence)
        or isinstance(raw_multiplicities, str)
        or not raw_multiplicities
        or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 2
            for n in raw_multiplicities
        )
    ):
        raise SelectedMadGraphError(
            "MadGraph runtime progress has invalid multiplicities"
        )
    multiplicities = tuple(int(n) for n in raw_multiplicities)
    if tuple(sorted(set(multiplicities))) != multiplicities:
        raise SelectedMadGraphError(
            "MadGraph runtime progress multiplicities are not sorted and unique"
        )
    if policy.get("process_families") != list(FAMILIES):
        raise SelectedMadGraphError(
            "MadGraph runtime progress has invalid process families"
        )
    if (
        policy.get("maximum_measured_multiplicity")
        != MAX_PROTOCOL_MEASURED_MULTIPLICITY
        or policy.get("higher_multiplicity_policy")
        != "not-applicable-protocol-scope"
    ):
        raise SelectedMadGraphError(
            "MadGraph runtime progress has invalid multiplicity scope"
        )
    raw_series = _mapping(
        payload.get("runtime_series"), context="MadGraph runtime progress series"
    )
    if set(raw_series) != set(FAMILIES):
        raise SelectedMadGraphError(
            "MadGraph runtime progress series has invalid process families"
        )
    completed_identities: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    for family in FAMILIES:
        family_series = _mapping(
            raw_series.get(family),
            context=f"MadGraph runtime progress series.{family}",
        )
        if set(family_series) != {MODE}:
            raise SelectedMadGraphError(
                f"MadGraph runtime progress series.{family} has invalid modes"
            )
        mode_cells = _mapping(
            family_series.get(MODE),
            context=f"MadGraph runtime progress series.{family}.{MODE}",
        )
        for raw_n, raw_cell in mode_cells.items():
            try:
                n = int(raw_n)
            except (TypeError, ValueError) as error:
                raise SelectedMadGraphError(
                    "MadGraph runtime progress has an invalid cell key"
                ) from error
            if str(n) != raw_n or n not in multiplicities:
                raise SelectedMadGraphError(
                    "MadGraph runtime progress has an out-of-policy cell"
                )
            cell = _mapping(
                raw_cell,
                context=f"MadGraph runtime progress series.{family}.{MODE}.{n}",
            )
            expected = _progress_cell_identity(
                family, n, helicity_workload=helicity_workload
            )
            if any(cell.get(key) != value for key, value in expected.items()):
                raise SelectedMadGraphError(
                    "MadGraph runtime progress cell identity is invalid"
                )
            cell_status = cell.get("status")
            if cell_status not in {
                "measured",
                "failed",
                "skipped",
                "not-applicable",
            }:
                raise SelectedMadGraphError(
                    "MadGraph runtime progress cell status is invalid"
                )
            if (cell_status == "not-applicable") != (
                n > MAX_PROTOCOL_MEASURED_MULTIPLICITY
            ):
                raise SelectedMadGraphError(
                    "MadGraph runtime progress protocol scope is invalid"
                )
            completed_identities.add((family, n))
            counts[str(cell_status)] += 1
    raw_pending = payload.get("pending_cells")
    if not isinstance(raw_pending, Sequence) or isinstance(raw_pending, str):
        raise SelectedMadGraphError(
            "MadGraph runtime progress pending_cells is invalid"
        )
    pending_identities: set[tuple[str, int]] = set()
    for raw_pending_cell in raw_pending:
        pending_cell = _mapping(
            raw_pending_cell, context="MadGraph runtime progress pending cell"
        )
        family = pending_cell.get("family")
        n = pending_cell.get("n")
        if (
            family not in FAMILIES
            or isinstance(n, bool)
            or not isinstance(n, int)
            or n not in multiplicities
            or any(
                pending_cell.get(key) != value
                for key, value in _progress_cell_identity(
                    str(family), n, helicity_workload=helicity_workload
                ).items()
            )
        ):
            raise SelectedMadGraphError(
                "MadGraph runtime progress pending cell identity is invalid"
            )
        pending_identities.add((str(family), n))
    all_identities = {
        (family, n) for family in FAMILIES for n in multiplicities
    }
    if (
        len(pending_identities) != len(raw_pending)
        or completed_identities & pending_identities
        or completed_identities | pending_identities != all_identities
    ):
        raise SelectedMadGraphError(
            "MadGraph runtime progress completed/pending partition is invalid"
        )
    summary = _mapping(
        payload.get("summary"), context="MadGraph runtime progress summary"
    )
    expected_summary = {
        "completed_cell_count": len(completed_identities),
        "pending_cell_count": len(pending_identities),
        "total_cell_count": len(all_identities),
        "runtime_series_status_counts": dict(sorted(counts.items())),
    }
    if dict(summary) != expected_summary:
        raise SelectedMadGraphError("MadGraph runtime progress summary is invalid")
    current = payload.get("current_cell")
    if current is not None:
        current_cell = _mapping(
            current, context="MadGraph runtime progress current_cell"
        )
        family = current_cell.get("family")
        n = current_cell.get("n")
        if (
            family not in FAMILIES
            or isinstance(n, bool)
            or not isinstance(n, int)
            or (str(family), n) not in pending_identities
            or any(
                current_cell.get(key) != value
                for key, value in _progress_cell_identity(
                    str(family), n, helicity_workload=helicity_workload
                ).items()
            )
            or not isinstance(current_cell.get("stage"), str)
            or not current_cell["stage"]
        ):
            raise SelectedMadGraphError(
                "MadGraph runtime progress current_cell is invalid"
            )
    if status == "running" and not pending_identities:
        raise SelectedMadGraphError(
            "running MadGraph runtime progress has no pending cells"
        )
    if status != "running" and (pending_identities or current is not None):
        raise SelectedMadGraphError(
            "terminal MadGraph runtime progress still has pending work"
        )
    expected_status = "complete-with-failures" if counts["failed"] else "complete"
    if status != "running" and status != expected_status:
        raise SelectedMadGraphError(
            "terminal MadGraph runtime progress status disagrees with its cells"
        )
    return payload


def _default_progress_output(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def build_runtime_report(
    *,
    source_report: Path,
    cache_dir: Path,
    fc: str,
    fflags: str,
    timeout_seconds: float,
    mg5_root: Path,
    family: str | None = None,
    min_n: int = 2,
    max_n: int = 9,
    multiplicities: Sequence[int] | None = None,
    memory_limit_gib: float = MAX_MEMORY_GIB,
    keep_generated: bool = False,
    progress_output: Path | None = None,
    helicity_workload: str = "fixed",
) -> dict[str, Any]:
    resolved_cache = cache_dir.expanduser().resolve(strict=False)
    lock = _acquire_cache_lock(resolved_cache)
    try:
        return _build_runtime_report_locked(
            source_report=source_report,
            cache_dir=resolved_cache,
            fc=fc,
            fflags=fflags,
            timeout_seconds=timeout_seconds,
            mg5_root=mg5_root,
            family=family,
            min_n=min_n,
            max_n=max_n,
            multiplicities=multiplicities,
            memory_limit_gib=memory_limit_gib,
            keep_generated=keep_generated,
            progress_output=progress_output,
            helicity_workload=helicity_workload,
        )
    finally:
        lock.close()


def _build_runtime_report_locked(
    *,
    source_report: Path,
    cache_dir: Path,
    fc: str,
    fflags: str,
    timeout_seconds: float,
    mg5_root: Path,
    family: str | None = None,
    min_n: int = 2,
    max_n: int = 9,
    multiplicities: Sequence[int] | None = None,
    memory_limit_gib: float = MAX_MEMORY_GIB,
    keep_generated: bool = False,
    progress_output: Path | None = None,
    helicity_workload: str = "fixed",
) -> dict[str, Any]:
    _workload_label(helicity_workload)
    if family is not None and family not in FAMILIES:
        raise SelectedMadGraphError(f"unsupported family {family!r}")
    raw_multiplicities: Sequence[int] = (
        multiplicities if multiplicities is not None else range(min_n, max_n + 1)
    )
    if not raw_multiplicities or any(
        isinstance(n, bool) or not isinstance(n, int) or n < 2
        for n in raw_multiplicities
    ):
        raise SelectedMadGraphError("multiplicities must be integers >=2")
    selected_multiplicities = tuple(sorted(set(raw_multiplicities)))
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
    ):
        raise SelectedMadGraphError("generation timeout must be positive and finite")
    if (
        isinstance(memory_limit_gib, bool)
        or not math.isfinite(memory_limit_gib)
        or memory_limit_gib <= 0.0
    ):
        raise SelectedMadGraphError("memory limit must be positive and finite")
    mg5_root = mg5_root.expanduser().resolve(strict=True)
    if (
        not (mg5_root / "bin" / "mg5_aMC").is_file()
        or not (mg5_root / "VERSION").is_file()
    ):
        raise SelectedMadGraphError(f"invalid MadGraph installation: {mg5_root}")
    if not WATCHDOG.is_file():
        raise SelectedMadGraphError(f"memory watchdog is missing: {WATCHDOG}")
    measured_multiplicities = tuple(
        n
        for n in selected_multiplicities
        if n <= MAX_PROTOCOL_MEASURED_MULTIPLICITY
    )
    source = load_source_selection(
        source_report,
        multiplicities=measured_multiplicities,
        helicity_workload=helicity_workload,
    )
    cache_dir = cache_dir.expanduser().resolve(strict=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected_families = FAMILIES if family is None else (family,)
    runtime_series: dict[str, dict[str, dict[str, Any]]] = {
        name: {MODE: {}} for name in FAMILIES
    }
    started_at_utc = datetime.now(UTC).isoformat()
    progress_path = (
        progress_output.expanduser().resolve(strict=False)
        if progress_output is not None
        else None
    )

    def load_reusable_checkpoint(name: str, n: int) -> dict[str, Any] | None:
        if (
            name not in selected_families
            or n > MAX_PROTOCOL_MEASURED_MULTIPLICITY
            or n in source.unavailable[name]
        ):
            return None
        selected = source.cells[name][n]
        cell_dir = cache_dir / name / f"n{n}"
        checkpoint = cell_dir / "cell.json"
        if not checkpoint.is_file():
            return None
        identity = _checkpoint_identity(
            source,
            selected,
            mg5_root,
            fc,
            fflags,
            timeout_seconds,
            memory_limit_gib,
        )
        cell = _load_checkpoint_cell(
            checkpoint,
            identity=identity,
            cell_dir=cell_dir,
            keep_generated=keep_generated,
        )
        cell, provenance_changed = _rebind_source_provenance(
            cell, source=source, selected=selected
        )
        if provenance_changed:
            _json_atomic(checkpoint, {"checkpoint_identity": identity, "cell": cell})
        return cell

    # Make a resumed/expanded run's first atomic progress snapshot at least as
    # informative as its reusable cache.  Otherwise the initial empty running
    # snapshot temporarily hides a previously published terminal prefix.
    for name in selected_families:
        for n in selected_multiplicities:
            reusable = load_reusable_checkpoint(name, n)
            if reusable is not None:
                runtime_series[name][MODE][str(n)] = reusable

    def publish_progress(current_cell: Mapping[str, Any] | None) -> None:
        if progress_path is None:
            return
        progress = _runtime_progress_report(
            source=source,
            runtime_series=runtime_series,
            selected_multiplicities=selected_multiplicities,
            selected_families=selected_families,
            timeout_seconds=timeout_seconds,
            memory_limit_gib=memory_limit_gib,
            started_at_utc=started_at_utc,
            current_cell=current_cell,
        )
        _json_atomic(progress_path, progress)

    publish_progress(None)
    for name in FAMILIES:
        if name not in selected_families:
            for n in selected_multiplicities:
                if n > MAX_PROTOCOL_MEASURED_MULTIPLICITY:
                    publish_progress(
                        _progress_cell_identity(
                            name, n, helicity_workload=helicity_workload
                        )
                        | {"stage": "protocol-scope-not-applicable"}
                    )
                    runtime_series[name][MODE][str(n)] = _protocol_scope_cell(
                        name, n, helicity_workload=helicity_workload
                    )
                    publish_progress(None)
                    continue
                publish_progress(
                    _progress_cell_identity(
                        name, n, helicity_workload=helicity_workload
                    )
                    | {"stage": "diagnostic-family-omission"}
                )
                runtime_series[name][MODE][str(n)] = _diagnostic_skip_cell(
                    name,
                    n,
                    "family omitted by diagnostic selection",
                    helicity_workload=helicity_workload,
                )
                publish_progress(None)
            continue
        failure_n: int | None = None
        failure_reason = ""
        for n in selected_multiplicities:
            if n > MAX_PROTOCOL_MEASURED_MULTIPLICITY:
                publish_progress(
                    _progress_cell_identity(
                        name, n, helicity_workload=helicity_workload
                    )
                    | {"stage": "protocol-scope-not-applicable"}
                )
                runtime_series[name][MODE][str(n)] = _protocol_scope_cell(
                    name, n, helicity_workload=helicity_workload
                )
                publish_progress(None)
                continue
            if failure_n is not None:
                publish_progress(
                    _progress_cell_identity(
                        name, n, helicity_workload=helicity_workload
                    )
                    | {"stage": "frontier-skip"}
                )
                runtime_series[name][MODE][str(n)] = _frontier_cell(
                    name,
                    n,
                    failure_n=failure_n,
                    reason=failure_reason,
                    helicity_workload=helicity_workload,
                )
                publish_progress(None)
                continue
            reusable = runtime_series[name][MODE].get(str(n))
            if reusable is not None:
                if reusable.get("status") == "failed":
                    failure_n = n
                    failure_reason = str(reusable.get("failure_reason"))
                continue
            unavailable_reason = source.unavailable[name].get(n)
            if unavailable_reason is not None:
                publish_progress(
                    _progress_cell_identity(
                        name, n, helicity_workload=helicity_workload
                    )
                    | {"stage": "source-dependency-frontier"}
                )
                cell = _dependency_failure_cell(
                    name,
                    n,
                    unavailable_reason,
                    helicity_workload=helicity_workload,
                )
                runtime_series[name][MODE][str(n)] = cell
                failure_n = n
                failure_reason = str(cell["failure_reason"])
                publish_progress(None)
                continue
            selected = source.cells[name][n]
            cell_dir = cache_dir / name / f"n{n}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = cell_dir / "cell.json"
            publish_progress(
                _progress_cell_identity(
                    name, n, helicity_workload=helicity_workload
                )
                | {
                    "stage": (
                        "checkpoint-reuse"
                        if checkpoint.is_file()
                        else "generate-build-benchmark"
                    )
                }
            )
            identity = _checkpoint_identity(
                source,
                selected,
                mg5_root,
                fc,
                fflags,
                timeout_seconds,
                memory_limit_gib,
            )
            if checkpoint.is_file():
                cell = _load_checkpoint_cell(
                    checkpoint,
                    identity=identity,
                    cell_dir=cell_dir,
                    keep_generated=keep_generated,
                )
                cell, provenance_changed = _rebind_source_provenance(
                    cell, source=source, selected=selected
                )
                if provenance_changed:
                    _json_atomic(
                        checkpoint,
                        {"checkpoint_identity": identity, "cell": cell},
                    )
            else:
                generation, generation_watchdog = _generation_attempt(
                    selected=selected,
                    cell_dir=cell_dir,
                    mg5_root=mg5_root,
                    timeout=timeout_seconds,
                    limit_gib=memory_limit_gib,
                )
                if generation.get("status") != "measured":
                    category = str(
                        generation.get("failure_category") or "generation-error"
                    )
                    reason = str(
                        generation.get("failure_reason") or "MadGraph generation failed"
                    )
                    if category not in RESOURCE_FRONTIER_CATEGORIES:
                        if not keep_generated:
                            _prune_cell_generated(cell_dir)
                        raise SelectedMadGraphError(
                            f"{name}/n{n} generation diagnostic ({category}): {reason}"
                        )
                    cell = _failure_cell(selected, category, reason)
                else:
                    try:
                        cell = _measure_generated_cell(
                            source=source,
                            selected=selected,
                            generation=generation,
                            generation_watchdog=generation_watchdog,
                            cell_dir=cell_dir,
                            fc=fc,
                            fflags=fflags,
                            limit_gib=memory_limit_gib,
                            mg5_root=mg5_root,
                            cold_to_ready_limit_seconds=timeout_seconds,
                        )
                    except ResourceFrontierError as error:
                        cell = _failure_cell(selected, error.category, str(error))
                    except (OSError, SelectedMadGraphError) as error:
                        if not keep_generated:
                            _prune_cell_generated(cell_dir)
                        context = (
                            "campaign filesystem ran out of space"
                            if _is_disk_exhaustion(error)
                            else "MadGraph build/runtime diagnostic failed"
                        )
                        raise SelectedMadGraphError(
                            f"{name}/n{n}: {context}: {error}"
                        ) from error
                _json_atomic(
                    checkpoint, {"checkpoint_identity": identity, "cell": cell}
                )
                if not keep_generated:
                    _prune_cell_generated(cell_dir)
            runtime_series[name][MODE][str(n)] = cell
            if cell.get("status") == "failed":
                failure_n = n
                failure_reason = str(cell.get("failure_reason"))
            publish_progress(None)

    counts = Counter(
        str(cell.get("status"))
        for family_series in runtime_series.values()
        for mode_cells in family_series.values()
        for cell in mode_cells.values()
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete" if counts.get("failed", 0) == 0 else "complete-with-failures"
        ),
        "failure_count": counts.get("failed", 0),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "policy": _runtime_policy(
            selected_multiplicities,
            timeout_seconds=timeout_seconds,
            memory_limit_gib=memory_limit_gib,
            helicity_workload=helicity_workload,
        ),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "summary": {"runtime_series_status_counts": dict(sorted(counts.items()))},
        "runtime_series": runtime_series,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-report",
        type=Path,
        required=True,
        help="authenticated campaign report supplying event and helicity selections",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--progress-output",
        type=Path,
        help=(
            "atomic sparse progress JSON (default: <output-stem>.progress.json)"
        ),
    )
    parser.add_argument("--mg5-root", type=Path, required=True)
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--fflags", default="-O3")
    parser.add_argument(
        "--generation-timeout-seconds",
        "--timeout-seconds",
        dest="timeout_seconds",
        type=float,
        default=DEFAULT_GENERATION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--memory-limit-gib", type=float, default=MAX_MEMORY_GIB)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=9)
    parser.add_argument(
        "--multiplicity",
        dest="multiplicities",
        action="append",
        type=int,
        help="exact final-state multiplicity; repeat for a sparse scan",
    )
    parser.add_argument(
        "--protocol-scope-multiplicity",
        dest="protocol_scope_multiplicities",
        action="append",
        type=int,
        help=(
            "requested n>6 recorded as not-applicable without measurement; "
            "repeat for a sparse final-plot scan"
        ),
    )
    parser.add_argument("--keep-generated", action="store_true")
    parser.add_argument(
        "--compare-helicity-sums",
        action="store_true",
        help=(
            "benchmark generated SMATRIX(P,ANS) with USERHEL=-1 and warmed "
            "GOODHEL pruning instead of one fixed direct-MATRIX helicity"
        ),
    )
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mg5-root", type=Path, required=True)
    parser.add_argument("--generated-output", type=Path, required=True)
    parser.add_argument("--worker-result", type=Path, required=True)
    parser.add_argument("--process", required=True)
    parser.add_argument("--generation-timeout-seconds", type=float, required=True)
    return parser


def _timeout_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "_run-with-timeout":
        try:
            return _timeout_worker(_timeout_parser().parse_args(raw[1:]))
        except SelectedMadGraphError as error:
            print(f"MadGraph benchmark error: {error}", file=sys.stderr)
            return 2
    if raw and raw[0] == "_generate-cell":
        try:
            return _generation_worker(_worker_parser().parse_args(raw[1:]))
        except SelectedMadGraphError as error:
            print(f"MadGraph benchmark error: {error}", file=sys.stderr)
            return 2
    arguments = _parser().parse_args(raw)
    try:
        measured = tuple(arguments.multiplicities or ())
        scoped = tuple(arguments.protocol_scope_multiplicities or ())
        if scoped and any(
            n <= MAX_PROTOCOL_MEASURED_MULTIPLICITY for n in scoped
        ):
            raise SelectedMadGraphError(
                "--protocol-scope-multiplicity accepts only n>6"
            )
        if measured and any(
            n > MAX_PROTOCOL_MEASURED_MULTIPLICITY for n in measured
        ):
            raise SelectedMadGraphError(
                "--multiplicity measures only n<=6; use "
                "--protocol-scope-multiplicity for n>6"
            )
        if set(measured) & set(scoped):
            raise SelectedMadGraphError(
                "measured and protocol-scoped multiplicities must be disjoint"
            )
        selected_multiplicities = (
            tuple(sorted(set(measured) | set(scoped)))
            if measured or scoped
            else None
        )
        output = arguments.output.expanduser().resolve(strict=False)
        progress_output = (
            arguments.progress_output.expanduser().resolve(strict=False)
            if arguments.progress_output is not None
            else _default_progress_output(output)
        )
        report = build_runtime_report(
            source_report=arguments.source_report,
            cache_dir=arguments.cache_dir,
            fc=arguments.fc,
            fflags=arguments.fflags,
            timeout_seconds=arguments.timeout_seconds,
            family=arguments.family,
            min_n=arguments.min_n,
            max_n=arguments.max_n,
            multiplicities=selected_multiplicities,
            mg5_root=arguments.mg5_root,
            memory_limit_gib=arguments.memory_limit_gib,
            keep_generated=arguments.keep_generated,
            progress_output=progress_output,
            helicity_workload=(
                "sum" if arguments.compare_helicity_sums else "fixed"
            ),
        )
        _json_atomic(output, report)
    except SelectedMadGraphError as error:
        print(f"MadGraph benchmark error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
