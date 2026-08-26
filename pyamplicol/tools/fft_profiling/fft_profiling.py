#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run and render the FullColor FFT profiling scan on a workstation or cluster.

This is deliberately an orchestration layer.  Physics generation, timings,
per-cell resource enforcement, numerical validation, report validation,
plotting, and PDF assembly remain owned by the existing developer tools.

Typical use::

    python tools/fft_profiling/fft_profiling.py --dry-run --cores 8
    python tools/fft_profiling/fft_profiling.py --lines pyamplicol-recurrence
    python tools/fft_profiling/fft_profiling.py --cores 8 --candidate-cores 2
    python tools/fft_profiling/fft_profiling.py --compare-helicity-sums --cores 8
    python tools/fft_profiling/fft_profiling.py --resume --cores 16
    python tools/fft_profiling/fft_profiling.py --render

``--render`` never waits for active workers.  It freezes one already-published
campaign snapshot and delegates the six PNGs and PDF to the authoritative
renderers.  ``--campaign-report`` can render another live scaling-study report,
which is useful for an existing non-orchestrated campaign.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import colorama
    import progressbar
except ModuleNotFoundError as error:  # pragma: no cover - installation dependent
    raise SystemExit(
        "fft_profiling.py requires the project dependencies progressbar2 and "
        "colorama; install pyAmpliCol in this Python environment"
    ) from error

from tools.ci import memory_watchdog  # noqa: E402
from tools.developer import fft_madgraph_selected_runtime as madgraph  # noqa: E402
from tools.developer import (  # noqa: E402
    fft_scaling_final_publication_report as publication,
)
from tools.developer import fft_scaling_selected_scalar_report as selected  # noqa: E402
from tools.developer import fft_scaling_study as study  # noqa: E402

KIND = "pyamplicol-fft-profiling-orchestrator"
SCHEMA_VERSION = 1
DEFAULT_RUNS_ROOT = ROOT / "IMPLEMENTATION_DOCS" / "RESULTS" / "fft-profiling" / "runs"
DEFAULT_FIXED_OUTPUT = DEFAULT_RUNS_ROOT / "cluster-fullcolor-n2-n9"
DEFAULT_SUM_OUTPUT = DEFAULT_RUNS_ROOT / "cluster-fullcolor-helicity-sum-n2-n9"
CANONICAL_RESULTS_ROOT = ROOT / "IMPLEMENTATION_DOCS" / "RESULTS"
STUDY_TOOL = ROOT / "tools" / "developer" / "fft_scaling_study.py"
MERGE_TOOL = ROOT / "tools" / "developer" / "fft_scaling_final_publication_report.py"
PLOT_TOOL = ROOT / "tools" / "developer" / "fft_scaling_study_plots.py"
PDF_TOOL = ROOT / "tools" / "developer" / "fft_results_summary_pdf.py"
MADGRAPH_TOOL = ROOT / "tools" / "developer" / "fft_madgraph_selected_runtime.py"
RENDER_REQUIREMENTS = ("matplotlib==3.10.8", "reportlab==4.4.4")
TERMINAL_STATUSES = frozenset({"complete", "complete-with-failures"})
ORCHESTRATOR_TERMINAL_CELL_STATUSES = frozenset(
    {"measured", "failed", "skipped", "not-applicable"}
)
MUTABLE_RESOURCE_IDENTITY_PATHS = frozenset(
    {
        "scan.target_seconds",
        "resources.per_cell_generation_timeout_seconds",
        "resources.per_cell_memory_limit_gib",
        "resources.per_cell_runtime_timeout_seconds",
    }
)
RETRYABLE_CELL_STATUSES = frozenset({"failed", "skipped"})
LEGACY_MADGRAPH_NOTE = (
    "MadGraph points are retained from a same-workstation snapshot that "
    "predates the node-fingerprint field; strict terminal host authentication "
    "is therefore unavailable for this anytime render."
)


class ProfilingError(RuntimeError):
    """The cluster profiling orchestration contract could not be satisfied."""


class CampaignSignal(KeyboardInterrupt):
    """A termination signal requested a resumable campaign shutdown."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@dataclass(frozen=True, slots=True)
class Shard:
    name: str
    family: str
    modes: tuple[str, ...]
    owned_modes: tuple[str, ...]
    dependency: tuple[str, str] | None
    phase: int
    candidate: bool


@dataclass(frozen=True, slots=True)
class ShardWork:
    shard: Shard
    final_multiplicity: int
    owned_mode: str


@dataclass(slots=True)
class ActiveJob:
    shard: Shard
    final_multiplicity: int | None
    owned_mode: str | None
    command: tuple[str, ...]
    process: subprocess.Popen[bytes]
    stdout: BinaryIO
    stderr: BinaryIO
    sampler: memory_watchdog.ProcessTreeSampler
    claimed_cores: int
    detail: str


@dataclass(frozen=True, slots=True)
class RenderSelection:
    source: Path
    overlay: Path | None


SHARDS = (
    Shard("gg-reference", "gg", ("reference-fft",), ("reference-fft",), None, 1, False),
    Shard("ddbar-amplicol", "ddbar", ("amplicol",), ("amplicol",), None, 1, False),
    Shard(
        "gg-recurrence",
        "gg",
        ("reference-fft", "recurrence-direct", "recurrence-fft"),
        ("recurrence-direct", "recurrence-fft"),
        ("gg-reference", "reference-fft"),
        2,
        True,
    ),
    Shard(
        "gg-otf",
        "gg",
        ("reference-fft", "otf-direct", "otf-fft"),
        ("otf-direct", "otf-fft"),
        ("gg-reference", "reference-fft"),
        2,
        True,
    ),
    Shard(
        "ddbar-recurrence",
        "ddbar",
        ("amplicol", "recurrence-direct", "recurrence-fft"),
        ("recurrence-direct", "recurrence-fft"),
        ("ddbar-amplicol", "amplicol"),
        2,
        True,
    ),
    Shard(
        "ddbar-otf",
        "ddbar",
        ("amplicol", "otf-direct", "otf-fft"),
        ("otf-direct", "otf-fft"),
        ("ddbar-amplicol", "amplicol"),
        2,
        True,
    ),
    Shard(
        "gg-amplicol",
        "gg",
        ("reference-fft", "amplicol"),
        ("amplicol",),
        ("gg-reference", "reference-fft"),
        2,
        False,
    ),
)
SHARD_BY_NAME = {shard.name: shard for shard in SHARDS}
MODE_OWNER = {
    (shard.family, mode): shard.name for shard in SHARDS for mode in shard.owned_modes
}
FAMILIES = ("gg", "ddbar")
LINE_GROUPS = (
    "reference-fft",
    "amplicol",
    "pyamplicol-recurrence",
    "pyamplicol-otf",
    "madgraph",
)
LINE_MODE_SELECTORS = (
    "recurrence-direct",
    "recurrence-fft",
    "otf-direct",
    "otf-fft",
)
LINE_CHOICES = (*LINE_GROUPS, *LINE_MODE_SELECTORS)
RENDER_LINE_CHOICES = (
    *LINE_CHOICES,
    "madgraph-standalone",
    "recola",
    "compiled-direct",
    "compiled-fft",
)
LINE_SELECTOR_SHARDS = {
    "reference-fft": ("gg-reference",),
    "amplicol": ("ddbar-amplicol", "gg-amplicol"),
    "pyamplicol-recurrence": ("gg-recurrence", "ddbar-recurrence"),
    "pyamplicol-otf": ("gg-otf", "ddbar-otf"),
    "madgraph": (),
    "recurrence-direct": ("gg-recurrence", "ddbar-recurrence"),
    "recurrence-fft": ("gg-recurrence", "ddbar-recurrence"),
    "otf-direct": ("gg-otf", "ddbar-otf"),
    "otf-fft": ("gg-otf", "ddbar-otf"),
}
LINE_SELECTOR_OWNED_MODES = {
    "reference-fft": ("reference-fft",),
    "amplicol": ("amplicol",),
    "pyamplicol-recurrence": ("recurrence-direct", "recurrence-fft"),
    "pyamplicol-otf": ("otf-direct", "otf-fft"),
    "madgraph": (),
    "recurrence-direct": ("recurrence-direct",),
    "recurrence-fft": ("recurrence-fft",),
    "otf-direct": ("otf-direct",),
    "otf-fft": ("otf-fft",),
}


def _positive_finite(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return value


def _finite_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _positive_integer(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _default_madgraph_root() -> Path | None:
    configured = os.environ.get("PYAMPLICOL_MADGRAPH_ROOT")
    candidates = (
        *((Path(configured).expanduser(),) if configured else ()),
        ROOT / "dependencies" / "checkouts" / "MG5_aMC",
        ROOT.parent / "MG5_aMC",
        Path.home() / "MG5" / "MG5_aMC_v3_7_1",
    )
    return next(
        (
            candidate.resolve(strict=False)
            for candidate in candidates
            if (candidate / "bin" / "mg5_aMC").is_file()
            and (candidate / "VERSION").is_file()
        ),
        None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "remove the exact recognized profiling --output and restart from scratch"
        ),
    )
    action.add_argument(
        "--render",
        action="store_true",
        help="render the latest frozen snapshot immediately; never wait for workers",
    )
    action.add_argument("--status", action="store_true", help="print JSON status")
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="print the deterministic plan as JSON without writing or launching",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="explicit alias for the default automatic resume behavior",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help=(
            "rerun failed/skipped cells in the current --families/--lines/"
            "--multiplicities selection without refreshing the whole output"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "complete profiling run/render directory (default is a distinct "
            "fixed-helicity or helicity-sum run directory)"
        ),
    )
    parser.add_argument(
        "--compare-helicity-sums",
        action="store_true",
        help=(
            "profile the complete physical-helicity-summed matrix element in "
            "an independent campaign, including a genuine summed MadGraph "
            "standalone series when --madgraph-root is available"
        ),
    )
    parser.add_argument(
        "--multiplicities",
        type=int,
        nargs="+",
        default=list(range(2, 10)),
        metavar="N",
        help=(
            "final-state multiplicities to add to the resumable scan; unfinished "
            "prior requests are also recovered (default: 2 3 ... 9)"
        ),
    )
    parser.add_argument(
        "--lines",
        nargs="+",
        choices=LINE_CHOICES,
        default=list(LINE_GROUPS),
        metavar="LINE",
        help=(
            "plot line groups to add to the resumable scan: reference-fft, "
            "amplicol, pyamplicol-recurrence (direct and FFT), "
            "pyamplicol-otf (direct and FFT), concrete pyAmpliCol modes "
            "recurrence-direct, recurrence-fft, otf-direct, otf-fft, and/or "
            "madgraph; repeated runs add exactly the requested selectors and "
            "multiplicities in the same --output (default: all groups)"
        ),
    )
    parser.add_argument(
        "--families",
        "--process-families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        metavar="FAMILY",
        help=(
            "process families to add to the scan: gg for g g > g g + gluons, "
            "ddbar for d d~ > d d~ + gluons (default: both)"
        ),
    )
    parser.add_argument(
        "--cores",
        type=_positive_integer,
        default=max(1, os.cpu_count() or 1),
        help="total scheduler core budget (default: detected logical cores)",
    )
    parser.add_argument(
        "--candidate-cores",
        type=_positive_integer,
        default=1,
        help=(
            "cores claimed by one candidate shard and passed exactly as "
            "evaluator.optimization.cores (default: 1)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=study.BENCHMARK_BATCH_SIZE,
        help=(
            "points in each pyAmpliCol public profile call; Reference FFT and "
            "AmpliCol remain scalar aggregates normalized per point "
            f"(default: {study.BENCHMARK_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=_positive_finite,
        default=study.REQUESTED_MEMORY_LIMIT_GIB,
        help="strict RAM cutoff for each measurement child (default: 30)",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=_positive_finite,
        default=study.DEFAULT_TIME_LIMIT_SECONDS,
        help="generation and runtime cutoff for each cell (default: 3600)",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--fc", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--make", default=os.environ.get("MAKE", "make"))
    parser.add_argument(
        "--target-seconds",
        type=_positive_finite,
        default=study.DEFAULT_PROFILE_TARGET_SECONDS,
        help=(
            "target runtime for each profiling cell after its first "
            "calibration/probe evaluation (default: 180)"
        ),
    )
    parser.add_argument(
        "--otf-max-multiplicity",
        type=int,
        default=study.DEFAULT_OTF_MAX_MULTIPLICITY,
        help=(
            "largest final-state multiplicity to admit for pyAmpliCol OTF "
            "measurements; raise this for an explicit high-multiplicity top-up "
            f"(default: {study.DEFAULT_OTF_MAX_MULTIPLICITY})"
        ),
    )
    parser.add_argument(
        "--amplicol-root",
        "--amplicol-repository",
        dest="amplicol_root",
        type=Path,
        default=study.legacy_amplicol.DEFAULT_REPOSITORY,
        help=(
            "original-AmpliCol checkout (default path is populated by "
            "dev-install --with-legacy-amplicol)"
        ),
    )
    parser.add_argument(
        "--reference-fft-root",
        type=Path,
        default=study.performance.DEFAULT_REFERENCE_ROOT,
        help=(
            "pinned AllGluonsMultipletFFT checkout for Reference FFT "
            "(default path is populated by dev-install --with-reference-fft)"
        ),
    )
    parser.add_argument(
        "--build-amplicol",
        action="store_true",
        help="allow the prerequisite AmpliCol shard to build its probe once",
    )
    parser.add_argument(
        "--madgraph-root",
        type=Path,
        default=_default_madgraph_root(),
        help=(
            "MadGraph installation used for the same-host standalone series "
            "(defaults to PYAMPLICOL_MADGRAPH_ROOT or a developer checkout)"
        ),
    )
    parser.add_argument(
        "--madgraph-overlay",
        type=Path,
        help="with --render --campaign-report, an existing authenticated overlay",
    )
    parser.add_argument(
        "--campaign-report",
        type=Path,
        help="with --render, freeze and render this existing live report",
    )
    parser.add_argument(
        "--recola-results",
        type=Path,
        help="with --render, overlay a Recola profiling JSON as a Recola line",
    )
    parser.add_argument(
        "--main-y-range",
        type=_finite_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="with --render, force the main-panel log y-range for every plot",
    )
    parser.add_argument(
        "--ratio-y-range",
        type=_finite_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="with --render, force the ratio-panel y-range for every plot",
    )
    parser.add_argument(
        "--ratio-y-scale",
        choices=("log", "linear"),
        default="log",
        help="with --render, ratio-panel y-axis scale (default: log)",
    )
    parser.add_argument(
        "--main-include-lines",
        nargs="+",
        choices=RENDER_LINE_CHOICES,
        metavar="LINE",
        help="with --render, main-panel line ids to include; overrides vetoes",
    )
    parser.add_argument(
        "--main-veto-lines",
        nargs="+",
        choices=RENDER_LINE_CHOICES,
        metavar="LINE",
        help="with --render, main-panel line ids to hide when include is absent",
    )
    parser.add_argument(
        "--ratio-include-lines",
        nargs="+",
        choices=RENDER_LINE_CHOICES,
        metavar="LINE",
        help="with --render, ratio-panel line ids to include; overrides vetoes",
    )
    parser.add_argument(
        "--ratio-veto-lines",
        nargs="+",
        choices=RENDER_LINE_CHOICES,
        metavar="LINE",
        help="with --render, ratio-panel line ids to hide when include is absent",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--poll-seconds", type=_positive_finite, default=0.5)
    parser.add_argument("--no-color", action="store_true")
    return parser


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _configured_run_path(arguments: argparse.Namespace) -> Path:
    if arguments.output is not None:
        return arguments.output
    return (
        DEFAULT_SUM_OUTPUT if arguments.compare_helicity_sums else DEFAULT_FIXED_OUTPUT
    )


def _run_directory(arguments: argparse.Namespace) -> Path:
    return _absolute(_configured_run_path(arguments))


def _helicity_workload(arguments: argparse.Namespace) -> str:
    return "sum" if arguments.compare_helicity_sums else "fixed"


def _pdf_filename(arguments: argparse.Namespace) -> str:
    return (
        "summary_plots_final_helicity_sum.pdf"
        if arguments.compare_helicity_sums
        else "summary_plots_final.pdf"
    )


def _canonical_pdf_path(arguments: argparse.Namespace) -> Path:
    return CANONICAL_RESULTS_ROOT / _pdf_filename(arguments)


def _json_filename(arguments: argparse.Namespace) -> str:
    return Path(_pdf_filename(arguments)).with_suffix(".json").name


def _canonical_publication_enabled(arguments: argparse.Namespace) -> bool:
    return arguments.output is None and arguments.campaign_report is None


def _universe(arguments: argparse.Namespace) -> tuple[int, ...]:
    stored = getattr(arguments, "universe_multiplicities", None)
    if stored is not None:
        return tuple(int(value) for value in stored)
    return tuple(range(2, 10))


def _selection(arguments: argparse.Namespace) -> tuple[int, ...]:
    return tuple(sorted(set(arguments.multiplicities)))


def _family_selection(arguments: argparse.Namespace) -> tuple[str, ...]:
    selected = set(arguments.families)
    return tuple(family for family in FAMILIES if family in selected)


def _line_selection(arguments: argparse.Namespace) -> tuple[str, ...]:
    selected = set(arguments.lines)
    return tuple(line for line in LINE_CHOICES if line in selected)


def _requested_line_groups(
    arguments: argparse.Namespace, run_directory: Path
) -> tuple[str, ...]:
    _ = run_directory
    selected = set(_line_selection(arguments))
    return tuple(line for line in LINE_CHOICES if line in selected)


def _madgraph_requested(arguments: argparse.Namespace, run_directory: Path) -> bool:
    return "madgraph" in _requested_line_groups(arguments, run_directory)


def _requested_shards(
    arguments: argparse.Namespace, run_directory: Path
) -> tuple[Shard, ...]:
    selected_families = set(_family_selection(arguments))
    names = {
        name
        for line in _requested_line_groups(arguments, run_directory)
        for name in LINE_SELECTOR_SHARDS[line]
        if SHARD_BY_NAME[name].family in selected_families
    }
    if _madgraph_requested(arguments, run_directory):
        source_modes = (
            publication.SUM_SOURCE_MODE
            if arguments.compare_helicity_sums
            else publication.SOURCE_MODE
        )
        names.update(
            MODE_OWNER[(family, mode)]
            for family, mode in source_modes.items()
            if family in selected_families
        )
    pending = list(names)
    while pending:
        shard = SHARD_BY_NAME[pending.pop()]
        if shard.dependency is None:
            continue
        dependency_name, _ = shard.dependency
        if dependency_name not in names:
            names.add(dependency_name)
            pending.append(dependency_name)
    return tuple(shard for shard in SHARDS if shard.name in names)


def _line_selected_owned_modes(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[str, ...]:
    selected: set[str] = set()
    for line in _requested_line_groups(arguments, run_directory):
        for mode in LINE_SELECTOR_OWNED_MODES[line]:
            if mode in shard.owned_modes:
                selected.add(mode)
    if _madgraph_requested(arguments, run_directory):
        source_modes = (
            publication.SUM_SOURCE_MODE
            if arguments.compare_helicity_sums
            else publication.SOURCE_MODE
        )
        for family, mode in source_modes.items():
            if MODE_OWNER[(family, mode)] == shard.name:
                selected.add(mode)
    return tuple(mode for mode in shard.owned_modes if mode in selected)


def _shard_owned_modes(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[str, ...]:
    selected = set(_line_selected_owned_modes(arguments, run_directory, shard))
    requested_names = {
        requested.name for requested in _requested_shards(arguments, run_directory)
    }
    for requested in SHARDS:
        if requested.name not in requested_names or requested.dependency is None:
            continue
        dependency_name, dependency_mode = requested.dependency
        if dependency_name == shard.name and dependency_mode in shard.owned_modes:
            selected.add(dependency_mode)
    return tuple(mode for mode in shard.owned_modes if mode in selected)


def _shard_modes(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[str, ...]:
    selected = set(_shard_owned_modes(arguments, run_directory, shard))
    if selected and shard.dependency is not None:
        _, dependency_mode = shard.dependency
        selected.add(dependency_mode)
    return tuple(mode for mode in shard.modes if mode in selected)


def _counted_owned_modes(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[str, ...]:
    if arguments.retry:
        return _line_selected_owned_modes(arguments, run_directory, shard)
    return _shard_owned_modes(arguments, run_directory, shard)


def _scheduled_shards(
    arguments: argparse.Namespace, run_directory: Path
) -> tuple[Shard, ...]:
    if not arguments.retry:
        return _requested_shards(arguments, run_directory)
    return tuple(
        shard
        for shard in _requested_shards(arguments, run_directory)
        if _counted_owned_modes(arguments, run_directory, shard)
    )


def _shard_run_id(shard: Shard, final_multiplicity: int | None = None) -> str:
    if final_multiplicity is None:
        return shard.name
    return f"{shard.name}-n{final_multiplicity}"


def _work_split_path(work: ShardWork) -> bool:
    return len(work.shard.owned_modes) > 1


def _work_run_id(work: ShardWork) -> str:
    if not _work_split_path(work):
        return _shard_run_id(work.shard, work.final_multiplicity)
    return f"{work.shard.name}-{work.owned_mode}-n{work.final_multiplicity}"


def _shard_cell_root(
    run_directory: Path, shard: Shard, final_multiplicity: int
) -> Path:
    return run_directory / "shards" / shard.name / "cells" / f"n{final_multiplicity}"


def _work_cell_root(run_directory: Path, work: ShardWork) -> Path:
    if not _work_split_path(work):
        return _shard_cell_root(run_directory, work.shard, work.final_multiplicity)
    return (
        run_directory
        / "shards"
        / work.shard.name
        / "mode-cells"
        / work.owned_mode
        / f"n{work.final_multiplicity}"
    )


def _shard_study_root(
    run_directory: Path, shard: Shard, final_multiplicity: int | None = None
) -> Path:
    if final_multiplicity is None:
        return run_directory / "shards" / shard.name / "study"
    return _shard_cell_root(run_directory, shard, final_multiplicity) / "study"


def _work_study_root(run_directory: Path, work: ShardWork) -> Path:
    return _work_cell_root(run_directory, work) / "study"


def _shard_report_path(
    run_directory: Path, shard: Shard, final_multiplicity: int | None = None
) -> Path:
    return (
        _shard_study_root(run_directory, shard, final_multiplicity)
        / "runs"
        / _shard_run_id(shard, final_multiplicity)
        / "report.json"
    )


def _work_report_path(run_directory: Path, work: ShardWork) -> Path:
    return (
        _work_study_root(run_directory, work)
        / "runs"
        / _work_run_id(work)
        / "report.json"
    )


def _master_study_root(run_directory: Path) -> Path:
    return run_directory / "master"


def _master_report_path(run_directory: Path) -> Path:
    return _master_study_root(run_directory) / "runs" / "campaign" / "report.json"


def _madgraph_overlay_path(run_directory: Path) -> Path:
    return run_directory / "madgraph" / "overlay.json"


def _madgraph_progress_path(run_directory: Path) -> Path:
    return run_directory / "madgraph" / "overlay.progress.json"


def _madgraph_cache_path(run_directory: Path) -> Path:
    return run_directory / "madgraph" / "cache"


def _requested_multiplicities(
    arguments: argparse.Namespace, run_directory: Path
) -> tuple[int, ...]:
    _ = run_directory
    return _selection(arguments)


def _manifest_int_history(
    manifest: Mapping[str, Any], primary: str, fallback: str
) -> list[int]:
    values = manifest.get(primary, manifest.get(fallback, []))
    if not isinstance(values, list) or any(
        not isinstance(value, int) for value in values
    ):
        raise ProfilingError("profiling manifest has invalid fill history")
    return values


def _manifest_line_history(manifest: Mapping[str, Any]) -> list[str]:
    values = manifest.get(
        "line_group_history", manifest.get("requested_line_groups", [])
    )
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        or not set(values).issubset(LINE_CHOICES)
    ):
        raise ProfilingError("profiling manifest has invalid line-group history")
    selected = set(values)
    return [line for line in LINE_CHOICES if line in selected]


def _manifest_family_history(manifest: Mapping[str, Any]) -> list[str]:
    identity = manifest.get("identity")
    scan = identity.get("scan") if isinstance(identity, Mapping) else None
    fallback = (
        scan.get("families")
        if isinstance(scan, Mapping) and isinstance(scan.get("families"), list)
        else []
    )
    values = manifest.get(
        "process_family_history", manifest.get("requested_families", fallback)
    )
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        or not set(values).issubset(FAMILIES)
    ):
        raise ProfilingError("profiling manifest has invalid process-family history")
    selected = set(values)
    return [family for family in FAMILIES if family in selected]


def _manifest_backed_arguments(
    arguments: argparse.Namespace, run_directory: Path
) -> argparse.Namespace:
    manifest_path = _manifest_path(run_directory)
    if not manifest_path.is_file():
        return arguments
    manifest = _load_json(manifest_path, context="profiling manifest")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        return arguments
    backed = copy.copy(arguments)
    scan = identity.get("scan")
    if isinstance(scan, Mapping):
        workload = scan.get("helicity_workload")
        if workload in {"fixed", "sum"}:
            backed.compare_helicity_sums = workload == "sum"
        if isinstance(scan.get("batch_size"), int):
            backed.batch_size = scan["batch_size"]
        if isinstance(scan.get("target_seconds"), int | float):
            backed.target_seconds = float(scan["target_seconds"])
    resources = identity.get("resources")
    if isinstance(resources, Mapping):
        if isinstance(resources.get("candidate_optimization_cores"), int):
            backed.candidate_cores = resources["candidate_optimization_cores"]
        if isinstance(resources.get("per_cell_memory_limit_gib"), int | float):
            backed.memory_limit_gib = float(resources["per_cell_memory_limit_gib"])
        generation_timeout = resources.get("per_cell_generation_timeout_seconds")
        runtime_timeout = resources.get("per_cell_runtime_timeout_seconds")
        if isinstance(generation_timeout, int | float) and isinstance(
            runtime_timeout, int | float
        ):
            backed.time_limit_seconds = float(max(generation_timeout, runtime_timeout))
    tools = identity.get("tools")
    if isinstance(tools, Mapping):
        for attribute in ("python", "cxx", "fc"):
            value = tools.get(attribute)
            if isinstance(value, str) and value:
                setattr(backed, attribute, value)
        for attribute in ("amplicol_root", "reference_fft_root", "madgraph_root"):
            value = tools.get(attribute)
            if isinstance(value, str) and value:
                setattr(backed, attribute, Path(value))
    return backed


def _composition_arguments(
    arguments: argparse.Namespace, run_directory: Path
) -> argparse.Namespace:
    backed = _manifest_backed_arguments(arguments, run_directory)
    manifest_path = _manifest_path(run_directory)
    if not manifest_path.is_file():
        return backed
    manifest = _load_json(manifest_path, context="profiling manifest")
    composed = copy.copy(backed)
    composed.multiplicities = _manifest_int_history(
        manifest, "fill_history_multiplicities", "requested_multiplicities"
    )
    composed.lines = _manifest_line_history(manifest)
    composed.families = _manifest_family_history(manifest)
    return composed


def _madgraph_source_path(arguments: argparse.Namespace, run_directory: Path) -> Path:
    multiplicities = _requested_multiplicities(arguments, run_directory)
    selection = "-".join(str(value) for value in multiplicities)
    return run_directory / "madgraph" / "source-reports" / f"n-{selection}.json"


def _manifest_path(run_directory: Path) -> Path:
    return run_directory / "manifest.json"


def _study_cli_arguments(
    arguments: argparse.Namespace,
    *,
    study_root: Path,
    run_id: str,
    families: Sequence[str],
    modes: Sequence[str],
    resume: bool,
    build_amplicol: bool = False,
    fill_multiplicities: Sequence[int] | None = None,
    fill_modes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    result = [
        "--study-root",
        str(study_root),
        "--run-id",
        run_id,
        "--fft",
        "--explicit-modes",
        "--batch-size",
        str(arguments.batch_size),
        "--optimization-cores",
        str(arguments.candidate_cores),
        "--python",
        str(arguments.python),
        "--cxx",
        str(arguments.cxx),
        "--fc",
        str(arguments.fc),
        "--alpha-s",
        format(study.DEFAULT_ALPHA_S, ".17g"),
        "--target-seconds",
        format(arguments.target_seconds, ".17g"),
        "--generation-timeout",
        format(arguments.time_limit_seconds, ".17g"),
        "--runtime-timeout",
        format(arguments.time_limit_seconds, ".17g"),
        "--memory-limit-gib",
        format(arguments.memory_limit_gib, ".17g"),
        "--otf-max-multiplicity",
        str(arguments.otf_max_multiplicity),
        "--amplicol-repository",
        str(_absolute(arguments.amplicol_root)),
        "--reference-fft-root",
        str(_absolute(arguments.reference_fft_root)),
    ]
    for multiplicity in _universe(arguments):
        result.extend(("--multiplicity", str(multiplicity)))
    for multiplicity in (
        _selection(arguments)
        if fill_multiplicities is None
        else tuple(fill_multiplicities)
    ):
        result.extend(("--fill-multiplicity", str(multiplicity)))
    if fill_modes is not None:
        for mode in fill_modes:
            result.extend(("--fill-mode", mode))
    for family in families:
        result.extend(("--family", family))
    for mode in modes:
        result.extend(("--mode", mode))
    if resume:
        result.append("--resume")
    if build_amplicol:
        result.append("--build-amplicol")
    if arguments.compare_helicity_sums:
        result.append("--compare-helicity-sums")
    return tuple(result)


def _shard_cli_arguments(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    final_multiplicity: int | None = None,
    *,
    modes: Sequence[str] | None = None,
    fill_modes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    return _study_cli_arguments(
        arguments,
        study_root=_shard_study_root(run_directory, shard, final_multiplicity),
        run_id=_shard_run_id(shard, final_multiplicity),
        families=(shard.family,),
        modes=(
            tuple(modes)
            if modes is not None
            else _shard_modes(arguments, run_directory, shard)
        ),
        resume=True,
        build_amplicol=arguments.build_amplicol and shard.name == "ddbar-amplicol",
        fill_multiplicities=(
            _requested_multiplicities(arguments, run_directory)
            if final_multiplicity is None
            else (final_multiplicity,)
        ),
        fill_modes=fill_modes,
    )


def _master_arguments(
    arguments: argparse.Namespace, run_directory: Path
) -> argparse.Namespace:
    modes = tuple(
        dict.fromkeys(
            mode
            for family in FAMILIES
            for mode in publication.FAMILY_MODES[family]
        )
    )
    original = arguments.multiplicities
    arguments.multiplicities = list(_universe(arguments))
    try:
        raw_arguments = _study_cli_arguments(
            arguments,
            study_root=_master_study_root(run_directory),
            run_id="campaign",
            families=FAMILIES,
            modes=modes,
            resume=True,
        )
    finally:
        arguments.multiplicities = original
    namespace = study._parser().parse_args(raw_arguments)
    study._validate_arguments(namespace)
    return namespace


def _shard_arguments(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    final_multiplicity: int | None = None,
    *,
    modes: Sequence[str] | None = None,
) -> argparse.Namespace:
    namespace = study._parser().parse_args(
        _shard_cli_arguments(
            arguments,
            run_directory,
            shard,
            final_multiplicity,
            modes=modes,
        )
    )
    study._validate_arguments(namespace)
    return namespace


def _work_owned_modes(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> tuple[str, ...]:
    counted = _counted_owned_modes(arguments, run_directory, work.shard)
    if work.owned_mode not in counted:
        return ()
    return (work.owned_mode,)


def _work_modes(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> tuple[str, ...]:
    selected = set(_work_owned_modes(arguments, run_directory, work))
    if selected and work.shard.dependency is not None:
        _, dependency_mode = work.shard.dependency
        selected.add(dependency_mode)
    return tuple(mode for mode in work.shard.modes if mode in selected)


def _work_cli_arguments(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> tuple[str, ...]:
    return _study_cli_arguments(
        arguments,
        study_root=_work_study_root(run_directory, work),
        run_id=_work_run_id(work),
        families=(work.shard.family,),
        modes=_work_modes(arguments, run_directory, work),
        resume=True,
        build_amplicol=arguments.build_amplicol and work.shard.name == "ddbar-amplicol",
        fill_multiplicities=(work.final_multiplicity,),
        fill_modes=_work_owned_modes(arguments, run_directory, work),
    )


def _work_arguments(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> argparse.Namespace:
    namespace = study._parser().parse_args(
        _work_cli_arguments(arguments, run_directory, work)
    )
    study._validate_arguments(namespace)
    return namespace


def _validate_y_range(
    values: Sequence[float] | None,
    *,
    option: str,
    positive: bool,
) -> None:
    if values is None:
        return
    if len(values) != 2:
        raise ProfilingError(f"{option} requires exactly two values")
    low, high = (float(values[0]), float(values[1]))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ProfilingError(f"{option} values must be finite")
    if high <= low:
        raise ProfilingError(f"{option} requires MIN < MAX")
    if positive and low <= 0.0:
        raise ProfilingError(f"{option} must be positive for a logarithmic y-axis")


def _validate_render_line_filter(values: Sequence[str] | None, *, option: str) -> None:
    if values is None:
        return
    selected = tuple(str(value) for value in values)
    if len(set(selected)) != len(selected):
        raise ProfilingError(f"{option} must not contain duplicates")
    invalid = sorted(set(selected) - set(RENDER_LINE_CHOICES))
    if invalid:
        raise ProfilingError(f"{option} has unknown line id(s): {', '.join(invalid)}")


def _validate_arguments(arguments: argparse.Namespace) -> None:
    _normalize_executables(arguments)
    if arguments.refresh:
        _reject_refresh_symlinks(_configured_run_path(arguments))
    if arguments.retry and arguments.refresh:
        raise ProfilingError("--retry is a targeted alternative to --refresh")
    if arguments.retry and (arguments.render or arguments.status):
        raise ProfilingError("--retry is valid only for profiling runs")
    if arguments.candidate_cores > arguments.cores:
        raise ProfilingError("--candidate-cores cannot exceed --cores")
    if arguments.dpi < 72 or arguments.dpi > 600:
        raise ProfilingError("--dpi must be between 72 and 600")
    _validate_y_range(arguments.main_y_range, option="--main-y-range", positive=True)
    _validate_y_range(
        arguments.ratio_y_range,
        option="--ratio-y-range",
        positive=arguments.ratio_y_scale == "log",
    )
    _validate_render_line_filter(
        arguments.main_include_lines,
        option="--main-include-lines",
    )
    _validate_render_line_filter(
        arguments.main_veto_lines,
        option="--main-veto-lines",
    )
    _validate_render_line_filter(
        arguments.ratio_include_lines,
        option="--ratio-include-lines",
    )
    _validate_render_line_filter(
        arguments.ratio_veto_lines,
        option="--ratio-veto-lines",
    )
    if arguments.target_seconds < 0.25:
        raise ProfilingError("--target-seconds must be at least 0.25")
    if not 2 <= arguments.otf_max_multiplicity <= 9:
        raise ProfilingError("--otf-max-multiplicity must be in the range 2..9")
    if (
        not arguments.multiplicities
        or min(arguments.multiplicities) < 2
        or max(arguments.multiplicities) > 9
    ):
        raise ProfilingError(
            "--multiplicities values must be integers in the final-PDF scan range 2..9"
        )
    if len(set(arguments.multiplicities)) != len(arguments.multiplicities):
        raise ProfilingError("--multiplicities must not contain duplicates")
    if len(set(arguments.lines)) != len(arguments.lines):
        raise ProfilingError("--lines must not contain duplicates")
    if len(set(arguments.families)) != len(arguments.families):
        raise ProfilingError("--families must not contain duplicates")
    # The authoritative parser owns run-id, n-range, mode, and resource checks.
    _master_arguments(arguments, _run_directory(arguments))
    if arguments.campaign_report is not None and not arguments.render:
        raise ProfilingError("--campaign-report is valid only with --render")
    if arguments.recola_results is not None:
        if not arguments.render:
            raise ProfilingError("--recola-results is valid only with --render")
        if not _absolute(arguments.recola_results).is_file():
            raise ProfilingError(
                f"Recola results JSON is missing: {_absolute(arguments.recola_results)}"
            )
    render_only_options = {
        "--main-y-range": arguments.main_y_range is not None,
        "--ratio-y-range": arguments.ratio_y_range is not None,
        "--ratio-y-scale": arguments.ratio_y_scale != "log",
        "--main-include-lines": arguments.main_include_lines is not None,
        "--main-veto-lines": arguments.main_veto_lines is not None,
        "--ratio-include-lines": arguments.ratio_include_lines is not None,
        "--ratio-veto-lines": arguments.ratio_veto_lines is not None,
    }
    for option, configured in render_only_options.items():
        if configured and not arguments.render:
            raise ProfilingError(f"{option} is valid only with --render")
    run_directory = _run_directory(arguments)
    if (
        not arguments.render
        and not _requested_shards(arguments, run_directory)
        and not _madgraph_requested(arguments, run_directory)
    ):
        raise ProfilingError(
            "the selected --families and --lines leave no applicable profiling work"
        )


def _normalize_executables(arguments: argparse.Namespace) -> None:
    for attribute in ("python", "cxx", "fc", "make"):
        raw = str(getattr(arguments, attribute))
        path = Path(raw).expanduser()
        if path.parent != Path("."):
            # Preserve virtual-environment launcher symlinks: resolving the
            # final component would silently bypass the venv's site-packages.
            setattr(arguments, attribute, str(Path(os.path.abspath(path))))
            continue
        located = shutil.which(raw)
        if located is not None:
            setattr(arguments, attribute, str(Path(os.path.abspath(located))))


def _identity(arguments: argparse.Namespace, run_directory: Path) -> dict[str, Any]:
    return {
        "run_directory": str(run_directory),
        "measurement_host": madgraph.measurement_host_identity(),
        "scan": {
            "multiplicity_universe": list(_universe(arguments)),
            "helicity_workload": _helicity_workload(arguments),
            "families": list(FAMILIES),
            "modes": {
                family: list(publication.FAMILY_MODES[family])
                for family in FAMILIES
            },
            "batch_size": arguments.batch_size,
            "fft_enabled": True,
            "target_seconds": arguments.target_seconds,
        },
        "resources": {
            "candidate_optimization_cores": arguments.candidate_cores,
            "per_cell_memory_limit_gib": arguments.memory_limit_gib,
            "per_cell_generation_timeout_seconds": arguments.time_limit_seconds,
            "per_cell_runtime_timeout_seconds": arguments.time_limit_seconds,
        },
        "tools": {
            "python": str(arguments.python),
            "cxx": str(arguments.cxx),
            "fc": str(arguments.fc),
            "amplicol_root": str(_absolute(arguments.amplicol_root)),
            "reference_fft_root": str(_absolute(arguments.reference_fft_root)),
            "madgraph_root": (
                str(_absolute(arguments.madgraph_root))
                if arguments.madgraph_root is not None
                else None
            ),
        },
    }


def _child_command(
    arguments: argparse.Namespace,
    run_directory: Path,
    work: ShardWork,
) -> tuple[str, ...]:
    return (
        str(arguments.python),
        str(STUDY_TOOL),
        *_work_cli_arguments(arguments, run_directory, work),
    )


def _measurement_multiplicities(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[int, ...]:
    owned_modes = _counted_owned_modes(arguments, run_directory, shard)
    return _measurement_multiplicities_for_modes(
        arguments, run_directory, shard, owned_modes
    )


def _measurement_multiplicities_for_modes(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    owned_modes: Sequence[str],
) -> tuple[int, ...]:
    _ = run_directory
    if not owned_modes:
        return ()
    return tuple(
        final_multiplicity
        for final_multiplicity in _selection(arguments)
        if not all(
            study.otf_protocol_scope_cell(
                family=shard.family,
                mode=study.MODE_BY_KEY[mode],
                final_multiplicity=final_multiplicity,
                sum_helicities=arguments.compare_helicity_sums,
                maximum_multiplicity=arguments.otf_max_multiplicity,
            )
            is not None
            for mode in owned_modes
        )
    )


def _work_measurement_multiplicities(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> tuple[int, ...]:
    return _measurement_multiplicities_for_modes(
        arguments,
        run_directory,
        work.shard,
        _work_owned_modes(arguments, run_directory, work),
    )


def _render_commands(
    arguments: argparse.Namespace,
    *,
    source_report: Path,
    staged_report: Path,
    staged_plots: Path,
    staged_pdf: Path,
    strict: bool,
    madgraph_overlay: Path | None,
) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    plot_report = source_report
    if strict:
        if madgraph_overlay is None:
            raise ProfilingError(
                "strict terminal rendering requires an authenticated MadGraph overlay"
            )
        commands.append(
            (
                str(arguments.python),
                str(MERGE_TOOL),
                "--campaign-report",
                str(source_report),
                "--madgraph-overlay",
                str(_absolute(madgraph_overlay)),
                "--output",
                str(staged_report),
            )
        )
        plot_report = staged_report
    plot_command = [
        str(arguments.python),
        str(PLOT_TOOL),
        str(plot_report),
        str(staged_plots),
        "--dpi",
        str(arguments.dpi),
    ]
    if arguments.recola_results is not None:
        plot_command.extend(
            ("--recola-results", str(_absolute(arguments.recola_results)))
        )
    if arguments.main_y_range is not None:
        plot_command.extend(
            ("--main-y-range", *(str(value) for value in arguments.main_y_range))
        )
    if arguments.ratio_y_range is not None:
        plot_command.extend(
            ("--ratio-y-range", *(str(value) for value in arguments.ratio_y_range))
        )
    if arguments.ratio_y_scale != "log":
        plot_command.extend(("--ratio-y-scale", arguments.ratio_y_scale))
    for option, values in (
        ("--main-include-lines", arguments.main_include_lines),
        ("--main-veto-lines", arguments.main_veto_lines),
        ("--ratio-include-lines", arguments.ratio_include_lines),
        ("--ratio-veto-lines", arguments.ratio_veto_lines),
    ):
        if values is not None:
            plot_command.extend((option, *(str(value) for value in values)))
    commands.extend(
        (
            tuple(plot_command),
            (
                str(arguments.python),
                str(PDF_TOOL),
                "--campaign-report",
                str(plot_report),
                "--plot-directory",
                str(staged_plots),
                "--output",
                str(staged_pdf),
            ),
        )
    )
    return tuple(commands)


def dry_run_plan(arguments: argparse.Namespace) -> dict[str, Any]:
    run_directory = _run_directory(arguments)
    example_stage = run_directory / "render" / ".staging-EXAMPLE"
    pdf_filename = _pdf_filename(arguments)
    requested_shard_names = {
        shard.name for shard in _scheduled_shards(arguments, run_directory)
    }
    works_by_shard: dict[str, list[ShardWork]] = {shard.name: [] for shard in SHARDS}
    for phase_number in _scheduled_phase_numbers(arguments, run_directory):
        for work in _phase_work_items(arguments, run_directory, phase_number):
            works_by_shard[work.shard.name].append(work)
    requested_lines = _requested_line_groups(arguments, run_directory)
    requested_families = _family_selection(arguments)
    madgraph_requested = _madgraph_requested(arguments, run_directory)
    shard_jobs = {
        shard.name: [
            {
                "n": work.final_multiplicity,
                "mode": work.owned_mode,
                "study_root": str(_work_study_root(run_directory, work)),
                "report": str(_work_report_path(run_directory, work)),
                "argv": list(_child_command(arguments, run_directory, work)),
                "shell_command": shlex.join(
                    _child_command(arguments, run_directory, work)
                ),
            }
            for work in works_by_shard[shard.name]
        ]
        for shard in SHARDS
    }
    commands = {
        shard.name: {
            "scheduled": shard.name in requested_shard_names,
            "phase": shard.phase,
            "family": shard.family,
            "modes": list(_shard_modes(arguments, run_directory, shard)),
            "owned_modes": list(_counted_owned_modes(arguments, run_directory, shard)),
            "dependency": list(shard.dependency) if shard.dependency else None,
            "claimed_cores": (arguments.candidate_cores if shard.candidate else 1),
            "measurement_multiplicities": list(
                _measurement_multiplicities(arguments, run_directory, shard)
            ),
            "protocol_skip_multiplicities": sorted(
                set(_selection(arguments))
                - set(_measurement_multiplicities(arguments, run_directory, shard))
            ),
            "study_root": str(_shard_study_root(run_directory, shard)),
            "report": str(_shard_report_path(run_directory, shard)),
            "jobs": shard_jobs[shard.name],
            "argv": (
                shard_jobs[shard.name][0]["argv"] if shard_jobs[shard.name] else None
            ),
            "shell_command": (
                shard_jobs[shard.name][0]["shell_command"]
                if shard_jobs[shard.name]
                else None
            ),
        }
        for shard in SHARDS
    }
    strict = (
        set(requested_lines) == set(LINE_GROUPS)
        and requested_families == FAMILIES
        and _publication_profile(arguments)
    )
    render_commands = _render_commands(
        arguments,
        source_report=_master_report_path(run_directory),
        staged_report=example_stage / "report.json",
        staged_plots=example_stage / "plots",
        staged_pdf=example_stage / pdf_filename,
        strict=strict,
        madgraph_overlay=_madgraph_overlay_path(run_directory),
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "helicity_workload": _helicity_workload(arguments),
        "batch_size": arguments.batch_size,
        "requested_fill_multiplicities": list(_selection(arguments)),
        "requested_process_families": list(requested_families),
        "requested_line_groups": list(requested_lines),
        "identity": _identity(arguments, run_directory),
        "scheduler": {
            "total_core_budget": arguments.cores,
            "candidate_cores_per_active_candidate_work_item": arguments.candidate_cores,
            "baseline_cores_per_active_work_item": 1,
            "parallelism": "weighted dependency-aware shard-multiplicity workers",
        },
        "shards": commands,
        "madgraph": {
            "phase": 4,
            "applicable": madgraph_requested,
            "helicity_workload": _helicity_workload(arguments),
            "process_families": list(requested_families),
            "measurement_multiplicities": [
                n
                for n in _requested_multiplicities(arguments, run_directory)
                if any(
                    madgraph.protocol_measures_multiplicity(family, n)
                    for family in requested_families
                )
            ],
            "protocol_scope_multiplicities": [
                n
                for n in _requested_multiplicities(arguments, run_directory)
                if not any(
                    madgraph.protocol_measures_multiplicity(family, n)
                    for family in requested_families
                )
            ],
            "measurement_multiplicities_by_family": {
                family: [
                    n
                    for n in _requested_multiplicities(arguments, run_directory)
                    if madgraph.protocol_measures_multiplicity(family, n)
                ]
                for family in requested_families
            },
            "protocol_scope_multiplicities_by_family": {
                family: [
                    n
                    for n in _requested_multiplicities(arguments, run_directory)
                    if not madgraph.protocol_measures_multiplicity(family, n)
                ]
                for family in requested_families
            },
            "dependency": (
                "completed pyAmpliCol source cells for the requested MadGraph fill"
                if madgraph_requested
                else None
            ),
            "not_applicable_reason": (
                None if madgraph_requested else "madgraph line group not selected"
            ),
            "report": str(_madgraph_overlay_path(run_directory)),
            "argv": (
                list(_madgraph_command(arguments, run_directory))
                if madgraph_requested and arguments.madgraph_root is not None
                else None
            ),
            "shell_command": (
                shlex.join(_madgraph_command(arguments, run_directory))
                if madgraph_requested and arguments.madgraph_root is not None
                else None
            ),
        },
        "outputs": {
            "manifest": str(_manifest_path(run_directory)),
            "master_report": str(_master_report_path(run_directory)),
            "render_pointer": str(run_directory / "render" / "current"),
            "final_pdf": str(run_directory / "render" / "current" / pdf_filename),
            "final_json": str(
                run_directory / "render" / "current" / _json_filename(arguments)
            ),
            "canonical_pdf": (
                str(_canonical_pdf_path(arguments))
                if _canonical_publication_enabled(arguments)
                else None
            ),
            "canonical_json": (
                str(_canonical_pdf_path(arguments).with_suffix(".json"))
                if _canonical_publication_enabled(arguments)
                else None
            ),
        },
        "terminal_render": {
            "strict_publication_profile": strict,
            "commands": [list(command) for command in render_commands],
            "shell_commands": [shlex.join(command) for command in render_commands],
        },
        "prerequisites": [
            "Python environment containing pyAmpliCol, progressbar2, colorama, "
            "matplotlib, and reportlab",
            f"C++ compiler: {arguments.cxx}",
            f"Fortran compiler: {arguments.fc}",
            f"make: {arguments.make}",
            f"legacy AmpliCol checkout: {_absolute(arguments.amplicol_root)}",
            f"Reference FFT checkout: {_absolute(arguments.reference_fft_root)}",
            (
                f"MadGraph installation: {_absolute(arguments.madgraph_root)}"
                if madgraph_requested and arguments.madgraph_root is not None
                else (
                    "MadGraph installation: REQUIRED (--madgraph-root)"
                    if madgraph_requested
                    else "MadGraph installation: not required by selected lines"
                )
            ),
        ],
    }


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json(value)
    if path.is_file() and path.read_bytes() == raw:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ProfilingError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProfilingError(f"{context} {path} must be a JSON object")
    return value


def _identity_diff(stored: object, requested: object, prefix: str = "") -> list[str]:
    if isinstance(stored, Mapping) and isinstance(requested, Mapping):
        differences: list[str] = []
        for key in sorted(set(stored) | set(requested)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored:
                differences.append(f"{child}: missing -> {requested[key]!r}")
            elif key not in requested:
                differences.append(f"{child}: {stored[key]!r} -> missing")
            else:
                differences.extend(_identity_diff(stored[key], requested[key], child))
        return differences
    return [] if stored == requested else [f"{prefix}: {stored!r} -> {requested!r}"]


def _identity_path_value(identity: object, path: str) -> object:
    value = identity
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _identity_difference_path(difference: str) -> str:
    return difference.split(":", 1)[0]


def _mutable_resource_identity_update_allowed(
    stored: object, requested: object, differences: Sequence[str]
) -> bool:
    changed_paths = {
        _identity_difference_path(difference) for difference in differences
    }
    if not changed_paths or not changed_paths <= MUTABLE_RESOURCE_IDENTITY_PATHS:
        return False
    for path in changed_paths:
        old = _identity_path_value(stored, path)
        new = _identity_path_value(requested, path)
        if (
            not isinstance(old, int | float)
            or isinstance(old, bool)
            or not isinstance(new, int | float)
            or isinstance(new, bool)
            or float(new) <= 0
        ):
            return False
    return True


def _immutable_identity_differences(differences: Sequence[str]) -> list[str]:
    return [
        difference
        for difference in differences
        if _identity_difference_path(difference) not in MUTABLE_RESOURCE_IDENTITY_PATHS
    ]


def _create_or_resume_manifest(
    arguments: argparse.Namespace, run_directory: Path
) -> dict[str, Any]:
    path = _manifest_path(run_directory)
    requested_identity = _identity(arguments, run_directory)
    if path.is_file():
        manifest = _load_json(path, context="profiling manifest")
        if (
            manifest.get("kind") != KIND
            or manifest.get("schema_version") != SCHEMA_VERSION
        ):
            raise ProfilingError("profiling manifest has the wrong schema")
        differences = _identity_diff(manifest.get("identity"), requested_identity)
        identity_updated = False
        if differences:
            if _mutable_resource_identity_update_allowed(
                manifest.get("identity"), requested_identity, differences
            ):
                manifest["identity"] = requested_identity
                identity_updated = True
            else:
                preview = "\n  ".join(differences[:12])
                raise ProfilingError(
                    "resume measurement identity differs from the manifest:\n  "
                    + preview
                )
        selected = list(_selection(arguments))
        selected_lines = list(_line_selection(arguments))
        selected_families = list(_family_selection(arguments))
        prior = _manifest_int_history(
            manifest, "fill_history_multiplicities", "requested_multiplicities"
        )
        history = sorted(set(prior) | set(selected))
        prior_lines = _manifest_line_history(manifest)
        line_history = [
            line
            for line in LINE_CHOICES
            if line in set(prior_lines) | set(selected_lines)
        ]
        prior_families = _manifest_family_history(manifest)
        family_history = [
            family
            for family in FAMILIES
            if family in set(prior_families) | set(selected_families)
        ]
        updates = {
            "requested_multiplicities": selected,
            "requested_families": selected_families,
            "requested_line_groups": selected_lines,
            "active_multiplicities": selected,
            "active_families": selected_families,
            "active_line_groups": selected_lines,
            "fill_history_multiplicities": history,
            "process_family_history": family_history,
            "line_group_history": line_history,
        }
        if identity_updated or any(
            manifest.get(key) != value for key, value in updates.items()
        ):
            manifest.update(updates)
            _write_json_atomic(path, manifest)
        return manifest
    selected = list(_selection(arguments))
    selected_lines = list(_line_selection(arguments))
    selected_families = list(_family_selection(arguments))
    manifest = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "identity": requested_identity,
        "initial_scheduler_core_budget": arguments.cores,
        "requested_multiplicities": selected,
        "requested_families": selected_families,
        "requested_line_groups": selected_lines,
        "active_multiplicities": selected,
        "active_families": selected_families,
        "active_line_groups": selected_lines,
        "fill_history_multiplicities": selected,
        "process_family_history": selected_families,
        "line_group_history": selected_lines,
    }
    _write_json_atomic(path, manifest)
    return manifest


def _expected_shard_policy(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    final_multiplicity: int | None = None,
    *,
    modes: Sequence[str] | None = None,
) -> dict[str, object]:
    return study.dry_run_plan(
        _shard_arguments(
            arguments,
            run_directory,
            shard,
            final_multiplicity,
            modes=modes,
        )
    )


def _pyamplicol_contractions_for_modes(
    modes: Sequence[str],
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for mode in modes:
        if mode == "recurrence-direct":
            selected.setdefault("recurrence", []).append("direct")
        elif mode == "recurrence-fft":
            selected.setdefault("recurrence", []).append("symmetric-group-fft")
        elif mode == "otf-direct":
            selected.setdefault("on-the-fly", []).append("direct")
        elif mode == "otf-fft":
            selected.setdefault("on-the-fly", []).append("symmetric-group-fft")
    return selected


def _policy_with_expected_mode_subset(
    policy: object, *, family: str, modes: Sequence[str]
) -> object:
    if not isinstance(policy, Mapping):
        return policy
    normalized = copy.deepcopy(dict(policy))
    normalized.pop("run_root", None)
    process_families = normalized.get("process_families")
    if isinstance(process_families, dict):
        family_policy = process_families.get(family)
        if isinstance(family_policy, dict):
            actual_modes = family_policy.get("modes")
            if (
                isinstance(actual_modes, list)
                and set(modes).issubset(set(actual_modes))
            ):
                family_policy["modes"] = list(modes)

    expected_contractions = _pyamplicol_contractions_for_modes(modes)
    contractions = normalized.get("selected_pyamplicol_color_contractions")
    if expected_contractions:
        if isinstance(contractions, Mapping) and all(
            set(values).issubset(set(contractions.get(key, ())))
            for key, values in expected_contractions.items()
        ):
            normalized["selected_pyamplicol_color_contractions"] = (
                expected_contractions
            )
    else:
        normalized.pop("selected_pyamplicol_color_contractions", None)
    measurement = normalized.get("measurement")
    if isinstance(measurement, dict):
        for key in (
            "cell_admission_limits",
            *study.MUTABLE_RESUME_MEASUREMENT_POLICY_KEYS,
            "generation_timeout_seconds",
            "memory_policy",
            "memory_watchdog_gib",
            "requested_memory_ceiling_gib",
            "runtime_timeout_seconds",
        ):
            measurement.pop(key, None)
    return normalized


def _shard_policy_matches_requested_modes(
    stored: object,
    expected: object,
    *,
    family: str,
    modes: Sequence[str],
) -> bool:
    return _policy_with_expected_mode_subset(
        stored, family=family, modes=modes
    ) == _policy_with_expected_mode_subset(expected, family=family, modes=modes)


def _validate_shard_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    report: Mapping[str, Any],
    *,
    context: str,
    final_multiplicity: int | None = None,
    modes: Sequence[str] | None = None,
    allow_partial_modes: bool = False,
) -> None:
    expected_modes = (
        tuple(modes)
        if modes is not None
        else _shard_modes(arguments, run_directory, shard)
    )
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ProfilingError(f"{context} has no cells")
    family_cells = cells.get(shard.family)
    if not isinstance(family_cells, Mapping):
        raise ProfilingError(f"{context} mode set differs")
    present_modes = tuple(mode for mode in expected_modes if mode in family_cells)
    if not present_modes or (
        not allow_partial_modes and set(present_modes) != set(expected_modes)
    ):
        raise ProfilingError(f"{context} mode set differs")
    if report.get("kind") != study.KIND or report.get(
        "schema_version"
    ) != study.SCHEMA_VERSION or not _shard_policy_matches_requested_modes(
        report.get("policy"),
        _expected_shard_policy(
            arguments,
            run_directory,
            shard,
            final_multiplicity,
            modes=present_modes,
        ),
        family=shard.family,
        modes=present_modes,
    ):
        raise ProfilingError(f"{context} policy/schema differs")


def _load_single_shard_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    path: Path,
    *,
    final_multiplicity: int | None = None,
    modes: Sequence[str] | None = None,
    allow_partial_modes: bool = False,
) -> dict[str, Any]:
    report = _load_json(path, context=f"{shard.name} shard report")
    _validate_shard_report(
        arguments,
        run_directory,
        shard,
        report,
        context=f"{shard.name} shard report",
        final_multiplicity=final_multiplicity,
        modes=modes,
        allow_partial_modes=allow_partial_modes,
    )
    study.normalize_loaded_failure_cells(report, path.parent)
    return report


def _mode_set_validation_error(error: ProfilingError) -> bool:
    return str(error).endswith(" mode set differs")


def _load_compatible_shard_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    path: Path,
    *,
    final_multiplicity: int | None = None,
    modes: Sequence[str] | None = None,
    allow_partial_modes: bool = False,
) -> dict[str, Any] | None:
    try:
        return _load_single_shard_report(
            arguments,
            run_directory,
            shard,
            path,
            final_multiplicity=final_multiplicity,
            modes=modes,
            allow_partial_modes=allow_partial_modes,
        )
    except ProfilingError as error:
        if allow_partial_modes:
            return None
        if not _mode_set_validation_error(error):
            raise
    try:
        return _load_single_shard_report(
            arguments,
            run_directory,
            shard,
            path,
            final_multiplicity=final_multiplicity,
            modes=modes,
            allow_partial_modes=True,
        )
    except ProfilingError as error:
        if _mode_set_validation_error(error):
            return None
        raise


def _merge_cell_record(
    target_curve: dict[str, object],
    raw_n: str,
    cell: Mapping[str, Any],
) -> None:
    existing = target_curve.get(raw_n)
    if (
        isinstance(existing, Mapping)
        and existing.get("status") == "measured"
        and cell.get("status") != "measured"
    ):
        return
    target_curve[raw_n] = copy.deepcopy(dict(cell))


def _obsolete_otf_protocol_scope_cell(
    arguments: argparse.Namespace,
    *,
    mode: str,
    raw_n: str,
    cell: Mapping[str, Any],
) -> bool:
    return (
        raw_n.isdigit()
        and mode in study.MODE_BY_KEY
        and study.MODE_BY_KEY[mode].execution_mode == "on-the-fly"
        and int(raw_n) <= arguments.otf_max_multiplicity
        and cell.get("failure_category") == "publication-protocol-scope"
    )


def _cell_terminal_for_orchestrator(cell: object) -> bool:
    return (
        isinstance(cell, Mapping)
        and cell.get("status") in ORCHESTRATOR_TERMINAL_CELL_STATUSES
    )


def _selected_cells_terminal_for_orchestrator(
    report: Mapping[str, Any],
    *,
    family: str,
    modes: Sequence[str],
    multiplicities: Sequence[int],
) -> bool:
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return False
    family_cells = cells.get(family)
    if not isinstance(family_cells, Mapping):
        return False
    for mode in modes:
        curve = family_cells.get(mode)
        if not isinstance(curve, Mapping):
            return False
        for final_multiplicity in multiplicities:
            if not _cell_terminal_for_orchestrator(
                curve.get(str(final_multiplicity))
            ):
                return False
    return True


def _report_modes_for_shard(report: Mapping[str, Any], shard: Shard) -> tuple[str, ...]:
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return ()
    family_cells = cells.get(shard.family)
    if not isinstance(family_cells, Mapping):
        return ()
    return tuple(mode for mode in shard.modes if mode in family_cells)


def _remove_retry_cells_from_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    report: dict[str, Any],
    *,
    multiplicities: Sequence[int],
) -> set[tuple[str, str, int]]:
    removed: set[tuple[str, str, int]] = set()
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return removed
    family_cells = cells.get(shard.family)
    if not isinstance(family_cells, Mapping):
        return removed
    for mode in _counted_owned_modes(arguments, run_directory, shard):
        curve = family_cells.get(mode)
        if not isinstance(curve, dict):
            continue
        for final_multiplicity in multiplicities:
            raw_n = str(final_multiplicity)
            cell = curve.get(raw_n)
            if (
                isinstance(cell, Mapping)
                and cell.get("status") in RETRYABLE_CELL_STATUSES
            ):
                del curve[raw_n]
                removed.add((shard.family, mode, final_multiplicity))
    return removed


def _rewrite_retry_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    path: Path,
    *,
    final_multiplicity: int | None,
    multiplicities: Sequence[int],
) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    report = _load_json(path, context=f"{shard.name} retry report")
    modes = _report_modes_for_shard(report, shard)
    if not modes:
        return set()
    _validate_shard_report(
        arguments,
        run_directory,
        shard,
        report,
        context=f"{shard.name} retry report",
        final_multiplicity=final_multiplicity,
        modes=modes,
        allow_partial_modes=True,
    )
    study.normalize_loaded_failure_cells(report, path.parent)
    removed = _remove_retry_cells_from_report(
        arguments,
        run_directory,
        shard,
        report,
        multiplicities=multiplicities,
    )
    if not removed:
        return set()
    rewritten = study.compose_report(
        _shard_arguments(
            arguments,
            run_directory,
            shard,
            final_multiplicity,
            modes=modes,
        ),
        report["cells"],
    )
    _write_json_atomic(path, rewritten)
    return removed


def _apply_retry_invalidation(
    arguments: argparse.Namespace, run_directory: Path
) -> int:
    if not arguments.retry:
        return 0
    requested = _requested_multiplicities(arguments, run_directory)
    removed: set[tuple[str, str, int]] = set()
    for shard in _scheduled_shards(arguments, run_directory):
        removed.update(
            _rewrite_retry_report(
                arguments,
                run_directory,
                shard,
                _shard_report_path(run_directory, shard),
                final_multiplicity=None,
                multiplicities=requested,
            )
        )
        for final_multiplicity in requested:
            removed.update(
                _rewrite_retry_report(
                    arguments,
                    run_directory,
                    shard,
                    _shard_report_path(run_directory, shard, final_multiplicity),
                    final_multiplicity=final_multiplicity,
                    multiplicities=(final_multiplicity,),
                )
            )
            for mode in _counted_owned_modes(arguments, run_directory, shard):
                removed.update(
                    _rewrite_retry_report(
                        arguments,
                        run_directory,
                        shard,
                        _work_report_path(
                            run_directory,
                            ShardWork(shard, final_multiplicity, mode),
                        ),
                        final_multiplicity=final_multiplicity,
                        multiplicities=(final_multiplicity,),
                    )
                )
    return len(removed)


def _resource_frontier_failure_at(
    cell: Mapping[str, Any], fallback_multiplicity: int
) -> int | None:
    if cell.get("censors_higher_multiplicities") is not True or cell.get(
        "status"
    ) not in {"failed", "skipped"}:
        return None
    failed_at = cell.get("failed_at_n")
    if isinstance(failed_at, int) and not isinstance(failed_at, bool):
        return failed_at
    return fallback_multiplicity


def _mode_resource_frontier(
    curve: Mapping[str, Any], final_multiplicity: int
) -> int | None:
    frontier: int | None = None
    for raw_n, cell in curve.items():
        if not str(raw_n).isdigit() or not isinstance(cell, Mapping):
            continue
        failed_at = _resource_frontier_failure_at(cell, int(raw_n))
        if failed_at is not None and failed_at < final_multiplicity:
            frontier = min(frontier if frontier is not None else failed_at, failed_at)
    return frontier


def _resource_frontier_skip_cell(
    arguments: argparse.Namespace,
    *,
    family: str,
    mode: str,
    final_multiplicity: int,
    failed_at: int,
) -> dict[str, object]:
    reason = f"curve censored after resource limit at n={failed_at}"
    return study._cell_base(
        family,
        study.MODE_BY_KEY[mode],
        final_multiplicity,
        sum_helicities=arguments.compare_helicity_sums,
    ) | {
        "status": "skipped",
        "failure_reason": reason,
        "failed_at_n": failed_at,
        "censors_higher_multiplicities": True,
    }


def _apply_resource_frontier_censoring(
    arguments: argparse.Namespace,
    report: dict[str, Any],
    shard: Shard,
    *,
    modes: Sequence[str],
    multiplicities: Sequence[int],
) -> bool:
    cells = report["cells"]
    assert isinstance(cells, dict)
    family_cells = cells[shard.family]
    assert isinstance(family_cells, dict)
    changed = False
    for mode in modes:
        curve = family_cells[mode]
        assert isinstance(curve, dict)
        for final_multiplicity in multiplicities:
            frontier = _mode_resource_frontier(curve, final_multiplicity)
            if frontier is None:
                continue
            raw_n = str(final_multiplicity)
            existing = curve.get(raw_n)
            if isinstance(existing, Mapping):
                if (
                    existing.get("status") in {"failed", "skipped"}
                    and existing.get("censors_higher_multiplicities") is True
                ):
                    continue
                if existing.get("status") == "measured":
                    continue
            curve[raw_n] = _resource_frontier_skip_cell(
                arguments,
                family=shard.family,
                mode=mode,
                final_multiplicity=final_multiplicity,
                failed_at=frontier,
            )
            changed = True
    return changed


def _dependency_skip_cell(
    arguments: argparse.Namespace,
    *,
    shard: Shard,
    mode: str,
    final_multiplicity: int,
    dependency_name: str,
    dependency_mode: str,
    dependency_cell: Mapping[str, Any],
) -> dict[str, object]:
    failed_at = _resource_frontier_failure_at(dependency_cell, final_multiplicity)
    dependency: dict[str, object] = {
        "shard": dependency_name,
        "family": shard.family,
        "mode": dependency_mode,
        "n": final_multiplicity,
        "status": str(dependency_cell.get("status")),
    }
    if failed_at is not None:
        dependency["resource_failure_at_n"] = failed_at
    return study._cell_base(
        shard.family,
        study.MODE_BY_KEY[mode],
        final_multiplicity,
        sum_helicities=arguments.compare_helicity_sums,
    ) | {
        "status": "skipped",
        "failure_category": "dependency-unavailable",
        "failure_reason": (
            f"{dependency_mode} baseline/input unavailable because "
            f"{dependency_name} recorded "
            f"{dependency_cell.get('status')} at n={final_multiplicity}"
        ),
        "censors_higher_multiplicities": False,
        "dependency": dependency,
    }


def _apply_dependency_skips(
    arguments: argparse.Namespace,
    run_directory: Path,
    report: dict[str, Any],
    shard: Shard,
    *,
    multiplicities: Sequence[int],
) -> bool:
    if shard.dependency is None:
        return False
    dependency_name, dependency_mode = shard.dependency
    dependency_shard = SHARD_BY_NAME[dependency_name]
    source = _load_shard_report(arguments, run_directory, dependency_shard)
    if source is None:
        return False
    source_cells = source["cells"]
    target_cells = report["cells"]
    assert isinstance(source_cells, Mapping)
    assert isinstance(target_cells, dict)
    source_family = source_cells[shard.family]
    target_family = target_cells[shard.family]
    assert isinstance(source_family, Mapping)
    assert isinstance(target_family, dict)
    dependency_curve = source_family[dependency_mode]
    assert isinstance(dependency_curve, Mapping)
    changed = False
    for final_multiplicity in multiplicities:
        if not _selected_cells_terminal_for_orchestrator(
            source,
            family=shard.family,
            modes=(dependency_mode,),
            multiplicities=(final_multiplicity,),
        ):
            continue
        dependency_cell = dependency_curve.get(str(final_multiplicity))
        if (
            not isinstance(dependency_cell, Mapping)
            or dependency_cell.get("status") == "measured"
        ):
            continue
        for mode in _shard_owned_modes(arguments, run_directory, shard):
            curve = target_family[mode]
            assert isinstance(curve, dict)
            if str(final_multiplicity) in curve:
                continue
            curve[str(final_multiplicity)] = _dependency_skip_cell(
                arguments,
                shard=shard,
                mode=mode,
                final_multiplicity=final_multiplicity,
                dependency_name=dependency_name,
                dependency_mode=dependency_mode,
                dependency_cell=dependency_cell,
            )
            changed = True
    return changed


def _load_shard_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    *,
    required: bool = False,
    modes: Sequence[str] | None = None,
    allow_partial_modes: bool = False,
) -> dict[str, Any] | None:
    shard_modes = (
        tuple(modes)
        if modes is not None
        else _shard_modes(arguments, run_directory, shard)
    )
    sources: list[tuple[dict[str, Any], int | None]] = []
    aggregate_path = _shard_report_path(run_directory, shard)
    if aggregate_path.is_file():
        aggregate = _load_compatible_shard_report(
            arguments,
            run_directory,
            shard,
            aggregate_path,
            final_multiplicity=None,
            modes=shard_modes,
            allow_partial_modes=allow_partial_modes,
        )
        if aggregate is not None:
            sources.append((aggregate, None))
    cell_reports: dict[Path, int] = {}
    for path in (run_directory / "shards" / shard.name / "cells").glob(
        "n*/study/runs/*/report.json"
    ):
        raw_n = path.parent.parent.parent.parent.name
        if raw_n.startswith("n") and raw_n[1:].isdigit():
            cell_reports[path] = int(raw_n[1:])
    for path in (run_directory / "shards" / shard.name / "mode-cells").glob(
        "*/n*/study/runs/*/report.json"
    ):
        raw_n = path.parent.parent.parent.parent.name
        if raw_n.startswith("n") and raw_n[1:].isdigit():
            cell_reports[path] = int(raw_n[1:])
    for final_multiplicity in _requested_multiplicities(arguments, run_directory):
        legacy_path = _shard_report_path(run_directory, shard, final_multiplicity)
        if legacy_path.is_file():
            cell_reports[legacy_path] = final_multiplicity
        for mode in shard_modes:
            if mode not in shard.owned_modes:
                continue
            work = ShardWork(shard, final_multiplicity, mode)
            work_path = _work_report_path(run_directory, work)
            if work_path.is_file():
                cell_reports[work_path] = final_multiplicity
    for path, final_multiplicity in sorted(
        cell_reports.items(), key=lambda item: (item[1], str(item[0]))
    ):
        if not path.is_file():
            continue
        report = _load_compatible_shard_report(
            arguments,
            run_directory,
            shard,
            path,
            final_multiplicity=final_multiplicity,
            modes=shard_modes,
            allow_partial_modes=allow_partial_modes,
        )
        if report is not None:
            sources.append((report, final_multiplicity))
    if not sources:
        if required:
            raise ProfilingError(f"required shard report is missing: {aggregate_path}")
        return None
    curve_sources: dict[str, dict[str, dict[str, object]]] = {
        shard.family: {mode: {} for mode in shard_modes}
    }
    target_family = curve_sources[shard.family]
    for source, source_multiplicity in sources:
        cells = source["cells"]
        assert isinstance(cells, Mapping)
        family_cells = cells[shard.family]
        assert isinstance(family_cells, Mapping)
        for mode in shard_modes:
            source_curve = family_cells.get(mode)
            if not isinstance(source_curve, Mapping):
                continue
            target_curve = target_family[mode]
            if source_multiplicity is None:
                for raw_n, cell in source_curve.items():
                    if (
                        isinstance(raw_n, str)
                        and raw_n.isdigit()
                        and isinstance(cell, Mapping)
                        and not _obsolete_otf_protocol_scope_cell(
                            arguments,
                            mode=mode,
                            raw_n=raw_n,
                            cell=cell,
                        )
                    ):
                        _merge_cell_record(target_curve, raw_n, cell)
                continue
            cell = source_curve.get(str(source_multiplicity))
            if isinstance(cell, Mapping) and not _obsolete_otf_protocol_scope_cell(
                arguments,
                mode=mode,
                raw_n=str(source_multiplicity),
                cell=cell,
            ):
                _merge_cell_record(
                    target_curve, str(source_multiplicity), cell
                )
    report = study.compose_report(
        _shard_arguments(arguments, run_directory, shard, modes=shard_modes),
        curve_sources,
    )
    _apply_resource_frontier_censoring(
        arguments,
        report,
        shard,
        modes=shard_modes,
        multiplicities=_requested_multiplicities(arguments, run_directory),
    )
    return study.compose_report(
        _shard_arguments(arguments, run_directory, shard, modes=shard_modes),
        report["cells"],
    )


def _publish_shard_report(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> dict[str, Any] | None:
    report = _load_shard_report(arguments, run_directory, shard)
    if report is not None:
        _write_json_atomic(_shard_report_path(run_directory, shard), report)
    return report


def _selected_shard_complete(
    arguments: argparse.Namespace,
    shard: Shard,
    report: Mapping[str, Any] | None,
    *,
    multiplicities: Sequence[int] | None = None,
) -> bool:
    return (
        report is not None
        and not str(report.get("status", "")).startswith("stopped")
        and _selected_cells_terminal_for_orchestrator(
            report,
            family=shard.family,
            modes=_counted_owned_modes(arguments, _run_directory(arguments), shard),
            multiplicities=(
                _selection(arguments)
                if multiplicities is None
                else tuple(multiplicities)
            ),
        )
    )


def _selected_work_complete(
    arguments: argparse.Namespace,
    run_directory: Path,
    work: ShardWork,
    report: Mapping[str, Any] | None,
) -> bool:
    return report is not None and _selected_cells_terminal_for_orchestrator(
        report,
        family=work.shard.family,
        modes=_work_owned_modes(arguments, run_directory, work),
        multiplicities=(work.final_multiplicity,),
    )


def _phase_work_items(
    arguments: argparse.Namespace, run_directory: Path, phase_number: int
) -> tuple[ShardWork, ...]:
    works = [
        ShardWork(shard, final_multiplicity, owned_mode)
        for final_multiplicity in _requested_multiplicities(arguments, run_directory)
        for shard in _scheduled_shards(arguments, run_directory)
        if shard.phase == phase_number
        for owned_mode in _counted_owned_modes(arguments, run_directory, shard)
        if final_multiplicity
        in _work_measurement_multiplicities(
            arguments,
            run_directory,
            ShardWork(shard, final_multiplicity, owned_mode),
        )
    ]
    return tuple(
        sorted(
            works,
            key=lambda work: (
                work.final_multiplicity,
                0 if work.shard.name == "gg-amplicol" else 1,
                work.shard.name,
            ),
        )
    )


def _scheduled_phase_numbers(
    arguments: argparse.Namespace, run_directory: Path
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                shard.phase
                for shard in _scheduled_shards(arguments, run_directory)
                if _counted_owned_modes(arguments, run_directory, shard)
            }
        )
    )


def _work_dependency_ready(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> bool:
    if work.shard.dependency is None:
        return True
    dependency_name, dependency_mode = work.shard.dependency
    dependency_shard = SHARD_BY_NAME[dependency_name]
    source = _load_shard_report(arguments, run_directory, dependency_shard)
    return source is not None and _selected_cells_terminal_for_orchestrator(
        source,
        family=work.shard.family,
        modes=(dependency_mode,),
        multiplicities=(work.final_multiplicity,),
    )


def _empty_cell_report(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> dict[str, Any]:
    return study.compose_report(_work_arguments(arguments, run_directory, work), {})


def _load_cell_report(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> dict[str, Any]:
    path = _work_report_path(run_directory, work)
    if not path.is_file():
        legacy_path = _shard_report_path(
            run_directory, work.shard, work.final_multiplicity
        )
        if legacy_path.is_file() and legacy_path != path:
            return _load_single_shard_report(
                arguments,
                run_directory,
                work.shard,
                legacy_path,
                final_multiplicity=work.final_multiplicity,
                modes=_work_modes(arguments, run_directory, work),
            )
        return _empty_cell_report(arguments, run_directory, work)
    return _load_single_shard_report(
        arguments,
        run_directory,
        work.shard,
        path,
        final_multiplicity=work.final_multiplicity,
        modes=_work_modes(arguments, run_directory, work),
    )


def _seed_cell_inputs(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> None:
    report = _load_cell_report(arguments, run_directory, work)
    target_cells = report["cells"]
    assert isinstance(target_cells, dict)
    target_family = target_cells[work.shard.family]
    assert isinstance(target_family, dict)

    existing = _load_shard_report(arguments, run_directory, work.shard)
    if existing is not None:
        existing_cells = existing["cells"]
        assert isinstance(existing_cells, Mapping)
        existing_family = existing_cells[work.shard.family]
        assert isinstance(existing_family, Mapping)
        for mode in _work_owned_modes(arguments, run_directory, work):
            source_curve = existing_family[mode]
            target_curve = target_family[mode]
            assert isinstance(source_curve, Mapping)
            assert isinstance(target_curve, dict)
            target_curve.update(copy.deepcopy(dict(source_curve)))

    if work.shard.dependency is not None:
        dependency_name, dependency_mode = work.shard.dependency
        dependency_shard = SHARD_BY_NAME[dependency_name]
        source = _load_shard_report(
            arguments, run_directory, dependency_shard, required=True
        )
        if not _selected_cells_terminal_for_orchestrator(
            source,
            family=work.shard.family,
            modes=(dependency_mode,),
            multiplicities=(work.final_multiplicity,),
        ):
            raise ProfilingError(
                f"dependency {dependency_name} has not completed "
                f"{work.shard.family}/{dependency_mode}/n{work.final_multiplicity}"
            )
        source_cells = source["cells"]
        assert isinstance(source_cells, Mapping)
        source_family = source_cells[work.shard.family]
        assert isinstance(source_family, Mapping)
        dependency_curve = source_family[dependency_mode]
        target_curve = target_family[dependency_mode]
        assert isinstance(dependency_curve, Mapping)
        assert isinstance(target_curve, dict)
        dependency_cell = dependency_curve.get(str(work.final_multiplicity))
        if isinstance(dependency_cell, Mapping):
            target_curve[str(work.final_multiplicity)] = copy.deepcopy(
                dict(dependency_cell)
            )

    report["status"] = "running"
    _write_json_atomic(_work_report_path(run_directory, work), report)


def _seed_protocol_scope(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> None:
    owned_modes = _shard_owned_modes(arguments, run_directory, shard)
    if not any(mode.startswith("otf-") for mode in owned_modes):
        return
    target = _shard_report_path(run_directory, shard)
    namespace = _shard_arguments(arguments, run_directory, shard)
    report = _load_shard_report(
        arguments, run_directory, shard
    ) or study.compose_report(namespace, {})
    if study.apply_protocol_scope_cells(
        report,
        family=shard.family,
        modes=owned_modes,
        multiplicities=_requested_multiplicities(arguments, run_directory),
        maximum_multiplicity=arguments.otf_max_multiplicity,
    ):
        report["status"] = "running"
        _write_json_atomic(target, report)


def _publication_profile(arguments: argparse.Namespace) -> bool:
    run_directory = _run_directory(arguments)
    namespace = _master_arguments(
        _manifest_backed_arguments(arguments, run_directory), run_directory
    )
    return publication.campaign_policy_is_publication_profile(
        study.compose_report(namespace, {})
    )


def _report_publication_profile(report: Mapping[str, Any]) -> bool:
    return report.get(
        "status"
    ) in TERMINAL_STATUSES and publication.campaign_policy_is_publication_profile(
        report
    )


def _compose_master(
    arguments: argparse.Namespace,
    run_directory: Path,
    *,
    active: Sequence[str] = (),
    halt_reason: str | None = None,
) -> dict[str, Any]:
    compose_arguments = _composition_arguments(arguments, run_directory)
    namespace = _master_arguments(compose_arguments, run_directory)
    reports: dict[str, dict[str, Any] | None] = {}
    shard_status: dict[str, str] = {}
    for shard in SHARDS:
        shard_modes = _shard_modes(compose_arguments, run_directory, shard)
        shard_report = (
            _load_shard_report(
                compose_arguments,
                run_directory,
                shard,
                modes=shard_modes,
                allow_partial_modes=True,
            )
            if shard_modes
            else None
        )
        reports[shard.name] = shard_report
        shard_status[shard.name] = (
            str(shard_report.get("status")) if shard_report is not None else "pending"
        )
    curve_sources: dict[str, dict[str, Mapping[str, object]]] = {
        "gg": {},
        "ddbar": {},
    }
    for family in FAMILIES:
        for mode in publication.FAMILY_MODES[family]:
            owner = SHARD_BY_NAME[MODE_OWNER[(family, mode)]]
            owner_report = reports[owner.name]
            if owner_report is None:
                continue
            owner_cells = owner_report["cells"]
            assert isinstance(owner_cells, Mapping)
            owner_family = owner_cells[family]
            assert isinstance(owner_family, Mapping)
            if mode not in owner_family:
                continue
            curve = owner_family[mode]
            assert isinstance(curve, Mapping)
            curve_sources[family][mode] = curve

    report = study.compose_report(
        namespace,
        curve_sources,
        halt_reason=halt_reason,
    )
    report["measurement_host"] = madgraph.measurement_host_identity()
    policy = report["policy"]
    assert isinstance(policy, dict)
    measurement = policy["measurement"]
    assert isinstance(measurement, dict)
    measurement["schedule_order"] = (
        "dependency-ordered-parallel-shard-multiplicity-workers"
    )
    plot = policy.setdefault("plot", {})
    assert isinstance(plot, dict)
    if arguments.compare_helicity_sums:
        plot["notes"] = [
            "Complete physical-helicity-summed matrix-element workload; "
            "MadGraph standalone uses generated SMATRIX with USERHEL=-1; "
            "warmed GOODHEL pruning remains enabled."
        ]
    elif not _publication_profile(compose_arguments):
        plot["notes"] = [
            "Cluster scan with configured per-cell resource caps; this is not the "
            "canonical 30-GiB/one-hour publication profile."
        ]
    report["profiling_orchestration"] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "shard_status": shard_status,
        "active_shards": list(active),
        "total_core_budget": arguments.cores,
        "candidate_optimization_cores": arguments.candidate_cores,
    }
    return report


def _publish_master(
    arguments: argparse.Namespace,
    run_directory: Path,
    *,
    active: Sequence[str] = (),
    halt_reason: str | None = None,
) -> dict[str, Any]:
    report = _compose_master(
        arguments, run_directory, active=active, halt_reason=halt_reason
    )
    _write_json_atomic(_master_report_path(run_directory), report)
    return report


def _completed_cells(report: Mapping[str, Any]) -> int:
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        return 0
    return sum(
        len(mode_cells)
        for family_cells in cells.values()
        if isinstance(family_cells, Mapping)
        for mode_cells in family_cells.values()
        if isinstance(mode_cells, Mapping)
    )


def _selected_completed_cells(
    arguments: argparse.Namespace,
    report: Mapping[str, Any],
    *,
    multiplicities: Sequence[int] | None = None,
) -> int:
    run_directory = _run_directory(arguments)
    requested = (
        _selection(arguments) if multiplicities is None else tuple(multiplicities)
    )
    return sum(
        _selected_cells_terminal_for_orchestrator(
            report,
            family=shard.family,
            modes=(mode,),
            multiplicities=(final_multiplicity,),
        )
        for shard in _scheduled_shards(arguments, run_directory)
        for mode in _counted_owned_modes(arguments, run_directory, shard)
        for final_multiplicity in requested
    )


def _selected_master_complete(
    arguments: argparse.Namespace,
    report: Mapping[str, Any],
    *,
    multiplicities: Sequence[int] | None = None,
) -> bool:
    run_directory = _run_directory(arguments)
    return not str(report.get("status", "")).startswith("stopped") and all(
        _selected_cells_terminal_for_orchestrator(
            report,
            family=shard.family,
            modes=_counted_owned_modes(arguments, run_directory, shard),
            multiplicities=(
                _selection(arguments)
                if multiplicities is None
                else tuple(multiplicities)
            ),
        )
        for shard in _scheduled_shards(arguments, run_directory)
    )


def _selected_pending_cells(
    arguments: argparse.Namespace,
    report: Mapping[str, Any],
    *,
    multiplicities: Sequence[int] | None = None,
) -> int:
    run_directory = _run_directory(arguments)
    requested = (
        _selection(arguments) if multiplicities is None else tuple(multiplicities)
    )
    return sum(
        not _selected_cells_terminal_for_orchestrator(
            report,
            family=shard.family,
            modes=(mode,),
            multiplicities=(final_multiplicity,),
        )
        for shard in _scheduled_shards(arguments, run_directory)
        for mode in _counted_owned_modes(arguments, run_directory, shard)
        for final_multiplicity in requested
    )


def _run_checked(
    command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> None:
    completed = subprocess.run(
        tuple(command),
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        stdout = completed.stdout.strip()
        detail = (
            f"command failed with status {completed.returncode}: {shlex.join(command)}"
        )
        if stdout:
            detail = f"{detail}\nstdout:\n{stdout}"
        raise ProfilingError(detail)


def _source_snapshot(path: Path) -> tuple[bytes, dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        report = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ProfilingError(
            f"cannot freeze campaign report {path}: {error}"
        ) from error
    if not isinstance(report, dict):
        raise ProfilingError("campaign report must be a JSON object")
    return raw, report, hashlib.sha256(raw).hexdigest()


def _fallback_render_candidates() -> tuple[Path, ...]:
    """Return legacy/current campaign reports eligible for implicit rendering."""

    study_root = CANONICAL_RESULTS_ROOT / "fft-scaling-study"
    profiling_root = CANONICAL_RESULTS_ROOT / "fft-profiling" / "runs"
    candidates = {
        *study_root.joinpath("data").glob("campaign-report*.json"),
        *study_root.joinpath("raw", "runs").glob("*/report.json"),
        *profiling_root.glob("*/master/runs/*/report.json"),
        *profiling_root.glob("*/composites/*/report.json"),
        *profiling_root.glob("*/source-snapshots/*/report.json"),
    }
    # A render generation is an immutable report snapshot.  Discover the few
    # structural nesting depths produced by the orchestrator and isolated
    # extension tooling without recursively walking generated artifacts.
    for depth in range(1, 4):
        prefix = "/".join("*" for _ in range(depth))
        candidates.update(
            profiling_root.glob(f"{prefix}/render/generations/*/report.json")
        )
    return tuple(sorted((_absolute(path) for path in candidates), key=str))


def _fallback_madgraph_candidates() -> tuple[Path, ...]:
    """Return discoverable terminal and partial MadGraph overlays."""

    study_root = CANONICAL_RESULTS_ROOT / "fft-scaling-study"
    profiling_root = CANONICAL_RESULTS_ROOT / "fft-profiling" / "runs"
    candidates = {
        *study_root.joinpath("data").glob("*madgraph*overlay*.json"),
        *profiling_root.glob("*/overlay.json"),
        *profiling_root.glob("*/overlay.progress.json"),
        *profiling_root.glob("*/madgraph/overlay.json"),
        *profiling_root.glob("*/madgraph/overlay.progress.json"),
    }
    return tuple(sorted((_absolute(path) for path in candidates), key=str))


def _render_inventory(
    report: Mapping[str, Any], *, requested: set[int]
) -> tuple[int, int, int, int, int]:
    measured: set[tuple[str, str, int]] = set()
    recorded: set[tuple[str, str, int]] = set()
    madgraph_measured: set[tuple[str, str, int]] = set()
    for section in ("cells", "runtime_series"):
        families = report.get(section)
        if not isinstance(families, Mapping):
            continue
        for family, family_cells in families.items():
            if not isinstance(family_cells, Mapping):
                continue
            for mode, mode_cells in family_cells.items():
                if not isinstance(mode_cells, Mapping):
                    continue
                for raw_n, cell in mode_cells.items():
                    if not str(raw_n).isdigit() or not isinstance(cell, Mapping):
                        continue
                    identity = (str(family), str(mode), int(raw_n))
                    recorded.add(identity)
                    if cell.get("status") == "measured":
                        measured.add(identity)
                        if mode == "madgraph-standalone":
                            madgraph_measured.add(identity)
    return (
        len({identity for identity in measured if identity[2] in requested}),
        len({identity for identity in madgraph_measured if identity[2] in requested}),
        len({identity for identity in recorded if identity[2] in requested}),
        len(measured),
        len(recorded),
    )


def _render_candidate_score(
    arguments: argparse.Namespace,
    report: Mapping[str, Any],
    *,
    source: Path,
    overlay: Path | None,
) -> tuple[int, int, int, int, int, int, int, int, str, str]:
    requested = set(_selection(arguments))
    policy = report.get("policy")
    multiplicities = (
        policy.get("final_state_multiplicities")
        if isinstance(policy, Mapping)
        else None
    )
    observed = (
        {int(value) for value in multiplicities}
        if isinstance(multiplicities, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in multiplicities
        )
        else set()
    )
    measured, madgraph_measured, recorded, measured_all, recorded_all = (
        _render_inventory(report, requested=requested)
    )
    overlay_kind = ""
    if overlay is not None:
        with contextlib.suppress(ProfilingError):
            overlay_kind = str(
                _load_json(overlay, context="MadGraph render overlay").get("kind", "")
            )
    return (
        measured,
        madgraph_measured,
        measured_all,
        recorded,
        recorded_all,
        int(requested <= observed),
        len(observed & requested),
        int(overlay_kind == madgraph.KIND),
        str(source),
        str(overlay or ""),
    )


def _compatible_render_report(
    arguments: argparse.Namespace, path: Path
) -> dict[str, Any] | None:
    try:
        _raw, report, _digest = _source_snapshot(path)
        if (
            report.get("kind") != study.KIND
            or report.get("schema_version") != study.SCHEMA_VERSION
            or not isinstance(report.get("cells"), Mapping)
        ):
            return None
        _validate_render_workload(arguments, report)
        if report.get("measurement_host") is None and _has_madgraph_series(report):
            _append_plot_note(report, LEGACY_MADGRAPH_NOTE)
    except (OSError, ProfilingError, TypeError, ValueError):
        return None
    return report


def _implicit_render_selection(
    arguments: argparse.Namespace, run_directory: Path
) -> RenderSelection:
    primary = _master_report_path(run_directory)
    candidates = {primary, *_fallback_render_candidates()}
    overlays = _fallback_madgraph_candidates()
    ranked: list[
        tuple[
            tuple[int, int, int, int, int, int, int, int, str, str],
            RenderSelection,
        ]
    ] = []
    for source in sorted(candidates, key=str):
        if not source.is_file():
            continue
        report = _compatible_render_report(arguments, source)
        if report is None:
            continue
        variants: list[tuple[dict[str, Any], Path | None]] = [(report, None)]
        for overlay in overlays:
            enriched = copy.deepcopy(report)
            try:
                _attach_partial_overlay(enriched, overlay)
            except ProfilingError:
                continue
            variants.append((enriched, overlay))
        for candidate_report, overlay in variants:
            ranked.append(
                (
                    _render_candidate_score(
                        arguments,
                        candidate_report,
                        source=source,
                        overlay=overlay,
                    ),
                    RenderSelection(source=source, overlay=overlay),
                )
            )
    if not ranked:
        return RenderSelection(source=primary, overlay=None)
    return max(ranked, key=lambda item: item[0])[1]


def _fallback_render_source(arguments: argparse.Namespace) -> Path | None:
    """Choose the richest compatible report for an otherwise bare render."""

    primary = _master_report_path(_run_directory(arguments))
    selection = _implicit_render_selection(arguments, _run_directory(arguments))
    return selection.source if selection.source != primary else None


def _render_source(arguments: argparse.Namespace, run_directory: Path) -> Path:
    if arguments.campaign_report is not None:
        return _absolute(arguments.campaign_report)
    primary = _master_report_path(run_directory)
    if arguments.output is not None:
        return primary
    return _implicit_render_selection(arguments, run_directory).source


def _validate_render_workload(
    arguments: argparse.Namespace, report: Mapping[str, Any]
) -> None:
    try:
        observed = study.report_helicity_workload(report)
    except study.StudyError as error:
        raise ProfilingError(f"invalid campaign helicity workload: {error}") from error
    expected = _helicity_workload(arguments)
    if observed != expected:
        raise ProfilingError(
            f"requested {expected!r} helicity workload but the frozen campaign "
            f"report declares {observed!r}; select the matching "
            "--compare-helicity-sums setting"
        )


def _has_madgraph_series(report: Mapping[str, Any]) -> bool:
    for section in ("cells", "runtime_series"):
        families = report.get(section)
        if not isinstance(families, Mapping):
            continue
        if any(
            isinstance(family, Mapping) and "madgraph-standalone" in family
            for family in families.values()
        ):
            return True
    return False


def _append_plot_note(report: dict[str, Any], note: str) -> None:
    policy = report.get("policy")
    if not isinstance(policy, dict):
        return
    plot = policy.setdefault("plot", {})
    if not isinstance(plot, dict):
        return
    raw_notes = plot.get("notes")
    notes = [str(item) for item in raw_notes] if isinstance(raw_notes, list) else []
    if note not in notes:
        notes.append(note)
    plot["notes"] = notes


def _legacy_madgraph_host_matches_current(value: object) -> bool:
    legacy_keys = {"system", "machine", "python"}
    if not isinstance(value, Mapping) or set(value) != legacy_keys:
        return False
    current = madgraph.measurement_host_identity()
    return all(value[key] == current[key] for key in legacy_keys)


def _attach_partial_overlay(report: dict[str, Any], path: Path) -> None:
    raw, payload, digest = _source_snapshot(path)
    report.setdefault("summary", {})
    try:
        if payload.get("kind") == madgraph.PROGRESS_KIND:
            with tempfile.TemporaryDirectory(prefix="fft-mg-progress-") as directory:
                frozen_progress = Path(directory) / "progress.json"
                frozen_progress.write_bytes(raw)
                payload = madgraph.load_runtime_progress(frozen_progress)
        raw_campaign_host = report.get("measurement_host")
        if raw_campaign_host is not None:
            campaign_host = madgraph.validate_measurement_host(
                raw_campaign_host, context="campaign measurement_host"
            )
            overlay_host = madgraph.validate_measurement_host(
                payload.get("host"), context="MadGraph overlay host"
            )
            if campaign_host != overlay_host:
                raise madgraph.SelectedMadGraphError(
                    "campaign and MadGraph overlay use different measurement hosts"
                )
        elif not _legacy_madgraph_host_matches_current(payload.get("host")):
            raise madgraph.SelectedMadGraphError(
                "legacy MadGraph overlay does not match this workstation"
            )
        selected.apply_runtime_series_source(
            report,
            selected.SourceReport(
                key=(
                    "madgraph-progress"
                    if payload.get("kind") == madgraph.PROGRESS_KIND
                    else "madgraph-overlay"
                ),
                path=path,
                sha256=digest,
                payload=payload,
            ),
        )
        if raw_campaign_host is None:
            _append_plot_note(report, LEGACY_MADGRAPH_NOTE)
        overlay_policy = payload.get("policy")
        if (
            isinstance(overlay_policy, Mapping)
            and madgraph._declared_helicity_workload(overlay_policy) == "sum"
        ):
            report_policy = report.get("policy")
            if isinstance(report_policy, dict):
                plot = report_policy.setdefault("plot", {})
                if isinstance(plot, dict):
                    raw_notes = plot.get("notes")
                    notes = (
                        [str(note) for note in raw_notes]
                        if isinstance(raw_notes, list)
                        else []
                    )
                    notes = [
                        note
                        for note in notes
                        if not (
                            "MadGraph" in note
                            and (
                                "omitted" in note.lower()
                                or "fixed-helicity" in note.lower()
                            )
                        )
                    ]
                    replacement = (
                        "MadGraph standalone uses generated SMATRIX with "
                        "USERHEL=-1; warmed GOODHEL pruning remains enabled."
                    )
                    if replacement not in notes:
                        notes.append(replacement)
                    plot["notes"] = notes
    except (
        KeyError,
        selected.CompositeError,
        madgraph.SelectedMadGraphError,
    ) as error:
        raise ProfilingError(f"invalid MadGraph overlay {path}: {error}") from error


def _matching_madgraph_overlay(
    arguments: argparse.Namespace,
    run_directory: Path,
    *,
    require_exact: bool = True,
) -> Path | None:
    path = _madgraph_overlay_path(run_directory)
    if not path.is_file():
        return None
    try:
        payload = _load_json(path, context="MadGraph overlay")
    except ProfilingError:
        return None
    policy = payload.get("policy")
    raw_multiplicities = (
        policy.get("final_state_multiplicities")
        if isinstance(policy, Mapping)
        else None
    )
    try:
        overlay_workload = (
            madgraph._declared_helicity_workload(policy)
            if isinstance(policy, Mapping)
            else None
        )
    except madgraph.SelectedMadGraphError:
        return None
    try:
        overlay_host = madgraph.validate_measurement_host(
            payload.get("host"), context="MadGraph overlay host"
        )
    except madgraph.SelectedMadGraphError:
        if require_exact:
            return None
        # Historical overlays remain usable for explicitly nonterminal
        # anytime rendering, but can never enter the strict final merger.
        if not _legacy_madgraph_host_matches_current(payload.get("host")):
            return None
        overlay_host = None
    if (
        not isinstance(policy, Mapping)
        or overlay_workload != _helicity_workload(arguments)
        or (
            overlay_host is not None
            and overlay_host != madgraph.measurement_host_identity()
        )
        or policy.get("maximum_measured_multiplicity")
        != madgraph.MAX_PROTOCOL_MEASURED_MULTIPLICITY
        or policy.get("family_maximum_measured_multiplicity")
        != madgraph.protocol_measured_multiplicity_limits()
        or policy.get("higher_multiplicity_policy") != "not-applicable-protocol-scope"
    ):
        return None
    if (
        not isinstance(raw_multiplicities, list)
        or not raw_multiplicities
        or any(not isinstance(value, int) for value in raw_multiplicities)
    ):
        return None
    requested = _requested_multiplicities(arguments, run_directory)
    observed = tuple(raw_multiplicities)
    raw_families = policy.get("selected_process_families")
    if isinstance(raw_families, list) and set(raw_families) < set(
        _family_selection(arguments)
    ):
        return None
    if require_exact:
        if observed != requested:
            return None
    elif observed != tuple(value for value in requested if value in set(observed)):
        return None
    return path


def _madgraph_render_source(
    arguments: argparse.Namespace, run_directory: Path
) -> Path | None:
    progress = _matching_madgraph_progress(arguments, run_directory)
    if progress is not None and progress.get("status") == "running":
        return _madgraph_progress_path(run_directory)
    terminal = _matching_madgraph_overlay(arguments, run_directory, require_exact=False)
    if terminal is not None:
        return terminal
    if progress is not None:
        return _madgraph_progress_path(run_directory)
    return None


def _publish_render_generation(
    stage: Path, render_root: Path, digest: str, pdf_filename: str
) -> Path:
    generations = render_root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    destination = generations / f"{digest[:16]}-{uuid.uuid4().hex[:8]}"
    stage.replace(destination)
    current = render_root / "current"
    if current.exists() and not current.is_symlink():
        raise ProfilingError(
            f"refusing to replace non-symlink render pointer: {current}"
        )
    temporary_link = render_root / f".current-{uuid.uuid4().hex}.tmp"
    relative = os.path.relpath(destination, render_root)
    os.symlink(relative, temporary_link)
    os.replace(temporary_link, current)
    return current / pdf_filename


def _publish_canonical_pdf(source: Path, arguments: argparse.Namespace) -> Path:
    destination = _canonical_pdf_path(arguments)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source_path, destination_path in (
        (source, destination),
        (source.with_suffix(".json"), destination.with_suffix(".json")),
    ):
        if not source_path.is_file():
            raise ProfilingError(f"canonical render source is missing: {source_path}")
        temporary = destination_path.with_name(
            f".{destination_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, destination_path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    return destination


def render_snapshot(
    arguments: argparse.Namespace, *, renderer_preflight: bool = True
) -> Path:
    run_directory = _run_directory(arguments)
    lock = _acquire_render_lock(run_directory)
    try:
        return _render_snapshot_locked(
            arguments,
            run_directory=run_directory,
            renderer_preflight=renderer_preflight,
        )
    finally:
        lock.close()


def _render_snapshot_locked(
    arguments: argparse.Namespace,
    *,
    run_directory: Path,
    renderer_preflight: bool,
) -> Path:
    if arguments.campaign_report is not None:
        render_arguments = arguments
    elif arguments.output is not None:
        render_arguments = _composition_arguments(arguments, run_directory)
    else:
        render_arguments = _manifest_backed_arguments(arguments, run_directory)
    if (
        arguments.output is not None
        and arguments.campaign_report is None
        and _manifest_path(run_directory).is_file()
    ):
        _publish_master(render_arguments, run_directory)
    if arguments.campaign_report is not None:
        selection = RenderSelection(
            source=_absolute(arguments.campaign_report),
            overlay=(
                _absolute(arguments.madgraph_overlay)
                if arguments.madgraph_overlay is not None
                else None
            ),
        )
    elif arguments.output is not None:
        selection = RenderSelection(
            source=_master_report_path(run_directory),
            overlay=_madgraph_render_source(render_arguments, run_directory),
        )
    else:
        selection = _implicit_render_selection(render_arguments, run_directory)
    source = selection.source
    if not source.is_file():
        raise ProfilingError(
            f"no campaign snapshot yet: {source}; start the scan before --render"
        )
    if renderer_preflight:
        _preflight_renderer(render_arguments)
    raw, report, digest = _source_snapshot(source)
    _validate_render_workload(render_arguments, report)
    pdf_filename = _pdf_filename(render_arguments)
    terminal_overlay = (
        _matching_madgraph_overlay(render_arguments, run_directory)
        if arguments.campaign_report is None
        else None
    )
    selected_overlay_kind = ""
    if selection.overlay is not None and selection.overlay.is_file():
        with contextlib.suppress(ProfilingError):
            selected_overlay_kind = str(
                _load_json(
                    selection.overlay,
                    context="selected MadGraph render overlay",
                ).get("kind", "")
            )
    strict = bool(
        _report_publication_profile(report)
        and selection.overlay is not None
        and selection.overlay.is_file()
        and selected_overlay_kind == madgraph.KIND
        and (
            arguments.campaign_report is not None
            or (
                source == _master_report_path(run_directory)
                and selection.overlay == terminal_overlay
            )
        )
    )
    overlay = selection.overlay if strict else None
    partial_overlay = selection.overlay or (run_directory / "madgraph" / ".unavailable")

    render_root = run_directory / "render"
    render_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=render_root))
    try:
        if strict:
            # The strict merger checks that the campaign file lives inside its
            # recorded run_root.  Keep an immutable digest-named freeze beside
            # the source so all downstream provenance remains resolvable.
            frozen_source = source.parent / f"render-snapshot-{digest}.json"
            if frozen_source.is_file() and frozen_source.read_bytes() != raw:
                raise ProfilingError(
                    f"render snapshot digest collision: {frozen_source}"
                )
            if not frozen_source.is_file():
                temporary = frozen_source.with_suffix(".json.tmp")
                temporary.write_bytes(raw)
                temporary.replace(frozen_source)
            staged_report = stage / "report.json"
        else:
            frozen_source = stage / "report.json"
            frozen = copy.deepcopy(report)
            if _report_publication_profile(report) and not strict:
                # The candidate campaign is terminal, but the same-host MG
                # phase is not.  Keep anytime rendering visibly nonterminal.
                frozen["status"] = "running-madgraph-series"
            if partial_overlay.is_file():
                _attach_partial_overlay(frozen, partial_overlay)
            frozen["render_snapshot"] = {
                "status": "in-progress"
                if report.get("status") not in TERMINAL_STATUSES
                else "cluster-terminal",
                "source_report": str(source),
                "source_sha256": digest,
            }
            _write_json_atomic(frozen_source, frozen)
            staged_report = frozen_source
        staged_plots = stage / "plots"
        staged_pdf = stage / pdf_filename
        commands = _render_commands(
            render_arguments,
            source_report=frozen_source,
            staged_report=staged_report,
            staged_plots=staged_plots,
            staged_pdf=staged_pdf,
            strict=strict,
            madgraph_overlay=overlay,
        )
        render_environment = _renderer_environment(render_arguments)
        for command in commands:
            _run_checked(command, environment=render_environment)
        expected_plots = {
            f"fullcolor-{family}-{metric}.png"
            for family in ("gg", "ddbar")
            for metric in ("generation", "warm-runtime", "rss")
        }
        observed_plots = {
            path.name for path in staged_plots.glob("*.png") if path.stat().st_size > 0
        }
        staged_json = staged_pdf.with_suffix(".json")
        if observed_plots != expected_plots or not (
            staged_pdf.is_file() and staged_pdf.stat().st_size > 0
        ) or not (
            staged_json.is_file() and staged_json.stat().st_size > 0
        ):
            raise ProfilingError(
                "render commands did not produce one complete output set"
            )
        published = _publish_render_generation(stage, render_root, digest, pdf_filename)
        if _canonical_publication_enabled(render_arguments):
            _publish_canonical_pdf(published, render_arguments)
        return published
    except BaseException:
        if stage.is_dir():
            shutil.rmtree(stage)
        raise


def _safe_refresh(run_directory: Path) -> bool:
    _validate_refresh_target(run_directory)
    render_lock = _acquire_render_lock(run_directory)
    try:
        try:
            cache_lock = madgraph._acquire_cache_lock(
                _madgraph_cache_path(run_directory)
            )
        except madgraph.SelectedMadGraphError as error:
            raise ProfilingError(f"cannot refresh while {error}") from error
        try:
            return _safe_refresh_locked(run_directory)
        finally:
            cache_lock.close()
    finally:
        render_lock.close()


def _safe_refresh_locked(run_directory: Path) -> bool:
    _validate_refresh_target(run_directory)
    resolved = run_directory.resolve(strict=False)
    if not resolved.exists():
        return False
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise ProfilingError(
            "--refresh requires an existing recognized profiling manifest at "
            f"{manifest_path}"
        )
    manifest = _load_json(manifest_path, context="profiling refresh manifest")
    identity = manifest.get("identity")
    if (
        manifest.get("kind") != KIND
        or manifest.get("schema_version") != SCHEMA_VERSION
        or not isinstance(identity, Mapping)
        or identity.get("run_directory") != str(resolved)
    ):
        raise ProfilingError(
            f"refusing --refresh for unrecognized profiling directory: {resolved}"
        )
    shutil.rmtree(resolved)
    return True


def _validate_refresh_target(run_directory: Path) -> None:
    _reject_refresh_symlinks(run_directory)
    unsafe = {
        Path("/").resolve(),
        Path.home().resolve(),
        ROOT.resolve(),
        DEFAULT_RUNS_ROOT.resolve(strict=False),
    }
    resolved = run_directory.resolve(strict=False)
    if resolved in unsafe or len(resolved.parts) < 4:
        raise ProfilingError(f"refusing ambiguous --refresh target: {resolved}")


def _reject_refresh_symlinks(path: Path) -> None:
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ProfilingError(
                f"refusing --refresh through symlinked path component: {candidate}"
            )


def _command_available(command: str) -> bool:
    path = Path(command).expanduser()
    return (
        path.is_file() and os.access(path, os.X_OK)
        if path.parent != Path(".")
        else shutil.which(command) is not None
    )


def _dependency_status(arguments: argparse.Namespace) -> dict[str, Any]:
    amplicol = _absolute(arguments.amplicol_root)
    reference_fft = _absolute(arguments.reference_fft_root)
    madgraph = (
        _absolute(arguments.madgraph_root)
        if arguments.madgraph_root is not None
        else None
    )
    return {
        "python": {
            "path": str(arguments.python),
            "available": _command_available(str(arguments.python)),
        },
        "cxx": {
            "path": str(arguments.cxx),
            "available": _command_available(str(arguments.cxx)),
        },
        "fortran": {
            "path": str(arguments.fc),
            "available": _command_available(str(arguments.fc)),
        },
        "make": {
            "path": str(arguments.make),
            "available": _command_available(str(arguments.make)),
        },
        "amplicol_root": {
            "path": str(amplicol),
            "compatible": (amplicol / "process_list.py").is_file(),
            "probe": str(amplicol / "amplicol_color_probe"),
            "probe_available": (amplicol / "amplicol_color_probe").is_file(),
        },
        "reference_fft_root": {
            "path": str(reference_fft),
            "compatible": (reference_fft / "Benchmark" / "run_benchmark.py").is_file(),
        },
        "madgraph_root": {
            "path": str(madgraph) if madgraph is not None else None,
            "applicable": True,
            "compatible": bool(
                madgraph is not None
                and (madgraph / "bin" / "mg5_aMC").is_file()
                and (madgraph / "VERSION").is_file()
            ),
        },
    }


def _preflight(arguments: argparse.Namespace) -> None:
    status = _dependency_status(arguments)
    run_directory = _run_directory(arguments)
    requested_shards = _requested_shards(arguments, run_directory)
    for key in ("python", "cxx", "fortran"):
        if status[key]["available"] is not True:
            raise ProfilingError(
                f"required {key} executable is unavailable: {status[key]['path']}"
            )
    if _pyamplicol_runtime_required(requested_shards):
        _preflight_pyamplicol_runtime(arguments)
    needs_amplicol = any(
        "amplicol" in _shard_modes(arguments, run_directory, shard)
        for shard in requested_shards
    )
    if needs_amplicol:
        amplicol = status["amplicol_root"]
        if amplicol["compatible"] is not True:
            raise ProfilingError(
                f"invalid --amplicol-root: {amplicol['path']}; pass a compatible "
                "checkout or run `just dev-install --with-legacy-amplicol`"
            )
        if (
            not arguments.compare_helicity_sums
            and amplicol["probe_available"] is not True
            and not arguments.build_amplicol
        ):
            raise ProfilingError(
                "AmpliCol probe is missing; pass --build-amplicol to build it once: "
                f"{amplicol['probe']}"
            )
    needs_reference_fft = any(
        "reference-fft" in _shard_modes(arguments, run_directory, shard)
        for shard in requested_shards
    )
    if needs_reference_fft:
        reference_fft = status["reference_fft_root"]
        if reference_fft["compatible"] is not True:
            raise ProfilingError(
                "invalid --reference-fft-root: expected "
                f"Benchmark/run_benchmark.py below {reference_fft['path']}; pass a "
                "compatible checkout or run "
                "`just dev-install --with-reference-fft`"
            )
    if _madgraph_requested(arguments, run_directory):
        if status["make"]["available"] is not True:
            raise ProfilingError(
                f"required make executable is unavailable: {status['make']['path']}"
            )
        madgraph = status["madgraph_root"]
        if madgraph["compatible"] is not True:
            raise ProfilingError(
                "MadGraph installation is missing or incompatible; pass "
                "--madgraph-root PATH containing bin/mg5_aMC and VERSION"
            )
    _preflight_renderer(arguments)


def _preflight_renderer(arguments: argparse.Namespace) -> None:
    if not _command_available(str(arguments.python)):
        raise ProfilingError(
            f"renderer --python executable is unavailable: {arguments.python}"
        )
    renderer = subprocess.run(
        (
            str(arguments.python),
            "-c",
            "import matplotlib, platform, reportlab; print(platform.python_version())",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_renderer_environment(arguments),
    )
    if renderer.returncode != 0:
        install_command = shlex.join(
            (
                str(arguments.python),
                "-m",
                "pip",
                "install",
                *RENDER_REQUIREMENTS,
            )
        )
        raise ProfilingError(
            "plot/PDF dependencies are missing from --python; install them with "
            f"{install_command}"
        )
    selected_version = renderer.stdout.strip()
    driver_version = str(madgraph.measurement_host_identity()["python"])
    if selected_version != driver_version:
        raise ProfilingError(
            "--python uses Python "
            f"{selected_version!r}, but the profiling driver uses {driver_version!r}; "
            "run fft_profiling.py with the selected interpreter"
        )


def _renderer_environment(arguments: argparse.Namespace) -> dict[str, str]:
    cache = _run_directory(arguments) / "render" / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(cache)
    return environment


def _madgraph_timeout(arguments: argparse.Namespace) -> float:
    # The canonical publication overlay has historically used a strict 3595-s
    # MG limit while the pyAmpliCol cells use the strict 3600-s admission edge.
    # Custom cluster limits are transmitted without clamping.
    return (
        study.DEFAULT_TIME_LIMIT_SECONDS - 5.0
        if arguments.time_limit_seconds == study.DEFAULT_TIME_LIMIT_SECONDS
        else arguments.time_limit_seconds
    )


def _madgraph_command(
    arguments: argparse.Namespace, run_directory: Path
) -> tuple[str, ...]:
    if arguments.madgraph_root is None:
        raise ProfilingError("--madgraph-root is required for the same-host series")
    result = [
        str(arguments.python),
        str(MADGRAPH_TOOL),
        "--source-report",
        str(_madgraph_source_path(arguments, run_directory)),
        "--cache-dir",
        str(_madgraph_cache_path(run_directory)),
        "--output",
        str(_madgraph_overlay_path(run_directory)),
        "--mg5-root",
        str(_absolute(arguments.madgraph_root)),
        "--fc",
        str(arguments.fc),
        "--make",
        str(arguments.make),
        "--generation-timeout-seconds",
        format(_madgraph_timeout(arguments), ".17g"),
        "--memory-limit-gib",
        format(arguments.memory_limit_gib, ".17g"),
        "--target-seconds",
        format(arguments.target_seconds, ".17g"),
    ]
    selected_families = _family_selection(arguments)
    for multiplicity in _requested_multiplicities(arguments, run_directory):
        measured_by_requested_family = any(
            madgraph.protocol_measures_multiplicity(family, multiplicity)
            for family in selected_families
        )
        result.extend(
            (
                (
                    "--multiplicity"
                    if measured_by_requested_family
                    else "--protocol-scope-multiplicity"
                ),
                str(multiplicity),
            )
        )
    if len(selected_families) == 1:
        result.extend(("--family", selected_families[0]))
    if arguments.compare_helicity_sums:
        result.append("--compare-helicity-sums")
    return tuple(result)


def _madgraph_source_projection(
    arguments: argparse.Namespace,
    run_directory: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    policy = report.get("policy")
    if not isinstance(policy, Mapping):
        raise ProfilingError("MadGraph source report has no policy")
    measurement = policy.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ProfilingError("MadGraph source report has no measurement policy")
    alpha_s = measurement.get("alpha_s")
    if not isinstance(alpha_s, int | float) or isinstance(alpha_s, bool):
        raise ProfilingError("MadGraph source report has invalid alpha_s")
    workload = _helicity_workload(arguments)
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ProfilingError("MadGraph source report has no cells")
    requested = {
        str(value) for value in _requested_multiplicities(arguments, run_directory)
    }
    source_cells: dict[str, dict[str, dict[str, object]]] = {}
    for family in _family_selection(arguments):
        mode = madgraph.source_mode(family, _helicity_workload(arguments))
        family_cells = cells.get(family)
        if not isinstance(family_cells, Mapping):
            raise ProfilingError(f"MadGraph source report has no {family} cells")
        curve = family_cells.get(mode)
        if not isinstance(curve, Mapping):
            raise ProfilingError(f"MadGraph source report has no {family}/{mode} curve")
        source_cells[family] = {
            mode: {
                str(raw_n): copy.deepcopy(cell)
                for raw_n, cell in curve.items()
                if str(raw_n) in requested
            }
        }
    projected: dict[str, Any] = {
        "kind": madgraph.SOURCE_KIND,
        "schema_version": 1,
        "status": "complete",
        "policy": {
            "helicity_workload": workload,
            "measurement": {
                "alpha_s": float(alpha_s),
                "helicity_workload": workload,
                "warm_fixed_helicity": workload == "fixed",
                "warm_helicity_sum": workload == "sum",
            },
        },
        "cells": source_cells,
    }
    if isinstance(report.get("measurement_host"), Mapping):
        projected["measurement_host"] = copy.deepcopy(report["measurement_host"])
    return projected


def _freeze_madgraph_source(arguments: argparse.Namespace, run_directory: Path) -> Path:
    master_path = _master_report_path(run_directory)
    _raw, master, _ = _source_snapshot(master_path)
    if not _madgraph_source_ready(arguments, run_directory, master):
        raise ProfilingError(
            "cannot freeze MadGraph source before its selected source cells "
            "are complete"
        )
    source = _madgraph_source_projection(arguments, run_directory, master)
    destination = _madgraph_source_path(arguments, run_directory)
    if destination.is_file():
        existing = _load_json(destination, context="frozen MadGraph source")
        normalized_existing = _madgraph_source_projection(
            arguments, run_directory, existing
        )
        if normalized_existing == source:
            return destination
    _write_json_atomic(destination, source)
    return destination


def _madgraph_source_ready(
    arguments: argparse.Namespace,
    run_directory: Path,
    master: Mapping[str, Any] | None = None,
) -> bool:
    if not _madgraph_requested(arguments, run_directory):
        return False
    if master is None:
        master_path = _master_report_path(run_directory)
        if not master_path.is_file():
            return False
        _raw, master, _ = _source_snapshot(master_path)
    for family in _family_selection(arguments):
        source_mode = madgraph.source_mode(family, _helicity_workload(arguments))
        if not _selected_cells_terminal_for_orchestrator(
            master,
            family=family,
            modes=(source_mode,),
            multiplicities=_requested_multiplicities(arguments, run_directory),
        ):
            return False
    return True


def _ensure_madgraph_overlay(
    arguments: argparse.Namespace,
    run_directory: Path,
    dashboard: Dashboard,
) -> None:
    if not _madgraph_requested(arguments, run_directory):
        return
    _freeze_madgraph_source(arguments, run_directory)
    if _matching_madgraph_overlay(arguments, run_directory) is not None:
        _diagnostic(
            "Reusing the matching MadGraph overlay.",
            colorama.Fore.CYAN,
        )
        return
    _run_madgraph(arguments, run_directory, dashboard)


def _start_madgraph_overlay(
    arguments: argparse.Namespace,
    run_directory: Path,
) -> ActiveJob | None:
    if not _madgraph_requested(arguments, run_directory):
        return None
    _freeze_madgraph_source(arguments, run_directory)
    if _matching_madgraph_overlay(arguments, run_directory) is not None:
        _diagnostic(
            "Reusing the matching MadGraph overlay.",
            colorama.Fore.CYAN,
        )
        return None
    return _launch_madgraph_job(arguments, run_directory)


def _maybe_start_madgraph_overlay(
    arguments: argparse.Namespace,
    run_directory: Path,
    active: list[ActiveJob],
) -> bool:
    if (
        active
        or not _madgraph_requested(arguments, run_directory)
        or not _madgraph_source_ready(arguments, run_directory)
    ):
        return False
    job = _start_madgraph_overlay(arguments, run_directory)
    if job is None:
        return True
    active.append(job)
    return False


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_i = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds_i:02d}"


def _aggregate_rss(active: Sequence[ActiveJob]) -> int | None:
    if not active:
        return 0
    try:
        records = memory_watchdog.process_snapshot()
        return sum(job.sampler.sample(records).rss_bytes for job in active)
    except memory_watchdog.ProbeError:
        return None


def _matching_madgraph_progress(
    arguments: argparse.Namespace, run_directory: Path
) -> dict[str, Any] | None:
    path = _madgraph_progress_path(run_directory)
    if not path.is_file():
        return None
    try:
        payload = madgraph.load_runtime_progress(path)
    except madgraph.SelectedMadGraphError:
        return None
    policy = payload.get("policy")
    try:
        progress_workload = (
            madgraph._declared_helicity_workload(policy)
            if isinstance(policy, Mapping)
            else None
        )
        progress_host = madgraph.validate_measurement_host(
            payload.get("host"), context="MadGraph runtime progress host"
        )
    except madgraph.SelectedMadGraphError:
        return None
    requested_multiplicities = _requested_multiplicities(arguments, run_directory)
    raw_multiplicities = policy.get("final_state_multiplicities")
    if (
        not isinstance(policy, Mapping)
        or progress_workload != _helicity_workload(arguments)
        or progress_host != madgraph.measurement_host_identity()
        or policy.get("family_maximum_measured_multiplicity")
        != madgraph.protocol_measured_multiplicity_limits()
    ):
        return None
    if (
        not isinstance(raw_multiplicities, list)
        or not raw_multiplicities
        or any(not isinstance(value, int) for value in raw_multiplicities)
    ):
        return None
    observed_multiplicities = tuple(raw_multiplicities)
    observed_set = set(observed_multiplicities)
    if observed_multiplicities != tuple(
        value for value in requested_multiplicities if value in observed_set
    ):
        return None
    raw_families = policy.get("selected_process_families")
    if isinstance(raw_families, list) and set(raw_families) < set(
        _family_selection(arguments)
    ):
        return None
    return payload


def _madgraph_completed(arguments: argparse.Namespace, run_directory: Path) -> int:
    if not _madgraph_requested(arguments, run_directory):
        return 0
    terminal = _matching_madgraph_overlay(arguments, run_directory)
    progress_payload = _matching_madgraph_progress(arguments, run_directory)
    if progress_payload is not None and progress_payload.get("status") == "running":
        report = progress_payload
    elif terminal is not None:
        report = _load_json(terminal, context="MadGraph overlay")
    elif progress_payload is not None:
        report = progress_payload
    else:
        return 0
    series = report.get("runtime_series")
    if not isinstance(series, Mapping):
        return 0
    requested_families = set(_family_selection(arguments))
    requested_multiplicities = {
        str(value) for value in _requested_multiplicities(arguments, run_directory)
    }
    return sum(
        1
        for family, family_cells in series.items()
        if family in requested_families
        if isinstance(family_cells, Mapping)
        for mode_cells in family_cells.values()
        if isinstance(mode_cells, Mapping)
        for raw_n in mode_cells
        if str(raw_n) in requested_multiplicities
    )


class Dashboard:
    def __init__(
        self,
        *,
        total: int,
        core_budget: int,
        helicity_workload: str,
        batch_size: int,
    ) -> None:
        self.total = total
        self.core_budget = core_budget
        self.started = time.monotonic()
        self.workload = helicity_workload
        self.batch_size = batch_size
        self.bar = progressbar.ProgressBar(
            max_value=total,
            fd=sys.stderr,
            widgets=[
                progressbar.Variable("headline", width=96),
                " ",
                progressbar.Bar(),
                " ",
                progressbar.Percentage(),
            ],
        )
        self.bar.start()

    def update(
        self,
        completed: int,
        *,
        phase: str,
        active: Sequence[ActiveJob],
    ) -> None:
        elapsed = time.monotonic() - self.started
        rss = _aggregate_rss(active)
        rss_text = "RSS n/a" if rss is None else f"RSS {rss / 1024**3:.2f} GiB"
        active_cores = sum(job.claimed_cores for job in active)
        jobs = ",".join(job.detail for job in active) or "idle"
        headline = (
            f"{completed}/{self.total} cells | {self.workload} "
            f"batch={self.batch_size} | "
            f"{phase}: {jobs} | "
            f"cores active={active_cores}/{self.core_budget} | {rss_text} | "
            f"elapsed {_format_duration(elapsed)}"
        )
        self.bar.update(min(completed, self.total), headline=headline, force=True)

    def finish(self) -> None:
        self.bar.finish(dirty=True)


def _color_enabled(arguments: argparse.Namespace) -> bool:
    return (
        not arguments.no_color and "NO_COLOR" not in os.environ and sys.stderr.isatty()
    )


def _diagnostic(message: str, color: str = "") -> None:
    suffix = colorama.Style.RESET_ALL if color else ""
    print(f"{color}{message}{suffix}", file=sys.stderr, flush=True)


def _python_source_environment(arguments: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source if not existing else os.pathsep.join((source, existing))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    environment["FFT_ACCEPTANCE_PYTHON"] = str(arguments.python)
    return environment


def _pyamplicol_runtime_required(requested_shards: Sequence[Shard]) -> bool:
    return any(
        shard.family == "ddbar"
        or any(study.MODE_BY_KEY[mode].kind == "candidate" for mode in shard.modes)
        for shard in requested_shards
    )


def _preflight_pyamplicol_runtime(arguments: argparse.Namespace) -> None:
    completed = subprocess.run(
        (
            str(arguments.python),
            "-c",
            (
                "import pyamplicol; "
                "from pyamplicol.generation.phase_space import "
                "massive_rambo_final_state; "
                "print(pyamplicol.__version__)"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=_python_source_environment(arguments),
    )
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout).strip()
    if len(detail) > 4000:
        detail = detail[-4000:]
    raise ProfilingError(
        "selected --python cannot import pyAmpliCol from this checkout; "
        "rerun `just dev-install` in this workspace or fix the Python/native "
        f"runtime before profiling.\n{detail}"
    )


def _claimed_cores(arguments: argparse.Namespace, shard: Shard | ShardWork) -> int:
    if isinstance(shard, ShardWork):
        shard = shard.shard
    return arguments.candidate_cores if shard.candidate else 1


def _launch_shard(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> ActiveJob:
    command = _child_command(arguments, run_directory, work)
    log_root = _work_cell_root(run_directory, work) / "orchestrator-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = (log_root / "stdout.log").open("wb")
    stderr = (log_root / "stderr.log").open("wb")
    claimed = _claimed_cores(arguments, work)
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        environment[key] = str(claimed)
    try:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        stdout.close()
        stderr.close()
        raise
    return ActiveJob(
        shard=work.shard,
        final_multiplicity=work.final_multiplicity,
        owned_mode=work.owned_mode,
        command=command,
        process=process,
        stdout=stdout,
        stderr=stderr,
        sampler=memory_watchdog.ProcessTreeSampler(process.pid, process.pid),
        claimed_cores=claimed,
        detail=f"{work.shard.family}/{work.owned_mode}/n{work.final_multiplicity}",
    )


def _work_detail(
    arguments: argparse.Namespace, run_directory: Path, work: ShardWork
) -> str:
    report = _load_shard_report(arguments, run_directory, work.shard)
    for mode in _work_owned_modes(arguments, run_directory, work):
        if report is None or not _selected_cells_terminal_for_orchestrator(
            report,
            family=work.shard.family,
            modes=(mode,),
            multiplicities=(work.final_multiplicity,),
        ):
            return f"{work.shard.family}/{mode}/n{work.final_multiplicity}"
    return f"{work.shard.name}/n{work.final_multiplicity}/finalizing"


def _shard_detail(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> str:
    report = _load_shard_report(arguments, run_directory, shard)
    for final_multiplicity in _requested_multiplicities(arguments, run_directory):
        for mode in _counted_owned_modes(arguments, run_directory, shard):
            work = ShardWork(shard, final_multiplicity, mode)
            if not _selected_work_complete(arguments, run_directory, work, report):
                return _work_detail(arguments, run_directory, work)
    return f"{shard.name}/finalizing"


def _terminate_jobs(active: Sequence[ActiveJob]) -> None:
    for job in active:
        if job.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(job.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    for job in active:
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(subprocess.TimeoutExpired):
            job.process.wait(timeout=remaining)
    for job in active:
        if job.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(job.process.pid, signal.SIGKILL)
            job.process.wait()
        job.stdout.close()
        job.stderr.close()


@contextlib.contextmanager
def _termination_signal_handlers():
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise CampaignSignal(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _prepare_phase_shard_state(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> dict[str, Any] | None:
    _seed_protocol_scope(arguments, run_directory, shard)
    report = _load_shard_report(arguments, run_directory, shard)
    loaded = report is not None
    if report is None and shard.dependency is None:
        return None
    if report is None:
        report = study.compose_report(
            _shard_arguments(arguments, run_directory, shard), {}
        )
    changed = _apply_resource_frontier_censoring(
        arguments,
        report,
        shard,
        modes=_counted_owned_modes(arguments, run_directory, shard),
        multiplicities=_requested_multiplicities(arguments, run_directory),
    )
    changed = (
        _apply_dependency_skips(
            arguments,
            run_directory,
            report,
            shard,
            multiplicities=_requested_multiplicities(arguments, run_directory),
        )
        or changed
    )
    if changed:
        report = study.compose_report(
            _shard_arguments(arguments, run_directory, shard), report["cells"]
        )
    if loaded or changed:
        _write_json_atomic(_shard_report_path(run_directory, shard), report)
    return report


def _refresh_phase_pending(
    arguments: argparse.Namespace,
    run_directory: Path,
    pending: Sequence[ShardWork],
    active: Sequence[ActiveJob],
) -> list[ShardWork]:
    reports: dict[str, dict[str, Any] | None] = {}
    for shard in {work.shard for work in pending}:
        reports[shard.name] = _prepare_phase_shard_state(
            arguments, run_directory, shard
        )
    active_keys = {
        (job.shard.name, job.final_multiplicity, job.owned_mode)
        for job in active
        if job.final_multiplicity is not None
    }
    return [
        work
        for work in pending
        if (work.shard.name, work.final_multiplicity, work.owned_mode)
        not in active_keys
        and not _selected_work_complete(
            arguments,
            run_directory,
            work,
            reports.get(work.shard.name)
            if work.shard.name in reports
            else _load_shard_report(arguments, run_directory, work.shard),
        )
    ]


def _work_launch_conflicts(
    arguments: argparse.Namespace, work: ShardWork, active: Sequence[ActiveJob]
) -> bool:
    if arguments.compare_helicity_sums and work.owned_mode == "amplicol":
        return any(job.owned_mode == "amplicol" for job in active)
    if not arguments.build_amplicol or work.shard.name != "ddbar-amplicol":
        return False
    probe = _absolute(arguments.amplicol_root) / "amplicol_color_probe"
    return not probe.is_file() and any(
        job.shard.name == work.shard.name for job in active
    )


def _phase(
    arguments: argparse.Namespace,
    run_directory: Path,
    phase_number: int,
    dashboard: Dashboard,
    *,
    background: list[ActiveJob] | None = None,
    madgraph_overlay_ready: bool = False,
) -> bool:
    pending = list(_phase_work_items(arguments, run_directory, phase_number))
    active: list[ActiveJob] = []
    failure: str | None = None
    background_jobs = background if background is not None else []
    try:
        while pending or active:
            failure = _poll_madgraph_jobs(arguments, run_directory, background_jobs)
            if failure is not None:
                break
            pending = _refresh_phase_pending(arguments, run_directory, pending, active)
            used = sum(job.claimed_cores for job in active + background_jobs)
            launched = True
            while launched:
                launched = False
                for index, work in enumerate(pending):
                    if not _work_dependency_ready(arguments, run_directory, work):
                        continue
                    claim = _claimed_cores(arguments, work)
                    if used + claim > arguments.cores:
                        continue
                    if _work_launch_conflicts(arguments, work, active):
                        continue
                    _seed_cell_inputs(arguments, run_directory, work)
                    report = _publish_shard_report(arguments, run_directory, work.shard)
                    if _selected_work_complete(arguments, run_directory, work, report):
                        pending.pop(index)
                        launched = True
                        break
                    job = _launch_shard(arguments, run_directory, work)
                    active.append(job)
                    pending.pop(index)
                    used += claim
                    launched = True
                    break

            for job in active:
                assert job.final_multiplicity is not None
                assert job.owned_mode is not None
                job.detail = _work_detail(
                    arguments,
                    run_directory,
                    ShardWork(job.shard, job.final_multiplicity, job.owned_mode),
                )
            names = tuple(
                (
                    f"{job.shard.name}:{job.owned_mode}:n{job.final_multiplicity}"
                    if job.owned_mode is not None
                    else f"{job.shard.name}:n{job.final_multiplicity}"
                )
                for job in active
            )
            master = _publish_master(arguments, run_directory, active=names)
            if not madgraph_overlay_ready:
                madgraph_overlay_ready = _maybe_start_madgraph_overlay(
                    arguments, run_directory, background_jobs
                )
            dashboard.update(
                _selected_completed_cells(arguments, master)
                + _madgraph_completed(arguments, run_directory),
                phase=f"phase {phase_number}",
                active=tuple(background_jobs) + tuple(active),
            )
            if not active and pending:
                if background_jobs and any(
                    _work_dependency_ready(arguments, run_directory, work)
                    for work in pending
                ):
                    time.sleep(arguments.poll_seconds)
                    continue
                unready = next(
                    (
                        work
                        for work in pending
                        if not _work_dependency_ready(arguments, run_directory, work)
                    ),
                    None,
                )
                if unready is not None:
                    dependency_name = (
                        unready.shard.dependency[0]
                        if unready.shard.dependency is not None
                        else "unknown"
                    )
                    failure = (
                        f"{unready.shard.name} cannot start "
                        f"n{unready.final_multiplicity} because dependency "
                        f"{dependency_name} is incomplete"
                    )
                    break
            if not active:
                break
            time.sleep(arguments.poll_seconds)
            failure = _poll_madgraph_jobs(arguments, run_directory, background_jobs)
            if failure is not None:
                break
            for job in tuple(active):
                returncode = job.process.poll()
                if returncode is None:
                    continue
                job.stdout.close()
                job.stderr.close()
                active.remove(job)
                assert job.final_multiplicity is not None
                assert job.owned_mode is not None
                report = _publish_shard_report(arguments, run_directory, job.shard)
                work = ShardWork(job.shard, job.final_multiplicity, job.owned_mode)
                if not _selected_work_complete(
                    arguments, run_directory, work, report
                ):
                    failure = (
                        f"{job.shard.name} n{job.final_multiplicity} exited "
                        f"{returncode} without completing the requested "
                        "authoritative cells; inspect its "
                        "orchestrator logs"
                    )
                    break
            if failure is not None:
                break
    finally:
        if active:
            _terminate_jobs(active)
    if failure is not None:
        _publish_master(arguments, run_directory, halt_reason=failure)
        raise ProfilingError(failure)
    return madgraph_overlay_ready


def _run_madgraph(
    arguments: argparse.Namespace,
    run_directory: Path,
    dashboard: Dashboard,
) -> None:
    active = [_launch_madgraph_job(arguments, run_directory)]
    try:
        _wait_madgraph_jobs(arguments, run_directory, dashboard, active)
    except BaseException:
        if active:
            _terminate_jobs(active)
        raise


def _launch_madgraph_job(
    arguments: argparse.Namespace, run_directory: Path
) -> ActiveJob:
    command = _madgraph_command(arguments, run_directory)
    log_root = run_directory / "madgraph" / "orchestrator-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = (log_root / "stdout.log").open("wb")
    stderr = (log_root / "stderr.log").open("wb")
    process = subprocess.Popen(
        command,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    pseudo = Shard("madgraph", "gg", (), (), None, 4, False)
    job = ActiveJob(
        shard=pseudo,
        final_multiplicity=None,
        owned_mode=None,
        command=command,
        process=process,
        stdout=stdout,
        stderr=stderr,
        sampler=memory_watchdog.ProcessTreeSampler(process.pid, process.pid),
        claimed_cores=1,
        detail="madgraph/initializing",
    )
    return job


def _update_madgraph_job_detail(run_directory: Path, job: ActiveJob) -> None:
    progress_path = _madgraph_progress_path(run_directory)
    if not progress_path.is_file():
        return
    try:
        progress = madgraph.load_runtime_progress(progress_path)
        current = progress.get("current_cell")
        if isinstance(current, Mapping):
            job.detail = (
                f"{current.get('family')}/{current.get('mode')}/"
                f"n{current.get('n')}"
            )
        else:
            job.detail = "madgraph/finalizing"
    except madgraph.SelectedMadGraphError:
        job.detail = "madgraph/progress-unavailable"


def _poll_madgraph_jobs(
    arguments: argparse.Namespace,
    run_directory: Path,
    active: list[ActiveJob],
) -> str | None:
    log_root = run_directory / "madgraph" / "orchestrator-logs"
    for job in tuple(active):
        _update_madgraph_job_detail(run_directory, job)
        returncode = job.process.poll()
        if returncode is None:
            continue
        job.stdout.close()
        job.stderr.close()
        active.remove(job)
        if (
            returncode != 0
            or _matching_madgraph_overlay(arguments, run_directory) is None
        ):
            return f"MadGraph series exited {returncode}; inspect {log_root}"
    return None


def _wait_madgraph_jobs(
    arguments: argparse.Namespace,
    run_directory: Path,
    dashboard: Dashboard,
    active: list[ActiveJob],
) -> None:
    while active:
        failure = _poll_madgraph_jobs(arguments, run_directory, active)
        if failure is not None:
            raise ProfilingError(failure)
        master = _load_json(_master_report_path(run_directory), context="master report")
        dashboard.update(
            _selected_completed_cells(arguments, master)
            + _madgraph_completed(arguments, run_directory),
            phase="phase 4",
            active=active,
        )
        if active:
            time.sleep(arguments.poll_seconds)


def _canonical_madgraph_overlay_authenticated(
    arguments: argparse.Namespace,
    run_directory: Path,
    master: Mapping[str, Any],
) -> bool:
    """Use the strict publication boundary to skip a completed canonical overlay."""

    if not _report_publication_profile(master):
        return False
    overlay = _matching_madgraph_overlay(arguments, run_directory)
    if overlay is None:
        return False
    try:
        publication.build_final_report(
            campaign_path=_master_report_path(run_directory),
            madgraph_overlay_path=overlay,
        )
    except publication.PublicationMergeError:
        return False
    return True


def _lock_path(run_directory: Path, purpose: str) -> Path:
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(run_directory).encode("utf-8")).hexdigest()[:12]
    return run_directory.parent / (f".{run_directory.name}.{digest}.{purpose}.lock")


def _acquire_render_lock(run_directory: Path) -> BinaryIO:
    """Serialize render generations and refresh without blocking workers."""

    path = _lock_path(run_directory, "render")
    handle = path.open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _acquire_execution_lock(run_directory: Path) -> BinaryIO:
    # The lock must survive --refresh deleting/recreating the selected output.
    # Keeping it beside the output closes the destructive race without locking
    # unrelated profiling runs in the same parent directory.
    path = _lock_path(run_directory, "orchestrator")
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise ProfilingError(f"another profiling driver holds {path}") from error
    return handle


def _render_phase_boundary(arguments: argparse.Namespace, run_directory: Path) -> Path:
    _publish_master(arguments, run_directory)
    return render_snapshot(arguments, renderer_preflight=False)


def run_campaign(arguments: argparse.Namespace) -> Path:
    if arguments.refresh:
        _reject_refresh_symlinks(_configured_run_path(arguments))
    run_directory = _run_directory(arguments)
    if arguments.refresh:
        _validate_refresh_target(run_directory)
    lock = _acquire_execution_lock(run_directory)
    dashboard: Dashboard | None = None
    madgraph_jobs: list[ActiveJob] = []
    try:
        with _termination_signal_handlers():
            if arguments.refresh:
                removed = _safe_refresh(run_directory)
                _diagnostic(
                    (
                        f"Removed recognized profiling output {run_directory}; "
                        "restarting."
                        if removed
                        else f"No profiling output exists at {run_directory}; starting."
                    ),
                    colorama.Fore.YELLOW,
                )
            _create_or_resume_manifest(arguments, run_directory)
            _preflight(arguments)
            retried = _apply_retry_invalidation(arguments, run_directory)
            if retried:
                _diagnostic(
                    f"Retrying {retried} failed/skipped profiling cell"
                    f"{'' if retried == 1 else 's'}.",
                    colorama.Fore.CYAN,
                )
            initial = _publish_master(arguments, run_directory)
            requested = _requested_multiplicities(arguments, run_directory)
            completed = _selected_completed_cells(
                arguments, initial, multiplicities=requested
            )
            total = (
                completed
                + _selected_pending_cells(arguments, initial, multiplicities=requested)
                + (
                    len(_family_selection(arguments)) * len(requested)
                    if _madgraph_requested(arguments, run_directory)
                    else 0
                )
            )
            dashboard = Dashboard(
                total=total,
                core_budget=arguments.cores,
                helicity_workload=_helicity_workload(arguments),
                batch_size=arguments.batch_size,
            )
            dashboard.update(completed, phase="initializing", active=())
            madgraph_overlay_ready = False
            for phase_number in _scheduled_phase_numbers(arguments, run_directory):
                madgraph_overlay_ready = _phase(
                    arguments,
                    run_directory,
                    phase_number,
                    dashboard,
                    background=madgraph_jobs,
                    madgraph_overlay_ready=madgraph_overlay_ready,
                ) or madgraph_overlay_ready
                if not madgraph_overlay_ready:
                    madgraph_overlay_ready = _maybe_start_madgraph_overlay(
                        arguments, run_directory, madgraph_jobs
                    )
                _render_phase_boundary(arguments, run_directory)
            terminal = _publish_master(arguments, run_directory)
            if not _selected_master_complete(
                arguments, terminal, multiplicities=requested
            ):
                raise ProfilingError(
                    "pyAmpliCol master did not complete the requested fill"
                )
            if _madgraph_requested(arguments, run_directory):
                if madgraph_jobs:
                    _wait_madgraph_jobs(
                        arguments, run_directory, dashboard, madgraph_jobs
                    )
                    madgraph_overlay_ready = True
                elif not madgraph_overlay_ready:
                    _ensure_madgraph_overlay(arguments, run_directory, dashboard)
                    madgraph_overlay_ready = True
                if _canonical_madgraph_overlay_authenticated(
                    arguments, run_directory, terminal
                ):
                    _diagnostic(
                        "MadGraph overlay authenticated for the terminal snapshot.",
                        colorama.Fore.CYAN,
                    )
            final_count = _selected_completed_cells(
                arguments, terminal, multiplicities=requested
            ) + _madgraph_completed(arguments, run_directory)
            dashboard.update(final_count, phase="rendering", active=())
            pdf = render_snapshot(arguments, renderer_preflight=False)
            message = (
                "Completed profiling scan"
                if terminal.get("status") in TERMINAL_STATUSES
                else (
                    "Completed requested line groups and multiplicities; "
                    "campaign snapshot remains resumable"
                )
            )
            _diagnostic(f"{message}: {pdf}", colorama.Fore.GREEN)
            return pdf
    except (KeyboardInterrupt, CampaignSignal):
        with contextlib.suppress(Exception):
            _publish_master(
                arguments,
                run_directory,
                halt_reason="interrupted by user; rerun the same command to resume",
            )
        raise
    finally:
        if madgraph_jobs:
            _terminate_jobs(madgraph_jobs)
        if dashboard is not None:
            dashboard.finish()
        lock.close()


def status_payload(arguments: argparse.Namespace) -> dict[str, Any]:
    run_directory = _run_directory(arguments)
    manifest_path = _manifest_path(run_directory)
    master_path = _master_report_path(run_directory)
    manifest = (
        _load_json(manifest_path, context="profiling manifest")
        if manifest_path.is_file()
        else None
    )
    master = (
        _load_json(master_path, context="master report")
        if master_path.is_file()
        else None
    )
    if manifest is not None:
        identity_diff = _identity_diff(
            manifest.get("identity"), _identity(arguments, run_directory)
        )
        differences = _immutable_identity_differences(identity_diff)
        if differences:
            preview = "\n  ".join(differences[:12])
            raise ProfilingError(
                "status measurement identity differs from the manifest:\n  " + preview
            )
    if master is not None:
        _validate_render_workload(arguments, master)
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "helicity_workload": _helicity_workload(arguments),
        "batch_size": arguments.batch_size,
        "output": str(run_directory),
        "requested_multiplicities": list(_selection(arguments)),
        "requested_families": list(_family_selection(arguments)),
        "requested_line_groups": list(_requested_line_groups(arguments, run_directory)),
        "fill_history_multiplicities": (
            _manifest_int_history(
                manifest, "fill_history_multiplicities", "requested_multiplicities"
            )
            if manifest is not None
            else []
        ),
        "line_group_history": (
            _manifest_line_history(manifest) if manifest is not None else []
        ),
        "process_family_history": (
            _manifest_family_history(manifest) if manifest is not None else []
        ),
        "manifest": str(manifest_path),
        "manifest_available": manifest is not None,
        "configured_tools": (
            manifest.get("identity", {}).get("tools")
            if isinstance(manifest, Mapping)
            and isinstance(manifest.get("identity"), Mapping)
            else None
        ),
        "master_report": str(master_path),
        "campaign_status": master.get("status")
        if master is not None
        else "not-started",
        "completed_requested_measurement_cells": (
            _selected_completed_cells(arguments, master)
            + _madgraph_completed(arguments, run_directory)
            if master is not None
            else 0
        ),
        "completed_measurement_cells": (
            _completed_cells(master) + _madgraph_completed(arguments, run_directory)
            if master is not None
            else 0
        ),
        "dependencies": _dependency_status(arguments),
        "madgraph_overlay": {
            "path": str(_madgraph_overlay_path(run_directory)),
            "applicable": (
                _madgraph_requested(arguments, run_directory)
                if manifest is not None
                else "madgraph" in _line_selection(arguments)
            ),
            "helicity_workload": _helicity_workload(arguments),
            "available": (
                _matching_madgraph_overlay(arguments, run_directory) is not None
            ),
        },
        "rendered_pdf": {
            "path": str(
                run_directory / "render" / "current" / _pdf_filename(arguments)
            ),
            "available": (
                run_directory / "render" / "current" / _pdf_filename(arguments)
            ).is_file(),
        },
        "canonical_pdf": {
            "path": str(_canonical_pdf_path(arguments)),
            "applicable": _canonical_publication_enabled(arguments),
            "available": (
                _canonical_publication_enabled(arguments)
                and _canonical_pdf_path(arguments).is_file()
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    enabled_color = _color_enabled(arguments)
    colorama.init(strip=not enabled_color, convert=False)
    try:
        _validate_arguments(arguments)
        if arguments.dry_run:
            print(json.dumps(dry_run_plan(arguments), indent=2, sort_keys=True))
            return 0
        if arguments.status:
            print(json.dumps(status_payload(arguments), indent=2, sort_keys=True))
            return 0
        if arguments.render:
            pdf = render_snapshot(arguments)
            print(pdf)
            return 0
        pdf = run_campaign(arguments)
        print(pdf)
        return 0
    except CampaignSignal as error:
        _diagnostic(
            f"Profiling stopped by signal {error.signum}; rerun the same command "
            "to resume."
        )
        return 128 + error.signum
    except KeyboardInterrupt:
        _diagnostic("Profiling interrupted; rerun the same command to resume.")
        return 130
    except (
        OSError,
        ProfilingError,
        madgraph.SelectedMadGraphError,
        study.StudyError,
    ) as error:
        _diagnostic(f"FFT profiling: {error}", colorama.Fore.RED)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
