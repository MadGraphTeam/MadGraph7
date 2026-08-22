#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run and render the FullColor FFT profiling scan on a workstation or cluster.

This is deliberately an orchestration layer.  Physics generation, timings,
per-cell resource enforcement, numerical validation, report validation,
plotting, and PDF assembly remain owned by the existing developer tools.

Typical use::

    python tools/fft_profiling/fft_profiling.py --dry-run --cores 8
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
TERMINAL_STATUSES = frozenset({"complete", "complete-with-failures"})
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


@dataclass(slots=True)
class ActiveJob:
    shard: Shard
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
        3,
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
        3,
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


def _positive_finite(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
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
    parser.add_argument("--target-seconds", type=_positive_finite, default=0.25)
    parser.add_argument(
        "--amplicol-root",
        "--amplicol-repository",
        dest="amplicol_root",
        type=Path,
        default=study.legacy_amplicol.DEFAULT_REPOSITORY,
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
        DEFAULT_SUM_OUTPUT
        if arguments.compare_helicity_sums
        else DEFAULT_FIXED_OUTPUT
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


def _canonical_publication_enabled(arguments: argparse.Namespace) -> bool:
    return arguments.output is None and arguments.campaign_report is None


def _universe(arguments: argparse.Namespace) -> tuple[int, ...]:
    stored = getattr(arguments, "universe_multiplicities", None)
    if stored is not None:
        return tuple(int(value) for value in stored)
    return tuple(range(2, 10))


def _selection(arguments: argparse.Namespace) -> tuple[int, ...]:
    return tuple(sorted(set(arguments.multiplicities)))


def _shard_study_root(run_directory: Path, shard: Shard) -> Path:
    return run_directory / "shards" / shard.name / "study"


def _shard_report_path(run_directory: Path, shard: Shard) -> Path:
    return _shard_study_root(run_directory, shard) / "runs" / shard.name / "report.json"


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
    path = _manifest_path(run_directory)
    if not path.is_file():
        return _selection(arguments)
    manifest = _load_json(path, context="profiling manifest")
    values = manifest.get("requested_multiplicities")
    if not isinstance(values, list) or any(
        not isinstance(value, int) for value in values
    ):
        raise ProfilingError("profiling manifest has invalid fill history")
    return tuple(values)


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
) -> tuple[str, ...]:
    result = [
        "--study-root",
        str(study_root),
        "--run-id",
        run_id,
        "--fft",
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
        "--amplicol-repository",
        str(_absolute(arguments.amplicol_root)),
    ]
    for multiplicity in _universe(arguments):
        result.extend(("--multiplicity", str(multiplicity)))
    for multiplicity in (
        _selection(arguments)
        if fill_multiplicities is None
        else tuple(fill_multiplicities)
    ):
        result.extend(("--fill-multiplicity", str(multiplicity)))
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
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[str, ...]:
    return _study_cli_arguments(
        arguments,
        study_root=_shard_study_root(run_directory, shard),
        run_id=shard.name,
        families=(shard.family,),
        modes=shard.modes,
        resume=True,
        build_amplicol=arguments.build_amplicol and shard.name == "ddbar-amplicol",
        fill_multiplicities=_requested_multiplicities(arguments, run_directory),
    )


def _master_arguments(
    arguments: argparse.Namespace, run_directory: Path
) -> argparse.Namespace:
    modes = tuple(
        dict.fromkeys(
            mode
            for family in ("gg", "ddbar")
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
            families=("gg", "ddbar"),
            modes=modes,
            resume=True,
        )
    finally:
        arguments.multiplicities = original
    namespace = study._parser().parse_args(raw_arguments)
    study._validate_arguments(namespace)
    return namespace


def _shard_arguments(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> argparse.Namespace:
    namespace = study._parser().parse_args(
        _shard_cli_arguments(arguments, run_directory, shard)
    )
    study._validate_arguments(namespace)
    return namespace


def _validate_arguments(arguments: argparse.Namespace) -> None:
    _normalize_executables(arguments)
    if arguments.refresh:
        _reject_refresh_symlinks(_configured_run_path(arguments))
    if arguments.candidate_cores > arguments.cores:
        raise ProfilingError("--candidate-cores cannot exceed --cores")
    if arguments.dpi < 72 or arguments.dpi > 600:
        raise ProfilingError("--dpi must be between 72 and 600")
    if arguments.target_seconds < 0.25:
        raise ProfilingError("--target-seconds must be at least 0.25")
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
    # The authoritative parser owns run-id, n-range, mode, and resource checks.
    _master_arguments(arguments, _run_directory(arguments))
    if arguments.campaign_report is not None and not arguments.render:
        raise ProfilingError("--campaign-report is valid only with --render")


def _normalize_executables(arguments: argparse.Namespace) -> None:
    for attribute in ("python", "cxx", "fc"):
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
            "families": ["gg", "ddbar"],
            "modes": {
                family: list(publication.FAMILY_MODES[family])
                for family in ("gg", "ddbar")
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
            "madgraph_root": (
                str(_absolute(arguments.madgraph_root))
                if arguments.madgraph_root is not None
                else None
            ),
        },
    }


def _child_command(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> tuple[str, ...]:
    return (
        str(arguments.python),
        str(STUDY_TOOL),
        *_shard_cli_arguments(arguments, run_directory, shard),
    )


def _measurement_multiplicities(
    arguments: argparse.Namespace, shard: Shard
) -> tuple[int, ...]:
    return tuple(
        final_multiplicity
        for final_multiplicity in _selection(arguments)
        if not all(
            study.otf_protocol_scope_cell(
                family=shard.family,
                mode=study.MODE_BY_KEY[mode],
                final_multiplicity=final_multiplicity,
                sum_helicities=arguments.compare_helicity_sums,
            )
            is not None
            for mode in shard.owned_modes
        )
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
    commands.extend(
        (
            (
                str(arguments.python),
                str(PLOT_TOOL),
                str(plot_report),
                str(staged_plots),
                "--dpi",
                str(arguments.dpi),
            ),
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
    commands = {
        shard.name: {
            "phase": shard.phase,
            "family": shard.family,
            "modes": list(shard.modes),
            "owned_modes": list(shard.owned_modes),
            "dependency": list(shard.dependency) if shard.dependency else None,
            "claimed_cores": (arguments.candidate_cores if shard.candidate else 1),
            "measurement_multiplicities": list(
                _measurement_multiplicities(arguments, shard)
            ),
            "protocol_skip_multiplicities": sorted(
                set(_selection(arguments))
                - set(_measurement_multiplicities(arguments, shard))
            ),
            "study_root": str(_shard_study_root(run_directory, shard)),
            "report": str(_shard_report_path(run_directory, shard)),
            "argv": (
                list(_child_command(arguments, run_directory, shard))
                if _measurement_multiplicities(arguments, shard)
                else None
            ),
            "shell_command": (
                shlex.join(_child_command(arguments, run_directory, shard))
                if _measurement_multiplicities(arguments, shard)
                else None
            ),
        }
        for shard in SHARDS
    }
    strict = _publication_profile(arguments)
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
        "identity": _identity(arguments, run_directory),
        "scheduler": {
            "total_core_budget": arguments.cores,
            "candidate_cores_per_active_candidate_shard": arguments.candidate_cores,
            "baseline_cores_per_active_shard": 1,
            "parallelism": "weighted dependency-aware curve shards",
        },
        "shards": commands,
        "madgraph": {
            "phase": 4,
            "applicable": True,
            "helicity_workload": _helicity_workload(arguments),
            "measurement_multiplicities": [
                n
                for n in _requested_multiplicities(arguments, run_directory)
                if n <= madgraph.MAX_PROTOCOL_MEASURED_MULTIPLICITY
            ],
            "protocol_scope_multiplicities": [
                n
                for n in _requested_multiplicities(arguments, run_directory)
                if n > madgraph.MAX_PROTOCOL_MEASURED_MULTIPLICITY
            ],
            "dependency": "completed pyAmpliCol cells for the requested fill",
            "not_applicable_reason": None,
            "report": str(_madgraph_overlay_path(run_directory)),
            "argv": (
                list(_madgraph_command(arguments, run_directory))
                if arguments.madgraph_root is not None
                else None
            ),
            "shell_command": (
                shlex.join(_madgraph_command(arguments, run_directory))
                if arguments.madgraph_root is not None
                else None
            ),
        },
        "outputs": {
            "manifest": str(_manifest_path(run_directory)),
            "master_report": str(_master_report_path(run_directory)),
            "render_pointer": str(run_directory / "render" / "current"),
            "final_pdf": str(
                run_directory / "render" / "current" / pdf_filename
            ),
            "canonical_pdf": (
                str(_canonical_pdf_path(arguments))
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
            f"legacy AmpliCol checkout: {_absolute(arguments.amplicol_root)}",
            (
                f"MadGraph installation: {_absolute(arguments.madgraph_root)}"
                if arguments.madgraph_root is not None
                else "MadGraph installation: REQUIRED (--madgraph-root)"
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
        if differences:
            preview = "\n  ".join(differences[:12])
            raise ProfilingError(
                "resume measurement identity differs from the manifest:\n  " + preview
            )
        prior = manifest.get("requested_multiplicities", [])
        if not isinstance(prior, list) or any(
            not isinstance(value, int) for value in prior
        ):
            raise ProfilingError("profiling manifest has invalid fill history")
        cumulative = sorted(set(prior) | set(_selection(arguments)))
        if cumulative != prior:
            manifest["requested_multiplicities"] = cumulative
            _write_json_atomic(path, manifest)
        return manifest
    selected = list(_selection(arguments))
    manifest = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "identity": requested_identity,
        "initial_scheduler_core_budget": arguments.cores,
        "requested_multiplicities": selected,
    }
    _write_json_atomic(path, manifest)
    return manifest


def _expected_shard_policy(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> dict[str, object]:
    return study.dry_run_plan(_shard_arguments(arguments, run_directory, shard))


def _load_shard_report(
    arguments: argparse.Namespace,
    run_directory: Path,
    shard: Shard,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    path = _shard_report_path(run_directory, shard)
    if not path.is_file():
        if required:
            raise ProfilingError(f"required shard report is missing: {path}")
        return None
    report = _load_json(path, context=f"{shard.name} shard report")
    if (
        report.get("kind") != study.KIND
        or report.get("schema_version") != study.SCHEMA_VERSION
        or report.get("policy")
        != _expected_shard_policy(arguments, run_directory, shard)
    ):
        raise ProfilingError(f"{shard.name} shard report policy/schema differs")
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ProfilingError(f"{shard.name} shard report has no cells")
    family_cells = cells.get(shard.family)
    if not isinstance(family_cells, Mapping) or set(family_cells) != set(shard.modes):
        raise ProfilingError(f"{shard.name} shard mode set differs")
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
        and study.selected_cells_complete(
            report,
            family=shard.family,
            modes=shard.owned_modes,
            multiplicities=(
                _selection(arguments)
                if multiplicities is None
                else tuple(multiplicities)
            ),
        )
    )


def _seed_dependency(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> None:
    if shard.dependency is None:
        return
    target = _shard_report_path(run_directory, shard)
    dependency_name, dependency_mode = shard.dependency
    dependency_shard = SHARD_BY_NAME[dependency_name]
    source = _load_shard_report(
        arguments, run_directory, dependency_shard, required=True
    )
    if not study.selected_cells_complete(
        source,
        family=shard.family,
        modes=(dependency_mode,),
        multiplicities=_requested_multiplicities(arguments, run_directory),
    ):
        raise ProfilingError(
            f"dependency {dependency_name} has not completed this fill selection"
        )
    namespace = _shard_arguments(arguments, run_directory, shard)
    report = (
        _load_shard_report(arguments, run_directory, shard, required=True)
        if target.is_file()
        else study.compose_report(namespace, {})
    )
    source_cells = source["cells"]
    target_cells = report["cells"]
    assert isinstance(source_cells, Mapping)
    assert isinstance(target_cells, dict)
    family_source = source_cells[shard.family]
    family_target = target_cells[shard.family]
    assert isinstance(family_source, Mapping)
    assert isinstance(family_target, dict)
    family_target[dependency_mode] = copy.deepcopy(family_source[dependency_mode])
    report["status"] = "running"
    _write_json_atomic(target, report)


def _seed_protocol_scope(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> None:
    if not any(mode.startswith("otf-") for mode in shard.owned_modes):
        return
    target = _shard_report_path(run_directory, shard)
    namespace = _shard_arguments(arguments, run_directory, shard)
    report = (
        _load_shard_report(arguments, run_directory, shard, required=True)
        if target.is_file()
        else study.compose_report(namespace, {})
    )
    if study.apply_protocol_scope_cells(
        report,
        family=shard.family,
        modes=shard.owned_modes,
        multiplicities=_requested_multiplicities(arguments, run_directory),
    ):
        report["status"] = "running"
        _write_json_atomic(target, report)


def _publication_profile(arguments: argparse.Namespace) -> bool:
    namespace = _master_arguments(arguments, _run_directory(arguments))
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
    namespace = _master_arguments(arguments, run_directory)
    reports: dict[str, dict[str, Any] | None] = {}
    shard_status: dict[str, str] = {}
    for shard in SHARDS:
        shard_report = _load_shard_report(arguments, run_directory, shard)
        reports[shard.name] = shard_report
        shard_status[shard.name] = (
            str(shard_report.get("status")) if shard_report is not None else "pending"
        )
    curve_sources: dict[str, dict[str, Mapping[str, object]]] = {
        "gg": {},
        "ddbar": {},
    }
    for family in ("gg", "ddbar"):
        for mode in publication.FAMILY_MODES[family]:
            owner = SHARD_BY_NAME[MODE_OWNER[(family, mode)]]
            owner_report = reports[owner.name]
            if owner_report is None:
                continue
            owner_cells = owner_report["cells"]
            assert isinstance(owner_cells, Mapping)
            owner_family = owner_cells[family]
            assert isinstance(owner_family, Mapping)
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
    measurement["schedule_order"] = "dependency-ordered-parallel-curve-shards"
    plot = policy.setdefault("plot", {})
    assert isinstance(plot, dict)
    if arguments.compare_helicity_sums:
        plot["notes"] = [
            "Complete physical-helicity-summed matrix-element workload; "
            "MadGraph standalone uses generated SMATRIX with USERHEL=-1; "
            "warmed GOODHEL pruning remains enabled."
        ]
    elif not _publication_profile(arguments):
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


def _selected_master_complete(
    arguments: argparse.Namespace,
    report: Mapping[str, Any],
    *,
    multiplicities: Sequence[int] | None = None,
) -> bool:
    return not str(report.get("status", "")).startswith("stopped") and all(
        study.selected_cells_complete(
            report,
            family=family,
            modes=publication.FAMILY_MODES[family],
            multiplicities=(
                _selection(arguments)
                if multiplicities is None
                else tuple(multiplicities)
            ),
        )
        for family in ("gg", "ddbar")
    )


def _selected_pending_cells(
    arguments: argparse.Namespace,
    report: Mapping[str, Any],
    *,
    multiplicities: Sequence[int] | None = None,
) -> int:
    requested = (
        _selection(arguments) if multiplicities is None else tuple(multiplicities)
    )
    return sum(
        not study.selected_cells_complete(
            report,
            family=family,
            modes=(mode,),
            multiplicities=(final_multiplicity,),
        )
        for family in ("gg", "ddbar")
        for mode in publication.FAMILY_MODES[family]
        for final_multiplicity in requested
    )


def _run_checked(
    command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> None:
    completed = subprocess.run(tuple(command), check=False, env=environment)
    if completed.returncode != 0:
        raise ProfilingError(
            f"command failed with status {completed.returncode}: {shlex.join(command)}"
        )


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
                    if (
                        not str(raw_n).isdigit()
                        or not isinstance(cell, Mapping)
                    ):
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
    if not isinstance(value, Mapping):
        return False
    current = madgraph.measurement_host_identity()
    return all(
        value.get(key) == current[key] for key in ("system", "machine", "python")
    )


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
        or policy.get("higher_multiplicity_policy")
        != "not-applicable-protocol-scope"
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
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
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
            overlay=_madgraph_render_source(arguments, run_directory),
        )
    else:
        selection = _implicit_render_selection(arguments, run_directory)
    source = selection.source
    if not source.is_file():
        raise ProfilingError(
            f"no campaign snapshot yet: {source}; start the scan before --render"
        )
    if renderer_preflight:
        _preflight_renderer(arguments)
    raw, report, digest = _source_snapshot(source)
    _validate_render_workload(arguments, report)
    pdf_filename = _pdf_filename(arguments)
    terminal_overlay = (
        _matching_madgraph_overlay(arguments, run_directory)
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
    partial_overlay = selection.overlay or (
        run_directory / "madgraph" / ".unavailable"
    )

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
            arguments,
            source_report=frozen_source,
            staged_report=staged_report,
            staged_plots=staged_plots,
            staged_pdf=staged_pdf,
            strict=strict,
            madgraph_overlay=overlay,
        )
        render_environment = _renderer_environment(arguments)
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
        if observed_plots != expected_plots or not (
            staged_pdf.is_file() and staged_pdf.stat().st_size > 0
        ):
            raise ProfilingError(
                "render commands did not produce one complete output set"
            )
        published = _publish_render_generation(
            stage, render_root, digest, pdf_filename
        )
        if _canonical_publication_enabled(arguments):
            _publish_canonical_pdf(published, arguments)
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
        "amplicol_root": {
            "path": str(amplicol),
            "compatible": (amplicol / "process_list.py").is_file(),
            "probe": str(amplicol / "amplicol_color_probe"),
            "probe_available": (amplicol / "amplicol_color_probe").is_file(),
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
    for key in ("python", "cxx", "fortran"):
        if status[key]["available"] is not True:
            raise ProfilingError(
                f"required {key} executable is unavailable: {status[key]['path']}"
            )
    amplicol = status["amplicol_root"]
    if amplicol["compatible"] is not True:
        raise ProfilingError(f"invalid --amplicol-root: {amplicol['path']}")
    if (
        not arguments.compare_helicity_sums
        and amplicol["probe_available"] is not True
        and not arguments.build_amplicol
    ):
        raise ProfilingError(
            "AmpliCol probe is missing; pass --build-amplicol to build it once: "
            f"{amplicol['probe']}"
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
        raise ProfilingError(
            "plot/PDF dependencies are missing from --python; install them with "
            f"{arguments.python} -m pip install -e '.[fft-profiling]'"
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
        "--generation-timeout-seconds",
        format(_madgraph_timeout(arguments), ".17g"),
        "--memory-limit-gib",
        format(arguments.memory_limit_gib, ".17g"),
    ]
    for multiplicity in _requested_multiplicities(arguments, run_directory):
        result.extend(
            (
                (
                    "--multiplicity"
                    if multiplicity
                    <= madgraph.MAX_PROTOCOL_MEASURED_MULTIPLICITY
                    else "--protocol-scope-multiplicity"
                ),
                str(multiplicity),
            )
        )
    if arguments.compare_helicity_sums:
        result.append("--compare-helicity-sums")
    return tuple(result)


def _freeze_madgraph_source(arguments: argparse.Namespace, run_directory: Path) -> Path:
    master_path = _master_report_path(run_directory)
    raw, master, _ = _source_snapshot(master_path)
    for family in ("gg", "ddbar"):
        source_mode = madgraph.source_mode(family, _helicity_workload(arguments))
        if not study.selected_cells_complete(
            master,
            family=family,
            modes=(source_mode,),
            multiplicities=_requested_multiplicities(arguments, run_directory),
        ):
            raise ProfilingError(
                "cannot freeze MadGraph source before its selected source cells "
                "are complete"
            )
    destination = _madgraph_source_path(arguments, run_directory)
    if destination.is_file():
        existing = _load_json(destination, context="frozen MadGraph source")
        normalized_master = copy.deepcopy(master)
        normalized_existing = copy.deepcopy(existing)
        normalized_master.pop("profiling_orchestration", None)
        normalized_existing.pop("profiling_orchestration", None)
        if normalized_existing != normalized_master:
            raise ProfilingError(
                "candidate measurements changed after this MadGraph selection "
                "source was frozen; use --refresh to restart this output safely"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_bytes(raw)
    temporary.replace(destination)
    return destination


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
    if (
        not isinstance(policy, Mapping)
        or progress_workload != _helicity_workload(arguments)
        or progress_host != madgraph.measurement_host_identity()
        or policy.get("final_state_multiplicities")
        != list(_requested_multiplicities(arguments, run_directory))
    ):
        return None
    return payload


def _madgraph_completed(arguments: argparse.Namespace, run_directory: Path) -> int:
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
    summary = report.get("summary")
    if isinstance(summary, Mapping) and isinstance(
        summary.get("completed_cell_count"), int
    ):
        return int(summary["completed_cell_count"])
    series = report.get("runtime_series")
    if not isinstance(series, Mapping):
        return 0
    return sum(
        len(mode_cells)
        for family_cells in series.values()
        if isinstance(family_cells, Mapping)
        for mode_cells in family_cells.values()
        if isinstance(mode_cells, Mapping)
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


def _claimed_cores(arguments: argparse.Namespace, shard: Shard) -> int:
    return arguments.candidate_cores if shard.candidate else 1


def _launch_shard(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> ActiveJob:
    command = _child_command(arguments, run_directory, shard)
    log_root = run_directory / "shards" / shard.name / "orchestrator-logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = (log_root / "stdout.log").open("wb")
    stderr = (log_root / "stderr.log").open("wb")
    claimed = _claimed_cores(arguments, shard)
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
        shard=shard,
        command=command,
        process=process,
        stdout=stdout,
        stderr=stderr,
        sampler=memory_watchdog.ProcessTreeSampler(process.pid, process.pid),
        claimed_cores=claimed,
        detail=f"{shard.family}/{shard.owned_modes[0]}/n{_selection(arguments)[0]}",
    )


def _shard_detail(
    arguments: argparse.Namespace, run_directory: Path, shard: Shard
) -> str:
    report = _load_shard_report(arguments, run_directory, shard)
    for final_multiplicity in _requested_multiplicities(arguments, run_directory):
        for mode in shard.owned_modes:
            if report is None or not study.selected_cells_complete(
                report,
                family=shard.family,
                modes=(mode,),
                multiplicities=(final_multiplicity,),
            ):
                return f"{shard.family}/{mode}/n{final_multiplicity}"
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


def _phase(
    arguments: argparse.Namespace,
    run_directory: Path,
    phase_number: int,
    dashboard: Dashboard,
) -> None:
    pending = []
    requested = _requested_multiplicities(arguments, run_directory)
    for shard in SHARDS:
        if shard.phase != phase_number:
            continue
        _seed_protocol_scope(arguments, run_directory, shard)
        report = _load_shard_report(arguments, run_directory, shard)
        if not _selected_shard_complete(
            arguments, shard, report, multiplicities=requested
        ):
            pending.append(shard)
    pending.sort(
        key=lambda shard: (
            0 if shard.name == "gg-amplicol" else 1,
            shard.name,
        )
    )
    active: list[ActiveJob] = []
    failure: str | None = None
    try:
        while pending or active:
            used = sum(job.claimed_cores for job in active)
            launched = True
            while launched:
                launched = False
                for index, shard in enumerate(pending):
                    claim = _claimed_cores(arguments, shard)
                    if used + claim > arguments.cores:
                        continue
                    if shard.dependency is not None:
                        _seed_dependency(arguments, run_directory, shard)
                    job = _launch_shard(arguments, run_directory, shard)
                    active.append(job)
                    pending.pop(index)
                    used += claim
                    launched = True
                    break

            for job in active:
                job.detail = _shard_detail(arguments, run_directory, job.shard)
            names = tuple(job.shard.name for job in active)
            master = _publish_master(arguments, run_directory, active=names)
            dashboard.update(
                _completed_cells(master)
                + _madgraph_completed(arguments, run_directory),
                phase=f"phase {phase_number}",
                active=active,
            )
            if not active:
                break
            time.sleep(arguments.poll_seconds)
            for job in tuple(active):
                returncode = job.process.poll()
                if returncode is None:
                    continue
                job.stdout.close()
                job.stderr.close()
                active.remove(job)
                report = _load_shard_report(
                    arguments, run_directory, job.shard, required=True
                )
                if returncode != 0 or not _selected_shard_complete(
                    arguments,
                    job.shard,
                    report,
                    multiplicities=requested,
                ):
                    failure = (
                        f"{job.shard.name} exited {returncode} without completing "
                        "the requested authoritative cells; inspect its "
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


def _run_madgraph(
    arguments: argparse.Namespace,
    run_directory: Path,
    dashboard: Dashboard,
) -> None:
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
        command=command,
        process=process,
        stdout=stdout,
        stderr=stderr,
        sampler=memory_watchdog.ProcessTreeSampler(process.pid, process.pid),
        claimed_cores=1,
        detail="madgraph/initializing",
    )
    try:
        while process.poll() is None:
            progress_path = _madgraph_progress_path(run_directory)
            if progress_path.is_file():
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
            master = _load_json(
                _master_report_path(run_directory), context="master report"
            )
            dashboard.update(
                _completed_cells(master)
                + _madgraph_completed(arguments, run_directory),
                phase="phase 4",
                active=(job,),
            )
            time.sleep(arguments.poll_seconds)
    except BaseException:
        _terminate_jobs((job,))
        raise
    finally:
        stdout.close()
        stderr.close()
    if (
        process.returncode != 0
        or _matching_madgraph_overlay(arguments, run_directory) is None
    ):
        raise ProfilingError(
            f"MadGraph series exited {process.returncode}; inspect {log_root}"
        )


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


def _render_phase_boundary(
    arguments: argparse.Namespace, run_directory: Path, phase_number: int
) -> Path:
    _publish_master(arguments, run_directory)
    pdf = render_snapshot(arguments, renderer_preflight=False)
    _diagnostic(
        f"Updated phase-{phase_number} profiling PDF: {pdf}",
        colorama.Fore.CYAN,
    )
    return pdf


def run_campaign(arguments: argparse.Namespace) -> Path:
    if arguments.refresh:
        _reject_refresh_symlinks(_configured_run_path(arguments))
    run_directory = _run_directory(arguments)
    if arguments.refresh:
        _validate_refresh_target(run_directory)
    lock = _acquire_execution_lock(run_directory)
    dashboard: Dashboard | None = None
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
            initial = _publish_master(arguments, run_directory)
            requested = _requested_multiplicities(arguments, run_directory)
            total = (
                _completed_cells(initial)
                + _selected_pending_cells(arguments, initial, multiplicities=requested)
                + (
                    2 * len(_requested_multiplicities(arguments, run_directory))
                )
            )
            dashboard = Dashboard(
                total=total,
                core_budget=arguments.cores,
                helicity_workload=_helicity_workload(arguments),
                batch_size=arguments.batch_size,
            )
            dashboard.update(_completed_cells(initial), phase="initializing", active=())
            _phase(arguments, run_directory, 1, dashboard)
            _render_phase_boundary(arguments, run_directory, 1)
            _phase(arguments, run_directory, 2, dashboard)
            _render_phase_boundary(arguments, run_directory, 2)
            _phase(arguments, run_directory, 3, dashboard)
            _render_phase_boundary(arguments, run_directory, 3)
            terminal = _publish_master(arguments, run_directory)
            if not _selected_master_complete(
                arguments, terminal, multiplicities=requested
            ):
                raise ProfilingError(
                    "pyAmpliCol master did not complete the requested fill"
                )
            _freeze_madgraph_source(arguments, run_directory)
            if _canonical_madgraph_overlay_authenticated(
                arguments, run_directory, terminal
            ):
                _diagnostic(
                    "Reusing the authenticated terminal MadGraph overlay.",
                    colorama.Fore.CYAN,
                )
            else:
                _run_madgraph(arguments, run_directory, dashboard)
            final_count = _completed_cells(terminal) + _madgraph_completed(
                arguments, run_directory
            )
            dashboard.update(final_count, phase="rendering", active=())
            pdf = render_snapshot(arguments, renderer_preflight=False)
            message = (
                "Completed profiling scan"
                if terminal.get("status") in TERMINAL_STATUSES
                else "Completed requested fill; campaign snapshot remains in progress"
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
        differences = _identity_diff(
            manifest.get("identity"), _identity(arguments, run_directory)
        )
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
        "completed_measurement_cells": (
            _completed_cells(master) + _madgraph_completed(arguments, run_directory)
            if master is not None
            else 0
        ),
        "dependencies": _dependency_status(arguments),
        "madgraph_overlay": {
            "path": str(_madgraph_overlay_path(run_directory)),
            "applicable": True,
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
