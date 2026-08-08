# SPDX-License-Identifier: 0BSD
"""Human-steerable pyAmpliCol profiling campaign.

This module is intentionally a thin controller.  Cell definition, dependency
planning, generation, profiling, atomic attempt publication, rendering, and PDF
compilation remain owned by the existing performance-report and public
``pyamplicol`` command APIs.
"""

from __future__ import annotations

import argparse
import base64
import copy
import difflib
import errno
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TextIO

from colorama import Fore, Style, just_fix_windows_console
from prettytable import PrettyTable

from .artifacts import (
    ATTEMPT_SCHEMA,
    ArtifactStore,
    CurrentRecord,
    LockTimeoutError,
    ManifestValidationError,
    _raise_disk_full,
)
from .cache import reset_entry, validate_measurement
from .campaign_policy import (
    PolicyMeasurementState,
    policy_measurement_state_hint,
    policy_status_label,
)
from .catalog import PROCESS_FAMILIES, REPORT_CATALOG, ReportCatalog
from .measurement import failure_measurement
from .models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ModelKey,
    ResultStatus,
    Workload,
)
from .publication import PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI
from .publisher import (
    DEFAULT_PDF_TIMEOUT_SECONDS,
    _compile_pdf,
    _report_source_copy_ignore,
)
from .resources import (
    DEFAULT_ATTEMPT_OUTPUT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
)
from .scheduler import (
    CampaignResult,
    CampaignScheduler,
    CampaignSettings,
    CellSelection,
    PlannedCell,
    plan_campaign,
    reconcile_attempt_history,
    select_cells,
)
from .service import ReportPaths, ReportService, validate_profile_name
from .source_identity import ReportSourceIdentity, _generated_report_path
from .timing import (
    evaluator_total_timing_record,
    recurrence_core_seconds_per_point,
)
from .workspace import (
    ENVIRONMENT_JSON,
    ENVIRONMENT_TEX,
    record_authenticated_profile_environment,
)

PROFILE = "macbook_M3_manual"
_ACTIVE_PROFILE = PROFILE
_PRESENTATION_PROFILE = "campaign-local-v1"
DEFAULT_GENERATION_LIMIT_SECONDS = 60.0 * 60.0
DEFAULT_WORKER_WALL_LIMIT_SECONDS = 60.0 * 60.0
DEFAULT_RAM_BYTES = 30_000_000_000
DEFAULT_CORES_PER_WORKER = 1
DEFAULT_TARGET_RUNTIME_SECONDS = 5.0
_NUMERICAL_RELATION_CORRECTNESS_ABI = (
    "pyamplicol-numerical-current-relation-correctness-v1"
)
DEFAULT_BATCH_SIZE = 128
DEFAULT_WARMUP_RUNS = 2
DEFAULT_MINIMUM_SAMPLES = 5
DEFAULT_WORKER_STALE_SECONDS = 15.0
DEFAULT_MANUAL_EXPECTED_PAGE_COUNT = 73
_REPORT_SECTION_MARKER = re.compile(
    r"^% pyamplicol-report-section-(begin|end): ([a-z][a-z0-9-]*)$"
)
_ORIGINAL_AMPLICOL_REQUIRED_FILES = (
    "makefile",
    "process_list.py",
    "amplicol_color_probe.f03",
    "amplicol_color_library_probe.f03",
    "amplicol_library_benchmark.f03",
)
_ORIGINAL_AMPLICOL_REQUIRED_MAKE_TARGETS = (
    "amplicol_generate_library",
    "amplicol_library_benchmark",
    "amplicol_color_probe",
    "amplicol_color_library_probe",
)
_LOCAL_AMPLICOL_CONFIG = ".pyamplicol-original-amplicol"
_LOCAL_MADGRAPH_CONFIG = ".pyamplicol-madgraph"
MAX_TAIL_READ_BYTES = 64 * 1024
MAX_LOG_TAIL_LINES = 8
MANUAL_STATE_SCHEMA = "pyamplicol-manual-campaign-state-v1"
SOURCE_MARKER_SCHEMA = "pyamplicol-manual-source-v1"
PRESENTATION_OUTCOME_SCHEMA = "pyamplicol-manual-presentation-outcome-v1"
_FULL_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_OUTCOME_SLUG = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SAFE_OUTCOME_CELL_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,127}")
_PRESENTATION_FAILURE_KIND_PREFIX = "ManualCampaignOutcome:"
_UNRECORDED_MEASUREMENT_NUMPY = "unavailable (not recorded by measurement)"
_MAX_PRESENTATION_COMPLETED_AT_NS = (1 << 63) - 1
_PRESENTATION_OUTCOME_KEYS = frozenset(
    {
        "schema",
        "profile",
        "cell_id",
        "source_revision",
        "campaign_invocation_id",
        "attempt_id",
        "status",
        "label",
        "completed_at_ns",
    }
)
_DASHBOARD_COUNTER_KEYS = (
    "selected",
    "recycled",
    "active",
    "selected_active",
    "dependency_active",
    "completed",
    "remaining",
    "static_na",
    "capped",
    "failed",
    "unverified",
    "dependency_only",
    "dependency_completed",
    "dependency_issues",
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ACTIVE_WORKER_STATUSES = frozenset({"queued", "preparing", "running"})
_RECYCLED_SUCCESS_STATUSES = frozenset({"recycled", "reused", "skipped-current"})
_COMPLETED_WORKER_STATUSES = frozenset(
    {"ok", "success", *_RECYCLED_SUCCESS_STATUSES}
)
_TERMINAL_CAP_STATUSES = frozenset(
    {
        PolicyMeasurementState.GENERATION_LIMIT.value,
        PolicyMeasurementState.MEMORY_LIMIT.value,
        PolicyMeasurementState.WORKER_TIMEOUT.value,
        PolicyMeasurementState.PROFILING_TIMEOUT.value,
        PolicyMeasurementState.VALIDATION_TIMEOUT.value,
    }
)
_TERMINAL_CAP_STATES = frozenset(
    {
        PolicyMeasurementState.GENERATION_LIMIT,
        PolicyMeasurementState.MEMORY_LIMIT,
        PolicyMeasurementState.WORKER_TIMEOUT,
        PolicyMeasurementState.PROFILING_TIMEOUT,
        PolicyMeasurementState.VALIDATION_TIMEOUT,
    }
)
_PHASE_TIMELINE_SCHEMA = "pyamplicol-manual-campaign-phase-timeline-v1"
_REPRODUCTION_STAGES = ("prepare", "generate", "profile")
_SUMMARY_SUCCESS_STATUSES = frozenset(
    {"ok", "success", "recycled", "reused", "skipped-current"}
)
_FAIL_FAST_SUCCESS_STATUSES = frozenset(
    {"ok", "success", "reused", "skipped-current"}
)
FAIL_FAST_FAILURE_LOG = "fail_fast_failure.log"


class _DashboardTerminalOutcome(StrEnum):
    """One normalized terminal category for a dashboard-owned cell.

    The map keyed by cell ID is the sole terminal-state authority. Counter
    sets are derived views, so retries cannot leave a cell in two categories.
    """

    SUCCESS = "success"
    CAPPED = "capped"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class ManualCampaignError(RuntimeError):
    """One concise, user-facing steering error."""


def _require_literal_directory(
    path: Path,
    *,
    label: str,
    required: bool,
) -> None:
    """Reject symlink and special-file campaign roots before resolution."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if required:
            raise ManualCampaignError(f"{label} does not exist: {path}") from None
        return
    except OSError as error:
        raise ManualCampaignError(f"cannot inspect {label} {path}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ManualCampaignError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ManualCampaignError(f"{label} must be a directory: {path}")


def _campaign_report_paths(repo_root: Path, docs_dir: Path) -> ReportPaths:
    """Derive every private campaign path from one literal campaign directory."""

    expanded = docs_dir.expanduser()
    literal_docs = expanded if expanded.is_absolute() else Path.cwd() / expanded
    _require_literal_directory(
        literal_docs,
        label="campaign directory",
        required=True,
    )
    literal_artifacts = literal_docs / "campaign_artifacts"
    literal_coordination = literal_artifacts / "coordination"
    _require_literal_directory(
        literal_artifacts,
        label="campaign artifact root",
        required=False,
    )
    _require_literal_directory(
        literal_coordination,
        label="campaign coordination root",
        required=False,
    )
    resolved_docs = literal_docs.resolve(strict=True)
    if resolved_docs != Path(os.path.abspath(literal_docs)):
        raise ManualCampaignError(
            f"campaign directory must not traverse a symbolic link: {literal_docs}"
        )
    artifact_root = resolved_docs / "campaign_artifacts"
    return ReportPaths.from_repo(
        repo_root,
        docs_dir=resolved_docs,
        artifact_root=artifact_root,
        coordination_root=artifact_root / "coordination",
    )


def _require_launcher_working_directory_identity(
    *, launcher_path_checked: bool
) -> None:
    """Reject a launcher left inside a renamed campaign directory."""

    if launcher_path_checked:
        return
    invocation = Path(sys.argv[0])
    if invocation.name != "steer_performance_campaign.py":
        return
    logical_text = os.environ.get("PWD")
    logical_cwd = Path(logical_text) if logical_text else None
    try:
        physical_cwd = Path.cwd()
        same_directory = (
            logical_cwd is not None
            and logical_cwd.is_absolute()
            and logical_cwd.samefile(physical_cwd)
        )
    except OSError:
        same_directory = False
    if not same_directory:
        raise ManualCampaignError(
            "campaign launcher cannot identify its working directory: "
            "the shell PWD is missing or no longer names the physical directory. "
            "Run `cd ..` and re-enter the intended campaign directory, then retry "
            "(or invoke its absolute path from another working directory)."
        )


def _acquire_campaign_directory_lock(docs_dir: Path) -> int:
    """Hold a shared lifetime claim so ``copy --force`` cannot reset state."""

    import fcntl

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(docs_dir, flags)
    except OSError as error:
        raise ManualCampaignError(
            f"cannot open campaign directory for coordination: {docs_dir}"
        ) from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
    except OSError as error:
        os.close(descriptor)
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            raise ManualCampaignError(
                f"campaign directory is being reset: {docs_dir}"
            ) from None
        raise ManualCampaignError(
            f"cannot coordinate campaign directory: {docs_dir}"
        ) from error
    return descriptor


def _release_campaign_directory_lock(descriptor: int) -> None:
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _missing_ratatui_bindings() -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in ("ratatui", "ratatui_py"):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return tuple(missing)


def _configure_dashboard_capability(
    arguments: argparse.Namespace,
    *,
    installed: bool,
) -> bool:
    """Apply the release-wheel fallback and report whether it changed a run."""

    if not installed or arguments.command not in {"dashboard-snapshot", "run"}:
        return False
    missing = _missing_ratatui_bindings()
    if not missing:
        return False
    if arguments.command == "dashboard-snapshot":
        missing_text = ", ".join(f"{name!r}" for name in missing)
        raise ManualCampaignError(
            "dashboard rendering is a contributor feature unless the optional "
            f"ratatui bindings are installed (missing: {missing_text}); use "
            "`run --no-dashboard` with a release wheel, or run from a contributor "
            "checkout built with `just dev-install`"
        )
    changed = not arguments.no_dashboard
    arguments.no_dashboard = True
    return changed


def _manual_static_na_reason(cell: CellSpec) -> str | None:
    """Return only stable catalog policy exclusions.

    Runtime implementation limits belong under the ordinary worker resource
    caps rather than becoming controller-specific static N/A rows.
    """

    return REPORT_CATALOG.static_na_reason(cell)


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep examples readable while displaying option defaults."""

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        if (
            action.default is not None
            and action.default is not argparse.SUPPRESS
            and "%(default)" not in text
            and action.option_strings
        ):
            text += " (default: %(default)s)"
        return text


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One stable top-level section that can be omitted from a PDF build."""

    identifier: str
    title: str


REPORT_SECTIONS = (
    ReportSection("scope", "Scope"),
    ReportSection("recursion", "Current Recursion and Model Lowering"),
    ReportSection("colour-runtime", "Colour and Runtime Semantics"),
    ReportSection("methodology", "Benchmark Methodology"),
    ReportSection("process-matrices", "Standard-Model Process Matrices"),
    ReportSection("z-ladders", "Z-plus-Jets Evaluator Ladders"),
    ReportSection("scalar-ladders", "Colour-Singlet Model Ladders"),
    ReportSection("interpretation", "Interpretation"),
    ReportSection("worked-zgg", "Worked d dbar -> Z g g Example"),
    ReportSection(
        "shared-current-dag",
        "Shared-Current DAG and Numerical Execution",
    ),
    ReportSection("lc-layouts", "Reusable Leading-Colour Execution Layouts"),
    ReportSection(
        "execution-modes",
        "Compiled, Eager, and Recurrence Execution",
    ),
    ReportSection("ufo-support", "Supported UFO and Serialized-JSON Models"),
)


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        just_fix_windows_console()

    def paint(self, color: str, value: object, *, bright: bool = False) -> str:
        text = str(value)
        if not self.enabled:
            return text
        prefix = Style.BRIGHT if bright else ""
        return f"{prefix}{color}{text}{Style.RESET_ALL}"

    def key(self, value: object) -> str:
        return self.paint(Fore.CYAN, value, bright=True)

    def success(self, value: object) -> str:
        return self.paint(Fore.GREEN, value)

    def warning(self, value: object) -> str:
        return self.paint(Fore.YELLOW, value)

    def failure(self, value: object) -> str:
        return self.paint(Fore.RED, value)

    def neutral(self, value: object) -> str:
        return str(value)


class _SnapshotProgress(Protocol):
    def begin(self, *, total: int, attempt: int, maximum_attempts: int) -> None: ...

    def update(self, completed: int, total: int, message: str) -> None: ...

    def end(self, *, success: bool, message: str, elapsed_seconds: float) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _RefreshScanProgress:
    """One compact coloured progress line owned by ``refresh-pdf``."""

    stream: TextIO
    palette: Palette
    _active: bool = False
    _completed: int = 0
    _total: int = 0
    _last_width: int = 0

    def begin(self, *, total: int, attempt: int, maximum_attempts: int) -> None:
        self._active = True
        self._completed = 0
        self._total = total
        self.update(
            0,
            total,
            f"starting snapshot attempt {attempt}/{maximum_attempts}",
        )

    def update(self, completed: int, total: int, message: str) -> None:
        self._completed = completed
        self._total = total
        fraction = 1.0 if total == 0 else min(1.0, completed / total)
        width = 28
        filled = min(width, int(fraction * width))
        bar = self.palette.key("█" * filled) + "·" * (width - filled)
        line = (
            f"{self.palette.key('Scanning campaign artifacts')} "
            f"[{bar}] {completed:>{len(str(max(total, 1)))}}/{total} {message}"
        )
        visible_padding = max(0, self._last_width - len(line))
        self.stream.write("\r" + line + " " * visible_padding)
        self.stream.flush()
        self._last_width = len(line)

    def end(self, *, success: bool, message: str, elapsed_seconds: float) -> None:
        if not self._active:
            return
        color = self.palette.success if success else self.palette.failure
        suffix = color(f"{message} in {elapsed_seconds:.2f}s")
        self.update(self._completed, self._total, suffix)
        self.stream.write("\n")
        self.stream.flush()
        self._active = False
        self._last_width = 0

    def close(self) -> None:
        if self._active:
            self.stream.write("\n")
            self.stream.flush()
            self._active = False
            self._last_width = 0


def _color_enabled(arguments: argparse.Namespace, *, json_output: bool = False) -> bool:
    return (
        not json_output
        and not bool(getattr(arguments, "no_color", False))
        and "NO_COLOR" not in os.environ
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: object, *, sync: bool = True) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("x", encoding="ascii") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            if sync:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        _raise_disk_full(error, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _summary_status(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if normalized == "cancelled":
        return "interrupted"
    if not normalized:
        raise ManualCampaignError("campaign outcome has an empty status")
    return normalized


def _campaign_summary_categories(
    *,
    static_na_ids: Iterable[str] = (),
    result: CampaignResult | None = None,
    state: DashboardState | None = None,
    interrupted: bool = False,
) -> dict[str, set[str]]:
    categories: dict[str, set[str]] = {}

    def add(status: str, cell_id: str) -> None:
        normalized = _summary_status(status)
        if normalized in _SUMMARY_SUCCESS_STATUSES:
            return
        categories.setdefault(normalized, set()).add(cell_id)

    for cell_id in static_na_ids:
        add("static_na", cell_id)
    observed_workers = (
        {}
        if state is None
        else {
            cell_id: worker
            for cell_id, worker in state.workers.items()
            if worker.peer_instance is None
            and cell_id in state.invocation_evidence_ids
        }
    )
    outcome_ids: set[str] = set()
    if result is not None:
        for outcome in result.outcomes:
            if (
                outcome.status == "cancelled"
                and outcome.cell_id not in observed_workers
            ):
                continue
            outcome_ids.add(outcome.cell_id)
            add(outcome.status, outcome.cell_id)
    if state is not None:
        workers = tuple(observed_workers.values())
        for worker in workers:
            if worker.recycled or worker.cell_id in state.recycled_ids:
                continue
            if worker.cell_id in outcome_ids:
                continue
            if interrupted and worker.status in _ACTIVE_WORKER_STATUSES:
                add("interrupted", worker.cell_id)
            elif worker.status not in _ACTIVE_WORKER_STATUSES:
                add(worker.status, worker.cell_id)
    return categories


def _has_invocation_summary_evidence(
    result: CampaignResult | None,
    state: DashboardState,
) -> bool:
    if state.invocation_evidence_ids:
        return True
    return result is not None and any(
        outcome.status != "cancelled" for outcome in result.outcomes
    )


def _validate_existing_summary_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ManualCampaignError(
            f"campaign summary target is not a regular directory: {path}"
        )


def _publish_campaign_summary_ids(
    service: ReportService,
    categories: Mapping[str, Iterable[str]],
    *,
    fail_fast_failure_log: str | None = None,
) -> tuple[Path, dict[str, int]]:
    target = service.paths.docs_dir / "campaign_summary_ids"
    publication_root = service.paths.coordination_root / "campaign-summary-publication"
    stage: Path | None = None
    backup: Path | None = None
    counts: dict[str, int] = {}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        publication_root.mkdir(parents=True, exist_ok=True)
        if publication_root.is_symlink() or not publication_root.is_dir():
            raise ManualCampaignError(
                "campaign summary staging root is not a regular directory: "
                f"{publication_root}"
            )
        if publication_root.stat().st_dev != target.parent.stat().st_dev:
            raise ManualCampaignError(
                "campaign summary cannot be replaced atomically because its "
                "documentation and coordination roots are on different "
                f"filesystems: {target.parent} and {publication_root}"
            )
        token = uuid.uuid4().hex
        stage = publication_root / f"{target.name}-{token}.stage"
        backup = publication_root / f"{target.name}-{token}.backup"
        stage.mkdir()
        for raw_status, raw_ids in sorted(categories.items()):
            status = _summary_status(raw_status)
            if status in _SUMMARY_SUCCESS_STATUSES:
                continue
            cell_ids = tuple(sorted(set(raw_ids)))
            if not cell_ids:
                continue
            path = stage / f"{status}.txt"
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write("".join(f"{cell_id}\n" for cell_id in cell_ids))
                stream.flush()
                os.fsync(stream.fileno())
            counts[status] = len(cell_ids)
        if fail_fast_failure_log is not None:
            if not fail_fast_failure_log.strip():
                raise ManualCampaignError("fail-fast failure log is empty")
            failure_path = stage / FAIL_FAST_FAILURE_LOG
            with failure_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(fail_fast_failure_log.rstrip("\n"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(stage)
        with service.store.named_lock("campaign-summary-ids"):
            moved_old = False
            if target.exists() or target.is_symlink():
                _validate_existing_summary_directory(target)
                os.replace(target, backup)
                moved_old = True
            try:
                os.replace(stage, target)
                _fsync_directory(target.parent)
            except BaseException:
                if moved_old and not target.exists():
                    os.replace(backup, target)
                    _fsync_directory(target.parent)
                raise
            if moved_old:
                shutil.rmtree(backup)
                _fsync_directory(target.parent)
    except OSError as error:
        _raise_disk_full(error, target)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists() and target.exists():
            shutil.rmtree(backup)
    return target.resolve(strict=True), counts


def _print_campaign_summary_ids(
    path: Path,
    counts: Mapping[str, int],
    palette: Palette,
) -> None:
    print()
    print(
        _table(
            ("campaign summary", "entries"),
            (
                (palette.key(status), palette.warning(count))
                for status, count in sorted(counts.items())
            )
            if counts
            else ((palette.success("non-OK entries"), palette.success(0)),),
            align={"entries": "r"},
        )
    )
    print(f"Campaign summary IDs: {path}")


def _normal(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _positive_finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a finite number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _nonnegative_finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a finite number") from error
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative finite number")
    return result


def _flatten(values: Sequence[Sequence[str]] | None) -> tuple[str, ...]:
    if not values:
        return ()
    flattened = tuple(item for group in values for item in group)
    if any(_normal(item) in {"*", "all"} for item in flattened):
        return ()
    return flattened


def _shell_join(arguments: Iterable[object]) -> str:
    return shlex.join(tuple(str(argument) for argument in arguments))


def _table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    align: Mapping[str, str] | None = None,
    max_width: Mapping[str, int] | None = None,
) -> str:
    value = PrettyTable()
    value.field_names = list(headers)
    value.hrules = 1
    value.vrules = 1
    value.padding_width = 1
    for header in headers:
        value.align[header] = "l"
    if align is not None:
        for header, direction in align.items():
            value.align[header] = direction
    if max_width is not None:
        for header, width in max_width.items():
            value.max_width[header] = width
    for row in rows:
        value.add_row(list(row))
    return value.get_string()


_MODEL_ALIASES = {
    "builtin": ModelKey.BUILTIN_SM,
    "built_in": ModelKey.BUILTIN_SM,
    "built_in_sm": ModelKey.BUILTIN_SM,
    "builtin_sm": ModelKey.BUILTIN_SM,
    "sm": ModelKey.BUILTIN_SM,
    "ufo": ModelKey.UFO_SM,
    "sm_ufo": ModelKey.UFO_SM,
    "ufo_sm": ModelKey.UFO_SM,
    "external_sm": ModelKey.UFO_SM,
    "scalar_contact": ModelKey.SCALAR_CONTACT,
    "scalar_gravity": ModelKey.SCALAR_GRAVITY,
}
_ACCURACY_ALIASES = {
    "lc": Accuracy.LC,
    "leading_color": Accuracy.LC,
    "leading_colour": Accuracy.LC,
    "nlc": Accuracy.NLC,
    "next_to_leading_color": Accuracy.NLC,
    "next_to_leading_colour": Accuracy.NLC,
    "full": Accuracy.FULL,
    "full_color": Accuracy.FULL,
    "full_colour": Accuracy.FULL,
}
_MODE_ALIASES = {
    "amplicol": ExecutionMode.AMPLICOL,
    "legacy": ExecutionMode.AMPLICOL,
    "recurrence": ExecutionMode.RECURRENCE,
    "ampli_col": ExecutionMode.RECURRENCE,
    "compiled": ExecutionMode.COMPILED,
    "eager": ExecutionMode.EAGER,
    "on_the_fly": ExecutionMode.ON_THE_FLY,
    "onthefly": ExecutionMode.ON_THE_FLY,
    "otf": ExecutionMode.ON_THE_FLY,
}
_WORKLOAD_ALIASES = {
    "non_union_flow": Workload.SELECTED_FLOW,
    "selected_flow": Workload.SELECTED_FLOW,
    "single_flow": Workload.SELECTED_FLOW,
    "single_flow_hel_sum": Workload.SELECTED_FLOW,
    "topology_replay": Workload.SELECTED_FLOW,
    "union_flow": Workload.ALL_FLOW,
    "all_flow": Workload.ALL_FLOW,
    "all_flows": Workload.ALL_FLOW,
    "all_flows_single_hel": Workload.ALL_FLOW,
    "all_flow_union": Workload.ALL_FLOW,
    "contracted": Workload.CONTRACTED,
}


def _dataset_aliases(catalog: ReportCatalog) -> dict[str, frozenset[str]]:
    dataset_ids = frozenset(cell.dataset_id for cell in catalog.measurement_cells())
    aliases: dict[str, frozenset[str]] = {
        _normal(dataset): frozenset({dataset}) for dataset in dataset_ids
    }
    aliases.update(
        {
            "z": frozenset(
                dataset for dataset in dataset_ids if dataset.startswith("z_")
            ),
            "z_table": frozenset(
                dataset for dataset in dataset_ids if dataset.startswith("z_")
            ),
            "matrix": frozenset(
                dataset for dataset in dataset_ids if dataset.startswith("matrix_")
            ),
            "matrix_table": frozenset(
                dataset for dataset in dataset_ids if dataset.startswith("matrix_")
            ),
            "matrix_best": frozenset(
                dataset
                for dataset in dataset_ids
                if dataset.startswith("matrix_") and "_builtin_sm_" in dataset
            ),
            "reference": frozenset(
                dataset
                for dataset in dataset_ids
                if dataset.startswith("reference_amplicol_")
            ),
            "scalar": frozenset(
                dataset for dataset in dataset_ids if dataset.startswith("scalar_")
            ),
        }
    )
    return aliases


def _invalid_value(
    option: str,
    value: str,
    allowed: Iterable[str],
) -> ManualCampaignError:
    choices = tuple(sorted(set(allowed)))
    suggestions = difflib.get_close_matches(_normal(value), choices, n=3, cutoff=0.45)
    suffix = "" if not suggestions else f"; did you mean {', '.join(suggestions)}?"
    if len(choices) > 24:
        displayed = (
            *choices[:24],
            f"… {len(choices) - 24} more (use --help or a dry run to browse)",
        )
    else:
        displayed = choices
    return ManualCampaignError(
        f"{option}: unsupported value {value!r}{suffix}\n"
        + _table(("option", "allowed values"), ((option, ", ".join(displayed)),))
    )


def _empty_selection_error(
    arguments: argparse.Namespace,
    catalog: ReportCatalog,
) -> ManualCampaignError:
    provided = tuple(
        option
        for attribute, option in (
            ("table", "--table"),
            ("process_id", "--process-id"),
            ("multiplicity", "--multiplicity"),
            ("color_approximation", "--color-approximation"),
            ("generation_mode", "--generation-mode"),
            ("generation_engine", "--generation-engine"),
            ("model", "--model"),
            ("variant", "--variant"),
            ("cell_id", "--cell-id"),
            ("cell_id_file", "--cell-id-file"),
        )
        if getattr(arguments, attribute, None)
    )
    variants = sorted(
        {
            str(cell.variant)
            for cell in catalog.measurement_cells()
            if cell.variant is not None
        }
    )
    multiplicities = sorted({cell.n_final for cell in catalog.measurement_cells()})
    processes = tuple(
        f"{family.identifier}:{family.key}" for family in PROCESS_FAMILIES
    )
    rows = (
        (
            "--table",
            "z_table, matrix_table, matrix_best, scalar_contact, "
            "scalar_gravity, reference",
        ),
        ("--process-id", ", ".join(processes)),
        ("--multiplicity", ", ".join(str(value) for value in multiplicities)),
        ("--color-approximation", "lc, nlc, full"),
        ("--generation-mode", "non-union-flow, union-flow, contracted"),
        (
            "--generation-engine",
            "amplicol, recurrence, compiled, eager, on-the-fly (otf)",
        ),
        ("--model", "built_in, sm_ufo, scalar_contact, scalar_gravity"),
        ("--variant", ", ".join(variants)),
        (
            "--cell-id",
            "one exact canonical cell ID (see --help or a broader dry run)",
        ),
        (
            "--cell-id-file",
            "UTF-8 files containing one exact canonical cell ID per line",
        ),
    )
    suggestion = (
        "Remove one selector at a time to find the conflicting dimension"
        if not provided
        else "Try removing one of " + ", ".join(provided)
    )
    return ManualCampaignError(
        "the selector intersection contains no catalog entries. "
        f"{suggestion}; wildcard `*`/`all` is also accepted.\n"
        + _table(
            ("selector", "canonical values / useful aliases"),
            rows,
            max_width={"canonical values / useful aliases": 96},
        )
    )


def _resolve_map(
    option: str,
    raw_values: Sequence[str],
    aliases: Mapping[str, Any],
) -> frozenset[Any]:
    resolved: set[Any] = set()
    for raw in raw_values:
        key = _normal(raw)
        try:
            resolved.add(aliases[key])
        except KeyError:
            raise _invalid_value(option, raw, aliases) from None
    return frozenset(resolved)


def _resolve_datasets(
    raw_values: Sequence[str],
    catalog: ReportCatalog,
) -> frozenset[str]:
    aliases = _dataset_aliases(catalog)
    resolved: set[str] = set()
    for raw in raw_values:
        key = _normal(raw)
        try:
            resolved.update(aliases[key])
        except KeyError:
            raise _invalid_value("--table", raw, aliases) from None
    return frozenset(resolved)


def _resolve_processes(
    raw_values: Sequence[str],
) -> tuple[frozenset[str], frozenset[str]]:
    by_id = {str(family.identifier): family.key for family in PROCESS_FAMILIES}
    by_key = {_normal(family.key): family.key for family in PROCESS_FAMILIES}
    process_keys: set[str] = set()
    concrete: set[str] = set()
    allowed = {**by_id, **by_key}
    for raw in raw_values:
        key = _normal(raw)
        if key in allowed:
            process_keys.add(allowed[key])
        elif ">" in raw:
            concrete.add(" ".join(raw.split()))
        else:
            raise _invalid_value("--process-id", raw, allowed) from None
    return frozenset(process_keys), frozenset(concrete)


def _cell_ids_from_files(
    raw_groups: Sequence[Sequence[Path]] | None,
    *,
    known_cells: frozenset[str],
) -> frozenset[str]:
    selected: set[str] = set()
    for group in raw_groups or ():
        for raw_path in group:
            path = Path(raw_path).expanduser()
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                raise ManualCampaignError(
                    f"--cell-id-file could not read {path}: {error}"
                ) from error
            for line_number, line in enumerate(lines, start=1):
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                location = f"{path}:{line_number}"
                if _normal(value) in {"*", "all"}:
                    raise ManualCampaignError(
                        f"{location}: wildcards are not allowed in a cell-ID file"
                    )
                if value not in known_cells:
                    suggestions = difflib.get_close_matches(
                        value,
                        sorted(known_cells),
                        n=3,
                        cutoff=0.45,
                    )
                    suffix = (
                        ""
                        if not suggestions
                        else f"; did you mean {', '.join(suggestions)}?"
                    )
                    raise ManualCampaignError(
                        f"{location}: unknown canonical cell ID {value!r}{suffix}"
                    )
                selected.add(value)
    return frozenset(selected)


def selection_from_arguments(
    arguments: argparse.Namespace,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> tuple[CellSelection, tuple[CellSpec, ...]]:
    raw_tables = _flatten(getattr(arguments, "table", None))
    raw_processes = _flatten(getattr(arguments, "process_id", None))
    raw_multiplicities = _flatten(getattr(arguments, "multiplicity", None))
    raw_accuracies = _flatten(getattr(arguments, "color_approximation", None))
    raw_workloads = _flatten(getattr(arguments, "generation_mode", None))
    raw_modes = _flatten(getattr(arguments, "generation_engine", None))
    raw_models = _flatten(getattr(arguments, "model", None))
    raw_variants = _flatten(getattr(arguments, "variant", None))
    raw_cell_id_groups = getattr(arguments, "cell_id", None)
    raw_cell_ids = _flatten(raw_cell_id_groups)
    direct_cell_wildcard = any(
        _normal(value) in {"*", "all"}
        for group in raw_cell_id_groups or ()
        for value in group
    )

    multiplicities: set[int] = set()
    for raw in raw_multiplicities:
        try:
            value = int(raw)
        except ValueError:
            raise ManualCampaignError(
                f"--multiplicity expects positive integers, not {raw!r}"
            ) from None
        if value < 1:
            raise ManualCampaignError("--multiplicity values must be positive")
        multiplicities.add(value)

    process_keys, processes = _resolve_processes(raw_processes)
    known_variants = frozenset(
        cell.variant for cell in catalog.measurement_cells() if cell.variant is not None
    )
    variants: set[str] = set()
    for raw in raw_variants:
        matches = {
            variant for variant in known_variants if _normal(variant) == _normal(raw)
        }
        if not matches:
            raise _invalid_value("--variant", raw, map(_normal, known_variants))
        variants.update(matches)

    known_cells = frozenset(cell.cell_id for cell in catalog.measurement_cells())
    cell_ids: set[str] = set()
    for raw in raw_cell_ids:
        normalized = raw.strip()
        if _normal(normalized) in {"*", "all"}:
            continue
        if normalized not in known_cells:
            raise _invalid_value("--cell-id", normalized, known_cells)
        cell_ids.add(normalized)
    file_cell_ids = _cell_ids_from_files(
        getattr(arguments, "cell_id_file", None),
        known_cells=known_cells,
    )
    if (
        getattr(arguments, "cell_id_file", None)
        and not file_cell_ids
        and not raw_cell_ids
        and not direct_cell_wildcard
    ):
        raise ManualCampaignError(
            "--cell-id-file inputs contain no canonical cell IDs"
        )
    if not direct_cell_wildcard:
        cell_ids.update(file_cell_ids)

    resolved_modes = _resolve_map(
        "--generation-engine",
        raw_modes,
        _MODE_ALIASES,
    )
    resolved_models = _resolve_map("--model", raw_models, _MODEL_ALIASES)
    selection = CellSelection(
        datasets=_resolve_datasets(raw_tables, catalog),
        modes=resolved_modes,
        # Legacy AmpliCol represents the built-in SM but stores model=None.
        # Apply the friendly model filter below so built_in + amplicol works.
        models=frozenset(),
        accuracies=_resolve_map(
            "--color-approximation",
            raw_accuracies,
            _ACCURACY_ALIASES,
        ),
        process_keys=process_keys,
        processes=processes,
        multiplicities=frozenset(multiplicities),
        variants=frozenset(variants),
        workloads=_resolve_map(
            "--generation-mode",
            raw_workloads,
            _WORKLOAD_ALIASES,
        ),
        cell_ids=frozenset(cell_ids),
    )
    selected = list(select_cells(selection, catalog=catalog))

    normalized_tables = {_normal(value) for value in raw_tables}
    if ExecutionMode.AMPLICOL in resolved_modes:
        supplemental_datasets: frozenset[str] = frozenset()
        supplemental_process_keys = selection.process_keys
        supplemental_workloads = selection.workloads
        supplemental_accuracies = selection.accuracies
        if "z" in normalized_tables or "z_table" in normalized_tables:
            supplemental_datasets = frozenset({"reference_amplicol_lc"})
            supplemental_process_keys = frozenset({"dd_z_jets"})
            supplemental_workloads = frozenset({Workload.SELECTED_FLOW})
            supplemental_accuracies = frozenset({Accuracy.LC})
        elif normalized_tables.intersection({"matrix", "matrix_table", "matrix_best"}):
            supplemental_datasets = frozenset(
                {
                    "reference_amplicol_lc",
                    "reference_amplicol_nlc",
                    "reference_amplicol_full",
                }
            )
        if supplemental_datasets:
            supplement = CellSelection(
                datasets=supplemental_datasets,
                modes=frozenset({ExecutionMode.AMPLICOL}),
                accuracies=supplemental_accuracies,
                process_keys=supplemental_process_keys,
                processes=selection.processes,
                multiplicities=selection.multiplicities,
                variants=selection.variants,
                workloads=supplemental_workloads,
                cell_ids=selection.cell_ids,
            )
            selected.extend(select_cells(supplement, catalog=catalog))

    if resolved_models:
        selected = [
            cell
            for cell in selected
            if cell.measurement.model in resolved_models
            or (
                cell.measurement.execution_mode is ExecutionMode.AMPLICOL
                and ModelKey.BUILTIN_SM in resolved_models
            )
        ]
    cells = tuple(
        sorted(
            {cell.cell_id: cell for cell in selected}.values(),
            key=lambda item: item.cell_id,
        )
    )
    if not cells:
        raise _empty_selection_error(arguments, catalog)
    return selection, cells


def _model_cli_source(cell: CellSpec, repo_root: Path) -> str:
    from .runner import _model_source_path

    model = cell.measurement.model
    if model is None:
        return "<legacy-AmpliCol-has-no-pyamplicol-model>"
    source = _model_source_path(repo_root, model)
    return "built-in-sm" if source is None else os.fspath(source)


def _reproduction_root(artifact_root: Path, cell: CellSpec) -> Path:
    return artifact_root / "manual-reproductions" / cell.cell_id


def _pyamplicol_cli(repo_root: Path) -> tuple[str, ...]:
    """Return a cwd-independent source CLI prefix for contributor installs."""

    source_package = repo_root / "src/pyamplicol"
    executable = repo_root / ".venv/bin/pyamplicol"
    if not source_package.is_dir() or not executable.is_file():
        return (sys.executable, "-m", "pyamplicol")
    return (
        "env",
        f"PYTHONPATH={os.fspath((repo_root / 'src').resolve(strict=False))}",
        os.fspath(executable.expanduser().resolve(strict=False)),
    )


def _materialize_reproduction_momenta(
    cell: CellSpec,
    *,
    artifact_root: Path,
    momenta: object,
) -> Path:
    """Publish the exact recorded report points for public recurrence."""

    path = _reproduction_root(artifact_root, cell) / "momenta.json"
    _atomic_json(path, momenta)
    return path


@dataclass(frozen=True, slots=True)
class ReproductionRecipe:
    kind: str
    prepare: tuple[str, ...] | None
    generate: tuple[str, ...] | None
    profile: tuple[str, ...] | None
    note: str
    exact: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "abi": PORTABLE_CURRENT_REPRODUCTION_RECIPE_ABI,
            "kind": self.kind,
            "prepare": None if self.prepare is None else list(self.prepare),
            "generate": None if self.generate is None else list(self.generate),
            "profile": None if self.profile is None else list(self.profile),
            "note": self.note,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class ReproductionSettings:
    """Effective invocation settings used for active public CLI recipes."""

    cores: int = DEFAULT_CORES_PER_WORKER
    target_runtime: float = DEFAULT_TARGET_RUNTIME_SECONDS
    batch_size: int = DEFAULT_BATCH_SIZE
    warmups: int = DEFAULT_WARMUP_RUNS
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES


def _validated_recipe_numerical_relation_fallback(
    provenance: object,
) -> Mapping[str, object] | None:
    if not isinstance(provenance, Mapping):
        return None
    raw = provenance.get("numerical_relation_fallback")
    if raw is None:
        return None
    from pyamplicol.generation.recurrence_numerical_current_warmup import (
        EVIDENCE_ENVELOPE_FALLBACK_REASON,
        NUMERICAL_RELATION_FALLBACK_ABI,
        RecurrenceNumericalEvidenceGeometry,
    )

    expected_fields = {
        "abi",
        "requested_mode",
        "effective_mode",
        "effective_reuse_state",
        "reason",
        "geometry",
        "certified_relation_count",
        "applied_relation_count",
    }
    geometry_fields = {
        "current_count",
        "component_count",
        "candidate_probe_count",
        "verification_probe_count",
        "runtime_parameter_count",
        "scalar_count",
        "row_count",
    }
    geometry = raw.get("geometry") if isinstance(raw, Mapping) else None
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected_fields
        or raw.get("abi") != NUMERICAL_RELATION_FALLBACK_ABI
        or raw.get("requested_mode") != "certified-reuse"
        or raw.get("effective_mode") != "off"
        or raw.get("effective_reuse_state") != "disabled"
        or raw.get("reason") != EVIDENCE_ENVELOPE_FALLBACK_REASON
        or raw.get("certified_relation_count") != 0
        or isinstance(raw.get("certified_relation_count"), bool)
        or raw.get("applied_relation_count") != 0
        or isinstance(raw.get("applied_relation_count"), bool)
        or not isinstance(geometry, Mapping)
        or set(geometry) != geometry_fields
        or any(type(geometry.get(field)) is not int for field in geometry_fields)
    ):
        raise ManualCampaignError(
            "measurement numerical-relation fallback provenance is malformed"
        )
    try:
        expected_geometry = RecurrenceNumericalEvidenceGeometry.from_counts(
            current_count=int(geometry["current_count"]),
            component_count=int(geometry["component_count"]),
            candidate_probe_count=int(geometry["candidate_probe_count"]),
            verification_probe_count=int(geometry["verification_probe_count"]),
            runtime_parameter_count=int(geometry["runtime_parameter_count"]),
        ).to_json_dict()
    except (OverflowError, TypeError, ValueError) as error:
        raise ManualCampaignError(
            "measurement numerical-relation fallback provenance is malformed"
        ) from error
    if dict(geometry) != expected_geometry:
        raise ManualCampaignError(
            "measurement numerical-relation fallback geometry is inconsistent"
        )
    return raw


def reproduction_recipe(
    cell: CellSpec,
    *,
    repo_root: Path,
    artifact_root: Path,
    artifact_path: str = "<ARTIFACT_DIR>",
    cores: int = DEFAULT_CORES_PER_WORKER,
    target_runtime: float = DEFAULT_TARGET_RUNTIME_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmups: int = DEFAULT_WARMUP_RUNS,
    minimum_samples: int = DEFAULT_MINIMUM_SAMPLES,
    measurement: Mapping[str, object] | None = None,
) -> ReproductionRecipe:
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        return ReproductionRecipe(
            kind="legacy-report-adapter",
            prepare=None,
            generate=None,
            profile=None,
            note="Original AmpliCol has no public pyamplicol CLI subcommand.",
            exact=False,
        )
    cli = _pyamplicol_cli(repo_root)
    if artifact_path == "<ARTIFACT_DIR>":
        artifact_path = os.fspath(_reproduction_root(artifact_root, cell) / "artifact")
    model_cache = _reproduction_root(artifact_root, cell) / "model-cache"
    provenance = (
        measurement.get("provenance") if isinstance(measurement, Mapping) else None
    )
    numerical_relation_fallback = _validated_recipe_numerical_relation_fallback(
        provenance
    )
    requested = (
        provenance.get("requested_config") if isinstance(provenance, Mapping) else None
    )
    requested_model = requested.get("model") if isinstance(requested, Mapping) else None
    recorded_model_source = (
        requested_model.get("source") if isinstance(requested_model, Mapping) else None
    )
    raw_model_source = (
        recorded_model_source
        if isinstance(recorded_model_source, str) and recorded_model_source
        else _model_cli_source(cell, repo_root)
    )
    artifact_record = (
        measurement.get("artifact") if isinstance(measurement, Mapping) else None
    )
    process_selector = cell.process
    completed = (
        isinstance(measurement, Mapping)
        and measurement.get("status") == ResultStatus.OK.value
        and isinstance(artifact_record, Mapping)
        and isinstance(artifact_record.get("path"), str)
        and isinstance(artifact_record.get("process_id"), str)
    )
    prepare: tuple[str, ...] | None = None
    model_source = raw_model_source
    prepared_execution = cell.measurement.execution_mode in {
        ExecutionMode.EAGER,
        ExecutionMode.RECURRENCE,
        ExecutionMode.ON_THE_FLY,
    }
    if prepared_execution and cell.measurement.model in {
        ModelKey.BUILTIN_SM,
        ModelKey.UFO_SM,
    }:
        prepared_source = _model_cli_source(cell, repo_root)
        prepared_stem = (
            "built-in-sm" if cell.measurement.model is ModelKey.BUILTIN_SM else "ufo-sm"
        )
        prepared_path = (
            _reproduction_root(artifact_root, cell)
            / "prepared-models"
            / f"{prepared_stem}-jit-o2.pyamplicol-model"
        )
        prepare = (
            *cli,
            "model",
            "compile",
            prepared_source,
            os.fspath(prepared_path),
            "--model-cache-dir",
            os.fspath(model_cache),
            "--model-cache",
            "--backend",
            "jit",
            "--cores",
            str(cores),
            "--jit-optimization-level",
            "2",
            "--set",
            "evaluator.execution_mode=recurrence",
            "--json",
            "--color",
            "always",
            "--progress",
            "tty",
        )
        model_source = os.fspath(prepared_path)
    on_the_fly = cell.measurement.execution_mode is ExecutionMode.ON_THE_FLY
    layout = (
        "topology-replay"
        if on_the_fly
        else (
            "all-flow-union"
            if cell.workload is Workload.ALL_FLOW
            else "topology-replay"
        )
    )
    numerical_reuse_flag = (
        "--no-numerical-current-reuse"
        if on_the_fly or numerical_relation_fallback is not None
        else "--numerical-current-reuse"
    )
    generate = (
        *cli,
        "generate",
        cell.process,
        artifact_path,
        "--model",
        model_source,
        "--model-cache",
        "--model-cache-dir",
        os.fspath(model_cache),
        "--color-accuracy",
        cell.measurement.accuracy.value,
        "--lc-flow-layout",
        layout,
        "--execution-mode",
        cell.measurement.execution_mode.value,
        "--backend",
        cell.measurement.backend,
        "--workers",
        str(cores),
        "--cores",
        str(cores),
        "--batch-size",
        str(batch_size),
        "--output-chunk-size",
        "512",
        "--horner-iterations",
        "10",
        "--cpe-iterations",
        "none",
        "--max-horner-variables",
        "1000",
        "--max-common-pair-cache-entries",
        "5000000",
        "--max-common-pair-distance",
        "1000",
        "--jit-optimization-level",
        str(cell.measurement.jit_optimization_level or 2),
        "--cpp-optimization",
        "O3",
        "--no-emit-api-bundle" if on_the_fly else "--emit-api-bundle",
        "--no-validation" if on_the_fly else "--validation",
        "--validation-samples",
        "1",
        "--validation-seed",
        "12345",
        "--relative-tolerance",
        "1e-12",
        "--absolute-tolerance",
        "1e-300",
        "--no-post-build-validation",
        numerical_reuse_flag,
        "--force",
        "--color",
        "always",
        "--progress",
        "tty",
    )
    selector_contract = (
        measurement.get("selector_contract")
        if isinstance(measurement, Mapping)
        else None
    )
    selector: tuple[str, ...] = ()
    if isinstance(selector_contract, Mapping):
        if cell.workload is Workload.SELECTED_FLOW:
            raw_flows = selector_contract.get("selected_color_flow_ids")
            if isinstance(raw_flows, Sequence) and not isinstance(
                raw_flows, (str, bytes)
            ):
                selector = tuple(
                    argument
                    for value in raw_flows
                    for argument in ("--color-flow", str(value))
                )
        elif cell.workload is Workload.ALL_FLOW:
            try:
                from .runner import SelectorContract

                raw_helicities: object = SelectorContract.from_mapping(
                    selector_contract
                ).runtime_all_flow_helicity_ids
            except (TypeError, ValueError):
                raw_helicities = ()
            if isinstance(raw_helicities, Sequence):
                selector = tuple(
                    argument
                    for value in raw_helicities
                    for argument in ("--helicity", str(value))
                )
    selector_ready = cell.workload is Workload.CONTRACTED or bool(selector)
    recorded_momenta = (
        provenance.get("report_momenta") if isinstance(provenance, Mapping) else None
    )
    momenta_ready = isinstance(recorded_momenta, Sequence) and not isinstance(
        recorded_momenta,
        (str, bytes),
    )
    exact = (
        completed
        and cell.measurement.execution_mode
        in {ExecutionMode.RECURRENCE, ExecutionMode.ON_THE_FLY}
        and selector_ready
        and momenta_ready
    )
    momenta: tuple[str, ...] = ()
    if exact:
        momenta = (
            "--momenta",
            os.fspath(
                _materialize_reproduction_momenta(
                    cell,
                    artifact_root=artifact_root,
                    momenta=recorded_momenta,
                )
            ),
        )
    profile = (
        *cli,
        "profile",
        artifact_path,
        "--process",
        process_selector,
        "--target-runtime",
        f"{target_runtime:g}",
        "--batch-size",
        str(batch_size),
        "--warmup-runs",
        str(warmups),
        "--minimum-samples",
        str(minimum_samples),
        "--precision",
        "16",
        *momenta,
        *selector,
        "--json",
        "--color",
        "always",
        "--progress",
        "tty",
    )
    if cell.measurement.execution_mode is ExecutionMode.COMPILED:
        kind = "public-cli+precompiled-generation+paired-arena-exceptions"
        note = (
            "Diagnostic only: the report injects a precompiled model outside "
            "generation_seconds and publishes a paired private Arena profile; "
            "the public commands do not reproduce either timing boundary."
        )
    elif cell.measurement.execution_mode is ExecutionMode.EAGER:
        kind = "public-cli+paired-arena-profile-exception"
        note = (
            "Generation uses the public CLI path. Profile is diagnostic only: "
            "the report publishes a paired private Arena timing boundary."
        )
    elif exact:
        kind = "public-cli-exact"
        note = (
            "Exact completed-cell public commands with the published stable "
            "selector and materialized report momenta."
        )
    else:
        kind = "public-cli-template"
        note = (
            "Prepare (when shown) and generate are runnable. Profile is a "
            "template only: its stable flow/helicity selector and exact "
            "momenta are published after successful generation/measurement."
        )
    if numerical_relation_fallback is not None:
        kind += "+effective-reuse-off-fallback"
        note += (
            " The measured generation took the authenticated "
            "evidence-envelope fallback: numerical-current reuse was effectively "
            "disabled, so the reproduction command uses "
            "--no-numerical-current-reuse."
        )
    if prepare is not None:
        kind += "+model-compile-prerequisite"
    return ReproductionRecipe(
        kind=kind,
        prepare=prepare,
        generate=generate,
        profile=profile,
        note=note,
        exact=exact,
    )


def _git_directory(repo_root: Path) -> Path:
    git_marker = repo_root / ".git"
    if git_marker.is_file():
        text = git_marker.read_text(encoding="ascii").strip()
        prefix = "gitdir:"
        if not text.startswith(prefix):
            raise ManualCampaignError(".git worktree marker is malformed")
        git_dir = Path(text[len(prefix) :].strip())
        if not git_dir.is_absolute():
            git_dir = (repo_root / git_dir).resolve(strict=False)
    else:
        git_dir = git_marker
    if not git_dir.is_dir():
        raise ManualCampaignError("checkout Git metadata directory is unavailable")
    return git_dir


def _git_common_directory(git_dir: Path) -> Path:
    commondir = git_dir / "commondir"
    if not commondir.is_file():
        return git_dir
    return (git_dir / commondir.read_text(encoding="ascii").strip()).resolve(
        strict=False
    )


def _repo_head(repo_root: Path) -> str:
    """Read the checkout revision without invoking Git."""

    git_dir = _git_directory(repo_root)
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if head.startswith("ref: "):
        reference = head[5:]
        loose = git_dir / reference
        if loose.is_file():
            head = loose.read_text(encoding="ascii").strip()
        else:
            common_dir = _git_common_directory(git_dir)
            common_loose = common_dir / reference
            if common_loose.is_file():
                head = common_loose.read_text(encoding="ascii").strip()
            else:
                packed = common_dir / "packed-refs"
                value = None
                if packed.is_file():
                    for line in packed.read_text(encoding="ascii").splitlines():
                        if line.startswith(("#", "^")):
                            continue
                        candidate, separator, name = line.partition(" ")
                        if separator and name == reference:
                            value = candidate
                            break
                if value is None:
                    raise ManualCampaignError(
                        f"cannot resolve Git reference {reference!r}"
                    )
                head = value
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ManualCampaignError("checkout HEAD is not a lowercase 40-hex revision")
    return head


def _commit_tree_marker(repo_root: Path, revision: str) -> str:
    """Read HEAD's tree, using one Git query only when the commit is packed."""

    common_dir = _git_common_directory(_git_directory(repo_root))
    object_path = common_dir / "objects" / revision[:2] / revision[2:]
    try:
        raw = zlib.decompress(object_path.read_bytes())
    except (OSError, zlib.error):
        # Ordinary clones and repositories after ``git gc`` store commits in
        # pack files.  Parsing Git's pack format here would duplicate Git and
        # be substantially more fragile than this single read-only fallback.
        try:
            completed = subprocess.run(
                ("git", "rev-parse", "--verify", f"{revision}^{{tree}}"),
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return f"manual-revision:{revision}"
        tree = completed.stdout.strip()
        if completed.returncode == 0 and _FULL_SOURCE_REVISION.fullmatch(tree):
            return tree
        return f"manual-revision:{revision}"
    _header, separator, body = raw.partition(b"\0")
    if not separator:
        return f"manual-revision:{revision}"
    for line in body.splitlines():
        if not line.startswith(b"tree "):
            continue
        value = line[5:].decode("ascii", errors="ignore")
        if len(value) == 40 and all(
            character in "0123456789abcdef" for character in value
        ):
            return value
    return f"manual-revision:{revision}"


def _index_tree_marker(
    entries: Sequence[tuple[bytes, int, bytes]],
) -> str:
    """Build Git's tree identity from index object IDs, never file contents."""

    root: dict[bytes, Any] = {}
    for raw_path, indexed_mode, object_id in entries:
        parts = raw_path.split(b"/")
        if not parts or any(part in {b"", b".", b".."} for part in parts):
            raise ManualCampaignError("checkout index contains an unsafe path")
        node = root
        for part in parts[:-1]:
            existing = node.get(part)
            if existing is None:
                child: dict[bytes, Any] = {}
                node[part] = child
                node = child
            elif isinstance(existing, dict):
                node = existing
            else:
                raise ManualCampaignError(
                    "checkout index contains a file/directory path collision"
                )
        leaf = parts[-1]
        if leaf in node:
            raise ManualCampaignError("checkout index contains duplicate paths")
        kind = indexed_mode & 0o170000
        if kind == 0o100000:
            tree_mode = 0o100755 if indexed_mode & 0o111 else 0o100644
        elif kind in {0o120000, 0o160000}:
            tree_mode = kind
        else:
            raise ManualCampaignError(
                f"checkout index contains unsupported mode {indexed_mode:o}"
            )
        node[leaf] = (tree_mode, object_id)

    def encode_tree(node: Mapping[bytes, Any]) -> bytes:
        records: list[bytes] = []
        ordered = sorted(
            node.items(),
            key=lambda item: item[0] + (b"/" if isinstance(item[1], dict) else b""),
        )
        for name, value in ordered:
            if isinstance(value, dict):
                mode = b"40000"
                object_id = encode_tree(value)
            else:
                indexed_mode, object_id = value
                mode = f"{indexed_mode:o}".encode("ascii")
            records.append(mode + b" " + name + b"\0" + object_id)
        payload = b"".join(records)
        # SHA-1 here is Git's tree-object address over index metadata.  No
        # worktree or performance artifact bytes are read or hashed.
        return hashlib.sha1(
            b"tree " + str(len(payload)).encode("ascii") + b"\0" + payload,
            usedforsecurity=False,
        ).digest()

    return encode_tree(root).hex()


def _index_metadata_dirty_paths(
    repo_root: Path,
    *,
    committed_tree: str | None = None,
) -> tuple[str, ...]:
    """Check index/worktree metadata and index-vs-HEAD with a packed-HEAD fallback."""

    index_path = _git_directory(repo_root) / "index"
    try:
        data = index_path.read_bytes()
    except OSError as error:
        raise ManualCampaignError(f"checkout index is unreadable: {error}") from error
    if len(data) < 12:
        raise ManualCampaignError("checkout index is truncated")
    signature, version, count = struct.unpack("!4sII", data[:12])
    if signature != b"DIRC" or version not in {2, 3}:
        raise ManualCampaignError(
            f"unsupported checkout index format: signature={signature!r}, "
            f"version={version}"
        )
    offset = 12
    tracked: set[str] = set()
    dirty: list[str] = []
    tree_entries: list[tuple[bytes, int, bytes]] = []
    unmerged = False
    for _entry in range(count):
        start = offset
        if offset + 62 > len(data):
            raise ManualCampaignError("checkout index entry is truncated")
        fields = struct.unpack("!10I20sH", data[offset : offset + 62])
        offset += 62
        mtime_seconds, mtime_nanoseconds = fields[2], fields[3]
        indexed_mode, indexed_size, flags = fields[6], fields[9], fields[11]
        object_id = fields[10]
        if version >= 3 and flags & 0x4000:
            offset += 2
        encoded_length = flags & 0x0FFF
        if encoded_length < 0x0FFF:
            end = offset + encoded_length
        else:
            try:
                end = data.index(b"\0", offset)
            except ValueError as error:
                raise ManualCampaignError(
                    "checkout index path is unterminated"
                ) from error
        if end >= len(data):
            raise ManualCampaignError("checkout index path is truncated")
        raw_relative = data[offset:end]
        relative = raw_relative.decode("utf-8", errors="surrogateescape")
        tracked.add(relative)
        offset = end + 1
        while (offset - start) % 8:
            offset += 1
        if flags & 0x3000:
            unmerged = True
        else:
            tree_entries.append((raw_relative, indexed_mode, object_id))
        if _generated_report_path(relative):
            continue
        path = repo_root / relative
        try:
            observed = path.lstat()
        except OSError:
            dirty.append(relative)
            continue
        metadata = (
            observed.st_mtime_ns // 1_000_000_000,
            observed.st_mtime_ns % 1_000_000_000,
            observed.st_size,
            observed.st_mode & 0o170000,
            observed.st_mode & 0o111,
        )
        expected = (
            mtime_seconds,
            mtime_nanoseconds,
            indexed_size,
            indexed_mode & 0o170000,
            indexed_mode & 0o111,
        )
        if metadata != expected:
            dirty.append(relative)
    if committed_tree is None:
        try:
            revision = _repo_head(repo_root)
        except (ManualCampaignError, OSError):
            # Keep the metadata helper usable for deliberately minimal test
            # indexes.  lightweight_source_identity() resolves HEAD first and
            # therefore never takes this compatibility path.
            revision = None
        if revision is not None:
            committed_tree = _commit_tree_marker(repo_root, revision)
    if committed_tree is not None:
        if unmerged:
            dirty.append("<unmerged Git index>")
        elif committed_tree.startswith("manual-revision:"):
            dirty.append("<index-vs-HEAD tree unavailable>")
        elif _index_tree_marker(tree_entries) != committed_tree:
            dirty.append("<staged index differs from HEAD>")
    critical = (
        "tools/performance_report/manual_campaign.py",
        "src/pyamplicol/_profiling_campaign/steer_performance_campaign.py",
    )
    dirty.extend(
        relative
        for relative in critical
        if (repo_root / relative).exists() and relative not in tracked
    )
    return tuple(sorted(set(dirty)))


def lightweight_source_identity(repo_root: Path) -> ReportSourceIdentity:
    revision = _repo_head(repo_root)
    tree = _commit_tree_marker(repo_root, revision)
    return ReportSourceIdentity(
        revision,
        tree,
        _index_metadata_dirty_paths(repo_root, committed_tree=tree),
    )


def installed_source_identity() -> ReportSourceIdentity:
    """Bind a copied campaign to the source revision recorded by its wheel."""

    from pyamplicol._internal.versions import active_source_revision

    revision = active_source_revision()
    if revision is None:
        raise ManualCampaignError(
            "installed pyAmpliCol has no source revision in its build info; "
            "reinstall a release wheel that retains build provenance"
        )
    tree = hashlib.sha1(
        f"pyamplicol-installed-report-tree-v1:{revision}".encode("ascii"),
        usedforsecurity=False,
    ).hexdigest()
    return ReportSourceIdentity(revision, tree, ())


@dataclass(frozen=True, slots=True)
class LightweightCurrent:
    cell_id: str
    attempt_id: str
    result_path: Path
    result: Mapping[str, object]
    complete: bool
    reusable: bool
    reason: str
    record: CurrentRecord | None = None


@dataclass(frozen=True, slots=True)
class _ProfileEnvironmentWitness:
    active_runtime: Mapping[str, object]
    host_environment: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RecordedSourcePolicy:
    """The active measurement revision and persisted read-only reuse policy."""

    source_revision: str
    continue_across_revisions: bool = False


@dataclass(frozen=True, slots=True)
class LightweightPresentationOutcome:
    """One tiny presentation-only terminal outcome for an otherwise empty cell."""

    profile: str
    cell_id: str
    source_revision: str
    campaign_invocation_id: str
    attempt_id: str | None
    status: str
    label: str
    completed_at_ns: int

    @property
    def ordering_key(self) -> tuple[int, str, str, str]:
        return (
            self.completed_at_ns,
            self.campaign_invocation_id,
            self.attempt_id or "",
            self.status,
        )

    @property
    def successful(self) -> bool:
        return self.status in _SUMMARY_SUCCESS_STATUSES

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PRESENTATION_OUTCOME_SCHEMA,
            "profile": self.profile,
            "cell_id": self.cell_id,
            "source_revision": self.source_revision,
            "campaign_invocation_id": self.campaign_invocation_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "label": self.label,
            "completed_at_ns": self.completed_at_ns,
        }


def _read_object(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _presentation_outcome_path(
    service: ReportService,
    cell_id: str,
    *,
    catalog_validated: bool = False,
) -> Path:
    if not catalog_validated:
        service.catalog.cell(cell_id)
    if _SAFE_OUTCOME_CELL_ID.fullmatch(cell_id) is None:
        raise ManualCampaignError(
            f"presentation outcome cell ID is unsafe: {cell_id!r}"
        )
    return (
        service.paths.coordination_root
        / "manual-presentation-outcomes"
        / f"{cell_id}.json"
    )


def _canonical_attempt_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    return value if str(parsed) == value else None


def _humanized_outcome_label(status: str) -> str:
    if status == "cancelled":
        return "interrupted"
    return " ".join(status.replace("_", " ").replace("-", " ").split())


def _parse_presentation_outcome(
    raw: Mapping[str, object] | None,
    *,
    expected_profile: str,
    expected_cell_id: str,
    source_revision: str,
    accept_historical_source: bool,
) -> LightweightPresentationOutcome | None:
    if raw is None or set(raw) != _PRESENTATION_OUTCOME_KEYS:
        return None
    profile = raw.get("profile")
    cell_id = raw.get("cell_id")
    revision = raw.get("source_revision")
    invocation_id = raw.get("campaign_invocation_id")
    status = raw.get("status")
    label = raw.get("label")
    completed_at_ns = raw.get("completed_at_ns")
    if (
        raw.get("schema") != PRESENTATION_OUTCOME_SCHEMA
        or not isinstance(profile, str)
        or profile != expected_profile
        or not isinstance(cell_id, str)
        or cell_id != expected_cell_id
        or not isinstance(revision, str)
        or _FULL_SOURCE_REVISION.fullmatch(revision) is None
        or (revision != source_revision and not accept_historical_source)
        or not isinstance(invocation_id, str)
        or not (1 <= len(invocation_id) <= 128)
        or not invocation_id.isascii()
        or not invocation_id.isprintable()
        or not isinstance(status, str)
        or _SAFE_OUTCOME_SLUG.fullmatch(status) is None
        or not isinstance(label, str)
        or not (1 <= len(label) <= 64)
        or label != label.strip()
        or not label.isascii()
        or not label.isprintable()
        or isinstance(completed_at_ns, bool)
        or not isinstance(completed_at_ns, int)
        or completed_at_ns <= 0
        or completed_at_ns > _MAX_PRESENTATION_COMPLETED_AT_NS
    ):
        return None
    raw_attempt_id = raw.get("attempt_id")
    attempt_id = _canonical_attempt_uuid(raw_attempt_id)
    if raw_attempt_id is not None and attempt_id is None:
        return None
    return LightweightPresentationOutcome(
        profile=profile,
        cell_id=cell_id,
        source_revision=revision,
        campaign_invocation_id=invocation_id,
        attempt_id=attempt_id,
        status=status,
        label=label,
        completed_at_ns=completed_at_ns,
    )


def lightweight_presentation_outcome(
    service: ReportService,
    cell: CellSpec,
    *,
    source_revision: str,
    accept_historical_source: bool = False,
) -> LightweightPresentationOutcome | None:
    """Read one direct compact outcome, ignoring unsafe or malformed metadata."""

    path = _presentation_outcome_path(
        service,
        cell.cell_id,
        catalog_validated=True,
    )
    if path.is_symlink() or not path.is_file():
        return None
    return _parse_presentation_outcome(
        _read_object(path),
        expected_profile=_PRESENTATION_PROFILE,
        expected_cell_id=cell.cell_id,
        source_revision=source_revision,
        accept_historical_source=accept_historical_source,
    )


def lightweight_presentation_outcomes(
    service: ReportService,
    cells: Sequence[CellSpec],
    *,
    source_revision: str,
    accept_historical_source: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, LightweightPresentationOutcome]:
    result: dict[str, LightweightPresentationOutcome] = {}
    total = len(cells)
    for completed, cell in enumerate(cells, start=1):
        outcome = lightweight_presentation_outcome(
            service,
            cell,
            source_revision=source_revision,
            accept_historical_source=accept_historical_source,
        )
        if outcome is not None:
            result[cell.cell_id] = outcome
        if progress is not None:
            progress(completed, total)
    return result


def _replace_presentation_outcome_locked(
    service: ReportService,
    outcome: LightweightPresentationOutcome,
    *,
    accept_historical_source: bool,
    only_if_replacing_failure: bool = False,
    catalog_validated: bool = False,
) -> bool:
    """Replace one outcome while the shared presentation lock is held."""

    validated = _parse_presentation_outcome(
        outcome.as_dict(),
        expected_profile=_PRESENTATION_PROFILE,
        expected_cell_id=outcome.cell_id,
        source_revision=outcome.source_revision,
        accept_historical_source=False,
    )
    if validated is None:
        raise ManualCampaignError("presentation outcome metadata is invalid")
    outcome = validated
    path = _presentation_outcome_path(
        service,
        outcome.cell_id,
        catalog_validated=catalog_validated,
    )
    raw_existing = (
        None if path.is_symlink() or not path.is_file() else _read_object(path)
    )
    existing = (
        None
        if raw_existing is None
        else _parse_presentation_outcome(
            raw_existing,
            expected_profile=_PRESENTATION_PROFILE,
            expected_cell_id=outcome.cell_id,
            source_revision=outcome.source_revision,
            accept_historical_source=accept_historical_source,
        )
    )
    if only_if_replacing_failure:
        replacement_target = existing
        if replacement_target is None and not accept_historical_source:
            historical = (
                None
                if raw_existing is None
                else _parse_presentation_outcome(
                    raw_existing,
                    expected_profile=_PRESENTATION_PROFILE,
                    expected_cell_id=outcome.cell_id,
                    source_revision=outcome.source_revision,
                    accept_historical_source=True,
                )
            )
            if historical is not None and not historical.successful:
                replacement_target = historical
        if replacement_target is None or replacement_target.successful:
            return False
    if existing is not None and existing.ordering_key > outcome.ordering_key:
        return False
    _atomic_json(path, outcome.as_dict(), sync=False)
    return True


def _publish_presentation_outcome(
    service: ReportService,
    outcome: LightweightPresentationOutcome,
    *,
    only_if_replacing_failure: bool = False,
    catalog_validated: bool = False,
) -> bool:
    """Atomically retain only the deterministically latest presentation outcome."""

    with service.store.named_lock("manual-source-marker"):
        source_policy = _recorded_measurement_source_policy_locked(
            service,
            checkout_revision=outcome.source_revision,
            allow_missing=False,
        )
        if (
            not source_policy.continue_across_revisions
            and source_policy.source_revision != outcome.source_revision
        ):
            return False
        with service.store.named_lock(
            f"manual-presentation-outcome-{outcome.cell_id}"
        ):
            return _replace_presentation_outcome_locked(
                service,
                outcome,
                accept_historical_source=(
                    source_policy.continue_across_revisions
                ),
                only_if_replacing_failure=only_if_replacing_failure,
                catalog_validated=catalog_validated,
            )


def _publish_recycled_presentation_outcomes(
    service: ReportService,
    currents: Mapping[str, LightweightCurrent],
    recycled_ids: Iterable[str],
    *,
    source_revision: str,
    campaign_invocation_id: str,
    accept_historical_source: bool = False,
) -> tuple[str, ...]:
    """Publish front-end recycle outcomes that bypass scheduler observers."""

    cells_by_id = {
        cell.cell_id: cell for cell in service.catalog.measurement_cells()
    }
    warnings: list[str] = []
    for cell_id in sorted(set(recycled_ids)):
        current = currents.get(cell_id)
        if (
            current is None
            or not current.reusable
            or current.result.get("status") != ResultStatus.OK.value
        ):
            continue
        cell = cells_by_id[cell_id]
        observed = lightweight_presentation_outcome(
            service,
            cell,
            source_revision=source_revision,
            accept_historical_source=accept_historical_source,
        )
        if observed is not None and observed.successful:
            continue
        try:
            with service.store.named_lock(
                f"campaign-cell-{cell_id}",
                timeout=0.0,
            ):
                _publish_presentation_outcome(
                    service,
                    LightweightPresentationOutcome(
                        profile=_PRESENTATION_PROFILE,
                        cell_id=cell_id,
                        source_revision=source_revision,
                        campaign_invocation_id=campaign_invocation_id,
                        attempt_id=current.attempt_id,
                        status="reused",
                        label="reused",
                        completed_at_ns=time.time_ns(),
                    ),
                    only_if_replacing_failure=True,
                    catalog_validated=True,
                )
        except LockTimeoutError:
            continue
        except Exception as error:
            warnings.append(f"{cell_id}: {type(error).__name__}: {error}")
    return tuple(warnings)


def _presentation_failure_measurement(
    outcome: LightweightPresentationOutcome,
) -> dict[str, object]:
    try:
        status = ResultStatus(outcome.status)
    except ValueError:
        status = ResultStatus.FAILED
    # A durable ``unverified`` measurement is valid only when it is loaded
    # from the exact authenticated sealed attempt below.  Never synthesize an
    # otherwise schema-invalid measured result from the tiny presentation
    # ledger when that attempt is absent, malformed, or mismatched.
    if status in {
        ResultStatus.NOT_AVAILABLE,
        ResultStatus.OK,
        ResultStatus.UNVERIFIED,
    }:
        status = ResultStatus.FAILED
    measurement = failure_measurement(status, outcome.label)
    failure = measurement["failure"]
    assert isinstance(failure, dict)
    failure["kind"] = _PRESENTATION_FAILURE_KIND_PREFIX + outcome.status
    return measurement


def _presentation_measurement(
    service: ReportService,
    cell: CellSpec,
    outcome: LightweightPresentationOutcome,
) -> dict[str, object]:
    """Return a presentation-only terminal measurement, failing closed.

    Unverified compiled/eager attempts are the sole terminal class whose
    measured timings are useful.  The tiny ledger supplies its exact attempt
    UUID; the artifact store authenticates that sealed worker result without
    scanning history.  Every malformed or mismatched record falls back to the
    ordinary compact terminal label.
    """

    if outcome.status != ResultStatus.UNVERIFIED.value or outcome.attempt_id is None:
        return _presentation_failure_measurement(outcome)
    try:
        loaded = dict(
            service.store.load_sealed_unsuccessful_worker_result(
                cell.cell_id,
                outcome.attempt_id,
            )
        )
        if (
            loaded.get("status") != ResultStatus.UNVERIFIED.value
            or not _result_matches_cell(loaded, cell)
            or _measurement_source_revision(loaded) != outcome.source_revision
        ):
            raise ValueError("unverified result identity differs from presentation")
        validate_measurement(loaded, expected_cell=cell)
    except (OSError, TypeError, ValueError, ManifestValidationError):
        return _presentation_failure_measurement(outcome)
    return loaded


def _result_matches_cell(
    result: Mapping[str, object],
    cell: CellSpec,
) -> bool:
    provenance = result.get("provenance")
    manual = (
        provenance.get("manual_campaign") if isinstance(provenance, Mapping) else None
    )
    identity = manual.get("cell_identity") if isinstance(manual, Mapping) else None
    if not isinstance(identity, Mapping):
        return False
    expected = {
        "cell_id": cell.cell_id,
        "dataset_id": cell.dataset_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "workload": cell.workload.value,
        "execution_mode": cell.measurement.execution_mode.value,
        "model": (
            None if cell.measurement.model is None else cell.measurement.model.value
        ),
        "accuracy": cell.measurement.accuracy.value,
        "backend": cell.measurement.backend,
        "variant": cell.variant,
    }
    return all(identity.get(key) == value for key, value in expected.items())


def _required_result_artifact_exists(result: Mapping[str, object]) -> bool:
    if result.get("status") != ResultStatus.OK.value:
        return True
    artifact = result.get("artifact")
    if not isinstance(artifact, Mapping):
        return False
    raw_path = artifact.get("path")
    return isinstance(raw_path, str) and Path(raw_path).exists()


def _measurement_source_revision(result: Mapping[str, object]) -> str | None:
    provenance = result.get("provenance")
    revision = (
        provenance.get("report_source_revision")
        if isinstance(provenance, Mapping)
        else None
    )
    return (
        revision
        if isinstance(revision, str)
        and _FULL_SOURCE_REVISION.fullmatch(revision) is not None
        else None
    )


def _historical_numerical_relation_compatible(
    cell: CellSpec,
    result: Mapping[str, object],
) -> bool:
    """Accept only historical results whose compact metadata proves safety."""

    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        return True
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        return False
    correctness = provenance.get("numerical_relation_correctness")
    if isinstance(correctness, Mapping):
        applied = correctness.get("applied_relation_count")
        state = correctness.get("state")
        if (
            correctness.get("abi") != _NUMERICAL_RELATION_CORRECTNESS_ABI
            or isinstance(applied, bool)
            or not isinstance(applied, int)
            or applied < 0
        ):
            return False
        if state == "no-applied-relations":
            return applied == 0
        if state == "member-scoped-v1":
            return applied > 0
        return False
    effective = provenance.get("effective_config")
    generation = effective.get("generation") if isinstance(effective, Mapping) else None
    relation = (
        generation.get("relation_discovery")
        if isinstance(generation, Mapping)
        else None
    )
    return isinstance(relation, Mapping) and relation.get("mode") == "off"


def _legacy_numerical_authority_compatible(
    cell: CellSpec,
    result: Mapping[str, object],
) -> bool:
    """Require the corrected legacy numerical path for affected currents.

    Older selected-flow measurements already used the generated library as
    their numerical authority, and the special three-quark-line contracted
    path remains a direct ``imode2`` measurement.  Retain those inexpensive
    metadata-identifiable currents while refreshing only the paths whose
    authority changed.
    """

    if (
        cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
        or result.get("status") != ResultStatus.OK.value
    ):
        return True
    from .legacy import (
        LEGACY_NUMERICAL_AUTHORITY_ABI,
        LEGACY_NUMERICAL_AUTHORITY_FIELD,
    )

    validation = result.get("validation")
    authority = (
        validation.get(LEGACY_NUMERICAL_AUTHORITY_FIELD)
        if isinstance(validation, Mapping)
        else None
    )
    expected_sources = {
        Workload.SELECTED_FLOW: {"selected-flow-generated-library"},
        Workload.ALL_FLOW: {"all-flow-selected-provider-replay"},
        Workload.CONTRACTED: {
            "contracted-generated-library",
            "direct-imode2-three-quark-line",
        },
    }
    if authority is not None:
        return (
            isinstance(authority, Mapping)
            and authority.get("abi") == LEGACY_NUMERICAL_AUTHORITY_ABI
            and authority.get("source")
            in expected_sources.get(cell.workload, set())
        )

    provenance = result.get("provenance")
    generation_source = (
        provenance.get("generation_source")
        if isinstance(provenance, Mapping)
        else None
    )
    return (
        cell.workload is Workload.SELECTED_FLOW
        and generation_source == "generated-library-create-mode-1"
    ) or (
        cell.workload is Workload.CONTRACTED
        and generation_source == "direct-imode2-three-quark-line-setup"
    )


def lightweight_current(
    store: ArtifactStore,
    cell: CellSpec,
    *,
    source_revision: str,
    accept_historical_source: bool = False,
) -> LightweightCurrent | None:
    """Read only the pointer, manifest metadata, and result required for reuse."""

    root = store._cell_root(cell.cell_id)
    pointer = _read_object(root / "current.json")
    raw_attempt_id = None if pointer is None else pointer.get("attempt_id")
    if (
        pointer is None
        or pointer.get("schema") != store.current_schema
        or pointer.get("cell_id") != cell.cell_id
        or not isinstance(raw_attempt_id, str)
    ):
        return None
    attempt_id = raw_attempt_id
    try:
        parsed_attempt_id = uuid.UUID(attempt_id)
    except ValueError:
        parsed_attempt_id = None
    canonical_manifest = f"attempts/{attempt_id}/manifest.json"
    manifest_sha256 = pointer.get("manifest_sha256")
    if (
        parsed_attempt_id is None
        or str(parsed_attempt_id) != attempt_id
        or pointer.get("manifest_path") != canonical_manifest
        or not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            root,
            {},
            False,
            False,
            "incomplete pointer metadata",
        )
    manifest_path = root / canonical_manifest
    manifest = _read_object(manifest_path)
    if (
        manifest is None
        or manifest.get("schema") != ATTEMPT_SCHEMA
        or manifest.get("cell_id") != cell.cell_id
        or manifest.get("attempt_id") != attempt_id
        or manifest.get("status") != "ok"
        or manifest.get("error") is not None
        or not isinstance(manifest.get("result_path"), str)
    ):
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            root,
            {},
            False,
            False,
            "incomplete metadata",
        )
    try:
        artifact_policy = ArtifactPolicy(str(manifest.get("artifact_policy")))
    except ValueError:
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            manifest_path,
            {},
            False,
            False,
            "unsupported artifact policy",
        )
    relative = Path(str(manifest["result_path"]))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            root,
            {},
            False,
            False,
            "unsafe result path",
        )
    artifact_records = manifest.get("artifacts")
    if not isinstance(artifact_records, list) or not any(
        isinstance(record, Mapping) and record.get("path") == relative.as_posix()
        for record in artifact_records
    ):
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            manifest_path,
            {},
            False,
            False,
            "result missing from manifest",
        )
    result_path = root / "attempts" / attempt_id / relative
    result = _read_object(result_path)
    if result is None:
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            result_path,
            {},
            False,
            False,
            "result unreadable",
        )
    try:
        result = dict(store.materialize_current_result(result))
    except ManifestValidationError:
        return LightweightCurrent(
            cell.cell_id,
            attempt_id,
            result_path,
            {},
            False,
            False,
            "unsafe result locator",
        )
    revision = _measurement_source_revision(result)
    revision_valid = revision is not None
    status = str(result.get("status", ""))
    complete = status in {
        ResultStatus.OK.value,
        ResultStatus.TIMEOUT.value,
        ResultStatus.MEMORY_LIMIT.value,
    }
    measurement_valid = False
    if complete:
        try:
            validate_measurement(result, expected_cell=cell)
        except ValueError:
            pass
        else:
            measurement_valid = True
    complete = complete and measurement_valid
    cell_match = _result_matches_cell(result, cell)
    artifact_exists = _required_result_artifact_exists(result)
    historical_relation_compatible = (
        revision == source_revision
        or not revision_valid
        or _historical_numerical_relation_compatible(cell, result)
    )
    legacy_authority_compatible = _legacy_numerical_authority_compatible(
        cell,
        result,
    )
    source_accepted = revision == source_revision or (
        accept_historical_source and revision_valid and historical_relation_compatible
    )
    reusable = (
        complete
        and source_accepted
        and cell_match
        and artifact_exists
        and legacy_authority_compatible
    )
    if status not in {
        ResultStatus.OK.value,
        ResultStatus.TIMEOUT.value,
        ResultStatus.MEMORY_LIMIT.value,
    }:
        reason = "incomplete status"
    elif not measurement_valid:
        reason = "invalid measurement schema"
    elif not revision_valid:
        reason = "invalid source revision"
    elif revision != source_revision and not accept_historical_source:
        reason = "source mismatch"
    elif revision != source_revision and not historical_relation_compatible:
        reason = "historical numerical-relation policy is not reusable"
    elif not legacy_authority_compatible:
        reason = "legacy numerical authority requires refresh"
    elif not cell_match:
        reason = "cell metadata mismatch"
    elif not artifact_exists:
        reason = "required artifact missing"
    elif status in {ResultStatus.TIMEOUT.value, ResultStatus.MEMORY_LIMIT.value}:
        reason = (
            "historical resource-capped terminal"
            if revision != source_revision
            else "resource-capped terminal"
        )
    elif revision != source_revision:
        reason = "historical source current"
    else:
        reason = "reusable"
    record = (
        CurrentRecord(
            cell_id=cell.cell_id,
            attempt_id=attempt_id,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            result_path=result_path,
            result=result,
            artifacts=(),
            artifact_policy=artifact_policy,
        )
        if complete
        else None
    )
    return LightweightCurrent(
        cell.cell_id,
        attempt_id,
        result_path,
        result,
        complete,
        reusable,
        reason,
        record,
    )


def lightweight_currents(
    service: ReportService,
    cells: Sequence[CellSpec],
    *,
    source_revision: str,
    accept_historical_source: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, LightweightCurrent]:
    result: dict[str, LightweightCurrent] = {}
    total = len(cells)
    for completed, cell in enumerate(cells, start=1):
        current = lightweight_current(
            service.store,
            cell,
            source_revision=source_revision,
            accept_historical_source=accept_historical_source,
        )
        if current is not None:
            result[cell.cell_id] = current
        if progress is not None:
            progress(completed, total)
    return result


def _selective_retry_requested(arguments: argparse.Namespace) -> bool:
    return bool(
        getattr(arguments, "rerun_failed", False)
        or getattr(arguments, "rerun_capped", False)
    )


def _fresh_attempt_requested(arguments: argparse.Namespace) -> bool:
    return bool(
        getattr(arguments, "force_refresh", False)
        or _selective_retry_requested(arguments)
    )


def _artifact_cleanup_enabled(arguments: argparse.Namespace) -> bool:
    """Compact disposable payloads unless full debug workspaces were requested."""

    return not bool(getattr(arguments, "retain_workspaces", False))


def _selective_retry_category(
    current: LightweightCurrent | None,
    outcome: LightweightPresentationOutcome | None,
) -> str | None:
    """Classify the latest reusable terminal for selective retry.

    Tiny presentation outcomes are authoritative for ordering, but a cap is
    trusted only when it names the exact attempt of a validated reusable cap
    current.  This keeps future/unauthenticated timeout-like slugs in the
    ordinary failure class rather than silently granting them policy meaning.
    """

    cap_state: PolicyMeasurementState | None = None
    current_state: PolicyMeasurementState | None = None
    if current is not None and current.reusable:
        current_state = policy_measurement_state_hint(current.result)
        if (
            current_state in _TERMINAL_CAP_STATES
            and policy_status_label(current.result) is not None
        ):
            cap_state = current_state

    if outcome is not None:
        if outcome.successful:
            return None
        if (
            cap_state is not None
            and current is not None
            and outcome.attempt_id == current.attempt_id
            and outcome.status == cap_state.value
        ):
            return "capped"
        return "failed"

    if current is None or not current.reusable:
        return None
    if current_state is PolicyMeasurementState.SUCCESS:
        return None
    return "capped" if cap_state is not None else "failed"


def _selective_retry_cells(
    cells: Sequence[CellSpec],
    currents: Mapping[str, LightweightCurrent],
    outcomes: Mapping[str, LightweightPresentationOutcome],
    *,
    rerun_failed: bool,
    rerun_capped: bool,
) -> tuple[CellSpec, ...]:
    selected_categories = {
        category
        for category, enabled in (
            ("failed", rerun_failed),
            ("capped", rerun_capped),
        )
        if enabled
    }
    return tuple(
        cell
        for cell in cells
        if _selective_retry_category(
            currents.get(cell.cell_id),
            outcomes.get(cell.cell_id),
        )
        in selected_categories
    )


def _source_cohort_counts(
    currents: Mapping[str, LightweightCurrent],
    *,
    cell_ids: set[str] | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for cell_id, current in currents.items():
        if cell_ids is not None and cell_id not in cell_ids:
            continue
        if not current.reusable:
            continue
        revision = _measurement_source_revision(current.result)
        if revision is not None:
            counts[revision] += 1
    return dict(sorted(counts.items()))


def _format_source_cohorts(cohorts: Mapping[str, int]) -> str:
    return (
        ", ".join(f"{revision}={count}" for revision, count in cohorts.items())
        or "none"
    )


def _lightweight_current_resolver(
    service: ReportService,
    *,
    source_revision: str,
    initial: Mapping[str, LightweightCurrent] | None = None,
) -> Callable[
    [CellSpec],
    tuple[CurrentRecord, PolicyMeasurementState] | None,
]:
    """Resolve planner currents from cheap metadata without artifact hashing."""

    cache: dict[str, LightweightCurrent | None] = dict(initial or {})

    def resolve(
        cell: CellSpec,
    ) -> tuple[CurrentRecord, PolicyMeasurementState] | None:
        if cell.cell_id not in cache:
            cache[cell.cell_id] = lightweight_current(
                service.store,
                cell,
                source_revision=source_revision,
            )
        current = cache[cell.cell_id]
        if current is None or not current.reusable or current.record is None:
            return None
        state = policy_measurement_state_hint(current.result)
        return None if state is None else (current.record, state)

    return resolve


def _installed_source_revision() -> str | None:
    try:
        from pyamplicol._internal.versions import (
            active_source_revision,
            package_version,
        )
    except (ImportError, RuntimeError, ValueError) as error:
        raise ManualCampaignError(
            f"measurement runtime provenance cannot be loaded: {error}; rerun "
            "`just dev-install`"
        ) from error
    try:
        package_version()
        return active_source_revision()
    except (RuntimeError, ValueError) as error:
        raise ManualCampaignError(
            f"measurement runtime provenance check failed: {error}"
        ) from error


def require_measurement_ready(source: ReportSourceIdentity) -> None:
    if not source.eligible:
        examples = ", ".join(source.dirty_paths[:5])
        remainder = max(0, len(source.dirty_paths) - 5)
        suffix = "" if remainder == 0 else f", … {remainder} more"
        raise ManualCampaignError(
            "measurement source is not a clean committed checkout "
            f"({examples}{suffix}); commit the campaign implementation and "
            "rebuild before running measurements"
        )
    installed = _installed_source_revision()
    if installed is None:
        raise ManualCampaignError(
            "measurement runtime contains no clean source revision for this "
            f"checkout ({source.revision}); rebuild it with `just dev-install` "
            "after committing evaluator-source changes. Copied profiling "
            "campaigns and generated report outputs do not invalidate the build."
        )
    if installed != source.revision:
        raise ManualCampaignError(
            "measurement runtime is not a clean build of this checkout: "
            f"checkout={source.revision}, installed={installed}. Commit the "
            "implementation, ensure the worktree is clean, then run "
            "`just dev-install`; dirty-build evidence is never reusable."
        )


def _source_marker_path(service: ReportService) -> Path:
    return service.paths.coordination_root / "manual-source.json"


def update_source_marker(
    service: ReportService,
    source: ReportSourceIdentity,
    *,
    continue_across_revisions: bool = False,
) -> bool:
    """Record the active source and lightweight publication policy once."""

    path = _source_marker_path(service)
    with service.store.named_lock("manual-source-marker"):
        previous = _read_object(path)
        changed = previous is not None and (
            previous.get("source_revision") != source.revision
            or previous.get("continue_across_revisions", False)
            != continue_across_revisions
        )
        _atomic_json(
            path,
            {
                "schema": SOURCE_MARKER_SCHEMA,
                "source_revision": source.revision,
                "continue_across_revisions": continue_across_revisions,
                "recorded_at_utc": _utc_now(),
            },
        )
    return changed


def recorded_measurement_source_policy(
    service: ReportService,
    *,
    checkout_revision: str,
) -> RecordedSourcePolicy:
    """Read the recorded measurement epoch and its lightweight reuse policy.

    Renderer-only commits must not make already published measurements vanish.
    Real campaign runs update this marker only after the checkout and installed
    runtime have passed the strict measurement-readiness checks.  The optional
    continuation policy lets read-only views merge per-cell currents from more
    than one valid recorded source revision without weakening active-revision
    dependency planning.
    """

    with service.store.named_lock("manual-source-marker"):
        return _recorded_measurement_source_policy_locked(
            service,
            checkout_revision=checkout_revision,
            allow_missing=True,
        )


def _recorded_measurement_source_policy_locked(
    service: ReportService,
    *,
    checkout_revision: str,
    allow_missing: bool,
) -> RecordedSourcePolicy:
    """Decode the source marker while its named lock is already held."""

    path = _source_marker_path(service)
    exists = path.exists()
    marker = _read_object(path)
    if marker is None:
        if exists or not allow_missing:
            raise ManualCampaignError(
                f"campaign source marker is unreadable: {path}; refusing to "
                "guess which measurement cohort to publish"
            )
        return RecordedSourcePolicy(checkout_revision)
    revision = marker.get("source_revision")
    continue_across_revisions = marker.get("continue_across_revisions", False)
    if (
        marker.get("schema") != SOURCE_MARKER_SCHEMA
        or not isinstance(revision, str)
        or _FULL_SOURCE_REVISION.fullmatch(revision) is None
        or not isinstance(continue_across_revisions, bool)
    ):
        raise ManualCampaignError(
            f"campaign source marker is malformed: {path}; refusing to guess "
            "which measurement cohort to publish"
        )
    return RecordedSourcePolicy(revision, continue_across_revisions)


def recorded_measurement_source_revision(
    service: ReportService,
    *,
    checkout_revision: str,
) -> str:
    """Return the active recorded measurement revision for compatibility."""

    return recorded_measurement_source_policy(
        service,
        checkout_revision=checkout_revision,
    ).source_revision


@dataclass(slots=True)
class _IncrementalTailState:
    identity: tuple[int, int] | None = None
    offset: int = 0
    pending: bytes = b""
    last_read_bytes: int = 0


@dataclass(frozen=True, slots=True)
class PhaseTimelineRow:
    """One normalized, presentation-only phase record from persisted evidence."""

    phase: str
    wall_seconds: float | None = None
    cpu_seconds: float | None = None
    peak_memory_bytes: int | None = None
    status: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class WorkerView:
    cell_id: str
    dependency: bool = False
    peer_instance: str | None = None
    generation_engine: str | None = None
    status: str = "queued"
    step: str = "waiting"
    phase: str = "preparation"
    attempt_id: str | None = None
    pid: int | None = None
    member_pids: tuple[int, ...] = ()
    wall_seconds: float = 0.0
    cpu_seconds: float | None = None
    current_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    current_physical_footprint_bytes: int | None = None
    peak_physical_footprint_bytes: int | None = None
    current_guard_bytes: int = 0
    peak_guard_bytes: int = 0
    child_count: int = 0
    progress_completed: int | None = None
    progress_total: int | None = None
    progress_message: str = ""
    progress_task_id: str | None = None
    progress_details: dict[str, object] = field(default_factory=dict)
    log_path: str | None = None
    progress_path: str | None = None
    reproduce_prepare: str | None = None
    reproduce_generate: str | None = None
    reproduce_profile: str | None = None
    published_wall_seconds_per_point: float | None = None
    published_evaluator_total_seconds_per_point: float | None = None
    published_recurrence_core_seconds_per_point: float | None = None
    phase_timeline: tuple[PhaseTimelineRow, ...] = ()
    provenance_summary: str | None = None
    reuse_explanation: str | None = None
    blocked_prerequisite_ids: tuple[str, ...] = ()
    recycled: bool = False
    events: list[str] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    _progress_tail_state: _IncrementalTailState = field(
        default_factory=_IncrementalTailState,
        repr=False,
    )
    _log_tail_state: _IncrementalTailState = field(
        default_factory=_IncrementalTailState,
        repr=False,
    )

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("_progress_tail_state", None)
        payload.pop("_log_tail_state", None)
        return payload


@dataclass(slots=True)
class DashboardState:
    instance_id: str
    selected_ids: tuple[str, ...]
    recycled_ids: set[str]
    static_na_ids: set[str]
    source_revision: str = ""
    generation_time_limit_seconds: float = DEFAULT_GENERATION_LIMIT_SECONDS
    memory_limit_bytes: int = DEFAULT_RAM_BYTES
    worker_wall_limit_seconds: float | None = DEFAULT_WORKER_WALL_LIMIT_SECONDS
    reproduction_settings: ReproductionSettings = field(
        default_factory=ReproductionSettings
    )
    dependency_ids: set[str] = field(default_factory=set)
    workers: dict[str, WorkerView] = field(default_factory=dict)
    terminal_outcomes: dict[str, _DashboardTerminalOutcome] = field(
        default_factory=dict
    )
    selected_index: int = 0
    detail_scroll: int = 0
    show_completed: bool = False
    show_errors: bool = True
    show_help: bool = False
    command_stage: str | None = None
    command_scroll: int = 0
    command_notice: str | None = None
    pending_clipboard: tuple[str, str] | None = None
    pending_print: tuple[str, str] | None = None
    started_at: float = field(default_factory=time.time)
    interrupted: bool = False
    counter_snapshot: dict[str, int] | None = None
    lease_updated_at: float | None = None
    live_snapshot: bool = False
    ready_count: int = 0
    waiting_dependency_count: int = 0
    waiting_coordination_lock_count: int = 0
    cleanup_warnings: list[tuple[str, str]] = field(default_factory=list)
    invocation_evidence_ids: set[str] = field(default_factory=set)

    def _clear_cell_disposition(self, cell_id: str) -> None:
        self.terminal_outcomes.pop(cell_id, None)
        self.recycled_ids.discard(cell_id)

    def _reduce_started(self, cell_id: str, *, dependency: bool) -> WorkerView:
        """Apply one started event and return a wholly fresh worker view."""

        self._clear_cell_disposition(cell_id)
        worker = WorkerView(
            cell_id,
            dependency=dependency,
            status="preparing",
            step="dependency preparation",
            phase="preparation",
        )
        self.workers[cell_id] = worker
        return worker

    def _reduce_finished(
        self,
        cell_id: str,
        status: str,
        *,
        recycled: bool = False,
    ) -> None:
        """Reduce a finished event into one normalized terminal category."""

        self._clear_cell_disposition(cell_id)
        if recycled or status in _RECYCLED_SUCCESS_STATUSES:
            self.recycled_ids.add(cell_id)
        if status in _SUMMARY_SUCCESS_STATUSES:
            outcome = _DashboardTerminalOutcome.SUCCESS
        elif status in _TERMINAL_CAP_STATUSES:
            outcome = _DashboardTerminalOutcome.CAPPED
        elif status == ResultStatus.UNVERIFIED.value:
            outcome = _DashboardTerminalOutcome.UNVERIFIED
        elif status != "cancelled":
            outcome = _DashboardTerminalOutcome.FAILED
        else:
            return
        self.terminal_outcomes[cell_id] = outcome

    @property
    def completed_ids(self) -> set[str]:
        return {
            cell_id
            for cell_id, outcome in self.terminal_outcomes.items()
            if cell_id not in self.recycled_ids
            and outcome is _DashboardTerminalOutcome.SUCCESS
        }

    @property
    def capped_ids(self) -> set[str]:
        return {
            cell_id
            for cell_id, outcome in self.terminal_outcomes.items()
            if outcome is _DashboardTerminalOutcome.CAPPED
        }

    @property
    def failed_ids(self) -> set[str]:
        return {
            cell_id
            for cell_id, outcome in self.terminal_outcomes.items()
            if outcome is _DashboardTerminalOutcome.FAILED
        }

    @property
    def unverified_ids(self) -> set[str]:
        return {
            cell_id
            for cell_id, outcome in self.terminal_outcomes.items()
            if outcome is _DashboardTerminalOutcome.UNVERIFIED
        }

    @property
    def active(self) -> tuple[WorkerView, ...]:
        """Return active work owned by this invocation only.

        Peer leases are rendered to make cross-process activity observable, but
        they must not inflate the primary ``Active`` counter for this runner.
        """

        return tuple(
            worker
            for worker in self.workers.values()
            if worker.peer_instance is None and worker.status in _ACTIVE_WORKER_STATUSES
        )

    @property
    def peer_active(self) -> tuple[WorkerView, ...]:
        """Return visible active work owned by concurrent peer invocations."""

        return tuple(
            worker
            for worker in self.workers.values()
            if worker.peer_instance is not None
            and worker.status in _ACTIVE_WORKER_STATUSES
        )

    def visible_workers(self) -> tuple[WorkerView, ...]:
        """Return the single ordered worker view used by table and details."""

        def rank(worker: WorkerView) -> tuple[int, str, str]:
            status_rank = {
                "running": 0,
                "preparing": 1,
                "queued": 2,
            }.get(
                worker.status,
                4 if worker.status in _COMPLETED_WORKER_STATUSES else 3,
            )
            return status_rank, worker.cell_id, worker.peer_instance or ""

        return tuple(
            sorted(
                (
                    worker
                    for worker in self.workers.values()
                    if (
                        self.show_completed
                        or worker.status not in _COMPLETED_WORKER_STATUSES
                    )
                    and (
                        self.show_errors
                        or worker.status in _ACTIVE_WORKER_STATUSES
                        or worker.status in _COMPLETED_WORKER_STATUSES
                    )
                ),
                key=rank,
            )
        )

    def counters(self) -> dict[str, int]:
        """Return one snapshot with non-overlapping selection dispositions.

        ``Selected`` progress is one exclusive partition.  Active work is not
        a terminal disposition, so it remains in ``remaining`` until it
        finishes.  Slot utilisation deliberately spans direct and dependency
        work, and is exposed separately as ``active`` with an explicit split.
        """

        if self.counter_snapshot is not None:
            return dict(self.counter_snapshot)
        selected = set(self.selected_ids)
        recycled_ids = self.recycled_ids & selected
        selected_outcomes = {
            cell_id: outcome
            for cell_id, outcome in self.terminal_outcomes.items()
            if cell_id in selected and cell_id not in recycled_ids
        }
        selected_terminal_ids = set(selected_outcomes)
        static_na_ids = (
            (self.static_na_ids & selected)
            - recycled_ids
            - selected_terminal_ids
        )
        remaining_ids = selected - recycled_ids - selected_terminal_ids - static_na_ids
        dependency_only_ids = self.dependency_ids - selected
        active_ids = {worker.cell_id for worker in self.active}
        selected_active_ids = active_ids & selected
        dependency_active_ids = active_ids & dependency_only_ids
        dependency_outcomes = {
            cell_id: outcome
            for cell_id, outcome in self.terminal_outcomes.items()
            if cell_id in dependency_only_ids
        }
        dependency_completed = sum(
            outcome is _DashboardTerminalOutcome.SUCCESS
            for outcome in dependency_outcomes.values()
        )
        dependency_issues = sum(
            outcome is not _DashboardTerminalOutcome.SUCCESS
            for outcome in dependency_outcomes.values()
        )
        return {
            "selected": len(self.selected_ids),
            "recycled": len(recycled_ids),
            "active": len(active_ids),
            "selected_active": len(selected_active_ids),
            "dependency_active": len(dependency_active_ids),
            "completed": sum(
                outcome is _DashboardTerminalOutcome.SUCCESS
                for outcome in selected_outcomes.values()
            ),
            "remaining": len(remaining_ids),
            "static_na": len(static_na_ids),
            "capped": sum(
                outcome is _DashboardTerminalOutcome.CAPPED
                for outcome in selected_outcomes.values()
            ),
            "failed": sum(
                outcome is _DashboardTerminalOutcome.FAILED
                for outcome in selected_outcomes.values()
            ),
            "unverified": sum(
                outcome is _DashboardTerminalOutcome.UNVERIFIED
                for outcome in selected_outcomes.values()
            ),
            "dependency_only": len(dependency_only_ids),
            "dependency_completed": dependency_completed,
            "dependency_issues": dependency_issues,
        }

    def selected_worker(self) -> WorkerView | None:
        rows = self.visible_workers()
        if not rows:
            self.selected_index = 0
            return None
        self.selected_index %= len(rows)
        return rows[self.selected_index]


class LeaseManager:
    """Publish compact atomic informational state for concurrent dashboards."""

    def __init__(
        self,
        service: ReportService,
        state: DashboardState,
    ) -> None:
        self.service = service
        self.state = state
        self._guard = threading.Lock()
        self._closed = False
        self.path = (
            service.paths.coordination_root
            / "manual-leases"
            / f"{state.instance_id}.json"
        )

    def _write(self) -> None:
        if self._closed:
            return
        _atomic_json(
            self.path,
            {
                "schema": MANUAL_STATE_SCHEMA,
                "instance_id": self.state.instance_id,
                "controller_pid": os.getpid(),
                "source_revision": self.state.source_revision,
                "started_at": self.state.started_at,
                "updated_at": time.time(),
                "counters": self.state.counters(),
                "limits": {
                    "generation_time_limit_seconds": (
                        self.state.generation_time_limit_seconds
                    ),
                    "memory_limit_bytes": self.state.memory_limit_bytes,
                    "worker_wall_limit_seconds": (self.state.worker_wall_limit_seconds),
                },
                "scheduler": {
                    "ready": self.state.ready_count,
                    "waiting_dependency": self.state.waiting_dependency_count,
                    "waiting_coordination_lock": (
                        self.state.waiting_coordination_lock_count
                    ),
                },
                "workers": {
                    key: worker.as_dict()
                    for key, worker in sorted(self.state.workers.items())
                    if worker.peer_instance is None
                },
            },
        )

    def publish(self) -> None:
        with self._guard:
            self._write()

    def dashboard_snapshot(self) -> DashboardState:
        with self._guard:
            _merge_peer_workers(self.service, self.state)
            return copy.deepcopy(self.state)

    def observe(self, payload: Mapping[str, object]) -> None:
        event = str(payload.get("event", "update"))
        with self._guard:
            if self._closed:
                return
            if event == "scheduler-state":
                self.state.ready_count = max(
                    0,
                    _optional_int(payload.get("ready")) or 0,
                )
                self.state.waiting_dependency_count = max(
                    0,
                    _optional_int(payload.get("waiting_dependency")) or 0,
                )
                self.state.waiting_coordination_lock_count = max(
                    0,
                    _optional_int(payload.get("waiting_coordination_lock")) or 0,
                )
                self._write()
                return
            cell_id = str(payload.get("cell_id", ""))
            if not cell_id:
                return
            if event == "preflight-finished":
                attempt_id = str(payload.get("attempt_id") or "")
                worker = self.state.workers.get(cell_id)
                if worker is not None and worker.attempt_id == attempt_id:
                    del self.state.workers[cell_id]
                self._write()
                return
            worker = self.state.workers.setdefault(cell_id, WorkerView(cell_id))
            worker.updated_at = time.time()
            cleanup_detail: str | None = None
            if event == "started":
                self.state.invocation_evidence_ids.add(cell_id)
                worker = self.state._reduce_started(
                    cell_id,
                    dependency=bool(payload.get("dependency", False)),
                )
                try:
                    cell = self.service.catalog.cell(cell_id)
                except KeyError:
                    cell = None
                if cell is not None:
                    worker.generation_engine = cell.measurement.execution_mode.value
                    settings = self.state.reproduction_settings
                    recipe = reproduction_recipe(
                        cell,
                        repo_root=self.service.paths.repo_root,
                        artifact_root=self.service.paths.artifact_root,
                        cores=settings.cores,
                        target_runtime=settings.target_runtime,
                        batch_size=settings.batch_size,
                        warmups=settings.warmups,
                        minimum_samples=settings.minimum_samples,
                    )
                    worker.reproduce_prepare = (
                        None if recipe.prepare is None else _shell_join(recipe.prepare)
                    )
                    worker.reproduce_generate = (
                        None
                        if recipe.generate is None
                        else _shell_join(recipe.generate)
                    )
                    worker.reproduce_profile = (
                        None if recipe.profile is None else _shell_join(recipe.profile)
                    )
            elif event == "worker":
                if worker.status not in _ACTIVE_WORKER_STATUSES:
                    self._write()
                    return
                self.state.invocation_evidence_ids.add(cell_id)
                worker.status = "running"
                worker.step = "worker launched"
                worker.attempt_id = str(payload.get("attempt_id") or "") or None
                worker.log_path = str(payload.get("log_path") or "") or None
                worker.progress_path = str(payload.get("progress_path") or "") or None
            elif event == "resource":
                resource_attempt_id = str(payload.get("attempt_id") or "") or None
                if worker.status not in _ACTIVE_WORKER_STATUSES or (
                    resource_attempt_id is not None
                    and resource_attempt_id != worker.attempt_id
                ):
                    self._write()
                    return
                self.state.invocation_evidence_ids.add(cell_id)
                worker.status = "running"
                worker.pid = _optional_int(payload.get("pid"))
                worker.member_pids = tuple(
                    int(value) for value in payload.get("member_pids", ())
                )
                worker.phase = str(payload.get("phase") or "unknown")
                worker.step = worker.phase.replace("_", " ")
                worker.wall_seconds = _finite_float(payload.get("wall_seconds"))
                worker.cpu_seconds = _optional_finite_float(payload.get("cpu_seconds"))
                worker.current_rss_bytes = (
                    _optional_int(payload.get("current_rss_bytes")) or 0
                )
                worker.peak_rss_bytes = (
                    _optional_int(payload.get("peak_rss_bytes")) or 0
                )
                worker.current_physical_footprint_bytes = _optional_int(
                    payload.get("current_physical_footprint_bytes")
                )
                worker.peak_physical_footprint_bytes = _optional_int(
                    payload.get("peak_physical_footprint_bytes")
                )
                worker.current_guard_bytes = (
                    _optional_int(payload.get("current_guard_bytes")) or 0
                )
                worker.peak_guard_bytes = (
                    _optional_int(payload.get("peak_guard_bytes")) or 0
                )
                worker.child_count = _optional_int(payload.get("child_count")) or 0
                _tail_progress(worker)
                _tail_log(worker)
            elif event == "finished":
                reported_status = str(payload.get("status") or "unknown")
                reported_detail = str(payload.get("detail") or "")
                raw_terminal_detail = payload.get("terminal_detail")
                reported_terminal_detail = (
                    raw_terminal_detail
                    if isinstance(raw_terminal_detail, str)
                    else None
                )
                outcome_attempt_id = worker.attempt_id
                reported_attempt_id = _canonical_attempt_uuid(outcome_attempt_id)
                if reported_attempt_id is None:
                    reported_attempt_id = _canonical_attempt_uuid(reported_detail)
                reported_success = reported_status in _SUMMARY_SUCCESS_STATUSES
                if not (
                    reported_status == "cancelled"
                    and reported_detail in {"not started", "lock wait cancelled"}
                ):
                    self.state.invocation_evidence_ids.add(cell_id)
                worker.status = (
                    "recycled"
                    if reported_status in _RECYCLED_SUCCESS_STATUSES
                    else reported_status
                )
                worker.recycled = reported_status in _RECYCLED_SUCCESS_STATUSES
                worker.step = reported_detail or worker.status
                raw_prerequisites = payload.get("prerequisite_cell_ids")
                if isinstance(raw_prerequisites, Sequence) and not isinstance(
                    raw_prerequisites,
                    (str, bytes),
                ):
                    worker.blocked_prerequisite_ids = tuple(
                        sorted(
                            {
                                str(value)
                                for value in raw_prerequisites
                                if isinstance(value, str) and value
                            }
                        )
                    )
                try:
                    cell = self.service.catalog.cell(cell_id)
                except KeyError:
                    cell = None
                current: LightweightCurrent | None = None
                terminal_result: Mapping[str, object] | None = None
                if cell is not None:
                    worker.generation_engine = cell.measurement.execution_mode.value
                    current = lightweight_current(
                        self.service.store,
                        cell,
                        source_revision=self.state.source_revision,
                    )
                    current_is_observed_attempt = (
                        current is not None
                        and current.complete
                        and (
                            reported_success
                            or reported_attempt_id == current.attempt_id
                        )
                    )
                    if current_is_observed_attempt:
                        assert current is not None
                        terminal_result = current.result
                        worker.attempt_id = current.attempt_id
                        _apply_persisted_worker_result(
                            worker,
                            current.result,
                            recycled=worker.recycled,
                            reuse_explanation=(
                                current.reason if worker.recycled else None
                            ),
                        )
                        worker.published_wall_seconds_per_point = _number(
                            current.result,
                            "wall_seconds_per_point",
                        )
                        worker.published_evaluator_total_seconds_per_point = (
                            _evaluator_total_number(current.result)
                        )
                        worker.published_recurrence_core_seconds_per_point = (
                            _recurrence_core_number(current.result)
                        )
                        provenance = current.result.get("provenance")
                        manual = (
                            provenance.get("manual_campaign")
                            if isinstance(provenance, Mapping)
                            else None
                        )
                        recipe = (
                            manual.get("public_cli_reproduction")
                            if isinstance(manual, Mapping)
                            else None
                        )
                        if isinstance(recipe, Mapping):
                            worker.reproduce_prepare = _stored_reproduction_command(
                                recipe,
                                "prepare",
                            )
                            worker.reproduce_generate = _stored_reproduction_command(
                                recipe,
                                "generate",
                            )
                            worker.reproduce_profile = _stored_reproduction_command(
                                recipe,
                                "profile",
                            )
                terminal_step = _concise_terminal_step(reported_terminal_detail)
                if terminal_step is None and terminal_result is not None:
                    terminal_step = _terminal_result_step(terminal_result)
                if terminal_step is not None:
                    worker.step = terminal_step
                if worker.recycled and worker.reuse_explanation is None:
                    worker.reuse_explanation = worker.step
                _tail_progress(worker)
                _tail_log(worker)
                self.state._reduce_finished(
                    cell_id,
                    worker.status,
                    recycled=worker.recycled,
                )
                successful_current = (
                    current
                    if reported_success
                    and current is not None
                    and current.reusable
                    and current.result.get("status") == ResultStatus.OK.value
                    else None
                )
                if (
                    cell is not None
                    and reported_status != ResultStatus.NOT_AVAILABLE.value
                    and (not reported_success or successful_current is not None)
                    and not (
                        reported_status == "cancelled"
                        and reported_detail in {"not started", "lock wait cancelled"}
                    )
                ):
                    normalized_status = (
                        reported_status
                        if _SAFE_OUTCOME_SLUG.fullmatch(reported_status) is not None
                        else _summary_status(reported_status)
                    )
                    attempt_id = _canonical_attempt_uuid(
                        None
                        if successful_current is None
                        else successful_current.attempt_id
                    )
                    if attempt_id is None:
                        attempt_id = _canonical_attempt_uuid(outcome_attempt_id)
                    if attempt_id is None:
                        attempt_id = _canonical_attempt_uuid(reported_detail)
                    label = _humanized_outcome_label(normalized_status)
                    if (
                        current is not None
                        and attempt_id is not None
                        and current.attempt_id == attempt_id
                    ):
                        label = policy_status_label(current.result) or label
                    completed_at_ns = (
                        _optional_int(payload.get("completed_at_ns")) or time.time_ns()
                    )
                    try:
                        _publish_presentation_outcome(
                            self.service,
                            LightweightPresentationOutcome(
                                profile=_PRESENTATION_PROFILE,
                                cell_id=cell_id,
                                source_revision=self.state.source_revision,
                                campaign_invocation_id=self.state.instance_id,
                                attempt_id=attempt_id,
                                status=normalized_status,
                                label=label,
                                completed_at_ns=completed_at_ns,
                            ),
                            only_if_replacing_failure=reported_success,
                            catalog_validated=True,
                        )
                    except Exception as error:
                        worker.events.append(
                            "presentation outcome warning: "
                            f"{type(error).__name__}: {error}"
                        )
            elif event == "cleanup-warning":
                cleanup_detail = str(payload.get("detail") or "unknown cleanup error")
                warning = (cell_id, cleanup_detail)
                if warning not in self.state.cleanup_warnings:
                    self.state.cleanup_warnings.append(warning)
            observed_event = (
                f"cleanup warning: {cleanup_detail}"
                if cleanup_detail is not None
                else (
                    f"resource: {worker.phase}"
                    if event == "resource"
                    else f"{event}: {worker.step}"
                )
            )
            if not (
                event == "resource"
                and worker.events
                and worker.events[-1] == observed_event
            ):
                worker.events.append(observed_event)
            del worker.events[:-8]
            self._write()

    def close(self) -> None:
        with self._guard:
            self._closed = True
            self.path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _FailFastTerminalFailure:
    observed_at_utc: str
    cell_id: str
    status: str
    detail: str
    terminal_detail: str | None
    completed_at_ns: int | None


@dataclass(frozen=True, slots=True)
class _FailFastReport:
    text: str
    cell_id: str
    attempt_root: str | None
    artifact_path: str | None
    worker_log_path: str | None
    worker_progress_path: str | None


class _FailFastObserver:
    """Turn the first terminal failure event into the campaign stop signal."""

    def __init__(
        self,
        delegate: Callable[[Mapping[str, object]], None],
        cancellation: threading.Event,
    ) -> None:
        self._delegate = delegate
        self._cancellation = cancellation
        self._guard = threading.Lock()
        self._failure: _FailFastTerminalFailure | None = None

    @property
    def failure(self) -> _FailFastTerminalFailure | None:
        with self._guard:
            return self._failure

    def observe(self, payload: Mapping[str, object]) -> None:
        if payload.get("event") == "finished":
            status = str(payload.get("status") or "unknown")
            if status not in _FAIL_FAST_SUCCESS_STATUSES and status != "cancelled":
                with self._guard:
                    if self._failure is None and not self._cancellation.is_set():
                        raw_terminal_detail = payload.get("terminal_detail")
                        raw_completed_at_ns = payload.get("completed_at_ns")
                        self._failure = _FailFastTerminalFailure(
                            observed_at_utc=_utc_now(),
                            cell_id=str(payload.get("cell_id") or ""),
                            status=status,
                            detail=str(payload.get("detail") or ""),
                            terminal_detail=(
                                raw_terminal_detail
                                if isinstance(raw_terminal_detail, str)
                                else None
                            ),
                            completed_at_ns=(
                                raw_completed_at_ns
                                if isinstance(raw_completed_at_ns, int)
                                and not isinstance(raw_completed_at_ns, bool)
                                else None
                            ),
                        )
                        # The scheduler consults this same event before filling
                        # another slot, and live supervisors consult it while
                        # sampling their process trees.
                        self._cancellation.set()
        self._delegate(payload)


def _attempt_failure_evidence(
    service: ReportService,
    *,
    cell_id: str,
    attempt_id: str | None,
) -> tuple[Path | None, Mapping[str, object] | None]:
    if attempt_id is None:
        return None, None
    current: CurrentRecord | None = None
    with suppress(Exception):
        current = service.store.load_current(cell_id, missing_ok=True)
    if current is not None and current.attempt_id == attempt_id:
        return current.manifest_path.parent, current.result

    live_root: Path | None = None
    archived_root: Path | None = None
    with suppress(Exception):
        live_root = service.store._existing_live_attempt_root(cell_id, attempt_id)
    with suppress(Exception):
        archived_root = service.store._existing_archived_attempt_root(
            cell_id,
            attempt_id,
        )
    roots = tuple(root for root in (live_root, archived_root) if root is not None)
    root = roots[0] if len(roots) == 1 else None
    result: Mapping[str, object] | None = None
    with suppress(Exception):
        result = service.store.load_sealed_unsuccessful_worker_result(
            cell_id,
            attempt_id,
        )
    return root, result


def _fail_fast_report_value(value: object) -> str:
    if value is None:
        return "<unavailable>"
    text = str(value)
    return text if text else "<unavailable>"


def _fail_fast_report_line(label: str, value: object) -> str:
    rendered = (
        _fail_fast_report_value(value)
        .replace("\r\n", "\n")
        .replace(
            "\r",
            "\n",
        )
    )
    return f"{label}: " + rendered.replace("\n", "\n  ")


def _build_fail_fast_report(
    service: ReportService,
    state: DashboardState,
    failure: _FailFastTerminalFailure,
    *,
    invocation_command: str,
) -> _FailFastReport:
    try:
        cell = service.catalog.cell(failure.cell_id)
    except KeyError as error:
        raise ManualCampaignError(
            f"fail-fast terminal references unknown cell {failure.cell_id!r}"
        ) from error
    worker = state.workers.get(failure.cell_id)
    attempt_id = _canonical_attempt_uuid(
        None if worker is None else worker.attempt_id
    )
    if attempt_id is None:
        attempt_id = _canonical_attempt_uuid(failure.detail)
    attempt_root, result = _attempt_failure_evidence(
        service,
        cell_id=cell.cell_id,
        attempt_id=attempt_id,
    )
    failure_record = result.get("failure") if isinstance(result, Mapping) else None
    failure_class = (
        failure_record.get("kind") if isinstance(failure_record, Mapping) else None
    )
    failure_message = (
        failure_record.get("message")
        if isinstance(failure_record, Mapping)
        else None
    )
    if not isinstance(failure_class, str) or not failure_class:
        failure_class = None
    if not isinstance(failure_message, str) or not failure_message:
        failure_message = failure.terminal_detail or failure.detail or None
    artifact = result.get("artifact") if isinstance(result, Mapping) else None
    raw_artifact_path = artifact.get("path") if isinstance(artifact, Mapping) else None
    artifact_path = (
        raw_artifact_path
        if isinstance(raw_artifact_path, str) and raw_artifact_path
        else None
    )
    if (
        artifact_path is None
        and attempt_root is not None
        and (attempt_root / "artifact").is_dir()
    ):
        artifact_path = str(attempt_root / "artifact")
    worker_log_path = None if worker is None else worker.log_path
    worker_progress_path = None if worker is None else worker.progress_path
    if attempt_root is not None:
        if worker_log_path is None and (attempt_root / "worker.log").is_file():
            worker_log_path = str(attempt_root / "worker.log")
        if (
            worker_progress_path is None
            and (attempt_root / "worker-progress.jsonl").is_file()
        ):
            worker_progress_path = str(attempt_root / "worker-progress.jsonl")
    process_family_id = next(
        (
            family.identifier
            for family in PROCESS_FAMILIES
            if family.key == cell.process_key
        ),
        None,
    )
    rows = (
        ("timestamp_utc", failure.observed_at_utc),
        ("campaign_invocation_id", state.instance_id),
        ("campaign_invocation", invocation_command),
        ("cell_id", cell.cell_id),
        ("process_family_id", process_family_id),
        ("process_key", cell.process_key),
        ("process", cell.process),
        ("n_final", cell.n_final),
        ("mode", cell.measurement.execution_mode.value),
        ("workload", cell.workload.value),
        ("accuracy", cell.measurement.accuracy.value),
        ("terminal_status", failure.status),
        ("outcome_detail", failure.detail),
        ("terminal_detail", failure.terminal_detail),
        ("failure_class", failure_class),
        ("failure_message", failure_message),
        ("attempt_uuid", attempt_id),
        ("attempt_root", attempt_root),
        ("artifact_path", artifact_path),
        ("worker_log_path", worker_log_path),
        ("worker_progress_path", worker_progress_path),
        (
            "reproduce_prepare",
            None if worker is None else worker.reproduce_prepare,
        ),
        (
            "reproduce_generate",
            None if worker is None else worker.reproduce_generate,
        ),
        (
            "reproduce_profile",
            None if worker is None else worker.reproduce_profile,
        ),
    )
    text = "\n".join(
        (
            "pyAmpliCol campaign fail-fast failure",
            *(_fail_fast_report_line(label, value) for label, value in rows),
        )
    )
    return _FailFastReport(
        text=text,
        cell_id=cell.cell_id,
        attempt_root=None if attempt_root is None else str(attempt_root),
        artifact_path=artifact_path,
        worker_log_path=worker_log_path,
        worker_progress_path=worker_progress_path,
    )


def _publish_completion_summary(
    service: ReportService,
    categories: Mapping[str, Iterable[str]],
    *,
    state: DashboardState,
    fail_fast_failure: _FailFastTerminalFailure | None,
    invocation_command: str,
    palette: Palette,
) -> Path:
    report = (
        None
        if fail_fast_failure is None
        else _build_fail_fast_report(
            service,
            state,
            fail_fast_failure,
            invocation_command=invocation_command,
        )
    )
    summary_path, summary_counts = _publish_campaign_summary_ids(
        service,
        categories,
        fail_fast_failure_log=None if report is None else report.text,
    )
    _print_campaign_summary_ids(summary_path, summary_counts, palette)
    if report is not None:
        report_path = summary_path / FAIL_FAST_FAILURE_LOG
        print(file=sys.stderr)
        print(
            palette.failure(
                "Fail-fast stopped dispatch after the first terminal failure."
            ),
            file=sys.stderr,
        )
        print(report.text, file=sys.stderr)
        print(f"fail_fast_report: {report_path}", file=sys.stderr)
    return summary_path


def _print_cleanup_warnings(state: DashboardState, palette: Palette) -> None:
    for cell_id, detail in state.cleanup_warnings:
        print(
            palette.warning(f"artifact cleanup skipped for {cell_id}: {detail}"),
            file=sys.stderr,
        )


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def _optional_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _first_mapping_text(
    values: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> str | None:
    for value in values:
        for key in keys:
            observed = value.get(key)
            if isinstance(observed, str) and observed.strip():
                return observed.strip()
    return None


def _manual_phase_timeline(
    result: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return only the versioned manual-campaign phase-timeline contract."""

    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    manual = provenance.get("manual_campaign")
    if not isinstance(manual, Mapping):
        return None
    timeline = manual.get("phase_timeline")
    if (
        not isinstance(timeline, Mapping)
        or timeline.get("schema") != _PHASE_TIMELINE_SCHEMA
    ):
        return None
    return timeline


def _persisted_phase_timeline_row(value: object) -> PhaseTimelineRow | None:
    """Decode exactly one entry from the versioned persisted contract."""

    if not isinstance(value, Mapping):
        return None
    phase = _first_mapping_text((value,), ("label", "key"))
    if phase is None:
        return None
    seconds = _optional_finite_float(value.get("seconds"))
    cpu_seconds = _optional_finite_float(value.get("cpu_seconds"))
    peak_memory = _optional_int(value.get("peak_memory_bytes"))
    return PhaseTimelineRow(
        phase=phase,
        wall_seconds=seconds if seconds is not None and seconds >= 0.0 else None,
        cpu_seconds=(
            cpu_seconds if cpu_seconds is not None and cpu_seconds >= 0.0 else None
        ),
        peak_memory_bytes=(
            peak_memory if peak_memory is not None and peak_memory >= 0 else None
        ),
        status=_first_mapping_text((value,), ("status",)),
        detail=_first_mapping_text((value,), ("detail",)),
    )


def _persisted_phase_timeline_rows(
    timeline: Mapping[str, object],
) -> tuple[PhaseTimelineRow, ...]:
    entries = timeline.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    return tuple(
        row
        for entry in entries[:16]
        if (row := _persisted_phase_timeline_row(entry)) is not None
    )


def _lease_phase_timeline_rows(value: object) -> tuple[PhaseTimelineRow, ...]:
    """Decode only the controller's own compact WorkerView lease shape."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[PhaseTimelineRow] = []
    for item in value[:16]:
        if isinstance(item, PhaseTimelineRow):
            rows.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        phase = item.get("phase")
        if not isinstance(phase, str) or not phase:
            continue
        wall = _optional_finite_float(item.get("wall_seconds"))
        cpu = _optional_finite_float(item.get("cpu_seconds"))
        peak = _optional_int(item.get("peak_memory_bytes"))
        status = item.get("status")
        detail = item.get("detail")
        rows.append(
            PhaseTimelineRow(
                phase=phase,
                wall_seconds=wall if wall is not None and wall >= 0.0 else None,
                cpu_seconds=cpu if cpu is not None and cpu >= 0.0 else None,
                peak_memory_bytes=peak if peak is not None and peak >= 0 else None,
                status=status if isinstance(status, str) and status else None,
                detail=detail if isinstance(detail, str) and detail else None,
            )
        )
    return tuple(rows)


def _extract_phase_timeline(
    result: Mapping[str, object],
) -> tuple[PhaseTimelineRow, ...]:
    """Normalize a persisted timeline without requiring it for result reuse."""

    timeline = _manual_phase_timeline(result)
    if timeline is not None:
        return _persisted_phase_timeline_rows(timeline)

    # Older results expose a few exact durations without a timeline wrapper.
    # Use only those recorded numbers; never derive one phase from another.
    rows: list[PhaseTimelineRow] = []
    provenance = result.get("provenance")
    provenance_values = provenance if isinstance(provenance, Mapping) else {}
    preparation = _optional_finite_float(
        provenance_values.get("model_preparation_seconds")
    )
    if preparation is not None and preparation >= 0.0:
        rows.append(
            PhaseTimelineRow(
                "Model preparation",
                wall_seconds=preparation,
                status=(
                    "reused"
                    if provenance_values.get("model_preparation_reused") is True
                    else "measured"
                ),
            )
        )
    generation = _optional_finite_float(result.get("generation_seconds"))
    if generation is not None and generation >= 0.0:
        rows.append(PhaseTimelineRow("Generation", wall_seconds=generation))
    benchmark = provenance_values.get("runtime_profile")
    benchmark_values = benchmark if isinstance(benchmark, Mapping) else {}
    legacy_warmup = benchmark_values.get("warmup")
    legacy_measurement = benchmark_values.get("measurement")
    if isinstance(legacy_warmup, Mapping):
        seconds = _optional_finite_float(legacy_warmup.get("elapsed_seconds"))
        if seconds is not None and seconds >= 0.0:
            rows.append(PhaseTimelineRow("Runtime warm-up", wall_seconds=seconds))
    if isinstance(legacy_measurement, Mapping):
        chunks = legacy_measurement.get("chunks")
        durations: list[float] = []
        if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    durations = []
                    break
                seconds = _optional_finite_float(chunk.get("elapsed_seconds"))
                if seconds is None or seconds < 0.0:
                    durations = []
                    break
                durations.append(seconds)
        if durations:
            rows.append(
                PhaseTimelineRow(
                    "Timed profiling",
                    wall_seconds=math.fsum(durations),
                )
            )
    for key, label in (
        ("warmup_elapsed_seconds", "Runtime warm-up"),
        ("calibration_outer_elapsed_seconds", "Runtime calibration"),
        ("profile_attribution_elapsed_seconds", "Profiling attribution"),
        ("evaluator_elapsed_seconds", "Profiling attribution"),
        ("profile_total_elapsed_seconds", "Complete runtime profile"),
    ):
        seconds = _optional_finite_float(benchmark_values.get(key))
        if seconds is not None and seconds >= 0.0:
            rows.append(PhaseTimelineRow(label, wall_seconds=seconds))
    if not any(
        row.phase in {"Timed profiling", "Timed headline measurement"} for row in rows
    ):
        seconds = _optional_finite_float(
            benchmark_values.get("achieved_runtime_seconds")
        )
        if seconds is None:
            seconds = _optional_finite_float(
                benchmark_values.get("measurement_phase_elapsed_seconds")
            )
        if seconds is not None and seconds >= 0.0:
            rows.append(
                PhaseTimelineRow(
                    "Timed headline measurement",
                    wall_seconds=seconds,
                )
            )
    return tuple(rows)


def _extract_provenance_summary(result: Mapping[str, object]) -> str | None:
    provenance = result.get("provenance")
    provenance_values = provenance if isinstance(provenance, Mapping) else {}
    manual = provenance_values.get("manual_campaign")
    manual_values = manual if isinstance(manual, Mapping) else {}
    identity = manual_values.get("cell_identity")
    identity_values = identity if isinstance(identity, Mapping) else {}
    revision = _first_mapping_text(
        (manual_values, provenance_values, result),
        (
            "report_measured_source_revision",
            "report_source_revision",
            "source_revision",
        ),
    )
    engine = _first_mapping_text(
        (identity_values, manual_values),
        ("execution_mode", "generation_engine", "engine"),
    )
    model = _first_mapping_text(
        (identity_values, manual_values),
        ("model", "model_source"),
    )
    fragments: list[str] = []
    if revision is not None:
        fragments.append(f"source {revision[:12]}")
    if engine is not None:
        fragments.append(f"engine {engine}")
    if model is not None:
        fragments.append(f"model {model}")
    recorded_at = _first_mapping_text((manual_values,), ("recorded_at_utc",))
    if recorded_at is not None:
        fragments.append(f"recorded {recorded_at}")
    return " · ".join(fragments) or None


def _stored_reproduction_command(
    recipe: object,
    key: str,
) -> str | None:
    value = recipe.get(key) if isinstance(recipe, Mapping) else None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(not isinstance(argument, str) or not argument for argument in value)
    ):
        return None
    return _shell_join(value)


def _apply_persisted_worker_result(
    worker: WorkerView,
    result: Mapping[str, object],
    *,
    recycled: bool,
    reuse_explanation: str | None = None,
) -> None:
    """Populate optional dashboard evidence without making reuse depend on it."""

    worker.phase_timeline = _extract_phase_timeline(result)
    worker.provenance_summary = _extract_provenance_summary(result)
    worker.recycled = recycled
    worker.reuse_explanation = reuse_explanation if recycled else None
    timeline = _manual_phase_timeline(result)
    if isinstance(timeline, Mapping):
        total_wall = _optional_finite_float(timeline.get("total_worker_wall_seconds"))
        peak_rss = _optional_int(timeline.get("peak_rss_bytes"))
        peak_guard = _optional_int(timeline.get("peak_guard_bytes"))
        if worker.wall_seconds <= 0.0 and total_wall is not None and total_wall >= 0.0:
            worker.wall_seconds = total_wall
        if worker.peak_rss_bytes <= 0 and peak_rss is not None and peak_rss >= 0:
            worker.peak_rss_bytes = peak_rss
        if worker.peak_guard_bytes <= 0 and peak_guard is not None and peak_guard >= 0:
            worker.peak_guard_bytes = peak_guard


def _concise_terminal_step(value: str | None) -> str | None:
    """Normalize one event-carried terminal reason for bounded display."""

    if value is None:
        return None
    concise = " ".join(_dashboard_log_line(value).split())
    if not concise:
        return None
    return concise if len(concise) <= 240 else f"{concise[:237]}..."


def _terminal_result_step(result: Mapping[str, object]) -> str | None:
    """Return one concise terminal reason from an already loaded current."""

    resources = result.get("resources")
    supervisor = resources.get("supervisor") if isinstance(resources, Mapping) else None
    if (
        isinstance(supervisor, Mapping)
        and supervisor.get("reason") == "phase_state_error"
    ):
        phase_error = supervisor.get("phase_state_error")
        if isinstance(phase_error, str) and phase_error.strip():
            value = phase_error
        else:
            value = "worker phase-state validation failed"
    else:
        policy_label = policy_status_label(result)
        failure = result.get("failure")
        failure_message = (
            failure.get("message") if isinstance(failure, Mapping) else None
        )
        if isinstance(policy_label, str) and policy_label.strip():
            value = policy_label
        elif isinstance(failure_message, str) and failure_message.strip():
            value = failure_message
        else:
            return None
    return _concise_terminal_step(value)


def _coalesced_worker_events(events: Sequence[str]) -> tuple[str, ...]:
    """Collapse repeated resource samples while retaining phase transitions."""

    rows: list[str] = []
    previous_resource: str | None = None
    for event in events:
        resource = event if event.startswith("resource:") else None
        if resource is not None and resource == previous_resource:
            continue
        rows.append(event)
        previous_resource = resource
    return tuple(rows)


def _incremental_tail_lines(
    path: Path,
    state: _IncrementalTailState,
    *,
    max_read_bytes: int = MAX_TAIL_READ_BYTES,
) -> tuple[str, ...]:
    """Read only newly appended complete lines, bounded after large jumps."""

    if max_read_bytes <= 0:
        raise ValueError("max_read_bytes must be positive")
    try:
        metadata = path.stat()
    except OSError:
        state.last_read_bytes = 0
        return ()
    identity = (metadata.st_dev, metadata.st_ino)
    if state.identity != identity or metadata.st_size < state.offset:
        state.identity = identity
        state.offset = 0
        state.pending = b""
    start = state.offset
    discard_partial_prefix = False
    if metadata.st_size - start > max_read_bytes:
        start = metadata.st_size - max_read_bytes
        state.pending = b""
        discard_partial_prefix = start > 0
    try:
        with path.open("rb") as stream:
            stream.seek(start)
            payload = stream.read(max_read_bytes)
    except OSError:
        state.last_read_bytes = 0
        return ()
    state.offset = start + len(payload)
    state.last_read_bytes = len(payload)
    if discard_partial_prefix:
        separator = payload.find(b"\n")
        payload = b"" if separator < 0 else payload[separator + 1 :]
    combined = state.pending + payload
    chunks = combined.split(b"\n")
    if combined.endswith(b"\n"):
        state.pending = b""
        chunks.pop()
    else:
        state.pending = chunks.pop()[-max_read_bytes:]
    return tuple(
        chunk.rstrip(b"\r").decode("utf-8", errors="replace") for chunk in chunks
    )


def _tail_progress(worker: WorkerView) -> None:
    if worker.progress_path is None:
        return
    lines = _incremental_tail_lines(
        Path(worker.progress_path),
        worker._progress_tail_state,
    )
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        task_id = event.get("task_id")
        worker.progress_task_id = (
            str(task_id) if isinstance(task_id, str) and task_id else None
        )
        completed = _optional_int(event.get("completed"))
        total = _optional_int(event.get("total"))
        event_kind = event.get("event")
        if completed is not None:
            worker.progress_completed = completed
        elif event_kind == "start":
            worker.progress_completed = 0
        if total is not None:
            worker.progress_total = total
        details = event.get("details")
        if isinstance(details, Mapping):
            worker.progress_details = {
                str(key): value
                for key, value in details.items()
                if value is None or isinstance(value, (bool, int, float, str))
            }
        else:
            worker.progress_details = {}
        detail_step = worker.progress_details.get("step")
        message = (
            event.get("message")
            or event.get("description")
            or (detail_step if isinstance(detail_step, str) else None)
        )
        worker.progress_message = "" if message is None else str(message)
        if worker.progress_message:
            worker.step = worker.progress_message
        return


def _dashboard_log_line(value: str) -> str:
    value = _ANSI_ESCAPE.sub("", value)
    cleaned = "".join(
        character
        for character in value.replace("\t", "  ")
        if ord(character) >= 32 and ord(character) != 127
    )
    return cleaned[-500:]


def _tail_log(worker: WorkerView) -> None:
    if worker.log_path is None:
        return
    lines = _incremental_tail_lines(Path(worker.log_path), worker._log_tail_state)
    worker.log_tail.extend(
        cleaned for line in lines if (cleaned := _dashboard_log_line(line))
    )
    del worker.log_tail[:-MAX_LOG_TAIL_LINES]


def _human_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}"


def _human_phase_duration(seconds: float) -> str:
    """Keep short phase timings visible without widening the worker runtime."""

    seconds = max(0.0, seconds)
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        minutes, remainder = divmod(seconds, 60.0)
        return f"{int(minutes):02d}:{remainder:04.1f}"
    return _human_duration(seconds)


def _human_bytes(value: int) -> str:
    if value < 1000:
        return f"{value} B"
    units = ("kB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        amount /= 1000.0
        if amount < 1000.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
    return f"{value} B"


def _progress_detail_summary(details: Mapping[str, object]) -> str:
    """Render typed progress details without discarding native stage counters."""

    ignored = {"process", "step"}
    ordered_keys = (
        "stage_index",
        "stage_total",
        "subset_size",
        "candidate_parent_pair_count",
        "candidate_parent_pair_total",
        "current_count",
        "contribution_count",
        "dynamic_color_state_count",
        "color_target_prune_count",
        "inspected_current_count",
        "certified_relation_count",
        "applied_relation_count",
        "generation_evidence_encoding",
        "generation_raw_evidence_bytes",
        "generation_evidence_transport_bytes",
    )
    ordered = [key for key in ordered_keys if key in details]
    ordered.extend(
        sorted(key for key in details if key not in ignored and key not in ordered)
    )
    fragments: list[str] = []
    consumed: set[str] = set()
    if "stage_index" in details and "stage_total" in details:
        fragments.append(f"stage {details['stage_index']}/{details['stage_total']}")
        consumed.update({"stage_index", "stage_total"})
    if (
        "candidate_parent_pair_count" in details
        and "candidate_parent_pair_total" in details
    ):
        count = details["candidate_parent_pair_count"]
        total = details["candidate_parent_pair_total"]
        if isinstance(count, int) and isinstance(total, int):
            fragments.append(f"candidate pairs {count:,}/{total:,}")
            consumed.update(
                {"candidate_parent_pair_count", "candidate_parent_pair_total"}
            )
    labels = {
        "subset_size": "subset",
        "current_count": "currents",
        "contribution_count": "contributions",
        "dynamic_color_state_count": "colour states",
        "color_target_prune_count": "colour pruned",
        "inspected_current_count": "currents inspected",
        "certified_relation_count": "relations certified",
        "applied_relation_count": "relations applied",
        "generation_evidence_encoding": "evidence",
        "generation_raw_evidence_bytes": "evidence raw",
        "generation_evidence_transport_bytes": "evidence transport",
    }
    byte_fields = {
        "generation_raw_evidence_bytes",
        "generation_evidence_transport_bytes",
    }
    for key in ordered:
        if key in consumed:
            continue
        value = details[key]
        label = labels.get(key, key.replace("_", " "))
        if key in byte_fields and isinstance(value, int):
            rendered = _human_bytes(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            rendered = f"{value:,}"
        else:
            rendered = str(value)
        fragments.append(f"{label} {rendered}")
    return " · ".join(fragments)


def _worker_progress_ratio(worker: WorkerView) -> float:
    if (
        worker.progress_completed is not None
        and worker.progress_total is not None
        and worker.progress_total > 0
    ):
        return min(1.0, worker.progress_completed / worker.progress_total)
    phase_ratio = {
        "preparation": 0.05,
        "generation": 0.35,
        "profiling": 0.75,
        "completed": 1.0,
    }
    return phase_ratio.get(worker.phase, 0.1)


def _worker_progress_indicator(worker: WorkerView, *, compact: bool) -> str:
    ratio = _worker_progress_ratio(worker)
    slots = 4 if compact else 6
    filled = min(slots, max(0, round(ratio * slots)))
    return f"{'▰' * filled}{'▱' * (slots - filled)} {ratio * 100:3.0f}%"


def _worker_enforcement_memory(worker: WorkerView, *, peak: bool = False) -> int:
    guard = worker.peak_guard_bytes if peak else worker.current_guard_bytes
    rss = worker.peak_rss_bytes if peak else worker.current_rss_bytes
    return guard if guard > 0 else rss


def _reproduction_command(worker: WorkerView | None, stage: str | None) -> str | None:
    """Return one exact persisted command without regenerating or normalizing it."""

    if worker is None or stage not in _REPRODUCTION_STAGES:
        return None
    value = getattr(worker, f"reproduce_{stage}")
    return value if isinstance(value, str) and value else None


def _available_reproduction_stages(worker: WorkerView | None) -> tuple[str, ...]:
    return tuple(
        stage
        for stage in _REPRODUCTION_STAGES
        if _reproduction_command(worker, stage) is not None
    )


def _cycle_reproduction_stage(state: DashboardState, step: int = 1) -> None:
    available = _available_reproduction_stages(state.selected_worker())
    if not available:
        state.command_stage = None
        state.command_notice = "No public CLI reproduction is available"
        state.command_scroll = 0
        return
    try:
        index = available.index(state.command_stage or "")
    except ValueError:
        index = -1 if step >= 0 else 0
    state.command_stage = available[(index + step) % len(available)]
    state.command_scroll = 0
    state.command_notice = None


def _osc52_clipboard_sequence(value: str) -> str:
    """Encode an explicitly requested clipboard copy without a subprocess."""

    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{encoded}\x07"


def _phase_timeline_lines(
    rows: Sequence[PhaseTimelineRow],
    *,
    compact: bool,
    limit: int,
) -> tuple[str, ...]:
    """Render a small aligned phase table that remains legible at 80 columns."""

    rendered: list[str] = []
    for row in rows[: max(0, limit)]:
        wall = (
            "unavailable"
            if row.wall_seconds is None
            else _human_phase_duration(row.wall_seconds)
        )
        outcome = (
            " · ".join(value for value in (row.status, row.detail) if value)
            or "recorded"
        )
        if compact:
            rendered.append(f"{row.phase[:20]} · {wall} · {outcome}")
            continue
        cpu = (
            "unavailable"
            if row.cpu_seconds is None
            else _human_phase_duration(row.cpu_seconds)
        )
        peak = (
            "unavailable"
            if row.peak_memory_bytes is None
            else _human_bytes(row.peak_memory_bytes)
        )
        rendered.append(
            f"{row.phase[:20]:<20}  {wall:>11}  {cpu:>11}  {peak:>11}  {outcome}"
        )
    if len(rows) > len(rendered):
        rendered.append(f"… {len(rows) - len(rendered)} more persisted phases")
    return tuple(rendered)


def _worker_from_lease(
    cell_id: str,
    raw: Mapping[str, object],
    *,
    peer_instance: str | None,
) -> WorkerView:
    """Decode one informational worker row without following referenced paths."""

    def optional_text(key: str) -> str | None:
        value = raw.get(key)
        return value if isinstance(value, str) and value else None

    def text(key: str, default: str) -> str:
        value = optional_text(key)
        return default if value is None else value

    raw_status = text("status", "unknown")
    recycled = raw.get("recycled") is True or raw_status in _RECYCLED_SUCCESS_STATUSES
    worker = WorkerView(
        cell_id=cell_id,
        dependency=raw.get("dependency") is True,
        peer_instance=peer_instance,
        generation_engine=optional_text("generation_engine"),
        status="recycled" if raw_status in _RECYCLED_SUCCESS_STATUSES else raw_status,
        step=text("step", "waiting"),
        phase=text("phase", "unknown"),
        attempt_id=optional_text("attempt_id"),
        pid=_optional_int(raw.get("pid")),
        wall_seconds=max(0.0, _finite_float(raw.get("wall_seconds"))),
        cpu_seconds=(
            None
            if (cpu_seconds := _optional_finite_float(raw.get("cpu_seconds"))) is None
            else max(0.0, cpu_seconds)
        ),
        current_rss_bytes=max(
            0,
            _optional_int(raw.get("current_rss_bytes")) or 0,
        ),
        peak_rss_bytes=max(
            0,
            _optional_int(raw.get("peak_rss_bytes")) or 0,
        ),
        current_physical_footprint_bytes=_optional_int(
            raw.get("current_physical_footprint_bytes")
        ),
        peak_physical_footprint_bytes=_optional_int(
            raw.get("peak_physical_footprint_bytes")
        ),
        current_guard_bytes=max(
            0,
            _optional_int(raw.get("current_guard_bytes")) or 0,
        ),
        peak_guard_bytes=max(
            0,
            _optional_int(raw.get("peak_guard_bytes")) or 0,
        ),
        child_count=max(0, _optional_int(raw.get("child_count")) or 0),
        progress_completed=_optional_int(raw.get("progress_completed")),
        progress_total=_optional_int(raw.get("progress_total")),
        progress_message=text("progress_message", ""),
        progress_task_id=optional_text("progress_task_id"),
        log_path=optional_text("log_path"),
        progress_path=optional_text("progress_path"),
        reproduce_prepare=optional_text("reproduce_prepare"),
        reproduce_generate=optional_text("reproduce_generate"),
        reproduce_profile=optional_text("reproduce_profile"),
        published_wall_seconds_per_point=_optional_finite_float(
            raw.get("published_wall_seconds_per_point")
        ),
        published_evaluator_total_seconds_per_point=_optional_finite_float(
            raw.get("published_evaluator_total_seconds_per_point")
        ),
        published_recurrence_core_seconds_per_point=_optional_finite_float(
            raw.get("published_recurrence_core_seconds_per_point")
        ),
        phase_timeline=_lease_phase_timeline_rows(raw.get("phase_timeline")),
        provenance_summary=optional_text("provenance_summary"),
        reuse_explanation=optional_text("reuse_explanation"),
        blocked_prerequisite_ids=tuple(
            sorted(
                {
                    str(value)
                    for value in raw.get("blocked_prerequisite_ids", ())
                    if isinstance(value, str) and value
                }
            )
        )
        if isinstance(raw.get("blocked_prerequisite_ids"), Sequence)
        and not isinstance(raw.get("blocked_prerequisite_ids"), (str, bytes))
        else (),
        recycled=recycled,
        updated_at=max(0.0, _finite_float(raw.get("updated_at"))),
    )
    raw_member_pids = raw.get("member_pids")
    if isinstance(raw_member_pids, Sequence) and not isinstance(
        raw_member_pids,
        (str, bytes),
    ):
        worker.member_pids = tuple(
            value
            for item in raw_member_pids
            if (value := _optional_int(item)) is not None and value >= 0
        )
    raw_progress_details = raw.get("progress_details")
    if isinstance(raw_progress_details, Mapping):
        worker.progress_details = {
            str(key): value
            for key, value in raw_progress_details.items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
    raw_events = raw.get("events")
    if isinstance(raw_events, Sequence) and not isinstance(
        raw_events,
        (str, bytes),
    ):
        worker.events = [str(value) for value in raw_events[-8:]]
    raw_log_tail = raw.get("log_tail")
    if isinstance(raw_log_tail, Sequence) and not isinstance(
        raw_log_tail,
        (str, bytes),
    ):
        worker.log_tail = [
            _dashboard_log_line(str(value))
            for value in raw_log_tail[-MAX_LOG_TAIL_LINES:]
        ]
    return worker


def _live_lease_workers(
    service: ReportService,
    selected_ids: frozenset[str],
    *,
    exclude_instance: str,
    source_revision: str,
) -> tuple[WorkerView, ...]:
    root = service.paths.coordination_root / "manual-leases"
    now = time.time()
    workers: list[WorkerView] = []
    if not root.is_dir():
        return ()
    for path in root.glob("*.json"):
        payload = _read_object(path)
        if payload is None:
            continue
        updated = _finite_float(payload.get("updated_at"))
        if now - updated > DEFAULT_WORKER_STALE_SECONDS:
            continue
        instance_id = str(payload.get("instance_id") or "")
        if not instance_id or instance_id == exclude_instance:
            continue
        if payload.get("source_revision") != source_revision:
            continue
        raw_workers = payload.get("workers")
        if not isinstance(raw_workers, Mapping):
            continue
        for cell_id, raw in raw_workers.items():
            if cell_id not in selected_ids or not isinstance(raw, Mapping):
                continue
            worker = _worker_from_lease(
                str(cell_id),
                raw,
                peer_instance=instance_id,
            )
            if worker.status in {"queued", "preparing", "running"}:
                workers.append(worker)
    return tuple(workers)


def _merge_peer_workers(
    service: ReportService,
    state: DashboardState,
) -> None:
    for key in tuple(state.workers):
        if state.workers[key].peer_instance is not None:
            del state.workers[key]
    peers = _live_lease_workers(
        service,
        frozenset(state.selected_ids),
        exclude_instance=state.instance_id,
        source_revision=state.source_revision,
    )
    for worker in peers:
        state.workers[f"peer:{worker.peer_instance}:{worker.cell_id}"] = worker


def _lease_counter_snapshot(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    counters: dict[str, int] = {}
    for key in _DASHBOARD_COUNTER_KEYS:
        observed = _optional_int(
            value.get(key, 0 if key == "unverified" else None)
        )
        if observed is None or observed < 0:
            return None
        counters[key] = observed
    return counters


def _lease_limits(
    value: object,
) -> tuple[float, int, float | None] | None:
    """Decode effective invocation caps, retaining defaults for old leases."""

    if value is None:
        return (
            DEFAULT_GENERATION_LIMIT_SECONDS,
            DEFAULT_RAM_BYTES,
            DEFAULT_WORKER_WALL_LIMIT_SECONDS,
        )
    if not isinstance(value, Mapping):
        return None
    generation = _optional_finite_float(value.get("generation_time_limit_seconds"))
    memory = _optional_int(value.get("memory_limit_bytes"))
    raw_wall = value.get("worker_wall_limit_seconds")
    wall = None if raw_wall is None else _optional_finite_float(raw_wall)
    if (
        generation is None
        or generation <= 0.0
        or memory is None
        or memory <= 0
        or (raw_wall is not None and (wall is None or wall <= 0.0))
    ):
        return None
    return generation, memory, wall


def _lease_scheduler_state(value: object) -> tuple[int, int, int]:
    if not isinstance(value, Mapping):
        return (0, 0, 0)
    observed = tuple(
        _optional_int(value.get(key))
        for key in ("ready", "waiting_dependency", "waiting_coordination_lock")
    )
    if any(item is None or item < 0 for item in observed):
        return (0, 0, 0)
    return tuple(int(item) for item in observed)  # type: ignore[arg-type,return-value]


def _live_dashboard_snapshot(
    coordination_root: Path,
    *,
    instance: str | None,
    stale_after_seconds: float,
    now: float | None = None,
) -> DashboardState:
    """Build one read-only frame from compact, non-stale lease JSON only."""

    if not math.isfinite(stale_after_seconds) or stale_after_seconds <= 0.0:
        raise ManualCampaignError("--stale-after must be a positive finite value")
    lease_root = coordination_root / "manual-leases"
    current_time = time.time() if now is None else now
    records: list[dict[str, object]] = []
    if lease_root.is_dir():
        for path in sorted(lease_root.glob("*.json")):
            payload = _read_object(path)
            if payload is None or payload.get("schema") != MANUAL_STATE_SCHEMA:
                continue
            instance_id = payload.get("instance_id")
            source_revision = payload.get("source_revision")
            updated_at = _optional_finite_float(payload.get("updated_at"))
            counters = _lease_counter_snapshot(payload.get("counters"))
            limits = _lease_limits(payload.get("limits"))
            workers = payload.get("workers")
            if (
                not isinstance(instance_id, str)
                or not instance_id
                or len(instance_id) > 128
                or not isinstance(source_revision, str)
                or not source_revision
                or updated_at is None
                or updated_at <= 0.0
                or current_time - updated_at > stale_after_seconds
                or counters is None
                or limits is None
                or not isinstance(workers, Mapping)
            ):
                continue
            records.append(payload)
    if not records:
        raise ManualCampaignError(
            "no active manual-campaign lease is available; start `run` in "
            "another terminal or increase --stale-after"
        )

    chosen: dict[str, object] | None = None
    if instance is not None:
        exact = [
            payload for payload in records if payload.get("instance_id") == instance
        ]
        matches = (
            exact
            if exact
            else [
                payload
                for payload in records
                if str(payload.get("instance_id")).startswith(instance)
            ]
        )
        identifiers = sorted({str(payload.get("instance_id")) for payload in matches})
        if not matches:
            available = ", ".join(
                sorted(str(payload.get("instance_id")) for payload in records)
            )
            raise ManualCampaignError(
                f"no active lease matches instance {instance!r}; available: {available}"
            )
        if len(identifiers) > 1:
            raise ManualCampaignError(
                f"instance prefix {instance!r} is ambiguous: " + ", ".join(identifiers)
            )
        chosen = max(
            matches,
            key=lambda payload: _finite_float(payload.get("updated_at")),
        )
    else:
        chosen = max(
            records,
            key=lambda payload: _finite_float(payload.get("updated_at")),
        )

    instance_id = str(chosen["instance_id"])
    source_revision = str(chosen["source_revision"])
    updated_at = _finite_float(chosen["updated_at"])
    started_at = _optional_finite_float(chosen.get("started_at")) or updated_at
    counters = _lease_counter_snapshot(chosen["counters"])
    assert counters is not None
    limits = _lease_limits(chosen.get("limits"))
    assert limits is not None
    generation_limit, memory_limit, wall_limit = limits
    ready_count, waiting_dependency_count, waiting_coordination_lock_count = (
        _lease_scheduler_state(chosen.get("scheduler"))
    )
    state = DashboardState(
        instance_id=instance_id,
        selected_ids=(),
        recycled_ids=set(),
        static_na_ids=set(),
        source_revision=source_revision,
        generation_time_limit_seconds=generation_limit,
        memory_limit_bytes=memory_limit,
        worker_wall_limit_seconds=wall_limit,
        started_at=min(started_at, current_time),
        counter_snapshot=counters,
        lease_updated_at=updated_at,
        live_snapshot=True,
        ready_count=ready_count,
        waiting_dependency_count=waiting_dependency_count,
        waiting_coordination_lock_count=waiting_coordination_lock_count,
    )
    raw_workers = chosen["workers"]
    assert isinstance(raw_workers, Mapping)
    for cell_id, raw in raw_workers.items():
        if not isinstance(cell_id, str) or not cell_id or not isinstance(raw, Mapping):
            continue
        state.workers[cell_id] = _worker_from_lease(
            cell_id,
            raw,
            peer_instance=None,
        )

    # Show active workers from concurrent same-source instances as peer rows.
    # The overview counters remain the chosen invocation's own atomic snapshot,
    # avoiding guesses or double-counting when selections overlap.
    for payload in records:
        peer_instance = str(payload["instance_id"])
        if (
            peer_instance == instance_id
            or payload.get("source_revision") != source_revision
        ):
            continue
        raw_peer_workers = payload["workers"]
        assert isinstance(raw_peer_workers, Mapping)
        for cell_id, raw in raw_peer_workers.items():
            if (
                not isinstance(cell_id, str)
                or not cell_id
                or not isinstance(raw, Mapping)
            ):
                continue
            worker = _worker_from_lease(
                cell_id,
                raw,
                peer_instance=peer_instance,
            )
            if worker.status not in {"queued", "preparing", "running"}:
                continue
            state.workers[f"peer:{peer_instance}:{cell_id}"] = worker

    ordered_workers = state.visible_workers()
    state.selected_index = next(
        (
            index
            for index, worker in enumerate(ordered_workers)
            if worker.status in _ACTIVE_WORKER_STATUSES
        ),
        0,
    )
    return state


def _snapshot_fixture(
    *,
    selected: int = 1796,
    recycled: int = 318,
    completed: int = 41,
) -> DashboardState:
    state = DashboardState(
        instance_id="snapshot-demo",
        selected_ids=tuple(f"cell-{index:04d}" for index in range(selected)),
        recycled_ids={f"cell-{index:04d}" for index in range(recycled)},
        static_na_ids={f"cell-{index:04d}" for index in range(12)},
        source_revision="5b58aeb7600548e84e7214ee0ef62e5f159ec3fb",
    )
    for index in range(recycled, recycled + completed):
        state._reduce_finished(f"cell-{index:04d}", "ok")
    state._reduce_finished("cell-0412", "memory_limit")
    state._reduce_finished("cell-0413", "error")
    state.dependency_ids = {"reference-amplicol-lc-n6-dd-zzz-jets-selected-flow"}
    state.workers = {
        "matrix-compiled-builtin-sm-lc-n9-dd-zzz-jets-selected-flow": WorkerView(
            "matrix-compiled-builtin-sm-lc-n9-dd-zzz-jets-selected-flow",
            generation_engine=ExecutionMode.COMPILED.value,
            status="running",
            step="native Arena sample 4/8",
            phase="profiling",
            attempt_id="b92b46a0-demo",
            pid=42173,
            member_pids=(42173, 42174, 42175),
            wall_seconds=412.8,
            cpu_seconds=971.2,
            current_rss_bytes=8_724_300_000,
            peak_rss_bytes=11_941_200_000,
            child_count=2,
            progress_completed=4,
            progress_total=8,
            progress_message="native Arena sample 4/8",
            published_wall_seconds_per_point=0.000_669_99,
            published_evaluator_total_seconds_per_point=0.000_654_31,
            events=[
                "generation: compiled Direct-Arena complete",
                "profiling: warmups complete",
                "profiling: native Arena sample 4/8",
            ],
        ),
        "matrix-recurrence-ufo-sm-lc-n7-gg-gluons-all-flow": WorkerView(
            "matrix-recurrence-ufo-sm-lc-n7-gg-gluons-all-flow",
            generation_engine=ExecutionMode.RECURRENCE.value,
            status="ok",
            step="measurement published",
            phase="complete",
            attempt_id="dd84040e-demo",
            pid=42191,
            member_pids=(42191,),
            wall_seconds=224.7,
            cpu_seconds=216.3,
            current_rss_bytes=3_252_000_000,
            peak_rss_bytes=4_018_000_000,
            progress_completed=72,
            progress_total=72,
            progress_message="measurement published",
            published_wall_seconds_per_point=218.105e-6,
            published_evaluator_total_seconds_per_point=217.812e-6,
            published_recurrence_core_seconds_per_point=205.431e-6,
            events=[
                "generation: model loaded",
                "generation: recurrence plan projected",
                "profiling: independent clocks published",
            ],
        ),
        "reference-amplicol-full-n6-dd-3q-lines-contracted": WorkerView(
            "reference-amplicol-full-n6-dd-3q-lines-contracted",
            dependency=True,
            generation_engine=ExecutionMode.AMPLICOL.value,
            status="preparing",
            step="prewarming original AmpliCol generator",
            phase="preparation",
            wall_seconds=8.3,
            current_rss_bytes=181_000_000,
            peak_rss_bytes=205_000_000,
            events=["preparation: legacy generator prewarm"],
        ),
    }
    worker_ids = tuple(state.workers)
    state._reduce_finished(
        "matrix-recurrence-ufo-sm-lc-n7-gg-gluons-all-flow",
        "ok",
    )
    state.selected_ids = (
        *worker_ids,
        *(f"cell-{index:04d}" for index in range(max(0, selected - len(worker_ids)))),
    )
    return state


def _append_dashboard_line(
    paragraph: Any,
    spans: Sequence[tuple[str, object]],
) -> None:
    """Append a styled line through Ratatui's stable single-span API."""

    for text, style in spans:
        paragraph.append_span(text, style)
    paragraph.line_break()


def _fit_dashboard_field(value: object, width: int) -> str:
    text = str(value)
    if width <= 0:
        return ""
    if len(text) > width:
        text = text[: max(0, width - 1)] + "…"
    return text.ljust(width)


def _append_detail_fields(
    paragraph: Any,
    fields: Sequence[tuple[str, object, object]],
    *,
    width: int,
    compact: bool,
    key_style: object,
) -> None:
    """Render fixed detail columns so keys and values retain vertical anchors."""

    content_width = max(20, width - 4)
    gutter = 1 if compact else 3
    group_width = max(20, (content_width - gutter) // 2)
    key_width = min(16 if compact else 18, max(12, group_width // 2))
    value_width = max(1, group_width - key_width)
    for offset in range(0, len(fields), 2):
        spans: list[tuple[str, object]] = []
        for local_index, (key, value, value_style) in enumerate(
            fields[offset : offset + 2]
        ):
            if local_index:
                spans.append((" " * gutter, key_style))
            spans.extend(
                (
                    (f"{key:<{key_width}}", key_style),
                    (_fit_dashboard_field(value, value_width), value_style),
                )
            )
        _append_dashboard_line(paragraph, spans)


def _ratatui_commands(
    state: DashboardState,
    *,
    width: int,
    height: int,
    color: bool = True,
) -> list[object]:
    try:
        from ratatui import Color, DrawCmd, Gauge, Paragraph
        from ratatui import Style as RStyle
    except ImportError as error:
        raise ManualCampaignError(
            "Ratatui is unavailable; run `just dev-install`"
        ) from error

    counters = state.counters()
    peer_active_count = len(state.peer_active)
    compact = width < 105 or height < 30
    overview_height = 7 if compact else 8
    table_height = max(7, min(13, height // 3) - (1 if compact else 0))
    footer_height = 1
    detail_y = overview_height + table_height
    detail_height = max(4, height - detail_y - footer_height)

    if color:
        title_style = RStyle(fg=Color.Cyan).bold()
        key_style = RStyle(fg=Color.Cyan).bold()
        success_style = RStyle(fg=Color.Green)
        warning_style = RStyle(fg=Color.Yellow)
        failure_style = RStyle(fg=Color.Red)
        neutral_style = RStyle(fg=Color.White)
        muted_style = RStyle(fg=Color.DarkGray).dim()
        metric_style = RStyle(fg=Color.LightCyan)
        memory_style = RStyle(fg=Color.LightMagenta)
        table_header_style = RStyle(fg=Color.LightCyan).bold().underlined()
        summary_style = RStyle(fg=Color.LightCyan, bg=Color.DarkGray).bold()
        highlight_style = RStyle(fg=Color.Black, bg=Color.Cyan).bold()
        gauge_label_style = RStyle(fg=Color.White).bold()
        gauge_value_style = RStyle(fg=Color.Green, bg=Color.Black).bold()
    else:
        title_style = RStyle()
        key_style = RStyle()
        success_style = RStyle()
        warning_style = RStyle()
        failure_style = RStyle()
        neutral_style = RStyle()
        muted_style = RStyle()
        metric_style = RStyle()
        memory_style = RStyle()
        table_header_style = RStyle()
        summary_style = RStyle()
        highlight_style = RStyle()
        gauge_label_style = RStyle()
        gauge_value_style = RStyle()

    def status_style(status: str) -> object:
        if status in {"error", "failed", "blocked_dependency"}:
            return failure_style
        if status in {
            *_TERMINAL_CAP_STATUSES,
            "timeout",
            "cancelled",
            ResultStatus.UNVERIFIED.value,
        }:
            return warning_style
        if status in _COMPLETED_WORKER_STATUSES:
            return success_style
        if status in _ACTIVE_WORKER_STATUSES:
            return warning_style
        return muted_style

    def phase_style(phase: str) -> object:
        normalized = phase.casefold().replace("_", "-")
        if "profil" in normalized:
            return metric_style
        if "generat" in normalized or "post-generation" in normalized:
            return memory_style
        if normalized in {"complete", "completed"}:
            return success_style
        if "prepar" in normalized:
            return warning_style
        return neutral_style

    def outcome_style(status: str | None) -> object:
        normalized = (status or "").casefold()
        if any(value in normalized for value in ("error", "failed")):
            return failure_style
        if any(value in normalized for value in ("limit", "capped", "timeout")):
            return warning_style
        if any(
            value in normalized
            for value in ("measured", "completed", "reused", "observed")
        ):
            return success_style
        if any(value in normalized for value in ("unavailable", "not_applicable")):
            return muted_style
        return neutral_style

    overview = Paragraph.new_empty()
    overview.set_block_title(
        f" LIVE pyAmpliCol profiling campaign · {state.instance_id[:12]} "
        if state.live_snapshot
        else " pyAmpliCol profiling campaign "
    )
    selection_progress = [
        ("Selected ", key_style),
        (str(counters["selected"]), neutral_style),
        ("   Recycled ", key_style),
        (str(counters["recycled"]), success_style),
        ("   Completed ", key_style),
        (str(counters["completed"]), success_style),
        ("   Remaining ", key_style),
        (str(counters["remaining"]), neutral_style),
    ]
    selection_issues = [
        ("   Capped ", key_style),
        (str(counters["capped"]), warning_style),
        ("   Errors ", key_style),
        (str(counters["failed"]), failure_style),
        ("   Unverified ", key_style),
        (str(counters["unverified"]), warning_style),
    ]
    if compact:
        _append_dashboard_line(overview, selection_progress)
        _append_dashboard_line(
            overview,
            [
                *selection_issues,
                ("   Static N/A ", key_style),
                (str(counters["static_na"]), warning_style),
            ],
        )
    else:
        _append_dashboard_line(overview, [*selection_progress, *selection_issues])
    worker_scope = [
        ("Workers active ", key_style),
        (str(counters["active"]), warning_style),
        ("   selected ", key_style),
        (str(counters["selected_active"]), warning_style),
        ("   dependency ", key_style),
        (str(counters["dependency_active"]), warning_style),
        *(
            (
                ("   Peer active ", key_style),
                (str(peer_active_count), metric_style),
            )
            if peer_active_count
            else ()
        ),
        *(
            ()
            if compact
            else (
                ("   Static N/A ", key_style),
                (str(counters["static_na"]), warning_style),
            )
        ),
        ("   Dependency-only ", key_style),
        (str(counters["dependency_only"]), neutral_style),
        *(
            (
                ("   Dependency done ", key_style),
                (str(counters["dependency_completed"]), success_style),
                ("   issues ", key_style),
                (str(counters["dependency_issues"]), failure_style),
            )
            if not compact
            else ()
        ),
    ]
    _append_dashboard_line(overview, worker_scope)
    _append_dashboard_line(
        overview,
        [
            ("Ready ", key_style),
            (str(state.ready_count), success_style),
            ("   Waiting dependency ", key_style),
            (str(state.waiting_dependency_count), muted_style),
            ("   Waiting coordination lock ", key_style),
            (str(state.waiting_coordination_lock_count), warning_style),
        ],
    )
    _append_dashboard_line(
        overview,
        [
            ("Caps ", key_style),
            ("Generation ", key_style),
            (
                _human_duration(state.generation_time_limit_seconds),
                warning_style,
            ),
            ("   RAM ", key_style),
            (_human_bytes(state.memory_limit_bytes), warning_style),
            ("   Total wall ", key_style),
            (
                (
                    "disabled"
                    if state.worker_wall_limit_seconds is None
                    else _human_duration(state.worker_wall_limit_seconds)
                ),
                warning_style,
            ),
        ],
    )
    if not compact:
        _append_dashboard_line(
            overview,
            [
                ("Elapsed ", key_style),
                (_human_duration(time.time() - state.started_at), neutral_style),
                ("   Source ", key_style),
                (state.source_revision[:12] or "unavailable", neutral_style),
                *(
                    (
                        ("   Lease age ", key_style),
                        (
                            _human_duration(
                                max(0.0, time.time() - state.lease_updated_at)
                            ),
                            success_style,
                        ),
                    )
                    if state.lease_updated_at is not None
                    else ()
                ),
            ],
        )
        _append_dashboard_line(
            overview,
            [
                ("Clocks ", key_style),
                (
                    "outer wall, evaluator-total, and recurrence core stay independent",
                    neutral_style,
                ),
            ],
        )

    workers = state.visible_workers()
    selected_index = 0
    viewport_start = 0
    viewport_capacity = max(1, table_height - 3)
    if workers:
        selected_index = state.selected_index % len(workers)
        state.selected_index = selected_index
        viewport_start = min(
            max(0, selected_index - viewport_capacity + 1),
            max(0, len(workers) - viewport_capacity),
        )
    viewport = workers[viewport_start : viewport_start + viewport_capacity]
    viewport_end = viewport_start + len(viewport)
    worker_table = Paragraph.new_empty()
    worker_table.set_block_title(
        " Workers ↑/↓"
        f" · {viewport_start + 1 if viewport else 0}-{viewport_end}/{len(workers)}"
        f" · d done:{'on' if state.show_completed else 'off'}"
        f" · e errors:{'on' if state.show_errors else 'off'} "
    )
    if compact:
        headers = [
            " ",
            "cell",
            "status",
            "phase / step",
            "progress",
            "runtime",
            "cap guard",
        ]
        percentages = [3, 25, 10, 24, 16, 10, 12]
    else:
        headers = [
            " ",
            "cell",
            "status",
            "phase / step",
            "progress",
            "runtime",
            "cap guard",
            "PID tree",
        ]
        percentages = [3, 23, 9, 22, 13, 8, 12, 10]
    table_content_width = max(len(headers), width - 2)
    spacing_width = len(headers) - 1
    distributable_width = max(len(headers), table_content_width - spacing_width)
    column_widths = [
        max(1, distributable_width * percentage // 100) for percentage in percentages
    ]
    column_widths[-1] += max(
        0,
        distributable_width - sum(column_widths),
    )
    header_spans: list[tuple[str, object]] = []
    for header_index, (header, column_width) in enumerate(
        zip(headers, column_widths, strict=True)
    ):
        if header_index:
            header_spans.append((" ", table_header_style))
        header_spans.append(
            (_fit_dashboard_field(header, column_width), table_header_style)
        )
    _append_dashboard_line(worker_table, header_spans)
    for local_index, worker in enumerate(viewport):
        index = viewport_start + local_index
        marker = "▶" if index == selected_index else " "
        enforcement = _worker_enforcement_memory(worker)
        guard_style = (
            failure_style
            if enforcement > state.memory_limit_bytes
            else (
                warning_style
                if enforcement >= int(0.8 * state.memory_limit_bytes)
                else muted_style
            )
        )
        row_values = [
            marker,
            worker.cell_id,
            (
                "blocked by dependency"
                if worker.status == "blocked_dependency"
                else worker.status
            ),
            worker.step,
            _worker_progress_indicator(worker, compact=compact),
            _human_duration(worker.wall_seconds),
            _human_bytes(enforcement),
        ]
        row_styles = [
            title_style,
            title_style if index == selected_index else neutral_style,
            status_style(worker.status),
            phase_style(worker.phase),
            status_style(worker.status),
            metric_style,
            guard_style,
        ]
        if not compact:
            row_values.append(
                "—"
                if worker.pid is None
                else f"{worker.pid}+{max(0, len(worker.member_pids) - 1)}"
            )
            row_styles.append(muted_style)
        row_spans: list[tuple[str, object]] = []
        for column_index, (value, value_style, column_width) in enumerate(
            zip(row_values, row_styles, column_widths, strict=True)
        ):
            rendered_style = highlight_style if index == selected_index else value_style
            if column_index:
                row_spans.append((" ", rendered_style))
            row_spans.append(
                (_fit_dashboard_field(value, column_width), rendered_style)
            )
        _append_dashboard_line(worker_table, row_spans)
    worker_table.set_wrap(False)

    selected = state.selected_worker()
    details = Paragraph.new_empty()
    details.set_block_title(
        " Dashboard help " if state.show_help else " Selected worker "
    )
    if state.show_help:
        _append_dashboard_line(
            details, [("↑/↓ or j/k ", key_style), ("select worker", neutral_style)]
        )
        _append_dashboard_line(
            details,
            [("PgUp/PgDn ", key_style), ("scroll worker details", neutral_style)],
        )
        _append_dashboard_line(
            details,
            [
                ("d ", key_style),
                ("show/hide completed and successful reused workers", neutral_style),
            ],
        )
        _append_dashboard_line(
            details,
            [
                ("e ", key_style),
                ("show/hide errors, caps, and recycled attention rows", neutral_style),
            ],
        )
        _append_dashboard_line(
            details,
            [
                ("1/2/3 ", key_style),
                ("open full prepare/generate/profile command", neutral_style),
            ],
        )
        _append_dashboard_line(
            details,
            [
                ("y / p ", key_style),
                ("copy exact command / print it for terminal selection", neutral_style),
            ],
        )
        _append_dashboard_line(
            details, [("? ", key_style), ("close this help", neutral_style)]
        )
        _append_dashboard_line(
            details,
            [
                ("Ctrl-C/Esc ", key_style),
                ("stop dispatch and workers safely", warning_style),
            ],
        )
        ratio = 0.0 if selected is None else _worker_progress_ratio(selected)
        gauge_label = "keyboard help"
    elif selected is None:
        _append_dashboard_line(
            details,
            [
                ("No workers match the current d/e filters", warning_style),
            ],
        )
        ratio = 0.0
        gauge_label = "filtered"
    else:
        _append_dashboard_line(
            details,
            [
                ("Cell ", key_style),
                (selected.cell_id, title_style),
                *(
                    (
                        ("   Peer ", key_style),
                        (selected.peer_instance[:12], neutral_style),
                    )
                    if selected.peer_instance is not None
                    else ()
                ),
            ],
        )
        persisted_summary = (
            selected.status in _COMPLETED_WORKER_STATUSES or selected.recycled
        )
        _append_detail_fields(
            details,
            (
                ("Phase", selected.phase, phase_style(selected.phase)),
                ("Step", selected.step, neutral_style),
            ),
            width=width,
            compact=compact,
            key_style=key_style,
        )
        if selected.blocked_prerequisite_ids:
            _append_dashboard_line(
                details,
                [
                    ("Blocked by ", key_style),
                    (", ".join(selected.blocked_prerequisite_ids), failure_style),
                ],
            )
        if compact and selected.log_tail and not persisted_summary:
            _append_dashboard_line(
                details,
                [
                    ("Recent log ", key_style),
                    (selected.log_tail[-1], neutral_style),
                ],
            )
        enforcement_current = _worker_enforcement_memory(selected)
        enforcement_peak = _worker_enforcement_memory(selected, peak=True)
        enforcement_style = (
            failure_style
            if enforcement_current > state.memory_limit_bytes
            else warning_style
        )
        if compact and not persisted_summary:
            _append_detail_fields(
                details,
                (
                    (
                        "Guard current",
                        _human_bytes(enforcement_current),
                        enforcement_style,
                    ),
                    ("Guard peak", _human_bytes(enforcement_peak), enforcement_style),
                    ("RAM cap", _human_bytes(state.memory_limit_bytes), warning_style),
                    (
                        "Physical current",
                        (
                            "unavailable"
                            if selected.current_physical_footprint_bytes is None
                            else _human_bytes(selected.current_physical_footprint_bytes)
                        ),
                        (
                            muted_style
                            if selected.current_physical_footprint_bytes is None
                            else memory_style
                        ),
                    ),
                ),
                width=width,
                compact=compact,
                key_style=key_style,
            )
        if persisted_summary:
            if selected.recycled:
                _append_dashboard_line(
                    details,
                    [
                        (
                            "No work executed by this invocation",
                            success_style,
                        ),
                    ],
                )
                _append_dashboard_line(
                    details,
                    [
                        ("Reuse ", key_style),
                        (
                            selected.reuse_explanation or "existing current reused",
                            neutral_style,
                        ),
                    ],
                )
            _append_dashboard_line(
                details,
                [
                    ("Provenance ", key_style),
                    (selected.provenance_summary or "unavailable", neutral_style),
                ],
            )
            if selected.phase_timeline:
                timeline_phase_width = 38 if width >= 150 else 28
                _append_dashboard_line(
                    details,
                    [("Phase timeline", key_style)],
                )
                if compact:
                    _append_dashboard_line(
                        details,
                        [
                            ("  phase", table_header_style),
                            (" · wall · outcome", table_header_style),
                        ],
                    )
                else:
                    _append_dashboard_line(
                        details,
                        [
                            (
                                f"  {'phase':<{timeline_phase_width}}",
                                table_header_style,
                            ),
                            (f"{'wall':>12}  ", table_header_style),
                            (f"{'CPU':>12}  ", table_header_style),
                            (f"{'peak RAM':>12}  ", table_header_style),
                            ("outcome", table_header_style),
                        ],
                    )
                for timeline_row in selected.phase_timeline:
                    wall = (
                        "unavailable"
                        if timeline_row.wall_seconds is None
                        else _human_phase_duration(timeline_row.wall_seconds)
                    )
                    outcome = (
                        " · ".join(
                            value
                            for value in (timeline_row.status, timeline_row.detail)
                            if value
                        )
                        or "recorded"
                    )
                    is_summary = timeline_row.phase.casefold() == "worker supervision"
                    if compact:
                        row_style = summary_style if is_summary else neutral_style
                        _append_dashboard_line(
                            details,
                            [
                                ("Σ " if is_summary else "  ", row_style),
                                (timeline_row.phase, row_style),
                                (" · ", row_style),
                                (
                                    wall,
                                    summary_style
                                    if is_summary
                                    else (
                                        muted_style
                                        if timeline_row.wall_seconds is None
                                        else metric_style
                                    ),
                                ),
                                (" · ", row_style),
                                (
                                    outcome,
                                    summary_style
                                    if is_summary
                                    else outcome_style(timeline_row.status),
                                ),
                            ],
                        )
                        continue
                    cpu = (
                        "unavailable"
                        if timeline_row.cpu_seconds is None
                        else _human_phase_duration(timeline_row.cpu_seconds)
                    )
                    peak = (
                        "unavailable"
                        if timeline_row.peak_memory_bytes is None
                        else _human_bytes(timeline_row.peak_memory_bytes)
                    )
                    row_style = summary_style if is_summary else neutral_style
                    metric_row_style = summary_style if is_summary else metric_style
                    unavailable_row_style = summary_style if is_summary else muted_style
                    _append_dashboard_line(
                        details,
                        [
                            ("Σ " if is_summary else "  ", row_style),
                            (
                                _fit_dashboard_field(
                                    timeline_row.phase,
                                    timeline_phase_width,
                                ),
                                row_style,
                            ),
                            (
                                f"{wall:>12}  ",
                                unavailable_row_style
                                if timeline_row.wall_seconds is None
                                else metric_row_style,
                            ),
                            (
                                f"{cpu:>12}  ",
                                unavailable_row_style
                                if timeline_row.cpu_seconds is None
                                else (summary_style if is_summary else title_style),
                            ),
                            (
                                f"{peak:>12}  ",
                                unavailable_row_style
                                if timeline_row.peak_memory_bytes is None
                                else (summary_style if is_summary else memory_style),
                            ),
                            (
                                outcome,
                                summary_style
                                if is_summary
                                else outcome_style(timeline_row.status),
                            ),
                        ],
                    )
            else:
                _append_dashboard_line(
                    details,
                    [
                        ("Phase timeline ", key_style),
                        ("unavailable (older result)", warning_style),
                    ],
                )
        _append_detail_fields(
            details,
            (
                (
                    "Wall",
                    _human_phase_duration(selected.wall_seconds),
                    metric_style,
                ),
                (
                    "CPU",
                    (
                        "unavailable"
                        if selected.cpu_seconds is None
                        else _human_phase_duration(selected.cpu_seconds)
                    ),
                    muted_style if selected.cpu_seconds is None else metric_style,
                ),
            ),
            width=width,
            compact=compact,
            key_style=key_style,
        )
        _append_detail_fields(
            details,
            (
                ("RSS current", _human_bytes(selected.current_rss_bytes), memory_style),
                ("RSS peak", _human_bytes(selected.peak_rss_bytes), memory_style),
            ),
            width=width,
            compact=compact,
            key_style=key_style,
        )
        if selected.current_physical_footprint_bytes is not None and not compact:
            physical_peak = (
                selected.peak_physical_footprint_bytes
                if selected.peak_physical_footprint_bytes is not None
                else selected.current_physical_footprint_bytes
            )
            _append_detail_fields(
                details,
                (
                    (
                        "Physical current",
                        _human_bytes(selected.current_physical_footprint_bytes),
                        memory_style,
                    ),
                    ("Physical peak", _human_bytes(physical_peak), memory_style),
                ),
                width=width,
                compact=compact,
                key_style=key_style,
            )
        if not (compact and not persisted_summary):
            _append_detail_fields(
                details,
                (
                    (
                        "Guard current",
                        _human_bytes(enforcement_current),
                        enforcement_style,
                    ),
                    ("Guard peak", _human_bytes(enforcement_peak), enforcement_style),
                    ("RAM cap", _human_bytes(state.memory_limit_bytes), warning_style),
                    (
                        "Physical current" if compact else "Worker wall cap",
                        (
                            (
                                "unavailable"
                                if selected.current_physical_footprint_bytes is None
                                else _human_bytes(
                                    selected.current_physical_footprint_bytes
                                )
                            )
                            if compact
                            else (
                                "disabled"
                                if state.worker_wall_limit_seconds is None
                                else _human_duration(state.worker_wall_limit_seconds)
                            )
                        ),
                        (
                            muted_style
                            if (
                                (
                                    compact
                                    and selected.current_physical_footprint_bytes
                                    is None
                                )
                                or (
                                    not compact
                                    and state.worker_wall_limit_seconds is None
                                )
                            )
                            else (memory_style if compact else warning_style)
                        ),
                    ),
                ),
                width=width,
                compact=compact,
                key_style=key_style,
            )
        evaluator_total = selected.published_evaluator_total_seconds_per_point
        recurrence_core = selected.published_recurrence_core_seconds_per_point
        missing_published = (
            "pending"
            if selected.status in {"queued", "preparing", "running"}
            else (
                "not exposed"
                if selected.status in _COMPLETED_WORKER_STATUSES
                else "unavailable"
            )
        )
        missing_recurrence_core = (
            "not applicable"
            if selected.generation_engine is not None
            and selected.generation_engine != ExecutionMode.RECURRENCE.value
            else missing_published
        )
        outer_wall_value = (
            missing_published
            if selected.published_wall_seconds_per_point is None
            else f"{selected.published_wall_seconds_per_point * 1.0e6:.6g} μs/pt"
        )
        evaluator_total_value = (
            missing_published
            if evaluator_total is None
            else f"{evaluator_total * 1.0e6:.6g} μs/pt"
        )
        _append_detail_fields(
            details,
            (
                (
                    "Outer wall",
                    outer_wall_value,
                    (
                        muted_style
                        if selected.published_wall_seconds_per_point is None
                        else metric_style
                    ),
                ),
                (
                    "Evaluator total",
                    evaluator_total_value,
                    muted_style if evaluator_total is None else metric_style,
                ),
            ),
            width=width,
            compact=compact,
            key_style=key_style,
        )
        recurrence_value = (
            missing_recurrence_core
            if recurrence_core is None
            else f"{recurrence_core * 1.0e6:.6g} μs/pt"
        )
        _append_detail_fields(
            details,
            (
                (
                    "Recurrence core",
                    recurrence_value,
                    muted_style if recurrence_core is None else metric_style,
                ),
                ("Attribution", "independent narrow clock", muted_style),
            ),
            width=width,
            compact=compact,
            key_style=key_style,
        )
        if not compact:
            progress_details = _progress_detail_summary(selected.progress_details)
            if progress_details:
                _append_dashboard_line(
                    details,
                    [
                        ("Progress data ", key_style),
                        (progress_details, neutral_style),
                    ],
                )
            _append_detail_fields(
                details,
                (
                    (
                        "PID tree",
                        ", ".join(str(value) for value in selected.member_pids) or "—",
                        metric_style,
                    ),
                    ("Attempt", selected.attempt_id or "—", neutral_style),
                ),
                width=width,
                compact=compact,
                key_style=key_style,
            )
        if selected.log_tail and not compact and not persisted_summary:
            for index, line in enumerate(selected.log_tail[-3:]):
                _append_dashboard_line(
                    details,
                    [
                        ("Recent log " if index == 0 else "           ", key_style),
                        (line, neutral_style),
                    ],
                )
        if not compact:
            visible_events = _coalesced_worker_events(selected.events)
            if not persisted_summary:
                for event in visible_events[-max(1, detail_height - 6) :]:
                    _append_dashboard_line(
                        details,
                        [("• ", key_style), (event, neutral_style)],
                    )
            available_commands = _available_reproduction_stages(selected)
            if available_commands:
                _append_dashboard_line(
                    details,
                    [
                        ("Commands ", key_style),
                        *(
                            span
                            for stage in available_commands
                            for span in (
                                (
                                    f"{_REPRODUCTION_STAGES.index(stage) + 1} ",
                                    key_style,
                                ),
                                (stage, success_style),
                                ("   ", neutral_style),
                            )
                        ),
                        ("open full command · y copies there", muted_style),
                    ],
                )
            elif selected.generation_engine == ExecutionMode.AMPLICOL.value:
                _append_dashboard_line(
                    details,
                    [
                        ("Commands ", key_style),
                        (
                            "original AmpliCol uses the report adapter; "
                            "no public CLI reproduction is available",
                            muted_style,
                        ),
                    ],
                )
        ratio = _worker_progress_ratio(selected)
        gauge_label = (
            selected.progress_message or f"{selected.phase} {round(100.0 * ratio):d}%"
        )
    if selected is not None and state.command_stage is not None:
        stage = state.command_stage
        command = _reproduction_command(selected, stage)
        command_details = Paragraph.new_empty()
        command_details.set_block_title(
            f" Reproduction command · {stage} · {selected.cell_id} "
        )
        _append_dashboard_line(
            command_details,
            [
                ("Stage ", key_style),
                (stage, success_style if command is not None else warning_style),
                ("   1 prepare  2 generate  3 profile  Tab/←/→ cycle", muted_style),
            ],
        )
        _append_dashboard_line(
            command_details,
            [
                ("Copy ", key_style),
                ("y exact clipboard", success_style),
                ("   Print ", key_style),
                ("p normal terminal", metric_style),
                *(
                    ()
                    if compact
                    else (("   Scroll PgUp/PgDn   Close Esc", muted_style),)
                ),
            ],
        )
        if compact:
            _append_dashboard_line(
                command_details,
                [("Scroll PgUp/PgDn   Close Esc", muted_style)],
            )
        if state.command_notice:
            _append_dashboard_line(
                command_details,
                [
                    ("Status ", key_style),
                    (
                        state.command_notice,
                        warning_style
                        if "unavailable" in state.command_notice.casefold()
                        else success_style,
                    ),
                ],
            )
        if command is None:
            _append_dashboard_line(
                command_details,
                [
                    ("Unavailable ", warning_style),
                    (
                        "this worker does not expose that public CLI stage",
                        muted_style,
                    ),
                ],
            )
        else:
            wrapped = textwrap.wrap(
                command,
                width=max(24, width - 6),
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=True,
            ) or [command]
            header_lines = (4 if compact else 3) + bool(state.command_notice)
            command_capacity = max(1, detail_height - 4 - header_lines)
            command_start = min(
                max(0, state.command_scroll),
                max(0, len(wrapped) - command_capacity),
            )
            command_end = min(len(wrapped), command_start + command_capacity)
            _append_dashboard_line(
                command_details,
                [
                    ("Exact command ", key_style),
                    (
                        f"{len(command):,} characters · wrapped lines "
                        f"{command_start + 1}-{command_end}/{len(wrapped)}",
                        muted_style,
                    ),
                ],
            )
            for line in wrapped[command_start:command_end]:
                _append_dashboard_line(
                    command_details,
                    [("  ", key_style), (line, neutral_style)],
                )
        command_details.set_wrap(False)
        details = command_details
    else:
        details.set_scroll(state.detail_scroll)

    gauge = Gauge().ratio(ratio).label(gauge_label)
    gauge.set_styles(
        neutral_style,
        gauge_label_style,
        gauge_value_style,
    )
    if state.command_stage is not None:
        footer_text = (
            (
                " 1/2/3 stage  Tab cycle  Pg scroll  y copy"
                "  p print  Esc close  Ctrl-C stop "
            )
            if compact
            else (
                " 1/2/3 stage  Tab/←/→ cycle  PgUp/PgDn scroll"
                "  y copy  p print  Esc close  Ctrl-C stop "
            )
        )
    elif compact:
        footer_text = (
            " ↑/↓ select  d done  e errors  1/2/3 command  ? help  Ctrl-C stop safely "
        )
    else:
        footer_text = (
            " ↑/↓ select  d completed  e errors  1/2/3 command"
            "  PgUp/PgDn details  ? help  Ctrl-C stop safely "
        )
    footer = Paragraph.from_text(footer_text)

    return [
        DrawCmd.paragraph(overview, (0, 0, width, overview_height)),
        DrawCmd.paragraph(worker_table, (0, overview_height, width, table_height)),
        DrawCmd.paragraph(details, (0, detail_y, width, detail_height - 2)),
        DrawCmd.gauge(gauge, (0, height - 3, width, 2)),
        DrawCmd.paragraph(footer, (0, height - 1, width, footer_height)),
    ]


def render_dashboard_frame(
    state: DashboardState,
    *,
    width: int,
    height: int,
    cells: bool = False,
    color: bool = True,
) -> object:
    if width < 60 or height < 18:
        raise ManualCampaignError("dashboard dimensions must be at least 60x18")
    try:
        from ratatui_py import headless_render_frame, headless_render_frame_cells
    except ImportError as error:
        raise ManualCampaignError(
            "Ratatui is unavailable; run `just dev-install`"
        ) from error
    commands = _ratatui_commands(state, width=width, height=height, color=color)
    if cells:
        return headless_render_frame_cells(width, height, commands)
    return headless_render_frame(width, height, commands)


def _toggle_worker_filter(state: DashboardState, attribute: str) -> None:
    selected = state.selected_worker()
    selected_identity = (
        None if selected is None else (selected.cell_id, selected.peer_instance)
    )
    setattr(state, attribute, not bool(getattr(state, attribute)))
    rows = state.visible_workers()
    if not rows:
        state.selected_index = 0
    elif selected_identity is not None:
        state.selected_index = next(
            (
                index
                for index, worker in enumerate(rows)
                if (worker.cell_id, worker.peer_instance) == selected_identity
            ),
            min(state.selected_index, len(rows) - 1),
        )
    else:
        state.selected_index %= len(rows)
    state.detail_scroll = 0


def _handle_dashboard_key(
    state: DashboardState,
    event: Mapping[str, object],
    cancellation: threading.Event,
) -> bool:
    """Apply one Ratatui key event; return true when rendering should stop."""

    from ratatui import KeyCode, KeyMods

    if event.get("kind") != "key":
        return False
    code = event.get("code")
    ch = event.get("ch")
    mods = _optional_int(event.get("mods")) or 0
    if code == KeyCode.Char and ch in {ord("c"), ord("C")} and mods & int(KeyMods.CTRL):
        state.interrupted = True
        cancellation.set()
        return True
    if code == KeyCode.Esc and state.command_stage is not None:
        state.command_stage = None
        state.command_scroll = 0
        state.command_notice = None
        state.pending_clipboard = None
        state.pending_print = None
    elif code == KeyCode.Up or (code == KeyCode.Char and ch == ord("k")):
        state.selected_index -= 1
        state.detail_scroll = 0
        state.command_scroll = 0
        state.command_notice = None
    elif code == KeyCode.Down or (code == KeyCode.Char and ch == ord("j")):
        state.selected_index += 1
        state.detail_scroll = 0
        state.command_scroll = 0
        state.command_notice = None
    elif code == KeyCode.PageUp:
        if state.command_stage is None:
            state.detail_scroll = max(0, state.detail_scroll - 3)
        else:
            state.command_scroll = max(0, state.command_scroll - 3)
    elif code == KeyCode.PageDown:
        if state.command_stage is None:
            state.detail_scroll += 3
        else:
            state.command_scroll += 3
    elif code == KeyCode.Char and ch in {ord("1"), ord("2"), ord("3")}:
        assert isinstance(ch, int)
        state.command_stage = _REPRODUCTION_STAGES[ch - ord("1")]
        state.command_scroll = 0
        state.command_notice = None
        state.show_help = False
    elif code == KeyCode.Tab or code == KeyCode.Right:
        _cycle_reproduction_stage(state, 1)
        state.show_help = False
    elif code == KeyCode.Left:
        _cycle_reproduction_stage(state, -1)
        state.show_help = False
    elif code == KeyCode.Char and ch in {ord("y"), ord("Y")}:
        command = _reproduction_command(state.selected_worker(), state.command_stage)
        if command is None or state.command_stage is None:
            state.command_notice = "Selected command is unavailable"
        else:
            state.pending_clipboard = (state.command_stage, command)
    elif code == KeyCode.Char and ch in {ord("p"), ord("P")}:
        command = _reproduction_command(state.selected_worker(), state.command_stage)
        if command is None or state.command_stage is None:
            state.command_notice = "Selected command is unavailable"
        else:
            state.pending_print = (state.command_stage, command)
    elif code == KeyCode.Char and ch in {ord("d"), ord("D")}:
        _toggle_worker_filter(state, "show_completed")
    elif code == KeyCode.Char and ch in {ord("e"), ord("E")}:
        _toggle_worker_filter(state, "show_errors")
    elif code == KeyCode.Char and ch == ord("?"):
        state.show_help = not state.show_help
        state.command_stage = None
        state.detail_scroll = 0
    elif code == KeyCode.Esc:
        state.interrupted = True
        cancellation.set()
        return True
    return False


@contextmanager
def _ratatui_terminal_session() -> Iterator[Any]:
    """Open Ratatui without calling the package's unsafe mode wrappers.

    ``ratatui_py.terminal_session`` currently calls several native functions
    without the terminal handle required by the bundled FFI.  The native
    terminal constructor and destructor already own raw-mode and alternate-
    screen setup/cleanup, selected through environment variables, so use that
    supported lifecycle directly.
    """

    try:
        from ratatui_py import Terminal
    except ImportError as error:
        raise ManualCampaignError(
            "Ratatui is unavailable; run `just dev-install`"
        ) from error

    alt_was_set = "RATATUI_FFI_ALTSCR" in os.environ
    previous_alt = os.environ.get("RATATUI_FFI_ALTSCR")
    no_raw_was_set = "RATATUI_FFI_NO_RAW" in os.environ
    previous_no_raw = os.environ.get("RATATUI_FFI_NO_RAW")
    os.environ["RATATUI_FFI_ALTSCR"] = "1"
    os.environ.pop("RATATUI_FFI_NO_RAW", None)
    try:
        terminal = Terminal()
    finally:
        if alt_was_set:
            os.environ["RATATUI_FFI_ALTSCR"] = previous_alt or ""
        else:
            os.environ.pop("RATATUI_FFI_ALTSCR", None)
        if no_raw_was_set:
            os.environ["RATATUI_FFI_NO_RAW"] = previous_no_raw or ""
        else:
            os.environ.pop("RATATUI_FFI_NO_RAW", None)

    try:
        yield terminal
    finally:
        terminal.close()


def _run_live_dashboard(
    lease: LeaseManager,
    state: DashboardState,
    finished: threading.Event,
    cancellation: threading.Event,
    *,
    color: bool = True,
) -> None:
    while not finished.is_set():
        print_request: tuple[str, str] | None = None
        with _ratatui_terminal_session() as terminal:
            while not finished.is_set():
                display_state = lease.dashboard_snapshot()
                width, height = terminal.size()
                terminal.draw_frame(
                    _ratatui_commands(
                        display_state,
                        width=width,
                        height=height,
                        color=color,
                    )
                )
                event = terminal.next_event(timeout_ms=150)
                if event is None:
                    continue
                if _handle_dashboard_key(state, event, cancellation):
                    return
                if state.pending_clipboard is not None:
                    stage, command = state.pending_clipboard
                    state.pending_clipboard = None
                    try:
                        sys.stdout.write(_osc52_clipboard_sequence(command))
                        sys.stdout.flush()
                    except OSError:
                        state.command_notice = (
                            "Clipboard unavailable; press p to print the command"
                        )
                    else:
                        state.command_notice = (
                            f"Copied full {stage} command ({len(command):,} characters)"
                        )
                if state.pending_print is not None:
                    print_request = state.pending_print
                    state.pending_print = None
                    break
        if print_request is None:
            continue
        stage, command = print_request
        print(f"\nFull {stage} command ({len(command):,} characters):\n")
        print(command)
        print()
        with suppress(EOFError):
            input("Press Enter to return to the dashboard… ")
        state.command_notice = f"Returned from printed {stage} command"


def _manual_baseline(
    cell: CellSpec,
    *,
    catalog: ReportCatalog = REPORT_CATALOG,
) -> CellSpec | None:
    if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
        return None
    dataset_id = f"reference_amplicol_{cell.measurement.accuracy.value}"
    return next(
        (
            candidate
            for candidate in catalog.measurement_cells()
            if candidate.dataset_id == dataset_id
            and candidate.process_key == cell.process_key
            and candidate.n_final == cell.n_final
            and candidate.workload is cell.workload
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    cell: CellSpec
    baseline: CellSpec
    candidate: float
    reference: float

    @property
    def multiplier(self) -> float:
        return self.candidate / self.reference


def _number(measurement: Mapping[str, object], field: str) -> float | None:
    value = measurement.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        return None
    return float(value)


def _evaluator_total_number(measurement: Mapping[str, object]) -> float | None:
    record = evaluator_total_timing_record(measurement)
    return None if record is None else _number(record, "raw_seconds_per_point")


def _recurrence_core_number(measurement: Mapping[str, object]) -> float | None:
    return recurrence_core_seconds_per_point(measurement)


@dataclass(frozen=True, slots=True)
class AbsoluteMetric:
    cell: CellSpec
    value: float


def _absolute_metric_identity(item: AbsoluteMetric) -> dict[str, object]:
    cell = item.cell
    return {
        "table": cell.dataset_id,
        "cell_id": cell.cell_id,
        "process_id": cell.process_key,
        "process": cell.process,
        "multiplicity": cell.n_final,
        "color_approximation": cell.measurement.accuracy.value,
        "generation_mode": cell.workload.value,
        "generation_engine": cell.measurement.execution_mode.value,
        "backend": cell.measurement.backend,
        "model": (
            None if cell.measurement.model is None else cell.measurement.model.value
        ),
        "variant": cell.variant,
        "seconds_per_point": item.value,
        "microseconds_per_point": item.value * 1.0e6,
    }


def _inspect_recurrence_core(
    cells: Sequence[CellSpec],
    currents: Mapping[str, LightweightCurrent],
) -> tuple[dict[str, object], Counter[str]]:
    observations: list[AbsoluteMetric] = []
    exclusions: Counter[str] = Counter()
    for cell in cells:
        if cell.measurement.execution_mode is not ExecutionMode.RECURRENCE:
            exclusions["not_recurrence"] += 1
            continue
        current = currents.get(cell.cell_id)
        if current is None:
            exclusions["candidate_missing"] += 1
            continue
        if not current.reusable:
            exclusions[f"candidate_{current.reason.replace(' ', '_')}"] += 1
            continue
        if current.result.get("status") in {
            ResultStatus.TIMEOUT.value,
            ResultStatus.MEMORY_LIMIT.value,
        }:
            exclusions["candidate_resource_capped"] += 1
            continue
        value = _recurrence_core_number(current.result)
        if value is None:
            exclusions["value_unavailable"] += 1
            continue
        observations.append(AbsoluteMetric(cell=cell, value=value))
    ordered = sorted(observations, key=lambda item: item.value)
    values = [item.value for item in ordered]
    payload: dict[str, object] = {
        "count": len(ordered),
        "fastest": (None if not ordered else _absolute_metric_identity(ordered[0])),
        "slowest": (None if not ordered else _absolute_metric_identity(ordered[-1])),
        "median_seconds_per_point": (None if not values else statistics.median(values)),
        "mean_seconds_per_point": (None if not values else statistics.fmean(values)),
    }
    return payload, exclusions


def comparison_statistics(comparisons: Sequence[Comparison]) -> dict[str, object]:
    if not comparisons:
        return {
            "count": 0,
            "best": None,
            "worst": None,
            "median": None,
            "mean": None,
            "weighted_mean": None,
        }
    ordered = sorted(comparisons, key=lambda item: item.multiplier)
    ratios = [item.multiplier for item in ordered]
    return {
        "count": len(ordered),
        "best": ordered[0],
        "worst": ordered[-1],
        "median": statistics.median(ratios),
        "mean": statistics.fmean(ratios),
        "weighted_mean": (
            math.fsum(item.candidate for item in ordered)
            / math.fsum(item.reference for item in ordered)
        ),
    }


def _comparison_identity(item: Comparison) -> dict[str, object]:
    cell = item.cell
    return {
        "table": cell.dataset_id,
        "cell_id": cell.cell_id,
        "process_id": cell.process_key,
        "process": cell.process,
        "multiplicity": cell.n_final,
        "color_approximation": cell.measurement.accuracy.value,
        "generation_mode": cell.workload.value,
        "generation_engine": cell.measurement.execution_mode.value,
        "backend": cell.measurement.backend,
        "model": (
            None if cell.measurement.model is None else cell.measurement.model.value
        ),
        "variant": cell.variant,
        "candidate": item.candidate,
        "amplicol": item.reference,
        "multiplier": item.multiplier,
        "amplicol_cell_id": item.baseline.cell_id,
    }


def _inspect_metric(
    cells: Sequence[CellSpec],
    currents: Mapping[str, LightweightCurrent],
    *,
    field: str | None,
    generation: bool,
    evaluator_total: bool = False,
) -> tuple[dict[str, object], Counter[str]]:
    comparisons: list[Comparison] = []
    exclusions: Counter[str] = Counter()
    for cell in cells:
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL:
            exclusions["amplicol_candidate"] += 1
            continue
        baseline = _manual_baseline(cell)
        if baseline is None:
            exclusions["no_amplicol_baseline"] += 1
            continue
        if generation and cell.workload is Workload.ALL_FLOW:
            exclusions["incompatible_generation_layout"] += 1
            continue
        candidate_current = currents.get(cell.cell_id)
        baseline_current = currents.get(baseline.cell_id)
        if candidate_current is None:
            exclusions["candidate_missing"] += 1
            continue
        if not candidate_current.reusable:
            exclusions[f"candidate_{candidate_current.reason.replace(' ', '_')}"] += 1
            continue
        if baseline_current is None:
            exclusions["baseline_missing"] += 1
            continue
        if not baseline_current.reusable:
            exclusions[f"baseline_{baseline_current.reason.replace(' ', '_')}"] += 1
            continue
        if candidate_current.result.get("status") in {
            ResultStatus.TIMEOUT.value,
            ResultStatus.MEMORY_LIMIT.value,
        }:
            exclusions["candidate_resource_capped"] += 1
            continue
        if baseline_current.result.get("status") in {
            ResultStatus.TIMEOUT.value,
            ResultStatus.MEMORY_LIMIT.value,
        }:
            exclusions["baseline_resource_capped"] += 1
            continue
        if evaluator_total:
            candidate = _evaluator_total_number(candidate_current.result)
            reference = _evaluator_total_number(baseline_current.result)
        else:
            assert field is not None
            candidate = _number(candidate_current.result, field)
            reference = _number(baseline_current.result, field)
        if candidate is None or reference is None:
            exclusions["value_unavailable"] += 1
            continue
        comparisons.append(Comparison(cell, baseline, candidate, reference))
    raw = comparison_statistics(comparisons)
    payload = {
        key: (_comparison_identity(value) if isinstance(value, Comparison) else value)
        for key, value in raw.items()
    }
    return payload, exclusions


def _inspect_payload(
    service: ReportService,
    cells: Sequence[CellSpec],
    *,
    source_revision: str,
    renderer_revision: str,
    accept_historical_source: bool = False,
) -> dict[str, object]:
    required = {cell.cell_id: cell for cell in cells}
    for cell in cells:
        baseline = _manual_baseline(cell)
        if baseline is not None:
            required[baseline.cell_id] = baseline
    currents = lightweight_currents(
        service,
        tuple(required.values()),
        source_revision=source_revision,
        accept_historical_source=accept_historical_source,
    )
    outcomes = lightweight_presentation_outcomes(
        service,
        tuple(required.values()),
        source_revision=source_revision,
        accept_historical_source=accept_historical_source,
    )
    statuses: Counter[str] = Counter()
    unverified_diagnostics: list[dict[str, object]] = []
    for cell in cells:
        if _manual_static_na_reason(cell) is not None:
            statuses["static_not_available"] += 1
            continue
        current = currents.get(cell.cell_id)
        outcome = outcomes.get(cell.cell_id)
        if current is None:
            if outcome is None or outcome.successful:
                statuses[ResultStatus.NOT_AVAILABLE.value] += 1
            else:
                statuses[outcome.status] += 1
                if outcome.status == ResultStatus.UNVERIFIED.value:
                    diagnostic = _presentation_measurement(
                        service,
                        cell,
                        outcome,
                    )
                    if diagnostic.get("status") == ResultStatus.UNVERIFIED.value:
                        provenance = diagnostic.get("provenance")
                        evaluator = (
                            provenance.get("evaluator_total_timing")
                            if isinstance(provenance, Mapping)
                            else None
                        )
                        unverified_diagnostics.append(
                            {
                                "cell_id": cell.cell_id,
                                "attempt_id": outcome.attempt_id,
                                "generation_seconds": diagnostic.get(
                                    "generation_seconds"
                                ),
                                "wall_seconds_per_point": diagnostic.get(
                                    "wall_seconds_per_point"
                                ),
                                "evaluator_total_seconds_per_point": (
                                    evaluator.get("raw_seconds_per_point")
                                    if isinstance(evaluator, Mapping)
                                    else None
                                ),
                                "reason": (
                                    diagnostic.get("failure", {}).get("message")
                                    if isinstance(diagnostic.get("failure"), Mapping)
                                    else outcome.label
                                ),
                            }
                        )
        elif not current.reusable:
            statuses[current.reason.replace(" ", "_")] += 1
        else:
            statuses[str(current.result.get("status"))] += 1
    generation, generation_exclusions = _inspect_metric(
        cells,
        currents,
        field="generation_seconds",
        generation=True,
    )
    runtime, runtime_exclusions = _inspect_metric(
        cells,
        currents,
        field="wall_seconds_per_point",
        generation=False,
    )
    evaluator_total, evaluator_total_exclusions = _inspect_metric(
        cells,
        currents,
        field=None,
        generation=False,
        evaluator_total=True,
    )
    recurrence_core, recurrence_core_exclusions = _inspect_recurrence_core(
        cells,
        currents,
    )
    selected_currents = tuple(
        currents[cell.cell_id]
        for cell in cells
        if cell.cell_id in currents and currents[cell.cell_id].reusable
    )
    clock_coverage = {
        "outer_wall_available": sum(
            _number(current.result, "wall_seconds_per_point") is not None
            for current in selected_currents
        ),
        "evaluator_total_available": sum(
            _evaluator_total_number(current.result) is not None
            for current in selected_currents
        ),
        "recurrence_core_available": sum(
            _recurrence_core_number(current.result) is not None
            for current in selected_currents
        ),
    }
    return {
        "profile": _ACTIVE_PROFILE,
        "source_revision": source_revision,
        "renderer_source_revision": renderer_revision,
        "source_policy": (
            "continue_across_revisions"
            if accept_historical_source
            else "strict_same_source"
        ),
        "source_cohorts": _source_cohort_counts(currents),
        "selection_count": len(cells),
        "reusable_count": sum(
            bool(current.reusable)
            for cell_id, current in currents.items()
            if cell_id in {cell.cell_id for cell in cells}
        ),
        "statuses": dict(sorted(statuses.items())),
        "unverified_diagnostics": sorted(
            unverified_diagnostics,
            key=lambda item: str(item["cell_id"]),
        ),
        "generation_multiplier_vs_amplicol": generation,
        "runtime_wall_multiplier_vs_amplicol": runtime,
        "runtime_evaluator_total_multiplier_vs_amplicol": evaluator_total,
        "recurrence_core_absolute": recurrence_core,
        "clock_coverage": clock_coverage,
        "exclusions": {
            "generation": dict(sorted(generation_exclusions.items())),
            "runtime_wall": dict(sorted(runtime_exclusions.items())),
            "runtime_evaluator_total": dict(sorted(evaluator_total_exclusions.items())),
            "recurrence_core": dict(sorted(recurrence_core_exclusions.items())),
        },
        "clock_note": (
            "Outer wall, evaluator-total, and recurrence core are independent "
            "raw clocks; unavailable values are never copied or derived."
        ),
    }


def _format_multiplier(value: object) -> str:
    return "—" if value is None else f"{float(value):.4g}x"


def _paint_multiplier(palette: Palette, value: object) -> str:
    rendered = _format_multiplier(value)
    if value is None:
        return palette.neutral(rendered)
    multiplier = float(value)
    if math.isclose(multiplier, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return palette.neutral(rendered)
    return palette.success(rendered) if multiplier < 1.0 else palette.failure(rendered)


def _format_seconds_per_point(value: object) -> str:
    return "—" if value is None else f"{float(value) * 1.0e6:.6g} μs/pt"


def _print_inspect(payload: Mapping[str, object], palette: Palette) -> None:
    source_revision = str(payload["source_revision"])
    renderer_revision = str(payload["renderer_source_revision"])
    source_cohorts = payload["source_cohorts"]
    if not isinstance(source_cohorts, Mapping):
        raise ManualCampaignError("inspect source cohort summary is malformed")
    rendered_source_cohorts = _format_source_cohorts(
        {str(revision): int(count) for revision, count in source_cohorts.items()}
    )
    identity_rows: list[tuple[object, object]] = [
        (palette.key("profile"), payload["profile"]),
        (palette.key("source policy"), payload["source_policy"]),
        (palette.key("active measurement build"), source_revision),
        (
            palette.key("measurement source cohorts"),
            rendered_source_cohorts,
        ),
    ]
    if renderer_revision != source_revision:
        identity_rows.append((palette.key("renderer checkout"), renderer_revision[:12]))
    identity_rows.extend(
        (
            (palette.key("selected"), payload["selection_count"]),
            (palette.key("reusable"), palette.success(payload["reusable_count"])),
            (
                palette.key("clock coverage"),
                ", ".join(
                    f"{key}={value}" for key, value in payload["clock_coverage"].items()
                ),
            ),
            (palette.key("clock semantics"), payload["clock_note"]),
        )
    )
    print(
        _table(
            ("key", "value"),
            identity_rows,
        )
    )
    statuses = payload["statuses"]
    assert isinstance(statuses, Mapping)
    print()
    print(
        _table(
            ("status", "entries"),
            (
                (
                    palette.key(status),
                    (
                        palette.success(count)
                        if status == ResultStatus.OK.value
                        else (
                            palette.failure(count)
                            if any(
                                token in status
                                for token in ("error", "failed", "mismatch")
                            )
                            else palette.warning(count)
                        )
                    ),
                )
                for status, count in statuses.items()
            ),
            align={"entries": "r"},
        )
    )
    unverified = payload.get("unverified_diagnostics")
    if isinstance(unverified, list) and unverified:
        print()
        print(
            _table(
                (
                    "unverified cell",
                    "generation",
                    "wall",
                    "evaluator total",
                    "reason",
                ),
                (
                    (
                        item.get("cell_id"),
                        (
                            "—"
                            if item.get("generation_seconds") is None
                            else f"{float(item['generation_seconds']):.6g} s"
                        ),
                        _format_seconds_per_point(
                            item.get("wall_seconds_per_point")
                        ),
                        _format_seconds_per_point(
                            item.get("evaluator_total_seconds_per_point")
                        ),
                        item.get("reason"),
                    )
                    for item in unverified
                    if isinstance(item, Mapping)
                ),
            )
        )
    rows: list[tuple[object, ...]] = []
    for title, key in (
        ("Generation", "generation_multiplier_vs_amplicol"),
        ("Runtime wall", "runtime_wall_multiplier_vs_amplicol"),
        (
            "Runtime evaluator total",
            "runtime_evaluator_total_multiplier_vs_amplicol",
        ),
    ):
        summary = payload[key]
        assert isinstance(summary, Mapping)
        best = summary.get("best")
        worst = summary.get("worst")
        rows.append(
            (
                palette.key(title),
                summary.get("count"),
                _paint_multiplier(
                    palette,
                    (None if not isinstance(best, Mapping) else best.get("multiplier")),
                ),
                _paint_multiplier(palette, summary.get("median")),
                _paint_multiplier(palette, summary.get("mean")),
                _paint_multiplier(palette, summary.get("weighted_mean")),
                _paint_multiplier(
                    palette,
                    (
                        None
                        if not isinstance(worst, Mapping)
                        else worst.get("multiplier")
                    ),
                ),
            )
        )
    print()
    print(
        _table(
            (
                "metric",
                "pairs",
                "best",
                "median",
                "average",
                "weighted avg",
                "worst",
            ),
            rows,
            align={"pairs": "r"},
        )
    )
    core_summary = payload["recurrence_core_absolute"]
    assert isinstance(core_summary, Mapping)
    fastest = core_summary.get("fastest")
    slowest = core_summary.get("slowest")
    print()
    print(
        _table(
            ("metric", "samples", "fastest", "median", "average", "slowest"),
            (
                (
                    palette.key("Recurrence core (absolute)"),
                    core_summary.get("count"),
                    _format_seconds_per_point(
                        fastest.get("seconds_per_point")
                        if isinstance(fastest, Mapping)
                        else None
                    ),
                    _format_seconds_per_point(
                        core_summary.get("median_seconds_per_point")
                    ),
                    _format_seconds_per_point(
                        core_summary.get("mean_seconds_per_point")
                    ),
                    _format_seconds_per_point(
                        slowest.get("seconds_per_point")
                        if isinstance(slowest, Mapping)
                        else None
                    ),
                ),
            ),
            align={"samples": "r"},
        )
    )
    identities: list[tuple[object, ...]] = []
    for title, key in (
        ("generation best", "generation_multiplier_vs_amplicol"),
        ("generation worst", "generation_multiplier_vs_amplicol"),
        ("runtime best", "runtime_wall_multiplier_vs_amplicol"),
        ("runtime worst", "runtime_wall_multiplier_vs_amplicol"),
        (
            "evaluator-total best",
            "runtime_evaluator_total_multiplier_vs_amplicol",
        ),
        (
            "evaluator-total worst",
            "runtime_evaluator_total_multiplier_vs_amplicol",
        ),
    ):
        summary = payload[key]
        assert isinstance(summary, Mapping)
        identity = summary.get("best" if title.endswith("best") else "worst")
        if not isinstance(identity, Mapping):
            continue
        for label, identity_field in (
            ("table", "table"),
            ("cell", "cell_id"),
            ("process ID", "process_id"),
            ("concrete process", "process"),
            ("multiplicity", "multiplicity"),
            ("colour", "color_approximation"),
            ("layout/workload", "generation_mode"),
            ("engine", "generation_engine"),
            ("backend", "backend"),
            ("model", "model"),
            ("variant", "variant"),
            ("candidate value", "candidate"),
            ("AmpliCol baseline", "amplicol"),
            ("multiplier", "multiplier"),
        ):
            value = identity.get(identity_field)
            if identity_field == "multiplier":
                value = _paint_multiplier(palette, value)
            identities.append((palette.key(title), palette.key(label), value))
    for title, identity in (
        ("recurrence-core fastest", fastest),
        ("recurrence-core slowest", slowest),
    ):
        if not isinstance(identity, Mapping):
            continue
        for label, identity_field in (
            ("table", "table"),
            ("cell", "cell_id"),
            ("process ID", "process_id"),
            ("concrete process", "process"),
            ("multiplicity", "multiplicity"),
            ("colour", "color_approximation"),
            ("layout/workload", "generation_mode"),
            ("engine", "generation_engine"),
            ("backend", "backend"),
            ("model", "model"),
            ("variant", "variant"),
            ("absolute value", "seconds_per_point"),
        ):
            value = identity.get(identity_field)
            if identity_field == "seconds_per_point":
                value = _format_seconds_per_point(value)
            identities.append((palette.key(title), palette.key(label), value))
    if identities:
        print()
        print(
            _table(
                ("extreme", "key", "value"),
                identities,
            )
        )
    exclusions = payload["exclusions"]
    assert isinstance(exclusions, Mapping)
    print()
    print(
        _table(
            ("metric", "excluded reason", "count"),
            (
                (palette.key(metric), reason, count)
                for metric, raw in exclusions.items()
                if isinstance(raw, Mapping)
                for reason, count in raw.items()
            ),
            align={"count": "r"},
        )
    )


def _campaign_settings(
    arguments: argparse.Namespace,
    source: ReportSourceIdentity,
    *,
    original_amplicol_available: bool = False,
    observer: Any = None,
    cancelled: Any = None,
    campaign_invocation_id: str | None = None,
) -> CampaignSettings:
    workers = int(arguments.workers)
    cores = int(arguments.cores_per_worker)
    amplicol_build_jobs = int(getattr(arguments, "amplicol_build_jobs", 1))
    available = os.cpu_count() or 1
    effective_worker_cores = max(cores, amplicol_build_jobs)
    if workers * effective_worker_cores > available and not bool(
        arguments.allow_oversubscription
    ):
        raise ManualCampaignError(
            f"{workers} workers x {effective_worker_cores} maximum per-worker "
            f"cores exceeds {available} logical CPUs; lower --workers, "
            "--cores-per-worker, or --amplicol-build-jobs, or pass "
            "--allow-oversubscription"
        )
    fresh_attempt = _fresh_attempt_requested(arguments)
    original_amplicol_revision = getattr(
        arguments,
        "original_amplicol_revision",
        None,
    )
    return CampaignSettings(
        workers=workers,
        cell_cores=cores,
        amplicol_build_jobs=amplicol_build_jobs,
        target_runtime_seconds=float(arguments.target_measurement_duration),
        batch_size=int(arguments.batch_size),
        warmup_runs=int(arguments.warmups),
        minimum_samples=int(arguments.minimum_samples),
        timeout_seconds=arguments.worker_wall_limit,
        generation_time_limit_seconds=float(arguments.generation_time_limit),
        profiling_time_limit_seconds=(
            None
            if arguments.worker_wall_limit is None
            else float(arguments.worker_wall_limit)
        ),
        validation_time_limit_seconds=(
            None
            if arguments.worker_wall_limit is None
            else float(arguments.worker_wall_limit)
        ),
        max_rss_bytes=int(arguments.ram_limit),
        campaign_max_rss_bytes=(
            None
            if getattr(arguments, "campaign_ram_limit", None) is None
            else int(arguments.campaign_ram_limit)
        ),
        attempt_output_limit_bytes=int(
            getattr(
                arguments,
                "attempt_output_limit",
                DEFAULT_ATTEMPT_OUTPUT_LIMIT_BYTES,
            )
        ),
        minimum_free_disk_bytes=int(
            getattr(
                arguments,
                "minimum_free_disk",
                DEFAULT_MINIMUM_FREE_DISK_BYTES,
            )
        ),
        artifact_policy=(
            ArtifactPolicy.REGENERATE
            if fresh_attempt
            else ArtifactPolicy.REUSE
        ),
        missing_only=not fresh_attempt,
        rerun=fresh_attempt,
        add_optional_dependencies=not bool(
            getattr(arguments, "no_dependencies_added", False)
        ),
        original_amplicol_available=original_amplicol_available,
        allow_symbolica_parallel=bool(arguments.allow_engine_parallelism),
        resource_sample_interval_seconds=float(arguments.resource_sample_interval),
        termination_grace_seconds=float(arguments.termination_grace),
        source_identity_override=source,
        campaign_invocation_id=campaign_invocation_id,
        progress_observer=observer,
        cancellation_requested=cancelled,
        manual_terminal_censors=True,
        discard_cancelled_attempts=False,
        remove_heavy_attempt_artifacts=_artifact_cleanup_enabled(arguments),
        retain_workspaces=bool(getattr(arguments, "retain_workspaces", False)),
        report_profile=None,
        original_amplicol_repository=(
            arguments.original_amplicol
            if original_amplicol_revision is not None
            else None
        ),
        original_amplicol_revision=original_amplicol_revision,
        madgraph_installation=getattr(arguments, "madgraph", None),
        multiplicity_wave_barrier=bool(getattr(arguments, "fail_fast", False)),
    )


def _validated_original_amplicol_checkout(path: Path) -> tuple[Path, str]:
    """Return one clean checkout exposing the maintained profiling interface."""

    repository = path.expanduser().resolve(strict=True)
    missing = tuple(
        name
        for name in _ORIGINAL_AMPLICOL_REQUIRED_FILES
        if not (repository / name).is_file() or (repository / name).is_symlink()
    )
    if missing:
        raise ManualCampaignError(
            "--original-amplicol is not a complete patched checkout; missing "
            + ", ".join(missing)
        )
    makefile = (repository / "makefile").read_text(
        encoding="utf-8",
        errors="replace",
    )
    missing_targets = tuple(
        target
        for target in _ORIGINAL_AMPLICOL_REQUIRED_MAKE_TARGETS
        if re.search(rf"(?m)^{re.escape(target)}\s*:", makefile) is None
    )
    if missing_targets:
        raise ManualCampaignError(
            "--original-amplicol lacks required Make targets: "
            + ", ".join(missing_targets)
        )
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = head.stdout.strip()
    if head.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ManualCampaignError(
            "--original-amplicol must be a Git checkout with a concrete HEAD"
        )
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0 or status.stdout:
        raise ManualCampaignError(
            "--original-amplicol must be a clean checkout; commit, remove, "
            "or relocate local changes and build products first"
        )
    return repository, revision


def _configured_original_amplicol(docs_dir: Path) -> Path | None:
    config = docs_dir / _LOCAL_AMPLICOL_CONFIG
    if not config.exists():
        return None
    if config.is_symlink() or not config.is_file():
        raise ManualCampaignError(
            f"campaign local-AmpliCol configuration is not a regular file: {config}"
        )
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ManualCampaignError(
            f"cannot read campaign local-AmpliCol configuration: {config}"
        ) from error
    if len(lines) != 1 or not lines[0] or not Path(lines[0]).is_absolute():
        raise ManualCampaignError(
            f"campaign local-AmpliCol configuration is invalid: {config}"
        )
    return Path(lines[0])


def _configured_madgraph(docs_dir: Path) -> Path | None:
    config = docs_dir / _LOCAL_MADGRAPH_CONFIG
    if not config.exists():
        return None
    if config.is_symlink() or not config.is_file():
        raise ManualCampaignError(
            f"campaign MadGraph configuration is not a regular file: {config}"
        )
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ManualCampaignError(
            f"cannot read campaign MadGraph configuration: {config}"
        ) from error
    if len(lines) != 1 or not lines[0] or not Path(lines[0]).is_absolute():
        raise ManualCampaignError(
            f"campaign MadGraph configuration is invalid: {config}"
        )
    return Path(lines[0])


def _validated_madgraph_installation(path: Path) -> Path:
    try:
        installation = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ManualCampaignError(
            f"--madgraph installation does not exist: {path}"
        ) from error
    if not installation.is_dir():
        raise ManualCampaignError(
            f"--madgraph must name an installation directory: {installation}"
        )
    executable = installation / "bin/mg5_aMC"
    try:
        executable_mode = executable.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise ManualCampaignError(
            "--madgraph must contain a regular executable bin/mg5_aMC"
        ) from error
    if not stat.S_ISREG(executable_mode) or not os.access(executable, os.X_OK):
        raise ManualCampaignError(
            "--madgraph must contain a regular executable bin/mg5_aMC"
        )
    return installation


def _resolve_madgraph(
    arguments: argparse.Namespace,
    *,
    installed: bool,
    docs_dir: Path,
) -> Path | None:
    installation = arguments.madgraph
    if installation is None and installed:
        installation = _configured_madgraph(docs_dir)
    if installation is None:
        return None
    return _validated_madgraph_installation(installation)


def _original_amplicol_available_for_planning(
    arguments: argparse.Namespace,
    *,
    installed: bool,
    root: Path,
    docs_dir: Path,
) -> bool:
    """Whether optional validation dependencies may include original AmpliCol."""

    if bool(getattr(arguments, "no_dependencies_added", False)):
        return False
    if arguments.original_amplicol is not None:
        return True
    if not installed:
        return (root / "dependencies/checkouts/legacy-amplicol").is_dir()
    try:
        configured = _configured_original_amplicol(docs_dir)
    except ManualCampaignError:
        return False
    return configured is not None and configured.is_dir()


def _resolve_original_amplicol(
    arguments: argparse.Namespace,
    *,
    installed: bool,
    root: Path,
    docs_dir: Path,
) -> tuple[Path | None, str | None]:
    original = arguments.original_amplicol
    if original is None:
        if installed:
            original = _configured_original_amplicol(docs_dir)
        else:
            default_original = root / "dependencies/checkouts/legacy-amplicol"
            if default_original.is_dir():
                original = default_original
    if original is None:
        return None, None
    return _validated_original_amplicol_checkout(original)


def _bind_original_amplicol_if_required(
    arguments: argparse.Namespace,
    planned: Sequence[PlannedCell],
    *,
    installed: bool,
    root: Path,
    docs_dir: Path,
) -> None:
    """Validate the optional legacy checkout only for work that will use it."""

    if not any(
        item.cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        for item in planned
    ):
        arguments.original_amplicol = None
        arguments.original_amplicol_revision = None
        return
    repository, revision = _resolve_original_amplicol(
        arguments,
        installed=installed,
        root=root,
        docs_dir=docs_dir,
    )
    if repository is None:
        raise ManualCampaignError(
            "this campaign selection requires original AmpliCol; pass "
            "--original-amplicol PATH_TO_COMPLETE_CHECKOUT containing the "
            "profiling interface from AmpliCol PR #12"
        )
    arguments.original_amplicol = repository
    arguments.original_amplicol_revision = revision


def _dry_run_rows(
    direct: Sequence[CellSpec],
    planned: Sequence[PlannedCell],
    *,
    repo_root: Path,
    artifact_root: Path,
    arguments: argparse.Namespace,
) -> Iterable[tuple[object, ...]]:
    direct_ids = {cell.cell_id for cell in direct}
    for item in planned:
        recipe = reproduction_recipe(
            item.cell,
            repo_root=repo_root,
            artifact_root=artifact_root,
            cores=arguments.cores_per_worker,
            target_runtime=arguments.target_measurement_duration,
            batch_size=arguments.batch_size,
            warmups=arguments.warmups,
            minimum_samples=arguments.minimum_samples,
        )
        role = "direct" if item.cell.cell_id in direct_ids else "dependency"
        yield (
            f"{role} · rank {item.rank}",
            "\n".join(
                (
                    item.cell.cell_id,
                    f"engine: {item.cell.measurement.execution_mode.value}",
                    f"kind: {recipe.kind}",
                    f"exact: {'yes' if recipe.exact else 'no'}",
                    f"note: {recipe.note}",
                )
            ),
        )


def _wrapped_shell(
    arguments: Sequence[str],
    *,
    width: int,
    indent: str = "  ",
) -> str:
    """Wrap shell words without splitting quoted arguments."""

    words = [shlex.quote(str(argument)) for argument in arguments]
    lines: list[str] = []
    current = indent
    for word in words:
        separator = "" if current == indent else " "
        if current != indent and len(current) + len(separator) + len(word) > width:
            lines.append(current)
            current = indent + word
        else:
            current += separator + word
    if current != indent:
        lines.append(current)
    return " \\\n".join(lines)


def _dry_run_recipe_blocks(
    direct: Sequence[CellSpec],
    *,
    repo_root: Path,
    artifact_root: Path,
    arguments: argparse.Namespace,
    width: int,
) -> Iterable[str]:
    for index, cell in enumerate(direct, start=1):
        if _manual_static_na_reason(cell) is not None:
            continue
        recipe = reproduction_recipe(
            cell,
            repo_root=repo_root,
            artifact_root=artifact_root,
            cores=arguments.cores_per_worker,
            target_runtime=arguments.target_measurement_duration,
            batch_size=arguments.batch_size,
            warmups=arguments.warmups,
            minimum_samples=arguments.minimum_samples,
        )
        metadata = _table(
            ("key", "value"),
            (
                ("recipe", index),
                ("cell", cell.cell_id),
                ("kind", recipe.kind),
                ("exact", "yes" if recipe.exact else "no"),
                ("note", recipe.note),
            ),
            max_width={"value": max(36, width - 16)},
        )
        commands: list[str] = [metadata]
        for label, command in (
            ("Prepare prerequisite", recipe.prepare),
            ("Generate", recipe.generate),
            (
                "Profile" if recipe.exact else "Profile template/diagnostic",
                recipe.profile,
            ),
        ):
            if command is None:
                continue
            commands.extend((f"{label}:", _wrapped_shell(command, width=width)))
        yield "\n".join(commands)


def _cancel_and_join_campaign_worker(
    worker: threading.Thread,
    cancellation: threading.Event,
    *,
    timeout_seconds: float,
) -> tuple[bool, bool]:
    """Poll cancellation until the non-daemon controller has fully exited."""

    cancellation.set()
    interrupted = False
    poll_seconds = min(max(timeout_seconds, 0.05), 0.25)
    notice_polls = max(1, math.ceil(max(timeout_seconds, 1.0) / poll_seconds))
    polls = 0
    while worker.is_alive():
        try:
            worker.join(timeout=poll_seconds)
        except KeyboardInterrupt:
            interrupted = True
            cancellation.set()
            print(
                "Interrupt repeated; still waiting for supervised worker-tree "
                "cleanup...",
                file=sys.stderr,
                flush=True,
            )
        polls += 1
        if worker.is_alive() and polls % notice_polls == 0:
            cancellation.set()
            print(
                "Waiting for campaign cancellation cleanup; no leases or claims "
                "will be released early...",
                file=sys.stderr,
                flush=True,
            )
    return interrupted, False


def _recycled_attention_workers(
    cells: Sequence[CellSpec],
    currents: Mapping[str, LightweightCurrent],
    recycled_ids: set[str],
    *,
    repo_root: Path,
    artifact_root: Path,
    settings: ReproductionSettings,
) -> dict[str, WorkerView]:
    """Expose same-source recycled currents without scanning attempt history."""

    workers: dict[str, WorkerView] = {}
    for cell in cells:
        if cell.cell_id not in recycled_ids:
            continue
        current = currents.get(cell.cell_id)
        if current is None:
            continue
        result_status = str(current.result.get("status") or "")
        state = policy_measurement_state_hint(current.result)
        status = (
            "recycled"
            if result_status in {ResultStatus.OK.value, "reused", "skipped-current"}
            else (None if state is None else state.value)
        )
        if status is None:
            continue
        resources = current.result.get("resources")
        resource_values = resources if isinstance(resources, Mapping) else {}
        provenance = current.result.get("provenance")
        manual = (
            provenance.get("manual_campaign")
            if isinstance(provenance, Mapping)
            else None
        )
        stored_recipe = (
            manual.get("public_cli_reproduction")
            if isinstance(manual, Mapping)
            else None
        )
        recipe = None
        if not isinstance(stored_recipe, Mapping):
            recipe = reproduction_recipe(
                cell,
                repo_root=repo_root,
                artifact_root=artifact_root,
                cores=settings.cores,
                target_runtime=settings.target_runtime,
                batch_size=settings.batch_size,
                warmups=settings.warmups,
                minimum_samples=settings.minimum_samples,
                measurement=current.result,
            )

        worker = WorkerView(
            cell.cell_id,
            generation_engine=cell.measurement.execution_mode.value,
            status=status,
            step=(
                "recycled generation-time cap"
                if status == "generation_limit"
                else (
                    "recycled process-tree RAM cap"
                    if status == "memory_limit"
                    else (
                        f"recycled {status.replace('_', ' ')}"
                        if status in _TERMINAL_CAP_STATUSES
                        else "reused existing current"
                    )
                )
            ),
            phase="recycled current",
            attempt_id=current.attempt_id,
            wall_seconds=_finite_float(resource_values.get("wall_seconds")),
            cpu_seconds=_optional_finite_float(resource_values.get("cpu_seconds")),
            current_rss_bytes=_optional_int(resource_values.get("current_rss_bytes"))
            or 0,
            peak_rss_bytes=_optional_int(resource_values.get("peak_rss_bytes")) or 0,
            current_physical_footprint_bytes=_optional_int(
                resource_values.get("current_physical_footprint_bytes")
            ),
            peak_physical_footprint_bytes=_optional_int(
                resource_values.get("peak_physical_footprint_bytes")
            ),
            current_guard_bytes=_optional_int(
                resource_values.get("current_guard_bytes")
            )
            or 0,
            peak_guard_bytes=_optional_int(resource_values.get("peak_guard_bytes"))
            or 0,
            progress_completed=1,
            progress_total=1,
            progress_message=(
                "recycled capped result"
                if status in _TERMINAL_CAP_STATUSES
                else "recycled current"
            ),
            reproduce_prepare=(
                _stored_reproduction_command(stored_recipe, "prepare")
                if recipe is None
                else (None if recipe.prepare is None else _shell_join(recipe.prepare))
            ),
            reproduce_generate=(
                _stored_reproduction_command(stored_recipe, "generate")
                if recipe is None
                else (None if recipe.generate is None else _shell_join(recipe.generate))
            ),
            reproduce_profile=(
                _stored_reproduction_command(stored_recipe, "profile")
                if recipe is None
                else (None if recipe.profile is None else _shell_join(recipe.profile))
            ),
            published_wall_seconds_per_point=_number(
                current.result,
                "wall_seconds_per_point",
            ),
            published_evaluator_total_seconds_per_point=(
                _evaluator_total_number(current.result)
            ),
            published_recurrence_core_seconds_per_point=(
                _recurrence_core_number(current.result)
            ),
            recycled=True,
            reuse_explanation=current.reason,
            events=[f"recycled: {current.reason}"],
        )
        _apply_persisted_worker_result(
            worker,
            current.result,
            recycled=True,
            reuse_explanation=current.reason,
        )
        workers[cell.cell_id] = worker
    return workers


def _run_campaign(
    arguments: argparse.Namespace,
    *,
    repo_root: Path,
    service: ReportService,
    source: ReportSourceIdentity,
    cells: Sequence[CellSpec],
    palette: Palette,
    installed: bool = False,
) -> int:
    invocation_command = getattr(arguments, "_campaign_invocation_command", None)
    if not isinstance(invocation_command, str) or not invocation_command:
        invocation_command = (
            "steer_performance_campaign.py run "
            "<invocation arguments unavailable to Python API>"
        )
    selective_retry = _selective_retry_requested(arguments)
    if bool(getattr(arguments, "force_refresh", False)) and selective_retry:
        raise ManualCampaignError(
            "--force-refresh cannot be combined with --rerun-failed or "
            "--rerun-capped; use force refresh for every selected cell or the "
            "selective retry flags"
        )
    fresh_attempt = _fresh_attempt_requested(arguments)
    measurable = tuple(cell for cell in cells if _manual_static_na_reason(cell) is None)
    static_na = {
        cell.cell_id for cell in cells if _manual_static_na_reason(cell) is not None
    }
    if not measurable and selective_retry:
        print(
            palette.success(
                "No measurable selected entries match the requested selective "
                "retry; no workers or attempts were created."
            )
        )
        return 0
    if not measurable:
        print(
            palette.warning(
                f"All {len(cells)} selected entries are policy/static unavailable; "
                "no attempts were created."
            )
        )
        print(
            _table(
                ("cell", "reason"),
                ((cell.cell_id, _manual_static_na_reason(cell)) for cell in cells),
            )
        )
        if not arguments.dry_run:
            summary_path, summary_counts = _publish_campaign_summary_ids(
                service,
                {"static_na": static_na},
            )
            _print_campaign_summary_ids(summary_path, summary_counts, palette)
        return 0
    active_lightweight = lightweight_currents(
        service,
        measurable,
        source_revision=source.revision,
    )
    continue_across_revisions = bool(
        getattr(arguments, "continue_across_revisions", False)
    )
    lightweight = (
        lightweight_currents(
            service,
            measurable,
            source_revision=source.revision,
            accept_historical_source=True,
        )
        if continue_across_revisions
        else active_lightweight
    )
    if selective_retry:
        outcomes = lightweight_presentation_outcomes(
            service,
            measurable,
            source_revision=source.revision,
            accept_historical_source=continue_across_revisions,
        )
        base_measurable_count = len(measurable)
        measurable = _selective_retry_cells(
            measurable,
            lightweight,
            outcomes,
            rerun_failed=bool(getattr(arguments, "rerun_failed", False)),
            rerun_capped=bool(getattr(arguments, "rerun_capped", False)),
        )
        if not measurable:
            requested = " and ".join(
                option
                for option, enabled in (
                    ("--rerun-failed", getattr(arguments, "rerun_failed", False)),
                    ("--rerun-capped", getattr(arguments, "rerun_capped", False)),
                )
                if enabled
            )
            print(
                palette.success(
                    f"No entries among {base_measurable_count} measurable "
                    f"selected cells match {requested}; no workers or attempts "
                    "were created."
                )
            )
            return 0
        matched_ids = {cell.cell_id for cell in measurable}
        cells = measurable
        static_na = set()
        active_lightweight = {
            cell_id: current
            for cell_id, current in active_lightweight.items()
            if cell_id in matched_ids
        }
        lightweight = {
            cell_id: current
            for cell_id, current in lightweight.items()
            if cell_id in matched_ids
        }
        print(
            palette.warning(
                f"Selective retry matched {len(measurable)} of "
                f"{base_measurable_count} measurable selected entries."
            )
        )
    recycled = {
        cell_id
        for cell_id, current in lightweight.items()
        if current.reusable and not fresh_attempt
    }
    historical_recycled = {
        cell_id
        for cell_id in recycled
        if _measurement_source_revision(lightweight[cell_id].result) != source.revision
    }
    requested_for_plan = tuple(
        cell
        for cell in measurable
        if fresh_attempt or cell.cell_id not in historical_recycled
    )
    original_amplicol_available = _original_amplicol_available_for_planning(
        arguments,
        installed=installed,
        root=repo_root,
        docs_dir=service.paths.docs_dir,
    )
    arguments.madgraph = _resolve_madgraph(
        arguments,
        installed=installed,
        docs_dir=service.paths.docs_dir,
    )
    preliminary_settings = _campaign_settings(
        arguments,
        source,
        original_amplicol_available=original_amplicol_available,
    )
    planned = plan_campaign(
        requested_for_plan,
        store=service.store,
        settings=preliminary_settings,
        expected_revision=source.revision,
        expected_tree=None,
        current_resolver=_lightweight_current_resolver(
            service,
            source_revision=source.revision,
            initial=active_lightweight,
        ),
    )
    # A historical cell can still be required as an active-source dependency
    # of a newly measured cell.  In that case it is work, not recycled work.
    recycled.difference_update(item.cell.cell_id for item in planned)
    selected_cohorts = _source_cohort_counts(
        lightweight,
        cell_ids=recycled,
    )
    if continue_across_revisions:
        print(
            palette.warning(
                "Cross-revision continuation is enabled: completed selected "
                "currents are kept per cell; dependencies needed by new work "
                "still require the active source revision. Recycled cohorts: "
                f"{_format_source_cohorts(selected_cohorts)}"
            )
        )
    if arguments.dry_run:
        terminal_width = max(
            80,
            min(160, shutil.get_terminal_size(fallback=(120, 24)).columns),
        )
        print(
            _table(
                ("key", "value"),
                (
                    (palette.key("direct entries"), len(cells)),
                    (palette.key("static N/A"), len(static_na)),
                    (palette.key("recycled"), len(recycled)),
                    (
                        palette.key("historical recycled"),
                        len(recycled & historical_recycled),
                    ),
                    (palette.key("planned with dependencies"), len(planned)),
                    (palette.key("source"), source.revision),
                    (
                        palette.key("source policy"),
                        (
                            "continue across revisions"
                            if continue_across_revisions
                            else "strict same-source"
                        ),
                    ),
                    (
                        palette.key("artifact cleanup"),
                        (
                            "compact retention (default)"
                            if _artifact_cleanup_enabled(arguments)
                            else "full debug workspaces (--retain-workspaces)"
                        ),
                    ),
                ),
            )
        )
        print()
        if static_na:
            print(
                _table(
                    ("static/policy unavailable cell", "reason"),
                    (
                        (cell.cell_id, _manual_static_na_reason(cell))
                        for cell in cells
                        if cell.cell_id in static_na
                    ),
                )
            )
            print()
        print(
            _table(
                ("entry", "details"),
                _dry_run_rows(
                    cells,
                    planned,
                    repo_root=repo_root,
                    artifact_root=service.paths.artifact_root,
                    arguments=arguments,
                ),
                max_width={"entry": 18, "details": max(48, terminal_width - 27)},
            )
        )
        direct_recipes = tuple(
            cell for cell in cells if _manual_static_na_reason(cell) is None
        )
        for block in _dry_run_recipe_blocks(
            direct_recipes,
            repo_root=repo_root,
            artifact_root=service.paths.artifact_root,
            arguments=arguments,
            width=terminal_width,
        ):
            print()
            print(block)
        return 0

    _bind_original_amplicol_if_required(
        arguments,
        planned,
        installed=installed,
        root=repo_root,
        docs_dir=service.paths.docs_dir,
    )

    require_measurement_ready(source)
    if continue_across_revisions:
        update_source_marker(
            service,
            source,
            continue_across_revisions=True,
        )
    else:
        update_source_marker(service, source)
    campaign_invocation_id = uuid.uuid4().hex
    for warning in _publish_recycled_presentation_outcomes(
        service,
        lightweight,
        recycled,
        source_revision=source.revision,
        campaign_invocation_id=campaign_invocation_id,
        accept_historical_source=continue_across_revisions,
    ):
        print(
            palette.warning(f"presentation outcome warning: {warning}"),
            file=sys.stderr,
        )
    if _artifact_cleanup_enabled(arguments):
        cleanup_cells = tuple(
            {
                cell.cell_id: cell
                for cell in (
                    *measurable,
                    *(item.cell for item in planned),
                )
            }.values()
        )
        cleanup_warnings = reconcile_attempt_history(service, cleanup_cells)
        for cell_id, detail in cleanup_warnings:
            print(
                palette.warning(
                    f"artifact cleanup skipped for {cell_id}: {detail}"
                ),
                file=sys.stderr,
            )
    measurable_ids = {cell.cell_id for cell in measurable}
    if not planned and measurable_ids <= recycled:
        print(
            _table(
                ("key", "value"),
                (
                    (palette.key("selected entries"), len(cells)),
                    (palette.key("measurable entries"), len(measurable)),
                    (
                        palette.key("recycled measurable entries"),
                        palette.success(len(recycled)),
                    ),
                    (
                        palette.key("static N/A entries"),
                        (
                            palette.warning(len(static_na))
                            if static_na
                            else palette.neutral(0)
                        ),
                    ),
                    (palette.key("workers created"), palette.success(0)),
                    (palette.key("attempts created"), palette.success(0)),
                ),
                align={"value": "r"},
            )
        )
        print()
        print(
            palette.success(
                "All selected measurable entries were recycled "
                f"({len(recycled)} of {len(measurable)}); no workers or "
                "attempts were created."
            )
        )
        if static_na:
            print(
                palette.warning(
                    f"{len(static_na)} selected static N/A "
                    f"{'entry remains' if len(static_na) == 1 else 'entries remain'} "
                    "policy unavailable and accounted for separately."
                )
            )
        summary_path, summary_counts = _publish_campaign_summary_ids(
            service,
            _campaign_summary_categories(
                static_na_ids=static_na,
            ),
        )
        _print_campaign_summary_ids(summary_path, summary_counts, palette)
        return 0
    cancellation = threading.Event()
    finished = threading.Event()
    state = DashboardState(
        instance_id=campaign_invocation_id,
        selected_ids=tuple(cell.cell_id for cell in cells),
        recycled_ids=recycled,
        static_na_ids=static_na,
        source_revision=source.revision,
        generation_time_limit_seconds=float(arguments.generation_time_limit),
        memory_limit_bytes=(
            preliminary_settings.effective_cell_rss_limit()
            or int(arguments.ram_limit)
        ),
        worker_wall_limit_seconds=arguments.worker_wall_limit,
        reproduction_settings=ReproductionSettings(
            cores=int(arguments.cores_per_worker),
            target_runtime=float(arguments.target_measurement_duration),
            batch_size=int(arguments.batch_size),
            warmups=int(arguments.warmups),
            minimum_samples=int(arguments.minimum_samples),
        ),
        dependency_ids={item.cell.cell_id for item in planned if item.dependency},
    )
    state.workers.update(
        _recycled_attention_workers(
            measurable,
            lightweight,
            recycled,
            repo_root=repo_root,
            artifact_root=service.paths.artifact_root,
            settings=state.reproduction_settings,
        )
    )
    for cell_id, worker in state.workers.items():
        state._reduce_finished(
            cell_id,
            worker.status,
            recycled=worker.recycled,
        )
    lease = LeaseManager(service, state)
    lease.publish()
    fail_fast_observer = (
        _FailFastObserver(lease.observe, cancellation)
        if bool(getattr(arguments, "fail_fast", False))
        else None
    )
    settings = _campaign_settings(
        arguments,
        source,
        original_amplicol_available=original_amplicol_available,
        observer=(
            lease.observe
            if fail_fast_observer is None
            else fail_fast_observer.observe
        ),
        cancelled=cancellation.is_set,
        campaign_invocation_id=state.instance_id,
    )
    scheduler = CampaignScheduler(service, settings=settings)
    result_holder: list[CampaignResult] = []
    error_holder: list[BaseException] = []

    def execute() -> None:
        try:
            result_holder.append(scheduler.run(planned))
        except BaseException as error:
            error_holder.append(error)
        finally:
            finished.set()

    worker = threading.Thread(
        target=execute,
        name="manual-performance-campaign",
        daemon=False,
    )
    worker.start()
    interrupted = False
    join_interrupted = False
    worker_alive = False
    try:
        interactive = (
            not arguments.no_dashboard
            and bool(getattr(sys.stdin, "isatty", lambda: False)())
            and bool(getattr(sys.stdout, "isatty", lambda: False)())
        )
        if interactive:
            _run_live_dashboard(
                lease,
                state,
                finished,
                cancellation,
                color=palette.enabled,
            )
        else:
            while not finished.wait(timeout=0.25):
                pass
    except KeyboardInterrupt:
        interrupted = True
        state.interrupted = True
        cancellation.set()
    finally:
        if state.interrupted:
            interrupted = True
        try:
            join_interrupted, worker_alive = _cancel_and_join_campaign_worker(
                worker,
                cancellation,
                timeout_seconds=max(5.0, arguments.termination_grace + 2.0),
            )
        finally:
            lease.close()
    if join_interrupted or worker_alive:
        interrupted = True
        state.interrupted = True
    fail_fast_failure = (
        None if fail_fast_observer is None else fail_fast_observer.failure
    )
    _print_cleanup_warnings(state, palette)
    if interrupted:
        result = result_holder[0] if result_holder else None
        _publish_completion_summary(
            service,
            _campaign_summary_categories(
                static_na_ids=static_na,
                result=result,
                state=state,
                interrupted=True,
            ),
            state=state,
            fail_fast_failure=fail_fast_failure,
            invocation_command=invocation_command,
            palette=palette,
        )
        artifact_outcome = (
            "disposable workspaces were compacted (default)"
            if _artifact_cleanup_enabled(arguments)
            else "full debug workspaces were retained (--retain-workspaces)"
        )
        print(
            palette.warning(
                "Interrupted: dispatch stopped, process trees received cancellation, "
                "compact interrupted-attempt diagnostics and completed currents "
                f"were preserved, {artifact_outcome}, and leases were removed."
            ),
            file=sys.stderr,
        )
        return 130
    if error_holder:
        partial_result = result_holder[0] if result_holder else None
        if fail_fast_failure is not None or _has_invocation_summary_evidence(
            partial_result, state
        ):
            _publish_completion_summary(
                service,
                _campaign_summary_categories(
                    static_na_ids=static_na,
                    result=partial_result,
                    state=state,
                ),
                state=state,
                fail_fast_failure=fail_fast_failure,
                invocation_command=invocation_command,
                palette=palette,
            )
        raise error_holder[0]
    result = result_holder[0]
    statuses = Counter(outcome.status for outcome in result.outcomes)
    print(
        _table(
            ("status", "entries"),
            (
                (
                    palette.key(status),
                    (
                        palette.success(count)
                        if status in _FAIL_FAST_SUCCESS_STATUSES
                        else palette.warning(count)
                    ),
                )
                for status, count in sorted(statuses.items())
            ),
            align={"entries": "r"},
        )
    )
    _publish_completion_summary(
        service,
        _campaign_summary_categories(
            static_na_ids=static_na,
            result=result,
            state=state,
        ),
        state=state,
        fail_fast_failure=fail_fast_failure,
        invocation_command=invocation_command,
        palette=palette,
    )
    if bool(getattr(arguments, "fail_fast", False)):
        has_terminal_failure = any(
            outcome.status not in _FAIL_FAST_SUCCESS_STATUSES
            and outcome.status != "cancelled"
            for outcome in result.outcomes
        )
        return 1 if fail_fast_failure is not None or has_terminal_failure else 0
    return 1 if result.failed else 0


def _capture_lightweight_snapshot(
    service: ReportService,
    *,
    source_revision: str,
    accept_historical_source: bool = False,
    progress: _SnapshotProgress | None = None,
) -> tuple[
    dict[str, LightweightCurrent],
    dict[str, LightweightPresentationOutcome],
    tuple[tuple[str, str], ...],
]:
    cells = tuple(service.catalog.measurement_cells())
    total = 4 * len(cells)
    for attempt in range(3):
        started_at = time.monotonic()
        completed_offset = 0
        last_update = 0.0

        if progress is not None:
            progress.begin(
                total=total,
                attempt=attempt + 1,
                maximum_attempts=3,
            )

        def callback(
            stage: str,
            stage_offset: int,
        ) -> Callable[[int, int], None] | None:
            nonlocal last_update
            if progress is None:
                return None
            progress.update(stage_offset, total, stage)

            def update(completed: int, _stage_total: int) -> None:
                nonlocal last_update
                now = time.monotonic()
                overall = stage_offset + completed
                if completed == 1 or overall == total or now - last_update >= 0.08:
                    progress.update(overall, total, stage)
                    last_update = now

            return update

        try:
            currents = lightweight_currents(
                service,
                cells,
                source_revision=source_revision,
                accept_historical_source=accept_historical_source,
                progress=callback(
                    "Reading current records",
                    completed_offset,
                ),
            )
            completed_offset += len(cells)
            outcomes = lightweight_presentation_outcomes(
                service,
                cells,
                source_revision=source_revision,
                accept_historical_source=accept_historical_source,
                progress=callback(
                    "Reading terminal outcomes",
                    completed_offset,
                ),
            )
            completed_offset += len(cells)
            identity = tuple(
                sorted(
                    (cell_id, current.attempt_id)
                    for cell_id, current in currents.items()
                    if current.reusable
                )
            )
            outcome_identity = tuple(
                sorted(
                    (cell_id, outcome.ordering_key)
                    for cell_id, outcome in outcomes.items()
                )
            )
            confirmed = lightweight_currents(
                service,
                cells,
                source_revision=source_revision,
                accept_historical_source=accept_historical_source,
                progress=callback(
                    "Confirming current records",
                    completed_offset,
                ),
            )
            completed_offset += len(cells)
            confirmed_outcomes = lightweight_presentation_outcomes(
                service,
                cells,
                source_revision=source_revision,
                accept_historical_source=accept_historical_source,
                progress=callback(
                    "Confirming terminal outcomes",
                    completed_offset,
                ),
            )
            confirmed_identity = tuple(
                sorted(
                    (cell_id, current.attempt_id)
                    for cell_id, current in confirmed.items()
                    if current.reusable
                )
            )
            confirmed_outcome_identity = tuple(
                sorted(
                    (cell_id, outcome.ordering_key)
                    for cell_id, outcome in confirmed_outcomes.items()
                )
            )
            stable = (
                identity == confirmed_identity
                and outcome_identity == confirmed_outcome_identity
            )
        except BaseException:
            if progress is not None:
                progress.end(
                    success=False,
                    message="artifact scan failed",
                    elapsed_seconds=time.monotonic() - started_at,
                )
            raise
        final_attempt = attempt + 1 == 3
        if progress is not None:
            progress.end(
                success=stable or not final_attempt,
                message=(
                    "stable snapshot captured"
                    if stable
                    else (
                        "campaign state remained unstable after three scans"
                        if final_attempt
                        else "campaign state changed; rescanning"
                    )
                ),
                elapsed_seconds=time.monotonic() - started_at,
            )
        if stable:
            return currents, outcomes, identity
    raise ManualCampaignError(
        "current pointers or presentation outcomes changed during three "
        "snapshot reads; retry refresh-pdf"
    )


def _merge_lightweight_snapshot(
    service: ReportService,
    currents: Mapping[str, LightweightCurrent],
    outcomes: Mapping[str, LightweightPresentationOutcome] | None = None,
) -> tuple[dict[str, dict[str, object]], int]:
    caches = service.reset_payloads()
    visible_outcomes = {} if outcomes is None else outcomes
    by_cell = {cell.cell_id: cell for cell in service.catalog.measurement_cells()}
    merged = 0
    for payload in caches.values():
        entries = payload["entries"]
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, dict)
            cell_id = str(entry["cell_id"])
            cell = by_cell[cell_id]
            if service.catalog.static_na_reason(cell) is not None:
                entry["measurement"] = reset_entry(cell)["measurement"]
                continue
            current = currents.get(cell_id)
            outcome = visible_outcomes.get(cell_id)
            if (
                current is not None
                and current.reusable
                and current.result.get("status") == ResultStatus.OK.value
            ):
                measurement = dict(current.result)
            elif outcome is not None and not outcome.successful:
                if (
                    current is not None
                    and current.reusable
                    and outcome.attempt_id is not None
                    and outcome.attempt_id == current.attempt_id
                ):
                    measurement = dict(current.result)
                else:
                    measurement = _presentation_measurement(service, cell, outcome)
            elif outcome is not None and outcome.successful:
                if current is None or not current.reusable:
                    continue
                measurement = dict(current.result)
            elif current is not None and current.reusable:
                measurement = dict(current.result)
            else:
                continue
            validate_measurement(measurement, expected_cell=cell)
            entry["measurement"] = measurement
            merged += 1
    service.validate_payloads(caches)
    return caches, merged


def _copy_report_sources(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=_report_source_copy_ignore(
            source,
            "pyAmpliCol.pdf",
            "*.aux",
            "*.bbl",
            "*.bcf",
            "*.blg",
            "*.fdb_latexmk",
            "*.fls",
            "*.log",
            "*.out",
            "*.run.xml",
            "*.synctex.gz",
            "*.toc",
        ),
    )


def _selected_report_sections(raw: Sequence[str] | None) -> tuple[ReportSection, ...]:
    requested = set(raw or ())
    known = {section.identifier for section in REPORT_SECTIONS}
    unknown = tuple(sorted(requested - known))
    if unknown:
        raise ManualCampaignError(
            "unknown report section ID(s): "
            + ", ".join(unknown)
            + "; run `refresh-pdf --list-sections` to list valid IDs"
        )
    return tuple(
        section for section in REPORT_SECTIONS if section.identifier in requested
    )


def _filter_report_sections(
    master: Path,
    removed: Sequence[ReportSection],
) -> None:
    """Remove selected marked sections from one private staged TeX source."""

    removed_ids = {section.identifier for section in removed}
    if not removed_ids:
        return
    try:
        lines = master.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as error:
        raise ManualCampaignError(
            f"cannot read staged report source {master}"
        ) from error
    events = tuple(
        (match.group(1), match.group(2))
        for line in lines
        if (match := _REPORT_SECTION_MARKER.fullmatch(line.rstrip("\r\n")))
        is not None
    )
    expected = tuple(
        event
        for section in REPORT_SECTIONS
        for event in (("begin", section.identifier), ("end", section.identifier))
    )
    if events != expected:
        raise ManualCampaignError(
            "report section markers differ from the installed section registry; "
            "refresh the copied campaign template before using --remove-sections"
        )
    output: list[str] = []
    active: str | None = None
    for line in lines:
        match = _REPORT_SECTION_MARKER.fullmatch(line.rstrip("\r\n"))
        if match is not None:
            event, identifier = match.groups()
            if event == "begin":
                active = identifier
                if identifier not in removed_ids:
                    output.append(line)
            else:
                if identifier not in removed_ids:
                    output.append(line)
                active = None
            continue
        if active not in removed_ids:
            output.append(line)
    master.write_text("".join(output), encoding="utf-8")


def _print_report_sections(palette: Palette) -> None:
    print(
        _table(
            ("section ID", "PDF section"),
            (
                (palette.key(section.identifier), section.title)
                for section in REPORT_SECTIONS
            ),
        )
    )


def _stage_copied_profile_identity(staging_docs: Path, profile: str) -> None:
    """Rebind copied workspace labels inside the private PDF build only."""

    environment_path = staging_docs / "report_environment.json"
    environment = _read_object(environment_path)
    if environment is not None:
        environment["profile"] = profile
        _atomic_json(environment_path, environment)

    workspace_path = staging_docs / "report-workspace.json"
    workspace = _read_object(workspace_path)
    if workspace is not None:
        workspace["profile"] = profile
        workspace["artifact_root"] = "campaign_artifacts"
        workspace["coordination_root"] = "campaign_artifacts/coordination"
        initialized = workspace.get("initialized_environment")
        if isinstance(initialized, Mapping):
            workspace["initialized_environment"] = {
                **initialized,
                "profile": profile,
            }
        _atomic_json(workspace_path, workspace)

    tex_path = staging_docs / "report_environment.tex"
    try:
        tex = tex_path.read_text(encoding="utf-8")
    except OSError:
        return
    escaped_profile = profile.replace("_", r"\_")
    updated, replacements = re.subn(
        r"(\\renewcommand\{\\ReportProfileName\}\{)[^}\r\n]*(\})",
        lambda match: f"{match.group(1)}{escaped_profile}{match.group(2)}",
        tex,
        count=1,
    )
    if replacements != 1:
        raise ManualCampaignError(
            "staged report environment does not define ReportProfileName"
        )
    tex_path.write_text(updated, encoding="utf-8")


def _require_reusable_staged_environment(
    staging_docs: Path,
    profile: str,
    *,
    expected_source_revision: str,
) -> None:
    """Reject authenticated metadata from an older measurement-source epoch."""

    environment_path = staging_docs / ENVIRONMENT_JSON
    if not environment_path.exists():
        # The atomic installer reports a missing report member below.  This
        # helper is concerned only with preventing stale authentication reuse.
        return
    environment = _read_object(environment_path)
    if environment is None:
        raise ManualCampaignError(
            "staged report environment metadata is malformed"
        )
    if environment.get("profile") != profile:
        raise ManualCampaignError(
            "staged report environment profile was not rebound"
        )
    status = environment.get("status")
    source_revision = environment.get("source_revision")
    if status == "pending_exact_runtime" and source_revision == "pending":
        return
    if status == "authenticated" and source_revision == expected_source_revision:
        return
    raise ManualCampaignError(
        "no active-source runtime witness is available, and the staged report "
        "environment is not authenticated for the active measurement source"
    )


def _install_report_snapshot(
    service: ReportService,
    staging_docs: Path,
    table_names: Sequence[str],
) -> None:
    sources: list[tuple[Path, Path]] = [
        *(
            (
                path,
                service.paths.results_dir / path.name,
            )
            for path in sorted((staging_docs / "results").glob("*.json"))
        ),
        *(
            (
                staging_docs / name,
                service.paths.docs_dir / name,
            )
            for name in sorted(table_names)
        ),
        *(
            (
                staging_docs / name,
                service.paths.docs_dir / name,
            )
            for name in (ENVIRONMENT_JSON, ENVIRONMENT_TEX)
        ),
        (
            staging_docs / "pyAmpliCol.pdf",
            service.paths.docs_dir / "pyAmpliCol.pdf",
        ),
    ]
    for source_path, _destination in sources:
        if not source_path.is_file():
            raise ManualCampaignError(f"staged report member is missing: {source_path}")
    backup_root = staging_docs.parent / "previous"
    backup_root.mkdir()
    replaced: list[tuple[Path, Path | None]] = []
    with service.store.named_lock("report-writer"):
        try:
            for source_path, destination in sources:
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = None
                if destination.exists():
                    backup = backup_root / destination.relative_to(
                        service.paths.docs_dir
                    )
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, backup)
                temporary = destination.with_name(
                    f".{destination.name}.manual-{uuid.uuid4().hex}"
                )
                try:
                    shutil.copy2(source_path, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                replaced.append((destination, backup))
        except BaseException:
            for destination, backup in reversed(replaced):
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            raise


def _active_environment_runtime_witness(
    currents: Mapping[str, LightweightCurrent],
    *,
    source_revision: str,
) -> _ProfileEnvironmentWitness | None:
    """Select one consistent runtime witness from validated active currents."""

    selected: _ProfileEnvironmentWitness | None = None
    selected_identity: str | None = None
    for cell_id in sorted(currents):
        current = currents[cell_id]
        if (
            not current.reusable
            or current.result.get("status") != ResultStatus.OK.value
            or _measurement_source_revision(current.result) != source_revision
        ):
            continue
        provenance = current.result.get("provenance")
        runtime = (
            provenance.get("runtime_identity")
            if isinstance(provenance, Mapping)
            else None
        )
        if runtime is None:
            # Original AmpliCol results carry no pyAmpliCol runtime witness.
            continue
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("source_revision") != source_revision
        ):
            raise ManualCampaignError(
                f"{cell_id}: active-source runtime witness is malformed"
            )
        candidate = runtime.get("candidate_build_identity")
        package_tree = runtime.get("python_package_tree")
        native_extension = runtime.get("native_extension")
        native_target = runtime.get("native_target")
        if not all(
            isinstance(value, Mapping)
            for value in (candidate, package_tree, native_extension, native_target)
        ):
            raise ManualCampaignError(
                f"{cell_id}: active-source runtime witness is incomplete"
            )
        assert isinstance(candidate, Mapping)
        if candidate.get("source_revision") != source_revision:
            raise ManualCampaignError(
                f"{cell_id}: active-source runtime build identity is malformed"
            )
        numpy_version = provenance.get("numpy")
        if numpy_version is None:
            numpy_version = _UNRECORDED_MEASUREMENT_NUMPY
        host_environment = {
            "platform": provenance.get("platform"),
            "machine": provenance.get("machine"),
            "processor": provenance.get("processor"),
            "python": provenance.get("python"),
            "python_implementation": provenance.get(
                "python_implementation",
                "CPython",
            ),
            "numpy": numpy_version,
        }
        if not all(
            isinstance(host_environment[field], str) and bool(host_environment[field])
            for field in (
                "platform",
                "machine",
                "python",
                "python_implementation",
                "numpy",
            )
        ) or not isinstance(host_environment["processor"], str):
            raise ManualCampaignError(
                f"{cell_id}: active-source measurement-host witness is incomplete"
            )
        assert isinstance(package_tree, Mapping)
        assert isinstance(native_extension, Mapping)
        assert isinstance(native_target, Mapping)
        fingerprint = candidate.get("candidate_fingerprint")
        if fingerprint is None and candidate.get("publishable") is True:
            fingerprint = runtime.get("candidate_build_identity_sha256")
        projection = {
            "source_revision": runtime.get("source_revision"),
            "package_version": runtime.get("package_version"),
            "python_package_tree_sha256": package_tree.get("sha256"),
            "build_fingerprint": fingerprint,
            "native_build_inputs_sha256": runtime.get("native_build_inputs_sha256"),
            "native_extension_sha256": native_extension.get("sha256"),
            "native_target": native_target.get("triple"),
            "native_cpu_features": native_target.get("cpu_features"),
            "measurement_host": host_environment,
        }
        identity = json.dumps(
            projection,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if selected_identity is not None and identity != selected_identity:
            raise ManualCampaignError(
                "active measurement-source currents disagree on their "
                "authenticated runtime environment"
            )
        selected = _ProfileEnvironmentWitness(runtime, host_environment)
        selected_identity = identity
    return selected


def _refresh_pdf(
    arguments: argparse.Namespace,
    *,
    service: ReportService,
    source: ReportSourceIdentity,
    palette: Palette,
) -> int:
    removed_sections = _selected_report_sections(arguments.remove_sections)
    source_policy = recorded_measurement_source_policy(
        service,
        checkout_revision=source.revision,
    )
    scan_progress: _SnapshotProgress | None = None
    if not bool(arguments.quiet) and bool(
        getattr(sys.stderr, "isatty", lambda: False)()
    ):
        scan_progress = _RefreshScanProgress(sys.stderr, palette)
    try:
        snapshot_arguments = {
            "source_revision": source_policy.source_revision,
            "accept_historical_source": (
                source_policy.continue_across_revisions
            ),
        }
        if scan_progress is None:
            currents, outcomes, snapshot = _capture_lightweight_snapshot(
                service,
                **snapshot_arguments,
            )
        else:
            currents, outcomes, snapshot = _capture_lightweight_snapshot(
                service,
                progress=scan_progress,
                **snapshot_arguments,
            )
    finally:
        if scan_progress is not None:
            scan_progress.close()
    source_cohorts = _source_cohort_counts(currents)
    environment_runtime = _active_environment_runtime_witness(
        currents,
        source_revision=source_policy.source_revision,
    )
    publication_caches, merged = _merge_lightweight_snapshot(
        service,
        currents,
    )
    render_caches, _presentation_merged = _merge_lightweight_snapshot(
        service,
        currents,
        outcomes,
    )
    build_root = service.paths.artifact_root / "manual-publication-builds"
    build_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="refresh-", dir=build_root))
    try:
        staging_docs = staging / "docs"
        _copy_report_sources(service.paths.docs_dir, staging_docs)
        _stage_copied_profile_identity(
            staging_docs,
            service.paths.docs_dir.name,
        )
        if environment_runtime is not None:
            record_authenticated_profile_environment(
                staging_docs,
                service.paths.docs_dir.name,
                expected_source_revision=source_policy.source_revision,
                active_runtime=environment_runtime.active_runtime,
                host_environment=environment_runtime.host_environment,
            )
        else:
            _require_reusable_staged_environment(
                staging_docs,
                service.paths.docs_dir.name,
                expected_source_revision=source_policy.source_revision,
            )
        _filter_report_sections(
            staging_docs / "pyAmpliCol.tex",
            removed_sections,
        )
        staging_service = ReportService(
            ReportPaths.from_repo(
                service.paths.repo_root,
                docs_dir=staging_docs,
                artifact_root=service.paths.artifact_root,
                coordination_root=service.paths.coordination_root,
            ),
            catalog=service.catalog,
        )
        staging_service.bind_measurement_lineage(None)
        staging_service.bind_original_amplicol_seed(None)
        tables = staging_service._render_tables(render_caches)
        staging_service._snapshot_files(publication_caches, tables)
        page_count = _compile_pdf(
            staging_docs,
            expected_page_count=(
                None
                if arguments.expected_page_count is None
                else int(arguments.expected_page_count)
            ),
            timeout_seconds=float(arguments.pdf_timeout),
            allow_overfull_boxes=True,
            stream_output=not bool(arguments.quiet),
        )
        _install_report_snapshot(service, staging_docs, tuple(tables))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(
        _table(
            ("key", "value"),
            (
                (
                    palette.key("source policy"),
                    (
                        "continue across revisions"
                        if source_policy.continue_across_revisions
                        else "strict same-source"
                    ),
                ),
                (
                    palette.key("measurement source cohorts"),
                    _format_source_cohorts(source_cohorts),
                ),
                (palette.key("renderer checkout"), source.revision),
                (palette.key("current snapshot"), len(snapshot)),
                (palette.key("measurements merged"), merged),
                (palette.key("tables rebuilt"), len(tables)),
                (
                    palette.key("sections omitted"),
                    (
                        ", ".join(
                            section.identifier for section in removed_sections
                        )
                        or "none"
                    ),
                ),
                (palette.key("PDF pages"), page_count),
                (
                    palette.key("installed"),
                    palette.success(service.paths.docs_dir / "pyAmpliCol.pdf"),
                ),
            ),
        )
    )
    print(f"PDF: {(service.paths.docs_dir / 'pyAmpliCol.pdf').resolve(strict=False)}")
    return 0


_PROCESS_SELECTOR_GUIDE = "\n".join(
    f"  {family.identifier:>2}: {family.key}" for family in PROCESS_FAMILIES
)
_DATASET_SELECTOR_GUIDE = textwrap.fill(
    ", ".join(sorted({cell.dataset_id for cell in REPORT_CATALOG.measurement_cells()})),
    width=78,
    initial_indent="  ",
    subsequent_indent="  ",
)
_ALIAS_SELECTOR_GUIDE = """
  table groups: z/z_table; matrix/matrix_table; matrix_best; reference; scalar
  colour: lc/leading_color/leading_colour;
          nlc/next_to_leading_color/next_to_leading_colour;
          full/full_color/full_colour
  layouts: non_union_flow/selected_flow/single_flow/single_flow_hel_sum/
           topology_replay; union_flow/all_flow/all_flows/
           all_flows_single_hel/all_flow_union; contracted
  engines: amplicol/legacy; recurrence/ampli_col; compiled; eager;
           on-the-fly/otf
  models: builtin/built_in/built_in_sm/builtin_sm/sm;
          ufo/sm_ufo/ufo_sm/external_sm; scalar_contact; scalar_gravity
""".strip("\n")


STEERING_GUIDE = f"""
Steering model
--------------
Selectors are repeatable and accept several values after one option. Values
within one dimension are ORed; dimensions are ANDed. Omitted dimensions and a
quoted '*' (or 'all') mean all values. Hyphens, underscores, and case are
normalized. `non-union-flow` means one selected colour flow with the helicity
sum; `union-flow` means all colour flows at one helicity; NLC/full rows use the
contracted workload.

`--cell-id-file` reads exact IDs, one per UTF-8 line, ignoring blank lines and
full-line `#` comments. IDs from files and `--cell-id` are ORed before the
remaining selectors are applied. Each completed run atomically refreshes
`campaign_summary_ids/`; use one or more of its status files to target a retry.
`unverified.txt` needs no `--force-refresh`: those retained diagnostics are not
successful currents, and a later recurrence authority or AmpliCol diagnostic
dependency causes the selected cells to run and validate again automatically.
By default the controller retains results, provenance, commands, progress, and
bounded log tails while removing disposable legacy/MadGraph/build workspaces
and obsolete heavy payloads. Use `--retain-workspaces` only when a debugging
session needs the full workspaces; output and free-disk safety limits still
apply. `--cleanup-artifacts` remains an accepted compatibility spelling for
the default. `--attempt-output-limit` defaults to 64 MiB per watched file and
`--minimum-free-disk` reserves 5 GiB on the campaign artifact volume. Keyboard
interruption and fail-fast leave attempts sealed with compact diagnostics
rather than leaving unbounded partial workspaces behind.

Where the report protocol permits, the controller parses, resolves, and
dispatches the real public `pyamplicol generate` and `pyamplicol profile`
subcommands in-process. Dry runs print directly runnable generation commands
and clearly labelled pre-generation profile templates; completed recurrence
provenance replaces templates with stable selectors and a materialized exact
momenta input. Compiled generation's precompiled-model injection and
compiled/eager paired Arena publication remain explicit report-protocol
exceptions. Original AmpliCol uses its maintained legacy adapter because no
public pyamplicol subcommand exists.

Campaign copies
---------------
Create a complete reset campaign with
`pyamplicol profiling-campaign copy DESTINATION`. The destination directory
name is the profile identity, so separate copies receive distinct artifact,
coordination, source-marker, result, and PDF paths.

Resource and availability policy
--------------------------------
Multiplicity-eight all-flow recurrence is runnable. Current releases use
bounded numerical-current evidence; a shape that cannot fit the 1-GiB envelope
visibly falls back to reuse-off while ordinary generation continues under the
generation-time and 30-GB process-tree caps. Z-table C++/ASM variants above
multiplicity six remain catalog-defined static N/A and create no attempts.

By default the planner adds each selected cell's available validation
dependencies at the active source revision. The exact per-process order is the
original-AmpliCol diagnostic, recurrence authority, then compiled/eager
candidate; added cells are shown as dependency-only work. Independent processes
remain parallel. A missing or terminal authority releases its candidate to run
unverified rather than creating a blocked dependency. Use
`--no-dependencies-added` for an explicitly baseline-free selection; hard
construction and selector/provider dependencies are always retained. Selecting
`amplicol` explicitly still requires a clean complete checkout. An omitted
engine selector and quoted `*` both mean every engine.

Canonical selector values and aliases
-------------------------------------
  tables: z_table (z), matrix, matrix_best, reference, scalar, or exact dataset
  colour: lc, nlc, full
  layouts: non-union-flow (selected-flow/topology-replay),
           union-flow (all-flow), contracted
  engines: amplicol, recurrence, compiled, eager, on-the-fly (otf)
  models: built_in (builtin_sm), sm_ufo (ufo_sm), scalar_contact, scalar_gravity
  variants: recurrence_jit_o2, jit_o1, jit_o3, eager_jit_o2, cpp_o3, asm_o3
            (filters named Z implementations only; unvaried rows remain)
  wildcard: quoted '*' or all; omitted selectors also mean all

Accepted aliases
----------------
{_ALIAS_SELECTOR_GUIDE}

Exact dataset IDs
-----------------
{_DATASET_SELECTOR_GUIDE}

Process IDs
-----------
{_PROCESS_SELECTOR_GUIDE}

Common recipes
--------------
  One exact cell:
    steer_performance_campaign.py run --cell-id CELL_ID

  Multiplicities 3 and 4 for both SM models:
    steer_performance_campaign.py run --multiplicity 3 4 --model built_in sm_ufo

  Complete Z table:
    steer_performance_campaign.py run --table z_table

  Every catalog entry, planned only:
    steer_performance_campaign.py run --dry-run

  Four workers, two cores each:
    steer_performance_campaign.py run --workers 4 --cores-per-worker 2

  Stop dispatch on the first terminal failure and retain its full evidence:
    steer_performance_campaign.py run --fail-fast --workers 4 --table matrix

  All pyAmpliCol engines without adding optional validation dependencies:
    steer_performance_campaign.py run \\
      --generation-engine recurrence compiled eager on-the-fly \\
      --no-dependencies-added

  Recompute instead of reusing same-source currents:
    steer_performance_campaign.py run --force-refresh --table z_table

  Retry only failed or policy-capped cells inside the ordinary selection:
    steer_performance_campaign.py run --rerun-failed --rerun-capped \
      --continue-across-revisions --table matrix

  Retry exact error and validation-failure IDs:
    steer_performance_campaign.py run \
      --cell-id-file campaign_summary_ids/error.txt \
                     campaign_summary_ids/validation_failed.txt \
      --force-refresh

  Keep completed cells while complementing them with a new clean build:
    steer_performance_campaign.py run --continue-across-revisions \\
      --table z_table

  Inspect recurrence and compiled LC rows:
    steer_performance_campaign.py inspect --color-approximation lc \\
      --generation-engine recurrence compiled

  Rebuild every JSON/TeX table and the PDF:
    steer_performance_campaign.py refresh-pdf

  List stable PDF section IDs or omit selected sections from one build:
    steer_performance_campaign.py refresh-pdf --list-sections
    steer_performance_campaign.py refresh-pdf \\
      --remove-sections worked-zgg shared-current-dag

  Deterministic dashboard fixture for layout review:
    steer_performance_campaign.py dashboard-snapshot --width 120 --height 36

  Read-only snapshot of the newest active campaign lease:
    steer_performance_campaign.py dashboard-snapshot --live

  Snapshot one concurrent invocation by exact ID or unique prefix:
    steer_performance_campaign.py dashboard-snapshot --live --instance ID_PREFIX

Keyboard controls
-----------------
  ↑/↓ or j/k  select a worker     PgUp/PgDn  scroll worker details
  d           show/hide completed successful workers (hidden by default)
  e           show/hide errors, caps, and recycled non-success rows
  1/2/3       open the complete prepare/generate/profile command drawer
  Tab or ←/→  cycle available commands       y  copy the exact command
  p           print the exact command outside the dashboard for selection
  Esc         close the command drawer        ?  show dashboard help
  Ctrl-C      stop safely, preserve compact diagnostics, and summarize IDs

The selected-worker panel preserves typed engine details when available,
including recurrence stages, current/contribution counts, relation counts,
and evidence sizes. RAM is the sampled process-tree current/peak usage.
"""


def _selector_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--table",
        nargs="+",
        action="append",
        metavar="TABLE",
        help=(
            "Rendered table/dataset selector. Canonical groups: z_table, "
            "matrix, matrix_best, reference, scalar; exact dataset IDs are "
            "also accepted. Repeat or supply multiple values."
        ),
    )
    parent.add_argument(
        "--process-id",
        nargs="+",
        action="append",
        metavar="ID",
        help=(
            "Process family number 1..15, process key, or quoted concrete "
            "process. Numeric IDs follow the report's canonical process list."
        ),
    )
    parent.add_argument(
        "--multiplicity",
        "--multiplcity",
        nargs="+",
        action="append",
        metavar="N",
        help="Final-state multiplicity; positive integers such as 3 4.",
    )
    parent.add_argument(
        "--color-approximation",
        "--color_approximation",
        "--colour-approximation",
        "--colour_approximation",
        nargs="+",
        action="append",
        metavar="LEVEL",
        help="Colour accuracy: lc, nlc, or full (friendly long aliases accepted).",
    )
    parent.add_argument(
        "--generation-mode",
        nargs="+",
        action="append",
        metavar="LAYOUT",
        help=(
            "Workload/layout: union-flow, non-union-flow, or contracted. "
            "Aliases all-flow, selected-flow, and topology-replay are accepted."
        ),
    )
    parent.add_argument(
        "--generation-engine",
        nargs="+",
        action="append",
        metavar="ENGINE",
        help=(
            "Engine: amplicol, recurrence, compiled, eager, or on-the-fly (alias otf)."
        ),
    )
    parent.add_argument(
        "--model",
        nargs="+",
        action="append",
        metavar="MODEL",
        help=(
            "Model: built_in/builtin_sm, sm_ufo/ufo_sm, scalar_contact, or "
            "scalar_gravity."
        ),
    )
    parent.add_argument(
        "--variant",
        nargs="+",
        action="append",
        metavar="VARIANT",
        help=(
            "Z-table implementation variant; repeat or supply multiple values. "
            "This narrows only variant-bearing rows; ordinary matrix/reference "
            "rows have no variant dimension and remain eligible."
        ),
    )
    parent.add_argument(
        "--cell-id",
        nargs="+",
        action="append",
        metavar="CELL",
        help="Exact canonical cell identity; repeat or supply multiple values.",
    )
    parent.add_argument(
        "--cell-id-file",
        nargs="+",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "UTF-8 file containing one exact canonical cell ID per line; blank "
            "lines and full-line # comments are ignored. Repeat or supply "
            "multiple paths. IDs are unioned with --cell-id and intersected "
            "with every other selector. Wildcards are not accepted in files. "
            "Running blocked_dependency.txt plans its prerequisite closure; "
            "unverified.txt retries without --force-refresh when independent "
            "authority becomes available; "
            "use --force-refresh when a selected ID already has a reusable "
            "terminal current."
        ),
    )
    return parent


def _output_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--no-color",
        action="store_true",
        help=(
            "Disable ANSI colours and Ratatui styles. Output is coloured by "
            "default; NO_COLOR also disables colour. Inspect JSON never contains "
            "ANSI escapes; dashboard cell JSON retains numeric style metadata "
            "unless colour is disabled."
        ),
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steer_performance_campaign.py",
        description=(
            "Safely steer, observe, inspect, and publish a pyAmpliCol "
            "profiling campaign."
        ),
        epilog=STEERING_GUIDE,
        formatter_class=HelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    selectors = _selector_parent()
    output = _output_parent()

    run = commands.add_parser(
        "run",
        parents=[selectors, output],
        help="Select cells, preview dependencies, or run supervised workers.",
        description=(
            "Select report cells, reuse matching same-source currents by "
            "default, plan dependencies, and execute resource-supervised "
            "workers. Cross-revision continuation is explicit and keeps "
            "active-build dependency planning strict."
        ),
        epilog=STEERING_GUIDE,
        formatter_class=HelpFormatter,
    )
    resources = run.add_argument_group("workers and resource limits")
    resources.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent cell workers.",
    )
    resources.add_argument(
        "--cores-per-worker",
        type=int,
        default=DEFAULT_CORES_PER_WORKER,
        help="Generation/engine cores assigned to each worker.",
    )
    resources.add_argument(
        "--amplicol-build-jobs",
        type=int,
        default=1,
        metavar="JOBS",
        help=(
            "Make jobs used only for original AmpliCol builds (default: 1). "
            "Keep this at 1 for the maintained legacy checkout, whose "
            "generator target is not parallel-safe."
        ),
    )
    resources.add_argument(
        "--generation-time-limit",
        type=_positive_finite_float,
        default=DEFAULT_GENERATION_LIMIT_SECONDS,
        metavar="SECONDS",
        help=(
            "Generation-only cap per process tree; preparation and legacy "
            "generation count, adaptive profiling does not."
        ),
    )
    resources.add_argument(
        "--ram-limit",
        type=_positive_int,
        default=DEFAULT_RAM_BYTES,
        metavar="BYTES",
        help="Decimal process-tree RAM ceiling per worker (30 GB = 30000000000).",
    )
    resources.add_argument(
        "--campaign-ram-limit",
        type=_positive_int,
        metavar="BYTES",
        help=(
            "Optional decimal RAM ceiling across all concurrent worker trees. "
            "The controller enforces it conservatively by also limiting each "
            "worker to this value divided by --workers; the effective worker "
            "cap is the smaller of that share and --ram-limit."
        ),
    )
    resources.add_argument(
        "--attempt-output-limit",
        type=_positive_int,
        default=DEFAULT_ATTEMPT_OUTPUT_LIMIT_BYTES,
        metavar="BYTES",
        help=(
            "Maximum size of each supervised attempt log/output stream "
            f"(default: {DEFAULT_ATTEMPT_OUTPUT_LIMIT_BYTES} bytes). A breach "
            "terminates that worker tree and records an error."
        ),
    )
    resources.add_argument(
        "--minimum-free-disk",
        type=_positive_int,
        default=DEFAULT_MINIMUM_FREE_DISK_BYTES,
        metavar="BYTES",
        help=(
            "Minimum free bytes reserved on the campaign artifact volume "
            f"(default: {DEFAULT_MINIMUM_FREE_DISK_BYTES}). Crossing the "
            "reserve terminates the active worker tree and records an error."
        ),
    )
    resources.add_argument(
        "--worker-wall-limit",
        type=_positive_finite_float,
        default=DEFAULT_WORKER_WALL_LIMIT_SECONDS,
        metavar="SECONDS",
        help=(
            "Hard total process-tree wall-time cap per worker; defaults to "
            "3600 seconds and bounds generation, profiling, validation, and "
            "every descendant command."
        ),
    )
    resources.add_argument(
        "--resource-sample-interval",
        type=_positive_finite_float,
        default=1.0,
        metavar="SECONDS",
        help="Process-tree CPU/RAM/dashboard sampling interval.",
    )
    resources.add_argument(
        "--termination-grace",
        type=_nonnegative_finite_float,
        default=5.0,
        metavar="SECONDS",
        help=(
            "Grace after SIGTERM before a remaining process tree is killed; "
            "cancelled attempts are then sealed with compact diagnostics. "
            "Output/free-disk breaches escalate immediately for host safety."
        ),
    )
    resources.add_argument(
        "--allow-oversubscription",
        action="store_true",
        help="Permit workers x cores to exceed logical CPU count.",
    )
    resources.add_argument(
        "--allow-engine-parallelism",
        action="store_true",
        help=(
            "Allow concurrent engine preparation where the scheduler normally "
            "serializes it."
        ),
    )
    resources.add_argument(
        "--original-amplicol",
        type=Path,
        metavar="PATH_TO_COMPLETE_CHECKOUT",
        help=(
            "Clean, complete original-AmpliCol checkout containing the profiling "
            "interface from PR #12 (currently amplicol_with_patches; compatible "
            "upstream revisions are accepted after merge). Required when the "
            "direct selection or automatically added diagnostic dependency "
            "includes the amplicol engine. When no checkout is configured, "
            "recurrence remains available as the authority for compiled/eager "
            "work. Pass "
            "--no-dependencies-added for an explicitly baseline-free selection. "
            "Omitted or '*' engine selection means all engines."
        ),
    )
    resources.add_argument(
        "--madgraph",
        type=Path,
        metavar="PATH_TO_MADGRAPH_INSTALLATION",
        help=(
            "MadGraph installation containing an executable bin/mg5_aMC. "
            "Installed campaign copies default to the path saved in "
            ".pyamplicol-madgraph."
        ),
    )
    profiling = run.add_argument_group("profiling hyperparameters")
    profiling.add_argument(
        "--target-measurement-duration",
        type=_positive_finite_float,
        default=DEFAULT_TARGET_RUNTIME_SECONDS,
        metavar="SECONDS",
        help="Target accumulated profiling duration for each runtime.",
    )
    profiling.add_argument(
        "--minimum-samples",
        type=int,
        default=DEFAULT_MINIMUM_SAMPLES,
        help="Minimum independent timed profiling blocks (at least 5).",
    )
    profiling.add_argument(
        "--warmups",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="Untimed runtime warmup calls before calibration.",
    )
    profiling.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Phase-space points in each runtime batch.",
    )
    behavior = run.add_argument_group("reuse and display")
    behavior.add_argument(
        "--no-dependencies-added",
        action="store_true",
        help=(
            "Do not automatically add availability-optional numerical "
            "authorities outside the direct selection. Hard construction and "
            "selector/provider dependencies are always retained; explicitly "
            "selected authorities still run first."
        ),
    )
    behavior.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "Create a fresh generation+measurement attempt even when a "
            "complete current exists (including one accepted through "
            "--continue-across-revisions). Disposable workspaces are compacted "
            "after preserving bounded diagnostics by default."
        ),
    )
    behavior.add_argument(
        "--rerun-failed",
        action="store_true",
        help=(
            "Create fresh attempts only for selected cells whose latest "
            "terminal outcome is a non-cap failure. Valid, unseen, static-N/A, "
            "and authenticated capped cells are excluded. Missing or historical "
            "numerical authorities are added as dependency-only work unless "
            "--no-dependencies-added is given. May be combined with "
            "--rerun-capped, but not --force-refresh."
        ),
    )
    behavior.add_argument(
        "--rerun-capped",
        action="store_true",
        help=(
            "Create fresh attempts only for selected cells with an "
            "authenticated generation, RAM, worker, profiling, or validation "
            "cap. May be combined with --rerun-failed, but not "
            "--force-refresh."
        ),
    )
    behavior.add_argument(
        "--cleanup-artifacts",
        action="store_true",
        help=(
            "Compatibility spelling for the default compact-retention policy: "
            "preserve results, provenance, progress, and bounded diagnostics "
            "while removing disposable build/reference workspaces."
        ),
    )
    behavior.add_argument(
        "--retain-workspaces",
        action="store_true",
        help=(
            "Opt in to retaining full legacy, MadGraph, build, failed, and "
            "obsolete attempt workspaces for debugging. Output and free-disk "
            "safety limits remain enforced."
        ),
    )
    behavior.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop new dispatch after the first terminal non-success, cancel "
            "other live workers, complete multiplicities in n=1, n=2, ... "
            "waves, retain bounded failed/cancelled diagnostics, exit "
            "nonzero, and atomically write "
            "campaign_summary_ids/fail_fast_failure.log with exact diagnostic "
            "and reproduction paths. Cancelled outcomes caused by the stop are "
            "not treated as a second failure. With --dry-run no report is "
            "created."
        ),
    )
    behavior.add_argument(
        "--continue-across-revisions",
        action="store_true",
        help=(
            "Explicitly keep structurally valid completed selected cells from "
            "older pyAmpliCol source revisions while filling missing cells "
            "with this clean active build. Historical currents never satisfy "
            "dependencies required by newly run cells: those dependencies are "
            "planned again at the active revision. The policy is persisted so "
            "later inspect and refresh-pdf commands automatically merge the "
            "per-cell currents and report every exact source cohort/count."
        ),
    )
    behavior.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print direct cells, dependency closure, reuse counts, and "
            "copy-paste public CLI recipes without launching workers."
        ),
    )
    behavior.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run without the interactive Ratatui terminal dashboard.",
    )

    inspect = commands.add_parser(
        "inspect",
        parents=[selectors, output],
        help="Inspect coverage and AmpliCol-relative summary statistics.",
        description=(
            "Summarize lightweight current metadata and compute candidate / "
            "AmpliCol generation and outer-wall runtime multipliers. Dedicated "
            "evaluator-total multipliers are emitted only when both sides "
            "expose that clock; original AmpliCol normally does not. "
            "Recurrence core is reported separately as an absolute statistic. "
            "If the latest real run enabled cross-revision continuation, the "
            "persisted mixed current set and its exact source cohorts are used."
        ),
        epilog=(
            "Lower multipliers are better. The weighted average is exactly "
            "sum(candidate) / sum(AmpliCol). Missing, capped, incompatible, "
            "baseline-less, and unexposed clocks are excluded and counted. "
            "Outer wall, evaluator-total, and recurrence core are never copied "
            "or derived from one another.\n\n" + STEERING_GUIDE
        ),
        formatter_class=HelpFormatter,
    )
    inspect.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Human coloured tables or stable uncoloured JSON.",
    )

    refresh = commands.add_parser(
        "refresh-pdf",
        parents=[output],
        help="Rebuild every table and atomically replace the report PDF.",
        description=(
            "Capture one stable current snapshot, rebuild every "
            "result JSON and TeX table, compile in a fresh directory, and "
            "atomically install pyAmpliCol.pdf. Overfull-box diagnostics are "
            "reported by LaTeX but are non-fatal here; compilation errors and "
            "unresolved references remain fatal. Compilation output streams "
            "live by default; --quiet suppresses it. The final absolute PDF "
            "path is printed on success. A prior real run with "
            "--continue-across-revisions automatically publishes the mixed "
            "per-cell current set and reports every exact source cohort/count."
        ),
        epilog=STEERING_GUIDE,
        formatter_class=HelpFormatter,
    )
    refresh.add_argument(
        "--expected-page-count",
        type=int,
        default=None,
        help=(
            "Optional stable page-count assertion. Omit it to accept legitimate "
            "layout changes or a copied campaign profile with a different name."
        ),
    )
    refresh.add_argument(
        "--pdf-timeout",
        type=_positive_finite_float,
        default=DEFAULT_PDF_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="Direct latexmk process timeout.",
    )
    refresh.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress the live artifact-scan progress display and latexmk "
            "stdout/stderr. The final publication summary and absolute PDF "
            "path are still printed."
        ),
    )
    sections = refresh.add_mutually_exclusive_group()
    sections.add_argument(
        "--list-sections",
        action="store_true",
        help=(
            "List stable section IDs and exit without reading campaign "
            "artifacts or rebuilding the PDF."
        ),
    )
    sections.add_argument(
        "--remove-sections",
        nargs="+",
        metavar="SECTION_ID",
        help=(
            "Omit one or more listed top-level sections from this PDF build. "
            "Measurements, caches, tables, and the source template are unchanged."
        ),
    )

    snapshot = commands.add_parser(
        "dashboard-snapshot",
        parents=[output],
        help="Capture a deterministic fixture or read-only live lease frame.",
        description=(
            "Render a Ratatui headless frame for visual review and automated "
            "style/value assertions. The default is a deterministic synthetic "
            "fixture. --live instead reads only compact non-stale JSON leases "
            "from the manual profile's ignored coordination directory: it "
            "does not inspect results, artifacts, source identity, or Git."
        ),
        epilog=(
            "Selection rules:\n"
            "  Without --live, output is deterministic and independent of "
            "campaign state.\n"
            "  With --live, the newest active lease is selected unless "
            "--instance names\n"
            "  an exact ID or unique prefix. Active same-source peer workers "
            "are shown\n"
            "  as peer rows; overview totals remain the selected invocation's "
            "own atomic\n"
            "  counters so overlapping selections are never guessed or "
            "double-counted.\n\n"
            "Examples:\n"
            "  steer_performance_campaign.py dashboard-snapshot "
            "--width 80 --height 24\n"
            "  steer_performance_campaign.py dashboard-snapshot "
            "--width 160 --height 48 --cells-json\n"
            "  steer_performance_campaign.py dashboard-snapshot --live "
            "--width 160 --height 48\n"
            "  steer_performance_campaign.py dashboard-snapshot --live "
            "--instance 7f91c2 --stale-after 30"
        ),
        formatter_class=HelpFormatter,
    )
    snapshot.add_argument("--width", type=int, default=120, help="Frame columns.")
    snapshot.add_argument("--height", type=int, default=36, help="Frame rows.")
    snapshot.add_argument(
        "--cells-json",
        action="store_true",
        help=(
            "Emit ANSI-free styled-cell JSON instead of the terminal frame. "
            "Use --no-color or NO_COLOR to zero the Ratatui style metadata."
        ),
    )
    snapshot.add_argument(
        "--live",
        action="store_true",
        help=(
            "Capture actual running state from lightweight lease JSON only. "
            "No measurement/current/artifact/source/Git data is read, and "
            "nothing is written. Without this flag the synthetic fixture is "
            "rendered."
        ),
    )
    snapshot.add_argument(
        "--instance",
        "--instance-id",
        metavar="ID_OR_PREFIX",
        help=(
            "With --live, select one invocation by its exact instance ID or a "
            "unique prefix. Omit to select the most recently updated active "
            "lease."
        ),
    )
    snapshot.add_argument(
        "--stale-after",
        type=_positive_finite_float,
        default=DEFAULT_WORKER_STALE_SECONDS,
        metavar="SECONDS",
        help=(
            "With --live, ignore leases older than this positive interval. "
            "Increase only when resource sampling is intentionally slower."
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    profile: str = PROFILE,
    docs_dir: Path | None = None,
    installed: bool = False,
    launcher_path_checked: bool = False,
) -> int:
    global _ACTIVE_PROFILE

    selected_profile = validate_profile_name(profile)
    _ACTIVE_PROFILE = selected_profile
    parser = build_parser()
    invocation_arguments = tuple(sys.argv[1:] if argv is None else argv)
    invocation_program = sys.argv[0] if argv is None else parser.prog
    arguments = parser.parse_args(invocation_arguments)
    arguments._campaign_invocation_command = _shell_join(
        (invocation_program, *invocation_arguments)
    )
    root = (
        Path.cwd().resolve(strict=False)
        if repo_root is None
        else repo_root.expanduser().resolve(strict=False)
    )
    if installed and docs_dir is None:
        raise ValueError("installed campaign mode requires its destination directory")
    campaign_docs = (
        root / "docs/performance_reports" / selected_profile
        if docs_dir is None
        else docs_dir
    )
    json_output = arguments.command == "inspect" and arguments.format == "json"
    palette = Palette(_color_enabled(arguments, json_output=json_output))
    campaign_lock: int | None = None
    try:
        _require_launcher_working_directory_identity(
            launcher_path_checked=launcher_path_checked,
        )
        if arguments.command == "refresh-pdf" and arguments.list_sections:
            _print_report_sections(palette)
            return 0
        paths = _campaign_report_paths(root, campaign_docs)
        campaign_lock = _acquire_campaign_directory_lock(paths.docs_dir)
        dashboard_disabled = _configure_dashboard_capability(
            arguments,
            installed=installed,
        )
        if (
            dashboard_disabled
            and bool(getattr(sys.stdin, "isatty", lambda: False)())
            and bool(getattr(sys.stdout, "isatty", lambda: False)())
        ):
            print(
                palette.warning(
                    "Optional Ratatui bindings are unavailable; continuing with "
                    "--no-dashboard."
                ),
                file=sys.stderr,
            )
        if arguments.command == "dashboard-snapshot":
            if arguments.instance is not None and not arguments.live:
                raise ManualCampaignError("--instance requires --live")
            state = (
                _live_dashboard_snapshot(
                    paths.coordination_root,
                    instance=arguments.instance,
                    stale_after_seconds=arguments.stale_after,
                )
                if arguments.live
                else _snapshot_fixture()
            )
            result = render_dashboard_frame(
                state,
                width=arguments.width,
                height=arguments.height,
                cells=arguments.cells_json,
                color=palette.enabled,
            )
            if arguments.cells_json:
                json.dump(result, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
            else:
                print(result)
            return 0

        service = ReportService(paths, portable_current_results=True)
        service.bind_measurement_lineage(None)
        service.bind_original_amplicol_seed(None)
        source = (
            installed_source_identity()
            if installed
            else lightweight_source_identity(root)
        )

        if arguments.command == "refresh-pdf":
            return _refresh_pdf(
                arguments,
                service=service,
                source=source,
                palette=palette,
            )

        _selection, cells = selection_from_arguments(arguments)
        if arguments.command == "inspect":
            source_policy = recorded_measurement_source_policy(
                service,
                checkout_revision=source.revision,
            )
            payload = _inspect_payload(
                service,
                cells,
                source_revision=source_policy.source_revision,
                renderer_revision=source.revision,
                accept_historical_source=(source_policy.continue_across_revisions),
            )
            if arguments.format == "json":
                json.dump(payload, sys.stdout, indent=2, sort_keys=True)
                sys.stdout.write("\n")
            else:
                _print_inspect(payload, palette)
            return 0
        if arguments.command == "run":
            return _run_campaign(
                arguments,
                repo_root=root,
                service=service,
                source=source,
                cells=cells,
                palette=palette,
                installed=installed,
            )
        parser.error(f"unsupported command {arguments.command!r}")
    except KeyboardInterrupt:
        print(
            palette.warning("Interrupted cleanly; completed currents were preserved."),
            file=sys.stderr,
        )
        return 130
    except (ManualCampaignError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(palette.failure(f"error: {error}"), file=sys.stderr)
        return 2
    finally:
        if campaign_lock is not None:
            _release_campaign_directory_lock(campaign_lock)
    return 2


__all__ = [
    "Comparison",
    "DashboardState",
    "LightweightCurrent",
    "ManualCampaignError",
    "ReproductionRecipe",
    "ReproductionSettings",
    "WorkerView",
    "build_parser",
    "comparison_statistics",
    "lightweight_current",
    "main",
    "render_dashboard_frame",
    "reproduction_recipe",
    "selection_from_arguments",
]
