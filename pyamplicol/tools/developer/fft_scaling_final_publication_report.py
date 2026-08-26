#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Merge the fresh n=2..9 scaling campaign with its MadGraph series.

This is intentionally a narrow publication boundary.  It does not splice old
pyAmpliCol measurements into the fresh campaign: the campaign cells and policy
are retained as one source, while the independently measured MadGraph cells are
attached as an external series after their resource, event, and numerical
provenance has been authenticated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.developer import fft_madgraph_selected_runtime as madgraph  # noqa: E402
from tools.developer import fft_scaling_selected_scalar_report as selected  # noqa: E402

RESULTS = ROOT / "IMPLEMENTATION_DOCS" / "RESULTS" / "fft-scaling-study"
DATA = RESULTS / "data"

CAMPAIGN_KIND = "pyamplicol-fullcolor-fft-scaling-study"
FINAL_KIND = "pyamplicol-fullcolor-fft-scaling-final-publication"
SCHEMA_VERSION = 1
FINAL_MULTIPLICITIES = tuple(range(2, 10))
MADGRAPH_MAX_MEASURED_MULTIPLICITY = 6
FAMILY_MODES = {
    "gg": (
        "reference-fft",
        "amplicol",
        "recurrence-direct",
        "recurrence-fft",
        "otf-direct",
        "otf-fft",
    ),
    "ddbar": (
        "amplicol",
        "recurrence-direct",
        "recurrence-fft",
        "otf-direct",
        "otf-fft",
    ),
}
MADGRAPH_FAMILY_MAX_MEASURED_MULTIPLICITY = (
    madgraph.protocol_measured_multiplicity_limits()
)
SOURCE_MODE = {"gg": "reference-fft", "ddbar": "recurrence-fft"}
SUM_SOURCE_MODE = {"gg": "recurrence-fft", "ddbar": "recurrence-fft"}
CANDIDATE_MODE_CONTRACT = {
    "recurrence-direct": ("recurrence", "direct"),
    "recurrence-fft": ("recurrence", "symmetric-group-fft"),
    "otf-direct": ("on-the-fly", "direct"),
    "otf-fft": ("on-the-fly", "symmetric-group-fft"),
}
METRIC_KEYS = (
    "generation_seconds",
    "warm_seconds_per_point",
    "max_rss_kib",
)
EXPECTED_EXPRESSIONS = {
    "gg": "g g > g g + (n-2)*g",
    "ddbar": "d d~ > d d~ + (n-2)*g",
}
EXPECTED_RATIO_REFERENCES = {"gg": "reference-fft", "ddbar": "amplicol"}
MAX_GENERATION_SECONDS = 3600.0
MAX_RUNTIME_SECONDS = 3600.0
MAX_RSS_GIB = 30.0
MAX_RSS_KIB = MAX_RSS_GIB * 1024**2
POINT_COUNT = 10
WARM_SAMPLE_COUNT = madgraph.WARM_SAMPLE_COUNT


class PublicationMergeError(RuntimeError):
    """The two inputs cannot be merged without weakening their contracts."""


def _expected_madgraph_iden(family: str, n: int) -> int:
    if family == "gg":
        return 256 * math.factorial(n)
    if family == "ddbar":
        return 36 * math.factorial(n - 2)
    raise PublicationMergeError(f"unsupported MadGraph family {family!r}")


def _source_mode(family: str, helicity_workload: str) -> str:
    modes = SUM_SOURCE_MODE if helicity_workload == "sum" else SOURCE_MODE
    try:
        return modes[family]
    except KeyError as error:
        raise PublicationMergeError(
            f"unsupported MadGraph family {family!r}"
        ) from error


def _madgraph_normalization_factor(
    family: str, n: int, *, helicity_workload: str = "fixed"
) -> float:
    if helicity_workload == "sum":
        return 1.0
    if helicity_workload != "fixed":
        raise PublicationMergeError(
            f"unsupported helicity workload {helicity_workload!r}"
        )
    return 1.0 if family == "gg" else 1.0 / _expected_madgraph_iden(family, n)


def _helicity_workload(policy: Mapping[str, Any]) -> str:
    measurement = _mapping(
        policy.get("measurement"), context="campaign measurement policy"
    )
    declared = measurement.get("helicity_workload")
    if declared is None:
        workload = "sum" if measurement.get("warm_helicity_sum") is True else "fixed"
    else:
        workload = str(declared)
    if workload not in {"fixed", "sum"}:
        raise PublicationMergeError("campaign helicity workload is unsupported")
    if workload == "sum" and (
        measurement.get("warm_fixed_helicity") is not False
        or measurement.get("warm_helicity_sum") is not True
    ):
        raise PublicationMergeError("summed campaign helicity markers are incompatible")
    if workload == "fixed" and (
        measurement.get("warm_fixed_helicity") is not True
        or measurement.get("warm_helicity_sum") is True
    ):
        raise PublicationMergeError("fixed campaign helicity markers are incompatible")
    return workload


def _overlay_helicity_workload(policy: Mapping[str, Any]) -> str:
    declared = policy.get("helicity_workload")
    if declared is None:
        workload = "sum" if policy.get("warm_helicity_sum") is True else "fixed"
    else:
        workload = str(declared)
    if workload not in {"fixed", "sum"}:
        raise PublicationMergeError("MadGraph helicity workload is unsupported")
    if workload == "sum" and (
        policy.get("warm_fixed_helicity") is not False
        or policy.get("warm_helicity_sum") is not True
    ):
        raise PublicationMergeError("summed MadGraph helicity markers are incompatible")
    if workload == "fixed" and (
        policy.get("warm_fixed_helicity") is not True
        or policy.get("warm_helicity_sum") is True
    ):
        raise PublicationMergeError("fixed MadGraph helicity markers are incompatible")
    return workload


@dataclass(frozen=True, slots=True)
class Document:
    key: str
    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def display_path(self) -> str:
        try:
            return str(self.path.relative_to(ROOT))
        except ValueError:
            return str(self.path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-report",
        type=Path,
        default=(
            RESULTS / "raw" / "runs" / "final-publication-scalar-n2-n9" / "report.json"
        ),
        help="fresh all-helicity pyAmpliCol/Reference/AmpliCol report",
    )
    parser.add_argument(
        "--madgraph-overlay",
        type=Path,
        required=True,
        help="completed live MadGraph runtime-series overlay",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA / "campaign-report-scalar-selected-n2-n9-final.json",
        help="final publication report",
    )
    return parser


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PublicationMergeError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _load_document(key: str, path: Path) -> Document:
    path = path.expanduser().resolve(strict=False)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationMergeError(f"cannot read {key} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PublicationMergeError(f"{key} must be a JSON object")
    return Document(key=key, path=path, sha256=_sha256_bytes(raw), payload=payload)


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicationMergeError(f"{context} must be an object")
    return value


def _measurement_host(value: object, *, context: str) -> dict[str, Any]:
    try:
        return madgraph.validate_measurement_host(value, context=context)
    except madgraph.SelectedMadGraphError as error:
        raise PublicationMergeError(str(error)) from error


def _sequence(value: object, *, context: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise PublicationMergeError(f"{context} must be an array")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PublicationMergeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PublicationMergeError(f"{context} must be a finite number")
    return result


def _positive_number(value: object, *, context: str) -> float:
    result = _finite_number(value, context=context)
    if result <= 0.0:
        raise PublicationMergeError(f"{context} must be positive")
    return result


def _process_expression(family: str, n: int) -> str:
    extra = " ".join("g" for _ in range(n - 2))
    base = "g g > g g" if family == "gg" else "d d~ > d d~"
    return base + (f" {extra}" if extra else "")


def _relative_error(left: float, right: float) -> float:
    if left == right:
        return 0.0
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _validate_limit(
    raw_limits: Mapping[str, Any], key: str, expected_limit: float
) -> None:
    limit = _mapping(raw_limits.get(key), context=f"cell_admission_limits.{key}")
    if (
        limit.get("operator") != "<"
        or _finite_number(
            limit.get("limit"), context=f"cell_admission_limits.{key}.limit"
        )
        != expected_limit
    ):
        raise PublicationMergeError(
            f"cell_admission_limits.{key} must be strictly < {expected_limit:g}"
        )


def _validate_campaign_policy(document: Document) -> Mapping[str, Any]:
    report = document.payload
    if report.get("kind") != CAMPAIGN_KIND or report.get("schema_version") != 1:
        raise PublicationMergeError("campaign report has the wrong kind or schema")
    policy = _mapping(report.get("policy"), context="campaign policy")
    if policy.get("kind") != CAMPAIGN_KIND or policy.get("schema_version") != 1:
        raise PublicationMergeError("campaign policy has the wrong kind or schema")
    if policy.get("final_state_multiplicities") != list(FINAL_MULTIPLICITIES):
        raise PublicationMergeError("campaign must physically cover only n=2..9")
    if policy.get("total_external_particles") != [n + 2 for n in FINAL_MULTIPLICITIES]:
        raise PublicationMergeError("campaign total-external-particle range is wrong")
    if policy.get("fft_enabled") is not True:
        raise PublicationMergeError("campaign policy must record --fft")
    contractions = _mapping(
        policy.get("selected_pyamplicol_color_contractions"),
        context="selected pyAmpliCol contractions",
    )
    expected_contractions = {
        "recurrence": ["direct", "symmetric-group-fft"],
        "on-the-fly": ["direct", "symmetric-group-fft"],
    }
    if dict(contractions) != expected_contractions:
        raise PublicationMergeError(
            "campaign must select direct and symmetric-group FFT for recurrence "
            "and on-the-fly only"
        )

    raw_families = _mapping(
        policy.get("process_families"), context="campaign process families"
    )
    if set(raw_families) != set(FAMILY_MODES):
        raise PublicationMergeError("campaign must contain exactly gg and ddbar")
    for family, expected_modes in FAMILY_MODES.items():
        family_policy = _mapping(
            raw_families.get(family), context=f"campaign process family {family}"
        )
        if family_policy.get("expression") != EXPECTED_EXPRESSIONS[family]:
            raise PublicationMergeError(f"campaign {family} expression is wrong")
        if family_policy.get("ratio_reference") != EXPECTED_RATIO_REFERENCES[family]:
            raise PublicationMergeError(f"campaign {family} ratio baseline is wrong")
        if family_policy.get("modes") != list(expected_modes):
            raise PublicationMergeError(
                f"campaign {family} must select exactly {list(expected_modes)}"
            )

    measurement = _mapping(
        policy.get("measurement"), context="campaign measurement policy"
    )
    helicity_workload = _helicity_workload(policy)
    expected_measurement = {
        "color_accuracy": "full",
        "n_definition": "final-state-particle-count",
        "generation_helicity_coverage": "all",
        "warm_fixed_helicity": helicity_workload == "fixed",
        "warm_benchmark_batch_size": 128,
        "warm_sample_count": WARM_SAMPLE_COUNT,
        "compiled_fft_enabled": False,
        "memory_policy": "per-cell-strictly-below-publication-ceiling",
    }
    for key, expected in expected_measurement.items():
        if measurement.get(key) != expected:
            raise PublicationMergeError(
                f"campaign measurement policy has incompatible {key}"
            )
    if (
        _finite_number(
            measurement.get("requested_memory_ceiling_gib"),
            context="requested memory ceiling",
        )
        != MAX_RSS_GIB
        or _finite_number(
            measurement.get("memory_watchdog_gib"), context="memory watchdog"
        )
        != MAX_RSS_GIB
    ):
        raise PublicationMergeError("campaign memory policy must use the 30-GiB cap")
    limits = _mapping(
        measurement.get("cell_admission_limits"), context="cell admission limits"
    )
    _validate_limit(limits, "generation_seconds", MAX_GENERATION_SECONDS)
    _validate_limit(limits, "runtime_seconds", MAX_RUNTIME_SECONDS)
    _validate_limit(limits, "peak_rss_gib", MAX_RSS_GIB)

    run_root_raw = policy.get("run_root")
    if not isinstance(run_root_raw, str) or not run_root_raw:
        raise PublicationMergeError("campaign policy.run_root is required")
    if Path(run_root_raw).expanduser().resolve(strict=False) != document.path.parent:
        raise PublicationMergeError("campaign report is outside its recorded run_root")
    return policy


def campaign_policy_is_publication_profile(report: Mapping[str, Any]) -> bool:
    """Return whether a report declares the exact canonical publication policy.

    Cell and overlay authentication deliberately remain in the strict merger;
    callers use this only to decide whether that merger is the required render
    boundary.
    """

    try:
        payload = dict(report)
        policy = payload.get("policy")
        run_root = policy.get("run_root") if isinstance(policy, Mapping) else None
        if not isinstance(run_root, str) or not run_root:
            return False
        _validate_campaign_policy(
            Document(
                key="campaign policy probe",
                path=Path(run_root) / "report.json",
                sha256="0" * 64,
                payload=payload,
            )
        )
    except PublicationMergeError:
        return False
    return True


def _validate_helicity(value: object, *, n: int, context: str) -> tuple[int, ...]:
    raw = _sequence(value, context=context)
    if len(raw) != n + 2 or any(
        not isinstance(item, int) or isinstance(item, bool) or item not in (-1, 1)
        for item in raw
    ):
        raise PublicationMergeError(f"{context} is invalid")
    return tuple(int(item) for item in raw)


def _validate_points(
    value: object,
    *,
    context: str,
    expected_count: int = POINT_COUNT,
) -> tuple[float, ...]:
    raw = _sequence(value, context=context)
    points = tuple(_finite_number(item, context=context) for item in raw)
    if len(points) != expected_count or not any(point != 0.0 for point in points):
        raise PublicationMergeError(
            f"{context} must contain exactly {expected_count} values with a "
            "nonzero result"
        )
    return points


def _validate_campaign_cells(
    document: Document, *, helicity_workload: str = "fixed"
) -> Counter[str]:
    raw_cells = _mapping(document.payload.get("cells"), context="campaign cells")
    if set(raw_cells) != set(FAMILY_MODES):
        raise PublicationMergeError("campaign cells must contain exactly gg and ddbar")
    status_counts: Counter[str] = Counter()
    for family, expected_modes in FAMILY_MODES.items():
        family_cells = _mapping(
            raw_cells.get(family), context=f"campaign cells.{family}"
        )
        if set(family_cells) != set(expected_modes):
            raise PublicationMergeError(
                f"campaign cells.{family} has an unexpected mode set"
            )
        for mode in expected_modes:
            mode_cells = _mapping(
                family_cells.get(mode), context=f"campaign cells.{family}.{mode}"
            )
            if set(mode_cells) != {str(n) for n in FINAL_MULTIPLICITIES}:
                raise PublicationMergeError(
                    f"campaign cells.{family}.{mode} must describe n=2..9"
                )
            frontier_seen = False
            for n in FINAL_MULTIPLICITIES:
                context = f"campaign cells.{family}.{mode}.{n}"
                cell = _mapping(mode_cells.get(str(n)), context=context)
                status = cell.get("status")
                if status not in {"measured", "failed", "skipped"}:
                    raise PublicationMergeError(f"{context}.status is unsupported")
                if status == "measured" and frontier_seen:
                    raise PublicationMergeError(
                        f"{context} resumes measurement after its availability frontier"
                    )
                if status == "failed" and frontier_seen:
                    raise PublicationMergeError(
                        f"{context} repeats a failure after its availability frontier"
                    )
                if (
                    status == "failed"
                    and cell.get("censors_higher_multiplicities") is not True
                ):
                    raise PublicationMergeError(
                        f"{context} records a non-censoring failure in a "
                        "completed campaign"
                    )
                if status == "skipped" and not isinstance(
                    cell.get("censors_higher_multiplicities"), bool
                ):
                    raise PublicationMergeError(
                        f"{context} must state whether its skip censors higher n"
                    )
                if status != "measured" and not frontier_seen:
                    frontier_seen = True
                for key, expected in (
                    ("family", family),
                    ("mode", mode),
                    ("n", n),
                    ("total_external", n + 2),
                    ("process", _process_expression(family, n)),
                    ("color_accuracy", "full"),
                ):
                    if cell.get(key) != expected:
                        raise PublicationMergeError(f"{context}.{key} is wrong")
                label = cell.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise PublicationMergeError(f"{context}.label is required")
                candidate_contract = CANDIDATE_MODE_CONTRACT.get(mode)
                if candidate_contract is not None:
                    expected_execution, expected_contraction = candidate_contract
                    for key, expected in (
                        ("execution_mode", expected_execution),
                        ("color_contraction", expected_contraction),
                        ("generation_helicity_coverage", "all"),
                        ("warm_fixed_helicity", helicity_workload == "fixed"),
                    ):
                        if cell.get(key) != expected:
                            raise PublicationMergeError(f"{context}.{key} is wrong")
                    if helicity_workload == "sum" and cell.get(
                        "warm_helicity_sum"
                    ) is not True:
                        raise PublicationMergeError(
                            f"{context}.warm_helicity_sum is wrong"
                        )
                if status == "measured":
                    metrics = _mapping(
                        cell.get("metrics"), context=f"{context}.metrics"
                    )
                    values = {
                        key: _positive_number(
                            metrics.get(key), context=f"{context}.metrics.{key}"
                        )
                        for key in METRIC_KEYS
                    }
                    if values["generation_seconds"] >= MAX_GENERATION_SECONDS:
                        raise PublicationMergeError(
                            f"{context} violates the strict one-hour generation cap"
                        )
                    if values["max_rss_kib"] >= MAX_RSS_KIB:
                        raise PublicationMergeError(
                            f"{context} violates the strict 30-GiB RSS cap"
                        )
                    _validate_helicity(cell.get("helicity"), n=n, context=context)
                    _validate_points(
                        cell.get("point_values"),
                        context=context,
                        expected_count=1 if mode == "amplicol" else POINT_COUNT,
                    )
                    event_paths = _sequence(
                        cell.get("event_paths"), context=f"{context}.event_paths"
                    )
                    if len(event_paths) != POINT_COUNT or any(
                        not isinstance(path, str) or not path for path in event_paths
                    ):
                        raise PublicationMergeError(
                            f"{context}.event_paths must contain ten paths"
                        )
                    numerical = cell.get("numerical")
                    if isinstance(numerical, Mapping):
                        available = numerical.get("available")
                        if available is True and numerical.get("passes") is not True:
                            raise PublicationMergeError(
                                f"{context} failed its numerical comparison"
                            )
                else:
                    reason = cell.get("failure_reason")
                    if not isinstance(reason, str) or not reason.strip():
                        raise PublicationMergeError(
                            f"{context}.failure_reason is required"
                        )
                status_counts[str(status)] += 1

    expected_status = (
        "complete-with-failures" if status_counts["failed"] else "complete"
    )
    if document.payload.get("status") != expected_status:
        raise PublicationMergeError("campaign status disagrees with its cells")
    if document.payload.get("failure_count") != status_counts["failed"]:
        raise PublicationMergeError("campaign failure_count disagrees with its cells")
    return status_counts


def _validate_madgraph_policy(
    document: Document, *, helicity_workload: str = "fixed"
) -> None:
    report = document.payload
    if (
        report.get("kind") != selected.RUNTIME_SERIES_OVERLAY_KIND
        or report.get("schema_version") != SCHEMA_VERSION
    ):
        raise PublicationMergeError("MadGraph overlay has the wrong kind or schema")
    policy = _mapping(report.get("policy"), context="MadGraph policy")
    if _overlay_helicity_workload(policy) != helicity_workload:
        raise PublicationMergeError(
            "MadGraph overlay helicity workload differs from the campaign"
        )
    expected = {
        "final_state_multiplicities": list(FINAL_MULTIPLICITIES),
        "process_families": list(FAMILY_MODES),
        "point_validation_count": POINT_COUNT,
        "warm_timed_point_index": 1,
        "warm_sample_count": WARM_SAMPLE_COUNT,
        "watchdog_enforced_per_generation_build_and_runtime": True,
        "generation_helicity_coverage": "all",
        "warm_fixed_helicity": helicity_workload == "fixed",
        "maximum_measured_multiplicity": MADGRAPH_MAX_MEASURED_MULTIPLICITY,
        "family_maximum_measured_multiplicity": (
            MADGRAPH_FAMILY_MAX_MEASURED_MULTIPLICITY
        ),
        "higher_multiplicity_policy": "not-applicable-protocol-scope",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise PublicationMergeError(f"MadGraph policy has incompatible {key}")
    timeout = _positive_number(
        policy.get("generation_timeout_seconds"),
        context="MadGraph generation timeout",
    )
    if timeout >= MAX_GENERATION_SECONDS:
        raise PublicationMergeError(
            "MadGraph generation timeout must be below one hour"
        )
    memory = _positive_number(
        policy.get("outer_memory_watchdog_gib"), context="MadGraph memory watchdog"
    )
    if memory > MAX_RSS_GIB:
        raise PublicationMergeError("MadGraph memory watchdog exceeds 30 GiB")
    metric_scope = _mapping(policy.get("metric_scope"), context="MadGraph metric scope")
    if set(metric_scope) != set(METRIC_KEYS) or any(
        not isinstance(value, str) or not value.strip()
        for value in metric_scope.values()
    ):
        raise PublicationMergeError(
            "MadGraph metric scope must define all three metrics"
        )


def _madgraph_protocol_scope_categories(family: str, n: int, limit: int) -> set[str]:
    categories = {f"protocol-scope-n>{limit}"}
    if family == "gg" and n == 6:
        categories.add("protocol-scope-pure-gluon-n6")
    return categories


def _resolve_recorded_path(raw: object, *, context: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PublicationMergeError(f"{context} must be a nonempty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=False)


def _event_identity(cell: Mapping[str, Any], *, context: str) -> tuple[Path, ...]:
    raw_paths = _sequence(cell.get("event_paths"), context=f"{context}.event_paths")
    raw_hashes = _sequence(cell.get("event_sha256"), context=f"{context}.event_sha256")
    if len(raw_paths) != POINT_COUNT or len(raw_hashes) != POINT_COUNT:
        raise PublicationMergeError(f"{context} must identify ten events")
    paths = tuple(
        _resolve_recorded_path(path, context=f"{context}.event_paths")
        for path in raw_paths
    )
    for path, expected_hash in zip(paths, raw_hashes, strict=True):
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or _sha256_file(path) != expected_hash
        ):
            raise PublicationMergeError(f"{context} event SHA-256 provenance differs")
    return paths


def _source_cell(
    report: Mapping[str, Any], family: str, mode: str, n: int, *, context: str
) -> Mapping[str, Any]:
    cells = _mapping(report.get("cells"), context=f"{context}.cells")
    family_cells = _mapping(cells.get(family), context=f"{context}.cells.{family}")
    mode_cells = _mapping(
        family_cells.get(mode), context=f"{context}.cells.{family}.{mode}"
    )
    return _mapping(
        mode_cells.get(str(n)), context=f"{context}.cells.{family}.{mode}.{n}"
    )


def _cell_event_hashes(cell: Mapping[str, Any], *, context: str) -> tuple[str, ...]:
    raw_paths = _sequence(cell.get("event_paths"), context=f"{context}.event_paths")
    if len(raw_paths) != POINT_COUNT:
        raise PublicationMergeError(f"{context}.event_paths must contain ten paths")
    return tuple(
        _sha256_file(_resolve_recorded_path(path, context=f"{context}.event_paths"))
        for path in raw_paths
    )


def _validate_summed_retained_sources(
    provenance: Mapping[str, Any], *, n: int, context: str
) -> None:
    retained = _mapping(
        provenance.get("retained"), context=f"{context}.provenance.retained"
    )
    matrix_path = _resolve_recorded_path(
        retained.get("matrix_source"), context=f"{context}.retained.matrix_source"
    )
    check_path = _resolve_recorded_path(
        retained.get("check_source"), context=f"{context}.retained.check_source"
    )
    matrix_sha = _sha256_file(matrix_path)
    check_sha = _sha256_file(check_path)
    if (
        retained.get("matrix_sha256") != matrix_sha
        or provenance.get("matrix_sha256") != matrix_sha
        or retained.get("check_source_sha256") != check_sha
        or provenance.get("check_source_sha256") != check_sha
    ):
        raise PublicationMergeError(f"{context} retained source SHA-256 differs")
    try:
        matrix_source = matrix_path.read_text(encoding="utf-8")
        check_source = check_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise PublicationMergeError(
            f"{context} cannot read retained generated sources: {error}"
        ) from error
    expected_ncomb = 2 ** (n + 2)
    ncomb_values = {
        int(value)
        for value in re.findall(r"\bNCOMB\s*=\s*(\d+)\b", matrix_source, re.I)
    }
    if expected_ncomb not in ncomb_values or re.search(
        r"INTEGER\s+NHEL\s*\(\s*NEXTERNAL\s*,\s*NCOMB\s*\)",
        matrix_source,
        re.I,
    ) is None:
        raise PublicationMergeError(f"{context} has incompatible NCOMB/NHEL source")
    iden = provenance.get("smatrix_iden")
    if (
        not isinstance(iden, int)
        or len(
            re.findall(
                rf"^[ \t]{{6,}}DATA[ \t]+IDEN[ \t]*/[ \t]*{iden}[ \t]*/",
                matrix_source,
                re.I | re.M,
            )
        )
        != 1
        or len(
            re.findall(
                r"^[ \t]{6,}ANS[ \t]*=[ \t]*ANS[ \t]*/[ \t]*"
                r"DBLE[ \t]*\([ \t]*IDEN[ \t]*\)",
                matrix_source,
                re.I | re.M,
            )
        )
        != 1
    ):
        raise PublicationMergeError(f"{context} has incompatible SMATRIX IDEN use")
    if (
        "USERHEL=-1" not in check_source.replace(" ", "")
        or check_source.count("CALL SMATRIX(") != 3
        or re.search(r"(?<!S)\bMATRIX\s*\(", check_source, re.I)
    ):
        raise PublicationMergeError(
            f"{context} retained driver is not a native SMATRIX helicity sum"
        )


def _validate_madgraph_provenance(
    campaign: Document,
    overlay: Document,
    *,
    helicity_workload: str = "fixed",
    measurement_host: Mapping[str, Any],
) -> Document:
    campaign_policy = _mapping(
        campaign.payload.get("policy"), context="campaign policy"
    )
    campaign_measurement = _mapping(
        campaign_policy.get("measurement"), context="campaign measurement policy"
    )
    campaign_alpha_s = (
        _positive_number(
            campaign_measurement.get("alpha_s"), context="campaign alpha_s"
        )
        if helicity_workload == "sum"
        else None
    )
    raw_series = _mapping(
        overlay.payload.get("runtime_series"), context="MadGraph runtime_series"
    )
    source_identity: tuple[str, str] | None = None
    measured_count_by_family: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for family in FAMILY_MODES:
        measured_limit = MADGRAPH_FAMILY_MAX_MEASURED_MULTIPLICITY[family]
        family_series = _mapping(
            raw_series.get(family), context=f"MadGraph runtime_series.{family}"
        )
        mode_cells = _mapping(
            family_series.get("madgraph-standalone"),
            context=f"MadGraph runtime_series.{family}.madgraph-standalone",
        )
        for n in FINAL_MULTIPLICITIES:
            context = f"MadGraph runtime_series.{family}.madgraph-standalone.{n}"
            cell = _mapping(mode_cells.get(str(n)), context=context)
            status = str(cell.get("status"))
            if status not in {
                "measured",
                "failed",
                "skipped",
                "not-applicable",
            }:
                raise PublicationMergeError(f"{context}.status is unsupported")
            if n > measured_limit:
                if status != "not-applicable":
                    raise PublicationMergeError(
                        f"{context} must be protocol-scoped not-applicable"
                    )
                scope = _mapping(
                    cell.get("protocol_scope"),
                    context=f"{context}.protocol_scope",
                )
                if (
                    cell.get("failure_category")
                    not in _madgraph_protocol_scope_categories(
                        family, n, measured_limit
                    )
                    or cell.get("censors_higher_multiplicities") is not False
                    or scope.get("maximum_measured_multiplicity")
                    != measured_limit
                    or scope.get("disposition") != "not-applicable"
                    or not isinstance(cell.get("failure_reason"), str)
                    or not cell["failure_reason"].strip()
                ):
                    raise PublicationMergeError(
                        f"{context} has incompatible protocol-scope metadata"
                    )
            elif status == "not-applicable":
                raise PublicationMergeError(
                    f"{context} cannot be not-applicable at n<={measured_limit}"
                )
            status_counts[status] += 1
            if status != "measured":
                continue
            measured_count_by_family[family] += 1
            metrics = _mapping(cell.get("metrics"), context=f"{context}.metrics")
            values = {
                key: _positive_number(metrics.get(key), context=f"{context}.{key}")
                for key in METRIC_KEYS
            }
            for key, value in values.items():
                if _finite_number(cell.get(key), context=f"{context}.{key}") != value:
                    raise PublicationMergeError(
                        f"{context}.{key} disagrees with the metrics object"
                    )
            if values["generation_seconds"] >= MAX_GENERATION_SECONDS:
                raise PublicationMergeError(f"{context} exceeds one hour")
            if values["max_rss_kib"] >= MAX_RSS_KIB:
                raise PublicationMergeError(f"{context} reaches 30 GiB RSS")
            protocol = _mapping(cell.get("protocol"), context=f"{context}.protocol")
            expected_evaluator = (
                "SMATRIX(P,ANS)-generated-complete-helicity-sum"
                if helicity_workload == "sum"
                else "MATRIX(P,NHEL,IC)-direct"
            )
            expected_color_sum = (
                "generated-SMATRIX-summed-and-averaged"
                if helicity_workload == "sum"
                else "full-unaveraged"
            )
            if (
                protocol.get("generation_helicity_coverage") != "all"
                or protocol.get("warm_fixed_helicity")
                is not (helicity_workload == "fixed")
                or protocol.get("helicity_summed")
                is not (helicity_workload == "sum")
                or protocol.get("evaluator") != expected_evaluator
                or protocol.get("color_sum") != expected_color_sum
            ):
                raise PublicationMergeError(
                    f"{context} has the wrong helicity protocol"
                )
            if helicity_workload == "sum" and (
                protocol.get("warm_helicity_sum") is not True
                or protocol.get("timed_helicity_count") != 2 ** (n + 2)
                or protocol.get("helicity_sum_implementation")
                != "generated-SMATRIX-with-USERHEL-minus-one"
                or protocol.get("warmed_native_call_pruning")
                != "generated GOODHEL cache may skip structurally zero helicities"
                or _finite_number(cell.get("alpha_s"), context=f"{context}.alpha_s")
                != campaign_alpha_s
            ):
                raise PublicationMergeError(
                    f"{context} has incompatible summed-helicity metadata"
                )
            refresh = _mapping(
                cell.get("runtime_refresh"), context=f"{context}.runtime_refresh"
            )
            if (
                refresh.get("accepted") is not True
                or refresh.get("fresh_process") is not True
                or refresh.get("scope") != "generation-resource-and-warm-runtime"
            ):
                raise PublicationMergeError(f"{context} is not a fresh accepted cell")
            numerical = _mapping(cell.get("numerical"), context=f"{context}.numerical")
            expected_factor = _madgraph_normalization_factor(
                family, n, helicity_workload=helicity_workload
            )
            if (
                numerical.get("reference_mode")
                != _source_mode(family, helicity_workload)
                or numerical.get("passes") is not True
                or not math.isclose(
                    _positive_number(
                        numerical.get("normalization_factor_reference_per_madgraph"),
                        context=f"{context}.normalization_factor",
                    ),
                    expected_factor,
                    rel_tol=0.0,
                    abs_tol=1.0e-16,
                )
            ):
                raise PublicationMergeError(f"{context} has incompatible normalization")
            provenance = _mapping(
                cell.get("provenance"), context=f"{context}.provenance"
            )
            if provenance.get("smatrix_iden") != _expected_madgraph_iden(family, n):
                raise PublicationMergeError(
                    f"{context} has the wrong generated SMATRIX denominator"
                )
            if helicity_workload == "sum":
                if (
                    _finite_number(
                        provenance.get("source_alpha_s"),
                        context=f"{context}.provenance.source_alpha_s",
                    )
                    != campaign_alpha_s
                    or _finite_number(
                        provenance.get("runtime_alpha_s"),
                        context=f"{context}.provenance.runtime_alpha_s",
                    )
                    != campaign_alpha_s
                    or provenance.get("timed_helicity_count") != 2 ** (n + 2)
                ):
                    raise PublicationMergeError(
                        f"{context} has incompatible summed runtime provenance"
                    )
                _validate_summed_retained_sources(
                    provenance, n=n, context=context
                )
            source = _mapping(
                provenance.get("source_report"),
                context=f"{context}.provenance.source_report",
            )
            path = source.get("path")
            sha256 = source.get("sha256")
            expected_pointer = (
                f"cells.{family}.{_source_mode(family, helicity_workload)}.{n}"
            )
            if source.get("cell") != expected_pointer:
                raise PublicationMergeError(f"{context} has the wrong source cell")
            if not isinstance(path, str) or not isinstance(sha256, str):
                raise PublicationMergeError(f"{context} lacks source-report identity")
            identity = (path, sha256)
            if source_identity is None:
                source_identity = identity
            elif source_identity != identity:
                raise PublicationMergeError(
                    "measured MadGraph cells use different source reports"
                )

    if any(measured_count_by_family[family] < 1 for family in FAMILY_MODES):
        raise PublicationMergeError(
            "MadGraph must measure at least one cell per family"
        )
    expected_status = (
        "complete-with-failures" if status_counts["failed"] else "complete"
    )
    if overlay.payload.get("status") != expected_status:
        raise PublicationMergeError("MadGraph status disagrees with its cells")
    if overlay.payload.get("failure_count") != status_counts["failed"]:
        raise PublicationMergeError("MadGraph failure_count disagrees with its cells")
    summary = _mapping(overlay.payload.get("summary"), context="MadGraph summary")
    if summary.get("runtime_series_status_counts") != dict(
        sorted(status_counts.items())
    ):
        raise PublicationMergeError("MadGraph summary disagrees with its cells")
    if source_identity is None:
        raise PublicationMergeError("MadGraph overlay has no measured provenance")

    source_path = _resolve_recorded_path(
        source_identity[0], context="MadGraph source report path"
    )
    source_document = _load_document("MadGraph source report", source_path)
    if source_document.sha256 != source_identity[1]:
        raise PublicationMergeError("MadGraph source-report SHA-256 differs")
    if source_document.payload.get("schema_version") != SCHEMA_VERSION:
        raise PublicationMergeError("MadGraph source report has the wrong schema")
    source_host = _measurement_host(
        source_document.payload.get("measurement_host"),
        context="MadGraph source report measurement_host",
    )
    if source_host != measurement_host:
        raise PublicationMergeError(
            "campaign and MadGraph source report use different measurement hosts"
        )
    if helicity_workload == "sum":
        source_policy = _mapping(
            source_document.payload.get("policy"), context="MadGraph source policy"
        )
        if _helicity_workload(source_policy) != "sum":
            raise PublicationMergeError(
                "MadGraph summed overlay points to a non-summed source report"
            )
        source_measurement = _mapping(
            source_policy.get("measurement"),
            context="MadGraph source measurement policy",
        )
        if (
            _positive_number(
                source_measurement.get("alpha_s"), context="MadGraph source alpha_s"
            )
            != campaign_alpha_s
        ):
            raise PublicationMergeError("MadGraph source alpha_s differs")

    campaign_cells = _mapping(campaign.payload.get("cells"), context="campaign cells")
    for family in FAMILY_MODES:
        mode = _source_mode(family, helicity_workload)
        mode_cells = _mapping(
            _mapping(
                campaign_cells.get(family), context=f"campaign cells.{family}"
            ).get(mode),
            context=f"campaign cells.{family}.{mode}",
        )
        overlay_cells = _mapping(
            _mapping(
                raw_series.get(family), context=f"MadGraph runtime_series.{family}"
            ).get("madgraph-standalone"),
            context=f"MadGraph runtime_series.{family}.madgraph-standalone",
        )
        for n in FINAL_MULTIPLICITIES:
            context = f"MadGraph runtime_series.{family}.madgraph-standalone.{n}"
            madgraph_cell = _mapping(overlay_cells.get(str(n)), context=context)
            if madgraph_cell.get("status") != "measured":
                continue
            source_cell = _source_cell(
                source_document.payload,
                family,
                mode,
                n,
                context="MadGraph source report",
            )
            if source_cell.get("status") != "measured":
                raise PublicationMergeError(f"{context} points to an unmeasured source")
            helicity = _validate_helicity(
                madgraph_cell.get("helicity"), n=n, context=f"{context}.helicity"
            )
            if helicity != _validate_helicity(
                source_cell.get("helicity"),
                n=n,
                context=f"MadGraph source cells.{family}.{mode}.{n}.helicity",
            ):
                raise PublicationMergeError(
                    f"{context} helicity differs from its source"
                )
            madgraph_paths = _event_identity(madgraph_cell, context=context)
            source_paths = tuple(
                _resolve_recorded_path(path, context="MadGraph source event path")
                for path in _sequence(
                    source_cell.get("event_paths"),
                    context="MadGraph source event paths",
                )
            )
            if madgraph_paths != source_paths:
                raise PublicationMergeError(
                    f"{context} event paths differ from its source"
                )
            factor = _madgraph_normalization_factor(
                family, n, helicity_workload=helicity_workload
            )
            source_points = _validate_points(
                source_cell.get("point_values"), context="MadGraph source point values"
            )
            madgraph_points = _validate_points(
                madgraph_cell.get("point_values"), context=f"{context}.point_values"
            )
            if (
                max(
                    _relative_error(observed * factor, reference)
                    for observed, reference in zip(
                        madgraph_points, source_points, strict=True
                    )
                )
                > 1.0e-10
            ):
                raise PublicationMergeError(
                    f"{context} numerically differs from its source"
                )

            fresh_cell = _mapping(mode_cells.get(str(n)), context="fresh source cell")
            if fresh_cell.get("status") != "measured":
                continue
            if helicity != _validate_helicity(
                fresh_cell.get("helicity"), n=n, context="fresh source helicity"
            ):
                raise PublicationMergeError(f"{context} differs from fresh helicity")
            if tuple(madgraph_cell["event_sha256"]) != _cell_event_hashes(
                fresh_cell, context=f"fresh campaign cells.{family}.{mode}.{n}"
            ):
                raise PublicationMergeError(f"{context} differs from fresh events")
            fresh_points = _validate_points(
                fresh_cell.get("point_values"), context="fresh source point values"
            )
            if (
                max(
                    _relative_error(observed * factor, reference)
                    for observed, reference in zip(
                        madgraph_points, fresh_points, strict=True
                    )
                )
                > 1.0e-10
            ):
                raise PublicationMergeError(
                    f"{context} numerically differs from the fresh campaign"
                )
    return source_document


def build_final_report(
    *, campaign_path: Path, madgraph_overlay_path: Path
) -> dict[str, Any]:
    campaign = _load_document("campaign report", campaign_path)
    overlay = _load_document("MadGraph overlay", madgraph_overlay_path)
    campaign_host = _measurement_host(
        campaign.payload.get("measurement_host"),
        context="campaign measurement_host",
    )
    overlay_host = _measurement_host(
        overlay.payload.get("host"), context="MadGraph overlay host"
    )
    if campaign_host != overlay_host:
        raise PublicationMergeError(
            "campaign and MadGraph overlay use different measurement hosts"
        )
    campaign_policy = _validate_campaign_policy(campaign)
    helicity_workload = _helicity_workload(campaign_policy)
    campaign_counts = _validate_campaign_cells(
        campaign, helicity_workload=helicity_workload
    )
    _validate_madgraph_policy(
        overlay, helicity_workload=helicity_workload
    )
    source_document = _validate_madgraph_provenance(
        campaign,
        overlay,
        helicity_workload=helicity_workload,
        measurement_host=campaign_host,
    )

    report = deepcopy(campaign.payload)
    report_measurement = _mapping(
        _mapping(report.get("policy"), context="final report policy").get(
            "measurement"
        ),
        context="final report measurement policy",
    )
    report_measurement.pop("fixed_helicity", None)
    if helicity_workload == "sum":
        report_policy = _mapping(report.get("policy"), context="final report policy")
        plot = report_policy.get("plot")
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
                "MadGraph standalone uses generated SMATRIX with USERHEL=-1; "
                "warmed GOODHEL pruning remains enabled."
            )
            if replacement not in notes:
                notes.append(replacement)
            plot["notes"] = notes
    report["kind"] = FINAL_KIND
    report["summary"] = {
        "cell_status_counts": dict(sorted(campaign_counts.items())),
    }
    try:
        selected._apply_runtime_series_overlay(
            report,
            selected.SourceReport(
                key="madgraph-overlay",
                path=overlay.path,
                sha256=overlay.sha256,
                payload=overlay.payload,
            ),
        )
    except selected.CompositeError as error:
        raise PublicationMergeError(str(error)) from error
    external_counts = Counter(
        str(cell.get("status"))
        for family_series in report["runtime_series"].values()
        for mode_cells in family_series.values()
        for cell in mode_cells.values()
    )
    total_failures = campaign_counts["failed"] + external_counts["failed"]
    report["status"] = "complete-with-failures" if total_failures else "complete"
    report["failure_count"] = total_failures
    report["summary"]["total_failure_count"] = total_failures
    report["publication_provenance"] = {
        "campaign_report": {
            "path": campaign.display_path,
            "sha256": campaign.sha256,
            "kind": CAMPAIGN_KIND,
            "status": campaign.payload["status"],
            "failure_count": campaign_counts["failed"],
        },
        "madgraph_overlay": {
            "path": overlay.display_path,
            "sha256": overlay.sha256,
            "kind": selected.RUNTIME_SERIES_OVERLAY_KIND,
            "status": overlay.payload["status"],
            "failure_count": external_counts["failed"],
        },
        "madgraph_source_report": {
            "path": source_document.display_path,
            "sha256": source_document.sha256,
            "kind": source_document.payload.get("kind"),
        },
        "measurement_host": dict(campaign_host),
        "same_host_authenticated": True,
        "final_state_multiplicities": list(FINAL_MULTIPLICITIES),
        "merge_policy": "fresh-campaign-cells-plus-authenticated-external-series",
    }
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = build_final_report(
            campaign_path=arguments.campaign_report,
            madgraph_overlay_path=arguments.madgraph_overlay,
        )
        _write_report(arguments.output, report)
    except PublicationMergeError as error:
        raise SystemExit(f"error: {error}") from error
    print(arguments.output.expanduser().resolve(strict=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
