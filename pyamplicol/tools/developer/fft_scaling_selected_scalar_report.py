#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Build the selected-mode scalar FullColor report for final-state n=2..9."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "IMPLEMENTATION_DOCS" / "RESULTS" / "fft-scaling-study"
DATA = RESULTS / "data"

KIND = "pyamplicol-fullcolor-selected-scalar-composite"
OVERLAY_KIND = "pyamplicol-fullcolor-selected-scalar-runtime-overlay"
CLEAN_RUNTIME_OVERLAY_KIND = (
    "pyamplicol-fullcolor-selected-scalar-clean-runtime-overlay"
)
RUNTIME_SERIES_OVERLAY_KIND = (
    "pyamplicol-fullcolor-selected-scalar-runtime-series-overlay"
)
RUNTIME_SERIES_PROGRESS_KIND = (
    "pyamplicol-fullcolor-selected-scalar-runtime-series-progress"
)
SCHEMA_VERSION = 1
LOW_MULTIPLICITIES = tuple(range(2, 7))
HIGH_MULTIPLICITIES = tuple(range(7, 10))
FINAL_MULTIPLICITIES = (*LOW_MULTIPLICITIES, *HIGH_MULTIPLICITIES)
SOURCE_BOUNDARY_AFTER_N = 6
FAMILY_MODES = {
    "gg": (
        "reference-fft",
        "amplicol",
        "recurrence-direct",
        "recurrence-fft",
    ),
    "ddbar": ("amplicol", "recurrence-direct", "recurrence-fft"),
}
HIGH_MODES = {
    "gg": ("reference-fft", "amplicol", "recurrence-fft"),
    "ddbar": ("amplicol", "recurrence-fft"),
}
CANDIDATE_MODES = frozenset(("recurrence-direct", "recurrence-fft"))
METRIC_KEYS = (
    "generation_seconds",
    "warm_seconds_per_point",
    "max_rss_kib",
)
OVERLAY_MULTIPLICITIES = frozenset((7, 8))
DIRECT_LABEL = "pyAmpliCol - recurrence (n≤6)"
SCALAR_KIND = "pyamplicol-fullcolor-fft-scaling-study"
CANONICAL_KIND = "pyamplicol-fullcolor-fft-scaling-batch128-composite"
RUNTIME_TARGETS = {
    "gg": {
        "reference-fft": tuple(range(2, 10)),
        "amplicol": tuple(range(2, 8)),
        "recurrence-direct": tuple(range(2, 7)),
        "recurrence-fft": tuple(range(2, 9)),
    },
    "ddbar": {
        "amplicol": tuple(range(2, 9)),
        "recurrence-direct": tuple(range(2, 7)),
        "recurrence-fft": tuple(range(2, 10)),
    },
}
RUNTIME_TARGET_COUNT = sum(
    len(multiplicities)
    for family_modes in RUNTIME_TARGETS.values()
    for multiplicities in family_modes.values()
)
EXPECTED_CENSORED_CELLS = {
    ("gg", "amplicol", 8): "skipped",
    ("gg", "amplicol", 9): "skipped",
    ("gg", "recurrence-fft", 9): "failed",
    ("ddbar", "amplicol", 9): "failed",
}
RUNTIME_SERIES_MODES = frozenset(("madgraph-standalone",))
MADGRAPH_MAX_MEASURED_MULTIPLICITY = 6
MADGRAPH_FAMILY_MAX_MEASURED_MULTIPLICITY = {"gg": 5, "ddbar": 6}
LEGACY_MADGRAPH_WARM_SAMPLE_COUNT = 10


class CompositeError(RuntimeError):
    """The source reports cannot form the requested composite without guessing."""


@dataclass(frozen=True, slots=True)
class SourceReport:
    key: str
    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def display_path(self) -> str:
        resolved = self.path.expanduser().resolve(strict=False)
        try:
            return str(resolved.relative_to(ROOT))
        except ValueError:
            return str(resolved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scalar-report",
        type=Path,
        default=DATA / "campaign-report-scalar-latency-current.json",
        help="scalar warm-runtime source report",
    )
    parser.add_argument(
        "--canonical-report",
        type=Path,
        default=DATA / "campaign-report-final.json",
        help="canonical low-n cold/resource source report",
    )
    parser.add_argument(
        "--high-report",
        type=Path,
        default=DATA / "campaign-report-high-n-scalar.json",
        help="scalar high-n source report",
    )
    parser.add_argument(
        "--runtime-overlay",
        type=Path,
        help=(
            "optional accepted warm-runtime overlay; supports the historical "
            "gg recurrence-FFT n=7/n=8 refresh and the complete clean-runtime "
            "refresh"
        ),
    )
    parser.add_argument(
        "--runtime-series-overlay",
        type=Path,
        help=(
            "optional sparse runtime-only comparison series, such as MadGraph "
            "standalone"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA / "campaign-report-scalar-selected-n2-n9.json",
        help="output composite report",
    )
    return parser


def _load_source(key: str, path: Path, *, require_cells: bool = True) -> SourceReport:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise CompositeError(f"cannot read {key} report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CompositeError(f"{key} report must be a JSON object")
    if require_cells and not isinstance(payload.get("cells"), dict):
        raise CompositeError(f"{key} report must contain a cells object")
    return SourceReport(
        key=key,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
    )


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompositeError(f"{context} must be an object")
    return value


def _validate_source_contract(
    source: SourceReport,
    *,
    expected_kind: str,
    expected_multiplicities: Sequence[int],
) -> None:
    if source.payload.get("kind") != expected_kind:
        raise CompositeError(f"{source.key} report kind must be {expected_kind!r}")
    if source.payload.get("schema_version") != SCHEMA_VERSION:
        raise CompositeError(f"{source.key} report schema_version must be 1")
    policy = _mapping(source.payload.get("policy"), context=f"{source.key} policy")
    raw_multiplicities = policy.get("final_state_multiplicities")
    if not isinstance(raw_multiplicities, Sequence) or isinstance(
        raw_multiplicities, str
    ):
        raise CompositeError(
            f"{source.key} policy.final_state_multiplicities must be an array"
        )
    if tuple(raw_multiplicities) != tuple(expected_multiplicities):
        raise CompositeError(
            f"{source.key} report multiplicities must be "
            f"{list(expected_multiplicities)}"
        )


def _cell(source: SourceReport, family: str, mode: str, n: int) -> dict[str, Any]:
    context = f"{source.key} cells.{family}.{mode}.{n}"
    cells = _mapping(source.payload["cells"], context=f"{source.key} cells")
    family_cells = _mapping(cells.get(family), context=f"{source.key} cells.{family}")
    mode_cells = _mapping(
        family_cells.get(mode), context=f"{source.key} cells.{family}.{mode}"
    )
    raw_cell = _mapping(mode_cells.get(str(n)), context=context)
    return deepcopy(dict(raw_cell))


def _positive_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CompositeError(f"{context} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CompositeError(f"{context} must be a positive finite number")
    return result


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CompositeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CompositeError(f"{context} must be a finite number")
    return result


def _metric(cell: Mapping[str, Any], key: str, *, context: str) -> float:
    metrics = _mapping(cell.get("metrics"), context=f"{context}.metrics")
    return _positive_number(metrics.get(key), context=f"{context}.metrics.{key}")


def _source_pointer(source: SourceReport, family: str, mode: str, n: int) -> str:
    return f"{source.key}:cells.{family}.{mode}.{n}"


def _measured_cell(
    source: SourceReport, family: str, mode: str, n: int
) -> dict[str, Any]:
    cell = _cell(source, family, mode, n)
    if cell.get("status") != "measured":
        raise CompositeError(
            f"{_source_pointer(source, family, mode, n)} must be measured"
        )
    return cell


def _low_cell(
    scalar: SourceReport,
    canonical: SourceReport,
    family: str,
    mode: str,
    n: int,
) -> dict[str, Any]:
    scalar_cell = _measured_cell(scalar, family, mode, n)
    canonical_cell = _measured_cell(canonical, family, mode, n)
    rss_source = canonical if mode in CANDIDATE_MODES else scalar
    rss_cell = canonical_cell if mode in CANDIDATE_MODES else scalar_cell

    result = scalar_cell
    result["metrics"] = {
        "generation_seconds": _metric(
            canonical_cell,
            "generation_seconds",
            context=_source_pointer(canonical, family, mode, n),
        ),
        "warm_seconds_per_point": _metric(
            scalar_cell,
            "warm_seconds_per_point",
            context=_source_pointer(scalar, family, mode, n),
        ),
        "max_rss_kib": _metric(
            rss_cell,
            "max_rss_kib",
            context=_source_pointer(rss_source, family, mode, n),
        ),
    }
    if mode == "recurrence-direct":
        result["label"] = DIRECT_LABEL
    result["plot_provenance"] = {
        "status": _source_pointer(scalar, family, mode, n),
        "metrics": {
            "generation_seconds": _source_pointer(canonical, family, mode, n),
            "warm_seconds_per_point": _source_pointer(scalar, family, mode, n),
            "max_rss_kib": _source_pointer(rss_source, family, mode, n),
        },
    }
    return result


def _high_cell(high: SourceReport, family: str, mode: str, n: int) -> dict[str, Any]:
    result = _cell(high, family, mode, n)
    status = result.get("status")
    if status not in {"measured", "failed", "skipped"}:
        raise CompositeError(
            f"{_source_pointer(high, family, mode, n)} has unsupported status "
            f"{status!r}"
        )
    provenance: dict[str, Any] = {"status": _source_pointer(high, family, mode, n)}
    if status == "measured":
        for metric in METRIC_KEYS:
            _metric(
                result,
                metric,
                context=_source_pointer(high, family, mode, n),
            )
        provenance["metrics"] = {
            metric: _source_pointer(high, family, mode, n) for metric in METRIC_KEYS
        }
    result["plot_provenance"] = provenance
    return result


def _policy(high: SourceReport) -> dict[str, Any]:
    high_policy = _mapping(high.payload.get("policy"), context="high policy")
    high_families = _mapping(
        high_policy.get("process_families"), context="high policy.process_families"
    )
    process_families: dict[str, Any] = {}
    for family, modes in FAMILY_MODES.items():
        source_family = _mapping(
            high_families.get(family),
            context=f"high policy.process_families.{family}",
        )
        expression = source_family.get("expression")
        ratio_reference = source_family.get("ratio_reference")
        if not isinstance(expression, str) or not isinstance(ratio_reference, str):
            raise CompositeError(f"high policy for {family} lacks plot metadata")
        process_families[family] = {
            "expression": expression,
            "modes": list(modes),
            "ratio_reference": ratio_reference,
        }
    return {
        "final_state_multiplicities": list(FINAL_MULTIPLICITIES),
        "measurement": {
            "alpha_s": 0.118,
            "color_accuracy": "full",
            "fixed_helicity": True,
            "n_definition": "final-state-particle-count",
            "warm_benchmark_batch_size": 1,
            "warm_samples": 10,
            "source_boundary_after_n": SOURCE_BOUNDARY_AFTER_N,
            "warm_timing_metric": (
                "scalar latency: n=2..6 from scalar-latency-current and "
                "n=7..9 from high-n-scalar"
            ),
            "generation_metric": (
                "cold-to-ready telemetry: n=2..6 from canonical-final and "
                "n=7..9 from high-n-scalar"
            ),
            "rss_metric": (
                "implementation RSS: Reference FFT and AmpliCol n=2..6 use "
                "scalar implementation-self RSS; pyAmpliCol n=2..6 uses the "
                "canonical max of generation-child and runtime-self RSS; "
                "n=7..9 retains high-n-scalar definitions"
            ),
        },
        "plot": {
            "source_boundary_after_n": SOURCE_BOUNDARY_AFTER_N,
            "source_boundary_label": "n=2..6 / n=7..9 source boundary",
            "metric_source_boundaries": {
                metric: {
                    "source_boundary_after_n": SOURCE_BOUNDARY_AFTER_N,
                    "source_boundary_label": "n=2..6 / n=7..9 source boundary",
                }
                for metric in METRIC_KEYS
            },
        },
        "process_families": process_families,
    }


def _relative_error(left: float, right: float) -> float:
    if left == right:
        return 0.0
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _load_overlay(path: Path) -> SourceReport:
    source = _load_source("runtime-overlay", path)
    if source.payload.get("schema_version") != SCHEMA_VERSION:
        raise CompositeError("runtime overlay schema_version must be 1")
    return source


def _apply_runtime_overlay(report: dict[str, Any], overlay: SourceReport) -> None:
    if overlay.payload.get("kind") != OVERLAY_KIND:
        raise CompositeError(f"runtime overlay kind must be {OVERLAY_KIND!r}")
    raw_cells = _mapping(overlay.payload.get("cells"), context="runtime-overlay cells")
    if not raw_cells:
        raise CompositeError("runtime overlay cells must not be empty")
    unexpected = set(raw_cells) - {str(n) for n in OVERLAY_MULTIPLICITIES}
    if unexpected:
        raise CompositeError(
            "runtime overlay may target only gg recurrence-fft n=7 or n=8; "
            f"got {sorted(unexpected)}"
        )

    for raw_n, raw_overlay_cell in raw_cells.items():
        n = int(raw_n)
        overlay_cell = _mapping(
            raw_overlay_cell, context=f"runtime-overlay cells.{raw_n}"
        )
        warm = _positive_number(
            overlay_cell.get("warm_seconds_per_point"),
            context=f"runtime-overlay cells.{raw_n}.warm_seconds_per_point",
        )
        samples_raw = overlay_cell.get("warm_samples_seconds")
        if not isinstance(samples_raw, Sequence) or isinstance(samples_raw, str):
            raise CompositeError(
                f"runtime-overlay cells.{raw_n}.warm_samples_seconds "
                "must contain ten samples"
            )
        samples = [
            _positive_number(
                value,
                context=f"runtime-overlay cells.{raw_n}.warm_samples_seconds",
            )
            for value in samples_raw
        ]
        if len(samples) != 10:
            raise CompositeError(
                f"runtime-overlay cells.{raw_n}.warm_samples_seconds "
                "must contain ten samples"
            )
        if not math.isclose(statistics.median(samples), warm, rel_tol=1e-12):
            raise CompositeError(
                f"runtime-overlay cells.{raw_n} warm value is not the sample median"
            )

        refresh = deepcopy(
            dict(
                _mapping(
                    overlay_cell.get("runtime_refresh"),
                    context=f"runtime-overlay cells.{raw_n}.runtime_refresh",
                )
            )
        )
        if refresh.get("accepted") is not True:
            raise CompositeError(
                f"runtime-overlay cells.{raw_n}.runtime_refresh.accepted must be true"
            )
        points_raw = overlay_cell.get("point_values")
        if not isinstance(points_raw, Sequence) or isinstance(points_raw, str):
            raise CompositeError(
                f"runtime-overlay cells.{raw_n}.point_values must be an array"
            )
        points = [
            _finite_number(value, context=f"runtime-overlay cells.{raw_n}.point_values")
            for value in points_raw
        ]

        cell = report["cells"]["gg"]["recurrence-fft"][raw_n]
        base_points_raw = cell.get("point_values")
        if not isinstance(base_points_raw, Sequence) or isinstance(
            base_points_raw, str
        ):
            raise CompositeError(f"composite gg recurrence-fft n={n} lacks points")
        base_points = [float(value) for value in base_points_raw]
        if len(points) != len(base_points):
            raise CompositeError(
                f"runtime-overlay cells.{raw_n}.point_values has the wrong length"
            )
        maximum_error = max(
            (
                _relative_error(left, right)
                for left, right in zip(points, base_points, strict=True)
            ),
            default=0.0,
        )
        if maximum_error > 1.0e-10:
            raise CompositeError(
                f"runtime-overlay cells.{raw_n} point values disagree "
                f"(maximum relative error {maximum_error:.3e})"
            )

        cell["metrics"]["warm_seconds_per_point"] = warm
        probe = _mapping(cell.get("probe"), context=f"composite gg recurrence n={n}")
        updated_probe = deepcopy(dict(probe))
        updated_probe["warm_median_seconds"] = warm
        updated_probe["warm_samples_seconds"] = samples
        updated_probe["point_values"] = points
        cell["probe"] = updated_probe
        cell["point_values"] = points
        previous_refresh = cell.get("runtime_refresh")
        refresh["overlay_source"] = f"runtime-overlay:cells.{raw_n}"
        refresh["maximum_relative_point_error_against_high_report"] = maximum_error
        if isinstance(previous_refresh, Mapping):
            refresh["supersedes"] = deepcopy(dict(previous_refresh))
        cell["runtime_refresh"] = refresh
        cell["plot_provenance"]["metrics"]["warm_seconds_per_point"] = (
            f"runtime-overlay:cells.{raw_n}"
        )


def _runtime_target_keys() -> set[tuple[str, str, int]]:
    return {
        (family, mode, n)
        for family, family_modes in RUNTIME_TARGETS.items()
        for mode, multiplicities in family_modes.items()
        for n in multiplicities
    }


def _overlay_samples(
    raw_cell: Mapping[str, Any], *, context: str, expected_count: int | Sequence[int]
) -> tuple[float, list[float]]:
    if isinstance(expected_count, int) and not isinstance(expected_count, bool):
        expected_counts = {expected_count}
    elif isinstance(expected_count, Sequence) and not isinstance(expected_count, str):
        expected_counts = {
            value
            for value in expected_count
            if isinstance(value, int) and not isinstance(value, bool)
        }
    else:
        expected_counts = set()
    if not expected_counts:
        raise CompositeError(f"{context} has no valid expected warm sample count")
    warm = _positive_number(
        raw_cell.get("warm_seconds_per_point"),
        context=f"{context}.warm_seconds_per_point",
    )
    raw_samples = raw_cell.get("warm_samples_seconds")
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, str):
        raise CompositeError(f"{context}.warm_samples_seconds must be an array")
    samples = [
        _positive_number(value, context=f"{context}.warm_samples_seconds")
        for value in raw_samples
    ]
    if len(samples) not in expected_counts:
        expected = " or ".join(str(value) for value in sorted(expected_counts))
        raise CompositeError(
            f"{context}.warm_samples_seconds must contain {expected} samples"
        )
    if not math.isclose(statistics.median(samples), warm, rel_tol=1e-12):
        raise CompositeError(f"{context} warm value is not the sample median")
    return warm, samples


def _overlay_points(
    raw_cell: Mapping[str, Any],
    base_cell: Mapping[str, Any],
    *,
    context: str,
) -> tuple[list[float], float]:
    raw_points = raw_cell.get("point_values")
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, str):
        raise CompositeError(f"{context}.point_values must be an array")
    points = [
        _finite_number(value, context=f"{context}.point_values") for value in raw_points
    ]
    base_points_raw = base_cell.get("point_values")
    if not isinstance(base_points_raw, Sequence) or isinstance(base_points_raw, str):
        raise CompositeError(f"{context} base cell lacks point values")
    base_points = [
        _finite_number(value, context=f"{context} base point values")
        for value in base_points_raw
    ]
    if len(points) != len(base_points) or not points:
        raise CompositeError(f"{context}.point_values has the wrong length")
    maximum_error = max(
        _relative_error(left, right)
        for left, right in zip(points, base_points, strict=True)
    )
    if maximum_error > 1.0e-10:
        raise CompositeError(
            f"{context}.point_values disagree with the retained inputs "
            f"(maximum relative error {maximum_error:.3e})"
        )
    return points, maximum_error


def _apply_clean_runtime_overlay(report: dict[str, Any], overlay: SourceReport) -> None:
    if overlay.payload.get("kind") != CLEAN_RUNTIME_OVERLAY_KIND:
        raise CompositeError(
            f"clean runtime overlay kind must be {CLEAN_RUNTIME_OVERLAY_KIND!r}"
        )
    if overlay.payload.get("status") != "complete":
        raise CompositeError("clean runtime overlay status must be 'complete'")
    raw_cells = _mapping(
        overlay.payload.get("cells"), context="clean-runtime-overlay cells"
    )
    observed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for family, raw_family_cells in raw_cells.items():
        family_cells = _mapping(
            raw_family_cells, context=f"clean-runtime-overlay cells.{family}"
        )
        for mode, raw_mode_cells in family_cells.items():
            mode_cells = _mapping(
                raw_mode_cells,
                context=f"clean-runtime-overlay cells.{family}.{mode}",
            )
            for raw_n, raw_cell in mode_cells.items():
                try:
                    n = int(raw_n)
                except (TypeError, ValueError) as error:
                    raise CompositeError(
                        "clean runtime overlay multiplicities must be integers"
                    ) from error
                key = (str(family), str(mode), n)
                if key in observed:
                    raise CompositeError(f"clean runtime overlay repeats {key}")
                observed[key] = _mapping(
                    raw_cell,
                    context=(f"clean-runtime-overlay cells.{family}.{mode}.{raw_n}"),
                )
    expected = _runtime_target_keys()
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise CompositeError(
            "clean runtime overlay must contain exactly "
            f"{RUNTIME_TARGET_COUNT} measurable cells; missing={missing}, "
            f"extra={extra}"
        )

    raw_censored = _mapping(
        overlay.payload.get("censored_cells"),
        context="clean-runtime-overlay censored_cells",
    )
    observed_censored: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for family, raw_family_cells in raw_censored.items():
        for mode, raw_mode_cells in _mapping(
            raw_family_cells,
            context=f"clean-runtime-overlay censored_cells.{family}",
        ).items():
            for raw_n, raw_cell in _mapping(
                raw_mode_cells,
                context=(f"clean-runtime-overlay censored_cells.{family}.{mode}"),
            ).items():
                try:
                    n = int(raw_n)
                except (TypeError, ValueError) as error:
                    raise CompositeError(
                        "clean runtime overlay censored multiplicities must be integers"
                    ) from error
                key = (str(family), str(mode), n)
                if key in observed_censored:
                    raise CompositeError(
                        f"clean runtime overlay repeats censored cell {key}"
                    )
                observed_censored[key] = _mapping(
                    raw_cell,
                    context=(
                        f"clean-runtime-overlay censored_cells.{family}.{mode}.{raw_n}"
                    ),
                )
    if set(observed_censored) != set(EXPECTED_CENSORED_CELLS):
        raise CompositeError(
            "clean runtime overlay must preserve exactly the four resource-"
            "censored cells"
        )
    for key, expected_status in EXPECTED_CENSORED_CELLS.items():
        if observed_censored[key].get("status") != expected_status:
            raise CompositeError(
                f"clean runtime overlay censored cell {key} changed status"
            )
        family, mode, n = key
        if report["cells"][family][mode][str(n)].get("status") != expected_status:
            raise CompositeError(f"composite censored cell {key} changed status")

    for family, mode, n in sorted(
        expected, key=lambda value: (value[2], value[0], value[1])
    ):
        context = f"clean-runtime-overlay cells.{family}.{mode}.{n}"
        raw_cell = observed[(family, mode, n)]
        if raw_cell.get("status") != "measured":
            raise CompositeError(f"{context}.status must be 'measured'")
        base_cell = report["cells"][family][mode][str(n)]
        if base_cell.get("status") != "measured":
            raise CompositeError(f"{context} targets a non-measured base cell")
        for key, expected_value in (
            ("family", family),
            ("mode", mode),
            ("n", n),
            ("process", base_cell.get("process")),
            ("total_external", n + 2),
            ("helicity", base_cell.get("helicity")),
            ("event_paths", base_cell.get("event_paths")),
        ):
            if raw_cell.get(key) != expected_value:
                raise CompositeError(f"{context}.{key} has the wrong identity")
        event_sha256 = raw_cell.get("event_sha256")
        if (
            not isinstance(event_sha256, Sequence)
            or isinstance(event_sha256, str)
            or len(event_sha256) != 10
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in event_sha256
            )
        ):
            raise CompositeError(
                f"{context}.event_sha256 must contain ten lowercase SHA-256 values"
            )
        expected_samples = 1 if mode == "amplicol" else 10
        warm, samples = _overlay_samples(
            raw_cell, context=context, expected_count=expected_samples
        )
        points, maximum_error = _overlay_points(raw_cell, base_cell, context=context)
        refresh = deepcopy(
            dict(
                _mapping(
                    raw_cell.get("runtime_refresh"),
                    context=f"{context}.runtime_refresh",
                )
            )
        )
        if (
            refresh.get("accepted") is not True
            or refresh.get("fresh_process") is not True
            or refresh.get("scope") != "warm-runtime-only"
        ):
            raise CompositeError(
                f"{context}.runtime_refresh must describe one accepted fresh "
                "warm-runtime-only process"
            )
        refresh["overlay_source"] = f"clean-runtime-overlay:cells.{family}.{mode}.{n}"
        refresh["maximum_relative_point_error_against_retained_input"] = maximum_error
        refresh["event_sha256"] = deepcopy(list(event_sha256))
        previous_refresh = base_cell.get("runtime_refresh")
        if isinstance(previous_refresh, Mapping):
            refresh["supersedes"] = deepcopy(dict(previous_refresh))
        base_cell["metrics"]["warm_seconds_per_point"] = warm
        base_cell["point_values"] = points
        base_cell["runtime_refresh"] = refresh
        base_cell["plot_provenance"]["metrics"]["warm_seconds_per_point"] = (
            f"clean-runtime-overlay:cells.{family}.{mode}.{n}"
        )
        probe = base_cell.get("probe")
        if isinstance(probe, Mapping):
            updated_probe = deepcopy(dict(probe))
            updated_probe["warm_median_seconds"] = warm
            updated_probe["warm_samples_seconds"] = samples
            updated_probe["point_values"] = points
            base_cell["probe"] = updated_probe
        reference = base_cell.get("reference")
        if isinstance(reference, Mapping):
            updated_reference = deepcopy(dict(reference))
            updated_reference["warm_median_seconds"] = warm
            updated_reference["warm_samples_seconds"] = samples
            updated_reference["matrix_elements"] = points
            base_cell["reference"] = updated_reference
        adaptive_points = raw_cell.get("adaptive_runtime_points")
        if mode == "amplicol":
            if (
                not isinstance(adaptive_points, int)
                or isinstance(adaptive_points, bool)
                or adaptive_points < 1
            ):
                raise CompositeError(
                    f"{context}.adaptive_runtime_points must be positive"
                )
            base_cell["adaptive_runtime_points"] = adaptive_points

    measurement = report["policy"]["measurement"]
    measurement["warm_timing_metric"] = (
        "clean warmed-runtime refresh: all 46 measurable selected cells use "
        "fresh processes on retained deterministic inputs"
    )
    measurement["warm_samples"] = {
        "reference-fft": 10,
        "pyamplicol": 10,
        "amplicol": 1,
    }
    measurement["warm_timer_source"] = "process-cpu-time"
    measurement["warm_target_seconds"] = 0.25
    measurement["warm_runtime_artifact_policy"] = (
        "runtime-compatible generated artifacts and external standalone inputs "
        "are retained; generation and RSS metrics remain historical"
    )
    plot = report["policy"]["plot"]
    boundaries = plot.get("metric_source_boundaries")
    if not isinstance(boundaries, dict):
        raise CompositeError("composite plot metric boundaries are malformed")
    boundaries["warm_seconds_per_point"] = None
    report["runtime_refresh"] = {
        "scope": "warm-runtime-only",
        "coverage": "complete",
        "fresh_measured_cell_count": RUNTIME_TARGET_COUNT,
        "preserved_resource_censored_cell_count": len(EXPECTED_CENSORED_CELLS),
    }


def _process_expression(family: str, n: int) -> str:
    extra = " ".join("g" for _ in range(n - 2))
    if family == "gg":
        return "g g > g g" + (f" {extra}" if extra else "")
    return "d d~ > d d~" + (f" {extra}" if extra else "")


def _helicity_workload_from_policy(
    policy: Mapping[str, Any], *, legacy_default: str = "fixed"
) -> str:
    measurement = _mapping(
        policy.get("measurement", policy), context="helicity workload policy"
    )
    markers: set[str] = set()
    declared = measurement.get("helicity_workload")
    if declared is not None:
        markers.add(str(declared))
    if measurement.get("warm_fixed_helicity") is True:
        markers.add("fixed")
    if measurement.get("warm_helicity_sum") is True:
        markers.add("sum")
    if len(markers) > 1 or any(value not in {"fixed", "sum"} for value in markers):
        raise CompositeError("helicity workload policy is contradictory")
    return next(iter(markers), legacy_default)


def _madgraph_family_measured_limits(
    policy: Mapping[str, Any] | None,
) -> dict[str, int]:
    if policy is None:
        return {
            family: MADGRAPH_MAX_MEASURED_MULTIPLICITY for family in FAMILY_MODES
        }
    raw_limits = policy.get("family_maximum_measured_multiplicity")
    if raw_limits is None:
        return {
            family: MADGRAPH_MAX_MEASURED_MULTIPLICITY for family in FAMILY_MODES
        }
    if not isinstance(raw_limits, Mapping):
        raise CompositeError(
            "runtime series overlay has invalid family multiplicity limits"
        )
    limits: dict[str, int] = {}
    for family in FAMILY_MODES:
        value = raw_limits.get(family)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 2
            or value > MADGRAPH_MAX_MEASURED_MULTIPLICITY
        ):
            raise CompositeError(
                "runtime series overlay has invalid family multiplicity limits"
            )
        limits[family] = int(value)
    if set(raw_limits) != set(FAMILY_MODES):
        raise CompositeError(
            "runtime series overlay has invalid family multiplicity limits"
        )
    return limits


def _madgraph_protocol_scope_categories(family: str, n: int, limit: int) -> set[str]:
    categories = {f"protocol-scope-n>{limit}"}
    if family == "gg" and n == 6:
        categories.add("protocol-scope-pure-gluon-n6")
    return categories


def _madgraph_warm_sample_count(policy: Mapping[str, Any] | None) -> int:
    if policy is None:
        return LEGACY_MADGRAPH_WARM_SAMPLE_COUNT
    raw_count = policy.get("warm_sample_count")
    if raw_count is None:
        return LEGACY_MADGRAPH_WARM_SAMPLE_COUNT
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1:
        raise CompositeError("runtime series overlay has invalid warm_sample_count")
    return raw_count


def _apply_runtime_series_overlay(
    report: dict[str, Any], overlay: SourceReport
) -> None:
    overlay_kind = overlay.payload.get("kind")
    if overlay_kind not in {
        RUNTIME_SERIES_OVERLAY_KIND,
        RUNTIME_SERIES_PROGRESS_KIND,
    }:
        raise CompositeError(
            "runtime series source must be a terminal overlay or authenticated "
            "sparse progress snapshot"
        )
    sparse_progress = overlay_kind == RUNTIME_SERIES_PROGRESS_KIND
    raw_series = _mapping(
        overlay.payload.get("runtime_series"), context="runtime-series-overlay"
    )
    report_policy = _mapping(report.get("policy"), context="report policy")
    report_workload = _helicity_workload_from_policy(report_policy)
    raw_multiplicities = report_policy.get("final_state_multiplicities")
    if (
        not isinstance(raw_multiplicities, Sequence)
        or isinstance(raw_multiplicities, str)
        or not raw_multiplicities
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 2
            for value in raw_multiplicities
        )
    ):
        raise CompositeError("report has invalid final-state multiplicities")
    multiplicities = tuple(int(value) for value in raw_multiplicities)
    raw_overlay_policy = overlay.payload.get("policy")
    overlay_policy: Mapping[str, Any] | None = None
    if raw_overlay_policy is None and not sparse_progress:
        # Historical terminal overlays predate their explicit policy block and
        # cover the report's complete multiplicity set by contract.
        raw_overlay_multiplicities: object = list(multiplicities)
    else:
        overlay_policy = _mapping(
            raw_overlay_policy, context="runtime-series-overlay policy"
        )
        raw_overlay_multiplicities = overlay_policy.get("final_state_multiplicities")
        overlay_workload = _helicity_workload_from_policy(overlay_policy)
        if overlay_workload != report_workload:
            raise CompositeError(
                "runtime series overlay helicity workload differs from the report"
            )
    family_limits = _madgraph_family_measured_limits(overlay_policy)
    expected_warm_samples = _madgraph_warm_sample_count(overlay_policy)
    if (
        not isinstance(raw_overlay_multiplicities, Sequence)
        or isinstance(raw_overlay_multiplicities, str)
        or not raw_overlay_multiplicities
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_overlay_multiplicities
        )
    ):
        raise CompositeError("runtime series overlay has invalid multiplicities")
    overlay_multiplicities = tuple(int(value) for value in raw_overlay_multiplicities)
    if overlay_multiplicities != tuple(
        value for value in multiplicities if value in set(overlay_multiplicities)
    ):
        raise CompositeError(
            "runtime series overlay multiplicities must be an ordered subset "
            "of the report"
        )
    if set(raw_series) != set(FAMILY_MODES):
        raise CompositeError("runtime series overlay must contain gg and ddbar")
    copied: dict[str, dict[str, dict[str, Any]]] = {}
    status_counts: Counter[str] = Counter()
    for family, raw_family_series in raw_series.items():
        family_series = _mapping(
            raw_family_series, context=f"runtime-series-overlay.{family}"
        )
        if set(family_series) != RUNTIME_SERIES_MODES:
            raise CompositeError(
                f"runtime-series-overlay.{family} must contain only MadGraph standalone"
            )
        copied[family] = {}
        for mode, raw_mode_cells in family_series.items():
            mode_cells = _mapping(
                raw_mode_cells,
                context=f"runtime-series-overlay.{family}.{mode}",
            )
            expected_ns = {str(n) for n in overlay_multiplicities}
            observed_ns = set(mode_cells)
            if observed_ns - expected_ns or (
                not sparse_progress and observed_ns != expected_ns
            ):
                raise CompositeError(
                    f"runtime-series-overlay.{family}.{mode} must describe exactly "
                    f"n={list(overlay_multiplicities)}"
                )
            frontier_seen = False
            copied_cells: dict[str, Any] = {}
            for n in (
                value for value in overlay_multiplicities if str(value) in mode_cells
            ):
                context = f"runtime-series-overlay.{family}.{mode}.{n}"
                cell = deepcopy(dict(_mapping(mode_cells[str(n)], context=context)))
                status = cell.get("status")
                if status not in {
                    "measured",
                    "failed",
                    "skipped",
                    "not-applicable",
                }:
                    raise CompositeError(f"{context}.status is unsupported")
                measured_limit = family_limits[family]
                if n > measured_limit:
                    if status != "not-applicable":
                        raise CompositeError(
                            f"{context} must be protocol-scoped not-applicable"
                        )
                elif status == "not-applicable":
                    raise CompositeError(
                        f"{context} cannot be not-applicable at n<={measured_limit}"
                    )
                if status == "measured" and frontier_seen:
                    raise CompositeError(
                        f"{context} resumes measurement after the series frontier"
                    )
                if (
                    status == "failed"
                    and cell.get("censors_higher_multiplicities") is True
                ):
                    if frontier_seen:
                        raise CompositeError(
                            f"{context} repeats the series failure frontier"
                        )
                    frontier_seen = True
                if (
                    status == "skipped"
                    and cell.get("censors_higher_multiplicities") is True
                    and not frontier_seen
                ):
                    raise CompositeError(
                        f"{context} skips before a recorded failure frontier"
                    )
                for key, expected_value in (
                    ("family", family),
                    ("mode", mode),
                    ("n", n),
                    ("process", _process_expression(family, n)),
                    ("total_external", n + 2),
                ):
                    if cell.get(key) != expected_value:
                        raise CompositeError(f"{context}.{key} has the wrong identity")
                label = cell.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise CompositeError(f"{context}.label is required")
                if status == "not-applicable":
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
                    ):
                        raise CompositeError(
                            f"{context} has incompatible protocol-scope metadata"
                        )
                if status == "measured":
                    _overlay_samples(
                        cell,
                        context=context,
                        expected_count=(
                            expected_warm_samples,
                            LEGACY_MADGRAPH_WARM_SAMPLE_COUNT,
                        ),
                    )
                    metrics = _mapping(
                        cell.get("metrics"), context=f"{context}.metrics"
                    )
                    measured_metrics = {
                        key: _finite_number(
                            metrics.get(key), context=f"{context}.metrics.{key}"
                        )
                        for key in METRIC_KEYS
                    }
                    if any(value <= 0.0 for value in measured_metrics.values()):
                        raise CompositeError(
                            f"{context}.metrics must contain three positive values"
                        )
                    if not math.isclose(
                        measured_metrics["warm_seconds_per_point"],
                        _finite_number(
                            cell.get("warm_seconds_per_point"),
                            context=f"{context}.warm_seconds_per_point",
                        ),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ):
                        raise CompositeError(
                            f"{context} warm metric and timing median disagree"
                        )
                    helicity = cell.get("helicity")
                    if (
                        not isinstance(helicity, Sequence)
                        or isinstance(helicity, str)
                        or len(helicity) != n + 2
                        or any(
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value not in (-1, 1)
                            for value in helicity
                        )
                    ):
                        raise CompositeError(f"{context}.helicity is invalid")
                    matrix_element = _finite_number(
                        cell.get("matrix_element"),
                        context=f"{context}.matrix_element",
                    )
                    point_values_raw = cell.get("point_values")
                    if not isinstance(point_values_raw, Sequence) or isinstance(
                        point_values_raw, str
                    ):
                        raise CompositeError(
                            f"{context}.point_values must contain ten values"
                        )
                    point_values = [
                        _finite_number(value, context=f"{context}.point_values")
                        for value in point_values_raw
                    ]
                    if (
                        len(point_values) != 10
                        or _relative_error(matrix_element, point_values[0]) > 1.0e-12
                    ):
                        raise CompositeError(
                            f"{context}.matrix_element must match point_values[0]"
                        )
                    event_paths = cell.get("event_paths")
                    event_sha256 = cell.get("event_sha256")
                    if (
                        not isinstance(event_paths, Sequence)
                        or isinstance(event_paths, str)
                        or len(event_paths) != 10
                        or any(not isinstance(value, str) for value in event_paths)
                        or not isinstance(event_sha256, Sequence)
                        or isinstance(event_sha256, str)
                        or len(event_sha256) != 10
                        or any(
                            not isinstance(value, str)
                            or len(value) != 64
                            or any(
                                character not in "0123456789abcdef"
                                for character in value
                            )
                            for value in event_sha256
                        )
                    ):
                        raise CompositeError(
                            f"{context} event identity must contain ten paths and "
                            "SHA-256 values"
                        )
                    protocol = _mapping(
                        cell.get("protocol"), context=f"{context}.protocol"
                    )
                    expected_protocol = {
                        "evaluator": (
                            "SMATRIX(P,ANS)-generated-complete-helicity-sum"
                            if report_workload == "sum"
                            else "MATRIX(P,NHEL,IC)-direct"
                        ),
                        "helicity_summed": report_workload == "sum",
                        "color_sum": (
                            "generated-SMATRIX-summed-and-averaged"
                            if report_workload == "sum"
                            else "full-unaveraged"
                        ),
                        "batch_size": 1,
                        "initialization_included": False,
                    }
                    if report_workload == "sum":
                        expected_protocol.update(
                            {
                                "helicity_workload": "sum",
                                "warm_fixed_helicity": False,
                                "warm_helicity_sum": True,
                                "timed_helicity_count": 2 ** (n + 2),
                            }
                        )
                        if (
                            cell.get("helicity_workload") != "sum"
                            or cell.get("warm_fixed_helicity") is not False
                            or cell.get("warm_helicity_sum") is not True
                            or cell.get("timed_helicity_count") != 2 ** (n + 2)
                        ):
                            raise CompositeError(
                                f"{context} summed workload metadata is incompatible"
                            )
                    if any(
                        protocol.get(key) != value
                        for key, value in expected_protocol.items()
                    ):
                        raise CompositeError(f"{context}.protocol is incompatible")
                else:
                    if (
                        not isinstance(cell.get("failure_reason"), str)
                        or not cell["failure_reason"].strip()
                    ):
                        raise CompositeError(f"{context}.failure_reason is required")
                status_counts[str(status)] += 1
                copied_cells[str(n)] = cell
            copied[family][mode] = copied_cells
    report["runtime_series"] = copied
    report["summary"]["runtime_series_status_counts"] = dict(
        sorted(status_counts.items())
    )


def apply_runtime_series_source(report: dict[str, Any], overlay: SourceReport) -> None:
    """Validate, attach, and provenance-record one immutable runtime series."""

    _apply_runtime_series_overlay(report, overlay)
    plot_provenance = report.setdefault("plot_provenance", {})
    if not isinstance(plot_provenance, dict):
        raise CompositeError("report plot_provenance must be an object")
    source_reports = plot_provenance.setdefault("source_reports", {})
    if not isinstance(source_reports, dict):
        raise CompositeError("report source-report provenance must be an object")
    source_reports[overlay.key] = {
        "path": overlay.display_path,
        "sha256": overlay.sha256,
    }


def build_selected_report(
    *,
    scalar_path: Path,
    canonical_path: Path,
    high_path: Path,
    runtime_overlay_path: Path | None = None,
    runtime_series_overlay_path: Path | None = None,
) -> dict[str, Any]:
    scalar = _load_source("scalar-current", scalar_path)
    canonical = _load_source("canonical-final", canonical_path)
    high = _load_source("high-n-scalar", high_path)
    _validate_source_contract(
        scalar,
        expected_kind=SCALAR_KIND,
        expected_multiplicities=range(2, 8),
    )
    _validate_source_contract(
        canonical,
        expected_kind=CANONICAL_KIND,
        expected_multiplicities=range(2, 8),
    )
    _validate_source_contract(
        high,
        expected_kind=SCALAR_KIND,
        expected_multiplicities=range(7, 11),
    )
    sources = (scalar, canonical, high)

    cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for family, modes in FAMILY_MODES.items():
        family_cells: dict[str, dict[str, dict[str, Any]]] = {}
        for mode in modes:
            mode_cells = {
                str(n): _low_cell(scalar, canonical, family, mode, n)
                for n in LOW_MULTIPLICITIES
            }
            if mode in HIGH_MODES[family]:
                mode_cells.update(
                    {
                        str(n): _high_cell(high, family, mode, n)
                        for n in HIGH_MULTIPLICITIES
                    }
                )
            family_cells[mode] = mode_cells
        cells[family] = family_cells

    status_counts = Counter(
        str(cell.get("status"))
        for family_cells in cells.values()
        for mode_cells in family_cells.values()
        for cell in mode_cells.values()
    )
    report: dict[str, Any] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "complete-with-failures" if status_counts["failed"] else "complete",
        "failure_count": status_counts["failed"],
        "policy": _policy(high),
        "plot_provenance": {
            "source_reports": {
                source.key: {
                    "path": source.display_path,
                    "sha256": source.sha256,
                }
                for source in sources
            },
            "low_n": list(LOW_MULTIPLICITIES),
            "high_n": list(HIGH_MULTIPLICITIES),
            "recurrence_direct_max_n": max(LOW_MULTIPLICITIES),
            "excluded_n": [10],
        },
        "summary": {"cell_status_counts": dict(sorted(status_counts.items()))},
        "cells": cells,
    }

    if runtime_overlay_path is not None:
        overlay = _load_overlay(runtime_overlay_path)
        if overlay.payload.get("kind") == OVERLAY_KIND:
            _apply_runtime_overlay(report, overlay)
        elif overlay.payload.get("kind") == CLEAN_RUNTIME_OVERLAY_KIND:
            _apply_clean_runtime_overlay(report, overlay)
        else:
            raise CompositeError(
                "runtime overlay kind must be either the historical or clean "
                "runtime overlay kind"
            )
        report["plot_provenance"]["source_reports"][overlay.key] = {
            "path": overlay.display_path,
            "sha256": overlay.sha256,
        }
    if runtime_series_overlay_path is not None:
        series_overlay = _load_source(
            "runtime-series-overlay",
            runtime_series_overlay_path,
            require_cells=False,
        )
        if series_overlay.payload.get("schema_version") != SCHEMA_VERSION:
            raise CompositeError("runtime series overlay schema_version must be 1")
        apply_runtime_series_source(report, series_overlay)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = build_selected_report(
            scalar_path=arguments.scalar_report,
            canonical_path=arguments.canonical_report,
            high_path=arguments.high_report,
            runtime_overlay_path=arguments.runtime_overlay,
            runtime_series_overlay_path=arguments.runtime_series_overlay,
        )
        _write_report(arguments.output, report)
    except CompositeError as error:
        raise SystemExit(f"error: {error}") from error
    print(arguments.output.expanduser().resolve(strict=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
