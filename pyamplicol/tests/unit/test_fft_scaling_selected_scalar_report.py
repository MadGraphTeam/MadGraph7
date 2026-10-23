# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.developer import fft_scaling_selected_scalar_report as selected


def _measured(source: int, family: str, mode: str, n: int) -> dict[str, Any]:
    warm = source + n / 100.0
    return {
        "family": family,
        "mode": mode,
        "n": n,
        "total_external": n + 2,
        "process": selected._process_expression(family, n),
        "helicity": [1] * (n + 2),
        "event_paths": [
            f"/events/{family}/n{n}/point-{index:02d}.event" for index in range(1, 11)
        ],
        "label": mode,
        "status": "measured",
        "metrics": {
            "generation_seconds": source * 10 + n,
            "warm_seconds_per_point": warm,
            "max_rss_kib": source * 100 + n,
        },
        "point_values": [float(index) for index in range(1, 11)],
        "probe": {
            "warm_median_seconds": warm,
            "warm_samples_seconds": [warm] * 10,
            "point_values": [float(index) for index in range(1, 11)],
        },
    }


def _report(source: int, *, high: bool, canonical: bool = False) -> dict[str, Any]:
    multiplicities = range(7, 11) if high else range(2, 8)
    modes = selected.HIGH_MODES if high else selected.FAMILY_MODES
    cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for family, family_modes in modes.items():
        cells[family] = {}
        for mode in family_modes:
            cells[family][mode] = {
                str(n): _measured(source, family, mode, n) for n in multiplicities
            }
    if high:
        for n in (8, 9):
            cells["gg"]["amplicol"][str(n)] = {
                "status": "skipped",
                "failure_reason": "structural or prior resource censor",
            }
        cells["gg"]["recurrence-fft"]["9"] = {
            "status": "failed",
            "failure_category": "memory-limit",
        }
        cells["ddbar"]["amplicol"]["9"] = {
            "status": "failed",
            "failure_category": "memory-limit",
        }
    return {
        "kind": selected.CANONICAL_KIND if canonical else selected.SCALAR_KIND,
        "schema_version": 1,
        "status": "complete-with-failures" if high else "complete",
        "policy": {
            "final_state_multiplicities": list(multiplicities),
            "process_families": {
                "gg": {
                    "expression": "g g > g g + (n-2)*g",
                    "modes": list(modes["gg"]),
                    "ratio_reference": "reference-fft",
                },
                "ddbar": {
                    "expression": "d d~ > d d~ + (n-2)*g",
                    "modes": list(modes["ddbar"]),
                    "ratio_reference": "amplicol",
                },
            },
        },
        "cells": cells,
    }


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        _write(tmp_path / "scalar.json", _report(1, high=False)),
        _write(
            tmp_path / "canonical.json",
            _report(2, high=False, canonical=True),
        ),
        _write(tmp_path / "high.json", _report(3, high=True)),
    )


def _clean_runtime_overlay(report: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for family, family_modes in selected.RUNTIME_TARGETS.items():
        cells[family] = {}
        for mode, multiplicities in family_modes.items():
            cells[family][mode] = {}
            for n in multiplicities:
                base = report["cells"][family][mode][str(n)]
                warm = float(100 + n)
                sample_count = 1 if mode == "amplicol" else 10
                cell = {
                    "status": "measured",
                    "family": family,
                    "mode": mode,
                    "n": n,
                    "total_external": n + 2,
                    "process": base["process"],
                    "helicity": base["helicity"],
                    "event_paths": base["event_paths"],
                    "event_sha256": ["a" * 64] * 10,
                    "warm_seconds_per_point": warm,
                    "warm_samples_seconds": [warm] * sample_count,
                    "point_values": base["point_values"],
                    "runtime_refresh": {
                        "accepted": True,
                        "fresh_process": True,
                        "scope": "warm-runtime-only",
                    },
                }
                if mode == "amplicol":
                    cell["adaptive_runtime_points"] = 7
                cells[family][mode][str(n)] = cell
    censored: dict[str, dict[str, dict[str, Any]]] = {}
    for (family, mode, n), status in selected.EXPECTED_CENSORED_CELLS.items():
        censored.setdefault(family, {}).setdefault(mode, {})[str(n)] = {
            "status": status
        }
    return {
        "kind": selected.CLEAN_RUNTIME_OVERLAY_KIND,
        "schema_version": 1,
        "status": "complete",
        "cells": cells,
        "censored_cells": censored,
    }


def _madgraph_cell(family: str, n: int, status: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "status": status,
        "family": family,
        "mode": "madgraph-standalone",
        "label": "MadGraph standalone (fixed h)",
        "n": n,
        "total_external": n + 2,
        "process": selected._process_expression(family, n),
    }
    if status == "measured":
        warm = float(n)
        cell.update(
            {
                "helicity": [1] * (n + 2),
                "event_paths": [f"/events/mg/{family}/n{n}/{i}" for i in range(10)],
                "event_sha256": ["b" * 64] * 10,
                "warm_seconds_per_point": warm,
                "warm_samples_seconds": [warm] * 10,
                "metrics": {
                    "generation_seconds": warm * 10.0,
                    "warm_seconds_per_point": warm,
                    "max_rss_kib": warm * 1024.0,
                },
                "matrix_element": 1.0,
                "point_values": [1.0] * 10,
                "protocol": {
                    "evaluator": "MATRIX(P,NHEL,IC)-direct",
                    "helicity_summed": False,
                    "color_sum": "full-unaveraged",
                    "batch_size": 1,
                    "initialization_included": False,
                },
            }
        )
    elif status == "not-applicable":
        measured_limit = selected.MADGRAPH_FAMILY_MAX_MEASURED_MULTIPLICITY[family]
        cell.update(
            {
                "failure_category": (
                    "protocol-scope-pure-gluon-n6"
                    if (family, n) == ("gg", 6)
                    else f"protocol-scope-n>{measured_limit}"
                ),
                "failure_reason": (
                    f"final-plot protocol measures {family} only through "
                    f"n={measured_limit}"
                ),
                "censors_higher_multiplicities": False,
                "protocol_scope": {
                    "maximum_measured_multiplicity": measured_limit,
                    "disposition": "not-applicable",
                },
            }
        )
    else:
        cell["failure_reason"] = "standalone source frontier"
    return cell


def _madgraph_overlay() -> dict[str, Any]:
    return {
        "kind": selected.RUNTIME_SERIES_OVERLAY_KIND,
        "schema_version": 1,
        "policy": {
            "final_state_multiplicities": list(selected.FINAL_MULTIPLICITIES),
            "helicity_workload": "fixed",
            "warm_fixed_helicity": True,
            "warm_helicity_sum": False,
            "family_maximum_measured_multiplicity": (
                selected.MADGRAPH_FAMILY_MAX_MEASURED_MULTIPLICITY
            ),
        },
        "runtime_series": {
            "gg": {
                "madgraph-standalone": {
                    str(n): _madgraph_cell(
                        "gg",
                        n,
                        (
                            "measured"
                            if n <= 4
                            else "failed"
                            if n == 5
                            else "not-applicable"
                        ),
                    )
                    for n in range(2, 10)
                }
            },
            "ddbar": {
                "madgraph-standalone": {
                    str(n): _madgraph_cell(
                        "ddbar",
                        n,
                        (
                            "failed"
                            if n == 2
                            else "skipped"
                            if n <= 6
                            else "not-applicable"
                        ),
                    )
                    for n in range(2, 10)
                }
            },
        },
    }


def test_composite_selects_exact_cells_and_metric_sources(tmp_path: Path) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)

    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
    )

    assert report["policy"]["final_state_multiplicities"] == list(range(2, 10))
    assert report["policy"]["process_families"]["gg"]["modes"] == [
        "reference-fft",
        "amplicol",
        "recurrence-direct",
        "recurrence-fft",
    ]
    assert set(report["cells"]["gg"]["recurrence-direct"]) == {
        "2",
        "3",
        "4",
        "5",
        "6",
    }
    assert report["cells"]["gg"]["recurrence-direct"]["6"]["label"] == (
        "pyAmpliCol - recurrence (n≤6)"
    )
    assert all(
        "10" not in mode_cells
        for family_cells in report["cells"].values()
        for mode_cells in family_cells.values()
    )
    assert report["summary"]["cell_status_counts"] == {
        "failed": 2,
        "measured": 46,
        "skipped": 2,
    }

    low_reference = report["cells"]["gg"]["reference-fft"]["6"]
    assert low_reference["metrics"] == {
        "generation_seconds": 26.0,
        "warm_seconds_per_point": 1.06,
        "max_rss_kib": 106.0,
    }
    low_candidate = report["cells"]["gg"]["recurrence-fft"]["6"]
    assert low_candidate["metrics"] == {
        "generation_seconds": 26.0,
        "warm_seconds_per_point": 1.06,
        "max_rss_kib": 206.0,
    }
    assert low_candidate["plot_provenance"]["metrics"] == {
        "generation_seconds": "canonical-final:cells.gg.recurrence-fft.6",
        "warm_seconds_per_point": "scalar-current:cells.gg.recurrence-fft.6",
        "max_rss_kib": "canonical-final:cells.gg.recurrence-fft.6",
    }
    assert report["cells"]["gg"]["recurrence-fft"]["7"]["metrics"] == {
        "generation_seconds": 37.0,
        "warm_seconds_per_point": 3.07,
        "max_rss_kib": 307.0,
    }


def test_exact_runtime_overlay_updates_only_accepted_warm_cell(
    tmp_path: Path,
) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    overlay = {
        "kind": selected.OVERLAY_KIND,
        "schema_version": 1,
        "cells": {
            "8": {
                "warm_seconds_per_point": 0.125,
                "warm_samples_seconds": [0.125] * 10,
                "point_values": [float(index) for index in range(1, 11)],
                "runtime_refresh": {
                    "accepted": True,
                    "archive_sha256": "a" * 64,
                    "paired_run_logs": ["p1-B.log", "p2-B.log", "p3-B.log"],
                },
            }
        },
    }
    overlay_path = _write(tmp_path / "overlay.json", overlay)

    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
        runtime_overlay_path=overlay_path,
    )

    cell = report["cells"]["gg"]["recurrence-fft"]["8"]
    assert cell["metrics"]["warm_seconds_per_point"] == 0.125
    assert cell["probe"]["warm_samples_seconds"] == [0.125] * 10
    assert cell["runtime_refresh"]["accepted"] is True
    assert cell["plot_provenance"]["metrics"]["warm_seconds_per_point"] == (
        "runtime-overlay:cells.8"
    )
    assert (
        report["cells"]["gg"]["recurrence-fft"]["7"]["metrics"][
            "warm_seconds_per_point"
        ]
        == 3.07
    )


def test_builder_fails_closed_on_missing_source_cell(tmp_path: Path) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    scalar = json.loads(scalar_path.read_text(encoding="utf-8"))
    del scalar["cells"]["ddbar"]["recurrence-direct"]["6"]
    _write(scalar_path, scalar)

    with pytest.raises(
        selected.CompositeError, match=r"cells\.ddbar\.recurrence-direct\.6"
    ):
        selected.build_selected_report(
            scalar_path=scalar_path,
            canonical_path=canonical_path,
            high_path=high_path,
        )


def test_runtime_overlay_rejects_any_target_outside_n7_n8(tmp_path: Path) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    overlay_path = _write(
        tmp_path / "overlay.json",
        {
            "kind": selected.OVERLAY_KIND,
            "schema_version": 1,
            "cells": {"9": {}},
        },
    )

    with pytest.raises(selected.CompositeError, match="only gg recurrence-fft"):
        selected.build_selected_report(
            scalar_path=scalar_path,
            canonical_path=canonical_path,
            high_path=high_path,
            runtime_overlay_path=overlay_path,
        )


def test_clean_runtime_overlay_refreshes_exactly_all_measured_warm_cells(
    tmp_path: Path,
) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    base = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
    )
    overlay_path = _write(tmp_path / "clean-overlay.json", _clean_runtime_overlay(base))

    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
        runtime_overlay_path=overlay_path,
    )

    assert report["runtime_refresh"] == {
        "scope": "warm-runtime-only",
        "coverage": "complete",
        "fresh_measured_cell_count": 46,
        "preserved_resource_censored_cell_count": 4,
    }
    assert (
        report["summary"]["cell_status_counts"] == base["summary"]["cell_status_counts"]
    )
    assert report["policy"]["measurement"]["warm_samples"] == {
        "reference-fft": 10,
        "pyamplicol": 10,
        "amplicol": 1,
    }
    assert report["policy"]["plot"]["metric_source_boundaries"] == {
        "generation_seconds": {
            "source_boundary_after_n": 6,
            "source_boundary_label": "n=2..6 / n=7..9 source boundary",
        },
        "warm_seconds_per_point": None,
        "max_rss_kib": {
            "source_boundary_after_n": 6,
            "source_boundary_label": "n=2..6 / n=7..9 source boundary",
        },
    }
    refreshed = report["cells"]["ddbar"]["recurrence-fft"]["9"]
    original = base["cells"]["ddbar"]["recurrence-fft"]["9"]
    assert refreshed["metrics"]["warm_seconds_per_point"] == 109.0
    assert (
        refreshed["metrics"]["generation_seconds"]
        == original["metrics"]["generation_seconds"]
    )
    assert refreshed["metrics"]["max_rss_kib"] == original["metrics"]["max_rss_kib"]
    assert refreshed["runtime_refresh"]["fresh_process"] is True
    assert refreshed["runtime_refresh"]["event_sha256"] == ["a" * 64] * 10
    assert set(report["cells"]["gg"]["recurrence-direct"]) == set("23456")


def test_clean_runtime_overlay_fails_closed_when_one_target_is_missing(
    tmp_path: Path,
) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    base = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
    )
    overlay = _clean_runtime_overlay(base)
    del overlay["cells"]["ddbar"]["recurrence-fft"]["9"]
    overlay_path = _write(tmp_path / "incomplete.json", overlay)

    with pytest.raises(selected.CompositeError, match="exactly 46 measurable"):
        selected.build_selected_report(
            scalar_path=scalar_path,
            canonical_path=canonical_path,
            high_path=high_path,
            runtime_overlay_path=overlay_path,
        )


def test_runtime_series_is_separate_from_the_46_existing_cells(
    tmp_path: Path,
) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    overlay_path = _write(tmp_path / "madgraph.json", _madgraph_overlay())

    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
        runtime_series_overlay_path=overlay_path,
    )

    assert report["summary"]["cell_status_counts"] == {
        "failed": 2,
        "measured": 46,
        "skipped": 2,
    }
    assert report["summary"]["runtime_series_status_counts"] == {
        "failed": 2,
        "measured": 3,
        "not-applicable": 7,
        "skipped": 4,
    }
    assert (
        report["runtime_series"]["gg"]["madgraph-standalone"]["4"][
            "warm_seconds_per_point"
        ]
        == 4.0
    )
    assert report["runtime_series"]["gg"]["madgraph-standalone"]["4"]["metrics"] == {
        "generation_seconds": 40.0,
        "warm_seconds_per_point": 4.0,
        "max_rss_kib": 4096.0,
    }
    assert "madgraph-standalone" not in report["cells"]["gg"]


def test_runtime_series_uses_policy_warm_sample_count(tmp_path: Path) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    overlay = _madgraph_overlay()
    overlay["policy"]["warm_sample_count"] = 1
    for family_series in overlay["runtime_series"].values():
        for cell in family_series["madgraph-standalone"].values():
            if cell["status"] == "measured":
                cell["warm_samples_seconds"] = [cell["warm_seconds_per_point"]]
    overlay_path = _write(tmp_path / "madgraph.json", overlay)

    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
        runtime_series_overlay_path=overlay_path,
    )

    assert (
        report["runtime_series"]["gg"]["madgraph-standalone"]["2"][
            "warm_samples_seconds"
        ]
        == [2.0]
    )


def test_runtime_series_progress_accepts_authenticated_sparse_cells(
    tmp_path: Path,
) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
    )
    progress = _madgraph_overlay()
    progress["kind"] = selected.RUNTIME_SERIES_PROGRESS_KIND
    progress["policy"] = {"final_state_multiplicities": list(range(2, 10))}
    progress["runtime_series"]["gg"]["madgraph-standalone"] = {
        "2": progress["runtime_series"]["gg"]["madgraph-standalone"]["2"]
    }
    progress["runtime_series"]["ddbar"]["madgraph-standalone"] = {}
    path = _write(tmp_path / "progress.json", progress)

    selected.apply_runtime_series_source(
        report,
        selected.SourceReport(
            key="progress",
            path=path,
            sha256="0" * 64,
            payload=progress,
        ),
    )

    assert set(report["runtime_series"]["gg"]["madgraph-standalone"]) == {"2"}
    assert report["runtime_series"]["ddbar"]["madgraph-standalone"] == {}
    assert report["plot_provenance"]["source_reports"]["progress"] == {
        "path": str(path),
        "sha256": "0" * 64,
    }


def test_summed_report_rejects_a_fixed_helicity_madgraph_overlay(
    tmp_path: Path,
) -> None:
    scalar_path, canonical_path, high_path = _sources(tmp_path)
    report = selected.build_selected_report(
        scalar_path=scalar_path,
        canonical_path=canonical_path,
        high_path=high_path,
    )
    report["policy"]["measurement"].update(
        {
            "helicity_workload": "sum",
            "warm_fixed_helicity": False,
            "warm_helicity_sum": True,
        }
    )
    overlay = _madgraph_overlay()
    overlay["policy"] = {
        "final_state_multiplicities": list(range(2, 10)),
        "helicity_workload": "fixed",
        "warm_fixed_helicity": True,
        "warm_helicity_sum": False,
    }
    path = _write(tmp_path / "fixed-madgraph.json", overlay)

    with pytest.raises(selected.CompositeError, match="differs from the report"):
        selected.apply_runtime_series_source(
            report,
            selected.SourceReport(
                key="fixed-madgraph",
                path=path,
                sha256="0" * 64,
                payload=overlay,
            ),
        )
