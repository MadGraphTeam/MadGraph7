# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tools.developer import fft_scaling_final_publication_report as final_report
from tools.developer import fft_scaling_study_plots as plots


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(tmp_path: Path, family: str, n: int) -> tuple[list[str], list[str]]:
    root = tmp_path / "events" / family / f"n{n}"
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    hashes: list[str] = []
    for index in range(1, 11):
        path = root / f"point-{index:02d}.event"
        path.write_text(f"{family} n={n} point={index}\n", encoding="utf-8")
        paths.append(str(path))
        hashes.append(_sha256(path))
    return paths, hashes


def _campaign_cell(tmp_path: Path, family: str, mode: str, n: int) -> dict[str, Any]:
    paths, _ = _events(tmp_path, family, n)
    points = [float(100 * n + index) for index in range(1, 11)]
    cell: dict[str, Any] = {
        "status": "measured",
        "family": family,
        "mode": mode,
        "label": mode,
        "n": n,
        "total_external": n + 2,
        "process": final_report._process_expression(family, n),
        "color_accuracy": "full",
        "helicity": [1 if index % 2 else -1 for index in range(n + 2)],
        "event_paths": paths,
        "point_values": points[:1] if mode == "amplicol" else points,
        "metrics": {
            "generation_seconds": float(n),
            "warm_seconds_per_point": n / 1_000_000.0,
            "max_rss_kib": float(1000 + n),
        },
    }
    if mode in final_report.CANDIDATE_MODE_CONTRACT:
        execution_mode, contraction = final_report.CANDIDATE_MODE_CONTRACT[mode]
        cell.update(
            {
                "execution_mode": execution_mode,
                "color_contraction": contraction,
                "generation_helicity_coverage": "all",
                "warm_fixed_helicity": True,
                "numerical": {"available": True, "passes": True},
            }
        )
    return cell


def _campaign(tmp_path: Path) -> dict[str, Any]:
    cells = {
        family: {
            mode: {
                str(n): _campaign_cell(tmp_path, family, mode, n)
                for n in final_report.FINAL_MULTIPLICITIES
            }
            for mode in modes
        }
        for family, modes in final_report.FAMILY_MODES.items()
    }
    return {
        "kind": final_report.CAMPAIGN_KIND,
        "schema_version": 1,
        "status": "complete",
        "failure_count": 0,
        "policy": {
            "kind": final_report.CAMPAIGN_KIND,
            "schema_version": 1,
            "run_root": str(tmp_path),
            "fft_enabled": True,
            "selected_pyamplicol_color_contractions": {
                "recurrence": ["direct", "symmetric-group-fft"],
                "on-the-fly": ["direct", "symmetric-group-fft"],
            },
            "process_families": {
                family: {
                    "expression": final_report.EXPECTED_EXPRESSIONS[family],
                    "ratio_reference": final_report.EXPECTED_RATIO_REFERENCES[family],
                    "modes": list(modes),
                }
                for family, modes in final_report.FAMILY_MODES.items()
            },
            "final_state_multiplicities": list(final_report.FINAL_MULTIPLICITIES),
            "total_external_particles": [
                n + 2 for n in final_report.FINAL_MULTIPLICITIES
            ],
            "measurement": {
                "color_accuracy": "full",
                "n_definition": "final-state-particle-count",
                "generation_helicity_coverage": "all",
                "warm_fixed_helicity": True,
                "fixed_helicity": True,
                "warm_benchmark_batch_size": 128,
                "warm_sample_count": 10,
                "compiled_fft_enabled": False,
                "memory_policy": "per-cell-strictly-below-publication-ceiling",
                "requested_memory_ceiling_gib": 30.0,
                "memory_watchdog_gib": 30.0,
                "cell_admission_limits": {
                    "generation_seconds": {"operator": "<", "limit": 3600.0},
                    "runtime_seconds": {"operator": "<", "limit": 3600.0},
                    "peak_rss_gib": {"operator": "<", "limit": 30.0},
                },
            },
        },
        "cells": cells,
    }


def _madgraph_cell(
    campaign: dict[str, Any],
    source_path: Path,
    family: str,
    n: int,
) -> dict[str, Any]:
    mode = final_report.SOURCE_MODE[family]
    source = campaign["cells"][family][mode][str(n)]
    factor = final_report._madgraph_normalization_factor(family, n)
    points = [float(value) / factor for value in source["point_values"]]
    hashes = [_sha256(Path(path)) for path in source["event_paths"]]
    warm = n / 10_000.0
    return {
        "status": "measured",
        "family": family,
        "mode": "madgraph-standalone",
        "label": "MadGraph standalone (fixed h)",
        "n": n,
        "total_external": n + 2,
        "process": final_report._process_expression(family, n),
        "helicity": source["helicity"],
        "event_paths": source["event_paths"],
        "event_sha256": hashes,
        "point_values": points,
        "matrix_element": points[0],
        "generation_seconds": float(2 * n),
        "warm_seconds_per_point": warm,
        "max_rss_kib": float(2048 + n),
        "warm_samples_seconds": [warm] * 10,
        "metrics": {
            "generation_seconds": float(2 * n),
            "warm_seconds_per_point": warm,
            "max_rss_kib": float(2048 + n),
        },
        "numerical": {
            "reference_mode": mode,
            "normalization_factor_reference_per_madgraph": factor,
            "passes": True,
        },
        "protocol": {
            "evaluator": "MATRIX(P,NHEL,IC)-direct",
            "generation_helicity_coverage": "all",
            "warm_fixed_helicity": True,
            "helicity_summed": False,
            "color_sum": "full-unaveraged",
            "batch_size": 1,
            "initialization_included": False,
        },
        "runtime_refresh": {
            "accepted": True,
            "fresh_process": True,
            "scope": "generation-resource-and-warm-runtime",
        },
        "provenance": {
            "smatrix_iden": final_report._expected_madgraph_iden(family, n),
            "source_report": {
                "path": str(source_path),
                "sha256": _sha256(source_path),
                "cell": f"cells.{family}.{mode}.{n}",
            },
        },
    }


def _madgraph_scope_cell(family: str, n: int) -> dict[str, Any]:
    return {
        "status": "not-applicable",
        "family": family,
        "mode": "madgraph-standalone",
        "label": "MadGraph standalone (fixed h)",
        "n": n,
        "total_external": n + 2,
        "process": final_report._process_expression(family, n),
        "failure_category": "protocol-scope-n>6",
        "failure_reason": "final-plot protocol measures MadGraph only through n=6",
        "censors_higher_multiplicities": False,
        "protocol_scope": {
            "maximum_measured_multiplicity": 6,
            "disposition": "not-applicable",
        },
    }


def _overlay(campaign: dict[str, Any], source_path: Path) -> dict[str, Any]:
    cells = {
        family: {
            "madgraph-standalone": {
                str(n): (
                    _madgraph_cell(campaign, source_path, family, n)
                    if n <= final_report.MADGRAPH_MAX_MEASURED_MULTIPLICITY
                    else _madgraph_scope_cell(family, n)
                )
                for n in final_report.FINAL_MULTIPLICITIES
            }
        }
        for family in final_report.FAMILY_MODES
    }
    return {
        "kind": final_report.selected.RUNTIME_SERIES_OVERLAY_KIND,
        "schema_version": 1,
        "status": "complete",
        "failure_count": 0,
        "policy": {
            "final_state_multiplicities": list(final_report.FINAL_MULTIPLICITIES),
            "process_families": list(final_report.FAMILY_MODES),
            "point_validation_count": 10,
            "warm_timed_point_index": 1,
            "warm_sample_count": 10,
            "generation_timeout_seconds": 3595.0,
            "outer_memory_watchdog_gib": 30.0,
            "watchdog_enforced_per_generation_build_and_runtime": True,
            "generation_helicity_coverage": "all",
            "warm_fixed_helicity": True,
            "maximum_measured_multiplicity": 6,
            "higher_multiplicity_policy": "not-applicable-protocol-scope",
            "metric_scope": {
                "generation_seconds": "cold to ready",
                "warm_seconds_per_point": "fixed-helicity warm time",
                "max_rss_kib": "conservative process-tree peak",
            },
        },
        "summary": {
            "runtime_series_status_counts": {
                "measured": 10,
                "not-applicable": 6,
            }
        },
        "runtime_series": cells,
    }


def _inputs(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    campaign = _campaign(tmp_path)
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay = _overlay(campaign, campaign_path)
    overlay_path = _write_json(tmp_path / "madgraph.json", overlay)
    return campaign, campaign_path, overlay, overlay_path


def test_merge_preserves_fresh_cells_and_policy_and_adds_all_metrics(
    tmp_path: Path,
) -> None:
    campaign, campaign_path, _, overlay_path = _inputs(tmp_path)

    report = final_report.build_final_report(
        campaign_path=campaign_path, madgraph_overlay_path=overlay_path
    )
    repeated = final_report.build_final_report(
        campaign_path=campaign_path, madgraph_overlay_path=overlay_path
    )

    assert report == repeated
    assert report["kind"] == final_report.FINAL_KIND
    expected_policy = deepcopy(campaign["policy"])
    expected_policy["measurement"].pop("fixed_helicity")
    assert report["policy"] == expected_policy
    assert report["policy"]["measurement"]["generation_helicity_coverage"] == "all"
    assert report["policy"]["measurement"]["warm_fixed_helicity"] is True
    assert report["cells"] == campaign["cells"]
    assert report["cells"]["gg"]["otf-fft"]["9"]["execution_mode"] == (
        "on-the-fly"
    )
    assert report["cells"]["ddbar"]["otf-direct"]["9"]["color_contraction"] == (
        "direct"
    )
    assert report["policy"]["final_state_multiplicities"] == list(range(2, 10))
    assert (
        report["runtime_series"]["ddbar"]["madgraph-standalone"]["9"]["status"]
        == "not-applicable"
    )
    assert report["summary"] == {
        "cell_status_counts": {"measured": 88},
        "runtime_series_status_counts": {
            "measured": 10,
            "not-applicable": 6,
        },
        "total_failure_count": 0,
    }
    assert report["publication_provenance"]["campaign_report"]["sha256"] == (
        _sha256(campaign_path)
    )


def test_merge_authenticates_a_native_smatrix_helicity_sum(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    measurement = campaign["policy"]["measurement"]
    measurement.update(
        {
            "alpha_s": 0.118,
            "helicity_workload": "sum",
            "warm_fixed_helicity": False,
            "warm_helicity_sum": True,
        }
    )
    measurement.pop("fixed_helicity", None)
    campaign["policy"]["plot"] = {
        "notes": [
            "MadGraph is omitted because its available series is fixed-helicity."
        ]
    }
    for family_modes in campaign["cells"].values():
        for mode, mode_cells in family_modes.items():
            if mode not in final_report.CANDIDATE_MODE_CONTRACT:
                continue
            for cell in mode_cells.values():
                cell["warm_fixed_helicity"] = False
                cell["warm_helicity_sum"] = True
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay = _overlay(campaign, campaign_path)
    overlay["policy"].update(
        {
            "helicity_workload": "sum",
            "warm_fixed_helicity": False,
            "warm_helicity_sum": True,
        }
    )
    for family in final_report.FAMILY_MODES:
        source_mode = final_report.SUM_SOURCE_MODE[family]
        for n in range(2, final_report.MADGRAPH_MAX_MEASURED_MULTIPLICITY + 1):
            cell = overlay["runtime_series"][family]["madgraph-standalone"][str(n)]
            source = campaign["cells"][family][source_mode][str(n)]
            cell.update(
                {
                    "label": "MadGraph standalone (helicity sum)",
                    "alpha_s": 0.118,
                    "helicity_workload": "sum",
                    "warm_fixed_helicity": False,
                    "warm_helicity_sum": True,
                    "timed_helicity_count": 2 ** (n + 2),
                    "helicity": source["helicity"],
                    "event_paths": source["event_paths"],
                    "event_sha256": [
                        _sha256(Path(path)) for path in source["event_paths"]
                    ],
                    "point_values": source["point_values"],
                    "matrix_element": source["point_values"][0],
                }
            )
            cell["numerical"].update(
                {
                    "reference_mode": source_mode,
                    "normalization_factor_reference_per_madgraph": 1.0,
                }
            )
            cell["protocol"].update(
                {
                    "evaluator": "SMATRIX(P,ANS)-generated-complete-helicity-sum",
                    "helicity_workload": "sum",
                    "warm_fixed_helicity": False,
                    "warm_helicity_sum": True,
                    "helicity_summed": True,
                    "timed_helicity_count": 2 ** (n + 2),
                    "helicity_sum_implementation": (
                        "generated-SMATRIX-with-USERHEL-minus-one"
                    ),
                    "warmed_native_call_pruning": (
                        "generated GOODHEL cache may skip structurally zero helicities"
                    ),
                    "color_sum": "generated-SMATRIX-summed-and-averaged",
                }
            )
            retained = tmp_path / "retained" / family / f"n{n}"
            retained.mkdir(parents=True)
            iden = final_report._expected_madgraph_iden(family, n)
            matrix = retained / "matrix.f"
            matrix.write_text(
                "      SUBROUTINE SMATRIX(P,ANS)\n"
                f"      INTEGER NCOMB\n      PARAMETER (NCOMB={2 ** (n + 2)})\n"
                "      INTEGER NEXTERNAL\n"
                "      INTEGER NHEL(NEXTERNAL,NCOMB)\n"
                f"      DATA IDEN/{iden}/\n"
                "      ANS=ANS/DBLE(IDEN)\n"
                "      END\n",
                encoding="utf-8",
            )
            check = retained / "check_sa.f"
            check.write_text(
                "      USERHEL=-1\n"
                "      CALL SMATRIX(P,V)\n"
                "      CALL SMATRIX(P,V)\n"
                "      CALL SMATRIX(P,V)\n",
                encoding="ascii",
            )
            cell["provenance"].update(
                {
                    "source_alpha_s": 0.118,
                    "runtime_alpha_s": 0.118,
                    "timed_helicity_count": 2 ** (n + 2),
                    "matrix_sha256": _sha256(matrix),
                    "check_source_sha256": _sha256(check),
                    "retained": {
                        "matrix_source": str(matrix),
                        "matrix_sha256": _sha256(matrix),
                        "check_source": str(check),
                        "check_source_sha256": _sha256(check),
                    },
                    "source_report": {
                        "path": str(campaign_path),
                        "sha256": _sha256(campaign_path),
                        "cell": f"cells.{family}.{source_mode}.{n}",
                    },
                }
            )
    overlay_path = _write_json(tmp_path / "madgraph-sum.json", overlay)

    report = final_report.build_final_report(
        campaign_path=campaign_path, madgraph_overlay_path=overlay_path
    )

    assert report["status"] == "complete"
    assert (
        report["runtime_series"]["gg"]["madgraph-standalone"]["2"][
            "helicity_workload"
        ]
        == "sum"
    )
    assert report["policy"]["measurement"]["warm_fixed_helicity"] is False
    assert report["policy"]["measurement"]["warm_helicity_sum"] is True
    assert all(
        "omitted" not in note.lower()
        for note in report["policy"]["plot"]["notes"]
    )


def test_protocol_scoped_n2_n9_final_report_renders_all_six_plots(
    tmp_path: Path,
) -> None:
    _, campaign_path, _, overlay_path = _inputs(tmp_path)
    report = final_report.build_final_report(
        campaign_path=campaign_path, madgraph_overlay_path=overlay_path
    )
    output = tmp_path / "plots"

    plots._render(report, output, dpi=72)

    assert {path.name for path in output.glob("*.png")} == {
        f"fullcolor-{family}-{metric}.png"
        for family in ("gg", "ddbar")
        for metric in ("generation", "warm-runtime", "rss")
    }


def test_merge_accepts_one_amplicol_validation_value_per_measured_cell(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    for family_cells in campaign["cells"].values():
        for cell in family_cells["amplicol"].values():
            cell["point_values"] = cell["point_values"][:1]
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay_path = _write_json(
        tmp_path / "madgraph.json", _overlay(campaign, campaign_path)
    )

    report = final_report.build_final_report(
        campaign_path=campaign_path,
        madgraph_overlay_path=overlay_path,
    )

    assert report["cells"]["gg"]["amplicol"]["2"]["point_values"] == [201.0]


def test_merge_rejects_more_than_one_amplicol_validation_value(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    campaign["cells"]["gg"]["amplicol"]["2"]["point_values"] = [201.0, 202.0]
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay_path = _write_json(
        tmp_path / "madgraph.json", _overlay(campaign, campaign_path)
    )

    with pytest.raises(final_report.PublicationMergeError, match="exactly 1"):
        final_report.build_final_report(
            campaign_path=campaign_path,
            madgraph_overlay_path=overlay_path,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("fft_enabled",), False, "record --fft"),
        (
            ("measurement", "generation_helicity_coverage"),
            "selected",
            "generation_helicity_coverage",
        ),
        (
            ("measurement", "compiled_fft_enabled"),
            True,
            "compiled_fft_enabled",
        ),
    ),
)
def test_merge_rejects_incompatible_campaign_protocol(
    tmp_path: Path, path: tuple[str, ...], value: object, message: str
) -> None:
    campaign = _campaign(tmp_path)
    target = campaign["policy"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay_path = _write_json(
        tmp_path / "madgraph.json", _overlay(campaign, campaign_path)
    )

    with pytest.raises(final_report.PublicationMergeError, match=message):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )


def test_merge_rejects_measurement_after_a_campaign_frontier(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    failed = campaign["cells"]["gg"]["recurrence-direct"]["5"]
    failed["status"] = "failed"
    failed["failure_reason"] = "strict resource frontier"
    failed["censors_higher_multiplicities"] = True
    failed.pop("metrics")
    failed.pop("helicity")
    failed.pop("event_paths")
    failed.pop("point_values")
    campaign["status"] = "complete-with-failures"
    campaign["failure_count"] = 1
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay_path = _write_json(
        tmp_path / "madgraph.json", _overlay(campaign, campaign_path)
    )

    with pytest.raises(final_report.PublicationMergeError, match="resumes measurement"):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("execution_mode", "recurrence", r"execution_mode is wrong"),
        ("color_contraction", "direct", r"color_contraction is wrong"),
        (
            "generation_helicity_coverage",
            "selected",
            r"generation_helicity_coverage is wrong",
        ),
        ("warm_fixed_helicity", False, r"warm_fixed_helicity is wrong"),
    ),
)
def test_merge_authenticates_otf_cell_protocol(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    campaign = _campaign(tmp_path)
    campaign["cells"]["gg"]["otf-fft"]["4"][field] = value
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay_path = _write_json(
        tmp_path / "madgraph.json", _overlay(campaign, campaign_path)
    )

    with pytest.raises(final_report.PublicationMergeError, match=message):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("numerical", "failed its numerical comparison"),
        ("events", "must contain ten paths"),
    ),
)
def test_merge_preserves_otf_numerical_and_event_checks(
    tmp_path: Path, mutation: str, message: str
) -> None:
    campaign = _campaign(tmp_path)
    cell = campaign["cells"]["ddbar"]["otf-direct"]["3"]
    if mutation == "numerical":
        cell["numerical"]["passes"] = False
    else:
        cell["event_paths"] = cell["event_paths"][:-1]
    campaign_path = _write_json(tmp_path / "report.json", campaign)
    overlay_path = _write_json(
        tmp_path / "madgraph.json", _overlay(campaign, campaign_path)
    )

    with pytest.raises(final_report.PublicationMergeError, match=message):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )


def test_merge_preserves_a_madgraph_failure_frontier(tmp_path: Path) -> None:
    _, campaign_path, overlay, _ = _inputs(tmp_path)
    curve = overlay["runtime_series"]["gg"]["madgraph-standalone"]
    curve["6"] = {
        "status": "failed",
        "family": "gg",
        "mode": "madgraph-standalone",
        "label": "MadGraph standalone (fixed h)",
        "n": 6,
        "total_external": 8,
        "process": final_report._process_expression("gg", 6),
        "failure_reason": "strict resource frontier",
    }
    overlay["status"] = "complete-with-failures"
    overlay["failure_count"] = 1
    overlay["summary"]["runtime_series_status_counts"] = {
        "failed": 1,
        "measured": 9,
        "not-applicable": 6,
    }
    overlay_path = _write_json(tmp_path / "madgraph.json", overlay)

    report = final_report.build_final_report(
        campaign_path=campaign_path, madgraph_overlay_path=overlay_path
    )

    assert report["status"] == "complete-with-failures"
    assert report["failure_count"] == 1
    assert (
        report["runtime_series"]["gg"]["madgraph-standalone"]["9"]["status"]
        == "not-applicable"
    )


def test_merge_rejects_madgraph_event_or_source_provenance(tmp_path: Path) -> None:
    _, campaign_path, overlay, _ = _inputs(tmp_path)
    overlay["runtime_series"]["gg"]["madgraph-standalone"]["2"]["event_sha256"][0] = (
        "0" * 64
    )
    overlay_path = _write_json(tmp_path / "madgraph.json", overlay)

    with pytest.raises(final_report.PublicationMergeError, match="SHA-256 provenance"):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )


def test_merge_rejects_old_fixed_ddbar_normalization(tmp_path: Path) -> None:
    _, campaign_path, overlay, _ = _inputs(tmp_path)
    cell = overlay["runtime_series"]["ddbar"]["madgraph-standalone"]["4"]
    cell["numerical"]["normalization_factor_reference_per_madgraph"] = 1.0 / 36.0
    cell["provenance"]["smatrix_iden"] = 36
    overlay_path = _write_json(tmp_path / "madgraph.json", overlay)

    with pytest.raises(
        final_report.PublicationMergeError,
        match=r"incompatible normalization|wrong generated SMATRIX denominator",
    ):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )


def test_merge_rejects_non_strict_resource_metrics(tmp_path: Path) -> None:
    _, campaign_path, overlay, _ = _inputs(tmp_path)
    cell = overlay["runtime_series"]["ddbar"]["madgraph-standalone"]["3"]
    cell["generation_seconds"] = 3600.0
    cell["metrics"]["generation_seconds"] = 3600.0
    overlay_path = _write_json(tmp_path / "madgraph.json", overlay)

    with pytest.raises(final_report.PublicationMergeError, match="exceeds one hour"):
        final_report.build_final_report(
            campaign_path=campaign_path, madgraph_overlay_path=overlay_path
        )
