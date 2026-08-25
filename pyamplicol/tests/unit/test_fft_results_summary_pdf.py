# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer import fft_results_summary_pdf as summary_pdf


def test_final_summary_pdf_contains_only_fresh_fixed_helicity_section() -> None:
    pages = summary_pdf._plot_pages()

    assert len(pages) == 6
    assert {page.section for page in pages} == {
        "Current-host fixed-helicity comparison (n=2..9)"
    }
    assert {page.path.parent.name for page in pages} == {"scalar-selected-n2-n9-final"}
    assert summary_pdf._parser().get_default("output").name == "summary_plots_final.pdf"
    assert (
        summary_pdf._parser().get_default("campaign_report") == summary_pdf.FINAL_REPORT
    )


def test_pdf_note_requires_and_records_fft_and_helicity_provenance() -> None:
    note = summary_pdf._campaign_note(
        {
            "status": "complete",
            "policy": {
                "fft_enabled": True,
                "selected_pyamplicol_color_contractions": {
                    "recurrence": ["direct", "symmetric-group-fft"],
                    "on-the-fly": ["direct", "symmetric-group-fft"],
                },
                    "measurement": {
                        "generation_helicity_coverage": "all",
                        "warm_fixed_helicity": True,
                        "warm_benchmark_batch_size": 128,
                        "compiled_fft_enabled": False,
                        "memory_watchdog_gib": 30.0,
                        "generation_timeout_seconds": 3600.0,
                        "runtime_timeout_seconds": 3600.0,
                    },
            }
        }
    )

    assert note.splitlines() == [
        "--fft | recurrence: persisted all-helicity physical schedule | "
        "OTF compact artifact: all runtime helicities",
        "pyAmpliCol setup: generation + fresh load + first fixed-helicity "
        "evaluation; OTF includes family warm-up",
        "Reference setup: build/init/first pass | AmpliCol setup: "
        "process/color-object generation",
        "Warmed runtime measured separately | compiled FFT disabled",
        "pyAmpliCol CLI profile: batch 128 (cyclic 10 points) | "
        "Reference/AmpliCol: scalar aggregates normalized per point",
    ]
    assert "compiled FFT disabled" in note


def test_pdf_note_and_section_record_helicity_sum() -> None:
    note = summary_pdf._campaign_note(
        {
            "policy": {
                "fft_enabled": True,
                "helicity_workload": "sum",
                "selected_pyamplicol_color_contractions": {
                    "recurrence": ["direct", "symmetric-group-fft"],
                    "on-the-fly": ["direct", "symmetric-group-fft"],
                },
                "measurement": {
                    "generation_helicity_coverage": "all",
                    "warm_fixed_helicity": False,
                    "warm_helicity_sum": True,
                    "warm_benchmark_batch_size": 128,
                    "compiled_fft_enabled": False,
                },
            }
        }
    )

    assert "first complete helicity sum" in note
    assert "Warmed runtime measured separately" in note
    assert "immutable snapshot" in note
    assert "create-raw bulk H family" in note
    assert "analytic-nonzero H sweep" in note
    pages = summary_pdf._plot_pages(helicity_workload="sum")
    assert {page.section for page in pages} == {
        "Current-host helicity-summed comparison (n=2..9)"
    }
    assert {page.path.parent.name for page in pages} == {
        "scalar-helicity-sum-n2-n9-final"
    }


def test_pdf_header_reserves_space_for_every_wrapped_note_line() -> None:
    note = (
        "IN PROGRESS SNAPSHOT | absent cells are unattempted\n"
        "--fft | recurrence: persisted all-helicity physical schedule | "
        "OTF compact artifact: all runtime helicities\n"
        "pyAmpliCol setup: generation + fresh load + first complete helicity "
        "sum; OTF includes family warm-up\n"
        "Reference setup: build/init/first pass | AmpliCol setup: process/raw-"
        "library generation/build + immutable snapshot\n"
        "Warmed runtime measured separately | compiled FFT disabled\n"
        "pyAmpliCol CLI profile: batch 128 (cyclic 10 points) | "
        "Reference/AmpliCol: scalar aggregates normalized per point\n"
        "AmpliCol: create-raw bulk H family + probe-local zero pruning | "
        "Reference FFT: analytic-nonzero H sweep"
    )

    lines = summary_pdf._campaign_note_lines(note, available_width=805.0)

    assert len(lines) >= len(note.splitlines())
    assert lines[-1].endswith("Reference FFT: analytic-nonzero H sweep")
    header_height = summary_pdf._campaign_header_height(lines)
    last_baseline_from_top = 15.5 + 8.0 * (len(lines) - 1)
    assert header_height == 15.0 + 8.0 * len(lines)
    assert header_height - last_baseline_from_top >= 7.0


def test_pdf_note_rejects_fixed_helicity_cell_in_sum_campaign() -> None:
    report = {
        "policy": {
            "fft_enabled": True,
            "helicity_workload": "sum",
            "selected_pyamplicol_color_contractions": {
                "recurrence": ["direct", "symmetric-group-fft"],
                "on-the-fly": ["direct", "symmetric-group-fft"],
            },
            "measurement": {
                "generation_helicity_coverage": "all",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
                "warm_benchmark_batch_size": 128,
                "compiled_fft_enabled": False,
            },
        },
        "runtime_series": {
            "gg": {
                "madgraph-standalone": {
                    "2": {
                        "status": "measured",
                        "protocol": {"helicity_summed": False},
                    }
                }
            }
        },
    }

    with pytest.raises(ValueError, match="not authenticated as helicity-summed"):
        summary_pdf._campaign_note(report)


def test_pdf_rejects_contradictory_cell_workload_markers() -> None:
    with pytest.raises(ValueError, match="markers are contradictory"):
        summary_pdf._cell_helicity_workload(
            {"helicity_workload": "sum", "warm_fixed_helicity": True}
        )


def test_pdf_rejects_disagreeing_policy_and_measurement_workloads() -> None:
    with pytest.raises(ValueError, match="declarations disagree"):
        summary_pdf._campaign_helicity_workload(
            {"helicity_workload": "fixed"},
            {
                "helicity_workload": "sum",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
            },
        )


def test_pdf_plot_manifest_rejects_changed_plot(tmp_path: Path) -> None:
    campaign_report = tmp_path / "report.json"
    campaign_report.write_text("{}\n", encoding="utf-8")
    plot_directory = tmp_path / "plots"
    plot_directory.mkdir()
    hashes: dict[str, str] = {}
    for index, filename in enumerate(summary_pdf.PLOT_FILENAMES):
        plot_path = plot_directory / filename
        plot_path.write_bytes(f"png-{index}".encode())
        hashes[filename] = summary_pdf._sha256(plot_path)
    (plot_directory / summary_pdf.PLOT_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "kind": "pyamplicol-fft-scaling-plots",
                "schema_version": 1,
                "report_sha256": summary_pdf._sha256(campaign_report),
                "helicity_workload": "sum",
                "plots": hashes,
            }
        ),
        encoding="utf-8",
    )

    summary_pdf._validate_plot_manifest(
        plot_directory, campaign_report, "sum"
    )
    (plot_directory / summary_pdf.PLOT_FILENAMES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash does not match"):
        summary_pdf._validate_plot_manifest(
            plot_directory, campaign_report, "sum"
        )


def test_pdf_validates_and_publishes_raw_plot_data_sidecar(tmp_path: Path) -> None:
    campaign_report = tmp_path / "report.json"
    campaign_report.write_text("{}\n", encoding="utf-8")
    plot_directory = tmp_path / "plots"
    plot_directory.mkdir()
    raw_data = (
        json.dumps(
            {
                "kind": "pyamplicol-fft-scaling-plot-data",
                "schema_version": 1,
                "report_sha256": summary_pdf._sha256(campaign_report),
                "helicity_workload": "fixed",
                "families": {"gg": {}, "ddbar": {}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    (plot_directory / summary_pdf.PLOT_DATA_NAME).write_bytes(raw_data)

    assert (
        summary_pdf._validated_plot_data_bytes(
            plot_directory, campaign_report, "fixed"
        )
        == raw_data
    )
    destination = summary_pdf._publish_plot_data_sidecar(
        raw_data, tmp_path / "summary_plots_final.pdf"
    )

    assert destination == tmp_path / "summary_plots_final.json"
    assert destination.read_bytes() == raw_data


def test_pdf_note_rejects_missing_otf_comparison() -> None:
    report = {
        "policy": {
            "fft_enabled": True,
            "selected_pyamplicol_color_contractions": {
                "recurrence": ["direct", "symmetric-group-fft"]
            },
        }
    }

    with pytest.raises(ValueError, match="for on-the-fly"):
        summary_pdf._campaign_note(report)
