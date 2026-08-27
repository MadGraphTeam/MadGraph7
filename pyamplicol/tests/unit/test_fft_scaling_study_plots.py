# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer import fft_scaling_study_plots as plots


def _report(boundary: object = 6) -> dict[str, object]:
    plot_policy = (
        {}
        if boundary is None
        else {
            "source_boundary_after_n": boundary,
            "source_boundary_label": "n=2..6 / n=7..9 source boundary",
        }
    )
    return {
        "policy": {
            "final_state_multiplicities": list(range(2, 10)),
            "plot": plot_policy,
        },
        "cells": {},
    }


def test_source_boundary_is_drawn_between_n6_and_n7() -> None:
    assert plots._source_boundary(_report()) == (
        6.5,
        "n=2..6 / n=7..9 source boundary",
    )


def test_source_boundary_is_optional() -> None:
    assert plots._source_boundary(_report(None)) is None


def test_source_boundary_can_be_disabled_for_only_the_refreshed_metric() -> None:
    report = _report()
    report["policy"]["plot"]["metric_source_boundaries"] = {
        "generation_seconds": {
            "source_boundary_after_n": 6,
            "source_boundary_label": "generation boundary",
        },
        "warm_seconds_per_point": None,
    }

    assert plots._source_boundary(report, "warm_seconds_per_point") is None
    assert plots._source_boundary(report, "generation_seconds") == (
        6.5,
        "generation boundary",
    )
    assert plots._source_boundary(report, "max_rss_kib") == (
        6.5,
        "n=2..6 / n=7..9 source boundary",
    )


def test_protocol_summary_records_fft_and_helicity_contract() -> None:
    report = _report()
    report["policy"].update(
        {
            "fft_enabled": True,
            "measurement": {
                "generation_helicity_coverage": "all",
                "warm_fixed_helicity": True,
                "warm_benchmark_batch_size": 128,
                "compiled_fft_enabled": False,
            },
        }
    )

    assert plots._protocol_summary(report) == (
        "Campaign --fft enabled | artifacts: all runtime helicities | "
        "setup/warm workload: one fixed helicity | pyAmpliCol CLI profile: batch 128 "
        "(cyclic 10 points) | Reference/AmpliCol: scalar aggregates normalized "
        "per point | compiled FFT disabled | "
        "main Y log; ratio Y log"
    )
    assert plots.METRICS[0].title == "Setup time"


def test_protocol_summary_and_warm_title_record_helicity_sum() -> None:
    report = _report()
    report["policy"].update(
        {
            "fft_enabled": True,
            "helicity_workload": "sum",
            "measurement": {
                "generation_helicity_coverage": "all",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
                "warm_benchmark_batch_size": 128,
                "compiled_fft_enabled": False,
            },
        }
    )

    assert plots._protocol_summary(report) == (
        "Campaign --fft enabled | artifacts: all runtime helicities | "
        "setup/warm workload: complete helicity sum | "
        "AmpliCol: create-raw bulk H family | pyAmpliCol CLI profile: batch 128 "
        "(cyclic 10 points) | Reference/AmpliCol: scalar aggregates normalized "
        "per point | compiled FFT disabled | "
        "main Y log; ratio Y log"
    )
    assert plots._metric_title(report, plots.METRICS[1]) == (
        "Warmed helicity-summed runtime per point"
    )
    assert plots._metric_axis_label(report, plots.METRICS[1]) == (
        "Time per helicity-summed point [s]"
    )
    protocol_lines = plots._protocol_summary_lines(report)
    assert protocol_lines == (
        "Campaign --fft enabled | artifacts: all runtime helicities | "
        "setup/warm workload: complete helicity sum",
        "AmpliCol: create-raw bulk H family | pyAmpliCol CLI profile: batch 128 "
        "(cyclic 10 points)",
        "Reference/AmpliCol: scalar aggregates normalized per point | "
        "compiled FFT disabled | main Y log; ratio Y log",
    )
    assert max(map(len, protocol_lines)) <= 110


def test_sum_plot_rejects_fixed_helicity_external_series() -> None:
    report = _report(None)
    report["policy"].update(
        {
            "fft_enabled": True,
            "helicity_workload": "sum",
            "process_families": {"gg": {"modes": ["reference-fft"]}},
            "measurement": {
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
            },
        }
    )
    report["cells"] = {
        "gg": {
            "reference-fft": {
                "2": {
                    "status": "measured",
                    "helicity_workload": "sum",
                    "metrics": {"warm_seconds_per_point": 1.0},
                }
            }
        }
    }
    report["runtime_series"] = {
        "gg": {
            "madgraph-standalone": {
                "2": {
                    "status": "measured",
                    "protocol": {"helicity_summed": False},
                    "metrics": {"warm_seconds_per_point": 2.0},
                }
            }
        }
    }

    with pytest.raises(plots.PlotError, match="not authenticated as a helicity-summed"):
        plots._family_cells_for_metric(report, "gg", plots.METRICS[1])


def test_plot_rejects_contradictory_helicity_workload_markers() -> None:
    report = _report(None)
    report["policy"].update(
        {
            "fft_enabled": True,
            "helicity_workload": "sum",
            "measurement": {
                "warm_fixed_helicity": True,
                "warm_helicity_sum": True,
            },
        }
    )
    with pytest.raises(plots.PlotError, match="summed policy"):
        plots._protocol_summary(report)

    cell = {
        "helicity_workload": "sum",
        "warm_fixed_helicity": True,
    }
    with pytest.raises(plots.PlotError, match="markers are contradictory"):
        plots._declared_cell_workload(cell)


def test_plot_rejects_disagreeing_policy_and_measurement_workloads() -> None:
    report = _report(None)
    report["policy"].update(
        {
            "helicity_workload": "fixed",
            "measurement": {
                "helicity_workload": "sum",
                "warm_fixed_helicity": False,
                "warm_helicity_sum": True,
            },
        }
    )

    with pytest.raises(plots.PlotError, match="declarations disagree"):
        plots._helicity_workload(report)


def test_plot_manifest_binds_report_workload_and_exact_pngs(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report = _report(None)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_directory = tmp_path / "plots"
    output_directory.mkdir()
    for index, filename in enumerate(plots.PLOT_FILENAMES):
        (output_directory / filename).write_bytes(f"png-{index}".encode())

    manifest_path = plots._write_plot_manifest(
        plots._sha256(report_path), report, output_directory
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["report_sha256"] == plots._sha256(report_path)
    assert manifest["helicity_workload"] == "fixed"
    assert set(manifest["plots"]) == set(plots.PLOT_FILENAMES)
    assert manifest["plots"][plots.PLOT_FILENAMES[0]] == plots._sha256(
        output_directory / plots.PLOT_FILENAMES[0]
    )


def test_plot_data_payload_records_plotted_values_and_ratios(tmp_path: Path) -> None:
    report = _report(None)
    report["policy"].update(
        {
            "process_families": {
                "gg": {"modes": ["reference-fft", "recurrence-direct"]},
                "ddbar": {"modes": ["amplicol"]},
            },
            "measurement": {"warm_fixed_helicity": True},
        }
    )
    measured = {
        "status": "measured",
        "warm_fixed_helicity": True,
    }
    report["cells"] = {
        "gg": {
            "reference-fft": {
                "2": measured
                | {
                    "label": "Reference FFT",
                    "metrics": {
                        "generation_seconds": 2.0,
                        "warm_seconds_per_point": 0.2,
                        "max_rss_kib": 2048.0,
                    },
                }
            },
            "recurrence-direct": {
                "2": measured
                | {
                    "metrics": {
                        "generation_seconds": 4.0,
                        "warm_seconds_per_point": 0.1,
                        "max_rss_kib": 4096.0,
                    },
                }
            },
        },
        "ddbar": {
            "amplicol": {
                "2": measured
                | {
                    "metrics": {
                        "generation_seconds": 3.0,
                        "warm_seconds_per_point": 0.3,
                        "max_rss_kib": 1024.0,
                    },
                }
            }
        },
    }

    payload = plots._plot_data_payload("abc123", report)
    output_directory = tmp_path / "plots"
    output_directory.mkdir()
    data_path = plots._write_plot_data("abc123", report, output_directory)
    persisted = json.loads(data_path.read_text(encoding="utf-8"))

    generation = payload["families"]["gg"]["metrics"]["generation"]
    rss = payload["families"]["gg"]["metrics"]["rss"]

    assert persisted == payload
    assert payload["kind"] == "pyamplicol-fft-scaling-plot-data"
    assert generation["source_metric_key"] == "generation_seconds"
    assert generation["modes"]["reference-fft"]["series"] == [
        {"n": 2, "value": 2.0}
    ]
    assert generation["modes"]["recurrence-direct"]["ratio_to_baseline"] == [
        {"n": 2, "value": 2.0}
    ]
    assert rss["axis_label"] == "Peak RSS [MiB]"
    assert rss["modes"]["reference-fft"]["series"] == [{"n": 2, "value": 2.0}]


def test_plot_data_payload_records_custom_axis_options() -> None:
    report = _report(None)
    report["policy"].update(
        {
            "process_families": {
                "gg": {"modes": ["reference-fft", "recurrence-direct"]},
                "ddbar": {"modes": ["amplicol"]},
            },
            "measurement": {"warm_fixed_helicity": True},
        }
    )
    measured = {
        "status": "measured",
        "warm_fixed_helicity": True,
    }
    report["cells"] = {
        "gg": {
            "reference-fft": {
                "2": measured
                | {"metrics": {"generation_seconds": 2.0}},
            },
            "recurrence-direct": {
                "2": measured
                | {"metrics": {"generation_seconds": 4.0}},
            },
        },
        "ddbar": {
            "amplicol": {
                "2": measured
                | {"metrics": {"generation_seconds": 3.0}},
            }
        },
    }
    axis_options = plots.AxisOptions(
        main_y_range=(1.0, 10.0),
        ratio_y_range=(0.0, 3.0),
        ratio_y_scale="linear",
    )

    payload = plots._plot_data_payload(
        "abc123",
        report,
        axis_options=axis_options,
    )
    generation = payload["families"]["gg"]["metrics"]["generation"]

    assert generation["main_axis"] == {"scale": "log", "range": [1.0, 10.0]}
    assert generation["ratio_axis"] == {
        "scale": "linear",
        "range": [0.0, 3.0],
    }


def test_plot_data_payload_filters_main_and_ratio_lines_independently() -> None:
    report = _report(None)
    report["policy"].update(
        {
            "process_families": {
                "gg": {
                    "modes": [
                        "reference-fft",
                        "recurrence-direct",
                        "madgraph-standalone",
                    ]
                },
                "ddbar": {"modes": ["amplicol"]},
            },
            "measurement": {"warm_fixed_helicity": True},
        }
    )
    measured = {
        "status": "measured",
        "warm_fixed_helicity": True,
    }
    report["cells"] = {
        "gg": {
            "reference-fft": {
                "2": measured | {"metrics": {"generation_seconds": 2.0}},
            },
            "recurrence-direct": {
                "2": measured | {"metrics": {"generation_seconds": 4.0}},
            },
            "madgraph-standalone": {
                "2": measured | {"metrics": {"generation_seconds": 6.0}},
            },
        },
        "ddbar": {
            "amplicol": {
                "2": measured | {"metrics": {"generation_seconds": 3.0}},
            }
        },
    }
    line_filters = plots.LineFilterOptions(
        main_include_lines=("reference-fft",),
        main_veto_lines=("reference-fft",),
        ratio_veto_lines=("pyamplicol-recurrence",),
    )

    payload = plots._plot_data_payload(
        "abc123",
        report,
        line_filters=line_filters,
    )
    generation = payload["families"]["gg"]["metrics"]["generation"]

    assert payload["line_filters"]["main"]["include_lines"] == ["reference-fft"]
    assert payload["line_filters"]["main"]["veto_lines"] == ["reference-fft"]
    assert generation["main_axis"]["range"] == pytest.approx([1.0, 4.0])
    assert generation["ratio_axis"]["range"] == pytest.approx(
        [10**-0.1, 3.0 * 10**0.1]
    )
    assert set(generation["modes"]) == {"reference-fft", "madgraph-standalone"}
    assert generation["modes"]["reference-fft"]["visible_in_main"] is True
    assert generation["modes"]["reference-fft"]["visible_in_ratio"] is True
    assert generation["modes"]["reference-fft"]["series"] == [
        {"n": 2, "value": 2.0}
    ]
    assert generation["modes"]["madgraph-standalone"]["visible_in_main"] is False
    assert generation["modes"]["madgraph-standalone"]["visible_in_ratio"] is True
    assert generation["modes"]["madgraph-standalone"]["series"] == []
    assert generation["modes"]["madgraph-standalone"]["ratio_to_baseline"] == [
        {"n": 2, "value": 3.0}
    ]


def test_axis_options_validate_log_positive_ranges() -> None:
    linear = plots._parser().parse_args(
        [
            "report.json",
            "plots",
            "--ratio-y-scale",
            "linear",
            "--ratio-y-range",
            "0",
            "2",
        ]
    )
    assert plots._axis_options_from_arguments(linear).ratio_y_range == (0.0, 2.0)

    log = plots._parser().parse_args(
        ["report.json", "plots", "--ratio-y-range", "0", "2"]
    )
    with pytest.raises(plots.PlotError, match="logarithmic y-axis"):
        plots._axis_options_from_arguments(log)


def test_recola_results_overlay_is_plotted_and_exported(tmp_path: Path) -> None:
    report = _report(None)
    report["policy"].update(
        {
            "process_families": {
                "gg": {"modes": ["reference-fft"]},
                "ddbar": {"modes": ["amplicol"]},
            },
            "measurement": {"warm_fixed_helicity": True},
        }
    )
    report["status"] = "complete"
    measured = {
        "status": "measured",
        "warm_fixed_helicity": True,
    }
    report["cells"] = {
        "gg": {
            "reference-fft": {
                "2": measured
                | {
                    "metrics": {
                        "generation_seconds": 2.0,
                        "warm_seconds_per_point": 0.2,
                        "max_rss_kib": 2048.0,
                    }
                }
            }
        },
        "ddbar": {
            "amplicol": {
                "2": measured
                | {
                    "metrics": {
                        "generation_seconds": 4.0,
                        "warm_seconds_per_point": 0.4,
                        "max_rss_kib": 4096.0,
                    }
                }
            }
        },
    }
    recola_path = tmp_path / "recola.json"
    recola_path.write_text(
        json.dumps(
            {
                "config": {"polarized": True},
                "results": {
                    "all_gluon": {
                        "2": {
                            "family": "all_gluon",
                            "process": "g g -> g g",
                            "gen_time_s": 5.0,
                            "run_time_s": 0.5,
                            "ram_after_generation_mib": 7.0,
                            "ram_after_profile_mib": 8.0,
                            "peak_generation_rss_mib": 6.0,
                            "polarized": True,
                            "helicity": [1, 1, 1, 1],
                            "profiled_call": "get_polarized_squared_amplitude_rcl",
                        },
                        "3": None,
                        "4": {
                            "status": "limit_reached",
                            "error": "all_gluon, n=4: generation time exceeded",
                        },
                    },
                    "down_quark_qcd": {
                        "2": {
                            "family": "down_quark_qcd",
                            "process": "d d~ -> d d~",
                            "gen_time_s": 6.0,
                            "run_time_s": 0.6,
                            "peak_generation_rss_mib": 9.0,
                            "polarized": True,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = plots._attach_recola_results(report, recola_path)
    external_sources = {"recola_results": metadata}
    payload = plots._plot_data_payload(
        "abc123",
        report,
        external_sources=external_sources,
    )
    output_directory = tmp_path / "plots"
    output_directory.mkdir()
    data_path = plots._write_plot_data(
        "abc123",
        report,
        output_directory,
        external_sources=external_sources,
    )
    persisted = json.loads(data_path.read_text(encoding="utf-8"))

    generation = payload["families"]["gg"]["metrics"]["generation"]
    warm = payload["families"]["gg"]["metrics"]["warm-runtime"]
    rss = payload["families"]["gg"]["metrics"]["rss"]
    ddbar_generation = payload["families"]["ddbar"]["metrics"]["generation"]

    assert payload["external_sources"]["recola_results"]["sha256"] == plots._sha256(
        recola_path
    )
    assert persisted == payload
    assert generation["modes"]["recola"]["label"] == "Recola"
    assert generation["modes"]["recola"]["series"] == [{"n": 2, "value": 5.0}]
    assert generation["modes"]["recola"]["ratio_to_baseline"] == [
        {"n": 2, "value": 2.5}
    ]
    assert warm["modes"]["recola"]["series"] == [{"n": 2, "value": 0.5}]
    assert rss["modes"]["recola"]["series"] == [{"n": 2, "value": 8.0}]
    assert ddbar_generation["modes"]["recola"]["series"] == [
        {"n": 2, "value": 6.0}
    ]
    assert plots._status_notes(
        report,
        "gg",
        ("recola",),
        family_cells=report["runtime_series"]["gg"],
    ) == ["Recola: n=4 failed (configured setup-time limit)."]


def test_recola_results_overlay_rejects_workload_mismatch(tmp_path: Path) -> None:
    report = _report(None)
    report["policy"].update(
        {
            "process_families": {"gg": {"modes": ["reference-fft"]}},
            "measurement": {"warm_fixed_helicity": True},
        }
    )
    report["cells"] = {"gg": {"reference-fft": {}}}
    recola_path = tmp_path / "recola-unpolarized.json"
    recola_path.write_text(
        json.dumps(
            {
                "config": {"polarized": False},
                "results": {
                    "all_gluon": {
                        "2": {
                            "family": "all_gluon",
                            "gen_time_s": 1.0,
                            "run_time_s": 0.1,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(plots.PlotError, match="campaign report is fixed"):
        plots._attach_recola_results(report, recola_path)


def test_final_series_palette_line_styles_and_legend_grouping() -> None:
    assert plots.MODE_STYLES["reference-fft"] == ("#000000", "o", "-")
    assert plots.MODE_STYLES["amplicol"] == ("#F0E442", "D", "-")
    assert plots.MODE_STYLES["madgraph-standalone"] == ("#CC3311", "*", "-")
    assert plots.MODE_STYLES["recurrence-fft"] == ("#009E73", "s", "-")
    assert plots.MODE_STYLES["recurrence-direct"] == ("#009E73", "o", "--")
    assert plots.MODE_STYLES["otf-fft"] == ("#0072B2", "v", "-")
    assert plots.MODE_STYLES["otf-direct"] == ("#0072B2", "^", "--")
    assert plots.MODE_STYLES["recola"] == ("#AA4499", "h", "-")

    assert plots.LEGEND_MODE_ORDER["gg"] == (
        "reference-fft",
        "amplicol",
        "recurrence-fft",
        "recurrence-direct",
        "otf-fft",
        "otf-direct",
        "madgraph-standalone",
        "recola",
    )
    assert plots.LEGEND_MODE_ORDER["ddbar"] == (
        "amplicol",
        "madgraph-standalone",
        "recurrence-fft",
        "recurrence-direct",
        "otf-fft",
        "otf-direct",
        "recola",
    )


def test_renderer_includes_measured_otf_cells_beyond_default_frontier() -> None:
    family_cells = {
        mode: {
            str(n): {
                "status": "measured",
                "label": label,
                "metrics": {"warm_seconds_per_point": float(n)},
            }
            for n in (6, 7)
        }
        for mode, label in (
            ("otf-direct", "pyAmpliCol - OTF"),
            ("otf-fft", "pyAmpliCol - OTF - FFT"),
            ("recurrence-fft", "pyAmpliCol - recurrence - FFT"),
        )
    }

    assert plots._series(family_cells, "otf-direct", plots.METRICS[1]) == [
        (6, 6.0),
        (7, 7.0),
    ]
    assert plots._series(family_cells, "otf-fft", plots.METRICS[1]) == [
        (6, 6.0),
        (7, 7.0),
    ]
    assert plots._series(family_cells, "recurrence-fft", plots.METRICS[1]) == [
        (6, 6.0),
        (7, 7.0),
    ]
    assert (
        plots._plot_policy_notes(
            _report(None), "ddbar", plots.METRICS[1], family_cells
        )
        == []
    )


def test_madgraph_external_series_is_added_to_all_three_metrics() -> None:
    report = _report()
    report["policy"]["process_families"] = {"gg": {"modes": ["reference-fft"]}}
    report["cells"] = {
        "gg": {
            "reference-fft": {
                "2": {
                    "status": "measured",
                    "metrics": {
                        "generation_seconds": 2.0,
                        "warm_seconds_per_point": 3.0,
                    },
                }
            }
        }
    }
    report["runtime_series"] = {
        "gg": {
            "madgraph-standalone": {
                "2": {
                    "status": "measured",
                    "metrics": {
                        "generation_seconds": 5.0,
                        "warm_seconds_per_point": 4.0,
                        "max_rss_kib": 6144.0,
                    },
                }
            }
        }
    }

    warm_cells, warm_modes = plots._family_cells_for_metric(
        report, "gg", plots.METRICS[1]
    )
    generation_cells, generation_modes = plots._family_cells_for_metric(
        report, "gg", plots.METRICS[0]
    )
    rss_cells, rss_modes = plots._family_cells_for_metric(
        report, "gg", plots.METRICS[2]
    )

    assert warm_modes == ("reference-fft", "madgraph-standalone")
    assert plots._series(warm_cells, "madgraph-standalone", plots.METRICS[1]) == [
        (2, 4.0)
    ]
    assert generation_modes == ("reference-fft", "madgraph-standalone")
    assert plots._series(generation_cells, "madgraph-standalone", plots.METRICS[0]) == [
        (2, 5.0)
    ]
    assert rss_modes == ("reference-fft", "madgraph-standalone")
    assert plots._series(rss_cells, "madgraph-standalone", plots.METRICS[2]) == [
        (2, 6.0)
    ]


def test_metric_series_exclusion_omits_only_invalid_warm_curve() -> None:
    report = _report(None)
    report["policy"].update(
        {
            "process_families": {"gg": {"modes": ["reference-fft", "recurrence-fft"]}},
            "plot": {
                "metric_series_exclusions": {
                    "warm_seconds_per_point": {
                        "gg": {"recurrence-fft": "driver evaluated all helicities"}
                    }
                }
            },
        }
    )
    report["cells"] = {
        "gg": {
            mode: {
                "2": {
                    "status": "measured",
                    "label": label,
                    "metrics": {
                        "generation_seconds": generation,
                        "warm_seconds_per_point": warm,
                    },
                }
            }
            for mode, label, generation, warm in (
                ("reference-fft", "Reference FFT", 1.0, 2.0),
                ("recurrence-fft", "pyAmpliCol recurrence FFT", 3.0, 4.0),
            )
        }
    }

    warm_cells, warm_modes = plots._family_cells_for_metric(
        report, "gg", plots.METRICS[1]
    )
    generation_cells, generation_modes = plots._family_cells_for_metric(
        report, "gg", plots.METRICS[0]
    )

    assert warm_modes == ("reference-fft",)
    assert "recurrence-fft" not in warm_cells
    assert generation_modes == ("reference-fft", "recurrence-fft")
    assert plots._series(generation_cells, "recurrence-fft", plots.METRICS[0]) == [
        (2, 3.0)
    ]
    assert plots._plot_policy_notes(
        report, "gg", plots.METRICS[1], report["cells"]["gg"]
    ) == [
        "pyAmpliCol recurrence FFT: warmed runtime per point omitted "
        "(driver evaluated all helicities)."
    ]


def test_stopped_protocol_investigation_is_not_described_as_running() -> None:
    report = _report(None)
    report.update(
        {
            "status": "stopped-protocol-investigation",
            "status_reason": "selected-output timing mismatch",
        }
    )

    assert plots._status_notes(report, "gg", ()) == [
        "Campaign stopped: selected-output timing mismatch. Absent cells are "
        "unattempted and omitted."
    ]


def test_generation_timeout_is_presented_as_setup_time() -> None:
    report = _report(None)
    report["policy"]["measurement"] = {"generation_timeout_seconds": 300.0}
    cell = {"failure_category": "generation-time-limit"}

    assert plots._failure_detail(report, cell) == "5 min setup-time limit"


def test_stopped_protocol_investigation_allows_empty_metric_panel(tmp_path) -> None:
    report = _report(None)
    report.update(
        {
            "status": "stopped-protocol-investigation",
            "status_reason": "all warm measurements are withheld",
            "cells": {"ddbar": {"amplicol": {}}},
        }
    )
    report["policy"]["process_families"] = {"ddbar": {"modes": ["amplicol"]}}

    output = tmp_path / "empty-warm.png"
    plots._plot_metric(report, "ddbar", plots.METRICS[1], output, dpi=72)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_complete_report_rejects_empty_metric_panel(tmp_path) -> None:
    report = _report(None)
    report.update({"status": "complete", "cells": {"ddbar": {"amplicol": {}}}})
    report["policy"]["process_families"] = {"ddbar": {"modes": ["amplicol"]}}

    with pytest.raises(plots.PlotError, match="no measured positive"):
        plots._plot_metric(
            report,
            "ddbar",
            plots.METRICS[1],
            tmp_path / "must-not-render.png",
            dpi=72,
        )


def test_legacy_warm_quality_advisory_is_an_ordinary_series_point() -> None:
    family_cells = {
        "recurrence-fft": {
            "3": {
                "status": "measured",
                "metrics": {"warm_seconds_per_point": 3.0},
                "excluded_metrics": {
                    "warm_seconds_per_point": {
                        "status": "excluded-provisional-warm-quality",
                        "value": 4.0,
                    }
                },
            }
        }
    }

    assert plots._series(
        family_cells, "recurrence-fft", plots.METRICS[1]
    ) == [(3, 4.0)]


def test_other_metric_exclusion_reasons_remain_invisible() -> None:
    family_cells = {
        "recurrence-fft": {
            "3": {
                "status": "measured",
                "metrics": {"warm_seconds_per_point": 3.0},
                "excluded_metrics": {
                    "warm_seconds_per_point": {
                        "status": "excluded-runtime-selection-semantics",
                        "value": 4.0,
                    }
                },
            }
        }
    }

    assert plots._series(
        family_cells, "recurrence-fft", plots.METRICS[1]
    ) == []


def test_legacy_warm_quality_advisory_does_not_break_curve() -> None:
    family_cells = {
        "recurrence-fft": {
            "2": {
                "status": "measured",
                "metrics": {"warm_seconds_per_point": 2.0},
            },
            "3": {
                "status": "measured",
                "metrics": {},
                "excluded_metrics": {
                    "warm_seconds_per_point": {
                        "status": "excluded-provisional-warm-quality",
                        "value": 3.0,
                    }
                },
            },
            "4": {
                "status": "measured",
                "metrics": {"warm_seconds_per_point": 4.0},
            },
        }
    }

    series = plots._series(family_cells, "recurrence-fft", plots.METRICS[1])

    assert series == [(2, 2.0), (3, 3.0), (4, 4.0)]
    assert list(plots._consecutive_runs(series)) == [series]


def test_complete_panel_can_render_only_legacy_warm_quality_advisory(
    tmp_path,
) -> None:
    report = _report(None)
    report.update(
        {
            "status": "complete",
            "cells": {
                "ddbar": {
                    "amplicol": {
                        "2": {
                            "status": "measured",
                            "label": "AmpliCol",
                            "metrics": {},
                            "excluded_metrics": {
                                "warm_seconds_per_point": {
                                    "status": (
                                        "excluded-provisional-warm-quality"
                                    ),
                                    "value": 2.5,
                                }
                            },
                        }
                    }
                }
            },
        }
    )
    report["policy"]["process_families"] = {
        "ddbar": {"modes": ["amplicol"]}
    }

    output = tmp_path / "legacy-advisory-only-warm.png"
    plots._plot_metric(report, "ddbar", plots.METRICS[1], output, dpi=72)

    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("boundary", (6.0, True, 9))
def test_source_boundary_rejects_ambiguous_values(boundary: object) -> None:
    with pytest.raises(plots.PlotError, match="source_boundary_after_n"):
        plots._source_boundary(_report(boundary))
