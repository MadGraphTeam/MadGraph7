#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Assemble the published FullColor scaling plots into one summary PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
    raise SystemExit(
        "fft_results_summary_pdf.py requires reportlab; use the bundled "
        "Codex PDF runtime or install reportlab"
    ) from error


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "IMPLEMENTATION_DOCS" / "RESULTS"
FINAL_REPORT = (
    RESULTS
    / "fft-scaling-study"
    / "data"
    / "campaign-report-scalar-selected-n2-n9-final.json"
)
PLOT_MANIFEST_NAME = "plot-manifest.json"
PLOT_FILENAMES = (
    "fullcolor-gg-generation.png",
    "fullcolor-gg-warm-runtime.png",
    "fullcolor-gg-rss.png",
    "fullcolor-ddbar-generation.png",
    "fullcolor-ddbar-warm-runtime.png",
    "fullcolor-ddbar-rss.png",
)


@dataclass(frozen=True, slots=True)
class PlotPage:
    section: str
    path: Path


def _plot_pages(
    plot_directory: Path | None = None,
    *,
    helicity_workload: str = "fixed",
) -> tuple[PlotPage, ...]:
    if helicity_workload not in {"fixed", "sum"}:
        raise ValueError("helicity_workload must be 'fixed' or 'sum'")
    plot_root = RESULTS / "fft-scaling-study" / "plots"
    return tuple(
        PlotPage(
            (
                "Current-host fixed-helicity comparison (n=2..9)"
                if helicity_workload == "fixed"
                else "Current-host helicity-summed comparison (n=2..9)"
            ),
            (
                plot_root
                / (
                    "scalar-selected-n2-n9-final"
                    if helicity_workload == "fixed"
                    else "scalar-helicity-sum-n2-n9-final"
                )
                if plot_directory is None
                else plot_directory
            )
            / name,
        )
        for name in PLOT_FILENAMES
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "summary_plots_final.pdf",
        help=(
            "output PDF (default: IMPLEMENTATION_DOCS/RESULTS/summary_plots_final.pdf)"
        ),
    )
    parser.add_argument(
        "--campaign-report",
        type=Path,
        default=FINAL_REPORT,
        help=(
            "report supplying FFT and helicity-coverage provenance for the PDF header"
        ),
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        help=(
            "directory containing the six plot PNGs (default: published final "
            "plot directory)"
        ),
    )
    return parser


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _campaign_note(report: Mapping[str, Any]) -> str:
    policy = _mapping(report.get("policy"), context="campaign policy")
    if policy.get("fft_enabled") is not True:
        raise ValueError("campaign policy must record fft_enabled=true")
    contractions = _mapping(
        policy.get("selected_pyamplicol_color_contractions"),
        context="selected pyAmpliCol color contractions",
    )
    expected_contractions = {"direct", "symmetric-group-fft"}
    for execution_mode in ("recurrence", "on-the-fly"):
        raw = contractions.get(execution_mode)
        selected = (
            {str(contraction) for contraction in raw}
            if isinstance(raw, Sequence) and not isinstance(raw, str)
            else set()
        )
        if selected != expected_contractions:
            raise ValueError(
                "campaign policy must select direct and symmetric-group FFT for "
                f"{execution_mode}"
            )
    measurement = _mapping(
        policy.get("measurement"), context="campaign measurement policy"
    )
    if measurement.get("generation_helicity_coverage") != "all":
        raise ValueError(
            "campaign policy must record generation_helicity_coverage='all'"
        )
    helicity_workload = _campaign_helicity_workload(policy, measurement)
    batch_size = measurement.get("warm_benchmark_batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(
            "campaign policy must record a positive warm_benchmark_batch_size"
        )
    _validate_campaign_cell_workloads(report, helicity_workload)
    if measurement.get("compiled_fft_enabled") is not False:
        raise ValueError("campaign policy must record compiled_fft_enabled=false")
    if helicity_workload == "fixed":
        note = (
            "--fft | recurrence: persisted all-helicity physical schedule | "
            "OTF compact artifact: all runtime helicities\n"
            "pyAmpliCol setup: generation + fresh load + first fixed-helicity "
            "evaluation; OTF includes family warm-up\n"
            "Reference setup: build/init/first pass | AmpliCol setup: "
            "process/color-object generation\n"
            "Warmed runtime measured separately | compiled FFT disabled\n"
            f"pyAmpliCol CLI profile: batch {batch_size} (cyclic 10 points) | "
            "Reference/AmpliCol: scalar aggregates normalized per point"
        )
    else:
        note = (
            "--fft | recurrence: persisted all-helicity physical schedule | "
            "OTF compact artifact: all runtime helicities\n"
            "pyAmpliCol setup: generation + fresh load + first complete helicity "
            "sum; OTF includes family warm-up\n"
            "Reference setup: build/init/first pass | AmpliCol setup: process/raw-"
            "library generation/build + immutable snapshot\n"
            "Warmed runtime measured separately | compiled FFT disabled\n"
            f"pyAmpliCol CLI profile: batch {batch_size} (cyclic 10 points) | "
            "Reference/AmpliCol: scalar aggregates normalized per point\n"
            "AmpliCol: create-raw bulk H family + probe-local zero pruning | "
            "Reference FFT: analytic-nonzero H sweep"
        )
    status = report.get("status")
    if not isinstance(status, str) or not status.startswith("complete"):
        return "IN PROGRESS SNAPSHOT | absent cells are unattempted\n" + note
    memory_limit = measurement.get("memory_watchdog_gib")
    generation_limit = measurement.get("generation_timeout_seconds")
    runtime_limit = measurement.get("runtime_timeout_seconds")
    if (memory_limit, generation_limit, runtime_limit) != (30.0, 3600.0, 3600.0):
        return "CLUSTER SCAN | configured per-cell resource caps\n" + note
    return note


def _campaign_helicity_workload(
    policy: Mapping[str, Any], measurement: Mapping[str, Any]
) -> str:
    measurement_declared = measurement.get("helicity_workload")
    policy_declared = policy.get("helicity_workload")
    if (
        measurement_declared is not None
        and policy_declared is not None
        and str(measurement_declared) != str(policy_declared)
    ):
        raise ValueError(
            "policy and measurement helicity_workload declarations disagree"
        )
    declared = (
        measurement_declared
        if measurement_declared is not None
        else policy_declared
    )
    if declared is not None:
        workload = str(declared)
        if workload not in {"fixed", "sum"}:
            raise ValueError("campaign helicity_workload must be 'fixed' or 'sum'")
        if workload == "sum" and (
            measurement.get("warm_fixed_helicity") is not False
            or measurement.get("warm_helicity_sum") is not True
        ):
            raise ValueError(
                "summed campaign must record warm_fixed_helicity=false and "
                "warm_helicity_sum=true"
            )
        if workload == "fixed" and (
            measurement.get("warm_fixed_helicity") is not True
            or measurement.get("warm_helicity_sum") is True
        ):
            raise ValueError(
                "fixed campaign must record warm_fixed_helicity=true without "
                "warm_helicity_sum=true"
            )
    elif measurement.get("warm_helicity_sum") is True:
        if measurement.get("warm_fixed_helicity") is True:
            raise ValueError("campaign helicity workload markers are contradictory")
        workload = "sum"
    elif measurement.get("warm_fixed_helicity") is True:
        workload = "fixed"
    else:
        raise ValueError(
            "campaign policy must identify a fixed or summed warm-helicity workload"
        )
    if workload == "fixed" and measurement.get("warm_fixed_helicity") is not True:
        raise ValueError("fixed campaign must record warm_fixed_helicity=true")
    if workload == "sum" and measurement.get("warm_fixed_helicity") is True:
        raise ValueError("summed campaign cannot record warm_fixed_helicity=true")
    return workload


def _cell_helicity_workload(cell: Mapping[str, Any]) -> str | None:
    markers: set[str] = set()
    declared = cell.get("helicity_workload")
    if declared is not None:
        workload = str(declared)
        if workload not in {"fixed", "sum"}:
            raise ValueError("measured cell helicity_workload must be 'fixed' or 'sum'")
        markers.add(workload)
    if cell.get("warm_helicity_sum") is True:
        markers.add("sum")
    if cell.get("warm_fixed_helicity") is True:
        markers.add("fixed")
    protocol = cell.get("protocol")
    if isinstance(protocol, Mapping):
        if protocol.get("helicity_summed") is True:
            markers.add("sum")
        if (
            protocol.get("helicity_summed") is False
            or protocol.get("warm_fixed_helicity") is True
        ):
            markers.add("fixed")
    if len(markers) > 1:
        raise ValueError("measured cell helicity workload markers are contradictory")
    return next(iter(markers), None)


def _validate_campaign_cell_workloads(
    report: Mapping[str, Any], expected: str
) -> None:
    for section_name in ("cells", "runtime_series"):
        section = report.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for family, raw_modes in section.items():
            if not isinstance(raw_modes, Mapping):
                continue
            for mode, raw_cells in raw_modes.items():
                if not isinstance(raw_cells, Mapping):
                    continue
                for raw_n, raw_cell in raw_cells.items():
                    if not isinstance(raw_cell, Mapping):
                        continue
                    if raw_cell.get("status") != "measured":
                        continue
                    observed = _cell_helicity_workload(raw_cell)
                    context = f"{section_name}.{family}.{mode}.n={raw_n}"
                    if expected == "sum" and observed != "sum":
                        raise ValueError(
                            f"{context} is not authenticated as helicity-summed"
                        )
                    if expected == "fixed" and observed == "sum":
                        raise ValueError(
                            f"{context} is summed data in a fixed-helicity campaign"
                        )


def _load_campaign_note(path: Path) -> str:
    report = _load_campaign(path)
    return _campaign_note(report)


def _load_campaign(path: Path) -> Mapping[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read campaign report {path}: {error}") from error
    if not isinstance(report, Mapping):
        raise ValueError("campaign report must be a JSON object")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_plot_manifest(
    plot_directory: Path,
    campaign_report: Path,
    helicity_workload: str,
) -> None:
    manifest_path = plot_directory / PLOT_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read plot manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise ValueError("plot manifest must be a JSON object")
    if (
        manifest.get("kind") != "pyamplicol-fft-scaling-plots"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("plot manifest has an unsupported identity or schema")
    if manifest.get("report_sha256") != _sha256(campaign_report):
        raise ValueError("plot manifest was rendered from a different campaign report")
    if manifest.get("helicity_workload") != helicity_workload:
        raise ValueError("plot manifest helicity workload does not match the campaign")
    raw_plots = manifest.get("plots")
    if not isinstance(raw_plots, Mapping) or set(raw_plots) != set(PLOT_FILENAMES):
        raise ValueError("plot manifest does not name the exact published plot set")
    for filename in PLOT_FILENAMES:
        plot_path = plot_directory / filename
        if raw_plots.get(filename) != _sha256(plot_path):
            raise ValueError(f"plot manifest hash does not match {filename}")


def _campaign_note_lines(
    campaign_note: str,
    *,
    available_width: float,
    font_name: str = "Helvetica",
    font_size: float = 6.6,
) -> tuple[str, ...]:
    """Preserve authored line breaks and wrap any line to the page width."""

    wrapped: list[str] = []
    for source_line in campaign_note.splitlines():
        words = source_line.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            candidate_width = pdfmetrics.stringWidth(
                candidate, font_name, font_size
            )
            if candidate_width <= available_width:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return tuple(wrapped)


def _campaign_header_height(note_lines: Sequence[str]) -> float:
    """Reserve a title band plus one eight-point row for every note line."""

    return max(31.0, 15.0 + 8.0 * len(note_lines))


def _draw_page(
    document: canvas.Canvas,
    page: PlotPage,
    *,
    page_number: int,
    page_count: int,
    campaign_note: str,
) -> None:
    width, height = landscape(A4)
    margin = 18.0
    footer_height = 13.0
    available_width = width - 2 * margin
    note_lines = _campaign_note_lines(
        campaign_note,
        available_width=available_width,
    )
    header_height = _campaign_header_height(note_lines)
    available_height = height - 2 * margin - header_height - footer_height

    image_width, image_height = ImageReader(page.path).getSize()
    scale = min(available_width / image_width, available_height / image_height)
    drawn_width = image_width * scale
    drawn_height = image_height * scale
    x = (width - drawn_width) / 2
    y = margin + footer_height + (available_height - drawn_height) / 2

    document.setFillColorRGB(1, 1, 1)
    document.rect(0, 0, width, height, fill=1, stroke=0)
    document.setFillColorRGB(0.14, 0.17, 0.20)
    document.setFont("Helvetica-Bold", 9)
    document.drawString(margin, height - margin - 7, page.section)
    document.setFont("Helvetica", 6.6)
    document.setFillColorRGB(0.35, 0.39, 0.43)
    for line_index, line in enumerate(note_lines):
        document.drawString(
            margin,
            height - margin - 15.5 - 8.0 * line_index,
            line,
        )
    document.drawImage(
        str(page.path),
        x,
        y,
        width=drawn_width,
        height=drawn_height,
        preserveAspectRatio=True,
        anchor="c",
    )
    document.setFillColorRGB(0.35, 0.39, 0.43)
    document.setFont("Helvetica", 7.5)
    document.drawRightString(
        width - margin,
        margin - 1,
        f"pyAmpliCol FullColor scaling results - page {page_number}/{page_count}",
    )
    document.showPage()


def main() -> int:
    arguments = _parser().parse_args()
    try:
        campaign_report = _load_campaign(arguments.campaign_report)
        campaign_note = _campaign_note(campaign_report)
        policy = _mapping(campaign_report.get("policy"), context="campaign policy")
        measurement = _mapping(
            policy.get("measurement"), context="campaign measurement policy"
        )
        helicity_workload = _campaign_helicity_workload(policy, measurement)
    except ValueError as error:
        raise SystemExit(f"invalid campaign provenance: {error}") from error
    pages = _plot_pages(
        arguments.plot_directory,
        helicity_workload=helicity_workload,
    )
    missing = [page.path for page in pages if not page.path.is_file()]
    if missing:
        raise SystemExit("missing published plot(s): " + ", ".join(map(str, missing)))
    try:
        _validate_plot_manifest(
            pages[0].path.parent,
            arguments.campaign_report,
            helicity_workload,
        )
    except ValueError as error:
        raise SystemExit(f"invalid plot provenance: {error}") from error

    output = arguments.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    document = canvas.Canvas(
        str(temporary),
        pagesize=landscape(A4),
        pageCompression=1,
        invariant=1,
    )
    document.setTitle(
        "pyAmpliCol FullColor scaling summary plots"
        if helicity_workload == "fixed"
        else "pyAmpliCol FullColor helicity-summed scaling summary plots"
    )
    document.setAuthor("pyAmpliCol")
    for index, page in enumerate(pages, start=1):
        _draw_page(
            document,
            page,
            page_number=index,
            page_count=len(pages),
            campaign_note=campaign_note,
        )
    document.save()
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
