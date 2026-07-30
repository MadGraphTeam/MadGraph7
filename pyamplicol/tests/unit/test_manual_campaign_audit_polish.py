# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest
from colorama import Fore

import tools.performance_report.manual_campaign as manual_campaign
from pyamplicol.cli import parse_cli
from tools.performance_report.arena_profile import PAIRED_TIMING_SAMPLE_CONTRACT
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import (
    LightweightCurrent,
    ManualCampaignError,
    Palette,
    build_parser,
    reproduction_recipe,
    selection_from_arguments,
)
from tools.performance_report.models import ExecutionMode, ModelKey, ResultStatus
from tools.performance_report.timing import (
    MEASURED_EXECUTION_TIMING_ABI,
    RECURRENCE_EXECUTION_TIMING_SOURCE,
)

ROOT = Path(__file__).resolve().parents[2]


def _parse(*arguments: str):
    return build_parser().parse_args(arguments)


def test_empty_selector_intersection_shows_canonical_values_and_suggestions() -> None:
    arguments = _parse(
        "inspect",
        "--table",
        "scalar_gravity",
        "--generation-engine",
        "recurrence",
    )
    with pytest.raises(ManualCampaignError) as captured:
        selection_from_arguments(arguments)

    message = str(captured.value)
    assert "contains no catalog entries" in message
    assert "Try removing one of --table, --generation-engine" in message
    assert "canonical values / useful aliases" in message
    assert "scalar_gravity" in message
    assert "amplicol, recurrence, compiled, eager" in message
    assert "wildcard `*`/`all`" in message


@pytest.mark.parametrize(
    "arguments",
    (
        ("run", "--generation-time-limit", "nan"),
        ("run", "--worker-wall-limit", "inf"),
        ("run", "--resource-sample-interval", "-inf"),
        ("run", "--termination-grace", "nan"),
        ("run", "--target-measurement-duration", "inf"),
        ("refresh-pdf", "--pdf-timeout", "nan"),
        ("dashboard-snapshot", "--stale-after", "inf"),
    ),
)
def test_float_options_reject_non_finite_values(arguments: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as captured:
        _parse(*arguments)
    assert captured.value.code == 2


def test_generation_recipe_spells_out_measurement_relevant_hyperparameters() -> None:
    candidate = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.measurement.model is ModelKey.BUILTIN_SM
        and cell.measurement.backend == "jit"
        and REPORT_CATALOG.static_na_reason(cell) is None
    )
    recipe = reproduction_recipe(
        candidate,
        repo_root=ROOT,
        cores=2,
        target_runtime=0.75,
        batch_size=32,
        warmups=3,
        minimum_samples=7,
    )
    assert recipe.generate is not None
    generate = recipe.generate
    expected_pairs = {
        "--validation-samples": "10",
        "--validation-seed": "12345",
        "--relative-tolerance": "1e-12",
        "--absolute-tolerance": "1e-300",
        "--output-chunk-size": "512",
        "--horner-iterations": "10",
        "--cpe-iterations": "none",
        "--max-horner-variables": "1000",
        "--max-common-pair-cache-entries": "5000000",
        "--max-common-pair-distance": "1000",
        "--cpp-optimization": "O3",
        "--cores": "2",
        "--batch-size": "32",
    }
    for option, expected in expected_pairs.items():
        assert generate[generate.index(option) + 1] == expected
    assert "--model-cache" in generate
    assert "--model-cache-dir" in generate
    assert "--numerical-current-reuse" in generate
    assert parse_cli(generate[1:]).resolve().effective.action == "generate"

    assert recipe.profile is not None
    profile = recipe.profile
    for option, expected in (
        ("--target-runtime", "0.75"),
        ("--batch-size", "32"),
        ("--warmup-runs", "3"),
        ("--minimum-samples", "7"),
        ("--precision", "16"),
    ):
        assert profile[profile.index(option) + 1] == expected
    assert parse_cli(profile[1:]).resolve().effective.action == "benchmark"


def test_dry_run_recipe_blocks_cover_more_than_twenty_direct_cells() -> None:
    cells = tuple(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is not ExecutionMode.AMPLICOL
        and REPORT_CATALOG.static_na_reason(cell) is None
    )[:21]
    blocks = tuple(
        manual_campaign._dry_run_recipe_blocks(
            cells,
            repo_root=ROOT,
            arguments=_parse("run", "--dry-run"),
            width=120,
        )
    )
    assert len(blocks) == 21
    assert all(f"| recipe | {index}" in block for index, block in enumerate(blocks, 1))


def test_evaluator_total_has_independent_statistics_and_semantic_colors() -> None:
    candidate = REPORT_CATALOG.cell(
        "matrix-recurrence-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    baseline = manual_campaign._manual_baseline(candidate)
    assert baseline is not None

    def current(
        cell_id: str,
        wall: float,
        evaluator_total: float,
        *,
        recurrence_core: float | None = None,
    ) -> LightweightCurrent:
        sample_count = 5
        repetitions = 2
        batch_size = 1
        measured_points = sample_count * repetitions * batch_size
        result = {
            "status": ResultStatus.OK.value,
            "wall_seconds_per_point": wall,
            "execution_seconds_per_point": recurrence_core,
            "sample_count": sample_count,
            "provenance": {
                "evaluator_total_timing": {
                    "abi": "pyamplicol-report-evaluator-total-timing-v1",
                    "status": "measured",
                    "ratio_eligible": False,
                    "raw_seconds_per_point": evaluator_total,
                    "source": "runtime.evaluate.accumulated",
                    "execution_mode": "recurrence",
                    "sample_contract": (
                        "accumulated-repeated-warmed-evaluator-total-v1"
                    ),
                    "sample_count": sample_count,
                    "repetitions_per_sample": repetitions,
                    "batch_size": batch_size,
                    "points_per_sample": repetitions * batch_size,
                    "measured_point_count": measured_points,
                    "accumulated_seconds": evaluator_total * measured_points,
                }
            },
        }
        if recurrence_core is not None:
            result["provenance"]["execution_timing"] = {
                "abi": MEASURED_EXECUTION_TIMING_ABI,
                "status": "measured",
                "ratio_eligible": True,
                "raw_seconds_per_point": recurrence_core,
                "source": RECURRENCE_EXECUTION_TIMING_SOURCE,
                "compiled_direct_arena_active": False,
                "sample_count": sample_count,
                "native_profile_points_per_sample": repetitions * batch_size,
                "sample_contract": PAIRED_TIMING_SAMPLE_CONTRACT,
            }
        return LightweightCurrent(
            cell_id=cell_id,
            attempt_id=f"attempt-{cell_id}",
            result_path=ROOT / "unused-result.json",
            result=result,
            complete=True,
            reusable=True,
            reason="reusable",
        )

    summary, exclusions = manual_campaign._inspect_metric(
        (candidate,),
        {
            candidate.cell_id: current(candidate.cell_id, 2.0, 4.0),
            baseline.cell_id: current(baseline.cell_id, 1.0, 2.0),
        },
        field=None,
        generation=False,
        evaluator_total=True,
    )
    assert exclusions == {}
    assert summary["count"] == 1
    assert summary["median"] == pytest.approx(2.0)
    assert summary["weighted_mean"] == pytest.approx(2.0)

    palette = Palette(enabled=True)
    assert Fore.GREEN in manual_campaign._paint_multiplier(palette, 0.9)
    assert Fore.RED in manual_campaign._paint_multiplier(palette, 1.1)
    assert "\x1b[" not in manual_campaign._paint_multiplier(palette, 1.0)

    malformed = current(candidate.cell_id, 2.0, 4.0)
    malformed_total = malformed.result["provenance"]["evaluator_total_timing"]
    assert isinstance(malformed_total, dict)
    malformed_total["accumulated_seconds"] = 99.0
    assert manual_campaign._evaluator_total_number(malformed.result) is None

    other = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.cell_id != candidate.cell_id
    )
    core_summary, core_exclusions = manual_campaign._inspect_recurrence_core(
        (candidate, other),
        {
            candidate.cell_id: current(
                candidate.cell_id,
                2.0,
                4.0,
                recurrence_core=4.0e-6,
            ),
            other.cell_id: current(
                other.cell_id,
                3.0,
                6.0,
                recurrence_core=2.0e-6,
            ),
        },
    )
    assert core_exclusions == {}
    assert core_summary["count"] == 2
    assert core_summary["median_seconds_per_point"] == pytest.approx(3.0e-6)
    assert core_summary["mean_seconds_per_point"] == pytest.approx(3.0e-6)
    fastest = core_summary["fastest"]
    slowest = core_summary["slowest"]
    assert isinstance(fastest, dict)
    assert isinstance(slowest, dict)
    assert fastest["cell_id"] == other.cell_id
    assert fastest["seconds_per_point"] == pytest.approx(2.0e-6)
    assert slowest["cell_id"] == candidate.cell_id
    assert slowest["seconds_per_point"] == pytest.approx(4.0e-6)
    assert "multiplier" not in fastest
