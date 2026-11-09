"""Report qualification uses point counts and repetition counts separately."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "tirust_report", ROOT / "docs/benchmarkfolder/build_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def passing_summary():
    return {
        "baseline_revision": "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f",
        "candidate_revision": "candidate",
        "gate": {"passed": True, "complete": True,
                 "consistent_candidate_source": True},
        "cases": [{
            "id": str(i), "role": "main", "points": 100,
            "correctness": [{"backend": b, "passed": True, "samples": 7}
                            for b in (1, 6, 7, 8, 9)],
            "performance": {"passed": True, "complete": True,
                            "pairs": 7, "upper_95": 1.04},
            "existing_backend_regressions": [
                {"backend": b, "complete": True, "passed": True}
                for b in (1, 6, 7, 8)],
        } for i in range(8)],
    }


def test_hundred_points_with_seven_comparison_repetitions():
    data = passing_summary()
    assert REPORT.validate_summary(data) is data


@pytest.mark.parametrize("change", [
    lambda row: row.update(points=99),
    lambda row: row["correctness"][0].update(samples=6),
    lambda row: row["correctness"].pop(),
    lambda row: row["performance"].update(upper_95=1.051),
    lambda row: row["existing_backend_regressions"].pop(),
])
def test_unqualified_evidence_cannot_produce_passing_headline(change):
    data = passing_summary()
    change(data["cases"][0])
    with pytest.raises(ValueError):
        REPORT.validate_summary(data)
