#!/usr/bin/env python3
"""Compare complete finite coefficient columns without certifying failed rows."""
from pathlib import Path
import itertools
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "protocol-sources"))
from tirust_benchmark import compare_rows, digest


def main():
    records = []
    paths = list((HERE / "runs").glob("*/*/result.json"))
    paths.extend((HERE / "diagnostics").glob("*/*/result.json"))
    for path in sorted(paths):
        result = json.loads(path.read_text())
        record = {"case": result["case"]["id"], "label": path.parent.name,
                  "status": result["status"], "precision_request": result["precision_request"],
                  "record_path": str(path.relative_to(HERE)),
                  "record_sha256": digest(path), "points_sha256": result["points_sha256"],
                  "rows": result["rows"], "parameter_card_sha256": None,
                  "expected_orders": None, "stress_summary": []}
        inputs = path.parent / (result["source"] + "-inputs.json")
        if inputs.exists():
            data = json.loads(inputs.read_text())
            record["parameter_card_sha256"] = data["param_card_sha256"]
            if data["points_sha256"] != record["points_sha256"]:
                raise ValueError("Run input point binding differs")
            card = HERE / "inputs" / result["source"] / record["case"] / "param_card.dat"
            if digest(card) != record["parameter_card_sha256"]:
                raise ValueError("Physical card binding differs")
        dimensions = path.parent / "raw/dimensions.dat"
        if dimensions.exists():
            values = list(map(int, dimensions.read_text().split()))
            if len(values) != 3 or values[0] != max(values[1:]):
                raise ValueError("Invalid answer dimensions")
            record["expected_orders"] = list(range(values[0] + 1))
        for category, rows in record["rows"].items():
            record["stress_summary"].extend(dict(row, category=category) for row in rows
                                             if row["point"] == 2 and row["order"] == 0)
        records.append(record)
    comparisons = []
    for left, right in itertools.combinations(records, 2):
        if left["case"] != right["case"] or left["expected_orders"] is None:
            continue
        if left["expected_orders"] != right["expected_orders"]:
            continue
        if any(left[name] != right[name] for name in ("points_sha256", "parameter_card_sha256")):
            raise ValueError("Cannot compare different physical inputs")
        for point, role in ((1, "ordinary-control"), (2, "stress")):
            sides, checked = [], []
            for record in (left, right):
                rows = sorted([dict(row, category=category) for category, values in record["rows"].items()
                               for row in values if row["point"] == point and "coefficients" in row],
                              key=lambda row: row["order"])
                if [row["order"] for row in rows] != record["expected_orders"]:
                    break
                sides.append(rows)
                checked.append(record["status"] == "complete" and
                               record["precision_request"] == "auto" and
                               all(row["category"] == "accepted" for row in rows))
            if len(sides) != 2:
                continue
            result = compare_rows(*sides, tolerance=3e-6)
            comparisons.append({"case": left["case"], "role": role,
                "reference": left["label"], "candidate": right["label"],
                "both_complete_checked_runs": all(checked),
                "diagnostic_only": not all(checked),
                "finite_max_relative_error": max(row["relative_error"] for row in result["residuals"] if row["column"] == 1),
                "pole_max_relative_error": max(row["relative_error"] for row in result["residuals"] if row["column"] >= 2),
                **result})
    primary = [record for record in records if record["precision_request"] == "auto"]
    result = {"scope": "Finite complete-order comparisons include rejected host results and unchecked forced-QP values only as explicitly marked diagnostics. They never certify numerical stability. Nonfinite or incomplete points have no comparison.",
              "primary_requests_expected": 24, "primary_requests_recorded": len(primary),
              "all_primary_runs_stable": len(primary) == 24 and all(record["status"] == "complete" for record in primary),
              "records": [{name: value for name, value in record.items() if name != "rows"} for record in records],
              "comparisons": comparisons, "script_sha256": digest(Path(__file__))}
    (HERE / "adjudication.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
