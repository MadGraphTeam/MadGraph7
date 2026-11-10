#!/usr/bin/env python3
"""Check observed production rescue dispatch, preserving default DP failures."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error("Output exists; choose a new evidence path")
report = json.loads(args.input.read_text())
assert report["state"] == "complete"
assert len(report["rows"]) == 8
default, rescued = [], []
for row in report["rows"]:
    assert row["status"] == 0 and not any(row["backend_statuses"])
    assert row["threshold"] in (1e-3, 1e-6)
    assert row["rotations"] in (0, 2)
    modes = [1, 2] if row["rotations"] == 0 else [1, 2, 1, 1]
    if row["threshold"] == 1e-6:
        modes += [4, 5] if row["rotations"] == 0 else [4, 5, 4, 4]
        assert row["return_code"] == 323
        assert row["relative_error_to_same_input_qp"] == 0
        assert row["relative_error_to_fixture_truth"] < 1e-8
        rescued.append(row)
    else:
        assert row["return_code"] == 219
        assert row["relative_error_to_same_input_qp"] > 3e-6
        default.append(row)
    assert row["modes"] == modes
    assert row["calls"] == len(modes)
assert len(default) == len(rescued) == 4
result = {
    "schema_version": 1,
    "classification": "Default policy accepts four inaccurate DP projections; tighter shared policy rescues four projections in QP",
    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    "adjudicator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "cases": 8, "default_dp_failures_retained": 4, "tighter_policy_qp_rescues": 4,
    "default_return_code": 219, "rescued_return_code": 323,
    "max_rescued_relative_error_to_same_input_qp": max(row["relative_error_to_same_input_qp"] for row in rescued),
    "max_rescued_relative_error_to_fixture_truth": max(row["relative_error_to_fixture_truth"] for row in rescued),
    "checks": ["Every numerical callback returned success",
        "All actual CT mode sequences match production DP/reversal/rotation then QP rescue",
        "H/T/U report stable DP for default threshold and successful QP rescue for tighter threshold",
        "All tighter-policy results agree exactly with independently computed same-input QP",
        "Original default-policy inaccuracies remain visible"],
    "scope": "Verbatim production rescue-control block with actual Fortran tensor callback and native accuracy/return codes",
    "limitation": "Synthetic interference projections; complete physical SLOOPMATRIX/filter initialization is not tested"}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(result["classification"])
