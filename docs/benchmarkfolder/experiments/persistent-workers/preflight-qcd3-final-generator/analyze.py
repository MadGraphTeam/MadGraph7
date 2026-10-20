#!/usr/bin/env python3
"""Summarize this exploratory paired series; never promote it to acceptance."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from ninjars_benchmark import runtime_statistics

config = json.loads((ROOT / "docs/benchmarkfolder/cases.json").read_text())
groups = {}
hashes = {}
for source, backend in (("baseline", 6), ("candidate", 6), ("candidate", 8)):
    label = f"{source}-{backend}"
    groups[label] = []
    for repetition in range(7):
        path = HERE / "data" / f"qcd-3-{label}-{repetition}.json"
        hashes[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[label].append(json.loads(path.read_text()))

identities = {(d["points_sha256"], d["parameter_card_sha256"],
               json.dumps(d["shared_numerical_settings"], sort_keys=True))
              for records in groups.values() for d in records}
assert len(identities) == 1, "Numerical inputs/settings differ"
for records in groups.values():
    assert all(d["rows"] == records[0]["rows"] for d in records), "Repeated rows differ"
    assert len({(d["executable_sha256"], d["cpu"]) for d in records}) == 1

# Retained strict-setting original CutTools/Collier calibrations provide an
# additional numerical comparison only. Their timings are never paired here.
reference = groups["baseline-6"][0]
for backend in (1, 7):
    path = HERE / "additional-data" / f"baseline-qcd-3-{backend}-calibration.json"
    d = json.loads(path.read_text())
    hashes[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ("points_sha256", "parameter_card_sha256"):
        assert d[key] == reference[key]
    def common_settings(record):
        return {key: value for key, value in record["shared_numerical_settings"].items()
                if key != "selected_backend_qp_integrand"}
    assert common_settings(d) == common_settings(reference)
    groups[f"baseline-{backend}-calibration"] = [d]

scales = {}
for records in groups.values():
    for row in records[0]["rows"]:
        scales[row["point"]] = max(scales.get(row["point"], 0),
                                   *(abs(x) for x in row["coefficients"]))
comparisons = {}
for left, right in itertools.combinations(groups, 2):
    failures, worst = [], 0.
    arows, brows = groups[left][0]["rows"], groups[right][0]["rows"]
    assert len(arows) == len(brows)
    for arow, brow in zip(arows, brows):
        assert (arow["point"], arow["order"]) == (brow["point"], brow["order"])
        for coefficient, (a, b) in enumerate(zip(arow["coefficients"], brow["coefficients"])):
            tolerance = (config["relative_tolerance"] * max(abs(a), abs(b))
                         + config["scale_tolerance"] * scales[arow["point"]])
            fraction = abs(a-b) / tolerance if tolerance else (0. if a == b else float("inf"))
            worst = max(worst, fraction)
            if fraction > 1:
                failures.append({"point": arow["point"], "order": arow["order"],
                                 "coefficient": coefficient, "values": [a, b]})
    comparisons[f"{left}/{right}"] = {
        "passed": not failures, "failures": failures,
        "worst_tolerance_fraction": worst,
        "coefficient_rows_exact": all(a["coefficients"] == b["coefficients"]
                                      for a, b in zip(arows, brows))}

times = {label: [d["seconds_per_point"] for d in rows] for label, rows in groups.items()}
summary = {
    "acceptance_eligible": False,
    "scope": "Exploratory three-target series during independent builds and profiles",
    "input_record_sha256": hashes,
    "agreement": comparisons,
    "status_counts": {label: records[0]["status_counts"] for label, records in groups.items()},
    "rescued_points": {label: sorted({r["point"] for r in records[0]["rows"]
                                    if r["status"] // 100 == 3})
                       for label, records in groups.items()},
    "ninjars_vs_original_ninja": runtime_statistics(
        times["baseline-6"], times["candidate-8"], config["max_runtime_ratio"]),
    "optimized_ninja_vs_original_ninja": runtime_statistics(
        times["baseline-6"], times["candidate-6"], config["max_runtime_ratio"]),
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items()
                  if key != "input_record_sha256"}, indent=2))
