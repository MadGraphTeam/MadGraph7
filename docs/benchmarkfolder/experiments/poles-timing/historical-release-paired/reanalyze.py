#!/usr/bin/env python3
"""Repeat strict structural and numerical analysis without native execution."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "analysis-protocol"))
from tirust_benchmark_pole_timing import parse_results

rows = []
for fixture in json.loads((HERE / "specification.json").read_text())["fixtures"]:
    label = fixture["name"] + ("-dp" if fixture["ctmode"] == 1 else "-qp")
    raw = HERE / label / "pole-timing-values.dat"
    # JSON stores tuple-valued refusal frame keys as arrays. Compare the same
    # serialized representation used when the original evidence was written.
    result = json.loads(json.dumps(parse_results(raw.read_text(), fixture)))
    original = json.loads((HERE / label / "result.json").read_text())
    if result != original:
        raise RuntimeError(f"Hardened analysis changed the original result: {label}")
    frames = [f for f in result["frames"] if f["phase"] == 2]
    row = dict(name=fixture["name"], n=fixture["n"], rank=fixture["rank"],
        precision="DP" if fixture["ctmode"] == 1 else "QP", scope=fixture["scope"],
        status=result["status"], timing_qualified=result["timing_qualified"],
        maximum_scaled_coefficient_residual=result["maximum_scaled_coefficient_residual"],
        finite_placeholder=result["finite_placeholder"], original_result_unchanged=True,
        raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    if frames:
        row.update(wall_poles_over_full=result["wall_poles_over_full"],
            cpu_poles_over_full=result["cpu_poles_over_full"],
            median_full_wall_microseconds=1e6*statistics.median(f["wall_seconds"]/f["loops"] for f in frames if f["finite"]),
            median_poles_wall_microseconds=1e6*statistics.median(f["wall_seconds"]/f["loops"] for f in frames if not f["finite"]),
            measured_wall_range=[min(f["wall_seconds"] for f in frames), max(f["wall_seconds"] for f in frames)],
            maximum_wall_over_cpu=max(f["wall_seconds"]/f["cpu_seconds"] for f in frames),
            pairs=result["completed_pairs"])
    rows.append(row)
summary = {"scope": "Historical warm tensor diagnostic; no physical-amplitude performance acceptance",
           "performance_eligible": False, "native_executions_repeated": False,
           "all_original_results_unchanged": True, "settings": rows,
           "measured_settings": sum(r["status"] == "measured" for r in rows),
           "refused_settings": sum(r["status"] == "refused" for r in rows)}
(HERE / "analysis-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))
