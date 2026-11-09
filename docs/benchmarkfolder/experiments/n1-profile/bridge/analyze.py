#!/usr/bin/env python3
"""Summarize retained, non-acceptance bridge attribution without native calls."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def distribution(values):
    return {"median": median(values), "minimum": min(values), "maximum": max(values),
            "observations": values}


def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    points_per_batch = data["loops"] * 100
    runs = data["runs"]
    assert len(runs) == len(data["levels"]) * data["repetitions"]
    assert all(run["parity_maximum_relative_error"] == 0 for run in runs)
    result = {
        "performance_eligible": False,
        "input": str(path),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "points_per_batch": points_per_batch,
        "saved_final_traversal_outputs_bit_identical": True,
        "unit": "microseconds_per_amplitude",
        "levels": {},
        "limits": [
            "Diagnostic attribution, not a paired performance acceptance test.",
            "Three repetitions per level; report range and all observations, not a confidence interval.",
            "Some runs are substantially slower; timer medians are descriptive under shared-host variation.",
            "Inclusive timers overlap; do not sum FFI, state, reduction and upstream totals.",
            "Timer and bookkeeping costs remain included. Cross-level subtraction is not a causal speedup estimate.",
            "The workload has exactly two DP N3 rank2 calls per amplitude, with independent reversal checking.",
        ],
    }
    for level in data["levels"]:
        selected = sorted((run for run in runs if run["level"] == level),
                          key=lambda run: run["repetition"])
        row = {"whole_amplitude_wall": distribution(
            [run["observation"]["seconds_per_point"] * 1e6 for run in selected])}
        if level != "pristine":
            row["whole_amplitude_cpu"] = distribution([
                run["profile"]["clock"]["batch_cpu_seconds"] / points_per_batch * 1e6
                for run in selected])
        if level not in ("pristine", "0:0", "1:0"):
            for run in selected:
                bridge = run["profile"]["bridge"]
                assert len(bridge["rows"]) == 2
                assert {(r["n"], r["rank"], r["check"], r["cache"])
                        for r in bridge["rows"]} == {(3, 2, 0, 1), (3, 2, 1, 1)}
                for item in bridge["rows"]:
                    assert item["calls"] == points_per_batch
                    assert item["direct_lanes"] == points_per_batch
                    assert all(item[key] == 0 for key in (
                        "failed_calls", "method_choice_lanes", "precision_tier_lanes", "refused_lanes"))
            row["all_calls_direct_dp"] = True
            row["calls_per_amplitude"] = 2
            row["bridge_inclusive"] = {
                key.removesuffix("_ns"): distribution([
                    sum(item[key] for item in run["profile"]["bridge"]["rows"])
                    / points_per_batch / 1000 for run in selected])
                for key in selected[0]["profile"]["bridge"]["rows"][0]
                if key.endswith("_ns")
            }
            row["bridge_auxiliary"] = {
                key.removesuffix("_ns"): distribution([
                    run["profile"]["bridge"]["auxiliary"][key]
                    / points_per_batch / 1000 for run in selected])
                for key in selected[0]["profile"]["bridge"]["auxiliary"]
                if key.endswith("_ns")
            }
        result["levels"][level] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for level, row in result["levels"].items():
        print(level, row["whole_amplitude_wall"])


if __name__ == "__main__":
    main()
