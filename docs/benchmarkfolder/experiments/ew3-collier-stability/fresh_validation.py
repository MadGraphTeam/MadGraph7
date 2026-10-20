#!/usr/bin/env python3
"""Check all four reducers on the fresh corrected EW3 CLI export.

One warmed traversal is numerical evidence only; it cannot pass a runtime gate.
The same stricter Collier accuracy is supplied to every backend.
"""
import argparse
import itertools
import json
import shutil
import sys

from probe import HERE, ROOT, WORK, compare, sha

sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="1,6,7,8")
    args = parser.parse_args()
    bench.WORK = WORK / "fresh-export"
    bench.REPORT = HERE / "fresh-export"
    config = json.loads((ROOT / "docs/benchmarkfolder/cases.json").read_text())
    config["madloop_parameters"]["COLLIERRequiredAccuracy"] = 1e-12
    case = next(c for c in config["cases"] if c["id"] == "ew-3")
    points = ROOT / "docs/benchmarkfolder/points/ew-3.dat"
    destination = bench.REPORT / "validation"
    destination.mkdir(exist_ok=True)
    (destination / "configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    directory = bench.subprocess_dir("candidate", case)
    for backend in map(int, args.backends.split(",")):
        report = destination / str(backend)
        report.mkdir()
        result = bench.measure("candidate", case, backend, points, 1,
                               "corrected-model-diagnostic", 5, config=config,
                               persist=False)
        result["acceptance_eligible"] = False
        result["scope"] = "Single-traversal correctness diagnostic after fresh CLI generation"
        (report / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            shutil.copy2(directory / name, report / name)
        shutil.copy2(bench.output_dir("candidate", case) / "Cards/MadLoopParams.dat",
                     report / "MadLoopParams.dat")
        shutil.copy2(bench.WORK / "logs" /
                     f"run-candidate-ew-3-{backend}-corrected-model-diagnostic.log",
                     report / "run.log")
        print(backend, result["status_counts"], flush=True)
    records = {}
    for backend in (1, 6, 7, 8):
        path = destination / str(backend) / "result.json"
        if path.exists():
            records[backend] = json.loads(path.read_text())
    # The pairwise scale is at least as strict as the global four-backend scale.
    comparisons = {f"{a}-{b}": compare(records[a]["rows"], records[b]["rows"])
                   for a, b in itertools.combinations(records, 2)}
    result = {
        "acceptance_eligible": False,
        "complete": set(records) == {1, 6, 7, 8},
        "all_four_numerical_comparisons_passed":
            set(records) == {1, 6, 7, 8} and all(r["passed"] for r in comparisons.values()),
        "comparisons": comparisons,
        "points_sha256": sha(points),
        "files_sha256": {str(p.relative_to(destination)): sha(p)
                         for p in sorted(destination.rglob("*"))
                         if p.is_file() and p.name != "summary.json"},
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
