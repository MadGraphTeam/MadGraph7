#!/usr/bin/env python3
"""Use the exact already-tested kernel objects for focused ownership stress."""
import argparse
import json
import os
from pathlib import Path
import shutil

import check_static as fixture
import stress_driver

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    original = HERE / args.source / "exact.json"
    source = json.loads(original.read_text())
    assert source["complete"] and "failure" not in source
    report = HERE / args.tag
    report.mkdir(exist_ok=False)
    records = []
    for item in source["results"]:
        work = fixture.check.ROOT / ".benchmark-work/experiments/symmetric-static-owner" / args.source / item["key"]
        for filename, digest in item["sources"].items():
            assert fixture.check.sha(work / filename) == digest
        name = "TEST_" + ("TERMINAL" if item["terminal"] else "UPDATE") + f"_WL_{item['rank']}_1"
        (work / "stress.f90").write_text(stress_driver.driver(name, item["groups"], item["terminal"]))
        command = ["gfortran", *fixture.FLAGS[item["flags"]], "constants.o",
                   "full.o", "checked-static.o", "stress.f90", "-o", "stress"]
        fixture.run(command, work, "stress-build.log")
        output = fixture.run([str(work / "stress")], work, "stress.log")
        record = dict(key=item["key"], passed=True, output=output, command=command,
            driver_sha256=fixture.check.sha(work / "stress.f90"),
            executable_sha256=fixture.check.sha(work / "stress"),
            object_sha256={p.name: fixture.check.sha(p)
                           for p in work.glob("*.o")})
        records.append(record)
        destination = report / item["key"]
        destination.mkdir()
        for filename in ("stress.f90", "stress-build.log", "stress.log"):
            shutil.copy2(work / filename, destination / filename)
        (report / "stress.json").write_text(json.dumps(dict(complete=False,
            results=records), indent=2) + "\n")
        print(item["key"], output.strip(), flush=True)
    for path in (Path(__file__), HERE / "stress_driver.py"):
        shutil.copy2(path, report / path.name)
    result = dict(complete=True, acceptance_eligible=False, results=records,
        comparisons=500 * len(records), source_report=str(original),
        source_report_sha256=fixture.check.sha(original),
        scope="500 successive all-fast calls per exact compiled safety kernel. "
              "Each call checks every output/input bit and cumulative flags, "
              "with independently changing A sparsity, group B zeros and signed "
              "zeros, all LCUT_SIZE 0..4, full and diagonal terminal paths. "
              "No fallback may hide an ownership or stale-cache defect.")
    (report / "stress.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
