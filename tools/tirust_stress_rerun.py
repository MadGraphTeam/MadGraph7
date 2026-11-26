#!/usr/bin/env python3
"""Rerun recorded stress lanes against another TIRust bridge library.

For a bridge-only change (vendor/tirust/src/lib.rs), the compiled stress clones of a
campaign can be reused: each lane's clone is copied, its driver runs with the new
libmadloop_tirust.so first on LD_LIBRARY_PATH (checked with ldd), on the same warm-up
and stress points, and every point is compared with the recorded lane and with the
independent forced-QP references (CutTools, Ninja and NinjaRS), using the
comparison of tools/tirust_stress.py.

  tirust_stress_rerun.py --campaign C --library DIR --out DIR --case CASE [lane ...]

Lanes default to candidate-9-qp, candidate-9-auto and chain-9671-auto. Results go to
OUT/summary.json; copies and logs stay under OUT. The clones keep their original
MadLoop source, so a requalification on the final pin remains the acceptance step.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tirust_benchmark as bench  # noqa: E402
from tirust_benchmark_campaign import activate_campaign  # noqa: E402
import tirust_stress as stress  # noqa: E402

REFERENCES = ("candidate-8-qp", "candidate-1-qp", "candidate-6-qp")


def rerun(case, lane, library, out, report, warm):
    old = json.loads((report / lane / "result.json").read_text())
    clone = bench.WORK / "stress" / case / lane / "outputs" / old["source"] / case
    target = out / lane / case
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(clone, target, symlinks=True)
    (directory,) = [d for d in (target / "SubProcesses").glob("P*_*") if (d / "check").is_file()]
    for name in ("stress-values.dat", "stress-routes.dat", "stress-timing.dat", "warm.dat"):
        (directory / name).unlink(missing_ok=True)
    env = bench.environment()
    env["LD_LIBRARY_PATH"] = f"{library}:{env['LD_LIBRARY_PATH']}"
    ldd = subprocess.run(["ldd", "./check"], cwd=directory, env=env, text=True, capture_output=True).stdout
    resolved = [line.split("=>")[1].split("(")[0].strip() for line in ldd.splitlines()
                if "libmadloop_tirust" in line]
    if resolved != [str(library / "libmadloop_tirust.so")]:
        raise RuntimeError(f"{lane}: bridge resolves to {resolved}")
    points = report / lane / "points-run1.dat"
    if not points.is_file():
        # Lanes recorded before the resumable driver read the whole stress stream.
        points = warm.parent / f"{case}-stress.dat"
    if bench.digest(points) != old["points_sha256"]:
        raise RuntimeError(f"{lane}: stress points differ from the recorded lane")
    log = out / lane / "driver.log"
    with log.open("w") as stream:
        returncode = subprocess.run(["./check", str(warm), str(points),
                                     "checked" if old["mode"] == "auto" else "unchecked"],
                                    cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=3600).returncode
    ndim = int((directory / "dimensions.dat").read_text().split()[0])
    rows = stress.parse_values(directory / "stress-values.dat", ndim, partial=True) \
        if (directory / "stress-values.dat").is_file() else []
    ids = [p["id"] for p in old.get("stress_points", [])]
    entry = {"returncode": returncode, "done": "STRESS-DONE" in log.read_text(), "points": {}}
    for point in sorted({r["point"] for r in rows}):
        new = [r for r in rows if r["point"] == point]
        before = [r for r in old["rows"] if r["point"] == point]
        references = {}
        for name in REFERENCES:
            path = report / name / "result.json"
            if not path.is_file():
                continue
            reference = [r for r in json.loads(path.read_text())["rows"] if r["point"] == point]
            if reference and stress.comparable(reference) and stress.comparable(new):
                result = stress.compare(reference, new, 1e-8)
                references[name] = {"passed": result["passed"],
                                    "maximum_relative_error": result["maximum_relative_error"]}
        entry["points"][ids[point - 1] if point <= len(ids) else str(point)] = {
            "status": new[0]["status"], "recorded_status": before[0]["status"] if before else None,
            "qp_failure_latch": any(r["fpe_qp"] for r in new),
            "finite": new[0]["coefficients"][1] if new[0]["coefficients"] else None,
            "unchanged": [(r["status"], r["coefficients"], r["fpe_qp"]) for r in new]
                         == [(r["status"], r["coefficients"], r["fpe_qp"]) for r in before],
            "accepted": stress.accepted_backend(new, old["chain"]) if old["mode"] == "auto" else None,
            "vs_qp_references": references}
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--library", required=True, type=Path, help="directory holding libmadloop_tirust.so")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("lanes", nargs="*", default=["candidate-9-qp", "candidate-9-auto", "chain-9671-auto"])
    args = parser.parse_args(argv)
    activate_campaign(bench, args.campaign)
    report = bench.REPORT / "experiments/stress" / args.case
    warm = bench.REPORT / ("stress-points" if args.case in stress.EXTRA_CASES else "points") / f"{args.case}.dat"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {"campaign": args.campaign, "case": args.case, "library": str(args.library.resolve()),
               "library_sha256": bench.digest(args.library.resolve() / "libmadloop_tirust.so"),
               "lanes": {lane: rerun(args.case, lane, args.library.resolve(), out, report, warm)
                         for lane in args.lanes}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
