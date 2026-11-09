#!/usr/bin/env python3
"""Run the preserved four-call standalone probe in its isolated baseline copy.

This records fixed-QP coefficients without certifying their numerical stability.
The primary generated exports and original stress observations are untouched.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[5]
REPORT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--point", type=int, default=1)
    parser.add_argument("--backend", type=int, choices=(1, 6, 7, 8), default=8)
    parser.add_argument("--onshell-threshold", type=float, default=1e-8)
    parser.add_argument("--timeout", type=float, default=30.)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("Run from the repository root with an alphanumeric label")
    if not math.isfinite(args.onshell_threshold) or not 0 < args.onshell_threshold < 1:
        parser.error("Invalid on-shell threshold")
    work = ROOT / ".tirust-build/complex-stress-direct-qp"
    process = work / "SubProcesses/P0_epem_mupmum"
    destination = REPORT / args.label
    destination.mkdir(exist_ok=False)
    point_file = REPORT.parent / "points/cms_ew_n2_epem_mupmum.json"
    points = json.loads(point_file.read_text())
    if not 1 <= args.point <= len(points["points"]):
        parser.error("Point index is outside the frozen stream")
    if digest(point_file.with_suffix(".dat")) != points["sha256"]:
        raise ValueError("Frozen parent stream changed")
    if digest(process / "check_sa.f") != digest(REPORT / "check_sa.f"):
        raise ValueError("Direct driver source differs from the preserved source")
    momentum = "".join(" ".join("%.17e" % x for x in vector) + "\n"
                       for vector in points["points"][args.point-1]["momenta"])
    (destination / "PS.input").write_text(momentum)
    text = (REPORT / "MadLoopParams.dat").read_text()
    for name, value in (("MLReductionLib", str(args.backend)),
                        ("OSThres", str(args.onshell_threshold))):
        text, count = re.subn(r"(?mi)(^#" + name + r"\s*\n)[^\n]*",
                             lambda match: match[1] + value, text)
        if count != 1:
            raise ValueError("Unexpected numerical card layout")
    (destination / "MadLoopParams.dat").write_text(text)
    record = {"source": "baseline", "backend": args.backend, "point_index": args.point,
        "parent_points_sha256": points["sha256"], "onshell_threshold": args.onshell_threshold,
        "stability_certified": False, "performance_eligible": False,
        "precision_request": "QP", "calls": 4,
        "driver_sha256": digest(process / "check_sa.f"),
        "executable_sha256": digest(process / "check"),
        "parameter_card_sha256": digest(work / "Cards/param_card.dat"),
        "runner_sha256": digest(Path(__file__))}
    shutil.copy2(Path(__file__), destination / "run_case.py")
    with (ROOT / ".tirust-build/numerical-tests.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (process / "PS.input").write_text(momentum)
        (process / "MadLoopParams.dat").unlink(missing_ok=True)
        (work / "SubProcesses/MadLoop5_resources/MadLoopParams.dat").write_text(text)
        (process / "result.dat").unlink(missing_ok=True)
        started = time.monotonic()
        with (destination / "output.log").open("w") as stream:
            try:
                run = subprocess.run([str(process / "check")], cwd=process, stdout=stream,
                    stderr=subprocess.STDOUT, env=dict(os.environ, GFORTRAN_UNBUFFERED_ALL="y"),
                    timeout=args.timeout)
                record.update(returncode=run.returncode, timed_out=False)
            except subprocess.TimeoutExpired:
                record.update(timed_out=True, timeout_seconds=args.timeout)
        record["elapsed_seconds_diagnostic_only"] = time.monotonic() - started
        record["result_file_exists"] = (process / "result.dat").exists()
        if record["result_file_exists"]:
            shutil.copy2(process / "result.dat", destination / "result.dat")
    (destination / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
