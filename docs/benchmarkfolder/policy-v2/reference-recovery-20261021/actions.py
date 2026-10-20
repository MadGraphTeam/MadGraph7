#!/usr/bin/env python3
"""Execute only root-authorized orphan cleanup and one frozen fresh campaign."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUN_ID = "corrected-original-core-v2-fresh-20261021"
CASES = "qcd-4,ew-4,qcd-1,ew-1,qcd-2,ew-2,qcd-3,ew-3"
FROZEN = ("ninjars_benchmark.py", "ninjars_benchmark_driver.f",
          "ninjars_benchmark_workers.py", "ninjars_benchmark_worker.f",
          "ninjars_benchmark_references_v2.py", "ninjars_benchmark_policy_v2.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ticks(pid):
    raw = Path(f"/proc/{pid}/stat").read_text()
    return raw[raw.rindex(")") + 2:].split()


def dump_new(name, value):
    path = HERE / name
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def terminate():
    old = json.loads((HERE / "pretermination-audit.json").read_text())["process"]
    pid = old["pid"]
    if (HERE / "termination.json").exists():
        raise RuntimeError("Termination already recorded")
    fields = ticks(pid)
    if (int(fields[19]) != old["start_ticks"] or fields[0] != "T" or
            int(fields[1]) != 1 or int(fields[2]) != pid or
            Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode() != old["command"] or
            os.readlink(f"/proc/{pid}/cwd") != old["cwd"] or
            sha(f"/proc/{pid}/exe") != old["executable_sha256"]):
        raise RuntimeError("Orphan identity/state changed; refuse to signal")
    members = []
    for proc in Path("/proc").iterdir():
        if proc.name.isdecimal():
            try:
                if int(ticks(int(proc.name))[2]) == pid:
                    members.append(int(proc.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    if members != [pid]:
        raise RuntimeError(f"Orphan group has unexpected members: {members}")
    before = datetime.datetime.now(datetime.timezone.utc).isoformat()
    os.kill(pid, signal.SIGKILL)
    for _ in range(50):
        try:
            state = ticks(pid)[0]
        except FileNotFoundError:
            state = "absent"
        if state in ("Z", "absent"):
            break
        time.sleep(.02)
    if state not in ("Z", "absent"):
        raise RuntimeError(f"Termination did not finish: {state}")
    dump_new("termination.json", {
        "schema": "exact-orphan-termination-v1", "root_authorized": True,
        "authorization": "Root authorized preserving the interrupted attempt and terminating this exact PID/start-ticks orphan before one fresh full-eight-case campaign.",
        "pid": pid, "start_ticks": old["start_ticks"], "signal": "SIGKILL",
        "utc": before, "result_state": state, "group_members_before": members,
        "pretermination_audit_sha256": sha(HERE / "pretermination-audit.json"),
        "old_report_status_edited": False,
    })


def launch():
    if not (HERE / "termination.json").is_file():
        raise RuntimeError("Exact orphan termination must be recorded first")
    if (HERE / "launch.json").exists():
        raise RuntimeError("Campaign launch already recorded")
    if 207 not in os.sched_getaffinity(0):
        raise RuntimeError("CPU 207 unavailable")
    for path in [ROOT / "docs/benchmarkfolder/policy-v2/reference-runs" / RUN_ID,
                 ROOT / ".benchmark-work/policy-v2/reference-runs" / RUN_ID]:
        if path.exists():
            raise RuntimeError(f"Fresh campaign namespace already exists: {path}")
    args = [sys.executable, "-B", "tools/ninjars_benchmark_references_v2.py",
            "--correction", "docs/benchmarkfolder/policy-v2/model-correction.json",
            "--config", "docs/benchmarkfolder/policy-v2/cases.json",
            "--cases", CASES, "--run-id", RUN_ID, "--cpu", "207",
            "--backends", "1,6,7", "--timeout", "86400"]
    env = os.environ.copy()
    env.update(TMPDIR=str(ROOT / ".ninjars-build/tmp"),
               SYMBOLICA_LOCK_DIR=str(ROOT / ".ninjars-build/symbolica-locks"))
    log = ROOT / ".benchmark-work/logs" / (RUN_ID + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as stream:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    start_ticks = int(ticks(child.pid)[19])
    dump_new("launch.json", {
        "schema": "fresh-frozen-reference-campaign-launch-v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "root_authorized": True, "argv": args, "cwd": str(ROOT),
        "pid": child.pid, "start_ticks": start_ticks, "log": str(log.relative_to(ROOT)),
        "frozen_sources_sha256": {name: sha(ROOT / "tools" / name) for name in FROZEN},
        "configuration_sha256": sha(ROOT / "docs/benchmarkfolder/policy-v2/cases.json"),
        "correction_sha256": sha(ROOT / "docs/benchmarkfolder/policy-v2/model-correction.json"),
        "case_order": CASES.split(","), "cpu": 207, "timeout_seconds": 86400,
        "backends": [1, 6, 7], "points_per_case": 100,
        "old_report_status_edited": False, "acceptance_eligible": False,
    })
    print(json.dumps({"pid": child.pid, "start_ticks": start_ticks, "log": str(log)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("terminate", "launch"))
    args = parser.parse_args()
    {"terminate": terminate, "launch": launch}[args.action]()
