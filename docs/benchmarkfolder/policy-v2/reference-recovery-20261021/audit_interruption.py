#!/usr/bin/env python3
"""Read-only post-pause reference audit. Never signals or adopts a process."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers

OLD = ROOT / "docs/benchmarkfolder/policy-v2/reference-runs/corrected-original-core-v2-24h-20261020"
PID = 2962065
START_TICKS = 55987751


def process_snapshot(pid: int) -> dict:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return {"pid": pid, "exists": False}
    raw = (proc / "stat").read_text()
    fields = raw[raw.rindex(")") + 2:].split()
    current = {
        "pid": pid, "exists": True, "state": fields[0],
        "ppid": int(fields[1]), "process_group": int(fields[2]),
        "session": int(fields[3]), "start_ticks": int(fields[19]),
        "user_ticks": int(fields[11]), "system_ticks": int(fields[12]),
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
        "command": (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
        "cwd": os.readlink(proc / "cwd"),
        "fds": {fd.name: os.readlink(fd) for fd in (proc / "fd").iterdir()},
        "executable_sha256": workers.digest(proc / "exe"),
        "wchan": (proc / "wchan").read_text(),
    }
    if current["start_ticks"] != START_TICKS:
        raise RuntimeError("PID has been reused; do not act on it")
    peers = []
    links = set(current["fds"].values())
    for peer in Path("/proc").iterdir():
        if not peer.name.isdecimal() or int(peer.name) == pid:
            continue
        try:
            for fd in (peer / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target in links:
                    peers.append({"pid": int(peer.name), "fd": fd.name, "target": target})
        except OSError:
            pass
    current["remaining_fd_peers"] = peers
    return current


def validate_completed_prefix() -> dict:
    """Check actual complete cases without changing incomplete run status."""
    status = policy.read(OLD / "run-status.json")
    protocol = policy.read(OLD / "protocol.json")
    descriptor = workers.protocol_descriptor()
    if (protocol["protocol_descriptor"] != descriptor or
            protocol["protocol_identity_sha256"] != bench.content_digest(descriptor)):
        raise RuntimeError("Frozen protocol changed")
    for name, checksum in descriptor["sources"].items():
        if workers.digest(OLD / "sources" / name) != checksum:
            raise RuntimeError("Retained protocol source changed")
    config = protocol["configuration"]
    completed = status["completed_cases"]
    if completed != ["qcd-1", "ew-1", "qcd-2", "ew-2", "qcd-3", "ew-3"]:
        raise RuntimeError("Unexpected completed prefix")
    checks = {}
    for case in completed:
        firsts = {}
        for backend in (1, 6, 7):
            key = f"{case}-baseline-{backend}"
            build = policy.read(OLD / "builds" / key / "build.json")
            compiled = policy.read(OLD / "builds" / key / "compiled-sources.json")
            if (compiled["files_sha256"] != bench.content_digest(compiled["files"]) or
                    compiled["files_sha256"] != build["generated_sources_sha256"] or
                    workers.digest(OLD / "builds" / key / "worker.f") != build["driver_sha256"]):
                raise RuntimeError("Completed build evidence changed")
            calibration = policy.read(OLD / "data" / f"{key}-calibration.json")
            for repetition in (0, 1):
                value = policy.read(OLD / "data" / f"{key}-{repetition}.json")
                if (value["case"], value["source"], value["backend"], value["repetition"]) != (
                        case, "baseline", backend, repetition):
                    raise RuntimeError("Wrong completed record identity")
                if value["executable_sha256"] != build["executable_sha256"]:
                    raise RuntimeError("Completed executable binding differs")
                if value["protocol_identity_sha256"] != protocol["protocol_identity_sha256"]:
                    raise RuntimeError("Completed protocol binding differs")
                if value["cpu"] != 207 or value["acceptance_eligible"]:
                    raise RuntimeError("Wrong helper CPU/eligibility")
                workers.bench.validate_repetition(value, calibration, config)
                if len(value["rows"]) != 100 * (value["answer_dimensions"]["loop_orders"] + 1):
                    raise RuntimeError("Incomplete numerical row coverage")
                for row in value["rows"]:
                    bench.validate_status(row["status"], backend)
                    if not all(math.isfinite(x) for x in row["coefficients"] + [row["accuracy"]]):
                        raise RuntimeError("Nonfinite completed record")
                if repetition == 0:
                    firsts[str(backend)] = value["rows"]
        comparison = policy.compare_rows(firsts)
        if not comparison["passed"]:
            raise RuntimeError("Completed-prefix backend disagreement")
        checks[case] = comparison
    return {
        "old_run_status_unchanged": status, "validated_complete_cases": completed,
        "calibration_records": 18, "timed_records": 36,
        "all_three_legacy_backends_agree": True, "comparisons": checks,
        "complete_eight_case_reference_claim": False,
    }


def main() -> None:
    result = {
        "schema": "interrupted-reference-read-only-audit-v1",
        "signals_sent": [], "numerical_jobs_started": [],
        "process": process_snapshot(PID),
        "prefix": validate_completed_prefix(),
        "old_report_files_sha256": {
            str(p.relative_to(ROOT)): workers.digest(p)
            for p in sorted(OLD.rglob("*")) if p.is_file()
        },
    }
    result["identity_sha256"] = bench.content_digest(result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
