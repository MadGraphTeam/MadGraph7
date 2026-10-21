#!/usr/bin/env python3
"""Serialize one retained final-source libtest observation; never hide failures."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "bytes": path.stat().st_size}


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


label, executable, *arguments = sys.argv[1:]
executable = Path(executable).resolve()
assert executable.is_relative_to(ROOT)
build = json.loads((REPORT / "build-record.json").read_text())
assert binding(executable) == build["executable"]
bridge = ROOT / build["source_directory"]
sources = build["bridge_source_sha256"]
assert all(sha(bridge / name) == expected for name, expected in sources.items())
assert set(os.sched_getaffinity(0)) == {190}
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS"):
    assert os.environ.get(variable) == "1"
command = [str(executable), *arguments, "--test-threads=1", "--nocapture"]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
clock = time.monotonic()
outpath, errpath = REPORT / (label + ".stdout"), REPORT / (label + ".stderr")
with outpath.open("x") as out, errpath.open("x") as err:
    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                               stdout=out, stderr=err, start_new_session=True)
    startticks = int(Path(f"/proc/{process.pid}/stat").read_text().split(")", 1)[1].split()[19])
    dump(REPORT / (label + "-active.json"), {"pid": process.pid,
        "process_group_id": process.pid, "start_ticks": startticks,
        "command": command, "started_utc": started, "cpu": 190})
    try:
        exitcode = process.wait(timeout=3600)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
remaining = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        fields = (entry / "stat").read_text().split(")", 1)[1].split()
        if int(fields[2]) == process.pid:
            remaining.append({"pid": int(entry.name), "state": fields[0], "start_ticks": int(fields[19])})
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
after = {name: sha(bridge / name) for name in sources}
record = {"schema": "final-release-numerical-test-run-v1", "label": label,
    "revision": build["revision"], "dirty_checkout": build["dirty_checkout"],
    "bridge_source_sha256": sources, "executable": build["executable"],
    "command": command, "cwd": str(ROOT), "cpu": 190, "started_utc": started,
    "wall_seconds": time.monotonic()-clock, "exit_code": exitcode,
    "stdout": binding(outpath), "stderr": binding(errpath),
    "process_id": process.pid, "process_start_ticks": startticks,
    "remaining_process_group_members": remaining, "all_children_exited": not remaining,
    "source_sha256_before_equals_after": after == sources,
    "source_sha256_after": after, "performance_claim": False}
dump(REPORT / (label + "-process.json"), record)
print(json.dumps({key: record[key] for key in ("label", "exit_code", "wall_seconds",
    "all_children_exited", "source_sha256_before_equals_after")}), flush=True)
assert not remaining and after == sources
sys.exit(exitcode)
