#!/usr/bin/env python3
"""Read-only software-evidence check; never run numerical engines or timings."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = json.loads((HERE / "manifest.json").read_text())
    identity = manifest.pop("identity_sha256")
    require(hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == identity, "Changed report-evidence manifest identity")
    require(manifest["schema"] == "madloop-final-report-pipeline-evidence-v1", "Wrong evidence schema")
    for name, record in manifest["files"].items():
        path = HERE / name
        require(path.resolve().is_relative_to(HERE), "Evidence path leaves package")
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "Changed report-evidence file: " + name)
    record = json.loads((HERE / "validation.json").read_text())
    require(record["exit_code"] == 0, "Report unit tests failed")
    for name, expected in record["source_sha256"].items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and digest(path) == expected,
                "Changed tested report source: " + name)
    log = (HERE / "unittest.stderr").read_text()
    footer = re.findall(r"^Ran (\d+) tests in ([0-9.]+)s$", log, re.M)
    require(len(footer) == 1 and int(footer[0][0]) == record["tests"], "Invalid test-count footer")
    require(float(footer[0][1]) == record["unittest_seconds"], "Changed test duration")
    require(re.search(r"^OK$", log, re.M) is not None and "FAILED" not in log,
            "Incomplete or failed report-test log")
    require(record["historical_top_report_overwritten"] is False
            and record["final_physics_dataset_available"] is False,
            "Software evidence cannot certify the final physics campaign")
    return {"passed": True, "tests": record["tests"], "identity_sha256": identity,
            "physics_or_performance_gate_certified": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
