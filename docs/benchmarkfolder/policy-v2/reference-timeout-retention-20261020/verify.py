#!/usr/bin/env python3
"""Verify the failed reference attempt without promoting its completed prefix."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
V2 = HERE.parent
REPORT = V2 / "reference-runs/corrected-original-core-v2-20261020"
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def checks():
    status = policy.read(REPORT / "run-status.json")
    protocol = policy.read(REPORT / "protocol.json")
    config = protocol["configuration"]
    cases = [case["id"] for case in config["cases"]]
    if (status["status"] != "failed" or status["requested_cases"] != cases or
            status["completed_cases"] != cases[:6] or status["acceptance_eligible"] or
            status["source_role"] != "original_helper_corrected"):
        raise RuntimeError("Failed full-eight reference identity changed")
    if (protocol["repetitions"] != 2 or protocol["backends"] != [1, 6, 7] or
            protocol["sources"] != "baseline" or config["points_per_case"] != 100):
        raise RuntimeError("Reference collection settings changed")
    descriptor = workers.protocol_descriptor()
    identity = bench.content_digest(descriptor)
    if (protocol["protocol_descriptor"] != descriptor or
            protocol["protocol_identity_sha256"] != identity):
        raise RuntimeError("Frozen numerical protocol changed")
    for name, checksum in descriptor["sources"].items():
        if workers.digest(REPORT / "sources" / name) != checksum:
            raise RuntimeError("Retained numerical protocol changed")
    failure = policy.read(REPORT / "rejected/qcd-4-failure.json")
    if (failure["case"], failure["backend"], failure["stage"],
            failure["error_type"], failure["repetition"],
            failure["command_sequence"]) != (
                "qcd-4", 1, "initialize", "ProtocolError", None, None):
        raise RuntimeError("Initial QCD4 timeout identity changed")
    if "Worker protocol timeout" not in failure["error"]:
        raise RuntimeError("Unexpected failure")
    if any((REPORT / "data").glob("qcd-4-*.json")):
        raise RuntimeError("QCD4 unexpectedly contains accepted measurements")
    if any((REPORT / "data").glob("ew-4-*.json")):
        raise RuntimeError("EW4 unexpectedly contains measurements")
    # The frozen validator must continue to reject this failed whole series.
    try:
        workers.checked_series(REPORT)
    except RuntimeError as error:
        if "incomplete or failed" not in str(error):
            raise
    else:
        raise RuntimeError("Failed run unexpectedly qualifies as a complete series")
    summary = {}
    expected_files = set()
    for case in cases[:6]:
        data = {}
        for backend in (1, 6, 7):
            prefix = f"{case}-baseline-{backend}"
            build = policy.read(REPORT / "builds" / prefix / "build.json")
            calibration = policy.read(REPORT / "data" / (prefix + "-calibration.json"))
            for repetition in ("calibration", 0, 1):
                name = f"{prefix}-{repetition}.json"
                expected_files.add(name)
                record = policy.read(REPORT / "data" / name)
                if (record["case"], record["source"], record["backend"],
                        record["repetition"]) != (case, "baseline", backend, repetition):
                    raise RuntimeError("Record identity changed")
                if (record["protocol_identity_sha256"] != identity or
                        record["executable_sha256"] != build["executable_sha256"] or
                        record["generated_sources_sha256"] != build["generated_sources_sha256"]):
                    raise RuntimeError("Record source/executable binding changed")
                dimensions = record["answer_dimensions"]
                nso, born = dimensions["loop_orders"], dimensions["born_orders"]
                if dimensions["allocated"] != max(nso, born) or born > nso:
                    raise RuntimeError("Missing physical split-order columns")
                rows = record["rows"]
                if [(row["point"], row["order"]) for row in rows] != [
                        (point, order) for point in range(1, 101)
                        for order in range(nso + 1)]:
                    raise RuntimeError("Incomplete or reordered point stream")
                for row in rows:
                    bench.validate_status(row["status"], backend)
                    if (len(row["coefficients"]) != 4 or row["accuracy"] < 0 or
                            not all(math.isfinite(x) for x in
                                    row["coefficients"] + [row["accuracy"]])):
                        raise RuntimeError("Invalid retained numerical row")
                counts = dict(Counter(str(row["status"]) for row in rows
                                      if row["order"] == 0))
                if counts != record["status_counts"]:
                    raise RuntimeError("Status count does not match rows")
                for code, count in record["timed_status_counts"].items():
                    bench.validate_status(int(code), backend)
                    if count <= 0:
                        raise RuntimeError("Invalid timed status count")
                evaluations = record["loops"] * 100
                if (evaluations <= 0 or
                        sum(record["timed_status_counts"].values()) != evaluations or
                        not math.isfinite(record["seconds"]) or record["seconds"] <= 0 or
                        record["seconds_per_point"] != record["seconds"] / evaluations):
                    raise RuntimeError("Invalid timing bookkeeping")
                bench.validate_repetition(record, calibration, config)
                data[backend, repetition] = record
        comparisons = {}
        for repetition in (0, 1):
            comparisons[str(repetition)] = policy.compare_rows({
                str(backend): data[backend, repetition]["rows"]
                for backend in (1, 6, 7)})
        summary[case] = {
            "status_counts": {str(backend): data[backend, 0]["status_counts"]
                              for backend in (1, 6, 7)},
            "repetitions_numerically_identical": all(
                data[backend, 0]["rows"] == data[backend, 1]["rows"]
                for backend in (1, 6, 7)),
            "comparisons": comparisons,
        }
    if {p.name for p in (REPORT / "data").glob("*.json")} != expected_files:
        raise RuntimeError("Unexpected or missing calibration/repetition file")
    stale = policy.read(HERE / "stale-output-provenance.json")
    for name, text in failure["raw_outputs"].items():
        checksum = hashlib.sha256(text.encode()).hexdigest()
        if checksum != stale["outputs"][name]["snapshot_sha256"]:
            raise RuntimeError("Changed rejected raw snapshot")
        copy = HERE / "stale-parent-outputs" / name
        if stale["outputs"][name]["matches_parent_copy"]:
            if copy.read_bytes() != text.encode():
                raise RuntimeError("Retained parent output does not match stale snapshot")
    return {
        "status": "failed", "accepted_complete_series": False,
        "completed_case_count": 6, "completed_repetition_records": 36,
        "completed_calibration_records": 18,
        "requested_cases": cases, "failure_case": "qcd-4",
        "failure_stage": "initialize-before-READY",
        "qcd4_accepted_measurements": 0,
        "rejected_raw_outputs_are_measurements": False,
        "physics_evaluations_started": False, "cases": summary,
    }


def main():
    manifest = policy.verified(policy.read(HERE / "manifest.json"),
                               "failed-reference-retention-v2")
    for name, record in manifest["files"].items():
        path = policy.path(name)
        if path.stat().st_size != record["bytes"] or workers.digest(path) != record["sha256"]:
            raise RuntimeError("Changed retained file: " + name)
    result = checks()
    if result != policy.read(HERE / "validation.json"):
        raise RuntimeError("Verification summary differs from retained rows")
    print(json.dumps({"files": len(manifest["files"]), **result}, indent=2))


if __name__ == "__main__":
    main()
