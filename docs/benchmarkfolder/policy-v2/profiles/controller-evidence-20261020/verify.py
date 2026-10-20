#!/usr/bin/env python3
"""Verify retained controller smoke evidence without executing a worker/perf."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers

HERE = Path(__file__).resolve().parent
RUNS = (
    "profile-controller-original-qcd1-20261020",
    "profile-controller-original-qcd1-retry1-20261020",
    "profile-controller-original-qcd1-retry2-20261020",
)


def checks():
    results = {}
    for name in RUNS:
        report = HERE.parent / name
        data = policy.read(report / "profile.json")
        snapshot = report / "ninjars_profile_v2.py"
        assert workers.digest(snapshot) == data["script_sha256"]
        assert data["acceptance_eligible"] is False
        binding = policy.read(policy.check_file(data["binding"]))
        measurement = policy.check_file(data["measurement"])
        reference = policy.read(measurement)
        target = binding["targets"][data["target"]]
        assert data["target"] == f"{reference['case']}-{reference['source']}-{reference['backend']}"
        assert binding["run_files_sha256"][policy.relative(measurement)] == workers.digest(measurement)
        assert target["executable_sha256"] == reference["executable_sha256"] == data["executable_sha256"]
        assert workers.digest(policy.check_file(data["points"])) == reference["points_sha256"]
        if not data["complete"]:
            assert name == RUNS[0] and data["error"] and not data["blocks"]
            results[name] = {"complete": False, "error": data["error"]}
            continue
        assert data["every_physical_row_exact"] and data["original_worker_unchanged"]
        assert len(data["blocks"]) == 3
        for index, recorded in enumerate(data["blocks"]):
            count = sum(recorded["timed_status_counts"].values())
            assert count == 30000
            parsed = workers.parse_outputs(report / f"block-{index}", 6, 100, 300)
            assert parsed == recorded
            assert parsed["rows"] == reference["rows"]
        raw = HERE / "raw-perf" / (name + ".perf")
        assert workers.digest(raw) == data["partition"]["raw_perf_sha256"]
        stacks = policy.check_file(data["partition"]["stacks"])
        spec = importlib.util.spec_from_file_location(name, snapshot)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        counts = Counter()
        for block in stacks.read_text().split("\n\n"):
            lines = [line for line in block.splitlines()
                     if line.strip() and not line.startswith("#")]
            if lines:
                counts[source.category(lines)] += 1
        assert sum(counts.values()) == data["partition"]["samples"]
        assert dict(counts) == {key: value["samples"]
                                for key, value in data["partition"]["categories"].items()}
        results[name] = {"complete": True, "blocks": 3,
                         "evaluations": 90000, "points": 100,
                         "samples": sum(counts.values()), "all_rows_exact": True}
    return results


def main():
    manifest = policy.verified(policy.read(HERE / "manifest.json"),
                               "profile-controller-evidence-v2")
    for name, record in manifest["files"].items():
        file = ROOT / name
        assert file.stat().st_size == record["bytes"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"]
    result = checks()
    assert result == policy.read(HERE / "validation.json")["results"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
