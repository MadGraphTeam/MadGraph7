#!/usr/bin/env python3
"""Read-only retained-evidence check; never builds or launches numerical work."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers

HERE = Path(__file__).resolve().parent
V2 = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checks():
    correction = policy.validate_correction(V2 / "model-correction.json")
    counts = {"completed_v2_runs": 0, "failed_v2_runs": 0, "v2_targets": 0,
              "v2_repetition_records": 0, "fresh_parities": 0,
              "historical_prototype_targets": 0, "preparation_cases": 0}
    completed, failed = [], []
    # Future collections may add sibling run directories. Verify only this
    # immutable package's run inventory, without treating later runs as part of
    # the historical snapshot.
    retained = (policy.read(HERE / "manifest.json")["files"]
                if (HERE / "manifest.json").exists() else None)
    for namespace in ("representative-runs", "reference-runs"):
        if retained is None:
            reports = sorted((V2 / namespace).iterdir())
        else:
            prefix = policy.relative(V2 / namespace) + "/"
            reports = sorted({(ROOT / name).parent for name in retained
                              if name.startswith(prefix) and name.endswith("/run-status.json")})
        for report in reports:
            if not report.is_dir():
                continue
            status = policy.read(report / "run-status.json")
            if status["status"] == "failed":
                assert status["error_type"]
                failed.append(policy.relative(report))
                counts["failed_v2_runs"] += 1
                continue
            assert status["status"] == "complete"
            binding_path = V2 / "runtime-bindings-20261020" / (report.name + ".json")
            bindings = policy.checked_bindings(binding_path, correction)
            protocol, records = workers.checked_series(report, minimum=2)
            counts["completed_v2_runs"] += 1
            counts["v2_targets"] += len(bindings["targets"])
            counts["v2_repetition_records"] += len(records)
            for key, binding in bindings["targets"].items():
                first = policy.read(report / "data" / (key + "-0.json"))
                fresh = policy.read(report / "data" / (key + "-fresh.json"))
                assert first["executable_sha256"] == binding["executable_sha256"]
                assert fresh["executable_sha256"] == binding["fresh_executable_sha256"]
                parity = workers.numerical_parity(first, fresh, protocol["configuration"])
                recorded = policy.read(report / "data" / (key + "-parity.json"))
                assert all(recorded[name] == value for name, value in parity.items())
                counts["fresh_parities"] += 1
            completed.append(policy.relative(report))
    assert counts["completed_v2_runs"] == 17 and counts["failed_v2_runs"] == 5

    # These old experiments lack the final protocol's source snapshots and must
    # not be passed off as current checked_series/proof. Check their raw parity
    # and full seven-block inventory without adding or repairing any metadata.
    parent = V2.parent / "experiments/persistent-workers"
    for name in ("ninja-parity-20261020", "ninja-parity-audit-20261020"):
        report = parent / name
        protocol = policy.read(report / "protocol.json")
        assert protocol["acceptance_eligible"] is False
        if name.endswith("audit-20261020"):
            assert policy.read(report / "run-status.json")["status"] == "complete"
        else:
            assert not (report / "run-status.json").exists()
        for case in ("qcd-1", "qcd-2"):
            for source in ("baseline", "candidate"):
                key = f"{case}-{source}-6"
                fresh = policy.read(report / "data" / (key + "-fresh.json"))
                for repetition in range(7):
                    row = policy.read(report / "data" / f"{key}-{repetition}.json")
                    parity = workers.numerical_parity(row, fresh, protocol["configuration"])
                    assert parity == {"passed": True, "exact_rows": True,
                                      "status_counts_match": True}
                counts["historical_prototype_targets"] += 1

    status = policy.read(V2 / "reference-preparation/status.json")
    summary = policy.verified(policy.read(V2 / "reference-preparation/summary.json"),
                              "corrected-original-build-preparation-v2")
    assert status["status"] == "complete" and not status["numerical_evaluations_started"]
    for field in ("status", "script", "log"):
        policy.check_file(summary[field])
    for case, item in status["cases"].items():
        retained = V2 / "reference-preparation/retained-rebuilds" / (case + ".json")
        assert sha(retained) == item["rebuild"]["sha256"]
        build = policy.verified(policy.read(retained), "forced-model-rebuild-v2")
        assert build["correction_identity_sha256"] == correction["identity_sha256"]
        assert not build["prebuild_model_objects"]
        assert not build["prebuild_model_archive_exists"]
        assert not build["prelink_executable_exists"] and build["model_objects"]
        assert build["sources_sha256"] == item["compiled_manifest"]["files_sha256"]
        for log in build["logs"]:
            policy.check_file(log)
        counts["preparation_cases"] += 1
    assert counts["preparation_cases"] == 8
    legacy = policy.verified(policy.read(V2 / "legacy-coverage-status.json"),
                             "legacy-representative-coverage-status-v2")
    assert legacy["targets"] == 26 and len(legacy["bindings"]) == 12
    assert not legacy["full_representative_proof"] and not legacy["acceptance_evidence"]
    for record in legacy["bindings"]:
        policy.check_file(record)
    return {"counts": counts, "completed_runs": completed, "failed_runs": failed,
            "physics_evaluations_started_by_verifier": False}


def main():
    manifest = policy.verified(policy.read(HERE / "manifest.json"),
                               "legacy-evidence-retention-v2")
    for name, record in manifest["files"].items():
        target = ROOT / name
        assert target.is_file() and target.stat().st_size == record["bytes"], name
        assert sha(target) == record["sha256"], name
    result = checks()
    assert result == policy.read(HERE / "validation.json")["results"]
    print(json.dumps({"retained_files": len(manifest["files"]), **result}, indent=2))


if __name__ == "__main__":
    main()
