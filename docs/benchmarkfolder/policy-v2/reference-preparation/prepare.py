#!/usr/bin/env python3
"""Prepare exact corrected-original core outputs; never start numerical workers."""
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references

REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/policy-v2/prepared-references"
PAUSE = WORK / "pause-before-next-case"


def main():
    correction_path = REPORT.parent / "model-correction.json"
    config_path = REPORT.parent / "cases.json"
    correction = policy.validate_correction(correction_path, live=True)
    config = policy.read(config_path)
    policy.validate_config(config)
    WORK.mkdir(parents=True, exist_ok=True)
    status_path = REPORT / "status.json"
    identity = {"preparation_source": policy.bound_file(__file__),
                "adapter": policy.bound_file(ROOT / "tools/ninjars_benchmark_references_v2.py"),
                "policy_implementation": policy.bound_file(ROOT / "tools/ninjars_benchmark_policy_v2.py"),
                "configuration": policy.bound_file(config_path),
                "correction": policy.bound_file(correction_path)}
    if status_path.exists():
        status = policy.read(status_path)
        if status["identity"] != identity:
            raise RuntimeError("Preparation implementation/configuration changed; use a new namespace")
    else:
        status = {"status": "preparing", "identity": identity, "cases": {},
                  "numerical_evaluations_started": False,
                  "scope": "Clean MODEL and subprocess builds; original DHELAS/reducers copied unchanged"}
    policy.bench.dump(status_path, status)
    for case in config["cases"]:
        case_id = case["id"]
        target = WORK / "outputs/baseline" / case_id
        if case_id in status["cases"]:
            recorded = status["cases"][case_id]
            policy.verify_rebuild(target / "correction-build.json", target, correction["identity_sha256"])
            if policy.manifest(target) != recorded["compiled_manifest"]:
                raise RuntimeError("Prepared source changed")
            continue
        if PAUSE.exists():
            status["status"] = "paused-before-next-case"
            policy.bench.dump(status_path, status)
            print("Paused before", case_id, flush=True)
            return
        status.update(status="preparing", active_case=case_id)
        policy.bench.dump(status_path, status)
        print("Preparing", case_id, flush=True)
        try:
            references.prepare_corrected(case, correction, WORK, REPORT)
            status["cases"][case_id] = {
                "output": policy.relative(target),
                "compiled_manifest": policy.manifest(target),
                "rebuild": policy.bound_file(target / "correction-build.json"),
                "parameter_card_sha256": policy.workers.digest(target / "Cards/param_card.dat")}
        except BaseException as error:
            status.update(status="failed", error_type=type(error).__name__, error=str(error))
            policy.bench.dump(status_path, status)
            raise
        policy.bench.dump(status_path, status)
        print("Prepared", case_id, flush=True)
    status.pop("active_case", None)
    status["status"] = "complete"
    policy.bench.dump(status_path, status)
    print(json.dumps({"status": "complete", "cases": list(status["cases"]),
                      "numerical_evaluations_started": False}), flush=True)


if __name__ == "__main__":
    main()
