#!/usr/bin/env python3
"""Read-only verification of analysis-code and retained real-data evidence."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = json.loads((HERE / "manifest.json").read_text())
    identity = manifest.pop("identity_sha256")
    computed = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(identity == computed, "Changed fix-evidence manifest identity")
    for name, record in manifest["files"].items():
        path = HERE / name
        require(path.resolve().is_relative_to(HERE) and path.stat().st_size == record["bytes"]
                and sha(path) == record["sha256"], "Changed fix evidence: " + name)
    change = json.loads((HERE / "change-record.json").read_text())
    for stage in ("before", "after"):
        for name, digest in change["source_" + stage + "_sha256"].items():
            if name.startswith("tools/"):
                require(sha(HERE / ("source-" + stage) / name) == digest,
                        "Changed retained reader snapshot: " + stage + "/" + name)
    for name, digest in {**change["source_after_sha256"],
                         **change["frozen_sources_unchanged_sha256"]}.items():
        require(sha(ROOT / name) == digest, "Changed current tested/frozen source: " + name)
    software = json.loads((HERE / "software-attempt-1/result.json").read_text())
    require(software["exit_code"] == 0 and software["source_before_sha256"] == software["source_after_sha256"],
            "Software tests failed or source changed while running")
    log = (HERE / "software-attempt-1/stderr.log").read_text()
    footer = re.findall(r"^Ran (\d+) tests in ([0-9.]+)s$", log, re.M)
    require(len(footer) == 1 and int(footer[0][0]) == 81 and re.search(r"^OK$", log, re.M)
            and "FAILED" not in log, "Wrong or failed targeted software suite")
    actual = json.loads((HERE / "actual-data-analysis.json").read_text())
    for name, digest in actual["inputs_sha256"].items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and sha(path) == digest,
                "Changed retained actual data or prior rejected analysis: " + name)
    require(actual["analysis_script_sha256"] == sha(HERE / "analyze_actual.py"), "Changed actual-data analysis script")
    for name, digest in actual["analysis_reader_sources_sha256"].items():
        require(sha(ROOT / name) == digest, "Changed actual analysis code: " + name)
    total = 0
    for case in actual["cases"].values():
        agreement = case["agreement"]
        require(agreement["passed"] and agreement["pairs"] == 21 and not agreement["failures"],
                "Actual component comparison failed")
        total += agreement["comparisons"]
    require(total == 35616 and actual["common_settings_equal_after_required_qp_validation"] is True
            and actual["all_21_pair_physical_comparisons_unchanged"] is True,
            "Actual-data correctness coverage changed")
    require(actual["core_dataset_identity_sha256"] is None
            and actual["final_report_validation_passed"] is False
            and actual["timing_acceptance_evaluated"] is False and actual["physics_evaluations"] == 0,
            "Read-only analysis cannot certify final report/performance or invent a core identity")
    return {"passed": True, "tests": 81, "actual_component_comparisons": total,
            "physics_evaluations": 0, "final_report_validation_passed": False,
            "identity_sha256": identity}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
