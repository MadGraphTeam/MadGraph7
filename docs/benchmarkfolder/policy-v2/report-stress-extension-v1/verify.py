#!/usr/bin/env python3
"""Verify bounded report software evidence without compiling or evaluating."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OWNED = ("tools/ninjars_report_v2.py", "tests/unit_tests/loop/test_ninjars_report_v2.py",
         "docs/benchmarkfolder/policy-v2/template/benchmark.typ", "docs/benchmarkfolder/style.typ")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = json.loads((HERE / "manifest.json").read_text())
    identity = manifest.pop("identity_sha256")
    computed = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(identity == computed, "Changed software-evidence manifest")
    require(manifest["schema"] == "madloop-report-stress-extension-evidence-v1", "Wrong software schema")
    for name, record in manifest["files"].items():
        path = HERE / name
        require(path.resolve().is_relative_to(HERE), "Evidence path leaves package")
        require(path.stat().st_size == record["bytes"] and digest(path) == record["sha256"],
                "Changed software-evidence file: " + name)
    attempt = json.loads((HERE / "attempt-1/result.json").read_text())
    require(attempt["exit_code"] == 0, "Report tests did not pass")
    require(attempt["source_before_sha256"] == attempt["source_after_sha256"],
            "A report dependency changed while the tests were running")
    for name in OWNED:
        require(digest(ROOT / name) == attempt["source_after_sha256"][name],
                "Changed tested report source: " + name)
    # Independently owned transitive readers may subsequently gain their own
    # separately reviewed checkpoint. Their exact tested hashes remain retained
    # above, without silently claiming this attempt tested the newer reader.
    log = (HERE / "attempt-1/stderr.log").read_text()
    footer = re.findall(r"^Ran (\d+) tests in ([0-9.]+)s$", log, re.M)
    require(len(footer) == 1 and int(footer[0][0]) == 40, "Wrong or missing test footer")
    require(re.search(r"^OK$", log, re.M) is not None and "FAILED" not in log, "Incomplete test log")
    require(attempt["historical_evidence_changed"] is False
            and attempt["top_pdf_overwritten"] is False, "Historical evidence was overwritten")
    return {"passed": True, "tests": 40, "identity_sha256": identity,
            "scope": "software and synthetic PDF layout only",
            "physics_or_performance_gate_certified": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
