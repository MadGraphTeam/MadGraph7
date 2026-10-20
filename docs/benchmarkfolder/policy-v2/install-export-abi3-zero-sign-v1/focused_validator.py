#!/usr/bin/env python3
"""Bind focused Python checks and sealed Rust tests to the refreshed install."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import prepare
import runtime_validator as runtime

ROOT, REPORT, SOURCE = prepare.ROOT, prepare.REPORT, prepare.SOURCE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((REPORT / "runtime-results.json").read_text())
    assert result["pole_checks_passed"] == result["pole_checks_tested"] == 20
    assert result["nlo_cli_success"] is True
    assert result["standalone"]["qp"]["return_code"] == 120
    proof = ROOT / "docs/benchmarkfolder/experiments/ninjars-mp-transport-audit-v1"
    subprocess.run([sys.executable, "-B", str(proof / "verify.py")], cwd=ROOT,
                   check=True)
    prepared = json.loads((proof / "bridge-prepared.json").read_text())
    tested = json.loads((proof / "bridge-result.json").read_text())
    assert tested["complete"] and all(row["returncode"] == 0 for row in tested["runs"])
    assert tested["replay_summary"] == {
        "changed_fields": 0, "changed_records": 0, "complete": True,
        "failures": 0, "records": 1392,
    }
    assert "18 passed; 0 failed; 1 ignored;" in (proof / "unit.stdout").read_text()
    hidden = SOURCE / "HEPTools/ninjars-unavailable"
    assert hidden.is_dir() and not (SOURCE / "HEPTools/ninjars").exists()
    identities = {}
    for name, expected in prepared["proposed_files"].items():
        for directory in (ROOT / "vendor/ninjars", SOURCE / "vendor/ninjars", hidden):
            assert sha(directory / name) == expected, str(directory / name)
        identities[name] = expected
    command = [sys.executable, "-B", "-m", "unittest",
               "tests.unit_tests.loop.test_ninjars_fortran",
               "tests.unit_tests.various.test_ninjars_install",
               "tests.unit_tests.various.test_ninjars_statistics"]
    log = REPORT / "focused-unit-tests.log"
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=runtime.environment(),
                                   stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text()
    assert completed.returncode == 0 and "Ran 16 tests" in text and re.search(r"(?m)^OK$", text)
    (REPORT / "focused-unit-results.json").write_text(json.dumps({
        "passed": True, "python_tests": 16, "command": command,
        "log_sha256": sha(log), "corrected_transport_suite": {
            "evidence": str(proof.relative_to(ROOT)),
            "manifest_sha256": sha(proof / "manifest.json"),
            "bridge_result_sha256": sha(proof / "bridge-result.json"),
            "rust_tests_passed": 18, "requests_replayed": 1392,
            "unchanged_output_fields": 11136,
            "tested_archived_and_installed_source_sha256": identities,
            "execution": "Sealed Rust suite reused after exact source binding; installed smoke and relocated QP run are fresh.",
        },
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
