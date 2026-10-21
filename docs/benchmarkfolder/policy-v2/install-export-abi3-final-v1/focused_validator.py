#!/usr/bin/env python3
"""Bind the one-executable final Rust proof and fresh archived Python tests."""
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prepare
import runtime_validator as runtime
import rust_proof

ROOT, REPORT, SOURCE = prepare.ROOT, prepare.REPORT, prepare.SOURCE


def main():
    result = json.loads((REPORT / "runtime-results.json").read_text())
    assert result["pole_checks_passed"] == result["pole_checks_tested"] == 20
    assert result["nlo_cli_success"] is True
    assert result["standalone"]["qp"]["return_code"] == 120
    proof_binding = json.loads((REPORT / "rust-proof-binding.json").read_text())
    proof_path = rust_proof.checked(proof_binding["proof"])
    rust_proof.checked(proof_binding["manifest"])
    proof, summary = rust_proof.load(proof_path, proof_binding["expected_manifest_sha256"])
    assert summary == proof_binding["summary"]
    identities = proof["bridge_source_sha256"]
    assert rust_proof.source_inventory(SOURCE / "vendor/ninjars") == identities
    hidden = SOURCE / "HEPTools/ninjars-unavailable"
    assert hidden.is_dir() and not (SOURCE / "HEPTools/ninjars").exists()
    installation = json.loads((REPORT / "installation.json").read_text())
    assert installation["archived_bridge_source_sha256"] == identities
    assert installation["rust_proof_binding_sha256"] == prepare.digest(REPORT / "rust-proof-binding.json")
    published = {"Cargo.lock": "Cargo.lock", "Cargo.toml": "Cargo.toml",
                 "mninjars.f90": "include/mninjars.f90", "ninjars.h": "include/ninjars.h"}
    for source, installed in published.items():
        assert prepare.digest(hidden / installed) == identities[source], installed
    command = [sys.executable, "-B", str(REPORT / "python_checks.py"),
               "--source", str(SOURCE), "--result", str(REPORT / "focused-python-results.json")]
    log = REPORT / "focused-unit-tests.log"
    with log.open("x") as stream:
        completed = subprocess.run(command, cwd=SOURCE, env=runtime.environment(),
                                   stdout=stream, stderr=subprocess.STDOUT)
    observed = json.loads((REPORT / "focused-python-results.json").read_text())
    assert completed.returncode == 0 and observed["passed"]
    assert observed["tests"] == len(observed["test_ids"]) > 0
    assert observed["failures"] == observed["errors"] == observed["skipped"] == 0
    text = log.read_text()
    assert re.search(r"(?m)^Ran %d tests? in " % observed["tests"], text)
    assert re.search(r"(?m)^OK$", text)
    prepare.write_json(REPORT / "focused-unit-results.json", {
        "passed": True, "python_tests": observed["tests"], "command": command,
        "test_ids": observed["test_ids"], "log_sha256": prepare.digest(log),
        "python_results_sha256": prepare.digest(REPORT / "focused-python-results.json"),
        "final_rust_suite": {
            "proof": proof_binding["proof"], "manifest": proof_binding["manifest"],
            "summary": summary,
            "tested_and_archived_build_input_sha256": identities,
            "published_source_files": published,
            "installation_sha256": prepare.digest(REPORT / "installation.json"),
            "execution": "One final Rust executable is source-bound; installed smoke, relocated QP calls and archived Python checks are fresh.",
        },
    })


if __name__ == "__main__":
    main()
