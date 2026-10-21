#!/usr/bin/env python3
"""Run the focused checks against the real archived final checkout."""
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import prepare
import runtime_validator as runtime

ROOT, REPORT, SOURCE = prepare.ROOT, prepare.REPORT, prepare.SOURCE


def main():
    result = json.loads((REPORT / "runtime-results.json").read_text())
    assert result["pole_checks_passed"] == result["pole_checks_tested"] == 20
    assert result["nlo_cli_success"] is True
    assert result["standalone"]["qp"]["return_code"] == 120
    installation = json.loads((REPORT / "installation.json").read_text())
    assert installation["passed"]
    identities = installation["archived_bridge_source_sha256"]
    actual = {str(p.relative_to(SOURCE / "vendor/ninjars")): prepare.digest(p)
              for p in sorted((SOURCE / "vendor/ninjars").rglob("*")) if p.is_file()}
    assert actual == identities
    restoration = json.loads((REPORT / "installation-restored.json").read_text())
    assert restoration["restored"] and restoration["hidden_path_absent"]
    hidden = SOURCE / "HEPTools/ninjars"
    assert hidden.is_dir() and not (SOURCE / "HEPTools/ninjars-unavailable").exists()
    assert restoration["installed_file_sha256"] == installation["installed_file_sha256"]
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
        "archived_bridge_source_sha256": identities,
        "published_source_files": published,
        "installation_sha256": prepare.digest(REPORT / "installation.json"),
        "precision_authority": "Separate final dependency numerical proof; this record qualifies installation/export only.",
    })


if __name__ == "__main__":
    main()
