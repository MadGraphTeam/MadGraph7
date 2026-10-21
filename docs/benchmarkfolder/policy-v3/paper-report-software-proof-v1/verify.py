"""Read-only payload and source-snapshot verification; never run physics/tests."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if sys.flags.optimize:
    raise RuntimeError("Optimized Python is not supported for evidence verification")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(filename):
    return json.loads(filename.read_text())


def digest(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def verify(live=False):
    manifest = load(HERE / "manifest.json")
    require(manifest["schema"] == "ninjars-paper-report-software-proof-v1", "Wrong software capsule")
    require(manifest["scope"] == "software-and-static-source-only", "Wrong evidence scope")
    for name, record in manifest["files"].items():
        filename = (HERE / name).resolve()
        require(filename.is_relative_to(HERE), "Evidence path escapes capsule")
        require(filename.stat().st_size == record["bytes"] and digest(filename) == record["sha256"],
                "Changed software capsule payload: " + name)
    result = load(HERE / "result.json")
    before, after = load(HERE / "source-before.json"), load(HERE / "source-after.json")
    require(before == after == result["source_sha256"], "Tested source changed")
    for original, snapshot in manifest["source_snapshots"].items():
        require(digest(HERE / snapshot) == before[original], "Retained source snapshot differs")
        if live:
            require(digest(ROOT / original) == before[original], "Current tested source differs: " + original)
    require({k: result[k] for k in ("tests_run", "failures", "errors", "skipped", "exit_code", "cpu")} ==
            {"tests_run": 19, "failures": 0, "errors": 0, "skipped": 0, "exit_code": 0, "cpu": 5},
            "Unexpected test result")
    require(digest(HERE / "test.log") == result["stderr_sha256"], "Test log differs")
    log = (HERE / "test.log").read_text()
    require(re.search(r"Ran 19 tests in [0-9.]+s\n\nOK", log) is not None, "Missing retained successful test run")
    require(result["synthetic_visual"]["scope"] == "software fixture only, not physics", "Synthetic scope differs")
    inventory = load(HERE / "static-source-inventory.json")
    require(set(inventory["cases"]) == {f"{kind}-{n}" for kind in ("qcd", "ew") for n in range(1, 5)},
            "Static inventory omits a core process")
    checked = 0
    for case in inventory["cases"].values():
        require(case["status"] == "derived", "Unresolved static count extraction")
        for name, expected in case["inputs_sha256"].items():
            if live:
                filename = (ROOT / name).resolve()
                require(filename.is_relative_to(ROOT), "Generated input escapes repository")
                require(digest(filename) == expected, "Changed generated input: " + name)
            checked += 1
    require(checked == 67, "Static input inventory changed")
    return {"verified": True, "payload_files": len(manifest["files"]), "tests_passed": 19,
            "source_inventory_cases": 8, "source_inventory_inputs": checked,
            "live": live, "numerical_evaluations": 0,
            "scope": "retained software proof only; no physics qualification or test rerun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    print(json.dumps(verify(parser.parse_args().live), sort_keys=True))
