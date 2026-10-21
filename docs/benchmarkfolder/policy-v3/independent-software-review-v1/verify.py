#!/usr/bin/env python3
"""Check this software receipt; never launch a test or numerical engine."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((HERE / "manifest.json").read_text())
assert set(manifest["files"]) == {p.name for p in HERE.iterdir() if p.is_file()} - {"manifest.json"}
for name, record in manifest["files"].items():
    path = HERE / name
    assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], name
sources = json.loads((HERE / "source-bindings.json").read_text())
for name, record in sources.items():
    path = ROOT / name
    assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"], name
result = json.loads((HERE / "result.json").read_text())
assert result["passed"] and result["exit_code"] == 0 and result["tests_run"] == 73
assert result["source_before_sha256"] == result["source_after_sha256"] == {k: v["sha256"] for k, v in sources.items()}
assert digest(HERE / "unittest.stdout") == result["stdout_sha256"]
assert digest(HERE / "unittest.stderr") == result["stderr_sha256"]
log = (HERE / "unittest.stderr").read_text()
assert re.search(r"Ran 73 tests in [0-9.]+s\n\nOK\n$", log)
assert result["scientific_or_timing_acceptance_claim"] is False
print("73 software tests and exact source bindings verified; no physics gate claimed")
