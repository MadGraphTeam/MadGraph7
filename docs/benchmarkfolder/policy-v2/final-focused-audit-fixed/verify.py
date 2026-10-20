#!/usr/bin/env python3
"""Rehash and recompute retained evidence without writing files or running physics."""
import sys

# Set this before loading the companion verifier, even without python -B.
sys.dont_write_bytecode = True

import hashlib
import json
from pathlib import Path
import runpy
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(directory, manifest):
    assert len(manifest["files"]) == manifest["count"]
    for name, expected in manifest["files"].items():
        assert sha(directory / name) == expected, name
    return manifest["count"]


def main():
    if sys.flags.optimize:
        raise RuntimeError("Run without -O; verification uses assertions")
    own_manifest = HERE / "SHA256SUMS.json"
    own_count = None
    if own_manifest.exists():
        own_count = verify_manifest(HERE, json.loads(own_manifest.read_text()))
    audit = json.loads((HERE / "audit.json").read_text())
    assert audit["state"] == "reviewed-source-scoped-evidence"
    for name, record in audit["source_hashes"].items():
        assert sha(ROOT / name) == record["sha256"], name
        committed = subprocess.check_output(
            ["git", "show", f"{audit['source_revision']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == record["sha256"], name
    for record in audit["evidence"]:
        assert record["path"] != "references/PLAN.md", "Bind the retained snapshot"
        assert sha(ROOT / record["path"]) == record["sha256"], record["path"]
    for name, record in audit["sealed_evidence_verified"].items():
        path = ROOT / record["seal"]["path"]
        assert sha(path) == record["seal"]["sha256"]
        count = verify_manifest(ROOT / name, json.loads(path.read_text()))
        assert count == record["verified_files"]
    # This loads function definitions; capture.main() is not called.
    checks = runpy.run_path(str(HERE / "capture.py"), run_name="audit_verifier")
    assert checks["check_plan_snapshot"]() == audit["plan_snapshot"]
    assert checks["check_variants"]() == audit["generated_variants"]
    assert checks["check_reversal"](audit["installed_archive_sha256"]) == audit["physical_reversal"]
    print(json.dumps({
        "passed": True,
        "read_only": True,
        "sources_verified": len(audit["source_hashes"]),
        "evidence_files_verified": len(audit["evidence"]),
        "recomputed_variant_comparisons": len(audit["generated_variants"]["comparisons"]),
        "reversal_rows_verified": audit["physical_reversal"]["complete_rows"],
        "own_sealed_files_verified": own_count,
        "audit_sha256": sha(HERE / "audit.json"),
        "pending": audit["pending"],
    }, indent=2))


if __name__ == "__main__":
    main()
