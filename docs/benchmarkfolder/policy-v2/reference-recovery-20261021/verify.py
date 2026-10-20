#!/usr/bin/env python3
"""Verify sealed preparation and the untouched interrupted report, read-only."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(include_old):
    seal = json.loads((HERE / "evidence-seal.json").read_text())
    unsigned = {key: value for key, value in seal.items() if key != "identity_sha256"}
    identity = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != seal["identity_sha256"]:
        raise RuntimeError("Recovery seal identity differs")
    files = seal["files"]
    total = 0
    for name, item in files.items():
        path = ROOT / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise RuntimeError(f"Changed recovery/preparation evidence: {name}")
        total += item["bytes"]
    old_count = 0
    if include_old:
        audit = json.loads((HERE / "pretermination-audit.json").read_text())
        for name, checksum in audit["old_report_files_sha256"].items():
            if digest(ROOT / name) != checksum:
                raise RuntimeError(f"Interrupted report changed: {name}")
            old_count += 1
    return {"passed": True, "sealed_files": len(files), "sealed_bytes": total,
            "old_report_files_checked": old_count, "signals_sent": [],
            "numerical_jobs_started": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-old-report", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.include_old_report), indent=2))
