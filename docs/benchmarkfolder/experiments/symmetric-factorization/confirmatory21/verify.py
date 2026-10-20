#!/usr/bin/env python3
"""Offline complete-inventory and numerical/source confirmation verifier."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
MANIFEST = HERE / "SHA256SUMS.json"
RUN_ID = "qcd3-factor-confirmatory21-v1"


def inventory():
    files = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".o", ".mod"):
            continue
        data = path.read_bytes()
        assert not data.startswith(b"\x7fELF"), path
        files[str(path.relative_to(HERE))] = dict(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    for script in ("analyze_completed.py", "retain_sources.py"):
        subprocess.run([sys.executable, str(HERE / script), "--run-id", RUN_ID],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    files = inventory()
    record = dict(
        schema="independent-qp-factor-confirmation-seal-v1",
        files=files, file_count=len(files),
        total_bytes=sum(x["bytes"] for x in files.values()),
        exclusions=["SHA256SUMS.json", "__pycache__/", "*.pyc", "*.o", "*.mod"],
        scope="Completed preregistered independent21-pair prototype confirmation. "
              "Every retained sample, source, recipe, log, preflight and host "
              "monitor record is included. No production acceptance claim.")
    if args.seal:
        with MANIFEST.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    else:
        assert json.loads(MANIFEST.read_text()) == record
    print(json.dumps(dict(
        verified_files=len(files), bytes=record["total_bytes"],
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()),
        indent=2))


if __name__ == "__main__":
    main()
