#!/usr/bin/env python3
"""Seal completed final install/export evidence and run its offline checker."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Also rehash preserved installed and relocated artifacts")
    args = parser.parse_args()
    installation = json.loads((HERE / "installation.json").read_text())
    focused = json.loads((HERE / "focused-unit-results.json").read_text())
    if not installation["passed"] or not focused["passed"]:
        raise RuntimeError("Cannot seal an incomplete installation/export campaign")
    files = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        if path.is_symlink() or path.suffix in (".a", ".o", ".mod", ".so"):
            raise RuntimeError("Build products belong in the private work namespace: " + str(path))
        content = path.read_bytes()
        if content.startswith(b"\x7fELF"):
            raise RuntimeError("An executable is not retained as published evidence")
        files[str(path.relative_to(HERE))] = {
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    manifest = {"schema": "final-pins-install-export-abi3-manifest-v1",
        "revision": installation["revision"], "files": files, "count": len(files),
        "sealed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (HERE / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
    command = [sys.executable, "-B", str(HERE / "verify.py")]
    if args.live:
        command.append("--live")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
