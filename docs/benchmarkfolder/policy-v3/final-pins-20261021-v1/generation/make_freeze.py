#!/usr/bin/env python3
"""Record root-supplied final identities; no install, compiler or evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
ASSETS = ("Cargo.toml", "Cargo.lock", "include/mninjars.f90", "include/ninjars.h",
          "lib/libninjars.a", "lib/mninjars.mod", "lib/ninjars-config.json",
          "lib/ninjars-link-before.txt", "lib/ninjars-native-libs.txt", "install.log")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--resources-released", action="store_true", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise RuntimeError("Provide the exact committed final revision")
    if not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        raise RuntimeError("Provide the independently validated final archive SHA256")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != args.revision:
        raise RuntimeError("Actual HEAD differs from the root-provided final revision")
    cpus = [int(value) for value in args.cpus.split(",")]
    if not cpus or len(cpus) != len(set(cpus)) or 207 in cpus:
        raise RuntimeError("Choose a coordinated unique non-207 CPU list")
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("A selected generation CPU is unavailable")
    installation = ROOT / "HEPTools/ninjars"
    files = {name: digest(installation / name) for name in ASSETS}
    if files["lib/libninjars.a"] != args.archive_sha256:
        raise RuntimeError("Installed archive differs from root's validated final archive")
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "vendor/ninjars"],
                                      cwd=ROOT).decode().split("\0")
    bridge = {name: digest(ROOT / name) for name in filter(None, tracked)
              if (ROOT / name).is_file()}
    source_digest = hashlib.sha256(json.dumps(bridge, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
    result = {"schema": "final-core-generation-freeze-v1",
              "committed_revision": args.revision,
              "source_and_installation_frozen": True,
              "generation_resource_release": True,
              "generation_cpus": cpus, "bridge_sources_sha256": source_digest,
              "installation_files_sha256": files}
    with (HERE / "freeze.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Final source/installation freeze recorded; no generation or compiler started.")


if __name__ == "__main__":
    main()
