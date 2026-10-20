#!/usr/bin/env python3
"""Verify the held recipe bytes; never import numerical or benchmark modules."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    seal = json.loads((HERE / "preparation-seal.json").read_text())
    unsigned = {key: value for key, value in seal.items() if key != "identity_sha256"}
    identity = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
    if identity != seal["identity_sha256"]:
        raise RuntimeError("Held preparation seal identity differs")
    for name, record in seal["files"].items():
        path = ROOT / name
        if (path.stat().st_size != record["bytes"] or
                hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise RuntimeError("Held preparation changed: " + name)
    print(json.dumps({"passed": True, "files": len(seal["files"]),
                      "preparation_identity_sha256": identity,
                      "generation_or_compilers_started": False}, indent=2))


if __name__ == "__main__":
    main()
