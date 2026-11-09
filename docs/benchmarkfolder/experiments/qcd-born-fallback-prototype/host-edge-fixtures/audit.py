#!/usr/bin/env python3
"""Hash/verify portable host-edge evidence; no compiler or numerical execution."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE / "evidence-index.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    files = []
    for path in sorted(HERE.rglob("*")):
        rel = path.relative_to(HERE)
        if path.is_file() and (path.suffix == '.pyc' or path.read_bytes().startswith(b'\x7fELF')):
            raise ValueError(f"generated binary/bytecode inside portable report: {rel}")
        if path == INDEX or any(part in {"work", "__pycache__"} for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"unexpected portable-evidence symlink: {rel}")
        if path.is_file():
            # Portable evidence is source/text/JSON, never an executable image.
            path.read_text()
            files.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": digest(path)})
    return {"schema": 1, "scope": "Born host policy and wrapper IEEE fixtures; no native reducers or performance acceptance.",
            "excluded_directory_names": ["work", "__pycache__"], "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = inventory()
    if args.write:
        INDEX.write_text(json.dumps(actual, indent=2) + "\n")
    else:
        assert json.loads(INDEX.read_text()) == actual, "portable evidence inventory mismatch"
    policy = json.loads((HERE / "qualification-1/results.json").read_text())
    for run in policy["runs"]:
        assert run["build"]["returncode"] == 0
        for case in run["cases"]:
            assert case["returncode"] == case["expected_returncode"]
    print(f"PASS {len(actual['files'])} portable files; index SHA256 {digest(INDEX)}")


if __name__ == "__main__":
    main()
