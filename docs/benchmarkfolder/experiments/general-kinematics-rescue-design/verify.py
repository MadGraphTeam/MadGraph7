#!/usr/bin/env python3
"""Read-only inventory/source verification; no native imports or subprocesses."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPECTED = {"README.md", "source-bindings.json", "verify.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(name):
    path = Path(name)
    require(not path.is_absolute() and name and ".." not in path.parts,
            "unsafe relative path: " + name)
    return path


def check_file(base, name, entry):
    path = base / relative(name)
    require(not path.is_symlink() and path.is_file(), "nonregular file: " + name)
    require(path.resolve().is_relative_to(base.resolve()), "path escapes root: " + name)
    raw = path.read_bytes()
    require(len(raw) == entry["bytes"], "byte count mismatch: " + name)
    require(hashlib.sha256(raw).hexdigest() == entry["sha256"], "hash mismatch: " + name)
    return raw


def verify(sources=False):
    index = json.loads((HERE / "evidence-index.json").read_text())
    require(index["schema"] == 1 and index["production_change"] is False,
            "invalid inventory semantics")
    require(set(index["files"]) == EXPECTED, "unexpected inventory entries")
    entries = list(HERE.rglob("*"))
    require(not any(path.is_symlink() for path in entries), "symlink in package")
    actual = {str(path.relative_to(HERE)) for path in entries if path.is_file()}
    require(actual == EXPECTED | {"evidence-index.json"}, "incomplete or extra package files")
    for name, entry in index["files"].items():
        raw = check_file(HERE, name, entry)
        require(not raw.startswith(b"\x7fELF"), "native file in source-only package")
    bindings = json.loads((HERE / "source-bindings.json").read_text())
    require(bindings["schema"] == 1 and bindings["source_only"] is True,
            "invalid source bindings")
    for name in ("compiler_executed", "native_executed", "production_modified", "physics_qualified"):
        require(bindings[name] is False, "invalid source-only claim: " + name)
    checked = 0
    for category in ("sources", "evidence"):
        for name, entry in bindings[category].items():
            relative(name)
            require(isinstance(entry["bytes"], int) and entry["bytes"] >= 0,
                    "invalid byte count: " + name)
            require(len(entry["sha256"]) == 64 and
                    all(c in "0123456789abcdef" for c in entry["sha256"]),
                    "invalid hash: " + name)
            if sources:
                raw = check_file(ROOT, name, entry)
                lines = raw.decode("utf-8").splitlines()
                for anchor in entry.get("anchors", []):
                    found = [i for i, line in enumerate(lines, 1) if anchor["needle"] in line]
                    require(found == anchor["line_numbers"], "source anchor changed: " + name)
                checked += 1
    return {"verification": "pass", "package_files": len(EXPECTED) + 1,
            "bound_files": sum(len(bindings[k]) for k in ("sources", "evidence")),
            "external_files_checked": checked, "native_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", action="store_true")
    args = parser.parse_args()
    try:
        report = verify(args.sources)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, "Verification failed: " + str(exc) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
