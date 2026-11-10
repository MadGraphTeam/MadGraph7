#!/usr/bin/env python3
"""Verify static text bindings; never import or execute numerical code."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(root=None, source_check=False):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    index = json.loads((root / "index.json").read_text())
    assert index["schema"] == "internal-dp-precision-feasibility-index-v1"
    assert set(index["files"]) == {"README.md", "source-bindings.json", "verify.py"}
    for name, expected in index["files"].items():
        data = (root / name).read_bytes()
        assert len(data) == expected["bytes"] and digest(data) == expected["sha256"], name
    bindings = json.loads((root / "source-bindings.json").read_text())
    assert bindings["schema"] == "internal-dp-precision-static-source-bindings-v1"
    assert bindings["execution"] == {"builds": 0, "native_calls": 0, "active_outputs_read": 0}
    for source in bindings["sources"]:
        for excerpt in source["excerpts"]:
            assert digest(excerpt["text"].encode()) == excerpt["sha256"], source["path"]
            assert excerpt["end"] >= excerpt["start"] >= 1
        if source_check:
            repo = root.parents[3]
            path = repo / source["path"]
            data = path.read_bytes()
            assert len(data) == source["bytes"] and digest(data) == source["sha256"], source["path"]
            lines = data.decode().splitlines(keepends=True)
            for excerpt in source["excerpts"]:
                assert "".join(lines[excerpt["start"] - 1:excerpt["end"]]) == excerpt["text"], source["path"]
    return {"verified": True, "source_check": source_check, "source_files": len(bindings["sources"]),
            "index_sha256": digest((root / "index.json").read_bytes())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(source_check=args.source_check), sort_keys=True))
