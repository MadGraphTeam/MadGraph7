#!/usr/bin/env python3
"""Write a deterministic, portable hash index of this evidence directory."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
destination = root / "evidence-index.json"
files = {}
for source in sorted(root.rglob("*")):
    if not source.is_file() or source == destination:
        continue
    relative = source.relative_to(root)
    if "__pycache__" in relative.parts or source.suffix == ".pyc":
        continue
    content = source.read_bytes()
    if content.startswith(b"\x7fELF"):
        raise SystemExit(f"Native binary must not be packaged: {relative}")
    files[str(relative)] = {"bytes": len(content),
                           "sha256": hashlib.sha256(content).hexdigest()}
destination.write_text(json.dumps({
    "schema_version": 1,
    "scope": "Portable packaged evidence files; native binary identities are recorded separately",
    "excluded": ["evidence-index.json", "__pycache__ directories", "*.pyc"],
    "files": files}, indent=2) + "\n")
print(f"Indexed {len(files)} evidence files")
