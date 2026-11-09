#!/usr/bin/env python3
"""Freeze or verify the complete CMS evidence, including ignored raw logs."""
import argparse
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[3]
INDEX = REPORT / "artifact-index.json"


def inventory():
    paths = [p for p in REPORT.rglob("*") if p.is_file() and p != INDEX
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths += [ROOT / "tools/tirust_complex_stress.py",
              ROOT / "tests/unit_tests/various/test_tirust_complex_stress.py"]
    return [{"path":str(p.relative_to(ROOT)), "bytes":p.stat().st_size,
             "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(set(paths))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write the final evidence index")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the MadGraph repository root")
    files = inventory()
    if args.write:
        record = {"schema":1, "scope":"CMS primary observations and separately labeled causal diagnostics",
                  "primary_extreme_points_qualified":False,
                  "point20_cuttools_disagreement_cause":"Legacy scalar cutoff and DP denominator transport",
                  "production_cuttools_changed":False, "files":files}
        INDEX.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    else:
        if json.loads(INDEX.read_text())["files"] != files:
            raise SystemExit("CMS evidence index differs from current files")
    print(json.dumps({"indexed_files":len(files), "bytes":sum(p["bytes"] for p in files),
                      "logs":sum(p["path"].endswith(".log") for p in files),
                      "action":"written" if args.write else "verified"}))


if __name__ == "__main__":
    main()
