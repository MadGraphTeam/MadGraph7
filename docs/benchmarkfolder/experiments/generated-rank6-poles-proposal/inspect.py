#!/usr/bin/env python3
"""Read-only source inventory; never imports MG, compiles, or loads a reducer.

From the MG repository root, print JSON, or compare to the retained inventory:
  python -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/inspect.py
  python -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/inspect.py --check
The first form writes only stdout. The second checks current bound source bytes.
This inventories generated call sites; it does not establish executed coverage.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CAMPAIGN = Path(".benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate")
POINTS = Path("docs/benchmarkfolder/campaigns/split-dp-terminal-1/points")
PREFIX = Path(".tirust-build/qp-transport-install/mg/HEPTools/tirust")


def binding(path):
    data = (ROOT / path).read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def statements(path):
    """Fixed-form statements with physical line anchors, not a Fortran evaluator."""
    result = []
    for line_number, line in enumerate((ROOT / path).read_text().splitlines(), 1):
        if len(line) < 6 or line[0] in "cC*!":
            continue
        if line[5] != " ":
            if not result:
                raise ValueError("Orphan continuation: " + str(path))
            result[-1][1] += line[6:].strip()
        elif line.strip():
            result.append([line_number, line[6:].strip()])
    return result


def arguments(text):
    parts, start, depth = [], 0, 0
    for position, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if depth < 0:
            raise ValueError("Unbalanced argument list")
        if char == "," and depth == 0:
            parts.append(text[start:position])
            start = position + 1
    if depth:
        raise ValueError("Unbalanced argument list")
    return parts + [text[start:]]


def process(name, expected_rank, expected_count):
    output = CAMPAIGN / name
    subprocesses = sorted((ROOT / output / "SubProcesses").glob("P*"))
    if len(subprocesses) != 1:
        raise ValueError("Expected one generated subprocess")
    sub = subprocesses[0].relative_to(ROOT)
    coefficient_text = (ROOT / sub / "loop_max_coefs.inc").read_text()
    count = int(re.search(r"LOOPMAXCOEFS\s*=\s*(\d+)", coefficient_text)[1])
    ranks = [rank for rank in range(7) if math.comb(rank + 4, 4) == count]
    if ranks != [expected_rank]:
        raise ValueError("Generated rank capacity changed")
    calls = []
    call_files = sorted((ROOT / sub).glob("loop_CT_calls_*.f"))
    for file in call_files:
        path = file.relative_to(ROOT)
        for line, statement in statements(path):
            match = re.fullmatch(r"CALL ML5_0_LOOP_6\((.*)\)", statement)
            if not match:
                continue
            args = arguments(match[1])
            if len(args) != 15 or args[13] != "I_SO":
                raise ValueError("Hexagon call signature changed")
            calls.append({"path": str(path), "line": line,
                          "group_id": int(args[14]), "rank": int(args[12]),
                          "external_wavefunction_ids": list(map(int, args[:6])),
                          "unsquared_mass_arguments": args[6:12],
                          "statement": statement})
    if len(calls) != expected_count or {row["rank"] for row in calls} != {expected_rank}:
        raise ValueError("Hexagon inventory changed")
    if len({row["group_id"] for row in calls}) != len(calls):
        raise ValueError("Repeated group ID")
    point_json = POINTS / (name + ".json")
    point_dat = POINTS / (name + ".dat")
    payload = json.loads((ROOT / point_json).read_text())
    first_point = payload["points"][0]
    text = (ROOT / point_dat).read_text().splitlines()
    coordinates = [list(map(float, row.split())) for row in text[1:7]]
    if first_point["momenta"] != coordinates or len(coordinates) != 6:
        raise ValueError("Saved JSON and decimal point disagree")
    source_paths = [sub / item for item in (
        "loop_max_coefs.inc", "polynomial.f", "TIRUST_interface.f",
        "COLLIER_interface.f", "TIR_interface.f", "CT_interface.f",
        "helas_calls_ampb_1.f", "nexternal.inc", "coupl.inc", "makefile")]
    source_paths += [file.relative_to(ROOT) for file in call_files]
    source_paths += [output / item for item in (
        "Source/DHELAS/aloha_functions.f", "Source/DHELAS/coef_specs.inc",
        "SubProcesses/MadLoopCommons.f", "SubProcesses/global_specs.inc",
        "Cards/param_card.dat", "Cards/ident_card.dat")]
    source_paths += [point_json, point_dat]
    return {"process": name, "subprocess": str(sub), "coefficient_capacity": count,
            "maximum_rank": expected_rank, "hexagon_count": len(calls),
            "hexagon_rank_counts": dict(sorted(Counter(str(row["rank"]) for row in calls).items())),
            "native_group_inputs_captured": False,
            "saved_point_id": first_point["id"], "saved_point_index": 0,
            "saved_point_binary64_hex": [[value.hex() for value in row] for row in coordinates],
            "calls": calls, "bindings": [binding(path) for path in source_paths]}


def inventory():
    paths = [
        "madgraph/loop/loop_exporters.py", "madgraph/various/q_polynomial.py",
        "madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc",
        "madgraph/iolibs/template_files/loop_optimized/COLLIER_interface.inc",
        "vendor/tirust/mtirust.f90", "vendor/tirust/src/lib.rs", "vendor/tirust/tirust.h",
        "vendor/tirust/patches/0001-direct-poles-only.patch",
        "docs/benchmarkfolder/qualification/poles-only/index.json",
        "docs/benchmarkfolder/qualification/poles-only/tensor-commands.txt",
        "docs/benchmarkfolder/qualification/poles-only/tensor-unit-tests.log",
        "docs/benchmarkfolder/qualification/poles-only/tensor-integration-tests.log",
        "docs/benchmarkfolder/qualification/poles-only/scalar-regression.log",
        "docs/benchmarkfolder/qualification/poles-only/patch-validation.json",
        "docs/benchmarkfolder/qualification/tirust-bridge/rust-tests.log",
        "docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache/validate_release.py",
        "docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache/README.md",
        str(PREFIX / "lib/libmadloop_tirust.so"), str(PREFIX / "lib/tirust-config.json"),
    ]
    return {"schema": "generated-rank6-poles-source-proposal-v1",
            "scope": "read-only source inventory; no numerical execution",
            "processes": [process("ew_n4_epem_mupmumaa", 4, 54),
                          process("qcd_n4_uux_zzgg", 5, 40)],
            "shared_bindings": [binding(Path(path)) for path in paths]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = inventory()
    if args.check:
        if result != json.loads((HERE / "inventory.json").read_text()):
            raise SystemExit("Source inventory differs")
        print("PASS: source bytes, 94 hexagon call sites, rank capacities and saved inputs agree; no native execution")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
