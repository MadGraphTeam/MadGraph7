#!/usr/bin/env python3
"""Source-only inventory of identical color rows used by CREATE_LOOP_COEFS."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def statements(path):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip() or line[0].upper() in ("C", "*", "!"):
            continue
        if len(line) > 5 and line[5] not in (" ", "0"):
            result[-1] += line[6:].strip()
        else:
            result.append(line[6:].strip())
    return result


def arguments(call):
    text = call[call.index("(")+1:-1]
    result, part, depth = [], [], 0
    for char in text:
        if char == "," and depth == 0:
            result.append("".join(part))
            part = []
        else:
            part.append(char)
            depth += (char == "(") - (char == ")")
    result.append("".join(part))
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(output):
    directory = next(output.glob("SubProcesses/P*_*"))
    resource = output / "SubProcesses/MadLoop5_resources"
    num = next(resource.glob("*ColorNumFactors.dat"))
    den = next(resource.glob("*ColorDenomFactors.dat"))
    numerators = [tuple(map(int, line.split())) for line in num.read_text().splitlines()]
    denominators = [tuple(map(int, line.split())) for line in den.read_text().splitlines()]
    assert len(numerators) == len(denominators)
    rows = [tuple(zip(n, d)) for n, d in zip(numerators, denominators)]
    ids = []
    source_files = {}
    for path in sorted(directory.glob("coef_construction_*.f")):
        source_files[str(path.relative_to(ROOT))] = digest(path)
        for line in statements(path):
            if line.startswith("CALL ") and "_CREATE_LOOP_COEFS(" in line:
                args = arguments(line)
                if len(args) == 8:
                    ids.append(int(args[6]))
    if not ids:
        return {"case": output.name, "status": "No Born-interference CREATE_LOOP_COEFS calls",
                "matrix_rows": len(rows)}
    selected = [rows[i-1] for i in ids]
    groups = Counter(selected)
    nonzero = sum(any(n != 0 for n, d in row) for row in groups)
    matrix_source = directory / "loop_matrix.f"
    dimensions = {}
    for key in ("NCOMB", "NAMPSO", "NSQUAREDSO", "NBORNAMPS", "NCTAMPS"):
        match = re.search(r"\b" + key + r"\s*=\s*(\d+)", matrix_source.read_text())
        dimensions[key] = int(match[1]) if match else None
    return {
        "case": output.name, "status": "source-only",
        "dimensions": dimensions, "matrix_rows": len(rows),
        "matrix_columns": len(rows[0]), "all_distinct_exact_integer_rows": len(set(rows)),
        "create_calls_per_complete_helicity": len(ids),
        "distinct_color_ids_used": len(set(ids)),
        "distinct_exact_rows_used": len(groups),
        "nonzero_exact_rows_used": nonzero,
        "repeated_contractions_per_complete_helicity": len(ids) - len(groups),
        "potential_fraction_reused": 1 - len(groups)/len(ids),
        "row_class_multiplicities_descending": sorted(groups.values(), reverse=True),
        "maximum_dp_qp_cache_bytes_at_ncolorrows": (
            len(rows) * dimensions["NAMPSO"] * (16+32)),
        "inputs_sha256": {
            str(num.relative_to(ROOT)): digest(num),
            str(den.relative_to(ROOT)): digest(den),
            str(matrix_source.relative_to(ROOT)): digest(matrix_source),
            **source_files,
        },
        "scope": "All statically emitted CREATE calls; helicity/SO runtime filtering can reduce executed calls. Identical ordered integer rows only, without factoring signs or rescaling.",
    }


if __name__ == "__main__":
    outputs = ROOT / ".benchmark-work/outputs/candidate"
    cases = ["qcd-1", "qcd-2", "qcd-3", "qcd-4", "ew-1", "ew-2", "ew-3", "ew-4",
             "control-gg-h", "control-mixed"]
    data = [inventory(outputs / case) for case in cases if (outputs / case).is_dir()]
    (HERE / "row-inventory.json").write_text(json.dumps(data, indent=2) + "\n")
    for row in data:
        print(row["case"], row.get("create_calls_per_complete_helicity"),
              row.get("distinct_exact_rows_used"), row.get("potential_fraction_reused"))
