#!/usr/bin/env python3
"""Source-only QCD3/QCD4 QP operation inventory; never runs a generated process."""
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
RUN = "corrected-original-core-v2-20261020"
WORK = ROOT / ".benchmark-work/policy-v2/reference-runs" / RUN
REPORT = ROOT / "docs/benchmarkfolder/policy-v2/reference-runs" / RUN


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical(path):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip() or line[0] in "cC*!":
            continue
        text = re.sub(r"\s+", "", line[6:]).upper()
        if line[5] not in " 0":
            result[-1] += text
        else:
            result.append(text)
    return result


def inspect(case):
    output = WORK / "workers" / (case + "-baseline-1") / "outputs/baseline" / case
    process = next((output / "SubProcesses").glob("P*_*"))
    resources = output / "SubProcesses/MadLoop5_resources"
    helicity = next(resources.glob("*_HelFilter.dat"))
    values = list(map(int, helicity.read_text().split()))
    matrix = process / "loop_matrix.f"
    matrix_text = "".join(logical(matrix))
    ncomb = int(re.search(r"PARAMETER\(NCOMB=(\d+)\)", matrix_text)[1])
    offset = int(re.search(r"DATAHELOFFSET/(\d+)/", matrix_text)[1])
    assert len(values) == ncomb
    active = sum(value > -offset and value != 0 for value in values)
    polynomial = process / "polynomial.f"
    routines, current = {}, None
    for line in logical(polynomial):
        match = re.match(r"SUBROUTINE(MP_\w+UPDATE_WL_\d+_\d+)\(", line)
        if match:
            current = match[1]
            routines[current] = []
        if current:
            routines[current].append(line)
            if line == "END":
                current = None
    sources = {str(p.relative_to(ROOT)): digest(p)
               for p in (matrix, polynomial, helicity)}
    calls, vertices = [], {}
    pattern = re.compile(
        r"CALL(MP_\w+UPDATE_WL_(\d+)_(\d+))"
        r"\(WL\(1,0,1,-?\d+\),(\d+),COEFS,(\d+),(\d+),WL\(")
    for path in sorted(process.glob("mp_coef_construction_*.f")):
        sources[str(path.relative_to(ROOT))] = digest(path)
        vertex = None
        for line in logical(path):
            match = re.match(r"CALL(MP_\w+)\(.*?,COEFS\)$", line)
            if match:
                vertex = match[1]
            match = pattern.match(line)
            if not match:
                continue
            name, left, right, cut, ins, outs = match.groups()
            left, right, cut, ins, outs = map(int, (left, right, cut, ins, outs))
            assert vertex is not None
            has_b_skip = any(row.startswith("IF(UPDATER_COEF.EQ.")
                             for row in routines[name])
            if vertex not in vertices:
                vpath = output / "Source/DHELAS" / (vertex + ".f")
                sources[str(vpath.relative_to(ROOT))] = digest(vpath)
                assignments = {}
                for row in logical(vpath):
                    assignment = re.fullmatch(r"COEFF\((\d+),(\d+),(\d+)\)=(.*)", row)
                    if assignment:
                        key = tuple(map(int, assignment.groups()[:3]))
                        assert key not in assignments
                        assignments[key] = assignment[4]
                vertices[vertex] = assignments
            a_count, b_count = comb(left + 4, 4), comb(right + 4, 4)
            selected = [vertices[vertex][j, m, k] for j in range(1, outs + 1)
                        for m in range(b_count) for k in range(1, ins + 1)]
            zeros = sum(rhs in ("0", "0D0", "0Q0") for rhs in selected)
            dense = cut * a_count * len(selected)
            after_literals = cut * a_count * (len(selected) - zeros if has_b_skip
                                             else len(selected))
            calls.append({"routine": name, "ranks": [left, right],
                          "shape": [cut, ins, outs], "vertex": vertex,
                          "compact_B_zero_branch": has_b_skip,
                          "dense_complex_products": dense,
                          "products_after_literal_B_skips": after_literals})
    return {
        "ncomb": ncomb, "active_helicities": active,
        "helicity_filter_counts": dict(Counter(map(str, values))),
        "update_calls_per_full_helicity_pass": len(calls),
        "calls_by_rank": dict(Counter(f"{c['ranks'][0]}/{c['ranks'][1]}" for c in calls)),
        "dense_complex_products_per_helicity": sum(c["dense_complex_products"] for c in calls),
        "products_after_literal_B_skips_per_helicity": sum(
            c["products_after_literal_B_skips"] for c in calls),
        "sources_sha256": sources, "calls": calls,
    }


def main():
    data = {case: inspect(case) for case in ("qcd-3", "qcd-4")}
    for item in data.values():
        item["dense_complex_products_per_full_QP_point"] = (
            item["active_helicities"] * item["dense_complex_products_per_helicity"])
        item["products_after_literal_B_skips_per_full_QP_point"] = (
            item["active_helicities"] * item["products_after_literal_B_skips_per_helicity"])
    ratio = (data["qcd-4"]["products_after_literal_B_skips_per_full_QP_point"]
             / data["qcd-3"]["products_after_literal_B_skips_per_full_QP_point"])
    timing_file = REPORT / "data/qcd-3-baseline-1-0.json"
    timing = json.loads(timing_file.read_text())
    fraction_qp = sum(n for code, n in timing["status_counts"].items()
                      if int(code) // 100 == 3) / sum(timing["status_counts"].values())
    upper_qp_estimate = timing["seconds_per_point"] / fraction_qp
    result = {
        "schema": "static-QP-workload-scaling-v1",
        "cases": data, "qcd4_qcd3_static_product_ratio": ratio,
        "qcd3_measurement": {"path": str(timing_file.relative_to(ROOT)),
                            "sha256": digest(timing_file),
                            "seconds_per_point": timing["seconds_per_point"],
                            "QP_fraction": fraction_qp},
        "illustrative_qcd3_QP_seconds_assigning_all_mixed_time_to_QP": upper_qp_estimate,
        "illustrative_qcd4_seconds_per_all_QP_point": upper_qp_estimate * ratio,
        "illustrative_qcd4_seconds_per_100_all_QP_points": upper_qp_estimate * ratio * 100,
        "scope": "Static full QP numerator updater complex-product counts, after literal "
                 "B-zero skips only where the original compact kernel actually branches. "
                 "Nonliteral B values may also vanish. A-zero products remain in original "
                 "kernels. Counts exclude ALOHA construction, merging, clearing, scalar "
                 "integrals and reduction. Filter values are retained startup output. "
                 "A full point multiplies by active helicities, assuming every generated "
                 "coefficient batch is evaluated once per active helicity. This does not "
                 "measure the live point index, rescue fraction or initialization work. "
                 "The illustrative time assigns all QCD3 mixed-run time to its QP points "
                 "and scales linearly by static products; it is not a bound or ETA.",
        "acceptance_evidence": False, "physics_evaluations_started": False,
        "analyzer_sha256": digest(Path(__file__)),
    }
    destination = Path(__file__).with_name("analysis.json")
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**{k: v for k, v in result.items() if k != "cases"},
                      "cases": {key: {k: v for k, v in item.items()
                                     if k not in ("sources_sha256", "calls")}
                                for key, item in data.items()}}, indent=2))


if __name__ == "__main__":
    main()
