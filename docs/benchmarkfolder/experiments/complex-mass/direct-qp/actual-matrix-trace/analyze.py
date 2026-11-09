#!/usr/bin/env python3
"""Audit all actual traced QP matrix entries using 80-digit decimal arithmetic."""
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent


def main():
    source = REPORT / "cms_actual_mp_matrices.dat"
    matrices = []
    for line in source.read_text().splitlines():
        fields = line.split()
        if fields[0] == "BEGIN":
            current = {"count":int(fields[1]), "threshold":Decimal(fields[2]), "q":{}, "s":{}}
        elif fields[0] == "Q":
            current["q"][int(fields[1])] = list(map(Decimal,fields[2:]))
        elif fields[0] == "S":
            current["s"][int(fields[1]),int(fields[2])] = list(map(Decimal,fields[3:]))
        elif fields[0] == "END":
            assert len(current["q"]) == current["count"]
            assert len(current["s"]) == current["count"]**2
            matrices.append(current)
    audit = json.loads((REPORT.parent / "cms_mp_shell_audit.json").read_text())
    physical = next(row for row in audit["rows"] if row["point_index"] == 20)
    vectors = [[Decimal(x) for x in row] for row in physical["repaired_components"]]
    expected_t = Decimal(physical["physical_80digit_t_gev2"])
    counts = Counter()
    maxima = {name:Decimal(0) for name in ("matrix_identity_residual_gev2", "diagonal_residual_gev2",
        "external_shell_residual_gev2", "guarded_external_virtuality_gev2", "relative_t_error")}
    with localcontext() as context:
        context.prec = 80
        tvector = [a-b for a,b in zip(vectors[0],vectors[2])]
        def matches(vector,target):
            return min(max(abs(a-sign*b) for a,b in zip(vector,target))
                       for sign in (-1,1)) < Decimal("1e-28")
        for matrix in matrices:
            for (i,j), value in matrix["s"].items():
                left,right = matrix["q"][i],matrix["q"][j]
                diff = [a-b for a,b in zip(left[:4],right[:4])]
                invariant = diff[0]**2-sum(x*x for x in diff[1:])
                recovered = [value[k]+left[4+k]+right[4+k] for k in range(2)]
                residual = max(abs(recovered[0]-invariant),abs(recovered[1]))
                maxima["matrix_identity_residual_gev2"] = max(maxima["matrix_identity_residual_gev2"],residual)
                counts["entries"] += 1
                if i == j:
                    counts["diagonal_entries"] += 1
                    maxima["diagonal_residual_gev2"] = max(maxima["diagonal_residual_gev2"],*map(abs,recovered))
                if any(matches(diff,vector) for vector in vectors):
                    counts["external_momentum_entries"] += 1
                    maxima["external_shell_residual_gev2"] = max(maxima["external_shell_residual_gev2"],abs(invariant))
                    maxima["guarded_external_virtuality_gev2"] = max(maxima["guarded_external_virtuality_gev2"],*map(abs,recovered))
                if matches(diff,tvector):
                    counts["physical_t_entries"] += 1
                    maxima["relative_t_error"] = max(maxima["relative_t_error"],abs((recovered[0]-expected_t)/expected_t))
        assert counts["external_momentum_entries"] and counts["physical_t_entries"]
        assert maxima["matrix_identity_residual_gev2"] < Decimal("1e-20")
        assert maxima["guarded_external_virtuality_gev2"] < Decimal("1e-20")
        assert maxima["relative_t_error"] < Decimal("1e-15")
    specification = json.loads((REPORT / "specification.json").read_text())
    assert specification["reference_result_sha256"] == specification["trace_result_sha256"]
    summary = {"scope":"All actual MP_BUILD_KINEMATIC_MATRIX calls during the isolated four-call TIRust QP point20 diagnostic",
        "decimal_precision":80,"point_index":20,"onshell_threshold":1e-24,
        "definition":"Recover q_ij squared as S_ij + m_i squared + m_j squared; identify physical external momenta by their repaired components.",
        "matrices":len(matrices),"propagator_count_histogram":dict(Counter(m["count"] for m in matrices)),
        "counts":dict(counts),"maximum_residuals":{key:str(value) for key,value in maxima.items()},
        "raw_trace_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "amplitude_result_byte_identical_to_untraced_run":True,
        "stability_certified":False,"performance_eligible":False}
    (REPORT / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
