#!/usr/bin/env python3
"""Reproduce the 80-digit physical expectations and check the saved QP audit."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent


def main():
    point_file = REPORT.parent / "points/cms_ew_n2_epem_mupmum.dat"
    lines = point_file.read_text().splitlines()
    count = int(lines[0])
    audit = json.loads((REPORT / "cms_mp_shell_audit.json").read_text())
    assert hashlib.sha256(point_file.read_bytes()).hexdigest() == audit["point_stream_sha256"]
    raw_file = REPORT / "cms_mp_shell_probe.dat"
    assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == audit["probe_raw_sha256"]
    raw = {i:{"matrix":{},"shells":{},"improved":{}} for i in range(1,count+1)}
    for line in raw_file.read_text().splitlines():
        fields = line.split()
        point = raw[int(fields[1])]
        if fields[0] == "KM":
            point["matrix"][tuple(map(int,fields[2:5]))] = list(map(Decimal,fields[5:]))
        elif fields[0] == "SHELL":
            point["shells"][int(fields[2])] = Decimal(fields[3])
        elif fields[0] == "IMPROVED":
            point["improved"][int(fields[2])] = list(map(Decimal,fields[3:]))
    maximum_error = Decimal(0)
    with localcontext() as context:
        context.prec = 80
        for row in audit["rows"]:
            index = row["point_index"]
            p = [list(map(Decimal,line.split())) for line in lines[1+(index-1)*4:1+index*4]]
            energy = p[0][0]+p[1][0]
            norm = sum(x*x for x in p[2][1:]).sqrt()
            direction = [x*energy/(2*norm) for x in p[2][1:]]
            difference = [Decimal(0),-direction[0],-direction[1],energy/2-direction[2]]
            expected = difference[0]**2-sum(x*x for x in difference[1:])
            assert expected == Decimal(row["physical_80digit_t_gev2"])
            evidence = raw[index]
            assert list(evidence["shells"].values()) == list(map(Decimal,row["repaired_shells_gev2"]))
            assert max(map(abs,evidence["shells"].values())) < Decimal("1e-20")
            assert list(evidence["improved"].values()) == [[Decimal(x) for x in vector]
                                                          for vector in row["repaired_components"]]
            for trial,threshold in enumerate(row["threshold_trials"],1):
                matrix = evidence["matrix"]
                for i,j in [(1,2),(2,3),(3,4),(4,1),*( (i,i) for i in range(1,5))]:
                    assert matrix[trial,i,j][2] == 0
                actual_t = matrix[trial,2,4][2]
                assert actual_t == Decimal(threshold["t_gev2"])
                assert abs((actual_t-expected)/expected) == Decimal(
                    threshold["relative_t_error_against_serialized_physical_80digits"])
                if trial == 3:
                    maximum_error = max(maximum_error,abs((actual_t-expected)/expected))
        assert maximum_error < Decimal("1e-15")
    result = {"points":count,"matrix_entries":sum(len(x["matrix"]) for x in raw.values()),
        "decimal_precision":80,"all_repaired_external_shells_and_adjacent_matrix_entries_checked":True,
        "maximum_physical_t_relative_error_at_osthres_1e_minus24":str(maximum_error),
        "stability_certified":False,"performance_eligible":False}
    (REPORT / "cms_mp_shell_audit-validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
