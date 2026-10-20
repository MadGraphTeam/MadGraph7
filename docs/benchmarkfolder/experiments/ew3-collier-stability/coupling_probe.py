#!/usr/bin/env python3
"""Compare independently generated DP/QP model couplings before reduction."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from probe import HERE, ROOT, SOURCE, WORK, sha, workers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal", action="store_true")
    args = parser.parse_args()
    label = "internal-parameters" if args.internal else "couplings-v2"
    work = WORK / label
    report = HERE / label
    work.mkdir(parents=True, exist_ok=False)
    report.mkdir()
    model = SOURCE / "Source/MODEL"
    names = sorted(set(re.findall(r"\b(?:GC_\w+|R2_\w+|C_UV\w+)\b",
                                 (model / "coupl.inc").read_text())))
    mp = (model / "mp_coupl.inc").read_text()
    includes = []
    if args.internal:
        names = sorted(set(re.findall(r"\bMDL_\w*FIN_\w*\b",
                                      (model / "input.inc").read_text())))
        mp = (model / "mp_input.inc").read_text()
        includes = ["      INCLUDE 'input.inc'", "      INCLUDE 'mp_input.inc'"]
    assert all("MP__" + name in mp for name in names)
    lines = ["      PROGRAM MODEL_PRECISION", "      IMPLICIT NONE",
             "      INCLUDE 'coupl.inc'", "      INCLUDE 'mp_coupl.inc'",
             *includes,
             "      CALL SETPARA('Cards/param_card.dat')", "      MU_R=1000D0",
             "      CALL UPDATE_AS_PARAM()", "      CALL MP_UPDATE_AS_PARAM()",
             "      OPEN(73,FILE='couplings.dat',STATUS='REPLACE')"]
    for name in names:
        lines += [f"      WRITE(73,'(A,4ES26.17E3)') '{name} ',",
                  f"     $ DBLE({name}),DIMAG(CMPLX({name},KIND=8)),",
                  f"     $ REAL(MP__{name},KIND=8),",
                  f"     $ REAL(AIMAG(CMPLX(MP__{name},KIND=16)),KIND=8)"]
    lines += ["      CLOSE(73)", "      END"]
    driver = work / "probe.f"
    driver.write_text("\n".join(lines) + "\n")
    shutil.copy2(driver, report / driver.name)
    (work / "Cards").mkdir()
    for name in ("param_card.dat", "ident_card.dat"):
        shutil.copy2(SOURCE / "Cards" / name, work / "Cards" / name)
    command = ["gfortran", "-ffixed-line-length-132", "-I" + str(model), str(driver),
               "-L" + str(SOURCE / "lib"), "-lmodel", "-o", str(work / "probe")]
    with (report / "build.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=workers.environment(WORK),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    with (report / "run.log").open("w") as stream:
        subprocess.run(["taskset", "-c", "5", "./probe"], cwd=work,
                       env=workers.environment(WORK), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    shutil.copy2(work / "couplings.dat", report / "couplings.dat")
    differences = []
    for line in (work / "couplings.dat").read_text().splitlines():
        name, *values = line.split()
        dr, di, qr, qi = map(float, values)
        dp, qp = complex(dr, di), complex(qr, qi)
        absolute = abs(dp-qp)
        scale = max(abs(dp), abs(qp))
        differences.append({"name": name, "double": [dr, di], "quadruple": [qr, qi],
                            "absolute_difference": absolute,
                            "relative_difference": absolute/scale if scale else 0.})
    differences.sort(key=lambda x: x["relative_difference"], reverse=True)
    result = {"acceptance_eligible": False, "couplings": len(names),
              "internal_parameters": args.internal,
              "model_archive_sha256": sha(SOURCE / "lib/libmodel.a"),
              "parameter_card_sha256": sha(work / "Cards/param_card.dat"),
              "executable_sha256": sha(work / "probe"), "command": command,
              "differences": differences}
    (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(differences[:12], indent=2))


if __name__ == "__main__":
    main()
