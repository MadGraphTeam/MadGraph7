#!/usr/bin/env python3
"""Diagnostic only: isolate rounded QP counterterms in original Collier."""
import json
import os
import re
import shutil
import subprocess

from probe import HERE, ROOT, WORK, compare, rows, sha, workers


def main():
    source = WORK / "mode1-accuracy1d-12"
    target, report = WORK / "rounded-qp-counterterms", HERE / "rounded-qp-counterterms"
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite: {target}")
    report.mkdir()
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)], check=True)
    for path in target.rglob("*"):
        if path.is_symlink() and path.resolve().is_relative_to(source):
            replacement = target / path.resolve().relative_to(source)
            path.unlink()
            path.symlink_to(os.path.relpath(replacement, path.parent))
    directory = next((target / "SubProcesses").glob("P*"))
    names = sorted(set(re.findall(r"\bC_UV\w+\b", (target / "Source/MODEL/coupl.inc").read_text())))
    driver = (ROOT / "tools/ninjars_benchmark_driver.f").read_text()
    driver = driver.replace("@PREFIX@", "ML5_0_").replace("@NEXTERNAL@", "5").replace("@NBORN@", "1")
    driver = driver.replace("      INCLUDE 'coupl.inc'\n",
                            "      INCLUDE 'coupl.inc'\n      INCLUDE 'mp_coupl.inc'\n")
    driver = driver.replace("CALL UPDATE_AS_PARAM()", "CALL UPDATE_AS_PARAM()\n"
                            "      CALL MP_UPDATE_AS_PARAM()\n" +
                            "\n".join(f"      {name}=CMPLX(MP__{name},KIND=8)" for name in names))
    (directory / "check_sa.f").unlink()
    (directory / "check_sa.f").write_text(driver)
    (directory / "check_sa.o").unlink(missing_ok=True)
    shutil.copy2(directory / "check_sa.f", report / "driver.f")
    shutil.copy2(target / "Cards/MadLoopParams.dat", report / "MadLoopParams.dat")
    with (report / "build.log").open("w") as stream:
        subprocess.run(["make", "-j1", "check", "SHELL=" + shutil.which("bash")],
                       cwd=directory, env=workers.environment(WORK),
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
    points = HERE / "selected-points.dat"
    for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
        (directory / name).unlink(missing_ok=True)
    command = ["taskset", "-c", "5", "./check", str(points), "1"]
    with (report / "run.log").open("w") as stream:
        subprocess.run(command, cwd=directory, env=workers.environment(WORK),
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
    for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
        shutil.copy2(directory / name, report / name)
    output = rows(report / "values.dat")
    references = {}
    for backend in (1, 6):
        path = HERE / f"backend{backend}-qp-integrand1-forced-after-warm/result.json"
        references[f"backend{backend}-full-qp"] = json.loads(path.read_text())["rows"]
    result = {"acceptance_eligible": False,
              "scope": "Isolated diagnosis; QP counterterm couplings rounded into DP after each model update",
              "replaced_couplings": names, "command": command,
              "points_sha256": sha(points), "parameter_card_sha256": sha(target / "Cards/param_card.dat"),
              "executable_sha256": sha(directory / "check"), "rows": output,
              "comparisons": {name: compare(output, reference) for name, reference in references.items()}}
    (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
