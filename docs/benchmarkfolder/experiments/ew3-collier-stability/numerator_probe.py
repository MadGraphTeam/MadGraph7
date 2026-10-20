#!/usr/bin/env python3
"""Separate original Ninja/CutTools numerator precision from reduction."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from probe import HERE, ROOT, RUN, WORK, compare, rows, sha, workers


def main():
    source_points = ROOT / "docs/benchmarkfolder/points/ew-3.dat"
    lines = source_points.read_text().splitlines()
    ids = [1, 49, 78]
    points = HERE / "selected-points.dat"
    points.write_text("3\n" + "\n".join(
        line for index in ids for line in lines[1+(index-1)*5:1+index*5]) + "\n")
    references = {}
    for label in ("baseline-1", "baseline-6", "candidate-8"):
        original = json.loads((RUN / "data" / f"ew-3-{label}-calibration.json").read_text())
        references[label] = [dict(row, point=ids.index(row["point"])+1)
                             for row in original["rows"] if row["point"] in ids]
    results = {}
    for backend, qp in ((6, True), (6, False), (1, True), (1, False)):
        label = f"backend{backend}-qp-integrand{int(qp)}-forced-after-warm"
        source = ROOT / f".benchmark-work/persistent-workers/runtime-v1-core-small-20261020/ew-3-baseline-{backend}/outputs/baseline/ew-3"
        target, report = WORK / label, HERE / label
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite: {target}")
        report.mkdir(parents=True)
        subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)], check=True)
        for path in target.rglob("*"):
            if path.is_symlink() and path.resolve().is_relative_to(source):
                replacement = target / path.resolve().relative_to(source)
                path.unlink()
                path.symlink_to(os.path.relpath(replacement, path.parent))
        card = target / "Cards/MadLoopParams.dat"
        assert card.resolve().is_relative_to(target)
        text = card.read_text()
        settings = {"CTModeRun": "-1", "UseQPIntegrandForCutTools": f".{str(qp).lower()}.",
                    "UseQPIntegrandForNinja": f".{str(qp).lower()}."}
        for key, value in settings.items():
            text, count = re.subn(rf"(#{key}\n)[^\n]+", lambda m: m[1] + value, text)
            assert count == 1
        card.write_text(text)
        resources = target / "SubProcesses/MadLoop5_resources/MadLoopParams.dat"
        assert resources.resolve().is_relative_to(target)
        if resources.resolve() != card.resolve():
            resources.write_text(text)
        shutil.copy2(card, report / card.name)
        directory = next((target / "SubProcesses").glob("P*"))
        driver = (ROOT / "tools/ninjars_benchmark_driver.f").read_text()
        driver = driver.replace("@PREFIX@", "ML5_0_").replace("@NEXTERNAL@", "5").replace("@NBORN@", "1")
        driver = driver.replace("      IMPLICIT NONE\n", "      IMPLICIT NONE\n      INCLUDE 'MadLoopParams.inc'\n", 1)
        marker = "      IF(WARM_POINTS.LT.NPOINTS) STOP 3\n"
        assert driver.count(marker) == 1
        driver = driver.replace(marker, marker +
            "C     Diagnostic: force QP only after normal initialization.\n"
            "C     Forced mode has no stability estimate. Do not request\n"
            "C     the THRES wrapper's extra initialization-style retry.\n"
            "      CALL ML5_0_FORCE_STABILITY_CHECK(.FALSE.)\n"
            "      CTMODERUN=4\n")
        (directory / "check_sa.f").unlink()
        (directory / "check_sa.f").write_text(driver)
        (directory / "check_sa.o").unlink(missing_ok=True)
        shutil.copy2(directory / "check_sa.f", report / "driver.f")
        with (report / "build.log").open("w") as stream:
            subprocess.run(["make", "-j1", "check", "SHELL=" + shutil.which("bash")],
                           cwd=directory, env=workers.environment(WORK),
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        command = ["taskset", "-c", "5", "./check", str(points), "1"]
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            (directory / name).unlink(missing_ok=True)
        with (report / "run.log").open("w") as stream:
            subprocess.run(command, cwd=directory, env=workers.environment(WORK),
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            shutil.copy2(directory / name, report / name)
        output = rows(report / "values.dat")
        result = {"acceptance_eligible": False, "settings": settings,
                  "forced_mode_after_normal_initialization": 4,
                  "forced_mode_stability_estimate": False,
                  "original_point_indices": ids, "executable_sha256": sha(directory / "check"),
                  "points_sha256": sha(points), "source_points_sha256": sha(source_points),
                  "parameter_card_sha256": sha(target / "Cards/param_card.dat"),
                  "rows": output,
                  "comparisons": {name: compare(output, reference) for name, reference in references.items()}}
        (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results[label] = result
        (HERE / "numerator-summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(label, [(ids[row["point"]-1], row["status"], row["coefficients"][1])
                      for row in output if row["order"] == 0], flush=True)


if __name__ == "__main__":
    main()
