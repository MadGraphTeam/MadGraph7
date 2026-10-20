#!/usr/bin/env python3
"""Test corrected generated MP helpers without modifying the original baseline.

Each target is an isolated copy of a retained generated process. Only the MP
helper section is regenerated; its DP section and all physical inputs must be
byte-identical. These are correctness diagnostics, not performance submissions.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

from probe import HERE, ROOT, RUN, SOURCE, WORK, compare, rows, sha, workers

sys.path.insert(0, str(ROOT))
from madgraph.iolibs.export_v4 import UFO_model_to_mg4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="1,6,7")
    args = parser.parse_args()
    summary = {}
    output = HERE / "corrected-helpers-v2-summary.json"
    if output.exists():
        summary = json.loads(output.read_text())
    for backend in map(int, args.backends.split(",")):
        assert backend in (1, 6, 7, 8)
        label = f"corrected-helpers-v2-{backend}"
        target, report = WORK / label, HERE / label
        source = SOURCE
        if backend == 8:
            source = ROOT / (
                ".benchmark-work/persistent-workers/runtime-v1-core-small-20261020/"
                "ew-3-candidate-8/outputs/candidate/ew-3")
        if target.exists() or report.exists():
            raise RuntimeError(f"Refusing to overwrite {label}")
        report.mkdir()
        subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)],
                       check=True)
        for path in target.rglob("*"):
            if path.is_symlink() and path.resolve().is_relative_to(source):
                replacement = target / path.resolve().relative_to(source)
                path.unlink()
                path.symlink_to(os.path.relpath(replacement, path.parent))
        model = target / "Source/MODEL"
        helper = model / "model_functions.f"
        old = helper.read_text()
        assert not helper.is_symlink()
        exporter = UFO_model_to_mg4.__new__(UFO_model_to_mg4)
        exporter.dir_path = str(model)
        exporter.model = {"modelpath": str(model), "functions": []}
        exporter.opt = {"mp": True}
        with exporter.open("model_functions.f", format="fortran") as writer:
            exporter.open = lambda *args, **kwargs: writer
            exporter.create_model_functions_def()
        new = helper.read_text()
        # Keep any user-defined functions after the built-in MP_ARG, including
        # EW CRECMS/MP_CRECMS. The first exploratory version dropped that tail;
        # retain its observations but use this variant for scoped attribution.
        last_builtin = r"(?ms)^ +COMPLEX\*32 FUNCTION MP_ARG\(COMNUM\)\n.*?^ +END\n"
        old_end = re.search(last_builtin, old).end()
        new_end = re.search(last_builtin, new).end()
        new = new[:new_end] + old[old_end:]
        helper.write_text(new)
        marker = "      COMPLEX*32 FUNCTION MP_COND("
        assert old.split(marker)[0] == new.split(marker)[0]
        assert new[re.search(last_builtin, new).end():] == old[old_end:]
        (report / "original-model_functions.f").write_text(old)
        shutil.copy2(helper, report / "corrected-model_functions.f")
        card = target / "Cards/MadLoopParams.dat"
        text = card.read_text()
        for key, value in (("MLReductionLib", str(backend)),
                           ("COLLIERRequiredAccuracy", "1d-12")):
            text, count = re.subn(rf"(#{key}\n)[^\n]+", lambda m: m[1]+value, text)
            assert count == 1
        card.write_text(text)
        resource = target / "SubProcesses/MadLoop5_resources/MadLoopParams.dat"
        assert resource.resolve().is_relative_to(target)
        if resource.resolve() != card.resolve():
            resource.write_text(text)
        shutil.copy2(card, report / card.name)
        directory = next((target / "SubProcesses").glob("P*"))
        driver = (ROOT / "tools/ninjars_benchmark_driver.f").read_text()
        for key, value in (("@PREFIX@", "ML5_0_"), ("@NEXTERNAL@", "5"),
                           ("@NBORN@", "1")):
            driver = driver.replace(key, value)
        (directory / "check_sa.f").unlink()
        (directory / "check_sa.f").write_text(driver)
        (directory / "check_sa.o").unlink(missing_ok=True)
        shutil.copy2(directory / "check_sa.f", report / "driver.f")
        with (report / "build.log").open("w") as log:
            for cwd, command in (
                (model, ["make", "-j1"]),
                (directory, ["make", "-j1", "check"])):
                subprocess.run(command + ["SHELL=" + shutil.which("bash")],
                               cwd=cwd, env=workers.environment(WORK),
                               stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=600)
        points = ROOT / "docs/benchmarkfolder/points/ew-3.dat"
        command = ["taskset", "-c", "5", "./check", str(points), "1"]
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            (directory / name).unlink(missing_ok=True)
        with (report / "run.log").open("w") as log:
            subprocess.run(command, cwd=directory, env=workers.environment(WORK),
                           stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            shutil.copy2(directory / name, report / name)
        result_rows = rows(report / "values.dat")
        references = {
            "original-collier-accuracy1d-12":
                rows(HERE / "mode1-accuracy1d-12/values.dat"),
            "original-ninja": json.loads(
                (RUN / "data/ew-3-baseline-6-calibration.json").read_text())["rows"],
        }
        result = {
            "acceptance_eligible": False,
            "scope": "Isolated original export plus corrected MP helpers"
                     if backend != 8 else "Isolated candidate export plus corrected MP helpers",
            "backend": backend,
            "rows": result_rows,
            "status_counts": {str(s): sum(r["order"] == 0 and r["status"] == s
                                         for r in result_rows)
                              for s in sorted({r["status"] for r in result_rows})},
            "comparisons": {name: compare(result_rows, reference)
                            for name, reference in references.items()},
            "dp_helper_source_unchanged": True,
            "user_defined_function_tail_unchanged": True,
            "original_helpers_sha256": sha(report / "original-model_functions.f"),
            "corrected_helpers_sha256": sha(helper),
            "exporter_sha256": sha(ROOT / "madgraph/iolibs/export_v4.py"),
            "model_archive_sha256": sha(target / "lib/libmodel.a"),
            "executable_sha256": sha(directory / "check"),
            "parameter_card_sha256": sha(target / "Cards/param_card.dat"),
            "points_sha256": sha(points),
            "command": command,
        }
        (report / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        summary[label] = result
        output.write_text(json.dumps(summary, indent=2)+"\n")
        print(label, result["status_counts"],
              {k: v["worst_tolerance_fraction"]
               for k, v in result["comparisons"].items()}, flush=True)


if __name__ == "__main__":
    main()
