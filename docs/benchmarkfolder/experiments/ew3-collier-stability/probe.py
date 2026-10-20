#!/usr/bin/env python3
"""Retain isolated original-Collier diagnostics, including failed statuses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_workers as workers

RUN = ROOT / "docs/benchmarkfolder/experiments/persistent-workers/runtime-v1-core-small-20261020"
SOURCE = ROOT / ".benchmark-work/persistent-workers/runtime-v1-core-small-20261020/ew-3-baseline-7/outputs/baseline/ew-3"
WORK = ROOT / ".benchmark-work/experiments/ew3-collier-stability"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    answer = []
    for line in path.read_text().splitlines():
        x = line.split()
        answer.append(dict(point=int(x[0]), order=int(x[1]), status=int(x[2]),
                           coefficients=list(map(float, x[3:7])), accuracy=float(x[7])))
    return answer


def compare(left, right):
    worst, failures = 0., []
    for a, b in zip(left, right, strict=True):
        assert (a["point"], a["order"]) == (b["point"], b["order"])
        scale = max(map(abs, a["coefficients"] + b["coefficients"]))
        for index, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
            tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scale
            fraction = abs(x-y)/tolerance if tolerance else (0. if x == y else float("inf"))
            worst = max(worst, fraction)
            if fraction > 1:
                failures.append(dict(point=a["point"], order=a["order"],
                                     coefficient=index, tolerance_fraction=fraction,
                                     values=[x, y]))
    return dict(passed=not failures, worst_tolerance_fraction=worst, failures=failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", help="Comma-separated diagnostic variants")
    args = parser.parse_args()
    references = {}
    for name in ("baseline-1", "baseline-6", "candidate-8"):
        path = RUN / "data" / f"ew-3-{name}-calibration.json"
        references[name] = json.loads(path.read_text())["rows"]
    settings = [(1, "1d-10"), (1, "1d-12"), (1, "1d-14"),
                (2, "1d-10"), (2, "1d-12"), (3, "1d-10"), (3, "1d-12")]
    if args.labels:
        selected = args.labels.split(",")
        assert set(selected) <= {f"mode{m}-accuracy{a}" for m, a in settings}
        settings = [(m, a) for m, a in settings if f"mode{m}-accuracy{a}" in selected]
    summary_path = HERE / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for mode, accuracy in settings:
        label = f"mode{mode}-accuracy{accuracy}"
        target = WORK / label
        report = HERE / label
        report.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite diagnostic work: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", "--reflink=auto", str(SOURCE), str(target)], check=True)
        for path in target.rglob("*"):
            if path.is_symlink() and path.resolve().is_relative_to(SOURCE):
                replacement = target / path.resolve().relative_to(SOURCE)
                path.unlink()
                path.symlink_to(os.path.relpath(replacement, path.parent))
        directory = next((target / "SubProcesses").glob("P*"))
        card = target / "Cards/MadLoopParams.dat"
        assert card.resolve().is_relative_to(target)
        text = card.read_text()
        for key, value in (("COLLIERMode", str(mode)), ("COLLIERRequiredAccuracy", accuracy)):
            text, count = re.subn(rf"(#{key}\n)[^\n]+", lambda m: m[1] + value, text)
            assert count == 1
        card.write_text(text)
        # The generated resource reader follows the copied resource symlink.
        resources = target / "SubProcesses/MadLoop5_resources/MadLoopParams.dat"
        assert resources.resolve().is_relative_to(target)
        if resources.resolve() != card.resolve():
            resources.write_text(text)
        shutil.copy2(card, report / card.name)
        points = ROOT / "docs/benchmarkfolder/points/ew-3.dat"
        command = ["taskset", "-c", "5", "./check_fresh", str(points), "1"]
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            (directory / name).unlink(missing_ok=True)
        with (report / "run.log").open("w") as stream:
            process = subprocess.run(command, cwd=directory, env=workers.environment(WORK),
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        if process.returncode:
            result = {"acceptance_eligible": False, "returncode": process.returncode,
                      "settings": {"COLLIERMode": mode, "COLLIERRequiredAccuracy": accuracy},
                      "command": command, "executable_sha256": sha(directory / "check_fresh"),
                      "points_sha256": sha(points),
                      "parameter_card_sha256": sha(target / "Cards/param_card.dat")}
            (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            summary[label] = result
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(label, "failed", process.returncode, flush=True)
            continue
        for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
            shutil.copy2(directory / name, report / name)
        output = rows(report / "values.dat")
        result = {
            "acceptance_eligible": False,
            "settings": {"COLLIERMode": mode, "COLLIERRequiredAccuracy": accuracy},
            "command": command, "executable_sha256": sha(directory / "check_fresh"),
            "points_sha256": sha(points), "parameter_card_sha256": sha(target / "Cards/param_card.dat"),
            "status_counts": {str(code): sum(r["order"] == 0 and r["status"] == code for r in output)
                              for code in sorted({r["status"] for r in output})},
            "unstable_points": [r for r in output if r["order"] == 0 and r["status"] // 100 == 4],
            "comparisons": {name: compare(output, reference) for name, reference in references.items()},
        }
        (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        summary[label] = result
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(label, result["status_counts"],
              {name: r["worst_tolerance_fraction"] for name, r in result["comparisons"].items()},
              flush=True)


if __name__ == "__main__":
    main()
