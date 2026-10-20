#!/usr/bin/env python3
"""Seven paired full-evaluation diagnostics of ordinary DP retry overhead."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"tools")]
import ninjars_benchmark as bench

spec = importlib.util.spec_from_file_location(
    "comparison", HERE.parent/"t3-compensation-performance/run.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
spec = importlib.util.spec_from_file_location("retry_process", HERE/"process.py")
process = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process)
sha = comparison.sha
save = comparison.dump


def run(cpu, label):
    report = HERE/"overhead"/label
    report.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {cpu})
    preparation = json.loads((HERE.parent/
        "t3-compensation-performance/first-three-cases-20261020/preparation.json").read_text())
    summary = {"acceptance_eligible": False, "quiet_task_window": True, "cpu": cpu,
               "purpose": "Added ordinary-DP retry overhead, not the final original-Ninja performance gate.",
               "runner_sha256": sha(Path(__file__)), "cases": {}, "complete": False}
    try:
        for case, loops in [("qcd-1", 300), ("ew-2", 30)]:
            target = report/case
            target.mkdir()
            native = preparation["cases"][case]["variants"]["native"]
            built = json.loads((HERE/case/"build.json").read_text())
            paths = {"native": ROOT/native["directory"],
                     "retry": next((process.WORK/case).glob("SubProcesses/P*_*"))}
            archives = {"native": ROOT/preparation["libraries"]["native"]/"libninjars.a",
                        "retry": process.BRIDGE/"libninjars.a"}
            expected = {"native": native["executable_sha256"], "retry": built["executable_sha256"]}
            for variant, directory in paths.items():
                assert sha(directory/"check") == expected[variant]
                expected_source = native["source_sha256"] if variant == "native" else built["generated_sources_sha256"]
                manifest = bench.generated_source_manifest(directory.parent.parent)
                assert manifest["files_sha256"] == expected_source
                save(target/f"{variant}-generated-sources.json", manifest)
                for card in ["MadLoopParams.dat", "param_card.dat"]:
                    shutil.copy2(directory.parent.parent/"Cards"/card, target/f"{variant}-{card}")
            assert (target/"native-MadLoopParams.dat").read_bytes() == (target/"retry-MadLoopParams.dat").read_bytes()
            assert (target/"native-param_card.dat").read_bytes() == (target/"retry-param_card.dat").read_bytes()
            assert sha(archives["native"]) == preparation["archives"]["native"]
            assert sha(archives["retry"]) == built["archive_sha256"]
            points = ROOT/"docs/benchmarkfolder/points"/f"{case}.dat"
            shutil.copy2(points, target/"points.dat")
            item = {"loops": loops, "points_sha256": sha(points),
                    "archive_sha256": {k: sha(v) for k, v in archives.items()},
                    "executable_sha256": expected, "measurements": {"native": [], "retry": []}}
            summary["cases"][case] = item
            for repetition in range(7):
                for variant in (("native", "retry") if repetition % 2 == 0 else ("retry", "native")):
                    directory = paths[variant]
                    assert sha(directory/"check") == expected[variant]
                    for pattern in ["*HelFilter.dat", "*LoopFilter.dat"]:
                        for path in directory.parent.rglob(pattern):
                            path.unlink()
                    artifacts = ["values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"]
                    if variant == "retry":
                        artifacts.append("retry-stats.dat")
                    for name in artifacts:
                        (directory/name).unlink(missing_ok=True)
                    prefix = f"{variant}-{repetition}"
                    command = ["./check", str(points), str(loops)]
                    start = time.perf_counter()
                    with (target/f"{prefix}.log").open("w") as log:
                        result = subprocess.run(command, cwd=directory, env=process.environment(),
                                                stdout=log, stderr=subprocess.STDOUT)
                    for name in artifacts:
                        if (directory/name).exists():
                            shutil.copy2(directory/name, target/f"{prefix}-{name}")
                    result.check_returncode()
                    measured = comparison.parse(directory, loops)
                    assert measured["status_counts"] == {"218": 100}
                    if variant == "retry":
                        rows = [line.split() for line in (directory/"retry-stats.dat").read_text().splitlines()]
                        assert len(rows) == 100
                        assert [int(row[0]) for row in rows] == list(range(1, 101))
                        assert all(list(map(int, row[1:6])) == [218, 0, 0, 0, 0] and
                                   row[6:8] == ["F", "F"] for row in rows)
                    measured.update(repetition=repetition, command=command,
                                    process_wall_seconds=time.perf_counter()-start)
                    prior = item["measurements"][variant]
                    if prior:
                        assert measured["rows"] == prior[0]["rows"]
                    prior.append(measured)
                    print(case, variant, repetition, measured["seconds_per_point"], flush=True)
                    save(report/"result.json", summary)
                assert item["measurements"]["native"][0]["rows"] == item["measurements"]["retry"][0]["rows"]
            times = {variant: [m["seconds_per_point"] for m in runs]
                     for variant, runs in item["measurements"].items()}
            item["statistics"] = bench.runtime_statistics(times["native"], times["retry"], 1.0)
            for key in ["passed", "status", "maximum_ratio"]:
                item["statistics"].pop(key)
            item["ratio_of_medians"] = statistics.median(times["retry"])/statistics.median(times["native"])
            item["physical_rows_exact"] = True
            item["all100_native_dp_zero_retries"] = True
            save(report/"result.json", summary)
        summary["complete"] = True
    except Exception as error:
        summary["error"] = repr(error)
        raise
    finally:
        save(report/"result.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=207)
    parser.add_argument("--label", default="quiet-seven-pairs")
    args = parser.parse_args()
    run(args.cpu, args.label)
