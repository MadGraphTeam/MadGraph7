#!/usr/bin/env python3
"""Capture original ABI2 MP inputs/outputs in a separate untimed executable."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import importlib.util
spec = importlib.util.spec_from_file_location("probe", HERE / "process-probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
WORK = probe.WORK
OUTPUT = WORK / "qcd3-capture"
REPORT = HERE / "capture"


def prepare():
    if OUTPUT.exists():
        raise RuntimeError("Refusing to overwrite captured process")
    original = WORK / "qcd3-production"
    subprocess.run(["cp", "-a", "--reflink=auto", str(original), str(OUTPUT)], check=True)
    for path in OUTPUT.rglob("*"):
        if path.is_symlink() and path.resolve().is_relative_to(original):
            replacement = OUTPUT / path.resolve().relative_to(original)
            path.unlink()
            path.symlink_to(os.path.relpath(replacement, path.parent))
    REPORT.mkdir()
    source = HERE / "capture-mp.c"
    object_path = WORK / "capture-mp.o"
    with (REPORT / "build.log").open("w") as stream:
        subprocess.run(["gcc", "-O2", "-fPIC", "-Wall", "-Wextra", "-Werror",
                        "-c", str(source), "-o", str(object_path)], check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
        definitions = OUTPUT / "SubProcesses/MadLoop_makefile_definitions"
        text = definitions.read_text()
        text = text.replace("LINK_LOOP_LIBS = ", "LINK_LOOP_LIBS = " +
                            str(object_path) + " -Wl,--wrap=ninjars_evaluate_mp ", 1)
        definitions.write_text(text)
        directory = probe.proc(OUTPUT)
        (directory / "check").unlink()
        subprocess.run(["make", "-j1", "check", "SHELL=" + shutil.which("bash")],
                       cwd=directory, env=probe.workers.environment(WORK),
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    probe.dump(REPORT / "build.json", {
        "acceptance_eligible": False,
        "numerical_library": "unchanged production ABI2",
        "library_sha256": probe.sha(ROOT / "HEPTools/ninjars/lib/libninjars.a"),
        "executable_sha256": probe.sha(directory / "check"),
        "wrapper_sha256": probe.sha(source),
        "driver_sha256": probe.sha(directory / "check_sa.f"),
        "point_indices": probe.POINT_IDS,
        "points_sha256": probe.sha(HERE / "selected-points.dat"),
    })


def run(cpu):
    capture = WORK / "actual-mp-requests.jsonl"
    directory = probe.proc(OUTPUT)
    env = probe.workers.environment(WORK)
    env["NINJARS_TENSOR_CAPTURE"] = str(capture)
    worker = None
    result = {"acceptance_eligible": False, "status": "running", "cpu": cpu}
    try:
        worker = probe.workers.Worker(
            ["taskset", "-c", str(cpu), "./check", str(HERE / "selected-points.dat")],
            directory, REPORT / "run.log", env=env)
        worker.run(1)
        result["parsed"] = probe.workers.parse_outputs(directory, 8, 4, 1)
        reference = (HERE / "production/block-0/values.dat").read_bytes()
        assert (directory / "values.dat").read_bytes() == reference
        for name in (*probe.workers.OUTPUTS, "dimensions.dat"):
            shutil.copy2(directory / name, REPORT / name)
        worker.close()
        worker = None
        shapes = {}
        successes = 0
        count = 0
        for line in capture.read_text().splitlines():
            row = json.loads(line)
            key = f"{row['n']}/{row['rank']}"
            shapes[key] = shapes.get(key, 0) + 1
            successes += row["status"] == 0
            count += 1
        with capture.open("rb") as source, gzip.open(REPORT / "actual-mp-requests.jsonl.gz", "wb") as target:
            shutil.copyfileobj(source, target)
        result.update(status="complete", captured_requests=count,
                      successful_requests=successes, shapes=shapes,
                      capture_sha256=probe.sha(capture),
                      compressed_capture_sha256=probe.sha(REPORT / "actual-mp-requests.jsonl.gz"),
                      printed_process_rows_unchanged=True)
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if worker is not None:
            worker.close()
        probe.dump(REPORT / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.cpu)
