#!/usr/bin/env python3
"""Isolated tensor-plan diagnostic; numerical protocol sources remain frozen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/generic-tensor-plan"
SOURCE = ROOT / ".benchmark-work/outputs/candidate/qcd-3"
POINT_IDS = [1, 44, 67, 80]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import ninjars_benchmark_workers as workers
from madgraph.various.banner import MadLoopParam


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def proc(output):
    return next(output.glob("SubProcesses/P*_*"))


def prepare():
    HERE.mkdir(parents=True, exist_ok=True)
    prefix = WORK / "install/lib"
    prefix.mkdir(parents=True, exist_ok=True)
    archive = WORK / "target/release/libmadloop_ninjars_tensor_plan_prototype.a"
    if not archive.exists():
        raise RuntimeError("Build the isolated prototype static archive first")
    shutil.copy2(archive, prefix / "libninjars.a")
    # Cargo builds Rust only. Reuse the exact installed Fortran module object.
    with (prefix / "mninjars.o").open("wb") as stream:
        subprocess.run(["ar", "p", str(ROOT / "HEPTools/ninjars/lib/libninjars.a"),
                        "mninjars.o"], stdout=stream, check=True)
    subprocess.run(["ar", "rcs", str(prefix / "libninjars.a"),
                    str(prefix / "mninjars.o")], check=True)
    full_points = ROOT / "docs/benchmarkfolder/points/qcd-3.dat"
    lines = full_points.read_text().splitlines()
    assert int(lines[0]) == 100 and len(lines) == 501
    points = HERE / "selected-points.dat"
    points.write_text("4\n" + "\n".join(
        line for index in POINT_IDS
        for line in lines[1 + 5 * (index - 1):1 + 5 * index]) + "\n")
    metadata = {
        "acceptance_eligible": False,
        "purpose": "Isolated exploratory exact parity and full warmed evaluation timing",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_compilation": json.loads((SOURCE / "compilation.json").read_text()),
        "full_points_sha256": sha(full_points),
        "points_sha256": sha(points), "point_indices": POINT_IDS,
        "protocol": workers.protocol_descriptor(), "variants": {},
    }
    for variant in ("production", "planned"):
        output = WORK / f"qcd3-{variant}"
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite {output}")
        subprocess.run(["cp", "-a", "--reflink=auto", str(SOURCE), str(output)], check=True)
        for link in output.rglob("*"):
            if link.is_symlink() and link.resolve().is_relative_to(SOURCE):
                replacement = output / link.resolve().relative_to(SOURCE)
                link.unlink()
                link.symlink_to(os.path.relpath(replacement, link.parent))
        directory = proc(output)
        loop = directory / "loop_matrix.f"
        match = re.search(r"SUBROUTINE (\w*)SLOOPMATRIX_THRES\(", loop.read_text())
        prefix_name = match.group(1)
        driver = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        driver = driver.replace("@PREFIX@", prefix_name).replace(
            "@NEXTERNAL@", "5").replace("@NBORN@", "1")
        driver_path = directory / "check_sa.f"
        driver_path.unlink()
        driver_path.write_text(driver)
        definitions = output / "SubProcesses/MadLoop_makefile_definitions"
        native = ROOT / "HEPTools/ninjars/lib/libninjars.a"
        if variant == "planned":
            text = definitions.read_text()
            old = str(ROOT / "HEPTools/ninjars/lib")
            assert old in text
            # ABI and module are unchanged; retain the installed module include.
            lines = [line if line.startswith("LOOP_INCLUDE") else
                     line.replace(old, str(prefix)) for line in text.splitlines()]
            definitions.write_text("\n".join(lines) + "\n")
            native = prefix / "libninjars.a"
        card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
        card["MLReductionLib"] = "8"
        card["CTModeRun"] = -1
        card["CTModeInit"] = 1
        card["MLStabThres"] = 1e-9
        card["UseQPIntegrandForNinjaRS"] = True
        card["UseLoopFilter"] = False
        for path in (output / "Cards/MadLoopParams.dat",
                     *output.glob("SubProcesses/**/MadLoopParams.dat")):
            if not path.resolve().is_relative_to(output):
                raise RuntimeError(f"Card points outside isolated output: {path}")
            card.write(str(path), template=str(
                ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        for name in ("check_sa.o", "check"):
            (directory / name).unlink(missing_ok=True)
        report = HERE / variant
        report.mkdir()
        with (report / "build.log").open("w") as stream:
            subprocess.run(["make", "-j2", "check", "SHELL=" + shutil.which("bash")],
                           cwd=directory, env=workers.environment(WORK),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        workers.validate_param_card(directory, (output / "Cards/param_card.dat").read_bytes())
        assert (output / "Cards/param_card.dat").read_bytes() == (
            SOURCE / "Cards/param_card.dat").read_bytes()
        for path in (driver_path, definitions, output / "Cards/param_card.dat",
                     output / "Cards/MadLoopParams.dat"):
            shutil.copy2(path, report / path.name)
        metadata["variants"][variant] = {
            "output": str(output.relative_to(ROOT)),
            "executable_sha256": sha(directory / "check"),
            "library_sha256": sha(native),
            "driver_sha256": sha(driver_path),
            "loop_matrix_sha256": sha(loop),
            "parameter_card_sha256": sha(output / "Cards/param_card.dat"),
            "madloop_card_sha256": sha(output / "Cards/MadLoopParams.dat"),
            "generated_sources": workers.bench.generated_source_manifest(output),
        }
    dump(HERE / "build.json", metadata)


def run(cpu, repetitions):
    if repetitions < 3:
        raise ValueError("At least three alternating blocks are required")
    build = json.loads((HERE / "build.json").read_text())
    instances = {}
    result = {"acceptance_eligible": False, "status": "running", "cpu": cpu,
              "build_sha256": sha(HERE / "build.json"), "repetitions": repetitions,
              "point_indices": POINT_IDS, "measurements": [],
              "timing_scope": "Full warmed evaluation; exploratory concurrent-host conditions",
              "run_policy": "Alternating fresh workers; initialize and warm outside timing; "
                            "one timed block per process; only one process alive "
                            "because Symbolica retains its restricted license permit"}
    reference = None
    try:
        for repetition in range(repetitions):
            order = ("production", "planned") if repetition % 2 == 0 else ("planned", "production")
            for variant in order:
                directory = proc(WORK / f"qcd3-{variant}")
                expected = build["variants"][variant]
                assert sha(directory / "check") == expected["executable_sha256"]
                workers.validate_param_card(directory, (HERE / variant / "param_card.dat").read_bytes())
                instances[variant] = workers.Worker(
                    ["taskset", "-c", str(cpu), "./check", str(HERE / "selected-points.dat")],
                    directory, HERE / variant / f"run-{repetition}.log",
                    env=workers.environment(WORK))
                worker = instances[variant]
                worker.run(1)
                parsed = workers.parse_outputs(worker.directory, 8, 4, 1)
                raw = (worker.directory / "values.dat").read_bytes()
                if reference is None:
                    reference = raw
                if raw != reference:
                    raise RuntimeError(f"Exact printed physics/status/accuracy mismatch: {variant}/{repetition}")
                for row in parsed["rows"]:
                    expected_status = 218 if row["point"] == 1 else 322
                    if row["status"] != expected_status:
                        raise RuntimeError(f"Expected ordinary DP or genuine QP rescue: {row}")
                destination = HERE / variant / f"block-{repetition}"
                destination.mkdir()
                for name in (*workers.OUTPUTS, "dimensions.dat"):
                    shutil.copy2(worker.directory / name, destination / name)
                parsed.update(variant=variant, repetition=repetition,
                              values_sha256=sha(destination / "values.dat"))
                result["measurements"].append(parsed)
                worker.close()
                del instances[variant]
                dump(HERE / "results.json", result)
                print(variant, repetition, parsed["seconds"], parsed["status_counts"], flush=True)
        result["printed_results_identical"] = True
        result["medians"] = {
            variant: statistics.median(row["seconds"] for row in result["measurements"]
                                       if row["variant"] == variant)
            for variant in ("production", "planned")}
        result["planned_over_production"] = result["medians"]["planned"] / result["medians"]["production"]
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        errors = []
        for variant, instance in instances.items():
            try:
                instance.close()
            except BaseException as error:
                errors.append(f"{variant}: {error}")
        if errors:
            result.update(status="failed", shutdown_errors=errors)
        dump(HERE / "results.json", result)
        if errors:
            raise RuntimeError(str(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.cpu, args.repetitions)
