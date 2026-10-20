#!/usr/bin/env python3
"""Exploratory full-process zero-skip trials in isolated generated copies."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
EXPERIMENT = "expanded-wl-sparsity"
RUNNER_DEPENDENCIES = []
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(HERE)]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam
from prototype import transform


def modify_variant(directory, variant):
    path = directory / "polynomial.f"
    after = path.read_text()
    if variant != "original":
        for mode in (["expanded", "compact"] if variant == "combined" else [variant]):
            after, _ = transform(after, mode)
    return {"polynomial.f": after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="qcd-3")
    parser.add_argument("--variants", default="compact")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--quiet", action="store_true",
                        help="Record an explicitly coordinated quiet measurement window")
    parser.add_argument("--maximum-ratio", type=float, default=1.0)
    parser.add_argument("--reference-values", type=Path,
                        help="Require every run to reproduce retained original values verbatim")
    parser.add_argument("--reference-status", type=Path,
                        help="Require every timed status count to match a retained original pass")
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    work = bench.WORK / "experiments" / EXPERIMENT / args.run_id
    report = HERE / args.run_id
    report.mkdir(parents=True, exist_ok=False)
    variants = ["original", *args.variants.split(",")]
    assert len(set(variants)) == len(variants)
    source = bench.WORK / "outputs/candidate" / args.case
    points = bench.REPORT / "points" / f"{args.case}.dat"
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MAKEFLAGS="SHELL=" + shutil.which("bash"),
                       LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE / "HEPTools" / p)
                                                     for p in ("ninja/lib", "collier")))
    result = {"case": args.case, "backend": 6, "cpu": args.cpu,
              "exploratory_only": True, "acceptance_eligible": False,
              "coordinated_quiet_window": args.quiet,
              "source_manifest": bench.generated_source_manifest(source),
              "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
              "prototype_sha256": hashlib.sha256((HERE / "prototype.py").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "builds": {}, "measurements": {name: [] for name in variants}}
    for name in ("reference_values", "reference_status"):
        path = getattr(args, name)
        if path is not None:
            result[name] = {"path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    source_inputs = report / "source-inputs"
    source_inputs.mkdir()
    source_input_hashes = {}
    for relative in ("generation.json", "compilation.json", "compiled-sources.json",
                     "Cards/param_card.dat", "Cards/MadLoopParams.dat", "Source/make_opts"):
        path = source / relative
        if path.is_file():
            target = source_inputs / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            source_input_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["source_inputs_sha256"] = source_input_hashes
    snapshots = [Path(__file__), HERE / "prototype.py", *RUNNER_DEPENDENCIES]
    for index, path in enumerate(snapshots):
        shutil.copy2(path, report / f"source-{index}-{path.name}")
    result["runner_dependencies_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in snapshots}
    jobs = {}
    for variant in variants:
        output = work / variant
        output.parent.mkdir(parents=True, exist_ok=True)
        assert not output.exists()
        subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)], check=True)
        for path in output.rglob("*"):
            if path.is_symlink() and path.resolve().is_relative_to(source):
                target = output / path.resolve().relative_to(source)
                path.unlink()
                path.symlink_to(os.path.relpath(target, path.parent))
        directory = next(output.glob("SubProcesses/P*_*"))
        jobs[variant] = output, directory
        card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
        for key, value in {"MLReductionLib": "6", "CTModeRun": -1,
                           "MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-10}.items():
            card[key] = value
        for path in [output / "Cards/MadLoopParams.dat",
                     *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
            card.write(str(path), template=str(
                ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        files = modify_variant(directory, variant)
        patches = []
        for name, after in files.items():
            path = directory / name
            before = path.read_text()
            if after != before:
                if EXPERIMENT == "expanded-wl-sparsity" and name == "polynomial.f":
                    def dp_bodies(text):
                        return re.findall(r"(?m)^      SUBROUTINE (?!MP_)\w+\(.*?^      END\s*$",
                                          text, flags=re.S | re.M)
                    assert dp_bodies(before) == dp_bodies(after), "DP body changed"
                path.write_text(after)
                patches.extend(difflib.unified_diff(
                    before.splitlines(True), after.splitlines(True),
                    fromfile=f"original/{name}", tofile=f"{variant}/{name}"))
        if patches:
            (report / (variant + ".patch")).write_text("".join(patches))
        with (report / f"build-{variant}.log").open("w") as log:
            subprocess.run(["make", "-j2", "check"], cwd=directory, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result["builds"][variant] = {
            "executable_sha256": hashlib.sha256((directory / "check").read_bytes()).hexdigest(),
            "modified_sources_sha256": {
                name: hashlib.sha256(after.encode()).hexdigest() for name, after in files.items()}}
        manifest = bench.generated_source_manifest(output)
        bench.dump(report / f"{variant}-compiled-sources.json", manifest)
        result["builds"][variant]["generated_sources_sha256"] = manifest["files_sha256"]
    expected = args.reference_values.read_bytes() if args.reference_values else None
    expected_status = args.reference_status.read_bytes() if args.reference_status else None
    try:
        for rep in range(args.repetitions):
            for variant in variants if rep % 2 == 0 else list(reversed(variants)):
                output, directory = jobs[variant]
                for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
                    for path in (output / "SubProcesses").rglob(pattern):
                        path.unlink()
                with (report / f"run-{variant}-{rep}.log").open("w") as log:
                    subprocess.run(["./check", str(points), str(args.loops)], cwd=directory,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   check=True)
                for name in ("values.dat", "timing.dat", "timed_status.dat"):
                    shutil.copy2(directory / name, report / f"{variant}-{rep}-{name}")
                values = (directory / "values.dat").read_bytes()
                statuses = (directory / "timed_status.dat").read_bytes()
                if expected is None:
                    expected = values
                if expected_status is None:
                    expected_status = statuses
                if values != expected or statuses != expected_status:
                    raise RuntimeError("Printed coefficients, accuracy, row statuses or timed statuses differ")
                cold, elapsed = map(float, (directory / "timing.dat").read_text().split())
                result["measurements"][variant].append({
                    "repetition": rep, "loops": args.loops,
                    "seconds_per_point": elapsed/(100*args.loops),
                    "cold_initialization_and_warm_pass_seconds": cold,
                    "row_sha256": hashlib.sha256(values).hexdigest(),
                    "timed_status_sha256": hashlib.sha256(statuses).hexdigest()})
                print(variant, rep, elapsed/(100*args.loops), flush=True)
        medians = {v: statistics.median(r["seconds_per_point"]
                                       for r in result["measurements"][v]) for v in variants}
        result.update(complete=True, all_printed_rows_and_statuses_exact=True,
                      median_seconds_per_point=medians,
                      ratios={v: medians[v]/medians["original"] for v in variants[1:]})
        if args.repetitions >= 7:
            original = [r["seconds_per_point"] for r in result["measurements"]["original"]]
            result["paired_statistics"] = {
                v: bench.runtime_statistics(
                    original, [r["seconds_per_point"] for r in result["measurements"][v]],
                    args.maximum_ratio)
                for v in variants[1:]}
    except Exception as error:
        result.update(complete=False, error=repr(error))
        raise
    finally:
        (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["ratios"]), flush=True)


if __name__ == "__main__":
    main()
