#!/usr/bin/env python3
"""Isolated full-evaluation comparison; never constitutes a final benchmark gate."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/t3-compensation-performance"
SOURCES = ROOT / ".benchmark-work/policy-v2/generation/outputs/candidate"
GUARDED = ROOT / ".benchmark-work/experiments/qcd3-point67-t3/guarded-bridge/libninjars.a"
NATIVE = ROOT / "HEPTools/ninjars/lib"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    result = os.environ.copy()
    result.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MAKEFLAGS="SHELL=" + shutil.which("bash"),
        SYMBOLICA_LOCK_DIR=str(ROOT / ".ninjars-build/symbolica-locks"),
        LD_LIBRARY_PATH=os.pathsep.join(str(bench.BASELINE / "HEPTools" / p)
                                       for p in ("ninja/lib", "collier")),
    )
    return result


def command(args, cwd, log):
    start = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run(args, cwd=cwd, env=environment(),
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return time.perf_counter() - start


def prepare(args):
    report, work = HERE / args.run_id, WORK / args.run_id
    report.mkdir(exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, report / "runner.py")
    metadata = {"source_scope": "Existing fd72 corrected-model CLI exports; "
                "isolated library substitution only. Not current-source acceptance.",
                "acceptance_eligible": False, "runner_sha256": sha(Path(__file__)),
                "cases": {}, "archives": {}, "libraries": {}}
    for variant in ("native", "guarded"):
        library = work / "libraries" / variant
        library.mkdir(parents=True)
        archive = NATIVE / "libninjars.a" if variant == "native" else GUARDED
        before = sha(archive)
        shutil.copy2(archive, library / "libninjars.a")
        assert sha(library / "libninjars.a") == before == sha(archive)
        shutil.copy2(NATIVE / "mninjars.mod", library / "mninjars.mod")
        metadata["archives"][variant] = before
        metadata["libraries"][variant] = str(library.relative_to(ROOT))
    for case in args.cases.split(","):
        source = SOURCES / case
        points = ROOT / "docs/benchmarkfolder/points" / (case + ".dat")
        retained = report / case / "source-inputs"
        retained.mkdir(parents=True)
        inputs = {}
        for name in ("generation.json", "compilation.json", "compiled-sources.json",
                     "Cards/param_card.dat", "Cards/MadLoopParams.dat", "Source/make_opts"):
            path = source / name
            destination = retained / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            inputs[name] = sha(path)
        item = {"source": str(source.relative_to(ROOT)), "inputs": inputs,
                "points": str(points.relative_to(ROOT)), "points_sha256": sha(points),
                "variants": {}}
        for variant in ("native", "guarded"):
            output = work / case / variant
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)],
                           check=True)
            for path in output.rglob("*"):
                if path.is_symlink() and path.resolve().is_relative_to(source):
                    target = output / path.resolve().relative_to(source)
                    path.unlink()
                    path.symlink_to(os.path.relpath(target, path.parent))
            directory = next(output.glob("SubProcesses/P*_*"))
            definitions = output / "SubProcesses/MadLoop_makefile_definitions"
            text = definitions.read_text()
            assert str(NATIVE) in text
            definitions.write_text(text.replace(
                str(NATIVE), str(ROOT / metadata["libraries"][variant])))
            card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
            for key, value in {"MLReductionLib": "8", "CTModeRun": -1,
                               "MLStabThres": 1e-9,
                               "COLLIERRequiredAccuracy": 1e-12}.items():
                card[key] = value
            for path in [output / "Cards/MadLoopParams.dat",
                         *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
                card.write(str(path), template=str(
                    ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
            # Reuse numerators compiled by the clean CLI generation; relink both
            # variants against the same immutable driver, model and objects.
            (directory / "check").unlink()
            build_time = command(["make", "-j2", "check"], directory,
                                 report / case / f"build-{variant}.log")
            shutil.copy2(definitions, report / case / f"{variant}-link-definitions")
            shutil.copy2(output / "Cards/MadLoopParams.dat",
                         report / case / f"{variant}-MadLoopParams.dat")
            manifest = bench.generated_source_manifest(output)
            dump(report / case / f"{variant}-generated-sources.json", manifest)
            item["variants"][variant] = {
                "directory": str(directory.relative_to(ROOT)),
                "executable_sha256": sha(directory / "check"),
                "card_sha256": sha(output / "Cards/MadLoopParams.dat"),
                "parameter_card_sha256": sha(output / "Cards/param_card.dat"),
                "source_sha256": manifest["files_sha256"],
                "build_seconds": build_time,
            }
        metadata["cases"][case] = item
        dump(report / "preparation.json", metadata)
        print("Prepared", case, flush=True)


def parse(directory, loops):
    rows = []
    for line in (directory / "values.dat").read_text().splitlines():
        words = line.split()
        assert len(words) == 8
        row = {"point": int(words[0]), "order": int(words[1]),
               "status": int(words[2]), "coefficients": list(map(float, words[3:7])),
               "accuracy": float(words[7])}
        bench.validate_status(row["status"], 8)
        assert all(math.isfinite(x) for x in row["coefficients"] + [row["accuracy"]])
        assert row["accuracy"] >= 0
        rows.append(row)
    allocated, loop_orders, born_orders = map(
        int, (directory / "dimensions.dat").read_text().split())
    assert allocated == max(loop_orders, born_orders) and born_orders <= loop_orders
    assert [(r["point"], r["order"]) for r in rows] == [
        (p, o) for p in range(1, 101) for o in range(loop_orders + 1)]
    counts = {}
    for line in (directory / "timed_status.dat").read_text().splitlines():
        status, count = map(int, line.split())
        bench.validate_status(status, 8)
        assert count > 0 and str(status) not in counts
        counts[str(status)] = count
    row_counts = Counter(str(r["status"]) for r in rows if r["order"] == 0)
    assert counts == {k: v * loops for k, v in row_counts.items()}
    cold, elapsed = map(float, (directory / "timing.dat").read_text().split())
    assert cold > 0 and elapsed > 0 and math.isfinite(cold + elapsed)
    return {"rows": rows, "cold_seconds": cold, "seconds": elapsed,
            "seconds_per_point": elapsed / (100 * loops),
            "status_counts": dict(row_counts), "timed_status_counts": counts}


def agreement(left, right):
    assert len(left) == len(right)
    scales = {}
    for row in left + right:
        scales[row["point"]] = max(scales.get(row["point"], 0),
                                   *(abs(x) for x in row["coefficients"]))
    failures, worst = [], 0.
    for a, b in zip(left, right):
        assert (a["point"], a["order"]) == (b["point"], b["order"])
        for k, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
            tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a["point"]]
            ratio = abs(x-y) / tolerance if tolerance else 0.
            worst = max(worst, ratio)
            if abs(x-y) > tolerance:
                failures.append({"point": a["point"], "order": a["order"],
                                 "coefficient": k, "values": [x, y],
                                 "tolerance": tolerance})
    return {"passed": not failures, "worst_tolerance_fraction": worst,
            "failures": failures}


def run(args):
    report = HERE / args.run_id
    setup = json.loads((report / "preparation.json").read_text())
    target = report / args.case / args.series
    target.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    metadata = setup["cases"][args.case]
    points = ROOT / metadata["points"]
    assert sha(points) == metadata["points_sha256"]
    result = {"acceptance_eligible": False, "case": args.case, "cpu": args.cpu,
              "quiet_window": args.quiet, "loops": args.loops,
              "preparation_sha256": sha(report / "preparation.json"),
              "runner_sha256": sha(Path(__file__)), "measurements": {},
              "complete": False}
    for variant, expected in setup["archives"].items():
        assert sha(ROOT / setup["libraries"][variant] / "libninjars.a") == expected
    try:
        for rep in range(args.repetitions):
            for variant in (("native", "guarded") if rep % 2 == 0
                            else ("guarded", "native")):
                identity = metadata["variants"][variant]
                directory = ROOT / identity["directory"]
                assert sha(directory / "check") == identity["executable_sha256"]
                assert sha(directory.parent.parent / "Cards/param_card.dat") == identity["parameter_card_sha256"]
                assert sha(directory.parent.parent / "Cards/MadLoopParams.dat") == identity["card_sha256"]
                for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
                    for path in directory.parent.rglob(pattern):
                        path.unlink()
                prefix = f"{variant}-{rep}"
                wall = command(["./check", str(points), str(args.loops)], directory,
                               target / (prefix + ".log"))
                for name in ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat"):
                    shutil.copy2(directory / name, target / (prefix + "-" + name))
                measured = parse(directory, args.loops)
                measured.update(repetition=rep, process_wall_seconds=wall)
                previous = result["measurements"].setdefault(variant, [])
                if previous:
                    assert measured["rows"] == previous[0]["rows"]
                previous.append(measured)
                print(args.case, variant, rep, measured["seconds_per_point"],
                      measured["status_counts"], flush=True)
            result["agreement"] = agreement(
                result["measurements"]["native"][0]["rows"],
                result["measurements"]["guarded"][0]["rows"])
            if not result["agreement"]["passed"]:
                raise RuntimeError("Full-evaluation coefficients disagree")
        times = {v: [r["seconds_per_point"] for r in rows]
                 for v, rows in result["measurements"].items()}
        result["ratio"] = statistics.median(times["guarded"]) / statistics.median(times["native"])
        if args.repetitions >= 7:
            result["statistics"] = bench.runtime_statistics(times["native"], times["guarded"], 1.)
        result["complete"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        dump(target / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases", default="qcd-1,ew-2,qcd-3")
    parser.add_argument("--case", default="qcd-1")
    parser.add_argument("--series", default="exploratory")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else run(args)
