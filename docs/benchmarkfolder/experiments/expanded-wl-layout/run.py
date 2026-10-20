#!/usr/bin/env python3
"""Test contiguous expanded UPDATE_WL output traversal in isolated copies."""
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
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam

HERE = ROOT / ".benchmark-work/experiments/expanded-wl-layout"
REPORT = Path(__file__).resolve().parent


def interchange(source):
    pattern = (
        r"      DO I=1,LCUT_SIZE\n"
        r"        DO J=1,OUT_SIZE\n"
        r"          DO K=0,(\d+)\n"
        r"            OUT\(J,K,I\)=([^\n]*)\n"
        r"          ENDDO\n"
        r"          DO K=1,IN_SIZE\n"
        r"(.*?)"
        r"          ENDDO\n"
        r"        ENDDO\n"
        r"      ENDDO")
    def replace(match):
        return (
            "      DO I=1,LCUT_SIZE\n"
            f"        DO K=0,{match[1]}\n"
            "          DO J=1,OUT_SIZE\n"
            f"            OUT(J,K,I)={match[2]}\n"
            "          ENDDO\n"
            "        ENDDO\n"
            "        DO K=1,IN_SIZE\n"
            "          DO J=1,OUT_SIZE\n" +
            match[3] +
            "          ENDDO\n"
            "        ENDDO\n"
            "      ENDDO")
    text, count = re.subn(pattern, replace, source, flags=re.S)
    if not count:
        raise RuntimeError("No expanded UPDATE_WL routines matched")
    return text, count


def choose_source(case, source_root=None):
    """Use an unchanged pre-expansion output even after core regeneration."""
    candidates = ([source_root / case] if source_root is not None else [
        HERE / case / "original",
        bench.WORK / "outputs/pre-final-candidate" / case,
        bench.WORK / "outputs/candidate" / case,
    ])
    for source in candidates:
        files = list(source.glob("SubProcesses/P*_*/polynomial.f"))
        if len(files) != 1:
            continue
        text = files[0].read_text()
        try:
            interchange(text)
        except RuntimeError:
            continue
        if re.search(r"DO L=0,\d+\s+NEW_POSITION = COMB_COEF_POS\(L,M\)"
                     r"\s+DO I=1,LCUT_SIZE", text):
            raise RuntimeError("Source lacks the compact optimization shared by both variants")
        return source.resolve(), files[0]
    raise RuntimeError(
        f"No retained pre-expansion source for {case}; use --source-root with "
        "the archived candidate outputs or preserve the original experiment copy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="ew-2,qcd-2")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--source-root", type=Path,
                        help="Parent containing pre-expansion case outputs")
    parser.add_argument("--work-dir", type=Path, default=HERE,
                        help="New experiment output directory; existing copies are never reused")
    parser.add_argument("--report-dir", type=Path, default=REPORT,
                        help="New evidence directory; existing evidence is never overwritten")
    args = parser.parse_args()
    args.work_dir = args.work_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MAKEFLAGS="SHELL=" + shutil.which("bash"))
    libraries = [bench.BASELINE / "HEPTools/ninja/lib",
                 bench.BASELINE / "HEPTools/collier"]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries))
    for case in args.cases.split(","):
        if (args.report_dir / (case + "-evidence.json")).exists():
            raise RuntimeError("Existing evidence is retained; select a new --report-dir")
        source, source_polynomial = choose_source(case, args.source_root)
        source_snapshot = {
            "path": str(source), "selection": "Retained candidate with original expanded traversal",
            "polynomial_sha256": hashlib.sha256(source_polynomial.read_bytes()).hexdigest(),
            "generated_sources_sha256": bench.generated_source_manifest(source)["files_sha256"],
            "recording": "Recorded before creating either experimental copy",
        }
        points = bench.REPORT / "points" / (case + ".dat")
        jobs, builds = {}, {}
        for name in ("original", "interchange"):
            output = args.work_dir / case / name
            if output.exists():
                raise RuntimeError(f"Refusing to reuse an existing experiment: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(output)],
                           check=True)
            for path in output.rglob("*"):
                if path.is_symlink() and path.resolve().is_relative_to(source):
                    target = output / path.resolve().relative_to(source)
                    path.unlink()
                    path.symlink_to(os.path.relpath(target, path.parent))
            directory = next(output.glob("SubProcesses/P*_*"))
            jobs[name] = output, directory
            card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
            card["MLReductionLib"] = "6"
            card["MLStabThres"] = 1e-9
            card["COLLIERRequiredAccuracy"] = 1e-10
            for path in [output / "Cards/MadLoopParams.dat",
                         *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
                card.write(str(path), template=str(
                    ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
            if name == "interchange":
                path = directory / "polynomial.f"
                before = path.read_text()
                after, count = interchange(before)
                path.write_text(after)
                (args.report_dir / (case + "-generated.patch")).write_text(
                    "".join(difflib.unified_diff(before.splitlines(True),
                                                after.splitlines(True),
                                                fromfile="original/polynomial.f",
                                                tofile="interchange/polynomial.f")))
                print(case, "modified routines", count, flush=True)
            with (args.work_dir / case / ("build-" + name + ".log")).open("w") as log:
                subprocess.run(["make", "-j4", "check"], cwd=directory,
                               env=environment, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
            manifest = bench.generated_source_manifest(output)
            bench.dump(args.report_dir / f"{case}-{name}-compiled-sources.json", manifest)
            builds[name] = {
                "executable_sha256": hashlib.sha256((directory / "check").read_bytes()).hexdigest(),
                "generated_sources_sha256": manifest["files_sha256"],
            }
        runs = {name: [] for name in jobs}
        expected = None
        for rep in range(args.repetitions):
            order = list(jobs) if rep % 2 == 0 else list(reversed(jobs))
            for name in order:
                output, directory = jobs[name]
                for path in (output / "SubProcesses").rglob("*HelFilter.dat"):
                    path.unlink()
                with (args.work_dir / case / f"run-{name}-{rep}.log").open("w") as log:
                    subprocess.run(["taskset", "-c", str(args.cpu), "./check",
                                    str(points), "30"], cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                cold, elapsed = map(float, (directory / "timing.dat").read_text().split())
                values = (directory / "values.dat").read_bytes()
                counts = dict(map(int, line.split()) for line in
                              (directory / "timed_status.dat").read_text().splitlines())
                if counts != {216: 3000}:
                    raise RuntimeError(f"Unexpected statuses: {counts}")
                if expected is None:
                    expected = values
                if values != expected:
                    (args.report_dir / f"{case}-mismatch-{name}-{rep}.dat").write_bytes(values)
                    raise RuntimeError("Numerical rows, accuracies or statuses changed")
                runs[name].append({"seconds_per_point": elapsed/3000,
                                   "cold_seconds": cold, "loops": 30,
                                   "status_counts": counts})
                print(case, name, rep, elapsed/3000, flush=True)
        medians = {name: statistics.median(x["seconds_per_point"] for x in samples)
                   for name, samples in runs.items()}
        result = {
            "case": case, "cpu": args.cpu, "backend": 6,
            "exploratory_only": True, "concurrent_task_profiling": True,
            "shared_numerical_settings": {"MLStabThres": 1e-9,
                                          "COLLIERRequiredAccuracy": 1e-10},
            "measurements": runs, "median_seconds_per_point": medians,
            "interchange_over_original": medians["interchange"]/medians["original"],
            "all_100_points_bitwise_identical_in_printed_output": True,
            "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
            "source_snapshot": source_snapshot, "builds": builds,
            "experiment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        bench.dump(args.report_dir / (case + "-evidence.json"), result)
        print(case, result["interchange_over_original"], flush=True)


if __name__ == "__main__":
    main()
