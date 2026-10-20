#!/usr/bin/env python3
"""Profile a warmed production Zgg rescue; no prototype library or generator."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/mp-qcd3-point44"
SOURCE = ROOT / ".benchmark-work/experiments/generic-tensor-plan/qcd3-production"
PERF = "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf"
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_workers as workers

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def directory():
    return next((WORK / "qcd3").glob("SubProcesses/P*_*"))

def prepare():
    if WORK.exists():
        raise RuntimeError(f"Refusing to overwrite {WORK}")
    source_metadata = json.loads((HERE.parent / "generic-tensor-plan/build.json").read_text())
    production = source_metadata["variants"]["production"]
    source_proc = next(SOURCE.glob("SubProcesses/P*_*"))
    library = ROOT / "HEPTools/ninjars/lib/libninjars.a"
    assert digest(source_proc / "check") == production["executable_sha256"]
    assert digest(library) == production["library_sha256"]
    WORK.mkdir()
    shutil.copytree(SOURCE, WORK / "qcd3", symlinks=True)
    for link in (WORK / "qcd3").rglob("*"):
        if link.is_symlink() and link.resolve().is_relative_to(SOURCE):
            replacement = WORK / "qcd3" / link.resolve().relative_to(SOURCE)
            link.unlink()
            link.symlink_to(os.path.relpath(replacement, link.parent))
    lines = (ROOT / "docs/benchmarkfolder/points/qcd-3.dat").read_text().splitlines()
    (HERE / "points.dat").write_text("2\n" + "\n".join(
        line for index in (1, 44)
        for line in lines[1+5*(index-1):1+5*index])+"\n")
    proc = directory()
    for path in (proc / "check_sa.f", WORK / "qcd3/Cards/MadLoopParams.dat",
                 WORK / "qcd3/Cards/param_card.dat"):
        shutil.copy2(path, HERE / path.name)
    (HERE / "build.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_metadata": source_metadata,
        "executable_sha256": digest(proc / "check"),
        "library_sha256": digest(library),
        "points_sha256": digest(HERE / "points.dat"),
        "point_indices": [1, 44], "protocol": workers.protocol_descriptor(),
        "note": "Identical prebuilt production worker copied; no rebuild or numerical modification.",
    }, indent=2)+"\n")

def category(lines):
    stack = "\n".join(lines)
    owner = lambda pattern: any(re.match(r"\s*[0-9a-f]+ <"+pattern, line) for line in lines)
    if "mp_compute_loop_coefs_" in stack or "mp_coef_construction_" in stack:
        return "fortran_qp_numerator"
    if owner(r"ninjars_symbolica::masters::SymbolicaMasters"):
        return "scalar_masters"
    if owner(r"madloop_ninjars::tensor::TensorGenericNumerator"):
        return "tensor_expansion_and_contraction"
    if owner(r"ninjars_symbolica::(?:cuts|high_cuts)::"):
        return "generic_cut_geometry"
    if owner(r"ninjars::(?:boxes|triangles|bubbles|tadpoles|amplitude)::"):
        return "generic_reduction_and_bookkeeping"
    if any(name in stack for name in (
            "madloop_ninjars::parse_record", "madloop_ninjars::format_record",
            "madloop_ninjars::parse_records", "madloop_ninjars::pairs")):
        return "bridge_numeric_transport"
    if lines[0].startswith("madloop-ninjars"):
        return "other_mp_worker"
    return "other_main"

def partition():
    data = WORK / "profile.perf"
    command = [PERF, "script", "--no-inline", "-i", str(data)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    counts, leafs = Counter(), {}
    examples = {}
    block = []
    def consume(lines):
        if not lines:
            return
        kind = category(lines)
        counts[kind] += 1
        examples.setdefault(kind, lines.copy())
        leaf = re.sub(r"^\s*[0-9a-f]+ ", "", lines[1]) if len(lines) > 1 else "[missing callchain]"
        leafs.setdefault(kind, Counter())[leaf] += 1
    for line in process.stdout:
        if line.strip():
            block.append(line.rstrip())
        else:
            consume(block)
            block.clear()
    consume(block)
    if process.wait() != 0:
        raise RuntimeError("perf script failed")
    total = sum(counts.values())
    report = {
        "command": command, "samples": total,
        "categories": {key: {"samples": value, "percent": 100*value/total}
                       for key, value in counts.items()},
        "classification": "First matching actual owner/method frame; scalar, tensor and geometry take precedence over enclosing reducers. Reduction bucket includes generic algebra and allocation/bookkeeping. Unidentified worker work remains explicit.",
        "top_leaf_frames_by_category": {key: value.most_common(12) for key, value in leafs.items()},
    }
    (HERE / "perf-partition.json").write_text(json.dumps(report, indent=2)+"\n")
    (HERE / "perf-exemplars.txt").write_text("\n\n".join(
        key+"\n"+"\n".join(lines) for key, lines in examples.items())+"\n")
    with (HERE / "perf-flat.txt").open("w") as stream:
        subprocess.run([PERF, "report", "-i", str(data), "--stdio", "--no-children",
            "--no-inline", "--call-graph", "none", "--percent-limit", "0.5"],
            cwd=ROOT, stdout=stream, check=True)
    print(json.dumps(report["categories"], indent=2))

def run(cpu):
    build = json.loads((HERE / "build.json").read_text())
    proc = directory()
    assert digest(proc / "check") == build["executable_sha256"]
    command = ["taskset", "-c", str(cpu), "./check", str(HERE / "points.dat")]
    env = workers.environment(WORK)
    worker, profiler = None, None
    results = []
    reference = None
    perf_command = None
    try:
        # READY arrives only after initialization and the full two-point warmup.
        worker = workers.Worker(command, proc, HERE / "run.log", env=env)
        perf_command = [PERF, "record", "--no-buildid", "-e", "cpu-clock:u", "-F", "99",
                        "--call-graph", "dwarf,65528", "-p", str(worker.process.pid),
                        "-o", str(WORK / "profile.perf")]
        with (HERE / "perf-record.log").open("w") as stream:
            profiler = subprocess.Popen(perf_command, cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            time.sleep(.25)
            for block in range(3):
                worker.run(1)
                parsed = workers.parse_outputs(proc, 8, 2, 1)
                raw = (proc / "values.dat").read_bytes()
                if reference is None:
                    reference = raw
                assert raw == reference, "Warm repeated outputs changed"
                for row in parsed["rows"]:
                    assert row["status"] == (218 if row["point"] == 1 else 322)
                output = HERE / f"block-{block}"
                output.mkdir()
                for name in (*workers.OUTPUTS, "dimensions.dat"):
                    shutil.copy2(proc / name, output / name)
                results.append(parsed)
            profiler.send_signal(signal.SIGINT)
            code = profiler.wait()
            profiler = None
            if code not in (0, -signal.SIGINT):
                raise RuntimeError(f"perf exited {code}")
        worker.close()
        worker = None
    finally:
        if profiler is not None:
            profiler.send_signal(signal.SIGINT)
            profiler.wait()
        if worker is not None:
            worker.close(graceful=False)
    (HERE / "run.json").write_text(json.dumps({
        "command": command, "perf_command": perf_command, "cpu": cpu,
        "blocks": results, "exact_repeated_rows": True,
        "scope": "Three warmed point44 rescues, each preceded by stable DP point1; profiler overhead means diagnostic only.",
        "build_sha256": digest(HERE / "build.json"),
    }, indent=2)+"\n")
    partition()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "partition"))
    parser.add_argument("--cpu", type=int, default=207)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.cpu)
    else:
        partition()
