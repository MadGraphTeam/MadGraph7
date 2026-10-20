#!/usr/bin/env python3
"""Frozen-input loop-induced and threshold supplement, outside the core harness.

The preparation commands do not evaluate a loop amplitude.  Collection reuses
the unchanged persistent-worker protocol; its explicit launch flag is intended
for orchestration after the final numerical source/runtime freeze.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references
import ninjars_benchmark_workers as workers

ROOT = bench.ROOT
PREREG = ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-plan-v1"
DEFAULT_REPORT = ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-generation-v1"
DEFAULT_WORK = ROOT / ".benchmark-work/policy-v2/final-supplement-generation-v1"
SCHEMA = "madloop-final-supplement-v1"
MODEL_FILES = policy.MODEL_FILES
ROLES = {"original": (1, 6, 7), "candidate_corrected": (1, 6, 7, 8),
         "original_helper_corrected": (1, 6, 7)}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    path = policy.path(path)
    return {"path": policy.relative(path), "sha256": digest(path),
            "bytes": path.stat().st_size}


def checked(record, inputs=None):
    path = policy.path(record["path"])
    if digest(path) != record["sha256"] or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError("Changed supplement evidence: " + str(path))
    if inputs is not None:
        inputs[policy.relative(path)] = record["sha256"]
    return path


def frozen_configuration():
    manifest = read(PREREG / "manifest.json")
    for name, item in manifest["files"].items():
        if digest(PREREG / name) != item["sha256"] or (PREREG / name).stat().st_size != item["bytes"]:
            raise ValueError("Changed supplement preregistration")
    plan = read(PREREG / "configuration.json")
    if digest(ROOT / plan["policy"]["source_configuration"]) != plan["policy"]["source_configuration_sha256"]:
        raise ValueError("Changed control policy")
    return plan


def loop_configuration():
    plan = frozen_configuration()
    return {"schema_version": 1, "seed": plan["loop_induced"]["seed"],
            "points_per_case": plan["loop_induced"]["points_per_case"],
            "repetitions": plan["loop_induced"]["repetitions"],
            "energy_gev": plan["loop_induced"]["energy_gev"],
            "relative_tolerance": plan["policy"]["relative_tolerance"],
            "scale_tolerance": plan["policy"]["scale_tolerance"],
            "max_runtime_ratio": plan["loop_induced"]["max_runtime_ratio"],
            "madloop_parameters": plan["policy"]["madloop_parameters"],
            "cases": [{k: case[k] for k in ("id", "model", "process", "nfinal")}
                      for case in plan["loop_induced"]["cases"]]}


def validate_configuration(config, threshold=False):
    expected = loop_configuration()
    if threshold:
        expected["points_per_case"] = 12
        expected["repetitions"] = 2
        expected["cases"] = [{**expected["cases"][1], "id": "supplement-gg-hh-threshold"}]
    if config != expected:
        raise ValueError("Supplement input configuration differs from fixed plan")


@contextmanager
def harness(work, report):
    old_work, old_report = bench.WORK, bench.REPORT
    bench.WORK, bench.REPORT = Path(work), Path(report)
    try:
        yield
    finally:
        bench.WORK, bench.REPORT = old_work, old_report


def preparation(report=DEFAULT_REPORT, work=DEFAULT_WORK):
    report, work = policy.path(report), policy.path(work)
    if report.exists() or work.exists():
        raise RuntimeError("Preparation destination exists; use a new namespace")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    (report / "sources").mkdir()
    shutil.copy2(Path(__file__), report / "sources/preparation.py")
    config = loop_configuration()
    bench.dump(report / "cases.json", config)
    bench.dump(report / "preparation.json", {
        "schema": "supplement-preparation-v1", "configuration": bound(report / "cases.json"),
        "preregistration": bound(PREREG / "configuration.json"),
        "baseline_revision": bench.BASELINE_REVISION,
        "script": bound(report / "sources/preparation.py"), "work": policy.relative(work),
        "report": policy.relative(report), "final_candidate_bound": False,
        "loop_evaluations": 0, "initialized_symbolica_workers": 0})
    return report


def baseline_integrity():
    """Verify each baseline archive source against its literal Git blob."""
    command = ["git", "ls-tree", "-rz", "--full-tree", bench.BASELINE_REVISION]
    entries = subprocess.check_output(command, cwd=ROOT).split(b"\0")
    count, tree = 0, {}
    # Batch object bytes avoid one subprocess per source file.
    objects = []
    for entry in filter(None, entries):
        header, name = entry.split(b"\t", 1)
        mode, kind, sha = header.split()
        if kind == b"blob":
            objects.append((mode, sha.decode(), name.decode()))
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for mode, sha, name in objects:
            process.stdin.write((sha + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if header[:2] != [sha.encode(), b"blob"]:
                raise ValueError("Invalid original Git archive object")
            raw = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b"\n":
                raise ValueError("Malformed Git object stream")
            path = bench.BASELINE / name
            actual = os.readlink(path).encode() if mode == b"120000" else path.read_bytes()
            # Historical original-source host compatibility changes are
            # independently recorded in provenance, never silently ignored.
            if actual != raw:
                raise ValueError("Literal baseline archive changed: " + name)
            tree[name] = hashlib.sha256(raw).hexdigest()
            count += 1
    finally:
        process.stdin.close()
        process.wait()
    return {"revision": bench.BASELINE_REVISION, "files": count,
            "files_sha256": bench.content_digest(tree)}


def point_metadata(path, masses, nfinal):
    raw = Path(path).read_bytes()
    lines = raw.decode().splitlines()
    count = int(lines[0])
    values = [list(map(float, line.split())) for line in lines[1:]]
    nexternal = nfinal + 2
    if len(values) != count * nexternal or len(masses) != nexternal or any(len(p) != 4 for p in values):
        raise ValueError("Invalid supplemental momentum stream")
    events = [values[i*nexternal:(i+1)*nexternal] for i in range(count)]
    residual = shell = 0.0
    invariants = []
    for event in events:
        scale = max(p[0] for p in event)
        if scale <= 0 or any(not math.isfinite(x) for p in event for x in p):
            raise ValueError("Nonphysical supplemental point")
        r = max(abs(sum(p[j] for p in event[:2])-sum(p[j] for p in event[2:])) for j in range(4)) / scale
        s = max(abs(p[0]**2-sum(x*x for x in p[1:])-m*m) for p, m in zip(event, masses)) / scale**2
        if r > 1e-10 or s > 1e-10:
            raise ValueError("Off-shell or nonconserving supplemental point")
        residual, shell = max(residual, r), max(shell, s)
        total = [event[0][j]+event[1][j] for j in range(4)]
        invariants.append(total[0]**2-sum(x*x for x in total[1:]))
    return {"points": count, "external_masses": masses, "sha256": hashlib.sha256(raw).hexdigest(),
            "serialized_distinct_configurations": len({tuple(tuple(p) for p in e) for e in events}),
            "physical_distinct_invariant_s": 1 if nfinal == 1 else None,
            "serialized_invariant_s_min": min(invariants), "serialized_invariant_s_max": max(invariants),
            "invariant_interpretation": "2->1 has one constrained physical s; storage roundoff is not new phase space" if nfinal == 1 else "RAMBO orientations at fixed center-of-mass energy",
            "max_relative_conservation_residual": residual,
            "max_relative_mass_shell_residual": shell,
            "storage": "binary64 values serialized as 17 digits after decimal in exponential notation"}


def threshold_points(top_mass, higgs_mass):
    plan = frozen_configuration()["gg_hh_threshold"]
    if not all(math.isfinite(m) and m > 0 for m in (top_mass, higgs_mass)):
        raise ValueError("Invalid final-card top/Higgs masses")
    points, grid = [], []
    for delta in plan["delta"]:
        energy = 2 * top_mass * (1 + float(delta))
        if energy <= 2 * higgs_mass:
            raise ValueError("Fixed threshold plan is incompatible with the physical card")
        e = energy/2
        p = math.sqrt(e*e-higgs_mass*higgs_mass)
        for cosine in plan["cos_theta"]:
            c = float(cosine)
            x, z = p*math.sqrt(1-c*c), p*c
            points.append([[e, 0., 0., e], [e, 0., 0., -e],
                           [e, x, 0., z], [e, -x, 0., -z]])
            grid.append({"point": len(points), "delta": delta, "cos_theta": cosine,
                         "sqrt_s_gev": energy, "azimuth": "0"})
    return points, grid


def generate(source, report=DEFAULT_REPORT, work=DEFAULT_WORK):
    report, work = policy.path(report), policy.path(work)
    config = read(report / "cases.json")
    validate_configuration(config)
    if source not in ("baseline", "candidate"):
        raise ValueError("Invalid generator role")
    # Candidate is intentionally deferred until root declares source/runtime
    # final; the ordinary bench capture binds exact clean numerical inputs.
    if source == "candidate":
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *bench.GENERATOR_PATHS, "vendor/ninjars"], cwd=ROOT, check=True)
    status_path = report / (source + "-generation.json")
    if status_path.exists():
        raise RuntimeError("Do not overwrite previous supplemental generation")
    status = {"status": "started", "source": source, "cases": {},
              "loop_evaluations": 0, "initialized_symbolica_workers": 0,
              "affinity": sorted(os.sched_getaffinity(0))}
    bench.dump(status_path, status)
    try:
        with harness(work, report):
            for case in config["cases"]:
                bench.generate(source, case)
                masses = bench.build(source, case)
                output = bench.output_dir(source, case)
                if source == "baseline":
                    points = bench.make_points(case, masses, config)
                    bench.dump(points.with_suffix(".physical.json"), point_metadata(points, masses, case["nfinal"]))
                else:
                    points = report / "points" / (case["id"] + ".dat")
                    if read(points.with_suffix(".json"))["masses"] != masses:
                        raise ValueError("Candidate external masses differ from original")
                    baseline = bench.output_dir("baseline", case)
                    if (baseline / "Cards/param_card.dat").read_bytes() != (output / "Cards/param_card.dat").read_bytes():
                        raise ValueError("Candidate physical card differs from literal original")
                manifest = bench.generated_source_manifest(output)
                target = report / "manifests" / f"{case['id']}-{source}.json"
                bench.dump(target, manifest)
                status["cases"][case["id"]] = {
                    "output": policy.relative(output), "generated_manifest": bound(target),
                    "generation": bound(output / "generation.json"),
                    "compilation": bound(output / "compilation.json"), "masses": masses,
                    "point_stream": bound(points), "physical_metadata": bound(points.with_suffix(".physical.json"))}
                bench.dump(status_path, status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(status_path, status)
        raise
    status["status"] = "complete"
    bench.dump(status_path, status)
    return status


def prepare_threshold(report=DEFAULT_REPORT, work=DEFAULT_WORK):
    from madgraph.models.check_param_card import ParamCard
    report, work = policy.path(report), policy.path(work)
    config = read(report / "cases.json")
    case = config["cases"][1]
    output = work / "outputs/baseline" / case["id"]
    card = output / "Cards/param_card.dat"
    param = ParamCard(str(card))
    top, higgs = param["mass"][6].value, param["mass"][25].value
    events, grid = threshold_points(float(top), float(higgs))
    path = report / "points/supplement-gg-hh-threshold.dat"
    if path.exists():
        raise RuntimeError("Threshold points are already bound")
    path.write_text(str(len(events)) + "\n" + "".join(" ".join(f"{x:.17e}" for x in p) + "\n" for event in events for p in event))
    metadata = point_metadata(path, [0., 0., float(higgs), float(higgs)], 2)
    metadata.update(grid=grid, parameter_card=bound(card), top_mass_gev=top,
                    higgs_mass_gev=higgs, width_policy="Unchanged zero-width physical card")
    bench.dump(path.with_suffix(".json"), metadata)
    threshold = copy.deepcopy(config)
    threshold.update(points_per_case=12, repetitions=2,
                     cases=[{**case, "id": "supplement-gg-hh-threshold"}])
    bench.dump(report / "threshold.json", threshold)
    return metadata


def distribution(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Invalid estimated-accuracy population")
    ordered = sorted(values)
    return {"count": len(values), "minimum": ordered[0], "median": statistics.median(values),
            "maximum": ordered[-1], "p90": ordered[math.ceil(.9*len(values))-1],
            "zero_count": values.count(0.), "interpretation": "MadLoop internal estimate; not independently measured precision"}


def compare_rows(groups, pairs=None):
    keys = list(groups)
    if not keys:
        raise ValueError("Empty numerical supplement cohort")
    stream = [(r["point"], r["order"]) for r in groups[keys[0]]]
    if not stream or len(set(stream)) != len(stream):
        raise ValueError("Empty or duplicated point/order stream")
    scales = defaultdict(float)
    for rows in groups.values():
        if [(r["point"], r["order"]) for r in rows] != stream:
            raise ValueError("Mismatched supplement point/order streams")
        for row in rows:
            if len(row["coefficients"]) != 4 or not all(math.isfinite(x) for x in row["coefficients"]):
                raise ValueError("Nonfinite physical supplement coefficients")
            scales[row["point"]] = max(scales[row["point"]], *map(abs, row["coefficients"]))
    pairs = list(itertools.combinations(keys, 2) if pairs is None else pairs)
    failures, checks, worst, largest_scaled = [], 0, 0., 0.
    per_point = {p: {"maximum_tolerance_fraction": 0., "maximum_point_scaled_absolute_difference": 0.} for p, _ in stream}
    for left, right in pairs:
        for a, b in zip(groups[left], groups[right]):
            for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                tolerance = 1e-7*max(abs(x), abs(y)) + 1e-10*scales[a["point"]]
                error = abs(x-y)
                fraction = error/tolerance if tolerance else 0.
                scaled = error/scales[a["point"]] if scales[a["point"]] else 0.
                worst, largest_scaled = max(worst, fraction), max(largest_scaled, scaled)
                target = per_point[a["point"]]
                target["maximum_tolerance_fraction"] = max(target["maximum_tolerance_fraction"], fraction)
                target["maximum_point_scaled_absolute_difference"] = max(target["maximum_point_scaled_absolute_difference"], scaled)
                checks += 1
                if error > tolerance:
                    failures.append({"targets": [left, right], "point": a["point"], "order": a["order"],
                                     "coefficient": coefficient, "values": [x, y], "tolerance": tolerance})
    return {"passed": not failures, "status": "failed" if failures else "passed", "pairs": len(pairs),
            "points": len({p for p, _ in stream}), "orders": len({o for _, o in stream}),
            "comparisons": checks, "max_tolerance_fraction": worst,
            "maximum_point_scaled_absolute_difference": largest_scaled, "failures": failures,
            "per_point": {str(p): item for p, item in sorted(per_point.items())}}


def stability_summary(groups):
    """Separate estimates, observed discrepancies, and status/rescue use."""
    references = [name for name in groups if name.startswith("original_helper_corrected/")]
    if not references:
        raise ValueError("Independent backend references are missing")
    targets = {}
    for name, rows in groups.items():
        peers = [peer for peer in references if peer != name]
        discrepancy = compare_rows(groups, [(name, peer) for peer in peers])
        totals = [r for r in rows if r["order"] == 0]
        counts = Counter(str(r["status"]) for r in totals)
        targets[name] = {
            "estimated_accuracy": distribution([r["accuracy"] for r in totals]),
            "estimated_accuracy_by_order": {str(order): distribution([r["accuracy"] for r in rows if r["order"] == order])
                                             for order in sorted({r["order"] for r in rows})},
            "observed_discrepancy": {**discrepancy,
                "interpretation": "Observed cross-backend coefficient discrepancy; the references are not an exact high-precision amplitude oracle"},
            "status_counts": dict(sorted(counts.items())),
            "dp": sum(count for code, count in counts.items() if int(code)//100 == 2),
            "qp": sum(count for code, count in counts.items() if int(code)//100 == 3),
            "ninjars_genuine_qp": sum(count for code, count in counts.items() if code == "322") if name.endswith("/8") else None,
            "per_point": [{"point": r["point"], "status": r["status"], "H": r["status"]//100,
                           "T": r["status"]//10 % 10, "U": r["status"] % 10,
                           "estimated_accuracy": r["accuracy"],
                           **discrepancy["per_point"][str(r["point"])]} for r in totals]}
    return {"targets": targets,
            "coverage": "Fixed deterministic streams/grid only; no probabilistic instability-rate claim",
            "independent_high_precision_controls": "Reported separately under precision_controls; not inferred from MadLoop accuracy or cross-backend agreement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate-baseline", "generate-candidate", "threshold"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    if args.action == "prepare":
        result = preparation(args.report, args.work)
    elif args.action.startswith("generate-"):
        result = generate(args.action.removeprefix("generate-"), args.report, args.work)
    else:
        result = prepare_threshold(args.report, args.work)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
