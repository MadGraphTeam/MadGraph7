#!/usr/bin/env python3
"""Isolated original-code references with exact verified model corrections.

Clone original generated outputs, force a full MODEL/subprocess rebuild, then
reuse the unchanged persistent worker protocol. Timings here are diagnostic.
"""
import argparse
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from types import SimpleNamespace

import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as policy

ROOT = bench.ROOT


def clone_output(original, target):
    original, target = policy.path(original), policy.path(target)
    if target.exists():
        raise RuntimeError("Reference output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(original), str(target)], check=True)
    for item in target.rglob("*"):
        if item.is_symlink():
            resolved = item.resolve()
            if resolved.is_relative_to(original):
                replacement = target / resolved.relative_to(original)
                item.unlink()
                item.symlink_to(os.path.relpath(replacement, item.parent))
    return target


def local_write(target, root):
    if not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Refusing to mutate a shared/external reference path: {target}")
    # Parent symlinks are rejected even when their current target is internal;
    # generated model corrections must be concrete isolated files.
    if any(item.is_symlink() for item in [target, *target.parents] if item != root.parent):
        raise RuntimeError(f"Writable reference path is a symlink: {target}")


def force_rebuild(output, case, correction_identity, work, report):
    output = policy.path(output)
    model = output / "Source/MODEL"
    libraries_before = policy.library_hashes(output)
    source_before = policy.manifest(output)
    model_objects = list(model.rglob("*.o"))
    for item in [*model_objects, *model.rglob("*.mod"), *model.rglob("*.smod")]:
        local_write(item, output)
        item.unlink()
    archive = output / "lib/libmodel.a"
    local_write(archive, output)
    archive.unlink(missing_ok=True)
    directories = list(output.glob("SubProcesses/P*_*"))
    if len(directories) != 1:
        raise RuntimeError("Expected exactly one subprocess")
    directory = directories[0]
    for item in [*directory.glob("*.o"), directory / "check"]:
        if item.exists():
            local_write(item, output)
            item.unlink()
    pre = {"prebuild_model_objects": [str(item.relative_to(output)) for item in model.rglob("*.o")],
           "prebuild_model_archive_exists": archive.exists(),
           "prelink_executable_exists": (directory / "check").exists()}
    model_log = report / "logs" / f"model-rebuild-{case['id']}.log"
    start = time.perf_counter()
    bench.run(["make", "-j1", "all"], model, model_log)
    # build() installs the frozen fresh-process driver, recompiles every deleted
    # process object and records exact source/executable/compiler identities.
    with workers.isolated_harness(work, report):
        bench.build("baseline", case)
    if policy.manifest(output) != source_before:
        raise RuntimeError("MODEL/subprocess rebuild changed numerical source")
    libraries_after = policy.library_hashes(output)
    if {k: v for k, v in libraries_before.items() if k != "libmodel.a"} != {
            k: v for k, v in libraries_after.items() if k != "libmodel.a"}:
        raise RuntimeError("Rebuild changed original reduction/DHELAS libraries")
    objects = {str(item.relative_to(output)): workers.digest(item) for item in sorted(model.rglob("*.o"))}
    if not objects or not archive.exists():
        raise RuntimeError("MODEL rebuild produced no archive/objects")
    # Every makeinc-listed object must have been recreated after explicit removal.
    declared = re.findall(r"\b[\w-]+\.o\b", (model / "makeinc.inc").read_text())
    if not declared or any(not (model / name).is_file() for name in declared):
        raise RuntimeError("MODEL rebuild omitted declared object files")
    retained_link_log = report / "logs" / f"fresh-link-{case['id']}.log"
    shutil.copy2(work / "logs" / f"compile-baseline-{case['id']}.log", retained_link_log)
    result = policy.seal({"schema": "forced-model-rebuild-v2",
                          "correction_identity_sha256": correction_identity,
                          "scope": "All MODEL objects/archive and all subprocess objects/executable; original DHELAS/reduction libraries copied unchanged",
                          **pre, "model_objects": objects,
                          "model_archive_sha256": workers.digest(archive),
                          "fresh_executable_sha256": workers.digest(directory / "check"),
                          "fresh_driver_sha256": workers.digest(directory / "check_sa.f"),
                          "sources_sha256": source_before["files_sha256"],
                          "libraries_before": libraries_before, "libraries_after": libraries_after,
                          "seconds": time.perf_counter()-start,
                          "commands": [["make", "-j1", "all"], ["make", "-j4", "check"]],
                          "logs": [policy.bound_file(model_log), policy.bound_file(retained_link_log)]})
    bench.dump(output / "correction-build.json", result)
    policy.verify_rebuild(output / "correction-build.json", output, correction_identity)
    return result


def prepare_corrected(case, correction, work, report):
    record = correction["cases"][case["id"]]
    parent = policy.path(record["parent_output"])
    if policy.manifest(parent) != record["parent_manifest"]:
        raise RuntimeError("Original generated parent changed")
    target = work / "outputs/baseline" / case["id"]
    clone_output(parent, target)
    if policy.manifest(target) != record["parent_manifest"]:
        raise RuntimeError("Isolated clone differs from parent")
    for name in policy.MODEL_FILES:
        source = record["postimages"][name]
        if source:
            destination = target / name
            local_write(destination, target)
            destination.write_bytes(policy.check_file(source).read_bytes())
        elif (target / name).exists():
            raise RuntimeError("Unexpected model file instead of recorded absence")
    corrected = policy.manifest(target)
    expected = {name: record["fresh_after_manifest"]["files"].get(name) for name in policy.MODEL_FILES}
    policy.verify_delta(record["parent_manifest"], corrected, expected)
    if workers.digest(target / "Cards/param_card.dat") != record["physical_cards"][0]["sha256"]:
        raise RuntimeError("Reference physical card changed")
    bench.dump(target / "correction-source.json", policy.seal({
        "schema": "corrected-original-source-v2", "source_kind": "original-generated-output+verified-model-correction",
        "parent_generation": record["original_generation"],
        "correction_identity_sha256": correction["identity_sha256"],
        "parent_manifest": record["parent_manifest"], "corrected_manifest": corrected}))
    force_rebuild(target, case, correction["identity_sha256"], work, report)
    return target


def collect(correction_path, config_path, cases, run_id, cpu, timeout=7200., fresh_parity=False,
            representative=False, points_root=None, backends=(1, 6, 7)):
    correction = policy.validate_correction(correction_path, live=True)
    points_root = policy.path(points_root) if points_root else bench.REPORT / "points"
    config = policy.read(config_path)
    policy.validate_config(config, core=not representative)
    backends = tuple(backends)
    if (not backends or len(set(backends)) != len(backends) or not set(backends).issubset({1, 6, 7}) or
            (not representative and set(backends) != {1, 6, 7})):
        raise RuntimeError("Incomplete/invalid corrected-reference backends")
    if cpu not in os.sched_getaffinity(0) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise RuntimeError("Invalid CPU/run ID")
    definitions = {case["id"]: case for case in config["cases"]}
    if not cases or len(set(cases)) != len(cases) or not set(cases).issubset(definitions):
        raise RuntimeError("Unknown, duplicate or empty reference cases")
    report = bench.REPORT / "policy-v2/reference-runs" / run_id
    work = bench.WORK / "policy-v2/reference-runs" / run_id
    if report.exists() or work.exists():
        raise RuntimeError("Use a fresh run ID; old evidence is never overwritten")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    descriptor = workers.protocol_descriptor()
    for name in workers.SOURCE_NAMES:
        (report / "sources").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "tools" / name, report / "sources" / name)
    shutil.copy2(Path(__file__), report / "sources" / Path(__file__).name)
    bench.dump(report / "protocol.json", {
        "input_adapter": policy.bound_file(report / "sources" / Path(__file__).name),
        "schema": workers.SCHEME, "configuration": config, "protocol_descriptor": descriptor,
        "protocol_identity_sha256": bench.content_digest(descriptor),
        "cpu": cpu, "repetitions": 2, "sources": "baseline", "backends": list(backends),
        "fresh_parity": fresh_parity, "acceptance_eligible": False,
        "source_role": "original_helper_corrected", "timings": "diagnostic-only",
        "correction_identity_sha256": correction["identity_sha256"]})
    status = {"status": "incomplete", "requested_cases": cases, "completed_cases": [],
              "source_role": "original_helper_corrected", "acceptance_eligible": False}
    bench.dump(report / "run-status.json", status)
    args = SimpleNamespace(cpu=cpu, repetitions=2, minimum_seconds=1e-12,
                           timeout=timeout, fresh_parity=fresh_parity, clean_build=True)
    worker_work = work / "workers"
    outputs = {}
    try:
        for case_id in cases:
            case = definitions[case_id]
            prepare_corrected(case, correction, work, report)
            source_points = points_root / (case_id + ".dat")
            points = report / "points" / source_points.name
            points.parent.mkdir(exist_ok=True)
            shutil.copy2(source_points, points)
            with workers.isolated_harness(work, report):
                workers.run_case(case, ("baseline",), backends, config, args,
                                 worker_work, report, run_id)
            for backend in backends:
                key = f"{case_id}-baseline-{backend}"
                outputs[key] = policy.relative(worker_work / key / "outputs/baseline" / case_id)
            status["completed_cases"].append(case_id)
            bench.dump(report / "run-status.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "run-status.json", status)
        raise
    status["status"] = "complete"
    bench.dump(report / "run-status.json", status)
    workers.checked_series(report, minimum=2)
    binding = report / "role-bindings.json"
    policy.bind_run(report, {"baseline": "original_helper_corrected"}, outputs, correction_path, binding)
    return {"report": policy.relative(report), "bindings": policy.relative(binding)}


@contextmanager
def input_roots(roots, outer_work):
    """Resolve only the outer inputs; frozen prepare's inner clones stay private."""
    original = bench.output_dir
    normalized = {source: policy.path(value) for source, value in roots.items()}
    def resolve(source, case):
        if bench.WORK == outer_work:
            return normalized[source] / case["id"]
        return original(source, case)
    bench.output_dir = resolve
    try:
        yield
    finally:
        bench.output_dir = original


def collect_timing(spec_path):
    """Collect unchanged seven-target timing in a new namespace, with explicit roots."""
    spec = policy.read(spec_path)
    config = policy.read(policy.path(spec["configuration"]))
    representative = bool(spec.get("representative", False))
    policy.validate_config(config, core=not representative)
    sources = tuple(spec.get("sources", ["baseline", "candidate"]))
    backends = tuple(spec.get("backends", [1, 6, 7, 8]))
    if (not sources or len(set(sources)) != len(sources) or not set(sources).issubset({"baseline", "candidate"}) or
            not backends or len(set(backends)) != len(backends) or not set(backends).issubset({1, 6, 7, 8}) or
            (not representative and (set(sources) != {"baseline", "candidate"} or set(backends) != {1, 6, 7, 8}))):
        raise RuntimeError("Invalid/incomplete timing target set")
    if representative and not spec.get("fresh_parity"):
        raise RuntimeError("Representative collection requires fresh parity")
    if set(spec["output_roots"]) != {"baseline", "candidate"}:
        raise RuntimeError("Timing adapter requires original and candidate output roots")
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise RuntimeError("Timing destination exists")
    cpu = spec["cpu"]
    if cpu not in os.sched_getaffinity(0):
        raise RuntimeError("Unavailable pinned CPU")
    definitions = {case["id"]: case for case in config["cases"]}
    cases = spec["cases"]
    if not cases or len(set(cases)) != len(cases) or not set(cases).issubset(definitions):
        raise RuntimeError("Invalid timing case selection")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    (report / "sources").mkdir()
    (report / "points").mkdir()
    for name in workers.SOURCE_NAMES:
        shutil.copy2(ROOT / "tools" / name, report / "sources" / name)
    shutil.copy2(Path(__file__), report / "sources" / Path(__file__).name)
    shutil.copy2(policy.path(spec_path), report / "input-spec.json")
    for case in cases:
        shutil.copy2(policy.path(spec["points_root"]) / (case+".dat"), report / "points" / (case+".dat"))
    descriptor = workers.protocol_descriptor()
    repetitions = 2 if representative else config["repetitions"]
    bench.dump(report / "protocol.json", {
        "schema": workers.SCHEME, "configuration": config, "protocol_descriptor": descriptor,
        "protocol_identity_sha256": bench.content_digest(descriptor), "cpu": cpu,
        "repetitions": repetitions, "sources": "both" if len(sources) == 2 else sources[0], "backends": list(backends),
        "fresh_parity": bool(spec.get("fresh_parity", False)), "acceptance_eligible": False,
        "input_adapter": policy.bound_file(report / "sources" / Path(__file__).name),
        "input_spec": policy.bound_file(report / "input-spec.json"),
        "initialization_policy": "Unchanged persistent-worker-v1: initialize and warm once",
        "ordering": "Unchanged run_case: adjacent original6/candidate8, reverse on odd repetitions"})
    status = {"status": "incomplete", "requested_cases": cases, "completed_cases": [],
              "acceptance_eligible": False}
    bench.dump(report / "run-status.json", status)
    args = SimpleNamespace(cpu=cpu, repetitions=repetitions,
        minimum_seconds=float(spec.get("minimum_seconds", 1.)), timeout=float(spec.get("timeout", 7200.)),
        fresh_parity=bool(spec.get("fresh_parity", False)), clean_build=True)
    if args.minimum_seconds <= 0 or args.timeout <= 0:
        raise RuntimeError("Invalid timing duration")
    outputs = {}
    # This outer sentinel is never written. prepare() clones directly from the
    # explicit input roots; all configuration/build mutations occur in jobs.
    outer_work = work / "input-root-sentinel"
    try:
        with workers.isolated_harness(outer_work, report), input_roots(spec["output_roots"], outer_work):
            for case_id in cases:
                workers.run_case(definitions[case_id], sources, backends,
                                 config, args, work / "workers", report, spec["run_id"])
                for source in sources:
                    for backend in backends:
                        if source == "baseline" and backend == 8:
                            continue
                        key = f"{case_id}-{source}-{backend}"
                        outputs[key] = policy.relative(work / "workers" / key / "outputs" / source / case_id)
                status["completed_cases"].append(case_id)
                bench.dump(report / "run-status.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "run-status.json", status)
        raise
    status["status"] = "complete"
    bench.dump(report / "run-status.json", status)
    workers.checked_series(report, minimum=2 if representative else 7, require_core=not representative)
    binding_spec = {"report": policy.relative(report), "outputs": outputs,
                    "role_map": {"baseline": "original", "candidate": "candidate_corrected"}}
    bench.dump(report / "binding-spec.json", binding_spec)
    return binding_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-spec", type=Path, help="Run unchanged seven-target collection using explicit input roots")
    parser.add_argument("--correction", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cases")
    parser.add_argument("--run-id")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=7200.)
    parser.add_argument("--fresh-parity", action="store_true")
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--points-root", type=Path)
    parser.add_argument("--backends", default="1,6,7")
    args = parser.parse_args()
    if args.timing_spec:
        print(json.dumps(collect_timing(args.timing_spec), indent=2))
        return
    if any(value is None for value in (args.correction, args.config, args.cases, args.run_id, args.cpu)):
        parser.error("Reference collection requires correction/config/cases/run-id/cpu")
    cases = ([case["id"] for case in policy.read(args.config)["cases"]] if args.cases == "all"
             else args.cases.split(","))
    print(json.dumps(collect(args.correction, args.config, cases, args.run_id, args.cpu,
                            args.timeout, args.fresh_parity, args.representative, args.points_root,
                            tuple(map(int, args.backends.split(",")))), indent=2))


if __name__ == "__main__":
    main()
