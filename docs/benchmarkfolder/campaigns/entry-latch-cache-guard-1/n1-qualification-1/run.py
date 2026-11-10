#!/usr/bin/env python3
"""Supervise existing CLI stages for two n1 cases; default is source-only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
CAMPAIGN = "entry-latch-cache-guard-1"
MANIFEST_SHA = "4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076"
CASES = ["qcd_n1_uux_z", "ew_n1_udx_wp"]
CPUS = [12, 13, 14, 15]
NUMERICAL_CPU = 12
CLEANUP_SECONDS = 60.
SUPPORT = {
    "process_cleanup": ("docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/process_cleanup.py", "240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84"),
    "n1_preparation_support": ("docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/run.py", "726f629210a155b894f423ad76fa1de9d96b1f98270d4aceba2ba240250d6901"),
    "n1_numerical_support": ("docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-qualification-1/run.py", "36622d7aa41cd6fe19cb01a992a6ec527fb9899579f57ddd6c5f7725b2fd58b0"),
}
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
import tirust_benchmark_campaign as campaign
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_support():
    modules = []
    for name, (relative, checksum) in SUPPORT.items():
        path = ROOT / relative
        if digest(path) != checksum:
            raise ValueError("Reviewed supervision/replay support changed: " + relative)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    cleanup, preparation, numerical = modules
    # Reuse the exact reviewed readers against this new namespace; never invoke
    # either historical main() or its all-eleven preparation schedule.
    numerical.CAMPAIGN, numerical.REPORT = CAMPAIGN, REPORT
    numerical.EXPECTED_MANIFEST = MANIFEST_SHA
    return cleanup, preparation, numerical


def schedule():
    tool = ["taskset", "-c", "12-15", sys.executable, "-B", "tools/tirust_benchmark.py"]
    tail = ["--campaign", CAMPAIGN, "--cases", *CASES, "--cpu", "12", "--jobs", "4"]
    items = [
        {"id": "reuse-baseline", "cap_seconds": 600., "command": [*tool, "reuse-baseline", *tail]},
        {"id": "generate-candidate", "cap_seconds": 900., "command": [*tool, "generate", "--source", "candidate", *tail]},
        {"id": "build-both", "cap_seconds": 1200., "command": [*tool, "build", "--source", "both", *tail]},
    ]
    for case in CASES:
        label = "full100-warmup-" + case + "-1"
        items.append({"id": "numerical-" + case, "case": case, "label": label, "cap_seconds": 1800.,
            "command": [sys.executable, "-B", "tools/tirust_benchmark_numerical.py", "--campaign", CAMPAIGN,
                "--case", case, "--label", label, "--warmup-only", "--backends", "1", "6", "7", "8", "9",
                "--cpu", str(NUMERICAL_CPU)]})
    return items


def verify_seal():
    value = read(HERE / "source-index.json")
    for section, base in (("files", HERE), ("inputs", ROOT)):
        for relative, checksum in value[section].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or digest(base / path) != checksum:
                raise ValueError("Source admission seal changed: " + relative)
    if digest(REPORT / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Campaign manifest changed")


def configurations():
    baseline = ROOT / ".benchmark-work/tirust/baseline"
    paths = sorted(set((ROOT / "input").glob("*configuration*.txt")) |
                   set((baseline / "input").glob("*configuration*.txt")))
    return {str(p.relative_to(ROOT)): {"sha256": digest(p), "text": p.read_text()}
            for p in paths if p.is_file()}


def readonly_preflight():
    """Hash actual inputs only: no activation, children, compile or native call."""
    verify_seal()
    manifest = read(REPORT / "manifest.json")
    if not set(CPUS) <= os.sched_getaffinity(0):
        raise ValueError("Approved nonexclusive CPU affinity is unavailable")
    for name, checksum in manifest["candidate"]["exporter_identity"]["files"].items():
        path = ROOT / name
        if (digest(path) if path.is_file() else None) != checksum:
            raise ValueError("Current generator source differs: " + name)
    for name, checksum in manifest["shared_inputs"]["files"].items():
        if digest(REPORT / name) != checksum:
            raise ValueError("Frozen shared input differs: " + name)
    work = ROOT / ".benchmark-work/tirust/campaigns" / CAMPAIGN
    if campaign._inventory(bench, work / "models" / campaign.MODEL) != manifest["shared_inputs"]["model_tree"]:
        raise ValueError("Pinned campaign UFO model tree changed")
    prefix = Path(manifest["candidate"]["native"]["prefix"])
    for name, checksum in manifest["candidate"]["native"]["files"].items():
        if digest(prefix / name) != checksum:
            raise ValueError("Installed native artifact differs: " + name)
    baseline = read(REPORT / "inputs/baseline.json")
    if digest(REPORT / "inputs/baseline.json") != manifest["baseline"]["record_sha256"]:
        raise ValueError("Baseline source record changed")
    parent = ROOT / ".benchmark-work/tirust/campaigns" / manifest["predecessor"]["id"]
    records = {}
    for case in CASES:
        if manifest["administrative_worker_timeout_seconds"][case] != 3600.:
            raise ValueError("Registered n1 worker deadline changed")
        output = parent / "outputs/baseline" / case
        if any(output.rglob("worker-protocol.fifo")):
            raise ValueError("Predecessor baseline is active")
        generation = read(output / "generation.json")
        identity = generation["source_identity"]
        if (identity["revision"] != baseline["revision"] or identity["archive_sha256"] != baseline["archive_sha256"]
                or identity["files_sha256"] != bench.content_digest(baseline["files"])):
            raise ValueError("Predecessor baseline identity differs from the frozen starting source")
        card = manifest["physical_card_origins"]["cards"][case]["param_card_sha256"]
        if generation["parameter_card_sha256"] != card or digest(output / "Cards/param_card.dat") != card:
            raise ValueError("Predecessor baseline physical card differs")
        if bench.parse_chain(generation["default_chain"]) != [8, 6, 7, 1]:
            raise ValueError("Materialized baseline default chain changed")
        records[case] = {"generation_sha256": digest(output / "generation.json"),
                        "compilation_sha256": digest(output / "compilation.json"), "param_card_sha256": card}
    return {"status": "source-only-preflight-passed", "manifest_sha256": MANIFEST_SHA,
            "baseline_records": records, "configurations": configurations(),
            "child_commands_executed": 0, "native_evaluations": 0}


def declared_cards():
    manifest = read(REPORT / "manifest.json")
    records = {}
    for case in CASES:
        expected = manifest["physical_card_origins"]["cards"][case]["param_card_sha256"]
        for source in ("baseline", "candidate"):
            output = bench.WORK / "outputs" / source / case
            generation = read(output / "generation.json")
            if (generation.get("campaign_binding") != bench.campaign_binding()
                    or generation["parameter_card_sha256"] != expected
                    or digest(output / "Cards/param_card.dat") != expected
                    or bench.parse_chain(generation["default_chain"]) != [8, 6, 7, 1]):
                raise ValueError("Declared physical card/default chain changed: " + source + "/" + case)
            records[source + "/" + case] = {"param_card_sha256": expected,
                                            "generation_sha256": digest(output / "generation.json")}
    return records


def verify_builds(config, numerical):
    manifest, baseline = read(REPORT / "manifest.json"), read(REPORT / "inputs/baseline.json")
    declared_cards()
    records = {}
    for case in (c for c in config["cases"] if c["id"] in CASES):
        points = REPORT / "points" / (case["id"] + ".dat")
        metadata = read(points.with_suffix(".json"))
        pair = {}
        for source in ("baseline", "candidate"):
            output, directory = bench.output_dir(source, case), bench.subprocess_dir(source, case)
            generation, compiled = read(output / "generation.json"), read(output / "compilation.json")
            if (compiled.get("campaign_binding") != bench.campaign_binding()
                    or generation["case"] != case or generation["source"] != source
                    or bench.generated_sources(output) != compiled["sources"]
                    or digest(directory / "check") != compiled["executable_sha256"]
                    or compiled.get("numerical_build_from_sources") is not True):
                raise ValueError("Compiled source/executable provenance differs")
            campaign.validate_generation_environment(generation, required=True)
            bench.verify_worker_template(compiled, ROOT / "tools/tirust_benchmark_worker.f")
            if source == "candidate":
                if (compiled.get("full_numerical_rebuild_this_invocation") is not True or generation.get("reused_from")
                        or generation["source_identity"]["files_sha256"] != manifest["candidate"]["exporter_identity"]["files_sha256"]):
                    raise ValueError("Candidate was not freshly generated and fully source built")
            elif compiled.get("full_numerical_rebuild_this_invocation") is not False:
                raise ValueError("Baseline must preserve proven numerical objects")
            evidence = REPORT / "inputs" / source / case["id"]
            if read(evidence / "generation.json") != generation or read(evidence / "compilation.json") != compiled:
                raise ValueError("Retained build copies differ")
            if source == "baseline":
                numerical.verify_baseline(generation, compiled, manifest, baseline, evidence, output)
            physical = bench.resolved_parameter_card(directory)
            if physical is None or digest(physical) != generation["parameter_card_sha256"]:
                raise ValueError("Fortran parameter-card resolution differs")
            if (metadata["inputs"]["count"] != 100 or digest(points) != metadata["sha256"]
                    or metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"] != compiled["masses"]):
                raise ValueError("Saved full100 point stream/masses differ")
            dependencies = bench.dependency_inputs(directory)
            if any(compiled[k] != dependencies for k in ("link_inputs_before", "link_inputs_after")):
                raise ValueError("Link-time dependency bytes changed")
            if not bench.same_compiler_policy(capture_effective_flags(directory, env=bench.environment()), compiled["effective_compiler_flags"]):
                raise ValueError("Compiler policy changed")
            runtime = numerical.runtime_files(bench.runtime_identity(directory))
            if bench.dependency_inputs(directory) != dependencies:
                raise ValueError("Link-time dependency bytes changed during verification")
            if source == "candidate":
                numerical.verify_native_runtime(runtime, manifest)
            pair[source] = compiled
            records[source + "/" + case["id"]] = {
                "generation_content_sha256": bench.content_digest(generation),
                "compilation_content_sha256": bench.content_digest(compiled),
                "executable_sha256": compiled["executable_sha256"],
                "param_card_sha256": generation["parameter_card_sha256"],
                "resolved_parameter_card": str(physical.resolve()), "points_sha256": digest(points),
                "points_metadata_sha256": digest(points.with_suffix(".json")), "runtime": runtime}
        validate_compiler_equality(pair["baseline"]["effective_compiler_flags"], pair["candidate"]["effective_compiler_flags"])
    return records


def cleanup_owned(owned, end):
    # These phases fit inside the sixty seconds already reserved by StageDeadline.
    remaining = max(0., end - time.monotonic())
    budget = min(55., remaining)
    result = owned.stop(cooperative_seconds=budget * 45./55.,
                        terminate_seconds=budget * 5./55., kill_seconds=budget * 5./55.)
    if result["survivors"]:
        raise RuntimeError("Owned cleanup is unresolved; later admissions are blocked")
    return result


def run_stage(item, config, initial, environment, requests, state, support):
    cleanup, preparation, numerical = support
    preparation.STAGE_SECONDS, preparation.CLEANUP_RESERVE = item["cap_seconds"], CLEANUP_SECONDS
    directory = HERE / item["id"]
    directory.mkdir(exist_ok=False)
    entry = {**item, "status": "running", "started_utc": now()}
    state["stages"].append(entry)
    owned = None
    cleaned = False
    try:
        with preparation.StageDeadline() as deadline:
            entry["deadline_monotonic"] = deadline.end
            requests.check(); verify_seal()
            campaign.verify_campaign(bench, stage="n1-before")
            before = configurations()
            dump(directory / "configuration-before.json", before, True)
            if before != initial:
                raise ValueError("Configuration drift before stage")
            if item["id"] == "build-both" or "case" in item:
                dump(directory / "cards-before.json", declared_cards(), True)
            admitted = verify_builds(config, numerical) if "case" in item else None
            if admitted is not None:
                dump(directory / "compiled-inputs.json", admitted, True)
            with (directory / "console.log").open("xb") as stream:
                deadline.handoff = True
                try:
                    process = subprocess.Popen(item["command"], cwd=ROOT, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    owned = cleanup.OwnedChild(process)
                finally:
                    deadline.handoff = False
                    if deadline.deferred:
                        deadline.expired(None, None)
                entry["child_pid"] = process.pid
                dump(HERE / "status.json", state)
                print("START " + item["id"] + " child_pid=" + str(process.pid), flush=True)
                entry["returncode"] = owned.wait(requests)
                requests.check()
                entry["cleanup"] = cleanup_owned(owned, deadline.end)
                cleaned = True
            if "case" in item:
                result = numerical.observed_result(item, entry["returncode"], config, admitted)
                dump(directory / "numerical-result.json", result, True)
                if result["status"] != "passed":
                    raise RuntimeError("Numerical case failed; raw/refusal evidence retained")
            elif entry["returncode"]:
                raise RuntimeError("Preparation command failed")
            after = configurations()
            dump(directory / "configuration-after.json", after, True)
            if after != initial:
                raise ValueError("Configuration drift; no restoration performed")
            campaign.verify_campaign(bench, stage="n1-after")
            if item["id"] != "reuse-baseline":
                dump(directory / "cards-after.json", declared_cards(), True)
            if item["id"] == "build-both" or "case" in item:
                current = verify_builds(config, numerical)
                if admitted is not None and current != admitted:
                    raise ValueError("Source/runtime inputs changed during numerical evaluation")
                dump(directory / "verified-builds.json", current, True)
            requests.check(); verify_seal()
            entry.update(status="passed", finished_utc=now(), console_sha256=digest(directory / "console.log"))
    except BaseException as error:
        if owned is not None and not cleaned:
            try:
                entry["cleanup"] = cleanup_owned(owned, entry["deadline_monotonic"])
            except BaseException as cleanup_error:
                entry["cleanup_error"] = repr(cleanup_error)
        entry.update(status="failed", error=repr(error), finished_utc=now())
        # No hashing/replay/native work after expiry: only administrative closure.
        raise
    finally:
        entry["publication_scope"] = "administrative status; unavailable stages are not numerical passes"
        dump(HERE / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Hash source/input readiness without child commands")
    parser.add_argument("--execute", action="store_true", help="Only after explicit coordinator admission")
    args = parser.parse_args()
    if ROOT != bench.ROOT:
        raise ValueError("Run from the repository root")
    verify_seal()
    if not args.execute:
        preflight = readonly_preflight() if args.preflight else None
        print(json.dumps({"status": "dry-run-no-execution", "campaign": CAMPAIGN,
            "manifest_sha256": MANIFEST_SHA, "stages": schedule(), "preflight": preflight,
            "cases": CASES, "points_per_lane": 100, "lanes_per_case": 6,
            "worker_deadline_seconds": 3600., "cleanup_reserve_seconds": CLEANUP_SECONDS,
            "nonexclusive_build_cpus": CPUS, "numerical_cpu": NUMERICAL_CPU,
            "performance_acceptance_eligible": False}, indent=2))
        return
    if (HERE / "specification.json").exists():
        raise ValueError("Append-only attempt already exists; no automatic retry")
    ready = readonly_preflight()
    support = load_support()
    campaign.activate_campaign(bench, CAMPAIGN)
    for case in CASES:
        for source in ("baseline", "candidate"):
            if (bench.WORK / "outputs" / source / case).exists():
                raise ValueError("Selected output already exists; no overwrite")
    for item in schedule():
        if "label" in item and (REPORT / "experiments/numerical" / item["label"]).exists():
            raise ValueError("Append-only numerical label already exists")
    config = bench.load_config()
    initial = configurations()
    specification = {"schema": "madloop-tirust-n1-cli-supervision-v1", "campaign": CAMPAIGN,
        "manifest_sha256": MANIFEST_SHA, "source_index_sha256": digest(HERE / "source-index.json"),
        "created_utc": now(), "supervisor_sha256": digest(__file__), "stages": schedule(),
        "readiness": ready, "worker_deadline_seconds": 3600., "cleanup_reserve_seconds": CLEANUP_SECONDS,
        "source_identity": bench.source_identity(ROOT, persist=False), "nonexclusive_build_cpus": CPUS,
        "numerical_cpu": NUMERICAL_CPU, "cases": CASES, "performance_acceptance_eligible": False,
        "full_matrix_complete": False, "timing_authorized": False}
    dump(HERE / "specification.json", specification, True)
    dump(HERE / "configuration-initial.json", initial, True)
    state = {"status": "running", "started_utc": now(), "stages": [], "cases": CASES,
             "performance_acceptance_eligible": False, "full_matrix_complete": False}
    try:
        with support[0].SignalRequests() as requests:
            for item in schedule():
                run_stage(item, config, initial, bench.environment(), requests, state, support)
            requests.check()
        state.update(status="passed", finished_utc=now())
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_utc=now())
        raise
    finally:
        dump(HERE / "status.json", state)


if __name__ == "__main__":
    main()
