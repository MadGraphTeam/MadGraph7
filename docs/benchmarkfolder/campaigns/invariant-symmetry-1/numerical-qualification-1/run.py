#!/usr/bin/env python3
"""Serial full100 numerical qualification; default dry-run, --execute required."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
CAMPAIGN = "invariant-symmetry-1"
REPORT = ROOT / "docs/benchmarkfolder/campaigns" / CAMPAIGN
MANIFEST = REPORT / "manifest.json"
EXPECTED_MANIFEST = "ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0"
CASE_ORDER = ["qcd_n1_uux_z", "ew_n1_udx_wp", "li_gg_h", "qcd_n2_gg_ttx",
              "ew_n2_epem_mupmum", "mixed_uux_ga", "li_gg_hh", "qcd_n3_uux_zgg",
              "ew_n3_epem_ttxa", "qcd_n4_uux_zzgg", "ew_n4_epem_mupmumaa"]
PROTOCOLS = ["tirust_benchmark_numerical.py", "tirust_benchmark.py",
             "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
             "tirust_benchmark_warmup.py", "tirust_benchmark_compiler.py",
             "tirust_benchmark_campaign.py"]
EXPECTED_LANES = [("baseline", 1)] + [("candidate", value) for value in (1, 6, 7, 8, 9)]
PREPARATION_STAGES = ["reuse-baseline", "build-baseline", "generate-candidate",
                      "assert-declared-cards", "build-candidate"]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import tirust_benchmark as bench
import tirust_benchmark_campaign as campaign
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality
from tirust_benchmark_warmup import parse_warmup
from process_cleanup import OwnedChild, SignalRequests


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def configurations():
    paths = sorted(set((ROOT / "input").glob("*configuration*.txt")) |
                   set((bench.BASELINE / "input").glob("*configuration*.txt")))
    return {str(path.relative_to(ROOT)): {"sha256": digest(path), "text": path.read_text()}
            for path in paths if path.is_file()}


def verify_seal():
    seal = read(HERE / "source-index.json")
    required = {"run.py", "process_cleanup.py", "test_supervisor.py", "README.md"}
    if set(seal["files"]) != required or seal["manifest_sha256"] != EXPECTED_MANIFEST:
        raise ValueError("Numerical supervisor source seal is incomplete")
    for name, checksum in seal["files"].items():
        if digest(HERE / name) != checksum:
            raise ValueError("Numerical supervisor source changed: " + name)
    expected_inputs = {"tools/" + name for name in PROTOCOLS} | {
        str(MANIFEST.relative_to(ROOT)), str((REPORT / "cases.json").relative_to(ROOT)),
        str((REPORT / "preparation-1/source-index.json").relative_to(ROOT))}
    if set(seal["inputs"]) != expected_inputs:
        raise ValueError("Numerical supervisor input seal is incomplete")
    for name, checksum in seal["inputs"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError("Numerical supervisor input changed: " + name)
    for name, checksum in seal.get("validation_files", {}).items():
        if digest(HERE / name) != checksum:
            raise ValueError("Source-only validation record changed: " + name)


def verify_preparation():
    directory = REPORT / "preparation-1"
    status, specification = read(directory / "status.json"), read(directory / "specification.json")
    stages = status.get("stages", [])
    if (status.get("status") != "passed" or status.get("amplitude_evaluations") != 0
            or [stage.get("id") for stage in stages] != PREPARATION_STAGES
            or any(stage.get("status") != "passed" or stage.get("returncode") != 0 for stage in stages)
            or specification.get("campaign") != CAMPAIGN
            or specification.get("manifest_sha256") != EXPECTED_MANIFEST):
        raise ValueError("All five bound source-preparation stages must pass before numerical admission")
    seal = read(directory / "source-index.json")
    for name, checksum in seal["files"].items():
        if digest(directory / name) != checksum:
            raise ValueError("Preparation source seal changed: " + name)
    for name, checksum in seal["inputs"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError("Preparation source input changed: " + name)
    if (specification.get("supervisor_sha256") != digest(directory / "run.py")
            or specification.get("source_index_sha256") != digest(directory / "source-index.json")
            or specification.get("cleanup_sha256") != digest(directory / "process_cleanup.py")):
        raise ValueError("Preparation specification no longer binds its sources")
    return {name: digest(directory / name) for name in ("status.json", "specification.json", "source-index.json")}


def commands(cpu):
    return [{"case": case_id, "label": f"full100-warmup-{case_id}-1", "command": [
        sys.executable, "-B", "tools/tirust_benchmark_numerical.py", "--campaign", CAMPAIGN,
        "--case", case_id, "--label", f"full100-warmup-{case_id}-1", "--warmup-only",
        "--backends", "1", "6", "7", "8", "9", "--cpu", str(cpu)]}
        for case_id in CASE_ORDER]


def runtime_files(runtime):
    # ldd addresses vary between invocations. Bind its complete library closure
    # and static dependencies by path/hash, never flatten library basenames.
    return {section: runtime[section] for section in ("libraries", "archives", "metadata")}


def verify_native_runtime(runtime, manifest):
    native = manifest["candidate"]["native"]
    sections = {"lib/libmadloop_tirust.so": "libraries", "lib/libtirust.a": "archives",
                "lib/tirust-config.json": "metadata", "lib/tirust-link-before.txt": "metadata",
                "lib/tirust-native-libs.txt": "metadata"}
    for name, section in sections.items():
        expected_path = (Path(native["prefix"]) / name).resolve()
        matches = [checksum for path, checksum in runtime[section].items()
                   if Path(path).resolve() == expected_path]
        if not matches or any(value != native["files"][name] for value in matches):
            raise ValueError("Runtime TIRust artifact differs from registered prefix: " + name)
    # Cargo/header source pins are checked by the campaign guard. They are not
    # runtime observations and need not appear in runtime_identity's closure.


def verify_baseline(generation, compiled, manifest, baseline_record, evidence, output):
    identity = generation["source_identity"]
    if (identity.get("revision") != bench.BASELINE_REVISION
            or identity.get("revision") != baseline_record["revision"]
            or identity.get("revision") != manifest["baseline"]["revision"]
            or identity.get("files_sha256") != bench.content_digest(baseline_record["files"])
            or identity.get("archive_sha256") != baseline_record["archive_sha256"]):
        raise ValueError("Baseline source differs from the frozen starting revision")
    reused = generation["reused_from"]
    proof = read(evidence / "object-relink-proof.json")
    previous = Path(bench.LEGACY_REPORT) / manifest["predecessor"]["report"] / "inputs/baseline" / generation["case"]["id"]
    if (digest(previous / "generation.json") != reused["generation_sha256"]
            or digest(previous / "compilation.json") != reused["compilation_sha256"]):
        raise ValueError("Baseline reuse differs from its recorded predecessor hashes")
    old_generation, old_compiled = read(previous / "generation.json"), read(previous / "compilation.json")
    if (any(generation[key] != old_generation[key] for key in
            ("case", "source", "source_identity", "command_sha256", "parameter_card_sha256"))
            or compiled.get("previous_compilation_sha256") != bench.content_digest(old_compiled)
            or proof != reused["object_relink_proof"] or proof.get("passed") is not True
            or proof.get("objects_unchanged") is not True or proof.get("original_executable_unchanged") is not True
            or proof["expected_executable_sha256"] != proof["relinked_executable_sha256"]
            or proof["expected_executable_sha256"] != old_compiled["executable_sha256"]
            or digest(evidence / "tirust_benchmark_relink.py") != proof["protocol_sha256"]):
        raise ValueError("Baseline retained object-relink proof differs or failed")
    expected = {key: value for key, value in reused["numerical_objects"].items() if Path(key).name != "check_sa.o"}
    actual = {key: value for key, value in bench.numerical_objects(output).items() if Path(key).name != "check_sa.o"}
    if not expected or actual != expected:
        raise ValueError("Proven baseline numerical objects changed after worker relink")


def verify_builds(config):
    manifest = read(MANIFEST)
    origins = campaign.verify_physical_card_origins(bench, manifest)
    if origins is None or set(origins["cards"]) != set(CASE_ORDER):
        raise ValueError("Every case requires explicitly declared physical-card ancestry")
    if (config["points_per_case"] != 100 or len(config["cases"]) != 11
            or {case["id"] for case in config["cases"]} != set(CASE_ORDER)):
        raise ValueError("Expected all eleven distinct 100-point cases")
    binding = {"id": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST}
    baseline_path = REPORT / "inputs/baseline.json"
    if digest(baseline_path) != manifest["baseline"]["record_sha256"]:
        raise ValueError("Frozen baseline source record changed")
    baseline_record = read(baseline_path)
    records = {}
    for case in config["cases"]:
        pair = {}
        card_hash = origins["cards"][case["id"]]["param_card_sha256"]
        points = REPORT / "points" / (case["id"] + ".dat")
        metadata = read(points.with_suffix(".json"))
        if (digest(points) != metadata["sha256"] or metadata["inputs"]["count"] != 100
                or digest(points) != manifest["shared_inputs"]["files"]["points/" + points.name]
                or digest(points.with_suffix(".json")) != manifest["shared_inputs"]["files"]["points/" + points.with_suffix(".json").name]):
            raise ValueError("Saved points differ from the registered input: " + case["id"])
        for source in ("baseline", "candidate"):
            output = bench.output_dir(source, case)
            directory = bench.subprocess_dir(source, case)
            generation = read(output / "generation.json")
            compiled = read(output / "compilation.json")
            if (compiled.get("campaign_binding") != binding or generation.get("campaign_binding") != binding
                    or generation.get("case") != case or generation.get("source") != source):
                raise ValueError(f"Missing campaign build binding: {source}/{case['id']}")
            if (bench.generated_sources(output) != compiled["sources"]
                    or digest(directory / "check") != compiled["executable_sha256"]
                    or compiled.get("numerical_build_from_sources") is not True):
                raise ValueError(f"Source/executable mismatch: {source}/{case['id']}")
            bench.verify_worker_template(compiled, ROOT / "tools/tirust_benchmark_worker.f")
            campaign.validate_generation_environment(generation, required=True)
            actual_card = bench.resolved_parameter_card(directory)
            if (digest(output / "Cards/param_card.dat") != card_hash
                    or generation["parameter_card_sha256"] != card_hash
                    or actual_card is None or digest(actual_card) != card_hash):
                raise ValueError(f"Physical card differs from declared ancestry or is shadowed: {source}/{case['id']}")
            if metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"] != compiled["masses"]:
                raise ValueError(f"Point masses differ from the compiled process: {source}/{case['id']}")
            if source == "candidate" and (compiled.get("full_numerical_rebuild_this_invocation") is not True
                    or generation.get("reused_from")
                    or generation.get("source_identity", {}).get("files_sha256") != manifest["candidate"]["exporter_identity"]["files_sha256"]):
                raise ValueError(f"Candidate lacks its fresh numerical source build: {case['id']}")
            if source == "baseline" and (not generation.get("reused_from", {}).get("worker_only_relink_required")
                    or compiled.get("full_numerical_rebuild_this_invocation") is not False):
                raise ValueError(f"Baseline lacks its private reuse proof: {case['id']}")
            evidence = REPORT / "inputs" / source / case["id"]
            if (read(evidence / "generation.json") != generation or read(evidence / "compilation.json") != compiled
                    or digest(evidence / "param_card.dat") != card_hash):
                raise ValueError(f"Retained build/card provenance differs from working inputs: {source}/{case['id']}")
            if source == "baseline":
                verify_baseline(generation, compiled, manifest, baseline_record, evidence, output)
            dependencies = bench.dependency_inputs(directory)
            if compiled.get("link_inputs_before") != dependencies or compiled.get("link_inputs_after") != dependencies:
                raise ValueError(f"Dependencies differ from link-time evidence: {source}/{case['id']}")
            flags = capture_effective_flags(directory, env=bench.environment())
            if not bench.same_compiler_policy(flags, compiled.get("effective_compiler_flags")):
                raise ValueError(f"Effective compiler policy changed: {source}/{case['id']}")
            runtime = runtime_files(bench.runtime_identity(directory))
            if {key: runtime[key] for key in ("archives", "metadata")} != dependencies:
                raise ValueError(f"Dependencies changed during runtime inspection: {source}/{case['id']}")
            if source == "candidate":
                verify_native_runtime(runtime, manifest)
            pair[source] = compiled
            records[f"{source}/{case['id']}"] = {
                "generation_sha256": digest(output / "generation.json"),
                "compilation_sha256": digest(output / "compilation.json"),
                "generation_content_sha256": bench.content_digest(generation),
                "compilation_content_sha256": bench.content_digest(compiled),
                "executable_sha256": digest(directory / "check"),
                "param_card_sha256": card_hash,
                "resolved_parameter_card": str(actual_card.resolve()),
                "points_sha256": digest(points),
                "points_metadata_sha256": digest(points.with_suffix(".json")),
                "runtime": runtime,
            }
        validate_compiler_equality(pair["baseline"]["effective_compiler_flags"],
                                   pair["candidate"]["effective_compiler_flags"])
    return records


def replay_lane(directory, source, backend, value, config, record):
    raw = directory / "raw"
    observation = parse_warmup(raw, [backend], 100, read(raw / "ready.json"))
    if value.get("performance_eligible") is not False or value.get("observation") != observation:
        raise ValueError("Result differs from the exact final-warmup replay")
    if read(raw / "observation.json") != observation or read(directory / "result.json") != value:
        raise ValueError("Retained result and observation copies disagree")
    for name in ("warmup.dat", "dimensions.dat"):
        if digest(raw / name) != digest(directory / (source + "-initialization") / name):
            raise ValueError("Warmup input differs from the retained initialization")
    inputs = read(directory / (source + "-inputs.json"))
    after = read(directory / "inputs-after.json")
    if (inputs.get("campaign_binding") != {"id": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST}
            or bench.content_digest(inputs["generation"]) != record["generation_content_sha256"]
            or bench.content_digest(inputs["compilation"]) != record["compilation_content_sha256"]
            or runtime_files(inputs["runtime"]) != record["runtime"]
            or runtime_files(after["runtime"]) != record["runtime"]
            or after["executable_sha256"] != record["executable_sha256"]
            or after["numerical_cards"] != inputs["numerical_cards"]
            or after["points_sha256"] != record["points_sha256"]):
        raise ValueError("Lane inputs differ from admitted build/runtime evidence or failed their postcheck")
    for key in ("param_card_sha256", "resolved_parameter_card", "points_sha256", "points_metadata_sha256"):
        if inputs[key] != record[key]:
            raise ValueError("Lane input differs from the admitted identity: " + key)
    settings = inputs["settings"]
    if bench.parse_chain(settings["mlreductionlib"]) != [backend]:
        raise ValueError("Retained selected backend differs from the requested lane")
    options = dict(config["madloop_parameters"])
    if source == "candidate":
        options.update(config["candidate_parameters"])
    if any(settings[name.lower()] != value for name, value in options.items()):
        raise ValueError("Numerical settings differ from the frozen policy")
    bench.validate_worker_modes(settings)
    return observation


def observed_result(item, returncode, config, records):
    report = REPORT / "experiments/numerical" / item["label"]
    path = report / "results.json"
    if not path.is_file():
        return {"case": item["case"], "status": "failed", "returncode": returncode,
                "error": "Numerical controller did not create results.json", "lanes": []}
    try:
        results = read(path)
    except (OSError, ValueError) as error:
        return {"case": item["case"], "status": "failed", "returncode": returncode,
                "error": f"Numerical controller result is unreadable: {error}", "lanes": []}
    case = next(value for value in config["cases"] if value["id"] == item["case"])
    if results.get("case") != case or results.get("performance_eligible") is not False:
        raise ValueError("Retained result belongs to another case or measurement phase")
    spec = read(report / "specification.json")
    if (spec.get("campaign_binding") != {"id": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST}
            or spec.get("case") != case or spec.get("performance_eligible") is not False
            or spec.get("measurement_phase") != "steady-warmup"
            or spec["worker_timeout_seconds"] != bench.case_worker_timeout(case)
            or spec["configuration_sha256"] != digest(REPORT / "cases.json")
            or digest(report / "cases.json") != digest(REPORT / "cases.json")
            or digest(report / "campaign-manifest.json") != EXPECTED_MANIFEST
            or spec["protocols"] != {name: digest(ROOT / "tools" / name) for name in PROTOCOLS}
            or any(digest(report / name) != spec["protocols"][name] for name in PROTOCOLS)):
        raise ValueError("Retained numerical specification differs from the admitted protocol")
    actual_lanes = [(value["source"], value["backend"]) for value in results["results"]]
    if actual_lanes != EXPECTED_LANES[:len(actual_lanes)] or len(actual_lanes) > len(EXPECTED_LANES):
        raise ValueError("Numerical result lanes differ from the required order")
    lanes, reference = [], None
    for value in results["results"]:
        observation, comparison = value.get("observation"), value.get("comparison")
        source, backend = value["source"], value["backend"]
        directory = report / f"{source}-{backend}"
        if value["status"] == "complete":
            observation = replay_lane(directory, source, backend, value, config, records[source + "/" + case["id"]])
            if source == "baseline":
                reference = observation
            expected = (bench.compare_rows(reference["rows"], observation["rows"], config["relative_tolerance"])
                        if reference is not None else None)
            if comparison != expected:
                raise ValueError("Retained numerical comparison differs from replay")
        elif value["status"] != "failed" or read(directory / "failure.json") != value:
            raise ValueError("Numerical failure record is missing or changed")
        lane = {"source": value["source"], "backend": value["backend"],
                "status": value["status"], "error": value.get("error"),
                "numerical_agreement_passed": comparison.get("passed") if comparison else None,
                "ordinary_target_passed": comparison.get("ordinary_target_passed") if comparison else None,
                "status_counts": observation.get("status_counts") if observation else None,
                "accepted_backend_precision_counts": observation.get("backend_precision_counts") if observation else None,
                "matched_saved_points": len({row["point"] for row in observation["rows"]}) if observation else 0,
                "fpe_flags": None, "internal_retry_counts": None,
                "performance_acceptance_eligible": False}
        if observation and (observation.get("performance_eligible") is not False
                            or observation.get("seconds") is not None
                            or observation.get("measurement_phase") != "steady-warmup"):
            raise ValueError("Expected only untimed final-warmup observations")
        lanes.append(lane)
    complete = ([(lane["source"], lane["backend"]) for lane in lanes] == EXPECTED_LANES
                and all(lane["status"] == "complete" and lane["matched_saved_points"] == 100
                        and lane["numerical_agreement_passed"] for lane in lanes))
    return {"case": item["case"], "status": "passed" if complete and returncode == 0 else "failed",
            "returncode": returncode, "results_path": str(path.relative_to(REPORT)),
            "results_sha256": digest(path), "raw_replay_passed": True, "lanes": lanes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=9)
    parser.add_argument("--execute", action="store_true",
                        help="Launch only after coordinator approves completed source preparation")
    args = parser.parse_args()
    if ROOT != bench.ROOT or digest(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError("Unexpected working directory or campaign manifest")
    verify_seal()
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("Selected CPU is outside allowed affinity")
    schedule = commands(args.cpu)
    if not args.execute:
        print(json.dumps({"status": "dry-run-no-execution", "campaign": CAMPAIGN,
                          "manifest_sha256": EXPECTED_MANIFEST, "commands": schedule,
                          "cpu": args.cpu, "affinity_exclusive": False, "points_per_lane": 100,
                          "lanes_per_case": 6, "fpe_flags_observed": False,
                          "performance_acceptance_eligible": False, "extra_timing_loops": False}, indent=2))
        return
    if (HERE / "specification.json").exists() or (HERE / "status.json").exists():
        raise ValueError("Numerical supervisor already has an attempt; do not overwrite or retry")
    preparation = verify_preparation()
    campaign.activate_campaign(bench, CAMPAIGN)
    campaign.verify_campaign(bench, stage="numerical-supervisor-before")
    config = bench.load_config()
    if set(CASE_ORDER) != {case["id"] for case in config["cases"]} or len(config["cases"]) != 11:
        raise ValueError("Qualification schedule must contain all eleven cases exactly once")
    for item in schedule:
        if (REPORT / "experiments/numerical" / item["label"]).exists():
            raise ValueError("An append-only numerical label already exists; do not overwrite or silently retry")
    records = verify_builds(config)
    protocols = {name: digest(ROOT / "tools" / name) for name in PROTOCOLS}
    configs = configurations()
    specification = {"schema": "madloop-tirust-serial-full100-numerical-v2", "created_utc": now(),
        "campaign": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST,
        "supervisor_sha256": digest(__file__), "process_cleanup_sha256": digest(HERE / "process_cleanup.py"),
        "source_index_sha256": digest(HERE / "source-index.json"), "preparation": preparation,
        "supervisor_pid": os.getpid(), "cpu": args.cpu, "affinity_exclusive": False,
        "schedule": schedule, "compiled_inputs": records, "protocols": protocols,
        "worker_timeouts": bench.CAMPAIGN_MANIFEST["administrative_worker_timeout_seconds"],
        "machine": bench.machine_identity(args.cpu), "host_before": bench.host_load_snapshot(),
        "performance_acceptance_eligible": False, "extra_timing_loops": False,
        "fpe_flags_observed": False, "internal_retry_counts_observed": False,
        "precision_scope": "Final accepted backend/precision only; no inference about internal attempts",
        "admission": "One supervised numerical command at a time; reviewed isolated diagnostics may run on separate CPUs; existing common flock serializes every campaign and diagnostic backend8/9 worker; formal timing requires all owned native and compilation jobs to finish",
        "timing_authorized_after_success": False}
    dump(HERE / "specification.json", specification, exclusive=True)
    dump(HERE / "configuration-initial.json", configs, exclusive=True)
    state = {"status": "running", "started_utc": now(), "supervisor_pid": os.getpid(), "cases": []}
    dump(HERE / "status.json", state)
    child = owned = None
    requests = SignalRequests()
    requests.__enter__()
    try:
        for item in schedule:
            requests.check()
            verify_seal()
            if verify_preparation() != preparation:
                raise ValueError("Completed preparation evidence changed before numerical case")
            campaign.verify_campaign(bench, stage="numerical-supervisor-case-before")
            if configurations() != configs or any(digest(ROOT / "tools" / name) != value for name, value in protocols.items()):
                raise ValueError("Configuration or frozen protocol changed before numerical case")
            if verify_builds(config) != records:
                raise ValueError("One of the 22 admitted build/card/runtime identities changed before numerical case")
            requests.check()
            entry = {"case": item["case"], "status": "running", "started_utc": now()}
            state["cases"].append(entry)
            log = HERE / (item["case"] + ".log")
            with log.open("xb") as stream:
                child = subprocess.Popen(item["command"], cwd=ROOT, env=bench.environment(),
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                owned = OwnedChild(child)
                entry["child_pid"] = child.pid
                dump(HERE / "status.json", state)
                print(f"START {item['case']} child_pid={child.pid}", flush=True)
                code = owned.wait(requests)
                requests.check()
                survivors = [row for row in owned.live() if row["pid"] != child.pid]
                if survivors:
                    raise RuntimeError("Numerical controller exited with observed owned descendants still live")
                child = owned = None
            entry.update(observed_result(item, code, config, records), finished_utc=now(), console_sha256=digest(log))
            if configurations() != configs or any(digest(ROOT / "tools" / name) != value for name, value in protocols.items()):
                raise ValueError("Configuration or frozen protocol changed during numerical case")
            campaign.verify_campaign(bench, stage="numerical-supervisor-case-after")
            verify_seal()
            if verify_preparation() != preparation or verify_builds(config) != records:
                raise ValueError("Completed preparation or admitted build/card/runtime identities changed during numerical case")
            requests.check()
            dump(HERE / "status.json", state)
            print(f"{entry['status'].upper()} {item['case']}", flush=True)
        failed = [item["case"] for item in state["cases"] if item["status"] != "passed"]
        requests.check()
        state.update(status="failed" if failed else "passed", finished_utc=now(), failed_cases=failed,
                     performance_acceptance_eligible=False, timing_authorized=False)
        dump(HERE / "status.json", state)
        dump(HERE / "configuration-final.json", configurations(), exclusive=True)
        if failed:
            raise RuntimeError(f"Unresolved numerical qualification cases retained: {failed}")
    except BaseException as error:
        if owned is not None:
            state["cleanup"] = owned.stop()
            state["cleanup_verified"] = not state["cleanup"]["survivors"]
        elif child is not None and child.poll() is None:
            state["cleanup_verified"] = False
            state["cleanup_error"] = "Controller ownership was not established; manual positively-owned cleanup required before any new admission"
        dump(HERE / "configuration-final.json", configurations())
        state.update(status="failed", finished_utc=now(), error=repr(error))
        dump(HERE / "status.json", state)
        raise
    finally:
        requests.__exit__(None, None, None)


if __name__ == "__main__":
    main()
