#!/usr/bin/env python3
"""Draft serial numerical qualification; --execute requires coordinator clearance."""
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
CAMPAIGN = "split-dp-terminal-1"
REPORT = ROOT / "docs/benchmarkfolder/campaigns" / CAMPAIGN
MANIFEST = REPORT / "manifest.json"
EXPECTED_MANIFEST = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
CASE_ORDER = ["qcd_n1_uux_z", "ew_n1_udx_wp", "li_gg_h", "qcd_n2_gg_ttx",
              "ew_n2_epem_mupmum", "mixed_uux_ga", "li_gg_hh", "qcd_n3_uux_zgg",
              "ew_n3_epem_ttxa", "qcd_n4_uux_zzgg", "ew_n4_epem_mupmumaa"]
PROTOCOLS = ["tirust_benchmark_numerical.py", "tirust_benchmark.py",
             "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
             "tirust_benchmark_warmup.py", "tirust_benchmark_compiler.py",
             "tirust_benchmark_campaign.py"]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE))
import tirust_benchmark as bench
import tirust_benchmark_campaign as campaign
from tirust_benchmark_compiler import validate_compiler_equality
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


def commands(cpu):
    return [{"case": case_id, "label": f"full100-warmup-{case_id}-1", "command": [
        sys.executable, "-B", "tools/tirust_benchmark_numerical.py", "--campaign", CAMPAIGN,
        "--case", case_id, "--label", f"full100-warmup-{case_id}-1", "--warmup-only",
        "--backends", "1", "6", "7", "8", "9", "--cpu", str(cpu)]}
        for case_id in CASE_ORDER]


def verify_builds(config):
    records = {}
    for case in config["cases"]:
        pair = {}
        for source in ("baseline", "candidate"):
            output = bench.output_dir(source, case)
            directory = bench.subprocess_dir(source, case)
            generation = read(output / "generation.json")
            compiled = read(output / "compilation.json")
            if compiled.get("campaign_binding", {}).get("manifest_sha256") != EXPECTED_MANIFEST:
                raise ValueError(f"Missing campaign build binding: {source}/{case['id']}")
            if (bench.generated_sources(output) != compiled["sources"]
                    or digest(directory / "check") != compiled["executable_sha256"]
                    or compiled.get("numerical_build_from_sources") is not True):
                raise ValueError(f"Source/executable mismatch: {source}/{case['id']}")
            bench.verify_worker_template(compiled, ROOT / "tools/tirust_benchmark_worker.f")
            campaign.validate_generation_environment(generation, required=True)
            if source == "candidate" and compiled.get("full_numerical_rebuild_this_invocation") is not True:
                raise ValueError(f"Candidate lacks its fresh numerical source build: {case['id']}")
            if source == "baseline" and not generation.get("reused_from"):
                raise ValueError(f"Baseline lacks its private reuse proof: {case['id']}")
            pair[source] = compiled
            records[f"{source}/{case['id']}"] = {
                "generation_sha256": digest(output / "generation.json"),
                "compilation_sha256": digest(output / "compilation.json"),
                "executable_sha256": digest(directory / "check"),
                "points_sha256": digest(REPORT / "points" / (case["id"] + ".dat")),
                "points_metadata_sha256": digest(REPORT / "points" / (case["id"] + ".json")),
            }
        validate_compiler_equality(pair["baseline"]["effective_compiler_flags"],
                                   pair["candidate"]["effective_compiler_flags"])
    return records


def observed_result(item, returncode):
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
    lanes = []
    for value in results["results"]:
        observation, comparison = value.get("observation"), value.get("comparison")
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
    expected = [("baseline", 1)] + [("candidate", value) for value in (1, 6, 7, 8, 9)]
    complete = ([(lane["source"], lane["backend"]) for lane in lanes] == expected
                and all(lane["status"] == "complete" and lane["matched_saved_points"] == 100
                        and lane["numerical_agreement_passed"] for lane in lanes))
    return {"case": item["case"], "status": "passed" if complete and returncode == 0 else "failed",
            "returncode": returncode, "results_path": str(path.relative_to(REPORT)),
            "results_sha256": digest(path), "lanes": lanes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=9)
    parser.add_argument("--execute", action="store_true",
                        help="Launch only after coordinator approves completed source preparation")
    args = parser.parse_args()
    if ROOT != bench.ROOT or digest(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError("Unexpected working directory or campaign manifest")
    if args.cpu not in os.sched_getaffinity(0):
        raise ValueError("Selected CPU is outside allowed affinity")
    schedule = commands(args.cpu)
    if not args.execute:
        print(json.dumps({"status": "draft-no-execution", "commands": schedule,
                          "fpe_flags_observed": False, "extra_timing_loops": False}, indent=2))
        return
    campaign.activate_campaign(bench, CAMPAIGN)
    campaign.verify_campaign(bench, stage="numerical-supervisor-before")
    config = bench.load_config()
    if set(CASE_ORDER) != {case["id"] for case in config["cases"]} or len(CASE_ORDER) != 11:
        raise ValueError("Qualification schedule must contain all eleven cases exactly once")
    if read(REPORT / "preparation-1/status.json")["status"] != "passed":
        raise ValueError("Source preparation must finish successfully before numerical qualification")
    for item in schedule:
        if (REPORT / "experiments/numerical" / item["label"]).exists():
            raise ValueError("An append-only numerical label already exists; do not overwrite or silently retry")
    records = verify_builds(config)
    protocols = {name: digest(ROOT / "tools" / name) for name in PROTOCOLS}
    configs = configurations()
    specification = {"schema": "madloop-tirust-serial-full100-numerical-v1", "created_utc": now(),
        "campaign": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST,
        "supervisor_sha256": digest(__file__), "process_cleanup_sha256": digest(HERE / "process_cleanup.py"),
        "supervisor_pid": os.getpid(), "cpu": args.cpu,
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
            campaign.verify_campaign(bench, stage="numerical-supervisor-case-before")
            if configurations() != configs or any(digest(ROOT / "tools" / name) != value for name, value in protocols.items()):
                raise ValueError("Configuration or frozen protocol changed before numerical case")
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
            entry.update(observed_result(item, code), finished_utc=now(), console_sha256=digest(log))
            if configurations() != configs or any(digest(ROOT / "tools" / name) != value for name, value in protocols.items()):
                raise ValueError("Configuration or frozen protocol changed during numerical case")
            campaign.verify_campaign(bench, stage="numerical-supervisor-case-after")
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
