#!/usr/bin/env python3
"""One fresh full100 TIRust lane after the recorded user interruption."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
CASE = "ew_n4_epem_mupmumaa"
OLD = REPORT / "experiments/numerical" / ("full100-warmup-" + CASE + "-1")
INPUT_SHA = "6517159cbdea243e0767955d35f5935b09f56d53021720487e5f11803346225a"
ADAPTER_SHA = "2fc766cab10bf4822a0216f566944ad6e4386225b21633b544e3fed7870a8c49"
REMAINING_SHA = "e3ee0bf1d51fa46c391c6cba7477d297832704ed311c6e4fb36ffe945c004d59"
LANES = [("baseline", 1), ("candidate", 1), ("candidate", 6), ("candidate", 7), ("candidate", 8)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(name, path, expected):
    if digest(path) != expected:
        raise ValueError("Pinned supervision source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


remaining = load("ew_remaining_support", REPORT / "remaining-qualification-1/run.py", REMAINING_SHA)
adapter = remaining.adapter
if digest(REPORT / "n1-qualification-1/run.py") != ADAPTER_SHA:
    raise ValueError("Pinned n1 adapter changed")
bench, campaign = adapter.bench, adapter.campaign
dump, now = adapter.dump, adapter.now


def checked_files(index, base):
    for name, record in index.items():
        relative = Path(name)
        path = base / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(base.resolve())
                or digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]):
            raise ValueError("Sealed input changed: " + name)


def verify_seal():
    seal = read(HERE / "source-index.json")
    checked_files(seal["files"], HERE)
    checked_files(seal["dependencies"], ROOT)
    if digest(HERE / "inputs-index.json") != INPUT_SHA:
        raise ValueError("Original-lane input inventory changed")
    inputs = read(HERE / "inputs-index.json")
    checked_files(inputs["files"], ROOT)
    if (len(inputs["files"]) != inputs["file_count"]
            or sum(item["bytes"] for item in inputs["files"].values()) != inputs["total_bytes"]):
        raise ValueError("Input inventory size differs")


def setup():
    verify_seal()
    adapter.CASES = [CASE]
    adapter.HERE = HERE
    adapter.verify_seal = verify_seal
    adapter.declared_cards = remaining.declared_cards
    campaign.activate_campaign(bench, adapter.CAMPAIGN)
    support = adapter.load_support()
    config = bench.load_config()
    case = next(value for value in config["cases"] if value["id"] == CASE)
    if bench.case_worker_timeout(case) != 14400. or config["points_per_case"] != 100:
        raise ValueError("Registered worker scope changed")
    if 12 not in bench.os.sched_getaffinity(0):
        raise ValueError("CPU12 is outside allowed nonexclusive affinity")
    return support, config, case


def lane_summary(source, backend, observation, comparison, directory):
    return {"source": source, "backend": backend, "status": "complete", "error": None,
        "numerical_agreement_passed": comparison["passed"],
        "ordinary_target_passed": comparison["ordinary_target_passed"],
        "status_counts": observation["status_counts"],
        "accepted_backend_precision_counts": observation["backend_precision_counts"],
        "matched_saved_points": len({row["point"] for row in observation["rows"]}),
        "result_path": str((directory / "result.json").relative_to(ROOT)),
        "fpe_flags": None, "internal_retry_counts": None, "performance_acceptance_eligible": False}


def replay_original(config, records, numerical):
    specification = read(OLD / "specification.json")
    if (specification["campaign_binding"] != bench.campaign_binding()
            or specification["worker_timeout_seconds"] != 14400.
            or specification["measurement_phase"] != "steady-warmup"
            or specification["performance_eligible"] is not False
            or specification["configuration_sha256"] != digest(REPORT / "cases.json")
            or digest(OLD / "cases.json") != digest(REPORT / "cases.json")
            or digest(OLD / "campaign-manifest.json") != adapter.MANIFEST_SHA):
        raise ValueError("Original qualification specification differs")
    for name, expected in specification["protocols"].items():
        if digest(OLD / name) != expected or digest(ROOT / "tools" / name) != expected:
            raise ValueError("Original numerical protocol differs from current tools")
    reference, summaries = None, []
    for source, backend in LANES:
        directory = OLD / f"{source}-{backend}"
        value = read(directory / "result.json")
        if value["status"] != "complete" or (value["source"], value["backend"]) != (source, backend):
            raise ValueError("Original selected lane is not complete")
        observation = numerical.replay_lane(directory, source, backend, value, config, records[source + "/" + CASE])
        if reference is None:
            reference = observation
        comparison = bench.compare_rows(reference["rows"], observation["rows"], config["relative_tolerance"])
        if value["comparison"] != comparison or not comparison["passed"]:
            raise ValueError("Original lane lacks unchanged regression qualification")
        summaries.append(lane_summary(source, backend, observation, comparison, directory))
    return reference, summaries


def preflight():
    support, config, case = setup()
    campaign.verify_campaign(bench, stage="ew-continuation-preflight")
    configurations = adapter.configurations()
    expected = read(REPORT / "remaining-qualification-1" / CASE / ("numerical-" + CASE) / "configuration-before.json")
    if configurations != expected:
        raise ValueError("Configuration differs from original attempt")
    records = adapter.verify_builds(config, support[2])
    reference, lanes = replay_original(config, records, support[2])
    # The original partial worker used exactly backend9. Verify its current
    # source, executable, cards, points and dependencies before any reset.
    interrupted_inputs = read(OLD / "candidate-9/candidate-inputs.json")
    bench.verify_observation_inputs("candidate", case, interrupted_inputs)
    for source in ("baseline", "candidate"):
        if (bench.subprocess_dir(source, case) / "worker-protocol.fifo").exists():
            raise ValueError("A selected worker FIFO still exists")
    state = read(REPORT / "remaining-qualification-1" / CASE / "status.json")
    last = state["stages"][-1]
    if (state["status"] != "failed" or "InterruptedError" not in state.get("error", "")
            or last["cleanup"]["survivors"] or "signal 2" not in last.get("error", "")):
        raise ValueError("Original user interruption is not closed and preserved")
    return {"status": "source-and-closed-evidence-preflight-passed",
        "campaign": adapter.CAMPAIGN, "manifest_sha256": adapter.MANIFEST_SHA,
        "inputs_index_sha256": INPUT_SHA, "configurations": configurations,
        "compiled_inputs": records, "original_lanes": lanes,
        "reference_rows_sha256": bench.content_digest(reference["rows"]),
        "worker_timeout_seconds": 14400., "stage_cap_seconds": 28800.,
        "cleanup_reserve_seconds": 60., "new_native_lanes": [["candidate", 9]],
        "native_executions": 0, "timing_authorized": False}


def item():
    return {"id": "numerical-" + CASE, "case": CASE, "label": "fresh-full100-candidate9-after-user-stop",
        "cap_seconds": 28800., "command": [sys.executable, "-B", str(HERE.relative_to(ROOT) / "run.py"), "--lane"]}


def run_lane():
    support, config, case = setup()
    if not (HERE / "specification.json").is_file():
        raise ValueError("The supervised continuation has not been admitted")
    records = read(HERE / ("numerical-" + CASE) / "compiled-inputs.json")
    reference, _ = replay_original(config, records, support[2])
    report = HERE / "lane"
    report.mkdir(exist_ok=False)
    original_spec = read(OLD / "specification.json")
    for name in [*original_spec["protocols"], "cases.json", "campaign-manifest.json"]:
        shutil.copy2(OLD / name, report / name)
    specification = {**original_spec, "purpose": "Fresh full100 candidate9 after user-requested daemon-update interruption",
        "machine": bench.machine_identity(12), "host_load": bench.host_load_snapshot(),
        "selected_native_lanes": [["candidate", 9]], "original_inputs_index_sha256": INPUT_SHA,
        "continuation_source_index_sha256": digest(HERE / "source-index.json"), "concatenated": False}
    dump(report / "specification.json", specification, True)
    directory = report / "candidate-9"
    directory.mkdir()
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((ROOT / ".tirust-build/numerical-tests.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker, _ = bench.start_worker(stack, "candidate", case, [9], config, 12, directory, timeout=14400.)
            raw = directory / "raw"
            raw.mkdir(exist_ok=False)
            for name in ("warmup.dat", "dimensions.dat"):
                shutil.copy2(worker.directory / name, raw / name)
            ready = {key: getattr(worker, key) for key in ("points", "ndim", "nso", "nborn", "cold_seconds")}
            dump(raw / "ready.json", ready, True)
            from tirust_benchmark_warmup import parse_warmup
            observation = parse_warmup(raw, [9], 100, ready)
            dump(raw / "observation.json", observation, True)
        inputs = read(directory / "candidate-inputs.json")
        dump(directory / "inputs-after.json", bench.verify_observation_inputs("candidate", case, inputs), True)
        comparison = bench.compare_rows(reference["rows"], observation["rows"], config["relative_tolerance"])
        result = {"source": "candidate", "backend": 9, "status": "complete", "performance_eligible": False,
                  "observation": observation, "comparison": comparison}
        dump(directory / "result.json", result, True)
        print(json.dumps({"source": "candidate", "backend": 9, "status": "complete",
                          "regression_passed": comparison["passed"]}), flush=True)
        if not comparison["passed"]:
            raise RuntimeError("Fresh candidate9 numerical disagreement retained")
    except BaseException as error:
        dump(directory / "failure.json", {"source": "candidate", "backend": 9, "status": "failed",
            "performance_eligible": False, "error": repr(error)}, True)
        raise


def observed_result(stage, returncode, config, records):
    numerical = adapter.load_support()[2]
    reference, lanes = replay_original(config, records, numerical)
    directory = HERE / "lane/candidate-9"
    if returncode or not (directory / "result.json").is_file():
        return {"case": CASE, "status": "failed", "returncode": returncode, "lanes": lanes}
    value = read(directory / "result.json")
    observation = numerical.replay_lane(directory, "candidate", 9, value, config, records["candidate/" + CASE])
    comparison = bench.compare_rows(reference["rows"], observation["rows"], config["relative_tolerance"])
    if comparison != value["comparison"]:
        raise ValueError("Fresh comparison differs from replay")
    lanes.append(lane_summary("candidate", 9, observation, comparison, directory))
    complete = all(lane["matched_saved_points"] == 100 and lane["numerical_agreement_passed"] for lane in lanes)
    return {"case": CASE, "status": "passed" if complete else "failed", "returncode": returncode,
        "raw_replay_passed": True, "lanes": lanes, "performance_acceptance_eligible": False,
        "original_attempt_status": "interrupted", "fresh_candidate9_points": 100, "concatenated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--lane", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sum((args.preflight, args.execute, args.lane)) > 1:
        parser.error("Choose one mode")
    if args.lane:
        run_lane()
        return
    verify_seal()
    if not args.execute:
        print(json.dumps({"status": "no-native-execution", "stage": item(),
                         "preflight": preflight() if args.preflight else None}, indent=2, sort_keys=True))
        return
    if (HERE / "specification.json").exists() or (HERE / "lane").exists():
        raise ValueError("Append-only continuation already attempted; no automatic retry")
    readiness = preflight()
    if readiness != read(HERE / "preflight.json")["preflight"]:
        raise ValueError("Readiness differs from the preregistered continuation")
    support, config, _ = setup()
    support[2].observed_result = observed_result
    dump(HERE / "specification.json", {"schema": "madloop-tirust-single-ew-lane-continuation-v1",
        "created_utc": now(), "source_index_sha256": digest(HERE / "source-index.json"),
        "manifest_sha256": adapter.MANIFEST_SHA, "readiness": readiness, "stage": item(),
        "performance_acceptance_eligible": False, "timing_authorized": False}, True)
    state = {"status": "running", "started_utc": now(), "stages": [], "case": CASE,
             "performance_acceptance_eligible": False}
    try:
        with support[0].SignalRequests() as requests:
            adapter.run_stage(item(), config, readiness["configurations"], bench.environment(), requests, state, support)
            requests.check()
        state.update(status="passed", finished_utc=now())
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished_utc=now())
        raise
    finally:
        dump(HERE / "status.json", state)
    print("CLOSED fresh candidate9 lane; combined checkpoint awaits pure sealing/audit", flush=True)


if __name__ == "__main__":
    main()
