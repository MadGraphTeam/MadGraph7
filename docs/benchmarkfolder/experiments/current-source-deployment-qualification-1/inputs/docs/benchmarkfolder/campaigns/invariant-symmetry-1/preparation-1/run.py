#!/usr/bin/env python3
"""Serialize source preparation. Default dry-run; only --execute launches builds."""
import argparse
from datetime import datetime, timezone
import hashlib
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
CAMPAIGN = "invariant-symmetry-1"
REPORT = ROOT / "docs/benchmarkfolder/campaigns" / CAMPAIGN
WORK = ROOT / ".benchmark-work/tirust/campaigns" / CAMPAIGN
MANIFEST = REPORT / "manifest.json"
EXPECTED_MANIFEST = "ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0"
CPUS = [12, 13, 14, 15]
STAGE_SECONDS = 3600.0
CLEANUP_RESERVE = 10.0
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


def event(value):
    with (HERE / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": now(), **value}, sort_keys=True) + "\n")


def schedule():
    common = ["taskset", "-c", "12-15", sys.executable, "-B", "tools/tirust_benchmark.py"]
    tail = ["--campaign", CAMPAIGN, "--cpu", "12", "--jobs", "4"]
    return [
        {"id": "reuse-baseline", "command": [*common, "reuse-baseline", *tail]},
        {"id": "build-baseline", "command": [*common, "build", "--source", "baseline", *tail]},
        {"id": "generate-candidate", "command": [*common, "generate", "--source", "candidate", *tail]},
        {"id": "assert-declared-cards", "command": None, "action": "assert_declared_cards"},
        {"id": "build-candidate", "command": [*common, "build", "--source", "candidate", *tail]},
    ]


def configurations():
    paths = sorted(set((ROOT / "input").glob("*configuration*.txt")) |
                   set((bench.BASELINE / "input").glob("*configuration*.txt")))
    return {str(path.relative_to(ROOT)): {"sha256": digest(path), "text": path.read_text()}
            for path in paths if path.is_file()}


def source_snapshot():
    identity = bench.source_identity(ROOT, persist=False)
    expected = read(MANIFEST)["candidate"]["exporter_identity"]["files_sha256"]
    if identity["files_sha256"] != expected:
        raise ValueError("Registered generator source changed")
    return identity


def verify_seal():
    seal = read(HERE / "source-index.json")
    for name, checksum in seal["files"].items():
        if digest(HERE / name) != checksum:
            raise ValueError("Preparation source changed: " + name)
    for name, checksum in seal["inputs"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError("Preparation input changed: " + name)


def assert_declared_cards(config):
    manifest = read(MANIFEST)
    origins = campaign.verify_physical_card_origins(bench, manifest)
    records = {}
    for case in config["cases"]:
        expected = origins["cards"][case["id"]]["param_card_sha256"]
        for source in ("baseline", "candidate"):
            output = WORK / "outputs" / source / case["id"]
            generation = read(output / "generation.json")
            card = output / "Cards/param_card.dat"
            if (generation["case"] != case or generation["source"] != source
                    or generation.get("campaign_binding") != {"id": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST}
                    or digest(card) != expected or generation["parameter_card_sha256"] != expected):
                raise ValueError("Generated physical card differs from declared ancestry: " + source + "/" + case["id"])
            records[source + "/" + case["id"]] = {"card_sha256": expected,
                "generation_sha256": digest(output / "generation.json")}
    return records


def check_preparation(stage, config):
    for case in config["cases"]:
        base = WORK / "outputs/baseline" / case["id"]
        generation = read(base / "generation.json")
        if not generation.get("reused_from", {}).get("worker_only_relink_required"):
            raise ValueError("Baseline lacks the private reuse proof: " + case["id"])
        if stage == "reuse-baseline":
            continue
        baseline = read(base / "compilation.json")
        if (baseline.get("full_numerical_rebuild_this_invocation") is not False
                or baseline.get("campaign_binding", {}).get("manifest_sha256") != EXPECTED_MANIFEST):
            raise ValueError("Baseline lacks a bound worker-only relink: " + case["id"])
        if stage == "build-baseline":
            continue
        target = WORK / "outputs/candidate" / case["id"]
        candidate = read(target / "generation.json")
        if (candidate.get("reused_from") or candidate.get("export_environment") != {"PYTHONHASHSEED": "0"}
                or candidate.get("source_identity", {}).get("files_sha256") != read(MANIFEST)["candidate"]["exporter_identity"]["files_sha256"]):
            raise ValueError("Candidate lacks a fresh, registered CLI export: " + case["id"])
        if stage == "generate-candidate" and (target / "compilation.json").exists():
            raise ValueError("Fresh candidate export unexpectedly has benchmark compilation provenance")
        if stage != "build-candidate":
            continue
        compiled = read(target / "compilation.json")
        if (compiled.get("full_numerical_rebuild_this_invocation") is not True
                or compiled.get("numerical_build_from_sources") is not True
                or compiled.get("campaign_binding", {}).get("manifest_sha256") != EXPECTED_MANIFEST):
            raise ValueError("Candidate lacks a bound full numerical source build: " + case["id"])
        validate_compiler_equality(baseline["effective_compiler_flags"], compiled["effective_compiler_flags"])


class StageDeadline:
    """Bound synchronous checks too; reserve ten seconds inside the stage cap."""
    def __enter__(self):
        self.end = time.monotonic() + STAGE_SECONDS
        self.handoff = False
        self.deferred = False
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.expired)
        signal.setitimer(signal.ITIMER_REAL, STAGE_SECONDS - CLEANUP_RESERVE)
        return self

    def expired(self, number, frame):
        if getattr(self, "handoff", False):
            self.deferred = True
            return
        raise TimeoutError("Preparation stage reached its work deadline; cleanup reserve begins")

    def __exit__(self, *exception):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


def cleanup_owned(owned, deadline):
    remaining = max(0., deadline - time.monotonic())
    # Every stop phase fits inside the original stage allowance. The frozen
    # helper targets only observed descendants with matching start identities.
    budget = min(6., remaining)
    result = owned.stop(cooperative_seconds=budget / 3,
                        terminate_seconds=budget / 3, kill_seconds=budget / 3)
    if result["survivors"]:
        raise RuntimeError("Owned process cleanup is incomplete; no later stage may start")
    return result


def failure_snapshots(directory, entry):
    remaining = entry.get("deadline_monotonic", 0) - time.monotonic()
    if remaining <= 0:
        entry["postfailure_snapshots_unavailable"] = "stage deadline reached; administrative closure only"
        return
    previous = signal.getsignal(signal.SIGALRM)
    def expired(number, frame):
        raise TimeoutError("Failure snapshot reached the original stage deadline")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        for filename, capture in (("configuration-after-failure.json", configurations),
                                  ("source-after-failure.json", lambda: bench.source_identity(ROOT, persist=False))):
            dump(directory / filename, capture(), exclusive=True)
        if (directory / "console.log").exists():
            entry["console_sha256"] = digest(directory / "console.log")
    except BaseException as error:
        entry["failure_snapshot_error"] = repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_stage(item, config, initial, environment, requests, state):
    name, command = item["id"], item["command"]
    entry = {**item, "started_utc": now(), "status": "running", "cap_seconds": STAGE_SECONDS}
    state["stages"].append(entry)
    directory = HERE / name
    directory.mkdir(exist_ok=False)
    owned = None
    completed = False
    try:
        with StageDeadline() as deadline:
            entry["deadline_monotonic"] = deadline.end
            requests.check()
            verify_seal()
            before = configurations()
            dump(directory / "configuration-before.json", before, exclusive=True)
            dump(directory / "source-before.json", source_snapshot(), exclusive=True)
            if before != initial:
                raise ValueError("Configuration drift before " + name)
            campaign.verify_campaign(bench, stage="source-preparation-before")
            if name == "build-candidate":
                dump(directory / "cards-before.json", assert_declared_cards(config), exclusive=True)
            with (directory / "console.log").open("xb") as stream:
                if command is None:
                    cards = assert_declared_cards(config)
                    dump(directory / "cards.json", cards, exclusive=True)
                    stream.write(b"PASS: all 22 physical cards match declared ancestry\n")
                    code = 0
                else:
                    # Defer every asynchronous exception across the child-handle
                    # handoff. No blocked signal mask is inherited by the child.
                    deadline.handoff = True
                    try:
                        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        owned = OwnedChild(process)
                    finally:
                        deadline.handoff = False
                        if deadline.deferred:
                            deadline.expired(None, None)
                    entry["child_pid"] = process.pid
                    dump(HERE / "status.json", state)
                    event({"event": "stage-start", "stage": name, "child_pid": process.pid})
                    print("START " + name + " child_pid=" + str(process.pid), flush=True)
                    code = owned.wait(requests)
                    entry["cleanup"] = cleanup_owned(owned, deadline.end)
                    completed = True
                entry["returncode"] = code
            if code:
                raise RuntimeError("Preparation stage " + name + " exited " + str(code))
            after = configurations()
            dump(directory / "configuration-after.json", after, exclusive=True)
            dump(directory / "source-after.json", source_snapshot(), exclusive=True)
            if after != initial:
                raise ValueError("Configuration drift after " + name + "; snapshots retained, no restoration")
            campaign.verify_campaign(bench, stage="source-preparation-after")
            check_preparation(name, config)
            if name in ("assert-declared-cards", "build-candidate"):
                dump(directory / "cards-after.json", assert_declared_cards(config), exclusive=True)
            verify_seal()
            requests.check()
            entry.update(status="passed", finished_utc=now(), console_sha256=digest(directory / "console.log"))
    except BaseException as error:
        if owned is not None and not completed:
            try:
                entry["cleanup"] = cleanup_owned(owned, entry["deadline_monotonic"])
            except BaseException as cleanup_error:
                entry["cleanup_error"] = repr(cleanup_error)
        entry.update(status="failed", finished_utc=now(), error=repr(error))
        # Preserve failure snapshots, including changed bytes. No repair/retry.
        # If the cap was reached, only small administrative status is written;
        # do not begin potentially unbounded hashing after the stage deadline.
        failure_snapshots(directory, entry)
        raise
    finally:
        dump(HERE / "status.json", state)
        event({"event": "stage-" + entry["status"], "stage": name})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run only after root review and scheduling admission")
    args = parser.parse_args()
    if ROOT != bench.ROOT or digest(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError("Unexpected working directory or campaign manifest")
    verify_seal()
    stages = schedule()
    if not args.execute:
        print(json.dumps({"status": "dry-run-no-execution", "campaign": CAMPAIGN,
            "manifest_sha256": EXPECTED_MANIFEST, "cpus": CPUS, "affinity_exclusive": False,
            "jobs": 4, "per_stage_seconds": STAGE_SECONDS, "stages": stages}, indent=2))
        return
    if not set(CPUS) <= os.sched_getaffinity(0):
        raise ValueError("Approved preparation CPUs are unavailable")
    if (WORK / "outputs").exists() or (HERE / "specification.json").exists():
        raise ValueError("Preparation already has outputs or an attempt; no overwrite or retry")
    os.sched_setaffinity(0, CPUS)
    campaign.activate_campaign(bench, CAMPAIGN)
    campaign.verify_campaign(bench, stage="source-preparation")
    config = bench.load_config()
    if len(config["cases"]) != 11 or len({case["id"] for case in config["cases"]}) != 11:
        raise ValueError("Expected all eleven distinct registered cases")
    initial = configurations()
    environment = dict(os.environ, PYTHONHASHSEED="0")
    specification = {"schema": "madloop-tirust-serial-source-preparation-v2", "created_utc": now(),
        "campaign": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST,
        "supervisor_sha256": digest(__file__), "source_index_sha256": digest(HERE / "source-index.json"),
        "cleanup_sha256": digest(HERE / "process_cleanup.py"), "supervisor_pid": os.getpid(),
        "cpus": CPUS, "affinity_exclusive": False, "jobs": 4, "per_stage_seconds": STAGE_SECONDS,
        "case_ids": [case["id"] for case in config["cases"]], "stages": stages,
        "configuration_hashes": {name: value["sha256"] for name, value in initial.items()},
        "source_identity": source_snapshot(),
        "environment": {key: environment.get(key) for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")},
        "scope": "Baseline proof/relink, actual CLI candidate generation, 22 declared cards, full fresh candidate source builds; SETPARA mass probes only",
        "native_numerical_evaluations_authorized": False, "performance_acceptance_eligible": False}
    dump(HERE / "specification.json", specification, exclusive=True)
    dump(HERE / "configuration-initial.json", initial, exclusive=True)
    state = {"status": "running", "started_utc": now(), "supervisor_pid": os.getpid(), "stages": []}
    dump(HERE / "status.json", state)
    try:
        with SignalRequests() as requests:
            for item in stages:
                run_stage(item, config, initial, environment, requests, state)
        state.update(status="passed", finished_utc=now(), amplitude_evaluations=0)
    except BaseException as error:
        state.update(status="failed", finished_utc=now(), error=repr(error))
        raise
    finally:
        dump(HERE / "status.json", state)
        event({"event": "preparation-" + state["status"]})


if __name__ == "__main__":
    main()
