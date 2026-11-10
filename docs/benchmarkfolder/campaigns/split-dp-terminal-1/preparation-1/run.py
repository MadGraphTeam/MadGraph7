#!/usr/bin/env python3
"""Serialize approved source preparation; do not launch numerical workers."""
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
WORK = ROOT / ".benchmark-work/tirust/campaigns" / CAMPAIGN
MANIFEST = REPORT / "manifest.json"
EXPECTED_MANIFEST = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
CPUS = [12, 13, 14, 15]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
import tirust_benchmark_campaign as campaign
from tirust_benchmark_compiler import validate_compiler_equality


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def event(value):
    with (HERE / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": now(), **value}, sort_keys=True) + "\n")


def configurations():
    paths = sorted(set((ROOT / "input").glob("*configuration*.txt")) |
                   set((bench.BASELINE / "input").glob("*configuration*.txt")))
    return {str(path.relative_to(ROOT)): {"sha256": digest(path), "text": path.read_text()}
            for path in paths if path.is_file()}


def check_preparation(stage, config):
    for case in config["cases"]:
        base = WORK / "outputs/baseline" / case["id"]
        generation = json.loads((base / "generation.json").read_text())
        if not generation.get("reused_from", {}).get("worker_only_relink_required"):
            raise ValueError(f"Baseline lacks the private reuse proof: {case['id']}")
        if stage == "reuse-baseline":
            continue
        baseline_compilation = json.loads((base / "compilation.json").read_text())
        if baseline_compilation.get("full_numerical_rebuild_this_invocation") is not False:
            raise ValueError(f"Expected a proven baseline worker-only relink: {case['id']}")
        if baseline_compilation.get("campaign_binding", {}).get("manifest_sha256") != EXPECTED_MANIFEST:
            raise ValueError(f"Baseline relink lacks the campaign binding: {case['id']}")
        if stage != "build-candidate":
            continue
        target = WORK / "outputs/candidate" / case["id"]
        candidate_generation = json.loads((target / "generation.json").read_text())
        candidate_compilation = json.loads((target / "compilation.json").read_text())
        if candidate_generation.get("reused_from"):
            raise ValueError(f"Candidate must be a fresh CLI export: {case['id']}")
        if candidate_generation.get("export_environment") != {"PYTHONHASHSEED": "0"}:
            raise ValueError(f"Candidate export environment mismatch: {case['id']}")
        if candidate_compilation.get("full_numerical_rebuild_this_invocation") is not True:
            raise ValueError(f"Candidate must rebuild numerical objects from source: {case['id']}")
        validate_compiler_equality(baseline_compilation["effective_compiler_flags"],
                                   candidate_compilation["effective_compiler_flags"])


def main():
    if ROOT != bench.ROOT or digest(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError("Unexpected working directory or campaign manifest")
    if not set(CPUS) <= os.sched_getaffinity(0):
        raise ValueError("Approved preparation CPUs are unavailable")
    os.sched_setaffinity(0, CPUS)
    campaign.activate_campaign(bench, CAMPAIGN)
    campaign.verify_campaign(bench, stage="source-preparation")
    if (WORK / "outputs").exists():
        raise ValueError("Preparation namespace already has outputs; do not overwrite a previous attempt")
    config = bench.load_config()
    common = ["taskset", "-c", "12-15", sys.executable, "-B", "tools/tirust_benchmark.py"]
    tail = ["--campaign", CAMPAIGN, "--cpu", "12", "--jobs", "4"]
    stages = [
        ("reuse-baseline", [*common, "reuse-baseline", *tail]),
        ("build-baseline", [*common, "build", "--source", "baseline", *tail]),
        ("build-candidate", [*common, "build", "--source", "candidate", *tail]),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    configs = configurations()
    specification = {
        "schema": "madloop-tirust-serial-source-preparation-v1", "created_utc": now(),
        "campaign": CAMPAIGN, "manifest_sha256": EXPECTED_MANIFEST,
        "supervisor_sha256": digest(__file__), "supervisor_pid": os.getpid(),
        "cpus": CPUS, "jobs": 4, "case_ids": [case["id"] for case in config["cases"]],
        "stages": [{"id": name, "command": command} for name, command in stages],
        "configuration_hashes": {path: item["sha256"] for path, item in configs.items()},
        "environment": {key: environment.get(key) for key in (
            "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")},
        "scope": "Private baseline proof/worker relink plus fresh candidate CLI/source builds; mandatory --masses SETPARA only, zero amplitude/reducer evaluations",
        "native_numerical_evaluations_authorized": False,
        "performance_acceptance_eligible": False,
    }
    dump(HERE / "specification.json", specification, exclusive=True)
    dump(HERE / "configuration-initial.json", configs, exclusive=True)
    state = {"supervisor_pid": os.getpid(), "started_utc": now(), "status": "running", "stages": []}
    dump(HERE / "status.json", state)
    try:
        for name, command in stages:
            before = configurations()
            dump(HERE / name / "configuration-before.json", before, exclusive=True)
            if before != configs:
                raise ValueError(f"Configuration drift before {name}")
            campaign.verify_campaign(bench, stage="source-preparation-before")
            log = HERE / name / "console.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            entry = {"id": name, "started_utc": now(), "status": "running", "command": command}
            state["stages"].append(entry)
            with log.open("xb") as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                entry["child_pid"] = child.pid
                dump(HERE / "status.json", state)
                event({"event": "stage-start", "stage": name, "child_pid": child.pid})
                print(f"START {name} child_pid={child.pid}", flush=True)
                code = child.wait()
            entry.update(returncode=code, finished_utc=now(), console_sha256=digest(log))
            after = configurations()
            dump(HERE / name / "configuration-after.json", after, exclusive=True)
            if after != configs:
                raise ValueError(f"Configuration drift after {name}; original snapshots retained, no restoration attempted")
            if code:
                raise RuntimeError(f"Preparation stage {name} exited {code}; console retained")
            campaign.verify_campaign(bench, stage="source-preparation-after")
            check_preparation(name, config)
            entry["status"] = "passed"
            dump(HERE / "status.json", state)
            event({"event": "stage-pass", "stage": name, "returncode": code})
            print(f"PASS {name}", flush=True)
        state.update(status="passed", finished_utc=now(), amplitude_evaluations=0)
        dump(HERE / "status.json", state)
        event({"event": "preparation-pass"})
    except BaseException as error:
        state.update(status="failed", finished_utc=now(), error=repr(error))
        dump(HERE / "status.json", state)
        event({"event": "preparation-failure", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
