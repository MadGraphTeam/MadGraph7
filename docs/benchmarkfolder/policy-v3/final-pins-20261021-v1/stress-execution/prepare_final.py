#!/usr/bin/env python3
"""Final-pin two-case CLI/build preparation, with no amplitude evaluation.

Reuse the retained early preparation procedure. The only adapters validate the
new installed publication and own children through the reviewed v3 helper.
"""
import argparse
from contextlib import contextmanager
import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_precision_v2 as precision
import ninjars_stress_v2 as stress
import ninjars_supplement_v2 as supplement

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = ROOT / "docs/benchmarkfolder/policy-v2/final-stress-generation-v1"
OLD_HELPER = ROOT / "docs/benchmarkfolder/policy-v2/final-stress-execution-v1/early_build.py"
loader = importlib.util.spec_from_file_location("retained_stress_early_build", OLD_HELPER)
early = importlib.util.module_from_spec(loader)
loader.loader.exec_module(early)

REVISION = "856e76e669618ec8d0f799b9c8e6f86819dcd986"
GENERATOR = "77b16df0251af4104a2c541bab913c11f4e67496a290495ce1a1c5fc526dfe3c"
RUNTIME = "a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008"
ARCHIVE = "b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c"
IMAGE = {"Cargo.toml", "Cargo.lock", "include/mninjars.f90", "include/ninjars.h",
         "install.log", "lib/libninjars.a", "lib/mninjars.mod",
         "lib/ninjars-config.json", "lib/ninjars-link-before.txt",
         "lib/ninjars-native-libs.txt"}
USED = (Path(__file__), OLD_HELPER, *(ROOT / "tools" / (name + ".py") for name in (
    "ninjars_benchmark", "ninjars_benchmark_pair_v3", "ninjars_benchmark_workers",
    "ninjars_benchmark_policy_v2", "ninjars_precision_v2", "ninjars_stress_v2",
    "ninjars_supplement_v2")))


def read(path):
    return json.loads(Path(path).read_text())


def runtime(spec):
    """Verify all ten installed files and exact measured numerical descriptor."""
    if (spec["generation_revision"] != REVISION or spec["numerical_revision"] != REVISION
            or spec["generator_identity_sha256"] != GENERATOR
            or spec["archive_sha256"] != ARCHIVE):
        raise ValueError("The authorized final source/archive cohort changed")
    signature = policy.verified(spec["runtime_signature"], "reducer-runtime-signature-v2")
    if signature["identity_sha256"] != RUNTIME:
        raise ValueError("Use the actual observed final runtime descriptor")
    publication = read(supplement.checked(spec["publication"]))
    if (publication["schema"] != "main-ninjars-final-pin-transition-v1"
            or publication["status"] != "published"
            or publication["revision"] != REVISION
            or publication["main_destination"] != "HEPTools/ninjars"
            or set(publication["main_artifact_sha256"]) != IMAGE
            or publication["main_artifact_sha256"]["lib/libninjars.a"] != ARCHIVE):
        raise ValueError("The actual publication is not the frozen final ten-file image")
    installation = read(supplement.checked(publication["accepted_installation"]))
    if (installation["revision"] != REVISION or installation["passed"] is not True
            or installation["actual_cli_update"] is not True
            or installation["compiled_smoke_passed"] is not True
            or installation["files"] != publication["main_artifact_sha256"]):
        raise ValueError("Publication differs from the actual validated CLI installation")
    prefix = ROOT / "HEPTools/ninjars"
    actual_files = {str(p.relative_to(prefix)) for p in prefix.rglob("*") if p.is_file()}
    if actual_files != IMAGE:
        raise ValueError("Main installation has unexpected or missing files")
    for name, checksum in publication["main_artifact_sha256"].items():
        if supplement.digest(prefix / name) != checksum:
            raise ValueError("Published installation asset changed: " + name)
    for name, checksum in installation["archived_bridge_source_sha256"].items():
        if supplement.digest(ROOT / "vendor/ninjars" / name) != checksum:
            raise ValueError("Installed bridge source differs from source freeze: " + name)
    metadata = {name: supplement.digest(prefix / "lib" / name)
                for name in policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
    precision.match_runtime_archive(signature, ARCHIVE, metadata, RUNTIME)
    return signature


@contextmanager
def owned_generation(work, report):
    """One CLI/build job at a time, with registered and identity-checked groups."""
    with supplement.harness(work, report), pair.owned_preparation(report, label="stress-generation"):
        original = bench.run

        def run(command, cwd, log, **kwargs):
            actual = list(command)
            if actual == ["make", "-j4", "check"]:
                actual = ["make", "-j1", "check"]
            elif len(actual) == 4 and actual[2] == "--plain":
                card = Path(actual[3])
                if not card.resolve().is_relative_to(report / "commands"):
                    raise ValueError("Unexpected CLI command card outside stress namespace")
                text = card.read_text()
                if text.count("set nb_core 4\n") != 1:
                    raise ValueError("Retained CLI resource setting changed")
                card.write_text(text.replace("set nb_core 4\n", "set nb_core 1\n"))
            elif actual != ["./check", "--masses"]:
                raise ValueError("Unexpected operation in generation-only preparation")
            return original(actual, cwd, log, **kwargs)

        bench.run = run
        try:
            yield
        finally:
            bench.run = original


def specification():
    old = read(OLD / "early-input-spec.json")
    return {**old, "generation_revision": REVISION, "numerical_revision": REVISION,
            "generator_identity_sha256": GENERATOR, "archive_sha256": ARCHIVE,
            "runtime_signature": read(BASE / "generation/observed-qcd1-runtime.json"),
            "publication": supplement.bound(BASE / "main-install-publication/transition.json"),
            "report": policy.relative(BASE / "stress-generation"),
            "work": ".benchmark-work/policy-v3/final-pins-20261021-v1/stress-generation"}


def source_hashes():
    return {policy.relative(path): supplement.digest(path) for path in USED}


def validate_inputs(result):
    """Regeneration must reproduce all first-retained physical input bytes."""
    old = policy.verified(read(OLD / "early-result.json"), "stress-early-build-result-v1")
    old_registration = policy.verified(read(supplement.checked(old["registration"])), early.SCHEMA)
    registration = policy.verified(read(supplement.checked(result["registration"])), early.SCHEMA)
    if (registration["configuration"] != old_registration["configuration"]
            or registration["physical_cards"] != old_registration["physical_cards"]):
        raise ValueError("Stress fixed configuration or physical cards changed")
    checks = {}
    for group in ("configurations", "point_files", "ordinary_point_metadata"):
        if result[group].keys() != old[group].keys():
            raise ValueError("Stress input inventory changed: " + group)
        for name, record in result[group].items():
            current, retained = supplement.checked(record), supplement.checked(old[group][name])
            if current.read_bytes() != retained.read_bytes():
                raise ValueError("Stress input is not byte-identical: " + group + "/" + name)
            checks[group + "/" + name] = {"current": record, "original": old[group][name],
                                          "byte_identical": True}
    if supplement.checked(result["scale_schedule"]).read_bytes() != supplement.checked(old["scale_schedule"]).read_bytes():
        raise ValueError("Stress scale schedule changed")
    for cid, record in result["source_proofs"].items():
        proof = policy.verified(read(supplement.checked(record)), "stress-generation-source-v1")
        old_proof = policy.verified(read(supplement.checked(old["source_proofs"][cid])), "stress-generation-source-v1")
        current = supplement.checked(proof["physical_card"])
        retained = supplement.checked(old_proof["physical_card"])
        if current.read_bytes() != retained.read_bytes():
            raise ValueError("Stress physical card changed: " + cid)
        checks["physical_cards/" + cid] = {"current": proof["physical_card"],
            "original": old_proof["physical_card"], "byte_identical": True}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only source/installation preflight")
    args = parser.parse_args()
    spec = specification()
    before = source_hashes()
    runtime(spec)
    early.sources(REVISION)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != REVISION:
        raise ValueError("Generation HEAD is not the authorized held revision")
    if bench.content_digest(bench.generator_tree_state()) != GENERATOR:
        raise ValueError("Generator input tree is not the authorized final cohort")
    if args.check:
        print(json.dumps({"status": "preflight-passed", "amplitude_evaluations": 0,
                          "runtime_identity_sha256": RUNTIME, "source_sha256": before}, indent=2))
        return
    if os.sched_getaffinity(0) != {6, 7}:
        raise ValueError("Root released CPU6/7 for this generation-only controller")
    for name in pair.THREADS:
        if os.environ.get(name) != "1":
            raise ValueError("Generation environment requires one evaluation thread: " + name)
    early.runtime, early.owned_generation = runtime, owned_generation
    spec_path = HERE / "final-input-spec.json"
    if spec_path.exists():
        raise ValueError("Preserve the earlier preparation specification")
    bench.dump(spec_path, spec)
    fields = (Path("/proc") / str(os.getpid()) / "stat").read_text().rsplit(")", 1)[1].split()
    bench.dump(HERE / "controller.json", {"pid": os.getpid(), "pgid": os.getpgrp(),
        "start_ticks": int(fields[19]), "affinity": sorted(os.sched_getaffinity(0)),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "amplitude_evaluations": 0, "initialized_symbolica_workers": 0})
    def interrupted(sig, _frame):
        raise SystemExit("Stress preparation interrupted by signal " + str(sig))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    result = early.prepare(spec_path)
    report = policy.path(spec["report"])
    shutil.copy2(Path(__file__), report / "sources/prepare_final.py")
    checks = validate_inputs(result)
    if source_hashes() != before:
        raise ValueError("Stress preparation helper source changed during execution")
    runtime(spec)
    ledger = [json.loads(line) for line in (report / "process-ledger.jsonl").read_text().splitlines()]
    spawns = {(item["pid"], item["start_ticks"]) for item in ledger if item["event"] == "spawn"}
    reaped = {(item["pid"], item["start_ticks"]) for item in ledger if item["event"] == "reaped" and item.get("reaped")}
    if spawns != reaped:
        raise ValueError("A generation child has not been reaped")
    final = policy.seal({"schema": "final-pin-stress-generation-v1", "status": "complete",
        "early_build_result": supplement.bound(report / "early-result.json"),
        "fixed_inputs": checks, "scale_schedule": result["scale_schedule"],
        "source_sha256_before_and_after": before, "controller": supplement.bound(HERE / "controller.json"),
        "process_ledger": supplement.bound(report / "process-ledger.jsonl"),
        "all_registered_children_reaped": True, "amplitude_evaluations": 0,
        "initialized_symbolica_workers": 0, "runtime_identity_sha256": RUNTIME,
        "archive_sha256": ARCHIVE, "generator_identity_sha256": GENERATOR,
        "generation_revision": REVISION,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    bench.dump(report / "final-result.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
