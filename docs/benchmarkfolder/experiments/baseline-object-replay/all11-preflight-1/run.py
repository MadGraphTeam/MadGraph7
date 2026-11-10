#!/usr/bin/env python3
"""Private-copy baseline relink proofs; no amplitude or reduction execution."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
from tirust_benchmark_compiler import capture_effective_flags
from tirust_benchmark_relink import prove_object_relink

REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/tirust/preflights/baseline-all11-1"
CPU = 12
PROTOCOLS = ("tirust_benchmark.py", "tirust_benchmark_compiler.py",
             "tirust_benchmark_relink.py", "tirust_benchmark_campaign.py")


def snapshot(output, env):
    directories = sorted(output.glob("SubProcesses/P*_*"))
    if len(directories) != 1:
        raise ValueError("Expected exactly one historical subprocess")
    directory = directories[0]
    return {"directory": str(output), "generation_sha256": bench.digest(output / "generation.json"),
            "compilation_sha256": bench.digest(output / "compilation.json"),
            "sources": bench.generated_sources(output), "objects": bench.numerical_objects(output),
            "executable_sha256": bench.digest(directory / "check"),
            "effective_compiler_flags": capture_effective_flags(directory, env=env),
            "dependencies": bench.dependency_inputs(directory), "runtime": bench.runtime_identity(directory),
            "physical_card_sha256": bench.digest(output / "Cards/param_card.dat"),
            "numerical_cards": bench.numerical_card_inputs(output)}


def verify_unchanged(before, after):
    for key in ("directory", "generation_sha256", "compilation_sha256", "sources", "objects",
                "executable_sha256", "dependencies", "physical_card_sha256", "numerical_cards"):
        if before[key] != after[key]:
            raise ValueError("Baseline input changed: " + key)
    if not bench.same_compiler_policy(before["effective_compiler_flags"], after["effective_compiler_flags"]):
        raise ValueError("Baseline effective compiler policy changed")
    for section in ("libraries", "archives", "metadata"):
        if before["runtime"][section] != after["runtime"][section]:
            raise ValueError("Baseline runtime changed: " + section)


def case_proof(case, env):
    original = bench.LEGACY_WORK / "outputs/baseline" / case["id"]
    evidence = REPORT / case["id"]
    evidence.mkdir(exist_ok=False)
    before = snapshot(original, env)
    bench.dump(evidence / "inputs-before.json", before)
    generation = json.loads((original / "generation.json").read_text())
    compilation = json.loads((original / "compilation.json").read_text())
    if (generation["source"] != "baseline" or generation["case"] != case
            or generation["source_identity"]["revision"] != bench.BASELINE_REVISION
            or compilation.get("numerical_build_from_sources") is not True
            or compilation["sources"] != before["sources"]
            or compilation["executable_sha256"] != before["executable_sha256"]
            or compilation.get("link_inputs_before") != before["dependencies"]
            or compilation.get("link_inputs_after") != before["dependencies"]
            or not bench.same_compiler_policy(compilation.get("effective_compiler_flags"), before["effective_compiler_flags"])
            or any(original.rglob("worker-protocol.fifo"))):
        raise ValueError("Historical baseline source/build/dependency binding is not intact")
    clone = WORK / case["id"] / "export"
    shutil.copytree(original, clone, symlinks=True, ignore=shutil.ignore_patterns(
        "worker-raw", "build-history", "*.fifo", "timing.dat", "values.dat", "timed_status.dat", "dimensions.dat", "warmup.dat"))
    if bench.generated_sources(clone) != before["sources"] or bench.numerical_objects(clone) != before["objects"]:
        raise ValueError("Private copy differs from frozen baseline sources or objects")
    directories = sorted(clone.glob("SubProcesses/P*_*"))
    clone_flags = capture_effective_flags(directories[0], env=env)
    if not bench.same_compiler_policy(before["effective_compiler_flags"], clone_flags):
        raise ValueError("Private copy effective compiler policy changed")
    clone_dependencies = bench.dependency_inputs(directories[0])
    mapped_dependencies = {section: {str(path).replace(str(clone), str(original), 1): checksum
                                   for path, checksum in values.items()}
                           for section, values in clone_dependencies.items()}
    if mapped_dependencies != before["dependencies"]:
        raise ValueError("Private copy dependency bytes differ from original")
    bench.dump(evidence / "private-copy.json", {"directory": str(clone), "sources_identical": True,
        "objects_identical": True, "effective_compiler_flags": clone_flags,
        "dependencies": clone_dependencies, "dependency_path_mapping": {str(clone): str(original)},
        "recorded_original_export_environment": generation.get("export_environment"),
        "original_export_environment_recorded": "export_environment" in generation,
        "relink_environment": {"PYTHONHASHSEED": env["PYTHONHASHSEED"], "MAKEFLAGS": env["MAKEFLAGS"]}})
    proof_directory = WORK / case["id"] / "proof"
    proof = prove_object_relink(directories[0], before["executable_sha256"], proof_directory, env)
    after = snapshot(original, env)
    bench.dump(evidence / "inputs-after.json", after)
    verify_unchanged(before, after)
    if bench.generated_sources(clone) != before["sources"] or bench.numerical_objects(clone) != before["objects"]:
        raise ValueError("Private-copy numerical source or objects changed while linking")
    for name in ("proof.json", "make.log", "link.log"):
        shutil.copy2(proof_directory / name, evidence / name)
    result = {"case": case, "status": "passed", "baseline_revision": bench.BASELINE_REVISION,
              "original_directory": str(original), "private_directory": str(clone),
              "original_inputs_unchanged": True, "private_objects_unchanged": True,
              "expected_executable_sha256": before["executable_sha256"],
              "relinked_executable_sha256": proof["relinked_executable_sha256"],
              "performance_acceptance_eligible": False, "native_amplitude_calls": 0,
              "numerical_compilation_commands": 0, "link_commands": 1,
              "historical_export_environment_recorded": "export_environment" in generation,
              "files": {path.name: bench.digest(path) for path in sorted(evidence.iterdir()) if path.is_file()}}
    bench.dump(evidence / "result.json", result)
    return result


def main():
    if ROOT != bench.ROOT or CPU not in os.sched_getaffinity(0):
        raise SystemExit("Run from the repository with allowed CPU12")
    os.sched_setaffinity(0, {CPU})
    WORK.mkdir(parents=True, exist_ok=False)
    protocols = REPORT / "protocol-sources"
    protocols.mkdir(exist_ok=False)
    for name in PROTOCOLS:
        shutil.copy2(ROOT / "tools" / name, protocols / name)
    source_hashes = {name: bench.digest(protocols / name) for name in PROTOCOLS}
    env = bench.environment()
    env["MAKEFLAGS"] += " -j1"
    specification = {"schema": "madloop-baseline-private-relink-preflight-v1", "created_utc": bench.utc_now(),
        "baseline_revision": bench.BASELINE_REVISION, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "jobs": 1, "performance_acceptance_eligible": False, "native_amplitude_calls": 0,
        "scope": "Private copies and untouched-driver exact relink only; original baseline read-only",
        "protocols": source_hashes, "runner_sha256": bench.digest(__file__),
        "configuration_sha256": bench.digest(bench.LEGACY_REPORT / "cases.json"),
        "relink_environment": {"PYTHONHASHSEED": env["PYTHONHASHSEED"], "MAKEFLAGS": env["MAKEFLAGS"]}}
    bench.dump(REPORT / "specification.json", specification)
    cases = bench.load_config()["cases"]
    results = []
    for case in cases:
        try:
            result = case_proof(case, env)
        except Exception as error:
            result = {"case": case, "status": "failed", "error": repr(error),
                      "performance_acceptance_eligible": False, "native_amplitude_calls": 0}
            bench.dump(REPORT / case["id"] / "failure.json", result)
        results.append(result)
        bench.dump(REPORT / "results.json", {"results": results, "performance_acceptance_eligible": False,
                                            "native_amplitude_calls": 0})
        print(json.dumps({"case": case["id"], "status": result["status"], "error": result.get("error")}), flush=True)
    if source_hashes != {name: bench.digest(ROOT / "tools" / name) for name in PROTOCOLS}:
        raise ValueError("Preflight protocol sources changed during execution")
    if any(result["status"] != "passed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
