#!/usr/bin/env python3
"""Read fresh generated trees and retain their complete numerical runtimes."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers


def audit(generation):
    generation = policy.path(generation)
    status = policy.read(generation / "status.json")
    if not all(status.get(key) is True for key in
               ("complete", "preserved_existing_inputs", "dependencies_unchanged", "original_archive_unchanged")):
        raise RuntimeError("Fresh generation or preservation checks are incomplete")
    if not status.get("dependencies"):
        raise RuntimeError("Generation dependency hashes are missing")
    for record in status["dependencies"].values():
        policy.check_file(record)
    config_cases = policy.read(generation / "cases.json")["cases"]
    expected = {case["id"] for case in config_cases}
    if len(expected) != 8 or set(status["cases"]) != expected:
        raise RuntimeError("Expected exactly eight fresh core exports")
    cases, groups = {}, {}
    for case_id in sorted(expected):
        record = status["cases"][case_id]
        if record["status"] != "complete":
            raise RuntimeError("Incomplete export: " + case_id)
        output = policy.path(record["output"])
        retained = generation / "inputs/candidate" / case_id
        compiled = policy.read(retained / "compiled-sources.json")
        live = policy.manifest(output)
        if live != compiled:
            raise RuntimeError("Generated sources changed: " + case_id)
        generated = policy.read(retained / "generation.json")
        identity = generated["source_identity"]
        if (identity["revision"] != status["git_revision"] or
                identity["tree_state_sha256"] != status["generator_tree_state_sha256"] or
                policy.read(output / "generation.json") != generated):
            raise RuntimeError("Generation identity changed: " + case_id)
        if compiled["files_sha256"] != record["compilation"]["generated_sources_sha256"]:
            raise RuntimeError("Compiled-source manifest differs: " + case_id)
        executables = list(output.glob("SubProcesses/P*_*/check"))
        if (len(executables) != 1 or workers.digest(executables[0]) !=
                record["compilation"]["executable_sha256"]):
            raise RuntimeError("Clean-build executable changed: " + case_id)
        libraries = policy.library_hashes(output)
        metadata = {name: policy.bound_file(filename)
                    for name, filename in policy.runtime_metadata_files(output).items()}
        signatures = {}
        for backend in (1, 6, 7, 8):
            signature = policy.reducer_signature({
                "backend": backend, "libraries": libraries,
                "runtime_metadata": metadata, "compiled_manifest": compiled})
            signatures[str(backend)] = signature["identity_sha256"]
            groups.setdefault(str(backend), {}).setdefault(
                signature["identity_sha256"], {"signature": signature, "cases": []}
            )["cases"].append(case_id)
        cases[case_id] = {
            "output": policy.relative(output),
            "generation": policy.bound_file(retained / "generation.json"),
            "compiled_manifest": policy.bound_file(retained / "compiled-sources.json"),
            "generation_source_identity": identity,
            "executable_sha256": record["compilation"]["executable_sha256"],
            "libraries": libraries,
            "runtime_metadata": {name: item["sha256"] for name, item in metadata.items()},
            "signatures": signatures,
            "has_iregi": any(Path(name).name == "libiregi.a" for name in libraries),
        }
    return policy.seal({
        "schema": "fresh-generated-runtime-inventory-v2",
        "status": policy.bound_file(generation / "status.json"),
        "audit_source": policy.bound_file(Path(__file__)),
        "case_count": len(cases), "cases": cases, "runtime_groups": groups,
        "generation_dependencies_rehashed": status["dependencies"],
        "single_common_group_per_backend": all(len(value) == 1 for value in groups.values()),
        "scope": "Live generated sources and complete libraries/modules/link inputs; "
                 "no build, evaluation, proof promotion or timing acceptance.",
        "acceptance_evidence": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, default=ROOT /
                        "docs/benchmarkfolder/policy-v2/final-core-generation-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if policy.path(args.output).exists():
        raise RuntimeError("Preserve the prior inventory; choose a fresh output")
    result = audit(args.generation)
    policy.bench.dump(policy.path(args.output), result)
    print(result["identity_sha256"])
    if not result["single_common_group_per_backend"]:
        raise SystemExit("Multiple exact runtime groups: extend representative coverage before proof")
