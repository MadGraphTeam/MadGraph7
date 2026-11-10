#!/usr/bin/env python3
"""Verify the retained baseline proof ledger without running a linker or binary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def collect():
    specification = load(ROOT / "specification.json")
    if (specification["cpu_affinity"] != [12] or specification["jobs"] != 1
            or specification["performance_acceptance_eligible"] is not False
            or specification["native_amplitude_calls"] != 0
            or digest(ROOT / "run.py") != specification["runner_sha256"]):
        raise ValueError("Changed preflight scope or executed runner")
    for name, checksum in specification["protocols"].items():
        if digest(ROOT / "protocol-sources" / name) != checksum:
            raise ValueError("Changed retained helper: " + name)
    path = ROOT / "protocol-sources/tirust_benchmark_compiler.py"
    spec = importlib.util.spec_from_file_location("frozen_compiler_policy", path)
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    results = load(ROOT / "results.json")
    if len(results["results"]) != 11 or results["native_amplitude_calls"] != 0:
        raise ValueError("Incomplete eleven-case proof ledger")
    cases = []
    for result in results["results"]:
        case_id = result["case"]["id"]
        directory = ROOT / case_id
        if result != load(directory / "result.json") or result["status"] != "passed":
            raise ValueError("Incomplete case result: " + case_id)
        for name, checksum in result["files"].items():
            if digest(directory / name) != checksum:
                raise ValueError("Changed case evidence: " + case_id + "/" + name)
        before, after = (load(directory / name) for name in ("inputs-before.json", "inputs-after.json"))
        for key in ("directory", "generation_sha256", "compilation_sha256", "sources", "objects",
                    "executable_sha256", "dependencies", "physical_card_sha256", "numerical_cards"):
            if before[key] != after[key]:
                raise ValueError("Original input mutation: " + case_id + "/" + key)
        compiler.validate_compiler_equality(before["effective_compiler_flags"], after["effective_compiler_flags"])
        for section in ("libraries", "archives", "metadata"):
            if before["runtime"][section] != after["runtime"][section]:
                raise ValueError("Original runtime mutation: " + case_id)
        private, proof = (load(directory / name) for name in ("private-copy.json", "proof.json"))
        compiler.validate_compiler_equality(before["effective_compiler_flags"], private["effective_compiler_flags"])
        for scope in before["effective_compiler_flags"]["scopes"]:
            keys = set(before["effective_compiler_flags"]["scopes"][scope]["recipes"])
            if any(set(record["scopes"][scope]["recipes"]) != keys for record in
                   (after["effective_compiler_flags"], private["effective_compiler_flags"])):
                raise ValueError("Numerical compiler recipe set changed")
        mappings = private["dependency_path_mapping"]
        if mappings != {private["directory"]: before["directory"]}:
            raise ValueError("Changed private dependency path map")
        mapped = {section: {name.replace(private["directory"], before["directory"], 1): checksum
                           for name, checksum in values.items()}
                  for section, values in private["dependencies"].items()}
        if mapped != before["dependencies"]:
            raise ValueError("Private dependency bytes changed")
        process = Path(proof["working_directory"]).relative_to(private["directory"])
        expected_objects = {Path(name).name: checksum for name, checksum in before["objects"].items()
                            if Path(name).parent == process and Path(name).suffix == ".o"}
        if proof["objects"] != expected_objects or not all(proof[key] is True for key in
                ("passed", "objects_unchanged", "original_executable_unchanged")):
            raise ValueError("Incomplete object/executable equality proof")
        hashes = [before["executable_sha256"], proof["expected_executable_sha256"],
                  proof["relinked_executable_sha256"], result["expected_executable_sha256"],
                  result["relinked_executable_sha256"]]
        if len(set(hashes)) != 1 or proof["protocol_sha256"] != specification["protocols"]["tirust_benchmark_relink.py"]:
            raise ValueError("Executable or link proof protocol changed")
        if (result["native_amplitude_calls"] != 0 or result["numerical_compilation_commands"] != 0
                or result["link_commands"] != 1 or result["performance_acceptance_eligible"] is not False
                or not private["sources_identical"] or not private["objects_identical"]):
            raise ValueError("Preflight scope inflated")
        command = proof["proof_link_command"]
        original = proof["original_link_command"]
        normalized = list(command)
        normalized[normalized.index("-o") + 1] = "check"
        if normalized != original or "-c" in command or "--dry-run" not in proof["make_command"]:
            raise ValueError("Unexpected compilation or altered link recipe")
        cases.append({"id": case_id, "status": "passed", "executable_sha256": hashes[0],
                      "original_inputs_unchanged": True, "native_amplitude_calls": 0,
                      "numerical_compilation_commands": 0, "link_commands": 1})
    if len({case["id"] for case in cases}) != 11:
        raise ValueError("Duplicate case")
    files = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(ROOT.rglob("*")) if path.is_file()
             and path.name != "index.json" and "__pycache__" not in path.parts}
    return {"schema": "madloop-baseline-private-relink-index-v1", "status": "passed",
            "scope": "Retained text-ledger verification only; does not rerun links or validate later filesystem state",
            "baseline_revision": specification["baseline_revision"], "cpu_affinity": [12], "jobs": 1,
            "cases": cases, "performance_acceptance_eligible": False, "native_amplitude_calls": 0,
            "numerical_compilation_commands": 0, "link_commands": 11, "files": files}


if __name__ == "__main__":
    result = collect()
    path = ROOT / "index.json"
    if sys.argv[1:] == ["--create"]:
        with path.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
    elif sys.argv[1:]:
        raise SystemExit("Use no arguments to verify; --create exclusively publishes the first index")
    elif result != load(path):
        raise SystemExit("Frozen preflight index differs")
    print(json.dumps({"status": "passed", "cases": len(result["cases"]),
                      "files": len(result["files"]), "index_sha256": digest(path)}))
