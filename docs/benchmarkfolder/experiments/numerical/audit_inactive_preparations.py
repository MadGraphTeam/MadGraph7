#!/usr/bin/env python3
"""Audit three inactive preparations without reading active outputs or executing code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "inactive-preparation-audit.json"
LABELS = ("strict-policy-ew-n3", "strict-policy-ew-n4", "strict-policy-qcd-n4")
LANES = (("baseline", 1), ("candidate", 1), ("candidate", 6),
         ("candidate", 7), ("candidate", 8), ("candidate", 9))
TEMPLATE_ERROR = "RuntimeError('Worker template differs from compilation')"
ADMIN_ERROR = 'ProtocolError("Invalid READY message: [\'ERROR\', \'administrative-stop-for-warmup-diagnostics\']")'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def load(path):
    return json.loads(path.read_text())


def retained_files():
    result = {}
    for label in LABELS:
        for path in sorted((HERE / label).rglob("*")):
            require(not path.is_symlink(), "Evidence symlink: " + str(path))
            if path.is_file():
                result[str(path.relative_to(HERE))] = record(path)
    return result


def recorded_input(folder, source, specification, config):
    """Check internal bindings; the historical binaries/cards are not re-read."""
    path = folder / f"{source}-inputs.json"
    inputs = load(path)
    compiled, generated = inputs["compilation"], inputs["generation"]
    require(generated["case"] == specification["case"] and generated["source"] == source,
            "Generation case/source differs")
    require(inputs["param_card_sha256"] == generated["parameter_card_sha256"],
            "Recorded parameter-card identities differ")
    require(compiled["numerical_build_from_sources"] is True and
            compiled["link_inputs_before"] == compiled["link_inputs_after"],
            "Recorded numerical build/link inputs differ")
    template = folder.parent / "tirust_benchmark_worker.f"
    require(record(template)["sha256"] == compiled["worker_template_sha256"],
            "Aborted worker does not match its retained template")
    parameters = compiled["worker_render_parameters"]
    rendered = (template.read_text().replace("@PREFIX@", parameters["prefix"])
                .replace("@NEXTERNAL@", str(parameters["nexternal"]))
                .replace("@NBORN@", str(parameters["nborn"])))
    driver_hash = hashlib.sha256(rendered.encode()).hexdigest()
    driver_inventory = [value for name, value in compiled["sources"].items()
                        if re.fullmatch(r"SubProcesses/P[^/]+/check_sa\.f", name)]
    require(driver_hash == compiled["driver_sha256"] and driver_inventory == [driver_hash],
            "Retained template rendering differs from recorded driver")
    expected = dict(config["madloop_parameters"])
    if source == "candidate":
        expected.update(config["candidate_parameters"])
    require(inputs["settings"]["mlreductionlib"] == "1" and all(
            inputs["settings"][name.lower()] == value for name, value in expected.items()),
            "Aborted worker policy differs from frozen configuration")
    return {"manifest": str(path.relative_to(HERE)),
            "source_revision": generated["source_identity"]["revision"],
            "generation_identity_recorded": {key: value for key, value in
                generated["source_identity"].items() if key != "files"},
            "generation_source_file_hashes_recorded": len(generated["source_identity"].get("files", {})),
            "points_sha256": inputs["points_sha256"],
            "parameter_card_sha256": inputs["param_card_sha256"],
            "executable_sha256": compiled["executable_sha256"],
            "driver_sha256": driver_hash,
            "worker_template_sha256": compiled["worker_template_sha256"],
            "recorded_generated_source_files": len(compiled["sources"]),
            "internal_identity_checks_pass": True,
            "historical_binary_card_and_point_bytes_revalidated": False}


def audit():
    inventory = retained_files()
    require(len(inventory) == 48 and not any(Path(name).name in
            {"values.dat", "warmup.dat", "timing.dat", "timed_status.dat", "dimensions.dat",
             "result.json", "observation.json"} for name in inventory),
            "Inactive inventory changed or unexpectedly contains numerical observations")
    cases, totals = [], {"preflight_template_mismatches": 0, "administrative_aborts": 0}
    for label in LABELS:
        folder = HERE / label
        specification, results, config = (load(folder / name) for name in
                                          ("specification.json", "results.json", "cases.json"))
        require(specification["performance_eligible"] is False and
                results["performance_eligible"] is False, "Preparation promoted to acceptance")
        require(record(folder / "cases.json")["sha256"] == specification["configuration_sha256"],
                "Frozen configuration hash differs")
        for name, checksum in specification["protocols"].items():
            require(Path(name).name == name and record(folder / name)["sha256"] == checksum,
                    "Frozen protocol hash differs: " + name)
        require(results["case"] == specification["case"] and
                specification["case"] in config["cases"], "Case binding differs")
        require([(r["source"], r["backend"]) for r in results["results"]] == list(LANES),
                "Unexpected preparation lane set/order")
        # Inspect the retained guard to identify where the recorded error arose;
        # do not import or execute historical protocol code.
        tree = ast.parse((folder / "tirust_benchmark.py").read_text())
        guard = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "verify_worker_template")
        require(any(isinstance(node, ast.Constant) and node.value ==
                    "Worker template differs from compilation" for node in ast.walk(guard)),
                "Retained source lacks the reported preflight guard")
        lanes = []
        for outcome in results["results"]:
            source, backend = outcome["source"], outcome["backend"]
            lane = folder / f"{source}-{backend}"
            require(load(lane / "failure.json") == outcome and outcome["status"] == "failed"
                    and outcome["performance_eligible"] is False, "Failure/result disagreement")
            require(not (lane / "result.json").exists() and not (lane / "raw").exists(),
                    "Inactive lane unexpectedly contains numerical observations")
            is_admin = label == "strict-policy-qcd-n4" and backend == 1
            require(outcome["error"] == (ADMIN_ERROR if is_admin else TEMPLATE_ERROR),
                    "Unrecognized stop reason")
            kind = "administrative_abort_before_ready" if is_admin else "worker_template_preflight_mismatch"
            totals["administrative_aborts" if is_admin else "preflight_template_mismatches"] += 1
            item = {"source": source, "backend": backend, "category": kind,
                    "original_status": outcome["status"], "original_error": outcome["error"],
                    "physics_result": "unassessed", "performance_eligible": False,
                    "complete_point_evaluations_retained": 0,
                    "missing_evidence": ["raw coefficients", "completed point/status stream", "timing batches"]}
            if is_admin:
                name = "administrative-stop.json" if source == "baseline" else "administrative-stop-candidate.json"
                stop = load(folder / name)
                require("No numerical failure inferred" in stop["reason"], "Missing administrative qualification")
                item["administrative_record"] = {"path": str((folder / name).relative_to(HERE)), **stop}
                item["recorded_inputs"] = recorded_input(lane, source, specification, config)
                item["stdout"] = str((lane / f"{source}-output.txt").relative_to(HERE))
                require((lane / f"{source}-output.txt").is_file(), "Missing retained worker stdout")
                item["missing_evidence"] += ["in-folder point/card bytes", "warmup point/status trace"]
            else:
                require(not (lane / f"{source}-inputs.json").exists() and
                        not (lane / f"{source}-output.txt").exists(), "Unexpected preflight native evidence")
                item["missing_evidence"] += ["lane input/build/runtime manifest", "worker stdout",
                                              "exact conflicting compilation/template hashes"]
            lanes.append(item)
        cases.append({"label": label, **specification["case"],
                      "requested_points": config["points_per_case"],
                      "configuration_sha256": specification["configuration_sha256"],
                      "protocols": specification["protocols"],
                      "case_file_count": sum(name.startswith(label + "/") for name in inventory),
                      "lanes": lanes})
    require(inventory == retained_files(), "Original preparation evidence changed during audit")
    return {"schema_version": 1, "scope": "Inactive preparation/protocol stops only; no accuracy or timing claims",
            "audited_labels": list(LABELS), "active_warmup_v2_outputs_read": False,
            "native_executions": 0, "original_files_modified": False,
            "complete_physics_runs": 0, "performance_eligible": False,
            "totals": {**totals, "attempts": len(cases), "lanes": len(cases) * len(LANES),
                       "retained_files": len(inventory)},
            "cases": cases, "files": inventory,
            "audit_program": {"path": Path(__file__).name, **record(Path(__file__))},
            "readme": {"path": "README-inactive-preparations.md",
                       **record(HERE / "README-inactive-preparations.md")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create the new audit JSON; refuse to overwrite")
    args = parser.parse_args()
    result = audit()
    if args.write:
        with OUTPUT.open("x") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        require(result == load(OUTPUT), "Audit differs from frozen inactive-preparation-audit.json")
    print(json.dumps({**result["totals"], "all_internal_checks_pass": True,
                      "physics_result": "unassessed", "performance_eligible": False}, sort_keys=True))
