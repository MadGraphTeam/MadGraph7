"""Read-only audit of the two retained historical c91 CPU9 chain diagnostics.

Run from the repository root. Only the sibling audit JSON is created; no probe
evidence, executable, native numerical job, or compiler is modified or started.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(kind):
    directory = HERE / f"historical-c91-{kind}-n2-chain-cpu9-1"
    spec, result = read(directory / "specification.json"), read(directory / "summary.json")
    prep, cal = read(directory / "preparation.json"), read(directory / "calibration.json")
    config, metadata = read(directory / "inputs/cases.json"), read(directory / "inputs/points.json")
    sources = {source: read(directory / f"{source}-inputs.json") for source in ("baseline", "candidate")}
    require(result["status"] == "complete" and result["numerical_pass"], "Incomplete diagnostic")
    require(not (directory / "failure.json").exists(), "Retained failure accompanies summary")
    for record in (spec, result, prep, read(directory / "run-start.json"), read(directory / "scheduling.json")):
        require(record["performance_acceptance_eligible"] is False, "Diagnostic eligibility changed")
    require(spec["pairs"] == 7 and config["points_per_case"] == 100, "Wrong pair/point count")
    require(spec["machine"]["selected_cpu"] == 9, "Wrong selected CPU")
    require(spec["baseline_chain"] == [8, 6, 7, 1] and spec["candidate_chain"] == [9, 8, 6, 7, 1],
            "Wrong reduction chains")
    for folder, entries in (("protocol-sources", spec["protocols"]), ("inputs", spec["inputs"])):
        for name, checksum in entries.items():
            require(digest(directory / folder / name) == checksum, f"Changed retained {folder}/{name}")
    require(metadata["sha256"] == digest(directory / "inputs/points.dat"), "Point stream hash changed")
    require(metadata["inputs"]["count"] == 100 and len(metadata["points"]) == 100, "Wrong saved points")
    points_tool = load("audit_points", ROOT / "tools/tirust_benchmark_points.py")
    points_tool.validate_points(metadata["points"], metadata["inputs"]["masses"],
                                incoming_masses=metadata["inputs"]["incoming_masses"])

    # The retained parser/protocol, not a newer implementation, defines these
    # observations. Importing these modules does not launch native work.
    protocols = directory / "protocol-sources"
    worker = load("audit_worker_" + kind, protocols / "tirust_benchmark_worker.py")
    bench = load("audit_bench_" + kind, protocols / "tirust_benchmark.py")
    compiler = load("audit_compiler_" + kind, protocols / "tirust_benchmark_compiler.py")
    observations, calibration = {}, {}
    total_reparsed = 0
    for source in sources:
        chain = spec[source + "_chain"]
        observations[source] = []
        for raw in sorted((directory / "raw" / source).iterdir()):
            saved = read(raw / "observation.json")
            evaluations = sum(saved["timed_status_counts"].values())
            require(evaluations % 100 == 0, "Nonintegral number of traversals")
            loops = evaluations // 100
            if raw.name == "calibrate-0":
                require(loops == 1, "First calibration was not one traversal")
            elif raw.name == "calibrate-1":
                first = read(raw.with_name("calibrate-0") / "observation.json")
                require(loops == max(1, math.ceil(spec["calibration_target_seconds"] / first["seconds"])),
                        "Second calibration differs from registered procedure")
                require(cal["loops"][source] == max(1, math.ceil(
                    loops * spec["calibration_target_seconds"] / saved["seconds"])), "Repetition loops changed")
            else:
                require(raw.name.startswith("repetition-") and loops == cal["loops"][source], "Wrong traversal count")
            actual = worker.parse_outputs(raw, chain, 100, loops)
            require(actual == saved, f"Raw parser differs from {source}/{raw.name}/observation.json")
            if raw.name == "calibrate-1":
                calibration[source] = actual
            elif raw.name.startswith("repetition-"):
                repetition = int(raw.name.split("-")[-1])
                require(repetition == len(observations[source]), "Repetitions missing or unordered")
                require(actual == result["raw_observations"][source][repetition], "Summary observation differs")
                observations[source].append(actual)
            total_reparsed += 1
        require(len(observations[source]) == 7, "Missing repetitions")

    comparisons = [bench.compare_rows(calibration["baseline"]["rows"], calibration["candidate"]["rows"],
                                     config["relative_tolerance"])]
    comparisons.extend(bench.compare_rows(left["rows"], right["rows"], config["relative_tolerance"])
                       for left, right in zip(observations["baseline"], observations["candidate"]))
    require(comparisons == result["comparisons"] and comparisons[0] == cal["comparison"],
            "Numerical comparison does not reproduce")
    require(all(item["passed"] and item["ordinary_target_passed"] for item in comparisons), "Numerical mismatch")
    timings = {source: [item["seconds_per_point"] for item in values] for source, values in observations.items()}
    require(timings == result["timings"], "Timings do not reproduce")
    statistics_result = worker.paired_statistics(timings["baseline"], timings["candidate"])
    for key in ("passed", "acceptance_eligible", "threshold", "minimum_pairs"):
        statistics_result.pop(key)
        require(key not in result["statistics"], "Formal decision leaked into diagnostic statistics")
    statistics_result["median_seconds_per_point"] = {source: statistics.median(values) for source, values in timings.items()}
    require(statistics_result == result["statistics"], "Descriptive statistics do not reproduce")
    usage = {source: bench.aggregate_usage(values) for source, values in observations.items()}
    require(usage == result["actual_backend_precision_counts"], "Accepted backend counts differ")
    require(usage == {"baseline": {"8:DP": 3500}, "candidate": {"9:DP": 3500}}, "Unexpected accepted precision/backend")
    require(result["usage_scope"] == "final accepted amplitude results; internal attempts are not counted", "Usage scope changed")
    duration_ok = all(item["seconds"] >= spec["minimum_descriptive_batch_seconds"]
                      for values in observations.values() for item in values)
    require(duration_ok == result["minimum_descriptive_batch_duration_met"], "Duration label differs")
    schedule = [(source, f"calibrate-{index}") for source in sources for index in range(2)]
    schedule += [(source, f"repetition-{index}") for index in range(7)
                 for source in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline"))]
    require([(item["source"], item["phase"]) for item in result["host_load"]] == schedule,
            "Recorded execution order was not alternating")

    compiler.validate_compiler_equality(sources["baseline"]["compilation"]["effective_compiler_flags"],
                                       sources["candidate"]["compilation"]["effective_compiler_flags"])
    source_proofs = {}
    for source, inputs in sources.items():
        compilation, generation = inputs["compilation"], inputs["generation"]
        clone = prep["clones"][source]
        output = Path(clone["private_output"])
        subdirs = list(output.glob("SubProcesses/P*_*"))
        require(len(subdirs) == 1, "Ambiguous subprocess")
        require(clone["compiled_or_relinked"] is False and compilation["numerical_build_from_sources"] is True,
                "Unproven source-built clone")
        require(generation["case"] == spec["case"] and generation["source"] == source, "Wrong generated process")
        for name, expected in (("generation", generation), ("compilation", compilation)):
            require(read(output / (name + ".json")) == expected, "Build provenance changed")
            require(digest(output / (name + ".json")) == clone[name + "_sha256"], "Prepared provenance hash changed")
        require(digest(subdirs[0] / "check") == compilation["executable_sha256"] == clone["executable_sha256"],
                "Numerical executable changed")
        require(bench.generated_sources(output) == compilation["sources"] == clone["generated_sources"],
                "Generated source inventory changed")
        for name, checksum in clone["numerical_objects"].items():
            require(digest(output / name) == checksum, "Prepared numerical object changed")
        bench.verify_worker_template(compilation, protocols / "tirust_benchmark_worker.f")
        require(inputs["diagnostic_worker_template"]["performance_acceptance_eligible"] is False,
                "Historical worker mislabeled")
        require(inputs["diagnostic_worker_template"]["sha256"] == spec["protocols"]["tirust_benchmark_worker.f"],
                "Historical template provenance differs")
        for key in ("libraries", "archives", "metadata"):
            require(inputs["runtime"][key] == clone["runtime"][key] == result["inputs_after"][source]["runtime"][key],
                    "Runtime changed between preparation/start/end")
            for path, checksum in inputs["runtime"][key].items():
                require(digest(path) == checksum, "Retained runtime file no longer matches: " + path)
        dependencies = {key: inputs["runtime"][key] for key in ("archives", "metadata")}
        require(compilation["link_inputs_before"] == compilation["link_inputs_after"] == dependencies,
                "Runtime differs from source-build link-time dependencies")
        for path in (output / "Cards/param_card.dat", Path(inputs["resolved_parameter_card"])):
            require(digest(path) == inputs["param_card_sha256"] == generation["parameter_card_sha256"], "Physical card changed")
        require(bench.numerical_card_inputs(output) == inputs["numerical_cards"] == result["inputs_after"][source]["numerical_cards"],
                "Numerical settings cards changed")
        require(inputs["points_sha256"] == metadata["sha256"] == result["inputs_after"][source]["points_sha256"],
                "Frozen point binding differs")
        require(inputs["points_metadata_sha256"] == digest(directory / "inputs/points.json"), "Point metadata changed")
        require(compilation["masses"] == metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"], "External masses differ")
        require(bench.parse_chain(inputs["settings"]["mlreductionlib"]) == spec[source + "_chain"], "Backend card mismatch")
        identity = generation["source_identity"]
        if source == "baseline":
            frozen = read(ROOT / "docs/benchmarkfolder/inputs/baseline.json")
            require(identity["revision"] == frozen["revision"] == "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f", "Wrong baseline")
            require(identity["archive_sha256"] == frozen["archive_sha256"] and identity["files_sha256"] == canonical(frozen["files"]),
                    "Baseline source identity differs from frozen record")
        else:
            require(identity["revision"] == "c91cb85f3aa942c90299b3518bb1f195f55334a5" and identity["dirty"] is False,
                    "Wrong historical candidate identity")
            require(identity["files_sha256"] == canonical(identity["files"]), "Candidate source inventory digest differs")
        source_proofs[source] = {"revision": identity["revision"], "files_sha256": identity["files_sha256"],
                                 "generated_sources_checked": len(compilation["sources"]),
                                 "numerical_objects_checked": len(clone["numerical_objects"]),
                                 "executable_sha256": compilation["executable_sha256"]}
    require(sources["baseline"]["param_card_sha256"] == sources["candidate"]["param_card_sha256"], "Physics cards differ")
    common_settings = sources["baseline"]["settings"].keys() & sources["candidate"]["settings"].keys() - {"mlreductionlib"}
    require(all(sources["baseline"]["settings"][key] == sources["candidate"]["settings"][key] for key in common_settings),
            "Shared numerical policies differ")
    for category in ("libraries", "archives", "metadata"):
        left, right = sources["baseline"]["runtime"][category], sources["candidate"]["runtime"][category]
        require(all(left[key] == right[key] for key in left.keys() & right.keys()), "Shared runtime bytes differ")
    return {"label": spec["label"], "audit_status": "verified", "summary_sha256": digest(directory / "summary.json"),
            "raw_observations_reparsed": total_reparsed, "numerical_comparisons_recomputed": len(comparisons),
            "maximum_relative_error": max(item["maximum_relative_error"] for item in comparisons),
            "ordinary_target_passed": True, "statistics": statistics_result, "accepted_backend_precision_counts": usage,
            "source_proofs": source_proofs, "performance_acceptance_eligible": False,
            "minimum_descriptive_batch_duration_met": duration_ok}


if __name__ == "__main__":
    report = {"schema": "madloop-chain-probe-independent-audit-v1", "audit_scope": "read-only retained-evidence audit",
              "native_or_compiler_jobs_started": False, "performance_acceptance_eligible": False,
              "probes": [audit(kind) for kind in ("qcd", "ew")],
              "host_context_files": {path.name: digest(path) for path in sorted(HERE.glob("shared-host-*.json"))},
              "limitations": ["Historical c91 exports, not the final optimized candidate or formal campaign.",
                              "Backend/precision counts describe final accepted results, not internal reduction attempts.",
                              "An accepted DP status is not proof that NinjaRS made no internal compensated retry.",
                              "Retained host-context notes report severe shared-host conditions; no individual batch cause is inferred.",
                              "Source identity is checked against retained manifests and source-built private outputs; this audit does not rebuild historical source."]}
    destination = HERE / "audit-n2-probes.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit": str(destination.relative_to(ROOT)), "probes": [
        {key: probe[key] for key in ("label", "audit_status", "raw_observations_reparsed", "maximum_relative_error")}
        for probe in report["probes"]]}, sort_keys=True))
