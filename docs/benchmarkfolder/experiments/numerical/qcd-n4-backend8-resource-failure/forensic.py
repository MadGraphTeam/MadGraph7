"""Audit retained QCD4 backend-8 raw data without rerunning its failed request.

Run from the repository root. The only output is sibling forensic.json. No
native job, compiler, runtime postcheck, active export, or backend-9 directory
is accessed. Original evidence and its failed status remain unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "strict-policy-qcd-n4-warmup-v2"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forensic():
    candidate, baseline = EVIDENCE / "candidate-8", EVIDENCE / "baseline-1"
    failure = read(candidate / "failure.json")
    require(failure == {"source": "candidate", "backend": 8, "status": "failed",
                       "performance_eligible": False,
                       "error": "BlockingIOError(11, 'Resource temporarily unavailable')"},
            "Original failure record changed")
    require(not (candidate / "result.json").exists(), "Original failed request acquired a completed result")
    require(not (candidate / "raw/observation.json").exists(), "Original failed request acquired an observation record")
    require(not (candidate / "inputs-after.json").exists(), "Unexpected completed post-measurement proof")
    spec, config = read(EVIDENCE / "specification.json"), read(EVIDENCE / "cases.json")
    require(spec["performance_eligible"] is False, "Original diagnostic eligibility changed")
    require(config["points_per_case"] == 100, "Wrong requested point count")
    require(spec["configuration_sha256"] == digest(EVIDENCE / "cases.json"), "Configuration hash changed")
    for name, checksum in spec["protocols"].items():
        require(digest(EVIDENCE / name) == checksum, "Retained controller/protocol hash changed")
    worker = load("forensic_worker", EVIDENCE / "tirust_benchmark_worker.py")
    bench = load("forensic_benchmark", EVIDENCE / "tirust_benchmark.py")
    compiler = load("forensic_compiler", EVIDENCE / "tirust_benchmark_compiler.py")
    reference = worker.parse_outputs(baseline / "raw", 1, 100, 1)
    observed = worker.parse_outputs(candidate / "raw", 8, 100, 1)
    baseline_result = read(baseline / "result.json")
    require(reference == read(baseline / "raw/observation.json") == baseline_result["observation"],
            "CutTools reference differs from retained raw outputs")
    require(baseline_result["status"] == "complete" and baseline_result["performance_eligible"] is False,
            "CutTools reference status changed")
    comparison = bench.compare_rows(reference["rows"], observed["rows"], config["relative_tolerance"])

    inputs = {"baseline": read(baseline / "baseline-inputs.json"),
              "candidate": read(candidate / "candidate-inputs.json")}
    compiler.validate_compiler_equality(inputs["baseline"]["compilation"]["effective_compiler_flags"],
                                       inputs["candidate"]["compilation"]["effective_compiler_flags"])
    points = ROOT / "docs/benchmarkfolder/points/qcd_n4_uux_zzgg.dat"
    metadata = read(points.with_suffix(".json"))
    require(digest(points) == metadata["sha256"] == inputs["baseline"]["points_sha256"]
            == inputs["candidate"]["points_sha256"], "Frozen input streams differ")
    require(metadata["inputs"]["count"] == 100, "Wrong frozen point count")
    require(inputs["baseline"]["param_card_sha256"] == inputs["candidate"]["param_card_sha256"],
            "Recorded physical parameter cards differ")
    shared = inputs["baseline"]["settings"].keys() & inputs["candidate"]["settings"].keys() - {"mlreductionlib"}
    require(all(inputs["baseline"]["settings"][key] == inputs["candidate"]["settings"][key] for key in shared),
            "Recorded shared numerical policies differ")
    provenance = {}
    for source, backend in (("baseline", 1), ("candidate", 8)):
        value = inputs[source]
        compilation, generation = value["compilation"], value["generation"]
        require(generation["case"] == spec["case"] and generation["source"] == source, "Wrong recorded process/source")
        require(generation["parameter_card_sha256"] == value["param_card_sha256"], "Recorded card changed since generation")
        require(bench.parse_chain(value["settings"]["mlreductionlib"]) == [backend], "Wrong selected backend")
        require(compilation["masses"] == metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"],
                "Recorded compiled masses differ from frozen points")
        require(compilation["numerical_build_from_sources"] is True, "Missing recorded source-build provenance")
        bench.verify_worker_template(compilation, EVIDENCE / "tirust_benchmark_worker.f")
        dependencies = {key: value["runtime"][key] for key in ("archives", "metadata")}
        require(compilation["link_inputs_before"] == compilation["link_inputs_after"] == dependencies,
                "Recorded link-time/startup dependencies differ")
        provenance[source] = {
            "source_revision": generation["source_identity"]["revision"],
            "source_files_sha256": generation["source_identity"]["files_sha256"],
            "recorded_executable_sha256": compilation["executable_sha256"],
            "recorded_generated_source_count": len(compilation["sources"]),
            "recorded_source_build": True,
            "recorded_link_time_startup_dependency_equality": True,
        }
    for category in ("libraries", "archives", "metadata"):
        left, right = inputs["baseline"]["runtime"][category], inputs["candidate"]["runtime"][category]
        require(all(left[key] == right[key] for key in left.keys() & right.keys()), "Recorded shared runtime hashes differ")

    evidence_paths = [EVIDENCE / name for name in spec["protocols"]]
    evidence_paths += [EVIDENCE / "specification.json", EVIDENCE / "cases.json", points, points.with_suffix(".json"),
                       candidate / "failure.json", candidate / "candidate-inputs.json", candidate / "candidate-output.txt",
                       baseline / "baseline-inputs.json", baseline / "result.json", baseline / "raw/observation.json"]
    for directory in (candidate, baseline):
        evidence_paths.extend(directory / "raw" / name for name in ("timing.dat", "values.dat", "timed_status.dat", "dimensions.dat"))
    evidence_paths.extend(candidate / "candidate-initialization" / name for name in ("warmup.dat", "dimensions.dat"))
    rows = observed["rows"]
    target_failures = [item for item in comparison["residuals"] if not (
        (item["symmetric_relative_error"] is not None and item["symmetric_relative_error"] < 1e-8)
        or (item["pole_to_finite"] is not None and item["pole_to_finite"] < 1e-8))]
    worst = max(comparison["residuals"], key=lambda item: item["relative_error"])
    return {
        "schema": "madloop-retained-resource-failure-forensic-v1",
        "forensic_status": "retained raw data internally consistent",
        "original_status": "failed", "original_error": failure["error"],
        "performance_eligible": False, "formal_numerical_qualification": False,
        "retrospective_promotion": False, "native_or_compiler_jobs_started": False,
        "postchecks_rerun": False, "original_completed_postverification_evidence": False,
        "raw_shape": {"points": len({row["point"] for row in rows}), "coefficient_rows": len(rows),
                      "answer_dimensions": observed["answer_dimensions"]},
        "accepted_status_counts": observed["status_counts"],
        "accepted_backend_precision_counts": observed["backend_precision_counts"],
        "accepted_qp_points": sorted({row["point"] for row in rows if row["precision"] == "QP"}),
        "retained_clock_values_seconds": {"initialization_and_warmup": observed["cold_seconds"],
                                           "single_traversal": observed["seconds"]},
        "cuttools_reference_reparse_matches_retained_result": True,
        "forensic_comparison": {key: value for key, value in comparison.items() if key not in ("residuals", "failures")},
        "regression_failure_count": len(comparison["failures"]),
        "ordinary_target_failure_count": len(target_failures),
        "ordinary_target_failure_points": sorted({item["point"] for item in target_failures}),
        "largest_raw_relative_residual": worst,
        "recorded_provenance_cross_checks": provenance,
        "bound_evidence_sha256": {str(path.relative_to(ROOT)): digest(path) for path in evidence_paths},
        "limitations": [
            "The original request remains failed. No result.json, raw/observation.json or completed post-measurement input-verification record exists for candidate-8.",
            "Raw outputs support this forensic comparison only; complete controller acknowledgement and graceful shutdown are not established by these retained files.",
            "The failure record has no traceback. BlockingIOError cannot be assigned to an exact operation from this evidence.",
            "The archived numerical controller contains no explicit post-observation verify_observation_inputs step. This audit neither reruns nor retroactively supplies post-measurement verification.",
            "Provenance checks compare retained pre-measurement/source-build records and frozen points. Current executable/library/card state is deliberately not postchecked.",
            "Accepted status counts describe final MadLoop results, not internal reduction attempts or retries.",
            "The stored clocks are one traversal plus initialization/warm-up. They are not paired timing evidence or a formal performance result.",
            "The 3e-6 comparison passes, but the 1e-8 ordinary-point target does not. Shared-host context does not establish the cause of any individual residual or timing."
        ],
    }


if __name__ == "__main__":
    record = forensic()
    with (HERE / "forensic.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: record[key] for key in (
        "original_status", "raw_shape", "accepted_backend_precision_counts",
        "ordinary_target_failure_count", "performance_eligible")}, sort_keys=True))
