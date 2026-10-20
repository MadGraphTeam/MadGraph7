#!/usr/bin/env python3
"""Read-only source/evidence verification; writes only this audit's snapshot."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
POLICY = ROOT / "docs/benchmarkfolder/policy-v2"
REVISION = "b44d20e725f1ec68589df20284e4ce9116e9e2ab"
PORT = ROOT / "docs/benchmarkfolder/experiments/t3-dp-retry-port"
INSTALL = POLICY / "install-export-abi3-terminal-fixed"
VARIANTS = POLICY / "final-variant-controls-both-precision"
PLAN_PROVENANCE = HERE / "inputs/plan-provenance.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": sha(path)}


def check_map(files, base=ROOT):
    for name, expected in files.items():
        assert sha(base / name) == expected, name
    return len(files)


def check_ref(record):
    path = ROOT / record["path"]
    assert sha(path) == record["sha256"], path
    return path


def evidence_files(directory):
    """Exclude transient compiler/Python products from retained inventories."""
    ignored_directories = {"__pycache__", ".pytest_cache", ".mypy_cache",
                           ".ruff_cache", "target", "build"}
    ignored_suffixes = {".pyc", ".pyo", ".o", ".obj", ".mod", ".smod"}
    return (path for path in directory.rglob("*")
            if path.is_file()
            and not ignored_directories.intersection(path.relative_to(directory).parts[:-1])
            and path.suffix.lower() not in ignored_suffixes)


def check_plan_snapshot():
    provenance = load(PLAN_PROVENANCE)
    assert provenance["original_path"] == "references/PLAN.md"
    assert provenance["source_revision_under_audit"] == REVISION
    assert provenance["original_sha256"] == provenance["snapshot"]["sha256"]
    snapshot = check_ref(provenance["snapshot"])
    assert snapshot == HERE / "inputs/PLAN.md"
    return {"provenance": ref(PLAN_PROVENANCE), **provenance}


def check_identity(record):
    original = dict(record)
    expected = original.pop("identity_sha256")
    encoded = json.dumps(original, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(encoded.encode()).hexdigest() == expected


def check_variant_comparison(name, records):
    """Recompute all coefficients without calling a generated executable."""
    stored = load(VARIANTS / "data" / f"agreement-{name}.json")
    assert stored["passed"] and not stored["failures"], name
    assert not stored["measurement_validation_failures"], name
    assert stored["relative_tolerance"] == 1e-7
    assert stored["scale_tolerance"] == 1e-10
    keys = [[(row["point"], row["order"]) for row in value["rows"]]
            for value in records]
    assert all(key == keys[0] for key in keys)
    if not name.startswith("rotation-invariance"):
        assert len({(value["points_sha256"], value["parameter_card_sha256"])
                    for value in records}) == 1
    scales = {}
    for value in records:
        assert "measurement_validation_error" not in value
        for row in value["rows"]:
            assert len(row["coefficients"]) == 4
            assert all(math.isfinite(x) for x in row["coefficients"])
            scales[row["point"]] = max(scales.get(row["point"], 0.),
                                      *(abs(x) for x in row["coefficients"]))
    count, maximum = 0, 0.
    for left, right in itertools.combinations(records, 2):
        for a, b in zip(left["rows"], right["rows"]):
            for x, y in zip(a["coefficients"], b["coefficients"]):
                tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a["point"]]
                difference = abs(x-y)
                assert difference <= tolerance, (name, a["point"], a["order"])
                maximum = max(maximum, difference / tolerance if tolerance else 0.)
                count += 1
    assert count == stored["coefficient_comparisons"]
    assert maximum == stored["maximum_tolerance_fraction"]
    return {**ref(VARIANTS / "data" / f"agreement-{name}.json"),
            "comparisons_recomputed": count,
            "maximum_tolerance_fraction": maximum,
            "status_counts": stored["status_counts"], "passed": True}


def check_variants():
    prepared = load(VARIANTS / "prepared.json")
    check_identity(prepared)
    assert sha(VARIANTS / "execute.py") == prepared["script_sha256"]
    check_ref(prepared["control_driver"])
    check_ref(prepared["generation"])
    for source in prepared["sources"].values():
        check_ref(source)
    rotation = load(VARIANTS / "rotation-reference.json")
    assert rotation["passed"] and not rotation["acceptance_eligible"]
    check_ref(rotation["binding"])
    records, comparisons = {}, {}
    backends = {"control-rotation": (1, 6, 7, 8),
                "control-rebinding": (1, 6, 7, 8),
                "control-rebinding-fresh": (8,),
                "control-near-singular": (1, 6, 8)}
    for name, selected in backends.items():
        identity = prepared["variants"][name]
        check_ref(identity["points"])
        check_ref(identity["driver"])
        records[name] = {}
        for backend in selected:
            record = load(VARIANTS / "data" / f"candidate-{name}-{backend}-0.json")
            inputs = VARIANTS / "inputs/candidate" / name
            assert record["backend"] == backend and record["case"] == name
            assert record["git_revision"] == REVISION
            assert record["generation_source_identity"]["revision"] == REVISION
            assert record["executable_sha256"] == identity["executable_sha256"]
            assert record["generated_sources_sha256"] == identity["sources"]["files_sha256"]
            assert record["points_sha256"] == identity["points"]["sha256"]
            assert record["parameter_card_sha256"] == sha(inputs / "param_card.dat")
            assert record["madloop_card_sha256"] == sha(inputs / f"MadLoopParams-{backend}.dat")
            assert record["compilation_metadata_sha256"] == sha(inputs / "compilation.json")
            expected_rows = [(point, order) for point in range(1, 11)
                             for order in range(record["answer_dimensions"]["loop_orders"] + 1)]
            assert [(r["point"], r["order"]) for r in record["rows"]] == expected_rows
            allowed = {1: {211, 221, 319, 329}, 6: {216, 226, 318, 328},
                       7: {217, 227}, 8: {218, 228, 312, 322}}[backend]
            assert all(r["status"] in allowed and math.isfinite(r["accuracy"])
                       for r in record["rows"])
            if name == "control-near-singular":
                expected_status = {1: 329, 6: 328, 8: 322}[backend]
                assert record["status_counts"] == {str(expected_status): 10}
                assert record["timed_status_counts"] == record["status_counts"]
                assert record["shared_numerical_settings"]["mlstabthres"] == 1e-20
            records[name][backend] = record
        if len(selected) > 1:
            comparisons[name] = check_variant_comparison(name, list(records[name].values()))
    comparisons["rebinding-context-reset"] = check_variant_comparison(
        "rebinding-context-reset",
        [records["control-rebinding"][8], records["control-rebinding-fresh"][8]])
    for backend in (1, 6, 7, 8):
        name = f"rotation-invariance-{backend}"
        comparisons[name] = check_variant_comparison(
            name, [load(check_ref(rotation["references"][str(backend)])),
                   records["control-rotation"][backend]])
    return {"prepared_identity_sha256": prepared["identity_sha256"],
            "passed": True, "comparisons": comparisons,
            "measurement_count": sum(map(len, records.values())),
            "generated_ninjars_rebinding_precision": "DP; all ten statuses 218",
            "performance_acceptance": False}


def check_reversal(archive):
    directory = HERE / "reversal-physical"
    result = load(directory / "result.json")
    assert result["passed"] and result["exit_code"] == 0 and result["build_exit_code"] == 0
    for field, filename in (("driver_sha256", "driver.f90"),
                            ("runner_sha256", "control.py"),
                            ("inputs_sha256", "inputs.json"),
                            ("output_sha256", "run.log")):
        assert result[field] == sha(directory / filename)
    inputs = load(directory / "inputs.json")
    assert inputs["data_sha256"] == sha(directory / "inputs.dat")
    for item in inputs["exact_rational_dp_buffers"]:
        assert sha(ROOT / item["fixture"]) == item["fixture_sha256"]
    assert result["archive_sha256"] == archive
    rows = result["rows"]
    assert len(rows) == 30
    assert [tuple(map(int, line.split()[1:4])) for line in rows] == [
        (case, phase, mode) for case in (1, 2)
        for phase in range(1, 6) for mode in (1, 2, 3)]
    assert all(int(line.split()[4]) == 0 for line in rows)
    assert len(result["negative_controls"]) == len(result["mp_reversal_tolerance_fractions"]) == 2
    return {"result": ref(directory / "result.json"), "passed": True, "complete_rows": 30,
            "modes": ["native DP", "compensated DP", "MP"],
            "mp_tolerance": "1e-31 times common largest Laurent magnitude",
            "negative_controls": result["negative_controls"],
            "mp_reversal_tolerance_fractions": result["mp_reversal_tolerance_fractions"],
            "retained_and_fresh_results": "bitwise identical in each mode",
            "separate_rational_term": "recorded engine-basis diagnostic; not an invariant gate"}


def main():
    assert not (HERE / "SHA256SUMS.json").exists(), "Sealed audit is immutable"
    port = load(PORT / "validation.json")
    expected = {**port["source_sha256"], **port["additional_source_sha256"]}
    check_map(expected)
    focused = load(INSTALL / "focused-unit-results.json")
    check_map(focused["transport_and_guard_suite"]["current_bridge_sources_match_tested_sources"])
    source_paths = set(expected)
    source_paths.update(subprocess.check_output(
        ["git", "ls-files", "vendor/ninjars"], cwd=ROOT, text=True).splitlines())
    source_paths.update([
        "tests/unit_tests/various/test_ninjars_statistics.py",
        "madgraph/madevent/sum_html.py",
        "madgraph/interface/amcatnlo_run_interface.py",
        "madgraph/iolibs/template_files/loop_optimized/CT_interface.inc",
        "madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc",
        "Template/loop_material/StandAlone/SubProcesses/MadLoopCommons.inc",
        "Template/loop_material/StandAlone/SubProcesses/MadLoopParams.inc",
        "Template/loop_material/StandAlone/SubProcesses/MadLoopParamReader.f",
        "madgraph/various/q_polynomial.py",
        "madgraph/loop/loop_exporters.py",
    ])
    sources = {}
    for name in sorted(source_paths):
        committed = subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=ROOT)
        current = sha(ROOT / name)
        original = hashlib.sha256(committed).hexdigest()
        assert current == original, f"Changed since final source: {name}"
        sources[name] = {"sha256": current, "matches_b44": True,
                         "recorded_in_abi3_port": name in expected}
        if name.startswith(("vendor/ninjars/tests/data/cases/",
                            "vendor/ninjars/tests/data/ninja/")):
            previous = subprocess.check_output(
                ["git", "show", f"{port['generated_at_source_commit']}:{name}"], cwd=ROOT)
            assert hashlib.sha256(previous).hexdigest() == current, name
            sources[name]["matches_committed_fixture_at_port_revision"] = True

    sealed = {}
    for directory in (PORT, INSTALL, POLICY / "final-variant-controls-fixed", VARIANTS):
        manifest = load(directory / "SHA256SUMS.json")
        count = check_map(manifest["files"], directory)
        assert count == manifest["count"]
        sealed[str(directory.relative_to(ROOT))] = {
            "seal": ref(directory / "SHA256SUMS.json"), "verified_files": count}

    comparison = load(POLICY / "final-controls-plan-fixed/comparison.json")
    assert comparison["passed"]
    check_map(comparison["measurements_sha256"])
    proof = load(POLICY / "representative-proof-abi3-terminal-fixed.json")
    check_map(proof["files_sha256"])
    variants = check_variants()
    plan = check_plan_snapshot()
    archive = load(INSTALL / "validation.json")["installation"]["archive_sha256"]
    reversal = check_reversal(archive)
    evidence_paths = [
        PLAN_PROVENANCE,
        HERE / "inputs/PLAN.md",
        POLICY / "final-controls-checklist.md",
        PORT / "validation.json", PORT / "abi3-unit-tests.log",
        PORT / "mock-flow/results.json", PORT / "mock-flow/origin.json",
        INSTALL / "validation.json", INSTALL / "focused-unit-results.json",
        INSTALL / "installation.json", INSTALL / "standalone-results.json",
        INSTALL / "runtime-results.json", INSTALL / "absent-results.json",
        INSTALL / "local-source-results.json",
        POLICY / "generation-abi3-terminal-fixed/summary.json",
        POLICY / "generation-abi3-terminal-fixed/SHA256SUMS.json",
        POLICY / "final-runtime-audit-fixed/inventory.json",
        POLICY / "representative-proof-abi3-terminal-fixed.json",
        POLICY / "representative-retention-abi3-terminal-fixed/validation.json",
        POLICY / "final-controls-plan-fixed/comparison.json",
        POLICY / "final-controls-plan-fixed/comparison-spec.json",
        POLICY / "final-controls-plan-fixed/compare.py",
        POLICY / "final-variant-controls-fixed/prepared.json",
        POLICY / "final-variant-controls-fixed/execution-failure.json",
        POLICY / "final-variant-controls-fixed/data/agreement-control-rebinding.json",
        POLICY / "final-variant-controls-fixed/drivers/control-rebinding.f",
        HERE / "reversal-initial-diagnosis.json",
        ROOT / "docs/benchmarkfolder/experiments/qcd3-point67-dp/oracle-build.json",
        ROOT / "docs/benchmarkfolder/experiments/qcd3-point67-dp/oracle-results.json",
        ROOT / "docs/benchmarkfolder/experiments/qcd3-point67-dp/oracle-inputs.json",
    ]
    for entry in comparison["bindings"]:
        path = entry["path"] if isinstance(entry, dict) else entry
        evidence_paths.append(ROOT / path)
    evidence_paths.extend(evidence_files(VARIANTS))
    for directory in (HERE / "reversal", HERE / "reversal-physical"):
        evidence_paths.extend(evidence_files(directory))
    convergence_line = next(line for line in (PORT / "abi3-unit-tests.log").read_text().splitlines()
                            if "NINJARS_CONVERGENCE {" in line)
    convergence = json.loads(convergence_line.split("NINJARS_CONVERGENCE ", 1)[1])
    output = {
        "schema": "final-focused-evidence-audit-v1",
        "state": "reviewed-source-scoped-evidence",
        "source_revision": REVISION,
        "plan_snapshot": plan,
        "abi_version": 3,
        "installed_archive_sha256": archive,
        "capture_command_runs_no_build_or_numerical_process": True,
        "separately_authorized_reversal_attempts": [
            {**ref(p), "passed": load(p)["passed"]}
            for p in (HERE / "reversal/result.json", HERE / "reversal-physical/result.json")
            if p.exists()],
        "source_hashes": sources,
        "unchanged_port_source_count": len(expected),
        "sealed_evidence_verified": sealed,
        "evidence": [ref(p) for p in sorted(set(evidence_paths))],
        "generated_controls": {
            "comparison_identity_sha256": comparison["identity_sha256"],
            "all_passed": True,
            "cases": comparison["cases"],
            "measurement_files_rehashed": len(comparison["measurements_sha256"]),
        },
        "representative": {
            "identity_sha256": proof["identity_sha256"],
            "coverage": proof["coverage"],
            "bound_files_rehashed": len(proof["files_sha256"]),
        },
        "generated_variants": variants,
        "physical_reversal": reversal,
        "convergence_summary": {k: convergence[k] for k in (
            "fixture", "lane", "modification", "production_bits", "reference_bits",
            "convergence_tolerance", "convergence_max_scaled_error",
            "finite_split_identity_tolerance", "passed")},
        "pending": [
            "Complete final eight-core correctness/performance gates.",
        ],
        "limitations": [
            "Historical ABI2 reversal oracle is not relabeled final installed-ABI3 evidence.",
            "Mock wrapper and mock rescue tests prove transport/state transitions, not physical invariance.",
            "Historical runtime/build-cache provenance remains unchanged.",
            "Threshold preservation has source and generated-run evidence; no dedicated boundary scan is claimed.",
            "Reused-MP bubble test compares final Laurent values after decoding to binary64, not every binary128 output bit.",
            "The initial generated rebinding run omitted updating MP external mass; the separate dual-precision driver passes. The failed original evidence is retained.",
            "Generated NinjaRS mass/scale rebinding is DP-only; genuine reused/fresh MP mass/scale coverage comes from the unchanged ABI3 bubble tests.",
            "The synthetic reversal input returned UNSTABLE before comparison; only the separate physical-fixture attempt establishes the invariant.",
            "No performance acceptance is inferred from this checklist.",
        ],
    }
    (HERE / "audit.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sources_verified": len(sources), "reused_port_sources": len(expected),
                      "sealed_evidence": sealed, "pending": output["pending"]}, indent=2))


if __name__ == "__main__":
    main()
