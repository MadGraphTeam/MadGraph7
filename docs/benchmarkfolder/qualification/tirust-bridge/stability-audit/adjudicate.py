#!/usr/bin/env python3
"""Summarize the diagnostic without treating an expected DP failure as a pass."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; evidence is append-only")
    data = json.loads(args.input.read_text())
    native = data["generated_adapter"]
    checks = []
    for origin, case in data["cases"].items():
        for precision, rows in case.items():
            for name in ("primary", "repeat_cached", "reverse_check_bypass", "reverse_cache_off",
                         "reverse_shared_cache", "rotation_1", "rotation_2"):
                assert rows[name]["status"] == 0
            assert rows["primary"]["coefficients"] == rows["repeat_cached"]["coefficients"]
            for name in ("reverse_cache_off", "reverse_shared_cache"):
                assert rows[name]["coefficients"] == rows["reverse_check_bypass"]["coefficients"]
            for name in ("reverse_check_bypass", "reverse_cache_off"):
                assert rows[name]["cache_before"] == rows[name]["cache_after"]
            checks.append(f"{origin}/{precision}: repeat and reversal modes agree; check/cache-off bypass counters")
    for precision in ("dp", "qp"):
        rows = native[precision]
        for name in ("primary", "reverse_check_bypass", "rotation_1", "rotation_2"):
            assert rows[name]["status"] == 0
        for diagnostic in ("host_reversal_estimate", "host_reversal_plus_rotations_estimate"):
            for projection in rows[diagnostic].values():
                assert not projection["default_requests_rescue"]
                assert projection["diagnostic_requests_rescue"] == (precision == "dp")
                assert projection["accuracy"] == projection["native_fortran"]["accuracy"]
    primary = native["dp"]["primary"]
    assert primary["symmetric_error_against_same_input_qp"][0] > 3e-6
    assert native["qp"]["primary"]["reference_relative_error"] < 1e-8
    estimates = {}
    for projection, component in (("real_interference", 0), ("imaginary_interference", 1)):
        estimate = native["dp"]["host_reversal_estimate"][projection]["estimate"][0]
        truth = native["qp"]["primary"]["coefficients"][0][component]
        estimates[projection] = {
            "finite_estimate": estimate, "same_input_qp_finite": truth,
            "symmetric_error": abs(estimate - truth) / abs(estimate + truth),
            "passes_independent_3e_minus_6_gate": abs(estimate - truth) / abs(estimate + truth) < 3e-6}
    assert not any(row["passes_independent_3e_minus_6_gate"] for row in estimates.values())
    sources = [
        "vendor/tirust/src/lib.rs", "vendor/tirust/check_fixtures.py",
        "madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc",
        "madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc",
        "madgraph/iolibs/template_files/loop_optimized/CT_interface.inc",
        "madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc",
        "madgraph/iolibs/template_files/loop/improve_ps.inc",
        "Template/loop_material/StandAlone/Cards/MadLoopParams.dat",
        "Template/loop_material/StandAlone/SubProcesses/MadLoopParamReader.f",
        "docs/benchmarkfolder/cases.json", "tools/tirust_benchmark_worker.f",
        "../references/TIRust/crates/iregi-rust/src/dispatch/embedded.rs"]
    result = {
        "schema_version": 1,
        "classification": "Retained DP stress failure; default host check misses it; explicit QP resolves it",
        "input_sha256": sha256(args.input), "adjudicator_sha256": sha256(__file__),
        "fixture": data["label"], "bridge_library_sha256": data["library_sha256"],
        "native_harness_sha256": data["adapter_library_sha256"],
        "native_dp_finite_symmetric_error_vs_same_input_qp": primary["symmetric_error_against_same_input_qp"][0],
        "native_qp_reference_whole_lane_relative_error": native["qp"]["primary"]["reference_relative_error"],
        "default_threshold": 1e-3, "diagnostic_only_tighter_threshold": 1e-6,
        "independent_acceptance_threshold": 3e-6, "ordinary_target_threshold": 1e-8,
        "native_host_averaged_estimates": estimates,
        "full_dp_precision_policy": {
            "workspace": "TensorWorkspace::new_embedded(plan, 0.0f64, EmbedderOptions::default())",
            "method_choice": True, "partial_rescue": False, "precision_tiers": False,
            "hidden_higher_precision_numerical_rescue": False},
        "checks": checks + ["Actual Fortran accuracy routine agrees with Python formula",
            "Default policy misses both native projections with and without two rotations",
            "Diagnostic 1e-6 threshold requests rescue for both native DP projections",
            "Explicit QP original/reversal/rotations agree with reference",
            "Both default-accepted averaged finite projections fail independent 3e-6 gate"],
        "limits": ["Debug bridge correctness evidence, not release performance qualification",
            "Synthetic complex tensor contraction; real and imaginary projections are tested separately",
            "Native estimator and adapter are executed, but no full physical SLOOPMATRIX rescue flow is claimed",
            "Tighter threshold and rotations are diagnostic only; benchmark/default policy is unchanged",
            "Full-DP precision policy does not describe the separate poles-only workspace"],
        "source_sha256": {path: sha256(path) for path in sources}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
