#!/usr/bin/env python3
"""Validate and summarize the isolated CutTools cutoff causal experiment."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPORT = Path(__file__).resolve().parent
DIRECT = REPORT.parent
ROOT = REPORT.parents[5]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter(card, name):
    values = re.findall(r"(?mi)^#" + name + r"\s*\n([^\n]*)", card)
    if len(values) != 1:
        raise ValueError("Missing/ambiguous numerical parameter " + name)
    return values[0].strip()


def coefficients(folder):
    log = (folder / "output.log").read_text()
    values = []
    for field in ("born", "finite", "1eps", "2eps"):
        entries = re.findall(r"Matrix element " + field + r"\s*=\s*([^\s]+)", log)
        if len(entries) != 1:
            raise ValueError("Unexpected standalone coefficient layout")
        values.append(float(entries[0]))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite coefficients")
    result = (folder / "result.dat").read_text()
    if not re.search(r"(?m)^RETCODE\s+120\s*$", result):
        raise ValueError("Unexpected fixed-QP status")
    normalized = {name: float(re.findall(r"(?m)^" + name + r"\s+(\S+)", result)[0])
                  for name in ("BORN", "FIN", "1EPS", "2EPS", "ASO2PI")}
    return values, normalized


def compare(left, right):
    rows = [[{"point":20, "order":0, "coefficients":x}] for x in (left, right)]
    return bench.compare_rows(*rows)


def scalar_inputs(log):
    return re.findall(r"(?m)^\s*(pp|p[1234]|p12|p23|mm|m[1234]):(.*)$", log)


def main():
    point_file = DIRECT.parent / "points/cms_ew_n2_epem_mupmum.json"
    parent = json.loads(point_file.read_text())
    if digest(point_file.with_suffix(".dat")) != parent["sha256"]:
        raise ValueError("Parent stream changed")
    expected_point = "".join(" ".join("%.17e" % x for x in p) + "\n"
                             for p in parent["points"][19]["momenta"])
    physical = DIRECT.parent / "inputs/baseline/cms_ew_n2_epem_mupmum/param_card.dat"
    complete = {}
    logs = {}
    for status_file in sorted(REPORT.glob("*/status.json")):
        folder = status_file.parent
        status = json.loads(status_file.read_text())
        build_path = REPORT / ("build-" + status["source"]) / "build.json"
        build = json.loads(build_path.read_text())
        for filename, field in (("run.py", "runner_sha256"),
                                ("MadLoopParams.dat", "numerical_card_sha256"),
                                ("PS.input", "point_input_sha256"),
                                ("cuttools_onshell.input", "threshold_input_sha256"),
                                ("output.log", "output_sha256"), ("result.dat", "result_sha256")):
            if digest(folder / filename) != status[field]:
                raise ValueError("Run artifact binding changed: " + str(folder / filename))
        if digest(build_path) != status["build_manifest_sha256"]:
            raise ValueError("Build manifest changed")
        build_folder = build_path.parent
        if (digest(build_folder / "CT_interface.original.f") != build["original_interface_sha256"]
                or digest(build_folder / "CT_interface.instrumented.f") != build["instrumented_interface_sha256"]
                or digest(build_folder / "run.py") != build["runner_sha256"]):
            raise ValueError("Preserved build sources changed")
        changed = [name for name in build["objects_archives_before"]
                   if build["objects_archives_before"][name] != build["objects_archives_after"][name]]
        if changed != ["SubProcesses/P0_epem_mupmum/CT_interface.o"]:
            raise ValueError("Unexpected numerical object changes")
        card = (folder / "MadLoopParams.dat").read_text()
        if (status["point_index"] != 20 or status["backend"] != 1
                or status["parent_points_sha256"] != parent["sha256"]
                or status["physical_card_sha256"] != digest(physical)
                or build["physical_card_sha256"] != digest(physical)
                or (folder / "PS.input").read_text() != expected_point
                or status["timed_out"] or status["returncode"] != 0
                or status["stability_certified"] or status["performance_eligible"]
                or parameter(card, "MLReductionLib") != "1"
                or parameter(card, "CTModeRun") != str(status["mode"])
                or parameter(card, "CTModeInit") != str(status["mode"])
                or parameter(card, "UseQPIntegrandForCutTools").lower() != ".true."
                or float(parameter(card, "OSThres")) != 1e-14
                or float(parameter(card, "MLStabThres")) != 1e-3
                or float((folder / "cuttools_onshell.input").read_text()) != status["cuttools_scalar_threshold"]):
            raise ValueError("Input/policy binding failed")
        values, normalized = coefficients(folder)
        log = (folder / "output.log").read_text()
        before = re.findall(r"CT_SCALAR_THRESHOLD_BEFORE\s+(\S+)", log)
        after = re.findall(r"CT_SCALAR_THRESHOLD_AFTER\s+(\S+)", log)
        if ([float(x) for x in before] != [1e-6]
                or [float(x) for x in after] != [status["cuttools_scalar_threshold"]]):
            raise ValueError("Runtime cutoff differs from requested cutoff")
        logs[folder.name] = log
        complete[folder.name] = {"source":status["source"], "mode":status["mode"],
            "scalar_threshold":status["cuttools_scalar_threshold"], "coefficients":values,
            "normalized_coefficients":normalized, "one_loop_errors":log.count("ERROR"),
            "one_loop_warnings":log.count("WARNING"), "status_sha256":digest(status_file),
            "raw_result_sha256":digest(folder / "result.dat")}

    baseline_original = REPORT / "baseline-original-threshold"
    retained_original = DIRECT / "backend-1-point20-osthres-1e-14"
    if digest(baseline_original / "result.dat") != digest(retained_original / "result.dat"):
        raise ValueError("Instrumented original cutoff does not reproduce original result")
    reference = DIRECT / "backend-9-point20-osthres-1e-14"
    reference_status = json.loads((reference / "status.json").read_text())
    if (reference_status["parent_points_sha256"] != parent["sha256"]
            or reference_status["parameter_card_sha256"] != digest(physical)
            or (reference / "PS.input").read_text() != expected_point):
        raise ValueError("TIRust comparison physical binding failed")
    target, normalized = coefficients(reference)
    for observation in complete.values():
        observation["comparison_to_tirust_qp"] = compare(target, observation["coefficients"])

    original_inputs = scalar_inputs(logs["baseline-original-threshold"])
    cutoff_inputs_equal = all(scalar_inputs(logs[name]) == original_inputs for name in (
        "baseline-threshold-1e-10", "baseline-threshold-1e-9",
        "candidate-original-threshold", "candidate-threshold-1e-10"))
    if not cutoff_inputs_equal:
        raise ValueError("Cutoff intervention changed scalar input stream")
    invariants = [Decimal(value.strip()) for name, value in original_inputs if name.startswith("p")]
    small = Counter(str(x) for x in invariants if x != 0 and abs(x) < Decimal("1e-6"))
    log = logs["baseline-threshold-1e-10"]
    alpha_components = re.findall(r"mdl_aEW\s*=\s*\(\s*([^,]+),([^\)]+)\)", log)
    if len(alpha_components) != 1:
        raise ValueError("Missing/ambiguous model electromagnetic coupling")
    alpha_ew = complex(*(float(x) for x in alpha_components[0]))
    expected_double_raw = -4 * abs(alpha_ew) / (2 * math.pi) * target[0]
    expected_double_normalized = -4 * abs(alpha_ew) / (2 * math.pi) / normalized["ASO2PI"]
    fks = ROOT / "Template/NLO/SubProcesses/fks_singular.f"
    if "contr2=contr2-particle_charge(i)**2" not in fks.read_text():
        raise ValueError("Independent infrared formula changed")
    transport_file = REPORT / "qp-transport/summary.json"
    transport = json.loads(transport_file.read_text())
    if (transport["stability_certified"] or transport["performance_eligible"]
            or transport["analyzer_sha256"] != digest(transport_file.parent / "analyze.py")):
        raise ValueError("Transport diagnosis changed or was incorrectly qualified")
    transport_agreement = max(float(reference["comparison"]["maximum_relative_error"])
                              for reference in transport["references"].values())
    summary = {"scope":"Isolated cutoff causal test; fixed-QP diagnostics remain unchecked",
        "stability_certified":False, "performance_eligible":False,
        "point":20, "parent_points_sha256":parent["sha256"],
        "physical_card_sha256":digest(physical), "observations":complete,
        "instrumentation_reproduces_original_result_bytes":True,
        "cutoff_intervention_scalar_inputs_identical":cutoff_inputs_equal,
        "scalar_input_record_count":len(original_inputs), "invariant_record_count":len(invariants),
        "small_nonzero_invariant_counts":dict(small),
        "baseline_candidate_original_result_bytes_equal":digest(REPORT / "baseline-original-threshold/result.dat") == digest(REPORT / "candidate-original-threshold/result.dat"),
        "baseline_candidate_lower_cutoff_result_bytes_equal":digest(REPORT / "baseline-threshold-1e-10/result.dat") == digest(REPORT / "candidate-threshold-1e-10/result.dat"),
        "lower_cutoff_plateau_result_bytes_equal":digest(REPORT / "baseline-threshold-1e-10/result.dat") == digest(REPORT / "baseline-threshold-1e-9/result.dat"),
        "independent_double_pole":{
            "formula":"-sum(external_massless_charge_squared)*Born*abs(alphaEW)/(2*pi)",
            "external_charge_squared_sum":4, "external_masses":parent["inputs"]["incoming_masses"]+parent["inputs"]["masses"],
            "alpha_ew_real":alpha_ew.real, "alpha_ew_imaginary":alpha_ew.imag,
            "alpha_ew_absolute":abs(alpha_ew), "raw_expected":expected_double_raw,
            "normalized_expected":expected_double_normalized, "tirust_raw":target[3],
            "tirust_relative_error":abs(target[3]-expected_double_raw)/abs(expected_double_raw),
            "fks_source":"Template/NLO/SubProcesses/fks_singular.f:7789", "fks_source_sha256":digest(fks),
            "normalization_source":"../check_sa.f:132", "normalization_source_sha256":digest(DIRECT / "check_sa.f")},
        "separate_qp_transport_diagnosis":{
            "summary_sha256":digest(transport_file),
            "maximum_relative_error_against_ninja_ninjars_tirust":transport_agreement,
            "ordinary_point_target_met_in_private_diagnostic":transport_agreement <= 1e-8,
            "production_cuttools_modified":False, "primary_policy_qualified":False},
        "interpretation":"Two isolated interventions attribute the legacy CutTools discrepancy to its absolute scalar cutoff and DP denominator transport. The private corrected-QP diagnostic agrees with independent reducers and the FKS double pole; original primary failures remain retained and unqualified."}
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runs":len(complete), "input_records":len(original_inputs),
        "independent_double_pole_relative_error":summary["independent_double_pole"]["tirust_relative_error"]}))


if __name__ == "__main__":
    main()
