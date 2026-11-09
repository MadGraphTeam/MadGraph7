#!/usr/bin/env python3
"""Verify preserved hashes and summarize the isolated QP transport experiment."""
from collections import Counter
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path
import re

REPORT = Path(__file__).resolve().parent
PARENT = REPORT.parent
DIRECT = PARENT.parent
getcontext().prec = 60
SLOTS = ("BORN", "FIN", "1EPS", "2EPS")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path, expected):
    if digest(path) != expected:
        raise ValueError("Hash mismatch: " + str(path))


def result(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in (*SLOTS, "ASO2PI", "RETCODE", "ACC"):
            values[fields[0]] = Decimal(fields[1].replace("D", "E"))
    if values.get("RETCODE") != 120 or values.get("ACC") != -1:
        raise ValueError("Unexpected fixed-QP diagnostic status")
    if not all(values[k].is_finite() for k in SLOTS):
        raise ValueError("Nonfinite diagnostic coefficient")
    return values


def relative(a, b):
    scale = max(abs(a), abs(b))
    return Decimal(0) if scale == 0 else abs(a-b) / scale


def comparison(a, b):
    errors = {k: str(relative(a[k], b[k])) for k in SLOTS}
    return {"relative_errors": errors,
            "maximum_relative_error": str(max(Decimal(x) for x in errors.values())),
            "metric": "abs(a-b)/max(abs(a),abs(b)), zero-zero=0"}


def scalar_inputs(path):
    text = path.read_text()
    # The QP OneLOop printall output emits all actual scalar arguments.
    matches = re.findall(r"^\s*p(?:p|1|2|3|4|12|23):\s*([-.+0-9EeDd]+)\s*$", text, re.M)
    values = Counter(Decimal(x.replace("D", "E")) for x in matches)
    if not values:
        raise ValueError("Missing printed scalar inputs")
    near_shell = {x: n for x, n in values.items() if abs(x) < Decimal("1e-10")}
    physical_small = {x: n for x, n in values.items() if Decimal("1e-10") <= abs(x) < Decimal("1e-6")}
    return {"count": sum(values.values()),
            "near_shell_values": {str(x): n for x, n in sorted(near_shell.items())},
            "maximum_near_shell_magnitude": str(max(abs(x) for x in near_shell)),
            "physical_small_values": {str(x): n for x, n in sorted(physical_small.items())},
            "threshold_observations": re.findall(r".*CT_SCALAR_THRESHOLD.*", text)}


def verify_observation(directory, expected_build=None):
    status = json.loads((directory / "status.json").read_text())
    if status.get("returncode") != 0 or status.get("timed_out"):
        raise ValueError("Incomplete observation " + str(directory))
    if status.get("stability_certified") or status.get("performance_eligible"):
        raise ValueError("Diagnostic incorrectly marked qualified")
    fields = {"PS.input": "point_input_sha256", "MadLoopParams.dat": "numerical_card_sha256",
              "cuttools_onshell.input": "threshold_input_sha256", "output.log": "output_sha256",
              "result.dat": "result_sha256", "run.py": "runner_sha256"}
    for filename, key in fields.items():
        check(directory / filename, status[key])
    card = (directory / "MadLoopParams.dat").read_text()
    def card_value(name):
        matches = re.findall(r"(?mi)^#" + name + r"\s*\n([^\n]+)", card)
        if len(matches) != 1:
            raise ValueError("Missing or duplicate card field " + name)
        return matches[0].strip().lower()
    if status["backend"] != 1 or card_value("MLReductionLib") != "1":
        raise ValueError("Observation is not exclusive CutTools")
    if card_value("UseQPIntegrandForCutTools") != ".true.":
        raise ValueError("QP numerator was disabled")
    if status["mode"] not in (4, 5, 6) or any(
            int(card_value(name)) != status["mode"] for name in ("CTModeRun", "CTModeInit")):
        raise ValueError("Numerical card and recorded QP mode differ")
    log = (directory / "output.log").read_text()
    for marker, expected in (("BEFORE", 1e-6), ("AFTER", status["cuttools_scalar_threshold"])):
        values = re.findall(r"CT_SCALAR_THRESHOLD_" + marker + r"\s+(\S+)", log)
        if len(values) != 1 or float(values[0].replace("D", "E")) != expected:
            raise ValueError("Runtime scalar threshold differs from requested policy")
    if float((directory / "cuttools_onshell.input").read_text()) != status["cuttools_scalar_threshold"]:
        raise ValueError("Recorded threshold input differs from status")
    if expected_build is not None:
        check(expected_build, status["build_manifest_sha256"])
    return status, result(directory / "result.dat"), scalar_inputs(directory / "output.log")


def main():
    build_path = REPORT / "build/build.json"
    build = json.loads(build_path.read_text())
    parent_build_path = PARENT / "build-baseline/build.json"
    check(parent_build_path, build["parent_build_manifest_sha256"])
    parent_build = json.loads(parent_build_path.read_text())
    if build["objects_archives_before"] != parent_build["objects_archives_after"]:
        raise ValueError("QP transport does not start from the bound cutoff numerical basis")
    before, after = build["objects_archives_before"], build["objects_archives_after"]
    changed = [p for p in before if before[p] != after.get(p)]
    added = sorted(after.keys() - before.keys())
    if changed != ["SubProcesses/P0_epem_mupmum/CT_interface.o"] or added != [
            "SubProcesses/P0_epem_mupmum/cts_cutroutines_qpinput.o"]:
        raise ValueError("Unexpected object/archive delta")
    for path, key in (("CT_interface.original.f", "instrumented_interface_sha256"),):
        check(REPORT / "build" / path, parent_build[key])
    for path, key in (("CT_interface.instrumented.f", "interface_source_sha256"),
                      ("cts_cutroutines.original.f", "vendor_source_sha256"),
                      ("cts_cutroutines.instrumented.f", "private_source_sha256"),
                      ("qp_transport.mk", "make_overlay_sha256"), ("run.py", "runner_sha256")):
        check(REPORT / "build" / path, build[key])
    for path, expected in build["patches"].items():
        check(REPORT / "build" / path, expected)

    old_status, old_result, old_inputs = verify_observation(
        PARENT / "baseline-threshold-1e-10", parent_build_path)
    baseline_original = verify_observation(PARENT / "baseline-original-threshold", parent_build_path)
    observations = {}
    parsed = {}
    for label in ("threshold-1e-10", "original-threshold", "threshold-1e-10-reverse", "threshold-1e-10-both"):
        status, values, inputs = verify_observation(REPORT / label, build_path)
        for key in ("physical_card_sha256", "point_input_sha256", "parent_points_sha256"):
            if status[key] != old_status[key]:
                raise ValueError("Observation changed physical inputs: " + key)
        if status["physical_card_sha256"] != build["physical_card_sha256"]:
            raise ValueError("Observation physical card differs from build")
        parsed[label] = values
        observations[label] = {"status_sha256": digest(REPORT / label / "status.json"),
            "mode": status["mode"], "cuttools_scalar_threshold": status["cuttools_scalar_threshold"],
            "coefficients": {k: str(values[k]) for k in SLOTS}, "scalar_inputs": inputs}
    if (REPORT / "threshold-1e-10/MadLoopParams.dat").read_bytes() != (
            PARENT / "baseline-threshold-1e-10/MadLoopParams.dat").read_bytes():
        raise ValueError("Direct transport comparison changed numerical card")
    for label in observations:
        target = observations[label]["scalar_inputs"]["physical_small_values"]
        reference = old_inputs["physical_small_values"]
        # Mode 6 evaluates both propagator orders and doubles occurrences.
        multiplier = 2 if observations[label]["mode"] == 6 else 1
        if target != {k: n * multiplier for k, n in reference.items()}:
            raise ValueError("Physical small invariant or occurrence count changed")

    references = {}
    for backend in (6, 8, 9):
        label = "backend-%s-point20-osthres-1e-24" % backend
        directory = DIRECT / label
        status = json.loads((directory / "status.json").read_text())
        if status["parameter_card_sha256"] != build["physical_card_sha256"]:
            raise ValueError("Reference physical card differs")
        if digest(directory / "PS.input") != old_status["point_input_sha256"]:
            raise ValueError("Reference physical point differs")
        if status["parent_points_sha256"] != old_status["parent_points_sha256"]:
            raise ValueError("Reference point stream differs")
        values = result(directory / "result.dat")
        references[label] = {"result_sha256": digest(directory / "result.dat"),
            "status_sha256": digest(directory / "status.json"),
            "coefficients": {k: str(values[k]) for k in SLOTS},
            "comparison": comparison(values, parsed["threshold-1e-10"])}

    raw_log = (REPORT / "threshold-1e-10/output.log").read_text()
    aew_match = re.search(r"mdl_aEW =\s*\(\s*([^,]+),([^)]+)\)", raw_log)
    aew_real, aew_imag = Decimal(aew_match[1]), Decimal(aew_match[2])
    aew = (aew_real*aew_real + aew_imag*aew_imag).sqrt()
    alpha_s = Decimal(re.search(r"^ aS =\s*(\S+)", raw_log, re.M)[1])
    expected = -4 * aew / alpha_s
    output = {"scope": "Isolated CMS point20 fixed-QP causal diagnostic; no primary acceptance or runtime claim",
        "stability_certified": False, "performance_eligible": False,
        "build_manifest_sha256": digest(build_path),
        "unchanged_old_object_and_archive_count": len(before) - len(changed),
        "changed_objects": changed, "added_objects": added,
        "bound_baseline_cutoff_status_sha256": digest(PARENT / "baseline-threshold-1e-10/status.json"),
        "baseline_with_DP_input_transport": {"coefficients": {k: str(old_result[k]) for k in SLOTS},
                                              "scalar_inputs": old_inputs},
        "observations": observations, "references": references,
        "transport_change": comparison(old_result, parsed["threshold-1e-10"]),
        "forward_reverse": comparison(parsed["threshold-1e-10"], parsed["threshold-1e-10-reverse"]),
        "original_threshold_still_disagrees_with_lower_threshold": comparison(
            parsed["original-threshold"], parsed["threshold-1e-10"]),
        "double_pole_crosscheck": {"expression": "-4*abs(aEW)/alphaS",
            "observed_aew_real": str(aew_real), "observed_aew_imaginary": str(aew_imag),
            "observed_alpha_s": str(alpha_s),
            "expected": str(expected), "transport_relative_error": str(relative(expected, parsed["threshold-1e-10"]["2EPS"])),
            "old_transport_relative_error": str(relative(expected, old_result["2EPS"])),
            "source_note": "FKS uses -sum(Q_i^2)*GAL(1)^2/G^2; generated GAL(1)^2=4*pi*abs(aEW), equivalent here at recorded precision."},
        "analyzer_sha256": digest(Path(__file__))}
    (REPORT / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"forward_reverse": output["forward_reverse"],
                      "references": references, "double_pole_crosscheck": output["double_pole_crosscheck"]}, indent=2))


if __name__ == "__main__":
    main()
