#!/usr/bin/env python3
"""Strict NRS-first report authority with a separate v3 presentation.

The unchanged v2 template contains four-tool policy claims and must not be used
to present an NRS-first pass. This adapter reuses its numerical/proof readers,
without changing their globals, roles, status rules, template or old evidence.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninjars_benchmark_policy_v3 as authority
import ninjars_report_v2 as legacy
import ninjars_precision_v2 as precision

v2 = legacy.policy
ROOT = v2.ROOT
SCHEMA = "madloop-nrs-first-report-v3"
BACKENDS = (1, 6, 7, 8)
TEMPLATE = ROOT / "docs/benchmarkfolder/policy-v3/template/benchmark.typ"
STYLE = ROOT / "docs/benchmarkfolder/style.typ"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pending(reason):
    return {"status": "pending", "reason": reason}


class Inputs:
    """Reread the authority's already validated bytes, rejecting later changes."""
    def __init__(self, dataset, dataset_path):
        self.expected = dict(dataset["inputs_sha256"])
        self.checked = dict(self.expected)
        require(json.loads(self.retain(dataset_path)) == dataset, "Dataset changed after independent validation")

    def retain(self, filename, checksum=None, from_dataset=False):
        filename = v2.path(filename)
        relative = v2.relative(filename)
        raw = filename.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if from_dataset:
            require(relative in self.expected, "Report input was not validated by v3 authority: " + relative)
            checksum = self.expected[relative] if checksum is None else checksum
            require(self.expected[relative] == checksum, "Conflicting authority input binding")
        require(checksum is None or checksum == actual, "Changed report input: " + relative)
        self.checked[relative] = actual
        return raw

    def read(self, filename, from_dataset=False):
        return json.loads(self.retain(filename, from_dataset=from_dataset))

    def bound(self, record):
        return json.loads(self.file(record))

    def file(self, record):
        require(isinstance(record, dict) and set(("path", "sha256")).issubset(record), "Missing bound report input")
        raw = self.retain(record["path"], record["sha256"], from_dataset=True)
        require("bytes" not in record or len(raw) == record["bytes"], "Changed bound input size")
        return raw


def calibration_usage(record):
    require(record["repetition"] == "calibration" and record["loops"] == 1,
            "A calibration cannot become a timing repetition")
    authority.validate_rows(record, record["backend"])
    counts = dict(record["status_counts"])
    return {"status": "observed-calibration", "numerical_commands": 1,
            "physical_points": sum(counts.values()), "timing_repetitions": 0,
            "dp": sum(n for c, n in counts.items() if int(c)//100 == 2),
            "qp": sum(n for c, n in counts.items() if int(c)//100 == 3),
            "status_counts": counts, "median_us": None,
            "scope": "One complete accepted numerical calibration; no promoted timing samples"}


def numerical_records(inputs, binding, candidate_only=False):
    if binding is None:
        return {}
    data = v2.path(binding["report"]) / "data"
    records = {}
    for relative in sorted(inputs.expected):
        path = v2.path(relative)
        if path.parent != data:
            continue
        match = re.fullmatch(r"(.+)-(baseline|candidate)-(6|8)-(\d+)\.json", path.name)
        if match is None:
            continue
        cid, source, backend, rep = match.groups()
        if candidate_only and (source, int(backend)) != ("candidate", 8):
            raise ValueError("Correctness binding includes another backend")
        record = inputs.read(path, from_dataset=True)
        key = cid, source, int(backend), int(rep)
        require(key not in records and (record["case"], record["source"], record["backend"],
                record["repetition"]) == key, "Ambiguous or relabelled numerical record")
        authority.validate_rows(record, int(backend))
        records[key] = record
    return records


def correctness_source(inputs, record):
    if record is None:
        return None, pending("No NinjaRS correctness collection supplied")
    value = inputs.bound(record)
    if value.get("schema") != "nrs-first-core-correctness-finalization-v1":
        return value, {"kind": "direct-source-binding", "binding": record,
                       "original_controller_success": "not asserted by this report",
                       "fresh_process_parity_passed": False}
    require(value.get("status") == "complete-raw-correctness-only" and
            value.get("original_failure_promoted") is False and value.get("fresh_parity") is False and
            type(value.get("numerical_retries")) is int and value["numerical_retries"] == 0 and
            type(value.get("new_amplitude_evaluations")) is int and value["new_amplitude_evaluations"] == 0,
            "Derived correctness view cannot promote controller failure, retries or fresh parity")
    original_status = inputs.bound(value["original_status"])
    view_status = inputs.bound(value["view_status"])
    require(original_status.get("status") == "failed" and view_status.get("status") == "complete",
            "Original failed status and derived view status must remain distinct")
    inputs.file(value["original_protocol"])
    inputs.file(value["original_specification"])
    require(value.get("copies"), "Derived correctness requires its byte-exact copy map")
    for copied in value["copies"]:
        require(copied["source"]["path"] != copied["destination"]["path"] and
                inputs.file(copied["source"]) == inputs.file(copied["destination"]),
                "Derived correctness view must preserve every copied byte")
    binding = inputs.bound(value["bindings"])
    return binding, {"kind": "derived-raw-correctness-copy-view", "receipt": record,
        "original_controller_success": False, "original_status": original_status,
        "view_status": view_status, "copy_map": value["copies"],
        "original_protocol": value["original_protocol"], "original_specification": value["original_specification"],
        "fresh_process_parity_passed": False, "new_amplitude_evaluations": 0, "numerical_retries": 0,
        "scope": "Accepted raw observations independently checked in a copied view; original controller failed finalization. Existing representative qualification is separate."}


def external_snapshot(dataset, inputs):
    if dataset.get("external_findings") is None:
        return {"status": "pending", "findings": [], "vetoes_nrs_core": False,
                "scope": "External observations are missing; no backend pass is inferred"}
    specification = inputs.bound(dataset["external_findings"])
    require(specification.get("schema") == "nrs-external-findings-v3", "Wrong external evidence schema")
    findings = specification["findings"]
    require(isinstance(findings, list), "External findings must be a list")
    for item in findings:
        require(item.get("status") in ("failed", "pending", "accepted-observation") and item.get("evidence"),
                "External failure cannot be relabelled passed or lose its raw evidence")
        for record in item["evidence"]:
            inputs.file(record)
    status = "failed" if any(f["status"] == "failed" for f in findings) else "pending"
    return {"status": status, "findings": findings, "vetoes_nrs_core": False,
            "scope": "Bound external observations; accepted-observation is not an all-backend qualification pass"}


def qualification_snapshot(dataset, rules, proof, pair_binding, correction, inputs,
                           runtime_provenance_spec=None, validation_spec=None,
                           precision_spec=None, supplement_spec=None, profile_spec=None):
    """Use existing proof authorities; unsupported full scopes remain pending."""
    bindings = [pair_binding] if pair_binding is not None else []
    if not bindings:
        require(not any(p is not None for p in (runtime_provenance_spec, validation_spec,
                    precision_spec, supplement_spec, profile_spec)), "Qualification requires actual pair source/runtime bindings")
        return {"runtime": pending("Actual timing-pair binding is absent"),
                "qualifications": {k: pending("Independent proof authority not supplied") for k in authority.QUALIFICATION_KEYS},
                "profiles": [], "profile_status": "pending"}
    signature = legacy.candidate_runtime_signature(bindings)
    runtime = legacy.runtime_provenance_snapshot(runtime_provenance_spec, dataset, bindings, inputs.checked)
    regressions = legacy.regression_snapshot(validation_spec, bindings, inputs.checked)
    controls = (precision.validate(precision_spec, signature["identity_sha256"], inputs.checked,
                                  runtime_signature=signature) if precision_spec is not None else
                pending("Installed public-ABI and 256/384-bit proof not supplied"))
    groups = v2.proof_runtime_signatures(proof["entries"])
    coverage = groups.get("candidate_corrected/8", {}).get(signature["identity_sha256"], {})
    automatic_rescue = {"dp", "qp"}.issubset(coverage.get("precisions", []))
    precision_rescue = {"status": "failed" if controls.get("status") == "failed" else "passed" if controls.get("status") == "passed" and automatic_rescue else "pending",
                        "installed_precision": controls, "normal_dp_qp_runtime_proven": automatic_rescue,
                        "scope": "Existing independent installed precision and accepted representative-runtime authorities"}
    loop = legacy.supplement_snapshot(supplement_spec, dataset, bindings, inputs.checked)
    # A successful strict v2 supplement is sufficient; a future NRS-only loop
    # authority may be needed for an external-only failure. Never suppress it.
    qualifications = {
        "installation_export": pending("Installed runtime smoke is checked separately; complete relocated standalone/NLO/export authority is not supplied"),
        "precision_rescue": precision_rescue,
        "focused_controls": pending("Source-bound complex/mixed/rotation/state-control authority is not supplied"),
        "loop_induced": {"status": "passed" if loop.get("complete_campaign_passed") is True else loop["status"],
                         "evidence": loop, "scope": "Existing strict loop-induced/threshold/precision authority; no gate retired"},
        "regressions": regressions}
    profiles = legacy.profile_snapshot(profile_spec, dataset, bindings, correction, inputs.checked)
    require(all(type(p.get(k)) is int and p[k] > 0 for p in profiles for k in ("samples", "evaluations")),
            "Final profiles require positive actual sample and evaluation counts")
    return {"runtime": runtime, "qualifications": qualifications, "profiles": profiles,
            "profile_status": "passed" if profiles else "pending"}


def build_snapshot(dataset_path, timestamp=None, runtime_provenance_spec=None,
                   validation_spec=None, precision_spec=None, supplement_spec=None,
                   profile_spec=None):
    dataset_path = v2.path(dataset_path)
    dataset = authority.validate_dataset(dataset_path)
    require(dataset.get("schema") == authority.DATASET, "Wrong v3 dataset authority")
    inputs = Inputs(dataset, dataset_path)
    rules, correction, config, proof = authority.validate_policy(v2.check_file(dataset["policy"]))
    for record in (dataset["policy"], rules["configuration"], rules["correction"], rules["representative_proof"]):
        inputs.bound(record)
    require({c["id"] for c in config["cases"]} == set(dataset["cases"]) == authority.CASE_IDS,
            "Report must preserve all eight required cases")
    references = inputs.bound(dataset["reference_bindings"])
    ref_records = {}
    for entry in references["entries"]:
        key = entry["case"], entry["backend"]
        require(key not in ref_records and entry["backend"] in (1, 6), "Duplicate/injected calibration reference")
        ref_records[key] = inputs.bound(entry["record"])
    pair_binding = inputs.bound(dataset["timing_bindings"]) if dataset.get("timing_bindings") else None
    correct_binding, correctness_provenance = correctness_source(inputs, dataset.get("correctness_bindings"))
    timed = numerical_records(inputs, pair_binding)
    correct = numerical_records(inputs, correct_binding, candidate_only=True)
    cases = []
    for definition in config["cases"]:
        cid = definition["id"]
        references_usage = {str(b): calibration_usage(ref_records[cid, b]) if (cid, b) in ref_records else
                            dict(pending("Accepted calibration not supplied"), timing_repetitions=0)
                            for b in (1, 6)}
        observations = [r for (case, _, b, _), r in sorted(correct.items()) if case == cid and b == 8]
        observations += [r for (case, source, b, _), r in sorted(timed.items()) if case == cid and source == "candidate" and b == 8]
        checks, stability = [], pending("No accepted NinjaRS numerical commands")
        for record in observations if all((cid, b) in ref_records for b in (1, 6)) else []:
            groups = {"original_helper_corrected/1": ref_records[cid, 1]["rows"],
                      "original_helper_corrected/6": ref_records[cid, 6]["rows"],
                      "candidate_corrected/8": record["rows"]}
            checks.append(legacy.compare(groups))
            if len(checks) == 1:
                stability = legacy.stability_snapshot(groups)
                stability["selected_observation"] = {"source": record["source"], "backend": record["backend"],
                    "repetition": record["repetition"], "scope": "First accepted NinjaRS command and two reference calibrations; correctness checks every observation, timed usage aggregates independently"}
        passed = bool(checks) and all(c["passed"] for c in checks)
        agreement = {"status": "passed" if passed else "failed" if checks else "pending", "passed": passed,
                     "observations": len(checks), "checks": checks, "scale_scope": "Corrected CutTools/Ninja and candidate NinjaRS only"}
        recorded = dataset["cases"][cid]
        require(recorded["agreement"]["passed"] is passed, "Stale NRS correctness conclusion")
        require(len(recorded["agreement"]["comparisons"]) == len(checks), "Numerical observation count differs from authority")
        for actual, saved in zip(checks, recorded["agreement"]["comparisons"]):
            legacy.compare_recorded(actual, saved)
        timing_checks = []
        for (case, source, backend, rep), record in sorted(timed.items()):
            if (case, source, backend) != (cid, "candidate", 8):
                continue
            require((cid, "baseline", 6, rep) in timed, "Missing matched literal-original timing denominator")
            if all((cid, b) in ref_records for b in (1, 6)):
                groups = {"original_helper_corrected/1": ref_records[cid, 1]["rows"],
                          "original_helper_corrected/6": ref_records[cid, 6]["rows"],
                          "candidate_corrected/8": record["rows"],
                          "original/6": timed[cid, "baseline", 6, rep]["rows"]}
                timing_checks.append(legacy.compare(groups))
        timing_passed = bool(timing_checks) and all(c["passed"] for c in timing_checks)
        timing_agreement = {"status": "passed" if timing_passed else "failed" if timing_checks else "pending",
            "passed": timing_passed, "checks": timing_checks,
            "scale_scope": "Matched accepted literal-original Ninja, candidate NinjaRS and both corrected references; rejected external rows excluded"}
        require(recorded["timing_agreement"]["passed"] is timing_passed and
                len(recorded["timing_agreement"]["comparisons"]) == len(timing_checks),
                "Stale timing-denominator same-outcome gate")
        for actual, saved in zip(timing_checks, recorded["timing_agreement"]["comparisons"]):
            legacy.compare_recorded(actual, saved)
        timings = {source: {str(b): legacy.summary([], config["points_per_case"], config["repetitions"])
                            for b in BACKENDS} for source in ("baseline", "candidate")}
        series = {}
        for source, backend, label in (("baseline", 6, "baseline"), ("candidate", 8, "candidate")):
            runs = [r for (case, s, b, _), r in sorted(timed.items()) if (case, s, b) == (cid, source, backend)]
            series[source] = runs
            timings[label][str(backend)] = legacy.summary(runs, config["points_per_case"], config["repetitions"])
        performance = {"status": "pending", "passed": False}
        if timed:
            performance = legacy.bench.runtime_statistics([r["seconds_per_point"] for r in series["baseline"]],
                [r["seconds_per_point"] for r in series["candidate"]], config["max_runtime_ratio"], config["seed"])
            timings["candidate"]["8"]["ratio"] = performance["runtime_ratio"]
        require(performance == recorded["performance"], "Stale NRS performance conclusion")
        cases.append(dict(definition, agreement=agreement, performance=performance, timing_agreement=timing_agreement, timings=timings,
                          numerical_reference_usage=references_usage, stability=stability,
                          nrs_numerical_commands=len(observations),
                          external_backend_cells={"1": "numerical-reference-only", "6": "numerical-reference-only", "7": "pending"}))
    correctness = all(c["agreement"]["passed"] for c in cases)
    performance = all(c["performance"]["passed"] and c["timing_agreement"]["passed"] for c in cases)
    require(dataset["nrs_correctness_passed"] is correctness and dataset["nrs_performance_passed"] is performance and
            dataset["nrs_core_validation_passed"] is (correctness and performance), "Forged or inconsistent v3 core gates")
    external = external_snapshot(dataset, inputs)
    for case in cases:
        for finding in external["findings"]:
            if finding.get("case") == case["id"] and finding.get("backend") in BACKENDS:
                case["external_backend_cells"][str(finding["backend"])] = finding["status"]
    qualified = qualification_snapshot(dataset, rules, proof, pair_binding, correction, inputs,
        runtime_provenance_spec, validation_spec, precision_spec, supplement_spec, profile_spec)
    qualifications = qualified["qualifications"]
    qualification_gates = {k: qualifications[k]["status"] == "passed" for k in authority.QUALIFICATION_KEYS}
    publication = {"four_tool_report": False, "profiles": qualified["profile_status"] == "passed",
                   "reproducible_pdf": False, "pushed_milestones": False}
    gates = authority.combine_gates(correctness, performance, qualification_gates, publication, external["findings"])
    if not external["findings"]:
        gates["external_failures_recorded"] = None
    full = gates["nrs_validation_milestone_passed"] and qualified["runtime"]["status"] == "passed"
    gates["nrs_validation_milestone_passed"] = full
    gates["final_publication_passed"] = full and gates["final_publication_passed"]
    core_status = ("passed" if correctness and performance else "failed" if any(
        c[k]["status"] == "failed" for c in cases for k in ("agreement", "performance", "timing_agreement")) else
        "unresolved" if any(c["performance"]["status"] == "unresolved" for c in cases) else "pending")
    snapshot = {"schema": SCHEMA, "generated_utc": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d · %H:%M UTC"),
        "dataset_identity_sha256": dataset["identity_sha256"], "dataset_path": v2.relative(dataset_path),
        "policy_identity_sha256": rules["identity_sha256"], "config": config, "cases": cases,
        "correctness_collection_provenance": correctness_provenance,
        **gates, "nrs_full_qualification_passed": full,
        "nrs_core_status": core_status,
        "external_backend_comparison_status": external["status"], "external_diagnostics": external,
        "qualification_obligations": qualifications, "runtime_provenance": qualified["runtime"],
        "profiles": qualified["profiles"], "publication_obligations": {
            "four_tool_report": pending("Complete honest four-tool accounting and upstream MRE evidence remain separate"),
            "profiles": {"status": qualified["profile_status"], "count": len(qualified["profiles"])},
            "reproducible_pdf": pending("Actual v3 combined render, byte reproduction and visual authority are not supplied"),
            "pushed_milestones": pending("This core dataset does not certify Git publication")},
        "performance_denominator": "original/6", "legacy_v2_complete_campaign_claim": False,
        "presentation": {"full_v2_template_compatible": False,
            "reason": "Unchanged v2 template asserts 21-pair/four-tool gates; v3 publication needs separate reviewed presentation",
            "template": "docs/benchmarkfolder/policy-v2/template/benchmark.typ",
            "v3_template": v2.relative(TEMPLATE),
            "style": "docs/benchmarkfolder/style.typ"},
        "authority_sources_sha256": {v2.relative(Path(module.__file__)): legacy.workers.digest(Path(module.__file__))
            for module in (sys.modules[__name__], authority, legacy, precision, legacy.profiler,
                           precision.supplement, v2, legacy.bench, legacy.workers)},
        "presentation_sources_sha256": {name: legacy.workers.digest(ROOT / name) for name in
            ("docs/benchmarkfolder/policy-v2/template/benchmark.typ", "docs/benchmarkfolder/style.typ")},
        "inputs_sha256": dict(sorted(inputs.checked.items()))}
    return v2.seal(snapshot)


def render(dataset_path, destination, timestamp=None, compile_pdf=False, **specifications):
    """Render a separate v3 snapshot; no old report file or proof is overwritten."""
    destination = v2.path(destination)
    require(not destination.exists(), "Preserve existing reports; choose a fresh destination")
    snapshot = build_snapshot(dataset_path, timestamp=timestamp, **specifications)
    old_template = ROOT / snapshot["presentation"]["template"]
    source = old_template.read_text()
    start = source.index("= Measured optimizations and retained negative results")
    finish = source.index("\n#pagebreak()\n#section-label[MG7 REGRESSIONS", start)
    # Reuse the exact previously reviewed attribution block without importing
    # the old template's numerical policy claims or modifying that source.
    research = '#import "style.typ": *\n' + source[start:finish] + "\n"
    snapshot["render_sources_sha256"] = {v2.relative(p): legacy.workers.digest(p)
        for p in (Path(__file__).resolve(), Path(authority.__file__).resolve(), TEMPLATE, STYLE, old_template)}
    snapshot["historical_research_fragment_sha256"] = hashlib.sha256(research.encode()).hexdigest()
    command = ["typst", "compile", "--creation-timestamp", str(legacy.PDF_CREATION_TIMESTAMP),
               "benchmark.typ", "benchmark.pdf"]
    snapshot["pdf_build"] = {"command": command, "creation_timestamp": legacy.PDF_CREATION_TIMESTAMP,
                             "scope": "Presentation output; no publication qualification inferred"}
    snapshot = v2.seal(snapshot)
    destination.mkdir(parents=True)
    legacy.bench.dump(destination / "report-data.json", snapshot)
    shutil.copy2(TEMPLATE, destination / "benchmark.typ")
    shutil.copy2(STYLE, destination / "style.typ")
    (destination / "research.typ").write_text(research)
    if compile_pdf:
        subprocess.run(command, cwd=destination, check=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp")
    parser.add_argument("--render", action="store_true", help="Write separate v3 Typst/assets to a fresh output directory")
    parser.add_argument("--compile", action="store_true", help="Render and compile v3 presentation; never a publication pass")
    for name in ("runtime-provenance", "validation-spec", "precision-proof", "supplement", "profile-spec"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    target = v2.path(args.output)
    require(not target.exists(), "Preserve prior report snapshots; choose a new output")
    specifications = dict(runtime_provenance_spec=args.runtime_provenance, validation_spec=args.validation_spec,
                          precision_spec=args.precision_proof, supplement_spec=args.supplement, profile_spec=args.profile_spec)
    if args.render or args.compile:
        value = render(args.dataset, target, timestamp=args.timestamp, compile_pdf=args.compile, **specifications)
    else:
        value = build_snapshot(args.dataset, timestamp=args.timestamp, **specifications)
        legacy.bench.dump(target, value)
    print(json.dumps({k: value[k] for k in ("identity_sha256", "nrs_correctness_passed",
        "nrs_performance_passed", "nrs_core_validation_passed", "nrs_full_qualification_passed",
        "external_backend_comparison_status", "final_publication_passed")}, indent=2))


if __name__ == "__main__":
    main()
