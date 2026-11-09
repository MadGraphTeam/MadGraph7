#!/usr/bin/env python3
"""Recheck standalone fixed-QP raw results; never upgrade primary qualification."""
from pathlib import Path
import itertools
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
from tirust_complex_stress import CASE


def parameter(text, name):
    values = re.findall(r"(?mi)^#" + name + r"\s*\n([^\n]*)", text)
    if len(values) != 1:
        raise ValueError("Missing or repeated numerical parameter")
    return values[0].strip()


def main():
    parent_file = REPORT.parent / "points" / (CASE["id"] + ".dat")
    parent = json.loads(parent_file.with_suffix(".json").read_text())
    if bench.digest(parent_file) != parent["sha256"]:
        raise ValueError("Parent physical points changed")
    physical = REPORT.parent / "inputs/baseline" / CASE["id"] / "param_card.dat"
    card_hash = bench.digest(physical)
    complete, failed = {}, {}
    for folder in sorted(REPORT.glob("backend-*-point*")):
        if not (folder / "status.json").exists():
            failed[folder.name] = {"status": "incomplete", "scope": "No completed direct-driver attempt"}
            continue
        record = json.loads((folder / "status.json").read_text())
        point = record["point_index"]
        expected = "".join(" ".join("%.17e" % x for x in vector) + "\n"
                           for vector in parent["points"][point-1]["momenta"])
        card = (folder / "MadLoopParams.dat").read_text()
        if (record["parent_points_sha256"] != parent["sha256"]
                or record["parameter_card_sha256"] != card_hash
                or (folder / "PS.input").read_text() != expected
                or bench.digest(folder / "run_case.py") != record["runner_sha256"]
                or record["driver_sha256"] != bench.digest(REPORT / "check_sa.f")
                or parameter(card, "MLReductionLib") != str(record["backend"])
                or parameter(card, "CTModeRun") != "4" or parameter(card, "CTModeInit") != "4"
                or parameter(card, "CheckCycle") != "1"
                or float(parameter(card, "OSThres")) != record["onshell_threshold"]
                or record["stability_certified"] or record["performance_eligible"]):
            raise ValueError("Standalone diagnostic input/protocol binding changed")
        if record.get("timed_out") or record.get("returncode") != 0 or not record["result_file_exists"]:
            failed[folder.name] = record
            continue
        result = (folder / "result.dat").read_text()
        if not re.search(r"(?m)^RETCODE\s+120\s*$", result):
            failed[folder.name] = {**record, "error": "Missing unchecked QP status120"}
            continue
        log = (folder / "output.log").read_text()
        values = []
        for field in ("born", "finite", "1eps", "2eps"):
            entries = re.findall(r"Matrix element " + field + r"\s*=\s*([^\s]+)", log)
            if len(entries) != 1:
                raise ValueError("Unexpected standalone coefficient layout")
            values.append(float(entries[0]))
        if not all(math.isfinite(x) for x in values):
            failed[folder.name] = {**record, "error": "Nonfinite diagnostic coefficients"}
            continue
        complete[folder.name] = {**record, "coefficients": values, "status":120,
            "accuracy":-1., "scope":"Unchecked fixed-QP coefficient diagnostic only",
            "raw_result_sha256":bench.digest(folder / "result.dat"),
            "raw_log_sha256":bench.digest(folder / "output.log")}
    comparisons = []
    for (left_name, left), (right_name, right) in itertools.combinations(complete.items(), 2):
        if left["point_index"] != right["point_index"]:
            continue
        rows = [[{"point":x["point_index"], "order":0, "coefficients":x["coefficients"]}]
                for x in (left,right)]
        comparisons.append({"reference":left_name, "candidate":right_name,
            "point":left["point_index"], "onshell_thresholds":[left["onshell_threshold"],right["onshell_threshold"]],
            **bench.compare_rows(*rows)})
    summary = {"scope":"Separate fixed-QP diagnostics with verified repaired kinematics; original primary failures remain open",
        "stability_certified":False, "performance_eligible":False,
        "parent_points_sha256":parent["sha256"], "parameter_card_sha256":card_hash,
        "kinematic_audit_sha256":bench.digest(REPORT / "cms_mp_shell_audit.json"),
        "complete":complete, "failures":failed, "comparisons":comparisons,
        "diagnostic_pairwise_agreement":bool(comparisons) and all(x["passed"] for x in comparisons),
        "comparison_failures":[{key:entry[key] for key in ("reference","candidate","point")}
                               for entry in comparisons if not entry["passed"]]}
    bench.dump(REPORT / "summary.json", summary)
    print(json.dumps({"completed":len(complete), "failed":list(failed),
        "comparisons":len(comparisons), "all_comparisons_pass":all(x["passed"] for x in comparisons)}))


if __name__ == "__main__":
    main()
