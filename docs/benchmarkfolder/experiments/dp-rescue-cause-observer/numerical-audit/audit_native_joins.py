#!/usr/bin/env python3
"""Supplement completed cause journals; read-only, no numerical invocation.

This adds semantic joins to the unchanged frozen parser. It does not change its
protocol or certify all six arms. Results are JSON on stdout; no files are made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[3]
TARGET = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-qualified/controller-native/attempt1"
EXECUTION_SHA = "c308d6bdf48db24bc428e0c556618e4191fe07535720708f954b62331376f8b4"
_HELPERS = None
PINS = {
    "analyze.py": "473475dfdde70df2f721ca782f3ac1719ab08b92b72a9eadd4e8bbe216aed10b",
    "protocol.py": "b668433b0fe6d397d3416d408b17a1f7f54f2f8b24c9cc063c1a5710423d38ec",
    "prepare_host.py": "4a3b3c002af1a40329f19e20453a861895fadf87eb506359cfe00f1f993957a0",
    "native/observer.rs": "acf09e33fe3d0e57d01a64d6bcaca44e98abd1277d1ed52aa8204ec04fe708fe",
    "native/observed-lib.rs.patch": "5d4ea85cc80f2953bdd3561cdfabe97b2a7d6b810bfa347207b6bdde03273954",
}
STAGES = ("stage2", "stage3", "init4", "budget5", "stage6", "stage7",
          "stage8", "return32", "stage9", "stable33")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def helpers():
    global _HELPERS
    for name, expected in PINS.items():
        need(sha(PROPOSAL / name) == expected, "Frozen helper/source changed: " + name)
    if _HELPERS is not None:
        need(sys.modules.get("protocol") is _HELPERS[0], "Frozen protocol module replaced")
        return _HELPERS
    need("protocol" not in sys.modules, "Unexpected preloaded protocol")
    protocol = load(PROPOSAL / "protocol.py", "protocol")
    analyzer = load(PROPOSAL / "analyze.py", "cause_supplement_frozen_analyzer")
    _HELPERS = protocol, analyzer
    return _HELPERS


def detail_order(call):
    """Check exactly what the pinned observer emits, not inferred work purity."""
    begin, details, end = call["begin"], call["details"], call["end"]
    kinds = [item["kind"] for item in details]
    if begin["input_bits"] == 113:
        need(kinds in ([], ["qp_cache_hit"]), "QP details are not empty or a sole cache hit")
        if kinds:
            need(begin["cache"] == 1 and begin["check"] == 0 and end["status"] == 0,
                 "QP cache hit has incompatible controls/status")
        return
    if not details:
        need(end["status"] != 0, "Successful DP C call omitted workspace evidence")
        return  # A failure before workspace.reduce_batch has no last_* evidence.
    need(kinds[:2] == ["options", "upstream_completion"], "DP option/completion order differs")
    if not details[1]["result_ok"]:
        need(len(details) == 2 and end["status"] == 2,
             "Rust-level Err retained stale details or did not return ERROR")
        return
    need(len(details) >= 4 and kinds[2] == "lane_route" and details[2]["lane"] == 0,
         "Completed DP call lacks exactly one lane0 route before diagnostics")
    index = 3
    diagnostic_index = 0
    while index < len(details) and kinds[index] == "diagnostic":
        note = details[index]
        need(note["lane"] == 0 and note["index"] == diagnostic_index,
             "Upstream diagnostic lane/index is not contiguous")
        diagnostic_index += 1
        index += 1
    need(index < len(details) and kinds[index] == "execution"
         and details[index]["input_rows"] == 1, "Missing or misplaced one-row execution record")
    index += 1
    need(kinds[index:] in ([], ["fallback_preparation_error"]),
         "Extra lane/execution/tier record or invalid trailing detail")
    if details[2]["route"] == "refused":
        need(end["status"] != 0, "Refused lane was reported as C success")


def supplement(path, analyzer):
    """The frozen parser must pass first; this function only strengthens joins."""
    counts = Counter()
    per_call = []
    transports = []
    wrapper_failures = []
    errors, notes, routes, executions = Counter(), Counter(), Counter(), Counter()
    flush = 0
    diagnostic = native = None
    ordinal = 0
    current_counts = Counter()

    def finish_diagnostic():
        nonlocal diagnostic
        if diagnostic is None:
            return
        tokens = tuple(diagnostic["tokens"])
        need(tokens == STAGES[:len(tokens)], "Fortran diagnostic stages are not a source-order prefix")
        need(len(tokens) not in (2, 6, 7), "Diagnostic ended between a call and mandatory status event")
        if len(tokens) == 3:
            need(diagnostic["init"] != 0, "Successful initialization lacks following budget event")
        if len(tokens) == 4:
            need(diagnostic["budget"] != 0, "Successful budget event lacks following kinematic stage")
        if len(tokens) == 8:
            need(diagnostic["returned"] != 0, "Successful tensor wrapper lacks output-finiteness stage")
        if len(tokens) > 3:
            need(diagnostic["init"] == 0, "Diagnostic continued after initialization failure")
        if len(tokens) > 4:
            need(diagnostic["budget"] == 0, "Diagnostic continued after budget-setting failure")
        if len(tokens) > 8:
            need(diagnostic["returned"] == 0, "Diagnostic continued after Fortran failure")
        call = diagnostic["native"]
        if call is not None:
            need("end" in call and "returned" in diagnostic,
                 "Native call lacks completion or its Fortran return status")
            c_status, f_status = call["end"]["status"], diagnostic["returned"]
            if diagnostic["precision"] == 53:
                need(c_status == f_status, "DP C/Fortran status mismatch")
                current_counts["dp_equal_status_joins"] += 1
            elif c_status != f_status:
                need((c_status, f_status) == (0, 2), "Unexplained QP C/Fortran status mismatch")
                transports.append({"public_sequence": flush - 1, "diagnostic_ordinal": diagnostic["ordinal"],
                                   "native_id": call["id"], "c_status": c_status, "fortran_status": f_status,
                                   "scope": "Successful C output rejected by the Fortran unpack stage"})
                current_counts["qp_post_c_transport_rejections"] += 1
            else:
                current_counts["qp_equal_status_joins"] += 1
            detail_order(call)
            current_counts["native_calls"] += 1
            current_counts["qp_cache_hits"] += sum(d["kind"] == "qp_cache_hit" for d in call["details"])
            phase = "bootstrap" if flush == 1 else "points_1_to_100"
            if call["end"]["status"] != 0:
                errors[(phase, diagnostic["precision"], call["end"]["status"], call["end"]["error"])] += 1
            for detail in call["details"]:
                if detail["kind"] == "diagnostic":
                    notes[(phase, detail["text"])] += 1
                elif detail["kind"] == "lane_route":
                    routes[(phase, diagnostic["ctmode"], diagnostic["n"], diagnostic["rank"], detail["route"])] += 1
                elif detail["kind"] == "execution":
                    for field, value in detail.items():
                        if field != "kind":
                            executions[(phase, field)] += value
        elif "returned" in diagnostic:
            need(diagnostic["returned"] in (2, 3), "No-C wrapper return is not ERROR/UNSUPPORTED")
            wrapper_failures.append({"public_sequence": flush - 1,
                                     "diagnostic_ordinal": diagnostic["ordinal"],
                                     "precision": diagnostic["precision"],
                                     "fortran_status": diagnostic["returned"]})
            current_counts["wrapper_rejections_before_c"] += 1
        else:
            current_counts["diagnostics_returning_before_tensor_wrapper"] += 1
        current_counts["diagnostics"] += 1
        diagnostic = None

    for record in analyzer.json_records(path):
        kind = record["kind"]
        if kind == "flush_begin":
            need(diagnostic is None and native is None, "Unfinished call crosses flush")
            flush = record["flush"]
            ordinal = 0
            current_counts = Counter()
        elif kind == "flush_end":
            finish_diagnostic()
            need(native is None, "Native call crosses flush boundary")
            if flush <= 101:
                counts.update(current_counts)
                per_call.append({"public_sequence": flush - 1, "counts": dict(sorted(current_counts.items()))})
        elif kind != "event":
            continue
        else:
            data = analyzer.validate_payload(record["data"])
            event_kind = data["kind"]
            if event_kind == "host":
                tag = data["tag"]
                words = tuple(map(analyzer.signed_word, data["words"]))
                if diagnostic is not None and diagnostic["native"] is not None:
                    previous = diagnostic["native"]
                    if "end" in previous and "returned" not in diagnostic:
                        need(tag == 32, "C completion is not followed by its Fortran status event")
                if tag == 30:
                    finish_diagnostic()
                    ordinal += 1
                    diagnostic = {"ordinal": ordinal, "precision": words[0], "ctmode": words[1],
                                  "n": words[2], "rank": words[3], "tokens": [], "native": None}
                elif tag in (31, 32, 33, 34):
                    need(diagnostic is not None, "Unscoped Fortran diagnostic progress")
                    if tag == 31:
                        token = "stage" + str(words[0])
                    elif tag == 34:
                        token = "init4" if words[0] == 4 else "budget5"
                        diagnostic["init" if words[0] == 4 else "budget"] = words[1]
                    elif tag == 32:
                        need("returned" not in diagnostic, "Repeated Fortran status event")
                        token = "return32"
                        diagnostic["returned"] = words[0]
                    else:
                        token = "stable33"
                        need(words[0] == diagnostic["precision"], "Stable marker precision mismatch")
                    diagnostic["tokens"].append(token)
                    need(tuple(diagnostic["tokens"]) == STAGES[:len(diagnostic["tokens"])],
                         "Fortran diagnostic stage order differs")
                elif tag in (20, 22, 40, 45, 46, 12):
                    finish_diagnostic()
            elif event_kind == "native_begin":
                need(native is None and diagnostic is not None and diagnostic["native"] is None,
                     "Native call reused a diagnostic ordinal or lacks a diagnostic")
                need(tuple(diagnostic["tokens"]) == STAGES[:7], "Native call is outside tensor-evaluation stage")
                need((data["input_bits"], data["n"], data["rank"])
                     == tuple(diagnostic[k] for k in ("precision", "n", "rank")), "Native/diagnostic signature differs")
                need(data["finite"] == 1 and data["poles"] == 0 and data["cache"] == 1
                     and data["context"] > 0 and data["check"] == int(diagnostic["ctmode"] in (2, 5)),
                     "Default finite/both-pole/cache/direction/context changed")
                native = {"id": record["native"], "begin": data, "details": []}
                diagnostic["native"] = native
            elif event_kind == "native_end":
                need(native is not None, "Native end without begin")
                native["end"] = data
                native = None
            elif event_kind != "point":
                need(native is not None, "Native detail outside call")
                native["details"].append(data)
    need(flush == 102 and diagnostic is None and native is None and len(per_call) == 101,
         "Supplement did not cover all101 calls and close flush")
    return {"counts": dict(sorted(counts.items())), "per_public_call": per_call,
            "qp_post_c_transport_rejections": transports, "wrapper_rejections_before_c": wrapper_failures,
            "immediate_native_errors": [{"phase": phase, "input_bits": bits, "status": status,
                "error": error, "count": count} for (phase, bits, status, error), count in sorted(errors.items())],
            "upstream_diagnostic_texts": [{"phase": phase, "text": text, "count": count}
                for (phase, text), count in sorted(notes.items())],
            "lane_routes_by_shape_direction": [{"phase": phase, "ctmode": ctmode, "n": n, "rank": rank,
                "route": route, "count": count} for (phase, ctmode, n, rank, route), count in sorted(routes.items())],
            "engine_execution_sums": [{"phase": phase, "field": field, "sum": count}
                for (phase, field), count in sorted(executions.items())]}


def cause_summary(points):
    """Concurrent measured signatures; an accuracy-only label is not causal proof."""
    rows = []
    for point in points:
        failures = [row for row in point["native_failures"] if row["precision"] == 53]
        guards = [row for row in point["fortran_guard_returns"] if row["precision"] == 53]
        checks = point["host_dp_checks"]
        unstable = sum(row["final_stable"] == 0 for row in point["host_dp_groups"])
        contraction = sum(row["contraction_rejected"] for row in point["host_dp_groups"])
        latch = any(row["latch_after_override"] for row in checks)
        finite_bad = any(row["finite_pre_accuracy_exceeds_threshold"] for row in checks)
        nonfinite = any(row["pre_accuracy_nonfinite"] for row in checks)
        dp_calls = sum(row["count"] for row in point["native_counts"] if row["input_bits"] == 53)
        branches = len(point["qp_escalation_branches"])
        signature = bool(branches and dp_calls and finite_bad and not nonfinite
                         and not failures and not guards and not unstable and not latch)
        rows.append({"point": point["point"], "phase": point["phase"], "status": point["status"],
                     "native_dp_failures": len(failures), "fortran_dp_guard_returns": len(guards),
                     "unstable_dp_groups": unstable, "contraction_rejections": contraction,
                     "host_rejection_latch": latch, "finite_host_accuracy_exceeds_threshold": finite_bad,
                     "host_accuracy_nonfinite": nonfinite, "qp_escalation_branches": branches,
                     "successful_dp_with_finite_host_accuracy_rejection_signature": signature,
                     "observations": point["observations"]})
    final = [row for row in rows if row["phase"] == 1]
    return {"bootstrap": [row for row in rows if row["phase"] == 0], "all100_points": final,
            "rescued_points": [row for row in final if row["status"] == 323],
            "observation_point_counts": dict(sorted(Counter(tag for row in final for tag in row["observations"]).items())),
            "successful_dp_with_finite_host_accuracy_rejection_points": [row["point"] for row in final
                if row["successful_dp_with_finite_host_accuracy_rejection_signature"]],
            "scope": "Concurrent source-joined observations; no unique cause or counterfactual speedup is established"}


def audit(case):
    protocol, analyzer = helpers()
    need(case in protocol.CASES, "Unknown frozen case")
    directory = TARGET / case / "enabled"
    need(directory.resolve(strict=True) == directory, "Enabled arm path aliases another directory")
    execution = directory / "execution.json"
    before = execution.read_bytes()
    metadata = json.loads(before)
    need(metadata["complete"] is True and metadata["arm"] == "enabled"
         and metadata["process_spawned"] and metadata["process_reaped"] and metadata["returncode"] == 0,
         "Arm is not completed and reaped; do not read an active journal")
    files = ("cause-public.txt", "warmup.dat", "cause-native.jsonl")
    raw_pins = {name: sha(directory / name) for name in files}
    need(all(raw_pins[name] == metadata["output_sha256"][name] for name in files),
         "Completed raw output differs from retained execution record")
    arm = analyzer.read_arm(directory)
    frozen = analyzer.parse_native(directory / "cause-native.jsonl", arm["calls"], True)
    extra = supplement(directory / "cause-native.jsonl", analyzer)
    need(extra["counts"].get("native_calls", 0) == frozen["native_call_count"], "Supplement/frozen native counts differ")
    need(execution.read_bytes() == before and all(sha(directory / name) == raw_pins[name] for name in files),
         "Completed evidence changed while auditing")
    return {"schema": "dp-rescue-native-join-supplement-v1", "case": case,
            "completed_enabled_arm": str(directory), "supplemental_joins_passed": True,
            "frozen_native_parser_passed": True, "public_call_count": 101,
            "frozen_native_summary": {k: frozen[k] for k in ("event_count", "native_call_count", "flush_count")},
            "cause_observations": cause_summary(frozen["points"]),
            **extra, "execution_sha256": hashlib.sha256(before).hexdigest(), "raw_sha256": raw_pins,
            "auditor_sha256": sha(__file__), "frozen_source_sha256": PINS,
            "six_arm_qualification": False, "cause_attribution_qualified": False,
            "performance_eligible": False,
            "limitations": ["Only an already-completed enabled arm is audited here; all six-arm parity/runtime gates remain separate.",
                            "C-success to FortranERROR in QP is distinguished as transport rejection; it is not a native reduction failure.",
                            "Direct routes and aggregate master-row counters do not certify exclusively binary64 internal arithmetic.",
                            "Upstream diagnostic strings remain in the bound raw journal; this supplement does not collapse retry history into one cause."]}


def audit_all():
    protocol, analyzer = helpers()
    path = TARGET / "execution.json"
    need(sha(path) == EXECUTION_SHA, "Wrong final six-arm execution record")
    execution = json.loads(path.read_bytes())
    need(execution["complete"] is True and execution["runtime_provenance_validated"] is True
         and execution["numerical_parity_passed"] is True
         and execution["protected_before"] == execution["protected_after"], "Six-arm completion gates did not pass")
    expected = {case + "/" + arm for case in protocol.CASES for arm in ("original", "disabled", "enabled")}
    need(set(execution["arms"]) == expected and all(row["complete"] and row["process_reaped"]
         and row["returncode"] == 0 for row in execution["arms"].values()), "Six-arm matrix is incomplete")
    cases = {}
    for case in protocol.CASES:
        result = audit(case)
        arms = [analyzer.read_arm(TARGET / case / arm) for arm in ("original", "disabled", "enabled")]
        result["independent_public_triplet"] = analyzer.validate_public_triplet(case, *arms)
        cases[case] = result
    need(sha(path) == EXECUTION_SHA, "Final execution record changed during audit")
    return {"schema": "dp-rescue-six-arm-join-audit-v1", "supplemental_joins_passed": True,
            "all_six_completed_execution_record_bound": True, "all_public_triplets_independently_reparsed": True,
            "execution_sha256": EXECUTION_SHA, "auditor_sha256": sha(__file__), "cases": cases,
            "cause_attribution_qualified": False, "performance_eligible": False,
            "scope": "Read-only semantic joins and concurrent cause observations after all six parity/runtime gates; no unique-cause, physics, or timing acceptance"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("all", "qcd_n3_uux_zgg", "ew_n3_epem_ttxa"))
    args = parser.parse_args()
    print(json.dumps(audit_all() if args.case == "all" else audit(args.case), indent=2, sort_keys=True, allow_nan=False))
