"""Bounded final invocation of the existing CLI readers; no numerical calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
assert (ROOT / "tools/ninjars_profile_cli_v3.py").is_file()
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_profile_cli_v3 as authority

HERE = Path(__file__).resolve().parent
RUN = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(name, data):
    path = HERE / name
    assert not path.exists(), path
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


expected = read(HERE / "expected-pins.json")
filenames = read(RUN / "result-list.json")
assert len(filenames) == len(set(filenames)) == 72
inputs, records, groups, failures = {}, [], {}, []
for index, filename in enumerate(filenames):
    try:
        record = authority.report_snapshot(ROOT / filename, expected=expected, inputs=inputs)
        item = record["item"]
        records.append(record)
        groups.setdefault(item["case"], {}).setdefault(str(item["backend"]), []).append(record)
        print(json.dumps({"index": index, "item": item, "status": "passed"}), flush=True)
    except Exception as error:
        failures.append({"index": index, "path": filename, "error": repr(error)})
        print(json.dumps(failures[-1]), flush=True)

assert not failures, failures
spec = records[0]["declared_specification"]
assert [r["item"] for r in records] == spec["schedule"]
assert len(spec["cases"]) == 12 and len(spec["schedule"]) == 72
assert all(r["declared_specification"] == spec and r["specification"] == records[0]["specification"] for r in records)
assert all(r["acceptance_eligible"] is False and r["timed_status_counts_available"] is False for r in records)

controllers = [RUN / "pilot-launch-v1/result.json", RUN / "continuation-launch-v1/result.json"]
commands = []
for path in controllers:
    data = read(path)
    assert data["status"] == "passed"
    assert all(c["returncode"] == 0 for c in data["commands"])
    assert [c["index"] for c in data["commands"]] == data["completed_indices"]
    commands.extend(data["commands"])
assert [c["index"] for c in commands] == list(range(72))
cleanup = read(RUN / "continuation-launch-v1/terminal-cleanup.json")
assert cleanup["status"] == "passed" and cleanup["completed_cells"] == 72 and cleanup["failed_cells"] == cleanup["repeated_cells"] == 0

case_summary, zero_weights, signed = {}, [], []
scope_keys = ("physical_card_sha256", "timing_energy_source", "helicity_index", "physical_helicity", "helicity_table_sha256", "helicity_filter_sha256", "selected_helicity_weight", "loop_optimized_output", "direct_dp", "common_madloop_settings", "compiler_settings_sha256", "actual_split_orders_target")
for cid in spec["cases"]:
    backends = groups[cid]
    assert set(backends) == {"6", "8"}
    series = [r for b in ("6", "8") for r in sorted(backends[b], key=lambda r: r["item"]["repetition"])]
    assert all([r["item"]["repetition"] for r in sorted(backends[b], key=lambda r: r["item"]["repetition"])] == [0, 1, 2] for b in ("6", "8"))
    reference = series[0]
    differences = []
    for r in series:
        if not authority.same_stage_scope(reference["stage_scope"], r["stage_scope"]):
            differences.append({"item": r["item"], "different_keys": [k for k in scope_keys if reference["stage_scope"].get(k) != r["stage_scope"].get(k)]})
        if r["selected_helicity_weight"] == 0:
            zero_weights.append(r["item"])
        for mode in ("single_helicity", "helicity_sum"):
            for quantity, value in r["raw_timing"][mode].items():
                if value <= 0:
                    signed.append({"item": r["item"], "mode": mode, "quantity": quantity, "value": value})
    cache = {}
    for backend in ("6", "8"):
        values = sorted(backends[backend], key=lambda r: r["item"]["repetition"])
        assert all(r["stability"]["dp_points"] == 20 and r["stability"]["qp_points"] == 0 for r in values)
        assert [r["new_stability_points"] for r in values] == [20, 0, 0]
        hashes = {r["stability"]["cache"]["sha256"] for r in values}
        assert len(hashes) == 1, (cid, backend, hashes)
        cache[backend] = {"dp_points": 20, "qp_points": 0, "new_points_by_session": [20, 0, 0], "same_cache_across_sessions": True, "cache_sha256": next(iter(hashes))}
    case_summary[cid] = {"sessions": {b: len(backends[b]) for b in ("6", "8")}, "all_stage_scopes_complete": all(r["stage_scope_complete"] for r in series), "all_stage_scopes_identical": not differences, "scope_differences": differences, "selected_helicity_weight": {b: [r["selected_helicity_weight"] for r in sorted(backends[b], key=lambda r: r["item"]["repetition"])] for b in ("6", "8")}, "adaptive_base_points": {b: [r["adaptive_base_points"] for r in sorted(backends[b], key=lambda r: r["item"]["repetition"])] for b in ("6", "8")}, "cache": cache, "elapsed_seconds_by_session": {b: [(datetime.fromisoformat(c["finished_utc"]) - datetime.fromisoformat(c["started_utc"])).total_seconds() for c in commands if records[c["index"]]["item"]["case"] == cid and records[c["index"]]["item"]["backend"] == int(b)] for b in ("6", "8")}}

save("inputs-sha256.json", dict(sorted(inputs.items())))
save("validated-records.json", records)
save("result.json", {"schema": "final-retained-cli72-reader-invocation-v1", "utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "numerical_calls": 0, "native_archive_bytes_read": 0, "reader": bound(Path(authority.__file__)), "invocation": bound(Path(__file__)), "expected_pins": bound(HERE / "expected-pins.json"), "result_list": bound(RUN / "result-list.json"), "source_inventory": bound(HERE / "inputs-sha256.json"), "validated_records": bound(HERE / "validated-records.json"), "specification": records[0]["specification"], "controller_results": [bound(p) for p in controllers], "cleanup": bound(RUN / "continuation-launch-v1/terminal-cleanup.json"), "coverage": {"cases": 12, "backends": [6, 8], "sessions_per_backend": 3, "cells": 72, "controller_indices": list(range(72)), "failed_cells": 0, "repeated_cells": 0, "all_parent_returncodes_zero": True, "strict_reader_failures": []}, "cases": case_summary, "zero_selected_helicity_weight_cells": zero_weights, "nonpositive_raw_timing_observations": signed, "qualification_eligible": False, "fullsum_performance_gate_eligible": False, "timed_status_counts_available": False, "limits": ["MG7 direct-DP wall stages divided by actual adaptive traversal counts; loop evaluation is FULL-minus-SKIP estimate, not a direct reduction stopwatch", "Three empirical CLI sessions per backend; no small-sample bootstrap coverage guarantee and no matched-point timing claim", "20-point CLI stability cache reused unchanged with zero new points in sessions1/2; these are not timed rescue counts or retained matched-momentum CDF samples", "Zero selected helicity weight or nonpositive subtraction samples must suppress corresponding optional logarithmic comparison", "Literal-original full-helicity rescue-on LIGHT6 seven-pair acceptance study remains separate"]})
print(json.dumps({"status": "passed", "cases": case_summary, "zero_weight_cells": zero_weights, "nonpositive": signed}), flush=True)
