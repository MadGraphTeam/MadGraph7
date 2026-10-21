"""Aggregate validated Ninja/NinjaRS stock MG7 CLI profile observations.

The CLI full-minus-SKIP reduction estimate is separate from fixed-point,
full-helicity, rescue-on acceptance timing and from random-DP stability.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SCHEMA = "ninjars-cli-profile-report-v1"
AUTHORITY = "ninjars_profile_cli_v3"
PIN_KEYS = {"ninja_archive", "ninjars_archive", "process_checks"}
MODES = ("single_helicity", "helicity_sum")
QUANTITIES = {"total": "seconds_per_point", "construction": "construction_seconds_per_point",
              "reduction_estimate": "reduction_estimate_seconds_per_point"}
REPETITIONS = 7


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(filename):
    filename = Path(filename)
    filename = (filename if filename.is_absolute() else ROOT / filename).resolve()
    require(filename.is_relative_to(ROOT), "CLI report files must stay inside the repository")
    return filename


def retain(filename, inputs):
    filename = path(filename)
    inputs[str(filename.relative_to(ROOT))] = hashlib.sha256(filename.read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    offset = (len(values)-1)*fraction
    lo, hi = math.floor(offset), math.ceil(offset)
    return values[lo] + (values[hi]-values[lo])*(offset-lo)


def median_samples(values, draws, rng):
    return [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]


def summarize(values, draws=10000, seed=20261021):
    require(values and all(type(v) in (int, float) and math.isfinite(v) for v in values),
            "CLI timing samples must be finite; signed subtraction estimates are retained")
    require(type(draws) is int and draws > 0, "Invalid bootstrap repetition count")
    median = statistics.median(values)
    bootstrap = median_samples(values, draws, random.Random(seed))
    return {"median_seconds_per_point": median,
            "median_absolute_deviation": statistics.median(abs(v-median) for v in values),
            "confidence_interval95": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            "minimum_seconds_per_point": min(values), "maximum_seconds_per_point": max(values),
            "observed_seconds_per_point": list(values), "sessions": len(values),
            "bootstrap": {"method": "independent-session percentile median bootstrap; linear quantiles",
                          "draws": draws, "seed": seed}}


def independent_ratio(ninja, ninjars, draws=10000, seed=20261021):
    """No log-ratio color for signed/zero samples or incomparable sessions."""
    require(ninja and ninjars and all(type(v) in (int, float) and math.isfinite(v)
            for v in ninja+ ninjars), "Invalid CLI ratio samples")
    if min(ninja + ninjars) <= 0:
        return {"status": "nonpositive_samples", "ratio": None, "confidence_interval95": None,
                "reason": "Signed/zero subtraction observations remain visible; no logarithmic ratio or color"}
    rng = random.Random(seed)
    denominator = median_samples(ninja, draws, rng)
    numerator = median_samples(ninjars, draws, rng)
    ratios = [a/b for a, b in zip(numerator, denominator)]
    return {"status": "comparable", "ratio": statistics.median(ninjars)/statistics.median(ninja),
            "confidence_interval95": [percentile(ratios, .025), percentile(ratios, .975)],
            "bootstrap": {"method": "independent-session percentile ratio of medians; no paired-point claim",
                          "draws": draws, "seed": seed}}


def declared_scope(specification):
    repetitions = specification.get("repetitions", REPETITIONS)
    initial = specification.get("initial_stability_points", 100)
    require(type(repetitions) is int and repetitions >= 3 and
            type(initial) is int and initial in (20, 100),
            "CLI study requires at least three declared sessions and an explicit 20/100-point initial cache")
    return {"repetitions": repetitions, "initial_stability_points": initial}


def validate_schedule(specification):
    require(specification.get("phase") == "mg7-profile-cli-ninja-nrs-v1" and
            specification.get("ready") is True, "Wrong/unready CLI study specification")
    cases = specification["cases"]
    require(isinstance(cases, dict) and cases, "No declared CLI cases")
    scope = declared_scope(specification)
    cells = set()
    for item in specification["schedule"]:
        require(item["case"] in cases and type(item["backend"]) is int and item["backend"] in (6, 8) and
                type(item["repetition"]) is int and 0 <= item["repetition"] < scope["repetitions"] and
                type(item["npoints"]) is int and item["npoints"] ==
                    (scope["initial_stability_points"] if item["repetition"] == 0 else 0),
                "Wrong CLI Ninja/NinjaRS declared-session or initial-cache schedule")
        cell = (item["case"], item["backend"], item["repetition"])
        require(cell not in cells, "Duplicate declared CLI schedule cell")
        cells.add(cell)
    require(cells == {(cid, backend, rep) for cid in cases for backend in (6, 8)
                     for rep in range(scope["repetitions"])},
            "CLI schedule must declare both tools and all declared sessions for every case")
    return cells


def build_snapshot(results, expected, draws=10000, failures=None):
    require(isinstance(expected, dict) and set(expected) == PIN_KEYS and all(isinstance(v, str) and
            len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in expected.values()),
            "Select the exact Ninja/NinjaRS archives and CLI observer source")
    failures = [] if failures is None else failures
    require(isinstance(results, (list, tuple)) and isinstance(failures, (list, tuple)) and
            (results or failures), "No actual CLI result/failure selection")
    inputs, records, specification, specification_binding, schedule = {}, {}, None, None, None
    rejected, failure_sources = {}, set()
    authority = importlib.import_module(AUTHORITY)
    retain(authority.__file__, inputs)

    def accept_cell(value):
        nonlocal specification, specification_binding, schedule
        require(value["runtime_pins"] == expected, "CLI result runtime/source pins differ")
        if specification is None:
            specification, specification_binding = value["declared_specification"], value["specification"]
            schedule = validate_schedule(specification)
        require(value["declared_specification"] == specification and value["specification"] == specification_binding,
                "CLI results belong to different study specifications")
        item = value["item"]
        require(type(item["backend"]) is int and type(item["repetition"]) is int and
                type(item["npoints"]) is int, "CLI observation cell types differ from the declared schedule")
        key = (item["case"], item["backend"], item["repetition"])
        require(key in schedule and item in specification["schedule"], "Undeclared CLI observation")
        return key

    for filename in results:
        value = authority.report_snapshot(path(filename), expected=expected, inputs=inputs)
        retain(filename, inputs)
        require(value.get("schema") == "validated-mg7-cli-profile-result-v1" and
                value.get("acceptance_eligible") is False and value.get("timed_status_counts_available") is False,
                "CLI report requires the concrete diagnostic-only raw authority")
        key = accept_cell(value)
        require(key not in records, "Duplicate CLI case/tool/session observation")
        records[key] = value
    for filename in failures:
        value = authority.failure_snapshot(path(filename), expected=expected, inputs=inputs)
        retain(filename, inputs)
        require(value.get("schema") == "retained-mg7-cli-profile-failure-v1" and value.get("status") == "failed" and
                isinstance(value.get("error"), str) and value["error"] and
                not {"timing", "raw_timing", "acceptance_eligible"}.intersection(value),
                "CLI failures require a separate retained descriptor without timing promotion")
        key = accept_cell(value)
        identity = (value["source"]["path"], value["source"]["sha256"])
        require(identity not in failure_sources, "Duplicate retained CLI failed attempt")
        failure_sources.add(identity)
        rejected.setdefault(key, []).append({name: entry for name, entry in value.items() if name != "declared_specification"})
    scope = declared_scope(specification)
    repetitions = scope["repetitions"]
    cases = {}
    for cid, definition in specification["cases"].items():
        observed = {str(backend): [records[cid, backend, rep] for rep in range(repetitions)
                    if (cid, backend, rep) in records] for backend in (6, 8)}
        complete = all(len(series) == repetitions for series in observed.values())
        case_failures = {str(backend): [failure for rep in range(repetitions)
            for failure in rejected.get((cid, backend, rep), [])] for backend in (6, 8)}
        has_failed_attempts = any(case_failures.values())
        all_records = observed["6"] + observed["8"]
        comparable = complete and all(record["stage_scope_complete"] is True and
            authority.same_stage_scope(all_records[0]["stage_scope"], record["stage_scope"]) for record in all_records)
        modes = {}
        for mode in MODES:
            quantities = {}
            for quantity, field in QUANTITIES.items():
                samples = {backend: [record["raw_timing"][mode][field] for record in series]
                           for backend, series in observed.items()}
                quantities[quantity] = {"backend": {backend: summarize(values, draws) if values else
                    {"status": "failed_attempt" if case_failures[backend] else "pending", "sessions": 0}
                    for backend, values in samples.items()},
                    "ninjars_over_ninja": independent_ratio(samples["6"], samples["8"], draws) if comparable else
                    {"status": "incomparable_stage_scope" if complete else "failed_attempt" if has_failed_attempts else "pending", "ratio": None,
                     "confidence_interval95": None}}
            modes[mode] = quantities
        cases[cid] = {"process": definition["process"], "model": definition.get("model_name", definition.get("model")),
            "nfinal": definition.get("nfinal"), "status": "complete_comparable" if comparable else
                "complete_incomparable" if complete else "failed_attempt" if has_failed_attempts else "pending",
            "sessions": {backend: len(series) for backend, series in observed.items()},
            "required_sessions_per_backend": repetitions, "timings": modes,
            "failed_attempts": case_failures,
            "session_states": {str(backend): [{"repetition": rep,
                "state": "valid_with_retained_failures" if (cid, backend, rep) in records and (cid, backend, rep) in rejected else
                    "valid" if (cid, backend, rep) in records else "failed_attempt" if (cid, backend, rep) in rejected else "unattempted",
                "failed_attempts": len(rejected.get((cid, backend, rep), []))}
                for rep in range(repetitions)] for backend in (6, 8)},
            "observations": {backend: [{key: record[key] for key in ("source", "item", "stage_scope",
                "stage_scope_complete", "stage_point_counts", "selected_helicity_weight", "timing", "raw_timing",
                "runtime_files", "generated_sources", "timing_metadata", "stability", "new_stability_points")}
                for record in series] for backend, series in observed.items()}}
    retain(__file__, inputs)
    return {"schema": SCHEMA, "requested_tools": [6, 8], "specification": specification_binding,
            "declared_scope": scope,
            "runtime_pins": expected, "cases": cases,
            "complete": all(case["status"] == "complete_comparable" for case in cases.values()),
            "retained_failed_attempts": len(failure_sources),
            "attempt_inventory_scope": "Selected retained success/failure descriptors; final assembly must also review the controller launch ledger for attempts that produced no result file",
            "inputs_sha256": dict(sorted(inputs.items())), "qualification_eligible": False,
            "fullsum_acceptance_eligible": False, "random_dp_stability_eligible": False,
            "timed_status_counts_available": False,
            "scope": "Stock MG7 CLI direct-DP wall-time stages minus initialization, divided by actual adaptive traversal counts",
            "reduction_estimate_scope": "Full-minus-SKIP estimate including master integrals, callbacks and assembly; not a direct algorithm stopwatch",
            "uncertainty_scope": str(repetitions)+" declared independent CLI sessions per backend; empirical percentile-bootstrap intervals use only actual observations, with no padding or small-sample coverage guarantee; separate from printed three-significant-digit rounding bounds",
            "stability_scope": "First CLI sample/cache only; ordinary momenta are not retained as a matched-point random-DP authority"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="JSON list of actual CLI result paths")
    parser.add_argument("--expected", type=Path, required=True, help="Exact Ninja/NinjaRS archive and CLI source SHA256s")
    parser.add_argument("--failures", type=Path, help="JSON list of retained failed CLI result paths; no failed timing is promoted")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = path(args.output)
    require(not target.exists(), "Preserve previous CLI report snapshots")
    snapshot = build_snapshot(json.loads(path(args.results).read_text()), json.loads(path(args.expected).read_text()),
                              failures=json.loads(path(args.failures).read_text()) if args.failures else None)
    for filename in (args.results, args.expected): retain(filename, snapshot["inputs_sha256"])
    if args.failures: retain(args.failures, snapshot["inputs_sha256"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
