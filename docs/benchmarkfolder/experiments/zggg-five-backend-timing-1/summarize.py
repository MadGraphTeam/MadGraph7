#!/usr/bin/env python3
"""Pure replay of the requested zggg five-backend shared-host pilot.

``derive(directory)`` reads retained evidence and returns a deterministic JSON
object.  It neither imports a controller nor executes a native program.  The
command line writes only ``directory/summary.json`` after closure receipts exist.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import random
import statistics


BACKENDS = (1, 6, 7, 8, 9)
NAMES = {1: "CutTools", 6: "Ninja", 7: "COLLIER", 8: "NinjaRS", 9: "TIRust"}
CASE_ID = "qcd_n4_uux_zggg"
POINTS = 10
FIXTURE_POINTS = 100
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_RESAMPLES = 10000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and value > 0, "Invalid " + name)
    return value


def _integer(value, name):
    _require(not isinstance(value, bool) and isinstance(value, int) and value > 0,
             "Invalid " + name)
    return value


def _percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _bootstrap(ratios):
    if len(ratios) < 2:
        return None
    generator = random.Random(BOOTSTRAP_SEED)
    size = len(ratios)
    medians = sorted(statistics.median(
        [ratios[generator.randrange(size)] for _ in range(size)])
        for _ in range(BOOTSTRAP_RESAMPLES))
    return {
        "median_ratio": statistics.median(ratios),
        "lower_95": _percentile(medians, .025),
        "upper_95": _percentile(medians, .975),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "paired_cycles": size,
        "method": "Descriptive percentile bootstrap of the median within-cycle "
                  "backend/NinjaRS ratio; resample whole matched cycles. "
                  "Balanced five-backend cycles are not adjacent alternating pairs.",
        "performance_acceptance": False,
    }


def derive(directory):
    """Derive all five lanes, including failures, from a closed evidence folder.

    Complete numerically accepted lanes receive headline timing summaries.
    Failed/incomplete lanes retain their raw samples and paired ratios, while
    headline timing fields remain null.  Warm-up and calibration usage never
    contributes to the aggregate timed-cycle precision counts.
    """
    directory = Path(directory).resolve()
    inputs = {}

    def read(relative, required=True):
        path = directory / relative
        if not path.is_file():
            _require(not required, "Missing replay input: " + str(relative))
            return None
        raw = path.read_bytes()
        inputs[str(Path(relative).as_posix())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    spec = read("specification.json")
    config = read("configuration.json")
    compilation = read("compilation.json")
    observations = read("observations.json", required=False)
    global_failure = read("failure.json", required=False)
    _require(observations is not None or global_failure is not None,
             "No terminal observation or failure receipt; do not summarize a live run")
    shared_after = read("shared-after.json")
    shared_before = read("shared-before.json", required=False)
    shared_before_path = "shared-before.json"
    if shared_before is None:
        shared_before_path = "../shared-before.json"
        shared_before = read(shared_before_path, required=False)
    shared_unchanged = shared_before is not None and shared_after == shared_before
    shared_integrity_failure = read("shared-integrity-failure.json", required=False)
    if shared_integrity_failure is not None:
        shared_unchanged = False
    cleanup_failures = []
    for path in sorted(directory.glob("cleanup-failure-*.json")):
        cleanup_failures.append({"path": path.name, "result": read(path.name)})

    case = spec["case"]
    _require(case["id"] == CASE_ID, "This summarizer is scoped to " + CASE_ID)
    _require(spec["points_per_traversal"] == POINTS
             and config["points_per_case"] == POINTS, "Expected the ten-point pilot")
    _require(spec.get("fixture_points", FIXTURE_POINTS) == FIXTURE_POINTS,
             "Expected a 100-point frozen fixture")
    _require(spec["performance_acceptance"] is False,
             "Shared-host pilot must not become a performance acceptance result")
    _require(spec["frozen_membership_one_based"] == list(range(1, POINTS + 1)),
             "Pilot membership is not the frozen first ten points")
    schedule = spec["cycles"]
    expected_cycles = [row["cycle"] for row in schedule]
    _require(len(expected_cycles) == 10 and len(set(expected_cycles)) == 10,
             "Expected ten distinct balanced cycles")
    for row in schedule:
        _require(sorted(row["order"]) == sorted(BACKENDS),
                 "Invalid five-backend cycle order")

    terminal = observations if observations is not None else global_failure
    lane_records = terminal.get("lanes", {})
    checks_receipt = terminal.get("checks", {})
    receipt_cycles = terminal.get("cycles", [])
    disk_cycles = {}
    for path in sorted(directory.glob("cycle-*.json")):
        row = read(path.name)
        cycle = _integer(row["cycle"], "cycle number")
        _require(cycle not in disk_cycles, "Duplicate cycle receipt")
        disk_cycles[cycle] = row
    _require(set(disk_cycles).issubset(expected_cycles), "Unexpected timed cycle")
    _require(len(receipt_cycles) == len(disk_cycles),
             "Terminal and individual cycle receipts differ")
    for row in receipt_cycles:
        _require(disk_cycles.get(row["cycle"]) == row,
                 "Terminal and individual cycle observations differ")

    backends = {}
    for backend in BACKENDS:
        key = str(backend)
        lane = dict(lane_records.get(key, {}))
        lane_inputs = read("lanes/" + key + "/inputs.json", required=False)
        if lane_inputs is not None:
            _require(lane_inputs["physical_card_sha256"] == compilation["parameter_card_sha256"]
                     and lane_inputs["executable_sha256"] == compilation["executable_sha256"]
                     and lane_inputs["points_sha256"] == spec["pilot_points_sha256"]
                     and lane_inputs["compiler_flags"] == spec["compiler_flags"],
                     "Lane physical card, executable, points or compiler flags changed")
        failure = read("lanes/" + key + "/failure.json", required=False)
        ready = read("lanes/" + key + "/ready.json", required=False)
        if ready is None:
            ready = lane.get("initialization")
        if failure is not None:
            lane.update(failure)
        check_paths = []
        learning_path = Path("lanes") / key / "learning-comparison.json"
        if (directory / learning_path).is_file():
            check_paths.append(learning_path)
        check_paths.extend(path.relative_to(directory) for path in sorted(
            (directory / "lanes" / key / "raw").glob("*/comparison.json")))
        checks = [{"path": str(path.as_posix()), "result": read(path)}
                  for path in check_paths]
        check_by_path = {item["path"]: item["result"] for item in checks}
        # Join the controller's terminal checks to their actual retained files.
        for item in checks_receipt.get(key, []):
            parts = Path(item["path"]).parts
            _require("lanes" in parts, "Unrecognized numerical check path")
            relative = str(Path(*parts[parts.index("lanes"):]).as_posix())
            _require(check_by_path.get(relative) == item["result"],
                     "Retained comparison differs from terminal check receipt")
        numerical_pass = (all(item["result"]["passed"] for item in checks)
                          if checks else None)
        ordinary_pass = (all(item["result"]["ordinary_target_passed"]
                             for item in checks) if checks else None)
        numerical_failures = [
            {"path": item["path"], "failures": item["result"].get("failures", [])}
            for item in checks if not item["result"]["passed"]]
        samples = []
        usage = collections.Counter()
        statuses = collections.Counter()
        ratios = []
        for planned in schedule:
            cycle = planned["cycle"]
            row = disk_cycles.get(cycle)
            if row is None:
                continue
            _require(row["order"] == planned["order"], "Cycle order changed")
            entry = row["observations"].get(key)
            if entry is None:
                continue
            loops = _integer(entry["loops"], "timed loops")
            observation = entry["observation"]
            seconds = _positive(observation["seconds"], "Fortran batch seconds")
            per_point = _positive(observation["seconds_per_point"], "seconds per point")
            evaluations = loops * POINTS
            _require(math.isclose(per_point, seconds / evaluations, rel_tol=1e-12),
                     "Seconds per point does not match batch/evaluation count")
            _require(sum(observation["backend_precision_counts"].values()) == evaluations
                     and sum(observation["timed_status_counts"].values()) == evaluations,
                     "Cycle usage histogram does not cover every timed evaluation")
            position = _integer(entry["position"], "cycle position")
            _require(position == planned["order"].index(backend) + 1,
                     "Backend position differs from balanced schedule")
            observation_path = Path("lanes") / key / "raw" / ("cycle-" + str(cycle)) / "observation.json"
            _require(read(observation_path) == observation,
                     "Cycle receipt differs from retained observation")
            comparison_path = str((observation_path.parent / "comparison.json").as_posix())
            _require(comparison_path in check_by_path, "Timed traversal lacks numerical comparison")
            samples.append({
                "cycle": cycle, "position": position, "loops": loops,
                "evaluations": evaluations, "evals": evaluations, "seconds": seconds,
                "seconds_per_point": per_point,
                "backend_precision_counts": observation["backend_precision_counts"],
                "usage": observation["backend_precision_counts"],
                "status_counts": observation["status_counts"],
                "timed_status_counts": observation["timed_status_counts"],
                "native_dp_only": observation["native_dp_only"],
                "observation_path": str(observation_path.as_posix()),
                "comparison_path": comparison_path,
            })
            usage.update(observation["backend_precision_counts"])
            statuses.update(observation["timed_status_counts"])
            reference = row["observations"].get("8")
            if reference is not None:
                denominator = _positive(reference["observation"]["seconds_per_point"],
                                        "same-cycle NinjaRS seconds per point")
                ratios.append({"cycle": cycle, "ratio": per_point / denominator,
                               "numerator_seconds_per_point": per_point,
                               "denominator_seconds_per_point": denominator})

        status = lane.get("status", "unavailable")
        complete = status == "complete" and len(samples) == len(schedule)
        if numerical_pass is False:
            status = "failed-numerical-comparison"
        elif status == "complete" and not complete:
            status = "incomplete-cycles"
        elif status == "complete" and numerical_pass is None:
            status = "unqualified-missing-numerical-comparisons"
        elif status == "complete" and global_failure is not None:
            status = "unqualified-global-failure"
        elif status == "complete" and cleanup_failures:
            status = "unqualified-worker-cleanup"
        elif status == "complete" and not shared_unchanged:
            status = "unqualified-shared-input-closure"
        timing_available = (complete and numerical_pass is True
                            and not cleanup_failures and global_failure is None
                            and shared_unchanged)
        times = [sample["seconds_per_point"] for sample in samples]
        median = statistics.median(times) if timing_available else None
        headline_ratios = ([item["ratio"] for item in ratios]
                           if timing_available and len(ratios) == len(schedule) else [])
        backends[key] = {
            "backend": backend, "name": NAMES[backend], "status": status,
            "timing_available": timing_available,
            "sample_count": len(samples), "expected_sample_count": len(schedule),
            "median_seconds_per_point": median,
            "mad_seconds_per_point": (statistics.median(abs(value - median) for value in times)
                                      if timing_available else None),
            "min_seconds_per_point": min(times) if timing_available else None,
            "max_seconds_per_point": max(times) if timing_available else None,
            "samples": samples,
            "paired_ratios_to_8": ratios,
            "median_ratio_to_8": statistics.median(headline_ratios) if headline_ratios else None,
            "paired_ratio_to_8_bootstrap_95": _bootstrap(headline_ratios),
            "aggregate_backend_precision_counts": dict(sorted(usage.items())),
            "aggregate_timed_status_counts": dict(sorted(statuses.items())),
            "aggregate_evaluations": sum(usage.values()),
            "usage_scope": "Timed cycles only; excludes initialization, learning and calibration",
            "native_dp_only": all(sample["native_dp_only"] for sample in samples) if samples else None,
            "initialization": ready,
            "initialization_scope": spec["initialization_scope"],
            "numerical_regression_pass": numerical_pass,
            "ordinary_target_passed": ordinary_pass,
            "regression_tolerance": 3e-6, "ordinary_target_tolerance": 1e-8,
            "numerical_check_count": len(checks), "numerical_checks": checks,
            "numerical_failures": numerical_failures,
            "failure": failure,
            "settings": lane.get("settings"),
            "input_binding": lane_inputs,
        }

    # A headline ratio requires a complete, numerically accepted denominator.
    if not backends["8"]["timing_available"]:
        for lane in backends.values():
            lane["median_ratio_to_8"] = None
            lane["paired_ratio_to_8_bootstrap_95"] = None
    all_complete = all(lane["timing_available"] for lane in backends.values())
    return {
        "schema": "madloop-zggg-five-backend-pilot-summary-v1",
        "status": "complete" if all_complete else "closed-with-failures-or-incomplete-lanes",
        "case": case, "process_case": CASE_ID,
        "scope": "Descriptive shared-host timing pilot; ten frozen points from a 100-point fixture",
        "point_count": POINTS, "fixture_point_count": FIXTURE_POINTS,
        "frozen_membership_one_based": spec["frozen_membership_one_based"],
        "seed": spec["seed"], "full_points_sha256": spec["full_points_sha256"],
        "pilot_points_sha256": spec["pilot_points_sha256"],
        "mg_revision": spec["mg_revision"], "tirust_revision": spec["tirust_revision"],
        "tirust_dso_sha256": spec["tirust_dso_sha256"],
        "executable_sha256": compilation["executable_sha256"],
        "compiler": spec["compiler"], "compiler_flags": spec["compiler_flags"],
        "runtime": spec["runtime"], "machine": spec["machine"], "cpu": spec["cpu"],
        "cpu_reservation": spec["cpu_reservation"], "host_exclusivity": spec["host_exclusivity"],
        "finite_and_poles": spec["finite_and_poles"],
        "cycle_schedule": schedule, "completed_cycle_receipts": sorted(disk_cycles),
        "calibration_target_seconds": spec["calibration_target_seconds"],
        "dispersion": "Unscaled median absolute deviation of cycle seconds per point",
        "ratio_method": "Median of matched within-cycle backend/NinjaRS ratios; balanced five-backend order",
        "performance_acceptance": False,
        "acceptance_note": "No baseline/runtime acceptance conclusion. Precise local timing and the "
                           "100-point campaign remain separate requirements; bootstrap intervals are descriptive.",
        "closure": {"terminal_receipt": "observations.json" if observations is not None else "failure.json",
                    "shared_after_receipt": "shared-after.json",
                    "shared_before_receipt": shared_before_path if shared_before is not None else None,
                    "shared_unchanged": shared_unchanged,
                    "cleanup_failures": cleanup_failures},
        "global_failure": global_failure,
        "controller_physics_regression_pass": terminal.get("physics_regression_pass"),
        "shared_integrity_failure": shared_integrity_failure,
        "backends": backends,
        "input_sha256": dict(sorted(inputs.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "measurements")
    args = parser.parse_args()
    result = derive(args.directory)
    destination = args.directory / "summary.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(destination.resolve())


if __name__ == "__main__":
    main()
