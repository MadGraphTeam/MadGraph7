"""Pure descriptive summaries for one frozen current-source timing schedule.

No native code, filesystem discovery, confidence interval, or performance gate is
implemented here. Numerical coefficient comparisons belong to the controller;
stable backend statuses alone do not qualify their accuracy.
"""

import copy
import json
import math
import statistics


SCHEMA = "madloop-current-source-five-backend-descriptive-timing-v1"
BACKENDS = (1, 6, 7, 8, 9)
CASE_IDS = ("qcd_n1_uux_z", "qcd_n2_gg_ttx", "ew_n2_epem_mupmum", "qcd_n4_uux_epem_mupmum")
POINTS = 100
TARGET_SECONDS = 2.0
# The retained worker's H/U aliases; T=1 or 2 denotes a steady result.
STATUS_BACKENDS = {
    (2, 1): (1, "DP"), (3, 9): (1, "QP"),
    (2, 6): (6, "DP"), (3, 8): (6, "QP"),
    (2, 7): (7, "DP"),
    (2, 8): (8, "DP"), (3, 2): (8, "QP"),
    (2, 9): (9, "DP"), (3, 3): (9, "QP"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"Invalid {name}")
    return float(value)


def schedule():
    """Ten cycles; each requested backend occupies each position twice."""
    rotations = [list(BACKENDS[i:] + BACKENDS[:i]) for i in range(5)]
    orders = rotations + [list(reversed(order)) for order in rotations]
    return [{"cycle": i + 1, "order": order} for i, order in enumerate(orders)]


def selected_mapping(value, name):
    require(isinstance(value, dict) and set(value) == set(map(str, BACKENDS)),
            f"{name} must contain exactly backends 1,6,7,8,9")
    return value


def json_copy(value, name):
    try:
        # Reject NaN/Infinity and non-JSON metadata, retaining the original keys
        # and types rather than silently normalizing an invalid input object.
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        require(json.loads(encoded) == value, f"{name} is not JSON-stable")
    except (TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name} metadata") from error
    return copy.deepcopy(value)


def decode_status(code, backend):
    integer(code, "status")
    h, t, u = code // 100, code // 10 % 10, code % 10
    actual = STATUS_BACKENDS.get((h, u))
    require(t in (1, 2) and actual is not None and actual[0] == backend,
            f"Unexpected steady status {code} for backend {backend}")
    return f"{actual[0]}:{actual[1]}"


def histogram(value, backend, count, name):
    require(isinstance(value, dict) and bool(value), f"Missing {name}")
    usage = {}
    for key, number in value.items():
        require(isinstance(key, str) and key.isascii() and key.isdecimal()
                and str(int(key)) == key, f"Invalid {name} status key")
        integer(number, f"{name} count", 1)
        actual = decode_status(int(key), backend)
        usage[actual] = usage.get(actual, 0) + number
    require(sum(value.values()) == count, f"{name} evaluation count mismatch")
    return usage


def dimensions(value, name):
    require(isinstance(value, dict), f"Missing {name}")
    require(set(value) == {"allocated", "loop_orders", "born_orders"},
            f"Invalid {name} fields")
    for key in value:
        integer(value[key], f"{name}.{key}")
    require(value["allocated"] == max(value["loop_orders"], value["born_orders"]),
            f"{name} omits physical columns")
    return value


def ready(value, backend):
    require(isinstance(value, dict), f"Missing backend {backend} initialization")
    require(type(value.get("points")) is int and value["points"] == POINTS,
            "Initialization must use all 100 points")
    positive(value.get("cold_seconds"), "initialization/warmup seconds")
    return dimensions({"allocated": value.get("ndim"),
                       "loop_orders": value.get("nso"),
                       "born_orders": value.get("nborn")}, "READY dimensions")


def observation(value, backend, loops, initial):
    """Validate the unchanged parse_outputs result, without replacing its rows."""
    require(isinstance(value, dict), "Missing parsed observation")
    elapsed = positive(value.get("seconds"), "wall-clock seconds")
    evaluations = POINTS * loops
    per_point = positive(elapsed / evaluations, "derived seconds per point")
    require(math.isclose(positive(value.get("seconds_per_point"), "seconds per point"),
                         per_point, rel_tol=1e-12, abs_tol=0),
            "Observation normalization disagrees with loops and points")
    require(positive(value.get("cold_seconds"), "observation cold seconds") ==
            initial["cold_seconds"], "Initialization timing changed between batches")
    dims = dimensions(value.get("answer_dimensions"), "observation dimensions")
    require(dims == ready(initial, backend), "Observation dimensions differ from READY")
    final = value.get("status_counts")
    histogram(final, backend, POINTS, "final traversal")
    timed = value.get("timed_status_counts")
    usage = histogram(timed, backend, evaluations, "timed traversal")
    require(all(timed.get(key, 0) >= count for key, count in final.items()),
            "Final traversal status absent from timed counts")
    require(value.get("backend_precision_counts") == usage,
            "Actual backend/precision counts disagree with status codes")
    dp_only = set(usage) == {f"{backend}:DP"}
    require(type(value.get("native_dp_only")) is bool and
            value["native_dp_only"] == dp_only, "Incorrect native-DP usage label")
    rows = value.get("rows")
    require(isinstance(rows, list) and len(rows) == POINTS * (dims["allocated"] + 1),
            "Incomplete final 100-point traversal rows")
    reconstructed = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), "Malformed numerical row")
        point, order = divmod(index, dims["allocated"] + 1)
        require(type(row.get("point")) is int and row["point"] == point + 1 and
                type(row.get("order")) is int and row["order"] == order,
                "Missing, duplicated, or unordered numerical row")
        actual = decode_status(row.get("status"), backend)
        require(type(row.get("actual_backend")) is int and
                row["actual_backend"] == backend and row.get("precision") == actual.split(":")[1],
                "Numerical row precision does not match status")
        coefficients = row.get("coefficients")
        require(isinstance(coefficients, list) and len(coefficients) == 4 and
                all(type(x) in (int, float) and math.isfinite(x) for x in coefficients),
                "Invalid Born/finite/single-pole/double-pole row")
        accuracy = row.get("accuracy")
        require(type(accuracy) in (int, float) and math.isfinite(accuracy) and accuracy >= 0,
                "Missing finite nonnegative stability estimate")
        if order == 0:
            key = str(row["status"])
            reconstructed[key] = reconstructed.get(key, 0) + 1
        else:
            require(row["status"] == rows[index - order]["status"],
                    "Split orders have different statuses")
    require(reconstructed == final, "Final status histogram differs from numerical rows")
    return {"loops": loops, "points": POINTS, "evaluations": evaluations,
            "seconds": elapsed, "seconds_per_point": per_point,
            "status_counts": copy.deepcopy(timed),
            "final_traversal_status_counts": copy.deepcopy(final),
            "backend_precision_counts": usage, "native_dp_only": dp_only}


def summarize_case(case_id, cycles, initialization, calibration):
    """Summarize ten paired descriptive cycles; inputs are never mutated.

    Each cycle is {cycle:1..10, order:[...], observations:{backend string:
    {loops:positive integer, observation:parse_outputs result}}}. Initialization
    stores the worker READY fields points/ndim/nso/nborn/cold_seconds per backend.
    Calibration stores target_seconds=2.0 and final loops per backend; any extra
    calibration records are retained verbatim. Timings may fall below the target
    after calibration; their actual positive durations remain visible.
    """
    require(case_id in CASE_IDS, "Case is outside this descriptive timing scope")
    selected_mapping(initialization, "initialization")
    selected_mapping(calibration, "calibration")
    initialization = json_copy(initialization, "initialization")
    calibration = json_copy(calibration, "calibration")
    common_dimensions = ready(initialization["1"], 1)
    for backend in BACKENDS:
        key = str(backend)
        require(ready(initialization[key], backend) == common_dimensions,
                "Backend READY dimensions differ for the same process")
        item = calibration[key]
        require(isinstance(item, dict), "Missing backend calibration")
        require(positive(item.get("target_seconds"), "calibration target") == TARGET_SECONDS,
                "Calibration target must be two seconds")
        integer(item.get("loops"), "calibrated loops", 1)
        for flag in ("numerical_regression_pass", "ordinary_target_passed"):
            if flag in item:
                require(type(item[flag]) is bool, f"Invalid controller-supplied {flag}")
    expected = schedule()
    require(isinstance(cycles, list) and len(cycles) == len(expected),
            "Exactly ten balanced timing cycles are required")
    samples = {str(backend): [] for backend in BACKENDS}
    for cycle, required in zip(cycles, expected):
        require(isinstance(cycle, dict), "Malformed cycle")
        require(type(cycle.get("cycle")) is int and cycle["cycle"] == required["cycle"] and
                isinstance(cycle.get("order"), list) and
                all(type(x) is int for x in cycle["order"]) and cycle["order"] == required["order"],
                "Cycle order differs from the frozen Latin/reversed schedule")
        observations = selected_mapping(cycle.get("observations"), "cycle observations")
        for position, backend in enumerate(required["order"], 1):
            key = str(backend)
            wrapper = observations[key]
            require(isinstance(wrapper, dict), "Missing observation wrapper")
            loops = integer(wrapper.get("loops"), "batch loops", 1)
            require(loops == calibration[key]["loops"], "Batch loops changed after calibration")
            sample = observation(wrapper.get("observation"), backend, loops, initialization[key])
            samples[key].append({"cycle": required["cycle"], "position": position, **sample})
    result = {}
    for backend in BACKENDS:
        key = str(backend)
        values = [item["seconds_per_point"] for item in samples[key]]
        median = positive(statistics.median(values), "median seconds per point")
        ratios = [{"cycle": sample["cycle"],
                   "ratio": positive(sample["seconds_per_point"] / reference["seconds_per_point"],
                                     "within-cycle timing ratio")}
                  for sample, reference in zip(samples[key], samples["8"])]
        aggregate_status, aggregate_usage = {}, {}
        for sample in samples[key]:
            for code, count in sample["status_counts"].items():
                aggregate_status[code] = aggregate_status.get(code, 0) + count
            for actual, count in sample["backend_precision_counts"].items():
                aggregate_usage[actual] = aggregate_usage.get(actual, 0) + count
        mad = statistics.median(abs(x - median) for x in values)
        require(math.isfinite(mad) and mad >= 0, "Invalid derived MAD")
        result[key] = {
            "requested_backend": backend, "sample_count": len(values),
            "median_seconds_per_point": median,
            "mad_seconds_per_point": mad,
            "mad_scale": "unscaled",
            "min_seconds_per_point": min(values), "max_seconds_per_point": max(values),
            "samples": samples[key], "median_ratio_to_8": positive(
                statistics.median(x["ratio"] for x in ratios), "median within-cycle ratio"),
            "ratios_to_8": ratios, "aggregate_status_counts": aggregate_status,
            "aggregate_backend_precision_counts": aggregate_usage,
            "native_dp_only": set(aggregate_usage) == {f"{backend}:DP"},
        }
        for flag in ("numerical_regression_pass", "ordinary_target_passed"):
            if flag in calibration[key]:
                result[key][flag] = calibration[key][flag]
        if any(flag in calibration[key] for flag in ("numerical_regression_pass", "ordinary_target_passed")):
            result[key]["numerical_check_source"] = "controller-supplied calibration metadata"
    return {"schema": SCHEMA, "case_id": case_id, "performance_acceptance": False,
            "cpu_reservation": False, "clock": "SYSTEM_CLOCK wall", "scope": "full amplitude",
            "points_per_traversal": POINTS, "cycle_count": len(expected),
            "timed_status_scope": "all evaluations", "numerical_rows_scope": "final traversal only",
            "calibration_target_seconds": TARGET_SECONDS, "schedule": expected,
            "initialization": initialization, "calibration": calibration, "backends": result}
