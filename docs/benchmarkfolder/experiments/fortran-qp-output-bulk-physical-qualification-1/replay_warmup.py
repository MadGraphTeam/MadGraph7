"""File-free replay of the frozen explicit-TIRust physical learning trace.

Derived from tools/tirust_benchmark_warmup.py, preserving its final-steady-window
semantics, chronological grouping and per-call counts. This intentionally
qualifies only the recorded100-point/NDIM1 explicit9 cohort. No live helper is
imported, and no filesystem, process, loader or native interface is available.
"""
import math

ORIGINAL_WARMUP_SHA256 = "be6b33f8f51f49484f0b75ed1eabcb62f5bc4a3f25f37666adff0d4029d7617d"
ORIGINAL_WORKER_SHA256 = "71a460579b2cabe8a1485fa0eed406e98988ec019b6c195bd78af55eedf99d57"
REFERENCE_FIELDS = (
    "rows", "status_counts", "backend_precision_counts", "warmup_first_call",
    "warmup_last_call", "warmup_call_point_order", "answer_dimensions",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def _status(code):
    require(isinstance(code, int) and not isinstance(code, bool) and
            code in (219, 323), "Learning call is not a frozen stable TIRust9 result")
    return "DP" if code == 219 else "QP"


def _call_groups(text):
    groups = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        require(len(fields) == 9, "Malformed warm-up numerical row")
        call = int(fields[0])
        require(call >= 0, "Warm-up call indices must be nonnegative")
        if not groups or groups[-1][0] != call:
            require(not groups or call > groups[-1][0], "Warm-up call indices are duplicated or unordered")
            groups.append((call, []))
        groups[-1][1].append(fields)
    # Preserve the bootstrap record in raw text; its numerical values/status
    # intentionally do not contribute steady-window evidence.
    return [group for group in groups if group[0] > 0]


def replay_warmup(observation, expected, ready=None):
    """Reconstruct and authenticate one complete retained warmup.json value.

    ``observation`` supplies untouched raw_outputs text and a validated positive
    cold time. ``expected`` is the original closed warmup reference. Optional
    ``ready`` binds actual startup dimensions/cold time to the same observation.
    No observed result fields are trusted without rederivation from the text.
    """
    require(isinstance(observation, dict) and isinstance(expected, dict), "Expected observation/reference objects")
    raw = observation["raw_outputs"]
    require(isinstance(raw, dict) and set(raw) == {"warmup.dat", "dimensions.dat"} and
            all(isinstance(value, str) for value in raw.values()), "Missing/extra raw learning text")
    raw = dict(raw)
    dimensions = list(map(int, raw["dimensions.dat"].split()))
    require(dimensions == [1, 1, 1], "Physical learning requires complete NDIM1/two-order columns")
    if ready is None:
        ready = {"points": 100, "ndim": 1, "nso": 1, "nborn": 1,
                 "cold_seconds": observation["cold_seconds"]}
    for key, expected_value in [("points", 100), ("ndim", 1), ("nso", 1), ("nborn", 1)]:
        require(isinstance(ready[key], int) and not isinstance(ready[key], bool) and
                ready[key] == expected_value, "READY learning dimensions differ")
    cold = ready["cold_seconds"]
    require(isinstance(cold, (int, float)) and not isinstance(cold, bool) and
            math.isfinite(cold) and cold > 0, "READY cold time must be finite and positive")
    require(cold == observation["cold_seconds"], "Learning cold time differs from READY")
    groups = _call_groups(raw["warmup.dat"])
    require(len(groups) >= 100, "Warm-up trace lacks a complete steady traversal")
    window = groups[-100:]
    calls = [call for call, _ in window]
    require(calls == list(range(calls[0], calls[0]+100)), "Final learning calls have gaps")
    rows, counts, usage, call_point_order = [], {}, {}, []
    for call, records in window:
        point = 1 + (call-1) % 100
        call_rows = []
        for fields in records:
            code = int(fields[3])
            precision = _status(code)
            row = {
                "point": int(fields[1]), "order": int(fields[2]), "status": code,
                "coefficients": list(map(float, fields[4:8])), "accuracy": float(fields[8]),
                "actual_backend": 9, "precision": precision,
            }
            require(all(math.isfinite(value) for value in row["coefficients"]+[row["accuracy"]]),
                    "Nonfinite learning numerical row")
            require(row["accuracy"] >= 0, "Missing learning stability estimate")
            require(row["point"] == point, "Learning point does not follow cyclic call mapping")
            call_rows.append(row)
        require(sorted(row["order"] for row in call_rows) == [0, 1],
                "Missing or duplicated learning answer column")
        require(len({row["status"] for row in call_rows}) == 1,
                "Learning split-order status differs")
        status = str(call_rows[0]["status"])
        counts[status] = counts.get(status, 0)+1
        key = "9:"+call_rows[0]["precision"]
        usage[key] = usage.get(key, 0)+1
        rows.extend(call_rows)
        call_point_order.append({"call": call, "point": point})
    rows.sort(key=lambda row: (row["point"], row["order"]))
    require([(row["point"], row["order"]) for row in rows] ==
            [(point, order) for point in range(1,101) for order in (0,1)],
            "Final learning window does not cover every physical point once")
    parsed = {
        "cold_seconds": cold, "seconds": None, "seconds_per_point": None,
        "rows": rows, "status_counts": counts, "timed_status_counts": None,
        "backend_precision_counts": usage, "native_dp_only": set(usage) == {"9:DP"},
        "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
        "measurement_phase": "steady-warmup", "performance_eligible": False,
        "warmup_first_call": calls[0], "warmup_last_call": calls[-1],
        "warmup_call_point_order": call_point_order, "raw_outputs": raw,
    }
    require(parsed == observation, "Retained learning observation differs from its raw text")
    for field in REFERENCE_FIELDS:
        require(parsed[field] == expected[field], "Learning changed from original reference: "+field)
    return parsed
