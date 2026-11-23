"""Extract numerical evidence from a worker's final steady warm-up traversal.

Warm-up observations never supply a warmed runtime or performance-acceptance
input. The READY cold time includes initialization and the whole warm-up.
"""

import math
from pathlib import Path

import tirust_benchmark_worker as worker


MeasurementError = worker.MeasurementError


def _call_groups(text):
    """Group the chronological trace without discarding invalid trailing calls."""
    groups = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError("Malformed warm-up numerical row")
        call = int(fields[0])
        if call < 0:
            raise ValueError("Warm-up call indices must be nonnegative")
        if not groups or groups[-1][0] != call:
            if groups and call <= groups[-1][0]:
                raise ValueError("Warm-up call indices are duplicated or unordered")
            groups.append((call, []))
        groups[-1][1].append(fields)
    # Call zero bootstraps MadLoop. Its status and numerical fields can be
    # unknown; neither it nor any pre-window call contributes evidence.
    return [group for group in groups if group[0] > 0]


def parse_warmup(directory, backend, points, ready):
    """Validate the final ``points`` consecutive calls and return an observation.

    ``ready`` contains the worker's points, ndim, nso, nborn and cold_seconds.
    Each selected call must be steady, follow the worker's cyclic point order,
    and contain every answer column, including Born-only split orders. The
    returned rows are sorted by point/order like ``worker.parse_outputs``.
    Raw text is retained unchanged on both success and ``MeasurementError``.
    """
    raw = {}
    try:
        directory = Path(directory)
        for name in ("warmup.dat", "dimensions.dat"):
            path = directory / name
            if path.is_file():
                # Disable newline translation so the evidence retains CRLF,
                # whitespace and comments exactly as emitted by the worker.
                with path.open(encoding="utf-8", newline="") as stream:
                    raw[name] = stream.read()

        if isinstance(points, bool) or not isinstance(points, int) or points <= 0:
            raise ValueError("points must be a positive integer")
        selected = (backend,) if isinstance(backend, int) else tuple(backend)
        if not selected or any(isinstance(value, bool) or not isinstance(value, int)
                               or value not in (1, 6, 7, 8, 9) for value in selected):
            raise ValueError("Expected backend or nonempty chain drawn from 1,6,7,8,9")

        for name in ("points", "ndim", "nso", "nborn"):
            value = ready[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid READY dimensions")
        if ready["points"] != points:
            raise ValueError("READY point count differs from the requested traversal")
        cold = ready["cold_seconds"]
        if (isinstance(cold, bool) or not isinstance(cold, (int, float))
                or not math.isfinite(cold) or cold <= 0):
            raise ValueError("READY cold time must be finite and positive")
        dimensions = list(map(int, raw["dimensions.dat"].split()))
        if len(dimensions) != 3:
            raise ValueError("Invalid worker dimensions")
        ndim, nso, nborn = dimensions
        if min(nso, nborn) < 0 or ndim != max(nso, nborn):
            raise ValueError("Worker answer dimension omits physical columns")
        if dimensions != [ready[name] for name in ("ndim", "nso", "nborn")]:
            raise ValueError("Worker dimensions differ from READY")

        groups = _call_groups(raw["warmup.dat"])
        if len(groups) < points:
            raise ValueError("Warm-up trace lacks a complete steady traversal")
        window = groups[-points:]
        calls = [call for call, _ in window]
        if calls != list(range(calls[0], calls[0] + points)):
            raise ValueError("Final warm-up call indices are not consecutive")

        rows, counts, usage, call_point_order = [], {}, {}, []
        for call, records in window:
            point = 1 + (call - 1) % points
            call_rows = []
            for fields in records:
                code = int(fields[3])
                actual = worker.validate_status(code, selected)
                row = {"point": int(fields[1]), "order": int(fields[2]), "status": code,
                       "coefficients": list(map(float, fields[4:8])), "accuracy": float(fields[8]),
                       "actual_backend": actual["backend"], "precision": actual["precision"]}
                if not all(math.isfinite(value) for value in row["coefficients"] + [row["accuracy"]]):
                    raise ValueError("Nonfinite warm-up numerical row")
                if row["accuracy"] < 0:
                    raise ValueError("Missing warm-up stability estimate")
                if row["point"] != point:
                    raise ValueError("Warm-up point does not follow the cyclic call mapping")
                call_rows.append(row)
            if sorted(row["order"] for row in call_rows) != list(range(ndim + 1)):
                raise ValueError("Missing or duplicated warm-up answer columns")
            if len({row["status"] for row in call_rows}) != 1:
                raise ValueError("Split orders have inconsistent warm-up evaluation statuses")
            status = str(call_rows[0]["status"])
            counts[status] = counts.get(status, 0) + 1
            key = f"{call_rows[0]['actual_backend']}:{call_rows[0]['precision']}"
            usage[key] = usage.get(key, 0) + 1
            rows.extend(call_rows)
            call_point_order.append({"call": call, "point": point})

        rows.sort(key=lambda row: (row["point"], row["order"]))
        if [(row["point"], row["order"]) for row in rows] != [
                (point, order) for point in range(1, points + 1) for order in range(ndim + 1)]:
            raise ValueError("Final warm-up window does not cover every point exactly once")
        return {"cold_seconds": cold, "seconds": None, "seconds_per_point": None,
                "rows": rows, "status_counts": counts, "timed_status_counts": None,
                "backend_precision_counts": usage,
                "native_dp_only": set(usage) == {f"{selected[0]}:DP"},
                "answer_dimensions": {"allocated": ndim, "loop_orders": nso, "born_orders": nborn},
                "measurement_phase": "steady-warmup", "performance_eligible": False,
                "warmup_first_call": calls[0], "warmup_last_call": calls[-1],
                "warmup_call_point_order": call_point_order, "raw_outputs": raw}
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as error:
        raise MeasurementError(str(error), raw) from error
