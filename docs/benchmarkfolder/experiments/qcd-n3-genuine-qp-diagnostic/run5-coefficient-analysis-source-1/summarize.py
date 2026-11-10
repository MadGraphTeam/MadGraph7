"""Bounded arithmetic summary of explicitly supplied closed diagnostic data.

No discovery, reduction, gate change or physics oracle. Returned ANS values are
binary64 even when their internal reduction used higher precision.
"""
import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re

BACKENDS = (1, 6, 8, 9)
POINTS = (1, 93, 39)
ARMS = tuple((backend, mode) for backend in BACKENDS for mode in (4, 5))
THRESHOLDS = (1e-14, 1e-12, 1e-10, 1e-8, 3e-6)
METRICS = ("absolute", "relative_to_left", "symmetric")
SLOTS = tuple({"slot_zero_based": 4 * sector + order, "fortran_index": [order, sector],
               "sector": "total" if sector == 0 else "split_order_1",
               "coefficient": ("Born", "finite", "single_pole", "double_pole")[order]}
              for sector in (0, 1) for order in range(4))
MAX_INPUT_BYTES = 64 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    require(type(value) in (float, int), "Coefficient must be a numeric scalar, not bool")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("Coefficient outside binary64 reporting range") from error
    require(math.isfinite(value), "Nonfinite coefficient")
    return value


def delta(left, right):
    """The frozen analyze.delta formula, with explicit finite-intermediate guards."""
    left, right = number(left), number(right)
    difference, denominator = abs(left - right), left + right
    require(math.isfinite(difference) and math.isfinite(denominator), "Nonfinite metric intermediate")
    answer = {"absolute": difference,
              "relative_to_left": difference / abs(left) if left else (0.0 if not difference else None),
              "symmetric": difference / abs(denominator) if denominator else (0.0 if not difference else None),
              "opposite_nonzero_sum": denominator == 0 and difference != 0}
    require(all(v is None or math.isfinite(v) for k, v in answer.items() if k != "opposite_nonzero_sum"), "Nonfinite metric")
    return answer


def expected_comparisons(keyed):
    output = []
    for left, right in combinations(ARMS, 2):
        if left[0] != right[0] and left[1] != right[1]:
            continue
        for index, point in enumerate(POINTS):
            a, b = keyed[*left, 1]["points"][index], keyed[*right, 1]["points"][index]
            output.append({"left": list(left), "right": list(right), "point": point,
                           "shared": a["inputs"]["shared_input_sha256"] == b["inputs"]["shared_input_sha256"],
                           "left_values": a["coefficients"], "right_values": b["coefficients"]})
    require(len(output) == 48, "Expected exactly48 pair/point identities")
    return output


def summaries(entries, slot):
    output = {}
    for metric in METRICS:
        maximum, witness = None, None
        defined, undefined = 0, 0
        counts = {threshold: 0 for threshold in THRESHOLDS}
        for row in entries:
            value = row["deltas"][slot][metric]
            if value is None:
                undefined += 1
                continue
            defined += 1
            if maximum is None or value > maximum:
                maximum = value
                witness = {"left": row["left"], "right": row["right"], "point": row["point"],
                           "left_value": row["left_values"][slot], "right_value": row["right_values"][slot]}
            for threshold in THRESHOLDS:
                counts[threshold] += int(value > threshold)
        output[metric] = {"defined_count": defined, "undefined_count": undefined,
                          "maximum": maximum, "maximum_witness": witness,
                          "diagnostic_threshold_counts": [{"threshold": t, "strictly_above": counts[t], "defined_denominator": defined}
                                                          for t in THRESHOLDS]}
    return output


def summarize(data):
    combined = isinstance(data, dict)
    if combined:
        require(data.get("schema") == "qcd-n3-genuine-qp-diagnostic-v1", "Unexpected combined analysis schema")
        require(data.get("host_checked") is False and data.get("timing_or_campaign_acceptance") is False,
                "Diagnostic must remain unchecked and outside acceptance")
        workers = data.get("workers")
    else:
        workers = data
    require(isinstance(workers, list) and len(workers) == 16, "Exactly16 workers required")
    keyed = {}
    for worker in workers:
        require(isinstance(worker, dict), "Worker must be an object")
        identity = (worker.get("backend"), worker.get("mode"), worker.get("observed"))
        require(all(type(v) is int for v in identity), "Worker identity must contain integers")
        require(identity in {(b, m, o) for b, m in ARMS for o in (0, 1)} and identity not in keyed, "Invalid/duplicate worker identity")
        require(worker.get("host_checked") is False and worker.get("warnings") == [], "Unexpected checked worker or warnings")
        points = worker.get("points")
        require(isinstance(points, list) and len(points) == 3 and
                all(isinstance(p, dict) and type(p.get("point")) is int for p in points) and
                tuple(p.get("point") for p in points) == POINTS,
                "Exact ordered three-point set required")
        for point in points:
            require(type(point.get("status")) is int and point["status"] == 120 and point.get("host_checked") is False,
                    "Every fixed diagnostic must remain unchecked120")
            require(point.get("accuracy") == [-1.0, -1.0] and all(type(v) in (int, float) for v in point["accuracy"]),
                    "Unchecked accuracy must remain[-1,-1]")
            values = point.get("coefficients")
            require(isinstance(values, list) and len(values) == 8, "Exactly8 ANS coefficients required")
            for value in values:
                number(value)
            if identity[2]:
                inputs = point.get("inputs")
                require(isinstance(inputs, dict) and isinstance(inputs.get("shared_input_sha256"), str) and
                        re.fullmatch(r"[0-9a-f]{64}", inputs["shared_input_sha256"]), "Malformed shared-input identity")
            else:
                require(point.get("inputs") is None, "Plain workers must not claim observed native inputs")
        keyed[identity] = worker
    require(len(keyed) == 16, "Worker matrix incomplete")
    raw_present = ["public_raw" in w for w in workers]
    if any(raw_present):
        require(all(raw_present), "Partial public parity evidence")
        for backend, mode in ARMS:
            require(keyed[backend, mode, 0]["public_raw"] == keyed[backend, mode, 1]["public_raw"], "Plain/observed public parity mismatch")
    else:
        require(combined, "Parsed-worker input requires its complete public parity evidence")
    entries = expected_comparisons(keyed)
    admissible = all(row["shared"] for row in entries)
    if combined:
        supplied = data.get("comparisons")
        require(isinstance(supplied, list) and len(supplied) == 48, "Exactly48 supplied comparisons required")
        for actual, expected in zip(supplied, entries):
            require(isinstance(actual, dict), "Comparison must be an object")
            require((actual.get("left"), actual.get("right"), actual.get("point")) ==
                    (expected["left"], expected["right"], expected["point"]), "Comparison order/identity mismatch")
            require(actual.get("shared_physical_born_repair_numerator_equal") is expected["shared"], "Comparison shared-input flag disagrees")
            require(actual.get("host_checked") is False and actual.get("native_basis_identity_claimed") is False,
                    "Comparison overstates host/basis qualification")
            if not expected["shared"]:
                require(actual.get("coefficients") is None, "Rejected shared-input comparison must suppress coefficients")
        require(data.get("all_shared_input_gates_pass") is admissible, "Aggregate shared-input flag disagrees")
    result = {"schema": "qcd-n3-returned-coefficient-diagnostic-summary-1",
              "scope": "Discrepancies of unchecked returned REAL*8 ANS coefficients; no oracle or accuracy/acceptance claim.",
              "coefficient_storage": "Fortran ANS(0:3,0:1), column-major TRANSFER of8 REAL*8 values",
              "slots": list(SLOTS), "worker_count": 16, "observed_pair_point_comparisons": 48,
              "potential_scalar_comparisons": 384, "diagnostic_thresholds": list(THRESHOLDS),
              "threshold_interpretation": "Strict > discrepancy counts only; absolute thresholds use returned amplitude units.",
              "host_checked": False, "accuracy": [-1.0, -1.0], "status": 120,
              "independent_oracle_available": False, "scientific_acceptance": False, "timing_or_campaign_acceptance": False,
              "all_shared_input_gates_pass": admissible,
              "shared_input_gate_meaning": "Common-input consistency only; it does not test numerical coefficient agreement or true error.",
              "public_parity_scope": "Rechecked from all16 parsed public records" if all(raw_present) else "Inherited from supplied closed adjudicate analysis; public records are not included there",
              "precision_transport": {"1": "CutTools binary64 denominators and returned native values with genuine QP numerator callbacks",
                                      "6": "Ninja binary128 native transport and QP numerator construction",
                                      "8": "NinjaRS decimal input recovered at113bits;256bit engine",
                                      "9": "TIRust decimal input recovered at113bits;256bit engine",
                                      "returned_ANS": "All eight reported coefficients are REAL*8"}}
    if not admissible:
        result.update(numerical_summary_suppressed=True, scalar_comparisons_reported=0, slot_summaries=None,
                      pair_breakdown=[], cuttools_reference_rows=[], observed_coefficients=[],
                      shared_input_mismatches=[{k: row[k] for k in ("left", "right", "point")} for row in entries if not row["shared"]])
        return result
    for index, row in enumerate(entries):
        row["deltas"] = [delta(a, b) for a, b in zip(row["left_values"], row["right_values"])]
        if combined:
            require(data["comparisons"][index].get("coefficients") == row["deltas"], "Provided coefficient deltas do not reproduce frozen formula")
    by_pair = {}
    for row in entries:
        key = (*row["left"], *row["right"])
        by_pair.setdefault(key, []).append(row)
    breakdown = []
    for rows in by_pair.values():
        breakdown.append({"left": rows[0]["left"], "right": rows[0]["right"],
                          "kind": "direction_modes" if rows[0]["left"][0] == rows[0]["right"][0] else "backends_same_mode",
                          "points": [{"point": row["point"], **{metric + "_by_slot": [d[metric] for d in row["deltas"]] for metric in METRICS},
                                      "opposite_nonzero_sum_slots": [i for i, d in enumerate(row["deltas"]) if d["opposite_nonzero_sum"]]} for row in rows]})
    result.update(numerical_summary_suppressed=False, scalar_comparisons_reported=384, shared_input_mismatches=[],
                  observed_coefficients=[{"backend": b, "mode": m, "point": p["point"], "coefficients_by_slot": p["coefficients"]}
                                         for b, m in ARMS for p in keyed[b, m, 1]["points"]],
                  slot_summaries=[{**slot, "metrics": summaries(entries, i)} for i, slot in enumerate(SLOTS)],
                  pair_breakdown=breakdown,
                  cuttools_reference_rows=[{"reference": row["left"], "other": row["right"], "point": row["point"],
                                            "reference_is_truth": False, "deltas_by_slot": row["deltas"]}
                                           for row in entries if row["left"][0] == 1 and row["right"][0] != 1])
    return result


def read_closed(path, expected_sha256):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""), "Explicit SHA256 required for closed input")
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    require(len(raw) <= MAX_INPUT_BYTES, "Input exceeds64MiB bound")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Closed input SHA256 mismatch")

    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(raw, parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Explicit caller-provided closed JSON; no default/discovery")
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    answer = summarize(read_closed(args.input, args.sha256))
    answer["input_sha256"] = args.sha256
    print(json.dumps(answer, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
